"""CLI entry point for latex_academic synthetic document generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker data structures
# ---------------------------------------------------------------------------


def _load_config(config_path: str | None) -> dict:
    default = Path(__file__).parent / "config" / "config_base.yaml"
    path = Path(config_path) if config_path else default
    return yaml.safe_load(path.read_text())


def _split_name(full_text: str, thresholds: np.ndarray, splits: list[str]) -> str:
    label_hash = int(hashlib.sha256(full_text.encode()).hexdigest()[:16], 16)
    idx = min(int(np.searchsorted(thresholds, np.random.default_rng(label_hash).random())), len(splits) - 1)
    return splits[idx]


# ---------------------------------------------------------------------------
# Per-worker generation function (module-level for pickling)
# ---------------------------------------------------------------------------


def _worker(args: tuple) -> list[dict] | None:
    """Generate one document and return per-page result dicts.

    Returns None if all retries are exhausted.
    Each returned dict contains JPEG bytes + annotation data for one page.
    """
    sample_idx, cfg, seed, dpi = args

    # Lazy imports inside worker to avoid pickling issues
    from augmentation import augment, encode_jpeg, laplacian_variance
    from content import SyntheticContent
    from document import CompilationError, LaTeXDocument
    from extraction import extract_pages
    from layout import LayoutConfig
    from serialization import (
        block_annotation_to_dict,
        line_annotation_to_dict,
        word_annotation_to_dict,
    )

    max_retries = cfg.get("quality", {}).get("max_retries", 10)
    min_words = cfg.get("quality", {}).get("min_word_count", 20)
    min_sharpness = cfg.get("quality", {}).get("min_sharpness", 10.0)
    aug_enabled = cfg.get("augmentation", {}).get("enabled", True)
    aug_cfg = cfg.get("augmentation", {})

    doc = LaTeXDocument()

    for attempt in range(max_retries):
        rng = random.Random(seed + attempt * 100_000)

        page_images_obj = None
        try:
            layout = LayoutConfig.from_config(rng, cfg)
            content = SyntheticContent.generate(layout, rng)
            context = content.to_template_context()

            page_images_obj = doc.generate(context, dpi=dpi)
            images = page_images_obj.images
            page_sizes = [(img.width, img.height) for img in images]

            page_annotations = extract_pages(
                pdf_path=page_images_obj.pdf_path,
                page_images_sizes=page_sizes,
                dpi=dpi,
                layout_n_cols=layout.n_cols,
                source_math_tokens=content.math_token_sequence(),
            )

        except CompilationError as exc:
            logger.debug("Sample %d attempt %d: compilation failed: %s", sample_idx, attempt, exc)
            if page_images_obj is not None:
                page_images_obj.cleanup()
                page_images_obj = None
            continue
        except Exception as exc:
            logger.debug("Sample %d attempt %d: unexpected error: %s", sample_idx, attempt, exc)
            if page_images_obj is not None:
                page_images_obj.cleanup()
                page_images_obj = None
            continue
        finally:
            if page_images_obj is not None:
                page_images_obj.cleanup()
                page_images_obj = None

        # Quality gate
        total_words = sum(p.word_count for p in page_annotations)
        if total_words < min_words:
            logger.debug("Sample %d attempt %d: too few words (%d)", sample_idx, attempt, total_words)
            continue

        generation_params = {
            **layout.to_dict(),
            "dpi": dpi,
            "seed": seed,
            "attempt": attempt,
        }

        page_results = []
        all_passed = True
        for page_idx, (img, ann) in enumerate(zip(images, page_annotations)):
            sharpness = laplacian_variance(img)
            if sharpness < min_sharpness:
                logger.debug("Sample %d page %d: sharpness %.1f < %.1f", sample_idx, page_idx, sharpness, min_sharpness)
                all_passed = False
                break

            if aug_enabled:
                img = augment(img, rng, aug_cfg)

            jpeg_bytes = encode_jpeg(img, rng)

            quality_metrics = {
                "word_count": ann.word_count,
                "line_count": ann.line_count,
                "sharpness": round(sharpness, 1),
                "image_size": [img.width, img.height],
                "word_segmentation_method": "pdfplumber",
            }

            page_results.append(
                {
                    "jpeg_bytes": jpeg_bytes,
                    "words": [word_annotation_to_dict(w) for w in ann.words],
                    "lines": [line_annotation_to_dict(ln) for ln in ann.lines],
                    "blocks": [block_annotation_to_dict(blk) for blk in ann.blocks],
                    "quality_metrics": quality_metrics,
                    "generation_params": {**generation_params, "page_index": page_idx},
                    "full_text": content.full_text,
                }
            )

        if all_passed and page_results:
            return page_results

    logger.warning("Sample %d: exhausted %d retries — skipping", sample_idx, max_retries)
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Generate synthetic LaTeX academic documents.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--count", type=int, default=1000, help="Number of documents to generate.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes.")
    parser.add_argument("--dpi", type=int, default=150, help="Rasterization DPI.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    split_ratio = cfg.get("generation", {}).get("split_ratio", [0.8, 0.1, 0.1])
    splits = ["train", "validation", "test"]
    thresholds = np.cumsum([r / sum(split_ratio) for r in split_ratio])

    # Create output dirs + open JSONL files
    args.output.mkdir(parents=True, exist_ok=True)
    split_dirs = {}
    jsonl_files = {}
    for split in splits:
        split_dir = args.output / split
        split_dir.mkdir(exist_ok=True)
        split_dirs[split] = split_dir
        jsonl_files[split] = open(split_dir / "metadata.jsonl", "a", encoding="utf-8")  # noqa: WPS515

    image_counter = {split: _count_existing_images(split_dirs[split]) for split in splits}
    total_saved = sum(image_counter.values())

    logger.info("Generating %d documents with %d workers at %d DPI", args.count, args.workers, args.dpi)

    from serialization import (
        KEY_GENERATION_PARAMS,
        KEY_QUALITY_METRICS,
        KEY_TEXT_BLOCKS,
        KEY_TEXT_LINES,
        KEY_TEXT_WORDS,
        encode_metadata,
    )

    worker_args = [(i, cfg, args.seed + i, args.dpi) for i in range(args.count)]

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, wa): wa[0] for wa in worker_args}
            with tqdm(total=args.count, desc="Generating") as pbar:
                for future in as_completed(futures):
                    pbar.update(1)
                    page_results = future.result()
                    if not page_results:
                        continue

                    full_text = page_results[0]["full_text"]
                    split = _split_name(full_text, thresholds, splits)
                    split_dir = split_dirs[split]

                    for page_result in page_results:
                        img_idx = image_counter[split]
                        image_counter[split] += 1
                        total_saved += 1

                        filename = f"image_{img_idx}.jpg"
                        (split_dir / filename).write_bytes(page_result["jpeg_bytes"])

                        record = encode_metadata(
                            filename,
                            keys=[
                                KEY_TEXT_LINES,
                                KEY_TEXT_WORDS,
                                KEY_TEXT_BLOCKS,
                                KEY_QUALITY_METRICS,
                                KEY_GENERATION_PARAMS,
                            ],
                            values=[
                                page_result["lines"],
                                page_result["words"],
                                page_result["blocks"],
                                page_result["quality_metrics"],
                                page_result["generation_params"],
                            ],
                        )
                        jsonl_files[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                        jsonl_files[split].flush()

    finally:
        for f in jsonl_files.values():
            f.close()

    logger.info("Done. %d pages saved across %d splits.", total_saved, len(splits))
    for split in splits:
        logger.info("  %s: %d images", split, image_counter[split])


def _count_existing_images(split_dir: Path) -> int:
    return len(list(split_dir.glob("image_*.jpg")))


if __name__ == "__main__":
    main()
