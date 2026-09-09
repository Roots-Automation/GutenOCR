"""Main generator for index_definition synthetic dataset."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from annotations import build_pillow_annotations, compute_quality_metrics
from corpus.definitions import DefinitionCorpus
from effects.physical import (
    BookSpineShadowEffect,
    FoldCreaseEffect,
    LowTonerStreakEffect,
    MoireOverlayEffect,
    VignettingEffect,
    WatermarkEffect,
    apply_if_enabled,
)
from effects.pixel import apply_pixel_effects
from PIL import Image, ImageDraw
from serialization import (
    KEY_GENERATION_PARAMS,
    KEY_QUALITY_METRICS,
    KEY_TEXT_BLOCKS,
    KEY_TEXT_LINES,
    KEY_TEXT_WORDS,
    SPLITS,
    block_annotation_to_dict,
    encode_metadata,
    line_annotation_to_dict,
    word_annotation_to_dict,
)

from elements.entry import EntryRenderer
from elements.layout import ColumnCursor, sample_layout
from elements.page import PageRenderer

_PROJECT_ROOT = Path(__file__).resolve().parent

PHYSICAL_EFFECTS = [
    ("book_spine_shadow", BookSpineShadowEffect.apply),
    ("fold_crease", FoldCreaseEffect.apply),
    ("moire", MoireOverlayEffect.apply),
    ("low_toner_streaks", LowTonerStreakEffect.apply),
    ("watermark", WatermarkEffect.apply),
    ("vignetting", VignettingEffect.apply),
]


class IndexDefinitionGenerator:
    def __init__(self, config: dict):
        self._config = config
        self._corpus = DefinitionCorpus.load(
            min_definition_length=config.get("corpus", {}).get("min_definition_length", 10)
        )
        self._page_renderer = PageRenderer(config, _PROJECT_ROOT)
        self._entry_renderer = EntryRenderer(config, _PROJECT_ROOT)

        page_cfg = config.get("page", {})
        self._short_size = page_cfg.get("short_size", [720, 1440])
        self._aspect_ratio = page_cfg.get("aspect_ratio", [1.0, 2.5])
        self._landscape_prob = page_cfg.get("landscape", 0.1)

        quality_cfg = config.get("quality", {})
        self._jpeg_quality = config.get("quality_jpeg", [50, 95])
        self._min_word_count = quality_cfg.get("min_word_count", 20)
        self._min_line_height_px = quality_cfg.get("min_line_height_px", 8.0)
        self._min_sharpness = quality_cfg.get("min_sharpness", 5.0)
        self._max_cross_overlap = quality_cfg.get("max_cross_block_line_overlap", 0.50)
        self._max_retries = quality_cfg.get("max_retries", 10)
        self._min_bbox_area = config.get("min_bbox_area", 16.0)

        mode_cfg = config.get("mode", {})
        self._mode_choices = mode_cfg.get("choices", ["definition", "index"])
        self._mode_weights = np.array(mode_cfg.get("weights", [7, 3]), dtype=float)
        self._mode_weights /= self._mode_weights.sum()

        split_ratio = config.get("split_ratio", [0.8, 0.1, 0.1])
        self._split_thresholds = np.cumsum(split_ratio)

        self._effects_cfg = config.get("effects", {})
        self._physical_cfgs = {k: config.get(k, {}) for k, _ in PHYSICAL_EFFECTS}

    def _assign_split(self, text: str) -> str:
        h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        r = (h % 10000) / 10000.0
        for split, thresh in zip(SPLITS, self._split_thresholds):
            if r < thresh:
                return split
        return SPLITS[-1]

    def _quality_failure(self, metrics: dict) -> str | None:
        wc = metrics.get("word_count", 0) or 0
        if wc < self._min_word_count:
            return f"word_count={wc} < {self._min_word_count}"
        lh = metrics.get("min_line_height_px")
        if lh is not None and lh < self._min_line_height_px:
            return f"min_line_height_px={lh} < {self._min_line_height_px}"
        sh = metrics.get("sharpness")
        if sh is not None and sh < self._min_sharpness:
            return f"sharpness={sh} < {self._min_sharpness}"
        co = metrics.get("max_cross_block_line_overlap")
        if co is not None and co > self._max_cross_overlap:
            return f"max_cross_block_line_overlap={co} > {self._max_cross_overlap}"
        return None

    def generate(self, seed: int | None = None) -> dict | None:
        rng = np.random.default_rng(seed)

        # Sample page size
        short_side = int(rng.integers(self._short_size[0], self._short_size[1] + 1))
        aspect = float(rng.uniform(*self._aspect_ratio))
        long_side = int(short_side * aspect)
        landscape = bool(rng.random() < self._landscape_prob)
        width, height = (long_side, short_side) if landscape else (short_side, long_side)

        # Sample mode
        mode = str(self._mode_choices[int(rng.choice(len(self._mode_choices), p=self._mode_weights))])

        # Render background + paper
        page_img, paper_rgb = self._page_renderer.generate(width, height, rng)

        # Adaptive text color
        lum = 0.2989 * paper_rgb[0] + 0.5870 * paper_rgb[1] + 0.1140 * paper_rgb[2]
        if lum < 128:
            text_color: tuple[int, int, int] = (230, 230, 230)
        else:
            gray_val = int(rng.integers(0, 12))
            text_color = (gray_val, gray_val, gray_val)

        # Layout
        columns, _content_bbox = sample_layout(width, height, self._config, rng)

        # Page style (fonts, headword style, separator)
        style = self._entry_renderer.sample_page_style(rng)

        # Entry gap
        gap_range = self._config.get("layout", {}).get("entry_gap", [2, 8])
        entry_gap = int(rng.integers(gap_range[0], gap_range[1] + 1))

        # Draw onto page
        draw = ImageDraw.Draw(page_img)

        all_rendered_lines = []
        block_region_types: dict[int, str] = {}
        block_id = 0
        pool_cycles = 0
        entry_pool = self._corpus.shuffled_batch(rng)
        entry_idx = 0

        for col in columns:
            cursor = ColumnCursor(col)
            max_skips = 5
            skips = 0

            while not cursor.is_full:
                # Refill pool if exhausted
                if entry_idx >= len(entry_pool):
                    pool_cycles += 1
                    entry_pool = self._corpus.shuffled_batch(rng)
                    entry_idx = 0

                headword, definition = entry_pool[entry_idx]
                entry_idx += 1

                # Measure
                if mode == "definition":
                    needed = self._entry_renderer.measure_definition_entry(headword, definition, col, style)
                else:
                    needed = self._entry_renderer.measure_index_entry(col, style)

                if needed > cursor.remaining_height:
                    if needed > col.height:
                        # Pathologically tall — skip this entry
                        skips += 1
                        if skips >= max_skips:
                            break
                    else:
                        # Entry fits in column but not in remaining space
                        break
                    continue

                skips = 0
                block_region_types[block_id] = "body"

                if mode == "definition":
                    lines = self._entry_renderer.render_definition_entry(
                        draw, headword, definition, col, cursor, style, text_color, block_id, rng
                    )
                else:
                    lines = self._entry_renderer.render_index_entry(
                        draw, headword, col, cursor, style, text_color, block_id, rng
                    )

                all_rendered_lines.extend(lines)
                block_id += 1
                cursor.advance(entry_gap)

        # Convert to RGB numpy array
        image_rgb = np.array(page_img.convert("RGB"))

        # Build annotations (pre-effects)
        lines, words, blocks, deg_line_ct, deg_word_ct = build_pillow_annotations(
            all_rendered_lines, width, height, self._min_bbox_area, block_region_types
        )

        # Apply pixel effects
        image_rgb, pixel_provenance = apply_pixel_effects(image_rgb, self._effects_cfg, rng)

        # Apply physical effects (these use np.random global state — seed was already set)
        phys_provenance = {}
        for name, fn in PHYSICAL_EFFECTS:
            cfg = self._physical_cfgs.get(name, {})
            # Physical effects need RGBA; convert, apply, convert back
            rgba = np.dstack([image_rgb, np.full(image_rgb.shape[:2], 255, dtype=np.uint8)])
            rgba, fired = apply_if_enabled(cfg, fn, rgba)
            image_rgb = rgba[..., :3]
            phys_provenance[name] = {"applied": fired}

        # Quality metrics
        metrics = compute_quality_metrics(
            image_rgb,
            lines,
            words,
            width,
            height,
            deg_line_ct,
            deg_word_ct,
            null_ct=0,
            total_ct=max(1, block_id),
        )

        generation_params = {
            "canvas_size": [width, height],
            "landscape": landscape,
            "mode": mode,
            "num_cols": len(columns),
            "font_family": style["family_name"],
            "base_font_size_px": style["base_size"],
            "head_font_size_px": style["head_size"],
            "headword_style": style["head_style"],
            "entry_separator_style": style["sep_style"],
            "paper_rgb": list(paper_rgb),
            "text_color": list(text_color),
            "entries_rendered": block_id,
            "pool_cycles": pool_cycles,
            "effects": {**pixel_provenance, **phys_provenance},
        }

        return {
            "image": image_rgb,
            "lines": lines,
            "words": words,
            "blocks": blocks,
            "metrics": metrics,
            "generation_params": generation_params,
        }

    def save(self, output_root: str, data: dict, idx: int) -> bool:
        metrics = data["metrics"]
        failure = self._quality_failure(metrics)
        if failure:
            return False

        split_text = " ".join(ln.text for ln in data["lines"][:5])
        split = self._assign_split(split_text or str(idx))

        split_dir = Path(output_root) / split
        split_dir.mkdir(parents=True, exist_ok=True)

        img_name = f"image_{idx}.jpg"
        img_path = split_dir / img_name

        rng_q = np.random.default_rng(idx + 99991)
        jpeg_quality = int(rng_q.integers(self._jpeg_quality[0], self._jpeg_quality[1] + 1))
        Image.fromarray(data["image"]).save(img_path, format="JPEG", quality=jpeg_quality)

        record = encode_metadata(
            img_name,
            [KEY_TEXT_LINES, KEY_TEXT_WORDS, KEY_TEXT_BLOCKS, KEY_QUALITY_METRICS, KEY_GENERATION_PARAMS],
            [
                [line_annotation_to_dict(ln) for ln in data["lines"]],
                [word_annotation_to_dict(wd) for wd in data["words"]],
                [block_annotation_to_dict(blk) for blk in data["blocks"]],
                data["metrics"],
                data["generation_params"],
            ],
        )

        jsonl_path = split_dir / "metadata.jsonl"
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return True


def _worker(args_tuple: tuple) -> tuple[int, bool]:
    config_path, output_root, idx, seed = args_tuple
    with open(config_path) as f:
        config = yaml.safe_load(f)
    gen = IndexDefinitionGenerator(config)
    max_retries = config.get("quality", {}).get("max_retries", 10)
    for attempt in range(max_retries):
        data = gen.generate(seed=seed + attempt * 7919)
        if data is None:
            continue
        if gen.save(output_root, data, idx):
            return idx, True
    return idx, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate index/definition synthetic dataset")
    parser.add_argument("--config", default="config/config_base.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-idx", type=int, default=0)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    output_root = Path(args.output)
    for split in SPLITS:
        (output_root / split).mkdir(parents=True, exist_ok=True)

    tasks = [(str(config_path), str(output_root), args.start_idx + i, args.seed + i) for i in range(args.count)]

    success = fail = 0
    if args.workers <= 1:
        for t in tasks:
            idx, ok = _worker(t)
            if ok:
                success += 1
            else:
                fail += 1
            if (success + fail) % 100 == 0:
                print(f"  {success} ok, {fail} failed", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, t): t for t in tasks}
            for fut in as_completed(futures):
                idx, ok = fut.result()
                if ok:
                    success += 1
                else:
                    fail += 1
                if (success + fail) % 100 == 0:
                    print(f"  {success} ok, {fail} failed", flush=True)

    print(f"Done: {success} generated, {fail} failed")


if __name__ == "__main__":
    main()
