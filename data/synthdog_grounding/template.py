"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

# When synthtiger loads this file directly (not as a package), the package
# __init__.py never runs.  Ensure the package root is on sys.path so that
# sibling modules (pillow_compat, serialization, elements, …) can always
# be imported with plain bare imports.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pillow_compat  # noqa: E402, F401, I001

import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from synthtiger import components, layers, templates  # noqa: E402

from annotations import build_annotations, compute_quality_metrics  # noqa: E402
from elements import Background, Document  # noqa: E402
from serialization import (  # noqa: E402
    KEY_QUALITY_METRICS,
    KEY_TEXT_BLOCKS,
    KEY_TEXT_LINES,
    KEY_TEXT_WORDS,
    SPLITS,
    LineAnnotation,
    block_annotation_to_dict,
    encode_metadata,
    line_annotation_to_dict,
    word_annotation_to_dict,
)


def _resolve_config_paths(config: dict, base_dir: Path) -> dict:
    """Resolve relative resource paths in config to absolute paths.

    SynthTiger's BaseTexture and BaseFont resolve paths via os.path.exists()
    during __init__.  When the CLI spawns worker processes, the child's cwd
    may differ from the parent's, breaking relative paths.  Resolving them
    here (in the main process, where cwd is correct) makes the config
    portable across processes.
    """
    config = copy.deepcopy(config)

    def _resolve(node):
        if isinstance(node, dict):
            if "paths" in node and isinstance(node["paths"], list):
                node["paths"] = [str((base_dir / p).resolve()) if not os.path.isabs(p) else p for p in node["paths"]]
            if "path" in node and isinstance(node["path"], str):
                if not os.path.isabs(node["path"]):
                    node["path"] = str((base_dir / node["path"]).resolve())
            for v in node.values():
                _resolve(v)
        elif isinstance(node, list):
            for item in node:
                _resolve(item)

    _resolve(config)
    return config


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge *overlay* into a deep copy of *base*.

    Scalar / list values in *overlay* replace those in *base*;
    nested dicts are merged recursively.
    """
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _package_data(
    *,
    image: np.ndarray,
    label: str,
    quality: int,
    roi: np.ndarray,
    lines: list[LineAnnotation],
    words: list,
    blocks: list,
    quality_metrics: dict,
    emit_quads: bool,
) -> dict[str, Any]:
    """Assemble the final data dict returned by generate()."""
    data: dict[str, Any] = {
        "image": image,
        "label": label,
        "quality": quality,
        "roi": roi,
        "lines": lines,
        "words": words,
        "blocks": blocks,
        "quality_metrics": quality_metrics,
    }

    if emit_quads:
        data["text_quads"] = [ln.quad for ln in lines]

    return data


def _check_font_dirs(config: dict) -> None:
    """Raise if any configured font directory is empty (no .ttf/.otf files)."""
    font_cfg = config.get("document", {}).get("content", {}).get("font", {})
    for font_dir in font_cfg.get("paths", []):
        p = Path(font_dir)
        if not p.is_dir():
            continue
        has_fonts = any(p.glob("*.ttf")) or any(p.glob("*.otf"))
        if not has_fonts:
            raise FileNotFoundError(
                f"No font files found in {p}.\n"
                f"Run 'uv run python fetch_fonts.py' from the synthdog_grounding/ "
                f"directory to download the required fonts."
            )


class SynthDoG(templates.Template):
    def __init__(self, config=None, split_ratio: list[float] | None = None):
        super().__init__(config)
        if config is None:
            config = {}

        # Deep-copy so we never mutate the caller's dict — SynthTiger
        # passes the same config object to multiple read_template() calls.
        config = copy.deepcopy(config)

        # Resolve _base inheritance
        if "_base" in config:
            base_path = Path(config.pop("_base"))
            if not base_path.is_absolute():
                base_path = Path(__file__).resolve().parent / base_path
            with open(base_path, encoding="utf-8") as f:
                base_config = yaml.safe_load(f)
            config = _deep_merge(base_config, config)

        # Resolve relative resource paths to absolute so that SynthTiger
        # worker processes (which may have a different cwd) can find them.
        config = _resolve_config_paths(config, Path(__file__).resolve().parent)

        # Verify font directories aren't empty before proceeding.
        _check_font_dirs(config)

        if split_ratio is None:
            split_ratio = config.get("split_ratio", [0.8, 0.1, 0.1])

        self.quality = config.get("quality", [50, 95])
        self.landscape = config.get("landscape", 0.5)
        self.short_size = config.get("short_size", [720, 1024])
        self.aspect_ratio = config.get("aspect_ratio", [1, 2])
        self.background = Background(config.get("background", {}))
        self.document = Document(config.get("document", {}))
        self.emit_quads = config.get("emit_quads", False)
        self.min_bbox_area = config.get("min_bbox_area", 16)
        self.min_contrast_ratio: float = float(config.get("min_contrast_ratio", 3.0))
        # Shadow applied to bg layer only, before merge
        self.bg_effect = components.Iterator(
            [components.Switch(components.Shadow())],
            **config.get("bg_effect", {}),
        )
        # Weaker shadow applied to the merged document layer (paper + text),
        # before compositing with bg. Restores page-level shadow depth while
        # keeping intensity low enough that the contrast backstop rarely fires.
        self.doc_effect = components.Iterator(
            [components.Switch(components.Shadow())],
            **config.get("doc_effect", {}),
        )
        # Shadow removed; 5 components now
        self.effect = components.Iterator(
            [
                components.Switch(components.RGB()),
                components.Switch(components.Contrast()),
                components.Switch(components.Brightness()),
                components.Switch(components.MotionBlur()),
                components.Switch(components.GaussianBlur()),
            ],
            **config.get("effect", {}),
        )

        # config for splits
        self.splits = SPLITS
        if any(r < 0 for r in split_ratio):
            raise ValueError(f"split_ratio values must be non-negative, got {split_ratio}")
        ratio_sum = sum(split_ratio)
        if not (0.99 <= ratio_sum <= 1.01):
            raise ValueError(f"split_ratio must sum to 1.0 (got {ratio_sum})")
        self.split_ratio = [r / ratio_sum for r in split_ratio]
        self._split_thresholds = np.cumsum(self.split_ratio)

    def __del__(self):
        if hasattr(self, "document"):
            self.document.close()

    def _render(self, document_group, bg_layer, size: tuple[int, int]) -> np.ndarray:
        """Merge layers, apply effects, and rasterize to a numpy array."""
        # Apply shadow to background only.
        self.bg_effect.apply([bg_layer])
        # Merge paper + text into a single doc layer, then apply a weaker shadow
        # for page-level depth. This partially affects text-vs-paper contrast but
        # at reduced intensity; the backstop in save() catches any failures.
        doc_layer = document_group.merge()
        self.doc_effect.apply([doc_layer])
        layer = layers.Group([doc_layer, bg_layer]).merge()
        # Apply elastic distortion to the composited image. This runs *after*
        # annotations are captured from per-layer quads, so saved bboxes reflect
        # the pre-distortion geometry.
        #
        # Empirical analysis (check_elastic_distortion.py, n=50) confirmed this
        # misalignment is negligible with config params alpha=[0,1], sigma=[0,0.5]:
        #   mean pixel delta   1.4  (vs motion blur 2.6,  Gaussian blur 2.1)
        #   p95 pixel delta   10.1  (vs motion blur 24.1, Gaussian blur 27.5)
        #   centroid drift     2.3px (vs motion blur 2.6px, Gaussian blur 2.2px)
        #   text coverage      0.90  (vs motion blur 0.92, Gaussian blur 0.93)
        # Elastic distortion is at or below the level of the blur effects that are
        # also applied post-annotation, so no fix is warranted.
        self.document.elastic_distortion.apply([layer])
        self.effect.apply([layer])
        return layer.output(bbox=[0, 0, *size])

    def generate(self):
        landscape = np.random.rand() < self.landscape
        short_size = np.random.randint(self.short_size[0], self.short_size[1] + 1)
        aspect_ratio = np.random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
        long_size = int(short_size * aspect_ratio)
        size = (long_size, short_size) if landscape else (short_size, long_size)

        bg_layer = self.background.generate(size)
        paper_layer, text_layers, texts, block_ids, words_per_line, textbox_null_count, textbox_total_count = (
            self.document.generate(size)
        )

        document_group = layers.Group([*text_layers, paper_layer])
        document_space = np.clip(size - document_group.size, 0, None)
        document_group.left = np.random.randint(document_space[0] + 1)
        document_group.top = np.random.randint(document_space[1] + 1)
        roi = np.array(paper_layer.quad, dtype=int)

        image_width, image_height = size

        lines, words, blocks, deg_line_ct, deg_word_ct = build_annotations(
            text_layers,
            texts,
            block_ids,
            words_per_line,
            image_width,
            image_height,
            self.emit_quads,
            self.min_bbox_area,
        )

        image = self._render(document_group, bg_layer, size)

        quality_metrics = compute_quality_metrics(
            image,
            lines,
            words,
            image_width,
            image_height,
            deg_line_ct,
            deg_word_ct,
            textbox_null_count,
            textbox_total_count,
        )

        label = re.sub(r"\s+", " ", " ".join(ln.text for ln in lines)).strip()
        quality = np.random.randint(self.quality[0], self.quality[1] + 1)

        return _package_data(
            image=image,
            label=label,
            quality=quality,
            roi=roi,
            lines=lines,
            words=words,
            blocks=blocks,
            quality_metrics=quality_metrics,
            emit_quads=self.emit_quads,
        )

    def init_save(self, root):
        os.makedirs(root, exist_ok=True)

    def save(self, root, data, idx):
        lines: list[LineAnnotation] = data.get("lines", [])
        if not lines:
            return

        quality_metrics = data.get("quality_metrics", {})
        min_contrast = quality_metrics.get("min_line_contrast_ratio")
        if min_contrast is not None and min_contrast < self.min_contrast_ratio:
            return

        image = data["image"]
        quality = data["quality"]
        words = data.get("words", [])
        blocks = data.get("blocks", [])

        # Content-based split: hash the label so the same text always lands
        # in the same split regardless of generation order or worker count.
        label_hash = int(hashlib.sha256(data["label"].encode()).hexdigest()[:16], 16)
        split_idx = int(np.searchsorted(self._split_thresholds, np.random.default_rng(label_hash).random()))
        output_dirpath = os.path.join(root, self.splits[split_idx])

        # save image
        image_filename = f"image_{idx}.jpg"
        image_filepath = os.path.join(output_dirpath, image_filename)
        os.makedirs(os.path.dirname(image_filepath), exist_ok=True)
        image = Image.fromarray(np.clip(image[..., :3], 0, 255).astype(np.uint8))
        image.save(image_filepath, quality=quality)

        # save metadata
        metadata_filename = "metadata.jsonl"
        metadata_filepath = os.path.join(output_dirpath, metadata_filename)
        os.makedirs(os.path.dirname(metadata_filepath), exist_ok=True)

        text_lines_data = [line_annotation_to_dict(ln) for ln in lines]
        text_words_data = [word_annotation_to_dict(wd) for wd in words]
        text_blocks_data = [block_annotation_to_dict(b) for b in blocks]

        keys = [KEY_TEXT_LINES, KEY_TEXT_BLOCKS, KEY_TEXT_WORDS, KEY_QUALITY_METRICS]
        values = [text_lines_data, text_blocks_data, text_words_data, quality_metrics]

        metadata = self.format_metadata(
            image_filename=image_filename,
            keys=keys,
            values=values,
        )
        with open(metadata_filepath, "a") as fp:
            json.dump(metadata, fp, ensure_ascii=False)
            fp.write("\n")

    def end_save(self, root):
        pass

    def format_metadata(self, image_filename: str, keys: list[str], values: list[Any]) -> dict[str, str]:
        """
        Fit gt_parse contents to huggingface dataset's format
        keys and values, whose lengths are equal, are used to constrcut 'gt_parse' field in 'ground_truth' field
        Args:
            keys: List of task_name
            values: List of actual gt data corresponding to each task_name
        """
        return encode_metadata(image_filename, keys, values)
