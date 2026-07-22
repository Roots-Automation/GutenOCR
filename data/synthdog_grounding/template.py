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

# imgaug lazily initializes its GLOBAL_RNG on the first call to get_global_rng(),
# drawing one value from np.random to seed it. If that happens inside
# set_global_random_seed() (which has already set np.random to a deterministic
# state), it advances the MT19937 position by 1 and breaks reproducibility on
# the first generate(seed=N) call. Force initialization here at import time so
# GLOBAL_RNG is never None when set_global_random_seed() runs.
import imgaug.random as _imgaug_random  # noqa: E402

_imgaug_random.get_global_rng()

import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from annotations import build_annotations, compute_quality_metrics  # noqa: E402
from effects.physical import (  # noqa: E402
    BookSpineShadowEffect,
    FoldCreaseEffect,
    LowTonerStreakEffect,
    MoireOverlayEffect,
    VignettingEffect,
    WatermarkEffect,
    apply_if_enabled,
)
from PIL import Image  # noqa: E402
from serialization import (  # noqa: E402
    KEY_QUALITY_METRICS,
    KEY_TEXT_BLOCKS,
    KEY_TEXT_LINES,
    KEY_TEXT_WORDS,
    QUALITY_FILTER_DEFAULTS,
    SPLITS,
    LineAnnotation,
    block_annotation_to_dict,
    encode_metadata,
    line_annotation_to_dict,
    word_annotation_to_dict,
)
from synthtiger import components, layers, templates  # noqa: E402

from elements import Background, Document  # noqa: E402


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
            if "font_path" in node and isinstance(node["font_path"], str):
                if not os.path.isabs(node["font_path"]):
                    node["font_path"] = str((base_dir / node["font_path"]).resolve())
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


def _rotate_point(px, py, cx, cy, angle_deg):
    """Rotate (px, py) around (cx, cy) by angle_deg degrees."""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    dx, dy = px - cx, py - cy
    return cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a


def _rotate_quad(quad, cx, cy, angle_deg):
    return [list(_rotate_point(p[0], p[1], cx, cy, angle_deg)) for p in quad]


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
        _fd = {k: v for k, _op, v in QUALITY_FILTER_DEFAULTS}
        self.min_word_count: int = int(config.get("min_word_count", _fd["word_count"]))
        self.max_textbox_null_frac: float = float(config.get("max_textbox_null_frac", _fd["textbox_null_frac"]))
        self.min_line_height_px: float = float(config.get("min_line_height_px", _fd["min_line_height_px"]))
        self.min_sharpness: float = float(config.get("min_sharpness", _fd["sharpness"]))
        self.max_intra_block_line_overlap: float = float(
            config.get("max_intra_block_line_overlap", _fd["max_intra_block_line_overlap"])
        )
        self.max_cross_block_line_overlap: float = float(
            config.get("max_cross_block_line_overlap", _fd["max_cross_block_line_overlap"])
        )
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
        self.effect = components.Iterator(
            [
                components.Switch(components.RGB()),
                components.Switch(components.Grayscale()),
                components.Switch(components.Contrast()),
                components.Switch(components.Brightness()),
                components.Switch(components.MotionBlur()),
                components.Switch(components.GaussianBlur()),
                components.Switch(components.Resample()),
                components.Switch(components.JpegCompression()),
            ],
            **config.get("effect", {}),
        )

        self.skew_angle: tuple = config.get("skew", {}).get("angle", [0, 0])
        self.skew_prob: float = config.get("skew", {}).get("prob", 0.0)

        self.vignetting_cfg = config.get("vignetting", {})
        self.book_spine_cfg = config.get("book_spine_shadow", {})
        self.fold_crease_cfg = config.get("fold_crease", {})
        self.low_toner_cfg = config.get("low_toner_streaks", {})
        self.moire_cfg = config.get("moire", {})
        self.watermark_cfg = config.get("watermark", {})

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

    def _render(
        self,
        document_group,
        bg_layer,
        size: tuple[int, int],
    ) -> np.ndarray:
        """Merge layers, apply effects, and rasterize to a numpy array."""
        # Apply shadow to background only.
        self.bg_effect.apply([bg_layer])
        # Merge paper + text into a single doc layer, then apply a weaker shadow
        # for page-level depth. This partially affects text-vs-paper contrast but
        # at reduced intensity; the backstop in save() catches any failures.
        doc_layer = document_group.merge()
        self.doc_effect.apply([doc_layer])
        # Apply doc-layer physical effects (operate on the paper+text composite,
        # before compositing with the background).
        doc_img = np.clip(doc_layer.image, 0, 255).astype(np.uint8)
        doc_img = apply_if_enabled(self.book_spine_cfg, BookSpineShadowEffect.apply, doc_img)
        doc_img = apply_if_enabled(self.fold_crease_cfg, FoldCreaseEffect.apply, doc_img)
        doc_layer.image = doc_img.astype(np.float32)
        layer = layers.Group([doc_layer, bg_layer]).merge()
        # Apply elastic distortion to the composited image. This runs *after*
        # annotations are captured from per-layer quads, so saved bboxes reflect
        # pre-distortion geometry. Misalignment is at or below the level of the
        # blur effects also applied post-annotation, so no correction is warranted.
        self.document.elastic_distortion.apply([layer])
        self.effect.apply([layer])
        result = layer.output(bbox=[0, 0, *size])
        # Global physical effects applied to the final composite: moiré → streaks → watermark → vignetting
        result = apply_if_enabled(self.moire_cfg, MoireOverlayEffect.apply, result)
        result = apply_if_enabled(self.low_toner_cfg, LowTonerStreakEffect.apply, result)
        result = apply_if_enabled(self.watermark_cfg, WatermarkEffect.apply, result)
        result = apply_if_enabled(self.vignetting_cfg, VignettingEffect.apply, result)
        return result

    def generate(self, seed: int | None = None):
        if seed is not None:
            import synthtiger as _st

            _st.set_global_random_seed(seed)
        landscape = np.random.rand() < self.landscape
        short_size = np.random.randint(self.short_size[0], self.short_size[1] + 1)
        aspect_ratio = np.random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
        long_size = int(short_size * aspect_ratio)
        size = (long_size, short_size) if landscape else (short_size, long_size)

        bg_layer = self.background.generate(size)
        (
            paper_layer,
            text_layers,
            texts,
            block_ids,
            words_per_line,
            block_region_types,
            textbox_null_count,
            textbox_total_count,
        ) = self.document.generate(size)

        document_group = layers.Group([*text_layers, paper_layer])
        document_space = np.clip(size - document_group.size, 0, None)
        document_group.left = np.random.randint(document_space[0] + 1)
        document_group.top = np.random.randint(document_space[1] + 1)
        roi = np.array(paper_layer.quad, dtype=int)

        skew_angle = 0.0
        if np.random.rand() < self.skew_prob:
            skew_angle = float(np.random.uniform(self.skew_angle[0], self.skew_angle[1]))
            cx = document_group.left + document_group.width / 2
            cy = document_group.top + document_group.height / 2
            for layer in [*text_layers, paper_layer]:
                layer.quad = _rotate_quad(layer.quad, cx, cy, skew_angle)

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
            block_region_types=block_region_types,
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
        quality_metrics["skew_angle"] = round(skew_angle, 3)

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

    def _quality_failure(self, data: dict) -> str | None:
        """Return a human-readable failure reason, or None if the sample passes all filters."""
        lines = data.get("lines", [])
        if not lines:
            return "no lines"
        qm = data.get("quality_metrics", {})
        contrast = qm.get("min_line_contrast_ratio")
        if contrast is not None and contrast < self.min_contrast_ratio:
            return f"contrast {contrast:.3f} < {self.min_contrast_ratio}"
        words = qm.get("word_count", 0)
        if words < self.min_word_count:
            return f"words {words} < {self.min_word_count}"
        null_frac = qm.get("textbox_null_frac", 0.0) or 0.0
        if null_frac > self.max_textbox_null_frac:
            return f"null_frac {null_frac:.3f} > {self.max_textbox_null_frac}"
        min_h = qm.get("min_line_height_px")
        if min_h is not None and min_h < self.min_line_height_px:
            return f"min_line_height {min_h:.1f} < {self.min_line_height_px}"
        sharpness = qm.get("sharpness")
        if sharpness is not None and sharpness < self.min_sharpness:
            return f"sharpness {sharpness:.1f} < {self.min_sharpness}"
        intra = qm.get("max_intra_block_line_overlap")
        if intra is not None and intra > self.max_intra_block_line_overlap:
            return f"intra_overlap {intra:.3f} > {self.max_intra_block_line_overlap}"
        cross = qm.get("max_cross_block_line_overlap")
        if cross is not None and cross > self.max_cross_block_line_overlap:
            return f"cross_overlap {cross:.3f} > {self.max_cross_block_line_overlap}"
        return None

    _SAVE_MAX_RETRIES: int = 20

    def save(self, root, data, idx):
        # Retry with deterministic sub-seeds until the sample passes all quality
        # filters, so that requesting N samples always yields exactly N on disk.
        # Retry seeds are spaced far from the primary seed space: idx * 100_000 + attempt.
        for attempt in range(self._SAVE_MAX_RETRIES):
            failure = self._quality_failure(data)
            if failure is None:
                break
            if attempt == 0:
                retry_base = (idx + 1) * 100_000
            retry_seed = retry_base + attempt
            data = self.generate(seed=retry_seed)
        else:
            # All retries exhausted — log and skip rather than write a bad sample.
            import warnings

            warnings.warn(
                f"save idx={idx}: could not produce a passing sample after {self._SAVE_MAX_RETRIES} retries; skipping.",
                stacklevel=2,
            )
            return

        lines: list[LineAnnotation] = data.get("lines", [])
        quality_metrics = data.get("quality_metrics", {})
        image = data["image"]
        quality = data["quality"]
        words = data.get("words", [])
        blocks = data.get("blocks", [])

        # Content-based split: hash the label so the same text always lands
        # in the same split regardless of generation order or worker count.
        label_hash = int(hashlib.sha256(data["label"].encode()).hexdigest()[:16], 16)
        split_idx = min(
            int(np.searchsorted(self._split_thresholds, np.random.default_rng(label_hash).random())),
            len(self.splits) - 1,
        )
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
