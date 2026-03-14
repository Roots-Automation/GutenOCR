"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from annotations import BlockAnnotation, LineAnnotation, WordAnnotation
from constants import SPLITS
from PIL import Image
from synthtiger import components, layers, templates

from elements import Background, Document


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge *overlay* into a deep copy of *base*.

    Scalar / list values in *overlay* replace those in *base*;
    nested dicts are merged recursively.
    """
    merged = {}
    for key in set(base) | set(overlay):
        if key in overlay and key in base:
            if isinstance(base[key], dict) and isinstance(overlay[key], dict):
                merged[key] = _deep_merge(base[key], overlay[key])
            else:
                merged[key] = overlay[key]
        elif key in overlay:
            merged[key] = overlay[key]
        else:
            merged[key] = base[key]
    return merged


def _clamp01(v: float) -> float:
    """Clamp a value to [0, 1]."""
    return max(0.0, min(1.0, v))


def _norm(val: float, dim: int) -> float:
    """Normalize a pixel coordinate by a dimension and clamp to [0, 1]."""
    return round(_clamp01(val / dim), 3)


def _norm_pt(x: float, y: float, w: int, h: int) -> list[float]:
    """Normalize an (x, y) point by image dimensions."""
    return [_norm(x, w), _norm(y, h)]


def _bbox_area_px(bbox: list[float], image_width: int, image_height: int) -> float:
    """Area of a normalized [x1, y1, x2, y2] bbox in pixels."""
    return (bbox[2] - bbox[0]) * image_width * (bbox[3] - bbox[1]) * image_height


def _build_block_annotations(block_ids: list[int], line_bboxes: list[list[float]]) -> list[BlockAnnotation]:
    """Build block-level annotations by grouping lines that share a block_id."""
    block_to_lines: dict[int, list[int]] = defaultdict(list)
    for i, bid in enumerate(block_ids):
        block_to_lines[bid].append(i)

    blocks = []
    for bid, line_indices in sorted(block_to_lines.items()):
        bboxes = [line_bboxes[i] for i in line_indices]
        bx1 = _clamp01(min(b[0] for b in bboxes))
        by1 = _clamp01(min(b[1] for b in bboxes))
        bx2 = _clamp01(max(b[2] for b in bboxes))
        by2 = _clamp01(max(b[3] for b in bboxes))
        blocks.append(
            BlockAnnotation(
                block_id=bid,
                bbox=[round(bx1, 3), round(by1, 3), round(bx2, 3), round(by2, 3)],
                line_ids=line_indices,
            )
        )
    return blocks


def _capture_line_bboxes(text_layers, w: int, h: int) -> list[list[float]]:
    """Compute normalized [x1, y1, x2, y2] bboxes for each text layer."""
    bboxes = []
    for text_layer in text_layers:
        bbox = [
            _norm(text_layer.left, w),
            _norm(text_layer.top, h),
            _norm(text_layer.left + text_layer.width, w),
            _norm(text_layer.top + text_layer.height, h),
        ]
        bboxes.append(bbox)
    return bboxes


def _capture_line_quads(text_layers, w: int, h: int) -> list[list[list[float]]]:
    """Compute normalized quad coordinates for each text layer."""
    quads = []
    for text_layer in text_layers:
        quad = text_layer.quad
        normalized = [_norm_pt(float(pt[0]), float(pt[1]), w, h) for pt in quad]
        quads.append(normalized)
    return quads


def _build_word_annotations(
    text_layers, words_per_line: list[list[dict]], w: int, h: int, emit_quads: bool
) -> list[WordAnnotation]:
    """Build word-level annotations from quad interpolation."""
    words = []
    word_global_id = 0
    for line_idx, (text_layer, word_local_data) in enumerate(zip(text_layers, words_per_line)):
        line_quad = text_layer.quad
        tl, tr, br, bl = line_quad[0], line_quad[1], line_quad[2], line_quad[3]

        for word in word_local_data:
            r1, r2 = word["x1_ratio"], word["x2_ratio"]
            w_tl = tl + r1 * (tr - tl)
            w_tr = tl + r2 * (tr - tl)
            w_br = bl + r2 * (br - bl)
            w_bl = bl + r1 * (br - bl)

            xs = [float(w_tl[0]), float(w_tr[0]), float(w_br[0]), float(w_bl[0])]
            ys = [float(w_tl[1]), float(w_tr[1]), float(w_br[1]), float(w_bl[1])]
            wx1 = _norm(min(xs), w)
            wy1 = _norm(min(ys), h)
            wx2 = _norm(max(xs), w)
            wy2 = _norm(max(ys), h)

            quad = None
            if emit_quads:
                quad = [
                    _norm_pt(float(w_tl[0]), float(w_tl[1]), w, h),
                    _norm_pt(float(w_tr[0]), float(w_tr[1]), w, h),
                    _norm_pt(float(w_br[0]), float(w_br[1]), w, h),
                    _norm_pt(float(w_bl[0]), float(w_bl[1]), w, h),
                ]

            words.append(
                WordAnnotation(
                    text=word["text"],
                    bbox=[wx1, wy1, wx2, wy2],
                    line_id=line_idx,
                    word_id=word_global_id,
                    quad=quad,
                )
            )
            word_global_id += 1
    return words


def _filter_degenerate(
    lines: list[LineAnnotation],
    words: list[WordAnnotation],
    blocks: list[BlockAnnotation],
    min_area: float,
    w: int,
    h: int,
) -> tuple[
    list[LineAnnotation],
    list[WordAnnotation],
    list[BlockAnnotation],
    int,
    int,
]:
    """Remove lines/words whose bbox area is below *min_area* pixels.

    Returns (lines, words, blocks, degenerate_line_count, degenerate_word_count).
    """
    degenerate_mask = [_bbox_area_px(ln.bbox, w, h) < min_area for ln in lines]
    deg_line_ct = sum(degenerate_mask)
    deg_word_ct = sum(1 for wd in words if degenerate_mask[wd.line_id]) if deg_line_ct else 0

    if not deg_line_ct:
        return lines, words, blocks, 0, 0

    survive = [i for i, degen in enumerate(degenerate_mask) if not degen]
    old_to_new = {old: new for new, old in enumerate(survive)}

    lines = [lines[i] for i in survive]
    for new_idx, ln in enumerate(lines):
        ln.line_id = new_idx

    new_words = []
    new_word_id = 0
    for wd in words:
        if wd.line_id not in old_to_new:
            continue
        wd.line_id = old_to_new[wd.line_id]
        wd.word_id = new_word_id
        new_words.append(wd)
        new_word_id += 1

    # Rebuild blocks from surviving lines
    block_ids = [ln.block_id for ln in lines]
    line_bboxes = [ln.bbox for ln in lines]
    blocks = _build_block_annotations(block_ids, line_bboxes)

    return lines, new_words, blocks, deg_line_ct, deg_word_ct


def _compute_quality_metrics(
    image: np.ndarray,
    lines: list[LineAnnotation],
    words: list[WordAnnotation],
    w: int,
    h: int,
    deg_lines: int,
    deg_words: int,
    null_ct: int,
    total_ct: int,
) -> dict:
    """Compute per-sample quality metrics from the rendered image."""
    gray = 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
    line_contrasts = []
    line_bbox_areas_px = []
    for ln in lines:
        bbox = ln.bbox
        x1_px = int(round(bbox[0] * w))
        y1_px = int(round(bbox[1] * h))
        x2_px = int(round(bbox[2] * w))
        y2_px = int(round(bbox[3] * h))
        if x2_px <= x1_px or y2_px <= y1_px:
            continue
        region = gray[y1_px:y2_px, x1_px:x2_px]
        if region.size == 0:
            continue
        line_contrasts.append(float(np.std(region)))
        line_bbox_areas_px.append((x2_px - x1_px) * (y2_px - y1_px))

    word_bbox_areas_px = []
    for wd in words:
        wb = wd.bbox
        wx = int(round(wb[2] * w)) - int(round(wb[0] * w))
        wy = int(round(wb[3] * h)) - int(round(wb[1] * h))
        if wx > 0 and wy > 0:
            word_bbox_areas_px.append(wx * wy)

    return {
        "min_line_contrast": round(min(line_contrasts), 3) if line_contrasts else None,
        "mean_line_contrast": round(float(np.mean(line_contrasts)), 3) if line_contrasts else None,
        "min_line_bbox_area_px": int(min(line_bbox_areas_px)) if line_bbox_areas_px else None,
        "min_word_bbox_area_px": int(min(word_bbox_areas_px)) if word_bbox_areas_px else None,
        "degenerate_line_count": int(deg_lines),
        "degenerate_word_count": int(deg_words),
        "textbox_null_count": int(null_ct),
        "textbox_total_count": int(total_ct),
        "image_size": [int(w), int(h)],
    }


class SynthDoG(templates.Template):
    def __init__(self, config=None, split_ratio: list[float] | None = None):
        super().__init__(config)
        if config is None:
            config = {}

        # Resolve _base inheritance
        if "_base" in config:
            base_path = Path(config.pop("_base"))
            with open(base_path, encoding="utf-8") as f:
                base_config = yaml.safe_load(f)
            config = _deep_merge(base_config, config)

        if split_ratio is None:
            split_ratio = [0.8, 0.1, 0.1]

        self.quality = config.get("quality", [50, 95])
        self.landscape = config.get("landscape", 0.5)
        self.short_size = config.get("short_size", [720, 1024])
        self.aspect_ratio = config.get("aspect_ratio", [1, 2])
        self.background = Background(config.get("background", {}))
        self.document = Document(config.get("document", {}))
        self.emit_quads = config.get("emit_quads", False)
        self.min_bbox_area = config.get("min_bbox_area", 16)
        self.effect = components.Iterator(
            [
                components.Switch(components.RGB()),
                components.Switch(components.Shadow()),
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

        # Capture line bboxes and optional quads
        line_bboxes = _capture_line_bboxes(text_layers, image_width, image_height)
        line_quads = _capture_line_quads(text_layers, image_width, image_height) if self.emit_quads else []

        # Assemble LineAnnotation objects
        lines = []
        for i, text in enumerate(texts):
            lines.append(
                LineAnnotation(
                    text=text,
                    bbox=line_bboxes[i],
                    block_id=block_ids[i],
                    line_id=i,
                    quad=line_quads[i] if line_quads else None,
                )
            )

        # Build word and block annotations
        words = _build_word_annotations(text_layers, words_per_line, image_width, image_height, self.emit_quads)
        blocks = _build_block_annotations(block_ids, line_bboxes)

        # Filter degenerate bboxes
        lines, words, blocks, deg_line_ct, deg_word_ct = _filter_degenerate(
            lines, words, blocks, self.min_bbox_area, image_width, image_height
        )

        # Render final image
        layer = layers.Group([*document_group.layers, bg_layer]).merge()
        self.effect.apply([layer])
        image = layer.output(bbox=[0, 0, *size])

        # Quality metrics
        quality_metrics = _compute_quality_metrics(
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

        # Flatten annotations to dicts for SynthTiger interface
        text_blocks_dicts = [{"block_id": b.block_id, "bbox": b.bbox, "line_ids": b.line_ids} for b in blocks]
        text_words_dicts = []
        for wd in words:
            entry: dict[str, Any] = {
                "text": wd.text,
                "bbox": wd.bbox,
                "word_id": wd.word_id,
                "line_id": wd.line_id,
            }
            if wd.quad is not None:
                entry["quad"] = wd.quad
            text_words_dicts.append(entry)

        data: dict[str, Any] = {
            "image": image,
            "label": label,
            "quality": quality,
            "roi": roi,
            "lines": lines,
            "words": words,
            "blocks": blocks,
            "text_blocks": text_blocks_dicts,
            "text_words": text_words_dicts,
            "quality_metrics": quality_metrics,
        }

        if self.emit_quads:
            data["text_quads"] = [ln.quad for ln in lines]

        return data

    def init_save(self, root):
        os.makedirs(root, exist_ok=True)

    def save(self, root, data, idx):
        lines: list[LineAnnotation] = data.get("lines", [])
        if not lines:
            return

        image = data["image"]
        quality = data["quality"]
        text_blocks = data.get("text_blocks", [])
        text_words = data.get("text_words", [])
        quality_metrics = data.get("quality_metrics", {})

        # Deterministic split
        split_idx = int(np.searchsorted(self._split_thresholds, np.random.default_rng(idx).random()))
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

        # Build text_lines_data from LineAnnotation objects
        text_lines_data = []
        for ln in lines:
            entry: dict[str, Any] = {
                "text": ln.text,
                "bbox": ln.bbox,
                "line_id": ln.line_id,
                "block_id": ln.block_id,
            }
            if ln.quad is not None:
                entry["quad"] = ln.quad
            text_lines_data.append(entry)

        keys = ["text_lines", "text_blocks", "text_words", "quality_metrics"]
        values = [text_lines_data, text_blocks, text_words, quality_metrics]

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
        if len(keys) != len(values):
            raise ValueError(f"Length does not match: keys({len(keys)}), values({len(values)})")

        gt_parse = {"gt_parse": dict(zip(keys, values))}
        gt_parse_str = json.dumps(gt_parse, ensure_ascii=False)
        metadata = {"file_name": image_filename, "ground_truth": gt_parse_str}
        return metadata
