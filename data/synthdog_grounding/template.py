"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

# Ensure Pillow compatibility patch is loaded before anything else
try:
    import pillow_compat
except ImportError:
    # If running as part of the package vs script, try relative import or assume it's already patched
    try:
        from . import pillow_compat  # noqa: F401 (side-effect import)
    except ImportError:
        pass

import json
import os
import re
from collections import defaultdict
from typing import Any

import numpy as np
from PIL import Image
from synthtiger import components, layers, templates

from elements import Background, Document


def _clamp01(v: float) -> float:
    """Clamp a value to [0, 1]."""
    return max(0.0, min(1.0, v))


def _bbox_area_px(bbox: list[float], image_width: int, image_height: int) -> float:
    """Area of a normalized [x1, y1, x2, y2] bbox in pixels."""
    return (bbox[2] - bbox[0]) * image_width * (bbox[3] - bbox[1]) * image_height


def _build_text_blocks(block_ids: list[int], text_bboxes: list[list[float]]) -> list[dict]:
    """Build block-level bboxes by grouping lines that share a block_id."""
    block_to_lines = defaultdict(list)
    for i, bid in enumerate(block_ids):
        block_to_lines[bid].append(i)

    text_blocks = []
    for bid, line_indices in sorted(block_to_lines.items()):
        bboxes = [text_bboxes[i] for i in line_indices]
        bx1 = _clamp01(min(b[0] for b in bboxes))
        by1 = _clamp01(min(b[1] for b in bboxes))
        bx2 = _clamp01(max(b[2] for b in bboxes))
        by2 = _clamp01(max(b[3] for b in bboxes))
        text_blocks.append(
            {
                "block_id": bid,
                "bbox": [round(bx1, 3), round(by1, 3), round(bx2, 3), round(by2, 3)],
                "line_ids": line_indices,
            }
        )
    return text_blocks


class SynthDoG(templates.Template):
    def __init__(self, config=None, split_ratio: list[float] | None = None):
        super().__init__(config)
        if config is None:
            config = {}
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
        self.splits = ["train", "validation", "test"]
        if any(r < 0 for r in split_ratio):
            raise ValueError(f"split_ratio values must be non-negative, got {split_ratio}")
        ratio_sum = sum(split_ratio)
        if not (0.99 <= ratio_sum <= 1.01):
            raise ValueError(f"split_ratio must sum to 1.0 (got {ratio_sum})")
        self.split_ratio = [r / ratio_sum for r in split_ratio]
        # Cumulative thresholds for deterministic split assignment per sample
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

        # Capture bounding boxes after document-level geometric effects are
        # applied (perspective, etc.); template-level pixel effects follow later.
        text_bboxes = []
        image_width, image_height = size
        for i, text_layer in enumerate(text_layers):
            # Get the bounding box as [x1, y1, x2, y2] normalized coordinates
            x1 = _clamp01(text_layer.left / image_width)
            y1 = _clamp01(text_layer.top / image_height)
            x2 = _clamp01((text_layer.left + text_layer.width) / image_width)
            y2 = _clamp01((text_layer.top + text_layer.height) / image_height)

            # Round to 3 decimal places
            bbox = [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]
            text_bboxes.append(bbox)

        # Optionally capture quad (4-corner polygon) coordinates for each line
        text_quads = []
        if self.emit_quads:
            for text_layer in text_layers:
                # .quad is [TL, TR, BR, BL] — 4x2 numpy array (already post-transform)
                quad = text_layer.quad
                normalized = [
                    [round(_clamp01(float(pt[0]) / image_width), 3), round(_clamp01(float(pt[1]) / image_height), 3)]
                    for pt in quad
                ]
                text_quads.append(normalized)

        # Build block-level bboxes from line bboxes
        text_blocks = _build_text_blocks(block_ids, text_bboxes)

        # Compute word bboxes by always deriving from quad interpolation,
        # so that after perspective the AABB tightly encloses the actual word.
        text_words = []
        word_global_id = 0
        for line_idx, (text_layer, word_local_data) in enumerate(zip(text_layers, words_per_line)):
            line_quad = text_layer.quad  # [TL, TR, BR, BL] — always available
            tl, tr, br, bl = line_quad[0], line_quad[1], line_quad[2], line_quad[3]

            for word in word_local_data:
                r1, r2 = word["x1_ratio"], word["x2_ratio"]
                # Interpolate along top edge (TL→TR) and bottom edge (BL→BR)
                w_tl = tl + r1 * (tr - tl)
                w_tr = tl + r2 * (tr - tl)
                w_br = bl + r2 * (br - bl)
                w_bl = bl + r1 * (br - bl)

                # Derive AABB as bounding box of the four quad corners
                xs = [float(w_tl[0]), float(w_tr[0]), float(w_br[0]), float(w_bl[0])]
                ys = [float(w_tl[1]), float(w_tr[1]), float(w_br[1]), float(w_bl[1])]
                wx1 = round(_clamp01(min(xs) / image_width), 3)
                wy1 = round(_clamp01(min(ys) / image_height), 3)
                wx2 = round(_clamp01(max(xs) / image_width), 3)
                wy2 = round(_clamp01(max(ys) / image_height), 3)

                word_entry = {
                    "text": word["text"],
                    "bbox": [wx1, wy1, wx2, wy2],
                    "word_id": word_global_id,
                    "line_id": line_idx,
                }
                if self.emit_quads:
                    word_entry["quad"] = [
                        [
                            round(_clamp01(float(w_tl[0]) / image_width), 3),
                            round(_clamp01(float(w_tl[1]) / image_height), 3),
                        ],
                        [
                            round(_clamp01(float(w_tr[0]) / image_width), 3),
                            round(_clamp01(float(w_tr[1]) / image_height), 3),
                        ],
                        [
                            round(_clamp01(float(w_br[0]) / image_width), 3),
                            round(_clamp01(float(w_br[1]) / image_height), 3),
                        ],
                        [
                            round(_clamp01(float(w_bl[0]) / image_width), 3),
                            round(_clamp01(float(w_bl[1]) / image_height), 3),
                        ],
                    ]
                text_words.append(word_entry)
                word_global_id += 1

        # --- Degenerate bbox filtering ---
        # Identify lines whose bbox area (in pixels) is below the threshold.
        # Filtering affects annotations only; the rendered image is unchanged.
        degenerate_mask = [_bbox_area_px(bbox, image_width, image_height) < self.min_bbox_area for bbox in text_bboxes]
        degenerate_line_count = sum(degenerate_mask)
        degenerate_word_count = (
            sum(1 for w in text_words if degenerate_mask[w["line_id"]]) if degenerate_line_count else 0
        )

        if degenerate_line_count:
            # Build old→new line index mapping for surviving lines
            survive = [i for i, degen in enumerate(degenerate_mask) if not degen]
            old_to_new = {old: new for new, old in enumerate(survive)}

            text_layers = [text_layers[i] for i in survive]
            texts = [texts[i] for i in survive]
            text_bboxes = [text_bboxes[i] for i in survive]
            block_ids = [block_ids[i] for i in survive]
            words_per_line = [words_per_line[i] for i in survive]
            if text_quads:
                text_quads = [text_quads[i] for i in survive]

            # Rebuild text_words with remapped line_id and word_id
            new_words = []
            new_word_id = 0
            for w in text_words:
                old_line = w["line_id"]
                if old_line not in old_to_new:
                    continue
                w["line_id"] = old_to_new[old_line]
                w["word_id"] = new_word_id
                new_words.append(w)
                new_word_id += 1
            text_words = new_words

            # Rebuild text_blocks from surviving lines
            text_blocks = _build_text_blocks(block_ids, text_bboxes)

        layer = layers.Group([*document_group.layers, bg_layer]).merge()
        self.effect.apply([layer])

        image = layer.output(bbox=[0, 0, *size])

        # --- RMS contrast per surviving line bbox ---
        gray = 0.2989 * image[..., 0] + 0.5870 * image[..., 1] + 0.1140 * image[..., 2]
        line_contrasts = []
        line_bbox_areas_px = []
        for bbox in text_bboxes:
            x1_px = int(round(bbox[0] * image_width))
            y1_px = int(round(bbox[1] * image_height))
            x2_px = int(round(bbox[2] * image_width))
            y2_px = int(round(bbox[3] * image_height))
            if x2_px <= x1_px or y2_px <= y1_px:
                continue
            region = gray[y1_px:y2_px, x1_px:x2_px]
            if region.size == 0:
                continue
            line_contrasts.append(float(np.std(region)))
            line_bbox_areas_px.append((x2_px - x1_px) * (y2_px - y1_px))

        word_bbox_areas_px = []
        for w in text_words:
            wb = w["bbox"]
            wx = int(round(wb[2] * image_width)) - int(round(wb[0] * image_width))
            wy = int(round(wb[3] * image_height)) - int(round(wb[1] * image_height))
            if wx > 0 and wy > 0:
                word_bbox_areas_px.append(wx * wy)

        quality_metrics = {
            "min_line_contrast": round(min(line_contrasts), 3) if line_contrasts else None,
            "mean_line_contrast": round(float(np.mean(line_contrasts)), 3) if line_contrasts else None,
            "min_line_bbox_area_px": int(min(line_bbox_areas_px)) if line_bbox_areas_px else None,
            "min_word_bbox_area_px": int(min(word_bbox_areas_px)) if word_bbox_areas_px else None,
            "degenerate_line_count": int(degenerate_line_count),
            "degenerate_word_count": int(degenerate_word_count),
            "textbox_null_count": int(textbox_null_count),
            "textbox_total_count": int(textbox_total_count),
            "image_size": [int(image_width), int(image_height)],
        }

        label = " ".join(texts)
        label = label.strip()
        label = re.sub(r"\s+", " ", label)
        quality = np.random.randint(self.quality[0], self.quality[1] + 1)

        data = {
            "image": image,
            "label": label,
            "quality": quality,
            "roi": roi,
            "text_lines": texts,
            "text_bboxes": text_bboxes,
            "block_ids": block_ids,
            "text_blocks": text_blocks,
            "text_words": text_words,
            "quality_metrics": quality_metrics,
        }

        if self.emit_quads:
            data["text_quads"] = text_quads

        return data

    def init_save(self, root):
        os.makedirs(root, exist_ok=True)

    def save(self, root, data, idx):
        text_lines = data.get("text_lines")
        if not text_lines:
            return

        image = data["image"]
        quality = data["quality"]
        text_bboxes = data.get("text_bboxes", [])
        block_ids = data.get("block_ids", [])
        text_blocks = data.get("text_blocks", [])
        text_words = data.get("text_words", [])
        text_quads = data.get("text_quads", [])
        quality_metrics = data.get("quality_metrics", {})

        # Deterministic split: seed a local RNG per sample so the assignment
        # is independent of generation order and worker count.
        split_idx = int(np.searchsorted(self._split_thresholds, np.random.default_rng(idx).random()))
        output_dirpath = os.path.join(root, self.splits[split_idx])

        # save image
        image_filename = f"image_{idx}.jpg"
        image_filepath = os.path.join(output_dirpath, image_filename)
        os.makedirs(os.path.dirname(image_filepath), exist_ok=True)
        image = Image.fromarray(np.clip(image[..., :3], 0, 255).astype(np.uint8))
        image.save(image_filepath, quality=quality)

        # save metadata (gt_json)
        metadata_filename = "metadata.jsonl"
        metadata_filepath = os.path.join(output_dirpath, metadata_filename)
        os.makedirs(os.path.dirname(metadata_filepath), exist_ok=True)

        # Create structured data for text lines with bboxes and block_id
        text_lines_data = []
        for i, (text, bbox) in enumerate(zip(text_lines, text_bboxes)):
            entry = {"text": text, "bbox": bbox, "line_id": i}
            if i < len(block_ids):
                entry["block_id"] = block_ids[i]
            if i < len(text_quads):
                entry["quad"] = text_quads[i]
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
