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


class SynthDoG(templates.Template):
    def __init__(self, config=None, split_ratio: list[float] = None):
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
        self.split_ratio = split_ratio
        # Cumulative thresholds for deterministic split assignment per sample
        self._split_thresholds = np.cumsum(split_ratio)

    def generate(self):
        landscape = np.random.rand() < self.landscape
        short_size = np.random.randint(self.short_size[0], self.short_size[1] + 1)
        aspect_ratio = np.random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
        long_size = int(short_size * aspect_ratio)
        size = (long_size, short_size) if landscape else (short_size, long_size)

        bg_layer = self.background.generate(size)
        paper_layer, text_layers, texts, block_ids, words_per_line = self.document.generate(size)

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
            x1 = text_layer.left / image_width
            y1 = text_layer.top / image_height
            x2 = (text_layer.left + text_layer.width) / image_width
            y2 = (text_layer.top + text_layer.height) / image_height

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
                    [round(float(pt[0]) / image_width, 3), round(float(pt[1]) / image_height, 3)] for pt in quad
                ]
                text_quads.append(normalized)

        # Build block-level bboxes from line bboxes
        block_to_lines = defaultdict(list)
        for i, bid in enumerate(block_ids):
            block_to_lines[bid].append(i)

        text_blocks = []
        for bid, line_indices in sorted(block_to_lines.items()):
            bboxes = [text_bboxes[i] for i in line_indices]
            bx1 = min(b[0] for b in bboxes)
            by1 = min(b[1] for b in bboxes)
            bx2 = max(b[2] for b in bboxes)
            by2 = max(b[3] for b in bboxes)
            text_blocks.append(
                {
                    "block_id": bid,
                    "bbox": [round(bx1, 3), round(by1, 3), round(bx2, 3), round(by2, 3)],
                    "line_ids": line_indices,
                }
            )

        # Compute absolute word bboxes using ratios interpolated into final line bbox
        text_words = []
        word_global_id = 0
        for line_idx, (text_layer, word_local_data) in enumerate(zip(text_layers, words_per_line)):
            lx = text_layer.left
            ly = text_layer.top
            lw = text_layer.width
            lh = text_layer.height

            if self.emit_quads:
                line_quad = text_layer.quad  # [TL, TR, BR, BL]
                tl, tr, br, bl = line_quad[0], line_quad[1], line_quad[2], line_quad[3]

            for word in word_local_data:
                wx1 = round((lx + word["x1_ratio"] * lw) / image_width, 3)
                wy1 = round(ly / image_height, 3)
                wx2 = round((lx + word["x2_ratio"] * lw) / image_width, 3)
                wy2 = round((ly + lh) / image_height, 3)
                word_entry = {
                    "text": word["text"],
                    "bbox": [wx1, wy1, wx2, wy2],
                    "word_id": word_global_id,
                    "line_id": line_idx,
                }
                if self.emit_quads:
                    r1, r2 = word["x1_ratio"], word["x2_ratio"]
                    # Interpolate along top edge (TL→TR) and bottom edge (BL→BR)
                    w_tl = tl + r1 * (tr - tl)
                    w_tr = tl + r2 * (tr - tl)
                    w_br = bl + r2 * (br - bl)
                    w_bl = bl + r1 * (br - bl)
                    word_entry["quad"] = [
                        [round(float(w_tl[0]) / image_width, 3), round(float(w_tl[1]) / image_height, 3)],
                        [round(float(w_tr[0]) / image_width, 3), round(float(w_tr[1]) / image_height, 3)],
                        [round(float(w_br[0]) / image_width, 3), round(float(w_br[1]) / image_height, 3)],
                        [round(float(w_bl[0]) / image_width, 3), round(float(w_bl[1]) / image_height, 3)],
                    ]
                text_words.append(word_entry)
                word_global_id += 1

        layer = layers.Group([*document_group.layers, bg_layer]).merge()
        self.effect.apply([layer])

        image = layer.output(bbox=[0, 0, *size])
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
        }

        if self.emit_quads:
            data["text_quads"] = text_quads

        return data

    def init_save(self, root):
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)

    def save(self, root, data, idx):
        image = data["image"]
        quality = data["quality"]
        text_lines = data.get("text_lines", [])
        text_bboxes = data.get("text_bboxes", [])
        block_ids = data.get("block_ids", [])
        text_blocks = data.get("text_blocks", [])
        text_words = data.get("text_words", [])
        text_quads = data.get("text_quads", [])

        # Deterministic split: seed a local RNG per sample so the assignment
        # is independent of generation order and worker count.
        split_idx = int(np.searchsorted(self._split_thresholds, np.random.default_rng(idx).random()))
        output_dirpath = os.path.join(root, self.splits[split_idx])

        # save image
        image_filename = f"image_{idx}.jpg"
        image_filepath = os.path.join(output_dirpath, image_filename)
        os.makedirs(os.path.dirname(image_filepath), exist_ok=True)
        image = Image.fromarray(image[..., :3].astype(np.uint8))
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

        keys = ["text_lines", "text_bboxes", "text_blocks", "text_words"]
        values = [text_lines_data, text_bboxes, text_blocks, text_words]
        if text_quads:
            keys.append("text_quads")
            values.append(text_quads)

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

    def format_metadata(self, image_filename: str, keys: list[str], values: list[Any]):
        """
        Fit gt_parse contents to huggingface dataset's format
        keys and values, whose lengths are equal, are used to constrcut 'gt_parse' field in 'ground_truth' field
        Args:
            keys: List of task_name
            values: List of actual gt data corresponding to each task_name
        """
        assert len(keys) == len(values), f"Length does not match: keys({len(keys)}), values({len(values)})"

        _gt_parse_v = dict()
        for k, v in zip(keys, values):
            _gt_parse_v[k] = v
        gt_parse = {"gt_parse": _gt_parse_v}
        gt_parse_str = json.dumps(gt_parse, ensure_ascii=False)
        metadata = {"file_name": image_filename, "ground_truth": gt_parse_str}
        return metadata
