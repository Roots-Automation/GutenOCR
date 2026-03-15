"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import numpy as np
from synthtiger import components

from layouts import GridStack, Layout

from .readers import _READER_TYPES, TextCursor
from .textbox import TextBox


def _relative_luminance(r, g, b):
    def channel(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _make_adaptive_color(color_config: dict, gray_range: list[int], lum: float) -> components.Switch:
    """Build a Switch(Gray) component whose prob is forced to 1.0 on dark backgrounds."""
    args = {**color_config.get("args", {}), "gray": gray_range}
    prob = color_config.get("prob", 0)
    if lum < 0.5:
        prob = 1.0
    return components.Switch(components.Gray(), prob=prob, args=args)


def _compute_layout_bbox(width: int, height: int, margin: list[float]) -> list[float]:
    """Sample 4 independent margins and return [left, top, w, h] for the content area."""
    layout_left = width * np.random.uniform(margin[0], margin[1])
    layout_right = width * np.random.uniform(margin[0], margin[1])
    layout_top = height * np.random.uniform(margin[0], margin[1])
    layout_bottom = height * np.random.uniform(margin[0], margin[1])
    layout_width = max(width - layout_left - layout_right, 0)
    layout_height = max(height - layout_top - layout_bottom, 0)
    return [layout_left, layout_top, layout_width, layout_height]


_LAYOUT_TYPES: dict[str, type] = {
    "grid_stack": GridStack,
}


class Content:
    def __init__(self, config):
        self.margin = config.get("margin", [0, 0.1])

        # Choose text reader based on configuration
        text_config = config.get("text", {})
        reader_type = text_config.get("type", "file")
        reader_cls = _READER_TYPES[reader_type]  # KeyError = clear signal of bad config
        reader_kwargs = {k: v for k, v in text_config.items() if k != "type"}
        self.reader: TextCursor = reader_cls(**reader_kwargs)

        self.font = components.BaseFont(**config.get("font", {}))
        layout_config = config.get("layout", {})
        layout_type = layout_config.get("type", "grid_stack")
        layout_cls = _LAYOUT_TYPES[layout_type]
        self.layout: Layout = layout_cls(layout_config)
        self.textbox = TextBox(config.get("textbox", {}))
        self.textbox_color_config = config.get("textbox_color", {})
        self.content_color_config = config.get("content_color", {})

    def generate(self, size, bg_color=(255, 255, 255)):
        width, height = size

        lum = _relative_luminance(*bg_color)
        gray_range = [0, 64] if lum > 0.5 else [191, 255]

        textbox_color = _make_adaptive_color(self.textbox_color_config, gray_range, lum)
        content_color = _make_adaptive_color(self.content_color_config, gray_range, lum)
        layout_bbox = _compute_layout_bbox(width, height, self.margin)

        text_layers, texts, block_ids, words_per_line = [], [], [], []
        textbox_total_count = 0
        textbox_null_count = 0
        layouts = self.layout.generate(layout_bbox)
        self.reader.move(np.random.randint(len(self.reader)))

        # Each (grid_idx, col_idx) pair is a distinct visual block
        col_key_to_block_id = {}
        next_block_id = 0

        for grid_idx, layout in enumerate(layouts):
            font = self.font.sample()

            for bbox, align, col_idx in layout:
                textbox_total_count += 1
                x, y, w, h = bbox
                text_layer, text, word_local_data = self.textbox.generate((w, h), self.reader, font)

                if text_layer is None:
                    textbox_null_count += 1
                    continue

                text_layer.center = (x + w / 2, y + h / 2)
                if align == "left":
                    text_layer.left = x
                if align == "right":
                    text_layer.right = x + w

                col_key = (grid_idx, col_idx)
                if col_key not in col_key_to_block_id:
                    col_key_to_block_id[col_key] = next_block_id
                    next_block_id += 1

                textbox_color.apply([text_layer])
                text_layers.append(text_layer)
                texts.append(text)
                block_ids.append(col_key_to_block_id[col_key])
                words_per_line.append(word_local_data)

        content_color.apply(text_layers)

        return text_layers, texts, block_ids, words_per_line, textbox_null_count, textbox_total_count
