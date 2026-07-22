"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import re

import numpy as np
from PIL import ImageFont as PILImageFont
from synthtiger import layers


def _extract_word_ratios(
    chars: list[str],
    positions: list[tuple[float, float]],
    line_width: float,
) -> list[dict]:
    """Return per-word x-ratio dicts from per-character (left, right) positions."""
    inv_w = 1.0 / line_width if line_width > 0 else None
    words = []
    word_chars: list[str] = []
    x1 = x2 = 0.0

    for ch, (left, right) in zip(chars, positions):
        if ch.isspace():
            if word_chars:
                words.append(
                    {
                        "text": "".join(word_chars),
                        "x1_ratio": x1 * inv_w if inv_w else 0.0,
                        "x2_ratio": x2 * inv_w if inv_w else 1.0,
                    }
                )
                word_chars.clear()
        else:
            if not word_chars:
                x1 = left
            x2 = right
            word_chars.append(ch)

    if word_chars:
        words.append(
            {
                "text": "".join(word_chars),
                "x1_ratio": x1 * inv_w if inv_w else 0.0,
                "x2_ratio": x2 * inv_w if inv_w else 1.0,
            }
        )

    return words


class TextBox:
    """Renders a line of text from a cursor into a synthtiger Layer with word-level x-ratio annotations."""

    def __init__(self, config):
        self.fill = config.get("fill", [1, 1])

    def generate(self, size, text, font):
        """Fit one line of text into size, returning (layer, text_str, word_ratios) or (None, None, None)."""
        width, height = size

        chars = []
        fill = np.random.uniform(self.fill[0], self.fill[1])
        width = np.clip(width * fill, height, width)
        font = {**font, "size": int(height)}

        font_obj = PILImageFont.truetype(font["path"], size=int(height))

        ascent, descent = font_obj.getmetrics()
        pil_height = ascent + descent
        char_scale = height / pil_height if pil_height > 0 else 1.0

        positions: list[tuple[float, float]] = []
        prefix = ""
        x = 0.0

        for char in text:
            if char in "\r\n":
                continue
            next_prefix = prefix + char
            x_right = font_obj.getlength(next_prefix) * char_scale
            if x_right > width:
                text.prev()
                break
            positions.append((x, x_right))
            chars.append(char)
            prefix = next_prefix
            x = x_right

        last_space = next((i for i in range(len(chars) - 1, -1, -1) if chars[i].isspace()), None)

        if last_space is not None:
            n_restore = len(chars) - last_space
            for _ in range(n_restore):
                text.prev()
            chars = chars[:last_space]
            positions = positions[:last_space]

        text_str = "".join(chars).strip()
        text_alpha_only = re.sub(r"[^\w]", "", text_str)
        if not chars or not text_str or not text_alpha_only:
            return None, None, None

        text_layer = layers.TextLayer(text_str, **font)
        text_layer.bbox = [0, 0, *(text_layer.size * char_scale)]
        # Use advance width, not ink width, so ratios stay in [0, 1].
        line_width = positions[-1][1] if positions else 0.0
        word_local_data = _extract_word_ratios(chars, positions, line_width=line_width)

        return text_layer, text_str, word_local_data
