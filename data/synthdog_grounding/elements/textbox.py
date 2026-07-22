"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import re
import sys
from pathlib import Path

import numpy as np
from synthtiger import layers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pillow_compat import _cached_truetype

_NON_WORD_RE = re.compile(r"[^\w]")


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

    def generate(self, size, cursor, font):
        """Fit one line of text into size, returning (layer, text_str, word_ratios) or (None, None, None)."""
        width, height = size

        chars = []
        fill = np.random.uniform(self.fill[0], self.fill[1])
        width = np.clip(width * fill, height, width)
        font = {**font, "size": int(height)}

        font_obj = _cached_truetype(font["path"], int(height))

        ascent, descent = font_obj.getmetrics()
        pil_height = ascent + descent
        char_scale = height / pil_height if pil_height > 0 else 1.0

        positions: list[tuple[float, float]] = []
        prefix = ""
        x = 0.0

        for char in cursor:
            if char in "\r\n":
                continue
            next_prefix = prefix + char
            x_right = font_obj.getlength(next_prefix) * char_scale
            if x_right > width:
                cursor.prev()
                break
            positions.append((x, x_right))
            chars.append(char)
            prefix = next_prefix
            x = x_right

        last_space = next((i for i in range(len(chars) - 1, -1, -1) if chars[i].isspace()), None)

        if last_space is not None:
            n_restore = len(chars) - last_space
            for _ in range(n_restore):
                cursor.prev()
            chars = chars[:last_space]
            positions = positions[:last_space]

        text_str = "".join(chars).strip()
        text_alpha_only = _NON_WORD_RE.sub("", text_str)
        if not chars or not text_str or not text_alpha_only:
            return None, None, None

        # Strip leading spaces left by the previous call's backtrack; rebase
        # positions so the first visible character starts at x=0. Without this,
        # line_width includes the leading-space advance and x1_ratio for the
        # first word is non-zero even though it visually starts at the left edge.
        lead = next(i for i, ch in enumerate(chars) if not ch.isspace())
        if lead:
            x_off = positions[lead][0]
            chars = chars[lead:]
            positions = [(lo - x_off, ro - x_off) for lo, ro in positions[lead:]]

        text_layer = layers.TextLayer(text_str, **font)
        text_layer.bbox = [0, 0, *(text_layer.size * char_scale)]
        # Use advance width, not ink width, so ratios stay in [0, 1].
        line_width = positions[-1][1] if positions else 0.0
        word_local_data = _extract_word_ratios(chars, positions, line_width=line_width)

        return text_layer, text_str, word_local_data
