"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import re

import numpy as np
from synthtiger import layers


def _extract_word_ratios(chars: list[str], char_layers: list, line_width: float) -> list[dict]:
    """Compute per-word x-ratio dicts from character layers.

    Args:
        chars: List of character strings matching char_layers.
        char_layers: List of rendered character layers with .left/.right attributes.
        line_width: The width (in pixels) to use as the ratio denominator.  Pass
            ``text_layer.size[0]`` (the integer merged-layer canvas width) so that
            the denominator matches the span used during quad interpolation.
    """
    inv_w = 1.0 / line_width if line_width > 0 else None
    words = []
    word_chars: list[str] = []
    x1 = x2 = 0.0

    for ch, layer in zip(chars, char_layers):
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
                x1 = layer.left
            x2 = layer.right
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
    """
    Generates a single line of text rendered as an image layer.

    The TextBox handles character-by-character rendering with proper spacing,
    ensuring words are not split across lines (coherent text generation).

    Attributes:
        fill: Tuple of [min, max] fill ratios controlling how much of the
              available width the text should occupy.

    Example:
        >>> textbox = TextBox({"fill": [0.8, 1.0]})
        >>> layer, text = textbox.generate((400, 50), corpus_reader, font_config)
    """

    def __init__(self, config):
        """
        Initialize a TextBox with the given configuration.

        Args:
            config: Dictionary with optional keys:
                - fill: [min, max] fill ratio range (default: [1, 1])
        """
        self.fill = config.get("fill", [1, 1])

    def generate(self, size, text, font):
        """
        Generate a text layer for a single line.

        Args:
            size: Tuple of (width, height) for the text box area
            text: Text iterator/reader that provides characters
            font: Font configuration dictionary with keys like 'path', 'size', etc.

        Returns:
            Tuple of (text_layer, text_string, word_local_data) where:
                - text_layer: A merged synthtiger Layer containing the rendered text
                - text_string: The actual text that was rendered
                - word_local_data: Per-word x-ratio dicts from character layers
            Returns (None, None, None) if no valid text could be generated.
        """
        width, height = size

        char_layers, chars = [], []
        fill = np.random.uniform(self.fill[0], self.fill[1])
        width = np.clip(width * fill, height, width)
        font = {**font, "size": int(height)}
        left, top = 0, 0

        for char in text:
            if char in "\r\n":
                continue

            char_layer = layers.TextLayer(char, **font)
            char_scale = height / char_layer.height if char_layer.height > 0 else 1.0
            char_layer.bbox = [left, top, *(char_layer.size * char_scale)]
            if char_layer.right > width:
                text.prev()  # undo consumption of the character that didn't fit
                break

            char_layers.append(char_layer)
            chars.append(char)
            left = char_layer.right

        # Find the last space boundary so we don't split a word across cells.
        # If no space exists, accept the truncated token as-is: the cursor is
        # already at the overflow char (put back by the fill loop above), so
        # the next textbox starts correctly without any extra repositioning.
        last_space = next((i for i in range(len(chars) - 1, -1, -1) if chars[i].isspace()), None)

        if last_space is not None:
            # Put back everything from the space onward (inclusive) and trim.
            n_restore = len(chars) - last_space
            for _ in range(n_restore):
                text.prev()
            chars = chars[:last_space]
            char_layers = char_layers[:last_space]
        # else: no space — keep chars as-is, cursor already at overflow char.

        text = "".join(chars).strip()
        text_alpha_only = re.sub(r"[^\w]", "", text)
        if len(char_layers) == 0 or len(text) == 0 or len(text_alpha_only) == 0:
            return None, None, None

        text_layer = layers.Group(char_layers).merge()

        word_local_data = _extract_word_ratios(chars, char_layers, line_width=text_layer.size[0])

        return text_layer, text, word_local_data
