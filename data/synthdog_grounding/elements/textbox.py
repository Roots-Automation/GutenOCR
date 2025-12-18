"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import re 

import numpy as np
from synthtiger import layers


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
            Tuple of (text_layer, text_string) where:
                - text_layer: A merged synthtiger Layer containing the rendered text
                - text_string: The actual text that was rendered
            Returns (None, None) if no valid text could be generated.
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
            char_scale = height / char_layer.height
            char_layer.bbox = [left, top, *(char_layer.size * char_scale)]
            if char_layer.right > width:
                break

            char_layers.append(char_layer)
            chars.append(char)
            left = char_layer.right

        while len(chars) and not chars[-1].isspace():
            chars.pop()
            char_layers.pop()
            text.prev()

        if len(chars):
            text.prev()
            chars.pop()
            char_layers.pop()

        text = "".join(chars).strip()
        text_alpha_only = re.sub(r"[^\w]", "", text)
        if len(char_layers) == 0 or len(text) == 0 or len(text_alpha_only) == 0:
            return None, None

        text_layer = layers.Group(char_layers).merge()

        return text_layer, text
