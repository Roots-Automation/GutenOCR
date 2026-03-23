"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import numpy as np
from synthtiger import components

from .content import Content
from .paper import Paper


class Document:
    """
    Generates a complete document with paper texture and text content.

    The Document class orchestrates the creation of synthetic document images
    by combining a paper background with text content, applying visual effects
    like perspective transforms, elastic distortion, and noise.

    Attributes:
        fullscreen: Probability that the document fills the entire canvas
        landscape: Probability of landscape orientation
        short_size: [min, max] range for the shorter dimension in pixels
        aspect_ratio: [min, max] range for the aspect ratio
        paper: Paper instance for generating paper texture
        content: Content instance for generating text
        effect: Iterator of visual effects to apply

    Example:
        >>> doc = Document({"fullscreen": 0.3, "landscape": 0.5})
        >>> paper_layer, text_layers, texts = doc.generate((800, 600))
    """

    def __init__(self, config):
        """
        Initialize a Document with the given configuration.

        Args:
            config: Dictionary with optional keys:
                - fullscreen: Probability of fullscreen mode (default: 0.5)
                - landscape: Probability of landscape orientation (default: 0.5)
                - short_size: [min, max] short dimension range (default: [480, 1024])
                - aspect_ratio: [min, max] aspect ratio range (default: [1, 2])
                - paper: Paper configuration dict
                - content: Content configuration dict
                - effect: Effect configuration dict
        """
        self.fullscreen = config.get("fullscreen", 0.5)
        self.landscape = config.get("landscape", 0.5)
        self.short_size = config.get("short_size", [480, 1024])
        self.aspect_ratio = config.get("aspect_ratio", [1, 2])
        self.paper = Paper(config.get("paper", {}))
        self.content = Content(config.get("content", {}))

        # Separate elastic distortion from the per-layer effect pipeline.
        # Elastic distortion warps pixels but cannot update layer quads, so
        # applying it per-layer before annotation capture produces misaligned
        # bboxes.  Instead, expose it as a standalone component for the caller
        # to apply to the composited image *after* annotations are captured.
        effect_config = config.get("effect", {})
        effect_args = effect_config.get("args", [{}, {}, {}])
        elastic_config = effect_args[0] if len(effect_args) > 0 else {}
        remaining_args = effect_args[1:] if len(effect_args) > 1 else []

        self.elastic_distortion = components.Switch(components.ElasticDistortion(), **elastic_config)

        self.effect = components.Iterator(
            [
                components.Switch(components.AdditiveGaussianNoise()),
                components.Switch(components.Erode()),
                components.Switch(components.Dilate()),
                components.Switch(components.CoarseDropout()),
                components.Switch(
                    components.Selector(
                        [
                            components.Perspective(),
                            components.Perspective(),
                            components.Perspective(),
                            components.Perspective(),
                            components.Perspective(),
                            components.Perspective(),
                            components.Perspective(),
                            components.Perspective(),
                        ]
                    )
                ),
            ],
            args=remaining_args if remaining_args else None,
        )

    def close(self):
        self.content.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _compute_document_size(self, size: tuple[int, int]) -> tuple[int, int]:
        """Optionally shrink *size* based on fullscreen, landscape, and aspect-ratio config."""
        width, height = size
        if np.random.rand() < self.fullscreen:
            return size

        landscape = np.random.rand() < self.landscape
        max_size = width if landscape else height
        short_size = np.random.randint(
            min(width, height, self.short_size[0]),
            min(width, height, self.short_size[1]) + 1,
        )
        aspect_ratio = np.random.uniform(
            min(max_size / short_size, self.aspect_ratio[0]),
            min(max_size / short_size, self.aspect_ratio[1]),
        )
        long_size = int(short_size * aspect_ratio)
        return (long_size, short_size) if landscape else (short_size, long_size)

    def generate(self, size):
        """
        Generate a document with paper and text content.

        Args:
            size: Tuple of (width, height) for the document canvas

        Returns:
            Tuple of (paper_layer, text_layers, texts, block_ids,
            words_per_line, textbox_null_count, textbox_total_count) where:
                - paper_layer: A Layer containing the paper texture
                - text_layers: List of Layers, one per text line
                - texts: List of strings corresponding to each text layer
                - block_ids: List of int block IDs, one per text line
                - words_per_line: List of word-detail dicts per line
                - textbox_null_count: Number of textbox slots that produced no text
                - textbox_total_count: Total textbox slots attempted
        """
        size = self._compute_document_size(size)
        paper_layer, bg_color = self.paper.generate(size)
        text_layers, texts, block_ids, words_per_line, textbox_null_count, textbox_total_count = self.content.generate(
            size, bg_color
        )
        self.effect.apply([*text_layers, paper_layer])

        return paper_layer, text_layers, texts, block_ids, words_per_line, textbox_null_count, textbox_total_count
