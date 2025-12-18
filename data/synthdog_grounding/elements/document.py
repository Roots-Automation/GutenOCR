"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import numpy as np
from synthtiger import components

from elements.content import Content
from elements.paper import Paper


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
        self.effect = components.Iterator(
            [
                components.Switch(components.ElasticDistortion()),
                components.Switch(components.AdditiveGaussianNoise()),
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
            **config.get("effect", {}),
        )

    def generate(self, size):
        """
        Generate a document with paper and text content.
        
        Args:
            size: Tuple of (width, height) for the document canvas
        
        Returns:
            Tuple of (paper_layer, text_layers, texts) where:
                - paper_layer: A Layer containing the paper texture
                - text_layers: List of Layers, one per text line
                - texts: List of strings corresponding to each text layer
        """
        width, height = size
        fullscreen = np.random.rand() < self.fullscreen

        if not fullscreen:
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
            size = (long_size, short_size) if landscape else (short_size, long_size)

        text_layers, texts = self.content.generate(size)
        paper_layer = self.paper.generate(size)
        self.effect.apply([*text_layers, paper_layer])

        return paper_layer, text_layers, texts
