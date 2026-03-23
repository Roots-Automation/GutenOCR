"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import numpy as np
from effects.physical import StainOverlayEffect
from synthtiger import components, layers


class Paper:
    def __init__(self, config):
        self.image = components.BaseTexture(**config.get("image", {}))
        self.color_config = config.get("color", {})
        self.stain_cfg = config.get("stain", {})

    def generate(self, size):
        """Generate a paper layer with optional random color and texture.

        Returns the base RGB before texture overlay. Callers should rely on
        min_line_contrast quality metric for post-hoc contrast validation.
        """
        color_prob = self.color_config.get("prob", 0)
        rgb_ranges = self.color_config.get("rgb", [[255, 255], [255, 255], [255, 255]])

        if np.random.rand() < color_prob:
            r = np.random.randint(rgb_ranges[0][0], rgb_ranges[0][1] + 1)
            g = np.random.randint(rgb_ranges[1][0], rgb_ranges[1][1] + 1)
            b = np.random.randint(rgb_ranges[2][0], rgb_ranges[2][1] + 1)
        else:
            r, g, b = 255, 255, 255

        paper_layer = layers.RectLayer(size, (r, g, b, 255))
        self.image.apply([paper_layer])

        if np.random.rand() < self.stain_cfg.get("prob", 0):
            paper_layer.image = StainOverlayEffect.apply(
                np.clip(paper_layer.image, 0, 255).astype(np.uint8),
                self.stain_cfg.get("args", {}),
            ).astype(np.float32)

        return paper_layer, (r, g, b)
