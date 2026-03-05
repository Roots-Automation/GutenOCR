"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import numpy as np
from synthtiger import components, layers


class Paper:
    def __init__(self, config):
        self.image = components.BaseTexture(**config.get("image", {}))
        self.color_config = config.get("color", {})

    def generate(self, size):
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

        return paper_layer, (r, g, b)
