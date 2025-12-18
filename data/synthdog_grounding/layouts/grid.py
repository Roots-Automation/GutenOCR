"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import numpy as np


class Grid:
    """
    Generates a grid-based layout for text placement.
    
    The Grid layout divides a bounding box into rows and columns,
    providing positions for text boxes with configurable spacing,
    fill ratios, and text alignment.
    
    Attributes:
        text_scale: [min, max] range for text size relative to box dimensions
        max_row: Maximum number of rows in the grid
        max_col: Maximum number of columns in the grid
        fill: [min, max] range for horizontal fill ratio
        full: Probability of using full fill (fill=1)
        align: List of valid alignment options ("left", "right", "center")
    
    Example:
        >>> grid = Grid({"max_row": 10, "max_col": 2})
        >>> layout = grid.generate([0, 0, 800, 600])
        >>> for bbox, align in layout:
        ...     print(f"Box at {bbox} with {align} alignment")
    """
    
    def __init__(self, config):
        """
        Initialize a Grid layout with the given configuration.
        
        Args:
            config: Dictionary with optional keys:
                - text_scale: [min, max] text scale range (default: [0.05, 0.1])
                - max_row: Maximum rows (default: 5)
                - max_col: Maximum columns (default: 3)
                - fill: [min, max] fill ratio range (default: [0, 1])
                - full: Probability of full fill (default: 0)
                - align: List of alignments (default: ["left", "right", "center"])
        """
        self.text_scale = config.get("text_scale", [0.05, 0.1])
        self.max_row = config.get("max_row", 5)
        self.max_col = config.get("max_col", 3)
        self.fill = config.get("fill", [0, 1])
        self.full = config.get("full", 0)
        self.align = config.get("align", ["left", "right", "center"])

    def generate(self, bbox):
        """
        Generate a grid layout within the given bounding box.
        
        Args:
            bbox: List of [left, top, width, height] defining the area
        
        Returns:
            List of (bbox, align) tuples where:
                - bbox: [x, y, width, height] for each text cell
                - align: Text alignment ("left", "right", or "center")
            Returns None if no valid grid could be generated.
        """
        left, top, width, height = bbox

        text_scale = np.random.uniform(self.text_scale[0], self.text_scale[1])
        text_size = min(width, height) * text_scale
        grids = np.random.permutation(self.max_row * self.max_col)

        for grid in grids:
            row = grid // self.max_col + 1
            col = grid % self.max_col + 1
            if text_size * (col * 2 - 1) <= width and text_size * row <= height:
                break
        else:
            return None

        bound = max(1 - text_size / width * (col - 1), 0)
        full = np.random.rand() < self.full
        fill = np.random.uniform(self.fill[0], self.fill[1])
        fill = 1 if full else fill
        fill = np.clip(fill, 0, bound)

        padding = np.random.randint(4) if col > 1 else np.random.randint(1, 4)
        padding = (bool(padding // 2), bool(padding % 2))

        weights = np.zeros(col * 2 + 1)
        weights[1:-1] = text_size / width
        probs = 1 - np.random.rand(col * 2 + 1)
        probs[0] = 0 if not padding[0] else probs[0]
        probs[-1] = 0 if not padding[-1] else probs[-1]
        probs[1::2] *= max(fill - sum(weights[1::2]), 0) / sum(probs[1::2])
        probs[::2] *= max(1 - fill - sum(weights[::2]), 0) / sum(probs[::2])
        weights += probs

        widths = [width * weights[c] for c in range(col * 2 + 1)]
        heights = [text_size for _ in range(row)]

        xs = np.cumsum([0] + widths)
        ys = np.cumsum([0] + heights)

        layout = []

        for c in range(col):
            align = self.align[np.random.randint(len(self.align))]

            for r in range(row):
                x, y = xs[c * 2 + 1], ys[r]
                w, h = xs[c * 2 + 2] - x, ys[r + 1] - y
                bbox = [left + x, top + y, w, h]
                layout.append((bbox, align))

        return layout
