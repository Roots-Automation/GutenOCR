"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

from __future__ import annotations

import numpy as np

from ._utils import LayoutCell, sample_fill


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
        >>> for bbox, align, col_idx in layout:
        ...     print(f"Box at {bbox} with {align} alignment in column {col_idx}")
    """

    def __init__(self, config: dict[str, object]) -> None:
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

    def generate(
        self,
        bbox: list[float],
        *,
        fill_range: tuple[float, float] | None = None,
        text_scale_range: tuple[float, float] | None = None,
    ) -> list[LayoutCell] | None:
        """
        Generate a grid layout within the given bounding box.

        Args:
            bbox: List of [left, top, width, height] defining the area
            fill_range: Optional (min, max) fill ratio override (defaults to self.fill).
                Note: When used internally by GridStack, these are always provided,
                overriding the instance defaults.
            text_scale_range: Optional (min, max) text scale override (defaults to self.text_scale).
                Note: When used internally by GridStack, these are always provided,
                overriding the instance defaults.

        Returns:
            List of LayoutCell(bbox, align, col_idx) where:
                - bbox: [x, y, width, height] for each text cell
                - align: Text alignment ("left", "right", or "center")
                - col_idx: Zero-based column index of the cell within this grid
            Returns None if no valid grid could be generated.
        """
        left, top, width, height = bbox

        if width <= 0 or height <= 0:
            return None

        if text_scale_range is None:
            text_scale_range = self.text_scale
        if fill_range is None:
            fill_range = self.fill

        dims = self._pick_grid_dimensions(width, height, text_scale_range)
        if dims is None:
            return None
        row, col, text_size = dims

        widths, heights = self._compute_column_widths(col, row, width, text_size, fill_range)

        return self._emit_cells(left, top, col, row, widths, heights)

    def _pick_grid_dimensions(
        self,
        width: float,
        height: float,
        text_scale_range: tuple[float, float] | list[float],
    ) -> tuple[int, int, float] | None:
        """Sample a text size and find a (row, col) pair that fits.

        Returns (row, col, text_size) or None if no combination fits.
        """
        text_scale = np.random.uniform(text_scale_range[0], text_scale_range[1])
        text_size = min(width, height) * text_scale
        cell_indices = np.random.permutation(self.max_row * self.max_col)

        for cell_idx in cell_indices:
            row = int(cell_idx // self.max_col + 1)
            col = int(cell_idx % self.max_col + 1)
            # Each of `col` columns needs `text_size` width, and the (col-1)
            # gaps between them each need another `text_size`, giving
            # text_size * (col * 2 - 1) total. Rows just stack vertically.
            if text_size * (col * 2 - 1) <= width and text_size * row <= height:
                return row, col, text_size

        return None

    def _compute_column_widths(
        self,
        col: int,
        row: int,
        width: float,
        text_size: float,
        fill_range: tuple[float, float] | list[float],
    ) -> tuple[list[float], list[float]]:
        """Compute per-slot widths and row heights for the grid.

        The width array uses an interleaved slot model:
            [pad_L, text_0, gap_01, text_1, ..., text_{n-1}, pad_R]
        Odd indices (1, 3, ...) are text columns; even indices (0, 2, ...)
        are gaps or edge padding. The method distributes a random fill budget
        across text slots and the remaining space across gap slots.

        Returns (widths, heights) where widths has (col * 2 + 1) entries and
        heights has `row` entries.
        """
        max_fill = max(1 - text_size / width * (col - 1), 0)
        fill = sample_fill(fill_range, self.full)
        fill = np.clip(fill, 0, max_fill)

        # 2-bit encoding of (left_pad, right_pad): bit1=left, bit0=right.
        # Single column excludes 0 to guarantee at least one side has padding.
        pad_bits = np.random.randint(4) if col > 1 else np.random.randint(1, 4)
        has_left_pad = bool(pad_bits // 2)
        has_right_pad = bool(pad_bits % 2)

        # Base weights: each text column gets a minimum of text_size/width.
        weights = np.zeros(col * 2 + 1)
        weights[1:-1] = text_size / width

        # Random proportions for distributing extra space.
        probs = 1 - np.random.rand(col * 2 + 1)
        probs[0] = 0 if not has_left_pad else probs[0]
        probs[-1] = 0 if not has_right_pad else probs[-1]

        # Distribute fill budget across text (odd) slots.
        odd_prob_sum = sum(probs[1::2])
        if odd_prob_sum > 0:
            probs[1::2] *= max(fill - sum(weights[1::2]), 0) / odd_prob_sum

        # Distribute gap budget across gap/padding (even) slots.
        even_prob_sum = sum(probs[::2])
        if even_prob_sum > 0:
            probs[::2] *= max(1 - fill - sum(weights[::2]), 0) / even_prob_sum

        weights += probs

        widths = [width * weights[c] for c in range(col * 2 + 1)]
        heights = [text_size for _ in range(row)]
        return widths, heights

    def _emit_cells(
        self,
        left: float,
        top: float,
        col: int,
        row: int,
        widths: list[float],
        heights: list[float],
    ) -> list[LayoutCell]:
        """Convert slot widths and row heights into positioned LayoutCells."""
        xs = np.cumsum([0] + widths)
        ys = np.cumsum([0] + heights)

        layout = []
        for c in range(col):
            align = self.align[np.random.randint(len(self.align))]
            for r in range(row):
                x, y = xs[c * 2 + 1], ys[r]
                w, h = xs[c * 2 + 2] - x, ys[r + 1] - y
                cell_bbox = (left + x, top + y, w, h)
                layout.append(LayoutCell(cell_bbox, align, c))

        return layout
