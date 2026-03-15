"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

from __future__ import annotations

import numpy as np

from ._utils import LayoutCell, sample_fill
from .grid import Grid


class GridStack:
    """
    Generates stacked grid layouts for multi-section documents.

    The GridStack creates multiple Grid sections stacked vertically,
    useful for generating documents with multiple paragraphs or
    distinct text regions separated by spacing.

    Failure contract: GridStack.generate() returns an empty list when no
    valid grids fit, while Grid.generate() returns None. The difference is
    intentional — GridStack always returns an iterable of grid layouts
    (possibly empty), so callers can unconditionally iterate over the result.

    Attributes:
        text_scale: [min, max] range for text size relative to box dimensions
        fill: [min, max] range for horizontal fill ratio
        full: Probability of using full fill
        stack_spacing: [min, max] range for spacing between stacked grids
        stack_fill: [min, max] range for vertical fill of the stacked area
        stack_full: Probability of using full vertical fill

    Example:
        >>> stack = GridStack({"stack_spacing": [0.02, 0.05]})
        >>> layouts = stack.generate([0, 0, 800, 600])
        >>> for grid_layout in layouts:
        ...     for bbox, align, col_idx in grid_layout:
        ...         print(f"Box at {bbox} in column {col_idx}")
    """

    def __init__(self, config: dict[str, object], *, grid: Grid | None = None) -> None:
        """
        Initialize a GridStack layout with the given configuration.

        Args:
            config: Dictionary with optional keys:
                - text_scale: [min, max] text scale range (default: [0.05, 0.1])
                - max_row: Maximum rows per grid (default: 5)
                - max_col: Maximum columns per grid (default: 3)
                - fill: [min, max] fill ratio range (default: [0, 1])
                - full: Probability of full fill (default: 0)
                - align: List of alignments (default: ["left", "right", "center"])
                - stack_spacing: [min, max] vertical spacing range (default: [0, 0.05])
                - stack_fill: [min, max] vertical fill range (default: [1, 1])
                - stack_full: Probability of full vertical fill (default: 0)
            grid: Optional pre-configured Grid collaborator. If None, a Grid is
                built from the relevant keys in *config*.
        """
        self.text_scale = config.get("text_scale", [0.05, 0.1])
        self.fill = config.get("fill", [0, 1])
        self.full = config.get("full", 0)
        self.stack_spacing = config.get("stack_spacing", [0, 0.05])
        self.stack_fill = config.get("stack_fill", [1, 1])
        self.stack_full = config.get("stack_full", 0)
        # fill/full intentionally omitted: generate() always overrides them
        # via the fill_range keyword argument to Grid.generate().
        self._grid = (
            grid
            if grid is not None
            else Grid(
                {
                    "text_scale": self.text_scale,
                    "max_row": config.get("max_row", 5),
                    "max_col": config.get("max_col", 3),
                    "align": config.get("align", ["left", "right", "center"]),
                }
            )
        )

    def generate(self, bbox: list[float]) -> list[list[LayoutCell]]:
        """
        Generate stacked grid layouts within the given bounding box.

        Args:
            bbox: List of [left, top, width, height] defining the area.

        Returns:
            List of grid layouts, where each grid layout is a list of
            LayoutCell(bbox, align, col_idx). Returns an empty list if no valid
            grids could be generated.
        """
        left, top, width, height = bbox

        if width <= 0 or height <= 0:
            return []

        stack_spacing = np.random.uniform(self.stack_spacing[0], self.stack_spacing[1])
        stack_spacing *= min(width, height)

        stack_fill = sample_fill(self.stack_fill, self.stack_full)
        fill = sample_fill(self.fill, self.full)

        layouts = []
        line = 0

        while True:
            grid_size = (width, height * stack_fill - line)
            if grid_size[1] <= 0:
                break

            text_scale = np.random.uniform(self.text_scale[0], self.text_scale[1])
            text_size = min(width, height) * text_scale
            text_scale = text_size / min(grid_size)

            layout = self._grid.generate(
                [left, top + line, *grid_size],
                fill_range=(fill, fill),
                text_scale_range=(text_scale, text_scale),
            )
            if layout is None:
                break

            line = max(y + h - top for (_, y, _, h), *_ in layout) + stack_spacing
            layouts.append(layout)

        if not layouts:
            return []

        line = max(line - stack_spacing, 0)
        space = max(height - line, 0)
        spaces = np.random.rand(len(layouts) + 1)
        spaces *= space / sum(spaces) if sum(spaces) > 0 else 0
        spaces = np.cumsum(spaces)

        redistributed = []
        for layout, space in zip(layouts, spaces):
            new_layout = []
            for cell_bbox, align, col_idx in layout:
                x, y, w, h = cell_bbox
                new_layout.append(LayoutCell([x, y + space, w, h], align, col_idx))
            redistributed.append(new_layout)

        return redistributed
