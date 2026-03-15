"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import numpy as np


class LayoutCell(NamedTuple):
    """A single cell in a grid layout."""

    bbox: list[float]
    align: str
    col_idx: int


class Layout(Protocol):
    """Protocol for top-level layout generators consumed by content.py."""

    def generate(self, bbox: list[float]) -> list[list[LayoutCell]]: ...


def sample_fill(fill_range: tuple[float, float], full_prob: float) -> float:
    """Sample a fill ratio, with a chance of forcing full fill."""
    full = np.random.rand() < full_prob
    fill = np.random.uniform(fill_range[0], fill_range[1])
    return 1.0 if full else fill
