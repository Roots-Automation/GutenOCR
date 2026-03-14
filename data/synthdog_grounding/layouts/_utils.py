"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

from __future__ import annotations

import numpy as np


def sample_fill(fill_range: list[float], full_prob: float) -> float:
    """Sample a fill ratio, with a chance of forcing full fill."""
    full = np.random.rand() < full_prob
    fill = np.random.uniform(fill_range[0], fill_range[1])
    return 1.0 if full else fill
