"""Column layout for index_definition generator."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ColumnRect:
    col_idx: int
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


class ColumnCursor:
    def __init__(self, col: ColumnRect):
        self._col = col
        self._y = col.y1

    @property
    def current_y(self) -> int:
        return self._y

    @property
    def remaining_height(self) -> int:
        return self._col.y2 - self._y

    @property
    def is_full(self) -> bool:
        return self._y >= self._col.y2

    def advance(self, height: int) -> None:
        self._y += height


def sample_layout(
    page_width: int,
    page_height: int,
    config: dict,
    rng: np.random.Generator,
) -> tuple[list[ColumnRect], tuple[int, int, int, int]]:
    """Sample column rects for this page. Returns (columns, content_bbox_px)."""
    layout_cfg = config.get("layout", {})
    margin_range = layout_cfg.get("margin", [0.02, 0.08])
    ml = int(page_width * rng.uniform(*margin_range))
    mr = int(page_width * rng.uniform(*margin_range))
    mt = int(page_height * rng.uniform(*margin_range))
    mb = int(page_height * rng.uniform(*margin_range))

    cx1, cy1 = ml, mt
    cx2, cy2 = page_width - mr, page_height - mb

    choices = layout_cfg.get("num_cols_choices", list(range(1, 9)))
    weights = layout_cfg.get("num_cols_weights", [2, 5, 8, 6, 3, 2, 1, 1])
    weights_arr = np.array(weights, dtype=float)
    weights_arr /= weights_arr.sum()
    n_cols = int(choices[int(rng.choice(len(choices), p=weights_arr))])

    content_width = max(1, cx2 - cx1)
    gutter_frac = layout_cfg.get("gutter_frac", [0.01, 0.04])
    gutter_px = int(content_width * rng.uniform(*gutter_frac)) if n_cols > 1 else 0

    min_col_w = layout_cfg.get("min_col_width_px", 80)
    while n_cols > 1:
        col_w = (content_width - gutter_px * (n_cols - 1)) // n_cols
        if col_w >= min_col_w:
            break
        n_cols -= 1
    col_w = max(min_col_w, (content_width - gutter_px * (n_cols - 1)) // n_cols)

    columns = []
    for i in range(n_cols):
        x1 = cx1 + i * (col_w + gutter_px)
        x2 = x1 + col_w
        columns.append(ColumnRect(col_idx=i, x1=x1, y1=cy1, x2=x2, y2=cy2))

    return columns, (cx1, cy1, cx2, cy2)
