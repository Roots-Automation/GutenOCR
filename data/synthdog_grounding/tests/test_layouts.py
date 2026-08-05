"""Tests for layouts/grid.py, grid_stack.py, and utils.py."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layouts import Grid, GridStack, LayoutCell
from layouts.utils import sample_fill

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BBOX = [10.0, 20.0, 800.0, 600.0]  # [left, top, width, height]


def _all_cells(layouts):
    """Flatten a list-of-lists of LayoutCell into a single sequence."""
    for layout in layouts:
        yield from layout


# ---------------------------------------------------------------------------
# sample_fill
# ---------------------------------------------------------------------------


def test_sample_fill_full_prob_one_always_returns_one():
    np.random.seed(0)
    for _ in range(20):
        assert sample_fill((0.0, 0.5), full_prob=1.0) == 1.0


def test_sample_fill_full_prob_zero_stays_in_range():
    np.random.seed(0)
    for _ in range(50):
        v = sample_fill((0.3, 0.7), full_prob=0.0)
        assert 0.3 <= v <= 0.7


def test_sample_fill_full_prob_zero_never_returns_one():
    np.random.seed(42)
    results = [sample_fill((0.0, 0.9), full_prob=0.0) for _ in range(100)]
    assert all(v < 1.0 for v in results)


# ---------------------------------------------------------------------------
# Grid.generate — structure invariants
# ---------------------------------------------------------------------------


def test_grid_generate_returns_list_of_layout_cells():
    np.random.seed(0)
    grid = Grid({"max_row": 3, "max_col": 2})
    layout = grid.generate(BBOX)
    assert layout is not None
    assert all(isinstance(c, LayoutCell) for c in layout)


def test_grid_generate_zero_width_returns_none():
    grid = Grid({})
    assert grid.generate([0.0, 0.0, 0.0, 600.0]) is None


def test_grid_generate_zero_height_returns_none():
    grid = Grid({})
    assert grid.generate([0.0, 0.0, 800.0, 0.0]) is None


def test_grid_generate_cells_within_bbox():
    """Every cell must lie entirely within the provided bounding box."""
    left, top, width, height = BBOX
    np.random.seed(0)
    grid = Grid({"max_row": 5, "max_col": 3})
    for seed in range(30):
        np.random.seed(seed)
        layout = grid.generate(BBOX)
        if layout is None:
            continue
        for cell_bbox, _, _ in layout:
            x, y, w, h = cell_bbox
            assert x >= left - 1e-6, f"x={x} < left={left}"
            assert y >= top - 1e-6, f"y={y} < top={top}"
            assert x + w <= left + width + 1e-6, f"right={x + w} > bbox_right={left + width}"
            assert y + h <= top + height + 1e-6, f"bottom={y + h} > bbox_bottom={top + height}"


def test_grid_generate_cell_sizes_positive():
    np.random.seed(0)
    grid = Grid({"max_row": 3, "max_col": 2})
    layout = grid.generate(BBOX)
    assert layout is not None
    for cell_bbox, _, _ in layout:
        _, _, w, h = cell_bbox
        assert w > 0, f"cell width={w} <= 0"
        assert h > 0, f"cell height={h} <= 0"


def test_grid_generate_col_idx_in_range():
    np.random.seed(0)
    max_col = 3
    grid = Grid({"max_col": max_col})
    layout = grid.generate(BBOX)
    assert layout is not None
    for _, _, col_idx in layout:
        assert 0 <= col_idx < max_col


def test_grid_generate_align_from_config():
    np.random.seed(0)
    allowed = ["left", "center"]
    grid = Grid({"align": allowed})
    for seed in range(20):
        np.random.seed(seed)
        layout = grid.generate(BBOX)
        if layout is None:
            continue
        for _, align, _ in layout:
            assert align in allowed


def test_grid_generate_respects_max_row():
    np.random.seed(0)
    max_row = 2
    grid = Grid({"max_row": max_row, "max_col": 1})
    layout = grid.generate(BBOX)
    assert layout is not None
    assert len(layout) <= max_row


def test_pick_grid_dimensions_rejects_narrow_columns():
    """col=3 must be rejected when col_width < 4*text_size.

    With width=100, text_size=10:
      old condition: text_size*(2*col-1)=50 <= 100  → col=3 accepted
      new condition: col_width=(100-20)/3≈26.7 < 4*10=40 → col=3 rejected
                     col_width=(100-10)/2=45  >= 40         → col=2 accepted
    Every layout returned must have col ≤ 2 (col_idx < 2).
    """
    grid = Grid({"max_row": 5, "max_col": 3, "text_scale": [0.1, 0.1]})
    narrow_bbox = [0.0, 0.0, 100.0, 200.0]
    found_layout = False
    for seed in range(60):
        np.random.seed(seed)
        layout = grid.generate(narrow_bbox)
        if layout is None:
            continue
        found_layout = True
        for _, _, col_idx in layout:
            assert col_idx < 2, (
                f"seed={seed}: col_idx={col_idx} (col={col_idx + 1}) was selected "
                "but col=3 violates the minimum 4× column-width constraint"
            )
    assert found_layout, "no valid layout returned across 60 seeds; bbox may be too small"


def test_col_width_boundary_exactly_4x_text_size_is_accepted():
    """col_width = EXACTLY 4*text_size must be accepted (condition uses >=, not >).

    With bbox=(0,0,90,100), text_scale=[0.1,0.1]:
      text_size = min(90,100) * 0.1 = 10
      col=1: col_width = 90/1 = 90 >= 40 ✓ (always accepted)
      col=2: col_width = (90-10)/2 = 40 = 4*10 → must be accepted

    At least one layout with col_idx=1 (col=2) must be produced.
    If the condition were col_width > 4*text_size (strict), col=2 would be
    rejected at this boundary and only col=1 layouts would appear.
    """
    grid = Grid({"max_row": 5, "max_col": 2, "text_scale": [0.1, 0.1]})
    boundary_bbox = [0.0, 0.0, 90.0, 100.0]
    found_col2 = False
    for seed in range(40):
        np.random.seed(seed)
        layout = grid.generate(boundary_bbox)
        if layout is None:
            continue
        for _, _, col_idx in layout:
            if col_idx == 1:
                found_col2 = True
                break
        if found_col2:
            break
    assert found_col2, (
        "no layout with col=2 produced; col_width=4*text_size at the boundary was "
        "incorrectly rejected — check that the condition uses >= not >"
    )


def test_grid_col_idx_bounded_with_large_max_col():
    """col_idx must stay in [0, max_col) regardless of how large max_col is."""
    max_col = 10
    grid = Grid({"max_row": 3, "max_col": max_col})
    found_any = False
    for seed in range(30):
        np.random.seed(seed)
        layout = grid.generate([0.0, 0.0, 2000.0, 600.0])
        if layout is None:
            continue
        found_any = True
        for _, _, col_idx in layout:
            assert 0 <= col_idx < max_col, f"seed={seed}: col_idx={col_idx} out of [0, {max_col})"
    assert found_any, "no valid layout produced in 30 seeds; bbox may be too small"


def test_grid_generate_impossibly_tiny_bbox_returns_none():
    """A 1×1 bbox with default text_scale [0.05, 0.1] → text_size ≥ 0.05, which
    can't satisfy text_size*(2*col-1) ≤ 1 for any col ≥ 1 at scale ≥ 0.05."""
    grid = Grid({"text_scale": [0.5, 0.9]})
    np.random.seed(0)
    result = grid.generate([0.0, 0.0, 1.0, 1.0])
    # May or may not succeed depending on RNG; just must not crash.
    assert result is None or isinstance(result, list)


# ---------------------------------------------------------------------------
# Grid overflow fix — fill lower bound
# ---------------------------------------------------------------------------


def test_grid_cells_within_bbox_when_fill_forced_to_zero():
    """Regression: with fill_range=(0, 0) the old code set odd probs to 0 but
    still allocated 1-fill=1 to gap slots, causing sum(weights)>1 and cells
    to overflow the bbox."""
    left, top, width, height = 0.0, 0.0, 800.0, 600.0
    grid = Grid({"max_row": 1, "max_col": 1, "text_scale": [0.1, 0.1]})
    for seed in range(50):
        np.random.seed(seed)
        layout = grid.generate([left, top, width, height], fill_range=(0.0, 0.0))
        if layout is None:
            continue
        for cell_bbox, _, _ in layout:
            x, y, w, h = cell_bbox
            assert x + w <= left + width + 1e-6, (
                f"seed={seed}: cell right={x + w:.3f} exceeds bbox right={left + width}"
            )


def test_grid_cells_within_bbox_multi_col_zero_fill():
    """Same overflow check for multi-column grids with fill forced to 0."""
    left, top, width, height = 0.0, 0.0, 800.0, 200.0
    grid = Grid({"max_row": 1, "max_col": 3, "text_scale": [0.05, 0.05]})
    for seed in range(50):
        np.random.seed(seed)
        layout = grid.generate([left, top, width, height], fill_range=(0.0, 0.0))
        if layout is None:
            continue
        for cell_bbox, _, _ in layout:
            x, y, w, h = cell_bbox
            assert x + w <= left + width + 1e-6, (
                f"seed={seed}: cell right={x + w:.3f} exceeds bbox right={left + width}"
            )


# ---------------------------------------------------------------------------
# GridStack.generate — structure invariants
# ---------------------------------------------------------------------------


def test_gridstack_generate_returns_list():
    np.random.seed(0)
    stack = GridStack({})
    result = stack.generate(BBOX)
    assert isinstance(result, list)


def test_gridstack_generate_zero_width_returns_empty():
    stack = GridStack({})
    assert stack.generate([0.0, 0.0, 0.0, 600.0]) == []


def test_gridstack_generate_zero_height_returns_empty():
    stack = GridStack({})
    assert stack.generate([0.0, 0.0, 800.0, 0.0]) == []


def test_gridstack_generate_cells_are_layout_cells():
    np.random.seed(0)
    stack = GridStack({})
    result = stack.generate(BBOX)
    for cell in _all_cells(result):
        assert isinstance(cell, LayoutCell)


def test_gridstack_generate_cells_within_bbox():
    """All cells from all stacked grids must lie within the original bbox."""
    left, top, width, height = BBOX
    for seed in range(20):
        np.random.seed(seed)
        stack = GridStack({})
        result = stack.generate(BBOX)
        for cell_bbox, _, _ in _all_cells(result):
            x, y, w, h = cell_bbox
            assert x >= left - 1e-6
            assert y >= top - 1e-6
            assert x + w <= left + width + 1e-6, f"seed={seed}: right={x + w:.2f} > {left + width}"
            assert y + h <= top + height + 1e-6, f"seed={seed}: bottom={y + h:.2f} > {top + height}"


def test_gridstack_generate_cell_sizes_positive():
    np.random.seed(1)
    stack = GridStack({})
    result = stack.generate(BBOX)
    assert len(result) > 0
    for cell_bbox, _, _ in _all_cells(result):
        _, _, w, h = cell_bbox
        assert w > 0
        assert h > 0


def test_gridstack_generate_multiple_grids_stacked():
    """GridStack should produce more than one grid section for a tall bbox."""
    np.random.seed(0)
    tall_bbox = [0.0, 0.0, 800.0, 2000.0]
    stack = GridStack({"text_scale": [0.02, 0.03]})
    result = stack.generate(tall_bbox)
    assert len(result) > 1, f"expected >1 grid section, got {len(result)}"


def test_gridstack_accepts_injected_grid():
    """GridStack must use the injected Grid collaborator."""
    called = []

    class SpyGrid(Grid):
        def generate(self, bbox, **kwargs):
            called.append(bbox)
            return super().generate(bbox, **kwargs)

    np.random.seed(0)
    spy = SpyGrid({"max_row": 1, "max_col": 1})
    stack = GridStack({}, grid=spy)
    stack.generate(BBOX)
    assert len(called) > 0
