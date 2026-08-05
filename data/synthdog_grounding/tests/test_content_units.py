"""Unit tests for content.py free functions and Content._render_zone fast-return paths."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from elements.content import (
    Content,
    _compute_layout_bbox,
    _make_adaptive_color,
    _relative_luminance,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FONT_DIR = str(Path(__file__).resolve().parents[1] / "resources/font/en")
_CORPUS = str(Path(__file__).resolve().parents[1] / "resources/corpus/enwiki.txt")


def _content_cfg():
    return {
        "text": {"path": _CORPUS},
        "font": {"paths": [_FONT_DIR], "weights": [1]},
        "layout": {},
        "textbox": {"fill": [1.0, 1.0]},
        "textbox_color": {"prob": 0},
        "content_color": {"prob": 0},
        "text_sprinkle": {"prob": 0},
    }


# ---------------------------------------------------------------------------
# _relative_luminance — known WCAG values
# ---------------------------------------------------------------------------


def test_relative_luminance_white():
    assert _relative_luminance(255, 255, 255) == pytest.approx(1.0, rel=1e-4)


def test_relative_luminance_black():
    assert _relative_luminance(0, 0, 0) == pytest.approx(0.0, abs=1e-9)


def test_relative_luminance_pure_red():
    # R coefficient is 0.2126; fully-saturated red → lum ≈ 0.2126
    lum = _relative_luminance(255, 0, 0)
    assert 0.21 < lum < 0.22


def test_relative_luminance_monotone_with_brightness():
    l1 = _relative_luminance(64, 64, 64)
    l2 = _relative_luminance(128, 128, 128)
    l3 = _relative_luminance(192, 192, 192)
    assert l1 < l2 < l3


# ---------------------------------------------------------------------------
# _make_adaptive_color — WCAG crossover at 0.179
# ---------------------------------------------------------------------------


def test_make_adaptive_color_dark_bg_forces_prob_one():
    """lum < 0.179 → dark background → force light text (prob=1.0)."""
    sw = _make_adaptive_color({"prob": 0.0, "args": {}}, [191, 255], lum=0.05)
    assert sw.prob == pytest.approx(1.0)


def test_make_adaptive_color_light_bg_uses_config_prob():
    """lum > 0.179 → light background → use the configured prob unchanged."""
    sw = _make_adaptive_color({"prob": 0.3, "args": {}}, [0, 64], lum=0.5)
    assert sw.prob == pytest.approx(0.3)


def test_make_adaptive_color_crossover_boundary_is_exclusive():
    """lum == 0.179 is the light-bg side (condition is <, not <=)."""
    sw = _make_adaptive_color({"prob": 0.0, "args": {}}, [0, 64], lum=0.179)
    assert sw.prob == pytest.approx(0.0)


def test_make_adaptive_color_just_below_crossover():
    sw = _make_adaptive_color({"prob": 0.0, "args": {}}, [191, 255], lum=0.178)
    assert sw.prob == pytest.approx(1.0)


def test_make_adaptive_color_missing_args_key():
    """config without 'args' key must not raise."""
    sw = _make_adaptive_color({"prob": 0.5}, [0, 64], lum=0.5)
    assert sw.prob == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _compute_layout_bbox — margin bounds
# ---------------------------------------------------------------------------


def test_compute_layout_bbox_zero_margin_full_extent():
    """margin=[0,0] → all four random values are 0 → full width and height returned."""
    result = _compute_layout_bbox(200, 150, [0, 0])
    assert result[2] == pytest.approx(200.0)
    assert result[3] == pytest.approx(150.0)


def test_compute_layout_bbox_non_negative_dimensions():
    """Width and height must never be negative regardless of margin."""
    np.random.seed(0)
    result = _compute_layout_bbox(100, 100, [0.0, 0.6])
    assert result[2] >= 0
    assert result[3] >= 0


def test_compute_layout_bbox_returns_four_values():
    np.random.seed(1)
    left, top, w, h = _compute_layout_bbox(400, 300, [0.0, 0.1])
    assert left >= 0 and top >= 0 and w >= 0 and h >= 0


# ---------------------------------------------------------------------------
# Content._render_zone — zero-area fast-return (null count fix)
# ---------------------------------------------------------------------------


def _zone_call(content, zone_bbox, next_block_id=7, region_type="header"):
    text_layers, texts, block_ids, wpl, font_info = [], [], [], [], []
    return content._render_zone(
        cfg={},
        zone_bbox=zone_bbox,
        region_type=region_type,
        canvas_ref=500.0,
        next_block_id=next_block_id,
        block_region_types={},
        text_layers=text_layers,
        texts=texts,
        block_ids=block_ids,
        words_per_line=wpl,
        line_font_info=font_info,
    )


def test_render_zone_zero_width_returns_zero_counts():
    """Zero-width zone → (next_block_id, 0, 0); no null inflation."""
    np.random.seed(0)
    content = Content(_content_cfg())
    nid, null_ct, total_ct = _zone_call(content, [0.0, 0.0, 0.0, 50.0], next_block_id=7)
    assert nid == 7
    assert null_ct == 0
    assert total_ct == 0


def test_render_zone_zero_height_returns_zero_counts():
    np.random.seed(0)
    content = Content(_content_cfg())
    nid, null_ct, total_ct = _zone_call(content, [0.0, 0.0, 200.0, 0.0], next_block_id=3)
    assert nid == 3
    assert null_ct == 0
    assert total_ct == 0


def test_render_zone_zero_area_does_not_advance_block_id():
    """next_block_id must be returned unchanged on a zero-area zone."""
    np.random.seed(0)
    content = Content(_content_cfg())
    nid, _, _ = _zone_call(content, [0.0, 0.0, 0.0, 0.0], next_block_id=42)
    assert nid == 42


def test_render_zone_grid_failure_returns_zero_counts():
    """When Grid.generate() returns None, counts must be (0, 0), not (0, 1).

    The (0, 1) bug would inflate textbox_total_count without a matching null,
    deflating textbox_null_frac and making the page look healthier than it is.
    """
    np.random.seed(0)
    content = Content(_content_cfg())
    with patch("elements.content.Grid.generate", return_value=None):
        nid, null_ct, total_ct = _zone_call(content, [0.0, 0.0, 200.0, 50.0], next_block_id=5)
    assert nid == 5
    assert null_ct == 0
    assert total_ct == 0


# ---------------------------------------------------------------------------
# Content.generate() — integration smoke tests
# ---------------------------------------------------------------------------


def test_content_generate_returns_ten_tuple():
    np.random.seed(0)
    content = Content(_content_cfg())
    result = content.generate((800, 600))
    assert len(result) == 10


def test_content_generate_parallel_lists_same_length():
    """text_layers, texts, block_ids, words_per_line must all have equal length."""
    np.random.seed(1)
    content = Content(_content_cfg())
    text_layers, texts, block_ids, words_per_line, *_ = content.generate((800, 600))
    assert len(text_layers) == len(texts) == len(block_ids) == len(words_per_line)


def test_content_generate_null_accounting():
    """null_count + len(text_layers) must equal total_count exactly."""
    np.random.seed(2)
    content = Content(_content_cfg())
    text_layers, _, _, _, _, null_ct, total_ct, *_ = content.generate((800, 600))
    assert null_ct + len(text_layers) == total_ct


def test_content_generate_block_ids_have_region_types():
    """Every block_id in block_ids must appear in block_region_types."""
    np.random.seed(3)
    content = Content(_content_cfg())
    _, _, block_ids, _, block_region_types, _, _, *_ = content.generate((800, 600))
    for bid in block_ids:
        assert bid in block_region_types


def test_content_generate_word_ratios_in_bounds():
    np.random.seed(4)
    content = Content(_content_cfg())
    _, _, _, words_per_line, *_ = content.generate((800, 600))
    for line_words in words_per_line:
        for w in line_words:
            assert 0.0 <= w["x1_ratio"] <= w["x2_ratio"] <= 1.0, w["text"]


def test_content_generate_dark_background():
    """Dark background (lum < 0.179) must run without error."""
    np.random.seed(5)
    content = Content(_content_cfg())
    result = content.generate((800, 600), bg_color=(10, 10, 10))
    text_layers = result[0]
    assert isinstance(text_layers, list)
