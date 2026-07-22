"""Tests for annotations.py free functions."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotations import (
    _bbox_area_px,
    _clamp01,
    _contrast_ratio,
    _laplacian_variance,
    _linearize_channel,
    _norm,
    _norm_pt,
    build_block_annotations,
    build_word_annotations,
    capture_line_bboxes,
    capture_line_quads,
    compute_quality_metrics,
    filter_degenerate,
)
from serialization import LineAnnotation, WordAnnotation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_line(line_id, block_id, bbox, text="hello world"):
    return LineAnnotation(text=text, bbox=bbox, block_id=block_id, line_id=line_id, quad=None)


def _make_word(word_id, line_id, bbox, text="hello"):
    return WordAnnotation(text=text, bbox=bbox, line_id=line_id, word_id=word_id, quad=None)


def _mock_layer(quad):
    """MagicMock text layer with numpy quad corners."""
    layer = MagicMock()
    layer.quad = [np.array(pt, dtype=float) for pt in quad]
    return layer


def _axis_layer(x1, y1, x2, y2):
    """Axis-aligned quad as a mock text layer."""
    return _mock_layer([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


# ---------------------------------------------------------------------------
# _clamp01
# ---------------------------------------------------------------------------


def test_clamp01_mid_value_unchanged():
    assert _clamp01(0.5) == 0.5


def test_clamp01_below_zero_returns_zero():
    assert _clamp01(-1.0) == 0.0


def test_clamp01_above_one_returns_one():
    assert _clamp01(2.0) == 1.0


def test_clamp01_boundaries_exact():
    assert _clamp01(0.0) == 0.0
    assert _clamp01(1.0) == 1.0


# ---------------------------------------------------------------------------
# _linearize_channel
# ---------------------------------------------------------------------------


def test_linearize_channel_zero_is_zero():
    assert _linearize_channel(0.0) == pytest.approx(0.0)


def test_linearize_channel_255_is_one():
    assert _linearize_channel(255.0) == pytest.approx(1.0, rel=1e-4)


def test_linearize_channel_below_threshold_uses_linear():
    # 9/255 ≈ 0.0353 < 0.03928 → linear branch c/12.92
    v = 9.0
    assert _linearize_channel(v) == pytest.approx(v / 255.0 / 12.92)


def test_linearize_channel_above_threshold_uses_gamma():
    v = 128.0
    c = v / 255.0
    expected = ((c + 0.055) / 1.055) ** 2.4
    assert _linearize_channel(v) == pytest.approx(expected)


def test_linearize_channel_monotone():
    results = [_linearize_channel(float(v)) for v in range(0, 256, 16)]
    assert all(a <= b for a, b in zip(results, results[1:]))


# ---------------------------------------------------------------------------
# _contrast_ratio
# ---------------------------------------------------------------------------


def test_contrast_ratio_white_vs_black_is_21():
    assert _contrast_ratio(1.0, 0.0) == pytest.approx(21.0)


def test_contrast_ratio_equal_luminances_is_one():
    assert _contrast_ratio(0.5, 0.5) == pytest.approx(1.0)


def test_contrast_ratio_order_independent():
    assert _contrast_ratio(0.2, 0.8) == pytest.approx(_contrast_ratio(0.8, 0.2))


# ---------------------------------------------------------------------------
# _norm / _norm_pt / _bbox_area_px
# ---------------------------------------------------------------------------


def test_norm_half_dimension():
    assert _norm(50.0, 100) == pytest.approx(0.5)


def test_norm_clamps_above_one():
    assert _norm(200.0, 100) == 1.0


def test_norm_rounds_to_three_decimal_places():
    # 1/3 rounds to 0.333
    assert _norm(1.0, 3) == pytest.approx(0.333, abs=5e-4)


def test_norm_pt_returns_two_element_list():
    assert _norm_pt(50.0, 25.0, 100, 50) == [pytest.approx(0.5), pytest.approx(0.5)]


def test_bbox_area_px_known_value():
    # dx=0.5 of 100px, dy=0.5 of 100px → 50*50=2500
    assert _bbox_area_px([0.1, 0.2, 0.6, 0.7], 100, 100) == pytest.approx(2500.0)


# ---------------------------------------------------------------------------
# build_block_annotations
# ---------------------------------------------------------------------------


def test_build_block_annotations_hull_over_two_lines():
    blocks = build_block_annotations(
        [0, 0],
        [[0.1, 0.1, 0.4, 0.3], [0.2, 0.2, 0.6, 0.5]],
    )
    assert len(blocks) == 1
    b = blocks[0]
    assert b.bbox == pytest.approx([0.1, 0.1, 0.6, 0.5], abs=1e-3)


def test_build_block_annotations_two_separate_blocks():
    blocks = build_block_annotations(
        [0, 1],
        [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]],
    )
    assert len(blocks) == 2


def test_build_block_annotations_default_region_type_is_body():
    blocks = build_block_annotations([0], [[0.0, 0.0, 0.5, 0.5]])
    assert blocks[0].region_type == "body"


def test_build_block_annotations_custom_region_type():
    blocks = build_block_annotations([0], [[0.0, 0.0, 0.5, 0.5]], block_region_types={0: "header"})
    assert blocks[0].region_type == "header"


def test_build_block_annotations_line_ids_grouped_correctly():
    # Lines 0,1 → block 1; line 2 → block 0
    blocks = build_block_annotations(
        [1, 1, 0],
        [[0.0, 0.0, 0.1, 0.1], [0.1, 0.0, 0.2, 0.1], [0.2, 0.0, 0.3, 0.1]],
    )
    block_map = {b.block_id: b for b in blocks}
    assert sorted(block_map[1].line_ids) == [0, 1]
    assert block_map[0].line_ids == [2]


# ---------------------------------------------------------------------------
# capture_line_bboxes
# ---------------------------------------------------------------------------


def test_capture_line_bboxes_axis_aligned():
    layer = _mock_layer([[0, 0], [100, 0], [100, 50], [0, 50]])
    bboxes = capture_line_bboxes([layer], w=100, h=50)
    assert bboxes[0] == pytest.approx([0.0, 0.0, 1.0, 1.0], abs=1e-3)


def test_capture_line_bboxes_diamond_quad_uses_full_aabb():
    # Diamond corners: top=(50,0), right=(100,25), bottom=(50,50), left=(0,25)
    layer = _mock_layer([[50, 0], [100, 25], [50, 50], [0, 25]])
    bboxes = capture_line_bboxes([layer], w=100, h=50)
    b = bboxes[0]
    assert b[0] == pytest.approx(0.0)
    assert b[2] == pytest.approx(1.0)
    assert b[1] == pytest.approx(0.0)
    assert b[3] == pytest.approx(1.0)


def test_capture_line_bboxes_multiple_layers_count():
    layers = [_axis_layer(0, 0, 50, 10), _axis_layer(0, 20, 50, 30)]
    assert len(capture_line_bboxes(layers, w=100, h=50)) == 2


# ---------------------------------------------------------------------------
# capture_line_quads
# ---------------------------------------------------------------------------


def test_capture_line_quads_normalized_corners():
    layer = _mock_layer([[0, 0], [100, 0], [100, 50], [0, 50]])
    quads = capture_line_quads([layer], w=100, h=50)
    assert len(quads) == 1
    assert quads[0][0] == pytest.approx([0.0, 0.0], abs=1e-3)
    assert quads[0][1] == pytest.approx([1.0, 0.0], abs=1e-3)
    assert quads[0][2] == pytest.approx([1.0, 1.0], abs=1e-3)
    assert quads[0][3] == pytest.approx([0.0, 1.0], abs=1e-3)


# ---------------------------------------------------------------------------
# build_word_annotations
# ---------------------------------------------------------------------------


def test_build_word_full_span_matches_line_bbox():
    layer = _axis_layer(0, 0, 100, 50)
    words = build_word_annotations(
        [layer], [[{"text": "hello", "x1_ratio": 0.0, "x2_ratio": 1.0}]], w=100, h=50, emit_quads=False
    )
    assert words[0].bbox == pytest.approx([0.0, 0.0, 1.0, 1.0], abs=1e-3)


def test_build_word_half_span_ends_at_midpoint():
    layer = _axis_layer(0, 0, 100, 50)
    words = build_word_annotations(
        [layer], [[{"text": "hi", "x1_ratio": 0.0, "x2_ratio": 0.5}]], w=100, h=50, emit_quads=False
    )
    assert words[0].bbox[2] == pytest.approx(0.5, abs=1e-3)


def test_build_word_no_quad_when_emit_false():
    layer = _axis_layer(0, 0, 100, 50)
    words = build_word_annotations(
        [layer], [[{"text": "x", "x1_ratio": 0.0, "x2_ratio": 1.0}]], w=100, h=50, emit_quads=False
    )
    assert words[0].quad is None


def test_build_word_has_quad_when_emit_true():
    layer = _axis_layer(0, 0, 100, 50)
    words = build_word_annotations(
        [layer], [[{"text": "x", "x1_ratio": 0.0, "x2_ratio": 1.0}]], w=100, h=50, emit_quads=True
    )
    assert words[0].quad is not None
    assert len(words[0].quad) == 4


def test_build_word_global_ids_are_sequential_across_lines():
    layer = _axis_layer(0, 0, 100, 50)
    words_per_line = [
        [{"text": "a", "x1_ratio": 0.0, "x2_ratio": 0.5}, {"text": "b", "x1_ratio": 0.5, "x2_ratio": 1.0}],
        [{"text": "c", "x1_ratio": 0.0, "x2_ratio": 1.0}],
    ]
    words = build_word_annotations([layer, layer], words_per_line, w=100, h=50, emit_quads=False)
    assert [w.word_id for w in words] == [0, 1, 2]


def test_build_word_line_ids_follow_source_line():
    layer = _axis_layer(0, 0, 100, 50)
    words_per_line = [
        [{"text": "a", "x1_ratio": 0.0, "x2_ratio": 1.0}],
        [{"text": "b", "x1_ratio": 0.0, "x2_ratio": 1.0}],
    ]
    words = build_word_annotations([layer, layer], words_per_line, w=100, h=50, emit_quads=False)
    assert words[0].line_id == 0
    assert words[1].line_id == 1


# ---------------------------------------------------------------------------
# filter_degenerate
# ---------------------------------------------------------------------------


def test_filter_degenerate_no_degenerates_returns_same_objects():
    lines = [_make_line(0, 0, [0.0, 0.0, 0.5, 0.5])]
    words = [_make_word(0, 0, [0.0, 0.0, 0.5, 0.5])]
    out_lines, out_words, dl, dw = filter_degenerate(lines, words, min_area=1.0, w=100, h=100)
    assert out_lines is lines
    assert out_words is words
    assert dl == 0 and dw == 0


def test_filter_degenerate_removes_zero_area_line():
    lines = [
        _make_line(0, 0, [0.0, 0.0, 0.0, 0.0]),  # zero area → degenerate
        _make_line(1, 0, [0.0, 0.0, 0.5, 0.5]),  # survives
    ]
    words = [
        _make_word(0, 0, [0.0, 0.0, 0.1, 0.1]),  # removed with line 0
        _make_word(1, 1, [0.0, 0.0, 0.5, 0.5]),  # survives
    ]
    out_lines, out_words, dl, dw = filter_degenerate(lines, words, min_area=1.0, w=100, h=100)
    assert len(out_lines) == 1
    assert out_lines[0].line_id == 0  # reassigned from old id 1
    assert dl == 1 and dw == 1


def test_filter_degenerate_line_and_word_ids_reassigned_densely():
    lines = [
        _make_line(0, 0, [0.0, 0.0, 0.0, 0.0]),  # degenerate
        _make_line(1, 0, [0.0, 0.0, 0.5, 0.5]),  # survives → new id 0
    ]
    words = [
        _make_word(0, 0, [0.0, 0.0, 0.1, 0.1]),  # dropped with line 0
        _make_word(1, 1, [0.0, 0.0, 0.3, 0.3]),  # new word_id=0, line_id=0
        _make_word(2, 1, [0.3, 0.0, 0.5, 0.3]),  # new word_id=1, line_id=0
    ]
    _, out_words, _, _ = filter_degenerate(lines, words, min_area=1.0, w=100, h=100)
    assert [w.word_id for w in out_words] == [0, 1]
    assert all(w.line_id == 0 for w in out_words)


def test_filter_degenerate_counts_multiple_degenerate_lines():
    lines = [
        _make_line(0, 0, [0.0, 0.0, 0.0, 0.0]),  # degenerate
        _make_line(1, 0, [0.0, 0.0, 0.0, 0.0]),  # degenerate
        _make_line(2, 0, [0.0, 0.0, 0.5, 0.5]),  # survives
    ]
    words = [
        _make_word(0, 0, [0.0, 0.0, 0.1, 0.1]),
        _make_word(1, 1, [0.0, 0.0, 0.1, 0.1]),
        _make_word(2, 2, [0.0, 0.0, 0.3, 0.3]),
    ]
    _, _, dl, dw = filter_degenerate(lines, words, min_area=1.0, w=100, h=100)
    assert dl == 2 and dw == 2


# ---------------------------------------------------------------------------
# _laplacian_variance
# ---------------------------------------------------------------------------


def test_laplacian_variance_uniform_array_is_zero():
    gray = np.full((10, 10), 128.0, dtype=np.float32)
    assert _laplacian_variance(gray) == pytest.approx(0.0)


def test_laplacian_variance_linear_gradient_is_zero():
    # Second derivative of a linear ramp is zero → Laplacian = 0
    gray = np.tile(np.linspace(0.0, 255.0, 10, dtype=np.float32), (10, 1))
    assert _laplacian_variance(gray) == pytest.approx(0.0, abs=1e-3)


def test_laplacian_variance_checkerboard_is_nonzero():
    i, j = np.meshgrid(range(10), range(10))
    gray = ((i + j) % 2 * 255).astype(np.float32)
    assert _laplacian_variance(gray) > 0.0


def test_laplacian_variance_higher_frequency_gives_higher_value():
    low = np.tile(np.array([100, 105, 100, 105], dtype=np.float32), (6, 2))
    high = np.tile(np.array([0, 255, 0, 255], dtype=np.float32), (6, 2))
    assert _laplacian_variance(high) > _laplacian_variance(low)


# ---------------------------------------------------------------------------
# compute_quality_metrics — smoke / contract tests
# ---------------------------------------------------------------------------


def _uniform_image(v=200, h=50, w=100):
    return np.full((h, w, 4), v, dtype=np.uint8)


def test_compute_quality_metrics_returns_all_expected_keys():
    image = _uniform_image()
    lines = [_make_line(0, 0, [0.1, 0.1, 0.9, 0.9])]
    words = [_make_word(0, 0, [0.1, 0.1, 0.5, 0.9])]
    metrics = compute_quality_metrics(image, lines, words, w=100, h=50, deg_lines=0, deg_words=0, null_ct=0, total_ct=5)
    expected = {
        "min_line_contrast",
        "mean_line_contrast",
        "min_line_contrast_ratio",
        "min_line_bbox_area_px",
        "min_word_bbox_area_px",
        "degenerate_line_count",
        "degenerate_word_count",
        "textbox_null_count",
        "textbox_total_count",
        "image_size",
        "word_segmentation_method",
        "line_count",
        "word_count",
        "textbox_null_frac",
        "min_line_height_px",
        "mean_line_height_px",
        "sharpness",
        "max_intra_block_line_overlap",
        "max_cross_block_line_overlap",
    }
    assert expected <= set(metrics)


def test_compute_quality_metrics_uniform_image_has_zero_sharpness():
    image = _uniform_image()
    lines = [_make_line(0, 0, [0.1, 0.1, 0.9, 0.9])]
    metrics = compute_quality_metrics(image, lines, [], w=100, h=50, deg_lines=0, deg_words=0, null_ct=0, total_ct=1)
    assert metrics["sharpness"] == pytest.approx(0.0)


def test_compute_quality_metrics_null_frac_zero_when_no_textboxes():
    image = _uniform_image()
    metrics = compute_quality_metrics(image, [], [], w=100, h=50, deg_lines=0, deg_words=0, null_ct=0, total_ct=0)
    assert metrics["textbox_null_frac"] == 0.0


def test_compute_quality_metrics_cross_block_overlap_nonzero_for_overlapping_lines():
    # Two overlapping lines in different blocks → cross-block overlap > 0
    image = np.full((100, 100, 4), 200, dtype=np.uint8)
    lines = [
        _make_line(0, 0, [0.0, 0.0, 0.8, 0.5]),
        _make_line(1, 1, [0.1, 0.1, 0.9, 0.6]),
    ]
    metrics = compute_quality_metrics(image, lines, [], w=100, h=100, deg_lines=0, deg_words=0, null_ct=0, total_ct=2)
    assert metrics["max_cross_block_line_overlap"] > 0.0


def test_compute_quality_metrics_intra_block_overlap_nonzero_for_same_block():
    image = np.full((100, 100, 4), 200, dtype=np.uint8)
    lines = [
        _make_line(0, 0, [0.0, 0.0, 0.8, 0.5]),
        _make_line(1, 0, [0.1, 0.1, 0.9, 0.6]),  # same block
    ]
    metrics = compute_quality_metrics(image, lines, [], w=100, h=100, deg_lines=0, deg_words=0, null_ct=0, total_ct=2)
    assert metrics["max_intra_block_line_overlap"] > 0.0
    assert metrics["max_cross_block_line_overlap"] == 0.0
