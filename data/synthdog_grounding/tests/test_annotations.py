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


# ---------------------------------------------------------------------------
# Adversarial edge cases
# ---------------------------------------------------------------------------


def test_filter_degenerate_all_lines_degenerate_returns_empty():
    """When every line is degenerate the function must return empty lists, not crash."""
    lines = [
        _make_line(0, 0, [0.0, 0.0, 0.0, 0.0]),
        _make_line(1, 1, [0.0, 0.0, 0.0, 0.0]),
    ]
    words = [
        _make_word(0, 0, [0.0, 0.0, 0.1, 0.1]),
        _make_word(1, 1, [0.0, 0.0, 0.1, 0.1]),
    ]
    out_lines, out_words, dl, dw = filter_degenerate(lines, words, min_area=1.0, w=100, h=100)
    assert out_lines == []
    assert out_words == []
    assert dl == 2
    assert dw == 2


def test_filter_degenerate_block_ids_unchanged_on_surviving_lines():
    """filter_degenerate only reassigns line_id; block_id must stay intact.

    Incorrect implementation could remap block_id along with line_id, breaking
    the block grouping step in build_annotations.
    """
    lines = [
        _make_line(0, 7, [0.0, 0.0, 0.0, 0.0]),  # degenerate, block 7
        _make_line(1, 3, [0.0, 0.0, 0.5, 0.5]),  # survives, block 3
    ]
    words = [_make_word(0, 1, [0.0, 0.0, 0.3, 0.3])]
    out_lines, _, _, _ = filter_degenerate(lines, words, min_area=1.0, w=100, h=100)
    assert len(out_lines) == 1
    assert out_lines[0].block_id == 3  # unchanged, NOT remapped to 0
    assert out_lines[0].line_id == 0  # line_id IS remapped


def test_filter_degenerate_surviving_words_ids_start_from_zero():
    """Word IDs must be dense starting from 0 after filtering."""
    lines = [
        _make_line(0, 0, [0.0, 0.0, 0.0, 0.0]),  # degenerate
        _make_line(1, 0, [0.0, 0.0, 0.5, 0.5]),  # survives
    ]
    words = [
        _make_word(0, 0, [0.0, 0.0, 0.1, 0.1]),  # dropped
        _make_word(1, 1, [0.0, 0.0, 0.3, 0.3]),  # new id 0
        _make_word(2, 1, [0.3, 0.0, 0.5, 0.3]),  # new id 1
    ]
    _, out_words, _, _ = filter_degenerate(lines, words, min_area=1.0, w=100, h=100)
    assert [w.word_id for w in out_words] == [0, 1]


def test_build_block_annotations_empty_input_returns_empty_list():
    """Empty block_ids / line_bboxes must produce an empty block list, not crash."""
    blocks = build_block_annotations([], [])
    assert blocks == []


def test_build_block_annotations_non_contiguous_block_ids():
    """Block IDs need not be 0-based; each unique id becomes exactly one block."""
    blocks = build_block_annotations(
        [5, 5, 12],
        [
            [0.0, 0.0, 0.3, 0.3],
            [0.1, 0.1, 0.4, 0.4],
            [0.6, 0.6, 0.9, 0.9],
        ],
    )
    block_map = {b.block_id: b for b in blocks}
    assert set(block_map) == {5, 12}
    assert sorted(block_map[5].line_ids) == [0, 1]
    assert block_map[12].line_ids == [2]


def test_compute_quality_metrics_no_lines_has_none_contrast():
    """When there are no surviving lines, min_line_contrast_ratio must be None — not a crash."""
    image = np.full((50, 100, 4), 200, dtype=np.uint8)
    metrics = compute_quality_metrics(image, [], [], w=100, h=50, deg_lines=0, deg_words=0, null_ct=0, total_ct=0)
    assert metrics["min_line_contrast"] is None
    assert metrics["min_line_contrast_ratio"] is None
    assert metrics["min_line_height_px"] is None
    assert metrics["line_count"] == 0
    assert metrics["word_count"] == 0


def test_build_word_annotations_empty_words_for_one_line():
    """A line with zero words must produce no WordAnnotations — not crash."""
    layer = _axis_layer(0, 0, 100, 50)
    words_per_line = [
        [],  # first line: no words
        [{"text": "hello", "x1_ratio": 0.0, "x2_ratio": 1.0}],  # second line: one word
    ]
    words = build_word_annotations([layer, layer], words_per_line, w=100, h=50, emit_quads=False)
    assert len(words) == 1
    assert words[0].text == "hello"
    assert words[0].line_id == 1
    assert words[0].word_id == 0


def test_build_word_annotations_tilted_quad_word_bbox_valid():
    """Word bboxes from a rotated quad must still satisfy x1<=x2 and y1<=y2.

    A parallelogram quad (sheared horizontally) produces word corner points
    whose x and y values may be out of sorted order — the AABB min/max in
    build_word_annotations must handle this correctly.
    """
    # Tilted quad: tl=(10,0), tr=(110,0), br=(120,50), bl=(20,50)
    layer = _mock_layer([[10, 0], [110, 0], [120, 50], [20, 50]])
    words_per_line = [[{"text": "word", "x1_ratio": 0.0, "x2_ratio": 1.0}]]
    words = build_word_annotations([layer], words_per_line, w=200, h=100, emit_quads=False)
    assert len(words) == 1
    wx1, wy1, wx2, wy2 = words[0].bbox
    assert wx1 <= wx2, f"word bbox x-inverted: x1={wx1} > x2={wx2}"
    assert wy1 <= wy2, f"word bbox y-inverted: y1={wy1} > y2={wy2}"


def test_build_word_annotations_partial_span_tilted_quad_bbox_valid():
    """A word at x1_ratio=0.25, x2_ratio=0.75 on a tilted quad must still
    produce a valid (non-inverted) bbox.
    """
    # Strongly tilted: tl=(0,0) tr=(100,50) br=(100,100) bl=(0,50)
    layer = _mock_layer([[0, 0], [100, 50], [100, 100], [0, 50]])
    words_per_line = [[{"text": "mid", "x1_ratio": 0.25, "x2_ratio": 0.75}]]
    words = build_word_annotations([layer], words_per_line, w=200, h=200, emit_quads=False)
    wx1, wy1, wx2, wy2 = words[0].bbox
    assert wx1 <= wx2, f"x-inverted: {wx1} > {wx2}"
    assert wy1 <= wy2, f"y-inverted: {wy1} > {wy2}"


def test_compute_quality_metrics_zero_pixel_bbox_region_skipped():
    """A line bbox that collapses to 0 pixels at the image resolution must be
    silently skipped — the function must not crash and min_line_contrast must
    be None (no valid line region measured).

    bbox [0.49, 0.0, 0.51, 1.0] on w=10: x1_px=round(4.9)=5, x2_px=round(5.1)=5
    → x2_px <= x1_px → region skipped → line_contrasts stays empty → None.
    """
    image = np.full((100, 10, 4), 200, dtype=np.uint8)
    lines = [_make_line(0, 0, [0.49, 0.0, 0.51, 1.0])]
    metrics = compute_quality_metrics(image, lines, [], w=10, h=100, deg_lines=0, deg_words=0, null_ct=0, total_ct=1)
    assert metrics["min_line_contrast"] is None, "zero-pixel bbox should be skipped, leaving min_line_contrast as None"


def test_laplacian_variance_3x3_returns_float():
    """A 3×3 array is the minimum valid input for _laplacian_variance — it must
    return a float, not crash or produce NaN.

    With a uniform 3×3 array the Laplacian is identically 0 everywhere, so
    the variance must be 0.0.
    """
    gray = np.full((3, 3), 128.0, dtype=np.float32)
    result = _laplacian_variance(gray)
    assert isinstance(result, float)
    assert result == pytest.approx(0.0)


def test_contrast_ratio_equal_zero_luminances_is_one():
    """Two identical black surfaces (luminance=0) must yield contrast ratio 1.0,
    not a ZeroDivisionError.  Formula: (0+0.05)/(0+0.05) = 1.0.
    """
    assert _contrast_ratio(0.0, 0.0) == pytest.approx(1.0)


def test_filter_degenerate_mixed_lines_exact_counts():
    """With 3 lines (degen, survive, degen) and 4 words, filter_degenerate must
    reassign IDs densely and return exactly the right degenerate counts.

    Expected: 1 surviving line (new line_id=0, block_id preserved), 2 surviving
    words (new word_ids=0,1, line_id=0), deg_line_ct=2, deg_word_ct=2.
    """
    lines = [
        _make_line(0, 10, [0.0, 0.0, 0.0, 0.0]),  # degen (block 10)
        _make_line(1, 20, [0.1, 0.1, 0.9, 0.5]),  # survives (block 20)
        _make_line(2, 30, [0.0, 0.0, 0.0, 0.0]),  # degen (block 30)
    ]
    words = [
        _make_word(0, 0, [0.0, 0.0, 0.1, 0.1]),  # degen line → dropped
        _make_word(1, 1, [0.1, 0.1, 0.5, 0.5]),  # survives → new id 0
        _make_word(2, 1, [0.5, 0.1, 0.9, 0.5]),  # survives → new id 1
        _make_word(3, 2, [0.0, 0.0, 0.1, 0.1]),  # degen line → dropped
    ]
    out_lines, out_words, dl, dw = filter_degenerate(lines, words, min_area=1.0, w=100, h=100)

    assert dl == 2
    assert dw == 2
    assert len(out_lines) == 1
    assert out_lines[0].line_id == 0
    assert out_lines[0].block_id == 20  # block_id must NOT be remapped
    assert len(out_words) == 2
    assert [w.word_id for w in out_words] == [0, 1]
    assert all(w.line_id == 0 for w in out_words)


def test_compute_quality_metrics_single_line_overlap_is_zero():
    """A single line has no pairs to compare; both overlap metrics must be 0.0."""
    image = np.full((100, 100, 4), 200, dtype=np.uint8)
    lines = [_make_line(0, 0, [0.1, 0.1, 0.9, 0.5])]
    metrics = compute_quality_metrics(image, lines, [], w=100, h=100, deg_lines=0, deg_words=0, null_ct=0, total_ct=1)
    assert metrics["max_intra_block_line_overlap"] == 0.0
    assert metrics["max_cross_block_line_overlap"] == 0.0


# ---------------------------------------------------------------------------
# Block quad emission
# ---------------------------------------------------------------------------


def test_build_block_annotations_no_quads_when_line_quads_none():
    """When line_quads is None (emit_quads=False) block.quad must be None."""
    blocks = build_block_annotations(
        [0],
        [[0.1, 0.1, 0.5, 0.5]],
        line_quads=None,
    )
    assert blocks[0].quad is None


def test_build_block_annotations_quad_is_4_corners_when_emit():
    """When line_quads provided, block.quad must be [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]."""
    line_quads = [[[0.1, 0.1], [0.5, 0.1], [0.5, 0.3], [0.1, 0.3]]]
    blocks = build_block_annotations(
        [0],
        [[0.1, 0.1, 0.5, 0.3]],
        line_quads=line_quads,
    )
    q = blocks[0].quad
    assert q is not None
    assert len(q) == 4
    # Rectangular: TL→TR→BR→BL order
    assert q[0][0] == pytest.approx(0.1, abs=1e-3)  # TL x
    assert q[0][1] == pytest.approx(0.1, abs=1e-3)  # TL y
    assert q[2][0] == pytest.approx(0.5, abs=1e-3)  # BR x
    assert q[2][1] == pytest.approx(0.3, abs=1e-3)  # BR y


def test_build_block_annotations_quad_spans_all_line_quads():
    """Block quad must be the union AABB of all constituent line quads."""
    line_quads = [
        [[0.1, 0.1], [0.4, 0.1], [0.4, 0.3], [0.1, 0.3]],
        [[0.2, 0.35], [0.7, 0.35], [0.7, 0.5], [0.2, 0.5]],
    ]
    blocks = build_block_annotations(
        [0, 0],
        [[0.1, 0.1, 0.4, 0.3], [0.2, 0.35, 0.7, 0.5]],
        line_quads=line_quads,
    )
    q = blocks[0].quad
    assert q is not None
    assert q[0][0] == pytest.approx(0.1, abs=1e-3)  # min x across all corners
    assert q[0][1] == pytest.approx(0.1, abs=1e-3)  # min y
    assert q[2][0] == pytest.approx(0.7, abs=1e-3)  # max x
    assert q[2][1] == pytest.approx(0.5, abs=1e-3)  # max y


def test_build_block_annotations_quad_clamped_to_0_1():
    """Block quad corners must be clamped to [0, 1] even when line quads overflow."""
    line_quads = [[[-0.5, -0.1], [1.5, -0.1], [1.5, 1.2], [-0.5, 1.2]]]
    blocks = build_block_annotations(
        [0],
        [[0.0, 0.0, 1.0, 1.0]],
        line_quads=line_quads,
    )
    q = blocks[0].quad
    assert q is not None
    for pt in q:
        assert 0.0 <= pt[0] <= 1.0, f"x={pt[0]} out of [0,1]"
        assert 0.0 <= pt[1] <= 1.0, f"y={pt[1]} out of [0,1]"


def test_block_annotation_to_dict_includes_quad_when_present():
    """block_annotation_to_dict must include 'quad' key when block.quad is set."""
    from serialization import BlockAnnotation, block_annotation_to_dict

    blk = BlockAnnotation(
        block_id=0,
        bbox=[0.1, 0.1, 0.5, 0.5],
        line_ids=[0],
        region_type="body",
        quad=[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
    )
    d = block_annotation_to_dict(blk)
    assert "quad" in d
    assert len(d["quad"]) == 4


def test_block_annotation_to_dict_omits_quad_when_none():
    """block_annotation_to_dict must NOT include 'quad' key when block.quad is None."""
    from serialization import BlockAnnotation, block_annotation_to_dict

    blk = BlockAnnotation(block_id=0, bbox=[0.1, 0.1, 0.5, 0.5], line_ids=[0])
    d = block_annotation_to_dict(blk)
    assert "quad" not in d
