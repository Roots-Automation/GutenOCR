"""Unit tests for template.py free functions and SynthDoG.save() filter logic."""

import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import free functions directly — no SynthDoG instantiation needed.
from template import (
    _deep_merge,
    _package_data,
    _resolve_config_paths,
    _rotate_point,
    _rotate_quad,
)

# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------


def test_deep_merge_overlay_scalar_replaces_base():
    base = {"a": 1, "b": 2}
    overlay = {"b": 99}
    result = _deep_merge(base, overlay)
    assert result == {"a": 1, "b": 99}


def test_deep_merge_overlay_list_replaces_base_list():
    base = {"x": [1, 2, 3]}
    overlay = {"x": [4, 5]}
    result = _deep_merge(base, overlay)
    assert result["x"] == [4, 5]


def test_deep_merge_nested_dicts_merged_recursively():
    base = {"font": {"size": 12, "bold": False}}
    overlay = {"font": {"bold": True}}
    result = _deep_merge(base, overlay)
    assert result["font"] == {"size": 12, "bold": True}


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"x": 1}}
    overlay = {"a": {"x": 2}}
    _deep_merge(base, overlay)
    assert base["a"]["x"] == 1


def test_deep_merge_does_not_mutate_overlay():
    base = {"a": 1}
    overlay = {"b": 2}
    _deep_merge(base, overlay)
    assert "a" not in overlay


def test_deep_merge_adds_new_keys_from_overlay():
    base = {"a": 1}
    result = _deep_merge(base, {"b": 2})
    assert result["b"] == 2


def test_deep_merge_empty_overlay_returns_copy_of_base():
    base = {"a": {"b": 1}}
    result = _deep_merge(base, {})
    assert result == base
    assert result is not base


def test_deep_merge_overlay_wins_when_types_differ():
    """If base has a dict but overlay has a scalar, overlay wins (no merge)."""
    base = {"a": {"x": 1}}
    overlay = {"a": 42}
    result = _deep_merge(base, overlay)
    assert result["a"] == 42


# ---------------------------------------------------------------------------
# _resolve_config_paths
# ---------------------------------------------------------------------------


def test_resolve_config_paths_relative_path_key():
    base_dir = Path("/abs/base")
    config = {"text": {"path": "corpus/en.txt"}}
    result = _resolve_config_paths(config, base_dir)
    assert result["text"]["path"] == str(Path("/abs/base/corpus/en.txt").resolve())


def test_resolve_config_paths_absolute_path_unchanged():
    base_dir = Path("/abs/base")
    config = {"text": {"path": "/already/absolute.txt"}}
    result = _resolve_config_paths(config, base_dir)
    assert result["text"]["path"] == "/already/absolute.txt"


def test_resolve_config_paths_list_of_paths():
    base_dir = Path("/abs/base")
    config = {"font": {"paths": ["fonts/en", "/abs/fonts/zh"]}}
    result = _resolve_config_paths(config, base_dir)
    assert result["font"]["paths"][0] == str(Path("/abs/base/fonts/en").resolve())
    assert result["font"]["paths"][1] == "/abs/fonts/zh"


def test_resolve_config_paths_font_path_key():
    base_dir = Path("/abs/base")
    config = {"watermark": {"font_path": "resources/font/DejaVu.ttf"}}
    result = _resolve_config_paths(config, base_dir)
    assert result["watermark"]["font_path"] == str(Path("/abs/base/resources/font/DejaVu.ttf").resolve())


def test_resolve_config_paths_does_not_mutate_input():
    base_dir = Path("/abs/base")
    config = {"text": {"path": "corpus/en.txt"}}
    _resolve_config_paths(config, base_dir)
    assert config["text"]["path"] == "corpus/en.txt"


def test_resolve_config_paths_nested():
    base_dir = Path("/root")
    config = {"document": {"content": {"text": {"path": "data/en.txt"}}}}
    result = _resolve_config_paths(config, base_dir)
    assert result["document"]["content"]["text"]["path"] == str(Path("/root/data/en.txt").resolve())


# ---------------------------------------------------------------------------
# _rotate_point / _rotate_quad
# ---------------------------------------------------------------------------


def test_rotate_point_zero_angle_identity():
    x, y = _rotate_point(3.0, 4.0, 0.0, 0.0, 0.0)
    assert x == pytest.approx(3.0)
    assert y == pytest.approx(4.0)


def test_rotate_point_90_degrees():
    # (1, 0) rotated 90° around origin → (0, 1)
    x, y = _rotate_point(1.0, 0.0, 0.0, 0.0, 90.0)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(1.0, abs=1e-9)


def test_rotate_point_180_degrees():
    x, y = _rotate_point(1.0, 0.0, 0.0, 0.0, 180.0)
    assert x == pytest.approx(-1.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)


def test_rotate_point_360_degrees_returns_to_start():
    x, y = _rotate_point(3.0, 7.0, 1.0, 2.0, 360.0)
    assert x == pytest.approx(3.0, abs=1e-9)
    assert y == pytest.approx(7.0, abs=1e-9)


def test_rotate_point_around_non_origin_center():
    # (2, 0) rotated 90° around (1, 0): radius=1, should land at (1, 1)
    x, y = _rotate_point(2.0, 0.0, 1.0, 0.0, 90.0)
    assert x == pytest.approx(1.0, abs=1e-9)
    assert y == pytest.approx(1.0, abs=1e-9)


def test_rotate_quad_preserves_length():
    quad = [[0, 0], [10, 0], [10, 5], [0, 5]]
    result = _rotate_quad(quad, 5.0, 2.5, 45.0)
    assert len(result) == 4
    assert all(len(pt) == 2 for pt in result)


def test_rotate_quad_zero_angle_unchanged():
    quad = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    result = _rotate_quad(quad, 0.0, 0.0, 0.0)
    for orig, rot in zip(quad, result):
        assert rot[0] == pytest.approx(orig[0])
        assert rot[1] == pytest.approx(orig[1])


def test_rotate_quad_preserves_pairwise_distances():
    """Rotation is rigid: pairwise distances between corners must not change."""
    quad = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]]
    rotated = _rotate_quad(quad, 50.0, 25.0, 37.0)

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    for i in range(4):
        j = (i + 1) % 4
        assert dist(quad[i], quad[j]) == pytest.approx(dist(rotated[i], rotated[j]), rel=1e-6)


# ---------------------------------------------------------------------------
# _package_data
# ---------------------------------------------------------------------------


def test_package_data_keys_without_quads():
    data = _package_data(
        image=np.zeros((10, 10, 4), dtype=np.uint8),
        label="hello",
        quality=85,
        roi=np.zeros((4, 2), dtype=int),
        lines=[],
        words=[],
        blocks=[],
        quality_metrics={},
        generation_params={},
        emit_quads=False,
    )
    assert "image" in data
    assert "label" in data
    assert "quality" in data
    assert "roi" in data
    assert "text_quads" not in data
    assert "generation_params" in data


def test_package_data_includes_text_quads_when_emit_quads():
    fake_line = MagicMock()
    fake_line.quad = [[0, 0], [1, 0], [1, 1], [0, 1]]
    data = _package_data(
        image=np.zeros((10, 10, 4), dtype=np.uint8),
        label="hello",
        quality=85,
        roi=np.zeros((4, 2), dtype=int),
        lines=[fake_line],
        words=[],
        blocks=[],
        quality_metrics={},
        generation_params={},
        emit_quads=True,
    )
    assert "text_quads" in data
    assert data["text_quads"] == [fake_line.quad]


# ---------------------------------------------------------------------------
# SynthDoG.save() — quality filter logic (tested via a stub instance)
# ---------------------------------------------------------------------------


def _make_stub_synthdog():
    """Build a minimal SynthDoG-like object with only the save() attributes."""
    stub = MagicMock(
        spec=[
            "min_contrast_ratio",
            "min_word_count",
            "max_textbox_null_frac",
            "min_line_height_px",
            "min_sharpness",
            "max_intra_block_line_overlap",
            "max_cross_block_line_overlap",
            "splits",
            "_split_thresholds",
            "_SAVE_MAX_RETRIES",
            "_quality_failure",
            "generate",
            "save",
            "format_metadata",
        ]
    )
    stub.min_contrast_ratio = 3.0
    stub.min_word_count = 5
    stub.max_textbox_null_frac = 0.5
    stub.min_line_height_px = 8.0
    stub.min_sharpness = 10.0
    stub.max_intra_block_line_overlap = 0.1
    stub.max_cross_block_line_overlap = 0.05
    stub.splits = ["train", "val", "test"]
    stub._split_thresholds = np.array([0.8, 0.9, 1.0])
    stub._SAVE_MAX_RETRIES = 20
    # Bind the real save() and _quality_failure() methods to the stub.
    from template import SynthDoG

    stub._quality_failure = lambda data: SynthDoG._quality_failure(stub, data)
    stub.save = lambda root, data, idx: SynthDoG.save(stub, root, data, idx)
    stub.format_metadata = MagicMock(return_value={})
    # generate() returns no-lines data so retry attempts also fail cleanly.
    _empty = {
        "lines": [],
        "words": [],
        "blocks": [],
        "label": "",
        "quality": 85,
        "image": np.zeros((4, 4, 4), dtype=np.float32),
        "quality_metrics": {"word_count": 0, "textbox_null_frac": 0.0},
    }
    stub.generate = MagicMock(return_value=_empty)
    return stub


def _good_metrics():
    return {
        "min_line_contrast_ratio": 4.5,
        "word_count": 20,
        "textbox_null_frac": 0.1,
        "min_line_height_px": 16.0,
        "sharpness": 50.0,
        "max_intra_block_line_overlap": 0.0,
        "max_cross_block_line_overlap": 0.0,
    }


def _good_data(label="hello world foo bar baz"):
    fake_line = MagicMock()
    fake_line.text = label
    return {
        "label": label,
        "image": np.zeros((10, 10, 4), dtype=np.uint8),
        "quality": 85,
        "lines": [fake_line],
        "words": [],
        "blocks": [],
        "quality_metrics": _good_metrics(),
    }


def test_save_skips_when_no_lines():
    stub = _make_stub_synthdog()
    data = _good_data()
    data["lines"] = []
    with tempfile.TemporaryDirectory() as root:
        stub.save(root, data, 0)
    stub.format_metadata.assert_not_called()


def test_save_skips_low_contrast():
    stub = _make_stub_synthdog()
    data = _good_data()
    data["quality_metrics"]["min_line_contrast_ratio"] = 1.5  # below 3.0
    with tempfile.TemporaryDirectory() as root:
        stub.save(root, data, 0)
    stub.format_metadata.assert_not_called()


def test_save_skips_low_word_count():
    stub = _make_stub_synthdog()
    data = _good_data()
    data["quality_metrics"]["word_count"] = 3  # below 5
    with tempfile.TemporaryDirectory() as root:
        stub.save(root, data, 0)
    stub.format_metadata.assert_not_called()


def test_save_skips_high_null_frac():
    stub = _make_stub_synthdog()
    data = _good_data()
    data["quality_metrics"]["textbox_null_frac"] = 0.8  # above 0.5
    with tempfile.TemporaryDirectory() as root:
        stub.save(root, data, 0)
    stub.format_metadata.assert_not_called()


def test_save_skips_low_line_height():
    stub = _make_stub_synthdog()
    data = _good_data()
    data["quality_metrics"]["min_line_height_px"] = 4.0  # below 8.0
    with tempfile.TemporaryDirectory() as root:
        stub.save(root, data, 0)
    stub.format_metadata.assert_not_called()


def test_save_skips_low_sharpness():
    stub = _make_stub_synthdog()
    data = _good_data()
    data["quality_metrics"]["sharpness"] = 5.0  # below 10.0
    with tempfile.TemporaryDirectory() as root:
        stub.save(root, data, 0)
    stub.format_metadata.assert_not_called()


def test_save_skips_high_intra_overlap():
    stub = _make_stub_synthdog()
    data = _good_data()
    data["quality_metrics"]["max_intra_block_line_overlap"] = 0.2  # above 0.1
    with tempfile.TemporaryDirectory() as root:
        stub.save(root, data, 0)
    stub.format_metadata.assert_not_called()


def test_save_skips_high_cross_overlap():
    stub = _make_stub_synthdog()
    data = _good_data()
    data["quality_metrics"]["max_cross_block_line_overlap"] = 0.1  # above 0.05
    with tempfile.TemporaryDirectory() as root:
        stub.save(root, data, 0)
    stub.format_metadata.assert_not_called()


def test_save_writes_when_all_filters_pass():
    stub = _make_stub_synthdog()
    with tempfile.TemporaryDirectory() as root:
        stub.save(root, _good_data(), 0)
    stub.format_metadata.assert_called_once()


def test_save_contrast_at_exact_threshold_passes():
    """A sample at exactly min_contrast_ratio must pass the filter.

    The condition is ``< self.min_contrast_ratio`` (strict less-than), so a
    value equal to the threshold must NOT be rejected.
    """
    stub = _make_stub_synthdog()
    data = _good_data()
    # Set contrast equal to the configured threshold (3.0).
    data["quality_metrics"]["min_line_contrast_ratio"] = stub.min_contrast_ratio
    with tempfile.TemporaryDirectory() as root:
        stub.save(root, data, 0)
    stub.format_metadata.assert_called_once()


def test_save_null_frac_none_treated_as_zero():
    """If quality_metrics['textbox_null_frac'] is None (no textboxes rendered),
    save() must treat it as 0.0 — not reject the sample or crash."""
    stub = _make_stub_synthdog()
    data = _good_data()
    data["quality_metrics"]["textbox_null_frac"] = None
    with tempfile.TemporaryDirectory() as root:
        stub.save(root, data, 0)
    stub.format_metadata.assert_called_once()


def test_synthdog_negative_split_ratio_raises():
    """Negative split_ratio components must raise ValueError before any I/O."""
    from unittest.mock import patch

    from template import SynthDoG

    with (
        patch("template._check_font_dirs"),
        patch("template.Background"),
        patch("template.Document"),
        patch("template.components"),
    ):
        with pytest.raises(ValueError, match="non-negative"):
            SynthDoG({}, split_ratio=[-0.1, 0.6, 0.5])


def test_synthdog_split_ratio_not_summing_to_one_raises():
    """A split_ratio whose sum deviates from 1.0 by more than ±1 % must raise ValueError."""
    from unittest.mock import patch

    from template import SynthDoG

    with (
        patch("template._check_font_dirs"),
        patch("template.Background"),
        patch("template.Document"),
        patch("template.components"),
    ):
        with pytest.raises(ValueError, match="sum"):
            SynthDoG({}, split_ratio=[0.3, 0.3, 0.1])


def test_save_split_idx_clamped_when_threshold_below_one():
    """Regression: if _split_thresholds[-1] < 1.0 due to float arithmetic,
    searchsorted can return len(splits), causing an IndexError. The fix clamps
    split_idx to len(splits)-1."""
    stub = _make_stub_synthdog()
    # Simulate floating-point threshold that doesn't reach 1.0
    stub._split_thresholds = np.array([0.3333333333333333, 0.6666666666666666, 0.9999999999999999])
    with tempfile.TemporaryDirectory() as root:
        # Drive many labels to maximize the chance of hitting the edge
        for i in range(200):
            stub.save(root, _good_data(label=f"word{i} test label sample text"), i)
    # If split_idx was ever out of bounds, save() would have raised IndexError.
    stub.format_metadata.assert_called()
