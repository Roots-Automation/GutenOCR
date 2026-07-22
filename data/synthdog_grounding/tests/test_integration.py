"""Integration tests for SynthDoG.generate().

These tests instantiate the full pipeline with real resources (fonts, corpus,
background/paper textures) and a hand-built minimal config.  They are marked
slow because they run the complete rendering stack.

The config deliberately disables all stochastic effects (prob=0) so that the
only randomness comes from layout, font, and color choices.  This gives fast,
deterministic runs while still exercising the core rendering path.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Minimal pipeline config
# ---------------------------------------------------------------------------

_BASE = Path(__file__).resolve().parents[1]
_BG = str(_BASE / "resources/background")
_PAPER = str(_BASE / "resources/paper")
_CORPUS = str(_BASE / "resources/corpus/enwiki.txt")
_FONTS = str(_BASE / "resources/font/en")

# Build once; SynthDoG copies it internally so sharing is safe.
_MINIMAL_CFG = {
    "quality": [85, 85],
    "landscape": 0.0,
    "short_size": [480, 480],
    "aspect_ratio": [1.5, 1.5],
    "emit_quads": False,
    "min_bbox_area": 16,
    # Permissive quality filters so smoke tests are never spuriously skipped.
    "min_contrast_ratio": 1.0,
    "min_word_count": 1,
    "max_textbox_null_frac": 1.0,
    "min_line_height_px": 1.0,
    "min_sharpness": 0.0,
    "max_intra_block_line_overlap": 1.0,
    "max_cross_block_line_overlap": 1.0,
    "background": {
        "image": {"paths": [_BG], "weights": [1]},
        "effect": {"args": [{"prob": 0, "args": {"sigma": [0, 0]}}]},
    },
    "document": {
        "fullscreen": 1.0,
        "landscape": 0.0,
        "short_size": [480, 480],
        "aspect_ratio": [1.5, 1.5],
        "paper": {
            "image": {"paths": [_PAPER], "weights": [1], "alpha": [0, 0], "grayscale": 0, "crop": 0},
        },
        "content": {
            "text": {"path": _CORPUS},
            "font": {"paths": [_FONTS], "weights": [1]},
            "layout": {
                "text_scale": [0.05, 0.08],
                "max_row": 5,
                "max_col": 2,
                "fill": [0.5, 1],
                "full": 0,
            },
            "textbox": {"fill": [0.5, 1]},
        },
        # All doc effects disabled for speed.
        "effect": {
            "args": [
                {"prob": 0, "args": {"alpha": [0, 0], "sigma": [0, 0]}},
                {"prob": 0, "args": {"scale": [0, 0], "per_channel": 0}},
                {"prob": 0, "args": {"k": [1, 1]}},
                {"prob": 0, "args": {"k": [1, 1]}},
                {"prob": 0, "args": {"p": [0, 0], "size_percent": [0.1, 0.1], "per_channel": 0}},
                {"prob": 0, "args": {"weights": [1], "args": [{"percents": [[1, 1], [1, 1], [1, 1], [1, 1]]}]}},
            ]
        },
    },
    "bg_effect": {
        "args": [{"prob": 0, "args": {"intensity": [0, 0], "amount": [0, 0], "smoothing": [1, 1], "bidirectional": 0}}]
    },
    "doc_effect": {
        "args": [{"prob": 0, "args": {"intensity": [0, 0], "amount": [0, 0], "smoothing": [1, 1], "bidirectional": 0}}]
    },
    "effect": {
        "args": [
            {"prob": 0, "args": {"rgb": [[128, 128], [128, 128], [128, 128]], "alpha": [0, 0]}},
            {"prob": 0, "args": {}},
            {"prob": 1, "args": {"alpha": [1, 1]}},  # contrast — identity
            {"prob": 1, "args": {"beta": [0, 0]}},  # brightness — identity
            {"prob": 0, "args": {"k": [3, 3], "angle": [0, 0]}},
            {"prob": 1, "args": {"sigma": [0, 0]}},  # gaussian blur — identity
            {"prob": 0, "args": {"size": [1, 1]}},
            {"prob": 0, "args": {"compression": [10, 10]}},
        ]
    },
    "skew": {"prob": 0, "angle": [0, 0]},
}


@pytest.fixture(scope="module")
def dog():
    """Single SynthDoG instance shared across all tests in this module."""
    from template import SynthDoG

    return SynthDoG(_MINIMAL_CFG)


@pytest.fixture(scope="module")
def sample_42(dog):
    """Pre-generated sample for seed=42, computed once per module."""
    return dog.generate(seed=42)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_generate_returns_image_with_correct_shape(sample_42):
    img = sample_42["image"]
    assert img.ndim == 3
    assert img.shape[2] == 4
    assert np.isfinite(img).all()


@pytest.mark.slow
def test_generate_image_has_four_channels(sample_42):
    assert sample_42["image"].ndim == 3
    assert sample_42["image"].shape[2] == 4


@pytest.mark.slow
def test_generate_has_nonempty_lines(sample_42):
    assert len(sample_42["lines"]) > 0


@pytest.mark.slow
def test_generate_has_nonempty_words(sample_42):
    assert len(sample_42["words"]) > 0


@pytest.mark.slow
def test_generate_has_nonempty_blocks(sample_42):
    assert len(sample_42["blocks"]) > 0


@pytest.mark.slow
def test_generate_label_contains_line_words(sample_42):
    # label is space-joined, whitespace-normalized text from all lines
    label = sample_42["label"]
    assert label  # non-empty
    for ln in sample_42["lines"]:
        # Every line's first word must appear somewhere in the label
        first_word = ln.text.split()[0] if ln.text.split() else None
        if first_word:
            assert first_word in label, f"word {first_word!r} from line not found in label"


@pytest.mark.slow
def test_generate_quality_metrics_present(sample_42):
    qm = sample_42["quality_metrics"]
    assert "sharpness" in qm
    assert "min_line_contrast_ratio" in qm
    assert "word_count" in qm
    assert "textbox_null_frac" in qm


# ---------------------------------------------------------------------------
# Annotation validity
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_generate_line_bboxes_normalized(sample_42):
    for ln in sample_42["lines"]:
        x1, y1, x2, y2 = ln.bbox
        assert 0.0 <= x1 <= x2 <= 1.0, f"x out of range: {ln.bbox}"
        assert 0.0 <= y1 <= y2 <= 1.0, f"y out of range: {ln.bbox}"


@pytest.mark.slow
def test_generate_word_bboxes_normalized(sample_42):
    for wd in sample_42["words"]:
        x1, y1, x2, y2 = wd.bbox
        assert 0.0 <= x1 <= x2 <= 1.0
        assert 0.0 <= y1 <= y2 <= 1.0


@pytest.mark.slow
def test_generate_block_bboxes_normalized(sample_42):
    for blk in sample_42["blocks"]:
        x1, y1, x2, y2 = blk.bbox
        assert 0.0 <= x1 <= x2 <= 1.0
        assert 0.0 <= y1 <= y2 <= 1.0


@pytest.mark.slow
def test_generate_line_ids_dense_from_zero(sample_42):
    ids = [ln.line_id for ln in sample_42["lines"]]
    assert ids == list(range(len(ids)))


@pytest.mark.slow
def test_generate_word_line_ids_valid(sample_42):
    line_ids = {ln.line_id for ln in sample_42["lines"]}
    for wd in sample_42["words"]:
        assert wd.line_id in line_ids


@pytest.mark.slow
def test_generate_block_covers_member_lines(sample_42):
    """Each block's bbox must contain every line bbox assigned to it."""
    line_map = {ln.line_id: ln for ln in sample_42["lines"]}
    for blk in sample_42["blocks"]:
        for lid in blk.line_ids:
            ln_bbox = line_map[lid].bbox
            assert ln_bbox[0] >= blk.bbox[0] - 1e-3
            assert ln_bbox[1] >= blk.bbox[1] - 1e-3
            assert ln_bbox[2] <= blk.bbox[2] + 1e-3
            assert ln_bbox[3] <= blk.bbox[3] + 1e-3


# ---------------------------------------------------------------------------
# Quality metrics sanity
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_generate_sharpness_positive_for_text_image(sample_42):
    # A rendered text image has edges → Laplacian variance must be nonzero.
    assert sample_42["quality_metrics"]["sharpness"] > 0.0


@pytest.mark.slow
def test_generate_word_count_matches_metric(sample_42):
    assert sample_42["quality_metrics"]["word_count"] == len(sample_42["words"])


@pytest.mark.slow
def test_generate_line_count_matches_metric(sample_42):
    assert sample_42["quality_metrics"]["line_count"] == len(sample_42["lines"])


@pytest.mark.slow
def test_generate_null_frac_in_range(sample_42):
    frac = sample_42["quality_metrics"]["textbox_null_frac"]
    assert 0.0 <= frac <= 1.0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_generate_same_seed_produces_identical_image(dog):
    a = dog.generate(seed=42)
    b = dog.generate(seed=42)
    assert np.array_equal(a["image"], b["image"])


@pytest.mark.slow
def test_generate_same_seed_produces_identical_label(dog):
    a = dog.generate(seed=42)
    b = dog.generate(seed=42)
    assert a["label"] == b["label"]


@pytest.mark.slow
def test_generate_same_seed_produces_identical_annotations(dog):
    a = dog.generate(seed=42)
    b = dog.generate(seed=42)
    assert [ln.bbox for ln in a["lines"]] == [ln.bbox for ln in b["lines"]]
    assert [wd.bbox for wd in a["words"]] == [wd.bbox for wd in b["words"]]


@pytest.mark.slow
def test_generate_different_seeds_produce_different_images(dog):
    a = dog.generate(seed=42)
    b = dog.generate(seed=99)
    assert not np.array_equal(a["image"], b["image"])


# ---------------------------------------------------------------------------
# Quality filter catches bad samples
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_save_rejects_low_sharpness_sample(dog, sample_42, tmp_path):
    """save() must discard a sample whose sharpness is below the configured floor.

    We mutate only the quality_metrics dict (not the image), borrow the real
    SynthDoG.save() implementation via the module, and bind it to a minimal stub
    that has min_sharpness set impossibly high.
    """
    from unittest.mock import MagicMock

    from template import SynthDoG

    stub = MagicMock(
        min_contrast_ratio=1.0,
        min_word_count=1,
        max_textbox_null_frac=1.0,
        min_line_height_px=1.0,
        min_sharpness=1e9,  # impossibly high — any real sample fails
        max_intra_block_line_overlap=1.0,
        max_cross_block_line_overlap=1.0,
        splits=["train"],
        _split_thresholds=np.array([1.0]),
    )
    stub.format_metadata = MagicMock(return_value={})

    blurry_data = dict(sample_42)
    blurry_data["quality_metrics"] = dict(sample_42["quality_metrics"])
    blurry_data["quality_metrics"]["sharpness"] = 0.0

    SynthDoG.save(stub, str(tmp_path), blurry_data, 0)
    stub.format_metadata.assert_not_called()


@pytest.mark.slow
def test_save_writes_passing_sample(dog, sample_42, tmp_path):
    """save() must write a JPEG and a metadata.jsonl for a sample that passes all filters."""
    import glob

    dog.save(str(tmp_path), sample_42, 0)

    jpgs = glob.glob(str(tmp_path / "**" / "*.jpg"), recursive=True)
    jsonls = glob.glob(str(tmp_path / "**" / "*.jsonl"), recursive=True)
    assert jpgs, "expected at least one JPEG to be written"
    assert jsonls, "expected at least one metadata.jsonl to be written"
