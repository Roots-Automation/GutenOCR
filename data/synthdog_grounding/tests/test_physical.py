"""Tests for effects/physical.py."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effects.physical import (
    BookSpineShadowEffect,
    FoldCreaseEffect,
    LowTonerStreakEffect,
    MoireOverlayEffect,
    StainOverlayEffect,
    VignettingEffect,
    WatermarkEffect,
    apply_if_enabled,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rgba(h=100, w=120, v=200):
    """Uniform gray RGBA image."""
    img = np.full((h, w, 4), v, dtype=np.uint8)
    img[..., 3] = 255
    return img


def _assert_invariants(original, result):
    """Shape, dtype, alpha, and value-range invariants that every effect must satisfy."""
    assert result.shape == original.shape, "shape changed"
    assert result.dtype == np.uint8, "dtype changed"
    assert np.array_equal(result[..., 3], original[..., 3]), "alpha channel mutated"
    assert result[..., :3].min() >= 0
    assert result[..., :3].max() <= 255


# ---------------------------------------------------------------------------
# apply_if_enabled
# ---------------------------------------------------------------------------


def test_apply_if_enabled_zero_prob_skips():
    img = _rgba()
    sentinel = [0]

    def boom(image, args):
        sentinel[0] += 1
        return image

    np.random.seed(0)
    result = apply_if_enabled({"prob": 0.0}, boom, img)
    assert sentinel[0] == 0
    assert np.array_equal(result, img)


def test_apply_if_enabled_full_prob_always_fires():
    img = _rgba()
    called = [0]

    def mark(image, args):
        called[0] += 1
        return image

    np.random.seed(0)
    for _ in range(5):
        apply_if_enabled({"prob": 1.0}, mark, img)
    assert called[0] == 5


def test_apply_if_enabled_passes_args():
    img = _rgba()
    received = {}

    def capture(image, args):
        received.update(args)
        return image

    np.random.seed(0)
    apply_if_enabled({"prob": 1.0, "args": {"key": 42}}, capture, img)
    assert received == {"key": 42}


# ---------------------------------------------------------------------------
# VignettingEffect
# ---------------------------------------------------------------------------


def test_vignetting_invariants():
    np.random.seed(0)
    img = _rgba()
    result = VignettingEffect.apply(img, {"intensity": [50, 50], "shape": [2.0, 2.0]})
    _assert_invariants(img, result)


def test_vignetting_corners_darker_than_center():
    """Vignette must darken corners more than center."""
    np.random.seed(0)
    img = _rgba(v=200)
    result = VignettingEffect.apply(img, {"intensity": [80, 80], "shape": [2.0, 2.0]})
    H, W = result.shape[:2]
    center = int(result[H // 2, W // 2, 0])
    corner = int(result[0, 0, 0])
    assert corner < center, f"corner={corner} not darker than center={center}"


def test_vignetting_no_brightening():
    """Vignette only darkens; no pixel should be brighter than the input."""
    np.random.seed(1)
    img = _rgba(v=150)
    result = VignettingEffect.apply(img, {"intensity": [30, 80], "shape": [1.5, 3.0]})
    assert np.all(result[..., :3].astype(np.int16) <= img[..., :3].astype(np.int16))


# ---------------------------------------------------------------------------
# BookSpineShadowEffect
# ---------------------------------------------------------------------------


def test_book_spine_shadow_invariants():
    np.random.seed(0)
    img = _rgba()
    result = BookSpineShadowEffect.apply(img, {"intensity": [60, 60], "width": [0.1, 0.1], "side": "left"})
    _assert_invariants(img, result)


def test_book_spine_shadow_left_darkens_left_edge():
    np.random.seed(0)
    img = _rgba(v=200)
    result = BookSpineShadowEffect.apply(img, {"intensity": [80, 80], "width": [0.2, 0.2], "side": "left"})
    H, W = result.shape[:2]
    left_mean = result[:, :5, 0].astype(float).mean()
    right_mean = result[:, -5:, 0].astype(float).mean()
    assert left_mean < right_mean, f"left={left_mean:.1f} not darker than right={right_mean:.1f}"


def test_book_spine_shadow_right_darkens_right_edge():
    np.random.seed(0)
    img = _rgba(v=200)
    result = BookSpineShadowEffect.apply(img, {"intensity": [80, 80], "width": [0.2, 0.2], "side": "right"})
    H, W = result.shape[:2]
    left_mean = result[:, :5, 0].astype(float).mean()
    right_mean = result[:, -5:, 0].astype(float).mean()
    assert right_mean < left_mean, f"right={right_mean:.1f} not darker than left={left_mean:.1f}"


def test_book_spine_shadow_no_brightening():
    np.random.seed(2)
    img = _rgba(v=180)
    result = BookSpineShadowEffect.apply(img, {})
    assert np.all(result[..., :3].astype(np.int16) <= img[..., :3].astype(np.int16))


# ---------------------------------------------------------------------------
# StainOverlayEffect
# ---------------------------------------------------------------------------


def test_stain_overlay_invariants():
    np.random.seed(0)
    img = _rgba()
    result = StainOverlayEffect.apply(img, {"count": [1, 1], "alpha": [0.15, 0.15]})
    _assert_invariants(img, result)


def test_stain_overlay_modifies_image():
    np.random.seed(0)
    img = _rgba(v=200)
    result = StainOverlayEffect.apply(img, {"count": [1, 1], "alpha": [0.5, 0.5]})
    assert not np.array_equal(result[..., :3], img[..., :3])


# ---------------------------------------------------------------------------
# FoldCreaseEffect
# ---------------------------------------------------------------------------


def test_fold_crease_invariants_horizontal():
    np.random.seed(0)
    img = _rgba()
    result = FoldCreaseEffect.apply(img, {"count": [1, 1], "orientation": "horizontal", "intensity": [30, 30]})
    _assert_invariants(img, result)


def test_fold_crease_invariants_vertical():
    np.random.seed(0)
    img = _rgba()
    result = FoldCreaseEffect.apply(img, {"count": [1, 1], "orientation": "vertical", "intensity": [30, 30]})
    _assert_invariants(img, result)


def test_fold_crease_invariants_diagonal():
    np.random.seed(0)
    img = _rgba()
    result = FoldCreaseEffect.apply(img, {"count": [1, 1], "orientation": "diagonal", "intensity": [30, 30]})
    _assert_invariants(img, result)


def test_fold_crease_only_darkens():
    """Crease is a subtractive effect; no pixel should be brightened."""
    np.random.seed(1)
    img = _rgba(v=150)
    result = FoldCreaseEffect.apply(img, {"count": [1, 1], "orientation": "horizontal", "intensity": [40, 40]})
    assert np.all(result[..., :3].astype(np.int16) <= img[..., :3].astype(np.int16))


# ---------------------------------------------------------------------------
# LowTonerStreakEffect
# ---------------------------------------------------------------------------


def test_low_toner_invariants_horizontal():
    np.random.seed(0)
    img = _rgba()
    result = LowTonerStreakEffect.apply(img, {"count": [1, 1], "orientation": "horizontal"})
    _assert_invariants(img, result)


def test_low_toner_invariants_vertical():
    np.random.seed(0)
    img = _rgba()
    result = LowTonerStreakEffect.apply(img, {"count": [1, 1], "orientation": "vertical"})
    _assert_invariants(img, result)


def test_low_toner_only_lightens():
    """Streak is an additive effect; no pixel should be darkened."""
    np.random.seed(2)
    img = _rgba(v=100)
    result = LowTonerStreakEffect.apply(img, {"count": [1, 1], "orientation": "horizontal", "intensity": [0.2, 0.2]})
    assert np.all(result[..., :3].astype(np.int16) >= img[..., :3].astype(np.int16))


# ---------------------------------------------------------------------------
# MoireOverlayEffect
# ---------------------------------------------------------------------------


def test_moire_invariants():
    np.random.seed(0)
    img = _rgba()
    result = MoireOverlayEffect.apply(img, {"frequency": [0.04, 0.04], "alpha": [0.1, 0.1], "angle": [30, 30]})
    _assert_invariants(img, result)


def test_moire_only_darkens():
    """Moiré pattern subtracts from RGB; no pixel should be brightened."""
    np.random.seed(0)
    img = _rgba(v=200)
    result = MoireOverlayEffect.apply(img, {"frequency": [0.04, 0.04], "alpha": [0.1, 0.1], "angle": [0, 0]})
    assert np.all(result[..., :3].astype(np.int16) <= img[..., :3].astype(np.int16))


# ---------------------------------------------------------------------------
# WatermarkEffect
# ---------------------------------------------------------------------------


def test_watermark_shape_and_dtype():
    np.random.seed(0)
    img = _rgba()
    result = WatermarkEffect.apply(img, {"words": ["DRAFT"], "alpha": [0.2, 0.2], "angle": [0, 0]})
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_watermark_modifies_image():
    np.random.seed(0)
    img = _rgba(v=255)
    result = WatermarkEffect.apply(img, {"words": ["TEST"], "alpha": [0.5, 0.5], "angle": [0, 0]})
    assert not np.array_equal(result, img)


# ---------------------------------------------------------------------------
# Geometry — precise spatial invariants
# ---------------------------------------------------------------------------


def test_vignetting_center_pixel_exactly_unchanged():
    """At the image center Xn=Yn=0, r=0, mask=0, so the pixel is untouched."""
    # Even dimensions: pixel at (H//2, W//2) has Xn=(H//2 - H/2)/(H/2)=0 exactly.
    np.random.seed(0)
    img = _rgba(h=100, w=100, v=180)
    result = VignettingEffect.apply(img, {"intensity": [80, 80], "shape": [2.0, 2.0]})
    cy, cx = img.shape[0] // 2, img.shape[1] // 2
    np.testing.assert_array_equal(result[cy, cx], img[cy, cx])


def test_book_spine_shadow_far_side_exactly_unchanged():
    """The side opposite the shadow has gradient=0; those pixels must equal the input."""
    np.random.seed(0)
    img = _rgba(h=100, w=100, v=200)
    # Left shadow covering 10 % of width (10 px).  Blur sigma = 10*0.15 = 1.5,
    # radius = 2, so the blur can spread at most 2 px beyond the gradient edge.
    # Pixels beyond col 12 are guaranteed to be unaffected.
    result = BookSpineShadowEffect.apply(img, {"intensity": [60, 60], "width": [0.1, 0.1], "side": "left"})
    np.testing.assert_array_equal(result[:, 15:, :3], img[:, 15:, :3])


# ---------------------------------------------------------------------------
# Visibility floor — max default intensity must not black out the image
# ---------------------------------------------------------------------------


def test_vignetting_visibility_floor():
    """VignettingEffect at intensity=80 subtracts at most 80; white → ≥175."""
    # At corners r=1: mask = 1^shape * 80 = 80 → 255 - 80 = 175 (exact lower bound).
    np.random.seed(0)
    img = _rgba(v=255)
    result = VignettingEffect.apply(img, {"intensity": [80, 80], "shape": [2.0, 2.0]})
    assert int(result[..., :3].min()) >= 255 - 80


def test_book_spine_shadow_visibility_floor():
    """BookSpineShadowEffect at intensity=100 subtracts at most 100; white → ≥155."""
    np.random.seed(0)
    img = _rgba(v=255)
    result = BookSpineShadowEffect.apply(img, {"intensity": [100, 100], "width": [0.2, 0.2], "side": "left"})
    assert int(result[..., :3].min()) >= 255 - 100


def test_moire_visibility_floor():
    """MoireOverlayEffect at alpha=0.15 subtracts at most floor(0.15*255)=38; white → ≥217."""
    np.random.seed(0)
    img = _rgba(v=255)
    result = MoireOverlayEffect.apply(img, {"frequency": [0.04, 0.04], "alpha": [0.15, 0.15], "angle": [30, 30]})
    assert int(result[..., :3].min()) >= 255 - int(0.15 * 255)


def test_fold_crease_visibility_floor():
    """FoldCreaseEffect at intensity=45 subtracts at most 45; white → ≥210."""
    np.random.seed(0)
    img = _rgba(v=255)
    result = FoldCreaseEffect.apply(img, {"count": [1, 1], "orientation": "horizontal", "intensity": [45, 45]})
    assert int(result[..., :3].min()) >= 255 - 45
