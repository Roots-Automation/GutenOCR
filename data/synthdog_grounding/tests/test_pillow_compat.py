"""Tests for pillow_compat.py."""

import sys
from pathlib import Path

import numpy as np
from PIL import ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pillow_compat import _cached_truetype, _fast_to_rgb

# A real font shipped with the project — required for getsize/getlength tests.
_FONT_PATH = str(Path(__file__).resolve().parents[1] / "resources/font/en/CourierPrime-BoldItalic.ttf")
_FONT_SIZE = 24


# ---------------------------------------------------------------------------
# _cached_truetype
# ---------------------------------------------------------------------------


def test_cached_truetype_returns_freetype_font():
    font = _cached_truetype(_FONT_PATH, _FONT_SIZE)
    assert isinstance(font, ImageFont.FreeTypeFont)


def test_cached_truetype_same_object_on_second_call():
    font_a = _cached_truetype(_FONT_PATH, _FONT_SIZE)
    font_b = _cached_truetype(_FONT_PATH, _FONT_SIZE)
    assert font_a is font_b


def test_cached_truetype_different_sizes_distinct_objects():
    font_12 = _cached_truetype(_FONT_PATH, 12)
    font_24 = _cached_truetype(_FONT_PATH, 24)
    assert font_12 is not font_24


# ---------------------------------------------------------------------------
# _fast_to_rgb
# ---------------------------------------------------------------------------


def test_fast_to_rgb_no_colorize_returns_gray_triple():
    for gray in (0, 128, 255):
        assert _fast_to_rgb(gray, colorize=False) == (gray, gray, gray)


def test_fast_to_rgb_colorize_satisfies_grayscale_formula():
    """The returned triple must reconstruct the original gray within ±1 (rounding)."""
    np.random.seed(42)
    for gray in range(0, 256, 16):
        r, g, b = _fast_to_rgb(gray, colorize=True)
        reconstructed = round(r * 0.2989 + g * 0.5870 + b * 0.1140)
        assert abs(reconstructed - gray) <= 1, f"gray={gray}: ({r},{g},{b}) reconstructs to {reconstructed}"


def test_fast_to_rgb_colorize_values_in_byte_range():
    np.random.seed(0)
    for gray in range(0, 256, 8):
        r, g, b = _fast_to_rgb(gray, colorize=True)
        assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255


def test_fast_to_rgb_deterministic_under_seeded_global_rng():
    np.random.seed(7)
    result_a = _fast_to_rgb(100, colorize=True)
    np.random.seed(7)
    result_b = _fast_to_rgb(100, colorize=True)
    assert result_a == result_b


def test_fast_to_rgb_colorize_false_never_touches_rng():
    """colorize=False must not advance the RNG state."""
    np.random.seed(99)
    state_before = np.random.get_state()[1].copy()
    _fast_to_rgb(128, colorize=False)
    state_after = np.random.get_state()[1]
    np.testing.assert_array_equal(state_before, state_after)


# ---------------------------------------------------------------------------
# register_pillow_compat — getsize / getlength patching
# ---------------------------------------------------------------------------


def test_freetype_font_has_getsize_after_register():
    assert hasattr(ImageFont.FreeTypeFont, "getsize")


def test_getsize_returns_positive_dimensions():
    font = _cached_truetype(_FONT_PATH, _FONT_SIZE)
    w, h = font.getsize("Hello")
    assert w > 0 and h > 0


def _live_getsize_cache():
    """Return the _getsize_cache dict that the patched FreeTypeFont.getsize
    actually writes to (regardless of which pillow_compat module was loaded first).
    """
    from PIL import ImageFont

    return ImageFont.FreeTypeFont.getsize.__globals__["_getsize_cache"]


def test_getsize_cached_on_second_call():
    cache = _live_getsize_cache()
    font = _cached_truetype(_FONT_PATH, _FONT_SIZE)
    cache.clear()
    font.getsize("X")
    assert any(k[2] == "X" for k in cache)


def test_getsize_cache_key_uses_path_and_size_not_id():
    """Cache key must not use id(font) — stable across GC cycles."""
    cache = _live_getsize_cache()
    font = _cached_truetype(_FONT_PATH, _FONT_SIZE)
    cache.clear()
    font.getsize("A")
    key = next(iter(cache))
    # Key is (path, size, text, direction) — first element must be a str path.
    assert isinstance(key[0], str), f"expected str path as key[0], got {type(key[0])}"
    assert key[1] == _FONT_SIZE


def test_getsize_wider_for_longer_text():
    font = _cached_truetype(_FONT_PATH, _FONT_SIZE)
    w_short, _ = font.getsize("Hi")
    w_long, _ = font.getsize("Hello, World!")
    assert w_long > w_short


def _live_getlength_cache():
    """Return the _getlength_cache dict that the patched getlength actually writes to."""
    from PIL import ImageFont

    return ImageFont.FreeTypeFont.getlength.__globals__["_getlength_cache"]


def test_getlength_cache_key_uses_path_and_size_not_id():
    cache = _live_getlength_cache()
    font = _cached_truetype(_FONT_PATH, _FONT_SIZE)
    cache.clear()
    font.getlength("Z")
    key = next(iter(cache))
    assert isinstance(key[0], str), f"expected str path as key[0], got {type(key[0])}"
    assert key[1] == _FONT_SIZE


def test_getlength_cached_on_second_call():
    cache = _live_getlength_cache()
    font = _cached_truetype(_FONT_PATH, _FONT_SIZE)
    cache.clear()
    font.getlength("Q")
    size_after_first = len(cache)
    font.getlength("Q")
    assert len(cache) == size_after_first


def test_getlength_returns_positive_float():
    font = _cached_truetype(_FONT_PATH, _FONT_SIZE)
    length = font.getlength("abc")
    assert length > 0


# ---------------------------------------------------------------------------
# _patch_to_rgb — synthtiger.utils.to_rgb replaced
# ---------------------------------------------------------------------------


def test_patch_to_rgb_replaces_synthtiger_utils():
    import synthtiger.utils as _u

    assert _u.to_rgb is _fast_to_rgb


def test_patch_to_rgb_replaces_image_util():
    import synthtiger.utils.image_util as _iu

    assert _iu.to_rgb is _fast_to_rgb
