import math
import warnings
from functools import lru_cache

import numpy as np
from PIL import ImageFont


@lru_cache(maxsize=128)
def _cached_truetype(path: str, size: int):
    return ImageFont.truetype(path, size=size)


def _patch_font_cache():
    """Cache ImageFont.truetype by (path, size) — SynthTiger loads the same
    font hundreds of times per sample with no cache of its own."""
    from synthtiger.layers import text_layer as _tl

    _tl.TextLayer._read_font = staticmethod(lambda path, size: _cached_truetype(path, size))


def _fast_to_rgb(gray: int, colorize: bool = False):
    """Drop-in for synthtiger.utils.image_util.to_rgb.

    The original permutes all 65 536 (r, g) pairs to find a valid triple.
    We generate a batch of 512 candidates at once and check vectorized —
    3 numpy calls regardless of how many candidates are valid.

    Uses the global np.random state (seeded by set_global_random_seed) so that
    generate(seed=N) is fully deterministic.
    """
    if not colorize:
        return (gray, gray, gray)

    r = np.random.randint(0, 256, size=512, dtype=np.int32)
    g = np.random.randint(0, 256, size=512, dtype=np.int32)
    b = np.rint((gray - r * 0.2989 - g * 0.5870) / 0.1140).astype(np.int32)
    valid = (b >= 0) & (b < 256)
    if valid.any():
        idx = int(np.argmax(valid))
        return (int(r[idx]), int(g[idx]), int(b[idx]))
    return (gray, gray, gray)


def _patch_to_rgb():
    """Replace the permutation-based to_rgb in synthtiger with the fast version."""
    import synthtiger.components.color.gray as _gray
    import synthtiger.components.color.gray_map as _gray_map
    import synthtiger.utils as _u
    import synthtiger.utils.image_util as _iu

    _iu.to_rgb = _fast_to_rgb
    _u.to_rgb = _fast_to_rgb
    # The color components import `utils` and call `utils.to_rgb` directly,
    # so patch at the module level where they'll look it up.
    _gray.utils.to_rgb = _fast_to_rgb
    _gray_map.utils.to_rgb = _fast_to_rgb


_getsize_cache: dict = {}  # (path, size, text, direction) -> (w, h)
_getlength_cache: dict = {}  # (path, size, text, direction) -> float


def register_pillow_compat():
    """
    Monkey-patches Pillow 10+ to restore removed methods ``getsize`` and
    ``getmask2`` that synthtiger still relies on.
    """

    # Patch ImageFont.FreeTypeFont.getsize
    if not hasattr(ImageFont.FreeTypeFont, "getsize"):

        def getsize(self, text, direction=None, features=None, language=None):
            key = (self.path, self.size, text, direction)
            cached = _getsize_cache.get(key)
            if cached is not None:
                return cached

            # Width: prefer getlength (advance width) over getbbox (ink width)
            try:
                w = int(math.ceil(self.getlength(text, direction=direction, features=features, language=language)))
            except (AttributeError, KeyError):
                try:
                    left, _, right, _ = self.getbbox(text, direction=direction, features=features, language=language)
                except KeyError:
                    left, _, right, _ = self.getbbox(text)
                w = right - left

            # Height: ink height from getbbox is correct for layout purposes
            try:
                _, top, _, bottom = self.getbbox(text, direction=direction, features=features, language=language)
            except KeyError:
                _, top, _, bottom = self.getbbox(text)
            h = bottom - top

            result = (w, h)
            _getsize_cache[key] = result
            return result

        setattr(ImageFont.FreeTypeFont, "getsize", getsize)

    # Patch FreeTypeFont.getlength with a cache — it's called ~32k times per
    # 10 samples, always for single characters which repeat constantly.
    _original_getlength = ImageFont.FreeTypeFont.getlength

    def getlength(self, text, mode="", direction=None, features=None, language=None):
        key = (self.path, self.size, text, direction)
        cached = _getlength_cache.get(key)
        if cached is not None:
            return cached
        result = _original_getlength(self, text, mode=mode, direction=direction, features=features, language=language)
        _getlength_cache[key] = result
        return result

    setattr(ImageFont.FreeTypeFont, "getlength", getlength)

    # Patch ImageFont.FreeTypeFont.getmask2 to handle missing libraqm.
    # We always patch this because even if it exists, it may raise KeyError
    # in Pillow 10+ when libraqm is missing but direction/features are passed.
    if hasattr(ImageFont.FreeTypeFont, "getmask2"):
        original_getmask2 = ImageFont.FreeTypeFont.getmask2

        def getmask2(self, text, mode="", direction=None, features=None, language=None, *args, **kwargs):
            try:
                return original_getmask2(self, text, mode, direction, features, language, *args, **kwargs)
            except KeyError:
                warnings.warn(
                    "libraqm is not available — ignoring direction/features/language "
                    "for text rendering. Install libraqm for full CJK/RTL support.",
                    stacklevel=2,
                )
                # Only warn once
                warnings.filterwarnings("ignore", message="libraqm is not available")
                return original_getmask2(self, text, mode, *args, **kwargs)

        setattr(ImageFont.FreeTypeFont, "getmask2", getmask2)


# Apply patches immediately on import
register_pillow_compat()
_patch_font_cache()
_patch_to_rgb()
