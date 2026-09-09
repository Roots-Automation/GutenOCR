import math
import warnings
from collections import OrderedDict
from functools import lru_cache

import numpy as np
from PIL import ImageFont


@lru_cache(maxsize=128)
def _cached_truetype(path: str, size: int):
    return ImageFont.truetype(path, size=size)


_TEXT_CACHE_MAXSIZE = 8192
_getsize_cache: OrderedDict = OrderedDict()
_getlength_cache: OrderedDict = OrderedDict()


def _bounded_cache_get(cache: OrderedDict, key):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


def _bounded_cache_set(cache: OrderedDict, key, value, maxsize: int = _TEXT_CACHE_MAXSIZE):
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > maxsize:
        cache.popitem(last=False)


_notdef_mask_cache: dict = {}
_renderable_cache: dict = {}


def _is_renderable(font_obj, char: str) -> bool:
    """Return True if font_obj has a real glyph for char."""
    key = (font_obj.path, font_obj.size)
    if key not in _notdef_mask_cache:
        _notdef_mask_cache[key] = np.array(font_obj.getmask(""))
    cp = ord(char)
    cache_key = (*key, cp)
    if cache_key not in _renderable_cache:
        _renderable_cache[cache_key] = not np.array_equal(np.array(font_obj.getmask(char)), _notdef_mask_cache[key])
    return _renderable_cache[cache_key]


def register_pillow_compat():
    """Monkey-patches Pillow 10+ to restore removed methods getsize and getmask2."""

    if not hasattr(ImageFont.FreeTypeFont, "getsize"):

        def getsize(self, text, direction=None, features=None, language=None):
            key = (self.path, self.size, text, direction)
            cached = _bounded_cache_get(_getsize_cache, key)
            if cached is not None:
                return cached

            try:
                w = int(math.ceil(self.getlength(text, direction=direction, features=features, language=language)))
            except (AttributeError, KeyError):
                try:
                    left, _, right, _ = self.getbbox(text, direction=direction, features=features, language=language)
                except KeyError:
                    left, _, right, _ = self.getbbox(text)
                w = right - left

            try:
                _, top, _, bottom = self.getbbox(text, direction=direction, features=features, language=language)
            except KeyError:
                _, top, _, bottom = self.getbbox(text)
            h = bottom - top

            result = (w, h)
            _bounded_cache_set(_getsize_cache, key, result)
            return result

        setattr(ImageFont.FreeTypeFont, "getsize", getsize)

    _original_getlength = ImageFont.FreeTypeFont.getlength

    def getlength(self, text, mode="", direction=None, features=None, language=None):
        key = (self.path, self.size, text, direction)
        cached = _bounded_cache_get(_getlength_cache, key)
        if cached is not None:
            return cached
        result = _original_getlength(self, text, mode=mode, direction=direction, features=features, language=language)
        _bounded_cache_set(_getlength_cache, key, result)
        return result

    setattr(ImageFont.FreeTypeFont, "getlength", getlength)

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
                warnings.filterwarnings("ignore", message="libraqm is not available")
                return original_getmask2(self, text, mode, *args, **kwargs)

        setattr(ImageFont.FreeTypeFont, "getmask2", getmask2)


register_pillow_compat()
