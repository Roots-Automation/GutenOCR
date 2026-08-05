import math
import warnings
from collections import OrderedDict
from functools import lru_cache

import numpy as np
from PIL import Image, ImageFont, ImageOps


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


def _patch_layer_init():
    """Avoid a redundant full-canvas copy in synthtiger's Layer.__init__.

    Layer.__init__ does `np.array(image, dtype=np.float32)`, which always
    copies — even when `image` is already a freshly-created float32 ndarray
    that nothing else references (e.g. Group.output()'s return value, about
    to be wrapped by Group.merge()). Every Layer construction / Group.merge()
    call pays this. Switched to np.asarray, which only copies when the dtype
    doesn't already match.

    Layer.copy() passes `self.image` — the layer's own *live* array — into
    Layer(...), relying on the constructor to make an independent copy; with
    asarray that would alias instead, breaking the two real callers
    (text_extrusion.py, text_shadow.py) that mutate the copy separately from
    the original. So copy() is patched alongside to force an explicit copy.
    """
    from synthtiger import utils
    from synthtiger.layers.layer import Layer

    def _fast_init(self, image):
        image = np.asarray(image, dtype=np.float32)
        image = utils.add_alpha_channel(image)
        height, width = image.shape[:2]
        self.image = image
        self.bbox = [0, 0, width, height]

    def _copy(self):
        layer = Layer(self.image.copy())
        layer.quad = self.quad
        return layer

    Layer.__init__ = _fast_init
    Layer.copy = _copy


# Requested lower-bound for PIL's JPEG "draft" decode (libjpeg scaled IDCT —
# decodes directly at a reduced resolution instead of full-res-then-shrink).
# draft() only accepts power-of-2 scales (1, 1/2, 1/4, 1/8) and picks the
# smallest one that still keeps BOTH dimensions >= this target, so this must
# be well below half of the *short* side of the smallest source image we
# still want to benefit — chosen against this repo's actual background/paper
# textures (short sides from ~1400px up to ~3000px), not a config knob.
_TEXTURE_DRAFT_TARGET = (1024, 1024)


def _draft_open(path):
    """Open *path* and ask the decoder to draft down toward
    _TEXTURE_DRAFT_TARGET (a no-op for non-JPEG formats — draft() is only
    implemented for JPEG/MPO, PIL's base Image.draft() does nothing).
    Returns (image, width, height) with the EXIF-orientation swap applied to
    width/height, matching BaseTexture._get_size's existing convention.
    """
    image = Image.open(path)
    image.draft("RGB", _TEXTURE_DRAFT_TARGET)
    width, height = image.size
    exif = dict(image.getexif())
    if exif.get(0x0112, 1) >= 5:
        width, height = height, width
    return image, width, height


def _patch_texture_downscale():
    """Pre-downscale synthtiger's background/paper texture decoding.

    BaseTexture._read_texture decodes the full native-resolution source image
    (this repo's textures run up to ~4200px on a side, >150MB as float32
    RGBA) before BaseTexture.apply() resizes it down to the canvas size —
    every sample, for every texture. _get_size and _read_texture are patched
    together to both go through _draft_open, so they agree on the resulting
    (possibly-drafted) dimensions by construction: BaseTexture.sample() reads
    crop x/y/w/h against whatever _get_size reports, and _read_texture must
    decode at that same size for the crop indices in data() to line up.
    """
    from synthtiger.components.texture.base_texture import BaseTexture

    def _fast_get_size(self, path):
        _, width, height = _draft_open(path)
        return width, height

    def _fast_read_texture(self, path, grayscale=False):
        texture, _, _ = _draft_open(path)
        texture = ImageOps.exif_transpose(texture)
        if grayscale:
            texture = texture.convert("L")
        texture = texture.convert("RGBA")
        texture = np.array(texture, dtype=np.float32)
        return texture

    BaseTexture._get_size = _fast_get_size
    BaseTexture._read_texture = _fast_read_texture


_TEXT_CACHE_MAXSIZE = 8192  # bounds below: keys include the measured text, which
# rarely repeats verbatim across samples (see textbox.py's per-prefix getlength
# calls) — an unbounded dict here leaks one entry per unique text seen, forever.
_getsize_cache: OrderedDict = OrderedDict()  # (path, size, text, direction) -> (w, h)
_getlength_cache: OrderedDict = OrderedDict()  # (path, size, text, direction) -> float


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


# Glyph-presence detection caches.
# Reference mask: U+E000 (Private Use Area) is never mapped by any text font,
# so its rendered mask is always the .notdef fallback glyph.
_notdef_mask_cache: dict = {}  # (path, size) -> np.ndarray
_renderable_cache: dict = {}  # (path, size, codepoint) -> bool


def _is_renderable(font_obj, char: str) -> bool:
    """Return True if font_obj has a real glyph for char.

    Compares the rendered mask of char against the .notdef fallback (obtained
    via U+E000, which no text font maps).  Results are cached per (font, size,
    codepoint) so the cost is one getmask call per unique character seen.
    """
    key = (font_obj.path, font_obj.size)
    if key not in _notdef_mask_cache:
        _notdef_mask_cache[key] = np.array(font_obj.getmask(""))
    cp = ord(char)
    cache_key = (*key, cp)
    if cache_key not in _renderable_cache:
        _renderable_cache[cache_key] = not np.array_equal(np.array(font_obj.getmask(char)), _notdef_mask_cache[key])
    return _renderable_cache[cache_key]


def register_pillow_compat():
    """
    Monkey-patches Pillow 10+ to restore removed methods ``getsize`` and
    ``getmask2`` that synthtiger still relies on.
    """

    # Patch ImageFont.FreeTypeFont.getsize
    if not hasattr(ImageFont.FreeTypeFont, "getsize"):

        def getsize(self, text, direction=None, features=None, language=None):
            key = (self.path, self.size, text, direction)
            cached = _bounded_cache_get(_getsize_cache, key)
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
            _bounded_cache_set(_getsize_cache, key, result)
            return result

        setattr(ImageFont.FreeTypeFont, "getsize", getsize)

    # Patch FreeTypeFont.getlength with a cache — it's called ~32k times per
    # 10 samples, always for single characters which repeat constantly.
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
_patch_layer_init()
_patch_texture_downscale()
