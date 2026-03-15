import math
import warnings

from PIL import ImageFont


def register_pillow_compat():
    """
    Monkey-patches Pillow 10+ to restore removed methods ``getsize`` and
    ``getmask2`` that synthtiger still relies on.
    """

    # Patch ImageFont.FreeTypeFont.getsize
    if not hasattr(ImageFont.FreeTypeFont, "getsize"):

        def getsize(self, text, direction=None, features=None, language=None):
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

            return w, h

        setattr(ImageFont.FreeTypeFont, "getsize", getsize)

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
