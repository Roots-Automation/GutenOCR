import PIL
from PIL import ImageFont, ImageDraw

def register_pillow_compat():
    """
    Monkey-patches Pillow > 10 to include removed methods 'getsize' and 'textsize'
    which are required by synthtiger.
    """
    if int(PIL.__version__.split('.')[0]) < 10:
        return

    # Patch ImageFont.FreeTypeFont.getsize
    if not hasattr(ImageFont.FreeTypeFont, "getsize"):
        def getsize(self, text, direction=None, features=None, language=None):
            # getbbox returns (left, top, right, bottom)
            try:
                left, top, right, bottom = self.getbbox(text, direction=direction, features=features, language=language)
            except KeyError: 
                # Likely 'setting text direction... is not supported without libraqm'
                # If we don't have raqm, we can't support these features.
                # Fallback to simple bbox without direction/features
                left, top, right, bottom = self.getbbox(text)
            
            return right - left, bottom - top
        
        setattr(ImageFont.FreeTypeFont, "getsize", getsize)

    # Patch ImageFont.FreeTypeFont.getmask2 to handle missing libraqm
    # We always patch this because even if it exists, it might fail with KeyError in Pillow 10+
    # if libraqm is missing but direction/features are passed.
    if hasattr(ImageFont.FreeTypeFont, "getmask2"):
        original_getmask2 = ImageFont.FreeTypeFont.getmask2
        
        def getmask2(self, text, mode="", direction=None, features=None, language=None, *args, **kwargs):
            try:
                return original_getmask2(self, text, mode, direction, features, language, *args, **kwargs)
            except KeyError:
                # Fallback: try without direction/features/language
                return original_getmask2(self, text, mode, *args, **kwargs)
        
        setattr(ImageFont.FreeTypeFont, "getmask2", getmask2)

    # Patch ImageDraw.ImageDraw.textsize
    if not hasattr(ImageDraw.ImageDraw, "textsize"):
        def textsize(self, text, font=None, spacing=4, direction=None, features=None, language=None, stroke_width=0, embedded_color=False):
            if font is None:
                font = self.getfont()
            
            # textbbox returns (left, top, right, bottom)
            left, top, right, bottom = self.textbbox((0, 0), text, font=font, spacing=spacing, direction=direction, features=features, language=language, stroke_width=stroke_width, embedded_color=embedded_color)
            return right - left, bottom - top

        setattr(ImageDraw.ImageDraw, "textsize", textsize)

# Apply patches immediately on import
register_pillow_compat()
