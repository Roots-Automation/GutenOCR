"""PIL-only post-rasterization image augmentation.

Applied after bboxes are extracted from the PDF, so pixel transforms
do not invalidate annotations.
"""

from __future__ import annotations

import io
import random

from PIL import Image, ImageEnhance, ImageFilter


def augment(img: Image.Image, rng: random.Random, cfg: dict | None = None) -> Image.Image:
    """Apply random augmentation to a rasterized page image.

    Args:
        img: PIL Image (RGB).
        rng: Random instance for reproducibility.
        cfg: Optional augmentation config dict with keys:
            blur_sigma_max, brightness_jitter, contrast_jitter.

    Returns:
        Augmented PIL Image.
    """
    cfg = cfg or {}
    blur_max = cfg.get("blur_sigma_max", 1.5)
    brightness_jitter = cfg.get("brightness_jitter", 0.15)
    contrast_jitter = cfg.get("contrast_jitter", 0.15)

    # Gaussian blur (always, mild)
    sigma = rng.uniform(0.0, blur_max)
    if sigma > 0.1:
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

    # Brightness jitter
    if rng.random() < 0.6:
        factor = 1.0 + rng.uniform(-brightness_jitter, brightness_jitter)
        img = ImageEnhance.Brightness(img).enhance(factor)

    # Contrast jitter
    if rng.random() < 0.6:
        factor = 1.0 + rng.uniform(-contrast_jitter, contrast_jitter)
        img = ImageEnhance.Contrast(img).enhance(factor)

    return img


def encode_jpeg(img: Image.Image, rng: random.Random, quality_range: tuple[int, int] = (75, 95)) -> bytes:
    """JPEG-encode a PIL Image and return the bytes.

    The quality is sampled randomly from quality_range for realistic
    compression artifacts.
    """
    quality = rng.randint(*quality_range)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def laplacian_variance(img: Image.Image) -> float:
    """Laplacian variance of a PIL Image — blur/sharpness detection proxy.

    Mirrors synthdog_grounding/annotations.py:_laplacian_variance but
    accepts a PIL Image directly.
    """
    import numpy as np

    gray = np.array(img.convert("L")).astype(np.float32)
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    return float(np.var(lap))
