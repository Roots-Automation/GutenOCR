"""Pixel-level augmentation effects for index_definition generator."""

import io

import cv2
import numpy as np
from PIL import Image


def apply_if_prob(cfg: dict, fn, image: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    if rng.random() < cfg.get("prob", 0):
        return fn(image, cfg.get("args", {}), rng), True
    return image, False


def apply_noise(image, args, rng):
    scale = rng.uniform(*args.get("scale", [0, 8]))
    noise = rng.normal(0, scale, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_erode(image, args, rng):
    k = int(rng.integers(args.get("k", [1, 3])[0], args.get("k", [1, 3])[1] + 1))
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(image, kernel)


def apply_dilate(image, args, rng):
    k = int(rng.integers(args.get("k", [1, 3])[0], args.get("k", [1, 3])[1] + 1))
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(image, kernel)


def apply_coarse_dropout(image, args, rng):
    p = rng.uniform(*args.get("p", [0.003, 0.015]))
    size_frac = rng.uniform(*args.get("size_percent", [0.05, 0.2]))
    H, W = image.shape[:2]
    patch_h = max(1, int(H * size_frac))
    patch_w = max(1, int(W * size_frac))
    out = image.copy()
    n_patches = max(1, int(p * H * W / (patch_h * patch_w)))
    for _ in range(n_patches):
        y = int(rng.integers(0, max(1, H - patch_h + 1)))
        x = int(rng.integers(0, max(1, W - patch_w + 1)))
        out[y : y + patch_h, x : x + patch_w] = 255
    return out


def apply_elastic_distortion(image, args, rng):
    from scipy.ndimage import gaussian_filter, map_coordinates

    alpha = rng.uniform(*args.get("alpha", [0, 0.5]))
    sigma = rng.uniform(*args.get("sigma", [0, 0.3]))
    if alpha == 0:
        return image
    H, W = image.shape[:2]
    sigma_px = max(sigma * min(H, W), 1.0)
    dx = gaussian_filter(rng.random((H, W)) * 2 - 1, sigma=sigma_px) * alpha * W
    dy = gaussian_filter(rng.random((H, W)) * 2 - 1, sigma=sigma_px) * alpha * H
    y, x = np.mgrid[0:H, 0:W]
    map_x = np.clip(x + dx, 0, W - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, H - 1).astype(np.float32)
    if image.ndim == 3:
        channels = [
            map_coordinates(image[:, :, c], [map_y, map_x], order=1, mode="reflect") for c in range(image.shape[2])
        ]
        return np.stack(channels, axis=2).astype(np.uint8)
    return map_coordinates(image, [map_y, map_x], order=1, mode="reflect").astype(np.uint8)


def apply_rgb_tint(image, args, rng):
    rgb_ranges = args.get("rgb", [[0, 255], [0, 255], [0, 255]])
    alpha = rng.uniform(*args.get("alpha", [0, 0.2]))
    tint = np.array([rng.uniform(*r) for r in rgb_ranges], dtype=np.float32)
    out = image.astype(np.float32) * (1 - alpha) + tint * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_grayscale(image, args, rng):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def apply_contrast(image, args, rng):
    alpha = rng.uniform(*args.get("alpha", [1.0, 1.5]))
    out = image.astype(np.float32) * alpha + 128 * (1 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_brightness(image, args, rng):
    beta = rng.uniform(*args.get("beta", [-32, 32]))
    return np.clip(image.astype(np.float32) + beta, 0, 255).astype(np.uint8)


def apply_motion_blur(image, args, rng):
    k_range = args.get("k", [3, 5])
    k = int(rng.integers(k_range[0], k_range[1] + 1))
    if k < 3:
        k = 3
    angle = rng.uniform(*args.get("angle", [0, 360]))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return cv2.filter2D(image, -1, kernel)


def apply_gaussian_blur(image, args, rng):
    sigma = rng.uniform(*args.get("sigma", [0, 1.5]))
    if sigma <= 0:
        return image
    radius = int(np.ceil(3 * sigma))
    ksize = 2 * radius + 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def apply_resample(image, args, rng):
    scale = rng.uniform(*args.get("size", [0.4, 0.7]))
    H, W = image.shape[:2]
    small_h = max(1, int(H * scale))
    small_w = max(1, int(W * scale))
    pil = Image.fromarray(image)
    small = pil.resize((small_w, small_h), Image.BILINEAR)
    return np.array(small.resize((W, H), Image.BILINEAR))


def apply_jpeg_compression(image, args, rng):
    quality_range = args.get("compression", [10, 40])
    quality = int(rng.integers(quality_range[0], quality_range[1] + 1))
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


PIXEL_EFFECTS = [
    ("noise", apply_noise),
    ("erode", apply_erode),
    ("dilate", apply_dilate),
    ("coarse_dropout", apply_coarse_dropout),
    ("elastic_distortion", apply_elastic_distortion),
    ("rgb_tint", apply_rgb_tint),
    ("grayscale", apply_grayscale),
    ("contrast", apply_contrast),
    ("brightness", apply_brightness),
    ("motion_blur", apply_motion_blur),
    ("gaussian_blur", apply_gaussian_blur),
    ("resample", apply_resample),
    ("jpeg_compression", apply_jpeg_compression),
]


def apply_pixel_effects(image: np.ndarray, effects_cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Apply all pixel effects in sequence. Returns (image, provenance_dict)."""
    provenance = {}
    for name, fn in PIXEL_EFFECTS:
        cfg = effects_cfg.get(name, {})
        image, fired = apply_if_prob(cfg, fn, image, rng)
        provenance[name] = {"applied": fired}
    return image, provenance
