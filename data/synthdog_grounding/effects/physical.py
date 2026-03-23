"""
Tier 3 realistic document noise augmentation effects.

All effect classes expose a static ``apply(image, args)`` method that
accepts an RGBA uint8 numpy array and returns an RGBA uint8 numpy array.
Computation is done in float32 internally.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def apply_if_enabled(cfg: dict, effect_fn, image: np.ndarray) -> np.ndarray:
    """Call *effect_fn(image, args)* with probability ``cfg["prob"]``."""
    if np.random.rand() < cfg.get("prob", 0):
        return effect_fn(image, cfg.get("args", {}))
    return image


def _gaussian_blur_1d(arr: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    """Separable 1-D Gaussian blur along *axis*."""
    if sigma <= 0:
        return arr
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), axis=axis, arr=arr.astype(np.float32))


def _gaussian_blur_2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Separable 2-D Gaussian blur."""
    if sigma <= 0:
        return arr
    arr = _gaussian_blur_1d(arr, sigma, axis=1)
    arr = _gaussian_blur_1d(arr, sigma, axis=0)
    return arr


# ---------------------------------------------------------------------------
# Effect classes
# ---------------------------------------------------------------------------


class VignettingEffect:
    """Radial edge darkening that mimics scanner-optics light fall-off."""

    @staticmethod
    def apply(image: np.ndarray, args: dict) -> np.ndarray:
        H, W = image.shape[:2]
        intensity = np.random.uniform(*args.get("intensity", [30, 80]))
        shape = np.random.uniform(*args.get("shape", [1.5, 3.0]))

        Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
        Xn = (X - W / 2) / (W / 2)
        Yn = (Y - H / 2) / (H / 2)
        # Normalize so r==1 at image corners
        r = np.sqrt(Xn**2 + Yn**2) / np.sqrt(2)

        mask = (r**shape * intensity).astype(np.float32)  # 0 at center
        img = image.astype(np.float32)
        img[..., :3] = np.clip(img[..., :3] - mask[..., np.newaxis], 0, 255)
        return img.astype(np.uint8)


class BookSpineShadowEffect:
    """Linear edge-gradient simulating open-book spine curvature."""

    @staticmethod
    def apply(image: np.ndarray, args: dict) -> np.ndarray:
        H, W = image.shape[:2]
        intensity = np.random.uniform(*args.get("intensity", [30, 100]))
        width_frac = np.random.uniform(*args.get("width", [0.05, 0.20]))

        side_cfg = args.get("side", "random")
        side = np.random.choice(["left", "right"]) if side_cfg == "random" else side_cfg

        width_px = max(1, int(W * width_frac))
        gradient = np.zeros(W, dtype=np.float32)

        if side == "left":
            t = np.linspace(1.0, 0.0, width_px, dtype=np.float32)
            gradient[:width_px] = intensity * t
        else:
            t = np.linspace(0.0, 1.0, width_px, dtype=np.float32)
            gradient[W - width_px :] = intensity * t

        # Smooth the gradient
        gradient = _gaussian_blur_1d(gradient[np.newaxis, :], sigma=width_px * 0.15, axis=1)[0]

        img = image.astype(np.float32)
        img[..., :3] = np.clip(img[..., :3] - gradient[np.newaxis, :, np.newaxis], 0, 255)
        return img.astype(np.uint8)


class StainOverlayEffect:
    """Soft elliptical coffee/water stain blobs blended onto the paper layer."""

    @staticmethod
    def apply(image: np.ndarray, args: dict) -> np.ndarray:
        H, W = image.shape[:2]
        count_range = args.get("count", [1, 3])
        size_range = args.get("size", [0.03, 0.12])
        alpha_range = args.get("alpha", [0.05, 0.20])
        cr_range = args.get("color_r", [110, 180])
        cg_range = args.get("color_g", [80, 140])
        cb_range = args.get("color_b", [50, 110])

        count = np.random.randint(count_range[0], count_range[1] + 1)
        min_dim = min(H, W)
        img = image.astype(np.float32)

        Y, X = np.mgrid[0:H, 0:W].astype(np.float32)

        for _ in range(count):
            cx = np.random.uniform(0.1, 0.9) * W
            cy = np.random.uniform(0.1, 0.9) * H
            size = np.random.uniform(size_range[0], size_range[1]) * min_dim
            rx = size * np.random.uniform(0.6, 1.6)
            ry = size * np.random.uniform(0.6, 1.6)

            # Smooth paraboloid bump: 1 at center, 0 outside ellipse
            bump = np.maximum(0.0, 1.0 - ((X - cx) / rx) ** 2 - ((Y - cy) / ry) ** 2).astype(np.float32)

            # Feather the edge with a Gaussian blur
            sigma = max(1.0, size * 0.25)
            bump = _gaussian_blur_2d(bump, sigma)
            peak = bump.max()
            if peak < 1e-8:
                continue
            bump /= peak  # renormalize to [0, 1]

            alpha = np.random.uniform(alpha_range[0], alpha_range[1])
            cr = np.random.randint(cr_range[0], cr_range[1] + 1)
            cg = np.random.randint(cg_range[0], cg_range[1] + 1)
            cb = np.random.randint(cb_range[0], cb_range[1] + 1)
            stain_color = np.array([cr, cg, cb], dtype=np.float32)

            mask = bump[..., np.newaxis] * alpha
            img[..., :3] = img[..., :3] * (1.0 - mask) + stain_color * mask

        return np.clip(img, 0, 255).astype(np.uint8)


class FoldCreaseEffect:
    """Thin Gaussian-profiled dark line simulating a paper fold or crease."""

    @staticmethod
    def apply(image: np.ndarray, args: dict) -> np.ndarray:
        H, W = image.shape[:2]
        count_range = args.get("count", [1, 2])
        orientation_cfg = args.get("orientation", "random")
        width_range = args.get("width", [1, 4])
        intensity_range = args.get("intensity", [10, 45])

        count = np.random.randint(count_range[0], count_range[1] + 1)
        img = image.astype(np.float32)

        orientations = ["horizontal", "vertical", "diagonal"]

        for _ in range(count):
            half_w = np.random.uniform(width_range[0], width_range[1])
            intensity = np.random.uniform(intensity_range[0], intensity_range[1])

            orientation = np.random.choice(orientations) if orientation_cfg == "random" else orientation_cfg

            if orientation == "horizontal":
                pos = np.random.uniform(0.1, 0.9) * H
                Y = np.arange(H, dtype=np.float32)
                profile = np.exp(-0.5 * ((Y - pos) / half_w) ** 2) * intensity
                mask = profile[:, np.newaxis, np.newaxis]
            elif orientation == "vertical":
                pos = np.random.uniform(0.1, 0.9) * W
                X = np.arange(W, dtype=np.float32)
                profile = np.exp(-0.5 * ((X - pos) / half_w) ** 2) * intensity
                mask = profile[np.newaxis, :, np.newaxis]
            else:  # diagonal
                angle = np.random.uniform(25, 65)
                px = np.random.uniform(0.2, 0.8) * W
                py = np.random.uniform(0.2, 0.8) * H
                rad = np.radians(angle)
                Yg, Xg = np.mgrid[0:H, 0:W].astype(np.float32)
                # Perpendicular distance from a line through (px, py) at given angle
                dist = -(Xg - px) * np.sin(rad) + (Yg - py) * np.cos(rad)
                profile = np.exp(-0.5 * (dist / half_w) ** 2) * intensity
                mask = profile[..., np.newaxis]

            img[..., :3] = np.clip(img[..., :3] - mask, 0, 255)

        return img.astype(np.uint8)


class LowTonerStreakEffect:
    """Horizontal or vertical washed-out bands simulating low-toner printer artifacts."""

    @staticmethod
    def apply(image: np.ndarray, args: dict) -> np.ndarray:
        H, W = image.shape[:2]
        orientation_cfg = args.get("orientation", "random")
        count_range = args.get("count", [1, 4])
        width_range = args.get("width", [3, 20])
        intensity_range = args.get("intensity", [0.05, 0.20])

        count = np.random.randint(count_range[0], count_range[1] + 1)
        img = image.astype(np.float32)

        for _ in range(count):
            width = np.random.uniform(width_range[0], width_range[1])
            intensity = np.random.uniform(intensity_range[0], intensity_range[1])

            orientation = (
                np.random.choice(["horizontal", "vertical"]) if orientation_cfg == "random" else orientation_cfg
            )

            sigma = width * 0.5

            if orientation == "horizontal":
                pos = np.random.uniform(0.05, 0.95) * H
                Y = np.arange(H, dtype=np.float32)
                profile = np.exp(-0.5 * ((Y - pos) / sigma) ** 2)
                # Vary intensity along the streak length for a non-uniform look
                noise = np.random.uniform(0.75, 1.25, size=(W,)).astype(np.float32)
                lighten = (profile[:, np.newaxis] * noise[np.newaxis, :] * intensity * 255)[..., np.newaxis]
            else:  # vertical
                pos = np.random.uniform(0.05, 0.95) * W
                X = np.arange(W, dtype=np.float32)
                profile = np.exp(-0.5 * ((X - pos) / sigma) ** 2)
                noise = np.random.uniform(0.75, 1.25, size=(H,)).astype(np.float32)
                lighten = (noise[:, np.newaxis] * profile[np.newaxis, :] * intensity * 255)[..., np.newaxis]

            img[..., :3] = np.clip(img[..., :3] + lighten, 0, 255)

        return img.astype(np.uint8)


class MoireOverlayEffect:
    """Sinusoidal interference grid simulating halftone moiré from scanning."""

    @staticmethod
    def apply(image: np.ndarray, args: dict) -> np.ndarray:
        H, W = image.shape[:2]
        freq = np.random.uniform(*args.get("frequency", [0.02, 0.06]))
        alpha = np.random.uniform(*args.get("alpha", [0.05, 0.15]))
        angle = np.random.uniform(*args.get("angle", [0, 90]))

        Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
        rad = np.radians(angle)
        Xr = X * np.cos(rad) + Y * np.sin(rad)
        Yr = -X * np.sin(rad) + Y * np.cos(rad)

        pattern = 0.5 + 0.5 * np.sin(2.0 * np.pi * freq * Xr) * np.sin(2.0 * np.pi * freq * Yr)

        img = image.astype(np.float32)
        img[..., :3] = np.clip(img[..., :3] - pattern[..., np.newaxis] * alpha * 255, 0, 255)
        return img.astype(np.uint8)


class WatermarkEffect:
    """Semi-transparent rotated text stamp (DRAFT, CONFIDENTIAL, etc.) overlaid on the page."""

    @staticmethod
    def apply(image: np.ndarray, args: dict) -> np.ndarray:
        from PIL import Image as PILImage
        from PIL import ImageDraw, ImageFont

        H, W = image.shape[:2]
        words = args.get("words", ["DRAFT", "CONFIDENTIAL", "COPY", "VOID", "SAMPLE"])
        font_path = args.get("font_path", None)
        font_size_frac = args.get("font_size_frac", [0.12, 0.20])
        alpha_range = args.get("alpha", [0.08, 0.25])
        angle_range = args.get("angle", [-45, 45])
        color = tuple(int(c) for c in args.get("color", [60, 60, 60]))

        word = words[np.random.randint(len(words))]
        angle = float(np.random.uniform(angle_range[0], angle_range[1]))
        alpha = float(np.random.uniform(alpha_range[0], alpha_range[1]))
        font_size = max(int(min(H, W) * np.random.uniform(font_size_frac[0], font_size_frac[1])), 12)

        try:
            pil_font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except Exception:
            pil_font = ImageFont.load_default()

        overlay = PILImage.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        bbox = draw.textbbox((0, 0), word, font=pil_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (W - text_w) / 2 - bbox[0]
        y = (H - text_h) / 2 - bbox[1]

        alpha_int = int(alpha * 255)
        draw.text((x, y), word, fill=(*color, alpha_int), font=pil_font)

        overlay = overlay.rotate(-angle, expand=False)

        base = PILImage.fromarray(np.clip(image, 0, 255).astype(np.uint8), "RGBA")
        result = PILImage.alpha_composite(base, overlay)
        return np.array(result).astype(np.uint8)
