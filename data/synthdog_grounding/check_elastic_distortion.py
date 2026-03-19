#!/usr/bin/env python3
"""Empirically quantify elastic distortion annotation misalignment.

Captures pre-distortion and post-distortion composited images for N samples,
then reports:
  - max/p95/mean absolute pixel delta across the full image
  - per-line text pixel coverage (fraction of text pixels inside bbox, post-distortion)
  - per-line centroid drift in pixels
  - comparison baseline: same metrics for motion blur and Gaussian blur

Usage:
    cd data/synthdog_grounding
    uv run python check_elastic_distortion.py \\
        --config config/config_en.yaml \\
        --n-samples 50 \\
        --output ./outputs/elastic_audit \\
        [--save-visuals]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from synthtiger import layers as synthtiger_layers

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pillow_compat  # noqa: E402, F401, I001

from serialization import LineAnnotation  # noqa: E402
from template import SynthDoG  # noqa: E402


# ---------------------------------------------------------------------------
# Patched SynthDoG that captures pre/post elastic distortion images
# ---------------------------------------------------------------------------


class _PatchedSynthDoG(SynthDoG):
    """SynthDoG subclass that stores pre- and post-elastic-distortion images."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_pre: np.ndarray | None = None
        self._last_post: np.ndarray | None = None

    def _render(self, document_group, bg_layer, size: tuple[int, int]) -> np.ndarray:
        layer = synthtiger_layers.Group([*document_group.layers, bg_layer]).merge()
        # Capture image before elastic distortion
        self._last_pre = layer.output(bbox=[0, 0, *size]).copy()
        # Apply elastic distortion (modifies layer.image in-place)
        self.document.elastic_distortion.apply([layer])
        # Capture image immediately after elastic distortion, before other effects
        self._last_post = layer.output(bbox=[0, 0, *size]).copy()
        # Apply remaining template-level effects (motion blur, Gaussian blur, etc.)
        self.effect.apply([layer])
        return layer.output(bbox=[0, 0, *size])


# ---------------------------------------------------------------------------
# Pixel delta stats helpers
# ---------------------------------------------------------------------------


def pixel_delta_stats(pre: np.ndarray, post: np.ndarray) -> dict:
    """Compute per-pixel absolute difference stats between two images.

    Args:
        pre: float32 (H, W, C) pre-distortion image
        post: float32 (H, W, C) post-distortion image

    Returns:
        dict with keys: max, p95, mean (all on 0-255 scale)
    """
    diff = np.abs(post.astype(np.float64) - pre.astype(np.float64))
    # Collapse channels: take max abs diff per pixel
    per_pixel = diff.max(axis=-1) if diff.ndim == 3 else diff
    return {
        "max": float(per_pixel.max()),
        "p95": float(np.percentile(per_pixel, 95)),
        "mean": float(per_pixel.mean()),
    }


# ---------------------------------------------------------------------------
# Text pixel detection and coverage / centroid drift helpers
# ---------------------------------------------------------------------------


def _gray(img: np.ndarray) -> np.ndarray:
    """Convert float32 (H, W, C) image to grayscale float (H, W)."""
    if img.ndim == 2:
        return img.astype(np.float64)
    if img.shape[-1] >= 3:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float64)
    return img[..., 0].astype(np.float64)


def _text_mask(gray_roi: np.ndarray) -> np.ndarray:
    """Return boolean mask of text pixels (darker than background estimate)."""
    if gray_roi.size == 0:
        return np.zeros_like(gray_roi, dtype=bool)
    # Background = 80th percentile of the roi (paper is typically light)
    bg = np.percentile(gray_roi, 80)
    # Threshold: more than 12% darker than background OR absolute < 200
    threshold = min(bg * 0.88, bg - 20.0)
    return gray_roi < threshold


def _bbox_px(bbox: list[float], W: int, H: int) -> tuple[int, int, int, int]:
    """Convert normalized [x1,y1,x2,y2] bbox to integer pixel coords."""
    x1 = int(max(0, min(W - 1, round(bbox[0] * W))))
    y1 = int(max(0, min(H - 1, round(bbox[1] * H))))
    x2 = int(max(0, min(W - 1, round(bbox[2] * W))))
    y2 = int(max(0, min(H - 1, round(bbox[3] * H))))
    # Ensure at least 1px wide/tall
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    return x1, y1, x2, y2


def per_line_stats(
    pre: np.ndarray,
    post: np.ndarray,
    lines: list[LineAnnotation],
    W: int,
    H: int,
) -> list[dict]:
    """Compute per-line coverage and centroid drift stats.

    For each line:
    - text_pixel_coverage: fraction of bbox text pixels (from pre) still
      detected as text in the same bbox region of post
    - centroid_drift_px: distance between text-pixel centroids in pre vs post
      within the annotated bbox

    Args:
        pre: float32 (H, W, C) pre-distortion image
        post: float32 (H, W, C) post-distortion image
        lines: list of LineAnnotation (bbox is normalized [x1,y1,x2,y2])
        W, H: image dimensions

    Returns:
        list of dicts with keys: text_pixel_coverage, centroid_drift_px
    """
    pre_g = _gray(pre)
    post_g = _gray(post)

    results = []
    for ln in lines:
        x1, y1, x2, y2 = _bbox_px(ln.bbox, W, H)
        pre_roi = pre_g[y1:y2, x1:x2]
        post_roi = post_g[y1:y2, x1:x2]

        pre_mask = _text_mask(pre_roi)
        post_mask = _text_mask(post_roi)

        n_pre = int(pre_mask.sum())
        if n_pre == 0:
            continue

        # Fraction of pre-image text pixels that are still detected as text
        # at the same spatial positions in the post-distortion image.
        # Bounded [0, 1]: < 1.0 means some text pixels look lighter post-distortion.
        overlap = int((pre_mask & post_mask).sum())
        coverage = overlap / n_pre

        # Centroid drift: difference between pre and post text pixel centroids
        pre_ys, pre_xs = np.where(pre_mask)
        post_ys, post_xs = np.where(post_mask)

        if len(pre_ys) == 0 or len(post_ys) == 0:
            drift = 0.0
        else:
            pre_cy, pre_cx = pre_ys.mean(), pre_xs.mean()
            post_cy, post_cx = post_ys.mean(), post_xs.mean()
            drift = float(np.hypot(post_cx - pre_cx, post_cy - pre_cy))

        results.append({"text_pixel_coverage": coverage, "centroid_drift_px": drift})

    return results


# ---------------------------------------------------------------------------
# Blur baseline helpers
# ---------------------------------------------------------------------------


def apply_motion_blur(img: np.ndarray, k: int, angle_deg: float) -> np.ndarray:
    """Apply motion blur with kernel size k and angle (degrees) to a float32 image."""
    # Create a line kernel
    kernel = np.zeros((k, k), dtype=np.float32)
    center = k // 2
    angle_rad = np.deg2rad(angle_deg)
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    for i in range(k):
        t = i - center
        px = int(round(center + t * dx))
        py = int(round(center + t * dy))
        if 0 <= px < k and 0 <= py < k:
            kernel[py, px] = 1.0
    row_sums = kernel.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    kernel = kernel / kernel.sum() if kernel.sum() > 0 else kernel

    result = img.copy()
    if img.ndim == 3:
        for c in range(img.shape[-1]):
            from scipy.ndimage import convolve

            result[..., c] = convolve(img[..., c].astype(np.float64), kernel).astype(np.float32)
    else:
        from scipy.ndimage import convolve

        result = convolve(img.astype(np.float64), kernel).astype(np.float32)
    return result


def apply_gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur with given sigma to a float32 image."""
    if sigma <= 0:
        return img.copy()
    result = img.copy().astype(np.float64)
    if img.ndim == 3:
        for c in range(img.shape[-1]):
            result[..., c] = gaussian_filter(result[..., c], sigma=sigma)
    else:
        result = gaussian_filter(result, sigma=sigma)
    return result.astype(np.float32)


def blur_baseline_stats(
    pre_images: list[np.ndarray],
    all_lines: list[list[LineAnnotation]],
    sizes: list[tuple[int, int]],
    blur_type: str,
) -> dict:
    """Compute pixel delta and per-line stats for a blur baseline."""
    pixel_maxes, pixel_p95s, pixel_means = [], [], []
    coverages, drifts = [], []

    rng = np.random.default_rng(0)
    for pre, lines, (W, H) in zip(pre_images, all_lines, sizes):
        if blur_type == "motion":
            k = int(rng.integers(3, 6))
            angle = float(rng.uniform(0, 360))
            post = apply_motion_blur(pre, k, angle)
        elif blur_type == "gaussian":
            sigma = float(rng.uniform(0.0, 1.5))
            post = apply_gaussian_blur(pre, sigma)
        else:
            raise ValueError(f"Unknown blur_type: {blur_type}")

        pd = pixel_delta_stats(pre, post)
        pixel_maxes.append(pd["max"])
        pixel_p95s.append(pd["p95"])
        pixel_means.append(pd["mean"])

        for stat in per_line_stats(pre, post, lines, W, H):
            coverages.append(stat["text_pixel_coverage"])
            drifts.append(stat["centroid_drift_px"])

    return {
        "pixel_max": pixel_maxes,
        "pixel_p95": pixel_p95s,
        "pixel_mean": pixel_means,
        "coverage": coverages,
        "centroid_drift": drifts,
    }


# ---------------------------------------------------------------------------
# Visual output helpers
# ---------------------------------------------------------------------------


def _draw_bboxes_on_array(img_arr: np.ndarray, lines: list[LineAnnotation], W: int, H: int) -> Image.Image:
    """Draw normalized bboxes on a numpy image array and return a PIL Image."""
    # Clip to uint8 RGB
    rgb = np.clip(img_arr[..., :3], 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(pil_img)
    for ln in lines:
        x1, y1, x2, y2 = _bbox_px(ln.bbox, W, H)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
    return pil_img


def _diff_heatmap(pre: np.ndarray, post: np.ndarray) -> Image.Image:
    """Create an amplified per-pixel max-channel difference heatmap."""
    diff = np.abs(post.astype(np.float64) - pre.astype(np.float64))
    per_pixel = diff.max(axis=-1) if diff.ndim == 3 else diff
    # Amplify: scale so p99 → 200, clip to [0, 255]
    p99 = np.percentile(per_pixel, 99)
    scale = 200.0 / p99 if p99 > 0 else 1.0
    amplified = np.clip(per_pixel * scale, 0, 255).astype(np.uint8)
    # Render as a red-channel heatmap
    H, W = amplified.shape
    heatmap = np.zeros((H, W, 3), dtype=np.uint8)
    heatmap[..., 0] = amplified  # red channel
    return Image.fromarray(heatmap, mode="RGB")


def save_visuals(
    i: int,
    pre: np.ndarray,
    post: np.ndarray,
    lines: list[LineAnnotation],
    W: int,
    H: int,
    out_dir: Path,
) -> None:
    pre_img = _draw_bboxes_on_array(pre, lines, W, H)
    post_img = _draw_bboxes_on_array(post, lines, W, H)
    diff_img = _diff_heatmap(pre, post)
    pre_img.save(out_dir / f"{i:04d}_pre_distortion.jpg", quality=95)
    post_img.save(out_dir / f"{i:04d}_post_distortion.jpg", quality=95)
    diff_img.save(out_dir / f"{i:04d}_diff.png")


# ---------------------------------------------------------------------------
# Stats formatting helpers
# ---------------------------------------------------------------------------


def _summarize(values: list[float], label: str) -> str:
    if not values:
        return f"  {label}: no data"
    arr = np.array(values)
    return f"  {label}: mean={arr.mean():.3f}, p95={np.percentile(arr, 95):.3f}, max={arr.max():.3f}"


def print_stats(name: str, stats: dict) -> None:
    print(f"\n{name}:")
    print(_summarize(stats["pixel_max"], "max pixel delta    "))
    print(_summarize(stats["pixel_p95"], "p95 pixel delta    "))
    print(_summarize(stats["pixel_mean"], "mean pixel delta   "))
    if stats.get("coverage"):
        print(_summarize(stats["coverage"], "text pixel coverage"))
    if stats.get("centroid_drift"):
        print(_summarize(stats["centroid_drift"], "centroid drift (px)"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Empirically quantify elastic distortion annotation misalignment.")
    parser.add_argument("--config", type=Path, default=Path("config/config_en.yaml"))
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("./outputs/elastic_audit"))
    parser.add_argument(
        "--save-visuals",
        action="store_true",
        help="Save pre/post/diff images for first 20 samples",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load config
    config_path = args.config.resolve()
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Instantiate the patched template
    print(f"Loading SynthDoG from {config_path} ...")
    template = _PatchedSynthDoG(config)

    visuals_dir = args.output / "visuals"
    if args.save_visuals:
        visuals_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-sample stats
    elastic_pixel_maxes, elastic_pixel_p95s, elastic_pixel_means = [], [], []
    elastic_coverages, elastic_drifts = [], []

    pre_images: list[np.ndarray] = []
    all_lines: list[list[LineAnnotation]] = []
    sizes: list[tuple[int, int]] = []

    print(f"Generating {args.n_samples} samples ...")
    for i in range(args.n_samples):
        data = template.generate()

        pre = template._last_pre
        post = template._last_post
        lines: list[LineAnnotation] = data.get("lines", [])
        W, H = data["image"].shape[1], data["image"].shape[0]

        if pre is None or post is None:
            print(f"  [sample {i}] WARNING: pre/post images not captured, skipping")
            continue

        # Pixel delta stats
        pd = pixel_delta_stats(pre, post)
        elastic_pixel_maxes.append(pd["max"])
        elastic_pixel_p95s.append(pd["p95"])
        elastic_pixel_means.append(pd["mean"])

        # Per-line stats
        for stat in per_line_stats(pre, post, lines, W, H):
            elastic_coverages.append(stat["text_pixel_coverage"])
            elastic_drifts.append(stat["centroid_drift_px"])

        # Store for blur baselines
        pre_images.append(pre)
        all_lines.append(lines)
        sizes.append((W, H))

        # Save visuals for first 20 samples
        if args.save_visuals and i < 20:
            save_visuals(i, pre, post, lines, W, H, visuals_dir)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{args.n_samples} done")

    # Elastic distortion summary
    elastic_stats = {
        "pixel_max": elastic_pixel_maxes,
        "pixel_p95": elastic_pixel_p95s,
        "pixel_mean": elastic_pixel_means,
        "coverage": elastic_coverages,
        "centroid_drift": elastic_drifts,
    }
    print_stats(f"Elastic distortion stats across {len(elastic_pixel_maxes)} samples", elastic_stats)

    # Blur baselines
    print("\nComputing blur baselines ...")
    motion_stats = blur_baseline_stats(pre_images, all_lines, sizes, "motion")
    gaussian_stats = blur_baseline_stats(pre_images, all_lines, sizes, "gaussian")
    print_stats("Motion blur baseline (k=3–5, any angle)", motion_stats)
    print_stats("Gaussian blur baseline (sigma=0–1.5)", gaussian_stats)

    # Interpretation guide
    print("\n--- Interpretation thresholds ---")
    print("  max pixel delta < 2px       → elastic distortion negligible")
    print("  text_pixel_coverage > 0.98  → annotations overwhelmingly accurate")
    print("  centroid_drift_px < 1.0     → within JPEG compression noise floor")

    if args.save_visuals:
        print(f"\nVisuals saved to: {visuals_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
