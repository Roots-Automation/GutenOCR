"""
Empirical investigation of the luminance threshold used in content.py.

The current code (line 73) uses 0.5 as the crossover between dark and light text:
    gray_range = [0, 64] if lum > 0.5 else [191, 255]

The WCAG-optimal crossover is ≈ 0.179 (equal contrast against dark/light text).
For backgrounds with lum ∈ (0.179, 0.5], the current code chooses *light* text
even though *dark* text would give higher contrast.

Usage:
    uv run python analyze_luminance_threshold.py
    uv run python analyze_luminance_threshold.py --generate 20
"""

import argparse

import numpy as np

# ---------------------------------------------------------------------------
# WCAG helpers
# ---------------------------------------------------------------------------


def _channel(c: float) -> float:
    """Linearise a single 0-1 sRGB channel."""
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _relative_luminance_float(r: float, g: float, b: float) -> float:
    """Relative luminance from 0-1 floats."""
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def _relative_luminance_uint8(r: int, g: int, b: int) -> float:
    return _relative_luminance_float(r / 255.0, g / 255.0, b / 255.0)


def _wcag_ratio(lum_a: float, lum_b: float) -> float:
    """WCAG 2.x contrast ratio (always ≥ 1)."""
    lighter = max(lum_a, lum_b)
    darker = min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


def _gray_lum(g: int) -> float:
    """Relative luminance of a gray pixel with value g ∈ [0, 255]."""
    return _relative_luminance_float(g / 255.0, g / 255.0, g / 255.0)


# Precompute luminance for the two gray ranges once.
_DARK_TEXT_LUMS = np.array([_gray_lum(g) for g in range(0, 65)])  # [0, 64]
_LIGHT_TEXT_LUMS = np.array([_gray_lum(g) for g in range(191, 256)])  # [191, 255]

WCAG_THRESHOLD_OPT = 0.179  # WCAG crossover
CURRENT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Section 1 – Luminance distribution (Monte Carlo)
# ---------------------------------------------------------------------------


def section1_luminance_distribution(n: int = 2_000_000) -> None:
    print("=" * 60)
    print("Section 1 — Luminance Distribution (Monte Carlo)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, size=(n, 3))

    # Vectorised relative luminance
    r = rgb[:, 0] / 255.0
    g = rgb[:, 1] / 255.0
    b = rgb[:, 2] / 255.0

    def lin(c):
        out = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return out

    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)

    frac_low = np.mean(lum < WCAG_THRESHOLD_OPT)
    frac_affected = np.mean((lum >= WCAG_THRESHOLD_OPT) & (lum <= CURRENT_THRESHOLD))
    frac_high = np.mean(lum > CURRENT_THRESHOLD)

    # The paper background is 50% colored, 50% pure white (lum=1.0).
    # Pure white has lum > 0.5, so it is unaffected.
    # The "affected" fraction comes only from the colored half.
    overall_affected = 0.5 * frac_affected

    print(f"N = {n:,} uniform-RGB samples\n")
    print(f"  lum < {WCAG_THRESHOLD_OPT}          (always dark bg, correct) : {frac_low * 100:6.2f}%")
    print(f"  lum ∈ ({WCAG_THRESHOLD_OPT}, {CURRENT_THRESHOLD}]  (affected zone)         : {frac_affected * 100:6.2f}%")
    print(f"  lum > {CURRENT_THRESHOLD}          (light bg, correct)       : {frac_high * 100:6.2f}%")
    print()
    print("Accounting for 50% color / 50% white paper split:")
    print(f"  Overall affected fraction of all paper samples    : {overall_affected * 100:6.2f}%")
    print()
    return frac_affected, overall_affected


# ---------------------------------------------------------------------------
# Section 2 – WCAG contrast analysis
# ---------------------------------------------------------------------------


def _min_wcag_for_bg(lum_bg: float, gray_range: list) -> float:
    """Minimum possible WCAG ratio for a given background lum and gray range."""
    lo, hi = gray_range
    text_lums_arr = np.array([_gray_lum(g) for g in range(lo, hi + 1)])
    ratios = np.array([_wcag_ratio(lum_bg, tl) for tl in text_lums_arr])
    return float(ratios.min())


def _max_wcag_for_bg(lum_bg: float, gray_range: list) -> float:
    lo, hi = gray_range
    text_lums_arr = np.array([_gray_lum(g) for g in range(lo, hi + 1)])
    ratios = np.array([_wcag_ratio(lum_bg, tl) for tl in text_lums_arr])
    return float(ratios.max())


def section2_wcag_analysis() -> None:
    print("=" * 60)
    print("Section 2 — WCAG Contrast Analysis")
    print("=" * 60)
    print(f"Current threshold : {CURRENT_THRESHOLD}")
    print(f"Optimal threshold : {WCAG_THRESHOLD_OPT}")
    print()

    # Spot table
    spot_lums = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    header = f"{'lum_bg':>6}  {'min WCAG current':>17}  {'min WCAG optimal':>17}  {'delta':>7}"
    print(header)
    print("-" * len(header))

    for lum_bg in spot_lums:
        # Current choice: lum > 0.5 → dark text; else light text
        # For lum_bg ≤ 0.5, current gives light text [191,255]
        # Optimal: lum_bg > 0.179 → dark text; else light text
        current_range = [0, 64] if lum_bg > CURRENT_THRESHOLD else [191, 255]
        optimal_range = [0, 64] if lum_bg > WCAG_THRESHOLD_OPT else [191, 255]

        min_cur = _min_wcag_for_bg(lum_bg, current_range)
        min_opt = _min_wcag_for_bg(lum_bg, optimal_range)
        delta = min_opt - min_cur
        print(f"{lum_bg:>6.2f}  {min_cur:>14.2f}:1  {min_opt:>14.2f}:1  {delta:>+7.2f}")

    print()

    # Sweep over [0.179, 0.5] to find absolute worst case
    lum_sweep = np.arange(WCAG_THRESHOLD_OPT, CURRENT_THRESHOLD + 0.001, 0.001)
    min_ratios_current = []
    min_ratios_optimal = []
    for lum_bg in lum_sweep:
        current_range = [0, 64] if lum_bg > CURRENT_THRESHOLD else [191, 255]
        optimal_range = [0, 64] if lum_bg > WCAG_THRESHOLD_OPT else [191, 255]
        min_ratios_current.append(_min_wcag_for_bg(lum_bg, current_range))
        min_ratios_optimal.append(_min_wcag_for_bg(lum_bg, optimal_range))

    worst_current = min(min_ratios_current)
    worst_optimal = min(min_ratios_optimal)
    worst_current_lum = lum_sweep[int(np.argmin(min_ratios_current))]
    worst_optimal_lum = lum_sweep[int(np.argmin(min_ratios_optimal))]

    print("Minimum WCAG ratio in affected zone (current threshold):")
    print(f"  {worst_current:.2f}:1  at lum_bg ≈ {worst_current_lum:.3f}")
    print("Minimum WCAG ratio in affected zone (optimal threshold):")
    print(f"  {worst_optimal:.2f}:1  at lum_bg ≈ {worst_optimal_lum:.3f}")
    print()

    # Does correcting ever hurt?
    degraded = [
        (lum_sweep[i], min_ratios_optimal[i], min_ratios_current[i])
        for i in range(len(lum_sweep))
        if min_ratios_optimal[i] < min_ratios_current[i] - 1e-6
    ]
    if degraded:
        print("WARNING: optimal threshold produces worse contrast at some lum_bg values:")
        for lum_bg, opt, cur in degraded[:5]:
            print(f"  lum_bg={lum_bg:.3f}  optimal={opt:.2f}:1  current={cur:.2f}:1")
    else:
        print("Correcting to 0.179 never produces worse contrast than the current threshold.")
    print()

    return worst_current, worst_optimal


# ---------------------------------------------------------------------------
# Section 3 – Hard-sample fraction
# ---------------------------------------------------------------------------


def section3_hard_sample_fraction(n: int = 2_000_000) -> None:
    print("=" * 60)
    print("Section 3 — Hard-Sample Fraction")
    print("=" * 60)

    rng = np.random.default_rng(0)

    # Simulate the 50/50 split: half pure white, half uniform color
    n_white = n // 2
    n_color = n - n_white

    # White background: lum = 1.0 → dark text [0, 64]
    white_lum = 1.0
    # For white bg with dark text, worst case is text at gray=64
    white_min_wcag = _wcag_ratio(white_lum, _gray_lum(64))  # ≈ 15.3

    # Colored backgrounds
    rgb_color = rng.integers(0, 256, size=(n_color, 3))
    r = rgb_color[:, 0] / 255.0
    g = rgb_color[:, 1] / 255.0
    b = rgb_color[:, 2] / 255.0

    def lin(c):
        return np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)

    # Current rule: lum > 0.5 → dark text (worst lum at gray=64); else light (worst at gray=191)
    # We compute the MINIMUM wcag for each sample (worst-case text in range)
    min_wcag_color = np.where(
        lum > CURRENT_THRESHOLD,
        np.array([_wcag_ratio(bg, _gray_lum(64)) for bg in lum]),  # dark text worst case
        np.array([_wcag_ratio(bg, _gray_lum(191)) for bg in lum]),  # light text worst case
    )

    # Combine
    all_min_wcag = np.concatenate(
        [
            np.full(n_white, white_min_wcag),
            min_wcag_color,
        ]
    )

    frac_below_4_5 = np.mean(all_min_wcag < 4.5)
    frac_below_3_0 = np.mean(all_min_wcag < 3.0)

    print("(Using worst-case text luminance within chosen gray range)")
    print(f"  Samples with min contrast < 4.5:1 : {frac_below_4_5 * 100:6.2f}%")
    print(f"  Samples with min contrast < 3.0:1 : {frac_below_3_0 * 100:6.2f}%")
    print()

    # Break down by zone
    frac_affected_low = np.mean(min_wcag_color[(lum >= WCAG_THRESHOLD_OPT) & (lum <= CURRENT_THRESHOLD)] < 3.0)
    frac_correct_low = (
        np.mean(min_wcag_color[lum < WCAG_THRESHOLD_OPT] < 3.0) if np.any(lum < WCAG_THRESHOLD_OPT) else float("nan")
    )
    frac_correct_high = (
        np.mean(min_wcag_color[lum > CURRENT_THRESHOLD] < 3.0) if np.any(lum > CURRENT_THRESHOLD) else float("nan")
    )

    print("  Breakdown of min contrast < 3.0:1 by zone (colored bg only):")
    print(f"    lum ∈ ({WCAG_THRESHOLD_OPT}, {CURRENT_THRESHOLD}] (affected) : {frac_affected_low * 100:6.2f}%")
    print(f"    lum < {WCAG_THRESHOLD_OPT}                (dark, correct) : {frac_correct_low * 100:6.2f}%")
    print(f"    lum > {CURRENT_THRESHOLD}                (light, correct): {frac_correct_high * 100:6.2f}%")
    print()

    return frac_below_4_5, frac_below_3_0


# ---------------------------------------------------------------------------
# Section 4 – Visual samples (optional)
# ---------------------------------------------------------------------------


def section4_visual_samples(n: int) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("PIL not available — skipping visual output.")
        return

    import os

    out_dir = "outputs/luminance_threshold_check"
    os.makedirs(out_dir, exist_ok=True)

    def _lin_scalar(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    def _lum_uint8(r, g, b):
        return 0.2126 * _lin_scalar(r / 255) + 0.7152 * _lin_scalar(g / 255) + 0.0722 * _lin_scalar(b / 255)

    rng = np.random.default_rng(7)
    samples = []
    while len(samples) < n:
        r, g, b = rng.integers(0, 256, size=3)
        lum = _lum_uint8(int(r), int(g), int(b))
        if WCAG_THRESHOLD_OPT < lum < CURRENT_THRESHOLD:
            samples.append((int(r), int(g), int(b), lum))

    W, H = 400, 100
    for i, (r, g, b, lum) in enumerate(samples):
        img = Image.new("RGB", (W * 2, H))
        draw = ImageDraw.Draw(img)

        # Left panel: current (wrong) — light text on medium background
        current_gray = 223  # midpoint of [191, 255]
        draw.rectangle([0, 0, W - 1, H - 1], fill=(r, g, b))
        draw.text((10, 10), f"Current (light text)\nbg lum={lum:.3f}", fill=(current_gray, current_gray, current_gray))

        # Right panel: optimal (correct) — dark text on medium background
        optimal_gray = 32  # midpoint of [0, 64]
        draw.rectangle([W, 0, W * 2 - 1, H - 1], fill=(r, g, b))
        draw.text(
            (W + 10, 10), f"Optimal (dark text)\nbg lum={lum:.3f}", fill=(optimal_gray, optimal_gray, optimal_gray)
        )

        ratio_cur = _wcag_ratio(lum, _gray_lum(current_gray))
        ratio_opt = _wcag_ratio(lum, _gray_lum(optimal_gray))
        draw.text((10, 60), f"WCAG {ratio_cur:.2f}:1", fill=(current_gray, current_gray, current_gray))
        draw.text((W + 10, 60), f"WCAG {ratio_opt:.2f}:1", fill=(optimal_gray, optimal_gray, optimal_gray))

        path = os.path.join(out_dir, f"sample_{i:03d}_lum{lum:.3f}.png")
        img.save(path)

    print(f"Saved {n} side-by-side images to {out_dir}/")


# ---------------------------------------------------------------------------
# Post-processing analysis note
# ---------------------------------------------------------------------------


def section_postprocessing_note() -> None:
    print("=" * 60)
    print("Note — Post-processing effects on contrast")
    print("=" * 60)
    print(
        "After text color is chosen, the pipeline applies:\n"
        "  • Brightness adjustment  ±48 (additive, shifts both bg and text)\n"
        "  • Shadow overlay          (darkens bg locally → helps dark text)\n"
        "  • Contrast multiplier     ×1.0–1.5 (amplifies existing contrast)\n"
        "  • Blur                    (reduces effective contrast at edges)\n"
        "\n"
        "In the affected zone (lum ∈ (0.179, 0.5]):\n"
        "  • Light text on medium-gray bg already has low contrast.\n"
        "  • Brightness ±48 can push bg toward 0 or 255; if bg darkens,\n"
        "    light-text contrast improves, but the effect is stochastic.\n"
        "  • Contrast ×1.0–1.5 scales the existing (low) ratio — it cannot\n"
        "    recover from a fundamentally wrong text-color choice.\n"
        "  • Net effect: post-processing is NOT a reliable mitigating factor.\n"
    )


# ---------------------------------------------------------------------------
# Decision summary
# ---------------------------------------------------------------------------


def decision_summary(frac_affected: float, overall_affected: float, worst_current: float, worst_optimal: float) -> None:
    print("=" * 60)
    print("Decision Summary")
    print("=" * 60)
    minor_zone = overall_affected < 0.05
    good_contrast = worst_current >= 3.0

    print(f"  Overall affected fraction  : {overall_affected * 100:.2f}%  (threshold < 5%: {minor_zone})")
    print(f"  Worst-case WCAG (current)  : {worst_current:.2f}:1  (threshold ≥ 3:1: {good_contrast})")
    print()

    if minor_zone and good_contrast:
        print("VERDICT: Minor issue — affected zone < 5% AND min contrast ≥ 3:1.")
        print("  Recommendation: Document as minor; leave threshold at 0.5.")
    else:
        reasons = []
        if not minor_zone:
            reasons.append(f"affected zone is {overall_affected * 100:.2f}% ≥ 5%")
        if not good_contrast:
            reasons.append(f"worst-case contrast {worst_current:.2f}:1 < 3:1")
        print(f"VERDICT: Non-trivial issue ({'; '.join(reasons)}).")
        print("  Recommendation: Fix threshold to 0.179 in content.py lines 28 and 73.")
        print(f"  (Optimal threshold raises worst-case contrast to {worst_optimal:.2f}:1)")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Luminance threshold analysis for content.py")
    parser.add_argument(
        "--generate",
        type=int,
        default=0,
        metavar="N",
        help="Generate N side-by-side visual samples in outputs/luminance_threshold_check/",
    )
    parser.add_argument("--n", type=int, default=2_000_000, help="Monte Carlo sample count (default 2 000 000)")
    args = parser.parse_args()

    print()
    frac_affected, overall_affected = section1_luminance_distribution(args.n)
    worst_current, worst_optimal = section2_wcag_analysis()
    section3_hard_sample_fraction(args.n)
    section_postprocessing_note()
    decision_summary(frac_affected, overall_affected, worst_current, worst_optimal)

    if args.generate > 0:
        print("=" * 60)
        print(f"Section 4 — Visual Samples (N={args.generate})")
        print("=" * 60)
        section4_visual_samples(args.generate)


if __name__ == "__main__":
    main()
