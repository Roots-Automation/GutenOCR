"""
Empirical test: generate N samples with no contrast rejection gate,
collect min_line_contrast_ratio for every sample, and report the
full distribution + what each candidate threshold would reject.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pillow_compat  # noqa: F401, E402
import yaml  # noqa: E402
from template import SynthDoG  # noqa: E402


def main(n: int = 500) -> None:
    with open("config/config_en.yaml") as f:
        config = yaml.safe_load(f)

    # Force threshold to 0 so nothing is rejected — we want the raw distribution
    config["min_contrast_ratio"] = 0.0

    t = SynthDoG(config)

    ratios = []
    for i in range(n):
        data = t.generate()
        r = data["quality_metrics"].get("min_line_contrast_ratio")
        if r is not None:
            ratios.append(r)
        if (i + 1) % 50 == 0:
            print(f"  generated {i + 1}/{n}", flush=True)

    ratios = sorted(ratios)
    arr = np.array(ratios)
    print(f"\nn={len(arr)}  (of {n} generated, {n - len(arr)} had null ratio)")
    print(f"min={arr.min():.3f}  max={arr.max():.3f}  mean={arr.mean():.3f}  median={np.median(arr):.3f}")
    print()

    print("Percentile distribution:")
    for p in [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]:
        print(f"  p{p:3d}: {np.percentile(arr, p):.3f}")

    print()
    print("Rejection rate at candidate thresholds:")
    for threshold in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        rejected = int(np.sum(arr < threshold))
        pct = 100.0 * rejected / len(arr)
        print(f"  threshold={threshold:.1f}: {rejected:4d}/{len(arr)} rejected ({pct:.1f}%)")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    main(n)
