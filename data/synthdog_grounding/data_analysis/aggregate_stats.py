#!/usr/bin/env python3
"""Aggregate per-sample statistics from multiple .stats.csv files.

Reads CSV files produced by generate_stats.py, computes summary statistics,
and writes an aggregated CSV plus a summary JSON.

Usage:
    python aggregate_stats.py -d /path/to/directory -o aggregated_stats
    python aggregate_stats.py file1.stats.csv file2.stats.csv -o aggregated_stats
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def _safe_float(value: str) -> float | None:
    """Convert a CSV cell to float, returning None for empty/invalid values."""
    if value is None or value == "" or value == "None":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _stats_for(values: list[float]) -> dict[str, float]:
    """Compute min/max/mean/median/std for a list of floats."""
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
    n = len(values)
    sorted_vals = sorted(values)
    mean = sum(sorted_vals) / n
    if n % 2 == 0:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    else:
        median = sorted_vals[n // 2]
    if n > 1:
        variance = sum((x - mean) ** 2 for x in sorted_vals) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return {
        "min": round(sorted_vals[0], 4),
        "max": round(sorted_vals[-1], 4),
        "mean": round(mean, 4),
        "median": round(median, 4),
        "std": round(std, 4),
    }


def find_stats_files(directory: Path) -> list[Path]:
    """Find all .stats.csv files in a directory."""
    return sorted(directory.glob("*.stats.csv"))


def read_stats_csv(path: Path) -> list[dict[str, str]]:
    """Read a .stats.csv file and return rows as dicts."""
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def aggregate_stats(stats_files: list[Path], output_path: Path | None = None) -> None:
    """Aggregate statistics from multiple .stats.csv files."""
    all_rows: list[dict[str, str]] = []

    for stats_path in stats_files:
        print(f"Processing {stats_path.name}...", file=sys.stderr)
        rows = read_stats_csv(stats_path)
        source = stats_path.name.replace(".stats.csv", "")
        for row in rows:
            row["source"] = source
            row["stats_file"] = stats_path.name
        all_rows.extend(rows)

    if not all_rows:
        print("No valid stats found in any files!", file=sys.stderr)
        return

    # Determine all fieldnames (union of all keys, preserving order)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in all_rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    # Build summary
    summary: dict = {}
    summary["total_samples"] = len(all_rows)
    summary["unique_sources"] = len(set(r.get("source", "") for r in all_rows))

    # Totals
    total_lines = sum(int(r.get("num_lines", 0) or 0) for r in all_rows)
    total_words = sum(int(r.get("num_words", 0) or 0) for r in all_rows)
    summary["total_lines"] = total_lines
    summary["total_words"] = total_words

    # Overlap analysis
    samples_with_bbox = sum(1 for r in all_rows if int(r.get("lines_with_bbox", 0) or 0) > 0)
    samples_with_high_overlap = sum(1 for r in all_rows if int(r.get("high_overlap_pairs", 0) or 0) > 0)
    summary["samples_with_bbox"] = samples_with_bbox
    summary["samples_with_high_overlap"] = samples_with_high_overlap
    summary["pct_samples_with_high_overlap"] = round(100 * samples_with_high_overlap / max(samples_with_bbox, 1), 2)

    # Statistical summaries for numeric columns
    numeric_cols = [
        "num_lines",
        "num_words",
        "avg_words_per_line",
        "max_iou",
        "high_overlap_pairs",
        "width_mean",
        "height_mean",
        "aspect_ratio_mean",
        # Quality metric columns
        "min_line_contrast",
        "mean_line_contrast",
        "min_line_bbox_area_px",
        "min_word_bbox_area_px",
        "degenerate_line_count",
        "degenerate_word_count",
        "textbox_null_count",
        "textbox_total_count",
    ]

    for col in numeric_cols:
        values = []
        for r in all_rows:
            v = _safe_float(r.get(col, ""))
            if v is not None:
                values.append(v)
        if values:
            summary[f"{col}_stats"] = _stats_for(values)

    # Identify problematic samples
    problematic = []
    for r in all_rows:
        high_overlap = int(r.get("high_overlap_pairs", 0) or 0)
        max_iou = _safe_float(r.get("max_iou", "")) or 0.0
        num_lines = int(r.get("num_lines", 0) or 0)
        if high_overlap > 10 or max_iou > 0.5 or num_lines > 100:
            problematic.append(
                {
                    "sample_id": r.get("sample_id", ""),
                    "source": r.get("source", ""),
                    "high_overlap_pairs": high_overlap,
                    "max_iou": max_iou,
                    "num_lines": num_lines,
                }
            )
    problematic.sort(key=lambda x: x["max_iou"], reverse=True)
    summary["problematic_samples"] = {
        "count": len(problematic),
        "samples": problematic[:20],
    }

    # Write outputs
    if output_path:
        csv_path = output_path.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Aggregated CSV saved to: {csv_path}", file=sys.stderr)

        json_path = output_path.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary statistics saved to: {json_path}", file=sys.stderr)

    # Print summary to stdout
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate statistics from multiple .stats.csv files")
    parser.add_argument("stats_files", nargs="*", help="Paths to .stats.csv files")
    parser.add_argument("-d", "--directory", type=str, help="Directory to scan for .stats.csv files")
    parser.add_argument("-o", "--output", type=str, help="Output path prefix (creates .csv and .json)")

    args = parser.parse_args()

    if args.directory:
        directory = Path(args.directory).resolve()
        if not directory.exists() or not directory.is_dir():
            print(f"Directory not found: {directory}", file=sys.stderr)
            sys.exit(1)
        stats_paths = find_stats_files(directory)
        if not stats_paths:
            print(f"No .stats.csv files found in {directory}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(stats_paths)} .stats.csv files in {directory}", file=sys.stderr)
    elif args.stats_files:
        stats_paths = [Path(p).resolve() for p in args.stats_files]
    else:
        print("Either provide .stats.csv files or use --directory option", file=sys.stderr)
        sys.exit(1)

    for p in stats_paths:
        if not p.exists():
            print(f"Stats file not found: {p}", file=sys.stderr)
            sys.exit(1)

    output_path = Path(args.output).resolve() if args.output else None

    try:
        aggregate_stats(stats_paths, output_path)
    except Exception as e:
        print(f"Error aggregating stats: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
