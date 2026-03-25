#!/usr/bin/env python3
"""Analyze SynthDoG data: per-sample statistics, aggregation, and batch processing.

Subcommands:
    python analyze.py stats <input>              # per-sample stats CSV
    python analyze.py aggregate -d <dir>         # aggregate .stats.csv files
    python analyze.py batch <dir> [--workers N]  # batch process all sources
    python analyze.py report <dir>               # quality metric distribution report (single-threaded)
"""

import argparse
import csv
import io
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from data_readers import iter_samples
from serialization import QUALITY_FILTER_DEFAULTS
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def count_words(text: str) -> int:
    """Count words in text using simple whitespace splitting."""
    return len(text.split())


def calculate_iou(bbox1: list[float], bbox2: list[float]) -> float:
    """Calculate Intersection over Union between two [x1, y1, x2, y2] bounding boxes.

    Computes standard IoU (intersection / union).  Note: this differs from
    ``max_intra_block_line_overlap`` / ``max_cross_block_line_overlap`` in
    quality_metrics, which use containment fraction (intersection / min_area).
    Both metrics are stored in the stats CSV; they answer different questions —
    IoU penalizes large mutual overlap regardless of box sizes, while containment
    fraction detects when a small box is subsumed by a larger one.
    """
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])

    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0

    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def analyze_line_overlaps(lines: list[dict]) -> dict[str, Any]:
    """Analyze overlaps between text lines in a single sample using standard IoU.

    Uses ``calculate_iou`` (intersection / union) — see its docstring for the
    distinction from the containment-fraction metric stored as
    ``max_intra/cross_block_line_overlap`` in quality_metrics.
    """
    overlaps = []
    high_overlap_pairs = 0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if "bbox" in lines[i] and "bbox" in lines[j]:
                iou = calculate_iou(lines[i]["bbox"], lines[j]["bbox"])
                overlaps.append(iou)
                if iou > 0.1:
                    high_overlap_pairs += 1

    return {
        "total_pairs": len(overlaps),
        "high_overlap_pairs": high_overlap_pairs,
        "max_iou": max(overlaps) if overlaps else 0.0,
        "avg_iou": statistics.mean(overlaps) if overlaps else 0.0,
    }


_ZERO_BBOX_STATS: dict[str, int | float] = {
    "lines_with_bbox": 0,
    "total_line_pairs": 0,
    "high_overlap_pairs": 0,
    "max_iou": 0,
    "avg_iou": 0,
    "width_min": 0,
    "width_max": 0,
    "width_mean": 0,
    "width_std": 0,
    "height_min": 0,
    "height_max": 0,
    "height_mean": 0,
    "height_std": 0,
    "aspect_ratio_min": 0,
    "aspect_ratio_max": 0,
    "aspect_ratio_mean": 0,
    "x_center_min": 0,
    "x_center_max": 0,
    "x_center_mean": 0,
    "y_center_min": 0,
    "y_center_max": 0,
    "y_center_mean": 0,
}

# Quality metric columns (None when absent)
_QUALITY_METRIC_KEYS = [
    "min_line_contrast",
    "mean_line_contrast",
    "min_line_contrast_ratio",
    "degenerate_line_count",
    "degenerate_word_count",
    "textbox_null_count",
    "textbox_total_count",
    "textbox_null_frac",
    "min_line_bbox_area_px",
    "min_word_bbox_area_px",
    "line_count",
    "word_count",
    "min_line_height_px",
    "mean_line_height_px",
    "sharpness",
    "max_intra_block_line_overlap",
    "max_cross_block_line_overlap",
]

# ---------------------------------------------------------------------------
# Per-sample statistics (was generate_stats.py)
# ---------------------------------------------------------------------------


def analyze_sample(data: dict[str, Any], sample_id: str) -> dict[str, Any]:
    """Analyze a single sample and return per-sample statistics."""
    text_lines = data.get("text", {}).get("lines", [])
    img_info = data.get("image", {})
    quality_metrics = data.get("quality_metrics", {})

    num_lines = len(text_lines)
    num_words = 0
    for line in text_lines:
        if isinstance(line, str):
            num_words += count_words(line)
        elif isinstance(line, dict) and "text" in line:
            num_words += count_words(line["text"])

    img_path = img_info.get("path", "")
    img_width = img_info.get("width", 0) or 0
    img_height = img_info.get("height", 0) or 0
    img_dpi = img_info.get("dpi")

    sample_stats: dict[str, Any] = {
        "sample_id": sample_id,
        "image_path": img_path,
        "image_width": img_width,
        "image_height": img_height,
        "image_dpi": img_dpi,
        "num_lines": num_lines,
        "num_words": num_words,
        "avg_words_per_line": round(num_words / max(num_lines, 1), 2),
    }

    # Analyze overlaps and dimensions if we have bounding boxes
    lines_with_bbox = [line for line in text_lines if isinstance(line, dict) and "bbox" in line]

    if lines_with_bbox:
        overlap_analysis = analyze_line_overlaps(lines_with_bbox)
        sample_stats.update(
            {
                "lines_with_bbox": len(lines_with_bbox),
                "total_line_pairs": overlap_analysis["total_pairs"],
                "high_overlap_pairs": overlap_analysis["high_overlap_pairs"],
                "max_iou": round(overlap_analysis["max_iou"], 4),
                "avg_iou": round(overlap_analysis["avg_iou"], 4),
            }
        )

        if img_width <= 0 or img_height <= 0:
            # Populate all dimension keys with zeros so every sample has the
            # same fieldnames — prevents csv.DictWriter from crashing when the
            # first sample is complete but a later one is missing keys.
            sample_stats.update({k: v for k, v in _ZERO_BBOX_STATS.items() if k not in sample_stats})
        else:
            widths, heights, aspect_ratios = [], [], []
            x_centers, y_centers = [], []

            for line in lines_with_bbox:
                bbox = line["bbox"]
                width_norm = bbox[2] - bbox[0]
                height_norm = bbox[3] - bbox[1]
                width_px = width_norm * img_width
                height_px = height_norm * img_height

                widths.append(width_px)
                heights.append(height_px)
                if height_px > 0:
                    aspect_ratios.append(width_px / height_px)
                x_centers.append((bbox[0] + bbox[2]) / 2)
                y_centers.append((bbox[1] + bbox[3]) / 2)

            sample_stats.update(
                {
                    "width_min": round(min(widths), 1) if widths else 0,
                    "width_max": round(max(widths), 1) if widths else 0,
                    "width_mean": round(statistics.mean(widths), 1) if widths else 0,
                    "width_std": round(statistics.stdev(widths), 1) if len(widths) > 1 else 0,
                    "height_min": round(min(heights), 1) if heights else 0,
                    "height_max": round(max(heights), 1) if heights else 0,
                    "height_mean": round(statistics.mean(heights), 1) if heights else 0,
                    "height_std": round(statistics.stdev(heights), 1) if len(heights) > 1 else 0,
                    "aspect_ratio_min": round(min(aspect_ratios), 2) if aspect_ratios else 0,
                    "aspect_ratio_max": round(max(aspect_ratios), 2) if aspect_ratios else 0,
                    "aspect_ratio_mean": round(statistics.mean(aspect_ratios), 2) if aspect_ratios else 0,
                    "x_center_min": round(min(x_centers), 3) if x_centers else 0,
                    "x_center_max": round(max(x_centers), 3) if x_centers else 0,
                    "x_center_mean": round(statistics.mean(x_centers), 3) if x_centers else 0,
                    "y_center_min": round(min(y_centers), 3) if y_centers else 0,
                    "y_center_max": round(max(y_centers), 3) if y_centers else 0,
                    "y_center_mean": round(statistics.mean(y_centers), 3) if y_centers else 0,
                }
            )
    else:
        sample_stats.update(_ZERO_BBOX_STATS)

    # Add quality metrics when present
    for key in _QUALITY_METRIC_KEYS:
        sample_stats[key] = quality_metrics.get(key)

    return sample_stats


def collect_sample_stats(input_path: Path) -> list[dict]:
    """Collect per-sample statistics from any supported input format."""
    sample_stats = []
    for sample_id, data in tqdm(iter_samples(input_path), desc="Processing samples", unit="sample"):
        try:
            stats = analyze_sample(data, sample_id)
            sample_stats.append(stats)
        except Exception as e:
            print(f"[warning] Error processing {sample_id}: {e}", file=sys.stderr)
    return sample_stats


def _build_stats_csv(sample_stats: list[dict]) -> str:
    """Format sample statistics as a CSV string."""
    if not sample_stats:
        return ""
    csv_buffer = io.StringIO()
    fieldnames = list(sample_stats[0].keys())
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(sample_stats)
    return csv_buffer.getvalue()


def _print_summary(sample_stats: list[dict]) -> None:
    """Print aggregate summary statistics to stderr."""
    total_lines = sum(s["num_lines"] for s in sample_stats)
    total_words = sum(s["num_words"] for s in sample_stats)
    samples_with_high_overlap = sum(1 for s in sample_stats if s["high_overlap_pairs"] > 0)

    print("Summary:", file=sys.stderr)
    print(f"  Samples: {len(sample_stats)}", file=sys.stderr)
    print(f"  Total lines: {total_lines}", file=sys.stderr)
    print(f"  Total words: {total_words}", file=sys.stderr)
    print(
        f"  Samples with high overlap: {samples_with_high_overlap}"
        f" ({100 * samples_with_high_overlap / len(sample_stats):.1f}%)",
        file=sys.stderr,
    )


def create_stats_file(input_path: Path, output_path: str | None = None, force: bool = False) -> None:
    """Create a stats CSV file for any supported input format."""
    if output_path:
        stats_path = Path(output_path).resolve()
    else:
        stats_path = input_path.with_suffix(".stats.csv")

    if stats_path.exists() and not force:
        file_size = stats_path.stat().st_size
        print(f"Stats file already exists: {stats_path} (size: {file_size} bytes)", file=sys.stderr)
        print("Use --force to regenerate.", file=sys.stderr)
        return

    print(f"Analyzing: {input_path}", file=sys.stderr)

    sample_stats = collect_sample_stats(input_path)
    if not sample_stats:
        print("No valid samples found!", file=sys.stderr)
        return

    with open(stats_path, "w", encoding="utf-8", newline="") as f:
        f.write(_build_stats_csv(sample_stats))

    print(f"Stats for {len(sample_stats)} samples written to {stats_path.name}", file=sys.stderr)
    _print_summary(sample_stats)


# ---------------------------------------------------------------------------
# Aggregation (was aggregate_stats.py)
# ---------------------------------------------------------------------------


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
    return {
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
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
        "min_line_contrast_ratio",
        "min_line_bbox_area_px",
        "min_word_bbox_area_px",
        "degenerate_line_count",
        "degenerate_word_count",
        "textbox_null_count",
        "textbox_total_count",
        "textbox_null_frac",
        "line_count",
        "word_count",
        "min_line_height_px",
        "mean_line_height_px",
        "sharpness",
        "max_intra_block_line_overlap",
        "max_cross_block_line_overlap",
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

        is_quality_problematic = False
        for metric_key, op, threshold in QUALITY_FILTER_DEFAULTS:
            val = _safe_float(r.get(metric_key, ""))
            if val is None:
                continue
            if op == "<" and val < threshold:
                is_quality_problematic = True
                break
            elif op == ">" and val > threshold:
                is_quality_problematic = True
                break

        is_problematic = high_overlap > 10 or max_iou > 0.5 or num_lines > 100 or is_quality_problematic
        if is_problematic:
            entry: dict[str, Any] = {
                "sample_id": r.get("sample_id", ""),
                "source": r.get("source", ""),
                "high_overlap_pairs": high_overlap,
                "max_iou": max_iou,
                "num_lines": num_lines,
            }
            for metric_key, _op, _thr in QUALITY_FILTER_DEFAULTS:
                entry[metric_key] = _safe_float(r.get(metric_key, ""))
            problematic.append(entry)
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


# ---------------------------------------------------------------------------
# Batch processing (was batch_stats.py)
# ---------------------------------------------------------------------------


def find_data_sources(root: Path) -> list[Path]:
    """Find all tar files and metadata.jsonl directories under root."""
    sources: list[Path] = []

    # Find tar files
    for tar_path in sorted(root.rglob("*.tar")):
        sources.append(tar_path)

    # Find directories with metadata.jsonl
    for meta_path in sorted(root.rglob("metadata.jsonl")):
        sources.append(meta_path.parent)

    return sources


def _process_one(source: Path, force: bool) -> str:
    """Process a single data source. Returns status message."""
    stats_path = source.with_suffix(".stats.csv")
    if stats_path.exists() and not force:
        return f"SKIP {source.name} (stats exist)"
    try:
        create_stats_file(source, force=force)
        return f"OK   {source.name}"
    except Exception as e:
        return f"FAIL {source.name}: {e}"


# ---------------------------------------------------------------------------
# Report subcommand
# ---------------------------------------------------------------------------

_PERCENTILES = [0, 5, 25, 50, 75, 95, 99, 100]


def _build_report(input_path: Path) -> str:
    """Collect quality_metrics from all samples and return a formatted report string."""
    # Determine whether input_path is a single split dir or a parent with multiple splits.
    splits_to_scan: list[Path] = []
    if (input_path / "metadata.jsonl").exists():
        splits_to_scan = [input_path]
    else:
        for child in sorted(input_path.iterdir()):
            if child.is_dir() and (child / "metadata.jsonl").exists():
                splits_to_scan.append(child)

    if not splits_to_scan:
        return f"No metadata.jsonl found under {input_path}"

    # Collect quality_metrics from all samples
    metric_values: dict[str, list[float]] = {k: [] for k in _QUALITY_METRIC_KEYS}
    total_samples = 0

    for split_dir in splits_to_scan:
        for _sample_id, data in iter_samples(split_dir):
            qm = data.get("quality_metrics", {})
            total_samples += 1
            for key in _QUALITY_METRIC_KEYS:
                v = qm.get(key)
                if v is not None:
                    try:
                        metric_values[key].append(float(v))
                    except (TypeError, ValueError):
                        pass

    lines_out: list[str] = []
    lines_out.append(f"=== Quality Report: {input_path} ({total_samples} samples) ===")
    lines_out.append("")

    # Percentile table header
    col_w = 34
    pct_w = 7
    hdr = f"{'METRIC':<{col_w}}" + "".join(f"{'p' + str(p):>{pct_w}}" for p in _PERCENTILES)
    lines_out.append(hdr)
    lines_out.append("-" * col_w + "+" + (("-" * (pct_w - 1) + "+") * len(_PERCENTILES)).rstrip("+"))

    for key in _QUALITY_METRIC_KEYS:
        vals = metric_values[key]
        if not vals:
            row = f"{key:<{col_w}}" + f"{'N/A':>{pct_w}}"
        else:
            arr = np.array(vals)
            pcts = np.percentile(arr, _PERCENTILES)
            row = f"{key:<{col_w}}" + "".join(f"{v:>{pct_w}.2f}" if abs(v) < 1000 else f"{v:>{pct_w}.0f}" for v in pcts)
        lines_out.append(row)

    lines_out.append("")
    lines_out.append("FILTER IMPACT (samples that would be discarded per threshold):")

    for metric, op, threshold in QUALITY_FILTER_DEFAULTS:
        vals = metric_values[metric]
        n = len(vals)
        if n == 0:
            lines_out.append(f"  {metric} {op} {threshold:<10}  N/A")
            continue
        arr = np.array(vals)
        if op == "<":
            discarded = int(np.sum(arr < threshold))
            label = f"{metric} < {threshold}"
        else:
            discarded = int(np.sum(arr > threshold))
            label = f"{metric} > {threshold}"
        pct = 100.0 * discarded / n
        lines_out.append(f"  {label:<45}  {discarded:>4} / {n:>5}  ({pct:.1f}%)")

    return "\n".join(lines_out)


def _cmd_report(args: argparse.Namespace) -> None:
    """Handle the 'report' subcommand."""
    input_path = Path(args.directory).resolve()
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    print(_build_report(input_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_stats(args: argparse.Namespace) -> None:
    """Handle the 'stats' subcommand."""
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        create_stats_file(input_path, args.output, args.force)
    except Exception as e:
        print(f"Error analyzing input: {e}", file=sys.stderr)
        sys.exit(1)


def _cmd_aggregate(args: argparse.Namespace) -> None:
    """Handle the 'aggregate' subcommand."""
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


def _cmd_batch(args: argparse.Namespace) -> None:
    """Handle the 'batch' subcommand."""
    root = Path(args.directory).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    sources = find_data_sources(root)
    if not sources:
        print(f"No data sources found under {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(sources)} data source(s) under {root}", file=sys.stderr)

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process_one, src, args.force): src for src in sources}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="source"):
                print(future.result(), file=sys.stderr)
    else:
        for source in tqdm(sources, desc="Processing", unit="source"):
            print(_process_one(source, args.force), file=sys.stderr)

    print("Done.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze SynthDoG data: per-sample statistics, aggregation, and batch processing"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- stats subcommand ---
    p_stats = subparsers.add_parser(
        "stats",
        help="Generate per-sample statistics CSV from a data source (tar or directory)",
    )
    p_stats.add_argument("input", type=str, help="Path to a tar file or directory containing metadata.jsonl")
    p_stats.add_argument("-o", "--output", type=str, help="Output path for stats CSV (defaults to <input>.stats.csv)")
    p_stats.add_argument("--force", action="store_true", help="Regenerate even if stats file already exists")
    p_stats.set_defaults(func=_cmd_stats)

    # --- aggregate subcommand ---
    p_agg = subparsers.add_parser(
        "aggregate",
        help="Aggregate statistics from multiple .stats.csv files",
    )
    p_agg.add_argument("stats_files", nargs="*", help="Paths to .stats.csv files")
    p_agg.add_argument("-d", "--directory", type=str, help="Directory to scan for .stats.csv files")
    p_agg.add_argument("-o", "--output", type=str, help="Output path prefix (creates .csv and .json)")
    p_agg.set_defaults(func=_cmd_aggregate)

    # --- batch subcommand ---
    p_batch = subparsers.add_parser(
        "batch",
        help="Batch generate statistics for all data sources under a directory",
    )
    p_batch.add_argument("directory", type=str, help="Root directory to scan for data sources")
    p_batch.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    p_batch.add_argument("--force", action="store_true", help="Regenerate existing stats files")
    p_batch.set_defaults(func=_cmd_batch)

    # --- report subcommand ---
    p_report = subparsers.add_parser(
        "report",
        help="Print a human-readable quality metric distribution report",
    )
    p_report.add_argument(
        "directory",
        type=str,
        help="Split directory (contains metadata.jsonl) or parent directory (scans all splits)",
    )
    p_report.set_defaults(func=_cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
