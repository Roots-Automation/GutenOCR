#!/usr/bin/env python3
"""Generate per-sample statistics CSV from SynthDoG data.

Supports both raw generation output (metadata.jsonl directories) and
packaged tar archives.  Quality metrics are included when present.

Usage:
    python generate_stats.py /path/to/data.tar
    python generate_stats.py /path/to/output_dir
    python generate_stats.py /path/to/data.tar -o /path/to/output.stats.csv
    python generate_stats.py /path/to/data.tar --force
"""

import argparse
import csv
import io
import statistics
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Allow standalone execution from the data_analysis/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_readers import iter_samples


def count_words(text: str) -> int:
    """Count words in text using simple whitespace splitting."""
    return len(text.split())


def calculate_iou(bbox1: list[float], bbox2: list[float]) -> float:
    """Calculate Intersection over Union between two [x1, y1, x2, y2] bounding boxes."""
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
    """Analyze overlaps between text lines in a single sample."""
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
    "degenerate_line_count",
    "degenerate_word_count",
    "textbox_null_count",
    "textbox_total_count",
    "min_line_bbox_area_px",
    "min_word_bbox_area_px",
]


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

        if img_width > 0 and img_height > 0:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-sample statistics CSV from a SynthDoG data source (tar file or directory)"
    )
    parser.add_argument("input", type=str, help="Path to a tar file or directory containing metadata.jsonl")
    parser.add_argument("-o", "--output", type=str, help="Output path for stats CSV (defaults to <input>.stats.csv)")
    parser.add_argument("--force", action="store_true", help="Regenerate even if stats file already exists")

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        create_stats_file(input_path, args.output, args.force)
    except Exception as e:
        print(f"Error analyzing input: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
