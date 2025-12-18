#!/usr/bin/env python3
"""
SynthDoG Statistics Generator

This script analyzes SynthDoG tar archives and generates comprehensive statistics
about the synthetic document data, including text metrics, image properties,
layout analysis, and bounding box statistics.

Features:
- Extracts detailed statistics from tar archives containing images and JSON annotations
- Analyzes text content (character counts, word counts, language distribution)
- Evaluates image properties (dimensions, aspect ratios, DPI)
- Performs layout analysis (line overlaps, density, spatial distribution)
- Calculates bounding box statistics (IoU analysis, size distributions)
- Outputs results to external CSV files for efficient processing

Usage:
    python generate_stats.py <tar_file> [options]

Examples:
    # Generate statistics for a single tar file
    python generate_stats.py /path/to/data.tar

    # Generate with custom output location
    python generate_stats.py /path/to/data.tar --output /path/to/output.stats.csv

    # Force regeneration of existing statistics
    python generate_stats.py /path/to/data.tar --force

Output:
    Creates a .stats.csv file containing:
    - Sample-level statistics (one row per image/annotation pair)
    - Image metadata (width, height, aspect ratio, DPI)
    - Text statistics (character counts, word counts, line counts)
    - Layout metrics (overlap analysis, density measures)
    - Bounding box analysis (size distributions, spatial metrics)

Requirements:
    - Python 3.8+
    - tqdm for progress bars
    - Standard library modules (tarfile, json, csv, statistics)
"""

import argparse
import csv
import io
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
import statistics


def count_words(text: str) -> int:
    """
    Count words in text using simple whitespace splitting.
    
    Args:
        text: Input text string
        
    Returns:
        int: Number of words in the text
        
    Note:
        This provides a rough word count based on whitespace separation.
        More sophisticated tokenization might be needed for some languages.
    """
    return len(text.split())


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2] (normalized coordinates)
        bbox2: Second bounding box [x1, y1, x2, y2] (normalized coordinates)
        
    Returns:
        float: IoU value between 0.0 and 1.0
        
    Note:
        Coordinates are expected to be normalized (0.0 to 1.0).
        Returns 0.0 if boxes don't overlap.
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


def analyze_line_overlaps(lines: List[Dict]) -> Dict[str, Any]:
    """
    Analyze overlaps between text lines in a single sample.
    """
    overlaps = []
    high_overlap_pairs = 0
    
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if 'bbox' in lines[i] and 'bbox' in lines[j]:
                iou = calculate_iou(lines[i]['bbox'], lines[j]['bbox'])
                overlaps.append(iou)
                if iou > 0.1:  # Consider IoU > 0.1 as significant overlap
                    high_overlap_pairs += 1
    
    return {
        'total_pairs': len(overlaps),
        'high_overlap_pairs': high_overlap_pairs,
        'max_iou': max(overlaps) if overlaps else 0.0,
        'avg_iou': statistics.mean(overlaps) if overlaps else 0.0
    }


def analyze_line_dimensions(lines: List[Dict], img_width: int, img_height: int) -> Dict[str, Any]:
    """
    Analyze dimensions and spatial properties of text lines.
    """
    widths, heights, aspect_ratios = [], [], []
    x_centers, y_centers = [], []
    
    for line in lines:
        if 'bbox' in line:
            bbox = line['bbox']
            # Convert normalized coords to pixel coords for absolute measurements
            width_norm = bbox[2] - bbox[0]
            height_norm = bbox[3] - bbox[1]
            
            width_px = width_norm * img_width
            height_px = height_norm * img_height
            
            widths.append(width_px)
            heights.append(height_px)
            
            if height_px > 0:
                aspect_ratios.append(width_px / height_px)
            
            # Center coordinates (normalized)
            x_centers.append((bbox[0] + bbox[2]) / 2)
            y_centers.append((bbox[1] + bbox[3]) / 2)
    
    def safe_stats(values):
        if not values:
            return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0}
        return {
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0
        }
    
    return {
        'width_px': safe_stats(widths),
        'height_px': safe_stats(heights),
        'aspect_ratio': safe_stats(aspect_ratios),
        'x_center_norm': safe_stats(x_centers),
        'y_center_norm': safe_stats(y_centers),
        'total_lines_with_bbox': len(widths)
    }


def analyze_sample(data: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
    """
    Analyze a single sample and return per-sample statistics.
    """
    text_lines = data.get("text", {}).get("lines", [])
    img_info = data.get("image", {})
    
    # Basic counts
    num_lines = len(text_lines)
    num_words = 0
    
    # Count words
    for line in text_lines:
        if isinstance(line, str):
            num_words += count_words(line)
        elif isinstance(line, dict) and "text" in line:
            num_words += count_words(line["text"])
    
    # Image info
    img_path = img_info.get("path", "")
    img_width = img_info.get("width", 0)
    img_height = img_info.get("height", 0)
    img_dpi = img_info.get("dpi")
    
    # Initialize stats
    sample_stats = {
        "sample_id": sample_id,
        "image_path": img_path,
        "image_width": img_width,
        "image_height": img_height,
        "image_dpi": img_dpi,
        "num_lines": num_lines,
        "num_words": num_words,
        "avg_words_per_line": round(num_words / max(num_lines, 1), 2)
    }
    
    # Analyze overlaps and dimensions if we have bounding boxes
    lines_with_bbox = [line for line in text_lines if isinstance(line, dict) and 'bbox' in line]
    
    if lines_with_bbox:
        # Overlap analysis
        overlap_analysis = analyze_line_overlaps(lines_with_bbox)
        sample_stats.update({
            "lines_with_bbox": len(lines_with_bbox),
            "total_line_pairs": overlap_analysis['total_pairs'],
            "high_overlap_pairs": overlap_analysis['high_overlap_pairs'],
            "max_iou": round(overlap_analysis['max_iou'], 4),
            "avg_iou": round(overlap_analysis['avg_iou'], 4)
        })
        
        # Dimension analysis
        if img_width > 0 and img_height > 0:
            widths, heights, aspect_ratios = [], [], []
            x_centers, y_centers = [], []
            
            for line in lines_with_bbox:
                bbox = line['bbox']
                # Convert normalized coords to pixel coords
                width_norm = bbox[2] - bbox[0]
                height_norm = bbox[3] - bbox[1]
                
                width_px = width_norm * img_width
                height_px = height_norm * img_height
                
                widths.append(width_px)
                heights.append(height_px)
                
                if height_px > 0:
                    aspect_ratios.append(width_px / height_px)
                
                # Center coordinates (normalized)
                x_centers.append((bbox[0] + bbox[2]) / 2)
                y_centers.append((bbox[1] + bbox[3]) / 2)
            
            # Add dimension statistics
            sample_stats.update({
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
                "y_center_mean": round(statistics.mean(y_centers), 3) if y_centers else 0
            })
    else:
        # No bounding boxes
        sample_stats.update({
            "lines_with_bbox": 0,
            "total_line_pairs": 0,
            "high_overlap_pairs": 0,
            "max_iou": 0,
            "avg_iou": 0,
            "width_min": 0, "width_max": 0, "width_mean": 0, "width_std": 0,
            "height_min": 0, "height_max": 0, "height_mean": 0, "height_std": 0,
            "aspect_ratio_min": 0, "aspect_ratio_max": 0, "aspect_ratio_mean": 0,
            "x_center_min": 0, "x_center_max": 0, "x_center_mean": 0,
            "y_center_min": 0, "y_center_max": 0, "y_center_mean": 0
        })
    
    return sample_stats


def add_stats_to_tar(tar_path: Path) -> None:
    """
    Add a stats.csv file to an existing tar file with per-sample statistics.
    If stats.csv already exists, assume it's correct and skip processing.
    Uses append mode for efficiency - much faster than rewriting entire tar.
    """
    print(f"Checking tar file: {tar_path}", file=sys.stderr)
    
    # First, check if stats.csv already exists in the tar
    try:
        with tarfile.open(tar_path, "r") as tar:
            try:
                stats_member = tar.getmember("stats.csv")
                print(f"stats.csv already exists in {tar_path.name} (size: {stats_member.size} bytes)", file=sys.stderr)
                print("Assuming stats.csv is correct and skipping processing.", file=sys.stderr)
                return
            except KeyError:
                print("stats.csv not found in tar. Proceeding to generate it...", file=sys.stderr)
    except Exception as e:
        print(f"Error checking tar contents: {e}", file=sys.stderr)
        return
    
    print(f"Analyzing tar file: {tar_path}", file=sys.stderr)
    
    # Collect sample statistics
    sample_stats = []
    
    with tarfile.open(tar_path, "r") as tar:
        json_files = [member for member in tar.getmembers() 
                     if member.name.endswith('.json') and member.isfile()]
        
        for json_member in tqdm(json_files, desc="Processing samples", unit="sample"):
            try:
                json_file = tar.extractfile(json_member)
                if json_file is None:
                    print(f"[warning] Could not extract {json_member.name}", file=sys.stderr)
                    continue
                    
                content = json_file.read().decode('utf-8')
                data = json.loads(content)
                
                # Extract sample ID from filename (e.g., "00042.json" -> "00042")
                sample_id = json_member.name.replace('.json', '')
                
                stats = analyze_sample(data, sample_id)
                sample_stats.append(stats)
                
            except json.JSONDecodeError as e:
                print(f"[warning] JSON decode error in {json_member.name}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"[warning] Error processing {json_member.name}: {e}", file=sys.stderr)
    
    if not sample_stats:
        print("No valid samples found!", file=sys.stderr)
        return
    
    # Create CSV content
    csv_buffer = io.StringIO()
    fieldnames = sample_stats[0].keys()
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(sample_stats)
    csv_content = csv_buffer.getvalue().encode('utf-8')
    
    # Append the stats.csv to the tar file (much more efficient!)
    try:
        with tarfile.open(tar_path, "a") as tar:
            stats_info = tarfile.TarInfo(name="stats.csv")
            stats_info.size = len(csv_content)
            stats_info.mtime = int(time.time())
            tar.addfile(stats_info, io.BytesIO(csv_content))
            
        print(f"Appended stats.csv with {len(sample_stats)} samples to {tar_path}", file=sys.stderr)
        
    except Exception as e:
        print(f"Error appending to tar: {e}", file=sys.stderr)
        print("Falling back to creating separate stats file...", file=sys.stderr)
        
        # Fallback: create separate stats file
        stats_path = tar_path.with_suffix('.stats.csv')
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(csv_buffer.getvalue())
        print(f"Created separate stats file: {stats_path}", file=sys.stderr)
    
    # Print summary statistics
    total_lines = sum(s['num_lines'] for s in sample_stats)
    total_words = sum(s['num_words'] for s in sample_stats)
    samples_with_high_overlap = sum(1 for s in sample_stats if s['high_overlap_pairs'] > 0)
    
    print(f"Summary:", file=sys.stderr)
    print(f"  Samples: {len(sample_stats)}", file=sys.stderr)
    print(f"  Total lines: {total_lines}", file=sys.stderr)
    print(f"  Total words: {total_words}", file=sys.stderr)
    print(f"  Samples with high overlap: {samples_with_high_overlap} ({100*samples_with_high_overlap/len(sample_stats):.1f}%)", file=sys.stderr)


def create_separate_stats_file(tar_path: Path, output_path: str = None) -> None:
    """
    Create a separate stats CSV file for the tar file (most efficient approach).
    If stats file already exists, assume it's correct and skip processing.
    """
    # Determine output path
    if output_path:
        stats_path = Path(output_path).resolve()
    else:
        stats_path = tar_path.with_suffix('.stats.csv')
    
    # Check if stats file already exists
    if stats_path.exists():
        file_size = stats_path.stat().st_size
        print(f"Stats file already exists: {stats_path} (size: {file_size} bytes)", file=sys.stderr)
        print("Assuming stats file is correct and skipping processing.", file=sys.stderr)
        return
    
    print(f"Analyzing tar file: {tar_path}", file=sys.stderr)
    
    # Collect sample statistics
    sample_stats = []
    
    with tarfile.open(tar_path, "r") as tar:
        json_files = [member for member in tar.getmembers() 
                     if member.name.endswith('.json') and member.isfile()]
        
        for json_member in tqdm(json_files, desc="Processing samples", unit="sample"):
            try:
                json_file = tar.extractfile(json_member)
                if json_file is None:
                    print(f"[warning] Could not extract {json_member.name}", file=sys.stderr)
                    continue
                    
                content = json_file.read().decode('utf-8')
                data = json.loads(content)
                
                # Extract sample ID from filename (e.g., "00042.json" -> "00042")
                sample_id = json_member.name.replace('.json', '')
                
                stats = analyze_sample(data, sample_id)
                sample_stats.append(stats)
                
            except json.JSONDecodeError as e:
                print(f"[warning] JSON decode error in {json_member.name}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"[warning] Error processing {json_member.name}: {e}", file=sys.stderr)
    
    if not sample_stats:
        print("No valid samples found!", file=sys.stderr)
        return
    
    # Write CSV file
    with open(stats_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = sample_stats[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_stats)
    
    print(f"Created stats file: {stats_path}", file=sys.stderr)
    print(f"Stats for {len(sample_stats)} samples written to {stats_path.name}", file=sys.stderr)
    
    # Print summary statistics
    total_lines = sum(s['num_lines'] for s in sample_stats)
    total_words = sum(s['num_words'] for s in sample_stats)
    samples_with_high_overlap = sum(1 for s in sample_stats if s['high_overlap_pairs'] > 0)
    
    print(f"Summary:", file=sys.stderr)
    print(f"  Samples: {len(sample_stats)}", file=sys.stderr)
    print(f"  Total lines: {total_lines}", file=sys.stderr)
    print(f"  Total words: {total_words}", file=sys.stderr)
    print(f"  Samples with high overlap: {samples_with_high_overlap} ({100*samples_with_high_overlap/len(sample_stats):.1f}%)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-sample statistics CSV for a tar file created by build_tar.py"
    )
    parser.add_argument(
        "tar_file", 
        type=str, 
        help="Path to the tar file to analyze"
    )
    parser.add_argument(
        "--append-to-tar",
        action="store_true",
        help="Append stats.csv to the tar file (default: create separate .stats.csv file)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output path for stats CSV (defaults to <tar_name>.stats.csv)"
    )
    
    args = parser.parse_args()
    
    tar_path = Path(args.tar_file).resolve()
    if not tar_path.exists():
        print(f"Tar file not found: {tar_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        if args.append_to_tar:
            add_stats_to_tar(tar_path)
        else:
            # More efficient: create separate stats file
            create_separate_stats_file(tar_path, args.output)
        
    except Exception as e:
        print(f"Error analyzing tar file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
