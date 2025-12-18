#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import List
import pandas as pd


def find_external_stats_file(tar_path: Path) -> Path:
    """
    Find the external .stats.csv file for a given tar file.
    """
    # For /path/to/file.tar, look for /path/to/file.stats.csv
    stats_path = tar_path.with_suffix('.stats.csv')
    return stats_path


def find_tar_files_with_stats(directory: Path) -> List[Path]:
    """
    Find all tar files in a directory that have corresponding .stats.csv files.
    """
    tar_files_with_stats = []
    
    for tar_path in directory.glob("*.tar"):
        stats_path = find_external_stats_file(tar_path)
        if stats_path.exists():
            tar_files_with_stats.append(tar_path)
        else:
            print(f"[info] Skipping {tar_path.name} - no stats file found", file=sys.stderr)
    
    return sorted(tar_files_with_stats)


def extract_stats_from_external_csv(tar_path: Path) -> pd.DataFrame:
    """
    Extract stats from external .stats.csv file and return as DataFrame.
    """
    stats_path = find_external_stats_file(tar_path)
    
    if not stats_path.exists():
        print(f"[warning] No external stats file found: {stats_path}", file=sys.stderr)
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(stats_path)
        
        # Add source tar info
        df['source_tar'] = tar_path.name
        df['stats_file'] = stats_path.name
        
        return df
        
    except Exception as e:
        print(f"[error] Failed to read {stats_path}: {e}", file=sys.stderr)
        return pd.DataFrame()


def aggregate_stats(tar_files: List[Path], output_path: Path = None):
    """
    Aggregate statistics from multiple tar files using external .stats.csv files.
    """
    all_dfs = []
    
    for tar_path in tar_files:
        print(f"Processing {tar_path.name}...", file=sys.stderr)
        df = extract_stats_from_external_csv(tar_path)
        if not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        print("No valid stats found in any tar files!", file=sys.stderr)
        return
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Generate summary statistics
    summary = {}
    
    # Basic counts
    summary['total_samples'] = int(len(combined_df))
    summary['total_lines'] = int(combined_df['num_lines'].sum())
    summary['total_words'] = int(combined_df['num_words'].sum())
    summary['unique_tar_files'] = int(combined_df['source_tar'].nunique())
    
    # Overlap analysis
    samples_with_bbox = (combined_df['lines_with_bbox'] > 0).sum()
    samples_with_high_overlap = (combined_df['high_overlap_pairs'] > 0).sum()
    
    summary['samples_with_bbox'] = int(samples_with_bbox)
    summary['samples_with_high_overlap'] = int(samples_with_high_overlap)
    summary['pct_samples_with_high_overlap'] = round(100 * samples_with_high_overlap / max(samples_with_bbox, 1), 2)
    
    # Statistical summaries for key metrics
    numeric_cols = ['num_lines', 'num_words', 'avg_words_per_line', 'max_iou', 'high_overlap_pairs',
                   'width_mean', 'height_mean', 'aspect_ratio_mean']
    
    for col in numeric_cols:
        if col in combined_df.columns:
            values = combined_df[col].dropna()
            if len(values) > 0:
                summary[f'{col}_stats'] = {
                    'min': round(float(values.min()), 4),
                    'max': round(float(values.max()), 4),
                    'mean': round(float(values.mean()), 4),
                    'median': round(float(values.median()), 4),
                    'std': round(float(values.std()) if len(values) > 1 else 0.0, 4)
                }
    
    # Identify problematic samples
    problematic_samples = combined_df[
        (combined_df['high_overlap_pairs'] > 10) |  # Many overlapping pairs
        (combined_df['max_iou'] > 0.5) |            # Very high overlap
        (combined_df['num_lines'] > 100)            # Unusually many lines
    ][['sample_id', 'source_tar', 'high_overlap_pairs', 'max_iou', 'num_lines']].sort_values(by='max_iou', ascending=False).to_dict('records')

    summary['problematic_samples'] = {
        'count': len(problematic_samples),
        'samples': problematic_samples[:20]  # Show first 20
    }
    
    # Save aggregated CSV
    if output_path:
        csv_path = output_path.with_suffix('.csv')
        combined_df.to_csv(csv_path, index=False)
        print(f"Aggregated CSV saved to: {csv_path}", file=sys.stderr)
        
        # Save summary JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary statistics saved to: {json_path}", file=sys.stderr)
    
    # Print summary to stdout
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate statistics from multiple tar files using external .stats.csv files"
    )
    parser.add_argument(
        "tar_files",
        nargs='*',
        help="Paths to tar files (will look for corresponding .stats.csv files)"
    )
    parser.add_argument(
        "-d", "--directory",
        type=str,
        help="Directory to scan for tar files with corresponding .stats.csv files"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output path prefix for aggregated results (will create .csv and .json files)"
    )
    
    args = parser.parse_args()
    
    # Determine tar files to process
    if args.directory:
        directory = Path(args.directory).resolve()
        if not directory.exists() or not directory.is_dir():
            print(f"Directory not found: {directory}", file=sys.stderr)
            sys.exit(1)
        tar_paths = find_tar_files_with_stats(directory)
        if not tar_paths:
            print(f"No tar files with .stats.csv found in {directory}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(tar_paths)} tar files with stats in {directory}", file=sys.stderr)
    elif args.tar_files:
        tar_paths = [Path(p).resolve() for p in args.tar_files]
    else:
        print("Either provide tar files or use --directory option", file=sys.stderr)
        sys.exit(1)
    
    # Check that all files exist
    for tar_path in tar_paths:
        if not tar_path.exists():
            print(f"Tar file not found: {tar_path}", file=sys.stderr)
            sys.exit(1)
        
        # Check for corresponding stats file
        stats_path = find_external_stats_file(tar_path)
        if not stats_path.exists():
            print(f"Stats file not found: {stats_path}", file=sys.stderr)
            sys.exit(1)
    
    output_path = Path(args.output).resolve() if args.output else None
    
    try:
        aggregate_stats(tar_paths, output_path)
    except Exception as e:
        print(f"Error aggregating stats: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
