#!/usr/bin/env python3
"""Batch statistics generation for SynthDoG data.

Finds all tar files and metadata.jsonl directories under a given path,
generates .stats.csv files for each, optionally in parallel.

Usage:
    python batch_stats.py /path/to/outputs
    python batch_stats.py /path/to/outputs --workers 4
    python batch_stats.py /path/to/outputs --force
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# Allow standalone execution from the data_analysis/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_stats import create_stats_file


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch generate statistics for SynthDoG data sources")
    parser.add_argument("directory", type=str, help="Root directory to scan for data sources")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--force", action="store_true", help="Regenerate existing stats files")

    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
