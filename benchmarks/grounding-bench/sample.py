"""
Extract the top-k diverse samples from a flat image/JSON directory.

Usage:
    python sample.py <data_dir> <ranked_csv> <output_dir> [--top-k 100]

Reads the rankings CSV produced by diversity/rank.py, walks through images in
rank order, skips any sample whose text.lines annotation is missing or empty,
and copies both the image and its paired JSON to output_dir. Stops when top-k
valid samples have been collected.
"""

import argparse
import json
import os
import shutil
import sys

import pandas as pd
from tqdm import tqdm


IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def find_image(data_dir: str, stem: str) -> str | None:
    """Return the path of the image file for the given stem, or None."""
    for ext in IMAGE_EXTS:
        path = os.path.join(data_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


def has_lines(json_path: str) -> bool:
    """Return True if the JSON annotation has a non-empty text.lines array."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        lines = data.get("text", {}).get("lines")
        return bool(lines)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract top-k diverse samples from a flat image/JSON directory."
    )
    parser.add_argument("data_dir", help="Directory containing image and JSON pairs")
    parser.add_argument(
        "ranked_csv", help="Rankings CSV produced by diversity/rank.py"
    )
    parser.add_argument("output_dir", help="Directory to write selected samples")
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of valid samples to collect (default: 100)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Error: data_dir not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(args.ranked_csv)
    except Exception as e:
        print(f"Error reading ranked CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if "rank" not in df.columns or "filename" not in df.columns:
        print(
            "Error: ranked CSV must contain 'rank' and 'filename' columns.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = df.sort_values("rank")

    os.makedirs(args.output_dir, exist_ok=True)

    selected = 0
    skipped_missing = 0
    skipped_no_lines = 0

    for filename in tqdm(df["filename"], desc="Sampling"):
        if selected >= args.top_k:
            break

        # Derive stem — filename in rankings is relative to data_dir
        stem = os.path.splitext(os.path.basename(filename))[0]

        # Locate image
        img_path = find_image(args.data_dir, stem)
        json_path = os.path.join(args.data_dir, stem + ".json")

        if img_path is None or not os.path.exists(json_path):
            skipped_missing += 1
            continue

        # Check annotation quality
        if not has_lines(json_path):
            skipped_no_lines += 1
            continue

        # Copy both files to output
        shutil.copy2(img_path, os.path.join(args.output_dir, os.path.basename(img_path)))
        shutil.copy2(json_path, os.path.join(args.output_dir, stem + ".json"))
        selected += 1

    print(f"\nDone.")
    print(f"  Selected : {selected}")
    print(f"  Skipped (missing files)    : {skipped_missing}")
    print(f"  Skipped (no text.lines)    : {skipped_no_lines}")
    print(f"  Output written to          : {args.output_dir}")

    if selected < args.top_k:
        print(
            f"\nWarning: only {selected} valid samples found "
            f"(requested {args.top_k}).",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
