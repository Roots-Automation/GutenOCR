#!/usr/bin/env python3
"""
SynthDoG Tar Archive Builder

This script creates compressed tar archives from SynthDoG generated data directories.
It efficiently packages image files and their corresponding JSON annotations while
adding metadata and optimizing storage.

Features:
- Compresses image/annotation pairs into tar archives
- Adds image metadata (dimensions, DPI) to JSON annotations
- Validates file pairs and handles missing files gracefully
- Configurable compression levels
- Progress tracking for large datasets

Usage:
    python build_tar.py <input_directory> [options]

Examples:
    # Create tar archive with default settings
    python build_tar.py /path/to/data/directory

    # Specify output file and compression
    python build_tar.py /path/to/data/directory -o output.tar --compression 9

    # Process directory and save to default location
    python build_tar.py /mnt/data/synthdog_grounding/0001

Output:
    Creates a compressed tar file containing:
    - Image files (.jpg, .png, etc.)
    - Enhanced JSON annotation files with image metadata
    - Maintains original file structure and naming

Requirements:
    - Pillow (PIL) for image metadata extraction
    - Python 3.8+
"""

import argparse
import io
import json
import re
import sys
import tarfile
import time
from pathlib import Path

# Allow standalone execution from the data_packaging/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_readers import extract_image_metadata
from serialization import KEY_QUALITY_METRICS, KEY_TEXT_BLOCKS, KEY_TEXT_LINES, KEY_TEXT_WORDS, decode_metadata

# Regular expression to extract numeric IDs from filenames
NUM_RE = re.compile(r"(\d+)(?=\.[^.]+$)")  # capture digits before the final extension


def extract_numeric_id(file_name: str) -> int:
    """
    Extract the trailing numeric ID from a filename.

    Args:
        file_name: Filename like 'image_22.jpg' or '00042.png'

    Returns:
        int: The numeric ID (e.g., 22, 42)

    Raises:
        ValueError: If no trailing number is found in the filename

    Examples:
        >>> extract_numeric_id("image_22.jpg")
        22
        >>> extract_numeric_id("00042.png")
        42
    """
    m = NUM_RE.search(file_name)
    if not m:
        raise ValueError(f"No trailing number found in filename: {file_name}")
    return int(m.group(1))


def process_directory(input_dir: Path, output_tar: Path):
    meta_path = input_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.jsonl at: {meta_path}")

    # Open tar for writing (uncompressed .tar)
    with tarfile.open(output_tar, "w") as tar, meta_path.open("r", encoding="utf-8") as meta_f:
        lines_processed = 0
        warnings = 0

        for line_no, raw in enumerate(meta_f, start=1):
            line = raw.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warning] JSON decode error on line {line_no}: {e}", file=sys.stderr)
                warnings += 1
                continue

            file_name = rec.get("file_name")
            if not file_name:
                print(f"[warning] No 'file_name' in line {line_no}; skipping.", file=sys.stderr)
                warnings += 1
                continue

            try:
                numeric_id = extract_numeric_id(file_name)
            except ValueError as e:
                print(f"[warning] {e} (line {line_no}); skipping.", file=sys.stderr)
                warnings += 1
                continue

            img_path = input_dir / file_name
            if not img_path.exists():
                print(f"[warning] Image not found: {img_path} (line {line_no}); skipping.", file=sys.stderr)
                warnings += 1
                continue

            # Build new names inside the tar
            ext = img_path.suffix.lower()  # keep original extension
            new_img_name = f"{numeric_id:05d}{ext}"
            new_json_name = f"{numeric_id:05d}.json"

            # Read image metadata
            try:
                width, height, dpi = extract_image_metadata(img_path)
            except Exception as e:
                print(f"[warning] Failed reading image metadata for {img_path}: {e}", file=sys.stderr)
                warnings += 1
                # If metadata fails, we can still proceed with width/height as None
                width, height, dpi = None, None, None

            # Parse ground_truth via shared serialization schema
            gt_parse = decode_metadata(rec)
            text_lines = gt_parse.get(KEY_TEXT_LINES, [])
            text_words = gt_parse.get(KEY_TEXT_WORDS, [])
            text_blocks = gt_parse.get(KEY_TEXT_BLOCKS, [])
            quality_metrics = gt_parse.get(KEY_QUALITY_METRICS, {})

            # Create the new JSON payload
            new_obj = {
                "text": {"lines": text_lines, "words": text_words, "blocks": text_blocks},
                "image": {"path": new_img_name, "width": width, "height": height, "dpi": dpi},
                "quality_metrics": quality_metrics,
            }
            payload = json.dumps(new_obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

            # 1) Add the image file to the tar under the new name
            tar.add(img_path, arcname=new_img_name)

            # 2) Add the JSON as a file to the tar
            ti = tarfile.TarInfo(name=new_json_name)
            ti.size = len(payload)
            ti.mtime = int(time.time())
            tar.addfile(ti, io.BytesIO(payload))

            lines_processed += 1

        print(
            f"Done. Wrote {lines_processed} record(s) to {output_tar.name} with {warnings} warning(s).", file=sys.stderr
        )


def main():
    parser = argparse.ArgumentParser(
        description="Transform dataset into a tar with 00000.ext images and 00000.json files."
    )
    parser.add_argument("directory", type=str, help="Path to the input directory containing metadata.jsonl and images")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output tar path (defaults to <dir>.tar)")
    args = parser.parse_args()

    input_dir = Path(args.directory).resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_tar = Path(args.output).resolve() if args.output else input_dir.with_suffix(".tar")
    process_directory(input_dir, output_tar)


if __name__ == "__main__":
    main()
