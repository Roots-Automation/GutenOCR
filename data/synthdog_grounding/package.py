#!/usr/bin/env python3
"""Package SynthDoG data directories into tar archives.

Reads metadata.jsonl + images from one or more directories, normalizes
filenames, enriches JSON with image metadata and full annotation data,
and writes uncompressed tar archives.

Usage:
    # Single directory
    python package.py /path/to/data/directory
    python package.py /path/to/data/directory -o output.tar

    # Batch: package all numbered subdirectories with train/val/test splits
    python package.py --batch /path/to/outputs --start 0 --end 75
    python package.py --batch /path/to/outputs --start 0 --end 75 --workers 8

    # Dry run to see what would be processed
    python package.py --batch /path/to/outputs --dry-run
"""

import argparse
import io
import json
import os
import re
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_readers import extract_image_metadata
from serialization import (
    KEY_QUALITY_METRICS,
    KEY_TEXT_BLOCKS,
    KEY_TEXT_LINES,
    KEY_TEXT_WORDS,
    SPLITS,
    decode_metadata,
)

# Capture digits before the final extension
_NUM_RE = re.compile(r"(\d+)(?=\.[^.]+$)")


def _extract_numeric_id(file_name: str) -> int:
    """Extract the trailing numeric ID from a filename.

    >>> _extract_numeric_id("image_22.jpg")
    22
    """
    m = _NUM_RE.search(file_name)
    if not m:
        raise ValueError(f"No trailing number found in filename: {file_name}")
    return int(m.group(1))


def process_directory(input_dir: Path, output_tar: Path) -> None:
    """Package a single directory into a tar archive."""
    meta_path = input_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.jsonl at: {meta_path}")

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
                numeric_id = _extract_numeric_id(file_name)
            except ValueError as e:
                print(f"[warning] {e} (line {line_no}); skipping.", file=sys.stderr)
                warnings += 1
                continue

            img_path = input_dir / file_name
            if not img_path.exists():
                print(f"[warning] Image not found: {img_path} (line {line_no}); skipping.", file=sys.stderr)
                warnings += 1
                continue

            ext = img_path.suffix.lower()
            new_img_name = f"{numeric_id:05d}{ext}"
            new_json_name = f"{numeric_id:05d}.json"

            try:
                width, height, dpi = extract_image_metadata(img_path)
            except Exception as e:
                print(f"[warning] Failed reading image metadata for {img_path}: {e}", file=sys.stderr)
                warnings += 1
                width, height, dpi = None, None, None

            gt_parse = decode_metadata(rec)
            text_lines = gt_parse.get(KEY_TEXT_LINES, [])
            text_words = gt_parse.get(KEY_TEXT_WORDS, [])
            text_blocks = gt_parse.get(KEY_TEXT_BLOCKS, [])
            quality_metrics = gt_parse.get(KEY_QUALITY_METRICS, {})

            new_obj = {
                "text": {"lines": text_lines, "words": text_words, "blocks": text_blocks},
                "image": {"path": new_img_name, "width": width, "height": height, "dpi": dpi},
                "quality_metrics": quality_metrics,
            }
            payload = json.dumps(new_obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

            tar.add(img_path, arcname=new_img_name)

            ti = tarfile.TarInfo(name=new_json_name)
            ti.size = len(payload)
            ti.mtime = int(time.time())
            tar.addfile(ti, io.BytesIO(payload))

            lines_processed += 1

        print(
            f"Done. Wrote {lines_processed} record(s) to {output_tar.name} with {warnings} warning(s).",
            file=sys.stderr,
        )


def _process_one(args: tuple[str, str, str]) -> tuple[bool, str]:
    """Process a single (core_dir, directory_num, split) task."""
    core_dir, directory_num, split = args
    input_dir = Path(core_dir) / directory_num / split
    output_tar = Path(core_dir) / f"{split}-{directory_num}.tar"

    if output_tar.exists():
        return True, f"SKIP {output_tar.name} (already exists)"
    if not input_dir.exists():
        return False, f"SKIP {input_dir} (directory does not exist)"

    try:
        process_directory(input_dir, output_tar)
        return True, f"OK   {output_tar.name}"
    except Exception as e:
        return False, f"FAIL {output_tar.name}: {e}"


def _run_batch(core_dir: Path, start: int, end: int, workers: int | None, dry_run: bool) -> None:
    """Package numbered subdirectories with train/val/test splits."""
    tasks = []
    for i in range(start, end + 1):
        num = f"{i:04d}"
        for split in SPLITS:
            tasks.append((str(core_dir), num, split))

    print(f"Found {len(tasks)} tasks to process", file=sys.stderr)

    if dry_run:
        print("\nDry run — would process:", file=sys.stderr)
        for core_dir_str, directory_num, split in tasks:
            input_dir = Path(core_dir_str) / directory_num / split
            output_tar = Path(core_dir_str) / f"{split}-{directory_num}.tar"
            if output_tar.exists():
                status = "EXISTS"
            elif not input_dir.exists():
                status = "MISSING"
            else:
                status = "TODO"
            print(f"  {input_dir} -> {output_tar.name} [{status}]", file=sys.stderr)
        return

    max_workers = workers or os.cpu_count()
    print(f"Using {max_workers} parallel workers", file=sys.stderr)

    success_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(_process_one, task): task for task in tasks}
        for future in as_completed(future_to_task):
            try:
                success, message = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                print(message, file=sys.stderr)
            except Exception as e:
                error_count += 1
                _, directory_num, split = future_to_task[future]
                print(f"FAIL {split}-{directory_num}: {e}", file=sys.stderr)

    print(f"\nDone: {success_count} succeeded, {error_count} failed/skipped out of {len(tasks)}", file=sys.stderr)
    if error_count > 0:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package SynthDoG data directories into tar archives")
    # Single-directory mode
    parser.add_argument("directory", nargs="?", type=str, help="Input directory containing metadata.jsonl and images")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output tar path (defaults to <dir>.tar)")

    # Batch mode
    parser.add_argument("--batch", type=str, metavar="DIR", help="Batch mode: root directory with numbered subdirs")
    parser.add_argument("--start", type=int, default=0, help="Batch: start directory number (default: 0)")
    parser.add_argument("--end", type=int, default=75, help="Batch: end directory number (default: 75)")
    parser.add_argument("--workers", type=int, default=None, help="Batch: parallel workers (default: CPU count)")
    parser.add_argument("--dry-run", action="store_true", help="Batch: show what would be processed")

    args = parser.parse_args()

    if args.batch:
        core_dir = Path(args.batch).resolve()
        if not core_dir.exists():
            print(f"Directory not found: {core_dir}", file=sys.stderr)
            sys.exit(1)
        _run_batch(core_dir, args.start, args.end, args.workers, args.dry_run)
    elif args.directory:
        input_dir = Path(args.directory).resolve()
        if not input_dir.exists() or not input_dir.is_dir():
            print(f"Input directory not found: {input_dir}", file=sys.stderr)
            sys.exit(1)
        output_tar = Path(args.output).resolve() if args.output else input_dir.with_suffix(".tar")
        process_directory(input_dir, output_tar)
    else:
        parser.error("Provide a directory or use --batch")


if __name__ == "__main__":
    main()
