#!/usr/bin/env python3
"""Generate SynthDoG data for a range of directory IDs.

Replaces run_synthdog_range.sh with proper error handling and resume support.

Usage:
    python run_synthdog_range.py --start 35 --end 75
    python run_synthdog_range.py --start 10 --end 20 --samples 5000 --workers 64
    python run_synthdog_range.py --start 0 --end 5 --config ../config/config_en.yaml
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SynthDoG data for a range of directory IDs")
    parser.add_argument("--start", type=int, default=35, help="Starting directory ID (default: 35)")
    parser.add_argument("--end", type=int, default=75, help="Ending directory ID (default: 75)")
    parser.add_argument("--samples", type=int, default=16784, help="Samples per directory (default: 16784)")
    parser.add_argument("--workers", type=int, default=128, help="Number of workers (default: 128)")
    parser.add_argument(
        "--config",
        type=str,
        default="../config/config_en-pdfs.yaml",
        help="Config file path (default: ../config/config_en-pdfs.yaml)",
    )
    parser.add_argument("--base-dir", type=str, default="./outputs", help="Base output directory (default: ./outputs)")

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        print("Ensure you're running from the synthdog_grounding/ directory.", file=sys.stderr)
        sys.exit(1)

    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    print("SynthDoG Range Generator", file=sys.stderr)
    print(f"  Output: {base_dir}", file=sys.stderr)
    print(f"  Range: {args.start:04d} to {args.end:04d}", file=sys.stderr)
    print(f"  Samples/dir: {args.samples}", file=sys.stderr)
    print(f"  Workers: {args.workers}", file=sys.stderr)
    print(f"  Config: {args.config}", file=sys.stderr)
    print(file=sys.stderr)

    failures: list[str] = []

    for i in range(args.start, args.end + 1):
        num = f"{i:04d}"
        out_dir = base_dir / num

        # Skip non-empty directories (resume-safe)
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"[{num}] Skipping (directory not empty)", file=sys.stderr)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{num}] Generating {args.samples} samples...", file=sys.stderr)
        cmd = [
            sys.executable,
            "-m",
            "synthtiger",
            "-o",
            str(out_dir),
            "-c",
            str(args.samples),
            "-w",
            str(args.workers),
            "-v",
            "../template.py",
            "SynthDoG",
            str(args.config),
        ]

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"[{num}] FAILED (exit code {result.returncode})", file=sys.stderr)
            failures.append(num)
        else:
            print(f"[{num}] Done", file=sys.stderr)

    print(file=sys.stderr)
    if failures:
        print(f"Completed with {len(failures)} failure(s): {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)
    else:
        total = args.end - args.start + 1
        print(f"Successfully processed {total} directories ({args.start:04d} to {args.end:04d})", file=sys.stderr)


if __name__ == "__main__":
    main()
