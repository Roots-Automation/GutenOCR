"""
GroundingBench build script.

Subcommands
-----------
assign
    Iterate rankings.csv top-to-bottom, pre-validate each sample (non-empty
    text.lines), collect the first N*per_task valid rows, shuffle with a
    fixed seed, and write a 'task' column (values 1–N) back to rankings.csv.
    Run once; the result is checked into the repo.

    Usage:
        python build.py assign <data_dir> <rankings_csv> [--per-task 100] [--seed 42]

sample
    Read rankings.csv, filter rows where task == <task>, and copy each
    image + JSON pair to output_dir.

    Usage:
        python build.py sample <data_dir> <rankings_csv> <output_dir> --task 1
"""

import argparse
import json
import os
import random
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


# ---------------------------------------------------------------------------
# assign subcommand
# ---------------------------------------------------------------------------


def cmd_assign(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.data_dir):
        print(f"Error: data_dir not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(args.rankings_csv)
    except Exception as e:
        print(f"Error reading rankings CSV: {e}", file=sys.stderr)
        sys.exit(1)

    required_cols = {"rank", "filename"}
    if not required_cols.issubset(df.columns):
        print(
            f"Error: rankings CSV must contain columns: {required_cols}",
            file=sys.stderr,
        )
        sys.exit(1)

    if "task" in df.columns:
        existing = df["task"].notna().sum()
        if existing > 0:
            print(
                f"Warning: 'task' column already has {existing} assigned rows. "
                "Pass --force to overwrite.",
                file=sys.stderr,
            )
            if not getattr(args, "force", False):
                sys.exit(1)

    df = df.sort_values("rank").reset_index(drop=True)

    total_needed = args.num_tasks * args.per_task
    valid_indices: list[int] = []
    skipped_missing = 0
    skipped_no_lines = 0

    print(f"Scanning for {total_needed} valid samples (top rows in rank order)...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
        if len(valid_indices) >= total_needed:
            break

        stem = os.path.splitext(os.path.basename(row["filename"]))[0]
        json_path = os.path.join(args.data_dir, stem + ".json")
        img_path = find_image(args.data_dir, stem)

        if img_path is None or not os.path.exists(json_path):
            skipped_missing += 1
            continue

        if not has_lines(json_path):
            skipped_no_lines += 1
            continue

        valid_indices.append(idx)

    if len(valid_indices) < total_needed:
        print(
            f"Error: only {len(valid_indices)} valid samples found, "
            f"need {total_needed}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Shuffle with fixed seed, then assign tasks round-robin
    rng = random.Random(args.seed)
    rng.shuffle(valid_indices)

    # Build task labels: [1,1,...,1, 2,2,...,2, ...] → one per valid index
    task_labels = []
    for t in range(1, args.num_tasks + 1):
        task_labels.extend([t] * args.per_task)

    # Write task column — unassigned rows get NaN
    df["task"] = pd.NA
    for df_idx, task in zip(valid_indices, task_labels):
        df.at[df_idx, "task"] = task

    # Convert to nullable int so assigned rows show as integers, not floats
    df["task"] = df["task"].astype(pd.Int64Dtype())

    df.to_csv(args.rankings_csv, index=False)

    counts = {t: (df["task"] == t).sum() for t in range(1, args.num_tasks + 1)}
    print(f"\nDone. Task assignments written to {args.rankings_csv}")
    print(f"  Skipped (missing files)  : {skipped_missing}")
    print(f"  Skipped (no text.lines)  : {skipped_no_lines}")
    for t, n in counts.items():
        print(f"  Task {t}: {n} samples")


# ---------------------------------------------------------------------------
# sample subcommand
# ---------------------------------------------------------------------------


def cmd_sample(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.data_dir):
        print(f"Error: data_dir not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(args.rankings_csv)
    except Exception as e:
        print(f"Error reading rankings CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if "task" not in df.columns:
        print(
            "Error: rankings CSV has no 'task' column. Run 'build.py assign' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    task_df = df[df["task"] == args.task]
    if task_df.empty:
        print(
            f"Error: no rows with task == {args.task} in {args.rankings_csv}.",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    copied = 0
    skipped_missing = 0

    for _, row in tqdm(task_df.iterrows(), total=len(task_df), desc=f"Task {args.task}"):
        stem = os.path.splitext(os.path.basename(row["filename"]))[0]
        img_path = find_image(args.data_dir, stem)
        json_path = os.path.join(args.data_dir, stem + ".json")

        if img_path is None or not os.path.exists(json_path):
            skipped_missing += 1
            print(f"  Missing: {stem}", file=sys.stderr)
            continue

        shutil.copyfile(img_path, os.path.join(args.output_dir, os.path.basename(img_path)))
        shutil.copyfile(json_path, os.path.join(args.output_dir, stem + ".json"))
        copied += 1

    print(f"\nDone.")
    print(f"  Copied  : {copied}")
    print(f"  Missing : {skipped_missing}")
    print(f"  Output  : {args.output_dir}")

    if skipped_missing > 0:
        print(
            f"\nWarning: {skipped_missing} files were missing from {args.data_dir}.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GroundingBench build script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- assign ---
    p_assign = sub.add_parser(
        "assign",
        help="Assign task numbers to top-N diverse valid samples in rankings.csv.",
    )
    p_assign.add_argument("data_dir", help="Directory containing image and JSON pairs")
    p_assign.add_argument("rankings_csv", help="Rankings CSV (modified in-place)")
    p_assign.add_argument(
        "--per-task",
        type=int,
        default=100,
        help="Samples per task (default: 100)",
    )
    p_assign.add_argument(
        "--num-tasks",
        type=int,
        default=4,
        help="Number of tasks (default: 4)",
    )
    p_assign.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for task shuffling (default: 42)",
    )
    p_assign.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing task assignments without prompting",
    )
    p_assign.set_defaults(func=cmd_assign)

    # --- sample ---
    p_sample = sub.add_parser(
        "sample",
        help="Copy image+JSON pairs for a given task to output_dir.",
    )
    p_sample.add_argument("data_dir", help="Directory containing image and JSON pairs")
    p_sample.add_argument("rankings_csv", help="Rankings CSV with 'task' column")
    p_sample.add_argument("output_dir", help="Directory to write selected samples")
    p_sample.add_argument(
        "--task",
        type=int,
        required=True,
        help="Task number to sample (e.g. 1, 2, 3, 4)",
    )
    p_sample.set_defaults(func=cmd_sample)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
