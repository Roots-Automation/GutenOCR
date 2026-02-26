"""
Deprecated — use build.py instead.

    python build.py sample <data_dir> <rankings_csv> <output_dir> --task 1

This file is kept only to avoid breaking any cached references.
"""
import sys

print(
    "Error: sample.py has been replaced by build.py.\n"
    "Use: python build.py sample <data_dir> <rankings_csv> <output_dir> --task 1",
    file=sys.stderr,
)
sys.exit(1)
