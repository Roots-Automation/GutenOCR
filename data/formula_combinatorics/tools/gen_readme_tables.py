#!/usr/bin/env python3
"""Print domain table and CLI flags from the live registry.

Run from the package root to check README accuracy:

    python tools/gen_readme_tables.py

Output can be copy-pasted into README.md to keep the docs current.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the project root without installing.
sys.path.insert(0, str(Path(__file__).parent.parent))

from formula_combinatorics.domains._config import DOMAIN_CONFIG


def _domain_table() -> None:
    print("## Domain table (from live registry)\n")
    print("| Domain | File | Weight | Tags | Difficulty |")
    print("|---|---|---|---|---|")
    for name, meta in DOMAIN_CONFIG.items():
        tags = ", ".join(meta.tags) if meta.tags else "—"
        file_guess = f"domains/{name}.py"
        print(f"| `{name}` | `{file_guess}` | {meta.weight} | {tags} | {meta.difficulty} |")
    print(f"\nTotal domains: {len(DOMAIN_CONFIG)}")
    total_w = sum(m.weight for m in DOMAIN_CONFIG.values())
    print(f"Sum of weights: {total_w:.4f} (unnormalized; generate() renormalizes at call time)")


def _cli_flags() -> None:
    import subprocess

    print("\n## CLI flags (from formula-generate --help)\n")

    result = subprocess.run(
        [sys.executable, "-m", "formula_combinatorics.generate", "--help"],
        capture_output=True,
        text=True,
    )
    print(result.stdout or result.stderr)


if __name__ == "__main__":
    _domain_table()
    _cli_flags()
