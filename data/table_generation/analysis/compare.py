"""Compare synthetic generator distribution against real TSR datasets.

Usage:
    python -m analysis.compare [--datasets fintabnet] [--syn-samples 10000] [--seed 42]

Real dataset stats are cached to analysis/cache/<name>.json after first download
so subsequent runs skip streaming entirely.

Add new datasets by registering a loader in DATASET_LOADERS below.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset

_pkg_root = str(Path(__file__).parent.parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from analysis.html_to_otsl import html_table_to_otsl, mustard_to_otsl
from analysis.stats import analyze_otsl, print_comparison, summarize
from content_distribution import ContentDistribution
from otsl import structure_to_otsl
from table_structure import generate_table_structure

_CACHE_DIR = Path(__file__).parent / "cache"


# ── Dataset loaders ───────────────────────────────────────────────────────────
# Each loader is a zero-arg callable that yields lists of OTSL tokens.


def _stream(repo: str, split: str = "train") -> list[list[str]]:
    print(f"Loading {repo} {split} split (streaming)...", flush=True)
    ds = load_dataset(repo, split=split, streaming=True)
    # Drop image column to avoid decoding overhead and potential decode errors
    ds = ds.select_columns(["otsl"])
    result = []
    for i, row in enumerate(ds):
        result.append(row["otsl"])
        if (i + 1) % 50_000 == 0:
            print(f"  {i + 1} rows...", flush=True)
    print(f"  {len(result)} rows total", flush=True)
    return result


def _load_cached(name: str, loader_fn, *, no_cache: bool = False) -> dict[str, Any]:
    """Return a summarized report for a real dataset, using cache when available."""
    _CACHE_DIR.mkdir(exist_ok=True)
    cache_path = _CACHE_DIR / f"{name}.json"
    if cache_path.exists() and not no_cache:
        print(f"  {name}: loading from cache ({cache_path})", flush=True)
        report = json.loads(cache_path.read_text())
        # JSON keys are always strings; restore int keys for histogram dicts
        for key in ("rows_hist", "cols_hist"):
            if key in report:
                report[key] = {int(k): v for k, v in report[key].items()}
        return report
    tokens_list = loader_fn()
    stats = [s for t in tokens_list if (s := analyze_otsl(t))]
    print(f"  {len(stats)} valid samples from {name}", flush=True)
    if not stats:
        print(f"  WARNING: no valid samples for {name}, skipping.", flush=True)
        return {}
    report = summarize(stats)
    cache_path.write_text(json.dumps(report))
    print(f"  {name}: cached to {cache_path}", flush=True)
    return report


def _load_fintabnet() -> list[list[str]]:
    return _stream("docling-project/FinTabNet_OTSL")


def _load_pubtabnet() -> list[list[str]]:
    return _stream("docling-project/PubTabNet_OTSL")


def _load_synthtabnet() -> list[list[str]]:
    return _stream("docling-project/SynthTabNet_OTSL")


def _load_pubtables1m_v11() -> list[list[str]]:
    return _stream("docling-project/PubTables-1M_OTSL-v1.1")


def _load_mustard() -> list[list[str]]:
    print("Loading bevaya/MUSTARD test split (streaming)...", flush=True)
    ds = load_dataset("bevaya/MUSTARD", split="test", streaming=True)
    ds = ds.select_columns(["otsl"])
    result = []
    for row in ds:
        tokens = mustard_to_otsl(row["otsl"] or "")
        if tokens:
            result.append(tokens)
    print(f"  {len(result)} rows total", flush=True)
    return result


def _load_html_dataset(
    repo: str,
    html_col: str,
    split: str = "train",
    semantic: bool = False,
    config: str | None = None,
    max_rows: int | None = None,
) -> list[list[str]]:
    label = f"{repo}[{config}]" if config else repo
    print(f"Loading {label} {split} split (streaming)...", flush=True)
    kwargs = {"split": split, "streaming": True}
    if config:
        kwargs["name"] = config
    ds = load_dataset(repo, **kwargs)
    ds = ds.select_columns([html_col])
    result = []
    for i, row in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        tokens = html_table_to_otsl(row[html_col] or "", semantic=semantic)
        if tokens:
            result.append(tokens)
        if (i + 1) % 10_000 == 0:
            print(f"  {i + 1} rows...", flush=True)
    print(f"  {len(result)} rows total", flush=True)
    return result


def _load_multihiertt() -> list[list[str]]:
    # tables field is a list of HTML strings per example; flatten all tables
    print("Loading bevaya/MultiHiertt train split (streaming)...", flush=True)
    ds = load_dataset("bevaya/MultiHiertt", split="train", streaming=True)
    ds = ds.select_columns(["tables"])
    result = []
    for i, row in enumerate(ds):
        for html in row["tables"] or []:
            tokens = html_table_to_otsl(html, semantic=False)
            if tokens:
                result.append(tokens)
    print(f"  {len(result)} rows total", flush=True)
    return result


def _load_hitab() -> list[list[str]]:
    return _load_html_dataset("bevaya/HiTab-StatCan-NSF", "table_html", split="train", semantic=True)


def _load_entrant() -> list[list[str]]:
    configs = ["18-K", "485BPOS", "497", "10-KT", "S-1", "8-K", "20-F", "S-4", "10-K", "10-Q"]
    result = []
    for cfg in configs:
        result.extend(
            _load_html_dataset("bevaya/ENTRANT", "html", split="train", semantic=True, config=cfg, max_rows=100_000)
        )
    return result


DATASET_LOADERS: dict[str, Any] = {
    "fintabnet": _load_fintabnet,
    "pubtabnet": _load_pubtabnet,
    "synthtabnet": _load_synthtabnet,
    "pubtables1m-v1.1": _load_pubtables1m_v11,
    "mustard": _load_mustard,
    "multihiertt": _load_multihiertt,
    "hitab": _load_hitab,
    "entrant": _load_entrant,
}


# ── Synthetic generator ───────────────────────────────────────────────────────


def _generate_synthetic(n: int, seed: int) -> list[list[str]]:
    print(f"Generating {n} synthetic samples (seed={seed})...", flush=True)
    rng = random.Random(seed)
    dist = ContentDistribution()
    results = []
    for _ in range(n):
        structure = generate_table_structure(rng)
        grid = []
        for r in range(structure.rows):
            row = []
            for c in range(structure.cols):
                if rng.random() < 0.15:
                    row.append("")
                else:
                    row.append(dist.sample_cell_content(rng))
            grid.append(row)
        tokens = structure_to_otsl(structure, grid, flavor="semantic").split()
        results.append(tokens)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare table structure distributions.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_LOADERS),
        choices=list(DATASET_LOADERS),
        help="Which real datasets to include.",
    )
    parser.add_argument("--syn-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached dataset stats and re-stream.")
    args = parser.parse_args()

    reports: dict[str, dict] = {}

    syn_tokens = _generate_synthetic(args.syn_samples, args.seed)
    syn_stats = [s for t in syn_tokens if (s := analyze_otsl(t))]
    reports["synthetic"] = summarize(syn_stats)

    for name in args.datasets:
        report = _load_cached(name, DATASET_LOADERS[name], no_cache=args.no_cache)
        if report:
            reports[name] = report

    print()
    print_comparison(reports)


if __name__ == "__main__":
    main()
