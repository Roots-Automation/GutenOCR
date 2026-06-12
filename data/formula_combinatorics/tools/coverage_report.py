#!/usr/bin/env python3
"""
Coverage Report — symbol coverage analysis for formula domain generators.

Generates a corpus of LaTeX formulas and reports which declared symbols
from MUST_COVER appear in the output, at what frequency, and optionally
broken down by domain or by symbol stratum.

Usage:
    python3 coverage_report.py                        # default n=5000, seed=0
    python3 coverage_report.py --n 10000 --seed 42   # custom corpus size
    python3 coverage_report.py --by-stratum           # per-stratum breakdown
    python3 coverage_report.py --by-domain            # per-domain breakdown
    python3 coverage_report.py --by-stratum --by-domain  # both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formula_combinatorics._coverage import measure_coverage, measure_coverage_by_domain
from formula_combinatorics.corpus import generate
from formula_combinatorics.domains import DEFAULT_WEIGHTS, GENERATORS
from formula_combinatorics.symbol_inventory import MUST_COVER, SYMBOL_STRATA

# ── terminal colours ──────────────────────────────────────────────────────────
_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def BOLD(t: str) -> str:
    return _c("1", t)


def DIM(t: str) -> str:
    return _c("2", t)


def GREEN(t: str) -> str:
    return _c("32", t)


def RED(t: str) -> str:
    return _c("31", t)


def YELLOW(t: str) -> str:
    return _c("33", t)


# ── report helpers ────────────────────────────────────────────────────────────


def _coverage_bar(frac: float, width: int = 20) -> str:
    filled = round(frac * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _frac_colour(frac: float) -> str:
    if frac == 1.0:
        return GREEN(f"{frac:.1%}")
    if frac >= 0.9:
        return YELLOW(f"{frac:.1%}")
    return RED(f"{frac:.1%}")


# ── output modes ──────────────────────────────────────────────────────────────


def run_flat(corpus: list[str], n: int, seed: int) -> None:
    """Print flat symbol → count table plus summary."""
    symbols = sorted(MUST_COVER)
    report = measure_coverage(corpus, symbols)

    print(f"\n{BOLD('Symbol Coverage')}  {DIM(f'n={n:,}  seed={seed}  symbols={len(symbols)}')}\n")
    print(f"  {'symbol':<36}  {'count':>6}  status")
    print("  " + "─" * 58)

    for sym in symbols:
        count = report.per_symbol_count.get(sym, 0)
        status = GREEN("✓") if count > 0 else RED("✗ MISSING")
        print(f"  {sym:<36}  {count:>6,}  {status}")

    print()
    bar = _coverage_bar(report.coverage_fraction)
    frac_str = _frac_colour(report.coverage_fraction)
    print(f"  {BOLD('Total:')}  {bar}  {frac_str}  ({len(report.covered)}/{report.total_symbols} symbols)")
    if report.missing:
        print(f"  {RED('Missing:')} {sorted(report.missing)}")
    print()


def run_by_stratum(corpus: list[str], n: int, seed: int) -> None:
    """Print per-stratum coverage breakdown."""
    print(f"\n{BOLD('Coverage by Stratum')}  {DIM(f'n={n:,}  seed={seed}')}\n")
    print(f"  {'stratum':<18}  {'symbols':>7}  {'covered':>7}  {'%':>7}  progress")
    print("  " + "─" * 68)

    for stratum_name, symbols in sorted(SYMBOL_STRATA.items()):
        report = measure_coverage(corpus, sorted(symbols))
        bar = _coverage_bar(report.coverage_fraction, width=16)
        frac_str = _frac_colour(report.coverage_fraction)
        print(f"  {stratum_name:<18}  {len(symbols):>7}  {len(report.covered):>7}  {frac_str:>7}  {DIM(bar)}")
        if report.missing:
            for sym in sorted(report.missing):
                print(f"    {RED('✗')} {sym}")
    print()


def run_by_domain(corpus_with_meta: list[dict], n: int, seed: int) -> None:
    """Print per-domain coverage fraction for MUST_COVER symbols."""
    symbols = sorted(MUST_COVER)
    by_domain = measure_coverage_by_domain(corpus_with_meta, symbols)

    print(f"\n{BOLD('Coverage by Domain')}  {DIM(f'n={n:,}  seed={seed}  symbols={len(symbols)}')}\n")
    print(f"  {'domain':<28}  {'formulas':>8}  {'covered':>7}  {'%':>7}  progress")
    print("  " + "─" * 74)

    for domain, report in sorted(by_domain.items(), key=lambda kv: kv[1].coverage_fraction):
        # Count how many formulas this domain contributed
        domain_n = sum(1 for item in corpus_with_meta if item.get("domain") == domain)
        bar = _coverage_bar(report.coverage_fraction, width=16)
        frac_str = _frac_colour(report.coverage_fraction)
        print(f"  {domain:<28}  {domain_n:>8,}  {len(report.covered):>7}  {frac_str:>7}  {DIM(bar)}")
    print()


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5_000,
        metavar="N",
        help="Number of formulas to generate (default: 5000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--by-stratum",
        action="store_true",
        help="Show per-stratum (greek / calligraphic / …) coverage breakdown",
    )
    parser.add_argument(
        "--by-domain",
        action="store_true",
        help="Show per-domain coverage fraction table",
    )
    args = parser.parse_args()

    domains = list(DEFAULT_WEIGHTS.keys())
    need_meta = args.by_domain

    print(f"{DIM(f'Generating {args.n:,} formulas (seed={args.seed})...')}", end=" ", flush=True)
    corpus_dict = generate(
        count=args.n,
        domains=domains,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=args.seed,
        include_metadata=need_meta,
    )
    print(DIM("done."))

    if need_meta:
        corpus_with_meta = list(corpus_dict.values())
        corpus = [item["formula"] for item in corpus_with_meta]
    else:
        corpus = list(corpus_dict.values())
        corpus_with_meta = []

    # Always show flat report
    run_flat(corpus, args.n, args.seed)

    if args.by_stratum:
        run_by_stratum(corpus, args.n, args.seed)

    if args.by_domain:
        run_by_domain(corpus_with_meta, args.n, args.seed)


if __name__ == "__main__":
    main()
