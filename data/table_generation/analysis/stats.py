"""Common statistics extraction from OTSL token sequences."""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

# Base vocab
BASE_TOKENS = ("fcel", "ecel", "lcel", "ucel", "xcel", "nl")
# Semantic extensions
SEMANTIC_TOKENS = ("ched", "rhed", "srow")
ALL_TOKENS = BASE_TOKENS + SEMANTIC_TOKENS


def analyze_otsl(tokens: list[str]) -> dict[str, Any] | None:
    """Extract structural stats from a flat list of OTSL tokens.

    Handles both base (6-token) and semantic (9-token) flavors.
    ched/rhed/srow are treated as primary cells for structural counting.

    Returns None if the sequence is malformed (no nl tokens).
    """
    rows = tokens.count("nl")
    if rows == 0:
        return None

    cols = tokens.index("nl")  # tokens before first nl = cols
    total_cells = rows * cols
    if total_cells == 0:
        return None

    n_fcel = tokens.count("fcel")
    n_ecel = tokens.count("ecel")
    n_lcel = tokens.count("lcel")
    n_ucel = tokens.count("ucel")
    n_xcel = tokens.count("xcel")
    # Semantic tokens — count as primary cells for structural purposes
    n_ched = tokens.count("ched")
    n_rhed = tokens.count("rhed")
    n_srow = tokens.count("srow")

    ext_cells = n_lcel + n_ucel + n_xcel

    return dict(
        rows=rows,
        cols=cols,
        total_cells=total_cells,
        n_fcel=n_fcel,
        n_ecel=n_ecel,
        n_lcel=n_lcel,
        n_ucel=n_ucel,
        n_xcel=n_xcel,
        n_ched=n_ched,
        n_rhed=n_rhed,
        n_srow=n_srow,
        ext_cells=ext_cells,
        has_span=ext_cells > 0,
        has_2d_span=n_xcel > 0,
        has_ched=n_ched > 0,
        has_rhed=n_rhed > 0,
        has_srow=n_srow > 0,
        empty_frac=n_ecel / total_cells,
        ext_frac=ext_cells / total_cells,
    )


def summarize(all_stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a list of per-table stat dicts into a report dict."""

    def _summary(vals: list) -> dict:
        s = sorted(vals)
        n = len(s)
        return dict(
            min=s[0],
            max=s[-1],
            mean=round(statistics.mean(s), 2),
            median=statistics.median(s),
            p10=s[int(n * 0.10)],
            p90=s[int(n * 0.90)],
        )

    def _hist(vals: list[int]) -> dict[int, float]:
        c = Counter(vals)
        total = len(vals)
        return {k: round(100 * c[k] / total, 1) for k in sorted(c)}

    def _pct(key: str) -> float:
        return round(100 * sum(1 for s in all_stats if s[key]) / len(all_stats), 1)

    def _mean_frac(key: str) -> float:
        return round(100 * statistics.mean(s[key] for s in all_stats), 1)

    def _token_pct(key: str) -> float:
        num = sum(s[key] for s in all_stats)
        den = sum(s["total_cells"] for s in all_stats)
        return round(100 * num / den, 1) if den else 0.0

    return dict(
        n=len(all_stats),
        rows=_summary([s["rows"] for s in all_stats]),
        cols=_summary([s["cols"] for s in all_stats]),
        total_cells=_summary([s["total_cells"] for s in all_stats]),
        rows_hist=_hist([s["rows"] for s in all_stats]),
        cols_hist=_hist([s["cols"] for s in all_stats]),
        pct_has_span=_pct("has_span"),
        pct_has_2d_span=_pct("has_2d_span"),
        mean_ext_frac=_mean_frac("ext_frac"),
        mean_empty_frac=_mean_frac("empty_frac"),
        pct_has_ched=_pct("has_ched"),
        pct_has_rhed=_pct("has_rhed"),
        pct_has_srow=_pct("has_srow"),
        token_pct=dict(
            fcel=_token_pct("n_fcel"),
            ecel=_token_pct("n_ecel"),
            lcel=_token_pct("n_lcel"),
            ucel=_token_pct("n_ucel"),
            xcel=_token_pct("n_xcel"),
            ched=_token_pct("n_ched"),
            rhed=_token_pct("n_rhed"),
            srow=_token_pct("n_srow"),
        ),
    )


def print_comparison(reports: dict[str, dict[str, Any]]) -> None:
    """Print a side-by-side comparison of multiple dataset reports."""
    names = list(reports.keys())
    col_w = 14

    def _row(label: str, vals: list) -> None:
        print(f"  {label:<32}" + "".join(f"{str(v):>{col_w}}" for v in vals))

    def _header(title: str) -> None:
        print(f"\n  -- {title} --")

    width = 32 + col_w * len(names)
    print("=" * width)
    print(f"  {'METRIC':<32}" + "".join(f"{n:>{col_w}}" for n in names))
    print("=" * width)

    _header("Sample count")
    _row("n", [r["n"] for r in reports.values()])

    for dim in ("rows", "cols", "total_cells"):
        _header(dim)
        for k in ("min", "max", "mean", "median", "p10", "p90"):
            _row(k, [r[dim][k] for r in reports.values()])

    _header("Spans")
    _row("tables with any span (%)", [r["pct_has_span"] for r in reports.values()])
    _row("tables with 2D span (%)", [r["pct_has_2d_span"] for r in reports.values()])
    _row("ext cells / total (%)", [r["mean_ext_frac"] for r in reports.values()])

    _header("Cell content")
    _row("empty cells / total (%)", [r["mean_empty_frac"] for r in reports.values()])

    _header("Semantic roles (% of tables)")
    _row("tables with ched (%)", [r["pct_has_ched"] for r in reports.values()])
    _row("tables with rhed (%)", [r["pct_has_rhed"] for r in reports.values()])
    _row("tables with srow (%)", [r["pct_has_srow"] for r in reports.values()])

    _header("Token mix (% of all non-nl tokens)")
    for tok in ("fcel", "ecel", "lcel", "ucel", "xcel", "ched", "rhed", "srow"):
        _row(tok, [r["token_pct"][tok] for r in reports.values()])

    _header("Row count distribution (%)")
    all_rows = sorted({k for r in reports.values() for k in r["rows_hist"]})
    for k in all_rows:
        _row(f"  {k} rows", [r["rows_hist"].get(k, 0.0) for r in reports.values()])

    _header("Col count distribution (%)")
    all_cols = sorted({k for r in reports.values() for k in r["cols_hist"]})
    for k in all_cols:
        _row(f"  {k} cols", [r["cols_hist"].get(k, 0.0) for r in reports.values()])

    print()
    print("=" * width)
