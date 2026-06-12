"""Corpus symbol coverage measurement.

Given a list of LaTeX formula strings and a declared symbol set, measures
which symbols appear in the corpus and at what frequency.

Coverage check uses substring matching: a symbol is "covered" if it appears
as a substring in at least one formula string.  This is appropriate for
multi-character LaTeX commands (e.g. \\alpha, \\mathcal{A}) but unreliable
for single-character symbols — use symbol_inventory.MUST_COVER (multi-char
only) to avoid false positives.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CoverageReport:
    total_symbols: int
    covered: frozenset[str]
    missing: frozenset[str]
    per_symbol_count: dict[str, int]

    @property
    def coverage_fraction(self) -> float:
        return len(self.covered) / self.total_symbols if self.total_symbols else 1.0

    def summary(self) -> str:
        lines = [
            f"Coverage: {len(self.covered)}/{self.total_symbols} ({self.coverage_fraction:.1%})",
        ]
        if self.missing:
            lines.append(f"Missing ({len(self.missing)}): {sorted(self.missing)}")
        return "\n".join(lines)


def measure_coverage(corpus: list[str], symbol_set: list[str]) -> CoverageReport:
    """Check which symbols from symbol_set appear in the corpus.

    Args:
        corpus: List of LaTeX formula strings.
        symbol_set: Symbols/tokens to check for presence.

    Returns:
        CoverageReport with per-symbol counts and aggregate statistics.
    """
    per_symbol_count: dict[str, int] = {}
    for sym in symbol_set:
        per_symbol_count[sym] = sum(1 for formula in corpus if sym in formula)

    covered = frozenset(sym for sym, count in per_symbol_count.items() if count > 0)
    missing = frozenset(sym for sym in symbol_set if per_symbol_count.get(sym, 0) == 0)

    return CoverageReport(
        total_symbols=len(symbol_set),
        covered=covered,
        missing=missing,
        per_symbol_count=per_symbol_count,
    )
