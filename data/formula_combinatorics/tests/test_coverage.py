"""CI coverage tests: fixed-seed corpus must cover the declared symbol set.

MUST_COVER contains multi-character LaTeX tokens from the shared vocabulary
pools.  A corpus of COVERAGE_N formulas with COVERAGE_SEED must contain every
token as a substring in at least one formula.

Run:
    pytest tests/test_coverage.py -v
"""

from __future__ import annotations

import pytest
from formula_combinatorics._coverage import CoverageReport, measure_coverage
from formula_combinatorics.corpus import generate
from formula_combinatorics.domains import DEFAULT_WEIGHTS, GENERATORS
from formula_combinatorics.symbol_inventory import (
    COVERAGE_N,
    COVERAGE_SEED,
    MUST_COVER,
    collect_should_cover,
)

_DOMAINS = list(GENERATORS.keys())


@pytest.fixture(scope="module")
def fixed_corpus() -> list[str]:
    """Generate a fixed deterministic corpus once per test session."""
    corpus_dict = generate(
        count=COVERAGE_N,
        domains=_DOMAINS,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=COVERAGE_SEED,
    )
    return list(corpus_dict.values())


def test_must_cover_symbols(fixed_corpus: list[str]) -> None:
    """Every MUST_COVER symbol must appear as a substring in the fixed corpus."""
    report = measure_coverage(fixed_corpus, sorted(MUST_COVER))
    print(report.summary())
    assert report.coverage_fraction == 1.0, f"Missing {len(report.missing)} symbols from MUST_COVER:\n" + "\n".join(
        f"  {sym!r}" for sym in sorted(report.missing)
    )


def test_coverage_report_structure(fixed_corpus: list[str]) -> None:
    """Smoke test: measure_coverage returns a well-formed CoverageReport."""
    sample_symbols = [r"\alpha", r"\frac", "nonexistent_xyz_abc"]
    report = measure_coverage(fixed_corpus, sample_symbols)
    assert isinstance(report, CoverageReport)
    assert report.total_symbols == 3
    assert r"\alpha" in report.covered
    assert "nonexistent_xyz_abc" in report.missing
    assert 0.0 <= report.coverage_fraction <= 1.0


def test_should_cover_collection_smoke() -> None:
    """collect_should_cover() returns a non-empty frozenset without errors."""
    should_cover = collect_should_cover()
    assert isinstance(should_cover, frozenset)
    assert len(should_cover) > 0
    assert "" not in should_cover


def test_should_cover_contains_greek() -> None:
    """SHOULD_COVER (Slot pools) contains the core Greek letter pools.

    Greek letters are used as Slot pools in virtually every domain.
    This verifies that collect_should_cover() correctly introspects those slots.
    """
    from formula_combinatorics._vocab import _GREEK

    should_cover = collect_should_cover()
    missing = frozenset(_GREEK) - should_cover
    assert not missing, f"Greek letters missing from any Slot pool: {sorted(missing)}"
