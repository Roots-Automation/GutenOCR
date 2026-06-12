"""CI coverage tests: fixed-seed corpus must cover the declared symbol set.

MUST_COVER contains multi-character LaTeX tokens from the shared vocabulary
pools.  A corpus of COVERAGE_N formulas with COVERAGE_SEED must contain every
token as a substring in at least one formula.

Run:
    pytest tests/test_coverage.py -v
"""

from __future__ import annotations

import pytest
from formula_combinatorics._coverage import CoverageReport, measure_coverage, measure_coverage_by_domain
from formula_combinatorics.corpus import generate
from formula_combinatorics.domains import DEFAULT_WEIGHTS, GENERATORS
from formula_combinatorics.symbol_inventory import (
    COVERAGE_N,
    COVERAGE_SEED,
    MUST_COVER,
    SYMBOL_STRATA,
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


@pytest.fixture(scope="module")
def fixed_corpus_with_meta() -> list[dict]:
    """Generate a fixed metadata corpus (includes domain labels) once per session."""
    corpus_dict = generate(
        count=COVERAGE_N,
        domains=_DOMAINS,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=COVERAGE_SEED,
        include_metadata=True,
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


def test_coverage_by_stratum(fixed_corpus: list[str]) -> None:
    """Every stratum in SYMBOL_STRATA must be 100% covered by the fixed corpus.

    SYMBOL_STRATA partitions MUST_COVER into named groups (greek, calligraphic, …).
    Since each symbol already passes test_must_cover_symbols, these roll-up assertions
    verify that the stratum definitions are correct and complete subsets of MUST_COVER.
    """
    failures: list[str] = []
    for stratum_name, symbols in sorted(SYMBOL_STRATA.items()):
        report = measure_coverage(fixed_corpus, sorted(symbols))
        if report.coverage_fraction < 1.0:
            failures.append(f"  {stratum_name}: missing {sorted(report.missing)}")
    assert not failures, "Stratum coverage failures:\n" + "\n".join(failures)


def test_coverage_by_domain_structure(fixed_corpus_with_meta: list[dict]) -> None:
    """measure_coverage_by_domain() returns a report for every sampled domain."""
    by_domain = measure_coverage_by_domain(fixed_corpus_with_meta, sorted(MUST_COVER))
    assert len(by_domain) > 0
    for domain, report in by_domain.items():
        assert isinstance(domain, str)
        assert isinstance(report, CoverageReport)
        assert report.total_symbols == len(MUST_COVER)
        assert 0.0 <= report.coverage_fraction <= 1.0


def test_should_cover_contains_greek() -> None:
    """SHOULD_COVER (Slot pools) contains the core Greek letter pools.

    Greek letters are used as Slot pools in virtually every domain.
    This verifies that collect_should_cover() correctly introspects those slots.
    """
    from formula_combinatorics._vocab import _GREEK

    should_cover = collect_should_cover()
    missing = frozenset(_GREEK) - should_cover
    assert not missing, f"Greek letters missing from any Slot pool: {sorted(missing)}"
