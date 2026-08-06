"""Tests for WU6: --coverage-mode generation."""

from __future__ import annotations

import pytest
from formula_combinatorics.corpus import generate
from formula_combinatorics.domains import DEFAULT_WEIGHTS, GENERATORS, PACK_HASHES, PACK_META, TEMPLATES
from formula_combinatorics.engine.symbol_inventory import MUST_COVER


def _extract_formulas(records: dict) -> list[str]:
    result = []
    for v in records.values():
        if isinstance(v, str):
            result.append(v)
        elif isinstance(v, dict):
            result.append(v["formula"])
    return result


@pytest.mark.slow
def test_coverage_mode_satisfies_must_cover():
    """With enough samples, coverage_mode should cover all MUST_COVER symbols ≥ N times."""
    N = 3
    records = generate(
        count=8_000,
        domains=list(DEFAULT_WEIGHTS.keys()),
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=0,
        coverage_mode=True,
        coverage_n=N,
    )
    formulas = _extract_formulas(records)
    from collections import Counter

    counts: Counter[str] = Counter()
    for formula in formulas:
        for sym in MUST_COVER:
            if sym in formula:
                counts[sym] += 1

    missing = [sym for sym in MUST_COVER if counts[sym] < N]
    assert not missing, f"{len(missing)} MUST_COVER symbols appeared fewer than {N} times: {sorted(missing)}"


def test_coverage_mode_terminates_within_budget():
    """coverage_mode terminates; record count is bounded by count * 20 attempts."""
    count = 100
    records = generate(
        count=count,
        domains=list(DEFAULT_WEIGHTS.keys()),
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=1,
        coverage_mode=True,
        coverage_n=1,
    )
    # coverage_mode may generate more than count records (to satisfy coverage), but is
    # bounded by max_attempts = count * 20.
    assert len(records) <= count * 20


def test_coverage_mode_false_is_normal_generation():
    """coverage_mode=False should behave identically to the default (no change to output count)."""
    with_mode = generate(
        count=50,
        domains=["algebra"],
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=0,
        coverage_mode=False,
    )
    without_mode = generate(
        count=50,
        domains=["algebra"],
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=0,
    )
    assert set(with_mode.values()) == set(without_mode.values())


def test_coverage_mode_with_metadata():
    """coverage_mode works correctly when include_metadata=True."""
    records = generate(
        count=200,
        domains=list(DEFAULT_WEIGHTS.keys()),
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=2,
        include_metadata=True,
        pack_hashes=PACK_HASHES,
        pack_meta=PACK_META,
        templates=TEMPLATES,
        coverage_mode=True,
        coverage_n=1,
    )
    for record in records.values():
        assert "formula" in record
        assert "semantic_key" in record
