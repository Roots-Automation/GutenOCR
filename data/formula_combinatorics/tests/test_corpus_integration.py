"""Integration tests for corpus.generate() across all domains."""

from __future__ import annotations

import pytest
from formula_combinatorics.corpus import generate
from formula_combinatorics.domains import DEFAULT_WEIGHTS, DOMAIN_DIFFICULTY, DOMAIN_TAGS, GENERATORS

_ALL_DOMAINS = list(DEFAULT_WEIGHTS.keys())


# ---------------------------------------------------------------------------
# Basic generation contract
# ---------------------------------------------------------------------------


def test_generate_returns_requested_count() -> None:
    result = generate(
        count=200,
        domains=_ALL_DOMAINS,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=0,
    )
    assert len(result) == 200


def test_generate_keys_are_sequential_string_ints() -> None:
    result = generate(
        count=50,
        domains=_ALL_DOMAINS,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=1,
    )
    assert list(result.keys()) == [str(i) for i in range(50)]


def test_generate_values_are_nonempty_strings() -> None:
    result = generate(
        count=100,
        domains=_ALL_DOMAINS,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=2,
    )
    for k, v in result.items():
        assert isinstance(v, str) and len(v) > 0, f"Empty/non-str value at key {k}"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_generate_output_is_deduplicated() -> None:
    result = generate(
        count=500,
        domains=_ALL_DOMAINS,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=3,
    )
    formulas = list(result.values())
    assert len(formulas) == len(set(formulas)), "Duplicate formulas in output"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_generate_deterministic_with_seed() -> None:
    kwargs = dict(
        count=100,
        domains=_ALL_DOMAINS,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
    )
    r1 = generate(**kwargs, seed=42)
    r2 = generate(**kwargs, seed=42)
    assert r1 == r2


def test_generate_different_seeds_differ() -> None:
    kwargs = dict(count=100, domains=_ALL_DOMAINS, generators=GENERATORS, weights=DEFAULT_WEIGHTS)
    r1 = generate(**kwargs, seed=0)
    r2 = generate(**kwargs, seed=1)
    assert r1 != r2


# ---------------------------------------------------------------------------
# Metadata mode
# ---------------------------------------------------------------------------


def test_generate_metadata_mode_structure() -> None:
    result = generate(
        count=100,
        domains=_ALL_DOMAINS,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=5,
        include_metadata=True,
    )
    for k, v in result.items():
        assert isinstance(v, dict), f"Key {k}: expected dict, got {type(v)}"
        assert "formula" in v and "domain" in v, f"Key {k}: missing fields in {v}"
        assert isinstance(v["formula"], str) and len(v["formula"]) > 0
        assert v["domain"] in GENERATORS, f"Key {k}: unknown domain {v['domain']!r}"


def test_generate_metadata_domains_cover_all_requested() -> None:
    result = generate(
        count=500,
        domains=_ALL_DOMAINS,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=6,
        include_metadata=True,
    )
    seen_domains = {v["domain"] for v in result.values()}
    # With 500 samples across 33 domains, expect most to appear at least once.
    assert len(seen_domains) >= 20, f"Only {len(seen_domains)} domains seen in 500 samples"


# ---------------------------------------------------------------------------
# Tag filtering
# ---------------------------------------------------------------------------


def test_generate_tag_filter_restricts_domains() -> None:
    result = generate(
        count=200,
        domains=_ALL_DOMAINS,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=7,
        tags=["foundational"],
        include_metadata=True,
    )
    for v in result.values():
        d = v["domain"]
        assert "foundational" in DOMAIN_TAGS.get(d, []), (
            f"Domain {d!r} passed foundational filter but has tags {DOMAIN_TAGS.get(d)}"
        )


def test_generate_exclude_tag_filter_removes_domains() -> None:
    result = generate(
        count=200,
        domains=_ALL_DOMAINS,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=8,
        exclude_tags=["structural"],
        include_metadata=True,
    )
    for v in result.values():
        d = v["domain"]
        assert "structural" not in DOMAIN_TAGS.get(d, []), (
            f"Domain {d!r} survived structural exclusion but has tags {DOMAIN_TAGS.get(d)}"
        )


def test_domain_difficulty_index_covers_all_domains() -> None:
    for domain in GENERATORS:
        assert domain in DOMAIN_DIFFICULTY, f"{domain!r} missing from DOMAIN_DIFFICULTY"
        assert DOMAIN_DIFFICULTY[domain] in {"elementary", "undergraduate", "graduate", "research"}, (
            f"{domain!r}: unexpected difficulty {DOMAIN_DIFFICULTY[domain]!r}"
        )


def test_generate_raises_after_impossible_tag_filter() -> None:
    import pytest

    with pytest.raises(ValueError, match="No domains remaining"):
        generate(
            count=50,
            domains=_ALL_DOMAINS,
            generators=GENERATORS,
            weights=DEFAULT_WEIGHTS,
            seed=9,
            tags=["nonexistent_tag_xyz"],
        )


# ---------------------------------------------------------------------------
# Single-domain generation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", sorted(GENERATORS.keys()))
def test_single_domain_generates_100_formulas(domain: str) -> None:
    result = generate(
        count=100,
        domains=[domain],
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=0,
    )
    assert len(result) >= 1, f"Domain {domain!r} produced no output"
    for v in result.values():
        assert isinstance(v, str) and len(v) > 0
