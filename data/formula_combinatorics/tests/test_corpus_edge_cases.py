"""Edge case tests for formula_combinatorics/corpus.py."""

from __future__ import annotations

import logging
import random

import pytest
from formula_combinatorics.corpus import _TAG_POOL, generate
from formula_combinatorics.domains import DEFAULT_WEIGHTS, GENERATORS

# ---------------------------------------------------------------------------
# max_attempts exhaustion
# ---------------------------------------------------------------------------


def test_max_attempts_returns_partial_output(caplog: pytest.LogCaptureFixture) -> None:
    """A domain with n_eff=1 can produce at most 1 unique formula; requesting more
    should return a partial corpus with a WARNING logged.  Wrapping is disabled so
    the constant generator cannot produce wrapped variants that inflate uniqueness."""

    def constant_gen(rng: random.Random) -> str:
        return "ONLY_ONE_FORMULA"

    with caplog.at_level(logging.WARNING, logger="formula_combinatorics.corpus"):
        result = generate(
            count=50,
            domains=["test"],
            generators={"test": constant_gen},
            weights={"test": 1.0},
            seed=0,
            display_fraction=0.0,
            inline_fraction=0.0,
        )
    assert len(result) == 1
    assert result["0"] == "ONLY_ONE_FORMULA"
    warning_text = " ".join(caplog.messages)
    assert "Generated" in warning_text or "unique" in warning_text.lower()


# ---------------------------------------------------------------------------
# Exception handling
# ---------------------------------------------------------------------------


def test_exception_in_generator_is_skipped_silently() -> None:
    """Generators that raise are caught; remaining samples are still collected."""
    call_count = {"n": 0}

    def flaky_gen(rng: random.Random) -> str:
        call_count["n"] += 1
        if call_count["n"] <= 5:
            raise RuntimeError("deliberate test error")
        return f"formula_{call_count['n']}"

    result = generate(
        count=5,
        domains=["test"],
        generators={"test": flaky_gen},
        weights={"test": 1.0},
        seed=0,
        display_fraction=0.0,
        inline_fraction=0.0,
    )
    assert len(result) == 5
    for v in result.values():
        assert v.startswith("formula_")


def test_always_raising_generator_returns_empty() -> None:
    """If a generator always raises, generate() completes gracefully with 0 formulas."""

    def always_raises(rng: random.Random) -> str:
        raise ValueError("always broken")

    result = generate(
        count=10,
        domains=["test"],
        generators={"test": always_raises},
        weights={"test": 1.0},
        seed=0,
    )
    assert isinstance(result, dict)
    assert len(result) == 0


def test_per_domain_error_tally_warns_on_high_rate(caplog: pytest.LogCaptureFixture) -> None:
    """A domain with 100% error rate triggers the per-domain WARNING."""

    def always_raises(rng: random.Random) -> str:
        raise ValueError("always broken")

    def good_gen(rng: random.Random) -> str:
        return f"good_{rng.randint(0, 99999)}"

    with caplog.at_level(logging.WARNING, logger="formula_combinatorics.corpus"):
        generate(
            count=20,
            domains=["bad", "good"],
            generators={"bad": always_raises, "good": good_gen},
            weights={"bad": 0.5, "good": 0.5},
            seed=0,
            display_fraction=0.0,
            inline_fraction=0.0,
        )

    combined = " ".join(caplog.messages)
    assert "bad" in combined, "Expected WARNING mentioning the broken domain name"
    assert "error" in combined.lower(), "Expected WARNING to mention errors"


def test_per_domain_error_tally_strict_raises() -> None:
    """strict=True raises RuntimeError instead of warning when error rate is high."""

    def always_raises(rng: random.Random) -> str:
        raise ValueError("always broken")

    def good_gen(rng: random.Random) -> str:
        return f"good_{rng.randint(0, 99999)}"

    with pytest.raises(RuntimeError, match="error rate"):
        generate(
            count=20,
            domains=["bad", "good"],
            generators={"bad": always_raises, "good": good_gen},
            weights={"bad": 0.5, "good": 0.5},
            seed=0,
            display_fraction=0.0,
            inline_fraction=0.0,
            strict=True,
        )


def test_per_domain_error_tally_no_warn_below_threshold(caplog: pytest.LogCaptureFixture) -> None:
    """Domains with <10 attempts do not trigger the error-rate WARNING."""
    call_count = {"n": 0}

    def rare_raiser(rng: random.Random) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ValueError("one-off error")
        return f"formula_{call_count['n']}"

    def dominant_gen(rng: random.Random) -> str:
        return f"dom_{rng.randint(0, 99999)}"

    with caplog.at_level(logging.WARNING, logger="formula_combinatorics.corpus"):
        generate(
            count=5,
            domains=["rare", "dominant"],
            generators={"rare": rare_raiser, "dominant": dominant_gen},
            weights={"rare": 0.001, "dominant": 0.999},
            seed=0,
            display_fraction=0.0,
            inline_fraction=0.0,
        )

    # "rare" gets very few attempts due to low weight; should not fire the threshold warning.
    high_rate_warning = any("High generator error rate" in m for m in caplog.messages)
    assert not high_rate_warning, "Should not warn when domain has <10 attempts"


# ---------------------------------------------------------------------------
# Weight normalization
# ---------------------------------------------------------------------------


def test_scaled_weights_give_identical_output() -> None:
    """generate() normalises weights internally, so scaling all weights by a
    constant factor must produce bit-for-bit identical output."""

    def gen_a(rng: random.Random) -> str:
        return f"A{rng.randint(0, 9999)}"

    def gen_b(rng: random.Random) -> str:
        return f"B{rng.randint(0, 9999)}"

    domains = ["a", "b"]
    generators = {"a": gen_a, "b": gen_b}
    common = dict(count=50, domains=domains, generators=generators, seed=77)

    result1 = generate(**common, weights={"a": 0.3, "b": 0.7})
    result2 = generate(**common, weights={"a": 30.0, "b": 70.0})
    assert result1 == result2


def test_extreme_weight_skew_biases_domain() -> None:
    """A domain with 1000× more weight should dominate the corpus."""

    def gen_a(rng: random.Random) -> str:
        return f"A{rng.randint(0, 9999)}"

    def gen_b(rng: random.Random) -> str:
        return f"B{rng.randint(0, 9999)}"

    result = generate(
        count=200,
        domains=["a", "b"],
        generators={"a": gen_a, "b": gen_b},
        weights={"a": 1000.0, "b": 0.001},
        seed=0,
        display_fraction=0.0,
        inline_fraction=0.0,
        include_metadata=True,
    )
    a_count = sum(1 for v in result.values() if v["domain"] == "a")
    assert a_count >= 190, f"Expected 'a' to dominate, got {a_count}/200"


# ---------------------------------------------------------------------------
# Tag filtering edge cases
# ---------------------------------------------------------------------------


def test_impossible_tag_filter_returns_empty() -> None:
    def gen(rng: random.Random) -> str:
        return "formula"

    result = generate(
        count=10,
        domains=["algebra"],
        generators={"algebra": gen},
        weights={"algebra": 1.0},
        seed=0,
        tags=["this_tag_does_not_exist_xyz"],
    )
    assert result == {}


def test_empty_domains_list_returns_empty(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="formula_combinatorics.corpus"):
        result = generate(
            count=10,
            domains=[],
            generators={},
            weights={},
            seed=0,
        )
    assert result == {}


# ---------------------------------------------------------------------------
# Metadata mode
# ---------------------------------------------------------------------------


def test_metadata_mode_returns_domain_field() -> None:
    def _algebra(rng: random.Random) -> str:
        return "algebra_formula"

    def _calculus(rng: random.Random) -> str:
        return "calculus_formula"

    result = generate(
        count=20,
        domains=["algebra", "calculus"],
        generators={"algebra": _algebra, "calculus": _calculus},
        weights={"algebra": 0.5, "calculus": 0.5},
        seed=0,
        include_metadata=True,
    )
    for v in result.values():
        assert isinstance(v, dict)
        assert "formula" in v and "domain" in v
        assert v["domain"] in ("algebra", "calculus")


# ---------------------------------------------------------------------------
# tags + exclude_tags simultaneous interaction
# ---------------------------------------------------------------------------


def test_tags_and_exclude_same_tag_returns_empty(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="formula_combinatorics.corpus"):
        result = generate(
            count=10,
            domains=list(GENERATORS.keys()),
            generators=GENERATORS,
            weights=DEFAULT_WEIGHTS,
            seed=0,
            tags=["foundational"],
            exclude_tags=["foundational"],
        )
    assert result == {}, "Same tag in both tags and exclude_tags should yield empty corpus"


def test_tags_and_exclude_disjoint_leaves_tags_only(caplog: pytest.LogCaptureFixture) -> None:
    result = generate(
        count=20,
        domains=list(GENERATORS.keys()),
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=0,
        tags=["foundational"],
        exclude_tags=["applied"],
        include_metadata=True,
    )
    assert len(result) == 20
    from formula_combinatorics.domains import DOMAIN_TAGS

    for v in result.values():
        domain = v["domain"]
        assert "foundational" in DOMAIN_TAGS.get(domain, []), (
            f"Domain {domain!r} has tag {DOMAIN_TAGS.get(domain)} but should have 'foundational'"
        )
        assert "applied" not in DOMAIN_TAGS.get(domain, []), (
            f"Domain {domain!r} should have been excluded by 'applied' tag"
        )


def test_tags_and_exclude_overlap_trims_correctly() -> None:
    result = generate(
        count=20,
        domains=list(GENERATORS.keys()),
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=0,
        tags=["foundational", "applied"],
        exclude_tags=["applied"],
        include_metadata=True,
    )
    from formula_combinatorics.domains import DOMAIN_TAGS

    for v in result.values():
        domain = v["domain"]
        assert "applied" not in DOMAIN_TAGS.get(domain, []), (
            f"Domain {domain!r} with 'applied' tag should have been excluded"
        )


def test_nonexistent_tag_with_exclude_still_empty() -> None:
    result = generate(
        count=10,
        domains=list(GENERATORS.keys()),
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=0,
        tags=["this_tag_xyz_does_not_exist"],
        exclude_tags=["foundational"],
    )
    assert result == {}


# ---------------------------------------------------------------------------
# _TAG_POOL validation
# ---------------------------------------------------------------------------


def test_tag_pool_all_entries_are_nonempty_strings() -> None:
    for entry in _TAG_POOL:
        assert isinstance(entry, str) and len(entry) > 0, f"_TAG_POOL entry is empty or non-string: {entry!r}"


def test_tag_pool_no_unescaped_braces() -> None:
    for entry in _TAG_POOL:
        assert "{" not in entry and "}" not in entry, (
            f"_TAG_POOL entry {entry!r} contains literal braces — would break \\tag{{...}}"
        )


def test_tag_pool_embedded_in_tag_command_is_balanced() -> None:
    def _balanced(s: str) -> bool:
        stripped = s.replace(r"\{", "").replace(r"\}", "")
        depth = 0
        for ch in stripped:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    return False
        return depth == 0

    for entry in _TAG_POOL:
        wrapped = rf"\tag{{{entry}}}"
        assert _balanced(wrapped), f"\\tag{{{entry!r}}} has unbalanced braces"
