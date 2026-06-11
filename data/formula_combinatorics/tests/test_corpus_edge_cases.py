"""Edge case tests for formula_combinatorics/corpus.py."""

from __future__ import annotations

import logging
import random

import pytest
from formula_combinatorics.corpus import generate

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
