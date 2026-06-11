"""Unit tests for formula_combinatorics/_template_dsl.py."""

from __future__ import annotations

import math
import random

import pytest
from formula_combinatorics._template_dsl import (
    _IDX_POOL,
    _N_IDX,
    EP,
    E,
    S,
    Template,
    X,
    _decorate,
    compute_weights,
    make_dispatcher,
    n_eff,
    sample,
)

_SMALL_POOL: tuple[str, ...] = ("a", "b", "c", "d", "e")


def _rng(seed: int = 0) -> random.Random:
    return random.Random(seed)


# ---------------------------------------------------------------------------
# n_eff correctness
# ---------------------------------------------------------------------------


def test_n_eff_single_slot_no_idx() -> None:
    t = Template("t", "{v}", slots={"v": S(_SMALL_POOL)})
    assert n_eff(t) == len(_SMALL_POOL)


def test_n_eff_single_slot_with_idx() -> None:
    t = Template("t", "{v}", slots={"v": S(_SMALL_POOL, idx=0.5)})
    expected = len(_SMALL_POOL) * (1.0 + 0.5 * (_N_IDX - 1))
    assert abs(n_eff(t) - expected) < 1e-9


def test_n_eff_two_independent_slots() -> None:
    pool_a: tuple[str, ...] = ("x", "y", "z")
    pool_b: tuple[str, ...] = ("p", "q", "r", "s")
    t = Template("t", "{v1} + {v2}", slots={"v1": S(pool_a), "v2": S(pool_b)})
    assert n_eff(t) == len(pool_a) * len(pool_b)


def test_n_eff_exclude_slot() -> None:
    pool: tuple[str, ...] = ("a", "b", "c", "d")
    t = Template("t", "{v1} {v2}", slots={"v1": S(pool), "v2": X(pool, ["v1"])})
    expected = len(pool) * (len(pool) - 1)
    assert n_eff(t) == expected


def test_n_eff_sub_uses_estimate() -> None:
    t = Template("t", "{e}", slots={"e": E(lambda rng: "x", n=500)})
    assert n_eff(t) == 500.0


def test_n_eff_constant_template() -> None:
    t = Template("t", r"x^2 + y^2 = 1", slots={})
    assert n_eff(t) == 1.0


def test_n_eff_distinct_group() -> None:
    pool: tuple[str, ...] = ("x", "y", "z", "t", "u")
    t = Template(
        "t",
        r"{v1} \neq {v2}",
        slots={"v1": S(pool), "v2": S(pool)},
        distinct=[["v1", "v2"]],
    )
    assert n_eff(t) == math.perm(len(pool), 2)


def test_n_eff_variants_sums_leaves() -> None:
    v1 = Template("v1", "{a}", slots={"a": S(("x", "y", "z"))})
    v2 = Template("v2", "{b}", slots={"b": S(("p", "q"))})
    parent = Template("parent", "", slots={}, variants=[v1, v2])
    assert n_eff(parent) == n_eff(v1) + n_eff(v2)


# ---------------------------------------------------------------------------
# sample() correctness
# ---------------------------------------------------------------------------


def test_sample_constant_template_returns_verbatim() -> None:
    latex = r"x^2 + y^2 = 1"
    t = Template("t", latex, slots={})
    assert sample(t, _rng()) == latex


def test_sample_constant_template_no_rng_calls() -> None:
    """The constant short-circuit path must never touch the RNG."""

    class _BrokenRNG:
        def random(self) -> float:
            raise RuntimeError("RNG called on constant template")

        def choice(self, *a: object) -> object:
            raise RuntimeError("RNG called on constant template")

        def choices(self, *a: object, **kw: object) -> object:
            raise RuntimeError("RNG called on constant template")

        def sample(self, *a: object) -> object:
            raise RuntimeError("RNG called on constant template")

    t = Template("t", r"\pi^2 / 6", slots={})
    assert sample(t, _BrokenRNG()) == r"\pi^2 / 6"  # type: ignore[arg-type]


def test_sample_slot_draws_from_pool() -> None:
    pool: tuple[str, ...] = ("x", "y", "z")
    t = Template("t", "{v}", slots={"v": S(pool)})
    for seed in range(50):
        assert sample(t, _rng(seed)) in pool


def test_sample_exclude_slot_never_equal() -> None:
    pool: tuple[str, ...] = ("a", "b", "c", "d")
    t = Template("t", "{v1} {v2}", slots={"v1": S(pool), "v2": X(pool, ["v1"])})
    rng = random.Random(0)
    for _ in range(100):
        v1, v2 = sample(t, rng).split()
        assert v1 != v2, f"ExcludeSlot failed: {v1!r} == {v2!r}"


def test_sample_distinct_group_pairwise_different() -> None:
    pool: tuple[str, ...] = ("a", "b", "c", "d", "e", "f")
    t = Template(
        "t",
        r"{v1}|{v2}|{v3}",
        slots={"v1": S(pool), "v2": S(pool), "v3": S(pool)},
        distinct=[["v1", "v2", "v3"]],
    )
    rng = random.Random(0)
    for _ in range(100):
        parts = sample(t, rng).split("|")
        assert len(set(parts)) == 3, f"distinct group has duplicates: {parts}"


def test_sample_variants_reaches_all_branches() -> None:
    v1 = Template("v1", "AAA", slots={})
    v2 = Template("v2", "BBB", slots={})
    parent = Template("parent", "", slots={}, variants=[v1, v2])
    results = {sample(parent, _rng(i)) for i in range(50)}
    assert "AAA" in results and "BBB" in results


# ---------------------------------------------------------------------------
# _decorate
# ---------------------------------------------------------------------------


def test_decorate_idx_zero_never_decorates() -> None:
    rng = random.Random(0)
    for _ in range(100):
        assert _decorate("x", 0.0, rng) == "x"


def test_decorate_idx_one_always_adds_subscript() -> None:
    rng = random.Random(0)
    for _ in range(100):
        result = _decorate("x", 1.0, rng)
        assert result.startswith("x_{") and result.endswith("}")


def test_decorate_subscript_stays_within_idx_pool() -> None:
    rng = random.Random(0)
    seen: set[str] = set()
    for _ in range(300):
        result = _decorate("x", 1.0, rng)
        sub = result[len("x_{") : -1]
        seen.add(sub)
    assert seen <= set(_IDX_POOL), f"Unexpected subscripts: {seen - set(_IDX_POOL)}"
    assert len(seen) > 1, "Only one subscript seen across 300 draws"


# ---------------------------------------------------------------------------
# compute_weights and make_dispatcher
# ---------------------------------------------------------------------------


def test_compute_weights_returns_positive_finite_floats() -> None:
    templates = [
        Template("t1", "{v}", slots={"v": S(_SMALL_POOL)}),
        Template("t2", "{v1}{v2}", slots={"v1": S(_SMALL_POOL), "v2": S(_SMALL_POOL)}),
    ]
    weights = compute_weights(templates)
    assert len(weights) == 2
    for w in weights:
        assert w > 0
        assert math.isfinite(w)


def test_compute_weights_larger_n_eff_gets_higher_weight() -> None:
    small = Template("small", "{v}", slots={"v": S(("a", "b"))})
    large = Template("large", "{v}", slots={"v": S(tuple("abcdefghijklmnopqrstuvwxyz"))})
    w_small, w_large = compute_weights([small, large])
    assert w_large > w_small


def test_compute_weights_sqrt_of_min_n_eff_cap() -> None:
    huge = Template("huge", "{e}", slots={"e": E(lambda rng: "x", n=1e12)})
    cap = 1_000_000
    (w,) = compute_weights([huge], cap=cap)
    assert abs(w - math.sqrt(cap)) < 1e-6


def test_make_dispatcher_is_callable() -> None:
    templates = [Template("t", "{v}", slots={"v": S(_SMALL_POOL)})]
    weights = compute_weights(templates)
    dispatcher = make_dispatcher(templates, weights)
    assert callable(dispatcher)
    assert dispatcher(_rng()) in _SMALL_POOL


def test_make_dispatcher_samples_all_templates() -> None:
    t1 = Template("t1", "AAA", slots={})
    t2 = Template("t2", "BBB", slots={})
    dispatcher = make_dispatcher([t1, t2], [1.0, 1.0])
    rng = random.Random(0)
    results = {dispatcher(rng) for _ in range(100)}
    assert results == {"AAA", "BBB"}


# ---------------------------------------------------------------------------
# ExcludeSlot empty-pool guard
# ---------------------------------------------------------------------------


def test_exclude_slot_raises_on_empty_pool() -> None:
    """ExcludeSlot must raise ValueError (not IndexError) when all pool entries
    are excluded, so misconfigured templates surface loudly instead of being
    silently swallowed by corpus.generate()'s exception handler."""
    tiny_pool: tuple[str, ...] = ("a",)
    t = Template("t", "{v1}{v2}", slots={"v1": S(tiny_pool), "v2": X(tiny_pool, ["v1"])})
    with pytest.raises(ValueError, match="ExcludeSlot pool is empty"):
        sample(t, _rng(0))


# ---------------------------------------------------------------------------
# ExcludeParamSub
# ---------------------------------------------------------------------------


def test_exclude_param_sub_produces_non_empty_string() -> None:
    choices = ("p", "q", "r", "s")

    def _gen(rng: random.Random, param: str, exclude: frozenset[str]) -> str:
        pool = [x for x in choices if x not in exclude and x != param]
        return rng.choice(pool) if pool else "fallback"

    t = Template(
        "t",
        "{n}{c2}",
        slots={
            "n": S(("a", "b", "c")),
            "c2": EP(_gen, param="n", exclude=["n"], n=3),
        },
    )
    rng = random.Random(0)
    for _ in range(50):
        result = sample(t, rng)
        assert isinstance(result, str) and len(result) > 0
