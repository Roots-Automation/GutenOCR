"""Unit tests for formula_combinatorics/_vocab.py."""

from __future__ import annotations

import random

import pytest
from formula_combinatorics._vocab import (
    _BBOLD,
    _BOLD_GREEK,
    _BOLD_VECS,
    _BOUNDS,
    _CALLIGRAPHIC,
    _COEFF_POOL,
    _COMB_K,
    _COMB_N,
    _ELT_POOL,
    _EXP_OP,
    _FOURIER_N,
    _FRAKTUR,
    _FUNC_NAMES,
    _FUNCS,
    _GENERIC_IDX,
    _GREEK,
    _GREEK_SCALARS,
    _GREEK_UPPER,
    _GRP_NAMES,
    _HOMO_POOL,
    _INDICES,
    _LAM_STATS,
    _LOG_BASES,
    _LOOP_N,
    _MATRIX_NAMES,
    _MU_STATS,
    _POS_INTS,
    _PROB_OP_FULL,
    _RELATIONS,
    _RING_NAMES,
    _RV_BASE,
    _SCALARS,
    _SETS,
    _SIG_STATS,
    _STATS_N,
    _VARS,
    _VARS_SCALARS,
    _VEC_POOL,
    _atom,
    _bgreek,
    _bvec,
    _expr,
    _fn_rich,
    _fn_rich_nosub,
    _maybe_idx,
    _overbrace,
    _underbrace,
)

# ---------------------------------------------------------------------------
# Pool integrity
# ---------------------------------------------------------------------------

_ALL_POOLS = [
    ("_VARS", _VARS),
    ("_GREEK", _GREEK),
    ("_GREEK_UPPER", _GREEK_UPPER),
    ("_SCALARS", _SCALARS),
    ("_INDICES", _INDICES),
    ("_POS_INTS", _POS_INTS),
    ("_BOUNDS", _BOUNDS),
    ("_LOOP_N", _LOOP_N),
    ("_VARS_SCALARS", _VARS_SCALARS),
    ("_GREEK_SCALARS", _GREEK_SCALARS),
    ("_FUNC_NAMES", _FUNC_NAMES),
    ("_GENERIC_IDX", _GENERIC_IDX),
    ("_HOMO_POOL", _HOMO_POOL),
    ("_GRP_NAMES", _GRP_NAMES),
    ("_RING_NAMES", _RING_NAMES),
    ("_ELT_POOL", _ELT_POOL),
    ("_LOG_BASES", _LOG_BASES),
    ("_COEFF_POOL", _COEFF_POOL),
    ("_VEC_POOL", _VEC_POOL),
    ("_MATRIX_NAMES", _MATRIX_NAMES),
    ("_CALLIGRAPHIC", _CALLIGRAPHIC),
    ("_FRAKTUR", _FRAKTUR),
    ("_FUNCS", _FUNCS),
    ("_BBOLD", _BBOLD),
    ("_SETS", _SETS),
    ("_BOLD_VECS", _BOLD_VECS),
    ("_BOLD_GREEK", _BOLD_GREEK),
    ("_RELATIONS", _RELATIONS),
    ("_RV_BASE", _RV_BASE),
    ("_STATS_N", _STATS_N),
    ("_COMB_N", _COMB_N),
    ("_COMB_K", _COMB_K),
    ("_LAM_STATS", _LAM_STATS),
    ("_MU_STATS", _MU_STATS),
    ("_SIG_STATS", _SIG_STATS),
    ("_EXP_OP", _EXP_OP),
    ("_PROB_OP_FULL", _PROB_OP_FULL),
    ("_FOURIER_N", _FOURIER_N),
]


@pytest.mark.parametrize("name,pool", _ALL_POOLS)
def test_pool_is_nonempty(name: str, pool: tuple) -> None:
    assert len(pool) > 0, f"{name} is empty"


@pytest.mark.parametrize("name,pool", _ALL_POOLS)
def test_pool_has_no_duplicates(name: str, pool: tuple) -> None:
    dupes = [x for x in pool if pool.count(x) > 1]
    assert not dupes, f"{name} has duplicate entries: {sorted(set(dupes))}"


@pytest.mark.parametrize("name,pool", _ALL_POOLS)
def test_pool_entries_are_nonempty_strings(name: str, pool: tuple) -> None:
    for entry in pool:
        assert isinstance(entry, str) and len(entry) > 0, f"{name}: entry {entry!r} is empty or not a string"


# ---------------------------------------------------------------------------
# Atomic samplers
# ---------------------------------------------------------------------------


def test_atom_returns_nonempty_string() -> None:
    rng = random.Random(0)
    for _ in range(50):
        result = _atom(rng)
        assert isinstance(result, str) and len(result) > 0


def test_atom_balanced_braces() -> None:
    rng = random.Random(42)
    for i in range(100):
        result = _atom(rng)
        stripped = result.replace(r"\{", "").replace(r"\}", "")
        depth = sum(1 if c == "{" else (-1 if c == "}" else 0) for c in stripped)
        assert depth == 0, f"Unbalanced braces in _atom() result #{i}: {result!r}"


def test_expr_depth_zero_returns_atom() -> None:
    rng = random.Random(0)
    for _ in range(20):
        result = _expr(rng, depth=0)
        assert isinstance(result, str) and len(result) > 0


def test_expr_depth_two_returns_nonempty() -> None:
    rng = random.Random(0)
    for _ in range(50):
        result = _expr(rng, depth=2)
        assert isinstance(result, str) and len(result) > 0


def test_expr_balanced_braces() -> None:
    rng = random.Random(7)
    for i in range(100):
        result = _expr(rng, depth=2)
        stripped = result.replace(r"\{", "").replace(r"\}", "")
        depth = 0
        for ch in stripped:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            assert depth >= 0, f"Unmatched '}}' in _expr result #{i}: {result!r}"
        assert depth == 0, f"Unmatched '{{' in _expr result #{i}: {result!r}"


def test_fn_rich_returns_nonempty() -> None:
    rng = random.Random(0)
    for _ in range(50):
        result = _fn_rich(rng)
        assert isinstance(result, str) and len(result) > 0


def test_fn_rich_nosub_returns_nonempty() -> None:
    rng = random.Random(0)
    for _ in range(50):
        result = _fn_rich_nosub(rng)
        assert isinstance(result, str) and len(result) > 0


def test_fn_rich_nosub_safe_for_superscript_append() -> None:
    """_fn_rich_nosub must not end with _ or ^ so callers can safely append ^{...}."""
    rng = random.Random(0)
    for _ in range(200):
        result = _fn_rich_nosub(rng)
        assert not result.endswith("_"), f"Unsafe trailing '_' in: {result!r}"
        assert not result.endswith("^"), f"Unsafe trailing '^' in: {result!r}"


# ---------------------------------------------------------------------------
# _maybe_idx
# ---------------------------------------------------------------------------


def test_maybe_idx_prob_zero_never_subscripts() -> None:
    rng = random.Random(0)
    for _ in range(100):
        result = _maybe_idx(rng, "x", prob=0.0)
        assert result == "x", f"Expected bare 'x' but got {result!r}"


def test_maybe_idx_prob_one_always_subscripts() -> None:
    rng = random.Random(0)
    for _ in range(100):
        result = _maybe_idx(rng, "x", prob=1.0)
        assert result.startswith("x_{") and result.endswith("}"), f"Expected subscripted form but got {result!r}"


def test_maybe_idx_default_prob_sometimes_subscripts() -> None:
    rng = random.Random(0)
    results = [_maybe_idx(rng, "y") for _ in range(200)]
    bare = sum(1 for r in results if r == "y")
    subscripted = sum(1 for r in results if r != "y")
    assert bare > 0, "Default prob should sometimes leave variable bare"
    assert subscripted > 0, "Default prob should sometimes subscript"


# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------


def test_underbrace_contains_keyword() -> None:
    result = _underbrace("f(x)", r"\text{integrand}")
    assert r"\underbrace" in result
    assert "f(x)" in result


def test_overbrace_contains_keyword() -> None:
    result = _overbrace("a + b", r"\text{sum}")
    assert r"\overbrace" in result
    assert "a + b" in result


def test_bvec_contains_mathbf_or_boldsymbol() -> None:
    rng = random.Random(0)
    for _ in range(30):
        result = _bvec(rng)
        assert r"\mathbf{" in result or r"\boldsymbol{" in result, f"Unexpected _bvec output: {result!r}"


def test_bgreek_contains_boldsymbol() -> None:
    rng = random.Random(0)
    for _ in range(30):
        result = _bgreek(rng)
        assert r"\boldsymbol{" in result, f"Unexpected _bgreek output: {result!r}"
