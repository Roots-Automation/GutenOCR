"""Unit tests for formula_combinatorics/_vocab.py."""

from __future__ import annotations

import random
import re

import pytest
from formula_combinatorics.engine._vocab import (
    _BBOLD,
    _BOLD_GREEK,
    _BOLD_VECS,
    _BOUNDS,
    _CALLIGRAPHIC,
    _COEFF_POOL,
    _COMB_K,
    _COMB_N,
    _DECO_CMDS,
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
    _PROPS,
    _RELATIONS,
    _RING_NAMES,
    _RV_BASE,
    _SCALARS,
    _SETS,
    _SIG_STATS,
    _STATS_N,
    _TRIG_NAME_POOLS,
    _VARS,
    _VARS_SCALARS,
    _VEC_POOL,
    _atom,
    _bgreek,
    _bvec,
    _cal,
    _deco,
    _eps_sub,
    _expr,
    _fn_rich,
    _fn_rich_nosub,
    _g,
    _gu,
    _i,
    _idx_atom,
    _lower,
    _maybe_idx,
    _overbrace,
    _overset,
    _prime_deco,
    _s,
    _tol_sub,
    _trig_nm_factory,
    _two,
    _underbrace,
    _upper,
    _v,
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
    ("_PROPS", _PROPS),
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


# ---------------------------------------------------------------------------
# Simple pool samplers (_v, _g, _gu, _s, _i, _cal)
# ---------------------------------------------------------------------------

_SIMPLE_SAMPLERS = [
    ("_v", _v, _VARS),
    ("_g", _g, _GREEK),
    ("_gu", _gu, _GREEK_UPPER),
    ("_s", _s, _SCALARS),
    ("_i", _i, _INDICES),
    ("_cal", _cal, _CALLIGRAPHIC),
]


@pytest.mark.parametrize("name,fn,pool", _SIMPLE_SAMPLERS, ids=[x[0] for x in _SIMPLE_SAMPLERS])
def test_simple_sampler_always_in_pool(name: str, fn, pool: tuple) -> None:
    rng = random.Random(0)
    for _ in range(200):
        result = fn(rng)
        assert result in pool, f"{name}: {result!r} not in declared pool"


@pytest.mark.parametrize("name,fn,pool", _SIMPLE_SAMPLERS, ids=[x[0] for x in _SIMPLE_SAMPLERS])
def test_simple_sampler_coverage_floor(name: str, fn, pool: tuple) -> None:
    rng = random.Random(0)
    seen = {fn(rng) for _ in range(200)}
    coverage = len(seen) / len(pool)
    assert coverage >= 0.5, f"{name}: only {len(seen)}/{len(pool)} pool entries seen in 200 draws"


# ---------------------------------------------------------------------------
# _eps_sub / _tol_sub
# ---------------------------------------------------------------------------


def test_eps_sub_always_in_pool() -> None:
    pool = {r"\epsilon", r"\varepsilon"}
    rng = random.Random(0)
    for _ in range(1000):
        assert _eps_sub(rng) in pool


def test_eps_sub_both_forms_appear() -> None:
    rng = random.Random(0)
    results = {_eps_sub(rng) for _ in range(1000)}
    assert r"\epsilon" in results
    assert r"\varepsilon" in results


def test_tol_sub_always_in_pool() -> None:
    pool = {r"\epsilon", r"\varepsilon", r"\delta"}
    rng = random.Random(0)
    for _ in range(1000):
        assert _tol_sub(rng) in pool


def test_tol_sub_all_three_forms_appear() -> None:
    rng = random.Random(0)
    results = {_tol_sub(rng) for _ in range(1000)}
    assert r"\epsilon" in results
    assert r"\varepsilon" in results
    assert r"\delta" in results


# ---------------------------------------------------------------------------
# _deco
# ---------------------------------------------------------------------------

_DECO_PATTERN = re.compile(r"^\\[a-zA-Z]+\{[a-zA-Z]\}$")
_DECO_TARGET_POOL = set(_VARS) | set("abcfghpqrs")


def test_deco_matches_expected_pattern() -> None:
    rng = random.Random(0)
    for _ in range(100):
        result = _deco(rng)
        assert _DECO_PATTERN.match(result), f"_deco output {result!r} does not match pattern"


def test_deco_hat_most_common() -> None:
    rng = random.Random(0)
    results = [_deco(rng) for _ in range(2000)]
    hat_count = sum(r.startswith(r"\hat{") for r in results)
    for cmd in _DECO_CMDS:
        if cmd != r"\hat":
            other = sum(r.startswith(f"{cmd}{{") for r in results)
            assert hat_count > other, f"\\hat ({hat_count}) not more common than {cmd!r} ({other})"


def test_deco_target_in_expected_pool() -> None:
    rng = random.Random(0)
    for _ in range(200):
        result = _deco(rng)
        inner = result.rsplit("{", 1)[-1].rstrip("}")
        assert inner in _DECO_TARGET_POOL, f"_deco target {inner!r} not in expected pool: {result!r}"


def test_deco_no_double_decoration() -> None:
    rng = random.Random(0)
    for _ in range(200):
        result = _deco(rng)
        assert result.count("{") == 1, f"Unexpected nested structure in _deco: {result!r}"


# ---------------------------------------------------------------------------
# _prime_deco
# ---------------------------------------------------------------------------


def test_prime_deco_base_always_present() -> None:
    rng = random.Random(0)
    for _ in range(100):
        result = _prime_deco(rng, "f")
        assert "f" in result, f"Base 'f' missing from: {result!r}"


def test_prime_deco_forms_reachable() -> None:
    rng = random.Random(0)
    results = {_prime_deco(rng, "f") for _ in range(200)}
    assert "f'" in results, "f' form never appeared"
    assert "f''" in results, "f'' form never appeared"
    assert r"f^{\prime}" not in results, "^{\\prime} form reintroduced (causes double superscript)"


# ---------------------------------------------------------------------------
# _idx_atom
# ---------------------------------------------------------------------------

_DOUBLE_SUB_RE = re.compile(r"_\{[^}]+\}_\{")


def test_idx_atom_no_double_subscript() -> None:
    rng = random.Random(0)
    for _ in range(1000):
        result = _idx_atom(rng)
        assert not _DOUBLE_SUB_RE.search(result), f"Double subscript in _idx_atom: {result!r}"


def test_idx_atom_prob_one_mostly_subscripts() -> None:
    rng = random.Random(0)
    results = [_idx_atom(rng, prob=1.0) for _ in range(200)]
    subscripted = sum("_{" in r for r in results)
    assert subscripted > 100, f"Only {subscripted}/200 subscripted with prob=1.0"


def test_idx_atom_prob_zero_no_added_subscripts() -> None:
    rng = random.Random(0)
    results = [_idx_atom(rng, prob=0.0) for _ in range(200)]
    for r in results:
        assert not _DOUBLE_SUB_RE.search(r), f"Unexpected double-subscript with prob=0.0: {r!r}"


# ---------------------------------------------------------------------------
# _two
# ---------------------------------------------------------------------------


def test_two_returns_two_elements() -> None:
    pool = ("a", "b", "c", "d")
    rng = random.Random(0)
    result = _two(rng, pool)
    assert len(result) == 2


def test_two_both_in_pool() -> None:
    pool = ("a", "b", "c", "d")
    rng = random.Random(0)
    for _ in range(50):
        a, b = _two(rng, pool)
        assert a in pool and b in pool


def test_two_always_distinct() -> None:
    pool = ("a", "b", "c", "d")
    rng = random.Random(0)
    for _ in range(100):
        a, b = _two(rng, pool)
        assert a != b, f"_two returned duplicate: {a!r}"


def test_two_minimal_pool_exhausts_it() -> None:
    pool = ("x", "y")
    rng = random.Random(0)
    a, b = _two(rng, pool)
    assert {a, b} == {"x", "y"}


# ---------------------------------------------------------------------------
# _lower / _upper
# ---------------------------------------------------------------------------

_LOWER_POOL = _BOUNDS + (r"-\infty",)
_UPPER_POOL = _BOUNDS + (r"\infty", r"+\infty")


def test_lower_always_in_pool() -> None:
    rng = random.Random(0)
    for _ in range(200):
        result = _lower(rng)
        assert result in _LOWER_POOL, f"_lower: {result!r} not in declared pool"


def test_upper_always_in_pool() -> None:
    rng = random.Random(0)
    for _ in range(200):
        result = _upper(rng)
        assert result in _UPPER_POOL, f"_upper: {result!r} not in declared pool"


def test_lower_has_neg_infty() -> None:
    rng = random.Random(0)
    results = {_lower(rng) for _ in range(200)}
    assert r"-\infty" in results, "_lower never produced -\\infty"


def test_lower_never_produces_pos_infty() -> None:
    rng = random.Random(0)
    for _ in range(500):
        assert _lower(rng) != r"+\infty", "_lower produced +\\infty (belongs only in _upper)"


def test_upper_has_infty() -> None:
    rng = random.Random(0)
    results = {_upper(rng) for _ in range(200)}
    assert r"\infty" in results or r"+\infty" in results, "_upper never produced \\infty"


def test_upper_never_produces_neg_infty() -> None:
    rng = random.Random(0)
    for _ in range(500):
        assert _upper(rng) != r"-\infty", "_upper produced -\\infty (belongs only in _lower)"


# ---------------------------------------------------------------------------
# _overset
# ---------------------------------------------------------------------------


def test_overset_deterministic_format() -> None:
    assert _overset("=", r"\phi") == r"\overset{=}{\phi}"


def test_overset_balanced_braces() -> None:
    def _bal(s: str) -> bool:
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

    assert _bal(_overset(r"\sim", "A"))
    assert _bal(_overset(r"\to", r"\phi"))
    assert _bal(_overset("f", r"\Longrightarrow"))


# ---------------------------------------------------------------------------
# _trig_nm_factory
# ---------------------------------------------------------------------------


def test_trig_nm_factory_returns_callable() -> None:
    fn = _trig_nm_factory("arcsin")
    assert callable(fn)


def test_trig_nm_factory_output_in_pool() -> None:
    fn = _trig_nm_factory("arcsin")
    pool = _TRIG_NAME_POOLS["arcsin"]
    rng = random.Random(0)
    for _ in range(200):
        result = fn(rng)
        assert result in pool, f"_trig_nm_factory('arcsin'): {result!r} not in pool"


def test_trig_nm_factory_all_entries_reachable() -> None:
    fn = _trig_nm_factory("arcsin")
    pool = set(_TRIG_NAME_POOLS["arcsin"])
    rng = random.Random(0)
    seen = {fn(rng) for _ in range(1000)}
    assert seen == pool, f"Not all arcsin entries reachable: missing {pool - seen}"


def test_trig_nm_factory_sin_distinct_from_arcsin() -> None:
    sin_pool = set(_TRIG_NAME_POOLS["sin"])
    arcsin_pool = set(_TRIG_NAME_POOLS["arcsin"])
    assert sin_pool != arcsin_pool, "sin and arcsin pools are identical — factory is not key-specific"


# ---------------------------------------------------------------------------
# _TRIG_NAME_POOLS integrity
# ---------------------------------------------------------------------------

_TRIG_POOL_ITEMS = list(_TRIG_NAME_POOLS.items())


@pytest.mark.parametrize("key,pool", _TRIG_POOL_ITEMS, ids=[k for k, _ in _TRIG_POOL_ITEMS])
def test_trig_name_pool_nonempty(key: str, pool: tuple) -> None:
    assert len(pool) > 0, f"_TRIG_NAME_POOLS[{key!r}] is empty"


@pytest.mark.parametrize("key,pool", _TRIG_POOL_ITEMS, ids=[k for k, _ in _TRIG_POOL_ITEMS])
def test_trig_name_pool_entries_are_nonempty_strings(key: str, pool: tuple) -> None:
    for entry in pool:
        assert isinstance(entry, str) and len(entry) > 0, (
            f"_TRIG_NAME_POOLS[{key!r}]: empty or non-string entry {entry!r}"
        )
