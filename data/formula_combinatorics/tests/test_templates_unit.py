"""Unit tests for formula_combinatorics/_templates.py."""

from __future__ import annotations

import random

from formula_combinatorics._templates import (
    _def_integral,
    _func_apply,
    _indef_integral,
    _interval,
    _limit,
    _matrix_env,
    _matrix_with_ellipsis,
    _mixed_partial,
    _norm,
    _partial_deriv,
    _poly,
    _poly_mid_factory,
    _smallmatrix_inline,
    _substack_prod,
    _substack_sum,
    _sum_indexed,
)
from formula_combinatorics._vocab import _FUNCS, _SCALARS


def _rng(seed: int = 0) -> random.Random:
    return random.Random(seed)


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


def _run_n(fn, n: int = 50) -> list[str]:
    return [fn(_rng(i)) for i in range(n)]


# ---------------------------------------------------------------------------
# _limit
# ---------------------------------------------------------------------------


def test_limit_contains_lim() -> None:
    for result in _run_n(_limit):
        assert r"\lim" in result, f"Missing \\lim in: {result!r}"


def test_limit_contains_to() -> None:
    for result in _run_n(_limit):
        assert r"\to" in result, f"Missing \\to in: {result!r}"


def test_limit_balanced_braces() -> None:
    for result in _run_n(_limit):
        assert _balanced(result), f"Unbalanced braces in: {result!r}"


# ---------------------------------------------------------------------------
# _def_integral
# ---------------------------------------------------------------------------


def test_def_integral_contains_int() -> None:
    for result in _run_n(_def_integral):
        assert r"\int" in result, f"Missing \\int in: {result!r}"


def test_def_integral_has_bounds() -> None:
    for result in _run_n(_def_integral):
        # Both upper and lower bound markers must be present
        assert "^{" in result and "_{" in result, f"Missing bound markers in: {result!r}"


def test_def_integral_balanced_braces() -> None:
    for result in _run_n(_def_integral):
        assert _balanced(result), f"Unbalanced braces in: {result!r}"


# ---------------------------------------------------------------------------
# _indef_integral
# ---------------------------------------------------------------------------


def test_indef_integral_contains_int() -> None:
    for result in _run_n(_indef_integral):
        assert r"\int" in result, f"Missing \\int in: {result!r}"


def test_indef_integral_starts_with_int() -> None:
    for result in _run_n(_indef_integral):
        assert result.startswith(r"\int "), f"Indefinite integral should start with \\int , got: {result!r}"


def test_indef_integral_balanced_braces() -> None:
    for result in _run_n(_indef_integral):
        assert _balanced(result), f"Unbalanced braces in: {result!r}"


# ---------------------------------------------------------------------------
# _partial_deriv
# ---------------------------------------------------------------------------


def test_partial_deriv_order1_contains_partial() -> None:
    for result in _run_n(lambda rng: _partial_deriv(rng, order=1)):
        assert r"\partial" in result, f"Missing \\partial in: {result!r}"


def test_partial_deriv_order2_contains_partial_squared() -> None:
    for result in _run_n(lambda rng: _partial_deriv(rng, order=2)):
        assert r"\partial^{2}" in result, f"Missing \\partial^{{2}} in: {result!r}"


def test_partial_deriv_balanced_braces() -> None:
    for result in _run_n(lambda rng: _partial_deriv(rng, order=1)):
        assert _balanced(result), f"Unbalanced braces in: {result!r}"


# ---------------------------------------------------------------------------
# _mixed_partial
# ---------------------------------------------------------------------------


def test_mixed_partial_contains_partial_twice() -> None:
    for result in _run_n(_mixed_partial):
        assert result.count(r"\partial") >= 2, f"Expected ≥2 \\partial in: {result!r}"


def test_mixed_partial_is_second_order() -> None:
    for result in _run_n(_mixed_partial):
        assert r"\partial^2" in result, f"Missing \\partial^2 in: {result!r}"


def test_mixed_partial_balanced_braces() -> None:
    for result in _run_n(_mixed_partial):
        assert _balanced(result), f"Unbalanced braces in: {result!r}"


# ---------------------------------------------------------------------------
# _matrix_env
# ---------------------------------------------------------------------------


def test_matrix_env_correct_environment_name() -> None:
    for env in ("pmatrix", "bmatrix", "Bmatrix", "vmatrix"):
        result = _matrix_env(_rng(), 2, 2, env)
        assert rf"\begin{{{env}}}" in result
        assert rf"\end{{{env}}}" in result


def test_matrix_env_2x3_has_one_row_separator() -> None:
    for seed in range(20):
        result = _matrix_env(_rng(seed), 2, 3, "pmatrix")
        assert result.count(r"\\") == 1, f"2-row matrix should have exactly 1 '\\\\', got: {result!r}"


def test_matrix_env_3x3_has_two_row_separators() -> None:
    for seed in range(20):
        result = _matrix_env(_rng(seed), 3, 3, "pmatrix")
        assert result.count(r"\\") == 2, f"3-row matrix should have exactly 2 '\\\\', got: {result!r}"


def test_matrix_env_balanced_braces() -> None:
    for seed in range(30):
        result = _matrix_env(_rng(seed), 2, 2)
        assert _balanced(result), f"Unbalanced braces in: {result!r}"


# ---------------------------------------------------------------------------
# _matrix_with_ellipsis
# ---------------------------------------------------------------------------


def test_matrix_with_ellipsis_contains_ellipsis() -> None:
    ellipsis_cmds = (r"\vdots", r"\ddots", r"\cdots")
    for seed in range(50):
        result = _matrix_with_ellipsis(_rng(seed))
        assert any(cmd in result for cmd in ellipsis_cmds), f"No ellipsis command in: {result!r}"


def test_matrix_with_ellipsis_balanced_braces() -> None:
    for seed in range(30):
        result = _matrix_with_ellipsis(_rng(seed))
        assert _balanced(result), f"Unbalanced braces in: {result!r}"


# ---------------------------------------------------------------------------
# _substack_sum
# ---------------------------------------------------------------------------


def test_substack_sum_contains_substack() -> None:
    for result in _run_n(_substack_sum):
        assert r"\substack" in result, f"Missing \\substack in: {result!r}"


def test_substack_sum_contains_sum() -> None:
    for result in _run_n(_substack_sum):
        assert r"\sum" in result, f"Missing \\sum in: {result!r}"


def test_substack_sum_balanced_braces() -> None:
    for result in _run_n(_substack_sum):
        assert _balanced(result), f"Unbalanced braces in: {result!r}"


# ---------------------------------------------------------------------------
# _substack_prod
# ---------------------------------------------------------------------------


def test_substack_prod_contains_substack() -> None:
    for result in _run_n(_substack_prod):
        assert r"\substack" in result, f"Missing \\substack in: {result!r}"


def test_substack_prod_contains_prod() -> None:
    for result in _run_n(_substack_prod):
        assert r"\prod" in result, f"Missing \\prod in: {result!r}"


def test_substack_prod_balanced_braces() -> None:
    for result in _run_n(_substack_prod):
        assert _balanced(result), f"Unbalanced braces in: {result!r}"


# ---------------------------------------------------------------------------
# _poly
# ---------------------------------------------------------------------------


def test_poly_contains_variable() -> None:
    for seed in range(50):
        result = _poly(_rng(seed), "x")
        assert "x" in result, f"Missing variable 'x' in polynomial: {result!r}"


def test_poly_greek_variable_present() -> None:
    for seed in range(20):
        result = _poly(_rng(seed), r"\alpha")
        assert r"\alpha" in result, f"Missing \\alpha in polynomial: {result!r}"


def test_poly_has_at_least_two_terms() -> None:
    for seed in range(50):
        result = _poly(_rng(seed), "t", max_degree=3)
        # At least one + or - in the result (connecting terms)
        assert "+" in result or "-" in result, f"Polynomial has fewer than 2 terms: {result!r}"


def test_poly_degree_does_not_exceed_max() -> None:
    for seed in range(50):
        result = _poly(_rng(seed), "x", max_degree=3)
        for power in ("4", "5", "6", "7", "8", "9"):
            assert f"x^{{{power}}}" not in result, f"Degree {power} exceeds max_degree=3 in: {result!r}"


# ---------------------------------------------------------------------------
# _smallmatrix_inline
# ---------------------------------------------------------------------------

_SMALLMATRIX_LEFT_DELIMS = (r"\bigl(", r"\bigl[", r"\bigl\{", r"\bigl\langle")


def test_smallmatrix_inline_contains_environment() -> None:
    for seed in range(50):
        result = _smallmatrix_inline(_rng(seed), 2, 2)
        assert r"\begin{smallmatrix}" in result
        assert r"\end{smallmatrix}" in result


def test_smallmatrix_inline_2x2_one_row_separator() -> None:
    for seed in range(20):
        result = _smallmatrix_inline(_rng(seed), 2, 2)
        assert result.count(r"\\") == 1, f"2-row matrix should have 1 '\\\\', got: {result!r}"


def test_smallmatrix_inline_3x3_two_row_separators() -> None:
    for seed in range(20):
        result = _smallmatrix_inline(_rng(seed), 3, 3)
        assert result.count(r"\\") == 2, f"3-row matrix should have 2 '\\\\', got: {result!r}"


def test_smallmatrix_inline_column_separators() -> None:
    for seed in range(20):
        result = _smallmatrix_inline(_rng(seed), 2, 3)
        assert result.count("&") == 4, f"2×3 matrix should have 4 '&', got: {result!r}"


def test_smallmatrix_inline_uses_known_delimiter() -> None:
    for seed in range(50):
        result = _smallmatrix_inline(_rng(seed), 2, 2)
        assert any(d in result for d in _SMALLMATRIX_LEFT_DELIMS), f"Unknown delimiter in: {result[:40]!r}"


def test_smallmatrix_inline_balanced_braces() -> None:
    for seed in range(30):
        result = _smallmatrix_inline(_rng(seed), 3, 3)
        assert _balanced(result), f"Unbalanced braces in: {result!r}"


# ---------------------------------------------------------------------------
# _sum_indexed
# ---------------------------------------------------------------------------


def test_sum_indexed_contains_sum() -> None:
    for seed in range(50):
        result = _sum_indexed(_rng(seed), "0", "n")
        assert r"\sum" in result, f"Missing \\sum in: {result!r}"


def test_sum_indexed_contains_lo_and_hi() -> None:
    for seed in range(50):
        result = _sum_indexed(_rng(seed), "0", "N")
        assert "0" in result and "N" in result, f"Missing lo/hi in: {result!r}"


def test_sum_indexed_has_sub_and_sup() -> None:
    for seed in range(50):
        result = _sum_indexed(_rng(seed), "1", "N")
        assert "_{" in result and "^{" in result, f"Missing sub/sup in: {result!r}"


def test_sum_indexed_limits_approx_50_pct() -> None:
    results = [_sum_indexed(_rng(i), "0", "n") for i in range(1000)]
    hits = sum(r"\limits" in r for r in results)
    assert 350 <= hits <= 650, f"\\limits appeared in {hits}/1000 (expected ~500)"


def test_sum_indexed_balanced_braces() -> None:
    for seed in range(30):
        result = _sum_indexed(_rng(seed), "a", r"\infty")
        assert _balanced(result), f"Unbalanced braces in: {result!r}"


# ---------------------------------------------------------------------------
# _norm
# ---------------------------------------------------------------------------


def test_norm_contains_two_norm_delimiters() -> None:
    for seed in range(50):
        result = _norm(_rng(seed))
        assert result.count(r"\|") == 2, f"Expected 2 \\| in: {result!r}"


def test_norm_subscript_from_pool() -> None:
    _NORM_SUB_POOL = {"1", "2", r"\infty", "p", "F"}
    for seed in range(50):
        result = _norm(_rng(seed))
        assert any(f"_{{{s}}}" in result for s in _NORM_SUB_POOL), f"Unknown p-subscript in: {result!r}"


def test_norm_explicit_p_appears() -> None:
    result = _norm(_rng(), p="q")
    assert "_{q}" in result, f"Explicit p='q' not found in subscript of: {result!r}"


def test_norm_balanced_braces() -> None:
    for seed in range(30):
        result = _norm(_rng(seed))
        assert _balanced(result), f"Unbalanced braces in: {result!r}"


# ---------------------------------------------------------------------------
# _func_apply
# ---------------------------------------------------------------------------


def test_func_apply_contains_left_paren() -> None:
    for seed in range(50):
        result = _func_apply(_rng(seed))
        assert r"\!\left(" in result, f"Missing \\!\\left( in: {result!r}"
        assert r"\right)" in result, f"Missing \\right) in: {result!r}"


def test_func_apply_starts_with_known_func() -> None:
    for seed in range(50):
        result = _func_apply(_rng(seed))
        assert any(result.startswith(fn) for fn in _FUNCS), f"Unknown function in: {result!r}"


def test_func_apply_all_funcs_reachable() -> None:
    results = [_func_apply(_rng(i)) for i in range(500)]
    for fn in _FUNCS:
        assert any(r.startswith(fn) for r in results), f"Function {fn!r} never appeared in 500 draws"


def test_func_apply_balanced_braces() -> None:
    for seed in range(30):
        result = _func_apply(_rng(seed))
        assert _balanced(result), f"Unbalanced braces in: {result!r}"


# ---------------------------------------------------------------------------
# _interval
# ---------------------------------------------------------------------------


def test_interval_contains_comma() -> None:
    for seed in range(50):
        result = _interval(_rng(seed))
        assert "," in result, f"Missing comma in: {result!r}"


def test_interval_starts_with_known_left_delimiter() -> None:
    known_left = (r"\lbrack", "(")
    for seed in range(50):
        result = _interval(_rng(seed))
        assert any(result.startswith(d) for d in known_left), f"Unknown left delimiter in: {result!r}"


def test_interval_ends_with_known_right_delimiter() -> None:
    known_right = (r"\rbrack", ")")
    for seed in range(50):
        result = _interval(_rng(seed))
        assert any(result.endswith(d) for d in known_right), f"Unknown right delimiter in: {result!r}"


def test_interval_balanced_braces() -> None:
    for seed in range(30):
        result = _interval(_rng(seed))
        assert _balanced(result), f"Unbalanced braces in: {result!r}"


# ---------------------------------------------------------------------------
# _poly_mid_factory
# ---------------------------------------------------------------------------


def test_poly_mid_factory_returns_callable() -> None:
    fn = _poly_mid_factory(3)
    assert callable(fn)


def test_poly_mid_factory_closure_returns_string() -> None:
    fn = _poly_mid_factory(3)
    result = fn(_rng(), "x")
    assert isinstance(result, str) and len(result) > 0


def test_poly_mid_factory_contains_max_exp_power() -> None:
    fn = _poly_mid_factory(3)
    for seed in range(20):
        result = fn(_rng(seed), "x")
        assert "^{3}" in result, f"Missing ^{{3}} in max_exp=3 closure: {result!r}"


def test_poly_mid_factory_max2_contains_only_degree2() -> None:
    fn = _poly_mid_factory(2)
    for seed in range(20):
        result = fn(_rng(seed), "x")
        assert "^{2}" in result, f"Missing ^{{2}} in max_exp=2 closure: {result!r}"
        assert "^{3}" not in result, f"Unexpected ^{{3}} in max_exp=2 closure: {result!r}"


def test_poly_mid_factory_uses_scalar_coefficients() -> None:
    fn = _poly_mid_factory(3)
    for seed in range(50):
        result = fn(_rng(seed), "x")
        assert any(f"{s} x" in result for s in _SCALARS), f"No scalar coefficient found in: {result!r}"


def test_poly_mid_factory_balanced_braces() -> None:
    fn = _poly_mid_factory(4)
    for seed in range(30):
        result = fn(_rng(seed), "t")
        assert _balanced(result), f"Unbalanced braces in: {result!r}"
