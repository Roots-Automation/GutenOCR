"""Tests for display-math and inline-math wrapping in corpus generation."""

import random

from formula_combinatorics.corpus import (
    _is_wrapped,
    _wrap_display,
    _wrap_inline,
    generate,
)
from formula_combinatorics.domains import DEFAULT_WEIGHTS, GENERATORS


def test_is_wrapped_detects_begin_env() -> None:
    assert _is_wrapped(r"\begin{align*}x &= y\end{align*}")
    assert _is_wrapped(r"\begin{equation}E=mc^2\end{equation}")
    assert _is_wrapped(r"\begin{equation*}\sqrt{x}\end{equation*}")
    assert not _is_wrapped(r"x^2 + y^2 = z^2")
    assert not _is_wrapped(r"\frac{a}{b}")
    assert not _is_wrapped(r"\sqrt{x^2+1}")


def test_is_wrapped_detects_bracket() -> None:
    assert _is_wrapped(r"\[x^2 + 1\]")
    assert not _is_wrapped(r"x^2 + 1")


def test_is_wrapped_detects_dollar() -> None:
    assert _is_wrapped(r"$E=mc^2$")
    assert not _is_wrapped(r"E=mc^2")


def test_wrap_display_valid_environments() -> None:
    rng = random.Random(42)
    formula = r"\sqrt{x^2+1}"
    for _ in range(1000):
        result = _wrap_display(formula, rng)
        assert (result.startswith(r"\[") and result.endswith(r"\]")) or (
            result.startswith(r"\begin{equation}") and result.endswith(r"\end{equation}")
        ), f"Unexpected wrapper: {result}"


def test_wrap_display_ratio() -> None:
    rng = random.Random(0)
    formula = r"x + y"
    bracket_count = sum(1 for _ in range(10_000) if _wrap_display(formula, rng).startswith(r"\["))
    ratio = bracket_count / 10_000
    assert 0.60 <= ratio <= 0.80, f"\\[ ratio {ratio:.3f} outside [0.60, 0.80]"


def test_wrap_inline_valid() -> None:
    assert _wrap_inline(r"E=mc^2") == r"$E=mc^2$"
    assert _wrap_inline(r"\frac{a}{b}") == r"$\frac{a}{b}$"


def test_generate_display_fraction_zero() -> None:
    domains = list(DEFAULT_WEIGHTS.keys())
    results = generate(
        count=200,
        domains=domains,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=1,
        display_fraction=0.0,
        inline_fraction=0.0,
    )
    for v in results.values():
        # \[ is only ever added by _wrap_display; \begin{equation} can be domain-native.
        assert not v.startswith(r"\["), f"Unexpected \\[ in: {v}"


def test_generate_display_fraction_one() -> None:
    domains = list(DEFAULT_WEIGHTS.keys())
    results = generate(
        count=200,
        domains=domains,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=2,
        display_fraction=1.0,
        inline_fraction=0.0,
    )
    for v in results.values():
        assert v.startswith(r"\begin{") or v.startswith(r"\["), f"Unwrapped formula: {v}"


def test_generate_inline_fraction_one() -> None:
    domains = list(DEFAULT_WEIGHTS.keys())
    results = generate(
        count=200,
        domains=domains,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=3,
        display_fraction=0.0,
        inline_fraction=1.0,
    )
    for v in results.values():
        # Pre-wrapped formulas (e.g. equation* from linear_algebra) pass through unwrapped.
        assert v.startswith("$") or v.startswith(r"\begin{"), f"Formula not wrapped: {v}"


def test_no_double_wrapping() -> None:
    domains = list(DEFAULT_WEIGHTS.keys())
    results = generate(
        count=500,
        domains=domains,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=4,
        display_fraction=1.0,
        inline_fraction=0.0,
    )
    for v in results.values():
        assert r"\[\begin{" not in v, f"Double-wrapped: {v}"
        assert r"\begin{align*}\begin{" not in v, f"Double-wrapped align: {v}"


def test_param_plumbing() -> None:
    domains = list(DEFAULT_WEIGHTS.keys())
    kwargs = dict(
        count=200,
        domains=domains,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=99,
        inline_fraction=0.0,
    )
    r0 = generate(**kwargs, display_fraction=0.0)
    r1 = generate(**kwargs, display_fraction=0.5)
    assert r0 != r1
