"""Tests for newly added amsmath environments in the align domain."""

from __future__ import annotations

import random

from formula_combinatorics.corpus import generate
from formula_combinatorics.domains import DEFAULT_WEIGHTS, GENERATORS

_LABELS = ("eq1", "eq2", "main", "result", "key", "def", "prop")

_align = GENERATORS["align"]


def _sample_align(n: int = 5000, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [_align(rng) for _ in range(n)]


# --- presence tests ---


def test_multline_star_present() -> None:
    samples = _sample_align()
    assert any(r"\begin{multline*}" in s for s in samples)


def test_multline_numbered_present() -> None:
    samples = _sample_align()
    assert any(s.startswith(r"\begin{multline}") and not s.startswith(r"\begin{multline*}") for s in samples)


def test_gather_star_present() -> None:
    samples = _sample_align()
    assert any(r"\begin{gather*}" in s for s in samples)


def test_gather_numbered_present() -> None:
    samples = _sample_align()
    assert any(s.startswith(r"\begin{gather}") and not s.startswith(r"\begin{gather*}") for s in samples)


def test_equation_split_present() -> None:
    samples = _sample_align()
    assert any(r"\begin{equation}\begin{split}" in s for s in samples)


def test_multi_column_align_present() -> None:
    # Multi-column align uses "&&" as the column separator
    samples = _sample_align()
    assert any(r"&&" in s for s in samples)


def test_labeled_equation_present() -> None:
    samples = _sample_align()
    assert any(r"\label{" in s for s in samples)


def test_labeled_align_numbered_present() -> None:
    samples = _sample_align()
    assert any(s.startswith(r"\begin{align}") and r"\label{" in s for s in samples)


# --- structural correctness tests ---


def test_split_nesting() -> None:
    samples = _sample_align(n=8000)
    split_samples = [s for s in samples if r"\begin{split}" in s]
    assert split_samples, "No split samples found"
    for s in split_samples:
        assert s.startswith(r"\begin{equation}"), f"split not inside equation: {s}"
        assert r"\begin{equation}\begin{split}" in s
        assert r"\end{split}\end{equation}" in s


def test_label_values_from_pool() -> None:
    samples = _sample_align(n=8000)
    labeled = [s for s in samples if r"\label{" in s]
    assert labeled, "No labeled samples found"
    for s in labeled:
        # Extract label value from \label{...}
        start = s.index(r"\label{") + len(r"\label{")
        end = s.index("}", start)
        lbl = s[start:end]
        assert lbl in _LABELS, f"Unexpected label value: {lbl!r}"


def test_multline_star_structure() -> None:
    samples = _sample_align(n=8000)
    ml = [s for s in samples if s.startswith(r"\begin{multline*}")]
    assert ml
    for s in ml:
        assert s.endswith(r"\end{multline*}"), f"Bad multline* closing: {s}"
        assert r"\\" in s, f"multline* has no line break: {s}"


def test_gather_structure() -> None:
    samples = _sample_align(n=8000)
    gs = [s for s in samples if r"\begin{gather" in s]
    assert gs
    for s in gs:
        assert r"\\" in s, f"gather has no line break: {s}"


# --- no-double-wrap tests ---


def test_no_double_wrapping_new_envs() -> None:
    domains = list(DEFAULT_WEIGHTS.keys())
    results = generate(
        count=1000,
        domains=domains,
        generators=GENERATORS,
        weights=DEFAULT_WEIGHTS,
        seed=7,
        display_fraction=1.0,
        inline_fraction=0.0,
    )
    for v in results.values():
        assert r"\[\begin{" not in v, f"Double-wrapped: {v}"
        assert r"$\begin{" not in v, f"Dollar-double-wrapped: {v}"
