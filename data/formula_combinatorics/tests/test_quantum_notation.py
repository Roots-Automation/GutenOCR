"""Tests for the quantum_notation domain and related lVert/anticommutator additions."""

from __future__ import annotations

import random

import pytest
from formula_combinatorics._template_dsl import sample
from formula_combinatorics.domains import GENERATORS, TEMPLATES

_GEN_QN = GENERATORS["quantum_notation"]
_GEN_LA = GENERATORS["linear_algebra"]

_TMPLS_QN = TEMPLATES["quantum_notation"]
_TMPLS_PH = TEMPLATES["physics"]
_TMPL_BY_NAME = {t.name: t for t in _TMPLS_QN}
_TMPL_PH_BY_NAME = {t.name: t for t in _TMPLS_PH}


def _draw(gen, n: int, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [gen(rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_smoke_generates_nonempty():
    results = _draw(_GEN_QN, 500)
    assert all(isinstance(s, str) and len(s) > 0 for s in results)


# ---------------------------------------------------------------------------
# Key delimiter presence in quantum_notation outputs
# ---------------------------------------------------------------------------


def test_rangle_appears_frequently():
    results = _draw(_GEN_QN, 2000)
    hits = sum(r"\rangle" in s for s in results)
    assert hits >= 1600, f"\\rangle appeared in only {hits}/2000 samples"


def test_langle_appears_frequently():
    results = _draw(_GEN_QN, 2000)
    hits = sum(r"\langle" in s for s in results)
    assert hits >= 1000, f"\\langle appeared in only {hits}/2000 samples"


# ---------------------------------------------------------------------------
# Per-template structural assertions
# ---------------------------------------------------------------------------


def test_inner_product_bk_distinct_states():
    """Bra and ket state labels in inner_product_bk must differ."""
    tmpl = _TMPL_BY_NAME["inner_product_bk"]
    rng = random.Random(42)
    for _ in range(200):
        rendered = sample(tmpl, rng)
        assert r"\langle" in rendered and r"\rangle" in rendered


def test_outer_product_contains_rangle_langle():
    """outer_product_bk must emit ...⟩⟨... pattern."""
    tmpl = _TMPL_BY_NAME["outer_product_bk"]
    rng = random.Random(7)
    for _ in range(200):
        rendered = sample(tmpl, rng)
        assert r"\rangle\langle" in rendered, f"Missing \\rangle\\langle in: {rendered}"


def test_density_matrix_pure_structure():
    """density_matrix_pure must have the form rho = |...>⟨...|."""
    tmpl = _TMPL_BY_NAME["density_matrix_pure"]
    rng = random.Random(13)
    for _ in range(200):
        rendered = sample(tmpl, rng)
        assert r"\rho" in rendered
        assert r"\rangle\langle" in rendered


def test_completeness_discrete_has_identity():
    """completeness_discrete must contain the identity symbol."""
    tmpl = _TMPL_BY_NAME["completeness_discrete"]
    rng = random.Random(21)
    identities = {r"\hat{1}", r"\mathbf{1}", r"\mathbb{I}", r"\mathbf{I}"}
    for _ in range(200):
        rendered = sample(tmpl, rng)
        assert any(sym in rendered for sym in identities), f"No identity symbol in: {rendered}"


def test_wigner_3j_has_pmatrix():
    """wigner_3j_symbol must produce a pmatrix with two rows."""
    tmpl = _TMPL_BY_NAME["wigner_3j_symbol"]
    rng = random.Random(0)
    for _ in range(50):
        rendered = sample(tmpl, rng)
        assert r"\begin{pmatrix}" in rendered
        assert r"\\" in rendered


def test_wigner_6j_has_Bmatrix():
    """wigner_6j_symbol must produce a Bmatrix."""
    tmpl = _TMPL_BY_NAME["wigner_6j_symbol"]
    rng = random.Random(0)
    for _ in range(50):
        rendered = sample(tmpl, rng)
        assert r"\begin{Bmatrix}" in rendered


def test_ket_tensor_product_has_otimes():
    """ket_tensor_product must contain \\otimes."""
    tmpl = _TMPL_BY_NAME["ket_tensor_product"]
    rng = random.Random(5)
    for _ in range(200):
        rendered = sample(tmpl, rng)
        assert r"\otimes" in rendered


def test_qubit_superposition_has_zero_one_kets():
    """qubit_superposition must contain |0⟩ and |1⟩."""
    tmpl = _TMPL_BY_NAME["qubit_superposition"]
    rng = random.Random(3)
    for _ in range(200):
        rendered = sample(tmpl, rng)
        assert "|0" in rendered and "|1" in rendered


def test_vacuum_expectation_value_has_zero_bra_ket():
    """vacuum_expectation_value must wrap operator between |0⟩ and ⟨0|."""
    tmpl = _TMPL_BY_NAME["vacuum_expectation_value"]
    rng = random.Random(9)
    for _ in range(200):
        rendered = sample(tmpl, rng)
        assert r"\langle 0 |" in rendered and r"| 0 \rangle" in rendered


# ---------------------------------------------------------------------------
# lVert / rVert coverage in linear_algebra
# ---------------------------------------------------------------------------


def test_lVert_rVert_appear_in_linear_algebra():
    results = _draw(_GEN_LA, 2000)
    assert any(r"\lVert" in s for s in results), r"\lVert never appeared in 2000 LA samples"
    assert any(r"\rVert" in s for s in results), r"\rVert never appeared in 2000 LA samples"


# ---------------------------------------------------------------------------
# Anticommutator coverage in physics (direct template sampling)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tmpl_name",
    ["anticommutator_def", "canonical_anticommutation_relation", "anticommutation_annihilators"],
)
def test_anticommutator_template_emits_curly_brace(tmpl_name: str):
    """Each anticommutator template must emit \\{ in every draw."""
    tmpl = _TMPL_PH_BY_NAME[tmpl_name]
    rng = random.Random(42)
    for _ in range(200):
        rendered = sample(tmpl, rng)
        assert r"\{" in rendered, f"Missing \\{{ in {tmpl_name}: {rendered}"


# ---------------------------------------------------------------------------
# No exceptions over many draws
# ---------------------------------------------------------------------------


def test_no_exceptions_over_1000_draws():
    rng = random.Random(99)
    for _ in range(1000):
        _GEN_QN(rng)
