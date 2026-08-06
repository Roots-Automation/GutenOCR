"""Structural tests for the 30 domains lacking per-template assertions.

Modelled on test_math_fonts.py and test_quantum_notation.py.  Each domain
gets:
  1. Frequency threshold — characteristic token(s) appear in ≥N of 2000 draws
  2. Per-template structural assertions — for key templates, structural
     invariants are checked across 50–200 draws
  3. Parametrized no-double-subscript regression — catches DSL bugs specific
     to each domain

math_fonts and quantum_notation are excluded; they have dedicated test files.
"""

from __future__ import annotations

import random
import re

import pytest
from formula_combinatorics.domains import GENERATORS, TEMPLATES
from formula_combinatorics.engine._template_dsl import ExcludeSlot, Slot, sample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DOUBLE_SUB_RE = re.compile(r"[A-Za-z0-9]_[A-Za-z0-9]_")


def _draw(gen, n: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    return [gen(rng) for _ in range(n)]


def _sample_template(domain: str, name: str, n: int = 100, seed: int = 0) -> list[str]:
    # Search top-level templates first, then variants one level deep.
    t_map = {t.name: t for t in TEMPLATES[domain]}
    if name in t_map:
        t = t_map[name]
    else:
        variant_map = {v.name: v for t in TEMPLATES[domain] for v in getattr(t, "variants", [])}
        t = variant_map[name]
    rng = random.Random(seed)
    return [sample(t, rng) for _ in range(n)]


def _freq(samples: list[str], token: str) -> int:
    return sum(token in s for s in samples)


# ---------------------------------------------------------------------------
# Parametrized: no double-subscript across all 30 structural domains
# ---------------------------------------------------------------------------

_STRUCTURAL_DOMAINS = [d for d in GENERATORS if d not in ("math_fonts", "quantum_notation")]


@pytest.mark.parametrize("domain", _STRUCTURAL_DOMAINS)
def test_no_double_subscript_500_draws(domain: str) -> None:
    gen = GENERATORS[domain]
    rng = random.Random(42)
    for _ in range(500):
        result = gen(rng)
        assert not _DOUBLE_SUB_RE.search(result), f"{domain}: double-subscript in output: {result!r}"


# ---------------------------------------------------------------------------
# Static: pool-value superscript/subscript clash detection
# ---------------------------------------------------------------------------
# A slot immediately followed by ^ in the template latex combined with a pool
# value that already ends with ^{...} produces a double superscript (e.g.
# (k_B T)^{-1}^2).  The mirror holds for subscripts.  These are detected
# statically — no sampling needed.

# {slot}^ and {slot}_ in raw format strings; negative lookahead/behind
# excludes escaped {{ }} sequences.
_SLOT_THEN_SUPER_RE = re.compile(r"(?<!\{)\{(\w+)\}(?!\})\^")
_SLOT_THEN_SUB_RE = re.compile(r"(?<!\{)\{(\w+)\}(?!\})_")
# Pool values that already carry a trailing superscript or subscript.
_TRAILING_SUPER_RE = re.compile(r"\^(\{[^{}]*\}|[A-Za-z0-9])$")
_TRAILING_SUB_RE = re.compile(r"_(\{[^{}]*\}|\\[A-Za-z]+|[A-Za-z0-9])$")


def _pool_vals(slot: object) -> tuple[str, ...]:
    return slot.pool if isinstance(slot, (Slot, ExcludeSlot)) else ()


def _script_clash_errors(t: object) -> list[str]:
    errors: list[str] = []
    for v in getattr(t, "variants", []):
        errors.extend(_script_clash_errors(v))
    latex = getattr(t, "latex", "")
    slots = getattr(t, "slots", {})
    if not latex or not slots:
        return errors
    for m in _SLOT_THEN_SUPER_RE.finditer(latex):
        slot = slots.get(m.group(1))
        if slot is None:
            continue
        for val in _pool_vals(slot):
            if _TRAILING_SUPER_RE.search(val):
                errors.append(
                    f"template '{t.name}': slot '{m.group(1)}' is followed by '^' "  # type: ignore[attr-defined]
                    f"but pool value {val!r} already ends with a superscript"
                )
    for m in _SLOT_THEN_SUB_RE.finditer(latex):
        slot = slots.get(m.group(1))
        if slot is None:
            continue
        for val in _pool_vals(slot):
            if _TRAILING_SUB_RE.search(val):
                errors.append(
                    f"template '{t.name}': slot '{m.group(1)}' is followed by '_' "  # type: ignore[attr-defined]
                    f"but pool value {val!r} already ends with a subscript"
                )
    return errors


@pytest.mark.parametrize("domain", sorted(TEMPLATES))
def test_no_script_clash_in_pools(domain: str) -> None:
    """Pool value with trailing ^/_ cannot be used where the template appends ^/_."""
    errors: list[str] = []
    for t in TEMPLATES[domain]:
        errors.extend(_script_clash_errors(t))
    assert not errors, "\n".join(errors)


# ---------------------------------------------------------------------------
# calculus
# ---------------------------------------------------------------------------

_CALCULUS = _draw(GENERATORS["calculus"], 2000)


def test_calculus_frac_frequency() -> None:
    n = _freq(_CALCULUS, r"\frac")
    assert n >= 700, f"\\frac appeared only {n}/2000 times in calculus"


def test_calculus_int_frequency() -> None:
    n = _freq(_CALCULUS, r"\int")
    assert n >= 300, f"\\int appeared only {n}/2000 times in calculus"


def test_calculus_lim_frequency() -> None:
    n = _freq(_CALCULUS, r"\lim")
    assert n >= 400, f"\\lim appeared only {n}/2000 times in calculus"


def test_calculus_first_derivative_has_frac_d() -> None:
    results = _sample_template("calculus", "first_derivative", 100)
    for r in results:
        assert r"\frac{d}" in r, f"first_derivative missing \\frac{{d}}: {r!r}"


def test_calculus_limit_simple_has_lim_and_to() -> None:
    results = _sample_template("calculus", "limit_simple", 100)
    for r in results:
        assert r"\lim" in r and r"\to" in r, f"limit_simple missing \\lim or \\to: {r!r}"


# ---------------------------------------------------------------------------
# algebra
# ---------------------------------------------------------------------------

_ALGEBRA = _draw(GENERATORS["algebra"], 2000)


def test_algebra_power_frequency() -> None:
    n = _freq(_ALGEBRA, "^{")
    assert n >= 560, f"'^{{' appeared only {n}/2000 times in algebra"


def test_algebra_equals_frequency() -> None:
    n = _freq(_ALGEBRA, "=")
    assert n >= 1000, f"'=' appeared only {n}/2000 times in algebra"


def test_algebra_quadratic_pm_has_pm_and_sqrt() -> None:
    results = _sample_template("algebra", "quadratic_formula_pm", 100)
    for r in results:
        assert r"\pm" in r, f"quadratic_formula_pm missing \\pm: {r!r}"
        assert r"\sqrt" in r, f"quadratic_formula_pm missing \\sqrt: {r!r}"
        assert r"\frac" in r, f"quadratic_formula_pm missing \\frac: {r!r}"


def test_algebra_discriminant_def_has_minus() -> None:
    results = _sample_template("algebra", "discriminant_def", 50)
    for r in results:
        assert "-" in r, f"discriminant_def should contain '-': {r!r}"


# ---------------------------------------------------------------------------
# trigonometry
# ---------------------------------------------------------------------------

_TRIG = _draw(GENERATORS["trigonometry"], 2000)


def test_trigonometry_sin_frequency() -> None:
    n = _freq(_TRIG, r"\sin")
    assert n >= 550, f"\\sin appeared only {n}/2000 times in trigonometry"


def test_trigonometry_cos_frequency() -> None:
    n = _freq(_TRIG, r"\cos")
    assert n >= 110, f"\\cos appeared only {n}/2000 times in trigonometry"


# ---------------------------------------------------------------------------
# linear_algebra
# ---------------------------------------------------------------------------

_LA = _draw(GENERATORS["linear_algebra"], 2000)


def test_linear_algebra_matrix_env_frequency() -> None:
    n = _freq(_LA, "begin{")
    assert n >= 620, f"'begin{{' appeared only {n}/2000 times in linear_algebra"


def test_linear_algebra_ampersand_frequency() -> None:
    n = _freq(_LA, "&")
    assert n >= 570, f"'&' appeared only {n}/2000 times in linear_algebra"


def test_linear_algebra_eigenvalue_has_lambda() -> None:
    results = _sample_template("linear_algebra", "eigenvalue_equation", 100)
    for r in results:
        assert r"\lambda" in r, f"eigenvalue_equation missing \\lambda: {r!r}"


def test_linear_algebra_pmatrix_2x2_has_structure() -> None:
    results = _sample_template("linear_algebra", "pmatrix_2x2", 50)
    for r in results:
        assert r"\begin{pmatrix}" in r, f"pmatrix_2x2 missing \\begin{{pmatrix}}: {r!r}"
        assert r"\\" in r, f"pmatrix_2x2 missing row separator: {r!r}"
        assert "&" in r, f"pmatrix_2x2 missing column separator: {r!r}"


# ---------------------------------------------------------------------------
# probability
# ---------------------------------------------------------------------------

_PROB = _draw(GENERATORS["probability"], 2000)


def test_probability_expectation_frequency() -> None:
    n = _freq(_PROB, r"\mathbb{E}")
    assert n >= 300, f"\\mathbb{{E}} appeared only {n}/2000 in probability"


def test_probability_bayes_has_mid_and_frac() -> None:
    results = _sample_template("probability", "bayes_theorem", 100)
    for r in results:
        assert r"\mid" in r, f"bayes_theorem missing \\mid: {r!r}"
        assert r"\frac" in r, f"bayes_theorem missing \\frac: {r!r}"


def test_probability_normal_pdf_has_exp_and_pi() -> None:
    results = _sample_template("probability", "normal_pdf", 100)
    for r in results:
        assert r"\exp" in r, f"normal_pdf missing \\exp: {r!r}"
        assert r"\pi" in r, f"normal_pdf missing \\pi: {r!r}"


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

_STATS = _draw(GENERATORS["statistics"], 2000)


def test_statistics_hat_frequency() -> None:
    n = _freq(_STATS, r"\hat{")
    assert n >= 650, f"\\hat{{ appeared only {n}/2000 in statistics"


def test_statistics_bar_frequency() -> None:
    n = _freq(_STATS, r"\bar{")
    assert n >= 600, f"\\bar{{ appeared only {n}/2000 in statistics"


# ---------------------------------------------------------------------------
# complex_analysis
# ---------------------------------------------------------------------------

_CX = _draw(GENERATORS["complex_analysis"], 2000)


def test_complex_analysis_oint_frequency() -> None:
    n = _freq(_CX, r"\oint")
    assert n >= 400, f"\\oint appeared only {n}/2000 in complex_analysis"


def test_complex_analysis_cauchy_has_oint_and_frac() -> None:
    results = _sample_template("complex_analysis", "cauchy_integral_formula", 100)
    for r in results:
        assert r"\oint" in r, f"cauchy_integral_formula missing \\oint: {r!r}"
        assert r"\frac" in r, f"cauchy_integral_formula missing \\frac: {r!r}"


def test_complex_analysis_residue_theorem_has_res() -> None:
    results = _sample_template("complex_analysis", "residue_theorem", 100)
    for r in results:
        assert r"\oint" in r, f"residue_theorem missing \\oint: {r!r}"
        assert r"\operatorname{Res}" in r, f"residue_theorem missing \\operatorname{{Res}}: {r!r}"


# ---------------------------------------------------------------------------
# number_theory
# ---------------------------------------------------------------------------

_NT = _draw(GENERATORS["number_theory"], 2000)


def test_number_theory_equiv_frequency() -> None:
    n = _freq(_NT, r"\equiv")
    assert n >= 100, f"\\equiv appeared only {n}/2000 in number_theory"


def test_number_theory_gcd_frequency() -> None:
    n = _freq(_NT, r"\gcd")
    assert n >= 150, f"\\gcd appeared only {n}/2000 in number_theory"


# ---------------------------------------------------------------------------
# set_theory
# ---------------------------------------------------------------------------

_SET = _draw(GENERATORS["set_theory"], 2000)


def test_set_theory_in_frequency() -> None:
    n = _freq(_SET, r"\in")
    assert n >= 300, f"\\in appeared only {n}/2000 in set_theory"


def test_set_theory_subset_frequency() -> None:
    n = _freq(_SET, r"\subset")
    assert n >= 200, f"\\subset appeared only {n}/2000 in set_theory"


# ---------------------------------------------------------------------------
# logic
# ---------------------------------------------------------------------------

_LOGIC = _draw(GENERATORS["logic"], 2000)


def test_logic_forall_frequency() -> None:
    n = _freq(_LOGIC, r"\forall")
    assert n >= 200, f"\\forall appeared only {n}/2000 in logic"


def test_logic_exists_frequency() -> None:
    n = _freq(_LOGIC, r"\exists")
    assert n >= 250, f"\\exists appeared only {n}/2000 in logic"


def test_logic_rightarrow_frequency() -> None:
    n = _freq(_LOGIC, r"\Rightarrow")
    assert n >= 200, f"\\Rightarrow appeared only {n}/2000 in logic"


# ---------------------------------------------------------------------------
# proof_theory
# ---------------------------------------------------------------------------

_PT = _draw(GENERATORS["proof_theory"], 2000)


def test_proof_theory_dfrac_frequency() -> None:
    n = _freq(_PT, r"\dfrac")
    assert n >= 670, f"\\dfrac appeared only {n}/2000 in proof_theory"


def test_proof_theory_vdash_frequency() -> None:
    n = _freq(_PT, r"\vdash")
    assert n >= 680, f"\\vdash appeared only {n}/2000 in proof_theory"


def test_proof_theory_inference_rule_layout() -> None:
    r"""Inference-rule layout: \dfrac used as premises-over-conclusion divider."""
    n = sum(1 for f in _PT if r"\dfrac" in f and r"\vdash" in f)
    assert n >= 200, f"\\dfrac+\\vdash co-occurrence (inference-rule layout) only {n}/2000 in proof_theory"


# ---------------------------------------------------------------------------
# align — every sample must start with \begin{...}
# ---------------------------------------------------------------------------

_ALIGN = _draw(GENERATORS["align"], 2000)


def test_align_begin_env_frequency() -> None:
    n = _freq(_ALIGN, r"\begin{align")
    assert n >= 1100, f"\\begin{{align appeared only {n}/2000 in align"


def test_align_every_sample_starts_with_begin() -> None:
    for i, s in enumerate(_ALIGN):
        assert s.startswith(r"\begin{"), f"align sample #{i} does not start with \\begin{{: {s[:50]!r}"


def test_align_ampersand_equals_frequency() -> None:
    n = _freq(_ALIGN, "&=")
    assert n >= 1100, f"'&=' appeared only {n}/2000 in align"


# ---------------------------------------------------------------------------
# group_theory
# ---------------------------------------------------------------------------

_GT = _draw(GENERATORS["group_theory"], 2000)


def test_group_theory_G_frequency() -> None:
    n = _freq(_GT, "G")
    assert n >= 500, f"'G' appeared only {n}/2000 in group_theory"


def test_group_theory_cong_frequency() -> None:
    n = _freq(_GT, r"\cong")
    assert n >= 100, f"\\cong appeared only {n}/2000 in group_theory"


# ---------------------------------------------------------------------------
# category_theory
# ---------------------------------------------------------------------------

_CAT = _draw(GENERATORS["category_theory"], 2000)


def test_category_theory_circ_frequency() -> None:
    n = _freq(_CAT, r"\circ")
    assert n >= 700, f"\\circ appeared only {n}/2000 in category_theory"


def test_category_theory_to_frequency() -> None:
    n = _freq(_CAT, r"\to")
    assert n >= 600, f"\\to appeared only {n}/2000 in category_theory"


# ---------------------------------------------------------------------------
# combinatorics
# ---------------------------------------------------------------------------

_COMB = _draw(GENERATORS["combinatorics"], 2000)


def test_combinatorics_binom_frequency() -> None:
    n = _freq(_COMB, r"\binom")
    assert n >= 700, f"\\binom appeared only {n}/2000 in combinatorics"


def test_combinatorics_sum_frequency() -> None:
    n = _freq(_COMB, r"\sum")
    assert n >= 800, f"\\sum appeared only {n}/2000 in combinatorics"


# ---------------------------------------------------------------------------
# classical_mechanics
# ---------------------------------------------------------------------------

_CM = _draw(GENERATORS["classical_mechanics"], 2000)


def test_classical_mechanics_frac_frequency() -> None:
    n = _freq(_CM, r"\frac")
    assert n >= 800, f"\\frac appeared only {n}/2000 in classical_mechanics"


def test_classical_mechanics_dot_frequency() -> None:
    n = _freq(_CM, r"\dot")
    assert n >= 400, f"\\dot appeared only {n}/2000 in classical_mechanics"


# ---------------------------------------------------------------------------
# electromagnetism
# ---------------------------------------------------------------------------

_EM = _draw(GENERATORS["electromagnetism"], 2000)


def test_electromagnetism_nabla_frequency() -> None:
    n = _freq(_EM, r"\nabla")
    assert n >= 550, f"\\nabla appeared only {n}/2000 in electromagnetism"


def test_electromagnetism_Efield_frequency() -> None:
    n = _freq(_EM, r"\mathbf{E}")
    assert n >= 150, f"\\mathbf{{E}} appeared only {n}/2000 in electromagnetism"


# ---------------------------------------------------------------------------
# statistical_mechanics
# ---------------------------------------------------------------------------

_SM = _draw(GENERATORS["statistical_mechanics"], 2000)


def test_statistical_mechanics_exp_frequency() -> None:
    n = _freq(_SM, r"e^{")
    assert n >= 850, f"e^{{ appeared only {n}/2000 in statistical_mechanics"


def test_statistical_mechanics_beta_frequency() -> None:
    n = _freq(_SM, r"\beta")
    assert n >= 600, f"\\beta appeared only {n}/2000 in statistical_mechanics"


# ---------------------------------------------------------------------------
# quantum_mechanics
# ---------------------------------------------------------------------------

_QM = _draw(GENERATORS["quantum_mechanics"], 2000)


def test_quantum_mechanics_hat_frequency() -> None:
    n = _freq(_QM, r"\hat{")
    assert n >= 500, f"\\hat{{ appeared only {n}/2000 in quantum_mechanics"


def test_quantum_mechanics_hbar_frequency() -> None:
    n = _freq(_QM, r"\hbar")
    assert n >= 250, f"\\hbar appeared only {n}/2000 in quantum_mechanics"


# ---------------------------------------------------------------------------
# field_theory
# ---------------------------------------------------------------------------

_FT = _draw(GENERATORS["field_theory"], 2000)


def test_field_theory_partial_frequency() -> None:
    n = _freq(_FT, r"\partial")
    assert n >= 400, f"\\partial appeared only {n}/2000 in field_theory"


def test_field_theory_varepsilon_frequency() -> None:
    n = _freq(_FT, r"\varepsilon")
    assert n >= 350, f"\\varepsilon appeared only {n}/2000 in field_theory"


# ---------------------------------------------------------------------------
# differential_geometry
# ---------------------------------------------------------------------------

_DG = _draw(GENERATORS["differential_geometry"], 2000)


def test_differential_geometry_nabla_frequency() -> None:
    n = _freq(_DG, r"\nabla")
    assert n >= 300, f"\\nabla appeared only {n}/2000 in differential_geometry"


def test_differential_geometry_wedge_frequency() -> None:
    n = _freq(_DG, r"\wedge")
    assert n >= 100, f"\\wedge appeared only {n}/2000 in differential_geometry"


# ---------------------------------------------------------------------------
# information_theory
# ---------------------------------------------------------------------------

_INFO = _draw(GENERATORS["information_theory"], 2000)


def test_information_theory_log_frequency() -> None:
    n = _freq(_INFO, r"\log")
    assert n >= 250, f"\\log appeared only {n}/2000 in information_theory"


def test_information_theory_H_frequency() -> None:
    n = _freq(_INFO, "H(")
    assert n >= 500, f"'H(' appeared only {n}/2000 in information_theory"


# ---------------------------------------------------------------------------
# graph_theory
# ---------------------------------------------------------------------------

_GRAPH = _draw(GENERATORS["graph_theory"], 2000)


def test_graph_theory_edge_membership_frequency() -> None:
    n = _freq(_GRAPH, r"\in E(")
    assert n >= 550, f"'\\in E(' appeared only {n}/2000 in graph_theory"


def test_graph_theory_G_frequency() -> None:
    n = _freq(_GRAPH, "G")
    assert n >= 600, f"'G' appeared only {n}/2000 in graph_theory"


# ---------------------------------------------------------------------------
# topology
# ---------------------------------------------------------------------------

_TOPO = _draw(GENERATORS["topology"], 2000)


def test_topology_mathcal_frequency() -> None:
    n = _freq(_TOPO, r"\mathcal")
    assert n >= 900, f"\\mathcal appeared only {n}/2000 in topology"


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

_ANAL = _draw(GENERATORS["analysis"], 2000)


def test_analysis_forall_frequency() -> None:
    n = _freq(_ANAL, r"\forall")
    assert n >= 150, f"\\forall appeared only {n}/2000 in analysis"


def test_analysis_epsilon_frequency() -> None:
    hits = _freq(_ANAL, r"\epsilon") + _freq(_ANAL, r"\varepsilon")
    assert hits >= 50, f"epsilon/varepsilon appeared only {hits}/2000 in analysis"


# ---------------------------------------------------------------------------
# asymptotics
# ---------------------------------------------------------------------------

_ASYM = _draw(GENERATORS["asymptotics"], 2000)


def test_asymptotics_landau_frequency() -> None:
    n = _freq(_ASYM, r"O\!")
    assert n >= 435, f"O\\! appeared only {n}/2000 in asymptotics"


def test_asymptotics_sim_frequency() -> None:
    n = _freq(_ASYM, r"\sim")
    assert n >= 196, f"\\sim appeared only {n}/2000 in asymptotics"


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

_GEO = _draw(GENERATORS["geometry"], 2000)


def test_geometry_angle_frequency() -> None:
    n = _freq(_GEO, r"\angle")
    assert n >= 350, f"\\angle appeared only {n}/2000 in geometry"


def test_geometry_triangle_frequency() -> None:
    n = _freq(_GEO, r"\triangle")
    assert n >= 400, f"\\triangle appeared only {n}/2000 in geometry"


# ---------------------------------------------------------------------------
# chemistry
# ---------------------------------------------------------------------------

_CHEM = _draw(GENERATORS["chemistry"], 2000)


def test_chemistry_rightarrow_frequency() -> None:
    n = _freq(_CHEM, r"\rightarrow")
    assert n >= 600, f"\\rightarrow appeared only {n}/2000 in chemistry"


def test_chemistry_H_frequency() -> None:
    n = _freq(_CHEM, "H")
    assert n >= 700, f"'H' appeared only {n}/2000 in chemistry"


# ---------------------------------------------------------------------------
# differential_equations
# ---------------------------------------------------------------------------

_DE = _draw(GENERATORS["differential_equations"], 2000)


def test_differential_equations_frac_d_frequency() -> None:
    n = _freq(_DE, r"\frac{d")
    assert n >= 100, f"\\frac{{d appeared only {n}/2000 in differential_equations"


# ---------------------------------------------------------------------------
# fourier
# ---------------------------------------------------------------------------

_FOURIER = _draw(GENERATORS["fourier"], 2000)


def test_fourier_hat_frequency() -> None:
    n = _freq(_FOURIER, r"\hat")
    assert n >= 600, f"\\hat appeared only {n}/2000 in fourier"


def test_fourier_int_frequency() -> None:
    n = _freq(_FOURIER, r"\int")
    assert n >= 500, f"\\int appeared only {n}/2000 in fourier"


# ---------------------------------------------------------------------------
# measure_theory
# ---------------------------------------------------------------------------

_MT = _draw(GENERATORS["measure_theory"], 2000)


def test_measure_theory_int_frequency() -> None:
    n = _freq(_MT, r"\int")
    assert n >= 1000, f"\\int appeared only {n}/2000 in measure_theory"


def test_measure_theory_mu_frequency() -> None:
    n = _freq(_MT, r"\mu")
    assert n >= 200, f"\\mu appeared only {n}/2000 in measure_theory"


# ---------------------------------------------------------------------------
# optimization
# ---------------------------------------------------------------------------

_OPT = _draw(GENERATORS["optimization"], 2000)


def test_optimization_nabla_frequency() -> None:
    n = _freq(_OPT, r"\nabla")
    assert n >= 600, f"\\nabla appeared only {n}/2000 in optimization"


def test_optimization_min_frequency() -> None:
    n = _freq(_OPT, r"\min")
    assert n >= 50, f"\\min appeared only {n}/2000 in optimization"


# ---------------------------------------------------------------------------
# p_adic
# ---------------------------------------------------------------------------

_PADIC = _draw(GENERATORS["p_adic"], 2000)


def test_p_adic_Q_frequency() -> None:
    n = _freq(_PADIC, r"\mathbb{Q}")
    assert n >= 200, f"\\mathbb{{Q}} appeared only {n}/2000 in p_adic"


def test_p_adic_Z_frequency() -> None:
    n = _freq(_PADIC, r"\mathbb{Z}")
    assert n >= 100, f"\\mathbb{{Z}} appeared only {n}/2000 in p_adic"


# ---------------------------------------------------------------------------
# stochastic_processes
# ---------------------------------------------------------------------------

_STO = _draw(GENERATORS["stochastic_processes"], 2000)


def test_stochastic_processes_E_frequency() -> None:
    n = _freq(_STO, r"\mathbb{E}")
    assert n >= 500, f"\\mathbb{{E}} appeared only {n}/2000 in stochastic_processes"


def test_stochastic_processes_W_frequency() -> None:
    n = _freq(_STO, "W_")
    assert n >= 100, f"'W_' appeared only {n}/2000 in stochastic_processes"


# ---------------------------------------------------------------------------
# representation_theory
# ---------------------------------------------------------------------------

_REP = _draw(GENERATORS["representation_theory"], 2000)


def test_representation_theory_rho_frequency() -> None:
    n = _freq(_REP, r"\rho")
    assert n >= 50, f"\\rho appeared only {n}/2000 in representation_theory"


def test_representation_theory_V_frequency() -> None:
    n = _freq(_REP, "V")
    assert n >= 100, f"'V' appeared only {n}/2000 in representation_theory"


# ---------------------------------------------------------------------------
# ring_field_theory
# ---------------------------------------------------------------------------

_RING = _draw(GENERATORS["ring_field_theory"], 2000)


def test_ring_field_theory_cdot_frequency() -> None:
    n = _freq(_RING, r"\cdot")
    assert n >= 400, f"\\cdot appeared only {n}/2000 in ring_field_theory"


def test_ring_field_theory_F_frequency() -> None:
    n = _freq(_RING, r"\mathbb{F}")
    assert n >= 200, f"\\mathbb{{F}} appeared only {n}/2000 in ring_field_theory"


# ---------------------------------------------------------------------------
# custom_operators
# ---------------------------------------------------------------------------

_CUSTOM = _draw(GENERATORS["custom_operators"], 2000)


def test_custom_operators_operatorname_frequency() -> None:
    n = _freq(_CUSTOM, r"\operatorname")
    assert n >= 80, f"\\operatorname appeared only {n}/2000 in custom_operators"


def test_custom_operators_lceil_frequency() -> None:
    n = _freq(_CUSTOM, r"\lceil")
    assert n >= 100, f"\\lceil appeared only {n}/2000 in custom_operators"
