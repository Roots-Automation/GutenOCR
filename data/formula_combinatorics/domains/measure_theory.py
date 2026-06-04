"""Measure theory domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import S, Template, X, compute_weights, make_dispatcher

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_MU_POOL: list[str] = [r"\mu", r"\nu", r"\lambda"]
_NU_POOL: list[str] = [r"\mu", r"\nu", r"\rho"]
_P_NORM_POOL: list[str] = ["p", "2"]

# ---------------------------------------------------------------------------
# Measure theory templates
# ---------------------------------------------------------------------------

_MEASURE_TEMPLATES: list[Template] = [
    Template(
        name="sigma_additivity",
        latex=r"{mu}(\emptyset) = 0, \quad {mu}\!\left(\bigsqcup_n A_n\right) = \sum_n {mu}(A_n)",
        slots={"mu": S(_MU_POOL)},
    ),
    Template(
        name="nonnegativity_integral",
        latex=r"\int_X f \, d{mu} \geq 0 \text{{ for }} f \geq 0",
        slots={"mu": S(_MU_POOL)},
    ),
    Template(
        name="radon_nikodym",
        latex=r"\frac{{d{mu}}}{{d{nu}}} \geq 0, \quad {mu}(A) = \int_A \frac{{d{mu}}}{{d{nu}}} \, d{nu}",
        slots={"mu": S(_MU_POOL), "nu": X(_NU_POOL, ["mu"])},
    ),
    Template(
        name="l1_convergence",
        latex=r"\int_X |f_n - f| \, d{mu} \to 0",
        slots={"mu": S(_MU_POOL)},
    ),
    Template(
        name="fatou_lemma",
        latex=r"\int_X \liminf_n f_n \, d{mu} \leq \liminf_n \int_X f_n \, d{mu}",
        slots={"mu": S(_MU_POOL)},
    ),
    Template(
        name="fubini_tonelli",
        latex=r"\int_{{X \times Y}} f \, d({mu} \otimes {nu}) = \int_X \int_Y f(x,y) \, d{nu}(y) \, d{mu}(x)",
        slots={"mu": S(_MU_POOL), "nu": X(_NU_POOL, ["mu"])},
    ),
    Template(
        name="lp_norm",
        latex=r"\|f\|_{{L^{{{p}}}}} = \left(\int |f|^{{{p}}} \, d{mu}\right)^{{1/{p}}}",
        slots={"p": S(_P_NORM_POOL), "mu": S(_MU_POOL)},
    ),
    Template(
        name="absolute_continuity",
        latex=r"{mu} \ll {nu} \iff {mu}(A) = 0 \text{{ whenever }} {nu}(A) = 0",
        slots={"mu": S(_MU_POOL), "nu": X(_NU_POOL, ["mu"])},
    ),
    Template(
        name="lebesgue_decomposition",
        latex=r"{mu} = {mu}_{{ac}} + {mu}_{{sing}}",
        slots={"mu": S(_MU_POOL)},
    ),
    Template(
        name="dominated_convergence",
        latex=r"|f_n| \leq g,\; \int g \, d{mu} < \infty \implies \int f_n \, d{mu} \to \int f \, d{mu}",
        slots={"mu": S(_MU_POOL)},
    ),
    Template(
        name="sigma_algebra_generated",
        latex=r"\sigma(\mathcal{{C}}) = \bigcap \left\{{\mathcal{{F}} : \mathcal{{C}} \subseteq \mathcal{{F}},\, \mathcal{{F}} \text{{ is a }} \sigma\text{{-algebra}}\right\}}",
        slots={},
    ),
    Template(
        name="markov_inequality",
        latex=r"{mu}(\{{|f| \geq t\}}) \leq \frac{{1}}{{t}} \int |f| \, d{mu}",
        slots={"mu": S(_MU_POOL)},
    ),
    Template(
        name="jensens_inequality",
        latex=r"\varphi\!\left(\int f \, d{mu}\right) \leq \int \varphi(f) \, d{mu}",
        slots={"mu": S(_MU_POOL)},
    ),
    Template(
        name="product_measure",
        latex=r"({mu} \otimes {nu})(A \times B) = {mu}(A) \cdot {nu}(B)",
        slots={"mu": S(_MU_POOL), "nu": X(_NU_POOL, ["mu"])},
    ),
]

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

_W_MEASURE: list[float] = compute_weights(_MEASURE_TEMPLATES)

# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

_measure_theory = make_dispatcher(_MEASURE_TEMPLATES, _W_MEASURE)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "measure_theory": _measure_theory,
}

WEIGHTS: dict[str, float] = {
    "measure_theory": 0.02,
}

TEMPLATES: dict[str, list[Template]] = {
    "measure_theory": _MEASURE_TEMPLATES,
}
