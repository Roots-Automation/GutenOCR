"""Probability and statistics domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import S, Template, X, compute_weights, make_dispatcher

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_RV_POOL: list[str] = ["X", "Y", "Z"]
_K_POOL: list[str] = ["k", "m"]
_N_POOL: list[str] = ["n", "N"]
_P_POOL: list[str] = ["p", r"\theta"]
_LAM_POOL: list[str] = [r"\lambda", r"\mu"]
_MU_POOL: list[str] = [r"\mu", "0"]
_SIG_POOL: list[str] = [r"\sigma", r"\sigma_0"]

# ---------------------------------------------------------------------------
# Probability templates
# ---------------------------------------------------------------------------

_PROB_TEMPLATES: list[Template] = [
    Template(
        name="binomial_pmf",
        latex=r"P({x} = {k}) = \binom{{{n}}}{{{k}}} {p}^{{{k}}} (1-{p})^{{{n}-{k}}}",
        slots={"x": S(_RV_POOL), "k": S(_K_POOL), "n": S(_N_POOL), "p": S(_P_POOL)},
    ),
    Template(
        name="poisson_pmf",
        latex=r"P({x} = {k}) = \frac{{{lam}^{{{k}}} e^{{-{lam}}}}}{{{k}!}}",
        slots={"x": S(_RV_POOL), "k": S(_K_POOL), "lam": S(_LAM_POOL)},
    ),
    Template(
        name="normal_pdf",
        latex=(
            r"f(x) = \frac{{1}}{{\sqrt{{2\pi}} {sig}}} "
            r"\exp\!\left(-\frac{{(x - {mu})^2}}{{2 {sig}^2}}\right)"
        ),
        slots={"mu": S(_MU_POOL), "sig": S(_SIG_POOL)},
    ),
    Template(
        name="expected_value_discrete",
        latex=r"E[{x}] = \sum_{{k}} k \cdot P({x} = k)",
        slots={"x": S(_RV_POOL)},
    ),
    Template(
        name="expected_value_continuous",
        latex=r"E[{x}] = \int_{{-\infty}}^{{\infty}} x \, f(x) \, dx",
        slots={"x": S(_RV_POOL)},
    ),
    Template(
        name="variance",
        latex=r"\text{{Var}}({x}) = E\!\left[{x}^2\right] - \left(E[{x}]\right)^2",
        slots={"x": S(_RV_POOL)},
    ),
    Template(
        name="bayes_theorem",
        latex=r"P(A \mid B) = \frac{{P(B \mid A) \, P(A)}}{{P(B)}}",
        slots={},
    ),
    Template(
        name="law_of_total_probability",
        latex=r"P(A) = \sum_{{i=1}}^{{{n}}} P(A \mid B_i) \, P(B_i)",
        slots={"n": S(_N_POOL)},
    ),
    Template(
        name="mgf",
        latex=(
            r"M_{{{x}}}(t) = E\!\left[e^{{t {x}}}\right] = "
            r"\sum_{{k=0}}^{{\infty}} \frac{{E[{x}^k]}}{{k!}} t^k"
        ),
        slots={"x": S(_RV_POOL)},
    ),
    Template(
        name="covariance",
        latex=r"\text{{Cov}}({x}, {y}) = E[{x} {y}] - E[{x}] E[{y}]",
        slots={"x": S(_RV_POOL), "y": X(_RV_POOL, ["x"])},
    ),
    Template(
        name="cdf",
        latex=r"F(x) = P({x} \leq x) = \int_{{-\infty}}^{{x}} f(t) \, dt",
        slots={"x": S(_RV_POOL)},
    ),
    Template(
        name="geometric_pmf",
        latex=r"P({x} = {k}) = (1 - {p})^{{{k}-1}} {p}",
        slots={"x": S(_RV_POOL), "k": S(_K_POOL), "p": S(_P_POOL)},
    ),
    Template(
        name="jensens_inequality",
        latex=r"f\!\left(E[{x}]\right) \leq E\!\left[f({x})\right]",
        slots={"x": S(_RV_POOL)},
    ),
    Template(
        name="correlation",
        latex=(
            r"\rho_{{{x}{y}}} = "
            r"\frac{{\text{{Cov}}({x}, {y})}}{{\sqrt{{\text{{Var}}({x}) \, \text{{Var}}({y})}}}}"
        ),
        slots={"x": S(_RV_POOL), "y": X(_RV_POOL, ["x"])},
    ),
    Template(
        name="central_limit_theorem",
        latex=r"\frac{{\bar{{{x}}}_n - \mu}}{{\sigma / \sqrt{{{n}}}}} \xrightarrow{{d}} \mathcal{{N}}(0,1)",
        slots={"x": S(_RV_POOL), "n": S(_N_POOL)},
    ),
    Template(
        name="law_of_large_numbers",
        latex=r"\bar{{{x}}}_n \xrightarrow{{p}} \mu \text{{ as }} n \to \infty",
        slots={"x": S(_RV_POOL)},
    ),
    Template(
        name="characteristic_function",
        latex=r"\varphi_{{{x}}}(t) = E\!\left[e^{{it{x}}}\right]",
        slots={"x": S(_RV_POOL)},
    ),
    Template(
        name="tower_property",
        latex=r"E\!\left[E[{x} \mid \mathcal{{F}}]\right] = E[{x}]",
        slots={"x": S(_RV_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

_W_PROB: list[float] = compute_weights(_PROB_TEMPLATES)

# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

_probability = make_dispatcher(_PROB_TEMPLATES, _W_PROB)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "probability": _probability,
}

WEIGHTS: dict[str, float] = {
    "probability": 0.09,
}

TEMPLATES: dict[str, list[Template]] = {
    "probability": _PROB_TEMPLATES,
}
