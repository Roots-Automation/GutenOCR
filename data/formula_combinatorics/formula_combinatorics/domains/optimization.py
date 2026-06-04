"""Optimization domain generator."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import S, Template, compute_weights, make_dispatcher

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_F_POOL = ["f", "F", r"\mathcal{L}"]
_ETA_POOL = [r"\eta", r"\alpha", r"\gamma"]
_LAM_POOL = [r"\lambda", r"\mu"]

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_OPTIMIZATION_TEMPLATES: list[Template] = [
    Template(
        name="gradient_descent_step",
        latex=r"x_{{k+1}} = x_k - {eta} \nabla {f}(x_k)",
        slots={"f": S(_F_POOL), "eta": S(_ETA_POOL)},
    ),
    Template(
        name="lagrangian",
        latex=r"\mathcal{{L}}(x, {lam}) = {f}(x) + {lam}^\top g(x)",
        slots={"f": S(_F_POOL), "lam": S(_LAM_POOL)},
    ),
    Template(
        name="kkt_stationarity",
        latex=r"\nabla {f}(x^*) = {lam} \nabla g(x^*)",
        slots={"f": S(_F_POOL), "lam": S(_LAM_POOL)},
    ),
    Template(
        name="convexity_definition",
        latex=r"{f}({lam} x + (1-{lam}) y) \leq {lam} {f}(x) + (1-{lam}) {f}(y)",
        slots={"f": S(_F_POOL), "lam": S(_LAM_POOL)},
    ),
    Template(
        name="conjugate_function",
        latex=r"{f}^*(y) = \sup_x \left\langle y, x \right\rangle - {f}(x)",
        slots={"f": S(_F_POOL)},
    ),
    Template(
        name="proximal_operator",
        latex=r"\operatorname{{prox}}_{{{f}}}(v) = \arg\min_x \left\{{{f}(x) + \tfrac{{1}}{{2}} \|x-v\|^2\right\}}",
        slots={"f": S(_F_POOL)},
    ),
    Template(
        name="newtons_method_step",
        latex=r"x_{{k+1}} = x_k - \left[\nabla^2 {f}(x_k)\right]^{{-1}} \nabla {f}(x_k)",
        slots={"f": S(_F_POOL)},
    ),
    Template(
        name="constrained_minimization",
        latex=r"\min_x \; {f}(x) \quad \text{{s.t.}} \quad g_i(x) \leq 0",
        slots={"f": S(_F_POOL)},
    ),
    Template(
        name="lipschitz_gradient",
        latex=r"\|\nabla {f}(x) - \nabla {f}(y)\| \leq L \|x - y\|",
        slots={"f": S(_F_POOL)},
    ),
    Template(
        name="first_order_convexity",
        latex=r"{f}(y) \geq {f}(x) + \nabla {f}(x)^\top (y - x)",
        slots={"f": S(_F_POOL)},
    ),
    Template(
        name="sublinear_convergence_rate",
        latex=r"\frac{{1}}{{T}} \sum_{{t=1}}^T {f}(x_t) - {f}(x^*) \leq O\!\left(\frac{{1}}{{\sqrt{{T}}}}\right)",
        slots={"f": S(_F_POOL)},
    ),
    Template(
        name="linear_convergence",
        latex=r"\|x_{{k+1}} - x^*\| \leq {lam} \|x_k - x^*\|",
        slots={"lam": S(_LAM_POOL)},
    ),
    Template(
        name="strong_convexity",
        latex=r"{f}(y) \geq {f}(x) + \nabla {f}(x)^\top (y-x) + \frac{{{lam}}}{{2}} \|y-x\|^2",
        slots={"f": S(_F_POOL), "lam": S(_LAM_POOL)},
    ),
    Template(
        name="proximal_point_step",
        latex=r"x^{{k+1}} = \arg\min_x \left\{{{f}(x) + \frac{{{lam}}}{{2}} \|{lam} x + z^k\|^2\right\}}",
        slots={"f": S(_F_POOL), "lam": S(_LAM_POOL)},
    ),
    Template(
        name="subgradient_step",
        latex=r"x_{{k+1}} = x_k - {eta}_k g_k, \quad g_k \in \partial {f}(x_k)",
        slots={"f": S(_F_POOL), "eta": S(_ETA_POOL)},
    ),
]

_W = compute_weights(_OPTIMIZATION_TEMPLATES)

_optimization = make_dispatcher(_OPTIMIZATION_TEMPLATES, _W)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "optimization": _optimization,
}

WEIGHTS: dict[str, float] = {
    "optimization": 0.05,
}

TEMPLATES: dict = {
    "optimization": _OPTIMIZATION_TEMPLATES,
}
