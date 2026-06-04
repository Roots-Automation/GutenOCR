"""p-adic numbers domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import S, Template, X, compute_weights, make_dispatcher

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_P_ADIC_POOL: list[str] = ["p", "q", r"\ell"]
_X_ELEM_POOL: list[str] = ["x", "a", r"\alpha"]
_XY_ELEM_POOL: list[str] = ["x", "a"]
_XY2_POOL: list[str] = ["x", r"\alpha"]
_YY_POOL: list[str] = ["y", r"\beta"]
_N_POOL: list[str] = ["n", "m", "N"]
_NJ_POOL: list[str] = ["n", "m"]
_J_POOL: list[str] = ["j", "k"]

# ---------------------------------------------------------------------------
# p-adic templates
# ---------------------------------------------------------------------------

_PADIC_TEMPLATES: list[Template] = [
    Template(
        name="padic_valuation",
        latex=r"|{x}|_{{{p}}} = {p}^{{-v_{{{p}}}({x})}}",
        slots={"x": S(_X_ELEM_POOL), "p": S(_P_ADIC_POOL)},
    ),
    Template(
        name="padic_integers",
        latex=r"\mathbb{{Z}}_{{{p}}} = \left\{{{x} \in \mathbb{{Q}}_{{{p}}} : |{x}|_{{{p}}} \leq 1\right\}}",
        slots={"x": S(_X_ELEM_POOL), "p": S(_P_ADIC_POOL)},
    ),
    Template(
        name="padic_expansion",
        latex=r"{x} = \sum_{{k=0}}^{{\infty}} a_k {p}^k, \quad 0 \leq a_k < {p}",
        slots={"x": S(_XY_ELEM_POOL), "p": S(_P_ADIC_POOL)},
    ),
    Template(
        name="ultrametric_inequality",
        latex=r"|{x} + {y}|_{{{p}}} \leq \max\!\left(|{x}|_{{{p}}}, |{y}|_{{{p}}}\right)",
        slots={"x": S(_XY2_POOL), "y": S(_YY_POOL), "p": S(_P_ADIC_POOL)},
    ),
    Template(
        name="padic_multiplicativity",
        latex=r"|{x} {y}|_{{{p}}} = |{x}|_{{{p}}} |{y}|_{{{p}}}",
        slots={"x": S(_XY_ELEM_POOL), "y": S(_XY_ELEM_POOL), "p": S(_P_ADIC_POOL)},
    ),
    Template(
        name="valuation_subadditivity",
        latex=r"v_{{{p}}}({x} + {y}) \geq \min\!\left(v_{{{p}}}({x}),\, v_{{{p}}}({y})\right)",
        slots={"x": S(_XY2_POOL), "y": S(_YY_POOL), "p": S(_P_ADIC_POOL)},
    ),
    Template(
        name="padic_norm_prime",
        latex=r"|{p}|_{{{p}}} = {p}^{{-1}}",
        slots={"p": S(_P_ADIC_POOL)},
    ),
    Template(
        name="product_formula",
        latex=r"|{x}|_\infty \cdot \prod_{{{p}}} |{x}|_{{{p}}} = 1",
        slots={"x": S(_X_ELEM_POOL), "p": S(_P_ADIC_POOL)},
    ),
    Template(
        name="hensels_lemma",
        latex=(
            r"f(a) \equiv 0 \pmod{{{p}}},\; f'(a) \not\equiv 0 \pmod{{{p}}} "
            r"\implies \exists!\, \alpha \in \mathbb{{Z}}_{{{p}}} : f(\alpha) = 0"
        ),
        slots={"p": S(_P_ADIC_POOL)},
    ),
    Template(
        name="legendre_factorial_valuation",
        latex=r"v_{{{p}}}({n}!) = \sum_{{k=1}}^{{\infty}} \left\lfloor \frac{{{n}}}{{{p}^k}} \right\rfloor",
        slots={"p": S(_P_ADIC_POOL), "n": S(_N_POOL)},
    ),
    Template(
        name="qp_completion",
        latex=r"\mathbb{{Q}}_{{{p}}} = \widehat{{\mathbb{{Q}}}}_{{|\cdot|_{{{p}}}}}",
        slots={"p": S(_P_ADIC_POOL)},
    ),
    Template(
        name="padic_unique_absolute_value",
        latex=r"\|\cdot\|_{{{p}}} \text{{ is the unique }} {p}\text{{-adic absolute value on }} \mathbb{{Q}}",
        slots={"p": S(_P_ADIC_POOL)},
    ),
    Template(
        name="padic_exponential",
        latex=r"\exp_{{{p}}}({x}) = \sum_{{k=0}}^{{\infty}} \frac{{{x}^k}}{{k!}}, \quad |{x}|_{{{p}}} < {p}^{{-1/({p}-1)}}",
        slots={"p": S(_P_ADIC_POOL), "x": S(_XY2_POOL)},
    ),
    Template(
        name="padic_binomial_valuation",
        latex=r"v_{{{p}}}\!\left(\binom{{{p}^{{{n}}}}}{{{p}^{{{j}}}}}\right) = {n} - {j}",
        slots={"p": S(_P_ADIC_POOL), "n": S(_NJ_POOL), "j": X(_J_POOL, ["n"])},
    ),
]

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

_W_PADIC: list[float] = compute_weights(_PADIC_TEMPLATES)

# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

_p_adic = make_dispatcher(_PADIC_TEMPLATES, _W_PADIC)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "p_adic": _p_adic,
}

WEIGHTS: dict[str, float] = {
    "p_adic": 0.02,
}

TEMPLATES: dict[str, list[Template]] = {
    "p_adic": _PADIC_TEMPLATES,
}
