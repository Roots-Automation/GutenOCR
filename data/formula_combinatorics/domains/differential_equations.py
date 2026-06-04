"""Differential equations domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import E, S, Template, compute_weights, make_dispatcher
from .._vocab import _SCALARS, _VARS

# ---------------------------------------------------------------------------
# Shared pools
# ---------------------------------------------------------------------------

_FUNC3_POOL: tuple[str, ...] = ("y", "u", r"\phi")

# ---------------------------------------------------------------------------
# Inline sub-generators
# ---------------------------------------------------------------------------


def _om_second_order(rng: random.Random) -> str:
    return rng.choice([r"\omega^2", "b", "c"])


def _kappa_sym(rng: random.Random) -> str:
    return rng.choice([r"\alpha", r"\kappa", "k"])


def _rate_sym(rng: random.Random) -> str:
    return rng.choice([r"\lambda", r"\alpha", "k"])


def _om_eig(rng: random.Random) -> str:
    return rng.choice([r"\lambda", r"\omega^2", "k"])


def _om0_sym(rng: random.Random) -> str:
    return rng.choice([r"\omega_0^2", "k"])


def _om_drive(rng: random.Random) -> str:
    return rng.choice([r"\omega", r"\Omega"])


# ---------------------------------------------------------------------------
# Differential equations templates
# ---------------------------------------------------------------------------

_DIFFEQ_TEMPLATES: list[Template] = [
    # c=0 — first-order linear ODE
    Template(
        name="first_order_linear_ode",
        latex=r"\frac{{d{f}}}{{d{v}}} + {a}({v}) {f} = g({v})",
        slots={
            "v": S(_VARS),
            "f": S(_FUNC3_POOL),
            "a": S(_SCALARS),
        },
    ),
    # c=1 — second-order linear ODE
    Template(
        name="second_order_linear_ode",
        latex=r"\frac{{d^2{f}}}{{d{v}^2}} + {a} \frac{{d{f}}}{{d{v}}} + {om} {f} = 0",
        slots={
            "v": S(_VARS),
            "f": S(_FUNC3_POOL),
            "a": S(_SCALARS),
            "om": E(_om_second_order, n=3),
        },
    ),
    # c=2 — heat equation
    Template(
        name="heat_equation",
        latex=r"\frac{{\partial u}}{{\partial t}} = {kap} \frac{{\partial^2 u}}{{\partial x^2}}",
        slots={"kap": E(_kappa_sym, n=3)},
    ),
    # c=3 — wave equation (fixed)
    Template(
        name="wave_equation",
        latex=r"\frac{{\partial^2 u}}{{\partial t^2}} = c^2 \frac{{\partial^2 u}}{{\partial x^2}}",
        slots={},
    ),
    # c=4 — Laplace equation
    Template(
        name="laplace_equation",
        latex=(
            r"\nabla^2 {f} = "
            r"\frac{{\partial^2 {f}}}{{\partial x^2}} + "
            r"\frac{{\partial^2 {f}}}{{\partial y^2}} = 0"
        ),
        slots={"f": S(_FUNC3_POOL)},
    ),
    # c=5 — general homogeneous ODE solution
    Template(
        name="homogeneous_ode_solution",
        latex=r"{f}({v}) = C_1 e^{{\lambda_1 {v}}} + C_2 e^{{\lambda_2 {v}}}",
        slots={
            "v": S(_VARS),
            "f": S(_FUNC3_POOL),
        },
    ),
    # c=6 — separable ODE
    Template(
        name="separable_ode",
        latex=r"\frac{{1}}{{{f}}} \frac{{d{f}}}{{d{v}}} = {a}({v})",
        slots={
            "v": S(_VARS),
            "f": S(_FUNC3_POOL),
            "a": S(_SCALARS),
        },
    ),
    # c=7 — exponential growth/decay solution
    Template(
        name="exponential_growth_solution",
        latex=r"{f}(t) = {f}_0 \, e^{{{rate} t}}",
        slots={
            "f": S(_FUNC3_POOL),
            "rate": E(_rate_sym, n=3),
        },
    ),
    # c=8 — Sturm-Liouville BVP
    Template(
        name="sturm_liouville_bvp",
        latex=(
            r"{f}''({v}) + {om} {f}({v}) = 0, \quad "
            r"{f}(0) = 0, \; {f}(L) = 0"
        ),
        slots={
            "v": S(_VARS),
            "f": S(_FUNC3_POOL),
            "om": E(_om_eig, n=3),
        },
    ),
    # c=9 — logistic growth equation (fixed)
    Template(
        name="logistic_equation",
        latex=r"\frac{{dP}}{{dt}} = r P\!\left(1 - \frac{{P}}{{K}}\right)",
        slots={},
    ),
    # c=10 — driven harmonic oscillator
    Template(
        name="driven_harmonic_oscillator",
        latex=(
            r"\frac{{\partial^2 u}}{{\partial t^2}} + 2{a} \frac{{\partial u}}{{\partial t}} "
            r"+ {om0} u = F_0 \cos({om} t)"
        ),
        slots={
            "a": S(_SCALARS),
            "om0": E(_om0_sym, n=2),
            "om": E(_om_drive, n=2),
        },
    ),
    # c=11 — integrating factor
    Template(
        name="integrating_factor",
        latex=r"\mu({v}) = e^{{\int {a}({v}) \, d{v}}}",
        slots={
            "v": S(_VARS),
            "a": S(_SCALARS),
        },
    ),
    # c=12 — Green's function (fixed)
    Template(
        name="greens_function",
        latex=r"L G(x, \xi) = \delta(x - \xi)",
        slots={},
    ),
    # c=13 — variation of parameters
    Template(
        name="variation_of_parameters",
        latex=r"{f}_p({v}) = {f}_1({v}) \int \frac{{{f}_2 g}}{{{f}_1 {f}_2' - {f}_2 {f}_1'}} \, d{v}",
        slots={
            "v": S(_VARS),
            "f": S(_FUNC3_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

_W_DIFFEQ: list[float] = compute_weights(_DIFFEQ_TEMPLATES)

# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

_differential_equations = make_dispatcher(_DIFFEQ_TEMPLATES, _W_DIFFEQ)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "differential_equations": _differential_equations,
}

WEIGHTS: dict[str, float] = {
    "differential_equations": 0.03,
}

TEMPLATES: dict[str, list[Template]] = {
    "differential_equations": _DIFFEQ_TEMPLATES,
}
