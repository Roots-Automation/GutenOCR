"""Complex analysis domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import S, Template, compute_weights, make_dispatcher

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_VARS_POOL: list[str] = ["x", "y", "z", "a", "b", "u", "w"]
_SCALARS_POOL: list[str] = ["r", "s", "t", "c", "d"]
_THETA_POOL: list[str] = [r"\theta", r"\phi", r"\varphi"]
_A_POOL: list[str] = ["a", "z_0", r"\alpha"]
_A2_POOL: list[str] = ["a", "z_0"]
_MOBIUS_POOL: list[str] = ["a", "b", "c", "d"]

# ---------------------------------------------------------------------------
# Complex analysis templates
# ---------------------------------------------------------------------------

_COMPLEX_TEMPLATES: list[Template] = [
    Template(
        name="complex_cartesian",
        latex=r"z = {x} + i{y}",
        slots={"x": S(_VARS_POOL), "y": S(_VARS_POOL)},
    ),
    Template(
        name="complex_polar",
        latex=r"z = {r} e^{{i{th}}}",
        slots={"r": S(_SCALARS_POOL), "th": S(_THETA_POOL)},
    ),
    Template(
        name="modulus_squared",
        latex=r"|z|^2 = {x}^2 + {y}^2",
        slots={"x": S(_VARS_POOL), "y": S(_VARS_POOL)},
    ),
    Template(
        name="cauchy_riemann",
        latex=(
            r"\frac{{\partial u}}{{\partial x}} = \frac{{\partial v}}{{\partial y}}, \quad "
            r"\frac{{\partial u}}{{\partial y}} = -\frac{{\partial v}}{{\partial x}}"
        ),
        slots={},
    ),
    Template(
        name="cauchy_integral_formula",
        latex=r"f(a) = \frac{{1}}{{2\pi i}} \oint_C \frac{{f(z)}}{{z - {a}}} \, dz",
        slots={"a": S(_A_POOL)},
    ),
    Template(
        name="residue_theorem",
        latex=r"\oint_C f(z) \, dz = 2\pi i \sum_k \operatorname{{Res}}(f, z_k)",
        slots={},
    ),
    Template(
        name="laurent_series",
        latex=r"f(z) = \sum_{{n=-\infty}}^{{\infty}} c_n (z - {a})^n",
        slots={"a": S(_A2_POOL)},
    ),
    Template(
        name="mobius_transformation",
        latex=r"w = \frac{{{a}z + {b}}}{{{c}z + {d}}}, \quad {a}{d} - {b}{c} \neq 0",
        slots={"a": S(_MOBIUS_POOL), "b": S(_MOBIUS_POOL), "c": S(_MOBIUS_POOL), "d": S(_MOBIUS_POOL)},
    ),
    Template(
        name="eulers_identity",
        latex=r"e^{{i\pi}} + 1 = 0",
        slots={},
    ),
    Template(
        name="complex_conjugate",
        latex=r"\bar{{z}} = x - iy, \quad z\bar{{z}} = |z|^2",
        slots={},
    ),
    Template(
        name="real_imag_parts",
        latex=(
            r"\operatorname{{Re}}(z) = \frac{{z + \bar{{z}}}}{{2}}, \quad "
            r"\operatorname{{Im}}(z) = \frac{{z - \bar{{z}}}}{{2i}}"
        ),
        slots={},
    ),
    Template(
        name="argument_principle",
        latex=r"\frac{{1}}{{2\pi i}} \oint_C \frac{{f'(z)}}{{f(z)}} \, dz = N - P",
        slots={},
    ),
    Template(
        name="rouche_theorem",
        latex=r"|f(z) - g(z)| < |g(z)| \text{{ on }} C \implies Z_f = Z_g",
        slots={},
    ),
    Template(
        name="maximum_modulus",
        latex=r"|f(z)| \leq \max_{{|\zeta|=r}} |f(\zeta)| \text{{ for }} |z| \leq r",
        slots={},
    ),
]

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

_W_COMPLEX: list[float] = compute_weights(_COMPLEX_TEMPLATES)

# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

_complex_analysis = make_dispatcher(_COMPLEX_TEMPLATES, _W_COMPLEX)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "complex_analysis": _complex_analysis,
}

WEIGHTS: dict[str, float] = {
    "complex_analysis": 0.03,
}

TEMPLATES: dict[str, list[Template]] = {
    "complex_analysis": _COMPLEX_TEMPLATES,
}
