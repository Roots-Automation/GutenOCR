"""Differential geometry domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import Template, compute_weights, make_dispatcher

# ---------------------------------------------------------------------------
# Differential geometry templates
# ---------------------------------------------------------------------------

_DIFFGEOM_TEMPLATES: list[Template] = [
    Template(
        name="metric_tensor",
        latex=r"ds^2 = g_{{ij}} \, dx^i \, dx^j",
        slots={},
    ),
    Template(
        name="christoffel_symbols",
        latex=r"\Gamma^k_{{ij}} = \frac{{1}}{{2}} g^{{kl}} \left(\partial_i g_{{jl}} + \partial_j g_{{il}} - \partial_l g_{{ij}}\right)",
        slots={},
    ),
    Template(
        name="geodesic_equation",
        latex=r"\frac{{d^2 x^k}}{{d\tau^2}} + \Gamma^k_{{ij}} \frac{{dx^i}}{{d\tau}} \frac{{dx^j}}{{d\tau}} = 0",
        slots={},
    ),
    Template(
        name="riemann_curvature",
        latex=r"R^l{{}}\_{{kij}} = \partial_i \Gamma^l_{{jk}} - \partial_j \Gamma^l_{{ik}} + \Gamma^l_{{im}} \Gamma^m_{{jk}} - \Gamma^l_{{jm}} \Gamma^m_{{ik}}",
        slots={},
    ),
    Template(
        name="torsion_free",
        latex=r"\nabla_X Y - \nabla_Y X = [X, Y]",
        slots={},
    ),
    Template(
        name="exterior_derivative_squared",
        latex=r"d(d\omega) = 0",
        slots={},
    ),
    Template(
        name="stokes_theorem",
        latex=r"\int_M d\omega = \int_{{\partial M}} \omega",
        slots={},
    ),
    Template(
        name="cartan_magic_formula",
        latex=r"\mathcal{{L}}_X \omega = d(\iota_X \omega) + \iota_X \, d\omega",
        slots={},
    ),
    Template(
        name="gaussian_curvature",
        latex=r"K = \frac{{R_{{1212}}}}{{g_{{11}} g_{{22}} - g_{{12}}^2}}",
        slots={},
    ),
    Template(
        name="ricci_tensor",
        latex=r"R_{{ij}} = R^k{{}}\_{{ikj}}",
        slots={},
    ),
    Template(
        name="einstein_field_equations",
        latex=r"G_{{ij}} = R_{{ij}} - \frac{{1}}{{2}} g_{{ij}} R = \frac{{8\pi G}}{{c^4}} T_{{ij}}",
        slots={},
    ),
    Template(
        name="covariant_derivative_tensor",
        latex=r"\nabla_k T^{{ij}} = \partial_k T^{{ij}} + \Gamma^i_{{kl}} T^{{lj}} + \Gamma^j_{{kl}} T^{{il}}",
        slots={},
    ),
    Template(
        name="lie_bracket",
        latex=r"[X, Y]^i = X^j \partial_j Y^i - Y^j \partial_j X^i",
        slots={},
    ),
    Template(
        name="gauss_bonnet",
        latex=r"\int_M K \, dA + \int_{{\partial M}} \kappa_g \, ds = 2\pi \chi(M)",
        slots={},
    ),
]

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

_W_DIFFGEOM: list[float] = compute_weights(_DIFFGEOM_TEMPLATES)

# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

_differential_geometry = make_dispatcher(_DIFFGEOM_TEMPLATES, _W_DIFFGEOM)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "differential_geometry": _differential_geometry,
}

WEIGHTS: dict[str, float] = {
    "differential_geometry": 0.02,
}

TEMPLATES: dict[str, list[Template]] = {
    "differential_geometry": _DIFFGEOM_TEMPLATES,
}
