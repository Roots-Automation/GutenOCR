"""Physics domain generator (canonical equations, parameterized for variety)."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import S, Template, compute_weights, make_dispatcher

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_MASS_POOL = ["m", r"\mu", "M"]
_CHARGE_POOL = ["q", "e", "Q"]
_HBAR_POOL = [r"\hbar", r"\hbar/2\pi"]
_Q_GEN_POOL = ["q", r"q_i", r"q_k"]
_BETA_POOL = [r"\beta", r"1/k_B T"]
_EPS_POOL = [r"\epsilon_0", r"\varepsilon_0"]
_MU0_POOL = [r"\mu_0", r"\mu"]
_SIG_POOL = [r"\sigma", r"\sigma_{\mathrm{SB}}"]

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_PHYSICS_TEMPLATES: list[Template] = [
    Template(
        name="hamiltonian",
        latex=r"H = \frac{{p^2}}{{2{m}}} + V(q)",
        slots={"m": S(_MASS_POOL)},
    ),
    Template(
        name="lagrangian",
        latex=r"L = \frac{{1}}{{2}} {m} \dot{{q}}^2 - V(q)",
        slots={"m": S(_MASS_POOL)},
    ),
    Template(
        name="euler_lagrange",
        latex=r"\frac{{d}}{{dt}}\frac{{\partial L}}{{\partial \dot{{{q}}}}} - \frac{{\partial L}}{{\partial {q}}} = 0",
        slots={"q": S(_Q_GEN_POOL)},
    ),
    Template(
        name="schrodinger_time_dependent",
        latex=r"i{hbar} \frac{{\partial \psi}}{{\partial t}} = \hat{{H}} \psi",
        slots={"hbar": S(_HBAR_POOL)},
    ),
    Template(
        name="schrodinger_time_independent",
        latex=r"-\frac{{{hbar}^2}}{{2{m}}} \nabla^2 \psi + V \psi = E \psi",
        slots={"hbar": S(_HBAR_POOL), "m": S(["m", r"\mu"])},
    ),
    Template(
        name="newtons_second_law",
        latex=r"\mathbf{{F}} = {m} \ddot{{\mathbf{{r}}}}",
        slots={"m": S(_MASS_POOL)},
    ),
    Template(
        name="gauss_law",
        latex=r"\nabla \cdot \mathbf{{E}} = \frac{{\rho}}{{{eps}}}",
        slots={"eps": S(_EPS_POOL)},
    ),
    Template(
        name="ampere_maxwell",
        latex=r"\nabla \times \mathbf{{B}} = {mu0} \mathbf{{J}} + {mu0} {eps0} \frac{{\partial \mathbf{{E}}}}{{\partial t}}",
        slots={"mu0": S(_MU0_POOL), "eps0": S(_EPS_POOL)},
    ),
    Template(
        name="lorentz_force",
        latex=r"\mathbf{{F}} = {q}\!\left(\mathbf{{E}} + \mathbf{{v}} \times \mathbf{{B}}\right)",
        slots={"q": S(_CHARGE_POOL)},
    ),
    Template(
        name="energy_momentum_relation",
        latex=r"E^2 = (pc)^2 + (mc^2)^2",
        slots={},
    ),
    Template(
        name="angular_momentum",
        latex=r"\mathbf{{L}} = \mathbf{{r}} \times \mathbf{{p}}",
        slots={},
    ),
    Template(
        name="partition_function",
        latex=r"Z = \sum_i e^{{-{beta} E_i}}",
        slots={"beta": S(_BETA_POOL)},
    ),
    Template(
        name="poisson_bracket",
        latex=(
            r"\{{H, f\}} = "
            r"\frac{{\partial H}}{{\partial q}}\frac{{\partial f}}{{\partial p}} - "
            r"\frac{{\partial H}}{{\partial p}}\frac{{\partial f}}{{\partial q}}"
        ),
        slots={},
    ),
    Template(
        name="euler_lagrange_alt",
        latex=r"\frac{{d}}{{dt}} \frac{{\partial L}}{{\partial \dot{{{q}}}}} = \frac{{\partial L}}{{\partial {q}}}",
        slots={"q": S(["q", r"q_i"])},
    ),
    Template(
        name="stefan_boltzmann",
        latex=r"P = {sig} A T^4",
        slots={"sig": S(_SIG_POOL)},
    ),
    Template(
        name="heisenberg_uncertainty",
        latex=r"\Delta x \, \Delta p \geq \frac{{\hbar}}{{2}}",
        slots={},
    ),
]

_W = compute_weights(_PHYSICS_TEMPLATES)

_physics = make_dispatcher(_PHYSICS_TEMPLATES, _W)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "physics": _physics,
}

WEIGHTS: dict[str, float] = {
    "physics": 0.06,
}

TEMPLATES: dict = {
    "physics": _PHYSICS_TEMPLATES,
}
