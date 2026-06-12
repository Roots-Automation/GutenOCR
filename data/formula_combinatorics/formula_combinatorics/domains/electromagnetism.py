"""Electromagnetism domain — Maxwell equations, EM field tensor, waves."""

from __future__ import annotations

from ..engine._template_dsl import _FN_SLOT, S, Template, X
from ._config import register_domain
from ._physics_vocab import (
    _B_FIELD,
    _CHARGE_POOL,
    _E_FIELD,
    _EPS_POOL,
    _K_POOL,
    _MU0_POOL,
    _OMEGA_POOL,
    _RHO_POOL,
    _SIGMA_POOL,
    _SPACETIME_IDX_POOL,
    _VEC_F,
)

_TEMPLATES: list[Template] = [
    # ── Part A: Gauss / Ampère-Maxwell ───────────────────────────────────────
    Template(
        name="gauss_law",
        latex=r"\nabla \cdot {ef} = \frac{{{rho}}}{{{eps}}}",
        slots={"ef": S(_E_FIELD), "rho": S(_RHO_POOL), "eps": S(_EPS_POOL)},
    ),
    Template(
        name="ampere_maxwell",
        latex=(
            r"\nabla \times {bf} = {mu0}\,\mathbf{{J}}"
            r" + {mu0}\,{eps}\,\frac{{\partial {ef}}}{{\partial t}}"
        ),
        slots={
            "bf": S(_B_FIELD),
            "mu0": S(_MU0_POOL),
            "eps": S(_EPS_POOL),
            "ef": S(_E_FIELD),
        },
    ),
    # ── Part B: Maxwell equations and EM fields ──────────────────────────────
    Template(
        name="gauss_law_magnetic",
        latex=r"\nabla \cdot {bf} = 0",
        slots={"bf": S(_B_FIELD)},
    ),
    Template(
        name="faraday_law",
        latex=r"\nabla \times {ef} = -\frac{{\partial {bf}}}{{\partial t}}",
        slots={"ef": S(_E_FIELD), "bf": S(_B_FIELD)},
    ),
    Template(
        name="coulombs_law",
        latex=r"{ff} = \frac{{{q1}\,{q2}}}{{4\pi\,{eps}\,r^2}}\,\hat{{r}}",
        slots={
            "ff": S(_VEC_F),
            "q1": S(_CHARGE_POOL),
            "q2": X(_CHARGE_POOL, ("q1",)),
            "eps": S(_EPS_POOL),
        },
    ),
    Template(
        name="electric_field_point_charge",
        latex=r"{ef} = \frac{{{qq}}}{{4\pi\,{eps}\,r^2}}\,\hat{{r}}",
        slots={"ef": S(_E_FIELD), "qq": S(_CHARGE_POOL), "eps": S(_EPS_POOL)},
    ),
    Template(
        name="ohms_law_microscopic",
        latex=r"\mathbf{{J}} = {sg}\,{ef}",
        slots={"sg": S(_SIGMA_POOL), "ef": S(_E_FIELD)},
    ),
    Template(
        name="poynting_vector_def",
        latex=r"\mathbf{{S}} = \frac{{1}}{{{mu0}}}\,{ef} \times {bf}",
        slots={"mu0": S(_MU0_POOL), "ef": S(_E_FIELD), "bf": S(_B_FIELD)},
    ),
    Template(
        name="em_wave_equation",
        latex=(
            r"\nabla^2 {ef}"
            r" - \frac{{1}}{{c^2}}\frac{{\partial^2 {ef}}}{{\partial t^2}} = 0"
        ),
        slots={"ef": S(_E_FIELD)},
    ),
    Template(
        name="plane_wave_solution",
        latex=r"{ef} = E_0\cos\!\bigl({kk}\,x - {om}\,t\bigr)",
        slots={"ef": S(_E_FIELD), "kk": S(_K_POOL), "om": S(_OMEGA_POOL)},
    ),
    Template(
        name="em_energy_density",
        latex=(
            r"u = \tfrac{{1}}{{2}}\,{eps}\,{ef}^2"
            r" + \frac{{{bf}^2}}{{2\,{mu0}}}"
        ),
        slots={
            "eps": S(_EPS_POOL),
            "ef": S(_E_FIELD),
            "bf": S(_B_FIELD),
            "mu0": S(_MU0_POOL),
        },
    ),
    Template(
        name="continuity_equation_charge",
        latex=r"\frac{{\partial {rho}}}{{\partial t}} + \nabla \cdot \mathbf{{J}} = 0",
        slots={"rho": S(_RHO_POOL)},
    ),
    # ── Part C: EM field tensor (covariant notation) ─────────────────────────
    Template(
        name="em_field_tensor_def",
        latex=r"F_{{{mu}{nu}}} = \partial_{{{mu}}} A_{{{nu}}} - \partial_{{{nu}}} A_{{{mu}}}",
        slots={
            "mu": S(_SPACETIME_IDX_POOL),
            "nu": X(_SPACETIME_IDX_POOL, ("mu",)),
        },
    ),
    Template(
        name="em_maxwell_covariant",
        latex=r"\partial_{{{mu}}} F^{{{mu}{nu}}} = {mu0}\,j^{{{nu}}}",
        slots={
            "mu0": S(_MU0_POOL),
            "mu": S(_SPACETIME_IDX_POOL),
            "nu": X(_SPACETIME_IDX_POOL, ("mu",)),
        },
    ),
    Template(
        name="em_bianchi_covariant",
        latex=(
            r"\partial_{{{mu}}} F_{{{nu}{rho}}}"
            r" + \partial_{{{nu}}} F_{{{rho}{mu}}}"
            r" + \partial_{{{rho}}} F_{{{mu}{nu}}} = 0"
        ),
        slots={
            "mu": S(_SPACETIME_IDX_POOL),
            "nu": X(_SPACETIME_IDX_POOL, ("mu",)),
            "rho": X(_SPACETIME_IDX_POOL, ("mu", "nu")),
        },
    ),
    # ── Appendix: Circuit and wave notation ──────────────────────────────────
    Template(
        name="conductance_unit_mho",
        latex=r"G = \frac{{1}}{{R}},\quad [G] = \mho",
        slots={},
    ),
    Template(
        name="conductance_ohms_law_mho",
        latex=r"G = \frac{{I}}{{U}},\quad G \in \mho",
        slots={},
    ),
    Template(
        name="complex_impedance_re_im",
        latex=r"Z = \Re(Z) + i\,\Im(Z)",
        slots={},
    ),
    # ── Part C: Function-pair (EM context) ───────────────────────────────────
    Template(
        name="fn_em_wave_pair",
        latex=(
            r"{fn1}\!\left(\frac{{\partial^2 {ef}}}{{\partial t^2}}\right)"
            r" = {fn2}\!\left(c^2\,\nabla^2 {ef}\right)"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "ef": S(_E_FIELD),
        },
    ),
]

GENERATORS, WEIGHTS, TEMPLATES = register_domain("electromagnetism", _TEMPLATES)
