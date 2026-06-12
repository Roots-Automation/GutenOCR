"""Classical mechanics domain — Hamiltonian/Lagrangian, oscillators, SR, waves."""

from __future__ import annotations

from ..engine._template_dsl import _FN_SLOT, _LIM_MOD, S, Template, X
from ._config import register_domain
from ._physics_vocab import (
    _B_FIELD,
    _CHARGE_POOL,
    _COORD_POOL,
    _E_FIELD,
    _ENERGY_POOL,
    _EPS_POOL,
    _GAMMA_POOL,
    _HAM_POOL,
    _HBAR_POOL,
    _IDX_POOL,
    _LAG_POOL,
    _LAM_POOL,
    _MASS_POOL,
    _OMEGA_POOL,
    _P_POOL,
    _Q_POOL,
    _VEC_F,
    _VEL_POOL,
)

_TEMPLATES: list[Template] = [
    # ── Part A: Lagrangian / Hamiltonian mechanics ───────────────────────────
    Template(
        name="hamiltonian",
        latex=r"{ham} = \frac{{p^2}}{{2\,{mm}}} + V({qq})",
        slots={"ham": S(_HAM_POOL), "mm": S(_MASS_POOL), "qq": S(_Q_POOL)},
    ),
    Template(
        name="lagrangian",
        latex=r"{lag} = \frac{{1}}{{2}}\,{mm}\,\dot{{{qq}}}^2 - V({qq})",
        slots={"lag": S(_LAG_POOL), "mm": S(_MASS_POOL), "qq": S(_Q_POOL)},
    ),
    Template(
        name="euler_lagrange",
        latex=(
            r"\frac{{d}}{{dt}}\frac{{\partial {lag}}}{{\partial \dot{{{qq}}}}}"
            r" - \frac{{\partial {lag}}}{{\partial {qq}}} = 0"
        ),
        slots={"lag": S(_LAG_POOL), "qq": S(_Q_POOL)},
    ),
    Template(
        name="euler_lagrange_alt",
        latex=(
            r"\frac{{d}}{{dt}}\frac{{\partial {lag}}}{{\partial \dot{{{qq}}}}}"
            r" = \frac{{\partial {lag}}}{{\partial {qq}}}"
        ),
        slots={"lag": S(_LAG_POOL), "qq": S(_Q_POOL)},
    ),
    Template(
        name="newtons_second_law",
        latex=r"{ff} = {mm}\,\ddot{{\mathbf{{r}}}}",
        slots={"ff": S(_VEC_F), "mm": S(_MASS_POOL)},
    ),
    Template(
        name="angular_momentum",
        latex=r"\mathbf{{L}}_{{{ii}}} = \mathbf{{r}}_{{{ii}}} \times \mathbf{{p}}_{{{ii}}}",
        slots={"ii": S(_IDX_POOL)},
    ),
    Template(
        name="poisson_bracket",
        latex=(
            r"\{{{ham},\,{fn}\}}"
            r" = \frac{{\partial {ham}}}{{\partial q}}\frac{{\partial {fn}}}{{\partial p}}"
            r" - \frac{{\partial {ham}}}{{\partial p}}\frac{{\partial {fn}}}{{\partial q}}"
        ),
        slots={"ham": S(_HAM_POOL), "fn": _FN_SLOT},
    ),
    Template(
        name="lorentz_force",
        latex=r"{ff} = {qq}\!\left({ef} + \mathbf{{v}} \times {bf}\right)",
        slots={
            "ff": S(_VEC_F),
            "qq": S(_CHARGE_POOL),
            "ef": S(_E_FIELD),
            "bf": S(_B_FIELD),
        },
    ),
    # ── Part B: Classical mechanics ─────────────────────────────────────────
    Template(
        name="hamiltons_eom_q",
        latex=r"\dot{{{qq}}} = \frac{{\partial {ham}}}{{\partial {pp}}}",
        slots={"qq": S(_Q_POOL), "ham": S(_HAM_POOL), "pp": S(_P_POOL)},
    ),
    Template(
        name="hamiltons_eom_p",
        latex=r"\dot{{{pp}}} = -\frac{{\partial {ham}}}{{\partial {qq}}}",
        slots={"qq": S(_Q_POOL), "ham": S(_HAM_POOL), "pp": S(_P_POOL)},
    ),
    Template(
        name="conservation_energy",
        latex=r"{ee} = \tfrac{{1}}{{2}}\,{mm}\,\dot{{{qq}}}^2 + V({qq}) = \mathrm{{const}}",
        slots={"ee": S(_ENERGY_POOL), "mm": S(_MASS_POOL), "qq": S(_Q_POOL)},
    ),
    Template(
        name="work_energy_theorem",
        latex=r"W = \Delta K = \tfrac{{1}}{{2}}\,{mm}\,{vv}_f^2 - \tfrac{{1}}{{2}}\,{mm}\,{vv}_i^2",
        slots={"mm": S(_MASS_POOL), "vv": S(tuple(v for v in _VEL_POOL if "_" not in v))},
    ),
    Template(
        name="sho_equation",
        latex=r"\ddot{{{cc}}} + {om}^2\,{cc} = 0",
        slots={"cc": S(_COORD_POOL), "om": S(_OMEGA_POOL)},
    ),
    Template(
        name="sho_solution",
        latex=r"{cc}(t) = A\cos\!\bigl({om}\,t + \varphi\bigr)",
        slots={"cc": S(_COORD_POOL), "om": S(_OMEGA_POOL)},
    ),
    Template(
        name="damped_oscillator",
        latex=r"\ddot{{{cc}}} + 2\,{gm}\,\dot{{{cc}}} + {om}^2\,{cc} = 0",
        slots={"cc": S(_COORD_POOL), "gm": S(_GAMMA_POOL), "om": S(_OMEGA_POOL)},
    ),
    Template(
        name="gravitational_potential",
        latex=r"V(r) = -\frac{{G\,{m1}\,{m2}}}{{r}}",
        slots={"m1": S(_MASS_POOL), "m2": X(_MASS_POOL, ("m1",))},
    ),
    Template(
        name="keplers_third_law",
        latex=r"T^2 = \frac{{4\pi^2\,a^3}}{{G\,{mm}}}",
        slots={"mm": S(_MASS_POOL)},
    ),
    Template(
        name="centripetal_acceleration",
        latex=r"a_c = \frac{{{vv}^2}}{{r}} = {om}^2\,r",
        slots={"vv": S(_VEL_POOL), "om": S(_OMEGA_POOL)},
    ),
    Template(
        name="moment_of_inertia_sum",
        latex=r"I = \sum{lim_mod}_{{{ii}}} {mm}_{{{ii}}}\,r_{{{ii}}}^2",
        slots={
            "lim_mod": _LIM_MOD,
            "ii": S(_IDX_POOL),
            "mm": S(tuple(v for v in _MASS_POOL if "_" not in v)),
        },
    ),
    Template(
        name="torque_cross_product",
        latex=r"\boldsymbol{{\tau}} = \mathbf{{r}} \times {ff}",
        slots={"ff": S(_VEC_F)},
    ),
    # ── Part B: Special relativity ───────────────────────────────────────────
    Template(
        name="lorentz_factor_def",
        latex=r"{gm} = \frac{{1}}{{\sqrt{{1-{vv}^2/c^2}}}}",
        slots={"gm": S(_GAMMA_POOL), "vv": S(_VEL_POOL)},
    ),
    Template(
        name="time_dilation_sr",
        latex=r"\Delta t' = {gm}\,\Delta t",
        slots={"gm": S(_GAMMA_POOL)},
    ),
    Template(
        name="length_contraction_sr",
        latex=r"L' = \frac{{L}}{{{gm}}}",
        slots={"gm": S(_GAMMA_POOL)},
    ),
    Template(
        name="relativistic_momentum_sr",
        latex=r"\mathbf{{p}} = {gm}\,{mm}\,{vv}",
        slots={"gm": S(_GAMMA_POOL), "mm": S(_MASS_POOL), "vv": S(_VEL_POOL)},
    ),
    Template(
        name="relativistic_energy_sr",
        latex=r"{ee} = {gm}\,{mm}\,c^2",
        slots={"ee": S(_ENERGY_POOL), "gm": S(_GAMMA_POOL), "mm": S(_MASS_POOL)},
    ),
    Template(
        name="minkowski_metric",
        latex=r"ds^2 = -c^2\,dt^2 + dx^2 + dy^2 + dz^2",
        slots={},
    ),
    Template(
        name="relativistic_velocity_addition",
        latex=r"{vv} = \frac{{{v1}+{v2}}}{{1+{v1}\,{v2}/c^2}}",
        slots={"vv": S(_VEL_POOL), "v1": S(_VEL_POOL), "v2": X(_VEL_POOL, ("v1",))},
    ),
    # ── Part B: Waves and optics ─────────────────────────────────────────────
    Template(
        name="wave_equation_1d",
        latex=(
            r"\frac{{\partial^2 {fn}}}{{\partial t^2}}"
            r" = {vv}^2\,\frac{{\partial^2 {fn}}}{{\partial x^2}}"
        ),
        slots={"fn": _FN_SLOT, "vv": S(_VEL_POOL)},
    ),
    Template(
        name="speed_of_light_formula",
        latex=r"c = \frac{{1}}{{\sqrt{{\mu_0\,{eps}}}}}",
        slots={"eps": S(_EPS_POOL)},
    ),
    Template(
        name="snells_law_optics",
        latex=(
            r"n_{{{i1}}}\sin\theta_{{{i1}}}"
            r" = n_{{{i2}}}\sin\theta_{{{i2}}}"
        ),
        slots={"i1": S(_IDX_POOL), "i2": X(_IDX_POOL, ("i1",))},
    ),
    Template(
        name="de_broglie_wavelength",
        latex=r"{lm} = \frac{{h}}{{{pp}}}",
        slots={"lm": S(_LAM_POOL), "pp": S(_P_POOL)},
    ),
    Template(
        name="planck_relation",
        latex=r"{ee} = {hb}\,{om}",
        slots={"ee": S(_ENERGY_POOL), "hb": S(_HBAR_POOL), "om": S(_OMEGA_POOL)},
    ),
    # ── Appendix: SR with \upsilon notation ──────────────────────────────────
    Template(
        name="normalized_velocity",
        latex=r"\upsilon \equiv \frac{{{vv}}}{{c}}, \quad 0 \leq \upsilon < 1",
        slots={"vv": S(_VEL_POOL)},
    ),
    Template(
        name="lorentz_factor_upsilon",
        latex=r"\gamma = \frac{{1}}{{\sqrt{{1 - \upsilon^2}}}}, \quad \upsilon = {vv}/c",
        slots={"vv": S(_VEL_POOL)},
    ),
    # ── Appendix: Vector notation with \imath / \jmath ───────────────────────
    Template(
        name="position_vector_imath_jmath",
        latex=r"\mathbf{{r}} = x\,\imath + y\,\jmath + z\,\hat{{k}}",
        slots={},
    ),
    Template(
        name="quaternion_units_imath_jmath",
        latex=r"\imath^2 = \jmath^2 = -1,\quad \imath\,\jmath = -\jmath\,\imath",
        slots={},
    ),
    # ── Appendix: Small-angle approximation ──────────────────────────────────
    Template(
        name="small_angle_approx",
        latex=r"\sin {coord} \approx {coord} \quad ({coord} \ll 1)",
        slots={"coord": S(_COORD_POOL)},
    ),
    # ── Part C: Function-pair templates (classical/SR context) ───────────────
    Template(
        name="fn_lorentz_factor_pair",
        latex=r"{fn1}({gm}) = {fn2}\!\left(\frac{{1}}{{\sqrt{{1-{vv}^2/c^2}}}}\right)",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "gm": S(_GAMMA_POOL),
            "vv": S(_VEL_POOL),
        },
    ),
    Template(
        name="fn_heat_equation",
        latex=(
            r"\frac{{\partial {fn1}}}{{\partial t}}"
            r" = {fn2}\,\frac{{\partial^2 {fn1}}}{{\partial {cc}^2}}"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "cc": S(_COORD_POOL),
        },
    ),
]

GENERATORS, WEIGHTS, TEMPLATES = register_domain("classical_mechanics", _TEMPLATES)
