"""Statistical mechanics domain — Boltzmann, ensembles, thermodynamics."""

from __future__ import annotations

from ..engine._template_dsl import _FN_SLOT, _LIM_MOD, S, Template
from ._config import register_domain
from ._physics_vocab import (
    _BETA_POOL,
    _ENERGY_POOL,
    _FUG_POOL,
    _IDX_POOL,
    _KB_POOL,
    _MASS_POOL,
    _MU_POOL,
    _N_POOL,
    _OMEGA_POOL,
    _PRESS_POOL,
    _SIGMA_POOL,
    _TC_POOL,
    _TEMP_POOL,
    _TH_POOL,
    _VOL_POOL,
)

_TEMPLATES: list[Template] = [
    # ── Part A: Partition function / Stefan-Boltzmann ────────────────────────
    Template(
        name="partition_function",
        latex=r"Z = \sum{lim_mod}_{{{ii}}} e^{{-{bt}\,E_{{{ii}}}}}",
        slots={"lim_mod": _LIM_MOD, "bt": S(_BETA_POOL), "ii": S(_IDX_POOL)},
    ),
    Template(
        name="stefan_boltzmann",
        latex=r"P = {sg}\,A\,{temp}^4",
        slots={"sg": S(_SIGMA_POOL), "temp": S(_TEMP_POOL)},
    ),
    # ── Part B: Statistical mechanics ───────────────────────────────────────
    Template(
        name="boltzmann_distribution",
        latex=r"p_{{{ii}}} = \frac{{e^{{-{bt}\,E_{{{ii}}}}}}}{{Z}}",
        slots={"ii": S(_IDX_POOL), "bt": S(_BETA_POOL)},
    ),
    Template(
        name="helmholtz_free_energy",
        latex=r"F = -{kb}\,{temp}\ln Z",
        slots={"kb": S(_KB_POOL), "temp": S(_TEMP_POOL)},
    ),
    Template(
        name="entropy_statistical",
        latex=r"S = -{kb}\sum{lim_mod}_{{{ii}}} p_{{{ii}}}\ln p_{{{ii}}}",
        slots={"lim_mod": _LIM_MOD, "kb": S(_KB_POOL), "ii": S(_IDX_POOL)},
    ),
    Template(
        name="average_energy",
        latex=r"\langle {ee} \rangle = -\frac{{\partial \ln Z}}{{\partial {bt}}}",
        slots={"ee": S(_ENERGY_POOL), "bt": S(_BETA_POOL)},
    ),
    Template(
        name="heat_capacity_cv",
        latex=(
            r"C_V = {kb}\,{bt}^2\,"
            r"\bigl\langle (E - \langle E \rangle)^2 \bigr\rangle"
        ),
        slots={"kb": S(_KB_POOL), "bt": S(_BETA_POOL)},
    ),
    Template(
        name="equipartition_theorem",
        latex=r"\langle E \rangle = \frac{{f}}{{2}}\,k_B\,{temp}",
        slots={"temp": S(_TEMP_POOL)},
    ),
    Template(
        name="fermi_dirac_distribution",
        latex=r"f({ee}) = \frac{{1}}{{e^{{{bt}({ee}-{mu})}}+1}}",
        slots={"bt": S(_BETA_POOL), "ee": S(_ENERGY_POOL), "mu": S(_MU_POOL)},
    ),
    Template(
        name="bose_einstein_distribution",
        latex=r"n({ee}) = \frac{{1}}{{e^{{{bt}({ee}-{mu})}}-1}}",
        slots={"bt": S(_BETA_POOL), "ee": S(_ENERGY_POOL), "mu": S(_MU_POOL)},
    ),
    Template(
        name="entropy_from_partition",
        latex=r"S = {kb}\bigl(\ln Z + {bt}\,\langle E\rangle\bigr)",
        slots={"kb": S(_KB_POOL), "bt": S(_BETA_POOL)},
    ),
    # ── Part B: Thermodynamics ───────────────────────────────────────────────
    Template(
        name="first_law_thermo",
        latex=r"dU = \delta Q - \delta W",
        slots={},
    ),
    Template(
        name="ideal_gas_law",
        latex=r"{pp}\,V = {nn}\,R\,{temp}",
        slots={"pp": S(_PRESS_POOL), "nn": S(_N_POOL), "temp": S(_TEMP_POOL)},
    ),
    Template(
        name="van_der_waals_eq",
        latex=r"\left({pp} + \frac{{a}}{{V^2}}\right)(V-b) = R\,{temp}",
        slots={"pp": S(_PRESS_POOL), "temp": S(_TEMP_POOL)},
    ),
    Template(
        name="carnot_efficiency",
        latex=r"\eta = 1 - \frac{{{tc}}}{{{th}}}",
        slots={"tc": S(_TC_POOL), "th": S(_TH_POOL)},
    ),
    Template(
        name="gibbs_free_energy_def",
        latex=r"G = H - {temp}\,S",
        slots={"temp": S(_TEMP_POOL)},
    ),
    Template(
        name="maxwell_relation_thermo",
        latex=(
            r"\left(\frac{{\partial T}}{{\partial V}}\right)_S"
            r" = -\left(\frac{{\partial {pp}}}{{\partial S}}\right)_V"
        ),
        slots={"pp": S(_PRESS_POOL)},
    ),
    # ── Appendix: Grand canonical / \Xi notation ────────────────────────────
    Template(
        name="thermal_speed",
        latex=r"\upsilon_{{\mathrm{{th}}}} = \sqrt{{\frac{{2 {kb} {tt}}}{{{mm}}}}}",
        slots={"kb": S(_KB_POOL), "tt": S(_TEMP_POOL), "mm": S(_MASS_POOL)},
    ),
    Template(
        name="grand_canonical_partition",
        latex=(
            r"\Xi(\mu, {vv}, {tt}) = "
            r"\sum{lim_mod}_{{N=0}}^{{\infty}} {ff}^N Z_N({vv}, {tt})"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "vv": S(_VOL_POOL),
            "tt": S(_TEMP_POOL),
            "ff": S(_FUG_POOL),
        },
    ),
    Template(
        name="grand_potential_from_xi",
        latex=r"\Omega = -{kb} {tt} \ln \Xi",
        slots={"kb": S(_KB_POOL), "tt": S(_TEMP_POOL)},
    ),
    Template(
        name="mean_particle_number_xi",
        latex=r"\langle N \rangle = {kb} {tt} \frac{{\partial \ln \Xi}}{{\partial \mu}}\bigg|_{{T,V}}",
        slots={"kb": S(_KB_POOL), "tt": S(_TEMP_POOL)},
    ),
    # ── Part C: Function-pair (statistical mechanics context) ────────────────
    Template(
        name="fn_boltzmann_weight",
        latex=r"{fn1}\!\left(e^{{-{bt}\,E_{{{ii}}}}}\right) = {fn2}\!\left(\frac{{1}}{{Z}}\right)",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "bt": S(_BETA_POOL),
            "ii": S(_IDX_POOL),
        },
    ),
    Template(
        name="fn_canonical_ensemble",
        latex=r"{fn1}(Z) = {fn2}\!\left(\sum{lim_mod}_{{{ii}}} e^{{-{bt}\,E_{{{ii}}}}}\right)",
        slots={
            "lim_mod": _LIM_MOD,
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "bt": S(_BETA_POOL),
            "ii": S(_IDX_POOL),
        },
    ),
    # ── Appendix: High-temperature approximation ─────────────────────────────
    Template(
        name="classical_partition_high_t",
        latex=r"Z \approx \left(\frac{{k_B T}}{{\hbar\,{om}}}\right)^{{{nn}}} \quad (k_B T \gg \hbar\,{om})",
        slots={"om": S(_OMEGA_POOL), "nn": S(_N_POOL)},
    ),
]

GENERATORS, WEIGHTS, TEMPLATES = register_domain("statistical_mechanics", _TEMPLATES)
