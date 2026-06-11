"""Physics domain generator — radically expanded for OCR pretraining diversity."""

from __future__ import annotations

from .._template_dsl import _FN_SLOT, _LIM_MOD, S, Template, X
from ._config import register_domain

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

# Notation-style pools (same quantity, visually distinct typesetting)
_HAM_POOL = ("H", r"\hat{H}", r"\mathcal{H}", r"\hat{\mathcal{H}}", r"H_0", r"\tilde{H}")
_LAG_POOL = ("L", r"\mathcal{L}", r"\Lambda", r"\tilde{L}", r"L_0", r"\hat{L}")
_PSI_POOL = (r"\psi", r"\phi", r"\Psi", r"\varphi", r"\tilde{\psi}", r"\hat{\psi}")
_E_FIELD = (r"\mathbf{E}", r"\vec{E}", "E", r"\hat{E}", r"\mathcal{E}", r"\tilde{E}")
_B_FIELD = (r"\mathbf{B}", r"\vec{B}", "B", r"\hat{B}", r"\mathcal{B}", r"\tilde{B}")
_VEC_F = (r"\mathbf{F}", r"\vec{F}", "F", r"\hat{F}", r"\tilde{F}", r"\mathcal{F}")

# Physical quantity pools
_MASS_POOL = ("m", "M", r"\mu", r"m_1", r"m_2", r"\tilde{m}", r"m_0", r"M_0")
_CHARGE_POOL = ("q", "e", "Q", r"q_1", r"q_2")
_HBAR_POOL = (r"\hbar", r"h/(2\pi)", r"h", r"\hslash")
_EPS_POOL = (r"\varepsilon_0", r"\epsilon_0", r"\varepsilon", r"\epsilon_r\varepsilon_0", r"\varepsilon_{\mathrm{eff}}")
_MU0_POOL = (r"\mu_0", r"\mu_{\mathrm{vac}}", r"\mu", r"\mu_r\mu_0", r"\mu_{\mathrm{eff}}")
_BETA_POOL = (r"\beta", r"(k_B T)^{-1}", r"\beta_0", r"\tilde{\beta}", r"1/(k T)")
_OMEGA_POOL = (r"\omega", r"\omega_0", r"\Omega", r"\nu")
_GAMMA_POOL = (r"\gamma", r"\Gamma", r"\gamma_0")
_TEMP_POOL = ("T", r"T_0", r"T_H", r"T_C")
_COORD_POOL = ("x", "y", "z", "r", r"\theta", r"\phi")
_Q_POOL = ("q", r"q_i", r"q_k", r"q_j")
_P_POOL = ("p", r"p_i", r"p_k", r"\pi")
_IDX_POOL = ("i", "j", "k", "n", "m")
_SPACETIME_IDX_POOL = (r"\mu", r"\nu", r"\rho", r"\sigma", r"\lambda", r"\kappa")
_N_POOL = ("n", "m", "N", r"n_0")
_LAM_POOL = (r"\lambda", r"\Lambda", r"\mu", r"\kappa")
_SIGMA_POOL = (r"\sigma", r"\Sigma", r"\sigma_0")
_ENERGY_POOL = ("E", r"E_0", r"\mathcal{E}", "U")
_VEL_POOL = ("v", r"v_0", "u", r"v_1")
_RHO_POOL = (r"\rho", r"\rho_0", r"\varrho")
_K_POOL = ("k", r"k_0", r"\kappa", r"k_n")
_KB_POOL = ("k_B", r"k_{\mathrm{B}}", r"k", r"\kappa_B", r"k_{\mathrm{Boltz}}")
_MU_POOL = (r"\mu", r"\mu_0", r"E_F")
_TC_POOL = ("T_C", "T_c", r"T_1", r"\tau_c", r"T_{\mathrm{cold}}")
_TH_POOL = ("T_H", "T_h", r"T_2", r"\tau_h", r"T_{\mathrm{hot}}")
_PRESS_POOL = ("p", "P", r"p_0")

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_PHYSICS_TEMPLATES: list[Template] = [
    # ---- Part A: Reparameterized originals (16) ----------------------------
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
        name="schrodinger_time_dependent",
        latex=r"i\,{hb}\,\frac{{\partial {psi}}}{{\partial t}} = {ham}\,{psi}",
        slots={"hb": S(_HBAR_POOL), "psi": S(_PSI_POOL), "ham": S(_HAM_POOL)},
    ),
    Template(
        name="schrodinger_time_independent",
        latex=r"-\frac{{{hb}^2}}{{2\,{mm}}}\,\nabla^2 {psi} + V\,{psi} = E\,{psi}",
        slots={"hb": S(_HBAR_POOL), "mm": S(_MASS_POOL), "psi": S(_PSI_POOL)},
    ),
    Template(
        name="newtons_second_law",
        latex=r"{ff} = {mm}\,\ddot{{\mathbf{{r}}}}",
        slots={"ff": S(_VEC_F), "mm": S(_MASS_POOL)},
    ),
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
    Template(
        name="energy_momentum_relation",
        latex=r"{ee}^2 = (pc)^2 + ({mm}\,c^2)^2",
        slots={"ee": S(_ENERGY_POOL), "mm": S(_MASS_POOL)},
    ),
    Template(
        name="angular_momentum",
        latex=r"\mathbf{{L}}_{{{ii}}} = \mathbf{{r}}_{{{ii}}} \times \mathbf{{p}}_{{{ii}}}",
        slots={"ii": S(_IDX_POOL)},
    ),
    Template(
        name="partition_function",
        latex=r"Z = \sum{lim_mod}_{{{ii}}} e^{{-{bt}\,E_{{{ii}}}}}",
        slots={"lim_mod": _LIM_MOD, "bt": S(_BETA_POOL), "ii": S(_IDX_POOL)},
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
        name="euler_lagrange_alt",
        latex=(
            r"\frac{{d}}{{dt}}\frac{{\partial {lag}}}{{\partial \dot{{{qq}}}}}"
            r" = \frac{{\partial {lag}}}{{\partial {qq}}}"
        ),
        slots={"lag": S(_LAG_POOL), "qq": S(_Q_POOL)},
    ),
    Template(
        name="stefan_boltzmann",
        latex=r"P = {sg}\,A\,{temp}^4",
        slots={"sg": S(_SIGMA_POOL), "temp": S(_TEMP_POOL)},
    ),
    Template(
        name="heisenberg_uncertainty",
        latex=r"\Delta {cc}\,\Delta {pp} \geq \frac{{{hb}}}{{2}}",
        slots={"cc": S(_COORD_POOL), "pp": S(_P_POOL), "hb": S(_HBAR_POOL)},
    ),
    # ---- Part B: Classical Mechanics (12) ----------------------------------
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
    # ---- Part B: Electromagnetism (10) -------------------------------------
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
    # ---- Part B: Quantum Mechanics (12) ------------------------------------
    Template(
        name="position_momentum_commutator",
        latex=r"[{cc},\,{pp}] = i\,{hb}",
        slots={"cc": S(_COORD_POOL), "pp": S(_P_POOL), "hb": S(_HBAR_POOL)},
    ),
    Template(
        name="angular_momentum_commutator",
        latex=r"[L_i,\,L_j] = i\,{hb}\,\varepsilon_{{ijk}}\,L_k",
        slots={"hb": S(_HBAR_POOL)},
    ),
    Template(
        name="eigenvalue_equation",
        latex=r"{ham}\,{psi} = {ee}\,{psi}",
        slots={"ham": S(_HAM_POOL), "psi": S(_PSI_POOL), "ee": S(_ENERGY_POOL)},
    ),
    Template(
        name="expectation_value_qm",
        latex=(
            r"\langle {psi} | \hat{{A}} | {psi} \rangle"
            r" = \int{lim_mod}_{{-\infty}}^{{\infty}} {psi}^*\!(x)\,{fn}(x)\,{psi}(x)\,dx"
        ),
        slots={"lim_mod": _LIM_MOD, "psi": S(_PSI_POOL), "fn": _FN_SLOT},
    ),
    Template(
        name="time_evolution_state",
        latex=r"|{psi}(t)\rangle = e^{{-i\,{ham}\,t/{hb}}}\,|{psi}(0)\rangle",
        slots={"psi": S(_PSI_POOL), "ham": S(_HAM_POOL), "hb": S(_HBAR_POOL)},
    ),
    Template(
        name="harmonic_oscillator_energy",
        latex=r"E_{{{nn}}} = \Bigl({nn} + \tfrac{{1}}{{2}}\Bigr)\,{hb}\,{om}",
        slots={"nn": S(_N_POOL), "hb": S(_HBAR_POOL), "om": S(_OMEGA_POOL)},
    ),
    Template(
        name="hydrogen_energy_levels",
        latex=r"E_{{{nn}}} = -\frac{{13.6\,\mathrm{{eV}}}}{{{nn}^2}}",
        slots={"nn": S(_N_POOL)},
    ),
    Template(
        name="ladder_op_lowering",
        latex=r"\hat{{a}}\,|{nn}\rangle = \sqrt{{{nn}}}\,|{nn}-1\rangle",
        slots={"nn": S(_N_POOL)},
    ),
    Template(
        name="ladder_op_raising",
        latex=r"\hat{{a}}^\dagger\,|{nn}\rangle = \sqrt{{{nn}+1}}\,|{nn}+1\rangle",
        slots={"nn": S(_N_POOL)},
    ),
    Template(
        name="born_rule",
        latex=r"P(a) = |\langle a \mid {psi}\rangle|^2",
        slots={"psi": S(_PSI_POOL)},
    ),
    Template(
        name="probability_density_qm",
        latex=r"\rho({cc},t) = |{psi}({cc},t)|^2",
        slots={"psi": S(_PSI_POOL), "cc": S(_COORD_POOL)},
    ),
    Template(
        name="path_integral_propagator",
        latex=(
            r"\langle x'|e^{{-i\,{ham}\,t/{hb}}}|x\rangle"
            r" = \int \mathcal{{D}}[q]\,e^{{i\,S[q]/{hb}}}"
        ),
        slots={"ham": S(_HAM_POOL), "hb": S(_HBAR_POOL)},
    ),
    # ---- Part B: Statistical Mechanics (9) ---------------------------------
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
    # ---- Part B: Special Relativity (7) ------------------------------------
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
    # ---- Part B: Thermodynamics (6) ----------------------------------------
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
    # ---- Part B: Waves and Optics (5) --------------------------------------
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
    # ---- Part C: High-n_eff function-pair templates (6) --------------------
    Template(
        name="variational_action",
        latex=(
            r"\delta\int{lim_mod}_{{t_1}}^{{t_2}}"
            r" {fn}\!\left({qq},\,\dot{{{qq}}},\,t\right)\,dt = 0"
        ),
        slots={"lim_mod": _LIM_MOD, "fn": _FN_SLOT, "qq": S(_Q_POOL)},
    ),
    Template(
        name="euler_lagrange_functional",
        latex=(
            r"\frac{{d}}{{dt}}\frac{{\partial {fn}}}{{\partial \dot{{{qq}}}}}"
            r" - \frac{{\partial {fn}}}{{\partial {qq}}} = 0"
        ),
        slots={"fn": _FN_SLOT, "qq": S(_Q_POOL)},
    ),
    Template(
        name="generalized_uncertainty_principle",
        latex=(
            r"\Delta {fn1}\,\Delta {fn2}"
            r" \geq \tfrac{{1}}{{2}}\bigl|\bigl\langle"
            r" [{fn1},\,{fn2}]\bigr\rangle\bigr|"
        ),
        slots={"fn1": _FN_SLOT, "fn2": _FN_SLOT},
    ),
    Template(
        name="operator_commutator_def",
        latex=r"\bigl[{fn1},\,{fn2}\bigr] = {fn1}\,{fn2} - {fn2}\,{fn1}",
        slots={"fn1": _FN_SLOT, "fn2": _FN_SLOT},
    ),
    Template(
        name="greens_function_equation",
        latex=r"{fn1}({cc})\,{fn2}({cc}') = -4\pi\,\delta({cc}-{cc}')",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "cc": S(_COORD_POOL),
        },
    ),
    Template(
        name="propagator_bra_ket",
        latex=(
            r"\bigl\langle {fn1} \bigr|"
            r" e^{{-i\,{ham}\,t/{hb}}}"
            r" \bigl| {fn2} \bigr\rangle"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "ham": S(_HAM_POOL),
            "hb": S(_HBAR_POOL),
        },
    ),
    Template(
        name="action_functional",
        latex=(
            r"S[{fn1}] = \int{lim_mod}_{{t_0}}^{{t_1}}"
            r" {fn2}\!\left({qq},\,\dot{{{qq}}},\,t\right)\,dt"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "qq": S(_Q_POOL),
        },
    ),
    Template(
        name="lagrangian_density_field",
        latex=(
            r"\mathcal{{L}}\bigl({fn1},\,\partial_{{{mu}}}{fn1}\bigr)"
            r" = {fn2}({fn1}) - V({fn1})"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "mu": S(_SPACETIME_IDX_POOL),
        },
    ),
    Template(
        name="noether_current",
        latex=(
            r"j^{{{mu}}} ="
            r" \frac{{\partial \mathcal{{L}}}}{{\partial(\partial_{{{mu}}}{fn1})}}"
            r"\,\delta {fn2}"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "mu": S(_SPACETIME_IDX_POOL),
        },
    ),
    Template(
        name="scattering_amplitude",
        latex=(
            r"\mathcal{{M}} = \langle {fn1} | T | {fn2} \rangle"
            r" = \langle {fn1} | {ham} | {fn2} \rangle"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "ham": S(_HAM_POOL),
        },
    ),
    Template(
        name="field_equation_general",
        latex=(
            r"\partial_{{{mu}}}\frac{{\partial \mathcal{{L}}}}{{\partial(\partial_{{{mu}}}{fn1})}}"
            r" - \frac{{\partial \mathcal{{L}}}}{{\partial {fn2}}} = 0"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "mu": S(_SPACETIME_IDX_POOL),
        },
    ),
]

# Part C additions — 8 more fn-pair templates
_PHYSICS_TEMPLATES += [
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
        name="fn_dispersion_relation",
        latex=r"{fn1}({om}) = {fn2}({kk}\,c)",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "om": S(_OMEGA_POOL),
            "kk": S(_K_POOL),
        },
    ),
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
    Template(
        name="fn_correlation_fn",
        latex=r"{fn1}(r) = {fn2}\!\left(\langle {psi}(0)\,{psi}(r)\rangle\right)",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "psi": S(_PSI_POOL),
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
    Template(
        name="fn_quantum_evolution",
        latex=(
            r"{fn1}(|{psi}(t)\rangle)"
            r" = {fn2}\!\left(e^{{-i\,{ham}\,t/{hb}}}\,|{psi}(0)\rangle\right)"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "psi": S(_PSI_POOL),
            "ham": S(_HAM_POOL),
            "hb": S(_HBAR_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Appendix — symbols \upsilon, \Xi, \Upsilon (needed for full Greek coverage)
# ---------------------------------------------------------------------------

_VOL_POOL = ("V", r"V_0", r"\mathcal{V}", "L")
_FUG_POOL = ("z", r"e^{{\beta \mu}}", r"\lambda")

_PHYSICS_TEMPLATES += [
    # \upsilon — thermal/normalized velocity
    Template(
        name="thermal_speed",
        latex=r"\upsilon_{{\mathrm{{th}}}} = \sqrt{{\frac{{2 {kb} {tt}}}{{{mm}}}}}",
        slots={"kb": S(_KB_POOL), "tt": S(_TEMP_POOL), "mm": S(_MASS_POOL)},
    ),
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
    # \Xi — grand canonical partition function
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
    # \Upsilon — Upsilon meson and topological context
    Template(
        name="upsilon_meson_mass",
        latex=r"m_\Upsilon \approx 9.460\,\frac{{\mathrm{{GeV}}}}{{c^2}}",
        slots={},
    ),
    Template(
        name="upsilon_leptonic_width",
        latex=(
            r"\Gamma(\Upsilon \to \ell^+ \ell^-)"
            r" = \frac{{16\pi \alpha^2 e_b^2}}{{3\, {mm}^2}} |\psi(0)|^2"
        ),
        slots={"mm": S(_MASS_POOL)},
    ),
]

_DAG_OP_POOL = ("A", "B", "H", r"\hat{H}", "U", "O", r"\hat{O}")
_DAG_STATE_POOL = ("n", "m", "k", r"\psi", r"\phi", r"\alpha")

_PHYSICS_TEMPLATES += [
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
        name="position_vector_imath_jmath",
        latex=r"\mathbf{{r}} = x\,\imath + y\,\jmath + z\,\hat{{k}}",
        slots={},
    ),
    Template(
        name="quaternion_units_imath_jmath",
        latex=r"\imath^2 = \jmath^2 = -1,\quad \imath\,\jmath = -\jmath\,\imath",
        slots={},
    ),
    Template(
        name="adjoint_bra_rule",
        latex=r"\langle {aa} | {op}^{{\dagger}} = (\langle {aa} | {op})^*",
        slots={"aa": S(_DAG_STATE_POOL), "op": S(_DAG_OP_POOL)},
    ),
    Template(
        name="adjoint_anti_multiplicativity",
        latex=r"({op1}\,{op2})^{{\dagger}} = {op2}^{{\dagger}}\,{op1}^{{\dagger}}",
        slots={"op1": S(_DAG_OP_POOL), "op2": X(_DAG_OP_POOL, ("op1",))},
    ),
    Template(
        name="adjoint_matrix_element",
        latex=(
            r"\langle {aa} | {op}^{{\dagger}} | {bb} \rangle"
            r" = \langle {bb} | {op} | {aa} \rangle^*"
        ),
        slots={
            "aa": S(_DAG_STATE_POOL),
            "bb": X(_DAG_STATE_POOL, ("aa",)),
            "op": S(_DAG_OP_POOL),
        },
    ),
    Template(
        name="dirac_adjoint_derivative",
        latex=r"\bar{{\psi}}\,\overleftarrow{{\partial}}_{{\mu}} = -\partial_{{\mu}}\bar{{\psi}}",
        slots={},
    ),
    Template(
        name="conserved_current_lr_arrows",
        latex=(
            r"j^{{\mu}} = \bar{{\psi}}\,\gamma^{{\mu}}\,\psi,"
            r"\quad j^{{\mu}} = \bar{{\psi}}\,\overrightarrow{{\partial}}^{{\mu}}\psi"
            r" - \bar{{\psi}}\,\overleftarrow{{\partial}}^{{\mu}}\psi"
        ),
        slots={},
    ),
]

_PART_APPROX: list[Template] = [
    Template(
        name="wkb_wavefunction",
        latex=r"{psi}({qq}) \approx \frac{{C}}{{\sqrt{{p({qq})}}}}\exp\!\left(\frac{{i}}{{\hbar}}\int^{{{qq}}} p(q')\,dq'\right)",
        slots={"psi": S(_PSI_POOL), "qq": S(_Q_POOL)},
    ),
    Template(
        name="perturbation_energy_first_order",
        latex=r"E_n \approx E_n^{{(0)}} + \langle n^{{(0)}} | {ham}' | n^{{(0)}} \rangle",
        slots={"ham": S(_HAM_POOL)},
    ),
    Template(
        name="small_angle_approx",
        latex=r"\sin {coord} \approx {coord} \quad ({coord} \ll 1)",
        slots={"coord": S(_COORD_POOL)},
    ),
    Template(
        name="classical_partition_high_t",
        latex=r"Z \approx \left(\frac{{k_B T}}{{\hbar\,{om}}}\right)^{{{nn}}} \quad (k_B T \gg \hbar\,{om})",
        slots={"om": S(_OMEGA_POOL), "nn": S(_N_POOL)},
    ),
]

_PHYSICS_TEMPLATES += _PART_APPROX

_PHYSICS_TEMPLATES += [
    Template(
        name="wavefunction_real_part",
        latex=r"\Re({psi}(\mathbf{{r}},t)) = A\cos(\mathbf{{k}}\cdot\mathbf{{r}} - \omega t)",
        slots={"psi": S(_PSI_POOL)},
    ),
    Template(
        name="wavefunction_imaginary_part",
        latex=r"\Im({psi}(\mathbf{{r}},t)) = A\sin(\mathbf{{k}}\cdot\mathbf{{r}} - \omega t)",
        slots={"psi": S(_PSI_POOL)},
    ),
    Template(
        name="complex_impedance_re_im",
        latex=r"Z = \Re(Z) + i\,\Im(Z)",
        slots={},
    ),
]

# ---- Tensor/index notation additions ----------------------------------------
_PHYSICS_TEMPLATES += [
    # -- Electromagnetic field tensor --
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
    # -- 4-vector notation --
    Template(
        name="four_momentum_def",
        latex=r"p^{{{mu}}} = ({ee}/c,\, \mathbf{{p}})",
        slots={
            "mu": S(_SPACETIME_IDX_POOL),
            "ee": S(_ENERGY_POOL),
        },
    ),
    Template(
        name="four_vector_norm",
        latex=r"p_{{{mu}}} p^{{{mu}}} = -{mm}^2 c^2",
        slots={
            "mu": S(_SPACETIME_IDX_POOL),
            "mm": S(_MASS_POOL),
        },
    ),
    Template(
        name="minkowski_metric_idx",
        latex=r"ds^2 = \eta_{{{mu}{nu}}} dx^{{{mu}}} dx^{{{nu}}}",
        slots={
            "mu": S(_SPACETIME_IDX_POOL),
            "nu": X(_SPACETIME_IDX_POOL, ("mu",)),
        },
    ),
    Template(
        name="stress_energy_conservation",
        latex=r"\partial_{{{mu}}} T^{{{mu}{nu}}} = 0",
        slots={
            "mu": S(_SPACETIME_IDX_POOL),
            "nu": X(_SPACETIME_IDX_POOL, ("mu",)),
        },
    ),
    # -- Levi-Civita symbol (parameterized) --
    Template(
        name="levi_civita_3d_cross",
        latex=(
            r"(\mathbf{{A}}\times\mathbf{{B}})^{{{ii}}}"
            r" = \varepsilon^{{{ii}{jj}{kk}}} A_{{{jj}}} B_{{{kk}}}"
        ),
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
        },
    ),
    Template(
        name="levi_civita_3d_curl",
        latex=(
            r"(\nabla\times\mathbf{{A}})^{{{ii}}}"
            r" = \varepsilon^{{{ii}{jj}{kk}}} \partial_{{{jj}}} A_{{{kk}}}"
        ),
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
        },
    ),
    Template(
        name="levi_civita_contraction",
        latex=(
            r"\varepsilon_{{{ii}{jj}{kk}}} \varepsilon^{{{ii}{ll}{mm}}}"
            r" = \delta^{{{ll}}}_{{{jj}}} \delta^{{{mm}}}_{{{kk}}}"
            r" - \delta^{{{ll}}}_{{{kk}}} \delta^{{{mm}}}_{{{jj}}}"
        ),
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
            "ll": X(_IDX_POOL, ("ii", "jj", "kk")),
            "mm": X(_IDX_POOL, ("ii", "jj", "kk", "ll")),
        },
    ),
    Template(
        name="levi_civita_4d_dual",
        latex=(
            r"\tilde{{F}}^{{{mu}{nu}}}"
            r" = \tfrac{{1}}{{2}} \varepsilon^{{{mu}{nu}{rho}{sig}}} F_{{{rho}{sig}}}"
        ),
        slots={
            "mu": S(_SPACETIME_IDX_POOL),
            "nu": X(_SPACETIME_IDX_POOL, ("mu",)),
            "rho": X(_SPACETIME_IDX_POOL, ("mu", "nu")),
            "sig": X(_SPACETIME_IDX_POOL, ("mu", "nu", "rho")),
        },
    ),
    # -- Kronecker delta as tensor --
    Template(
        name="kronecker_mixed",
        latex=r"\delta^{{{ii}}}_{{{jj}}}",
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
    Template(
        name="kronecker_trace",
        latex=r"\delta^{{{ii}}}_{{{ii}}} = n",
        slots={"ii": S(_IDX_POOL)},
    ),
    Template(
        name="kronecker_contraction",
        latex=r"\delta^{{{ii}}}_{{{jj}}} T^{{{jj}}} = T^{{{ii}}}",
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
]

# ---------------------------------------------------------------------------
# Anticommutators and non-Poisson brackets
# ---------------------------------------------------------------------------

_FERM_IDX_POOL = ("i", "j", "k", "l", "m")

_PART_ANTICOMMUTATOR: list[Template] = [
    Template(
        name="anticommutator_def",
        latex=r"\{{{A},\,{B}\}} = {A}{B} + {B}{A}",
        slots={"A": S(_DAG_OP_POOL), "B": X(_DAG_OP_POOL, ("A",))},
    ),
    Template(
        name="canonical_anticommutation_relation",
        latex=r"\{{a_{{{ii}}},\,a_{{{jj}}}^\dagger\}} = \delta_{{{ii}{jj}}}",
        slots={"ii": S(_FERM_IDX_POOL), "jj": X(_FERM_IDX_POOL, ("ii",))},
    ),
    Template(
        name="anticommutation_annihilators",
        latex=r"\{{a_{{{ii}}},\,a_{{{jj}}}\}} = 0",
        slots={"ii": S(_FERM_IDX_POOL), "jj": X(_FERM_IDX_POOL, ("ii",))},
    ),
    Template(
        name="fermionic_number_operator",
        latex=r"\hat{{n}}_{{{ii}}} = a_{{{ii}}}^\dagger a_{{{ii}}},\quad \hat{{n}}_{{{ii}}}^2 = \hat{{n}}_{{{ii}}}",
        slots={"ii": S(_FERM_IDX_POOL)},
    ),
    Template(
        name="dirac_bracket",
        latex=r"\{f,\,g\}_D = \{f,\,g\} - \{f,\,\phi_a\}\,C^{ab}\,\{\phi_b,\,g\}",
        slots={},
    ),
    Template(
        name="moyal_bracket",
        latex=r"\{{f,\,g\}}_\star = \tfrac{{1}}{{i{hb}}}(f \star g - g \star f)",
        slots={"hb": S(_HBAR_POOL)},
    ),
]

_PHYSICS_TEMPLATES += _PART_ANTICOMMUTATOR


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("physics", _PHYSICS_TEMPLATES)
