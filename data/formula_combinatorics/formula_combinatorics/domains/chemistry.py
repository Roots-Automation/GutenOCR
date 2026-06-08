"""Chemistry domain: chemical equations, equilibria, kinetics, thermochemistry,
acid/base, and electrochemistry formula templates."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import S, Template, X, compute_weights, make_dispatcher

# ---------------------------------------------------------------------------
# Symbol pools
# ---------------------------------------------------------------------------

_COEFF_POOL = ("1", "2", "3", "4")

# \text{}-wrapped species for use in standard LaTeX math mode
_SPECIES_POOL = (
    r"\text{H}_2",
    r"\text{O}_2",
    r"\text{N}_2",
    r"\text{CO}_2",
    r"\text{H}_2\text{O}",
    r"\text{NH}_3",
    r"\text{CH}_4",
    r"\text{HCl}",
    r"\text{NaOH}",
    r"\text{SO}_2",
    r"\text{NO}",
    r"\text{Cl}_2",
    r"\text{Fe}_2\text{O}_3",
    r"\text{CaCO}_3",
    r"\text{C}_6\text{H}_{12}\text{O}_6",
)

# Plain names for use exclusively inside \ce{...} (mhchem handles formatting)
_CE_SPECIES_POOL = (
    "H2",
    "O2",
    "N2",
    "CO2",
    "H2O",
    "NH3",
    "CH4",
    "HCl",
    "NaOH",
    "SO2",
    "NO",
    "Cl2",
    "Fe2O3",
    "CaCO3",
)

_ION_POOL = (
    r"\text{Na}^+",
    r"\text{K}^+",
    r"\text{Ca}^{2+}",
    r"\text{Mg}^{2+}",
    r"\text{Fe}^{2+}",
    r"\text{Fe}^{3+}",
    r"\text{Al}^{3+}",
    r"\text{Cl}^-",
    r"\text{OH}^-",
    r"\text{SO}_4^{2-}",
    r"\text{NO}_3^-",
    r"\text{CO}_3^{2-}",
    r"\text{PO}_4^{3-}",
    r"\text{NH}_4^+",
    r"\text{H}^+",
)

_STATE_POOL = (r"(s)", r"(l)", r"(g)", r"(aq)")

_KEQV_POOL = ("K_{eq}", "K_c", "K_p", "K_a", "K_b", "K_w", "K_{sp}", "K_f")

_KRATE_POOL = ("k", "k_1", "k_{-1}", "k_f", "k_r", "k_2")

_ORDER_POOL = ("m", "n", "1", "2", r"\alpha", r"\beta")

_TEMP_POOL = ("T", "T_0", "T_1", "T_2", r"\theta")

_EA_POOL = ("E_a", "E_A", r"\Delta E^\ddagger", r"E^\ddagger")

_AFREQ_POOL = ("A", "A_0", r"\nu_0", "Z_{AB}")

_DH_POOL = (
    r"\Delta H",
    r"\Delta H^\circ",
    r"\Delta H_{rxn}",
    r"\Delta H_f^\circ",
    r"\Delta H_{comb}",
)

_DG_POOL = (
    r"\Delta G",
    r"\Delta G^\circ",
    r"\Delta G_{rxn}",
    r"\Delta G_f^\circ",
)

_DS_POOL = (
    r"\Delta S",
    r"\Delta S^\circ",
    r"\Delta S_{rxn}",
    r"\Delta S_{mix}",
)

_R_POOL = ("R", "R_0", "R_u", r"\mathcal{R}")

_PH_POOL = (r"\text{pH}", r"\text{pOH}", r"\text{p}K_a", r"\text{p}K_b")

_CONC_SPECIES_POOL = (
    r"\text{H}^+",
    r"\text{OH}^-",
    r"\text{HA}",
    r"\text{A}^-",
    r"\text{B}",
    r"\text{BH}^+",
    r"\text{H}_3\text{O}^+",
)

_EPOT_POOL = ("E", r"E^\circ", "E_{cell}", r"E^\circ_{cell}")

_FARAD_POOL = ("F", r"\mathcal{F}", "F_0")

_NELECTRON_POOL = ("n", "n_e", "z", "n_{el}")

_Q_POOL = ("Q", "Q_c", "Q_p", "Q_{rxn}")

_N_POOL = ("n", "n_A", "n_B", "m", r"\nu")

_TIME_POOL = ("t", r"t_{1/2}", r"\tau", "t_0", "t_r")

# Italic single-letter labels used inside [...] concentration expressions
_CEQ_POOL = ("A", "B", "C", "D", "P", "Q", "X", "Y")

_STOICH_EXP_POOL = ("a", "b", "c", "d", "m", "n", "p", "q")

_KA_KB_POOL = ("K_a", "K_b", "K_c")

# ---------------------------------------------------------------------------
# Section A — Chemical Reaction Equations (15)
# ---------------------------------------------------------------------------

_CHEM_TEMPLATES: list[Template] = [
    Template(
        name="rxn_irreversible_simple",
        latex=r"{ca}\,{A} \rightarrow {cb}\,{B}",
        slots={
            "ca": S(_COEFF_POOL),
            "A": S(_SPECIES_POOL),
            "cb": S(_COEFF_POOL),
            "B": S(_SPECIES_POOL),
        },
    ),
    Template(
        name="rxn_irreversible_two_plus_one",
        latex=r"{ca}\,{A} + {cb}\,{B} \rightarrow {cc}\,{C}",
        slots={
            "ca": S(_COEFF_POOL),
            "A": S(_SPECIES_POOL),
            "cb": S(_COEFF_POOL),
            "B": S(_SPECIES_POOL),
            "cc": S(_COEFF_POOL),
            "C": S(_SPECIES_POOL),
        },
    ),
    Template(
        name="rxn_irreversible_two_to_two",
        latex=r"{ca}\,{A} + {cb}\,{B} \rightarrow {cc}\,{C} + {cd}\,{D}",
        slots={
            "ca": S(_COEFF_POOL),
            "A": S(_SPECIES_POOL),
            "cb": S(_COEFF_POOL),
            "B": S(_SPECIES_POOL),
            "cc": S(_COEFF_POOL),
            "C": S(_SPECIES_POOL),
            "cd": S(_COEFF_POOL),
            "D": S(_SPECIES_POOL),
        },
    ),
    Template(
        name="rxn_reversible_simple",
        latex=r"{ca}\,{A} \rightleftharpoons {cb}\,{B}",
        slots={
            "ca": S(_COEFF_POOL),
            "A": S(_SPECIES_POOL),
            "cb": S(_COEFF_POOL),
            "B": S(_SPECIES_POOL),
        },
    ),
    Template(
        name="rxn_reversible_two_to_two",
        latex=r"{ca}\,{A} + {cb}\,{B} \rightleftharpoons {cc}\,{C} + {cd}\,{D}",
        slots={
            "ca": S(_COEFF_POOL),
            "A": S(_SPECIES_POOL),
            "cb": S(_COEFF_POOL),
            "B": S(_SPECIES_POOL),
            "cc": S(_COEFF_POOL),
            "C": S(_SPECIES_POOL),
            "cd": S(_COEFF_POOL),
            "D": S(_SPECIES_POOL),
        },
    ),
    Template(
        name="rxn_with_states",
        latex=r"{A}{sA} + {B}{sB} \rightarrow {C}{sC}",
        slots={
            "A": S(_SPECIES_POOL),
            "sA": S(_STATE_POOL),
            "B": S(_SPECIES_POOL),
            "sB": S(_STATE_POOL),
            "C": S(_SPECIES_POOL),
            "sC": S(_STATE_POOL),
        },
    ),
    Template(
        name="rxn_ionic_simple",
        latex=r"{I1} + {I2} \rightarrow {P}{sp}",
        slots={
            "I1": S(_ION_POOL),
            "I2": S(_ION_POOL),
            "P": S(_SPECIES_POOL),
            "sp": S(_STATE_POOL),
        },
    ),
    Template(
        name="rxn_net_ionic",
        latex=r"{I1} + {I2} \rightarrow {I3}(aq)",
        slots={
            "I1": S(_ION_POOL),
            "I2": S(_ION_POOL),
            "I3": S(_ION_POOL),
        },
    ),
    Template(
        name="half_rxn_oxidation",
        latex=r"{Ia} \rightarrow {Ib} + {ne}\,e^{{-}}",
        slots={
            "Ia": S(_ION_POOL),
            "Ib": S(_ION_POOL),
            "ne": S(_COEFF_POOL),
        },
    ),
    Template(
        name="half_rxn_reduction",
        latex=r"{Ia} + {ne}\,e^{{-}} \rightarrow {Ib}",
        slots={
            "Ia": S(_ION_POOL),
            "ne": S(_COEFF_POOL),
            "Ib": S(_ION_POOL),
        },
    ),
    Template(
        name="rxn_combustion_hydrocarbon",
        latex=r"\text{{CH}}_4 + 2\,\text{{O}}_2 \rightarrow \text{{CO}}_2{sC} + 2\,\text{{H}}_2\text{{O}}{sW}",
        slots={
            "sC": S(_STATE_POOL),
            "sW": S(_STATE_POOL),
        },
    ),
    Template(
        name="rxn_decomposition",
        latex=r"{ca}\,{A}{sA} \rightarrow {cb}\,{B}{sB} + {cc}\,{C}{sC}",
        slots={
            "ca": S(_COEFF_POOL),
            "A": S(_SPECIES_POOL),
            "sA": S(_STATE_POOL),
            "cb": S(_COEFF_POOL),
            "B": S(_SPECIES_POOL),
            "sB": S(_STATE_POOL),
            "cc": S(_COEFF_POOL),
            "C": S(_SPECIES_POOL),
            "sC": S(_STATE_POOL),
        },
    ),
    Template(
        name="rxn_synthesis",
        latex=r"{ca}\,{A}{sA} + {cb}\,{B}{sB} \rightarrow {cc}\,{C}{sC}",
        slots={
            "ca": S(_COEFF_POOL),
            "A": S(_SPECIES_POOL),
            "sA": S(_STATE_POOL),
            "cb": S(_COEFF_POOL),
            "B": S(_SPECIES_POOL),
            "sB": S(_STATE_POOL),
            "cc": S(_COEFF_POOL),
            "C": S(_SPECIES_POOL),
            "sC": S(_STATE_POOL),
        },
    ),
    Template(
        name="rxn_precipitation",
        latex=r"{I1} + {I2} \rightarrow {P}(s) + {I3}(aq)",
        slots={
            "I1": S(_ION_POOL),
            "I2": S(_ION_POOL),
            "P": S(_SPECIES_POOL),
            "I3": S(_ION_POOL),
        },
    ),
    Template(
        name="rxn_ce_simple",
        latex=r"\ce{{{ca} {A} -> {cb} {B}}}",
        slots={
            "ca": S(_COEFF_POOL),
            "A": S(_CE_SPECIES_POOL),
            "cb": S(_COEFF_POOL),
            "B": S(_CE_SPECIES_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Section B — Equilibrium and Kinetics (16)
# ---------------------------------------------------------------------------

_CHEM_TEMPLATES += [
    Template(
        name="equilibrium_constant_def",
        latex=(
            r"{Keq} = \frac{{[\text{{C}}]^{{{cc}}}[\text{{D}}]^{{{dd}}}}}"
            r"{{[\text{{A}}]^{{{aa}}}[\text{{B}}]^{{{bb}}}}}"
        ),
        slots={
            "Keq": S(_KEQV_POOL),
            "aa": S(_STOICH_EXP_POOL),
            "bb": S(_STOICH_EXP_POOL),
            "cc": S(_STOICH_EXP_POOL),
            "dd": S(_STOICH_EXP_POOL),
        },
    ),
    Template(
        name="equilibrium_constant_two_two",
        latex=(
            r"{Keq} = \frac{{[{C}]^{{{ce}}}[{D}]^{{{de}}}}}"
            r"{{[{A}]^{{{ae}}}[{B}]^{{{be}}}}}"
        ),
        slots={
            "Keq": S(_KEQV_POOL),
            "A": S(_CEQ_POOL),
            "B": X(_CEQ_POOL, ("A",)),
            "C": X(_CEQ_POOL, ("A", "B")),
            "D": X(_CEQ_POOL, ("A", "B", "C")),
            "ae": S(_STOICH_EXP_POOL),
            "be": S(_STOICH_EXP_POOL),
            "ce": S(_STOICH_EXP_POOL),
            "de": S(_STOICH_EXP_POOL),
        },
    ),
    Template(
        name="keq_two_one",
        latex=(
            r"{Keq} = \frac{{[{C}]^{{{ce}}}}}"
            r"{{[{A}]^{{{ae}}}[{B}]^{{{be}}}}}"
        ),
        slots={
            "Keq": S(_KEQV_POOL),
            "A": S(_CEQ_POOL),
            "B": X(_CEQ_POOL, ("A",)),
            "C": X(_CEQ_POOL, ("A", "B")),
            "ae": S(_STOICH_EXP_POOL),
            "be": S(_STOICH_EXP_POOL),
            "ce": S(_STOICH_EXP_POOL),
        },
    ),
    Template(
        name="keq_one_two",
        latex=(
            r"{Keq} = \frac{{[{A}]^{{{ae}}}[{B}]^{{{be}}}}}"
            r"{{[{C}]^{{{ce}}}}}"
        ),
        slots={
            "Keq": S(_KEQV_POOL),
            "A": S(_CEQ_POOL),
            "B": X(_CEQ_POOL, ("A",)),
            "C": X(_CEQ_POOL, ("A", "B")),
            "ae": S(_STOICH_EXP_POOL),
            "be": S(_STOICH_EXP_POOL),
            "ce": S(_STOICH_EXP_POOL),
        },
    ),
    Template(
        name="kp_from_kc",
        latex=r"K_p = K_c\,({R}\,{T})^{{\Delta n}}",
        slots={
            "R": S(_R_POOL),
            "T": S(_TEMP_POOL),
        },
    ),
    Template(
        name="rate_law_two_species",
        latex=r"r = {kk}\,[{Ab}]^{{{m}}}\,[{Bb}]^{{{n}}}",
        slots={
            "kk": S(_KRATE_POOL),
            "Ab": S(_CEQ_POOL),
            "Bb": X(_CEQ_POOL, ("Ab",)),
            "m": S(_ORDER_POOL),
            "n": S(_ORDER_POOL),
        },
    ),
    Template(
        name="rate_law_one_species",
        latex=r"r = {kk}\,[{Ab}]^{{{m}}}",
        slots={
            "kk": S(_KRATE_POOL),
            "Ab": S(_CEQ_POOL),
            "m": S(_ORDER_POOL),
        },
    ),
    Template(
        name="arrhenius_equation",
        latex=r"{kk} = {Af}\,e^{{-{Ea}/({R}\,{T})}}",
        slots={
            "kk": S(_KRATE_POOL),
            "Af": S(_AFREQ_POOL),
            "Ea": S(_EA_POOL),
            "R": S(_R_POOL),
            "T": S(_TEMP_POOL),
        },
    ),
    Template(
        name="arrhenius_ln_form",
        latex=r"\ln {kk} = \ln {Af} - \frac{{{Ea}}}{{{R}\,{T}}}",
        slots={
            "kk": S(_KRATE_POOL),
            "Af": S(_AFREQ_POOL),
            "Ea": S(_EA_POOL),
            "R": S(_R_POOL),
            "T": S(_TEMP_POOL),
        },
    ),
    Template(
        name="integrated_rate_zero_order",
        latex=r"[{Ab}]_t = [{Ab}]_0 - {kk}\,t",
        slots={
            "Ab": S(_CEQ_POOL),
            "kk": S(_KRATE_POOL),
        },
    ),
    Template(
        name="integrated_rate_first_order",
        latex=r"\ln\frac{{[{Ab}]_t}}{{[{Ab}]_0}} = -{kk}\,t",
        slots={
            "Ab": S(_CEQ_POOL),
            "kk": S(_KRATE_POOL),
        },
    ),
    Template(
        name="integrated_rate_first_order_alt",
        latex=r"[{Ab}]_t = [{Ab}]_0\,e^{{-{kk}\,t}}",
        slots={
            "Ab": S(_CEQ_POOL),
            "kk": S(_KRATE_POOL),
        },
    ),
    Template(
        name="integrated_rate_second_order",
        latex=r"\frac{{1}}{{[{Ab}]_t}} - \frac{{1}}{{[{Ab}]_0}} = {kk}\,t",
        slots={
            "Ab": S(_CEQ_POOL),
            "kk": S(_KRATE_POOL),
        },
    ),
    Template(
        name="half_life_first_order",
        latex=r"t_{{1/2}} = \frac{{\ln 2}}{{{kk}}}",
        slots={"kk": S(_KRATE_POOL)},
    ),
    Template(
        name="half_life_second_order",
        latex=r"t_{{1/2}} = \frac{{1}}{{{kk}\,[{Ab}]_0}}",
        slots={
            "kk": S(_KRATE_POOL),
            "Ab": S(_CEQ_POOL),
        },
    ),
    Template(
        name="reaction_quotient_def",
        latex=(
            r"{QQ} = \frac{{[{C}]^{{{ce}}}[{D}]^{{{de}}}}}"
            r"{{[{A}]^{{{ae}}}[{B}]^{{{be}}}}}"
        ),
        slots={
            "QQ": S(_Q_POOL),
            "A": S(_CEQ_POOL),
            "B": X(_CEQ_POOL, ("A",)),
            "C": X(_CEQ_POOL, ("A", "B")),
            "D": X(_CEQ_POOL, ("A", "B", "C")),
            "ae": S(_STOICH_EXP_POOL),
            "be": S(_STOICH_EXP_POOL),
            "ce": S(_STOICH_EXP_POOL),
            "de": S(_STOICH_EXP_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Section C — Thermochemistry (12)
# ---------------------------------------------------------------------------

_CHEM_TEMPLATES += [
    Template(
        name="gibbs_chemistry",
        latex=r"{DG} = {DH} - {T}\,{DS}",
        slots={
            "DG": S(_DG_POOL),
            "DH": S(_DH_POOL),
            "T": S(_TEMP_POOL),
            "DS": S(_DS_POOL),
        },
    ),
    Template(
        name="gibbs_standard_lnK",
        latex=r"{DG} = -{R}\,{T}\,\ln {Keq}",
        slots={
            "DG": S(_DG_POOL),
            "R": S(_R_POOL),
            "T": S(_TEMP_POOL),
            "Keq": S(_KEQV_POOL),
        },
    ),
    Template(
        name="gibbs_from_Q",
        latex=r"{DG} = {DG0} + {R}\,{T}\,\ln {QQ}",
        slots={
            "DG": S(_DG_POOL),
            "DG0": S(_DG_POOL),
            "R": S(_R_POOL),
            "T": S(_TEMP_POOL),
            "QQ": S(_Q_POOL),
        },
        distinct=[["DG", "DG0"]],
    ),
    Template(
        name="hess_law",
        latex=(
            r"{dh} = \sum \Delta H_f^\circ(\text{{products}})"
            r" - \sum \Delta H_f^\circ(\text{{reactants}})"
        ),
        slots={"dh": S(_DH_POOL)},
    ),
    Template(
        name="enthalpy_of_formation",
        latex=r"{dh} = \Delta H_f^\circ\!\bigl({sp}\bigr)",
        slots={
            "dh": S(_DH_POOL),
            "sp": S(_SPECIES_POOL),
        },
    ),
    Template(
        name="entropy_change_rxn",
        latex=(
            r"{ds} = \sum S^\circ(\text{{products}})"
            r" - \sum S^\circ(\text{{reactants}})"
        ),
        slots={"ds": S(_DS_POOL)},
    ),
    Template(
        name="entropy_of_mixing",
        latex=r"\Delta S_{{mix}} = -{R}\,\sum{lm} x_i \ln x_i",
        slots={
            "R": S(_R_POOL),
            "lm": S(("", r"\limits")),
        },
    ),
    Template(
        name="clausius_clapeyron",
        latex=r"\frac{{d\ln P}}{{dT}} = \frac{{{DH}}}{{{R}\,{T}^2}}",
        slots={
            "DH": S(_DH_POOL),
            "R": S(_R_POOL),
            "T": S(_TEMP_POOL),
        },
    ),
    Template(
        name="clausius_clapeyron_two_point",
        latex=(
            r"\ln\frac{{P_2}}{{P_1}} = -\frac{{{DH}}}{{{R}}}"
            r"\!\left(\frac{{1}}{{{T2}}} - \frac{{1}}{{{T1}}}\right)"
        ),
        slots={
            "DH": S(_DH_POOL),
            "R": S(_R_POOL),
            "T1": S(_TEMP_POOL),
            "T2": X(_TEMP_POOL, ("T1",)),
        },
    ),
    Template(
        name="kirchhoff_law",
        latex=r"{dh2} = {dh1} + \int_{{{T1}}}^{{{T2}}} \Delta C_p\,dT",
        slots={
            "dh1": S(_DH_POOL),
            "dh2": S(_DH_POOL),
            "T1": S(_TEMP_POOL),
            "T2": X(_TEMP_POOL, ("T1",)),
        },
    ),
    Template(
        name="gibbs_helmholtz",
        latex=(
            r"\left(\frac{{\partial ({DG}/T)}}{{\partial T}}\right)_P"
            r" = -\frac{{{DH}}}{{T^2}}"
        ),
        slots={
            "DG": S(_DG_POOL),
            "DH": S(_DH_POOL),
        },
    ),
    Template(
        name="enthalpy_heat_capacity",
        latex=r"{DH} = {nn}\,C_p\,\Delta {T}",
        slots={
            "DH": S(_DH_POOL),
            "nn": S(_N_POOL),
            "T": S(_TEMP_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Section D — Acid/Base and Electrochemistry (12)
# ---------------------------------------------------------------------------

_CHEM_TEMPLATES += [
    Template(
        name="ph_definition",
        latex=r"\text{{pH}} = -\log\!\bigl[{sp}\bigr]",
        slots={"sp": S(_CONC_SPECIES_POOL)},
    ),
    Template(
        name="poh_definition",
        latex=r"{phx} = -\log\!\bigl[{sp}\bigr]",
        slots={
            "phx": S(_PH_POOL),
            "sp": S(_CONC_SPECIES_POOL),
        },
    ),
    Template(
        name="ph_poh_relation",
        latex=r"\text{pH} + \text{pOH} = 14",
        slots={},
    ),
    Template(
        name="ka_definition",
        latex=r"K_a = \frac{{[{Ab}][\text{{H}}^+]}}{{[{HA}]}}",
        slots={
            "Ab": S(_CONC_SPECIES_POOL),
            "HA": X(_CONC_SPECIES_POOL, ("Ab",)),
        },
    ),
    Template(
        name="henderson_hasselbalch",
        latex=r"\text{{pH}} = \text{{p}}K_a + \log\frac{{[{Ab}]}}{{[{HA}]}}",
        slots={
            "Ab": S(_CONC_SPECIES_POOL),
            "HA": X(_CONC_SPECIES_POOL, ("Ab",)),
        },
    ),
    Template(
        name="henderson_hasselbalch_general",
        latex=r"{phx} = \text{{p}}{Keq} + \log\frac{{[{Ab}]}}{{[{HA}]}}",
        slots={
            "phx": S(_PH_POOL),
            "Keq": S(_KA_KB_POOL),
            "Ab": S(_CONC_SPECIES_POOL),
            "HA": X(_CONC_SPECIES_POOL, ("Ab",)),
        },
    ),
    Template(
        name="nernst_equation",
        latex=r"{E} = {E0} - \frac{{{R}\,{T}}}{{{ne}\,{F}}}\ln {QQ}",
        slots={
            "E": S(_EPOT_POOL),
            "E0": X(_EPOT_POOL, ("E",)),
            "R": S(_R_POOL),
            "T": S(_TEMP_POOL),
            "ne": S(_NELECTRON_POOL),
            "F": S(_FARAD_POOL),
            "QQ": S(_Q_POOL),
        },
    ),
    Template(
        name="nernst_at_298",
        latex=r"{E} = {E0} - \frac{{0.0592\,\text{{V}}}}{{{ne}}}\log {QQ}",
        slots={
            "E": S(_EPOT_POOL),
            "E0": X(_EPOT_POOL, ("E",)),
            "ne": S(_NELECTRON_POOL),
            "QQ": S(_Q_POOL),
        },
    ),
    Template(
        name="cell_potential",
        latex=r"{Ecell} = E^\circ_{{\text{{cathode}}}} - E^\circ_{{\text{{anode}}}}",
        slots={"Ecell": S(_EPOT_POOL)},
    ),
    Template(
        name="gibbs_from_cell_potential",
        latex=r"{DG} = -{ne}\,{F}\,{E}",
        slots={
            "DG": S(_DG_POOL),
            "ne": S(_NELECTRON_POOL),
            "F": S(_FARAD_POOL),
            "E": S(_EPOT_POOL),
        },
    ),
    Template(
        name="faraday_electrolysis_mass",
        latex=r"m = \frac{{M\,I\,{tt}}}{{{F}\,{ne}}}",
        slots={
            "tt": S(_TIME_POOL),
            "F": S(_FARAD_POOL),
            "ne": S(_NELECTRON_POOL),
        },
    ),
    Template(
        name="faraday_charge",
        latex=r"Q = {ne}\,{F}\,{nn}",
        slots={
            "ne": S(_NELECTRON_POOL),
            "F": S(_FARAD_POOL),
            "nn": S(_N_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Section E — Chemistry-Specific Notation (10)
# ---------------------------------------------------------------------------

_CHEM_TEMPLATES += [
    Template(
        name="ce_arrow_reaction",
        latex=r"\ce{{{ca} {A} + {cb} {B} -> {cc} {C}}}",
        slots={
            "ca": S(_COEFF_POOL),
            "A": S(_CE_SPECIES_POOL),
            "cb": S(_COEFF_POOL),
            "B": S(_CE_SPECIES_POOL),
            "cc": S(_COEFF_POOL),
            "C": S(_CE_SPECIES_POOL),
        },
    ),
    Template(
        name="ce_equilibrium_arrow",
        latex=r"\ce{{{A} <=> {B}}}",
        slots={
            "A": S(_CE_SPECIES_POOL),
            "B": S(_CE_SPECIES_POOL),
        },
    ),
    Template(
        name="concentration_bracket",
        latex=r"[{sp}] = {ee}",
        slots={
            "sp": S(_CONC_SPECIES_POOL),
            "ee": S(_STOICH_EXP_POOL),
        },
    ),
    Template(
        name="ion_charge_notation",
        latex=r"{ion} + {sp}{state} \rightarrow \text{{precipitate}}",
        slots={
            "ion": S(_ION_POOL),
            "sp": S(_SPECIES_POOL),
            "state": S(_STATE_POOL),
        },
    ),
    Template(
        name="species_with_state",
        latex=r"{sp}{st}",
        slots={
            "sp": S(_SPECIES_POOL),
            "st": S(_STATE_POOL),
        },
    ),
    Template(
        name="stoich_ratio",
        latex=r"\frac{{{ca}\,\text{{mol}}\,{A}}}{{{cb}\,\text{{mol}}\,{B}}}",
        slots={
            "ca": S(_COEFF_POOL),
            "cb": S(_COEFF_POOL),
            "A": S(_SPECIES_POOL),
            "B": S(_SPECIES_POOL),
        },
    ),
    Template(
        name="molar_mass_fraction",
        latex=r"n = \frac{{m}}{{M_{{{sp}}}}}",
        slots={"sp": S(_SPECIES_POOL)},
    ),
    Template(
        name="concentration_molarity",
        latex=r"c = \frac{{{nn}}}{{V}}",
        slots={"nn": S(_N_POOL)},
    ),
    Template(
        name="dilution_equation",
        latex=r"c_1 V_1 = c_2 V_2",
        slots={},
    ),
    Template(
        name="colligative_boiling_point",
        latex=r"\Delta T_b = K_b\,{nn}\,m",
        slots={"nn": S(_N_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Dispatcher and exports
# ---------------------------------------------------------------------------

_W_CHEM = compute_weights(_CHEM_TEMPLATES)
_chemistry = make_dispatcher(_CHEM_TEMPLATES, _W_CHEM)

GENERATORS: dict[str, Callable[[random.Random], str]] = {"chemistry": _chemistry}
WEIGHTS: dict[str, float] = {"chemistry": 0.05}
TEMPLATES: dict[str, list[Template]] = {"chemistry": _CHEM_TEMPLATES}
