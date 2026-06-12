"""Quantum mechanics domain — Schrödinger, commutators, ladder operators, anticommutators."""

from __future__ import annotations

from ..engine._template_dsl import _FN_SLOT, _LIM_MOD, S, Template, X
from ._config import register_domain
from ._physics_vocab import (
    _COORD_POOL,
    _DAG_OP_POOL,
    _DAG_STATE_POOL,
    _ENERGY_POOL,
    _FERM_IDX_POOL,
    _HAM_POOL,
    _HBAR_POOL,
    _MASS_POOL,
    _N_POOL,
    _OMEGA_POOL,
    _P_POOL,
    _PSI_POOL,
    _Q_POOL,
)

_TEMPLATES: list[Template] = [
    # ── Part A: Schrödinger equations / uncertainty ──────────────────────────
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
        name="heisenberg_uncertainty",
        latex=r"\Delta {cc}\,\Delta {pp} \geq \frac{{{hb}}}{{2}}",
        slots={"cc": S(_COORD_POOL), "pp": S(_P_POOL), "hb": S(_HBAR_POOL)},
    ),
    Template(
        name="energy_momentum_relation",
        latex=r"{ee}^2 = (pc)^2 + ({mm}\,c^2)^2",
        slots={"ee": S(_ENERGY_POOL), "mm": S(_MASS_POOL)},
    ),
    # ── Part B: Quantum mechanics ────────────────────────────────────────────
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
    # ── Anticommutators (fermionic operators) ────────────────────────────────
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
    # ── Part C: Function-pair (QM context) ───────────────────────────────────
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
    # ── Adjoint operators ────────────────────────────────────────────────────
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
    # ── Approximation methods ────────────────────────────────────────────────
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
    # ── Wavefunction notation ────────────────────────────────────────────────
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
]

GENERATORS, WEIGHTS, TEMPLATES = register_domain("quantum_mechanics", _TEMPLATES)
