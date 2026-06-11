"""Quantum notation domain — systematic Dirac bra-ket coverage for OCR pretraining."""

from __future__ import annotations

from .._template_dsl import S, Template, X
from ._config import register_domain

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_STATE_POOL = (
    r"\psi",
    r"\phi",
    r"\Psi",
    r"\varphi",
    r"\chi",
    r"\xi",
    r"\alpha",
    r"\beta",
    r"\psi_0",
    r"\phi_0",
    r"\psi_n",
    r"\phi_k",
)
# Subset without pre-attached subscripts, for templates that add their own subscript
_PLAIN_STATE_POOL = (
    r"\psi",
    r"\phi",
    r"\Psi",
    r"\varphi",
    r"\chi",
    r"\xi",
    r"\alpha",
    r"\beta",
)

_OP_POOL = (
    r"\hat{A}",
    r"\hat{B}",
    r"\hat{H}",
    r"\hat{O}",
    r"\hat{U}",
    r"\hat{P}",
    r"\hat{Q}",
    r"\hat{S}",
    r"\hat{L}",
    r"\hat{J}",
    "A",
    "U",
    "H",
    "O",
)

_IDX_POOL = ("n", "m", "k", "i", "j", r"\ell", r"\nu", r"\mu")

_COEFF_POOL = (r"\alpha", r"\beta", r"\gamma", r"c_0", r"c_1", r"c_n", r"a", r"b")

_ANG_J_POOL = (r"\tfrac{1}{2}", "1", r"\tfrac{3}{2}", "2", "j", r"j_1", r"j_2")
_ANG_M_POOL = (r"-\tfrac{1}{2}", r"\tfrac{1}{2}", "m", r"m_1", "-m", "0")
_ANG_L_POOL = ("0", "1", "2", r"\ell", "3")
_ANG_ML_POOL = ("0", r"\pm 1", "m", r"m_\ell", "-1")

_IDENTITY_POOL = (r"\hat{1}", r"\mathbf{1}", r"\mathbb{I}", r"\mathbf{I}")

_P_NORM_POOL = ("2", "p", r"\infty", "1")
_VEC_POOL = (r"\mathbf{u}", r"\mathbf{v}", r"\mathbf{x}", "u", "v", "x")
_MAT_POOL = ("A", "B", "M", "T", "U", "H")


# ---------------------------------------------------------------------------
# Part A — Atomic bra-ket forms (8 templates)
# ---------------------------------------------------------------------------

_PART_A: list[Template] = [
    Template(
        name="pure_ket",
        latex=r"|{state}\rangle",
        slots={"state": S(_STATE_POOL)},
    ),
    Template(
        name="pure_bra",
        latex=r"\langle{state}|",
        slots={"state": S(_STATE_POOL)},
    ),
    Template(
        name="inner_product_bk",
        latex=r"\langle{bra}|{ket}\rangle",
        slots={"bra": S(_STATE_POOL), "ket": X(_STATE_POOL, ("bra",))},
    ),
    Template(
        name="outer_product_bk",
        latex=r"|{ket}\rangle\langle{bra}|",
        slots={"ket": S(_STATE_POOL), "bra": X(_STATE_POOL, ("ket",))},
    ),
    Template(
        name="operator_expectation_short",
        latex=r"\langle {op} \rangle",
        slots={"op": S(_OP_POOL)},
    ),
    Template(
        name="operator_matrix_element",
        latex=r"\langle {bra} | {op} | {ket} \rangle",
        slots={
            "bra": S(_STATE_POOL),
            "op": S(_OP_POOL),
            "ket": X(_STATE_POOL, ("bra",)),
        },
    ),
    Template(
        name="hermitian_expectation_bk",
        latex=r"\langle {psi} | {op} | {psi} \rangle",
        slots={"psi": S(_STATE_POOL), "op": S(_OP_POOL)},
    ),
    Template(
        name="ket_norm_lvert",
        latex=r"\lVert |{state}\rangle \rVert = 1",
        slots={"state": S(_STATE_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B — Completeness relations and density matrices (5 templates)
# ---------------------------------------------------------------------------

_PART_B: list[Template] = [
    Template(
        name="completeness_discrete",
        latex=r"\sum_{{{idx}}} |{idx}\rangle\langle{idx}| = {id}",
        slots={"idx": S(_IDX_POOL), "id": S(_IDENTITY_POOL)},
    ),
    Template(
        name="completeness_continuous",
        latex=r"\int |x\rangle\langle x|\,dx = {id}",
        slots={"id": S(_IDENTITY_POOL)},
    ),
    Template(
        name="density_matrix_pure",
        latex=r"\rho = |{psi}\rangle\langle{psi}|",
        slots={"psi": S(_STATE_POOL)},
    ),
    Template(
        name="density_matrix_mixed",
        latex=r"\rho = \sum_{{{idx}}} p_{{{idx}}}\,|{psi}_{{{idx}}}\rangle\langle{psi}_{{{idx}}}|",
        slots={"idx": S(_IDX_POOL), "psi": S(_PLAIN_STATE_POOL)},
    ),
    Template(
        name="density_matrix_trace",
        latex=r"\operatorname{tr}(\rho) = 1, \quad \rho \geq 0",
        slots={},
    ),
]

# ---------------------------------------------------------------------------
# Part C — Composite and multi-particle states (7 templates)
# ---------------------------------------------------------------------------

_PART_C: list[Template] = [
    Template(
        name="ket_quantum_numbers_3",
        latex=r"|{nn},\,{ll},\,{mm}\rangle",
        slots={
            "nn": S(("n", "1", "2", "3", r"n'")),
            "ll": S(_ANG_L_POOL),
            "mm": S(_ANG_ML_POOL),
        },
    ),
    Template(
        name="ket_angular_momentum",
        latex=r"|{jj},\,{mm}\rangle",
        slots={"jj": S(_ANG_J_POOL), "mm": S(_ANG_M_POOL)},
    ),
    Template(
        name="ket_tensor_product",
        latex=r"|{s1}\rangle \otimes |{s2}\rangle",
        slots={"s1": S(_STATE_POOL), "s2": X(_STATE_POOL, ("s1",))},
    ),
    Template(
        name="ket_tensor_shorthand",
        latex=r"|{s1}\rangle|{s2}\rangle",
        slots={"s1": S(_STATE_POOL), "s2": X(_STATE_POOL, ("s1",))},
    ),
    Template(
        name="qubit_superposition",
        latex=r"|{psi}\rangle = {c1}|0\rangle + {c2}|1\rangle",
        slots={
            "psi": S(_STATE_POOL),
            "c1": S(_COEFF_POOL),
            "c2": X(_COEFF_POOL, ("c1",)),
        },
    ),
    Template(
        name="qubit_bloch_superposition",
        latex=(
            r"|{psi}\rangle = \cos\tfrac{{\theta}}{{2}}|0\rangle"
            r" + e^{{i\varphi}}\sin\tfrac{{\theta}}{{2}}|1\rangle"
        ),
        slots={"psi": S(_STATE_POOL)},
    ),
    Template(
        name="two_particle_composite",
        latex=r"|{n1},\,{n2}\rangle",
        slots={"n1": S(_IDX_POOL), "n2": X(_IDX_POOL, ("n1",))},
    ),
]

# ---------------------------------------------------------------------------
# Part D — Angular momentum coupling (4 templates)
# ---------------------------------------------------------------------------

_PART_D: list[Template] = [
    Template(
        name="clebsch_gordan_coeff",
        latex=r"\langle j_1\,m_1;\,j_2\,m_2 | j\,m \rangle",
        slots={},
    ),
    Template(
        name="cg_state_expansion",
        latex=(
            r"|j_1 j_2;\,j\,m\rangle"
            r" = \sum_{{m_1,m_2}}\langle j_1\,m_1;\,j_2\,m_2 | j\,m\rangle"
            r"\,|j_1\,m_1\rangle|j_2\,m_2\rangle"
        ),
        slots={},
    ),
    Template(
        name="wigner_3j_symbol",
        latex=(
            r"\begin{pmatrix}"
            r" j_1 & j_2 & j_3 \\"
            r" m_1 & m_2 & m_3"
            r" \end{pmatrix}"
        ),
        slots={},
    ),
    Template(
        name="wigner_6j_symbol",
        latex=(
            r"\begin{Bmatrix}"
            r" j_1 & j_2 & j_3 \\"
            r" j_4 & j_5 & j_6"
            r" \end{Bmatrix}"
        ),
        slots={},
    ),
]

# ---------------------------------------------------------------------------
# Part E — Vacuum and thermal expectation values (5 templates)
# ---------------------------------------------------------------------------

_PART_E: list[Template] = [
    Template(
        name="vacuum_expectation_value",
        latex=r"\langle 0 | {op} | 0 \rangle",
        slots={"op": S(_OP_POOL)},
    ),
    Template(
        name="vacuum_expectation_general",
        latex=r"\langle \Omega | {op} | \Omega \rangle",
        slots={"op": S(_OP_POOL)},
    ),
    Template(
        name="time_ordered_vev",
        latex=r"\langle 0 | T\{\hat{\phi}(x)\,\hat{\phi}(y)\} | 0 \rangle",
        slots={},
    ),
    Template(
        name="thermal_expectation_beta",
        latex=r"\langle {op} \rangle_{{\beta}} = \operatorname{{tr}}\!\left(\rho_{{\beta}}\,{op}\right)",
        slots={"op": S(_OP_POOL)},
    ),
    Template(
        name="thermal_expectation_t",
        latex=r"\langle {op} \rangle_T",
        slots={"op": S(_OP_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_QUANTUM_NOTATION_TEMPLATES: list[Template] = _PART_A + _PART_B + _PART_C + _PART_D + _PART_E


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("quantum_notation", _QUANTUM_NOTATION_TEMPLATES)
