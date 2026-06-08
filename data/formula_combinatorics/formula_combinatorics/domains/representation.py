"""Representation theory and character theory domain generator."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import E, S, Template, X, compute_weights, make_dispatcher
from .._vocab import _fn_rich_nosub

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_REP_POOL = (r"\rho", r"\pi", r"\sigma", r"\tau", "V", "W", "U", r"\phi")  # 8
_GROUP_POOL = ("G", "H", "K", "N", "A", "B", r"\Gamma", r"\Delta")  # 8
_ELEM_POOL = (
    "g",
    "h",
    "a",
    "b",
    "x",
    r"\sigma",
    r"\tau",
    r"\alpha",
    r"\beta",
    r"\gamma",
)  # 10
_HOMO_POOL = (r"\phi", r"\psi", r"\theta", "f", "T", r"\Phi", r"\Psi", r"\chi")  # 8
_MODULE_POOL = ("M", "N", "V", "W", "U", "L", "S", "P")  # 8
_WEIGHT_POOL = (
    r"\lambda",
    r"\mu",
    r"\nu",
    r"\alpha",
    r"\beta",
    r"\omega",
    r"\Lambda",
)  # 7
# Values like r"\mathfrak{g}" are pool values, not format strings — safe as substitution targets.
_LIE_POOL = (
    r"\mathfrak{g}",
    r"\mathfrak{h}",
    r"\mathfrak{n}",
    r"\mathfrak{b}",
    r"\mathfrak{k}",
    r"\mathfrak{m}",
)  # 6
_FIELD_POOL = ("k", r"\mathbb{F}", "K", r"\mathbb{C}", r"\mathbb{R}")  # 5
_IDX_POOL2 = ("i", "j", "k", "m", "n")  # 5

# ---------------------------------------------------------------------------
# Part A: Reparameterized originals (12)
# ---------------------------------------------------------------------------

_TEMPLATES_A: list[Template] = [
    Template(
        name="character_trace",
        latex=r"\chi_{{{rr}}}({gg}) = \operatorname{{tr}}\!\left({rr}({gg})\right)",
        slots={"rr": S(_REP_POOL), "gg": S(_ELEM_POOL)},
    ),  # n_eff = 80
    Template(
        name="character_dimension",
        latex=r"\chi_{{{rr}}}(e) = \dim {rr}",
        slots={"rr": S(_REP_POOL)},
    ),  # n_eff = 8
    Template(
        name="character_orthogonality",
        latex=(
            r"\frac{{1}}{{|{GG}|}} \sum_{{{gg} \in {GG}}} "
            r"\chi_i({gg})\,\overline{{\chi_j({gg})}} = \delta_{{ij}}"
        ),
        slots={"GG": S(_GROUP_POOL), "gg": S(_ELEM_POOL)},
    ),  # n_eff = 80
    Template(
        name="character_centralizer",
        latex=r"\sum_{{i}} |\chi_i({gg})|^2 = |C_{{{GG}}}({gg})|",
        slots={"GG": S(_GROUP_POOL), "gg": S(_ELEM_POOL)},
    ),  # n_eff = 80
    Template(
        name="irreducible_decomposition",
        latex=(
            r"{rr} \cong \bigoplus_i {rr}_i^{{\oplus n_i}},\quad "
            r"{rr}_i \text{{ irreducible}}"
        ),
        slots={"rr": S(_REP_POOL)},
    ),  # n_eff = 8
    Template(
        name="sum_of_squares",
        latex=r"\sum_i (\dim {rr}_i)^2 = |{GG}|",
        slots={"rr": S(_REP_POOL), "GG": S(_GROUP_POOL)},
    ),  # n_eff = 64
    Template(
        name="burnside_orbit",
        latex=(r"|X/{GG}| = \frac{{1}}{{|{GG}|}} \sum_{{{gg} \in {GG}}} |X^{{{gg}}}|"),
        slots={"GG": S(_GROUP_POOL), "gg": S(_ELEM_POOL)},
    ),  # n_eff = 80
    Template(
        name="burnside_fixed_points",
        latex=(
            r"|\text{{orbits}}| = \frac{{1}}{{|{GG}|}} "
            r"\sum_{{{gg} \in {GG}}} |\{{x : {gg} \cdot x = x\}}|"
        ),
        slots={"GG": S(_GROUP_POOL), "gg": S(_ELEM_POOL)},
    ),  # n_eff = 80
    Template(
        name="schur_linear_map",
        latex=(
            r"{phi} : {rr} \to {rr} \text{{ {GG}-linear}} "
            r"\implies {phi} = \lambda \operatorname{{Id}}"
        ),
        slots={"phi": S(_HOMO_POOL), "rr": S(_REP_POOL), "GG": S(_GROUP_POOL)},
    ),  # n_eff = 512
    Template(
        name="schur_hom_space",
        latex=(
            r"\operatorname{{Hom}}_{{{GG}}}({rr}_i, {rr}_j) \cong "
            r"\begin{{cases}} \mathbb{{C}} & i = j \\ 0 & i \neq j \end{{cases}}"
        ),
        slots={"GG": S(_GROUP_POOL), "rr": S(_REP_POOL)},
    ),  # n_eff = 64
    Template(
        name="induced_representation",
        latex=(
            r"\operatorname{{Ind}}_{{{HH}}}^{{{GG}}} {rr} \cong "
            r"\mathbb{{C}}[{GG}] \otimes_{{\mathbb{{C}}[{HH}]}} {rr}"
        ),
        slots={
            "GG": S(_GROUP_POOL),
            "HH": X(_GROUP_POOL, ("GG",)),
            "rr": S(_REP_POOL),
        },
    ),  # n_eff = 448
    Template(
        name="frobenius_reciprocity",
        latex=(
            r"\langle \operatorname{{Ind}}_{{{HH}}}^{{{GG}}} {chi},"
            r" {psi} \rangle_{{{GG}}} = \langle {chi},"
            r" \operatorname{{Res}}_{{{HH}}}^{{{GG}}} {psi} \rangle_{{{HH}}}"
        ),
        slots={
            "GG": S(_GROUP_POOL),
            "HH": X(_GROUP_POOL, ("GG",)),
            "chi": S(_HOMO_POOL),
            "psi": X(_HOMO_POOL, ("chi",)),
        },
    ),  # n_eff = 3,136
]

# ---------------------------------------------------------------------------
# Part B1: Character Theory (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B1: list[Template] = [
    Template(
        name="inner_product_characters",
        latex=(
            r"\langle \chi_{{{rr}}}, \chi_{{{ss}}} \rangle_{{{GG}}} = "
            r"\frac{{1}}{{|{GG}|}} \sum_{{{gg} \in {GG}}} "
            r"\chi_{{{rr}}}({gg})\, \overline{{\chi_{{{ss}}}({gg})}}"
        ),
        slots={
            "rr": S(_REP_POOL),
            "ss": X(_REP_POOL, ("rr",)),
            "GG": S(_GROUP_POOL),
            "gg": S(_ELEM_POOL),
        },
    ),  # n_eff = 4,480
    Template(
        name="character_sum_formula",
        latex=(
            r"\sum_{{{gg} \in {GG}}} \chi_{{{rr}}}({gg}) = "
            r"\langle \chi_{{{rr}}}, \mathbf{{1}} \rangle \cdot |{GG}|"
        ),
        slots={"rr": S(_REP_POOL), "GG": S(_GROUP_POOL), "gg": S(_ELEM_POOL)},
    ),  # n_eff = 640
    Template(
        name="character_product",
        latex=(
            r"\chi_{{{rr} \otimes {ss}}}({gg}) = "
            r"\chi_{{{rr}}}({gg}) \cdot \chi_{{{ss}}}({gg})"
        ),
        slots={
            "rr": S(_REP_POOL),
            "ss": X(_REP_POOL, ("rr",)),
            "gg": S(_ELEM_POOL),
        },
    ),  # n_eff = 560
    Template(
        name="character_dual",
        latex=r"\chi_{{{rr}^*}}({gg}) = \overline{{\chi_{{{rr}}}({gg})}}",
        slots={"rr": S(_REP_POOL), "gg": S(_ELEM_POOL)},
    ),  # n_eff = 80
    Template(
        name="character_table_column_ortho",
        latex=(
            r"\sum_i \chi_i({gg})\, \overline{{\chi_i({hh})}} = "
            r"|C_{{{GG}}}({gg})| \cdot \delta_{{[{gg}][{hh}]}}"
        ),
        slots={
            "GG": S(_GROUP_POOL),
            "gg": S(_ELEM_POOL),
            "hh": X(_ELEM_POOL, ("gg",)),
        },
    ),  # n_eff = 720
    Template(
        name="multiplicity_formula",
        latex=(
            r"m_{{{rr}}} = \frac{{1}}{{|{GG}|}} "
            r"\sum_{{{gg} \in {GG}}} \chi_V({gg})\, \overline{{\chi_{{{rr}}}({gg})}}"
        ),
        slots={"rr": S(_REP_POOL), "GG": S(_GROUP_POOL), "gg": S(_ELEM_POOL)},
    ),  # n_eff = 640
]

# ---------------------------------------------------------------------------
# Part B2: Schur's Lemma & Hom Spaces (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B2: list[Template] = [
    Template(
        name="schur_irreducible",
        latex=(
            r"{phi} \in \operatorname{{Hom}}_{{{GG}}}({rr}, {rr}) "
            r"\Rightarrow {phi} = \lambda \cdot \operatorname{{Id}}_{{{rr}}}"
        ),
        slots={"phi": S(_HOMO_POOL), "rr": S(_REP_POOL), "GG": S(_GROUP_POOL)},
    ),  # n_eff = 512
    Template(
        name="schur_noniso",
        latex=(
            r"{rr} \not\cong {ss} \text{{ irreducible}} "
            r"\Rightarrow \operatorname{{Hom}}_{{{GG}}}({rr}, {ss}) = 0"
        ),
        slots={
            "rr": S(_REP_POOL),
            "ss": X(_REP_POOL, ("rr",)),
            "GG": S(_GROUP_POOL),
        },
    ),  # n_eff = 448
    Template(
        name="endomorphism_dim",
        latex=(
            r"\dim \operatorname{{Hom}}_{{{GG}}}({rr}, {ss}) = "
            r"\langle \chi_{{{rr}}}, \chi_{{{ss}}} \rangle_{{{GG}}}"
        ),
        slots={
            "GG": S(_GROUP_POOL),
            "rr": S(_REP_POOL),
            "ss": X(_REP_POOL, ("rr",)),
        },
    ),  # n_eff = 448
    Template(
        name="isotypic_component",
        latex=(
            r"{VV} = \bigoplus_{{[{rr}]}} {MM}_{{{rr}}},\quad "
            r"{MM}_{{{rr}}} \cong {rr}^{{\oplus m_{{{rr}}}}}"
        ),
        slots={
            "VV": S(_MODULE_POOL),
            "rr": X(_MODULE_POOL, ("VV",)),
            "MM": X(_MODULE_POOL, ("VV", "rr")),
        },
    ),  # n_eff = 336
    Template(
        name="intertwining_operator",
        latex=(
            r"{phi} \circ {rr}({gg}) = {ss}({gg}) \circ {phi} "
            r"\quad \forall {gg} \in {GG}"
        ),
        slots={
            "phi": S(_HOMO_POOL),
            "rr": S(_REP_POOL),
            "ss": X(_REP_POOL, ("rr",)),
            "gg": S(_ELEM_POOL),
            "GG": S(_GROUP_POOL),
        },
    ),  # n_eff = 35,840
    Template(
        name="decomp_multiplicity",
        latex=(
            r"{VV} \cong \bigoplus_{{{rr}}} {rr}^{{\oplus m_{{{rr}}}}},\quad "
            r"m_{{{rr}}} = \langle \chi_{{{VV}}}, \chi_{{{rr}}} \rangle"
        ),
        slots={"VV": S(_MODULE_POOL), "rr": X(_MODULE_POOL, ("VV",))},
    ),  # n_eff = 56
]

# ---------------------------------------------------------------------------
# Part B3: Induced/Restricted Representations (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B3: list[Template] = [
    Template(
        name="dimension_induction",
        latex=(
            r"\dim\!\left(\operatorname{{Ind}}_{{{HH}}}^{{{GG}}} {rr}\right) = "
            r"[{GG}:{HH}] \cdot \dim {rr}"
        ),
        slots={
            "GG": S(_GROUP_POOL),
            "HH": X(_GROUP_POOL, ("GG",)),
            "rr": S(_REP_POOL),
        },
    ),  # n_eff = 448
    Template(
        name="induced_character",
        latex=(
            r"\chi_{{\operatorname{{Ind}}_{{{HH}}}^{{{GG}}}{rr}}}({gg}) = "
            r"\frac{{1}}{{|{HH}|}} \sum_{{x \in {GG},\, x^{{-1}} {gg} x \in {HH}}} "
            r"\chi_{{{rr}}}(x^{{-1}} {gg} x)"
        ),
        slots={
            "GG": S(_GROUP_POOL),
            "HH": X(_GROUP_POOL, ("GG",)),
            "rr": S(_REP_POOL),
            "gg": S(_ELEM_POOL),
        },
    ),  # n_eff = 4,480
    Template(
        name="character_restriction",
        latex=(
            r"\chi_{{\operatorname{{Res}}_{{{HH}}}^{{{GG}}}{rr}}}({hh}) = "
            r"\chi_{{{rr}}}({hh}) \quad \forall {hh} \in {HH}"
        ),
        slots={
            "GG": S(_GROUP_POOL),
            "HH": X(_GROUP_POOL, ("GG",)),
            "rr": S(_REP_POOL),
            "hh": S(_ELEM_POOL),
        },
    ),  # n_eff = 4,480
    Template(
        name="tensor_product_reps",
        latex=(r"({rr} \otimes {ss})({gg}) = {rr}({gg}) \otimes {ss}({gg})"),
        slots={
            "rr": S(_REP_POOL),
            "ss": X(_REP_POOL, ("rr",)),
            "gg": S(_ELEM_POOL),
        },
    ),  # n_eff = 560
    Template(
        name="mackey_criterion",
        latex=(
            r"\operatorname{{Ind}}_{{{HH}}}^{{{GG}}} {chi} \text{{ irred}} "
            r"\iff {chi}^s \neq {chi} \;\forall s \in {GG} \setminus {HH}"
        ),
        slots={
            "GG": S(_GROUP_POOL),
            "HH": X(_GROUP_POOL, ("GG",)),
            "chi": S(_HOMO_POOL),
        },
    ),  # n_eff = 448
    Template(
        name="clifford_theory",
        latex=(
            r"\operatorname{{Res}}_{{{NN}}}^{{{GG}}} {rr} = "
            r"e \cdot \bigoplus_t {rr}^{{(t)}}, \quad {NN} \trianglelefteq {GG}"
        ),
        slots={
            "GG": S(_GROUP_POOL),
            "NN": X(_GROUP_POOL, ("GG",)),
            "rr": S(_REP_POOL),
        },
    ),  # n_eff = 448
]

# ---------------------------------------------------------------------------
# Part B4: Module Theory (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B4: list[Template] = [
    Template(
        name="maschke_theorem",
        latex=(
            r"\operatorname{{char}}({ff}) \nmid |{GG}| "
            r"\Rightarrow {ff}[{GG}] \text{{ is semisimple}}"
        ),
        slots={"ff": S(_FIELD_POOL), "GG": S(_GROUP_POOL)},
    ),  # n_eff = 40
    Template(
        name="krull_schmidt",
        latex=(
            r"{MM} \cong \bigoplus_i {MM}_i^{{n_i}},\quad "
            r"{MM}_i \text{{ indecomposable, unique up to iso}}"
        ),
        slots={"MM": S(_MODULE_POOL)},
    ),  # n_eff = 8
    Template(
        name="module_hom",
        latex=(
            r"\operatorname{{Hom}}_{{k{GG}}}({MM}, {NN}) = "
            r"\{{ f : f({gg} \cdot m) = {gg} \cdot f(m) \;\forall {gg} \in {GG} \}}"
        ),
        slots={
            "MM": S(_MODULE_POOL),
            "NN": X(_MODULE_POOL, ("MM",)),
            "GG": S(_GROUP_POOL),
            "gg": S(_ELEM_POOL),
        },
    ),  # n_eff = 4,480
    Template(
        name="semisimple_algebra_decomp",
        latex=r"{ff}[{GG}] \cong \bigoplus_i M_{{n_i}}({ff})",
        slots={"ff": S(_FIELD_POOL), "GG": S(_GROUP_POOL)},
    ),  # n_eff = 40
    Template(
        name="exact_sequence",
        latex=(
            r"0 \to {LL} \to {MM} \to {NN} \to 0 "
            r"\text{{ (short exact of }} {GG}\text{{-modules)}}"
        ),
        slots={
            "LL": S(_MODULE_POOL),
            "MM": X(_MODULE_POOL, ("LL",)),
            "NN": X(_MODULE_POOL, ("LL", "MM")),
            "GG": S(_GROUP_POOL),
        },
    ),  # n_eff = 2,688
    Template(
        name="projective_cover",
        latex=(
            r"{PP} \twoheadrightarrow {MM} \to 0, "
            r"\quad {PP} \text{{ projective }} {GG}\text{{-module}}"
        ),
        slots={
            "PP": S(_MODULE_POOL),
            "MM": X(_MODULE_POOL, ("PP",)),
            "GG": S(_GROUP_POOL),
        },
    ),  # n_eff = 448
]

# ---------------------------------------------------------------------------
# Part B5: Lie Algebra Representations (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B5: list[Template] = [
    Template(
        name="lie_module_action",
        latex=(
            r"X \cdot (Y \cdot v) - Y \cdot (X \cdot v) = [X, Y] \cdot v, "
            r"\quad X, Y \in {gg}"
        ),
        slots={"gg": S(_LIE_POOL)},
    ),  # n_eff = 6
    Template(
        name="weight_space",
        latex=(
            r"{VV}_{{{ww}}} = \{{ v \in {VV} : {hh} \cdot v = {ww}({hh})\, v "
            r"\;\forall {hh} \in {gg} \}}"
        ),
        slots={
            "VV": S(_MODULE_POOL),
            "ww": S(_WEIGHT_POOL),
            "hh": S(_ELEM_POOL),
            "gg": S(_LIE_POOL),
        },
    ),  # n_eff = 3,360
    Template(
        name="highest_weight_module",
        latex=(
            r"{VV} = \bigoplus_{{{ww} \leq {ll}}} {VV}_{{{ww}}}, "
            r"\quad {VV}_{{{ll}}} \neq 0"
        ),
        slots={
            "VV": S(_MODULE_POOL),
            "ll": S(_WEIGHT_POOL),
            "ww": X(_WEIGHT_POOL, ("ll",)),
        },
    ),  # n_eff = 336
    Template(
        name="root_decomposition",
        latex=(r"{gg} = {hh} \oplus \bigoplus_{{{alpha} \in \Phi}} {gg}_{{{alpha}}}"),
        slots={
            "gg": S(_LIE_POOL),
            "hh": X(_LIE_POOL, ("gg",)),
            "alpha": S(_WEIGHT_POOL),
        },
    ),  # n_eff = 210
    Template(
        name="killing_form",
        latex=(
            r"\kappa(X, Y) = \operatorname{{tr}}\!\left("
            r"\operatorname{{ad}} X \circ \operatorname{{ad}} Y\right), "
            r"\quad X, Y \in {gg}"
        ),
        slots={"gg": S(_LIE_POOL)},
    ),  # n_eff = 6
    Template(
        name="casimir_element",
        latex=(
            r"\Omega \cdot v = c_{{{ll}}} \cdot v \quad "
            r"\forall v \in {VV}_{{{ll}}} \text{{ (Casimir on irreducible)}}"
        ),
        slots={"ll": S(_WEIGHT_POOL), "VV": S(_MODULE_POOL)},
    ),  # n_eff = 56
]

# ---------------------------------------------------------------------------
# Part B6: Symmetric Group & Young Tableaux (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B6: list[Template] = [
    Template(
        name="hook_length_formula",
        latex=(r"\dim S^{{{ll}}} = \frac{{n!}}{{\prod_{{(i,j) \in {ll}}} h(i,j)}}"),
        slots={"ll": S(_WEIGHT_POOL)},
    ),  # n_eff = 7
    Template(
        name="young_symmetrizer",
        latex=r"c_{{{ll}}} = a_{{{ll}}} \cdot b_{{{ll}}} \in \mathbb{{C}}[S_n]",
        slots={"ll": S(_WEIGHT_POOL)},
    ),  # n_eff = 7
    Template(
        name="plancherel_measure",
        latex=(r"\mathbb{{P}}(\lambda = {ll}) = \frac{{(\dim S^{{{ll}}})^2}}{{n!}}"),
        slots={"ll": S(_WEIGHT_POOL)},
    ),  # n_eff = 7
    Template(
        name="murnaghan_nakayama",
        latex=(
            r"\chi^{{{ll}}}({gg}) = \sum_T (-1)^{{\operatorname{{ht}}(T)}} "
            r"\chi^{{{mm}}}(\tilde{{{gg}}}), \quad \text{{rim-hook tableaux}}"
        ),
        slots={
            "ll": S(_WEIGHT_POOL),
            "mm": X(_WEIGHT_POOL, ("ll",)),
            "gg": S(_ELEM_POOL),
        },
    ),  # n_eff = 420
    Template(
        name="specht_module",
        latex=(
            r"S^{{{ll}}} = \mathbb{{C}}[S_n] \cdot e_T, "
            r"\quad T \text{{ standard of shape }} {ll}"
        ),
        slots={"ll": S(_WEIGHT_POOL)},
    ),  # n_eff = 7
    Template(
        name="littlewood_richardson",
        latex=(
            r"c^{{{nu}}}_{{{ll},{mm}}} = "
            r"\#\text{{LR-tableaux of shape }} {nu}/{ll} "
            r"\text{{ and content }} {mm}"
        ),
        slots={
            "nu": S(_WEIGHT_POOL),
            "ll": X(_WEIGHT_POOL, ("nu",)),
            "mm": X(_WEIGHT_POOL, ("nu", "ll")),
        },
    ),  # n_eff = 210
]

# ---------------------------------------------------------------------------
# Part B7: Representation-Theoretic Identities (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B7: list[Template] = [
    Template(
        name="rep_ring_sum",
        latex=r"[{rr}] + [{ss}] = [{rr} \oplus {ss}] \in R({GG})",
        slots={
            "rr": S(_REP_POOL),
            "ss": X(_REP_POOL, ("rr",)),
            "GG": S(_GROUP_POOL),
        },
    ),  # n_eff = 448
    Template(
        name="rep_ring_product_rule",
        latex=r"[{rr}] \cdot [{ss}] = [{rr} \otimes {ss}] \in R({GG})",
        slots={
            "rr": S(_REP_POOL),
            "ss": X(_REP_POOL, ("rr",)),
            "GG": S(_GROUP_POOL),
        },
    ),  # n_eff = 448
    Template(
        name="adams_operation",
        latex=(r"\psi^{{{kk}}}(\chi_{{{rr}}})({gg}) = \chi_{{{rr}}}({gg}^{{{kk}}})"),
        slots={"kk": S(_IDX_POOL2), "rr": S(_REP_POOL), "gg": S(_ELEM_POOL)},
    ),  # n_eff = 400
    Template(
        name="regular_representation",
        latex=(r"\mathbb{{C}}[{GG}] \cong \bigoplus_i {rr}_i^{{\oplus \dim {rr}_i}}"),
        slots={"GG": S(_GROUP_POOL), "rr": S(_REP_POOL)},
    ),  # n_eff = 64
    Template(
        name="character_value_conjugacy",
        latex=(
            r"{gg} \sim {hh} \text{{ in }} {GG} "
            r"\Rightarrow \chi_{{{rr}}}({gg}) = \chi_{{{rr}}}({hh})"
        ),
        slots={
            "GG": S(_GROUP_POOL),
            "rr": S(_REP_POOL),
            "gg": S(_ELEM_POOL),
            "hh": X(_ELEM_POOL, ("gg",)),
        },
    ),  # n_eff = 5,760
    Template(
        name="group_algebra_iso",
        latex=(
            r"{ff}[{GG}] \cong \bigoplus_i M_{{n_i}}({ff}), "
            r"\quad \sum_i n_i^2 = |{GG}|"
        ),
        slots={"ff": S(_FIELD_POOL), "GG": S(_GROUP_POOL)},
    ),  # n_eff = 40
]

# ---------------------------------------------------------------------------
# Part C: High-n_eff function-pair templates (6)
# ---------------------------------------------------------------------------

_TEMPLATES_C: list[Template] = [
    Template(
        name="character_fn_sum",
        latex=(
            r"\frac{{1}}{{|{GG}|}} \sum_{{{gg} \in {GG}}} "
            r"{fn1}({gg}) \cdot \overline{{{fn2}({gg})}}"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "GG": S(_GROUP_POOL),
            "gg": S(_ELEM_POOL),
        },
    ),  # n_eff = 8,000,000
    Template(
        name="trace_fn_pair",
        latex=(
            r"\operatorname{{tr}}\!\left({fn1}({gg}) \cdot {fn2}({hh})\right) = "
            r"\sum_i {fn1}({gg})_{{ii}} \cdot {fn2}({hh})_{{ii}}"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "gg": S(_ELEM_POOL),
            "hh": X(_ELEM_POOL, ("gg",)),
        },
    ),  # n_eff = 9,000,000
    Template(
        name="weight_fn_module",
        latex=(
            r"{fn1}({hh}) \cdot {fn2}(v) = {ww}({hh}) \cdot {fn2}(v) "
            r"\text{{ in weight space }} {VV}_{{{ww}}}"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "ww": S(_WEIGHT_POOL),
            "VV": S(_MODULE_POOL),
            "hh": S(_ELEM_POOL),
        },
    ),  # n_eff = 56,000,000
    Template(
        name="fn_triple_rep",
        latex=(r"{fn1}({gg} \cdot {hh}) = {fn2}({gg}) \cdot {fn3}({hh})"),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "fn3": E(_fn_rich_nosub, n=100),
            "gg": S(_ELEM_POOL),
            "hh": X(_ELEM_POOL, ("gg",)),
        },
    ),  # n_eff = 900,000,000 (capped at 1M for weights)
    Template(
        name="rep_fn_equivariance",
        latex=(
            r"{fn1}({gg} \cdot x) = {rr}({gg}) \cdot {fn1}(x) "
            r"\quad \forall {gg} \in {GG}"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "rr": S(_REP_POOL),
            "gg": S(_ELEM_POOL),
            "GG": S(_GROUP_POOL),
        },
    ),  # n_eff = 64,000
    Template(
        name="matrix_coeff_fn",
        latex=(r"{fn1}({gg})_{{ij}} = \langle {fn1}({gg})\, e_j,\, e_i \rangle"),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "gg": S(_ELEM_POOL),
            "GG": S(_GROUP_POOL),
        },
    ),  # n_eff = 8,000
]

# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_TEMPLATES_D: list[Template] = [
    Template(
        name="rep_isomorphism_simeq",
        latex=r"{rr} \simeq {ss} \iff \exists\text{{ invertible intertwiner }} {phi}: {rr} \xrightarrow{{\sim}} {ss}",
        slots={"rr": S(_REP_POOL), "ss": X(_REP_POOL, ("rr",)), "phi": S(_HOMO_POOL)},
    ),
    Template(
        name="character_determines_rep",
        latex=r"\chi_{{{rr}}} = \chi_{{{ss}}} \implies {rr} \simeq {ss}",
        slots={"rr": S(_REP_POOL), "ss": X(_REP_POOL, ("rr",))},
    ),
    Template(
        name="external_tensor_product",
        latex=r"({rr} \boxtimes {ss})({gg},\,{hh}) = {rr}({gg}) \otimes {ss}({hh})",
        slots={
            "rr": S(_REP_POOL),
            "ss": X(_REP_POOL, ("rr",)),
            "gg": S(_ELEM_POOL),
            "hh": X(_ELEM_POOL, ("gg",)),
        },
    ),
    Template(
        name="boxtimes_character",
        latex=r"\chi_{{{rr} \boxtimes {ss}}}({gg},\,{hh}) = \chi_{{{rr}}}({gg})\,\chi_{{{ss}}}({hh})",
        slots={
            "rr": S(_REP_POOL),
            "ss": X(_REP_POOL, ("rr",)),
            "gg": S(_ELEM_POOL),
            "hh": X(_ELEM_POOL, ("gg",)),
        },
    ),
]

_REPR_TEMPLATES: list[Template] = (
    _TEMPLATES_A
    + _TEMPLATES_B1
    + _TEMPLATES_B2
    + _TEMPLATES_B3
    + _TEMPLATES_B4
    + _TEMPLATES_B5
    + _TEMPLATES_B6
    + _TEMPLATES_B7
    + _TEMPLATES_C
    + _TEMPLATES_D
)

_W_REPR: list[float] = compute_weights(_REPR_TEMPLATES)
_representation_theory = make_dispatcher(_REPR_TEMPLATES, _W_REPR)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "representation_theory": _representation_theory,
}

WEIGHTS: dict[str, float] = {
    "representation_theory": 0.01,
}

TEMPLATES: dict[str, list[Template]] = {
    "representation_theory": _REPR_TEMPLATES,
}
