"""Proof theory domain: 2D inference-rule layout (\\dfrac{premises}{conclusion})."""

from __future__ import annotations

from ..engine._template_dsl import _LIM_MOD, S, Template, X
from ._config import register_domain
from ._logic_vocab import (
    _LL_POOL,
    _PAIR_POOL,
    _PRED_POOL,
    _PROP_POOL,
    _TERM_POOL,
    _TYPE_POOL,
    _UNIV_POOL,
    _VAR_POOL,
)

# ---------------------------------------------------------------------------
# Part B3: Natural Deduction / Sequent Calculus (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B3: list[Template] = [
    Template(
        name="weakening_left",
        latex=(
            r"\dfrac{{\Gamma \vdash {qq}}}"
            r"{{{pp},\,\Gamma \vdash {qq}}} (\mathrm{{W}}_L)"
        ),
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="cut_rule",
        latex=(
            r"\dfrac{{\Gamma \vdash {pp} \quad {pp},\,\Delta \vdash {qq}}}"
            r"{{\Gamma,\,\Delta \vdash {qq}}} (\text{{Cut}})"
        ),
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="conjunction_intro",
        latex=(
            r"\dfrac{{\Gamma \vdash {pp} \quad \Gamma \vdash {qq}}}"
            r"{{\Gamma \vdash {pp} \land {qq}}} (\land I)"
        ),
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="implication_intro",
        latex=(
            r"\dfrac{{\Gamma,\,{pp} \vdash {qq}}}"
            r"{{\Gamma \vdash {pp} \Rightarrow {qq}}} (\Rightarrow I)"
        ),
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="disjunction_elim",
        latex=(
            r"\dfrac{{\Gamma \vdash {pp} \lor {qq}"
            r" \quad {pp} \vdash {rr} \quad {qq} \vdash {rr}}}"
            r"{{\Gamma \vdash {rr}}} (\lor E)"
        ),
        slots={
            "pp": S(_PROP_POOL),
            "qq": X(_PROP_POOL, ("pp",)),
            "rr": X(_PROP_POOL, ("pp", "qq")),
        },
    ),
    Template(
        name="negation_intro",
        latex=(
            r"\dfrac{{\Gamma,\,{pp} \vdash \bot}}"
            r"{{\Gamma \vdash \neg {pp}}} (\neg I)"
        ),
        slots={"pp": S(_PROP_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B6: Type Theory (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B6: list[Template] = [
    Template(
        name="function_type_intro",
        latex=(
            r"\dfrac{{\Gamma,\,{xx}:{sa} \vdash {ee}:{ta}}}"
            r"{{\Gamma \vdash \lambda {xx}.\,{ee}:{sa}\to {ta}}}"
        ),
        slots={
            "xx": S(_VAR_POOL),
            "ee": S(_TERM_POOL),
            "sa": S(_TYPE_POOL),
            "ta": X(_TYPE_POOL, ("sa",)),
        },
    ),
    Template(
        name="application_type",
        latex=(
            r"\dfrac{{\Gamma \vdash {ff}:{sa}\to {ta}"
            r" \quad \Gamma \vdash {aa}:{sa}}}"
            r"{{\Gamma \vdash {ff}\,{aa}:{ta}}}"
        ),
        slots={
            "ff": S(_TERM_POOL),
            "aa": X(_TERM_POOL, ("ff",)),
            "sa": S(_TYPE_POOL),
            "ta": X(_TYPE_POOL, ("sa",)),
        },
    ),
    Template(
        name="product_type",
        latex=r"{sa} \times {ta} \ni ({aa},{bb}),\quad {aa}:{sa},\;{bb}:{ta}",
        slots={
            "sa": S(_TYPE_POOL),
            "ta": X(_TYPE_POOL, ("sa",)),
            "aa": S(_TERM_POOL),
            "bb": X(_TERM_POOL, ("aa",)),
        },
    ),
    Template(
        name="sum_type",
        latex=(
            r"{sa} + {ta} \ni \mathrm{{inl}}\,{aa} \mid \mathrm{{inr}}\,{bb},"
            r"\quad {aa}:{sa},\;{bb}:{ta}"
        ),
        slots={
            "sa": S(_TYPE_POOL),
            "ta": X(_TYPE_POOL, ("sa",)),
            "aa": S(_TERM_POOL),
            "bb": X(_TERM_POOL, ("aa",)),
        },
    ),
    Template(
        name="curry_howard",
        latex=r"{pp} \Rightarrow {qq} \cong {sa} \to {ta}",
        slots={
            "pp": S(_PROP_POOL),
            "qq": X(_PROP_POOL, ("pp",)),
            "sa": S(_TYPE_POOL),
            "ta": X(_TYPE_POOL, ("sa",)),
        },
    ),
    Template(
        name="dependent_product",
        latex=r"\prod{lim_mod}_{{{xx}:{sa}}} {ta}({xx})",
        slots={
            "lim_mod": _LIM_MOD,
            "xx": S(_VAR_POOL),
            "sa": S(_TYPE_POOL),
            "ta": X(_TYPE_POOL, ("sa",)),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part H: Zero-Premise Axiom Rules (5)
# \dfrac{}{...} — empty-numerator pattern
# ---------------------------------------------------------------------------

_TEMPLATES_H: list[Template] = [
    Template(
        name="identity_axiom",
        latex=r"\dfrac{{}}{{{pp},\,\Gamma \vdash {pp}}} (\mathrm{{Ax}})",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="bottom_left",
        latex=r"\dfrac{{}}{{ \bot,\,\Gamma \vdash {pp}}} (\bot_L)",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="top_right",
        latex=r"\dfrac{{}}{{ \Gamma \vdash \top}} (\top_R)",
        slots={},
    ),
    Template(
        name="reflexivity_axiom",
        latex=r"\dfrac{{}}{{\Gamma \vdash {tt} = {tt}}} (\mathrm{{refl}})",
        slots={"tt": S(_TERM_POOL)},
    ),
    Template(
        name="lem_classical",
        latex=r"\dfrac{{}}{{\Gamma \vdash {pp} \lor \neg {pp}}} (\mathrm{{LEM}})",
        slots={"pp": S(_PROP_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part I: Structural Rules (4)
# ---------------------------------------------------------------------------

_TEMPLATES_I: list[Template] = [
    Template(
        name="weakening_right",
        latex=(
            r"\dfrac{{\Gamma \vdash \Delta}}"
            r"{{\Gamma \vdash \Delta,\,{pp}}} (\mathrm{{W}}_R)"
        ),
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="contraction_left",
        latex=(
            r"\dfrac{{{pp},\,{pp},\,\Gamma \vdash \Delta}}"
            r"{{{pp},\,\Gamma \vdash \Delta}} (\mathrm{{C}}_L)"
        ),
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="contraction_right",
        latex=(
            r"\dfrac{{\Gamma \vdash \Delta,\,{pp},\,{pp}}}"
            r"{{\Gamma \vdash \Delta,\,{pp}}} (\mathrm{{C}}_R)"
        ),
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="exchange_left",
        latex=(
            r"\dfrac{{\Gamma,\,{pp},\,{qq},\,\Delta \vdash \Theta}}"
            r"{{\Gamma,\,{qq},\,{pp},\,\Delta \vdash \Theta}} (\mathrm{{E}}_L)"
        ),
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
]

# ---------------------------------------------------------------------------
# Part J: LK Connective Rules (12)
# ---------------------------------------------------------------------------

_TEMPLATES_J: list[Template] = [
    Template(
        name="conjunction_left",
        latex=(
            r"\dfrac{{{pp},\,\Gamma \vdash \Delta}}"
            r"{{{pp} \land {qq},\,\Gamma \vdash \Delta}} (\land L)"
        ),
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="conjunction_right_lk",
        latex=(
            r"\dfrac{{\Gamma \vdash {pp},\,\Delta \quad \Gamma \vdash {qq},\,\Delta}}"
            r"{{\Gamma \vdash {pp} \land {qq},\,\Delta}} (\land R)"
        ),
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="disjunction_left_lk",
        latex=(
            r"\dfrac{{{pp},\,\Gamma \vdash \Delta \quad {qq},\,\Gamma \vdash \Delta}}"
            r"{{{pp} \lor {qq},\,\Gamma \vdash \Delta}} (\lor L)"
        ),
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="disjunction_right_lk",
        latex=(
            r"\dfrac{{\Gamma \vdash {pp},\,\Delta}}"
            r"{{\Gamma \vdash {pp} \lor {qq},\,\Delta}} (\lor R)"
        ),
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="implication_left_lk",
        latex=(
            r"\dfrac{{\Gamma \vdash {pp},\,\Delta \quad {qq},\,\Gamma \vdash \Delta}}"
            r"{{{pp} \Rightarrow {qq},\,\Gamma \vdash \Delta}} (\Rightarrow L)"
        ),
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="implication_right_lk",
        latex=(
            r"\dfrac{{\Gamma,\,{pp} \vdash {qq},\,\Delta}}"
            r"{{\Gamma \vdash {pp} \Rightarrow {qq},\,\Delta}} (\Rightarrow R)"
        ),
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="negation_left_lk",
        latex=(
            r"\dfrac{{\Gamma \vdash {pp},\,\Delta}}"
            r"{{\neg {pp},\,\Gamma \vdash \Delta}} (\neg L)"
        ),
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="negation_right_lk",
        latex=(
            r"\dfrac{{{pp},\,\Gamma \vdash \Delta}}"
            r"{{\Gamma \vdash \neg {pp},\,\Delta}} (\neg R)"
        ),
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="forall_left_lk",
        latex=(
            r"\dfrac{{{pp}({tt}),\,\Gamma \vdash \Delta}}"
            r"{{\forall {xx}\,{pp}({xx}),\,\Gamma \vdash \Delta}} (\forall L)"
        ),
        slots={"pp": S(_PRED_POOL), "tt": S(_TERM_POOL), "xx": S(_VAR_POOL)},
    ),
    Template(
        name="forall_right_lk",
        latex=(
            r"\dfrac{{\Gamma \vdash {pp}({yy}),\,\Delta}}"
            r"{{\Gamma \vdash \forall {xx}\,{pp}({xx}),\,\Delta}} (\forall R,\;{yy}\text{{ fresh}})"
        ),
        slots={
            "pp": S(_PRED_POOL),
            "xx": S(_VAR_POOL),
            "yy": X(_VAR_POOL, ("xx",)),
        },
    ),
    Template(
        name="exists_left_lk",
        latex=(
            r"\dfrac{{{pp}({yy}),\,\Gamma \vdash \Delta}}"
            r"{{\exists {xx}\,{pp}({xx}),\,\Gamma \vdash \Delta}} (\exists L,\;{yy}\text{{ fresh}})"
        ),
        slots={
            "pp": S(_PRED_POOL),
            "xx": S(_VAR_POOL),
            "yy": X(_VAR_POOL, ("xx",)),
        },
    ),
    Template(
        name="exists_right_lk",
        latex=(
            r"\dfrac{{\Gamma \vdash {pp}({tt}),\,\Delta}}"
            r"{{\Gamma \vdash \exists {xx}\,{pp}({xx}),\,\Delta}} (\exists R)"
        ),
        slots={"pp": S(_PRED_POOL), "tt": S(_TERM_POOL), "xx": S(_VAR_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part K: Nested / Multi-Level Proof Trees (4)
# ---------------------------------------------------------------------------

_TEMPLATES_K: list[Template] = [
    Template(
        name="derivation_ellipsis",
        latex=r"\dfrac{{\vdots}}{{\Gamma \vdash {pp}}}",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="nested_axiom_weakening",
        latex=(
            r"\dfrac{{\dfrac{{}}{{{pp} \vdash {pp}}}}}"
            r"{{{pp},\,{qq} \vdash {pp}}} (\mathrm{{W}}_L)"
        ),
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="nested_conjunction_proof",
        latex=(
            r"\dfrac{{\dfrac{{}}{{{pp} \vdash {pp}}}"
            r" \quad \dfrac{{}}{{{qq} \vdash {qq}}}}}"
            r"{{{pp},\,{qq} \vdash {pp} \land {qq}}} (\land I)"
        ),
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="nested_cut_derivation",
        latex=(
            r"\dfrac{{\dfrac{{\vdots}}{{\Gamma \vdash {pp}}}"
            r" \quad \dfrac{{\vdots}}{{{pp},\,\Delta \vdash {qq}}}}}"
            r"{{\Gamma,\,\Delta \vdash {qq}}} (\text{{Cut}})"
        ),
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
]

# ---------------------------------------------------------------------------
# Part L: Dependent Sum Type Σ (5)
# ---------------------------------------------------------------------------

_TEMPLATES_L: list[Template] = [
    Template(
        name="dependent_sum_formation",
        latex=(
            r"\dfrac{{\Gamma \vdash {sa}\ \mathsf{{type}}"
            r" \quad \Gamma,\,{xx}:{sa} \vdash {ta}\ \mathsf{{type}}}}"
            r"{{\Gamma \vdash \textstyle\sum_{{{xx}:{sa}}} {ta}\ \mathsf{{type}}}}"
        ),
        slots={
            "xx": S(_VAR_POOL),
            "sa": S(_TYPE_POOL),
            "ta": X(_TYPE_POOL, ("sa",)),
        },
    ),
    Template(
        name="dependent_sum_intro",
        latex=(
            r"({aa},{bb}) : \textstyle\sum_{{{xx}:{sa}}} {ta}({xx}),"
            r"\quad {aa}:{sa},\;{bb}:{ta}({aa})"
        ),
        slots={
            "xx": S(_VAR_POOL),
            "sa": S(_TYPE_POOL),
            "ta": X(_TYPE_POOL, ("sa",)),
            "aa": S(_PAIR_POOL),
            "bb": X(_PAIR_POOL, ("aa",)),
        },
    ),
    Template(
        name="dependent_sum_elim_fst",
        latex=(
            r"\dfrac{{\Gamma \vdash {pp}:\textstyle\sum_{{{xx}:{sa}}}{ta}({xx})}}"
            r"{{\Gamma \vdash \pi_1\,{pp}:{sa}}}"
        ),
        slots={
            "xx": S(_VAR_POOL),
            "sa": S(_TYPE_POOL),
            "ta": X(_TYPE_POOL, ("sa",)),
            "pp": S(_PAIR_POOL),
        },
    ),
    Template(
        name="dependent_sum_elim_snd",
        latex=(
            r"\dfrac{{\Gamma \vdash {pp}:\textstyle\sum_{{{xx}:{sa}}}{ta}({xx})}}"
            r"{{\Gamma \vdash \pi_2\,{pp}:{ta}(\pi_1\,{pp})}}"
        ),
        slots={
            "xx": S(_VAR_POOL),
            "sa": S(_TYPE_POOL),
            "ta": X(_TYPE_POOL, ("sa",)),
            "pp": S(_PAIR_POOL),
        },
    ),
    Template(
        name="sigma_curry_howard",
        latex=(
            r"\textstyle\sum_{{{xx}:{sa}}} {ta}({xx})"
            r" \cong \exists {xx}:{sa}.\;{ta}({xx})"
        ),
        slots={
            "xx": S(_VAR_POOL),
            "sa": S(_TYPE_POOL),
            "ta": X(_TYPE_POOL, ("sa",)),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part M: Identity Types / Propositional Equality (5)
# ---------------------------------------------------------------------------

_TEMPLATES_M: list[Template] = [
    Template(
        name="id_type_def",
        latex=r"{aa} =_{{{sa}}} {bb}",
        slots={
            "sa": S(_TYPE_POOL),
            "aa": S(_TERM_POOL),
            "bb": X(_TERM_POOL, ("aa",)),
        },
    ),
    Template(
        name="id_type_refl",
        latex=(
            r"\dfrac{{\Gamma \vdash {aa}:{sa}}}"
            r"{{\Gamma \vdash \mathsf{{refl}}_{{{aa}}}:{aa}=_{{{sa}}}{aa}}}"
        ),
        slots={"sa": S(_TYPE_POOL), "aa": S(_TERM_POOL)},
    ),
    Template(
        name="path_induction_j",
        latex=(
            r"\dfrac{{\Gamma \vdash {pp}:{aa}=_{{{sa}}}{aa}}}"
            r"{{\Gamma \vdash \mathsf{{J}}({pp}):C({aa},\mathsf{{refl}}_{{{aa}}})}}"
        ),
        slots={"sa": S(_TYPE_POOL), "aa": S(_TERM_POOL), "pp": X(_TERM_POOL, ("aa",))},
    ),
    Template(
        name="transport",
        latex=(
            r"\mathsf{{transport}}^{{P}}_{{{pp}}} : P({aa}) \to P({bb}),"
            r"\quad {pp}:{aa}=_{{{sa}}}{bb}"
        ),
        slots={
            "sa": S(_TYPE_POOL),
            "aa": S(_TERM_POOL),
            "bb": X(_TERM_POOL, ("aa",)),
            "pp": X(_TERM_POOL, ("aa", "bb")),
        },
    ),
    Template(
        name="function_ap",
        latex=(
            r"\mathsf{{ap}}_{{{ff}}}({pp}) : {ff}({aa}) =_{{{ta}}} {ff}({bb}),"
            r"\quad {pp}:{aa}=_{{{sa}}}{bb}"
        ),
        slots={
            "sa": S(_TYPE_POOL),
            "ta": X(_TYPE_POOL, ("sa",)),
            "aa": S(_TERM_POOL),
            "bb": X(_TERM_POOL, ("aa",)),
            "ff": X(_TERM_POOL, ("aa", "bb")),
            "pp": X(_TERM_POOL, ("aa", "bb", "ff")),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part N: Universe Levels and Type Judgment Forms (4)
# ---------------------------------------------------------------------------

_TEMPLATES_N: list[Template] = [
    Template(
        name="universe_typing",
        latex=r"{sa} : \mathcal{{U}}_{{{ii}}}",
        slots={"sa": S(_TYPE_POOL), "ii": S(_UNIV_POOL)},
    ),
    Template(
        name="universe_cumulative",
        latex=r"\mathcal{{U}}_{{{ii}}} : \mathcal{{U}}_{{{jj}}}",
        slots={"ii": S(_UNIV_POOL), "jj": X(_UNIV_POOL, ("ii",))},
    ),
    Template(
        name="type_judgment_form",
        latex=r"\Gamma \vdash {sa}\ \mathsf{{type}}",
        slots={"sa": S(_TYPE_POOL)},
    ),
    Template(
        name="context_formation",
        latex=(
            r"\dfrac{{\Gamma\ \mathsf{{ctx}} \quad \Gamma \vdash {sa}\ \mathsf{{type}}}}"
            r"{{(\Gamma,\,{xx}:{sa})\ \mathsf{{ctx}}}}"
        ),
        slots={"sa": S(_TYPE_POOL), "xx": S(_VAR_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part O: System F Polymorphism (5)
# ---------------------------------------------------------------------------

_TEMPLATES_O: list[Template] = [
    Template(
        name="systemf_type_abs",
        latex=r"\Lambda\alpha.\,{ee}",
        slots={"ee": S(_TERM_POOL)},
    ),
    Template(
        name="systemf_type_app",
        latex=r"{ee}\,[{sa}]",
        slots={"ee": S(_TERM_POOL), "sa": S(_TYPE_POOL)},
    ),
    Template(
        name="systemf_forall_intro",
        latex=(
            r"\dfrac{{\Gamma,\,\alpha\ \mathsf{{type}} \vdash {ee}:{ta}}}"
            r"{{\Gamma \vdash \Lambda\alpha.\,{ee}:\forall\alpha.\,{ta}}} (\forall I)"
        ),
        slots={"ee": S(_TERM_POOL), "ta": S(_TYPE_POOL)},
    ),
    Template(
        name="systemf_forall_elim",
        latex=(
            r"\dfrac{{\Gamma \vdash {ee}:\forall\alpha.\,{ta}}}"
            r"{{\Gamma \vdash {ee}\,[{sa}]:[{sa}/\alpha]{ta}}} (\forall E)"
        ),
        slots={"ee": S(_TERM_POOL), "sa": S(_TYPE_POOL), "ta": X(_TYPE_POOL, ("sa",))},
    ),
    Template(
        name="systemf_polymorphic_id",
        latex=(
            r"\mathbf{{id}} \equiv \Lambda\alpha.\,\lambda {xx}:\alpha.\,{xx}"
            r" : \forall\alpha.\,\alpha\to\alpha"
        ),
        slots={"xx": S(_VAR_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part P: Linear Logic (8)
# ---------------------------------------------------------------------------

_TEMPLATES_P: list[Template] = [
    Template(
        name="linear_impl_def",
        latex=r"{aa} \multimap {bb}",
        slots={"aa": S(_LL_POOL), "bb": X(_LL_POOL, ("aa",))},
    ),
    Template(
        name="tensor_right",
        latex=(
            r"\dfrac{{\Gamma \vdash {aa} \quad \Delta \vdash {bb}}}"
            r"{{\Gamma,\Delta \vdash {aa} \otimes {bb}}} (\otimes R)"
        ),
        slots={"aa": S(_LL_POOL), "bb": X(_LL_POOL, ("aa",))},
    ),
    Template(
        name="tensor_left",
        latex=(
            r"\dfrac{{\Gamma,{aa},{bb} \vdash \Delta}}"
            r"{{\Gamma,{aa} \otimes {bb} \vdash \Delta}} (\otimes L)"
        ),
        slots={"aa": S(_LL_POOL), "bb": X(_LL_POOL, ("aa",))},
    ),
    Template(
        name="linear_impl_right",
        latex=(
            r"\dfrac{{\Gamma,{aa} \vdash {bb}}}"
            r"{{\Gamma \vdash {aa} \multimap {bb}}} (\multimap R)"
        ),
        slots={"aa": S(_LL_POOL), "bb": X(_LL_POOL, ("aa",))},
    ),
    Template(
        name="linear_impl_left",
        latex=(
            r"\dfrac{{\Gamma \vdash {aa} \quad {bb},\Delta \vdash \Theta}}"
            r"{{\Gamma,{aa} \multimap {bb},\Delta \vdash \Theta}} (\multimap L)"
        ),
        slots={"aa": S(_LL_POOL), "bb": X(_LL_POOL, ("aa",))},
    ),
    Template(
        name="bang_promotion",
        latex=(r"\dfrac{{!\Gamma \vdash {aa}}}{{!\Gamma \vdash !{aa}}} (!)"),
        slots={"aa": S(_LL_POOL)},
    ),
    Template(
        name="bang_dereliction",
        latex=(
            r"\dfrac{{\Gamma,{aa} \vdash \Delta}}"
            r"{{\Gamma,!{aa} \vdash \Delta}} (\mathsf{{d}})"
        ),
        slots={"aa": S(_LL_POOL)},
    ),
    Template(
        name="additive_with_right",
        latex=(
            r"\dfrac{{\Gamma \vdash {aa} \quad \Gamma \vdash {bb}}}"
            r"{{\Gamma \vdash {aa} \mathbin{{\&}} {bb}}} (\mathbin{{\&}} R)"
        ),
        slots={"aa": S(_LL_POOL), "bb": X(_LL_POOL, ("aa",))},
    ),
]

# ---------------------------------------------------------------------------
# Part Q: Non-Provability + Left-Turnstile Symbols (4)
# ---------------------------------------------------------------------------

_TEMPLATES_Q: list[Template] = [
    Template(
        name="nvdash_formula",
        latex=r"\Gamma \nvdash {pp}",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="nvdash_model_formula",
        latex=r"\mathcal{{M}} \nvDash {pp}",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="dashv_biderivability",
        latex=r"{pp} \dashv\vdash {qq}",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="quantifier_adjunction",
        latex=r"\exists \dashv \Delta \dashv \forall",
        slots={},
    ),
]

# ---------------------------------------------------------------------------
# Part R: Typed Lambda Binders (4)
# ---------------------------------------------------------------------------

_TEMPLATES_R: list[Template] = [
    Template(
        name="typed_beta_reduction",
        latex=(
            r"(\lambda {xx}:{sa}.\,{ee1})\,{ee2}"
            r" \to_\beta [{ee2}/{xx}]\,{ee1}"
        ),
        slots={
            "xx": S(_VAR_POOL),
            "sa": S(_TYPE_POOL),
            "ee1": S(_TERM_POOL),
            "ee2": X(_TERM_POOL, ("ee1",)),
        },
    ),
    Template(
        name="typed_lambda_annotation",
        latex=r"\lambda ({xx}:{sa}).\,{ee} : {sa} \to {ta}",
        slots={
            "xx": S(_VAR_POOL),
            "sa": S(_TYPE_POOL),
            "ta": X(_TYPE_POOL, ("sa",)),
            "ee": S(_TERM_POOL),
        },
    ),
    Template(
        name="typed_abstraction_rule",
        latex=(
            r"\dfrac{{\Gamma,\,{xx}:{sa} \vdash {ee}:{ta}}}"
            r"{{\Gamma \vdash (\lambda {xx}:{sa}.\,{ee}):{sa}\to {ta}}}"
        ),
        slots={
            "xx": S(_VAR_POOL),
            "sa": S(_TYPE_POOL),
            "ta": X(_TYPE_POOL, ("sa",)),
            "ee": S(_TERM_POOL),
        },
    ),
    Template(
        name="let_binding",
        latex=(r"\mathsf{{let}}\;{xx}:{sa} = {ee1}\;\mathsf{{in}}\;{ee2} : {ta}"),
        slots={
            "xx": S(_VAR_POOL),
            "sa": S(_TYPE_POOL),
            "ta": X(_TYPE_POOL, ("sa",)),
            "ee1": S(_TERM_POOL),
            "ee2": X(_TERM_POOL, ("ee1",)),
        },
    ),
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TEMPLATES: list[Template] = (
    _TEMPLATES_B3
    + _TEMPLATES_B6
    + _TEMPLATES_H
    + _TEMPLATES_I
    + _TEMPLATES_J
    + _TEMPLATES_K
    + _TEMPLATES_L
    + _TEMPLATES_M
    + _TEMPLATES_N
    + _TEMPLATES_O
    + _TEMPLATES_P
    + _TEMPLATES_Q
    + _TEMPLATES_R
)

GENERATORS, WEIGHTS, TEMPLATES = register_domain("proof_theory", _TEMPLATES)
