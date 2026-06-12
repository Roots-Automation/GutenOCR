"""Logic domain generators."""

from __future__ import annotations

from ..engine._template_dsl import _FN_SLOT, _LIM_MOD, S, Template, X
from ..engine._vocab import _BBOLD
from ._config import register_domain

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_PROP_POOL = (
    "P",
    "Q",
    "R",
    "S",
    "A",
    "B",
    r"\phi",
    r"\psi",
    r"\chi",
    r"\varphi",
    r"\alpha",
    r"\beta",
)  # 12 — proposition letters
_PRED_POOL = ("P", "Q", "R", "F", "G", r"\phi", r"\psi", r"\chi", r"\Phi")  # 9 — predicate names
_VAR_POOL = ("x", "y", "z", "a", "b", "c", "u", "v", "w", "n", "m")  # 11 — individual vars
_BBOLD_POOL = tuple(_BBOLD)  # 9 — \mathbb{...}
_WORLD_POOL = ("w", "u", "v", "s", "t", r"\mathcal{W}", r"\mathcal{M}")  # 7 — Kripke worlds
_TYPE_POOL = (r"\sigma", r"\tau", r"\alpha", r"\beta", r"\gamma", "A", "B", "C")  # 8 — types
_TERM_POOL = ("t", "s", "r", "u", "v", "a", "b", "c")  # 8 — lambda terms
_CMD_POOL = ("C", "S", "T", "P", "Q")  # 5 — program commands
_NUM_POOL = ("0", "1", "2", "3", "n", "m")  # 6 — Church numeral indices
_LL_POOL = (r"\alpha", r"\beta", r"\gamma", r"\delta", "A", "B", "C", "D")  # 8 — linear logic formulas
_UNIV_POOL = ("0", "1", "2", r"\omega", "i", "j")  # 6 — universe levels
_PAIR_POOL = ("p", "q", "r", "s", "e", "d")  # 6 — pair / sigma terms

# ---------------------------------------------------------------------------
# Part A: Reparameterized original templates (15)
# Key fix: _PROPS = ["P","Q","R"] (3 items) -> _PROP_POOL (12 items)
# ---------------------------------------------------------------------------

_TEMPLATES_A: list[Template] = [
    Template(
        name="de_morgan_and",
        latex=r"\neg({pp} \land {qq}) \equiv \neg {pp} \lor \neg {qq}",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="de_morgan_or",
        latex=r"\neg({pp} \lor {qq}) \equiv \neg {pp} \land \neg {qq}",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="implication_disjunction",
        latex=r"{pp} \Rightarrow {qq} \equiv \neg {pp} \lor {qq}",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="biconditional",
        latex=r"{pp} \Leftrightarrow {qq} \equiv ({pp} \Rightarrow {qq}) \land ({qq} \Rightarrow {pp})",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="xor",
        latex=r"{pp} \oplus {qq} \equiv ({pp} \lor {qq}) \land \neg({pp} \land {qq})",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="universal_quantifier",
        latex=r"\forall {vv} \in {bb},\; {pp}({vv})",
        slots={"vv": S(_VAR_POOL), "bb": S(_BBOLD_POOL), "pp": S(_PRED_POOL)},
    ),
    Template(
        name="existential_quantifier",
        latex=r"\exists {vv} \in {bb} : {pp}({vv})",
        slots={"vv": S(_VAR_POOL), "bb": S(_BBOLD_POOL), "pp": S(_PRED_POOL)},
    ),
    Template(
        name="associativity_and",
        latex=r"({pp} \land {qq}) \land {rr} \equiv {pp} \land ({qq} \land {rr})",
        slots={
            "pp": S(_PROP_POOL),
            "qq": X(_PROP_POOL, ("pp",)),
            "rr": X(_PROP_POOL, ("pp", "qq")),
        },
    ),
    Template(
        name="distributive_and_or",
        latex=r"{pp} \land ({qq} \lor {rr}) \equiv ({pp} \land {qq}) \lor ({pp} \land {rr})",
        slots={
            "pp": S(_PROP_POOL),
            "qq": X(_PROP_POOL, ("pp",)),
            "rr": X(_PROP_POOL, ("pp", "qq")),
        },
    ),
    Template(
        name="excluded_middle",
        latex=r"{pp} \lor \neg {pp} \equiv \top",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="contradiction",
        latex=r"{pp} \land \neg {pp} \equiv \bot",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="double_negation",
        latex=r"\neg\neg {pp} \equiv {pp}",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="hypothetical_syllogism",
        latex=r"({pp} \Rightarrow {qq}) \land ({qq} \Rightarrow {rr}) \Rightarrow ({pp} \Rightarrow {rr})",
        slots={
            "pp": S(_PROP_POOL),
            "qq": X(_PROP_POOL, ("pp",)),
            "rr": X(_PROP_POOL, ("pp", "qq")),
        },
    ),
    Template(
        name="quantifier_negation",
        latex=r"\neg \forall {vv}\, {pp}({vv}) \equiv \exists {vv}\, \neg {pp}({vv})",
        slots={"vv": S(_VAR_POOL), "pp": S(_PRED_POOL)},
    ),
    Template(
        name="unique_existence",
        latex=r"\exists! {vv} \in {bb} : {pp}({vv})",
        slots={"vv": S(_VAR_POOL), "bb": S(_BBOLD_POOL), "pp": S(_PRED_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B1: Propositional Logic Extended (8)
# ---------------------------------------------------------------------------

_TEMPLATES_B1: list[Template] = [
    Template(
        name="contrapositive",
        latex=r"({pp} \Rightarrow {qq}) \equiv (\neg {qq} \Rightarrow \neg {pp})",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="modus_ponens",
        latex=r"{pp},\; {pp} \Rightarrow {qq} \vdash {qq}",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="modus_tollens",
        latex=r"\neg {qq},\; {pp} \Rightarrow {qq} \vdash \neg {pp}",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="disjunctive_syllogism",
        latex=r"{pp} \lor {qq},\; \neg {pp} \vdash {qq}",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="absorption",
        latex=r"{pp} \lor ({pp} \land {qq}) \equiv {pp}",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="resolution_rule",
        latex=r"({pp} \lor {qq}) \land (\neg {pp} \lor {rr}) \vdash {qq} \lor {rr}",
        slots={
            "pp": S(_PROP_POOL),
            "qq": X(_PROP_POOL, ("pp",)),
            "rr": X(_PROP_POOL, ("pp", "qq")),
        },
    ),
    Template(
        name="exportation",
        latex=r"({pp} \land {qq} \Rightarrow {rr}) \equiv ({pp} \Rightarrow ({qq} \Rightarrow {rr}))",
        slots={
            "pp": S(_PROP_POOL),
            "qq": X(_PROP_POOL, ("pp",)),
            "rr": X(_PROP_POOL, ("pp", "qq")),
        },
    ),
    Template(
        name="pierce_law",
        latex=r"(({pp} \Rightarrow {qq}) \Rightarrow {pp}) \Rightarrow {pp}",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
]

# ---------------------------------------------------------------------------
# Part B2: First-Order Logic (8)
# ---------------------------------------------------------------------------

_TEMPLATES_B2: list[Template] = [
    Template(
        name="universal_instantiation",
        latex=r"\forall {vv}\, {pp}({vv}) \vdash {pp}({tt})",
        slots={
            "pp": S(_PRED_POOL),
            "vv": S(_VAR_POOL),
            "tt": X(_VAR_POOL, ("vv",)),
        },
    ),
    Template(
        name="existential_generalization",
        latex=r"{pp}({tt}) \vdash \exists {vv}\, {pp}({vv})",
        slots={
            "pp": S(_PRED_POOL),
            "tt": S(_TERM_POOL),
            "vv": S(_VAR_POOL),
        },
    ),
    Template(
        name="prenex_universal",
        latex=r"(\forall {vv}\, {pp}({vv})) \land {qq} \equiv \forall {vv}\, ({pp}({vv}) \land {qq})",
        slots={
            "pp": S(_PRED_POOL),
            "vv": S(_VAR_POOL),
            "qq": S(_PROP_POOL),
        },
    ),
    Template(
        name="prenex_existential",
        latex=r"(\exists {vv}\, {pp}({vv})) \lor {qq} \equiv \exists {vv}\, ({pp}({vv}) \lor {qq})",
        slots={
            "pp": S(_PRED_POOL),
            "vv": S(_VAR_POOL),
            "qq": S(_PROP_POOL),
        },
    ),
    Template(
        name="equality_substitution",
        latex=r"{xx} = {yy} \Rightarrow ({pp}({xx}) \Rightarrow {pp}({yy}))",
        slots={
            "xx": S(_VAR_POOL),
            "yy": X(_VAR_POOL, ("xx",)),
            "pp": S(_PRED_POOL),
        },
    ),
    Template(
        name="uniqueness_expansion",
        latex=(
            r"\exists!{vv}\, {pp}({vv}) \equiv \exists {vv}\,"
            r" ({pp}({vv}) \land \forall {uu}\, ({pp}({uu}) \Rightarrow {uu} = {vv}))"
        ),
        slots={
            "pp": S(_PRED_POOL),
            "vv": S(_VAR_POOL),
            "uu": X(_VAR_POOL, ("vv",)),
        },
    ),
    Template(
        name="composition_relation",
        latex=(
            r"({RR} \circ {SS})({xx},{yy})"
            r" \equiv \exists {zz}\, ({RR}({xx},{zz}) \land {SS}({zz},{yy}))"
        ),
        slots={
            "RR": S(_PRED_POOL),
            "SS": X(_PRED_POOL, ("RR",)),
            "xx": S(_VAR_POOL),
            "yy": X(_VAR_POOL, ("xx",)),
            "zz": X(_VAR_POOL, ("xx", "yy")),
        },
    ),
    Template(
        name="compactness_instance",
        latex=(
            r"\Gamma \models {pp}"
            r" \iff \exists \Gamma_0 \subseteq \Gamma,\; |\Gamma_0| < \infty,\; \Gamma_0 \models {pp}"
        ),
        slots={"pp": S(_PROP_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B3: Proof Theory / Sequent Calculus (6)
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
# Part B4: Modal Logic (8)
# ---------------------------------------------------------------------------

_TEMPLATES_B4: list[Template] = [
    Template(
        name="k_axiom",
        latex=r"\Box({pp} \Rightarrow {qq}) \Rightarrow (\Box {pp} \Rightarrow \Box {qq})",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="t_axiom",
        latex=r"\Box {pp} \Rightarrow {pp}",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="b_axiom",
        latex=r"{pp} \Rightarrow \Box\Diamond {pp}",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="s4_axiom",
        latex=r"\Box {pp} \Rightarrow \Box\Box {pp}",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="s5_axiom",
        latex=r"\Diamond {pp} \Rightarrow \Box\Diamond {pp}",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="dual_modal",
        latex=r"\Diamond {pp} \equiv \neg\Box\neg {pp}",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="kripke_satisfaction",
        latex=(
            r"\mathcal{{M}},{ww} \models \Box {pp}"
            r" \iff \forall {vv}\,({ww}\,R\,{vv} \Rightarrow \mathcal{{M}},{vv} \models {pp})"
        ),
        slots={
            "ww": S(_WORLD_POOL),
            "vv": X(_WORLD_POOL, ("ww",)),
            "pp": S(_PROP_POOL),
        },
    ),
    Template(
        name="necessitation",
        latex=r"\vdash {pp} \implies \vdash \Box {pp}",
        slots={"pp": S(_PROP_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B5: Lambda Calculus (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B5: list[Template] = [
    Template(
        name="beta_reduction",
        latex=r"(\lambda {vv}.\,{ee1})\,{ee2} \to_\beta [{ee2}/{vv}]\,{ee1}",
        slots={
            "vv": S(_VAR_POOL),
            "ee1": S(_TERM_POOL),
            "ee2": X(_TERM_POOL, ("ee1",)),
        },
    ),
    Template(
        name="alpha_equivalence",
        latex=r"\lambda {xx}.\,{ee} =_\alpha \lambda {yy}.\,[{yy}/{xx}]\,{ee}",
        slots={
            "xx": S(_VAR_POOL),
            "yy": X(_VAR_POOL, ("xx",)),
            "ee": S(_TERM_POOL),
        },
    ),
    Template(
        name="eta_conversion",
        latex=r"\lambda {xx}.\,{ff}\,{xx} =_\eta {ff} \quad ({xx} \notin \mathrm{{FV}}({ff}))",
        slots={"xx": S(_VAR_POOL), "ff": S(_TERM_POOL)},
    ),
    Template(
        name="church_numeral_def",
        latex=r"\overline{{{nn}}} \equiv \lambda {ff}.\,\lambda {xx}.\,{ff}^{{{nn}}}\,{xx}",
        slots={
            "nn": S(_NUM_POOL),
            "ff": S(_VAR_POOL),
            "xx": X(_VAR_POOL, ("ff",)),
        },
    ),
    Template(
        name="church_succ",
        latex=(
            r"\mathbf{{S}} \equiv"
            r" \lambda {nn}.\,\lambda {ff}.\,\lambda {xx}.\,{ff}\,({nn}\,{ff}\,{xx})"
        ),
        slots={
            "nn": S(_VAR_POOL),
            "ff": X(_VAR_POOL, ("nn",)),
            "xx": X(_VAR_POOL, ("nn", "ff")),
        },
    ),
    Template(
        name="application_assoc",
        latex=r"{ee1}\,{ee2}\,{ee3} \equiv ({ee1}\,{ee2})\,{ee3}",
        slots={
            "ee1": S(_TERM_POOL),
            "ee2": X(_TERM_POOL, ("ee1",)),
            "ee3": X(_TERM_POOL, ("ee1", "ee2")),
        },
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
# Part B7: Boolean Algebra (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B7: list[Template] = [
    Template(
        name="idempotent_meet",
        latex=r"{pp} \land {pp} = {pp}",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="idempotent_join",
        latex=r"{pp} \lor {pp} = {pp}",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="absorption_meet",
        latex=r"{pp} \land ({pp} \lor {qq}) = {pp}",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="absorption_join",
        latex=r"{pp} \lor ({pp} \land {qq}) = {pp}",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="complement_meet",
        latex=r"{pp} \land \neg {pp} = 0",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="shannons_expansion",
        latex=r"{ff}({pp}) = ({pp} \land {ff}(\top)) \lor (\neg {pp} \land {ff}(\bot))",
        slots={"ff": S(_PRED_POOL), "pp": S(_PROP_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B8: Program Logic / Temporal Logic (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B8: list[Template] = [
    Template(
        name="hoare_triple",
        latex=r"\{{{pp}\}}\;{CC}\;\{{{qq}\}}",
        slots={
            "pp": S(_PROP_POOL),
            "qq": X(_PROP_POOL, ("pp",)),
            "CC": S(_CMD_POOL),
        },
    ),
    Template(
        name="hoare_sequence",
        latex=(
            r"\dfrac{{\{{{pp}\}}\,{CC1}\,\{{{rr}\}}"
            r" \quad \{{{rr}\}}\,{CC2}\,\{{{qq}\}}}}"
            r"{{\{{{pp}\}}\,{CC1};\,{CC2}\,\{{{qq}\}}}}"
        ),
        slots={
            "pp": S(_PROP_POOL),
            "qq": X(_PROP_POOL, ("pp",)),
            "rr": X(_PROP_POOL, ("pp", "qq")),
            "CC1": S(_CMD_POOL),
            "CC2": X(_CMD_POOL, ("CC1",)),
        },
    ),
    Template(
        name="weakest_precondition",
        latex=(
            r"\mathrm{{wp}}({CC}, {qq})"
            r" = \text{{weakest}}\;{pp}\;\text{{s.t.}}\;\{{{pp}\}}\,{CC}\,\{{{qq}\}}"
        ),
        slots={
            "pp": S(_PROP_POOL),
            "qq": X(_PROP_POOL, ("pp",)),
            "CC": S(_CMD_POOL),
        },
    ),
    Template(
        name="ltl_always",
        latex=r"\square {pp} \equiv \bigwedge_{{t \geq 0}} {pp}(t)",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="ltl_eventually",
        latex=r"\lozenge {pp} \equiv \bigvee_{{t \geq 0}} {pp}(t)",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="ltl_until",
        latex=r"{pp}\;\mathcal{{U}}\;{qq} \equiv \exists t,\;{qq}(t) \land \forall s < t,\;{pp}(s)",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
]

# ---------------------------------------------------------------------------
# Part C: High-n_eff predicate-pair templates (6)
# Each uses _FN_SLOT — n_eff 110k-11M per template
# ---------------------------------------------------------------------------

_TEMPLATES_C: list[Template] = [
    Template(
        name="predicate_implication_pair",
        latex=r"\forall {vv}\,({fn1}({vv}) \Rightarrow {fn2}({vv}))",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "vv": S(_VAR_POOL),
        },
    ),
    Template(
        name="predicate_equivalence_pair",
        latex=r"\forall {vv}\,({fn1}({vv}) \Leftrightarrow {fn2}({vv}))",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "vv": S(_VAR_POOL),
        },
    ),
    Template(
        name="predicate_conjunction_pair",
        latex=r"\forall {vv}\,({fn1}({vv}) \land {fn2}({vv}))",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "vv": S(_VAR_POOL),
        },
    ),
    Template(
        name="forall_exists_pair",
        latex=r"\forall {vv}\,\exists {uu}\,({fn1}({vv}) \land {fn2}({uu}))",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "vv": S(_VAR_POOL),
            "uu": X(_VAR_POOL, ("vv",)),
        },
    ),
    Template(
        name="binary_predicate_pair",
        latex=r"\forall {vv}\,\forall {uu}\,({fn1}({vv},{uu}) \Rightarrow {fn2}({vv},{uu}))",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "vv": S(_VAR_POOL),
            "uu": X(_VAR_POOL, ("vv",)),
        },
    ),
    Template(
        name="predicate_chain",
        latex=r"\forall {vv}\,({fn1}({vv}) \Rightarrow {fn2}({vv}) \Rightarrow {fn3}({vv}))",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "fn3": _FN_SLOT,
            "vv": S(_VAR_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part D: New symbols — \vDash, \Vdash, \therefore, \because, \barwedge, \veebar (11)
# ---------------------------------------------------------------------------

_TEMPLATES_D: list[Template] = [
    Template(
        name="semantic_entailment_world",
        latex=r"{MM} \vDash {pp}",
        slots={"MM": S(_WORLD_POOL), "pp": S(_PROP_POOL)},
    ),
    Template(
        name="semantic_entailment_conjunction",
        latex=r"{MM} \vDash {pp} \land {qq}",
        slots={"MM": S(_WORLD_POOL), "pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="semantic_entailment_gamma",
        latex=r"\Gamma \vDash {pp}",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="forcing_world",
        latex=r"{ww} \Vdash {pp}",
        slots={"ww": S(_WORLD_POOL), "pp": S(_PROP_POOL)},
    ),
    Template(
        name="forcing_implication",
        latex=r"{ww} \Vdash {pp} \Rightarrow {qq}",
        slots={"ww": S(_WORLD_POOL), "pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="therefore_modus_ponens",
        latex=r"{pp} \land ({pp} \Rightarrow {qq}) \therefore {qq}",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="because_disjunctive_syllogism",
        latex=r"{pp} \because {pp} \lor {qq},\; \neg {qq}",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="barwedge_nand_equiv",
        latex=r"{pp} \barwedge {qq} \equiv \neg({pp} \land {qq})",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="barwedge_nand_identity",
        latex=r"({pp} \barwedge {pp}) \barwedge ({qq} \barwedge {qq}) \equiv {pp} \lor {qq}",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="veebar_xor_via_or_and",
        latex=r"{pp} \veebar {qq} \equiv ({pp} \lor {qq}) \land \neg({pp} \land {qq})",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="veebar_xor_via_asymmetric_and",
        latex=r"{pp} \veebar {qq} \equiv ({pp} \land \neg {qq}) \lor (\neg {pp} \land {qq})",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
]

# ---------------------------------------------------------------------------
# Combine all templates
# ---------------------------------------------------------------------------

_TEMPLATES_E: list[Template] = [
    Template(
        name="lattice_join_def",
        latex=r"{pp} \vee {qq} = \sup\{{{pp},\,{qq}\}}",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="lattice_meet_def",
        latex=r"{pp} \wedge {qq} = \inf\{{{pp},\,{qq}\}}",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="boolean_complement_laws",
        latex=r"{pp} \vee \neg {pp} = \mathbf{{1}},\quad {pp} \wedge \neg {pp} = \mathbf{{0}}",
        slots={"pp": S(_PROP_POOL)},
    ),
    Template(
        name="distributive_lattice_law",
        latex=r"{pp} \vee ({qq} \wedge {rr}) = ({pp} \vee {qq}) \wedge ({pp} \vee {rr})",
        slots={
            "pp": S(_PROP_POOL),
            "qq": X(_PROP_POOL, ("pp",)),
            "rr": X(_PROP_POOL, ("pp", "qq")),
        },
    ),
    Template(
        name="de_morgan_lattice",
        latex=r"\neg({pp} \vee {qq}) = \neg {pp} \wedge \neg {qq}",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
]

_TEMPLATES_F: list[Template] = [
    Template(
        name="biconditional_leftrightarrow",
        latex=r"{pp} \leftrightarrow {qq} \equiv ({pp} \rightarrow {qq}) \wedge ({qq} \rightarrow {pp})",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="converse_leftarrow",
        latex=r"({pp} \leftarrow {qq}) \equiv ({qq} \rightarrow {pp})",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="necessary_condition_leftarrow",
        latex=r"{pp} \Leftarrow {qq} \iff {qq} \Rightarrow {pp}",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
]

_TEMPLATES_G: list[Template] = [
    Template(
        name="nexists_predicate",
        latex=r"\nexists\, {xx}:\; {pp}({xx})",
        slots={"xx": S(_VAR_POOL), "pp": S(_PRED_POOL)},
    ),
    Template(
        name="nexists_as_negation",
        latex=r"\nexists\, {xx}:\; {pp}({xx}) \equiv \forall {xx}\;\neg {pp}({xx})",
        slots={"xx": S(_VAR_POOL), "pp": S(_PRED_POOL)},
    ),
    Template(
        name="proof_end_blacksquare",
        latex=r"\therefore {pp} \Rightarrow {qq} \qquad \blacksquare",
        slots={"pp": S(_PROP_POOL), "qq": X(_PROP_POOL, ("pp",))},
    ),
    Template(
        name="contradiction_blacksquare",
        latex=r"\neg({pp} \land \neg {pp}) \qquad \blacksquare",
        slots={"pp": S(_PROP_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part H: Zero-Premise Axiom Rules (5)
# \dfrac{}{...} — the empty-numerator pattern absent from every prior section
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
# Part I: Missing Structural Rules (4)
# Complement the lone weakening_left in B3
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
# Part J: LK Right-Side and Symmetric Connective Rules (11)
# Existing B3 rules are all single-conclusion (ND-style).
# These add the symmetric classical LK rules with multi-formula sequents.
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
# Part K: Nested / Multi-Level Proof Trees + Derivation Ellipsis (4)
# \dfrac within \dfrac numerator — pattern entirely absent prior to this section
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
# dependent_product (Π) exists in B6; Σ is its dual and was entirely absent
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
# Martin-Löf / HoTT notation — entirely absent prior to this section
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
# Type formation / context formation judgments — absent prior to this section
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
# Type-level abstraction / application — absent prior to this section
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
# Entirely new logical system; symbols: \otimes, \oplus, \multimap, \mathbin{\&}, !, ?
# \parr excluded (non-standard MathJax rendering)
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
# \nvdash, \nvDash, \dashv — absent prior to this section
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
# B5 lambda templates use untyped λx.e; these add type annotations on binders
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

_LOGIC_TEMPLATES: list[Template] = (
    _TEMPLATES_A
    + _TEMPLATES_B1
    + _TEMPLATES_B2
    + _TEMPLATES_B3
    + _TEMPLATES_B4
    + _TEMPLATES_B5
    + _TEMPLATES_B6
    + _TEMPLATES_B7
    + _TEMPLATES_B8
    + _TEMPLATES_C
    + _TEMPLATES_D
    + _TEMPLATES_E
    + _TEMPLATES_F
    + _TEMPLATES_G
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

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("logic", _LOGIC_TEMPLATES)
