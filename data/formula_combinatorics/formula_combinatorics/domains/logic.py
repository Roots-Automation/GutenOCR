"""Logic domain generators."""

from __future__ import annotations

from ..engine._template_dsl import _FN_SLOT, S, Template, X
from ._config import register_domain
from ._logic_vocab import (
    _BBOLD_POOL,
    _CMD_POOL,
    _NUM_POOL,
    _PRED_POOL,
    _PROP_POOL,
    _TERM_POOL,
    _VAR_POOL,
    _WORLD_POOL,
)

# ---------------------------------------------------------------------------
# Part A: Reparameterized original templates (15)
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
# Part E: Lattice Operators (5)
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

# ---------------------------------------------------------------------------
# Part F: Arrow Variations (3)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Part G: Negated Quantifiers and Proof End (4)
# ---------------------------------------------------------------------------

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
# Registry
# ---------------------------------------------------------------------------

_LOGIC_TEMPLATES: list[Template] = (
    _TEMPLATES_A
    + _TEMPLATES_B1
    + _TEMPLATES_B2
    + _TEMPLATES_B4
    + _TEMPLATES_B5
    + _TEMPLATES_B7
    + _TEMPLATES_B8
    + _TEMPLATES_C
    + _TEMPLATES_D
    + _TEMPLATES_E
    + _TEMPLATES_F
    + _TEMPLATES_G
)

GENERATORS, WEIGHTS, TEMPLATES = register_domain("logic", _LOGIC_TEMPLATES)
