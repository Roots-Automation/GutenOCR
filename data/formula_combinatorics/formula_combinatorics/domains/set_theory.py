"""Set theory domain generators."""

from __future__ import annotations

from .._template_dsl import _FN_SLOT, _LIM_MOD, S, Template, X
from ._config import register_domain

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_SET_POOL = ("A", "B", "C", "D", "E", "S", "T", "U", "V", "W")  # 10
_SCRIPT_POOL = (
    r"\mathcal{A}",
    r"\mathcal{B}",
    r"\mathcal{C}",
    r"\mathcal{F}",
    r"\mathcal{G}",
    r"\mathcal{U}",
)  # 6
_ORD_POOL = (
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\delta",
    r"\lambda",
    r"\mu",
    r"\nu",
    r"\omega",
)  # 8
_CARD_POOL = (
    r"\aleph_0",
    r"\aleph_1",
    r"\aleph_2",
    r"\kappa",
    r"\lambda",
    r"\mu",
)  # 6
_VAR_POOL = ("x", "y", "z", "a", "b", "c", "u", "v", "w")  # 9
_PROP_POOL = ("P", "Q", "R", r"\varphi", r"\psi", "S", "T", "U")  # 8
_FUNC_POOL = ("f", "g", "h", r"\varphi", r"\psi", r"\phi", r"\xi", r"\eta")  # 8
_IDX_POOL = ("n", "m", "k", "i", "j", "r", "s", "l")  # 8
_BBOLD_POOL = (
    r"\mathbb{N}",
    r"\mathbb{Z}",
    r"\mathbb{Q}",
    r"\mathbb{R}",
    r"\mathbb{C}",
    r"\mathbb{P}",
    r"\mathbb{F}",
    r"\mathbb{H}",
    r"\mathbb{T}",
)  # 9
_REL_POOL = (r"\leq", r"\prec", r"\sqsubseteq", r"\preceq", r"\unlhd")  # 5

# ---------------------------------------------------------------------------
# Part A: reparameterized originals (16)
# ---------------------------------------------------------------------------

_SET_THEORY_TEMPLATES: list[Template] = [
    Template(
        name="commutativity",
        latex=(
            r"{AA} \cup {BB} = {BB} \cup {AA},"
            r"\quad {AA} \cap {CC} = {CC} \cap {AA}"
        ),
        slots={
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
            "CC": X(_SET_POOL, ("AA", "BB")),
        },
    ),
    Template(
        name="subset_intersection",
        latex=r"{AA} \subseteq {BB} \iff {AA} \cap {BB} = {AA}",
        slots={"AA": S(_SET_POOL), "BB": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="inclusion_exclusion_two",
        latex=r"|{AA} \cup {BB}| = |{AA}| + |{BB}| - |{AA} \cap {BB}|",
        slots={"AA": S(_SET_POOL), "BB": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="power_set_size",
        latex=r"|\mathcal{{P}}({AA})| = 2^{{|{AA}|}}",
        slots={"AA": S(_SET_POOL)},
    ),
    Template(
        name="set_difference",
        latex=r"{AA} \setminus {BB} = \{{{vv} \in {AA} \mid {PP}({vv})\}}",
        slots={
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
            "vv": S(_VAR_POOL),
            "PP": S(_PROP_POOL),
        },
    ),
    Template(
        name="de_morgan_union",
        latex=r"({AA} \cup {BB})^c = {AA}^c \cap {BB}^c",
        slots={"AA": S(_SET_POOL), "BB": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="de_morgan_intersection",
        latex=r"({AA} \cap {BB})^c = {AA}^c \cup {BB}^c",
        slots={"AA": S(_SET_POOL), "BB": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="distributive_law",
        latex=(
            r"{AA} \cap ({BB} \cup {CC})"
            r" = ({AA} \cap {BB}) \cup ({AA} \cap {CC})"
        ),
        slots={
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
            "CC": X(_SET_POOL, ("AA", "BB")),
        },
    ),
    Template(
        name="cartesian_product",
        latex=(
            r"{AA} \times {BB}"
            r" = \{{({vv},\,{ww}) \mid {vv} \in {AA},\; {ww} \in {BB}\}}"
        ),
        slots={
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
            "vv": S(_VAR_POOL),
            "ww": X(_VAR_POOL, ("vv",)),
        },
    ),
    Template(
        name="power_of_set",
        latex=r"|{AA}^{{{nn}}}| = |{AA}|^{{{nn}}}",
        slots={"AA": S(_SET_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="symmetric_difference",
        latex=(
            r"{AA} \triangle {BB}"
            r" = ({AA} \setminus {BB}) \cup ({BB} \setminus {AA})"
        ),
        slots={"AA": S(_SET_POOL), "BB": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="set_builder",
        latex=r"\{{{vv} \in {bb} \mid {PP}({vv})\}}",
        slots={"vv": S(_VAR_POOL), "bb": S(_BBOLD_POOL), "PP": S(_PROP_POOL)},
    ),
    Template(
        name="inclusion_exclusion_general",
        latex=(
            r"\left|\bigcup_{{{ii}=1}}^{{{nn}}} {AA}_{{{ii}}}\right|"
            r" = \sum{lim_mod}_{{{ii}}} |{AA}_{{{ii}}}|"
            r" - \sum{lim_mod}_{{{ii} < {jj}}} |{AA}_{{{ii}}} \cap {AA}_{{{jj}}}| + \cdots"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "AA": S(_SET_POOL),
            "nn": S(_IDX_POOL),
            "ii": X(_IDX_POOL, ("nn",)),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
    Template(
        name="set_equality",
        latex=r"{AA} = {BB} \iff {AA} \subseteq {BB} \land {BB} \subseteq {AA}",
        slots={"AA": S(_SET_POOL), "BB": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="cantor_theorem",
        latex=r"|{AA}| < |\mathcal{{P}}({AA})|",
        slots={"AA": S(_SET_POOL)},
    ),
    Template(
        name="continuum_hypothesis",
        latex=r"\aleph_0 < {kk},\quad |\mathbb{{R}}| = 2^{{\aleph_0}}",
        slots={"kk": S(_CARD_POOL)},
    ),
    # -----------------------------------------------------------------------
    # Part B1: set algebra identities (10)
    # -----------------------------------------------------------------------
    Template(
        name="absorption_union",
        latex=r"{AA} \cup ({AA} \cap {BB}) = {AA}",
        slots={"AA": S(_SET_POOL), "BB": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="absorption_intersection",
        latex=r"{AA} \cap ({AA} \cup {BB}) = {AA}",
        slots={"AA": S(_SET_POOL), "BB": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="identity_laws",
        latex=r"{AA} \cup \emptyset = {AA},\quad {AA} \cap {UU} = {AA}",
        slots={"AA": S(_SET_POOL), "UU": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="complement_laws",
        latex=r"{AA} \cup {AA}^c = {UU},\quad {AA} \cap {AA}^c = \emptyset",
        slots={"AA": S(_SET_POOL), "UU": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="double_complement",
        latex=r"({AA}^c)^c = {AA}",
        slots={"AA": S(_SET_POOL)},
    ),
    Template(
        name="distributive_union",
        latex=(
            r"{AA} \cup ({BB} \cap {CC})"
            r" = ({AA} \cup {BB}) \cap ({AA} \cup {CC})"
        ),
        slots={
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
            "CC": X(_SET_POOL, ("AA", "BB")),
        },
    ),
    Template(
        name="associativity_union",
        latex=r"({AA} \cup {BB}) \cup {CC} = {AA} \cup ({BB} \cup {CC})",
        slots={
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
            "CC": X(_SET_POOL, ("AA", "BB")),
        },
    ),
    Template(
        name="associativity_intersection",
        latex=r"({AA} \cap {BB}) \cap {CC} = {AA} \cap ({BB} \cap {CC})",
        slots={
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
            "CC": X(_SET_POOL, ("AA", "BB")),
        },
    ),
    Template(
        name="subset_union",
        latex=r"{AA} \subseteq {BB} \iff {AA} \cup {BB} = {BB}",
        slots={"AA": S(_SET_POOL), "BB": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="subset_transitivity",
        latex=(
            r"{AA} \subseteq {BB} \land {BB} \subseteq {CC}"
            r" \implies {AA} \subseteq {CC}"
        ),
        slots={
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
            "CC": X(_SET_POOL, ("AA", "BB")),
        },
    ),
    # -----------------------------------------------------------------------
    # Part B2: set comprehension / builder notation (6)
    # -----------------------------------------------------------------------
    Template(
        name="set_builder_restriction",
        latex=r"\{{{vv} \in {AA} \mid {PP}({vv})\}}",
        slots={"vv": S(_VAR_POOL), "AA": S(_SET_POOL), "PP": S(_PROP_POOL)},
    ),
    Template(
        name="set_builder_image",
        latex=r"\{{{ff}({vv}) \mid {vv} \in {AA}\}}",
        slots={"ff": S(_FUNC_POOL), "vv": S(_VAR_POOL), "AA": S(_SET_POOL)},
    ),
    Template(
        name="set_builder_product",
        latex=(r"\{{({vv},\,{ww}) \mid {vv} \in {AA},\; {ww} \in {BB}\}}"),
        slots={
            "vv": S(_VAR_POOL),
            "ww": X(_VAR_POOL, ("vv",)),
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
        },
    ),
    Template(
        name="axiom_extensionality",
        latex=(
            r"{AA} = {BB}"
            r" \iff \forall {vv}\,({vv} \in {AA} \iff {vv} \in {BB})"
        ),
        slots={
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
            "vv": S(_VAR_POOL),
        },
    ),
    Template(
        name="axiom_separation",
        latex=r"\{{{vv} \in {AA} \mid {PP}({vv})\}} \subseteq {AA}",
        slots={"vv": S(_VAR_POOL), "AA": S(_SET_POOL), "PP": S(_PROP_POOL)},
    ),
    Template(
        name="axiom_pairing",
        latex=(
            r"\forall {vv}\,\forall {ww}\,\exists {CC},\;"
            r"{vv} \in {CC} \land {ww} \in {CC}"
        ),
        slots={
            "vv": S(_VAR_POOL),
            "ww": X(_VAR_POOL, ("vv",)),
            "CC": S(_SET_POOL),
        },
    ),
    # -----------------------------------------------------------------------
    # Part B3: relations (8)
    # -----------------------------------------------------------------------
    Template(
        name="relation_def",
        latex=r"{RR} \subseteq {AA} \times {BB}",
        slots={"RR": S(_PROP_POOL), "AA": S(_SET_POOL), "BB": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="reflexive_relation",
        latex=r"\forall {vv} \in {AA},\quad {vv}\,{rr}\,{vv}",
        slots={"vv": S(_VAR_POOL), "AA": S(_SET_POOL), "rr": S(_PROP_POOL)},
    ),
    Template(
        name="symmetric_relation",
        latex=r"{vv}\,{rr}\,{ww} \implies {ww}\,{rr}\,{vv}",
        slots={
            "vv": S(_VAR_POOL),
            "ww": X(_VAR_POOL, ("vv",)),
            "rr": S(_PROP_POOL),
        },
    ),
    Template(
        name="transitive_relation",
        latex=(
            r"{vv}\,{rr}\,{ww} \land {ww}\,{rr}\,{zz}"
            r" \implies {vv}\,{rr}\,{zz}"
        ),
        slots={
            "vv": S(_VAR_POOL),
            "ww": X(_VAR_POOL, ("vv",)),
            "zz": X(_VAR_POOL, ("vv", "ww")),
            "rr": S(_PROP_POOL),
        },
    ),
    Template(
        name="equivalence_class",
        latex=r"[{vv}]_{{{rr}}} = \{{{ww} \in {AA} \mid {vv}\,{rr}\,{ww}\}}",
        slots={
            "vv": S(_VAR_POOL),
            "ww": X(_VAR_POOL, ("vv",)),
            "AA": S(_SET_POOL),
            "rr": S(_PROP_POOL),
        },
    ),
    Template(
        name="quotient_set",
        latex=r"{AA}/{rr} = \{{[{vv}]_{{{rr}}} \mid {vv} \in {AA}\}}",
        slots={"AA": S(_SET_POOL), "rr": S(_PROP_POOL), "vv": S(_VAR_POOL)},
    ),
    Template(
        name="partial_order_antisymmetry",
        latex=(
            r"{vv}\,{rl}\,{ww} \land {ww}\,{rl}\,{vv}"
            r" \implies {vv} = {ww}"
        ),
        slots={
            "vv": S(_VAR_POOL),
            "ww": X(_VAR_POOL, ("vv",)),
            "rl": S(_REL_POOL),
        },
    ),
    Template(
        name="well_order",
        latex=(
            r"\forall {SS} \subseteq {AA},\; {SS} \neq \emptyset"
            r" \implies \exists {vv} \in {SS},\;"
            r" \forall {ww} \in {SS},\; {vv}\,{rl}\,{ww}"
        ),
        slots={
            "SS": S(_SET_POOL),
            "AA": X(_SET_POOL, ("SS",)),
            "vv": S(_VAR_POOL),
            "ww": X(_VAR_POOL, ("vv",)),
            "rl": S(_REL_POOL),
        },
    ),
    # -----------------------------------------------------------------------
    # Part B4: functions / maps (8)
    # -----------------------------------------------------------------------
    Template(
        name="function_type",
        latex=r"{ff}: {AA} \to {BB}",
        slots={"ff": S(_FUNC_POOL), "AA": S(_SET_POOL), "BB": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="injection_def",
        latex=r"{ff}({vv}) = {ff}({ww}) \implies {vv} = {ww}",
        slots={
            "ff": S(_FUNC_POOL),
            "vv": S(_VAR_POOL),
            "ww": X(_VAR_POOL, ("vv",)),
        },
    ),
    Template(
        name="surjection_def",
        latex=(
            r"\forall {yy} \in {BB},\;"
            r"\exists {xx} \in {AA},\; {ff}({xx}) = {yy}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
            "xx": S(_VAR_POOL),
            "yy": X(_VAR_POOL, ("xx",)),
        },
    ),
    Template(
        name="function_composition",
        latex=r"({gg} \circ {ff})({vv}) = {gg}({ff}({vv}))",
        slots={
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "vv": S(_VAR_POOL),
        },
    ),
    Template(
        name="function_image",
        latex=r"{ff}({AA}) = \{{{ff}({vv}) \mid {vv} \in {AA}\}}",
        slots={"ff": S(_FUNC_POOL), "AA": S(_SET_POOL), "vv": S(_VAR_POOL)},
    ),
    Template(
        name="function_preimage",
        latex=(
            r"{ff}^{{-1}}({BB})"
            r" = \{{{vv} \in {AA} \mid {ff}({vv}) \in {BB}\}}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
            "vv": S(_VAR_POOL),
        },
    ),
    Template(
        name="bijection_cardinality",
        latex=(
            r"|{AA}| = |{BB}|"
            r" \iff \exists\; {ff}: {AA} \xrightarrow{{\sim}} {BB}"
        ),
        slots={
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
            "ff": S(_FUNC_POOL),
        },
    ),
    Template(
        name="function_inverse",
        latex=(
            r"{ff} \circ {ff}^{{-1}} = \mathrm{{id}}_{{{BB}}},"
            r"\quad {ff}^{{-1}} \circ {ff} = \mathrm{{id}}_{{{AA}}}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
        },
    ),
    # -----------------------------------------------------------------------
    # Part B5: cardinality (8)
    # -----------------------------------------------------------------------
    Template(
        name="cardinality_leq",
        latex=(
            r"|{AA}| \leq |{BB}|"
            r" \iff \exists\; {ff}: {AA} \hookrightarrow {BB}"
        ),
        slots={
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
            "ff": S(_FUNC_POOL),
        },
    ),
    Template(
        name="schroder_bernstein",
        latex=(
            r"|{AA}| \leq |{BB}| \land |{BB}| \leq |{AA}|"
            r" \implies |{AA}| = |{BB}|"
        ),
        slots={"AA": S(_SET_POOL), "BB": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="countable_def",
        latex=(
            r"|{AA}| = \aleph_0"
            r" \iff \exists\; {ff}: \mathbb{{N}} \xrightarrow{{\sim}} {AA}"
        ),
        slots={"AA": S(_SET_POOL), "ff": S(_FUNC_POOL)},
    ),
    Template(
        name="cardinal_addition",
        latex=r"{kk} + {ll} = |{AA} \sqcup {BB}|",
        slots={
            "kk": S(_CARD_POOL),
            "ll": X(_CARD_POOL, ("kk",)),
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
        },
    ),
    Template(
        name="cardinal_multiplication",
        latex=r"{kk} \cdot {ll} = |{AA} \times {BB}|",
        slots={
            "kk": S(_CARD_POOL),
            "ll": X(_CARD_POOL, ("kk",)),
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
        },
    ),
    Template(
        name="cardinal_exponentiation",
        latex=r"{kk}^{{{ll}}} = |\{{{ff}: {BB} \to {AA}\}}|",
        slots={
            "kk": S(_CARD_POOL),
            "ll": X(_CARD_POOL, ("kk",)),
            "ff": S(_FUNC_POOL),
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
        },
    ),
    Template(
        name="aleph_ordering",
        latex=r"\aleph_{{{nn}}} < \aleph_{{{mm}}}",
        slots={"nn": S(_IDX_POOL), "mm": X(_IDX_POOL, ("nn",))},
    ),
    Template(
        name="beth_recursion",
        latex=r"\beth_{{0}} = \aleph_0,\quad \beth_{{{nn}+1}} = 2^{{\beth_{{{nn}}}}}",
        slots={"nn": S(_IDX_POOL)},
    ),
    # -----------------------------------------------------------------------
    # Part B6: ordinals (6)
    # -----------------------------------------------------------------------
    Template(
        name="ordinal_successor",
        latex=r"S({aa}) = {aa} \cup \{{{aa}\}}",
        slots={"aa": S(_ORD_POOL)},
    ),
    Template(
        name="ordinal_addition_noncommutative",
        latex=r"{aa} + {bb} \neq {bb} + {aa}\text{{ (in general)}}",
        slots={"aa": S(_ORD_POOL), "bb": X(_ORD_POOL, ("aa",))},
    ),
    Template(
        name="ordinal_distributive",
        latex=(
            r"{aa} \cdot ({bb} + {cc})"
            r" = {aa} \cdot {bb} + {aa} \cdot {cc}"
        ),
        slots={
            "aa": S(_ORD_POOL),
            "bb": X(_ORD_POOL, ("aa",)),
            "cc": X(_ORD_POOL, ("aa", "bb")),
        },
    ),
    Template(
        name="transfinite_induction",
        latex=(
            r"{PP}(0) \land \forall {aa}\,({PP}({aa}) \to {PP}(S({aa})))"
            r" \implies \forall {aa},\; {PP}({aa})"
        ),
        slots={"PP": S(_PROP_POOL), "aa": S(_ORD_POOL)},
    ),
    Template(
        name="limit_ordinal",
        latex=r"{ll} = \sup\{{{aa} \mid {aa} < {ll}\}}",
        slots={"ll": S(_ORD_POOL), "aa": X(_ORD_POOL, ("ll",))},
    ),
    Template(
        name="ordinal_well_order",
        latex=(
            r"\forall {SS} \subseteq \mathrm{{Ord}},\; {SS} \neq \emptyset"
            r" \implies \exists {aa} \in {SS},\;"
            r" \forall {bb} \in {SS},\; {aa} \leq {bb}"
        ),
        slots={
            "SS": S(_SET_POOL),
            "aa": S(_ORD_POOL),
            "bb": X(_ORD_POOL, ("aa",)),
        },
    ),
    # -----------------------------------------------------------------------
    # Part B7: indexed families (4)
    # -----------------------------------------------------------------------
    Template(
        name="indexed_union",
        latex=r"\bigcup_{{{ii} \in {II}}} {AA}_{{{ii}}}",
        slots={"ii": S(_IDX_POOL), "II": S(_SET_POOL), "AA": X(_SET_POOL, ("II",))},
    ),
    Template(
        name="indexed_intersection",
        latex=r"\bigcap_{{{ii} \in {II}}} {AA}_{{{ii}}}",
        slots={"ii": S(_IDX_POOL), "II": S(_SET_POOL), "AA": X(_SET_POOL, ("II",))},
    ),
    Template(
        name="general_cartesian_product",
        latex=(
            r"\prod{lim_mod}_{{{ii} \in {II}}} {AA}_{{{ii}}}"
            r" = \{{{ff} \mid \forall {ii} \in {II},\;"
            r"{ff}({ii}) \in {AA}_{{{ii}}}\}}"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "ii": S(_IDX_POOL),
            "II": S(_SET_POOL),
            "AA": X(_SET_POOL, ("II",)),
            "ff": S(_FUNC_POOL),
        },
    ),
    Template(
        name="axiom_of_choice",
        latex=(
            r"\forall \{{{AA}_{{{ii}}}\}}_{{{ii} \in {II}}},"
            r"\quad \prod{lim_mod}_{{{ii} \in {II}}} {AA}_{{{ii}}} \neq \emptyset"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "AA": S(_SET_POOL),
            "ii": S(_IDX_POOL),
            "II": X(_SET_POOL, ("AA",)),
        },
    ),
    # -----------------------------------------------------------------------
    # Part B8: ZFC axioms (4)
    # -----------------------------------------------------------------------
    Template(
        name="axiom_union_set",
        latex=(
            r"\forall {AA}\;\exists {BB},\;"
            r"\forall {vv}\,({vv} \in {BB}"
            r" \iff \exists {CC} \in {AA},\; {vv} \in {CC})"
        ),
        slots={
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
            "CC": X(_SET_POOL, ("AA", "BB")),
            "vv": S(_VAR_POOL),
        },
    ),
    Template(
        name="axiom_power_set",
        latex=(
            r"\forall {AA}\;\exists {BB},\;"
            r"\forall {CC}\,({CC} \subseteq {AA} \iff {CC} \in {BB})"
        ),
        slots={
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
            "CC": X(_SET_POOL, ("AA", "BB")),
        },
    ),
    Template(
        name="axiom_infinity",
        latex=(
            r"\exists {SS},\; \emptyset \in {SS}"
            r" \land \forall {vv} \in {SS},\; {vv} \cup \{{{vv}\}} \in {SS}"
        ),
        slots={"SS": S(_SET_POOL), "vv": S(_VAR_POOL)},
    ),
    Template(
        name="axiom_foundation",
        latex=(
            r"\forall {AA} \neq \emptyset,\;"
            r"\exists {vv} \in {AA},\; {vv} \cap {AA} = \emptyset"
        ),
        slots={"AA": S(_SET_POOL), "vv": S(_VAR_POOL)},
    ),
    # -----------------------------------------------------------------------
    # Part B9: filters and ultrafilters (4)
    # -----------------------------------------------------------------------
    Template(
        name="filter_closed_intersection",
        latex=(
            r"{AA} \in \mathcal{{F}} \land {BB} \in \mathcal{{F}}"
            r" \implies {AA} \cap {BB} \in \mathcal{{F}}"
        ),
        slots={"AA": S(_SET_POOL), "BB": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="filter_upward_closed",
        latex=(
            r"{AA} \in \mathcal{{F}} \land {AA} \subseteq {BB}"
            r" \implies {BB} \in \mathcal{{F}}"
        ),
        slots={"AA": S(_SET_POOL), "BB": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="ultrafilter_property",
        latex=(
            r"\forall {BB} \subseteq {XX},"
            r"\quad {BB} \in \mathcal{{F}}"
            r" \lor {XX} \setminus {BB} \in \mathcal{{F}}"
        ),
        slots={"BB": S(_SET_POOL), "XX": X(_SET_POOL, ("BB",))},
    ),
    Template(
        name="principal_filter",
        latex=(
            r"\uparrow\!{AA}"
            r" = \{{{BB} \in \mathcal{{P}}({XX}) \mid {AA} \subseteq {BB}\}}"
        ),
        slots={
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
            "XX": X(_SET_POOL, ("AA", "BB")),
        },
    ),
    # -----------------------------------------------------------------------
    # Part B10: Boolean algebra / lattice (4)
    # -----------------------------------------------------------------------
    Template(
        name="power_set_boolean",
        latex=(
            r"(\mathcal{{P}}({AA}),\,\cup,\,\cap,\,{{}}^c,\,\emptyset,\,{AA})"
            r"\text{{ is a Boolean algebra}}"
        ),
        slots={"AA": S(_SET_POOL)},
    ),
    Template(
        name="symmetric_diff_group",
        latex=(
            r"(\mathcal{{P}}({AA}),\,\triangle)"
            r"\text{{ is an abelian group with identity }}\emptyset"
        ),
        slots={"AA": S(_SET_POOL)},
    ),
    Template(
        name="lattice_complement",
        latex=r"{AA} \vee {AA}^c = \mathbf{{1}},\quad {AA} \wedge {AA}^c = \mathbf{{0}}",
        slots={"AA": S(_SET_POOL)},
    ),
    Template(
        name="lattice_join_meet",
        latex=(
            r"{AA} \vee {BB} = {AA} \cup {BB},"
            r"\quad {AA} \wedge {BB} = {AA} \cap {BB}"
        ),
        slots={"AA": S(_SET_POOL), "BB": X(_SET_POOL, ("AA",))},
    ),
    # -----------------------------------------------------------------------
    # Part C: high-n_eff function-pair templates (8)
    # -----------------------------------------------------------------------
    Template(
        name="bijection_pair",
        latex=r"{fn1}: {AA} \xrightarrow{{\sim}} {BB}",
        slots={
            "fn1": _FN_SLOT,
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
        },
    ),
    Template(
        name="injection_pair",
        latex=r"{fn1} \circ {fn2}: {AA} \hookrightarrow {CC}",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "AA": S(_SET_POOL),
            "CC": X(_SET_POOL, ("AA",)),
        },
    ),
    Template(
        name="composition_pair",
        latex=r"({fn1} \circ {fn2})({vv}) = {fn1}({fn2}({vv}))",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "vv": S(_VAR_POOL),
        },
    ),
    Template(
        name="image_pair",
        latex=r"{fn1}({fn2}({AA})) \subseteq {fn1}({BB})",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
        },
    ),
    Template(
        name="preimage_pair",
        latex=(
            r"({fn1} \circ {fn2})^{{-1}}({AA})"
            r" = {fn2}^{{-1}}({fn1}^{{-1}}({AA}))"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "AA": S(_SET_POOL),
        },
    ),
    Template(
        name="cardinality_pair",
        latex=r"|{fn1}({AA})| \leq |{AA}|",
        slots={"fn1": _FN_SLOT, "AA": S(_SET_POOL)},
    ),
    Template(
        name="product_pair",
        latex=r"|{fn1} \times {fn2}| = |{fn1}| \cdot |{fn2}|",
        slots={"fn1": _FN_SLOT, "fn2": _FN_SLOT},
    ),
    Template(
        name="fixed_point_pair",
        latex=r"\exists {vv} \in {AA},\; {fn1}({vv}) = {vv}",
        slots={
            "fn1": _FN_SLOT,
            "AA": S(_SET_POOL),
            "vv": S(_VAR_POOL),
        },
    ),
]

# Part C additions
_SET_THEORY_TEMPLATES += [
    Template(
        name="fn_powerset_pair",
        latex=r"{fn1}(\mathcal{{P}}({ss})) = {fn2}\!\left(\{{T : T \subseteq {ss}\}}\right)",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "ss": S(_SET_POOL),
        },
    ),
    Template(
        name="fn_cardinal_pair",
        latex=r"{fn1}(|{ss1} \times {ss2}|) = {fn2}(|{ss1}| \cdot |{ss2}|)",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "ss1": S(_SET_POOL),
            "ss2": X(_SET_POOL, ("ss1",)),
        },
    ),
]

_SET_THEORY_TEMPLATES += [
    Template(
        name="complement_superscript_eq",
        latex=r"{AA}^c = \complement_{{{UU}}} {AA}",
        slots={
            "AA": S(_SET_POOL),
            "UU": X(_SET_POOL, ("AA",)),
        },
    ),
    Template(
        name="complement_de_morgan_union",
        latex=(
            r"\complement_{{{UU}}} ({AA} \cup {BB})"
            r" = \complement_{{{UU}}} {AA} \cap \complement_{{{UU}}} {BB}"
        ),
        slots={
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
            "UU": X(_SET_POOL, ("AA", "BB")),
        },
    ),
    Template(
        name="complement_de_morgan_intersection",
        latex=(
            r"\complement_{{{UU}}} ({AA} \cap {BB})"
            r" = \complement_{{{UU}}} {AA} \cup \complement_{{{UU}}} {BB}"
        ),
        slots={
            "AA": S(_SET_POOL),
            "BB": X(_SET_POOL, ("AA",)),
            "UU": X(_SET_POOL, ("AA", "BB")),
        },
    ),
    Template(
        name="gimel_beth_power",
        latex=r"\gimel(\aleph_{{{nn}}}) = 2^{{\aleph_{{{nn}}}}}",
        slots={"nn": S(_IDX_POOL)},
    ),
    Template(
        name="gimel_cofinality",
        latex=r"\gimel(\kappa) = \kappa^{{\mathrm{{cf}}(\kappa)}}",
        slots={},
    ),
    Template(
        name="daleth_recursion",
        latex=r"\daleth_0 = \aleph_0,\quad \daleth_{{n+1}} = 2^{{\daleth_n}}",
        slots={},
    ),
    Template(
        name="daleth_gimel_leq",
        latex=r"\daleth_{{{nn}}} \leq \gimel(\aleph_{{{nn}}})",
        slots={"nn": S(_IDX_POOL)},
    ),
]

_PART_ARROWS: list[Template] = [
    Template(
        name="set_equality_biconditional",
        latex=r"{AA} = {BB} \leftrightarrow ({AA} \subseteq {BB}) \wedge ({BB} \subseteq {AA})",
        slots={"AA": S(_SET_POOL), "BB": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="directed_family_uparrow",
        latex=r"{AA}_\alpha \uparrow {AA}: \quad {AA}_1 \subseteq {AA}_2 \subseteq \cdots,\;\bigcup_\alpha {AA}_\alpha = {AA}",
        slots={"AA": S(_SET_POOL)},
    ),
]
_SET_THEORY_TEMPLATES += _PART_ARROWS

_PART_MISC: list[Template] = [
    Template(
        name="nexists_element",
        latex=r"\nexists\, {xx} \in {AA}:\; {ff}({xx}) = 0",
        slots={"xx": S(_VAR_POOL), "AA": S(_SET_POOL), "ff": S(_FUNC_POOL)},
    ),
    Template(
        name="varnothing_disjoint",
        latex=r"{AA} \cap {BB} = \varnothing \iff {AA} \text{{ and }} {BB} \text{{ are disjoint}}",
        slots={"AA": S(_SET_POOL), "BB": X(_SET_POOL, ("AA",))},
    ),
    Template(
        name="varnothing_membership",
        latex=r"\forall\, {xx}:\; {xx} \notin \varnothing",
        slots={"xx": S(_VAR_POOL)},
    ),
]
_SET_THEORY_TEMPLATES += _PART_MISC

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("set_theory", _SET_THEORY_TEMPLATES)
