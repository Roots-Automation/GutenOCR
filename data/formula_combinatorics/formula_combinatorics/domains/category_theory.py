"""Category theory domain generators."""

from __future__ import annotations

from ..engine._template_dsl import _FN_SLOT, S, Template, X
from ._config import register_domain

# ---------------------------------------------------------------------------
# Shared pools
# ---------------------------------------------------------------------------

_CAT_POOL: tuple[str, ...] = (
    r"\mathcal{A}",
    r"\mathcal{B}",
    r"\mathcal{C}",
    r"\mathcal{D}",
    r"\mathcal{E}",
    r"\mathcal{M}",
    r"\mathcal{N}",
    r"\mathbf{C}",
    r"\mathbf{D}",
    r"\mathbf{E}",
)  # 10

_OBJ_POOL: tuple[str, ...] = (
    "A",
    "B",
    "C",
    "D",
    "E",
    "X",
    "Y",
    "Z",
    "P",
    "Q",
)  # 10

_MOR_POOL: tuple[str, ...] = (
    "f",
    "g",
    "h",
    "k",
    "u",
    "v",
    "p",
    "q",
    r"\phi",
    r"\psi",
    r"\alpha",
    r"\beta",
    r"\gamma",
)  # 13

_FUN_POOL: tuple[str, ...] = (
    "F",
    "G",
    "H",
    "K",
    "L",
    "R",
    r"\mathcal{F}",
    r"\mathcal{G}",
    r"\mathcal{H}",
)  # 9

_NAT_POOL: tuple[str, ...] = (
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\eta",
    r"\epsilon",
    r"\mu",
    r"\nu",
    r"\tau",
    r"\sigma",
)  # 9

_MONAD_POOL: tuple[str, ...] = (
    "T",
    "S",
    "M",
    r"\mathbb{T}",
    r"\mathbb{S}",
    r"\mathbb{M}",
)  # 6

_CT_IDX_POOL: tuple[str, ...] = ("n", "m", "k", "i", "j")  # 5

# ---------------------------------------------------------------------------
# Part A: Basic Category Theory (8 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_A: list[Template] = [
    Template(
        name="composition",
        latex=(
            r"{mor1}: {obj1} \to {obj2},\quad"
            r" {mor2}: {obj2} \to {obj3}"
            r" \;\Rightarrow\; {mor2} \circ {mor1}: {obj1} \to {obj3}"
        ),
        slots={
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
            "obj3": X(_OBJ_POOL, ("obj1", "obj2")),
            "mor1": S(_MOR_POOL),
            "mor2": X(_MOR_POOL, ("mor1",)),
        },
    ),  # n_eff = 10×9×8 × 13×12 = 112,320
    Template(
        name="identity_morphism",
        latex=(
            r"\operatorname{{id}}_{{{obj}}}: {obj} \to {obj}"
            r"\quad \text{{in }}{cat}"
        ),
        slots={
            "obj": S(_OBJ_POOL),
            "cat": S(_CAT_POOL),
        },
    ),  # n_eff = 10×10 = 100
    Template(
        name="associativity_law",
        latex=(
            r"{mor3} \circ ({mor2} \circ {mor1})"
            r" = ({mor3} \circ {mor2}) \circ {mor1}"
        ),
        slots={
            "mor1": S(_MOR_POOL),
            "mor2": X(_MOR_POOL, ("mor1",)),
            "mor3": X(_MOR_POOL, ("mor1", "mor2")),
        },
    ),  # n_eff = 13×12×11 = 1,716
    Template(
        name="left_unit_law",
        latex=(
            r"\operatorname{{id}}_{{{obj2}}} \circ {mor} = {mor}"
            r"\quad ({mor}: {obj1} \to {obj2})"
        ),
        slots={
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
            "mor": S(_MOR_POOL),
        },
    ),  # n_eff = 10×9×13 = 1,170
    Template(
        name="right_unit_law",
        latex=(
            r"{mor} \circ \operatorname{{id}}_{{{obj1}}} = {mor}"
            r"\quad ({mor}: {obj1} \to {obj2})"
        ),
        slots={
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
            "mor": S(_MOR_POOL),
        },
    ),  # n_eff = 10×9×13 = 1,170
    Template(
        name="hom_set",
        latex=r"\mathrm{{Hom}}_{{{cat}}}({obj1},\, {obj2})",
        slots={
            "cat": S(_CAT_POOL),
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
        },
    ),  # n_eff = 10×10×9 = 900
    Template(
        name="isomorphism_def",
        latex=(
            r"{mor1}: {obj1} \xrightarrow{{\sim}} {obj2}"
            r" \;\Leftrightarrow\; \exists\, {mor2}: {obj2} \to {obj1},\;"
            r"{mor2} \circ {mor1} = \operatorname{{id}}_{{{obj1}}},"
            r"\; {mor1} \circ {mor2} = \operatorname{{id}}_{{{obj2}}}"
        ),
        slots={
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
            "mor1": S(_MOR_POOL),
            "mor2": X(_MOR_POOL, ("mor1",)),
        },
    ),  # n_eff = 10×9×13×12 = 14,040
    Template(
        name="monomorphism_def",
        latex=(
            r"{mor1} \circ {mor2} = {mor1} \circ {mor3}"
            r" \;\Rightarrow\; {mor2} = {mor3}"
            r"\quad ({mor1}: {obj1} \to {obj2})"
        ),
        slots={
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
            "mor1": S(_MOR_POOL),
            "mor2": X(_MOR_POOL, ("mor1",)),
            "mor3": X(_MOR_POOL, ("mor1", "mor2")),
        },
    ),  # n_eff = 10×9×13×12×11 = 154,440
]

# ---------------------------------------------------------------------------
# Part B1: Functors (8 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B1: list[Template] = [
    Template(
        name="functor_def",
        latex=r"{fun}: {cat1} \to {cat2}",
        slots={
            "fun": S(_FUN_POOL),
            "cat1": S(_CAT_POOL),
            "cat2": X(_CAT_POOL, ("cat1",)),
        },
    ),  # n_eff = 9×10×9 = 810
    Template(
        name="functor_composition_law",
        latex=(
            r"{fun}({mor2} \circ {mor1})"
            r" = {fun}({mor2}) \circ {fun}({mor1})"
        ),
        slots={
            "fun": S(_FUN_POOL),
            "mor1": S(_MOR_POOL),
            "mor2": X(_MOR_POOL, ("mor1",)),
        },
    ),  # n_eff = 9×13×12 = 1,404
    Template(
        name="functor_identity_law",
        latex=(
            r"{fun}(\operatorname{{id}}_{{{obj}}})"
            r" = \operatorname{{id}}_{{{fun}({obj})}}"
        ),
        slots={
            "fun": S(_FUN_POOL),
            "obj": S(_OBJ_POOL),
        },
    ),  # n_eff = 9×10 = 90
    Template(
        name="contravariant_functor",
        latex=r"{fun}: {cat1}^{{op}} \to {cat2}",
        slots={
            "fun": S(_FUN_POOL),
            "cat1": S(_CAT_POOL),
            "cat2": X(_CAT_POOL, ("cat1",)),
        },
    ),  # n_eff = 9×10×9 = 810
    Template(
        name="hom_functor_covariant",
        latex=r"\mathrm{{Hom}}({obj}, -): {cat} \to \mathbf{{Set}}",
        slots={
            "obj": S(_OBJ_POOL),
            "cat": S(_CAT_POOL),
        },
    ),  # n_eff = 10×10 = 100
    Template(
        name="hom_functor_contravariant",
        latex=r"\mathrm{{Hom}}(-, {obj}): {cat}^{{op}} \to \mathbf{{Set}}",
        slots={
            "obj": S(_OBJ_POOL),
            "cat": S(_CAT_POOL),
        },
    ),  # n_eff = 10×10 = 100
    Template(
        name="faithful_functor_def",
        latex=(
            r"{fun}_{{A,B}}: \mathrm{{Hom}}({obj1},\, {obj2})"
            r" \hookrightarrow \mathrm{{Hom}}({fun}({obj1}),\, {fun}({obj2}))"
        ),
        slots={
            "fun": S(_FUN_POOL),
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
        },
    ),  # n_eff = 9×10×9 = 810
    Template(
        name="full_functor_def",
        latex=(
            r"{fun}_{{A,B}}: \mathrm{{Hom}}({obj1},\, {obj2})"
            r" \twoheadrightarrow \mathrm{{Hom}}({fun}({obj1}),\, {fun}({obj2}))"
        ),
        slots={
            "fun": S(_FUN_POOL),
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
        },
    ),  # n_eff = 9×10×9 = 810
]

# ---------------------------------------------------------------------------
# Part B2: Natural Transformations (7 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B2: list[Template] = [
    Template(
        name="natural_transformation_def",
        latex=r"{nat}: {fun1} \Rightarrow {fun2}",
        slots={
            "nat": S(_NAT_POOL),
            "fun1": S(_FUN_POOL),
            "fun2": X(_FUN_POOL, ("fun1",)),
        },
    ),  # n_eff = 9×9×8 = 648
    Template(
        name="naturality_square",
        latex=(
            r"{fun2}({mor}) \circ {nat}_{{{obj1}}}"
            r" = {nat}_{{{obj2}}} \circ {fun1}({mor})"
        ),
        slots={
            "nat": S(_NAT_POOL),
            "fun1": S(_FUN_POOL),
            "fun2": X(_FUN_POOL, ("fun1",)),
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
            "mor": S(_MOR_POOL),
        },
    ),  # n_eff = 9×9×8×10×9×13 = 758,160
    Template(
        name="natural_isomorphism",
        latex=(
            r"{nat}: {fun1} \xRightarrow{{\sim}} {fun2},\quad"
            r" {nat}_{{{obj}}}: {fun1}({obj}) \xrightarrow{{\sim}} {fun2}({obj})"
        ),
        slots={
            "nat": S(_NAT_POOL),
            "fun1": S(_FUN_POOL),
            "fun2": X(_FUN_POOL, ("fun1",)),
            "obj": S(_OBJ_POOL),
        },
    ),  # n_eff = 9×9×8×10 = 6,480
    Template(
        name="vertical_composition",
        latex=(
            r"{nat1}: {fun1} \Rightarrow {fun2},\;"
            r"{nat2}: {fun2} \Rightarrow {fun3}"
            r" \;\Rightarrow\; {nat2} \circ {nat1}: {fun1} \Rightarrow {fun3}"
        ),
        slots={
            "nat1": S(_NAT_POOL),
            "nat2": X(_NAT_POOL, ("nat1",)),
            "fun1": S(_FUN_POOL),
            "fun2": X(_FUN_POOL, ("fun1",)),
            "fun3": X(_FUN_POOL, ("fun1", "fun2")),
        },
    ),  # n_eff = 9×8×9×8×7 = 36,288
    Template(
        name="horizontal_composition",
        latex=(
            r"{nat2} * {nat1}: {fun3} \circ {fun1}"
            r" \Rightarrow {fun4} \circ {fun2}"
        ),
        slots={
            "nat1": S(_NAT_POOL),
            "nat2": X(_NAT_POOL, ("nat1",)),
            "fun1": S(_FUN_POOL),
            "fun2": X(_FUN_POOL, ("fun1",)),
            "fun3": X(_FUN_POOL, ("fun1", "fun2")),
            "fun4": X(_FUN_POOL, ("fun1", "fun2", "fun3")),
        },
    ),  # n_eff = 9×8×9×8×7×6 = 217,728
    Template(
        name="whiskering_left",
        latex=(
            r"{fun3} \circ {nat}:"
            r" {fun3} \circ {fun1} \Rightarrow {fun3} \circ {fun2}"
        ),
        slots={
            "nat": S(_NAT_POOL),
            "fun1": S(_FUN_POOL),
            "fun2": X(_FUN_POOL, ("fun1",)),
            "fun3": X(_FUN_POOL, ("fun1", "fun2")),
        },
    ),  # n_eff = 9×9×8×7 = 4,536
    Template(
        name="whiskering_right",
        latex=(
            r"{nat} \circ {fun3}:"
            r" {fun1} \circ {fun3} \Rightarrow {fun2} \circ {fun3}"
        ),
        slots={
            "nat": S(_NAT_POOL),
            "fun1": S(_FUN_POOL),
            "fun2": X(_FUN_POOL, ("fun1",)),
            "fun3": X(_FUN_POOL, ("fun1", "fun2")),
        },
    ),  # n_eff = 9×9×8×7 = 4,536
]

# ---------------------------------------------------------------------------
# Part B3: Limits & Colimits (8 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B3: list[Template] = [
    Template(
        name="product_universal_property",
        latex=(
            r"\exists!\, {mor3}: {obj3} \to {obj1} \times {obj2}"
            r"\text{{ s.t. }}"
            r"\pi_1 \circ {mor3} = {mor1},\;"
            r"\pi_2 \circ {mor3} = {mor2}"
        ),
        slots={
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
            "obj3": X(_OBJ_POOL, ("obj1", "obj2")),
            "mor1": S(_MOR_POOL),
            "mor2": X(_MOR_POOL, ("mor1",)),
            "mor3": X(_MOR_POOL, ("mor1", "mor2")),
        },
    ),  # n_eff = 10×9×8×13×12×11 = 1,235,520
    Template(
        name="coproduct_universal_property",
        latex=(
            r"\exists!\, {mor3}: {obj1} \sqcup {obj2} \to {obj3}"
            r"\text{{ s.t. }}"
            r"{mor3} \circ \iota_1 = {mor1},\;"
            r"{mor3} \circ \iota_2 = {mor2}"
        ),
        slots={
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
            "obj3": X(_OBJ_POOL, ("obj1", "obj2")),
            "mor1": S(_MOR_POOL),
            "mor2": X(_MOR_POOL, ("mor1",)),
            "mor3": X(_MOR_POOL, ("mor1", "mor2")),
        },
    ),  # n_eff = 10×9×8×13×12×11 = 1,235,520
    Template(
        name="pullback_def",
        latex=r"{obj4} = {obj1} \times_{{{obj3}}} {obj2}",
        slots={
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
            "obj3": X(_OBJ_POOL, ("obj1", "obj2")),
            "obj4": X(_OBJ_POOL, ("obj1", "obj2", "obj3")),
        },
    ),  # n_eff = 10×9×8×7 = 5,040
    Template(
        name="pushout_def",
        latex=r"{obj4} = {obj1} \sqcup_{{{obj3}}} {obj2}",
        slots={
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
            "obj3": X(_OBJ_POOL, ("obj1", "obj2")),
            "obj4": X(_OBJ_POOL, ("obj1", "obj2", "obj3")),
        },
    ),  # n_eff = 10×9×8×7 = 5,040
    Template(
        name="equalizer_def",
        latex=(
            r"\operatorname{{eq}}({mor1},{mor2})"
            r" \hookrightarrow {obj1}"
            r" \xrightarrow[{mor2}]{{{mor1}}} {obj2}"
        ),
        slots={
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
            "mor1": S(_MOR_POOL),
            "mor2": X(_MOR_POOL, ("mor1",)),
        },
    ),  # n_eff = 10×9×13×12 = 14,040
    Template(
        name="coequalizer_def",
        latex=(
            r"{obj1}"
            r" \xrightarrow[{mor2}]{{{mor1}}} {obj2}"
            r" \xrightarrow{{{mor3}}} \operatorname{{coeq}}({mor1},{mor2})"
        ),
        slots={
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
            "mor1": S(_MOR_POOL),
            "mor2": X(_MOR_POOL, ("mor1",)),
            "mor3": X(_MOR_POOL, ("mor1", "mor2")),
        },
    ),  # n_eff = 10×9×13×12×11 = 154,440
    Template(
        name="terminal_object",
        latex=(
            r"\forall\, {obj} \in {cat},\;"
            r"\exists!\, {mor}: {obj} \to \mathbf{{1}}"
        ),
        slots={
            "obj": S(_OBJ_POOL),
            "cat": S(_CAT_POOL),
            "mor": S(_MOR_POOL),
        },
    ),  # n_eff = 10×10×13 = 1,300
    Template(
        name="initial_object",
        latex=(
            r"\forall\, {obj} \in {cat},\;"
            r"\exists!\, {mor}: \mathbf{{0}} \to {obj}"
        ),
        slots={
            "obj": S(_OBJ_POOL),
            "cat": S(_CAT_POOL),
            "mor": S(_MOR_POOL),
        },
    ),  # n_eff = 10×10×13 = 1,300
]

# ---------------------------------------------------------------------------
# Part B4: Adjunctions (7 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B4: list[Template] = [
    Template(
        name="adjunction_hom_iso",
        latex=(
            r"\mathrm{{Hom}}_{{{cat2}}}({fun1}({obj1}),\, {obj2})"
            r" \cong \mathrm{{Hom}}_{{{cat1}}}({obj1},\, {fun2}({obj2}))"
        ),
        slots={
            "fun1": S(_FUN_POOL),
            "fun2": X(_FUN_POOL, ("fun1",)),
            "cat1": S(_CAT_POOL),
            "cat2": X(_CAT_POOL, ("cat1",)),
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
        },
    ),  # n_eff = 9×8×10×9×10×9 = 583,200
    Template(
        name="unit_natural_transformation",
        latex=(
            r"{nat}: \operatorname{{id}}_{{{cat}}}"
            r" \Rightarrow {fun2} \circ {fun1}"
        ),
        slots={
            "nat": S(_NAT_POOL),
            "cat": S(_CAT_POOL),
            "fun1": S(_FUN_POOL),
            "fun2": X(_FUN_POOL, ("fun1",)),
        },
    ),  # n_eff = 9×10×9×8 = 6,480
    Template(
        name="counit_natural_transformation",
        latex=(
            r"{nat}: {fun1} \circ {fun2}"
            r" \Rightarrow \operatorname{{id}}_{{{cat}}}"
        ),
        slots={
            "nat": S(_NAT_POOL),
            "cat": S(_CAT_POOL),
            "fun1": S(_FUN_POOL),
            "fun2": X(_FUN_POOL, ("fun1",)),
        },
    ),  # n_eff = 9×10×9×8 = 6,480
    Template(
        name="triangle_identity_left",
        latex=(
            r"{nat2}_{{{fun1}({obj})}} \circ {fun1}({nat1}_{{{obj}}})"
            r" = \operatorname{{id}}_{{{fun1}({obj})}}"
        ),
        slots={
            "nat1": S(_NAT_POOL),
            "nat2": X(_NAT_POOL, ("nat1",)),
            "fun1": S(_FUN_POOL),
            "obj": S(_OBJ_POOL),
        },
    ),  # n_eff = 9×8×9×10 = 6,480
    Template(
        name="triangle_identity_right",
        latex=(
            r"{fun2}({nat2}_{{{obj}}}) \circ {nat1}_{{{fun2}({obj})}}"
            r" = \operatorname{{id}}_{{{fun2}({obj})}}"
        ),
        slots={
            "nat1": S(_NAT_POOL),
            "nat2": X(_NAT_POOL, ("nat1",)),
            "fun2": S(_FUN_POOL),
            "obj": S(_OBJ_POOL),
        },
    ),  # n_eff = 9×8×9×10 = 6,480
    Template(
        name="left_adjoint_preserves_colimits",
        latex=(
            r"{fun1} \dashv {fun2}: {cat1} \to {cat2}"
            r" \;\Rightarrow\; {fun1}\text{{ preserves all colimits}}"
        ),
        slots={
            "fun1": S(_FUN_POOL),
            "fun2": X(_FUN_POOL, ("fun1",)),
            "cat1": S(_CAT_POOL),
            "cat2": X(_CAT_POOL, ("cat1",)),
        },
    ),  # n_eff = 9×8×10×9 = 6,480
    Template(
        name="right_adjoint_preserves_limits",
        latex=(
            r"{fun1} \dashv {fun2}: {cat1} \to {cat2}"
            r" \;\Rightarrow\; {fun2}\text{{ preserves all limits}}"
        ),
        slots={
            "fun1": S(_FUN_POOL),
            "fun2": X(_FUN_POOL, ("fun1",)),
            "cat1": S(_CAT_POOL),
            "cat2": X(_CAT_POOL, ("cat1",)),
        },
    ),  # n_eff = 9×8×10×9 = 6,480
]

# ---------------------------------------------------------------------------
# Part B5: Monads (7 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B5: list[Template] = [
    Template(
        name="monad_triple",
        latex=r"({mon}, {nat1}, {nat2})\text{{ monad on }}{cat}",
        slots={
            "mon": S(_MONAD_POOL),
            "nat1": S(_NAT_POOL),
            "nat2": X(_NAT_POOL, ("nat1",)),
            "cat": S(_CAT_POOL),
        },
    ),  # n_eff = 6×9×8×10 = 4,320
    Template(
        name="monad_multiplication",
        latex=r"{nat}: {mon}^2 \Rightarrow {mon}",
        slots={
            "nat": S(_NAT_POOL),
            "mon": S(_MONAD_POOL),
        },
    ),  # n_eff = 9×6 = 54
    Template(
        name="monad_unit_nat",
        latex=r"{nat}: \operatorname{{id}}_{{{cat}}} \Rightarrow {mon}",
        slots={
            "nat": S(_NAT_POOL),
            "mon": S(_MONAD_POOL),
            "cat": S(_CAT_POOL),
        },
    ),  # n_eff = 9×6×10 = 540
    Template(
        name="monad_associativity",
        latex=r"{nat} \circ {mon} {nat} = {nat} \circ {nat} {mon}",
        slots={
            "nat": S(_NAT_POOL),
            "mon": S(_MONAD_POOL),
        },
    ),  # n_eff = 9×6 = 54
    Template(
        name="monad_unit_left",
        latex=r"{nat1} \circ {mon}{nat2} = \operatorname{{id}}_{{{mon}}}",
        slots={
            "nat1": S(_NAT_POOL),
            "nat2": X(_NAT_POOL, ("nat1",)),
            "mon": S(_MONAD_POOL),
        },
    ),  # n_eff = 9×8×6 = 432
    Template(
        name="kleisli_composition",
        latex=(
            r"{mor2} \star {mor1}"
            r" = {nat}_{{{obj}}} \circ {mon}({mor2}) \circ {mor1}"
        ),
        slots={
            "nat": S(_NAT_POOL),
            "mon": S(_MONAD_POOL),
            "mor1": S(_MOR_POOL),
            "mor2": X(_MOR_POOL, ("mor1",)),
            "obj": S(_OBJ_POOL),
        },
    ),  # n_eff = 9×6×13×12×10 = 84,240
    Template(
        name="eilenberg_moore_algebra",
        latex=(
            r"({mon}({obj}),\; {nat}: {mon}({obj}) \to {obj})"
            r"\text{{ is a }}{mon}\text{{-algebra}}"
        ),
        slots={
            "mon": S(_MONAD_POOL),
            "obj": S(_OBJ_POOL),
            "nat": S(_NAT_POOL),
        },
    ),  # n_eff = 6×10×9 = 540
]

# ---------------------------------------------------------------------------
# Part B6: Abelian Categories (7 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B6: list[Template] = [
    Template(
        name="short_exact_sequence",
        latex=(
            r"0 \to {obj1}"
            r" \xrightarrow{{{mor1}}} {obj2}"
            r" \xrightarrow{{{mor2}}} {obj3} \to 0"
        ),
        slots={
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
            "obj3": X(_OBJ_POOL, ("obj1", "obj2")),
            "mor1": S(_MOR_POOL),
            "mor2": X(_MOR_POOL, ("mor1",)),
        },
    ),  # n_eff = 10×9×8×13×12 = 112,320
    Template(
        name="kernel_def",
        latex=(
            r"\ker({mor}) \hookrightarrow {obj1}"
            r" \xrightarrow{{{mor}}} {obj2}"
        ),
        slots={
            "mor": S(_MOR_POOL),
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
        },
    ),  # n_eff = 13×10×9 = 1,170
    Template(
        name="cokernel_def",
        latex=(
            r"\operatorname{{coker}}({mor}: {obj1} \to {obj2})"
            r" = {obj2} / \operatorname{{im}}({mor})"
        ),
        slots={
            "mor": S(_MOR_POOL),
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
        },
    ),  # n_eff = 13×10×9 = 1,170
    Template(
        name="exactness_at_b",
        latex=(
            r"\operatorname{{im}}({mor1}) = \ker({mor2})"
            r"\quad \text{{in }}\; {obj1} \to {obj2} \to {obj3}"
        ),
        slots={
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
            "obj3": X(_OBJ_POOL, ("obj1", "obj2")),
            "mor1": S(_MOR_POOL),
            "mor2": X(_MOR_POOL, ("mor1",)),
        },
    ),  # n_eff = 10×9×8×13×12 = 112,320
    Template(
        name="splitting_lemma",
        latex=(
            r"\exists\, {mor2}: {obj3} \to {obj2},\;"
            r"{mor1} \circ {mor2} = \operatorname{{id}}_{{{obj3}}}"
        ),
        slots={
            "obj2": S(_OBJ_POOL),
            "obj3": X(_OBJ_POOL, ("obj2",)),
            "mor1": S(_MOR_POOL),
            "mor2": X(_MOR_POOL, ("mor1",)),
        },
    ),  # n_eff = 10×9×13×12 = 14,040
    Template(
        name="long_exact_sequence",
        latex=(
            r"\cdots \to {obj1}"
            r" \xrightarrow{{{mor1}}} {obj2}"
            r" \xrightarrow{{{mor2}}} {obj3}"
            r" \xrightarrow{{{mor3}}} {obj4} \to \cdots"
        ),
        slots={
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
            "obj3": X(_OBJ_POOL, ("obj1", "obj2")),
            "obj4": X(_OBJ_POOL, ("obj1", "obj2", "obj3")),
            "mor1": S(_MOR_POOL),
            "mor2": X(_MOR_POOL, ("mor1",)),
            "mor3": X(_MOR_POOL, ("mor1", "mor2")),
        },
    ),  # n_eff = 10×9×8×7×13×12×11 = 86,486,400
    Template(
        name="connecting_homomorphism",
        latex=(
            r"\delta: \ker({mor2}: {obj2} \to {obj3})"
            r" \to \operatorname{{coker}}({mor1}: {obj1} \to {obj2})"
        ),
        slots={
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
            "obj3": X(_OBJ_POOL, ("obj1", "obj2")),
            "mor1": S(_MOR_POOL),
            "mor2": X(_MOR_POOL, ("mor1",)),
        },
    ),  # n_eff = 10×9×8×13×12 = 112,320
]

# ---------------------------------------------------------------------------
# Part B7: Yoneda & Representability (6 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B7: list[Template] = [
    Template(
        name="yoneda_lemma",
        latex=(
            r"\operatorname{{Nat}}(\mathrm{{Hom}}({obj}, -),\, {fun})"
            r" \cong {fun}({obj})"
        ),
        slots={
            "obj": S(_OBJ_POOL),
            "fun": S(_FUN_POOL),
        },
    ),  # n_eff = 10×9 = 90
    Template(
        name="yoneda_embedding",
        latex=r"{cat} \hookrightarrow [{cat}^{{op}}, \mathbf{{Set}}]",
        slots={
            "cat": S(_CAT_POOL),
        },
    ),  # n_eff = 10
    Template(
        name="representable_functor_def",
        latex=r"{fun} \cong \mathrm{{Hom}}_{{{cat}}}({obj}, -)",
        slots={
            "fun": S(_FUN_POOL),
            "cat": S(_CAT_POOL),
            "obj": S(_OBJ_POOL),
        },
    ),  # n_eff = 9×10×10 = 900
    Template(
        name="yoneda_fully_faithful",
        latex=(
            r"h^{{{obj1}}} \cong h^{{{obj2}}}"
            r" \;\Rightarrow\; {obj1} \cong {obj2}"
            r"\quad \text{{in }}{cat}"
        ),
        slots={
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
            "cat": S(_CAT_POOL),
        },
    ),  # n_eff = 10×9×10 = 900
    Template(
        name="presheaf_category",
        latex=(
            r"\widehat{{{cat}}} = [{cat}^{{op}}, \mathbf{{Set}}],"
            r"\quad {fun} \in \widehat{{{cat}}}"
        ),
        slots={
            "cat": S(_CAT_POOL),
            "fun": S(_FUN_POOL),
        },
    ),  # n_eff = 10×9 = 90
    Template(
        name="coend_formula",
        latex=(
            r"{fun}({obj1})"
            r" \cong \int^{{{obj2}}} {fun}({obj2})"
            r" \times \mathrm{{Hom}}({obj2},\, {obj1})"
        ),
        slots={
            "fun": S(_FUN_POOL),
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
        },
    ),  # n_eff = 9×10×9 = 810
]

# ---------------------------------------------------------------------------
# Part C: High-n_eff function-pair templates (6 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_C: list[Template] = [
    Template(
        name="fn_naturality_square",
        latex=(
            r"{fn1}({fun2}({mor}) \circ {nat}_{{{obj1}}})"
            r" = {fn2}({nat}_{{{obj2}}} \circ {fun1}({mor}))"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "nat": S(_NAT_POOL),
            "fun1": S(_FUN_POOL),
            "fun2": X(_FUN_POOL, ("fun1",)),
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
            "mor": S(_MOR_POOL),
        },
    ),  # n_eff = 100² × 9×9×8×10×9×13 ≈ 75.8B
    Template(
        name="fn_adjunction_unit",
        latex=(
            r"{fn1}({nat}_{{{obj}}})"
            r" = {fn2}({fun2}({fun1}({obj})))"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "nat": S(_NAT_POOL),
            "fun1": S(_FUN_POOL),
            "fun2": X(_FUN_POOL, ("fun1",)),
            "obj": S(_OBJ_POOL),
        },
    ),  # n_eff = 100² × 9×9×8×10 ≈ 6.5B
    Template(
        name="fn_kleisli_bind",
        latex=(
            r"{fn1}({mor2} \star {mor1})"
            r" = {fn2}({nat}_{{{obj}}} \circ {mon}({mor2}) \circ {mor1})"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "nat": S(_NAT_POOL),
            "mon": S(_MONAD_POOL),
            "mor1": S(_MOR_POOL),
            "mor2": X(_MOR_POOL, ("mor1",)),
            "obj": S(_OBJ_POOL),
        },
    ),  # n_eff = 100² × 9×6×13×12×10 ≈ 84.2B
    Template(
        name="fn_exact_image",
        latex=(
            r"{fn1}(\ker({mor2}: {obj2} \to {obj3}))"
            r" = {fn2}(\operatorname{{im}}({mor1}: {obj1} \to {obj2}))"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "obj1": S(_OBJ_POOL),
            "obj2": X(_OBJ_POOL, ("obj1",)),
            "obj3": X(_OBJ_POOL, ("obj1", "obj2")),
            "mor1": S(_MOR_POOL),
            "mor2": X(_MOR_POOL, ("mor1",)),
        },
    ),  # n_eff = 100² × 10×9×8×13×12 ≈ 1.1T
    Template(
        name="fn_functor_composition",
        latex=(
            r"{fn1}({fun}({mor2} \circ {mor1}))"
            r" = {fn2}({fun}({mor2}) \circ {fun}({mor1}))"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "fun": S(_FUN_POOL),
            "mor1": S(_MOR_POOL),
            "mor2": X(_MOR_POOL, ("mor1",)),
        },
    ),  # n_eff = 100² × 9×13×12 ≈ 140.4B
    Template(
        name="fn_yoneda_natural_iso",
        latex=(
            r"{fn1}(\operatorname{{Nat}}(\mathrm{{Hom}}({obj}, -),\, {fun}))"
            r" = {fn2}({fun}({obj}))"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "obj": S(_OBJ_POOL),
            "fun": S(_FUN_POOL),
        },
    ),  # n_eff = 100² × 10×9 = 9B
]

# ---------------------------------------------------------------------------
# Part D: \coprod, \amalg, \bigotimes, \bigodot, \rightsquigarrow, \multimap (10)
# ---------------------------------------------------------------------------

_TEMPLATES_D: list[Template] = [
    Template(
        name="coprod_indexed",
        latex=r"\coprod_{{{ii}}} {AA}_{{{ii}}}",
        slots={"ii": S(_CT_IDX_POOL), "AA": S(_OBJ_POOL)},
    ),
    Template(
        name="amalg_binary",
        latex=r"{aa} \amalg {bb}",
        slots={"aa": S(_OBJ_POOL), "bb": X(_OBJ_POOL, ("aa",))},
    ),
    Template(
        name="coprod_universal",
        latex=r"\exists!\, {uu} : \coprod_{{{ii}}} {AA}_{{{ii}}} \to {BB}",
        slots={
            "ii": S(_CT_IDX_POOL),
            "AA": S(_OBJ_POOL),
            "BB": X(_OBJ_POOL, ("AA",)),
            "uu": S(_MOR_POOL),
        },
    ),
    Template(
        name="bigotimes_indexed",
        latex=r"\bigotimes_{{{ii}=1}}^{{n}} {VV}_{{{ii}}}",
        slots={"ii": S(_CT_IDX_POOL), "VV": S(_OBJ_POOL)},
    ),
    Template(
        name="bigotimes_cat_indexed",
        latex=r"\bigotimes_{{i \in {II}}} {VV}_i",
        slots={"II": S(_CAT_POOL), "VV": S(_OBJ_POOL)},
    ),
    Template(
        name="bigodot_indexed",
        latex=r"\bigodot_{{{ii}}} {FF}_{{{ii}}}",
        slots={"ii": S(_CT_IDX_POOL), "FF": S(_FUN_POOL)},
    ),
    Template(
        name="rightsquigarrow_obj",
        latex=r"{AA} \rightsquigarrow {BB}",
        slots={"AA": S(_OBJ_POOL), "BB": X(_OBJ_POOL, ("AA",))},
    ),
    Template(
        name="rightsquigarrow_fun",
        latex=r"{FF} \rightsquigarrow {GG}",
        slots={"FF": S(_FUN_POOL), "GG": X(_FUN_POOL, ("FF",))},
    ),
    Template(
        name="multimap_internal_hom",
        latex=r"{AA} \multimap {BB} \cong \hom({AA}, {BB})",
        slots={"AA": S(_OBJ_POOL), "BB": X(_OBJ_POOL, ("AA",))},
    ),
    Template(
        name="multimap_currying",
        latex=(
            r"\hom({AA} \otimes {BB}, {CC})"
            r" \cong \hom({AA}, {BB} \multimap {CC})"
        ),
        slots={
            "AA": S(_OBJ_POOL),
            "BB": X(_OBJ_POOL, ("AA",)),
            "CC": X(_OBJ_POOL, ("AA", "BB")),
        },
    ),
]

# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_TEMPLATES_E: list[Template] = [
    Template(
        name="weak_equivalence",
        latex=r"{fun}: {cat1} \xrightarrow{{\;\sim\;}} {cat2},\quad {cat1} \simeq {cat2}",
        slots={"fun": S(_FUN_POOL), "cat1": S(_CAT_POOL), "cat2": X(_CAT_POOL, ("cat1",))},
    ),
    Template(
        name="external_product_functor",
        latex=r"{fun1} \boxtimes {fun2}: {cat1} \times {cat2} \to {cat3}",
        slots={
            "fun1": S(_FUN_POOL),
            "fun2": X(_FUN_POOL, ("fun1",)),
            "cat1": S(_CAT_POOL),
            "cat2": X(_CAT_POOL, ("cat1",)),
            "cat3": X(_CAT_POOL, ("cat1", "cat2")),
        },
    ),
    Template(
        name="boxtimes_symmetry",
        latex=r"{AA} \boxtimes {BB} \simeq {BB} \boxtimes {AA} \quad \text{{in }} {cat}",
        slots={"AA": S(_OBJ_POOL), "BB": X(_OBJ_POOL, ("AA",)), "cat": S(_CAT_POOL)},
    ),
]

_CATTHY_TEMPLATES: list[Template] = (
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
    + _TEMPLATES_E
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("category_theory", _CATTHY_TEMPLATES)
