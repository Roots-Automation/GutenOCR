"""Ring, field, module, and Galois theory domain generator."""

from __future__ import annotations

from .._template_dsl import _FN_SLOT, _LIM_MOD, S, Template, X
from .._vocab import _ELT_POOL, _HOMO_POOL, _RING_NAMES
from ._config import register_domain

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_RING_POOL = (
    "R",
    "S",
    "A",
    "B",
    r"\mathbb{Z}",
    r"\mathbb{Q}[x]",
    r"\mathbb{Z}[x]",
    r"\mathbb{F}_p",
    r"\mathcal{O}",
    r"\mathbb{Z}[\sqrt{d}]",
)  # 10
_IDEAL_POOL = (
    "I",
    "J",
    r"\mathfrak{m}",
    r"\mathfrak{p}",
    r"\mathfrak{a}",
    r"\mathfrak{b}",
    "P",
    "Q",
    r"\mathfrak{q}",
    r"\mathfrak{n}",
)  # 10
_FIELD_POOL = (
    "K",
    "L",
    "E",
    "F",
    "k",
    r"\mathbb{Q}",
    r"\mathbb{F}_p",
    r"\mathbb{Q}(\sqrt{d})",
    r"\mathbb{F}_{p^n}",
    r"\mathbb{C}",
    r"\mathbb{R}",
)  # 11
_MODULE_POOL = (
    "M",
    "N",
    "V",
    "W",
    r"\mathcal{M}",
    r"\mathcal{N}",
    "P",
    "Q",
    "L",
    "U",
)  # 10
_ELEM_POOL = (
    "a",
    "b",
    "c",
    "x",
    "y",
    "z",
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\xi",
    r"\eta",
)  # 11
_POLY_POOL = ("f", "g", "h", "p", "q", r"\phi", r"\psi", r"\chi")  # 8
_SUBGP_POOL = ("G", "H", "N", "A", "B", r"\Gamma")  # 6

# ---------------------------------------------------------------------------
# Part A: Reparameterized originals (21)
# ---------------------------------------------------------------------------

_TEMPLATES_A: list[Template] = [
    Template(
        name="ideal_quotient",
        latex=r"{II} \trianglelefteq {RR},\quad {RR}/{II}",
        slots={"II": S(_IDEAL_POOL), "RR": S(_RING_POOL)},
    ),
    Template(
        name="maximal_ideal_field",
        latex=r"{RR}/{II} \text{{ field}} \iff {II} \text{{ maximal in }} {RR}",
        slots={"RR": S(_RING_POOL), "II": S(_IDEAL_POOL)},
    ),
    Template(
        name="prime_ideal_domain",
        latex=r"{RR}/{II} \text{{ domain}} \iff {II} \text{{ prime in }} {RR}",
        slots={"RR": S(_RING_POOL), "II": S(_IDEAL_POOL)},
    ),
    Template(
        name="chinese_remainder",
        latex=(
            r"{II} + {JJ} = {RR} \implies"
            r" {RR}/({II} \cap {JJ}) \cong {RR}/{II} \times {RR}/{JJ}"
        ),
        slots={"II": S(_IDEAL_POOL), "JJ": X(_IDEAL_POOL, ("II",)), "RR": S(_RING_POOL)},
    ),
    Template(
        name="crt_embedding",
        latex=r"{RR}/({II} \cap {JJ}) \hookrightarrow {RR}/{II} \times {RR}/{JJ}",
        slots={"RR": S(_RING_POOL), "II": S(_IDEAL_POOL), "JJ": X(_IDEAL_POOL, ("II",))},
    ),
    Template(
        name="ring_homomorphism",
        latex=(
            r"{phi} : {RR} \to {SS},\quad"
            r" {phi}(ab) = {phi}(a){phi}(b),\quad {phi}(1) = 1"
        ),
        slots={"phi": S(_HOMO_POOL), "RR": S(_RING_POOL), "SS": X(_RING_POOL, ("RR",))},
    ),
    Template(
        name="first_isomorphism_ring",
        latex=r"{RR}/\ker {phi} \cong \operatorname{{im}}\, {phi} \subseteq {SS}",
        slots={"phi": S(_HOMO_POOL), "RR": S(_RING_POOL), "SS": X(_RING_POOL, ("RR",))},
    ),
    Template(
        name="tower_law",
        latex=r"[{KK}:{FF}] = [{KK}:{LL}][{LL}:{FF}]",
        slots={
            "KK": S(_FIELD_POOL),
            "LL": X(_FIELD_POOL, ("KK",)),
            "FF": X(_FIELD_POOL, ("KK", "LL")),
        },
    ),
    Template(
        name="extension_dimension",
        latex=r"[{KK}:{FF}] = \dim_{{{FF}}} {KK}",
        slots={"KK": S(_FIELD_POOL), "FF": X(_FIELD_POOL, ("KK",))},
    ),
    Template(
        name="simple_extension_degree",
        latex=r"{KK} = {FF}({g_el}),\quad [{KK}:{FF}] = \deg \min_{{{FF}}}({g_el})",
        slots={
            "KK": S(_FIELD_POOL),
            "FF": X(_FIELD_POOL, ("KK",)),
            "g_el": S(_ELEM_POOL),
        },
    ),
    Template(
        name="galois_group_order",
        latex=r"|\operatorname{{Gal}}({KK}/{FF})| = [{KK}:{FF}]",
        slots={"KK": S(_FIELD_POOL), "FF": X(_FIELD_POOL, ("KK",))},
    ),
    Template(
        name="galois_group_iso",
        latex=r"\operatorname{{Gal}}({KK}/{FF}) \cong {HH}",
        slots={"KK": S(_FIELD_POOL), "FF": X(_FIELD_POOL, ("KK",)), "HH": S(_SUBGP_POOL)},
    ),
    Template(
        name="fixed_field",
        latex=r"{KK}^{{\operatorname{{Gal}}({KK}/{FF})}} = {FF}",
        slots={"KK": S(_FIELD_POOL), "FF": X(_FIELD_POOL, ("KK",))},
    ),
    Template(
        name="galois_correspondence",
        latex=(
            r"\left\{{{HH} \leq \operatorname{{Gal}}({KK}/{FF})\right\}}"
            r" \longleftrightarrow"
            r" \left\{{{LL} : {FF} \subseteq {LL} \subseteq {KK}\right\}}"
        ),
        slots={
            "KK": S(_FIELD_POOL),
            "LL": X(_FIELD_POOL, ("KK",)),
            "FF": X(_FIELD_POOL, ("KK", "LL")),
            "HH": S(_SUBGP_POOL),
        },
    ),
    Template(
        name="tensor_product",
        latex=r"{MM} \otimes_{{{RR}}} {NN}",
        slots={"MM": S(_MODULE_POOL), "NN": X(_MODULE_POOL, ("MM",)), "RR": S(_RING_POOL)},
    ),
    Template(
        name="tensor_associativity",
        latex=(
            r"({MM} \otimes_{{{RR}}} {NN}) \otimes_{{{RR}}} {LL}"
            r" \cong {MM} \otimes_{{{RR}}} ({NN} \otimes_{{{RR}}} {LL})"
        ),
        slots={
            "MM": S(_MODULE_POOL),
            "NN": X(_MODULE_POOL, ("MM",)),
            "LL": X(_MODULE_POOL, ("MM", "NN")),
            "RR": S(_RING_POOL),
        },
    ),
    Template(
        name="tensor_unit",
        latex=r"{RR} \otimes_{{{RR}}} {MM} \cong {MM}",
        slots={"RR": S(_RING_POOL), "MM": S(_MODULE_POOL)},
    ),
    Template(
        name="hom_modules",
        latex=r"\operatorname{{Hom}}_{{{RR}}}({MM}, {NN})",
        slots={"RR": S(_RING_POOL), "MM": S(_MODULE_POOL), "NN": X(_MODULE_POOL, ("MM",))},
    ),
    Template(
        name="hom_direct_sum",
        latex=(
            r"\operatorname{{Hom}}_{{{RR}}}({MM} \oplus {NN}, {LL})"
            r" \cong \operatorname{{Hom}}_{{{RR}}}({MM},{LL})"
            r" \oplus \operatorname{{Hom}}_{{{RR}}}({NN},{LL})"
        ),
        slots={
            "RR": S(_RING_POOL),
            "MM": S(_MODULE_POOL),
            "NN": X(_MODULE_POOL, ("MM",)),
            "LL": X(_MODULE_POOL, ("MM", "NN")),
        },
    ),
    Template(
        name="hom_exact_sequence",
        latex=(
            r"0 \to \operatorname{{Hom}}_{{{RR}}}({LL},{MM})"
            r" \to \operatorname{{Hom}}_{{{RR}}}({LL},{NN})"
            r" \to \operatorname{{Hom}}_{{{RR}}}({LL},{PP})"
        ),
        slots={
            "RR": S(_RING_POOL),
            "LL": S(_MODULE_POOL),
            "MM": X(_MODULE_POOL, ("LL",)),
            "NN": X(_MODULE_POOL, ("LL", "MM")),
            "PP": X(_MODULE_POOL, ("LL", "MM", "NN")),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B1: Ideal Structure (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B1: list[Template] = [
    Template(
        name="radical_ideal",
        latex=r"\sqrt{{{II}}} = \{{ {xx} \in {RR} : \exists n,\, {xx}^n \in {II} \}}",
        slots={"II": S(_IDEAL_POOL), "RR": S(_RING_POOL), "xx": S(_ELEM_POOL)},
    ),
    Template(
        name="primary_ideal_def",
        latex=(
            r"{QQ} \text{{ primary in }} {RR}: {aa} {bb} \in {QQ}"
            r" \Rightarrow {aa} \in {QQ} \text{{ or }} {bb}^n \in {QQ}"
        ),
        slots={
            "QQ": S(_IDEAL_POOL),
            "RR": S(_RING_POOL),
            "aa": S(_ELEM_POOL),
            "bb": X(_ELEM_POOL, ("aa",)),
        },
    ),
    Template(
        name="coprime_ideals_product",
        latex=r"{II} + {JJ} = {RR} \Rightarrow {II} \cdot {JJ} = {II} \cap {JJ}",
        slots={"II": S(_IDEAL_POOL), "JJ": X(_IDEAL_POOL, ("II",)), "RR": S(_RING_POOL)},
    ),
    Template(
        name="ideal_containment_quotient",
        latex=r"{JJ} \subseteq {II} \Rightarrow {II}/{JJ} \trianglelefteq {RR}/{JJ}",
        slots={"II": S(_IDEAL_POOL), "JJ": X(_IDEAL_POOL, ("II",)), "RR": S(_RING_POOL)},
    ),
    Template(
        name="ideal_generated",
        latex=r"({aa}) = \{{ r \cdot {aa} : r \in {RR} \}} \trianglelefteq {RR}",
        slots={"aa": S(_ELEM_POOL), "RR": S(_RING_POOL)},
    ),
    Template(
        name="nilradical_intersection",
        latex=r"\operatorname{{nil}}({RR}) = \bigcap_{{{II} \text{{ prime}}}} {II}",
        slots={"RR": S(_RING_POOL), "II": S(_IDEAL_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B2: Isomorphism Theorems (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B2: list[Template] = [
    Template(
        name="second_iso_ring",
        latex=r"({SS} + {II})/{II} \cong {SS}/({SS} \cap {II})",
        slots={"SS": S(_RING_POOL), "II": S(_IDEAL_POOL), "RR": S(_RING_POOL)},
    ),
    Template(
        name="third_iso_ring",
        latex=r"({RR}/{II})/({JJ}/{II}) \cong {RR}/{JJ} \quad ({II} \subseteq {JJ})",
        slots={"RR": S(_RING_POOL), "II": S(_IDEAL_POOL), "JJ": X(_IDEAL_POOL, ("II",))},
    ),
    Template(
        name="ring_product_hom",
        latex=(
            r"{phi} : {RR} \to {SS} \times {TT},\quad"
            r" {phi}({aa}) = ({phi}_1({aa}), {phi}_2({aa}))"
        ),
        slots={
            "phi": S(_HOMO_POOL),
            "RR": S(_RING_POOL),
            "SS": X(_RING_POOL, ("RR",)),
            "TT": X(_RING_POOL, ("RR", "SS")),
            "aa": S(_ELEM_POOL),
        },
    ),
    Template(
        name="local_ring_hom",
        latex=(
            r"{phi} : ({RR}, {II}) \to ({SS}, {JJ})"
            r" \text{{ local}} \iff {phi}({II}) \subseteq {JJ}"
        ),
        slots={
            "phi": S(_HOMO_POOL),
            "RR": S(_RING_POOL),
            "SS": X(_RING_POOL, ("RR",)),
            "II": S(_IDEAL_POOL),
            "JJ": X(_IDEAL_POOL, ("II",)),
        },
    ),
    Template(
        name="kernel_image_iso",
        latex=r"{RR} / \ker {phi} \cong \operatorname{{im}}\, {phi} \leq {SS}",
        slots={"phi": S(_HOMO_POOL), "RR": S(_RING_POOL), "SS": X(_RING_POOL, ("RR",))},
    ),
    Template(
        name="surjective_ideal_correspondence",
        latex=(
            r"{phi} : {RR} \twoheadrightarrow {SS}:"
            r" \text{{ideals of }} {SS} \longleftrightarrow"
            r" \text{{ideals of }} {RR} \text{{ above }} \ker {phi}"
        ),
        slots={
            "phi": S(_HOMO_POOL),
            "RR": S(_RING_POOL),
            "SS": X(_RING_POOL, ("RR",)),
            "II": S(_IDEAL_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B3: PID / UFD / Polynomial Rings (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B3: list[Template] = [
    Template(
        name="bezout_identity",
        latex=r"\gcd({aa}, {bb}) = {rr} \cdot {aa} + {ss} \cdot {bb} \text{{ in }} {RR}",
        slots={
            "aa": S(_ELEM_POOL),
            "bb": X(_ELEM_POOL, ("aa",)),
            "rr": X(_ELEM_POOL, ("aa", "bb")),
            "ss": X(_ELEM_POOL, ("aa", "bb", "rr")),
            "RR": S(_RING_POOL),
        },
    ),
    Template(
        name="eisenstein_criterion",
        latex=(
            r"{ff} \text{{ irred over }} {FF}:"
            r" {II} \mid a_i\,(i<n),\; {II} \nmid a_n,\; {II}^2 \nmid a_0"
        ),
        slots={"ff": S(_POLY_POOL), "II": S(_IDEAL_POOL), "FF": S(_FIELD_POOL)},
    ),
    Template(
        name="gauss_primitive_product",
        latex=r"{ff},\, {gg} \text{{ primitive}} \Rightarrow {ff} \cdot {gg} \text{{ primitive}}",
        slots={"ff": S(_POLY_POOL), "gg": X(_POLY_POOL, ("ff",))},
    ),
    Template(
        name="polynomial_degree_product",
        latex=r"\deg({ff} \cdot {gg}) = \deg {ff} + \deg {gg} \text{{ in }} {RR}",
        slots={"ff": S(_POLY_POOL), "gg": X(_POLY_POOL, ("ff",)), "RR": S(_RING_POOL)},
    ),
    Template(
        name="hilbert_basis",
        latex=r"{RR} \text{{ Noetherian}} \Rightarrow {RR}[x] \text{{ Noetherian}}",
        slots={"RR": S(_RING_POOL)},
    ),
    Template(
        name="minimal_poly_degree",
        latex=r"[{KK}({aa}):{KK}] = \deg \min_{{{KK}}}({aa})",
        slots={
            "aa": S(_ELEM_POOL),
            "KK": S(_FIELD_POOL),
            "FF": X(_FIELD_POOL, ("KK",)),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B4: Field Theory (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B4: list[Template] = [
    Template(
        name="algebraic_element",
        latex=r"{aa} \in {KK} \text{{ algebraic over }} {FF},\quad [{FF}({aa}):{FF}] < \infty",
        slots={"aa": S(_ELEM_POOL), "KK": S(_FIELD_POOL), "FF": X(_FIELD_POOL, ("KK",))},
    ),
    Template(
        name="separable_closure",
        latex=r"{LL} = {KK}^{{\mathrm{{sep}}}} \subseteq {KK},\quad {LL}/{FF} \text{{ separable}}",
        slots={
            "KK": S(_FIELD_POOL),
            "FF": X(_FIELD_POOL, ("KK",)),
            "LL": X(_FIELD_POOL, ("KK", "FF")),
        },
    ),
    Template(
        name="primitive_element_theorem",
        latex=r"{KK} = {FF}({aa}),\quad {KK}/{FF} \text{{ finite separable}}",
        slots={"aa": S(_ELEM_POOL), "KK": S(_FIELD_POOL), "FF": X(_FIELD_POOL, ("KK",))},
    ),
    Template(
        name="frobenius_endomorphism",
        latex=r"{phi} : {KK} \to {KK},\quad {phi}({aa}) = {aa}^p,\quad \operatorname{{char}} {KK} = p",
        slots={"phi": S(_HOMO_POOL), "KK": S(_FIELD_POOL), "aa": S(_ELEM_POOL)},
    ),
    Template(
        name="finite_field_subfield",
        latex=r"{LL} \subseteq {KK} \iff [{LL}:{FF}] \mid [{KK}:{FF}]",
        slots={
            "KK": S(_FIELD_POOL),
            "LL": X(_FIELD_POOL, ("KK",)),
            "FF": X(_FIELD_POOL, ("KK", "LL")),
        },
    ),
    Template(
        name="norm_trace",
        latex=(
            r"N_{{{KK}/{FF}}}({aa})"
            r" = \prod{lim_mod}_{{{phi} \in \operatorname{{Gal}}({KK}/{FF})}} {phi}({aa})"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "aa": S(_ELEM_POOL),
            "KK": S(_FIELD_POOL),
            "FF": X(_FIELD_POOL, ("KK",)),
            "phi": S(_HOMO_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B5: Galois Theory Extended (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B5: list[Template] = [
    Template(
        name="artin_theorem",
        latex=(
            r"{KK}^{{{HH}}} = {FF} \text{{ and }}"
            r" [{KK}:{FF}] = |{HH}| \text{{ (Artin)}}"
        ),
        slots={"HH": S(_SUBGP_POOL), "KK": S(_FIELD_POOL), "FF": X(_FIELD_POOL, ("KK",))},
    ),
    Template(
        name="galois_orbit_stabilizer",
        latex=(
            r"|\operatorname{{Orb}}_{{\operatorname{{Gal}}({KK}/{FF})}}({aa})| ="
            r" [\operatorname{{Gal}}({KK}/{FF}) : \operatorname{{Stab}}({aa})]"
        ),
        slots={
            "aa": S(_ELEM_POOL),
            "KK": S(_FIELD_POOL),
            "FF": X(_FIELD_POOL, ("KK",)),
            "HH": S(_SUBGP_POOL),
        },
    ),
    Template(
        name="splitting_field_unique",
        latex=(
            r"{KK} \text{{ splitting field of }} {ff}"
            r" \text{{ over }} {FF},\quad \text{{unique up to iso}}"
        ),
        slots={"ff": S(_POLY_POOL), "KK": S(_FIELD_POOL), "FF": X(_FIELD_POOL, ("KK",))},
    ),
    Template(
        name="galois_action_composition",
        latex=(
            r"{sigma}, {tau} \in \operatorname{{Gal}}({KK}/{FF})"
            r" \Rightarrow {sigma} \circ {tau} \in \operatorname{{Gal}}({KK}/{FF})"
        ),
        slots={
            "sigma": S(_HOMO_POOL),
            "tau": X(_HOMO_POOL, ("sigma",)),
            "KK": S(_FIELD_POOL),
            "FF": X(_FIELD_POOL, ("KK",)),
        },
    ),
    Template(
        name="subgroup_field_correspondence",
        latex=(
            r"{HH} \leq \operatorname{{Gal}}({KK}/{FF})"
            r" \longleftrightarrow {LL} = {KK}^{{{HH}}},"
            r"\quad {FF} \subseteq {LL} \subseteq {KK}"
        ),
        slots={
            "HH": S(_SUBGP_POOL),
            "KK": S(_FIELD_POOL),
            "LL": X(_FIELD_POOL, ("KK",)),
            "FF": X(_FIELD_POOL, ("KK", "LL")),
        },
    ),
    Template(
        name="degree_orbit_formula",
        latex=r"[{KK}:{FF}({aa})] \cdot |\operatorname{{Orb}}({aa})| = [{KK}:{FF}]",
        slots={"aa": S(_ELEM_POOL), "KK": S(_FIELD_POOL), "FF": X(_FIELD_POOL, ("KK",))},
    ),
]

# ---------------------------------------------------------------------------
# Part B6: Commutative Algebra & Homological (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B6: list[Template] = [
    Template(
        name="localization_def",
        latex=(
            r"{SS}^{{-1}}{RR}"
            r" = \{{ {aa}/{bb} : {aa} \in {RR},\, {bb} \in {SS} \}}"
        ),
        slots={
            "SS": S(_RING_POOL),
            "RR": X(_RING_POOL, ("SS",)),
            "aa": S(_ELEM_POOL),
            "bb": X(_ELEM_POOL, ("aa",)),
        },
    ),
    Template(
        name="nakayama_lemma",
        latex=(
            r"{II} {MM} = {MM} \Rightarrow {MM} = 0"
            r" \quad \text{{(Nakayama, }} {RR} \text{{ local)}}"
        ),
        slots={"II": S(_IDEAL_POOL), "MM": S(_MODULE_POOL), "RR": S(_RING_POOL)},
    ),
    Template(
        name="primary_decomposition",
        latex=(
            r"{II} = {QQ}_1 \cap {QQ}_2 \cap \cdots"
            r" \text{{ (Lasker-Noether, }} {RR} \text{{ Noetherian)}}"
        ),
        slots={
            "II": S(_IDEAL_POOL),
            "QQ": X(_IDEAL_POOL, ("II",)),
            "PP": X(_IDEAL_POOL, ("II", "QQ")),
            "RR": S(_RING_POOL),
        },
    ),
    Template(
        name="projective_direct_summand",
        latex=(
            r"{PP} \text{{ projective}} \iff"
            r" \exists {QQ} : {PP} \oplus {QQ} \text{{ free}}"
        ),
        slots={"PP": S(_MODULE_POOL), "QQ": X(_MODULE_POOL, ("PP",)), "MM": S(_MODULE_POOL)},
    ),
    Template(
        name="flat_module_tensor_exact",
        latex=(
            r"{MM} \text{{ flat over }} {RR}"
            r" \iff {NN} \otimes_{{{RR}}} {MM} \text{{ preserves exact sequences}}"
        ),
        slots={"MM": S(_MODULE_POOL), "NN": X(_MODULE_POOL, ("MM",)), "RR": S(_RING_POOL)},
    ),
    Template(
        name="ext1_extension_class",
        latex=(
            r"\operatorname{{Ext}}^1_{{{RR}}}({MM}, {NN})"
            r" \longleftrightarrow \left[ 0 \to {NN} \to E \to {MM} \to 0 \right]"
        ),
        slots={"MM": S(_MODULE_POOL), "NN": X(_MODULE_POOL, ("MM",)), "RR": S(_RING_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part C: High-n_eff function-pair templates (6)
# ---------------------------------------------------------------------------

_TEMPLATES_C: list[Template] = [
    Template(
        name="fn_triple_ring",
        latex=r"{fn1}({aa} \cdot {bb}) = {fn2}({aa}) \cdot {fn3}({bb})",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "fn3": _FN_SLOT,
            "aa": S(_ELEM_POOL),
            "bb": X(_ELEM_POOL, ("aa",)),
        },
    ),
    Template(
        name="fn_pair_ring_hom",
        latex=r"{fn1}({phi}({aa})) = {fn2}({aa}) \quad \forall {aa} \in {RR}",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "phi": S(_HOMO_POOL),
            "RR": S(_RING_POOL),
            "aa": S(_ELEM_POOL),
        },
    ),
    Template(
        name="fn_field_conjugate",
        latex=(
            r"{fn1}({sigma}({aa})) = {fn2}({aa})"
            r" \quad ({sigma} \in \operatorname{{Gal}}({KK}/{FF}))"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "sigma": S(_HOMO_POOL),
            "KK": S(_FIELD_POOL),
            "FF": X(_FIELD_POOL, ("KK",)),
            "aa": S(_ELEM_POOL),
        },
    ),
    Template(
        name="fn_module_action",
        latex=(
            r"{fn1}(r \cdot m) = r \cdot {fn2}(m)"
            r" \quad \forall r \in {RR},\, m \in {MM}"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "RR": S(_RING_POOL),
            "MM": S(_MODULE_POOL),
        },
    ),
    Template(
        name="fn_ideal_coset",
        latex=r"{fn1}({aa} + {II}) = {fn2}({aa}) \quad \text{{in }} {RR}/{II}",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "II": S(_IDEAL_POOL),
            "RR": S(_RING_POOL),
            "aa": S(_ELEM_POOL),
        },
    ),
    Template(
        name="fn_norm_product",
        latex=(
            r"{fn1}({aa}) \cdot {fn2}({sigma}({aa}))"
            r" = N_{{{KK}/{FF}}}({aa})"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "aa": S(_ELEM_POOL),
            "sigma": S(_HOMO_POOL),
            "KK": S(_FIELD_POOL),
            "FF": X(_FIELD_POOL, ("KK",)),
        },
    ),
]

# ---------------------------------------------------------------------------
# Assembly & registry
# ---------------------------------------------------------------------------

_RING_TEMPLATES: list[Template] = (
    _TEMPLATES_A
    + _TEMPLATES_B1
    + _TEMPLATES_B2
    + _TEMPLATES_B3
    + _TEMPLATES_B4
    + _TEMPLATES_B5
    + _TEMPLATES_B6
    + _TEMPLATES_C
)


# Homomorphism / endomorphism templates (from algebra domain)
_RING_TEMPLATES += [
    Template(
        name="ring_homomorphism_rightarrow",
        latex=r"f: {RR} \rightarrow {SS},\quad f(1_{{{RR}}}) = 1_{{{SS}}}",
        slots={"RR": S(_RING_NAMES), "SS": X(_RING_NAMES, ("RR",))},
    ),
    Template(
        name="frobenius_endomorphism_longmapsto",
        latex=r"\mathrm{{Frob}}: {rr} \longmapsto {rr}^p \quad (p\text{{ prime}})",
        slots={"rr": S(_ELT_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("ring_field_theory", _RING_TEMPLATES)
