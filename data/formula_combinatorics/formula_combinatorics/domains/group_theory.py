"""Group theory domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import S, Template, X, compute_weights, make_dispatcher

# ---------------------------------------------------------------------------
# Shared pools (kept local to avoid cross-domain coupling)
# ---------------------------------------------------------------------------

_ELEMS = ["g", "h", "a", "b", "x", "y", r"\sigma", r"\tau", r"\alpha", r"\beta", r"\gamma"]
_HOMOS = [r"\phi", r"\varphi", r"\psi", "f", r"\theta", r"\rho", r"\pi"]
_SIMPLE = ["G", "H", "K", "N", "A", "B"]

_NV_POOL: tuple[str, ...] = ("n", "m", "4", "5", "6", "p")
_QV_POOL: tuple[str, ...] = ("q", "2", "p")
_P_POOL: tuple[str, ...] = ("p", "q", r"\ell")
_N_POOL: tuple[str, ...] = ("n", "m", "r")
_SIMPLE_T: tuple[str, ...] = tuple(_SIMPLE)
_ELEMS_T: tuple[str, ...] = tuple(_ELEMS)
_HOMOS_T: tuple[str, ...] = tuple(_HOMOS)

# Pre-expand the full named-group pool over all (nv, qv) combinations so that
# G can be sampled from a static pool, faithfully covering every string the
# original generator could produce.
_NAMED_GROUPS: list[str] = []
_seen_ng: set[str] = set()
for _nv in _NV_POOL:
    for _qv in _QV_POOL:
        for _candidate in [
            rf"S_{{{_nv}}}",
            rf"D_{{{_nv}}}",
            rf"A_{{{_nv}}}",
            rf"GL_{{{_nv}}}(\mathbb{{F}}_{{{_qv}}})",
            rf"SL_{{{_nv}}}(\mathbb{{F}}_{{{_qv}}})",
            rf"\mathbb{{Z}}_{{{_nv}}}",
            rf"\mathbb{{Z}}/{_nv}\mathbb{{Z}}",
        ]:
            if _candidate not in _seen_ng:
                _NAMED_GROUPS.append(_candidate)
                _seen_ng.add(_candidate)

_G_POOL: tuple[str, ...] = tuple(_SIMPLE + _NAMED_GROUPS)

# ---------------------------------------------------------------------------
# Group theory templates
# ---------------------------------------------------------------------------

_GROUP_THEORY_TEMPLATES: list[Template] = [
    # c=0 — Lagrange's theorem (3 forms)
    Template(
        name="lagrange_theorem",
        latex="",
        slots={},
        variants=[
            Template(
                name="lagrange_divisibility",
                latex=r"|{H}| \mid |{G}|",
                slots={
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE_T, ("G",)),
                },
            ),
            Template(
                name="lagrange_index_formula",
                latex=r"[{G}:{H}] = \frac{{|{G}|}}{{|{H}|}}",
                slots={
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE_T, ("G",)),
                },
            ),
            Template(
                name="lagrange_order_product",
                latex=r"|{G}| = [{G}:{H}] \cdot |{H}|",
                slots={
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE_T, ("G",)),
                },
            ),
        ],
    ),
    # c=1 — first isomorphism theorem
    Template(
        name="first_isomorphism_theorem",
        latex=r"{G}/\ker {phi} \cong \operatorname{{im}}\, {phi}",
        slots={
            "G": S(_G_POOL),
            "phi": S(_HOMOS_T),
        },
    ),
    # c=2 — second isomorphism theorem
    Template(
        name="second_isomorphism_theorem",
        latex=r"{H}/({H} \cap {N_sub}) \cong {H}{N_sub}/{N_sub}",
        slots={
            "H": S(_SIMPLE_T),
            "N_sub": X(_SIMPLE_T, ("H",)),
        },
    ),
    # c=3 — third isomorphism theorem
    Template(
        name="third_isomorphism_theorem",
        latex=r"({G}/{N_sub})/({H}/{N_sub}) \cong {G}/{H}",
        slots={
            "G": S(_G_POOL),
            "H": X(_SIMPLE_T, ("G",)),
            "N_sub": X(_SIMPLE_T, ("G", "H")),
        },
    ),
    # c=4 — element order (3 forms)
    Template(
        name="element_order",
        latex="",
        slots={},
        variants=[
            Template(
                name="element_order_divides",
                latex=r"\operatorname{{ord}}({g_el}) \mid |{G}|",
                slots={
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS_T),
                },
            ),
            Template(
                name="element_order_power",
                latex=r"{g_el}^{{|{G}|}} = e",
                slots={
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS_T),
                },
            ),
            Template(
                name="element_order_lcm",
                latex=r"\operatorname{{ord}}({g_el}{h_el}) \mid \operatorname{{lcm}}(\operatorname{{ord}}({g_el}), \operatorname{{ord}}({h_el}))",
                slots={
                    "g_el": S(_ELEMS_T),
                    "h_el": X(_ELEMS_T, ("g_el",)),
                },
            ),
        ],
    ),
    # c=5 — orbit-stabilizer theorem
    Template(
        name="orbit_stabilizer",
        latex=r"|{G}| = |\operatorname{{Orb}}({g_el})| \cdot |\operatorname{{Stab}}_{{{G}}}({g_el})|",
        slots={
            "G": S(_G_POOL),
            "g_el": S(_ELEMS_T),
        },
    ),
    # c=6 — class equation (2 forms)
    Template(
        name="class_equation",
        latex="",
        slots={},
        variants=[
            Template(
                name="class_equation_center",
                latex=r"|{G}| = |Z({G})| + \sum_{{{g_el}}} [{G} : C_{{{G}}}({g_el})]",
                slots={
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS_T),
                },
            ),
            Template(
                name="class_equation_conjugacy",
                latex=r"|{G}| = \sum_{{[{g_el}]}} \frac{{|{G}|}}{{|C_{{{G}}}({g_el})|}}",
                slots={
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS_T),
                },
            ),
        ],
    ),
    # c=7 — Sylow's theorem (3 forms)
    Template(
        name="sylow_theorem",
        latex="",
        slots={},
        variants=[
            Template(
                name="sylow_order_decomposition",
                latex=r"|{G}| = {p}^k m,\quad \gcd({p}, m) = 1",
                slots={
                    "G": S(_G_POOL),
                    "p": S(_P_POOL),
                },
            ),
            Template(
                name="sylow_p_power_divides",
                latex=r"{p}^k \mid |{G}|,\quad {p}^{{k+1}} \nmid |{G}|",
                slots={
                    "G": S(_G_POOL),
                    "p": S(_P_POOL),
                },
            ),
            Template(
                name="sylow_count_congruence",
                latex=r"n_{{{p}}}({G}) \mid m,\quad n_{{{p}}}({G}) \equiv 1 \pmod{{{p}}}",
                slots={
                    "G": S(_G_POOL),
                    "p": S(_P_POOL),
                },
            ),
        ],
    ),
    # c=8 — commutator (2 forms)
    Template(
        name="commutator",
        latex="",
        slots={},
        variants=[
            Template(
                name="commutator_definition",
                latex=r"[{g_el}, {h_el}] = {g_el}^{{-1}} {h_el}^{{-1}} {g_el} {h_el}",
                slots={
                    "g_el": S(_ELEMS_T),
                    "h_el": X(_ELEMS_T, ("g_el",)),
                },
            ),
            Template(
                name="commutator_product_rule",
                latex=r"[{g_el}, {h_el}{N_sub}] = [{g_el},{h_el}] \cdot [{g_el},{N_sub}]^{{{h_el}}}",
                slots={
                    "g_el": S(_ELEMS_T),
                    "h_el": X(_ELEMS_T, ("g_el",)),
                    "N_sub": S(_SIMPLE_T),
                },
            ),
        ],
    ),
    # c=9 — derived subgroup / abelianisation (2 forms)
    Template(
        name="derived_subgroup",
        latex="",
        slots={},
        variants=[
            Template(
                name="derived_subgroup_definition",
                latex=r"[{G}, {G}] = \langle [{g_el}, {h_el}] : {g_el}, {h_el} \in {G} \rangle",
                slots={
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS_T),
                    "h_el": X(_ELEMS_T, ("g_el",)),
                },
            ),
            Template(
                name="abelianization",
                latex=r"{G}/[{G},{G}] \cong {G}^{{\mathrm{{ab}}}}",
                slots={"G": S(_G_POOL)},
            ),
        ],
    ),
    # c=10 — center of a group (3 forms)
    Template(
        name="center_of_group",
        latex="",
        slots={},
        variants=[
            Template(
                name="center_definition",
                latex=r"Z({G}) = \left\{{{g_el} \in {G} \mid {g_el}{h_el} = {h_el}{g_el}\; \forall {h_el} \in {G}\right\}}",
                slots={
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS_T),
                    "h_el": X(_ELEMS_T, ("g_el",)),
                },
            ),
            Template(
                name="center_abelian_iff",
                latex=r"{G} \text{{ abelian}} \iff Z({G}) = {G}",
                slots={"G": S(_G_POOL)},
            ),
            Template(
                name="center_normal_inn",
                latex=r"Z({G}) \trianglelefteq {G},\quad {G}/Z({G}) \cong \operatorname{{Inn}}({G})",
                slots={"G": S(_G_POOL)},
            ),
        ],
    ),
    # c=11 — conjugacy classes (2 forms)
    Template(
        name="conjugacy",
        latex="",
        slots={},
        variants=[
            Template(
                name="conjugacy_relation",
                latex=r"{g_el} \sim {h_el} \iff \exists\, {n} \in {G} : {n}{g_el}{n}^{{-1}} = {h_el}",
                slots={
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS_T),
                    "h_el": X(_ELEMS_T, ("g_el",)),
                    "n": S(_N_POOL),
                },
            ),
            Template(
                name="conjugacy_class_size",
                latex=r"|[{g_el}]_{{{G}}}| = [{G} : C_{{{G}}}({g_el})]",
                slots={
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS_T),
                },
            ),
        ],
    ),
    # c=12 — normalizer (2 forms)
    Template(
        name="normalizer",
        latex="",
        slots={},
        variants=[
            Template(
                name="normalizer_definition",
                latex=r"N_{{{G}}}({H}) = \left\{{{g_el} \in {G} : {g_el}{H}{g_el}^{{-1}} = {H}\right\}}",
                slots={
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE_T, ("G",)),
                    "g_el": S(_ELEMS_T),
                },
            ),
            Template(
                name="normal_iff_normalizer",
                latex=r"{H} \trianglelefteq {G} \iff N_{{{G}}}({H}) = {G}",
                slots={
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE_T, ("G",)),
                },
            ),
        ],
    ),
    # c=13 — centralizer (2 forms)
    Template(
        name="centralizer",
        latex="",
        slots={},
        variants=[
            Template(
                name="centralizer_definition",
                latex=r"C_{{{G}}}({g_el}) = \left\{{{h_el} \in {G} : {h_el}{g_el} = {g_el}{h_el}\right\}}",
                slots={
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS_T),
                    "h_el": X(_ELEMS_T, ("g_el",)),
                },
            ),
            Template(
                name="centralizer_class_size",
                latex=r"|{G}| = |C_{{{G}}}({g_el})| \cdot |[{g_el}]|",
                slots={
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS_T),
                },
            ),
        ],
    ),
    # c=14 — homomorphism properties (3 forms)
    Template(
        name="homomorphism_properties",
        latex="",
        slots={},
        variants=[
            Template(
                name="homomorphism_multiplicativity",
                latex=r"{phi}({g_el}{h_el}) = {phi}({g_el})\,{phi}({h_el})",
                slots={
                    "phi": S(_HOMOS_T),
                    "g_el": S(_ELEMS_T),
                    "h_el": X(_ELEMS_T, ("g_el",)),
                },
            ),
            Template(
                name="homomorphism_identity_inverse",
                latex=r"{phi}({g_el}^{{-1}}) = {phi}({g_el})^{{-1}},\quad {phi}(e) = e",
                slots={
                    "phi": S(_HOMOS_T),
                    "g_el": S(_ELEMS_T),
                },
            ),
            Template(
                name="isomorphism_bijective_homo",
                latex=r"{phi} : {G} \to {H} \text{{ iso.}} \iff {phi} \text{{ bij. hom.}}",
                slots={
                    "phi": S(_HOMOS_T),
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE_T, ("G",)),
                },
            ),
        ],
    ),
    # c=15 — kernel (2 forms)
    Template(
        name="kernel",
        latex="",
        slots={},
        variants=[
            Template(
                name="kernel_definition",
                latex=r"\ker {phi} = \left\{{{g_el} \in {G} : {phi}({g_el}) = e\right\}}",
                slots={
                    "phi": S(_HOMOS_T),
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS_T),
                },
            ),
            Template(
                name="kernel_normal_embedding",
                latex=r"\ker {phi} \trianglelefteq {G},\quad {G}/\ker {phi} \hookrightarrow {H}",
                slots={
                    "phi": S(_HOMOS_T),
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE_T, ("G",)),
                },
            ),
        ],
    ),
    # c=16 — direct product (3 forms)
    Template(
        name="direct_product",
        latex="",
        slots={},
        variants=[
            Template(
                name="direct_product_two",
                latex=r"{G} \times {H}",
                slots={
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE_T, ("G",)),
                },
            ),
            Template(
                name="direct_product_three",
                latex=r"{G} \times {H} \times {N_sub}",
                slots={
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE_T, ("G",)),
                    "N_sub": X(_SIMPLE_T, ("G", "H")),
                },
            ),
            Template(
                name="direct_product_cyclic",
                latex=r"\mathbb{{Z}}_{{{p}}} \times \mathbb{{Z}}_{{{p}^2}}",
                slots={"p": S(_P_POOL)},
            ),
        ],
    ),
    # c=17 — semidirect product (2 forms)
    Template(
        name="semidirect_product",
        latex="",
        slots={},
        variants=[
            Template(
                name="semidirect_product_iso",
                latex=r"{G} \cong {N_sub} \rtimes {H}",
                slots={
                    "G": S(_G_POOL),
                    "N_sub": X(_SIMPLE_T, ("G",)),
                    "H": X(_SIMPLE_T, ("G", "N_sub")),
                },
            ),
            Template(
                name="semidirect_product_conditions",
                latex=r"{N_sub} \trianglelefteq {G},\quad {G} = {N_sub}{H},\quad {N_sub} \cap {H} = \{{e\}}",
                slots={
                    "G": S(_G_POOL),
                    "N_sub": X(_SIMPLE_T, ("G",)),
                    "H": X(_SIMPLE_T, ("G", "N_sub")),
                },
            ),
        ],
    ),
    # c=18 — group presentations (3 forms)
    Template(
        name="group_presentations",
        latex="",
        slots={},
        variants=[
            Template(
                name="dihedral_presentation",
                latex=(
                    r"D_{{{nv}}} = \langle {g_el}, {h_el} \mid "
                    r"{g_el}^{{{nv}}} = {h_el}^2 = e,\; {h_el}{g_el}{h_el}^{{-1}} = {g_el}^{{-1}} \rangle"
                ),
                slots={
                    "nv": S(_NV_POOL),
                    "g_el": S(_ELEMS_T),
                    "h_el": X(_ELEMS_T, ("g_el",)),
                },
            ),
            Template(
                name="cyclic_presentation",
                latex=r"\mathbb{{Z}}/{nv}\mathbb{{Z}} = \langle {g_el} \mid {g_el}^{{{nv}}} = e \rangle",
                slots={
                    "nv": S(_NV_POOL),
                    "g_el": S(_ELEMS_T),
                },
            ),
            Template(
                name="quaternion_presentation",
                latex=(
                    r"Q_8 = \langle {g_el}, {h_el} \mid "
                    r"{g_el}^4 = e,\; {g_el}^2 = {h_el}^2,\; {h_el}{g_el}{h_el}^{{-1}} = {g_el}^{{-1}} \rangle"
                ),
                slots={
                    "g_el": S(_ELEMS_T),
                    "h_el": X(_ELEMS_T, ("g_el",)),
                },
            ),
        ],
    ),
    # c=19 — free groups (2 forms)
    Template(
        name="free_groups",
        latex="",
        slots={},
        variants=[
            Template(
                name="free_group_rank_n",
                latex=r"F_{{{nv}}} = \langle {g_el}_1, \ldots, {g_el}_{{{nv}}} \mid \varnothing \rangle",
                slots={
                    "nv": S(_NV_POOL),
                    "g_el": S(_ELEMS_T),
                },
            ),
            Template(
                name="free_group_rank_2_nonabelian",
                latex=r"F_2 = \langle {g_el}, {h_el} \rangle,\quad [{g_el},{h_el}] \neq e",
                slots={
                    "g_el": S(_ELEMS_T),
                    "h_el": X(_ELEMS_T, ("g_el",)),
                },
            ),
        ],
    ),
    # c=20 — Cayley's theorem
    Template(
        name="cayley_theorem",
        latex=r"{G} \hookrightarrow S_{{|{G}|}}",
        slots={"G": S(_G_POOL)},
    ),
    # c=21 — short exact sequence (2 forms)
    Template(
        name="short_exact_sequence",
        latex="",
        slots={},
        variants=[
            Template(
                name="short_exact_sequence_maps",
                latex=r"0 \to {H} \xrightarrow{{{phi}}} {G} \xrightarrow{{{psi}}} {G}/{H} \to 0",
                slots={
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE_T, ("G",)),
                    "phi": S(_HOMOS_T),
                    "psi": X(_HOMOS_T, ("phi",)),
                },
            ),
            Template(
                name="short_exact_sequence_normal",
                latex=r"1 \to {N_sub} \to {G} \to {G}/{N_sub} \to 1",
                slots={
                    "G": S(_G_POOL),
                    "N_sub": X(_SIMPLE_T, ("G",)),
                },
            ),
        ],
    ),
    # c=22 — exactness (2 forms)
    Template(
        name="exactness",
        latex="",
        slots={},
        variants=[
            Template(
                name="exactness_ker_im",
                latex=r"\ker {psi} = \operatorname{{im}}\, {phi}",
                slots={
                    "phi": S(_HOMOS_T),
                    "psi": X(_HOMOS_T, ("phi",)),
                },
            ),
            Template(
                name="exactness_injective",
                latex=r"0 \to {H} \xrightarrow{{{phi}}} {G} \text{{ exact}} \iff {phi} \text{{ injective}}",
                slots={
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE_T, ("G",)),
                    "phi": S(_HOMOS_T),
                },
            ),
        ],
    ),
    # c=23 — inner automorphism group
    Template(
        name="inner_automorphisms",
        latex=r"\operatorname{{Inn}}({G}) \cong {G} / Z({G})",
        slots={"G": S(_G_POOL)},
    ),
    # c=24 — outer automorphism group
    Template(
        name="outer_automorphisms",
        latex=r"\operatorname{{Out}}({G}) = \operatorname{{Aut}}({G}) / \operatorname{{Inn}}({G})",
        slots={"G": S(_G_POOL)},
    ),
    # c=25 — abelianisation (2 forms)
    Template(
        name="abelianization_quotient",
        latex="",
        slots={},
        variants=[
            Template(
                name="abelianization_def",
                latex=r"{G}^{{\mathrm{{ab}}}} = {G}/[{G},{G}]",
                slots={"G": S(_G_POOL)},
            ),
            Template(
                name="abelian_iff_trivial_commutator",
                latex=r"{G} \text{{ abelian}} \iff [{G},{G}] = \{{e\}}",
                slots={"G": S(_G_POOL)},
            ),
        ],
    ),
    # c=26 — index product inequality
    Template(
        name="index_product_inequality",
        latex=r"[{G}:{H} \cap {N_sub}] \leq [{G}:{H}] \cdot [{G}:{N_sub}]",
        slots={
            "G": S(_G_POOL),
            "H": X(_SIMPLE_T, ("G",)),
            "N_sub": X(_SIMPLE_T, ("G", "H")),
        },
    ),
    # c=27 — p-group centre is non-trivial (2 forms)
    Template(
        name="p_group_center",
        latex="",
        slots={},
        variants=[
            Template(
                name="p_group_nontrivial_center",
                latex=r"|{G}| = {p}^{{{n}}} \implies Z({G}) \neq \{{e\}}",
                slots={
                    "G": S(_G_POOL),
                    "p": S(_P_POOL),
                    "n": S(_N_POOL),
                },
            ),
            Template(
                name="p_group_subgroup_normal",
                latex=r"|{G}| = {p}^{{{n}}},\; {H} \leq {G} \implies {H} \trianglelefteq {G}",
                slots={
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE_T, ("G",)),
                    "p": S(_P_POOL),
                    "n": S(_N_POOL),
                },
            ),
        ],
    ),
]

# c=28 — symmetric / alternating / GL group orders (3 forms)
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="named_group_orders",
        latex="",
        slots={},
        variants=[
            Template(
                name="symmetric_group_order",
                latex=r"|S_{{{nv}}}| = {nv}!",
                slots={"nv": S(_NV_POOL)},
            ),
            Template(
                name="alternating_normal_index2",
                latex=r"A_{{{nv}}} \trianglelefteq S_{{{nv}}},\quad [S_{{{nv}}} : A_{{{nv}}}] = 2",
                slots={"nv": S(_NV_POOL)},
            ),
            Template(
                name="gl_order_formula",
                latex=r"|GL_{{{nv}}}(\mathbb{{F}}_{{{qv}}})| = \prod_{{k=0}}^{{{nv}-1}} ({qv}^{{{nv}}} - {qv}^k)",
                slots={
                    "nv": S(_NV_POOL),
                    "qv": S(_QV_POOL),
                },
            ),
        ],
    )
)

# c=29 — dihedral group (2 forms)
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="dihedral_group",
        latex="",
        slots={},
        variants=[
            Template(
                name="dihedral_group_order",
                latex=r"|D_{{{nv}}}| = 2{nv}",
                slots={"nv": S(_NV_POOL)},
            ),
            Template(
                name="dihedral_group_center",
                latex=(
                    r"Z(D_{{{nv}}}) = \begin{{cases}} "
                    r"\{{e, {g_el}^{{{nv}/2}}\}} & {nv} \text{{ even}} \\ "
                    r"\{{e\}} & {nv} \text{{ odd}} \end{{cases}}"
                ),
                slots={
                    "nv": S(_NV_POOL),
                    "g_el": S(_ELEMS_T),
                },
            ),
        ],
    )
)

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

_W_GT: list[float] = compute_weights(_GROUP_THEORY_TEMPLATES)

# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

_group_theory = make_dispatcher(_GROUP_THEORY_TEMPLATES, _W_GT)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "group_theory": _group_theory,
}

WEIGHTS: dict[str, float] = {
    "group_theory": 0.04,
}

TEMPLATES: dict[str, list[Template]] = {
    "group_theory": _GROUP_THEORY_TEMPLATES,
}
