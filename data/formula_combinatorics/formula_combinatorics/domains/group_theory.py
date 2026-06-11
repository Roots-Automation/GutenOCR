"""Group theory domain generators."""

from __future__ import annotations

from .._template_dsl import _FN_SLOT, _LIM_MOD, S, Template, X
from .._vocab import _ELT_POOL, _GRP_NAMES
from .._vocab import _HOMO_POOL as _HOMOS
from ._config import register_domain

# ---------------------------------------------------------------------------
# Shared pools (kept local to avoid cross-domain coupling)
# ---------------------------------------------------------------------------

_ELEMS: tuple[str, ...] = (
    "g",
    "h",
    "a",
    "b",
    "x",
    "y",
    r"\sigma",
    r"\tau",
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\delta",
    r"\omega",
)
_SIMPLE: tuple[str, ...] = ("G", "H", "K", "N", "A", "B", "P", "Q", "L", "M", "T", "W")

_NV_POOL: tuple[str, ...] = ("n", "m", "4", "5", "6", "7", "8", "p", "q", "r")
_QV_POOL: tuple[str, ...] = ("q", "2", "3", "4", "5", "p")
_P_POOL: tuple[str, ...] = ("p", "q", r"\ell", "r", "s", r"p_1", r"p_2")
_N_POOL: tuple[str, ...] = ("n", "m", "r", "k", "d", r"n_1", r"n_2")


def _build_named_groups() -> tuple[str, ...]:
    # Pre-expand the full named-group pool over all (nv, qv) combinations so
    # that G can be sampled from a static pool, faithfully covering every
    # string the original generator could produce.
    seen: set[str] = set()
    result: list[str] = []
    for nv in _NV_POOL:
        for qv in _QV_POOL:
            for candidate in [
                rf"S_{{{nv}}}",
                rf"D_{{{nv}}}",
                rf"A_{{{nv}}}",
                rf"GL_{{{nv}}}(\mathbb{{F}}_{{{qv}}})",
                rf"SL_{{{nv}}}(\mathbb{{F}}_{{{qv}}})",
                rf"\mathbb{{Z}}_{{{nv}}}",
                rf"\mathbb{{Z}}/{nv}\mathbb{{Z}}",
            ]:
                if candidate not in seen:
                    result.append(candidate)
                    seen.add(candidate)
    return tuple(result)


_NAMED_GROUPS: tuple[str, ...] = _build_named_groups()
_G_POOL: tuple[str, ...] = _SIMPLE + _NAMED_GROUPS

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
                    "H": X(_SIMPLE, ("G",)),
                },
            ),
            Template(
                name="lagrange_index_formula",
                latex=r"[{G}:{H}] = \frac{{|{G}|}}{{|{H}|}}",
                slots={
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE, ("G",)),
                },
            ),
            Template(
                name="lagrange_order_product",
                latex=r"|{G}| = [{G}:{H}] \cdot |{H}|",
                slots={
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE, ("G",)),
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
            "phi": S(_HOMOS),
        },
    ),
    # c=2 — second isomorphism theorem
    Template(
        name="second_isomorphism_theorem",
        latex=r"{H}/({H} \cap {N_sub}) \cong {H}{N_sub}/{N_sub}",
        slots={
            "H": S(_SIMPLE),
            "N_sub": X(_SIMPLE, ("H",)),
        },
    ),
    # c=3 — third isomorphism theorem
    Template(
        name="third_isomorphism_theorem",
        latex=r"({G}/{N_sub})/({H}/{N_sub}) \cong {G}/{H}",
        slots={
            "G": S(_G_POOL),
            "H": X(_SIMPLE, ("G",)),
            "N_sub": X(_SIMPLE, ("G", "H")),
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
                    "g_el": S(_ELEMS),
                },
            ),
            Template(
                name="element_order_power",
                latex=r"{g_el}^{{|{G}|}} = e",
                slots={
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS),
                },
            ),
            Template(
                name="element_order_lcm",
                latex=r"\operatorname{{ord}}({g_el} {h_el}) \mid \operatorname{{lcm}}(\operatorname{{ord}}({g_el}), \operatorname{{ord}}({h_el}))",
                slots={
                    "g_el": S(_ELEMS),
                    "h_el": X(_ELEMS, ("g_el",)),
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
            "g_el": S(_ELEMS),
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
                latex=r"|{G}| = |Z({G})| + \sum{lim_mod}_{{{g_el}}} [{G} : C_{{{G}}}({g_el})]",
                slots={
                    "lim_mod": _LIM_MOD,
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS),
                },
            ),
            Template(
                name="class_equation_conjugacy",
                latex=r"|{G}| = \sum{lim_mod}_{{[{g_el}]}} \frac{{|{G}|}}{{|C_{{{G}}}({g_el})|}}",
                slots={
                    "lim_mod": _LIM_MOD,
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS),
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
                    "g_el": S(_ELEMS),
                    "h_el": X(_ELEMS, ("g_el",)),
                },
            ),
            Template(
                name="commutator_product_rule",
                latex=r"[{g_el}, {h_el} {N_sub}] = [{g_el},{h_el}] \cdot [{g_el},{N_sub}]^{{{h_el}}}",
                slots={
                    "g_el": S(_ELEMS),
                    "h_el": X(_ELEMS, ("g_el",)),
                    "N_sub": S(_SIMPLE),
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
                    "g_el": S(_ELEMS),
                    "h_el": X(_ELEMS, ("g_el",)),
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
                latex=r"Z({G}) = \left\{{{g_el} \in {G} \mid {g_el} {h_el} = {h_el} {g_el}\; \forall {h_el} \in {G}\right\}}",
                slots={
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS),
                    "h_el": X(_ELEMS, ("g_el",)),
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
                latex=r"{g_el} \sim {h_el} \iff \exists\, {n} \in {G} : {n} {g_el} {n}^{{-1}} = {h_el}",
                slots={
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS),
                    "h_el": X(_ELEMS, ("g_el",)),
                    "n": S(_N_POOL),
                },
            ),
            Template(
                name="conjugacy_class_size",
                latex=r"|[{g_el}]_{{{G}}}| = [{G} : C_{{{G}}}({g_el})]",
                slots={
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS),
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
                latex=r"N_{{{G}}}({H}) = \left\{{{g_el} \in {G} : {g_el} {H} {g_el}^{{-1}} = {H}\right\}}",
                slots={
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE, ("G",)),
                    "g_el": S(_ELEMS),
                },
            ),
            Template(
                name="normal_iff_normalizer",
                latex=r"{H} \trianglelefteq {G} \iff N_{{{G}}}({H}) = {G}",
                slots={
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE, ("G",)),
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
                latex=r"C_{{{G}}}({g_el}) = \left\{{{h_el} \in {G} : {h_el} {g_el} = {g_el} {h_el}\right\}}",
                slots={
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS),
                    "h_el": X(_ELEMS, ("g_el",)),
                },
            ),
            Template(
                name="centralizer_class_size",
                latex=r"|{G}| = |C_{{{G}}}({g_el})| \cdot |[{g_el}]|",
                slots={
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS),
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
                latex=r"{phi}({g_el} {h_el}) = {phi}({g_el})\,{phi}({h_el})",
                slots={
                    "phi": S(_HOMOS),
                    "g_el": S(_ELEMS),
                    "h_el": X(_ELEMS, ("g_el",)),
                },
            ),
            Template(
                name="homomorphism_identity_inverse",
                latex=r"{phi}({g_el}^{{-1}}) = {phi}({g_el})^{{-1}},\quad {phi}(e) = e",
                slots={
                    "phi": S(_HOMOS),
                    "g_el": S(_ELEMS),
                },
            ),
            Template(
                name="isomorphism_bijective_homo",
                latex=r"{phi} : {G} \to {H} \text{{ iso.}} \iff {phi} \text{{ bij. hom.}}",
                slots={
                    "phi": S(_HOMOS),
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE, ("G",)),
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
                    "phi": S(_HOMOS),
                    "G": S(_G_POOL),
                    "g_el": S(_ELEMS),
                },
            ),
            Template(
                name="kernel_normal_embedding",
                latex=r"\ker {phi} \trianglelefteq {G},\quad {G}/\ker {phi} \hookrightarrow {H}",
                slots={
                    "phi": S(_HOMOS),
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE, ("G",)),
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
                    "H": X(_SIMPLE, ("G",)),
                },
            ),
            Template(
                name="direct_product_three",
                latex=r"{G} \times {H} \times {N_sub}",
                slots={
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE, ("G",)),
                    "N_sub": X(_SIMPLE, ("G", "H")),
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
                    "N_sub": X(_SIMPLE, ("G",)),
                    "H": X(_SIMPLE, ("G", "N_sub")),
                },
            ),
            Template(
                name="semidirect_product_conditions",
                latex=r"{N_sub} \trianglelefteq {G},\quad {G} = {N_sub}{H},\quad {N_sub} \cap {H} = \{{e\}}",
                slots={
                    "G": S(_G_POOL),
                    "N_sub": X(_SIMPLE, ("G",)),
                    "H": X(_SIMPLE, ("G", "N_sub")),
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
                    r"{g_el}^{{{nv}}} = {h_el}^2 = e,\; {h_el} {g_el} {h_el}^{{-1}} = {g_el}^{{-1}} \rangle"
                ),
                slots={
                    "nv": S(_NV_POOL),
                    "g_el": S(_ELEMS),
                    "h_el": X(_ELEMS, ("g_el",)),
                },
            ),
            Template(
                name="cyclic_presentation",
                latex=r"\mathbb{{Z}}/{nv}\mathbb{{Z}} = \langle {g_el} \mid {g_el}^{{{nv}}} = e \rangle",
                slots={
                    "nv": S(_NV_POOL),
                    "g_el": S(_ELEMS),
                },
            ),
            Template(
                name="quaternion_presentation",
                latex=(
                    r"Q_8 = \langle {g_el}, {h_el} \mid "
                    r"{g_el}^4 = e,\; {g_el}^2 = {h_el}^2,\; {h_el} {g_el} {h_el}^{{-1}} = {g_el}^{{-1}} \rangle"
                ),
                slots={
                    "g_el": S(_ELEMS),
                    "h_el": X(_ELEMS, ("g_el",)),
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
                    "g_el": S(_ELEMS),
                },
            ),
            Template(
                name="free_group_rank_2_nonabelian",
                latex=r"F_2 = \langle {g_el}, {h_el} \rangle,\quad [{g_el},{h_el}] \neq e",
                slots={
                    "g_el": S(_ELEMS),
                    "h_el": X(_ELEMS, ("g_el",)),
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
                    "H": X(_SIMPLE, ("G",)),
                    "phi": S(_HOMOS),
                    "psi": X(_HOMOS, ("phi",)),
                },
            ),
            Template(
                name="short_exact_sequence_normal",
                latex=r"1 \to {N_sub} \to {G} \to {G}/{N_sub} \to 1",
                slots={
                    "G": S(_G_POOL),
                    "N_sub": X(_SIMPLE, ("G",)),
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
                    "phi": S(_HOMOS),
                    "psi": X(_HOMOS, ("phi",)),
                },
            ),
            Template(
                name="exactness_injective",
                latex=r"0 \to {H} \xrightarrow{{{phi}}} {G} \text{{ exact}} \iff {phi} \text{{ injective}}",
                slots={
                    "G": S(_G_POOL),
                    "H": X(_SIMPLE, ("G",)),
                    "phi": S(_HOMOS),
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
            "H": X(_SIMPLE, ("G",)),
            "N_sub": X(_SIMPLE, ("G", "H")),
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
                    "H": X(_SIMPLE, ("G",)),
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
                latex=r"|GL_{{{nv}}}(\mathbb{{F}}_{{{qv}}})| = \prod{lim_mod}_{{k=0}}^{{{nv}-1}} ({qv}^{{{nv}}} - {qv}^k)",
                slots={
                    "lim_mod": _LIM_MOD,
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
                    "g_el": S(_ELEMS),
                },
            ),
        ],
    )
)

# c=30 — Burnside / orbit partition / action maps
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="burnside_lemma",
        latex=(
            r"|{H}/{G}|"
            r" = \tfrac{{1}}{{|{G}|}}\sum{lim_mod}_{{{g_el} \in {G}}} \left|{H}^{{{g_el}}}\right|"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "G": S(_G_POOL),
            "H": X(_SIMPLE, ("G",)),
            "g_el": S(_ELEMS),
        },
    )
)

# c=31 — fixed-point subset
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="fixed_point_subset",
        latex=(
            r"{H}^{{{g_el}}}"
            r" = \left\{{{h_el} \in {H} : {g_el} \cdot {h_el} = {h_el}\right\}}"
        ),
        slots={
            "H": S(_SIMPLE),
            "g_el": S(_ELEMS),
            "h_el": X(_ELEMS, ("g_el",)),
        },
    )
)

# c=32 — orbit-coset isomorphism (transitive actions)
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="orbit_coset_iso",
        latex=r"{G}/\operatorname{{Stab}}_{{{G}}}({g_el}) \cong {G} \cdot {g_el}",
        slots={
            "G": S(_G_POOL),
            "g_el": S(_ELEMS),
        },
    )
)

# c=33 — faithful action injection
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="action_faithful",
        latex=(
            r"{phi} : {G} \hookrightarrow \operatorname{{Sym}}({H})"
            r"\text{{ faithful}} \iff"
            r" \bigcap_{{{g_el} \in {H}}} \operatorname{{Stab}}_{{{G}}}({g_el}) = \{{e\}}"
        ),
        slots={
            "phi": S(_HOMOS),
            "G": S(_G_POOL),
            "H": X(_SIMPLE, ("G",)),
            "g_el": S(_ELEMS),
        },
    )
)

# c=34 — group extension / five-term SES
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="group_extension_ses",
        latex=(
            r"1 \to {N_sub} \xrightarrow{{{phi}}} {G}"
            r" \xrightarrow{{{psi}}} {H} \to 1"
        ),
        slots={
            "G": S(_G_POOL),
            "N_sub": X(_SIMPLE, ("G",)),
            "H": X(_SIMPLE, ("G", "N_sub")),
            "phi": S(_HOMOS),
            "psi": X(_HOMOS, ("phi",)),
        },
    )
)

# c=35 — split extension / semidirect with explicit twist
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="split_extension",
        latex=(
            r"{G} \cong {N_sub} \rtimes_{{{phi}}} {H},"
            r"\quad {phi} : {H} \to \operatorname{{Aut}}({N_sub})"
        ),
        slots={
            "G": S(_G_POOL),
            "N_sub": X(_SIMPLE, ("G",)),
            "H": X(_SIMPLE, ("G", "N_sub")),
            "phi": S(_HOMOS),
        },
    )
)

# c=36 — second cohomology classifies extensions
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="second_cohomology_extensions",
        latex=r"H^2({H},\, {N_sub}) \cong \operatorname{{Ext}}({H},\, {N_sub})",
        slots={
            "H": S(_SIMPLE),
            "N_sub": X(_SIMPLE, ("H",)),
        },
    )
)

# c=37 — lower central series / nilpotency class
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="lower_central_series",
        latex=(
            r"{G} = {G}_0 \trianglerighteq {G}_1 \trianglerighteq \cdots"
            r" \trianglerighteq {G}_{{{n}}} = 1,"
            r"\quad {G}_k = [{G}, {G}_{{k-1}}]"
        ),
        slots={
            "G": S(_SIMPLE),  # named groups (e.g. \mathbb{Z}_{n}) already carry a subscript
            "n": S(_N_POOL),
        },
    )
)

# c=38 — derived series / solvability
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="derived_series_solvable",
        latex=(
            r"{G}^{{(0)}} = {G} \supset {G}^{{(1)}}"
            r" \supset \cdots \supset {G}^{{({n})}} = 1"
        ),
        slots={
            "G": S(_G_POOL),
            "n": S(_N_POOL),
        },
    )
)

# c=39 — nilpotent class definition
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="nilpotent_class",
        latex=(
            r"\gamma_{{{n}}}({G}) = 1,"
            r"\quad \gamma_{{{n}-1}}({G}) \neq 1,"
            r"\quad \operatorname{{cl}}({G}) = {n}"
        ),
        slots={
            "G": S(_G_POOL),
            "n": S(_N_POOL),
        },
    )
)

# c=40 — Schur–Zassenhaus
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="schur_zassenhaus",
        latex=(
            r"\gcd(|{N_sub}|, [{G}:{N_sub}]) = 1"
            r" \implies {G} \cong {N_sub} \rtimes {H}"
        ),
        slots={
            "G": S(_G_POOL),
            "N_sub": X(_SIMPLE, ("G",)),
            "H": X(_SIMPLE, ("G", "N_sub")),
        },
    )
)

# c=41 — Jordan–Hölder uniqueness statement
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="jordan_holder",
        latex=(
            r"\text{{Any two composition series of }} {G}"
            r" \text{{ have the same factors up to iso and reordering}}"
        ),
        slots={"G": S(_G_POOL)},
    )
)

# c=42 — Frattini subgroup
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="frattini_subgroup",
        latex=r"\Phi({G}) = \bigcap_{{M \text{{ max}} \leq {G}}} M \trianglelefteq {G}",
        slots={"G": S(_G_POOL)},
    )
)

# c=43 — transfer homomorphism
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="transfer_homomorphism",
        latex=(
            r"\operatorname{{Ver}} : {G} \to {H}/[{H},{H}],"
            r"\quad {phi}({g_el}) = \prod{lim_mod}_{{t}} t{g_el} t^{{-1}} \bmod [{H},{H}]"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "G": S(_G_POOL),
            "H": X(_SIMPLE, ("G",)),
            "phi": S(_HOMOS),
            "g_el": S(_ELEMS),
        },
    )
)

# ---------------------------------------------------------------------------
# Part C — high-n_eff function-decorated templates (_FN_SLOT)
# ---------------------------------------------------------------------------

# c=44
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="fn_group_order",
        latex=r"{fn1}(|{G}|) = {fn2}([{G}:{H}] \cdot |{H}|)",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "G": S(_G_POOL),
            "H": X(_SIMPLE, ("G",)),
        },
    )
)

# c=45
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="fn_index_tower",
        latex=r"{fn1}([{G}:{K}]) = {fn2}([{G}:{H}] \cdot [{H}:{K}])",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "G": S(_G_POOL),
            "H": X(_SIMPLE, ("G",)),
            "K": X(_SIMPLE, ("G", "H")),
        },
    )
)

# c=46
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="fn_conjugation_apply",
        latex=r"{fn1}({g_el} {h_el} {g_el}^{{-1}}) = {fn2}({h_el})",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "g_el": S(_ELEMS),
            "h_el": X(_ELEMS, ("g_el",)),
        },
    )
)

# c=47
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="fn_orbit_stabilizer",
        latex=(
            r"{fn1}(|{G}|)"
            r" = {fn2}(|\operatorname{{Orb}}_{{{G}}}({g_el})| \cdot |\operatorname{{Stab}}_{{{G}}}({g_el})|)"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "G": S(_G_POOL),
            "g_el": S(_ELEMS),
        },
    )
)

# c=48
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="fn_hom_kernel",
        latex=(
            r"{fn1}(\ker {phi})"
            r" = {fn2}\!\left(\{{g \in {G} : {phi}({g_el}) = e\}}\right)"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "phi": S(_HOMOS),
            "G": S(_G_POOL),
            "g_el": S(_ELEMS),
        },
    )
)

# c=49
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="fn_sylow_decomposition",
        latex=(
            r"{fn1}(|{G}|) = {fn2}({p}^{{{n}}} m),"
            r"\quad \gcd({p}, m) = 1"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "G": S(_G_POOL),
            "p": S(_P_POOL),
            "n": S(_N_POOL),
        },
    )
)

# c=50
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="fn_abelianization",
        latex=r"{fn1}({G}^{{\mathrm{{ab}}}}) = {fn2}({G}/[{G},{G}])",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "G": S(_G_POOL),
        },
    )
)

# c=51
_GROUP_THEORY_TEMPLATES.append(
    Template(
        name="fn_class_equation",
        latex=(
            r"{fn1}(|{G}|)"
            r" = {fn2}\!\left(|Z({G})| + \sum{lim_mod}_{{{g_el}}} [{G} : C_{{{G}}}({g_el})]\right)"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "G": S(_G_POOL),
            "g_el": S(_ELEMS),
        },
    )
)

_GROUP_THEORY_TEMPLATES += [
    # \vartriangleleft / \vartriangleright — normal subgroup relation
    Template(
        name="normal_subgroup_left",
        latex=r"{NN} \vartriangleleft {GG}",
        slots={"NN": S(_SIMPLE), "GG": S(_G_POOL)},
    ),
    Template(
        name="normal_subgroup_quotient",
        latex=r"{NN} \vartriangleleft {GG} \Rightarrow {GG}/{NN} \text{{ is a group}}",
        slots={"NN": S(_SIMPLE), "GG": S(_G_POOL)},
    ),
    Template(
        name="normal_subgroup_right",
        latex=r"{GG} \vartriangleright {NN}",
        slots={"NN": S(_SIMPLE), "GG": S(_G_POOL)},
    ),
    # \ltimes — left semidirect product
    Template(
        name="semidirect_left",
        latex=r"{GG} = {KK} \ltimes {NN}",
        slots={"GG": S(_SIMPLE), "KK": S(_SIMPLE), "NN": S(_SIMPLE)},
        distinct=[["GG", "KK", "NN"]],
    ),
    Template(
        name="semidirect_left_action",
        latex=r"{GG} = {NN} \rtimes {KK} \cong {KK} \ltimes {NN}",
        slots={"GG": S(_SIMPLE), "KK": S(_SIMPLE), "NN": S(_SIMPLE)},
        distinct=[["GG", "KK", "NN"]],
    ),
    # \wr — wreath product
    Template(
        name="wreath_product",
        latex=r"{GG} \wr S_{{{nn}}}",
        slots={"GG": S(_SIMPLE), "nn": S(_NV_POOL)},
    ),
    Template(
        name="wreath_product_iterated",
        latex=r"{GG} \wr {HH} \cong {GG}^{{|{HH}|}} \rtimes {HH}",
        slots={"GG": S(_SIMPLE), "HH": S(_SIMPLE)},
        distinct=[["GG", "HH"]],
    ),
]

# ---------------------------------------------------------------------------

# Homomorphism / conjugation templates (from algebra domain)
_GROUP_THEORY_TEMPLATES += [
    Template(
        name="group_homomorphism_rightarrow",
        latex=r"\phi: {GG} \rightarrow {HH} \text{{ is a group homomorphism}}",
        slots={"GG": S(_GRP_NAMES), "HH": X(_GRP_NAMES, ("GG",))},
    ),
    Template(
        name="conjugation_mapsto",
        latex=r"\phi_h: {gg} \mapsto h{gg}h^{{-1}}, \quad {gg} \in {GG}",
        slots={"gg": S(_ELT_POOL), "GG": S(_GRP_NAMES)},
    ),
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("group_theory", _GROUP_THEORY_TEMPLATES)
