"""Topology domain generators."""

from __future__ import annotations

from .._template_dsl import _FN_SLOT, _LIM_MOD, S, Template, X
from ._config import register_domain

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

# Notation-style pools (primary OCR diversity levers)
_SPACE_POOL = ("X", "Y", "M", r"\mathcal{M}", r"\mathcal{X}", "B", "E")
_OPEN_POOL = ("U", "V", "W", r"\mathcal{U}", r"\mathcal{V}")
_CLOSED_POOL = ("A", "B", "F", "K", "C")
_MAP_POOL = ("f", "g", r"\phi", r"\psi", r"\varphi", "h", "p")
_IDX_POOL = ("n", "m", "k", "p", "q")
_G_POOL = (
    r"\mathbb{Z}",
    r"\mathbb{Z}/p\mathbb{Z}",
    r"\mathbb{Q}",
    r"\mathbb{R}",
    r"\mathbb{Z}/2\mathbb{Z}",
)
_DIST_POOL = ("d", r"d_X", r"d_Y", r"\rho", r"\sigma")
_OMEGA_POOL = (r"\omega", r"\Omega", r"\eta", r"\theta")
_FIBER_POOL = ("F", r"\mathcal{F}", r"\mathcal{E}", "E")
_CC_POOL = ("c", "L", r"\kappa")
_EPS_POOL = (r"\varepsilon", r"\epsilon", r"\delta")
_PT_POOL = ("y", "z", "w", "u", "v")
_DM_POOL = ("n", "m", "d", "N")

# ---------------------------------------------------------------------------
# Topology templates
# ---------------------------------------------------------------------------

_TOPOLOGY_TEMPLATES: list[Template] = [
    # ------------------------------------------------------------------
    # Part A: Reparameterized originals (16)
    # ------------------------------------------------------------------
    Template(
        name="fundamental_group",
        latex=r"\pi_{{{nn}}}({sp}, x_0)",
        slots={"sp": S(_SPACE_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="closure",
        latex=(
            r"\overline{{{aa}}} = \bigcap\!\left\{{{ff}"
            r" \supseteq {aa} : {ff} \text{{ closed}}\right\}}"
        ),
        slots={"aa": S(_CLOSED_POOL), "ff": X(_CLOSED_POOL, ("aa",))},
    ),
    Template(
        name="triangle_inequality",
        latex=r"{dd}(x, z) \leq {dd}(x, y) + {dd}(y, z)",
        slots={"dd": S(_DIST_POOL)},
    ),
    Template(
        name="quotient_space",
        latex=r"{sp} / {{\sim}} \overset{{{mp}}}{{\longrightarrow}} {tp}",
        slots={"sp": S(_SPACE_POOL), "mp": S(_MAP_POOL), "tp": X(_SPACE_POOL, ("sp",))},
    ),
    Template(
        name="hausdorff",
        latex=(
            r"\forall\, x \neq y \in {sp},\;"
            r"\exists\, {uu} \ni x,\; {vv} \ni y :"
            r"\; {uu} \cap {vv} = \emptyset"
        ),
        slots={"sp": S(_SPACE_POOL), "uu": S(_OPEN_POOL), "vv": X(_OPEN_POOL, ("uu",))},
    ),
    Template(
        name="interior_closure",
        latex=r"\operatorname{{int}}({aa}) \subseteq {aa} \subseteq \overline{{{aa}}}",
        slots={"aa": S(_CLOSED_POOL)},
    ),
    Template(
        name="homotopy_group_sphere",
        latex=r"\pi_{{{nn}}}(S^{{{nn}}}) \cong \mathbb{{Z}}",
        slots={"nn": S(_IDX_POOL)},
    ),
    Template(
        name="euler_characteristic",
        latex=r"\chi({sp}) = \sum{lim_mod}_{{{nn} \geq 0}} (-1)^{{{nn}}} b_{{{nn}}}({sp})",
        slots={"lim_mod": _LIM_MOD, "sp": S(_SPACE_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="boundary_squared",
        latex=r"\partial_{{{nn}}} \circ \partial_{{{nn}+1}} = 0",
        slots={"nn": S(_IDX_POOL)},
    ),
    Template(
        name="induced_map_fundamental_group",
        latex=r"{ff}_* : \pi_1({sp}, x_0) \to \pi_1({tp}, {ff}(x_0))",
        slots={"ff": S(_MAP_POOL), "sp": S(_SPACE_POOL), "tp": X(_SPACE_POOL, ("sp",))},
    ),
    Template(
        name="stokes_topology",
        latex=r"\int{lim_mod}_{{\partial {mm}}} {om} = \int{lim_mod}_{{{mm}}} d{om}",
        slots={"lim_mod": _LIM_MOD, "mm": S(_SPACE_POOL), "om": S(_OMEGA_POOL)},
    ),
    Template(
        name="homology_disjoint_union",
        latex=(
            r"H_{{{nn}}}({sp} \sqcup {tp})"
            r" \cong H_{{{nn}}}({sp}) \oplus H_{{{nn}}}({tp})"
        ),
        slots={"sp": S(_SPACE_POOL), "tp": X(_SPACE_POOL, ("sp",)), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="long_exact_sequence",
        latex=(
            r"\cdots \to H_{{{nn}}}({aa})"
            r" \to H_{{{nn}}}({sp})"
            r" \to H_{{{nn}}}({sp}, {aa})"
            r" \to H_{{{nn}-1}}({aa}) \to \cdots"
        ),
        slots={"sp": S(_SPACE_POOL), "aa": S(_CLOSED_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="covering_space",
        latex=(
            r"{ff}_* : \pi_1(\tilde{{{sp}}}, \tilde{{x}}_0)"
            r" \hookrightarrow \pi_1({sp}, x_0)"
        ),
        slots={"sp": S(_SPACE_POOL), "ff": S(_MAP_POOL)},
    ),
    Template(
        name="homotopy_equivalence_homology",
        latex=(
            r"{sp} \simeq {tp}"
            r" \implies H_{{{nn}}}({sp}) \cong H_{{{nn}}}({tp})"
        ),
        slots={"sp": S(_SPACE_POOL), "tp": X(_SPACE_POOL, ("sp",)), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="brouwer_fixed_point",
        latex=(
            r"{ff} : D^{{{nn}}} \to D^{{{nn}}} \text{{ continuous}}"
            r" \implies \exists\, x : {ff}(x) = x"
        ),
        slots={"ff": S(_MAP_POOL), "nn": S(_IDX_POOL)},
    ),
    # ------------------------------------------------------------------
    # Part B: New flat standalone templates
    # ------------------------------------------------------------------
    # Point-set topology (10)
    Template(
        name="open_set_union",
        latex=(
            r"\bigcup_{{\alpha}} {uu}_\alpha \in \tau"
            r" \quad \text{{for any family of open sets}}"
        ),
        slots={"uu": S(_OPEN_POOL)},
    ),
    Template(
        name="open_set_finite_intersection",
        latex=r"{uu} \cap {vv} \in \tau \quad ({uu},\, {vv} \text{{ open}})",
        slots={"uu": S(_OPEN_POOL), "vv": X(_OPEN_POOL, ("uu",))},
    ),
    Template(
        name="continuous_preimage",
        latex=(
            r"{ff} \text{{ continuous}}"
            r" \iff {ff}^{{-1}}({uu}) \text{{ open}}"
            r" \;\forall\, {uu} \in \tau"
        ),
        slots={"ff": S(_MAP_POOL), "uu": S(_OPEN_POOL)},
    ),
    Template(
        name="compactness_finite_subcover",
        latex=(
            r"\forall \{{{uu}_\alpha\}}"
            r" \text{{ open cover of }} {sp},"
            r" \;\exists \text{{ finite subcover}}"
        ),
        slots={"sp": S(_SPACE_POOL), "uu": S(_OPEN_POOL)},
    ),
    Template(
        name="connected_def",
        latex=(
            r"{sp} = {uu} \cup {vv},\;"
            r"{uu} \cap {vv} = \emptyset,\;"
            r"{uu},{vv} \text{{ open}}"
            r" \implies {uu} = \emptyset \text{{ or }} {vv} = \emptyset"
        ),
        slots={"sp": S(_SPACE_POOL), "uu": S(_OPEN_POOL), "vv": X(_OPEN_POOL, ("uu",))},
    ),
    Template(
        name="path_connected_def",
        latex=(
            r"\forall\, x,y \in {sp},\;"
            r"\exists\, {gg} : [0,1] \to {sp} \text{{ cts}},"
            r"\; {gg}(0)=x,\; {gg}(1)=y"
        ),
        slots={"sp": S(_SPACE_POOL), "gg": S(_MAP_POOL)},
    ),
    Template(
        name="urysohn_lemma",
        latex=(
            r"{aa}, {bb} \text{{ disjoint closed in }} {sp}"
            r" \implies \exists\, {ff} : {sp} \to [0,1],"
            r"\; {ff}|_{{{aa}}} = 0,\; {ff}|_{{{bb}}} = 1"
        ),
        slots={
            "sp": S(_SPACE_POOL),
            "aa": S(_CLOSED_POOL),
            "bb": X(_CLOSED_POOL, ("aa",)),
            "ff": S(_MAP_POOL),
        },
    ),
    Template(
        name="product_topology_subbasis",
        latex=(
            r"\pi_{{{nn}}}^{{-1}}({uu})"
            r" \text{{ form a sub-basis for }} \prod{lim_mod}_{{{nn}}} {sp}_{{{nn}}}"
        ),
        slots={"lim_mod": _LIM_MOD, "sp": S(_SPACE_POOL), "uu": S(_OPEN_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="tychonoff_theorem",
        latex=(
            r"\Bigl(\forall\, {nn} : {sp}_{{{nn}}} \text{{ compact}}\Bigr)"
            r" \implies \prod{lim_mod}_{{{nn}}} {sp}_{{{nn}}} \text{{ compact}}"
        ),
        slots={"lim_mod": _LIM_MOD, "sp": S(_SPACE_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="baire_category_theorem",
        latex=(
            r"{sp} \text{{ complete metric}}"
            r" \implies \bigcap_{{k=1}}^\infty {uu}_k"
            r" \text{{ dense (dense open }} {uu}_k\text{{)}}"
        ),
        slots={"sp": S(_SPACE_POOL), "uu": S(_OPEN_POOL)},
    ),
    # Metric spaces (8)
    Template(
        name="open_ball_def",
        latex=r"B_{{{eps}}}(x) = \{{{yy} \in {sp} : {dd}(x, {yy}) < {eps}\}}",
        slots={
            "sp": S(_SPACE_POOL),
            "dd": S(_DIST_POOL),
            "eps": S(_EPS_POOL),
            "yy": S(_PT_POOL),
        },
    ),
    Template(
        name="cauchy_sequence",
        latex=(
            r"\forall\, {eps} > 0\;"
            r"\exists\, N :\; n, m \geq N"
            r" \implies {dd}(x_n, x_m) < {eps}"
        ),
        slots={"dd": S(_DIST_POOL), "eps": S(_EPS_POOL)},
    ),
    Template(
        name="completeness_def",
        latex=(
            r"{sp} \text{{ complete}}"
            r" \iff \text{{every Cauchy sequence in }} {sp} \text{{ converges}}"
        ),
        slots={"sp": S(_SPACE_POOL)},
    ),
    Template(
        name="isometry_def",
        latex=r"{dd}({ff}(x), {ff}(y)) = {dd}(x, y) \;\forall\, x, y",
        slots={"dd": S(_DIST_POOL), "ff": S(_MAP_POOL)},
    ),
    Template(
        name="lipschitz_map",
        latex=r"{dd}({ff}(x), {ff}(y)) \leq K\, {dd}(x, y)",
        slots={"dd": S(_DIST_POOL), "ff": S(_MAP_POOL)},
    ),
    Template(
        name="contraction_mapping",
        latex=(
            r"{dd}({ff}(x), {ff}(y)) \leq {cc}\, {dd}(x, y),\;"
            r"{cc} < 1"
            r" \implies \exists!\, x^* : {ff}(x^*) = x^*"
        ),
        slots={"dd": S(_DIST_POOL), "ff": S(_MAP_POOL), "cc": S(_CC_POOL)},
    ),
    Template(
        name="total_boundedness",
        latex=(
            r"\forall\, {eps} > 0,\;"
            r"{sp} \subseteq \bigcup_{{i=1}}^n B_{{{eps}}}(x_i)"
        ),
        slots={"sp": S(_SPACE_POOL), "eps": S(_EPS_POOL)},
    ),
    Template(
        name="metric_equivalence_norms",
        latex=r"{cc}_1 \|x\| \leq {dd}(x, 0) \leq {cc}_2 \|x\|",
        slots={"dd": S(_DIST_POOL), "cc": S(_CC_POOL)},
    ),
    # Fundamental group and homotopy (8)
    Template(
        name="seifert_van_kampen",
        latex=(
            r"\pi_1({sp})"
            r" \cong \pi_1({aa}) *_{{\pi_1({aa} \cap {bb})}}"
            r" \pi_1({bb})"
        ),
        slots={
            "sp": S(_SPACE_POOL),
            "aa": S(_CLOSED_POOL),
            "bb": X(_CLOSED_POOL, ("aa",)),
        },
    ),
    Template(
        name="fundamental_group_product",
        latex=(
            r"\pi_1({sp} \times {tp})"
            r" \cong \pi_1({sp}) \times \pi_1({tp})"
        ),
        slots={"sp": S(_SPACE_POOL), "tp": X(_SPACE_POOL, ("sp",))},
    ),
    Template(
        name="fundamental_group_circle",
        latex=r"\pi_1(S^1) \cong \mathbb{Z}",
        slots={},
    ),
    Template(
        name="homotopy_relative",
        latex=r"{ff} \simeq {gg} \;\mathrm{{rel}}\; A",
        slots={"ff": S(_MAP_POOL), "gg": X(_MAP_POOL, ("ff",))},
    ),
    Template(
        name="homotopy_class_composition",
        latex=r"[{ff}] \cdot [{gg}] \in \pi_1({sp}, x_0)",
        slots={"ff": S(_MAP_POOL), "gg": X(_MAP_POOL, ("ff",)), "sp": S(_SPACE_POOL)},
    ),
    Template(
        name="deck_transformation",
        latex=(
            r"\mathrm{{Deck}}(\tilde{{{sp}}}/{sp})"
            r" \cong \pi_1({sp}) / p_* \pi_1(\tilde{{{sp}}})"
        ),
        slots={"sp": S(_SPACE_POOL)},
    ),
    Template(
        name="van_kampen_free_product",
        latex=(
            r"\pi_1({sp} \vee {tp})"
            r" \cong \pi_1({sp}) * \pi_1({tp})"
        ),
        slots={"sp": S(_SPACE_POOL), "tp": X(_SPACE_POOL, ("sp",))},
    ),
    Template(
        name="homotopy_type_invariance",
        latex=(
            r"{sp} \simeq {tp}"
            r" \implies \pi_{{{nn}}}({sp}) \cong \pi_{{{nn}}}({tp})"
        ),
        slots={"sp": S(_SPACE_POOL), "tp": X(_SPACE_POOL, ("sp",)), "nn": S(_IDX_POOL)},
    ),
    # Homology (10)
    Template(
        name="singular_homology_def",
        latex=(
            r"H_{{{nn}}}({sp})"
            r" = \ker \partial_{{{nn}}} / \mathrm{{im}}\, \partial_{{{nn}+1}}"
        ),
        slots={"sp": S(_SPACE_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="homology_with_coeffs",
        latex=(
            r"H_{{{nn}}}({sp};\, {gc})"
            r" = H_{{{nn}}}(C_*({sp}) \otimes {gc})"
        ),
        slots={"sp": S(_SPACE_POOL), "nn": S(_IDX_POOL), "gc": S(_G_POOL)},
    ),
    Template(
        name="excision_theorem",
        latex=(
            r"H_{{{nn}}}({sp} \setminus {uu},\,"
            r"{aa} \setminus {uu})"
            r" \cong H_{{{nn}}}({sp}, {aa})"
        ),
        slots={
            "sp": S(_SPACE_POOL),
            "aa": S(_CLOSED_POOL),
            "uu": S(_OPEN_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="mayer_vietoris",
        latex=(
            r"\cdots \to H_{{{nn}}}({aa} \cap {bb})"
            r" \to H_{{{nn}}}({aa}) \oplus H_{{{nn}}}({bb})"
            r" \to H_{{{nn}}}({sp})"
            r" \to H_{{{nn}-1}}({aa} \cap {bb}) \to \cdots"
        ),
        slots={
            "sp": S(_SPACE_POOL),
            "aa": S(_CLOSED_POOL),
            "bb": X(_CLOSED_POOL, ("aa",)),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="kunneth_formula",
        latex=(
            r"H_{{{nn}}}({sp} \times {tp};\, {gc})"
            r" \cong \bigoplus_{{p+q={nn}}}"
            r" H_p({sp};\, {gc}) \otimes H_q({tp};\, {gc})"
        ),
        slots={
            "sp": S(_SPACE_POOL),
            "tp": X(_SPACE_POOL, ("sp",)),
            "gc": S(_G_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="universal_coefficients_thm",
        latex=(
            r"H^{{{nn}}}({sp};\, {gc})"
            r" \cong \mathrm{{Hom}}(H_{{{nn}}}({sp}), {gc})"
            r" \oplus \mathrm{{Ext}}(H_{{{nn}-1}}({sp}), {gc})"
        ),
        slots={"sp": S(_SPACE_POOL), "gc": S(_G_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="betti_number_def",
        latex=r"b_{{{nn}}}({sp}) = \mathrm{{rank}}\, H_{{{nn}}}({sp};\, \mathbb{{Z}})",
        slots={"sp": S(_SPACE_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="poincare_duality",
        latex=(
            r"H^{{{nn}}}({sp};\, {gc})"
            r" \cong H_{{{dm}-{nn}}}({sp};\, {gc})"
        ),
        slots={
            "sp": S(_SPACE_POOL),
            "gc": S(_G_POOL),
            "nn": S(_IDX_POOL),
            "dm": X(_DM_POOL, ("nn",)),
        },
    ),
    Template(
        name="relative_long_exact_sequence",
        latex=(
            r"\cdots \to H_{{{nn}}}({aa})"
            r" \xrightarrow{{i_*}} H_{{{nn}}}({sp})"
            r" \xrightarrow{{j_*}} H_{{{nn}}}({sp}, {aa})"
            r" \xrightarrow{{\partial}} H_{{{nn}-1}}({aa}) \to \cdots"
        ),
        slots={"sp": S(_SPACE_POOL), "aa": S(_CLOSED_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="reduced_homology_suspension",
        latex=(
            r"\tilde{{H}}_{{{nn}}}(\Sigma {sp})"
            r" \cong \tilde{{H}}_{{{nn}-1}}({sp})"
        ),
        slots={"sp": S(_SPACE_POOL), "nn": S(_IDX_POOL)},
    ),
    # Cohomology and de Rham (6)
    Template(
        name="de_rham_cohomology",
        latex=(
            r"H^{{{nn}}}_{{dR}}({sp})"
            r" = \ker d^{{{nn}}} / \mathrm{{im}}\, d^{{{nn}-1}}"
        ),
        slots={"sp": S(_SPACE_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="de_rham_isomorphism",
        latex=(
            r"H^{{{nn}}}_{{dR}}({sp})"
            r" \cong H^{{{nn}}}({sp};\, \mathbb{{R}})"
        ),
        slots={"sp": S(_SPACE_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="cup_product_cohomology",
        latex=(
            r"\smile : H^{{{pp}}}({sp}) \otimes H^{{{qq}}}({sp})"
            r" \to H^{{{pp}+{qq}}}({sp})"
        ),
        slots={"sp": S(_SPACE_POOL), "pp": S(_IDX_POOL), "qq": X(_IDX_POOL, ("pp",))},
    ),
    Template(
        name="cap_product_cohomology",
        latex=(
            r"\frown : H^{{{pp}}}({sp}) \otimes H_{{{nn}}}({sp})"
            r" \to H_{{{nn}-{pp}}}({sp})"
        ),
        slots={"sp": S(_SPACE_POOL), "pp": S(_IDX_POOL), "nn": X(_IDX_POOL, ("pp",))},
    ),
    Template(
        name="chern_class_first",
        latex=r"c_1({fb}) \in H^2({sp};\, \mathbb{{Z}})",
        slots={"fb": S(_FIBER_POOL), "sp": S(_SPACE_POOL)},
    ),
    Template(
        name="euler_class",
        latex=r"e({fb}) \in H^{{{nn}}}({sp};\, \mathbb{{Z}})",
        slots={"fb": S(_FIBER_POOL), "sp": S(_SPACE_POOL), "nn": S(_IDX_POOL)},
    ),
    # Manifolds and smooth topology (7)
    Template(
        name="tangent_bundle_dim",
        latex=r"\dim T{sp} = 2\, \dim {sp}",
        slots={"sp": S(_SPACE_POOL)},
    ),
    Template(
        name="differential_map",
        latex=r"d{ff}_x : T_x {sp} \to T_{{{ff}(x)}} {tp}",
        slots={"ff": S(_MAP_POOL), "sp": S(_SPACE_POOL), "tp": X(_SPACE_POOL, ("sp",))},
    ),
    Template(
        name="regular_value_theorem",
        latex=(
            r"{ff}^{{-1}}(y) \text{{ is a submanifold}}"
            r" \text{{ if }} y \text{{ is a regular value of }} {ff}"
        ),
        slots={"ff": S(_MAP_POOL)},
    ),
    Template(
        name="degree_of_map",
        latex=(
            r"\deg({ff})"
            r" = \sum{lim_mod}_{{x \in {ff}^{{-1}}(y)}}"
            r" \operatorname{{sign}}\, d{ff}_x"
        ),
        slots={"lim_mod": _LIM_MOD, "ff": S(_MAP_POOL)},
    ),
    Template(
        name="morse_inequality",
        latex=r"b_{{{nn}}}({sp}) \leq c_{{{nn}}}({sp})",
        slots={"sp": S(_SPACE_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="lefschetz_number",
        latex=(
            r"L({ff}) = \sum{lim_mod}_{{k=0}}^{{{nn}}} (-1)^k"
            r" \mathrm{{Tr}}\bigl({ff}_* : H_k \to H_k\bigr)"
        ),
        slots={"lim_mod": _LIM_MOD, "ff": S(_MAP_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="lefschetz_fixed_point",
        latex=r"L({ff}) \neq 0 \implies {ff} \text{{ has a fixed point}}",
        slots={"ff": S(_MAP_POOL)},
    ),
    # Covering spaces and fibrations (3)
    Template(
        name="lifting_criterion",
        latex=(
            r"{ff}_*(\pi_1({sp}, x_0))"
            r" \subseteq p_*(\pi_1(\tilde{{{tp}}}, \tilde{{x}}_0))"
        ),
        slots={"ff": S(_MAP_POOL), "sp": S(_SPACE_POOL), "tp": X(_SPACE_POOL, ("sp",))},
    ),
    Template(
        name="fibration_long_exact",
        latex=(
            r"\cdots \to \pi_{{{nn}}}({fb})"
            r" \to \pi_{{{nn}}}({sp})"
            r" \to \pi_{{{nn}}}({tp})"
            r" \to \pi_{{{nn}-1}}({fb}) \to \cdots"
        ),
        slots={
            "fb": S(_FIBER_POOL),
            "sp": S(_SPACE_POOL),
            "tp": X(_SPACE_POOL, ("sp",)),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="hurewicz_homomorphism",
        latex=(
            r"h : \pi_{{{nn}}}({sp}) \to H_{{{nn}}}({sp})"
            r" \text{{ iso if }} \pi_k({sp})=0 \text{{ for }} k < {nn}"
        ),
        slots={"sp": S(_SPACE_POOL), "nn": S(_IDX_POOL)},
    ),
    # ------------------------------------------------------------------
    # Part C: High-n_eff function-pair templates (8)
    # ------------------------------------------------------------------
    Template(
        name="homotopy_def",
        latex=(
            r"H : {sp} \times [0,1] \to {tp}"
            r" \text{{ with }} H(\cdot,0) = {fn1},"
            r"\; H(\cdot,1) = {fn2}"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "sp": S(_SPACE_POOL),
            "tp": X(_SPACE_POOL, ("sp",)),
        },
    ),
    Template(
        name="chain_map_homology",
        latex=(
            r"{fn1}_* : H_{{{nn}}}(C_*) \to H_{{{nn}}}(D_*)"
            r" \text{{ induced by chain map }} {fn1}"
        ),
        slots={"fn1": _FN_SLOT, "nn": S(_IDX_POOL)},
    ),
    Template(
        name="cohomology_functoriality",
        latex=(
            r"{fn1}^* \circ {fn2}^*"
            r" = ({fn2} \circ {fn1})^*"
            r" : H^*({sp}) \to H^*({tp})"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "sp": S(_SPACE_POOL),
            "tp": X(_SPACE_POOL, ("sp",)),
        },
    ),
    Template(
        name="degree_composition",
        latex=r"\deg({fn1} \circ {fn2}) = \deg({fn1}) \cdot \deg({fn2})",
        slots={"fn1": _FN_SLOT, "fn2": _FN_SLOT},
    ),
    Template(
        name="homology_functoriality",
        latex=r"({fn1} \circ {fn2})_* = {fn1}_* \circ {fn2}_*",
        slots={"fn1": _FN_SLOT, "fn2": _FN_SLOT},
    ),
    Template(
        name="natural_transformation_square",
        latex=r"{fn2}_* \circ i_* = j_* \circ {fn1}_*",
        slots={"fn1": _FN_SLOT, "fn2": _FN_SLOT},
    ),
    Template(
        name="induced_iso_homotopy_equiv",
        latex=(
            r"{fn1} \simeq {fn2}"
            r" \implies {fn1}_* = {fn2}_*"
            r" : H_{{{nn}}}({sp}) \to H_{{{nn}}}({tp})"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "sp": S(_SPACE_POOL),
            "tp": X(_SPACE_POOL, ("sp",)),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="relative_map_induced",
        latex=(
            r"{fn1}_* : H_{{{nn}}}({sp}, {aa})"
            r" \to H_{{{nn}}}({tp}, {bb})"
            r" \text{{ for }} {fn1} : ({sp},{aa}) \to ({tp},{bb})"
        ),
        slots={
            "fn1": _FN_SLOT,
            "nn": S(_IDX_POOL),
            "sp": S(_SPACE_POOL),
            "tp": X(_SPACE_POOL, ("sp",)),
            "aa": S(_CLOSED_POOL),
            "bb": X(_CLOSED_POOL, ("aa",)),
        },
    ),
]

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("topology", _TOPOLOGY_TEMPLATES)
