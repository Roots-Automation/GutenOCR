"""Graph theory domain generators."""

from __future__ import annotations

from .._template_dsl import _LIM_MOD, E, S, Template, X, register_domain
from .._vocab import _STATS_N as _N_POOL
from .._vocab import _fn_rich_nosub

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_GRAPH_POOL = ("G", "H", r"\Gamma", "D", "T")
_VTX_POOL = ("u", "v", "w", "x", "y")
_R_POOL = ("r", "s", "t", "k")
_P_POOL = ("p", "q", r"\rho")
_LAM_POOL = (
    r"\lambda_1",
    r"\lambda_2",
    r"\lambda_n",
    r"\lambda_{\min}",
    r"\lambda_{\max}",
)

# ---------------------------------------------------------------------------
# Graph theory templates
# ---------------------------------------------------------------------------

_GRAPH_THEORY_TEMPLATES: list[Template] = [
    # ------------------------------------------------------------------
    # Part A: reparameterized originals
    # ------------------------------------------------------------------
    Template(
        name="handshaking_lemma",
        latex=r"\sum{lim_mod}_{{v \in V({gg})}} \deg_{{{gg}}}(v) = 2|E({gg})|",
        slots={"lim_mod": _LIM_MOD, "gg": S(_GRAPH_POOL)},
    ),
    Template(
        name="complete_graph_edges",
        latex=(
            r"|E(K_{{{nn}}})| = \binom{{{nn}}}{{2}}"
            r" = \frac{{{nn}({nn}-1)}}{{2}}"
        ),
        slots={"nn": S(_N_POOL)},
    ),
    Template(
        name="chromatic_number_bound",
        latex=r"\chi({gg}) \leq \Delta({gg}) + 1",
        slots={"gg": S(_GRAPH_POOL)},
    ),
    Template(
        name="euler_planar_formula",
        latex=r"|V| - |E| + |F| = 2",
        slots={},
    ),
    Template(
        name="degree_definition",
        latex=(
            r"\deg_{{{gg}}}({vv})"
            r" = |\{{{uu} \in V({gg}) : \{{{uu},{vv}\}} \in E({gg})\}}|"
        ),
        slots={"gg": S(_GRAPH_POOL), "vv": S(_VTX_POOL), "uu": X(_VTX_POOL, ("vv",))},
    ),
    Template(
        name="planar_edge_bound",
        latex=r"|E({gg})| \leq 3|V({gg})| - 6",
        slots={"gg": S(_GRAPH_POOL)},
    ),
    Template(
        name="connectivity_inequality",
        latex=r"\kappa({gg}) \leq \kappa'({gg}) \leq \delta({gg})",
        slots={"gg": S(_GRAPH_POOL)},
    ),
    Template(
        name="chromatic_complete_graph",
        latex=r"\chi(K_{{{nn}}}) = {nn}",
        slots={"nn": S(_N_POOL)},
    ),
    Template(
        name="independence_vertex_cover",
        latex=r"\alpha({gg}) + \tau({gg}) = |V({gg})|",
        slots={"gg": S(_GRAPH_POOL)},
    ),
    Template(
        name="graph_distance_triangle",
        latex=r"d_{{{gg}}}({uu},{vv}) \leq d_{{{gg}}}({uu},{ww}) + d_{{{gg}}}({ww},{vv})",
        slots={
            "gg": S(_GRAPH_POOL),
            "uu": S(_VTX_POOL),
            "vv": X(_VTX_POOL, ("uu",)),
            "ww": X(_VTX_POOL, ("uu", "vv")),
        },
    ),
    Template(
        name="spectral_ordering",
        latex=(
            r"\lambda_1({gg}) \geq \lambda_2({gg})"
            r" \geq \cdots \geq \lambda_{{{nn}}}({gg})"
        ),
        slots={"gg": S(_GRAPH_POOL), "nn": S(_N_POOL)},
    ),
    Template(
        name="degree_sequence_sum",
        latex=r"2|E({gg})| = \sum{lim_mod}_{{k \geq 0}} k \cdot n_k({gg})",
        slots={"lim_mod": _LIM_MOD, "gg": S(_GRAPH_POOL)},
    ),
    Template(
        name="ramsey_bound",
        latex=r"R({ss},{tt}) \leq \binom{{{ss}+{tt}-2}}{{{ss}-1}}",
        slots={"ss": S(_R_POOL), "tt": X(_R_POOL, ("ss",))},
    ),
    Template(
        name="turan_theorem",
        latex=(
            r"\mathrm{{ex}}({nn}, K_{{{rr}+1}})"
            r" = \Bigl(1 - \tfrac{{1}}{{{rr}}}\Bigr)\frac{{{nn}^2}}{{2}}"
        ),
        slots={"nn": S(_N_POOL), "rr": S(_R_POOL)},
    ),
    # ------------------------------------------------------------------
    # Part B: new flat standalone templates
    # ------------------------------------------------------------------
    # Trees
    Template(
        name="tree_edge_count",
        latex=r"|E({gg})| = |V({gg})| - 1",
        slots={"gg": S(_GRAPH_POOL)},
    ),
    Template(
        name="matrix_tree_theorem",
        latex=r"\tau({gg}) = \det\!\bigl(L_{{ij}}({gg})\bigr)",
        slots={"gg": S(_GRAPH_POOL)},
    ),
    Template(
        name="cayley_formula",
        latex=r"\tau(K_{{{nn}}}) = {nn}^{{{nn}-2}}",
        slots={"nn": S(_N_POOL)},
    ),
    # Spectral
    Template(
        name="graph_laplacian_def",
        latex=r"L({gg}) = D({gg}) - A({gg})",
        slots={"gg": S(_GRAPH_POOL)},
    ),
    Template(
        name="laplacian_eigenvalue_zero",
        latex=r"\lambda_{{\min}}\!\bigl(L({gg})\bigr) = 0",
        slots={"gg": S(_GRAPH_POOL)},
    ),
    Template(
        name="fiedler_connectivity",
        latex=(
            r"\lambda_2\!\bigl(L({gg})\bigr) > 0"
            r" \iff {gg} \text{{ is connected}}"
        ),
        slots={"gg": S(_GRAPH_POOL)},
    ),
    Template(
        name="adjacency_eigenvalue_bound",
        latex=r"|{lam}(A({gg}))| \leq \Delta({gg})",
        slots={"gg": S(_GRAPH_POOL), "lam": S(_LAM_POOL)},
    ),
    Template(
        name="cheeger_inequality",
        latex=(
            r"\frac{{h({gg})^2}}{{2}}"
            r" \leq \lambda_2(L({gg}))"
            r" \leq 2\,h({gg})"
        ),
        slots={"gg": S(_GRAPH_POOL)},
    ),
    Template(
        name="path_count_adjacency",
        latex=(
            r"\bigl(A({gg})^{{{kk}}}\bigr)_{{{uu}{vv}}}"
            r" = \#\text{{walks of length }}{kk}"
            r" \text{{ from }}{uu}\text{{ to }}{vv}"
        ),
        slots={
            "gg": S(_GRAPH_POOL),
            "kk": S(_R_POOL),
            "uu": S(_VTX_POOL),
            "vv": X(_VTX_POOL, ("uu",)),
        },
    ),
    # Colorings
    Template(
        name="chromatic_polynomial_def",
        latex=r"P({gg}, k) = \#\text{{proper }}k\text{{-colorings of }}{gg}",
        slots={"gg": S(_GRAPH_POOL)},
    ),
    Template(
        name="edge_coloring_vizing",
        latex=r"\chi'({gg}) \in \{{\Delta({gg}),\;\Delta({gg})+1\}}",
        slots={"gg": S(_GRAPH_POOL)},
    ),
    Template(
        name="brooks_theorem",
        latex=(
            r"\chi({gg}) \leq \Delta({gg})"
            r" \text{{ for connected }}{gg}"
            r" \text{{ not }}K_{{{nn}}}\text{{ or odd cycle}}"
        ),
        slots={"gg": S(_GRAPH_POOL), "nn": S(_N_POOL)},
    ),
    Template(
        name="four_color_theorem",
        latex=r"\chi({gg}) \leq 4 \text{{ for every planar }}{gg}",
        slots={"gg": S(_GRAPH_POOL)},
    ),
    # Bipartite and matchings
    Template(
        name="bipartite_edge_bound",
        latex=r"|E({gg})| \leq \left\lfloor \frac{{|V({gg})|^2}}{{4}} \right\rfloor",
        slots={"gg": S(_GRAPH_POOL)},
    ),
    Template(
        name="konig_theorem",
        latex=r"\nu({gg}) = \tau({gg}) \quad \text{{({gg} bipartite)}}",
        slots={"gg": S(_GRAPH_POOL)},
    ),
    Template(
        name="hall_condition",
        latex=(
            r"{gg} \text{{ has a perfect matching}}"
            r" \iff |N(S)| \geq |S|"
            r" \;\forall\, S \subseteq A"
        ),
        slots={"gg": S(_GRAPH_POOL)},
    ),
    Template(
        name="matching_gallai",
        latex=r"\nu({gg}) + \rho({gg}) = |V({gg})|",
        slots={"gg": S(_GRAPH_POOL)},
    ),
    # Flows and cuts
    Template(
        name="max_flow_min_cut",
        latex=(
            r"\max_f \operatorname{{val}}(f)"
            r" = \operatorname{{cap}}(S^*, T^*)"
            r" \quad \text{{in }}{gg}"
        ),
        slots={"gg": S(_GRAPH_POOL)},
    ),
    Template(
        name="menger_theorem",
        latex=(
            r"\kappa({gg}) = \min_{{{uu},{vv}}}"
            r" \lambda\!\bigl({gg};\,{uu},{vv}\bigr)"
        ),
        slots={
            "gg": S(_GRAPH_POOL),
            "uu": S(_VTX_POOL),
            "vv": X(_VTX_POOL, ("uu",)),
        },
    ),
    # Random graphs
    Template(
        name="erdos_renyi_degree",
        latex=(
            r"\mathbb{{E}}[\deg(v)]"
            r" = ({nn}-1){pp}"
            r" \quad \text{{in }}G({nn},{pp})"
        ),
        slots={"nn": S(_N_POOL), "pp": S(_P_POOL)},
    ),
    Template(
        name="random_graph_connectivity_threshold",
        latex=(
            r"G({nn}, p)\text{{ is a.s.\ connected for }}"
            r"p = \frac{{\ln {nn}}}{{{nn}}}"
        ),
        slots={"nn": S(_N_POOL)},
    ),
    # Clique / independence
    Template(
        name="clique_complement",
        latex=r"\omega({gg}) = \alpha\!\left(\bar{{{gg}}}\right)",
        slots={"gg": S(_GRAPH_POOL)},
    ),
    Template(
        name="ramsey_lower_bound",
        latex=r"R(k,k) \geq 2^{{{kk}/2}}",
        slots={"kk": S(_R_POOL)},
    ),
    # Diameter / radius
    Template(
        name="diameter_eccentricity",
        latex=(
            r"\operatorname{{diam}}({gg})"
            r" = \max_{{{uu} \in V({gg})}} \varepsilon({uu})"
        ),
        slots={"gg": S(_GRAPH_POOL), "uu": S(_VTX_POOL)},
    ),
    Template(
        name="radius_diameter_bound",
        latex=(
            r"\operatorname{{rad}}({gg})"
            r" \leq \operatorname{{diam}}({gg})"
            r" \leq 2\,\operatorname{{rad}}({gg})"
        ),
        slots={"gg": S(_GRAPH_POOL)},
    ),
    # Density
    Template(
        name="graph_density",
        latex=(r"d({gg}) = \frac{{2\,|E({gg})|}}{{|V({gg})|\,(|V({gg})|-1)}}"),
        slots={"gg": S(_GRAPH_POOL)},
    ),
    # Isomorphism
    Template(
        name="graph_isomorphism_degree_seq",
        latex=(
            r"{g1} \cong {g2}"
            r" \implies \deg\text{{-seq}}({g1}) = \deg\text{{-seq}}({g2})"
        ),
        slots={"g1": S(_GRAPH_POOL), "g2": X(_GRAPH_POOL, ("g1",))},
    ),
    # Euler circuit
    Template(
        name="euler_circuit_condition",
        latex=(
            r"{gg} \text{{ has an Euler circuit}}"
            r" \iff {gg} \text{{ connected, all degrees even}}"
        ),
        slots={"gg": S(_GRAPH_POOL)},
    ),
    # ------------------------------------------------------------------
    # Part C: high-n_eff function-pair templates
    # ------------------------------------------------------------------
    Template(
        name="flow_capacity_bound",
        latex=(
            r"0 \leq {fn1}({uu},{vv}) \leq {fn2}({uu},{vv})"
            r" \quad \forall\,({uu},{vv}) \in E({gg})"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "uu": S(_VTX_POOL),
            "vv": X(_VTX_POOL, ("uu",)),
            "gg": S(_GRAPH_POOL),
        },
    ),
    Template(
        name="distance_weight_relaxation",
        latex=(
            r"{fn1}({vv}) \leq {fn1}({uu}) + {fn2}({uu},{vv})"
            r" \quad \forall\,{uu} \in N({vv})"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "uu": S(_VTX_POOL),
            "vv": X(_VTX_POOL, ("uu",)),
        },
    ),
    Template(
        name="bellman_ford_update",
        latex=(
            r"{fn1}({vv}) = \min_{{{uu} \in N({vv})}}"
            r" \bigl\{{{fn1}({uu}) + {fn2}({uu},{vv})\bigr\}}"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "uu": S(_VTX_POOL),
            "vv": X(_VTX_POOL, ("uu",)),
        },
    ),
    Template(
        name="graph_hom_composition",
        latex=(
            r"{fn1} \circ {fn2} : {g1} \to {g3}"
            r" \text{{ is a graph homomorphism}}"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "g1": S(_GRAPH_POOL),
            "g3": X(_GRAPH_POOL, ("g1",)),
        },
    ),
    Template(
        name="kirchhoff_potential",
        latex=(
            r"\sum{lim_mod}_{{{uu} \sim {vv}}} {fn2}({uu},{vv})"
            r"\,\bigl({fn1}({uu}) - {fn1}({vv})\bigr) = 0"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "uu": S(_VTX_POOL),
            "vv": X(_VTX_POOL, ("uu",)),
        },
    ),
    Template(
        name="spectral_quadratic_form",
        latex=(
            r"{fn1}^\top A({gg})\,{fn2}"
            r" = \sum{lim_mod}_{{({uu},{vv}) \in E({gg})}}"
            r" \bigl({fn1}({uu}){fn2}({vv})"
            r" + {fn1}({vv}){fn2}({uu})\bigr)"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "gg": S(_GRAPH_POOL),
            "uu": S(_VTX_POOL),
            "vv": X(_VTX_POOL, ("uu",)),
        },
    ),
]

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("graph_theory", _GRAPH_THEORY_TEMPLATES, 0.02)
