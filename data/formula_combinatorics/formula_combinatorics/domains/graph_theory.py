"""Graph theory domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import E, Template, compute_weights, make_dispatcher

# ---------------------------------------------------------------------------
# Inline sub-generators
# ---------------------------------------------------------------------------


def _n_graph(rng: random.Random) -> str:
    return rng.choice(["n", "m"])


# ---------------------------------------------------------------------------
# Graph theory templates
# ---------------------------------------------------------------------------

_GRAPH_THEORY_TEMPLATES: list[Template] = [
    # c=0 — handshaking lemma (fixed)
    Template(
        name="handshaking_lemma",
        latex=r"\sum_{v \in V} \deg(v) = 2|E|",
        slots={},
    ),
    # c=1 — complete graph edge count
    Template(
        name="complete_graph_edges",
        latex=r"|E(K_{{{n}}})| = \binom{{{n}}}{{2}} = \frac{{{n}({n}-1)}}{{2}}",
        slots={"n": E(_n_graph, n=2)},
    ),
    # c=2 — chromatic number bound (fixed)
    Template(
        name="chromatic_number_bound",
        latex=r"\chi(G) \leq \Delta(G) + 1",
        slots={},
    ),
    # c=3 — Euler's formula for planar graphs (fixed)
    Template(
        name="euler_planar_formula",
        latex=r"|V| - |E| + |F| = 2",
        slots={},
    ),
    # c=4 — degree as neighbourhood size (fixed)
    Template(
        name="degree_definition",
        latex=r"\deg(v) = |\{u \in V : \{u,v\} \in E\}|",
        slots={},
    ),
    # c=5 — planar graph edge bound (fixed)
    Template(
        name="planar_edge_bound",
        latex=r"|E| \leq 3|V| - 6",
        slots={},
    ),
    # c=6 — connectivity inequality (fixed)
    Template(
        name="connectivity_inequality",
        latex=r"\kappa(G) \leq \kappa'(G) \leq \delta(G)",
        slots={},
    ),
    # c=7 — chromatic number of complete graph
    Template(
        name="chromatic_complete_graph",
        latex=r"\chi(K_{{{n}}}) = {n}",
        slots={"n": E(_n_graph, n=2)},
    ),
    # c=8 — independence number + vertex cover (fixed)
    Template(
        name="independence_vertex_cover",
        latex=r"\alpha(G) + \tau(G) = |V|",
        slots={},
    ),
    # c=9 — metric triangle inequality for graph distance (fixed)
    Template(
        name="graph_distance_triangle",
        latex=r"d(u,v) \leq d(u,w) + d(w,v)",
        slots={},
    ),
    # c=10 — spectral ordering of eigenvalues (fixed)
    Template(
        name="spectral_ordering",
        latex=r"\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n",
        slots={},
    ),
    # c=11 — degree sequence sum (fixed)
    Template(
        name="degree_sequence_sum",
        latex=r"2|E| = \sum_{k \geq 0} k \cdot n_k",
        slots={},
    ),
    # c=12 — Ramsey bound (fixed)
    Template(
        name="ramsey_bound",
        latex=r"R(s,t) \leq \binom{s+t-2}{s-1}",
        slots={},
    ),
    # c=13 — Turán's theorem
    Template(
        name="turan_theorem",
        latex=r"ex({n}, K_{{r+1}}) = \left(1 - \frac{{1}}{{r}}\right) \frac{{{n}^2}}{{2}}",
        slots={"n": E(_n_graph, n=2)},
    ),
]

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

_W_GT: list[float] = compute_weights(_GRAPH_THEORY_TEMPLATES)

# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

_graph_theory = make_dispatcher(_GRAPH_THEORY_TEMPLATES, _W_GT)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "graph_theory": _graph_theory,
}

WEIGHTS: dict[str, float] = {
    "graph_theory": 0.02,
}

TEMPLATES: dict[str, list[Template]] = {
    "graph_theory": _GRAPH_THEORY_TEMPLATES,
}
