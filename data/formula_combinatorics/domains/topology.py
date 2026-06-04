"""Topology domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import S, Template, compute_weights, make_dispatcher

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_SPACE_POOL: list[str] = ["X", "Y", "M", "S"]
_SET_POOL: list[str] = ["A", "U", "V", "K"]
_MAP_POOL: list[str] = [r"\phi", r"\psi", "f"]

# ---------------------------------------------------------------------------
# Topology templates
# ---------------------------------------------------------------------------

_TOPOLOGY_TEMPLATES: list[Template] = [
    Template(
        name="fundamental_group",
        latex=r"\pi_1({X}, x_0)",
        slots={"X": S(_SPACE_POOL)},
    ),
    Template(
        name="closure",
        latex=r"\overline{{{A}}} = \bigcap \left\{{F \supseteq {A} : F \text{{ closed}}\right\}}",
        slots={"A": S(_SET_POOL)},
    ),
    Template(
        name="triangle_inequality",
        latex=r"d(x, z) \leq d(x, y) + d(y, z)",
        slots={},
    ),
    Template(
        name="quotient_space",
        latex=r"{X} / {{\sim}}",
        slots={"X": S(_SPACE_POOL)},
    ),
    Template(
        name="hausdorff",
        latex=r"\forall x \neq y \in {X},\; \exists U \ni x,\, V \ni y : U \cap V = \emptyset",
        slots={"X": S(_SPACE_POOL)},
    ),
    Template(
        name="interior_closure",
        latex=r"\operatorname{{int}}({A}) \subseteq {A} \subseteq \overline{{{A}}}",
        slots={"A": S(_SET_POOL)},
    ),
    Template(
        name="homotopy_group_sphere",
        latex=r"\pi_n(S^n) \cong \mathbb{{Z}}",
        slots={},
    ),
    Template(
        name="euler_characteristic",
        latex=r"\chi({X}) = \sum_k (-1)^k b_k",
        slots={"X": S(_SPACE_POOL)},
    ),
    Template(
        name="boundary_squared",
        latex=r"\partial^2 = 0",
        slots={},
    ),
    Template(
        name="induced_map_fundamental_group",
        latex=r"{f}_* : \pi_1({X}, x_0) \to \pi_1(Y, {f}(x_0))",
        slots={"f": S(_MAP_POOL), "X": S(_SPACE_POOL)},
    ),
    Template(
        name="stokes_topology",
        latex=r"\int_{{\partial M}} \omega = \int_M d\omega",
        slots={},
    ),
    Template(
        name="homology_disjoint_union",
        latex=r"H_n({X}) \oplus H_n(Y) \cong H_n({X} \sqcup Y)",
        slots={"X": S(_SPACE_POOL)},
    ),
    Template(
        name="long_exact_sequence",
        latex=r"\cdots \to H_n({A}) \to H_n({X}) \to H_n({X}, {A}) \to H_{{n-1}}({A}) \to \cdots",
        slots={"X": S(_SPACE_POOL), "A": S(_SET_POOL)},
    ),
    Template(
        name="covering_space",
        latex=r"p_* : \pi_1(\tilde{{{X}}}, \tilde{{x}}_0) \hookrightarrow \pi_1({X}, x_0)",
        slots={"X": S(_SPACE_POOL)},
    ),
    Template(
        name="homotopy_equivalence_homology",
        latex=r"{X} \simeq Y \implies H_n({X}) \cong H_n(Y)",
        slots={"X": S(_SPACE_POOL)},
    ),
    Template(
        name="brouwer_fixed_point",
        latex=r"f : D^n \to D^n \text{{ continuous}} \implies \exists x : f(x) = x",
        slots={},
    ),
]

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

_W_TOPOLOGY: list[float] = compute_weights(_TOPOLOGY_TEMPLATES)

# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

_topology = make_dispatcher(_TOPOLOGY_TEMPLATES, _W_TOPOLOGY)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "topology": _topology,
}

WEIGHTS: dict[str, float] = {
    "topology": 0.04,
}

TEMPLATES: dict[str, list[Template]] = {
    "topology": _TOPOLOGY_TEMPLATES,
}
