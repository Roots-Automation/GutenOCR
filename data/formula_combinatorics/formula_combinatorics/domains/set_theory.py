"""Set theory domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import S, Template, X, compute_weights, make_dispatcher
from .._vocab import _BBOLD, _PROPS, _SETS

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_SETS_POOL: list[str] = list(_SETS) if not isinstance(_SETS, list) else _SETS
_PROPS_POOL: list[str] = list(_PROPS) if not isinstance(_PROPS, list) else _PROPS
_BBOLD_POOL: list[str] = list(_BBOLD) if not isinstance(_BBOLD, list) else _BBOLD
_VARS_POOL: list[str] = ["x", "y", "z", "a", "b", "n", "k"]
_N_POOL: list[str] = ["n", "m"]

# ---------------------------------------------------------------------------
# Set theory templates
# ---------------------------------------------------------------------------

_SET_THEORY_TEMPLATES: list[Template] = [
    Template(
        name="commutativity",
        latex=r"{A} \cup {B} = {B} \cup {A}, \quad {A} \cap {B} = {B} \cap {A}",
        slots={"A": S(_SETS_POOL), "B": X(_SETS_POOL, ["A"])},
    ),
    Template(
        name="subset_intersection",
        latex=r"{A} \subseteq {B} \iff {A} \cap {B} = {A}",
        slots={"A": S(_SETS_POOL), "B": X(_SETS_POOL, ["A"])},
    ),
    Template(
        name="inclusion_exclusion_two",
        latex=r"|{A} \cup {B}| = |{A}| + |{B}| - |{A} \cap {B}|",
        slots={"A": S(_SETS_POOL), "B": X(_SETS_POOL, ["A"])},
    ),
    Template(
        name="power_set_size",
        latex=r"|\mathcal{{P}}({A})| = 2^{{|{A}|}}",
        slots={"A": S(_SETS_POOL)},
    ),
    Template(
        name="set_difference",
        latex=r"{A} \setminus {B} = \{{{v} \in {A} \mid {v} \notin {B}\}}",
        slots={"A": S(_SETS_POOL), "B": X(_SETS_POOL, ["A"]), "v": S(_VARS_POOL)},
    ),
    Template(
        name="de_morgan_union",
        latex=r"({A} \cup {B})^c = {A}^c \cap {B}^c",
        slots={"A": S(_SETS_POOL), "B": X(_SETS_POOL, ["A"])},
    ),
    Template(
        name="de_morgan_intersection",
        latex=r"({A} \cap {B})^c = {A}^c \cup {B}^c",
        slots={"A": S(_SETS_POOL), "B": X(_SETS_POOL, ["A"])},
    ),
    Template(
        name="distributive_law",
        latex=r"{A} \cap ({B} \cup {C}) = ({A} \cap {B}) \cup ({A} \cap {C})",
        slots={"A": S(_SETS_POOL), "B": X(_SETS_POOL, ["A"]), "C": X(_SETS_POOL, ["A", "B"])},
    ),
    Template(
        name="cartesian_product",
        latex=r"{A} \times {B} = \{{(a, b) \mid a \in {A},\; b \in {B}\}}",
        slots={"A": S(_SETS_POOL), "B": X(_SETS_POOL, ["A"])},
    ),
    Template(
        name="power_of_set",
        latex=r"|{A}^{{{n}}}| = |{A}|^{{{n}}}",
        slots={"A": S(_SETS_POOL), "n": S(_N_POOL)},
    ),
    Template(
        name="symmetric_difference",
        latex=r"{A} \triangle {B} = ({A} \setminus {B}) \cup ({B} \setminus {A})",
        slots={"A": S(_SETS_POOL), "B": X(_SETS_POOL, ["A"])},
    ),
    Template(
        name="set_builder",
        latex=r"\{{{v} \in {bb} \mid {P}({v})\}}",
        slots={"v": S(_VARS_POOL), "bb": S(_BBOLD_POOL), "P": S(_PROPS_POOL)},
    ),
    Template(
        name="inclusion_exclusion_general",
        latex=(
            r"\left|\bigcup_{{i=1}}^{{{n}}} {A}_i\right| "
            r"= \sum_i |{A}_i| - \sum_{{i < j}} |{A}_i \cap {A}_j| + \cdots"
        ),
        slots={"A": S(_SETS_POOL), "n": S(_N_POOL)},
    ),
    Template(
        name="set_equality",
        latex=r"{A} = {B} \iff {A} \subseteq {B} \land {B} \subseteq {A}",
        slots={"A": S(_SETS_POOL), "B": X(_SETS_POOL, ["A"])},
    ),
    Template(
        name="cantor_theorem",
        latex=r"|{bb}| < |\mathcal{{P}}({bb})|",
        slots={"bb": S(_BBOLD_POOL)},
    ),
    Template(
        name="continuum_hypothesis",
        latex=r"\aleph_0 < \aleph_1, \quad |\mathbb{{R}}| = 2^{{\aleph_0}}",
        slots={},
    ),
]

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

_W_SET: list[float] = compute_weights(_SET_THEORY_TEMPLATES)

# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

_set_theory = make_dispatcher(_SET_THEORY_TEMPLATES, _W_SET)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "set_theory": _set_theory,
}

WEIGHTS: dict[str, float] = {
    "set_theory": 0.05,
}

TEMPLATES: dict[str, list[Template]] = {
    "set_theory": _SET_THEORY_TEMPLATES,
}
