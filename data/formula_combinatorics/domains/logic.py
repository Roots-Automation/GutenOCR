"""Logic domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import S, Template, X, compute_weights, make_dispatcher
from .._vocab import _BBOLD, _PROPS

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_PROPS_POOL: list[str] = list(_PROPS) if not isinstance(_PROPS, list) else _PROPS
_BBOLD_POOL: list[str] = list(_BBOLD) if not isinstance(_BBOLD, list) else _BBOLD
_VARS_POOL: list[str] = ["x", "y", "z", "a", "b", "n", "k"]

# ---------------------------------------------------------------------------
# Logic templates
# ---------------------------------------------------------------------------

_LOGIC_TEMPLATES: list[Template] = [
    Template(
        name="de_morgan_and",
        latex=r"\neg({p} \land {q}) \equiv \neg {p} \lor \neg {q}",
        slots={"p": S(_PROPS_POOL), "q": X(_PROPS_POOL, ["p"])},
    ),
    Template(
        name="de_morgan_or",
        latex=r"\neg({p} \lor {q}) \equiv \neg {p} \land \neg {q}",
        slots={"p": S(_PROPS_POOL), "q": X(_PROPS_POOL, ["p"])},
    ),
    Template(
        name="implication_disjunction",
        latex=r"{p} \Rightarrow {q} \equiv \neg {p} \lor {q}",
        slots={"p": S(_PROPS_POOL), "q": X(_PROPS_POOL, ["p"])},
    ),
    Template(
        name="biconditional",
        latex=r"{p} \Leftrightarrow {q} \equiv ({p} \Rightarrow {q}) \land ({q} \Rightarrow {p})",
        slots={"p": S(_PROPS_POOL), "q": X(_PROPS_POOL, ["p"])},
    ),
    Template(
        name="xor",
        latex=r"{p} \oplus {q} \equiv ({p} \lor {q}) \land \neg({p} \land {q})",
        slots={"p": S(_PROPS_POOL), "q": X(_PROPS_POOL, ["p"])},
    ),
    Template(
        name="universal_quantifier",
        latex=r"\forall {v} \in {bb},\; {p}({v})",
        slots={"v": S(_VARS_POOL), "bb": S(_BBOLD_POOL), "p": S(_PROPS_POOL)},
    ),
    Template(
        name="existential_quantifier",
        latex=r"\exists {v} \in {bb} : {p}({v})",
        slots={"v": S(_VARS_POOL), "bb": S(_BBOLD_POOL), "p": S(_PROPS_POOL)},
    ),
    Template(
        name="associativity_and",
        latex=r"({p} \land {q}) \land {r} \equiv {p} \land ({q} \land {r})",
        slots={"p": S(_PROPS_POOL), "q": X(_PROPS_POOL, ["p"]), "r": X(_PROPS_POOL, ["p", "q"])},
    ),
    Template(
        name="distributive_and_or",
        latex=r"{p} \land ({q} \lor {r}) \equiv ({p} \land {q}) \lor ({p} \land {r})",
        slots={"p": S(_PROPS_POOL), "q": X(_PROPS_POOL, ["p"]), "r": X(_PROPS_POOL, ["p", "q"])},
    ),
    Template(
        name="excluded_middle",
        latex=r"{p} \lor \neg {p} \equiv \top",
        slots={"p": S(_PROPS_POOL)},
    ),
    Template(
        name="contradiction",
        latex=r"{p} \land \neg {p} \equiv \bot",
        slots={"p": S(_PROPS_POOL)},
    ),
    Template(
        name="double_negation",
        latex=r"\neg\neg {p} \equiv {p}",
        slots={"p": S(_PROPS_POOL)},
    ),
    Template(
        name="hypothetical_syllogism",
        latex=r"({p} \Rightarrow {q}) \land ({q} \Rightarrow {r}) \Rightarrow ({p} \Rightarrow {r})",
        slots={"p": S(_PROPS_POOL), "q": X(_PROPS_POOL, ["p"]), "r": X(_PROPS_POOL, ["p", "q"])},
    ),
    Template(
        name="quantifier_negation",
        latex=r"\neg \forall {v} \, {p}({v}) \equiv \exists {v} \, \neg {p}({v})",
        slots={"v": S(_VARS_POOL), "p": S(_PROPS_POOL)},
    ),
    Template(
        name="unique_existence",
        latex=r"\exists! {v} \in {bb} : {p}({v})",
        slots={"v": S(_VARS_POOL), "bb": S(_BBOLD_POOL), "p": S(_PROPS_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

_W_LOGIC: list[float] = compute_weights(_LOGIC_TEMPLATES)

# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

_logic = make_dispatcher(_LOGIC_TEMPLATES, _W_LOGIC)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "logic": _logic,
}

WEIGHTS: dict[str, float] = {
    "logic": 0.04,
}

TEMPLATES: dict[str, list[Template]] = {
    "logic": _LOGIC_TEMPLATES,
}
