"""Representation theory and character theory domain generator."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import S, Template, X, compute_weights, make_dispatcher

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_HOMOS = [r"\phi", r"\varphi", r"\psi", "f", r"\theta", r"\rho", r"\pi"]
_ELEMS = ["g", "h", "a", "b", "x", "y", r"\sigma", r"\tau", r"\alpha", r"\beta"]
_REPS = [r"\rho", r"\pi", r"\sigma", "V", "W"]
_SIMPLE = ["G", "H", "K", "N", "A", "B"]

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_REPR_TEMPLATES: list[Template] = [
    # c == 0: character trace / dimension
    Template(
        name="character_trace",
        latex=r"\chi_{{{rep}}}({g_el}) = \operatorname{{tr}}\!\left({rep}({g_el})\right)",
        slots={"rep": S(_REPS), "g_el": S(_ELEMS)},
    ),
    Template(
        name="character_dimension",
        latex=r"\chi_{{{rep}}}(e) = \dim {rep}",
        slots={"rep": S(_REPS)},
    ),
    # c == 1: orthogonality
    Template(
        name="character_orthogonality",
        latex=r"\frac{{1}}{{|{G}|}} \sum_{{{g_el} \in {G}}} \chi_i({g_el})\,\overline{{\chi_j({g_el})}} = \delta_{{ij}}",
        slots={"G": S(_SIMPLE), "g_el": S(_ELEMS)},
    ),
    Template(
        name="character_centralizer",
        latex=r"\sum_{{i}} |\chi_i({g_el})|^2 = |C_{{{G}}}({g_el})|",
        slots={"G": S(_SIMPLE), "g_el": S(_ELEMS)},
    ),
    # c == 2: decomposition
    Template(
        name="irreducible_decomposition",
        latex=r"{rep} \cong \bigoplus_i {rep}_i^{{\oplus n_i}},\quad {rep}_i \text{{ irreducible}}",
        slots={"rep": S(_REPS)},
    ),
    Template(
        name="sum_of_squares",
        latex=r"\sum_i (\dim {rep}_i)^2 = |{G}|",
        slots={"rep": S(_REPS), "G": S(_SIMPLE)},
    ),
    # c == 3: Burnside / orbit counting
    Template(
        name="burnside_orbit",
        latex=r"|X/{G}| = \frac{{1}}{{|{G}|}} \sum_{{{g_el} \in {G}}} |X^{{{g_el}}}|",
        slots={"G": S(_SIMPLE), "g_el": S(_ELEMS)},
    ),
    Template(
        name="burnside_fixed_points",
        latex=r"|\text{{orbits}}| = \frac{{1}}{{|{G}|}} \sum_{{{g_el} \in {G}}} |\{{x : {g_el} \cdot x = x\}}|",
        slots={"G": S(_SIMPLE), "g_el": S(_ELEMS)},
    ),
    # c == 4: Schur's lemma
    Template(
        name="schur_linear_map",
        latex=r"{phi} : {rep} \to {rep} \text{{ {G}-linear}} \implies {phi} = \lambda \operatorname{{Id}}",
        slots={"phi": S(_HOMOS), "rep": S(_REPS), "G": S(_SIMPLE)},
    ),
    Template(
        name="schur_hom_space",
        latex=r"\operatorname{{Hom}}_{{{G}}}({rep}_i, {rep}_j) \cong \begin{{cases}} \mathbb{{C}} & i = j \\ 0 & i \neq j \end{{cases}}",
        slots={"G": S(_SIMPLE), "rep": S(_REPS)},
    ),
    # c == 5: induced / restricted
    Template(
        name="induced_representation",
        latex=r"\operatorname{{Ind}}_{{{H}}}^{{{G}}} {rep} \cong \mathbb{{C}}[{G}] \otimes_{{\mathbb{{C}}[{H}]}} {rep}",
        slots={"G": S(_SIMPLE), "H": X(_SIMPLE, ["G"]), "rep": S(_REPS)},
    ),
    Template(
        name="frobenius_reciprocity",
        latex=r"\langle \operatorname{{Ind}}_{{{H}}}^{{{G}}} \chi, \psi \rangle_{{{G}}} = \langle \chi, \operatorname{{Res}}_{{{H}}}^{{{G}}} \psi \rangle_{{{H}}}",
        slots={"G": S(_SIMPLE), "H": X(_SIMPLE, ["G"])},
    ),
]

_W = compute_weights(_REPR_TEMPLATES)

_representation_theory = make_dispatcher(_REPR_TEMPLATES, _W)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "representation_theory": _representation_theory,
}

WEIGHTS: dict[str, float] = {
    "representation_theory": 0.01,
}

TEMPLATES: dict = {
    "representation_theory": _REPR_TEMPLATES,
}
