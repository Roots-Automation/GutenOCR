"""Ring, field, module, and Galois theory domain generator."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import S, Template, X, compute_weights, make_dispatcher

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_HOMOS = [r"\phi", r"\varphi", r"\psi", "f", r"\theta", r"\rho", r"\pi"]
_IDEALS = ["I", "J", r"\mathfrak{m}", r"\mathfrak{p}", r"\mathfrak{a}", r"\mathfrak{b}"]
_RINGS = ["R", "S", "A", r"\mathbb{Z}", r"\mathbb{F}_p", r"\mathbb{Q}[x]", r"\mathbb{Z}[x]"]
_KFIELDS = ["K", "L", "E", r"\mathbb{Q}(\sqrt{d})", r"\mathbb{F}_{p^n}"]
_BFIELDS = ["F", "k", r"\mathbb{Q}", r"\mathbb{F}_p"]
_MODS = ["M", "N", "V", "W", r"\mathcal{M}", r"\mathcal{N}", "P", "Q"]
_ELEMS = ["g", "h", "a", "b", "x", "y", r"\alpha", r"\beta"]
_SIMPLE = ["G", "H", "K", "N", "A", "B"]

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_RING_TEMPLATES: list[Template] = [
    # c == 0: ideals and quotient rings
    Template(
        name="ideal_quotient",
        latex=r"{I} \trianglelefteq {R},\quad {R}/{I}",
        slots={"I": S(_IDEALS), "R": S(_RINGS)},
    ),
    Template(
        name="maximal_ideal_field",
        latex=r"{R}/{I} \text{{ field}} \iff {I} \text{{ maximal in }} {R}",
        slots={"R": S(_RINGS), "I": S(_IDEALS)},
    ),
    Template(
        name="prime_ideal_domain",
        latex=r"{R}/{I} \text{{ domain}} \iff {I} \text{{ prime in }} {R}",
        slots={"R": S(_RINGS), "I": S(_IDEALS)},
    ),
    # c == 1: CRT
    Template(
        name="chinese_remainder",
        latex=r"{I} + {J} = {R} \implies {R}/({I} \cap {J}) \cong {R}/{I} \times {R}/{J}",
        slots={"I": S(_IDEALS), "J": X(_IDEALS, ["I"]), "R": S(_RINGS)},
    ),
    Template(
        name="crt_embedding",
        latex=r"{R}/({I} \cap {J}) \hookrightarrow {R}/{I} \times {R}/{J}",
        slots={"R": S(_RINGS), "I": S(_IDEALS), "J": X(_IDEALS, ["I"])},
    ),
    # c == 2: ring homomorphism
    Template(
        name="ring_homomorphism",
        latex=r"{phi} : {R} \to {S},\quad {phi}(ab) = {phi}(a){phi}(b),\quad {phi}(1) = 1",
        slots={"phi": S(_HOMOS), "R": S(_RINGS), "S": X(_RINGS, ["R"])},
    ),
    Template(
        name="first_isomorphism_ring",
        latex=r"{R}/\ker {phi} \cong \operatorname{{im}}\, {phi} \subseteq {S}",
        slots={"phi": S(_HOMOS), "R": S(_RINGS), "S": X(_RINGS, ["R"])},
    ),
    # c == 3: field extensions
    Template(
        name="tower_law",
        latex=r"[{K}:{F}] = [{K}:{L}][{L}:{F}]",
        slots={"K": S(_KFIELDS), "L": X(_KFIELDS, ["K"]), "F": S(_BFIELDS)},
    ),
    Template(
        name="extension_dimension",
        latex=r"[{K}:{F}] = \dim_{{{F}}} {K}",
        slots={"K": S(_KFIELDS), "F": S(_BFIELDS)},
    ),
    Template(
        name="simple_extension_degree",
        latex=r"{K} = {F}({g_el}),\quad [{K}:{F}] = \deg \min_{{{F}}}({g_el})",
        slots={"K": S(_KFIELDS), "F": S(_BFIELDS), "g_el": S(_ELEMS)},
    ),
    # c == 4: Galois group
    Template(
        name="galois_group_order",
        latex=r"|\operatorname{{Gal}}({K}/{F})| = [{K}:{F}]",
        slots={"K": S(_KFIELDS), "F": S(_BFIELDS)},
    ),
    Template(
        name="galois_group_iso",
        latex=r"\operatorname{{Gal}}({K}/{F}) \cong {H}",
        slots={"K": S(_KFIELDS), "F": S(_BFIELDS), "H": S(_SIMPLE)},
    ),
    Template(
        name="fixed_field",
        latex=r"{K}^{{\operatorname{{Gal}}({K}/{F})}} = {F}",
        slots={"K": S(_KFIELDS), "F": S(_BFIELDS)},
    ),
    # c == 5: Galois correspondence
    Template(
        name="galois_correspondence",
        latex=r"\left\{{{H} \leq \operatorname{{Gal}}({K}/{F})\right\}} \longleftrightarrow \left\{{{L} : {F} \subseteq {L} \subseteq {K}\right\}}",
        slots={"K": S(_KFIELDS), "L": X(_KFIELDS, ["K"]), "F": S(_BFIELDS), "H": S(_SIMPLE)},
    ),
    # c == 6: tensor product of modules
    Template(
        name="tensor_product",
        latex=r"{M} \otimes_{{{R}}} {N}",
        slots={"M": S(_MODS), "N": X(_MODS, ["M"]), "R": S(_RINGS)},
    ),
    Template(
        name="tensor_associativity",
        latex=r"({M} \otimes_{{{R}}} {N}) \otimes_{{{R}}} {M} \cong {M} \otimes_{{{R}}} ({N} \otimes_{{{R}}} {M})",
        slots={"M": S(_MODS), "N": X(_MODS, ["M"]), "R": S(_RINGS)},
    ),
    Template(
        name="tensor_unit",
        latex=r"{R} \otimes_{{{R}}} {M} \cong {M}",
        slots={"R": S(_RINGS), "M": S(_MODS)},
    ),
    # c == 7: Hom of modules
    Template(
        name="hom_modules",
        latex=r"\operatorname{{Hom}}_{{{R}}}({M}, {N})",
        slots={"R": S(_RINGS), "M": S(_MODS), "N": X(_MODS, ["M"])},
    ),
    Template(
        name="hom_direct_sum",
        latex=r"\operatorname{{Hom}}_{{{R}}}({M} \oplus {N}, {M}) \cong \operatorname{{Hom}}_{{{R}}}({M},{M}) \oplus \operatorname{{Hom}}_{{{R}}}({N},{M})",
        slots={"R": S(_RINGS), "M": S(_MODS), "N": X(_MODS, ["M"])},
    ),
    Template(
        name="hom_exact_sequence",
        latex=r"0 \to \operatorname{{Hom}}_{{{R}}}({M},{M}) \to \operatorname{{Hom}}_{{{R}}}({N},{M}) \to \operatorname{{Hom}}_{{{R}}}({N},{N})",
        slots={"R": S(_RINGS), "M": S(_MODS), "N": X(_MODS, ["M"])},
    ),
]

_W = compute_weights(_RING_TEMPLATES)

_ring_field_theory = make_dispatcher(_RING_TEMPLATES, _W)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "ring_field_theory": _ring_field_theory,
}

WEIGHTS: dict[str, float] = {
    "ring_field_theory": 0.01,
}

TEMPLATES: dict = {
    "ring_field_theory": _RING_TEMPLATES,
}
