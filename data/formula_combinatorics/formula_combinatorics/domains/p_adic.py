"""p-adic numbers domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import E, S, Template, X, compute_weights, make_dispatcher
from .._vocab import _fn_rich_nosub

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_PRIME_POOL: tuple[str, ...] = (
    "p",
    "q",
    r"\ell",
    r"\ell'",
    "r",
    "2",
    "3",
    "5",
    "7",
    r"\ell_0",
)  # 10

_ELEM_POOL: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "a",
    "b",
    "c",
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\xi",
    r"\eta",
)  # 11

_FIELD_POOL: tuple[str, ...] = (
    "K",
    "L",
    "F",
    "E",
    r"\mathbf{k}",
    r"\mathcal{K}",
    r"\mathcal{L}",
    r"\hat{K}",
    r"\hat{F}",
)  # 9

_RING_POOL: tuple[str, ...] = (
    r"\mathcal{O}",
    r"\mathcal{O}_K",
    r"\mathcal{O}_L",
    r"\mathcal{O}_F",
    "A",
    "B",
    "R",
    r"\mathcal{A}",
)  # 8

_IDEAL_POOL: tuple[str, ...] = (
    r"\mathfrak{m}",
    r"\mathfrak{p}",
    r"\mathfrak{q}",
    r"\mathfrak{a}",
    r"\mathfrak{b}",
    "I",
    "P",
)  # 7

_UNIF_POOL: tuple[str, ...] = (
    r"\pi",
    r"\varpi",
    r"\Pi",
    "u",
    "t",
    r"\pi_0",
    r"\varpi_0",
)  # 7
_UNIF_BASE_POOL: tuple[str, ...] = tuple(v for v in _UNIF_POOL if "_" not in v)  # 5

_POLY_POOL: tuple[str, ...] = (
    "f",
    "g",
    "h",
    "F",
    "G",
    r"\phi",
    r"\psi",
    r"\chi",
)  # 8

_IDX_POOL: tuple[str, ...] = (
    "n",
    "m",
    "k",
    "r",
    "s",
    "N",
    "M",
)  # 7

# ---------------------------------------------------------------------------
# Part A: 14 reparameterized originals
# ---------------------------------------------------------------------------

_TEMPLATES_A: list[Template] = [
    Template(
        name="padic_valuation",
        latex=r"|{xx}|_{{{pp}}} = {pp}^{{-v_{{{pp}}}({xx})}}",
        slots={"xx": S(_ELEM_POOL), "pp": S(_PRIME_POOL)},
    ),
    Template(
        name="padic_integers",
        latex=(
            r"\mathbb{{Z}}_{{{pp}}} = \left\{{{xx} \in \mathbb{{Q}}_{{{pp}}}"
            r" : |{xx}|_{{{pp}}} \leq 1\right\}}"
        ),
        slots={"xx": S(_ELEM_POOL), "pp": S(_PRIME_POOL)},
    ),
    Template(
        name="padic_expansion",
        latex=r"{xx} = \sum{lim_mod}_{{k=0}}^{{\infty}} a_k {pp}^k,\quad 0 \leq a_k < {pp}",
        slots={"lim_mod": S(("", r"\limits")), "xx": S(_ELEM_POOL), "pp": S(_PRIME_POOL)},
    ),
    Template(
        name="ultrametric_inequality",
        latex=(
            r"|{xx} + {yy}|_{{{pp}}}"
            r" \leq \max\!\left(|{xx}|_{{{pp}}}, |{yy}|_{{{pp}}}\right)"
        ),
        slots={
            "xx": S(_ELEM_POOL),
            "yy": X(_ELEM_POOL, ("xx",)),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="padic_multiplicativity",
        latex=r"|{xx} {yy}|_{{{pp}}} = |{xx}|_{{{pp}}} |{yy}|_{{{pp}}}",
        slots={
            "xx": S(_ELEM_POOL),
            "yy": X(_ELEM_POOL, ("xx",)),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="valuation_subadditivity",
        latex=(
            r"v_{{{pp}}}({xx} + {yy})"
            r" \geq \min\!\left(v_{{{pp}}}({xx}),\, v_{{{pp}}}({yy})\right)"
        ),
        slots={
            "xx": S(_ELEM_POOL),
            "yy": X(_ELEM_POOL, ("xx",)),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="padic_norm_prime",
        latex=r"|{pp}|_{{{pp}}} = {pp}^{{-1}}",
        slots={"pp": S(_PRIME_POOL)},
    ),
    Template(
        name="product_formula",
        latex=r"|{xx}|_\infty \cdot \prod{lim_mod}_{{{pp}}} |{xx}|_{{{pp}}} = 1",
        slots={"lim_mod": S(("", r"\limits")), "xx": S(_ELEM_POOL), "pp": S(_PRIME_POOL)},
    ),
    Template(
        name="hensels_lemma",
        latex=(
            r"{ff}({aa}) \equiv 0 \pmod{{{pp}}},\;"
            r" {ff}'({aa}) \not\equiv 0 \pmod{{{pp}}}"
            r" \implies \exists!\, {bb} \in \mathbb{{Z}}_{{{pp}}} : {ff}({bb}) = 0"
        ),
        slots={
            "ff": S(_POLY_POOL),
            "aa": S(_ELEM_POOL),
            "bb": X(_ELEM_POOL, ("aa",)),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="legendre_factorial_valuation",
        latex=(
            r"v_{{{pp}}}({nn}!) ="
            r" \sum{lim_mod}_{{k=1}}^{{\infty}} \left\lfloor \frac{{{nn}}}{{{pp}^k}} \right\rfloor"
        ),
        slots={"lim_mod": S(("", r"\limits")), "pp": S(_PRIME_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="qp_completion",
        latex=r"\mathbb{{Q}}_{{{pp}}} = \widehat{{\mathbb{{Q}}}}_{{|\cdot|_{{{pp}}}}}",
        slots={"pp": S(_PRIME_POOL)},
    ),
    Template(
        name="padic_unique_absolute_value",
        latex=(
            r"\|\cdot\|_{{{pp}}} \text{{ is the unique }}"
            r" {pp}\text{{-adic absolute value on }} \mathbb{{Q}}"
        ),
        slots={"pp": S(_PRIME_POOL)},
    ),
    Template(
        name="padic_exponential",
        latex=(
            r"\exp_{{{pp}}}({xx}) = \sum{lim_mod}_{{k=0}}^{{\infty}} \frac{{{xx}^k}}{{k!}},\quad"
            r" |{xx}|_{{{pp}}} < {pp}^{{-1/({pp}-1)}}"
        ),
        slots={"lim_mod": S(("", r"\limits")), "pp": S(_PRIME_POOL), "xx": S(_ELEM_POOL)},
    ),
    Template(
        name="padic_binomial_valuation",
        latex=r"v_{{{pp}}}\!\left(\binom{{{pp}^{{{nn}}}}}{{{pp}^{{{jj}}}}}\right) = {nn} - {jj}",
        slots={
            "pp": S(_PRIME_POOL),
            "nn": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("nn",)),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B1: Discrete valuation theory (8)
# ---------------------------------------------------------------------------

_TEMPLATES_B1: list[Template] = [
    Template(
        name="valuation_product",
        latex=r"v_{{{pp}}}({xx} \cdot {yy}) = v_{{{pp}}}({xx}) + v_{{{pp}}}({yy})",
        slots={
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
            "yy": X(_ELEM_POOL, ("xx",)),
        },
    ),
    Template(
        name="valuation_min_property",
        latex=(
            r"v_{{{pp}}}({xx} + {yy})"
            r" \geq \min\bigl(v_{{{pp}}}({xx}),\, v_{{{pp}}}({yy})\bigr)"
        ),
        slots={
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
            "yy": X(_ELEM_POOL, ("xx",)),
        },
    ),
    Template(
        name="valuation_ring_def",
        latex=(r"{RR} = \{{ {xx} \in {KK} : v_{{{pp}}}({xx}) \geq 0 \}}"),
        slots={
            "pp": S(_PRIME_POOL),
            "KK": S(_FIELD_POOL),
            "xx": S(_ELEM_POOL),
            "RR": S(_RING_POOL),
        },
    ),
    Template(
        name="maximal_ideal_def",
        latex=(r"{II} = \{{ {xx} \in {RR} : v_{{{pp}}}({xx}) \geq 1 \}}"),
        slots={
            "RR": S(_RING_POOL),
            "II": S(_IDEAL_POOL),
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
        },
    ),
    Template(
        name="residue_field_iso",
        latex=r"{RR} / {II} \cong \mathbb{{F}}_{{{pp}^f}}",
        slots={
            "RR": S(_RING_POOL),
            "II": S(_IDEAL_POOL),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="uniformizer_generates",
        latex=r"{II} = {uu} \cdot {RR}",
        slots={
            "uu": S(_UNIF_POOL),
            "RR": S(_RING_POOL),
            "II": S(_IDEAL_POOL),
        },
    ),
    Template(
        name="valuation_zero_unit",
        latex=r"v_{{{pp}}}({xx}) = 0 \iff {xx} \in {RR}^*",
        slots={
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
            "RR": S(_RING_POOL),
        },
    ),
    Template(
        name="discrete_valuation_surjective",
        latex=r"v_{{{pp}}}({KK}^*) = \mathbb{{Z}},\quad {xx} \in {KK}^*",
        slots={
            "KK": S(_FIELD_POOL),
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B2: Completions & expansions (8)
# ---------------------------------------------------------------------------

_TEMPLATES_B2: list[Template] = [
    Template(
        name="padic_fraction_form",
        latex=(
            r"{xx} = {pp}^{{{nn}}} \cdot {yy},\quad"
            r" v_{{{pp}}}({xx}) = {nn},\quad {yy} \in \mathbb{{Z}}_{{{pp}}}^*"
        ),
        slots={
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
            "yy": X(_ELEM_POOL, ("xx",)),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="padic_log_series",
        latex=(
            r"\log_{{{pp}}}(1 + {xx})"
            r" = \sum{lim_mod}_{{{nn}=1}}^{{\infty}} \frac{{(-1)^{{{nn}+1}} {xx}^{{{nn}}}}}{{{nn}}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="padic_exp_series",
        latex=(
            r"\exp_{{{pp}}}({xx})"
            r" = \sum{lim_mod}_{{{nn}=0}}^{{\infty}} \frac{{{xx}^{{{nn}}}}}{{{nn}!}},\quad"
            r" |{xx}|_{{{pp}}} < {pp}^{{-1/({pp}-1)}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="padic_power_series_conv",
        latex=(
            r"\sum{lim_mod}_{{{nn}=0}}^{{\infty}} a_{{{nn}}} {xx}^{{{nn}}}"
            r" \text{{ converges in }} \mathbb{{Q}}_{{{pp}}}"
            r" \iff a_{{{nn}}} \to 0"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "pp": S(_PRIME_POOL),
            "ff": S(_POLY_POOL),
            "xx": S(_ELEM_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="teichmuller_decomp",
        latex=(
            r"{xx} = \sum{lim_mod}_{{{nn}=0}}^{{\infty}} [{xx}_{{{nn}}}]\, {pp}^{{{nn}}},"
            r"\quad [{xx}_{{{nn}}}] \text{{ Teichm\"{{u}}ller lift}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="padic_complete_metric",
        latex=(
            r"\mathbb{{Q}}_{{{pp}}} \text{{ is complete w.r.t. }}"
            r" |\cdot|_{{{pp}}},\quad {xx} \in \mathbb{{Q}}_{{{pp}}}"
        ),
        slots={"pp": S(_PRIME_POOL), "xx": S(_ELEM_POOL)},
    ),
    Template(
        name="padic_series_cauchy",
        latex=(
            r"\{{a_{{{nn}}}\}} \text{{ Cauchy in }} \mathbb{{Q}}_{{{pp}}}"
            r" \iff |a_{{{nn}+1}} - a_{{{nn}}}|_{{{pp}}} \to 0"
        ),
        slots={
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="padic_radius_convergence",
        latex=(
            r"\sum a_{{{nn}}} {xx}^{{{nn}}} \text{{ conv. on }}"
            r" |{xx}|_{{{pp}}} < r,\quad"
            r" r = \liminf_{{{{{nn}}}\to\infty}} |a_{{{nn}}}|_{{{pp}}}^{{-1/{nn}}}"
        ),
        slots={
            "pp": S(_PRIME_POOL),
            "ff": S(_POLY_POOL),
            "xx": S(_ELEM_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B3: Local fields (7)
# ---------------------------------------------------------------------------

_TEMPLATES_B3: list[Template] = [
    Template(
        name="ramification_degree",
        latex=(
            r"e({KK}/\mathbb{{Q}}_{{{pp}}})"
            r" \cdot f({KK}/\mathbb{{Q}}_{{{pp}}})"
            r" = [{KK}:\mathbb{{Q}}_{{{pp}}}]"
        ),
        slots={
            "KK": S(_FIELD_POOL),
            "pp": S(_PRIME_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="ramification_composition",
        latex=(
            r"e({LL}/{KK}) = e({LL}/{FF}) \cdot e({FF}/{KK})"
            r" \text{{ for }} {LL} \supset {FF} \supset {KK}"
        ),
        slots={
            "LL": S(_FIELD_POOL),
            "FF": X(_FIELD_POOL, ("LL",)),
            "KK": X(_FIELD_POOL, ("LL", "FF")),
        },
    ),
    Template(
        name="residue_field_tower",
        latex=r"[k_{{{LL}}} : k_{{{KK}}}] = f({LL}/{KK})",
        slots={
            "LL": S(_FIELD_POOL),
            "KK": X(_FIELD_POOL, ("LL",)),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="totally_ramified_iff",
        latex=(
            r"{KK}/\mathbb{{Q}}_{{{pp}}} \text{{ tot. ramified}}"
            r" \iff f({KK}/\mathbb{{Q}}_{{{pp}}}) = 1"
            r" \iff \text{{ min poly of }} {uu} \text{{ is Eisenstein}}"
        ),
        slots={
            "KK": S(_FIELD_POOL),
            "uu": S(_UNIF_POOL),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="unramified_extension_unique",
        latex=(
            r"\exists!\text{{ unramified ext. of }} {KK}"
            r" \text{{ of degree }} {nn} \text{{ over }} \mathbb{{Q}}_{{{pp}}}"
        ),
        slots={
            "KK": S(_FIELD_POOL),
            "pp": S(_PRIME_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="local_field_finite_ext",
        latex=(
            r"[{KK}:\mathbb{{Q}}_{{{pp}}}] = {nn} < \infty,"
            r"\quad e \cdot f = {nn}"
        ),
        slots={
            "KK": S(_FIELD_POOL),
            "pp": S(_PRIME_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="norm_map_local",
        latex=(
            r"N_{{{LL}/{KK}}}: {LL}^* \to {KK}^*,\quad "
            r"N_{{{LL}/{KK}}}({uu}_{{{LL}}})^{{f({LL}/{KK})}} = (-1)^{{e}} a_0"
        ),
        slots={
            "LL": S(_FIELD_POOL),
            "KK": X(_FIELD_POOL, ("LL",)),
            "pp": S(_PRIME_POOL),
            "uu": S(_UNIF_BASE_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B4: Hensel's lemma variants (7)
# ---------------------------------------------------------------------------

_TEMPLATES_B4: list[Template] = [
    Template(
        name="hensel_strong",
        latex=(
            r"|{ff}({aa})|_{{{pp}}} < |{ff}'({aa})|_{{{pp}}}^2"
            r" \Rightarrow \exists!\, {bb} \in \mathbb{{Q}}_{{{pp}}}"
            r" : {ff}({bb}) = 0,\; |{bb} - {aa}|_{{{pp}}} < |{ff}'({aa})|_{{{pp}}}"
        ),
        slots={
            "ff": S(_POLY_POOL),
            "aa": S(_ELEM_POOL),
            "bb": X(_ELEM_POOL, ("aa",)),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="hensel_factorization",
        latex=(
            r"{ff} \equiv {gg} \cdot {hh} \pmod{{{pp}}},\;"
            r" \gcd(\bar{{{gg}}}, \bar{{{hh}}}) = 1"
            r" \Rightarrow {ff} = \tilde{{{gg}}} \cdot \tilde{{{hh}}}"
            r" \text{{ over }} \mathbb{{Z}}_{{{pp}}}"
        ),
        slots={
            "ff": S(_POLY_POOL),
            "gg": X(_POLY_POOL, ("ff",)),
            "hh": X(_POLY_POOL, ("ff", "gg")),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="hensel_system",
        latex=(
            r"\det J_{{{ff}}}({aa}) \not\equiv 0 \pmod{{{pp}}}"
            r" \Rightarrow \exists\, {bb} \in \mathbb{{Z}}_{{{pp}}}^{{{nn}}}"
            r" : {ff}({bb}) = 0"
        ),
        slots={
            "ff": S(_POLY_POOL),
            "aa": S(_ELEM_POOL),
            "bb": X(_ELEM_POOL, ("aa",)),
            "pp": S(_PRIME_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="hensel_newton_polygon",
        latex=(
            r"\text{{Newton polygon of }} {ff}"
            r" \text{{ has horiz. segment of length }} {nn}"
            r" \Rightarrow {ff} \text{{ has }} {nn} \text{{ roots in }} \mathbb{{Q}}_{{{pp}}}"
        ),
        slots={
            "ff": S(_POLY_POOL),
            "pp": S(_PRIME_POOL),
            "nn": S(_IDX_POOL),
            "mm": X(_IDX_POOL, ("nn",)),
        },
    ),
    Template(
        name="hensel_iterative",
        latex=(
            r"{aa}_{{{nn}+1}} = {aa}_{{{nn}}}"
            r" - \frac{{{ff}({aa}_{{{nn}}})}}{{ {ff}'({aa}_{{{nn}}})}}"
            r" \xrightarrow{{{pp}\text{{-adic}}}} \text{{root of }} {ff}"
        ),
        slots={
            "ff": S(_POLY_POOL),
            "aa": S(_ELEM_POOL),
            "pp": S(_PRIME_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="hensel_criterion_mod",
        latex=(
            r"{ff}({aa}) \equiv 0 \pmod{{{pp}^{{{nn}}}}},\;"
            r" v_{{{pp}}}({ff}'({aa})) = 0"
            r" \Rightarrow \exists\, {bb} : {ff}({bb}) \equiv 0 \pmod{{{pp}^{{2{nn}}}}}"
        ),
        slots={
            "ff": S(_POLY_POOL),
            "aa": S(_ELEM_POOL),
            "bb": X(_ELEM_POOL, ("aa",)),
            "pp": S(_PRIME_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="hensel_general_ring",
        latex=(
            r"{ff}({aa}) \in {pp} \cdot {RR},\;"
            r" {ff}'({aa}) \notin {pp} \cdot {RR}"
            r" \Rightarrow \exists!\, {bb} \in {RR} : {ff}({bb}) = 0"
        ),
        slots={
            "ff": S(_POLY_POOL),
            "RR": S(_RING_POOL),
            "aa": S(_ELEM_POOL),
            "bb": X(_ELEM_POOL, ("aa",)),
            "pp": S(_PRIME_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B5: p-adic analysis (7)
# ---------------------------------------------------------------------------

_TEMPLATES_B5: list[Template] = [
    Template(
        name="strassmann_zeros",
        latex=(
            r"{ff} \in \mathbb{{Z}}_{{{pp}}}[[{xx}]]"
            r" \text{{ with last non-zero coeff in degree }} {nn}"
            r" \Rightarrow {ff} \text{{ has }} \leq {nn} \text{{ zeros in }}"
            r" \mathbb{{Z}}_{{{pp}}}"
        ),
        slots={
            "ff": S(_POLY_POOL),
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="weierstrass_preparation",
        latex=(
            r"{ff} = {pp}^{{{mm}}} \cdot U({xx}) \cdot W({xx}),"
            r"\quad \deg W = {nn},\quad U \in \mathbb{{Z}}_{{{pp}}}[[{xx}]]^*"
        ),
        slots={
            "ff": S(_POLY_POOL),
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
            "nn": S(_IDX_POOL),
            "mm": X(_IDX_POOL, ("nn",)),
        },
    ),
    Template(
        name="mahler_expansion",
        latex=(
            r"{ff}({xx}) = \sum{lim_mod}_{{{nn}=0}}^{{\infty}}"
            r" c_{{{nn}}} \binom{{{xx}}}{{{nn}}},\quad"
            r" c_{{{nn}}} = \Delta^{{{nn}}} {ff}(0)"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "ff": S(_POLY_POOL),
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="padic_newton_iterate",
        latex=(
            r"|{ff}'({aa})|_{{{pp}}} > |{ff}({aa})|_{{{pp}}}"
            r" \Rightarrow \text{{Newton iterates starting at }} {aa}"
            r" \text{{ converge to a root of }} {ff}"
        ),
        slots={
            "ff": S(_POLY_POOL),
            "aa": S(_ELEM_POOL),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="locally_analytic",
        latex=(
            r"{ff} : {KK} \to {KK} \text{{ locally analytic}},"
            r"\quad \forall {xx} \in {KK}:\;"
            r" {ff}({xx}) = \sum{lim_mod}_{{n \geq 0}} a_n ({xx} - c)^n"
            r" \text{{ on some ball}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "ff": S(_POLY_POOL),
            "KK": S(_FIELD_POOL),
            "xx": S(_ELEM_POOL),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="overconvergent_series",
        latex=(
            r"\sum a_{{{nn}}} {xx}^{{{nn}}}"
            r" \text{{ overconvergent: conv. on }}"
            r" |{xx}|_{{{pp}}} < r \text{{ for some }} r > 1"
        ),
        slots={
            "ff": S(_POLY_POOL),
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="padic_integration",
        latex=(
            r"\int{lim_mod}_{{\mathbb{{Z}}_{{{pp}}}}} {ff}({xx})\, d\mu"
            r" = \lim_{{{nn}\to\infty}} \sum{lim_mod}_{{a=0}}^{{{pp}^{{{nn}}}-1}}"
            r" {ff}(a)\, \mu\!\left(a + {pp}^{{{nn}}} \mathbb{{Z}}_{{{pp}}}\right)"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "ff": S(_POLY_POOL),
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B6: Number theory applications (7)
# ---------------------------------------------------------------------------

_TEMPLATES_B6: list[Template] = [
    Template(
        name="kummer_carries",
        latex=(
            r"v_{{{pp}}}\!\binom{{{nn}+{mm}}}{{{nn}}}"
            r" = \text{{number of carries in base-}}{pp}"
            r" \text{{ addition of }} {nn} \text{{ and }} {mm}"
        ),
        slots={
            "pp": S(_PRIME_POOL),
            "nn": S(_IDX_POOL),
            "mm": X(_IDX_POOL, ("nn",)),
        },
    ),
    Template(
        name="product_formula_global",
        latex=r"\prod_v |{xx}|_v = 1 \quad \forall\, {xx} \in {KK}^*",
        slots={"xx": S(_ELEM_POOL), "KK": S(_FIELD_POOL)},
    ),
    Template(
        name="teichmuller_character",
        latex=(
            r"\omega : (\mathbb{{Z}}/{pp}\mathbb{{Z}})^*"
            r" \to \mu_{{{pp}-1}} \subset \mathbb{{Z}}_{{{pp}}}^*,\quad"
            r" \omega({xx}) \equiv {xx} \pmod{{{pp}}}"
        ),
        slots={"pp": S(_PRIME_POOL), "xx": S(_ELEM_POOL)},
    ),
    Template(
        name="iwasawa_algebra",
        latex=(
            r"\Lambda_{{{KK}}} = \mathbb{{Z}}_{{{pp}}}[[\Gamma]]"
            r" \cong \mathbb{{Z}}_{{{pp}}}[[T]],\quad \Gamma \cong \mathbb{{Z}}_{{{pp}}}"
        ),
        slots={"pp": S(_PRIME_POOL), "KK": S(_FIELD_POOL)},
    ),
    Template(
        name="norm_residue_symbol",
        latex=(
            r"\left({aa},\, {KK}/{FF}\right)_{{{pp}}} = \mathrm{{Frob}}"
            r" \text{{ when }} {aa} \in N_{{{KK}/{FF}}}({KK}^*)"
        ),
        slots={
            "aa": S(_ELEM_POOL),
            "KK": S(_FIELD_POOL),
            "FF": X(_FIELD_POOL, ("KK",)),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="lubin_tate_formal",
        latex=(
            r"\mathcal{{F}}_{{{uu}}}({xx}, {yy})"
            r" \text{{ Lubin-Tate formal {KK}-module for uniformizer }} {uu},"
            r" \quad [{pp}]_{{F}}({xx}) = {pp} {xx} + {xx}^{{{pp}}}"
        ),
        slots={
            "uu": S(_UNIF_POOL),
            "pp": S(_PRIME_POOL),
            "KK": S(_FIELD_POOL),
            "xx": S(_ELEM_POOL),
            "yy": X(_ELEM_POOL, ("xx",)),
        },
    ),
    Template(
        name="witt_vectors",
        latex=(
            r"W({RR}) = \text{{ring of Witt vectors over }} {RR}"
            r" \text{{ w.r.t. prime }} {pp},"
            r"\quad W_{{n}}({RR}) = {RR}^{{{nn}}}"
        ),
        slots={
            "RR": S(_RING_POOL),
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part C: High-n_eff function-pair templates (6)
# ---------------------------------------------------------------------------

_TEMPLATES_C: list[Template] = [
    Template(
        name="fn_padic_norm",
        latex=r"{fn1}(|{xx}|_{{{pp}}}) = {fn2}({xx})",
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
        },
    ),
    Template(
        name="fn_valuation_pair",
        latex=(
            r"{fn1}\!\left(v_{{{pp}}}({xx}) + v_{{{pp}}}({yy})\right)"
            r" = {fn2}\!\left(v_{{{pp}}}({xx} \cdot {yy})\right)"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
            "yy": X(_ELEM_POOL, ("xx",)),
        },
    ),
    Template(
        name="fn_padic_exp",
        latex=(
            r"{fn1}\!\left(\exp_{{{pp}}}({xx})\right) = {fn2}({xx})"
            r" \quad \bigl(|{xx}|_{{{pp}}} < {pp}^{{-1/({pp}-1)}}\bigr)"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
        },
    ),
    Template(
        name="fn_local_norm",
        latex=(
            r"{fn1}\!\left(N_{{{KK}/{FF}}}({xx})\right) = {fn2}({xx})"
            r" \quad ({xx} \in {KK})"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "KK": S(_FIELD_POOL),
            "FF": X(_FIELD_POOL, ("KK",)),
            "xx": S(_ELEM_POOL),
        },
    ),
    Template(
        name="fn_triple_padic",
        latex=(
            r"{fn1}({xx} \cdot {yy})"
            r" = {fn2}({xx}) \cdot {fn3}({yy})"
            r" \quad \text{{in }} \mathbb{{Q}}_{{{pp}}}"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "fn3": E(_fn_rich_nosub, n=100),
            "pp": S(_PRIME_POOL),
            "xx": S(_ELEM_POOL),
            "yy": X(_ELEM_POOL, ("xx",)),
        },
    ),
    Template(
        name="fn_frobenius_local",
        latex=(
            r"{fn1}(\mathrm{{Frob}}_{{{KK}}}({xx})) = {fn2}({xx})"
            r" \quad ({xx} \in {KK})"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "KK": S(_FIELD_POOL),
            "xx": S(_ELEM_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_PADIC_TEMPLATES: list[Template] = (
    _TEMPLATES_A
    + _TEMPLATES_B1
    + _TEMPLATES_B2
    + _TEMPLATES_B3
    + _TEMPLATES_B4
    + _TEMPLATES_B5
    + _TEMPLATES_B6
    + _TEMPLATES_C
)

_W: list[float] = compute_weights(_PADIC_TEMPLATES)
_p_adic = make_dispatcher(_PADIC_TEMPLATES, _W)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "p_adic": _p_adic,
}

WEIGHTS: dict[str, float] = {
    "p_adic": 0.02,
}

TEMPLATES: dict[str, list[Template]] = {
    "p_adic": _PADIC_TEMPLATES,
}
