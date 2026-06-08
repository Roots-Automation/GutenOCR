"""Number theory domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import E, S, Template, X, compute_weights, make_dispatcher
from .._vocab import _fn_rich_nosub

# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------

_IDX_POOL: tuple[str, ...] = ("n", "m", "k", "j", "r", "l", "s", "t", "i", "p", "N", "M")
_INT_POOL: tuple[str, ...] = ("a", "b", "c", "d", "u", "v", "x", "y")
_PRIME_POOL: tuple[str, ...] = ("p", "q", "r", "s", "l", "t", r"p_1", r"p_2", r"q_1", r"q_2", r"\ell", "u")
# Restricted pool for templates that append their own subscripts to {pp} — using
# pre-subscripted values like p_2 would produce invalid double-subscript LaTeX (p_2_i).
_PRIME_BASE_POOL: tuple[str, ...] = tuple(v for v in _PRIME_POOL if "_" not in v)
_MOD_POOL: tuple[str, ...] = ("m", "n", "p", "q", "N", "M", "k", "r", "P", "Q")
_FUNC_POOL: tuple[str, ...] = ("f", "g", "h", r"\phi", r"\psi", r"\chi", r"\varphi", r"\xi", r"\eta", r"\zeta")
_ALPHA_POOL: tuple[str, ...] = (
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\delta",
    r"\theta",
    r"\lambda",
    r"\mu",
    r"\nu",
)
_RING_POOL: tuple[str, ...] = (
    r"\mathbb{Z}",
    r"\mathbb{Z}[i]",
    r"\mathbb{Z}[\omega]",
    r"\mathcal{O}_K",
    r"\mathcal{O}_F",
)
_FIELD_POOL: tuple[str, ...] = (
    r"\mathbb{Q}",
    r"\mathbb{F}_p",
    r"\mathbb{F}_q",
    r"\mathbb{Q}(\sqrt{d})",
    r"\mathbb{Q}(\zeta_n)",
)

_CFRAC_ALIGN: tuple[str, ...] = ("l", "r")

# ---------------------------------------------------------------------------
# Part A: Reparameterized originals (20 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_A: list[Template] = [
    Template(
        name="binomial_coefficient",
        latex=r"\binom{{{nn}}}{{{kk}}}",
        slots={
            "nn": S(_IDX_POOL),
            "kk": X(_IDX_POOL, ("nn",)),
        },
    ),
    Template(
        name="binomial_coefficient_def",
        latex=r"\binom{{{nn}}}{{{kk}}} = \frac{{{nn}!}}{{{kk}!\,({nn}-{kk})!}}",
        slots={
            "nn": S(_IDX_POOL),
            "kk": X(_IDX_POOL, ("nn",)),
        },
    ),
    Template(
        name="modular_congruence",
        latex=r"{aa} \equiv {bb} \pmod{{{mm}}}",
        slots={
            "aa": S(_INT_POOL),
            "bb": X(_INT_POOL, ("aa",)),
            "mm": S(_MOD_POOL),
        },
    ),
    Template(
        name="gcd_lcm_identity",
        latex=(
            r"\gcd({aa}, {bb}) \cdot \operatorname{{lcm}}({aa}, {bb})"
            r" = {aa} \cdot {bb}"
        ),
        slots={
            "aa": S(_INT_POOL),
            "bb": X(_INT_POOL, ("aa",)),
        },
    ),
    Template(
        name="sum_first_n",
        latex=(
            r"\sum{lim_mod}_{{{kk}=1}}^{{{nn}}} {kk}"
            r" = \frac{{{nn}({nn}+1)}}{{2}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "nn": S(_IDX_POOL),
            "kk": X(_IDX_POOL, ("nn",)),
        },
    ),
    Template(
        name="sum_of_squares",
        latex=(
            r"\sum{lim_mod}_{{{kk}=1}}^{{{nn}}} {kk}^2"
            r" = \frac{{{nn}({nn}+1)(2{nn}+1)}}{{6}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "nn": S(_IDX_POOL),
            "kk": X(_IDX_POOL, ("nn",)),
        },
    ),
    Template(
        name="fibonacci_recurrence",
        latex=(
            r"F_{{{nn}+2}} = F_{{{nn}+1}} + F_{{{nn}}},"
            r"\quad F_0 = 0,\; F_1 = 1"
        ),
        slots={"nn": S(_IDX_POOL)},
    ),
    Template(
        name="euler_totient",
        latex=(
            r"\phi({nn}) = {nn} \prod{lim_mod}_{{{pp} \mid {nn}}}"
            r" \left(1 - \frac{{1}}{{{pp}}}\right)"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "nn": S(_IDX_POOL),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="fermat_little_theorem",
        latex=r"{aa}^{{{pp}-1}} \equiv 1 \pmod{{{pp}}}",
        slots={
            "aa": S(_INT_POOL),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="catalan_numbers",
        latex=r"C_{{{nn}}} = \frac{{1}}{{{nn}+1}} \binom{{2{nn}}}{{{nn}}}",
        slots={"nn": S(_IDX_POOL)},
    ),
    Template(
        name="geometric_series",
        latex=(
            r"\sum{lim_mod}_{{{kk}=0}}^{{{nn}}} {aa}^{{{kk}}}"
            r" = \frac{{1 - {aa}^{{{nn}+1}}}}{{1 - {aa}}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "nn": S(_IDX_POOL),
            "kk": X(_IDX_POOL, ("nn",)),
            "aa": S(_INT_POOL),
        },
    ),
    Template(
        name="factorial_product",
        latex=r"{nn}! = \prod{lim_mod}_{{{kk}=1}}^{{{nn}}} {kk}",
        slots={
            "lim_mod": S(("", r"\limits")),
            "nn": S(_IDX_POOL),
            "kk": X(_IDX_POOL, ("nn",)),
        },
    ),
    Template(
        name="leibniz_pi",
        latex=(
            r"\sum{lim_mod}_{{{kk}=0}}^{{\infty}} \frac{{(-1)^{{{kk}}}}}{{2{kk}+1}}"
            r" = \frac{{\pi}}{{4}}"
        ),
        slots={"lim_mod": S(("", r"\limits")), "kk": S(_IDX_POOL)},
    ),
    Template(
        name="wilson_theorem",
        latex=r"({pp}-1)! \equiv -1 \pmod{{{pp}}}",
        slots={"pp": S(_PRIME_POOL)},
    ),
    Template(
        name="sum_of_divisors",
        latex=r"\sigma_{{{kk}}}({nn}) = \sum{lim_mod}_{{d \mid {nn}}} d^{{{kk}}}",
        slots={
            "lim_mod": S(("", r"\limits")),
            "nn": S(_IDX_POOL),
            "kk": X(_IDX_POOL, ("nn",)),
        },
    ),
    Template(
        name="number_of_divisors",
        latex=r"\tau({nn}) = \sum{lim_mod}_{{d \mid {nn}}} 1 = \prod{lim_mod}_{{{pp}^k \| {nn}}} (k+1)",
        slots={
            "lim_mod": S(("", r"\limits")),
            "nn": S(_IDX_POOL),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="prime_number_theorem",
        latex=r"\pi({xx}) \sim \frac{{{xx}}}{{\ln {xx}}}",
        slots={"xx": S(_INT_POOL)},
    ),
    Template(
        name="mobius_inversion",
        latex=(
            r"{ff}({nn}) = \sum{lim_mod}_{{d \mid {nn}}} {gg}(d)"
            r" \implies {gg}({nn}) = \sum{lim_mod}_{{d \mid {nn}}} \mu(d)\,{ff}({nn}/d)"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="riemann_zeta_euler_product",
        latex=(
            r"\zeta({ss}) = \sum{lim_mod}_{{{nn}=1}}^{{\infty}}"
            r" \frac{{1}}{{{nn}^{{{ss}}}}}"
            r" = \prod{lim_mod}_{{{pp} \text{{ prime}}}}"
            r" \frac{{1}}{{1-{pp}^{{-{ss}}}}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "ss": S(_ALPHA_POOL),
            "nn": S(_IDX_POOL),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="sum_of_cubes",
        latex=(
            r"\sum{lim_mod}_{{{kk}=1}}^{{{nn}}} {kk}^3"
            r" = \left(\frac{{{nn}({nn}+1)}}{{2}}\right)^2"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "nn": S(_IDX_POOL),
            "kk": X(_IDX_POOL, ("nn",)),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B1: Divisibility and GCD (8 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B1: list[Template] = [
    Template(
        name="divisibility_def",
        latex=(
            r"{aa} \mid {bb}"
            r" \iff \exists {cc} \in \mathbb{{Z}},\; {bb} = {cc} \cdot {aa}"
        ),
        slots={
            "aa": S(_INT_POOL),
            "bb": X(_INT_POOL, ("aa",)),
            "cc": X(_INT_POOL, ("aa", "bb")),
        },
    ),
    Template(
        name="bezout_identity",
        latex=r"\gcd({aa},{bb}) = {xx}\,{aa} + {yy}\,{bb}",
        slots={
            "aa": S(_INT_POOL),
            "bb": X(_INT_POOL, ("aa",)),
            "xx": X(_INT_POOL, ("aa", "bb")),
            "yy": X(_INT_POOL, ("aa", "bb", "xx")),
        },
    ),
    Template(
        name="euclidean_step",
        latex=r"{aa} = {qq}\,{bb} + {rr},\quad 0 \leq {rr} < {bb}",
        slots={
            "aa": S(_INT_POOL),
            "bb": X(_INT_POOL, ("aa",)),
            "qq": X(_INT_POOL, ("aa", "bb")),
            "rr": X(_INT_POOL, ("aa", "bb", "qq")),
        },
    ),
    Template(
        name="lcm_via_gcd",
        latex=(
            r"\operatorname{{lcm}}({aa},{bb})"
            r" = \frac{{{aa} \cdot {bb}}}{{\gcd({aa},{bb})}}"
        ),
        slots={
            "aa": S(_INT_POOL),
            "bb": X(_INT_POOL, ("aa",)),
        },
    ),
    Template(
        name="divisor_sum_phi",
        latex=r"\sum{lim_mod}_{{d \mid {nn}}} \phi(d) = {nn}",
        slots={"lim_mod": S(("", r"\limits")), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="prime_factorization",
        latex=r"{nn} = \prod{lim_mod}_{{i=1}}^{{{kk}}} {pp}_i^{{e_i}}",
        slots={
            "lim_mod": S(("", r"\limits")),
            "nn": S(_IDX_POOL),
            "kk": X(_IDX_POOL, ("nn",)),
            "pp": S(_PRIME_BASE_POOL),
        },
    ),
    Template(
        name="coprime_reduction",
        latex=(
            r"\gcd({aa},{mm}) = 1 \implies"
            r" {aa}^{{{nn}}} \equiv {aa}^{{{nn} \bmod \phi({mm})}} \pmod{{{mm}}}"
        ),
        slots={
            "aa": S(_INT_POOL),
            "mm": S(_MOD_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="unique_factorization",
        latex=(
            r"{nn} = {pp}_1^{{a_1}} \cdots {pp}_{{{kk}}}^{{a_{{{kk}}}}}"
            r"\quad (\text{{unique up to order}})"
        ),
        slots={
            "nn": S(_IDX_POOL),
            "pp": S(_PRIME_BASE_POOL),
            "kk": X(_IDX_POOL, ("nn",)),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B2: Modular arithmetic (8 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B2: list[Template] = [
    Template(
        name="crt",
        latex=(
            r"{xx} \equiv {aa} \pmod{{{mm}}},\;"
            r" {xx} \equiv {bb} \pmod{{{nn}}}"
            r" \implies {xx} \equiv {cc} \pmod{{{mm}{nn}}}"
        ),
        slots={
            "xx": S(_INT_POOL),
            "aa": X(_INT_POOL, ("xx",)),
            "bb": X(_INT_POOL, ("xx", "aa")),
            "cc": X(_INT_POOL, ("xx", "aa", "bb")),
            "mm": S(_MOD_POOL),
            "nn": X(_MOD_POOL, ("mm",)),
        },
    ),
    Template(
        name="euler_theorem",
        latex=(
            r"\gcd({aa},{nn}) = 1"
            r" \implies {aa}^{{\phi({nn})}} \equiv 1 \pmod{{{nn}}}"
        ),
        slots={
            "aa": S(_INT_POOL),
            "nn": S(_MOD_POOL),
        },
    ),
    Template(
        name="modular_inverse",
        latex=r"{aa}\,{aa}^{{-1}} \equiv 1 \pmod{{{mm}}}",
        slots={
            "aa": S(_INT_POOL),
            "mm": S(_MOD_POOL),
        },
    ),
    Template(
        name="primitive_root_order",
        latex=(
            r"\operatorname{{ord}}_{{{pp}}}({aa}) = \phi({pp}),"
            r"\quad {aa} \text{{ is a primitive root mod }} {pp}"
        ),
        slots={
            "aa": S(_INT_POOL),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="discrete_log",
        latex=r"{aa} \equiv {gg}^{{{kk}}} \pmod{{{pp}}}",
        slots={
            "aa": S(_INT_POOL),
            "gg": X(_INT_POOL, ("aa",)),
            "kk": S(_IDX_POOL),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="quadratic_residue_def",
        latex=(
            r"{aa} \text{{ is a QR mod }} {pp}"
            r" \iff \exists {xx} \in \mathbb{{Z}},\;"
            r" {xx}^2 \equiv {aa} \pmod{{{pp}}}"
        ),
        slots={
            "aa": S(_INT_POOL),
            "pp": S(_PRIME_POOL),
            "xx": X(_INT_POOL, ("aa",)),
        },
    ),
    Template(
        name="legendre_symbol",
        latex=(
            r"\left(\frac{{{aa}}}{{{pp}}}\right)"
            r" \equiv {aa}^{{({pp}-1)/2}} \pmod{{{pp}}}"
        ),
        slots={
            "aa": S(_INT_POOL),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="quadratic_reciprocity",
        latex=(
            r"\left(\frac{{{pp}}}{{{qq}}}\right)"
            r"\left(\frac{{{qq}}}{{{pp}}}\right)"
            r" = (-1)^{{\frac{{{pp}-1}}{{2}}\cdot\frac{{{qq}-1}}{{2}}}}"
        ),
        slots={
            "pp": S(_PRIME_POOL),
            "qq": X(_PRIME_POOL, ("pp",)),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B3: Arithmetic functions (8 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B3: list[Template] = [
    Template(
        name="phi_prime_power",
        latex=r"\phi({pp}^{{{kk}}}) = {pp}^{{{kk}-1}}({pp}-1)",
        slots={
            "pp": S(_PRIME_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="phi_multiplicative",
        latex=(
            r"\gcd({mm},{nn})=1"
            r" \implies \phi({mm}{nn}) = \phi({mm})\phi({nn})"
        ),
        slots={
            "mm": S(_MOD_POOL),
            "nn": X(_MOD_POOL, ("mm",)),
        },
    ),
    Template(
        name="sigma_prime_power",
        latex=(r"\sigma({pp}^{{{kk}}}) = \frac{{{pp}^{{{kk}+1}}-1}}{{{pp}-1}}"),
        slots={
            "pp": S(_PRIME_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="dirichlet_convolution",
        latex=(
            r"({ff} * {gg})({nn})"
            r" = \sum{lim_mod}_{{d \mid {nn}}} {ff}(d)\,{gg}({nn}/d)"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="mobius_def",
        latex=(
            r"\mu({nn}) = \begin{{cases}}"
            r" 1 & {nn}=1 \\"
            r" (-1)^{{{kk}}} & {nn}=p_1\cdots p_{{{kk}}} \\"
            r" 0 & \text{{otherwise}}"
            r" \end{{cases}}"
        ),
        slots={
            "nn": S(_IDX_POOL),
            "kk": X(_IDX_POOL, ("nn",)),
        },
    ),
    Template(
        name="liouville_function",
        latex=(
            r"\lambda({nn}) = (-1)^{{\Omega({nn})}},"
            r"\quad \Omega({nn}) = \sum{lim_mod}_{{{pp}^k \mid\mid {nn}}} k"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "nn": S(_IDX_POOL),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="von_mangoldt_def",
        latex=(
            r"\Lambda({nn}) = \begin{{cases}}"
            r" \log {pp} & {nn}={pp}^{{{kk}}} \\"
            r" 0 & \text{{otherwise}}"
            r" \end{{cases}}"
        ),
        slots={
            "nn": S(_IDX_POOL),
            "pp": S(_PRIME_POOL),
            "kk": X(_IDX_POOL, ("nn",)),
        },
    ),
    Template(
        name="multiplicative_def",
        latex=(
            r"{ff}({mm}\,{nn}) = {ff}({mm})\,{ff}({nn})"
            r"\quad (\gcd({mm},{nn})=1)"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "mm": S(_MOD_POOL),
            "nn": X(_MOD_POOL, ("mm",)),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B4: Combinatorial identities (8 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B4: list[Template] = [
    Template(
        name="vandermonde_identity",
        latex=(
            r"\sum{lim_mod}_{{{kk}=0}}^{{{rr}}}"
            r" \binom{{{mm}}}{{{kk}}}\binom{{{nn}}}{{{rr}-{kk}}}"
            r" = \binom{{{mm}+{nn}}}{{{rr}}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "mm": S(_IDX_POOL),
            "nn": X(_IDX_POOL, ("mm",)),
            "kk": X(_IDX_POOL, ("mm", "nn")),
            "rr": X(_IDX_POOL, ("mm", "nn", "kk")),
        },
    ),
    Template(
        name="hockey_stick_identity",
        latex=(
            r"\sum{lim_mod}_{{{ii}={rr}}}^{{{nn}}} \binom{{{ii}}}{{{rr}}}"
            r" = \binom{{{nn}+1}}{{{rr}+1}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "nn": S(_IDX_POOL),
            "rr": X(_IDX_POOL, ("nn",)),
            "ii": X(_IDX_POOL, ("nn", "rr")),
        },
    ),
    Template(
        name="multinomial_coefficient",
        latex=(
            r"\binom{{{nn}}}{{{kk}_1,\ldots,{kk}_{{{mm}}}}}"
            r" = \frac{{{nn}!}}{{{kk}_1!\cdots {kk}_{{{mm}}}!}}"
        ),
        slots={
            "nn": S(_IDX_POOL),
            "kk": X(_IDX_POOL, ("nn",)),
            "mm": X(_IDX_POOL, ("nn", "kk")),
        },
    ),
    Template(
        name="lucas_theorem",
        latex=(
            r"\binom{{{mm}}}{{{nn}}}"
            r" \equiv \prod{lim_mod}_{{i=0}}^{{{kk}}} \binom{{m_i}}{{n_i}}"
            r" \pmod{{{pp}}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "mm": S(_IDX_POOL),
            "nn": X(_IDX_POOL, ("mm",)),
            "kk": X(_IDX_POOL, ("mm", "nn")),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="stirling_second_kind",
        latex=(
            r"S({nn},{kk}) = \frac{{1}}{{{kk}!}}"
            r"\sum{lim_mod}_{{j=0}}^{{{kk}}} (-1)^j \binom{{{kk}}}{{j}} ({kk}-j)^{{{nn}}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "nn": S(_IDX_POOL),
            "kk": X(_IDX_POOL, ("nn",)),
        },
    ),
    Template(
        name="stirling_first_kind",
        latex=(
            r"\genfrac{{[}}{{]}}{{0pt}}{{}}{{{nn}}}{{{kk}}}"
            r" = ({nn}-1)\,\genfrac{{[}}{{]}}{{0pt}}{{}}{{{nn}-1}}{{{kk}}}"
            r" + \genfrac{{[}}{{]}}{{0pt}}{{}}{{{nn}-1}}{{{kk}-1}}"
        ),
        slots={
            "nn": S(_IDX_POOL),
            "kk": X(_IDX_POOL, ("nn",)),
        },
    ),
    Template(
        name="bell_number",
        latex=r"B_{{{nn}}} = \sum{lim_mod}_{{{kk}=0}}^{{{nn}}} S({nn},{kk})",
        slots={
            "lim_mod": S(("", r"\limits")),
            "nn": S(_IDX_POOL),
            "kk": X(_IDX_POOL, ("nn",)),
        },
    ),
    Template(
        name="catalan_ballot",
        latex=(
            r"C_{{{nn}}} = \frac{{1}}{{{nn}+1}}\binom{{2{nn}}}{{{nn}}}"
            r" = \binom{{2{nn}}}{{{nn}}} - \binom{{2{nn}}}{{{nn}+1}}"
        ),
        slots={"nn": S(_IDX_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B5: Diophantine equations and continued fractions (6 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B5: list[Template] = [
    Template(
        name="linear_diophantine",
        latex=r"{aa}\,{xx} + {bb}\,{yy} = \gcd({aa},{bb})",
        slots={
            "aa": S(_INT_POOL),
            "bb": X(_INT_POOL, ("aa",)),
            "xx": X(_INT_POOL, ("aa", "bb")),
            "yy": X(_INT_POOL, ("aa", "bb", "xx")),
        },
    ),
    Template(
        name="pell_equation",
        latex=r"{xx}^2 - {DD}\,{yy}^2 = 1",
        slots={
            "xx": S(_INT_POOL),
            "yy": X(_INT_POOL, ("xx",)),
            "DD": S(_IDX_POOL),
        },
    ),
    Template(
        name="pythagorean_triple",
        latex=(
            r"{aa} = {mm}^2 - {nn}^2,\;"
            r" {bb} = 2{mm}{nn},\;"
            r" {cc} = {mm}^2+{nn}^2"
        ),
        slots={
            "aa": S(_INT_POOL),
            "bb": X(_INT_POOL, ("aa",)),
            "cc": X(_INT_POOL, ("aa", "bb")),
            "mm": S(_IDX_POOL),
            "nn": X(_IDX_POOL, ("mm",)),
        },
    ),
    Template(
        name="continued_fraction_def",
        latex=(
            r"{xx} = {aa}_0 + \cfrac{{1}}"
            r"{{{aa}_1 + \cfrac{{1}}{{{aa}_2 + \cdots}}}}"
        ),
        slots={
            "xx": S(_INT_POOL),
            "aa": X(_INT_POOL, ("xx",)),
        },
    ),
    Template(
        name="convergent_recurrence",
        latex=(r"{pp}_{{{nn}}} = {aa}_{{{nn}}}\,{pp}_{{{nn}-1}} + {pp}_{{{nn}-2}}"),
        slots={
            "pp": S(_INT_POOL),
            "aa": X(_INT_POOL, ("pp",)),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="fermat_sum_two_squares",
        latex=(
            r"{pp} \equiv 1 \pmod{{4}}"
            r" \implies {pp} = {aa}^2 + {bb}^2"
        ),
        slots={
            "pp": S(_PRIME_POOL),
            "aa": S(_INT_POOL),
            "bb": X(_INT_POOL, ("aa",)),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B5b: cfrac alignment variants and deep nesting (3 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B5b: list[Template] = [
    Template(
        name="cfrac_aligned_variant",
        latex="",
        slots={},
        variants=[
            Template(
                name="cfrac_default_two_level",
                latex=(
                    r"{xx} = {aa}_0 + \cfrac{{1}}"
                    r"{{{aa}_1 + \cfrac{{1}}{{{aa}_2 + \cdots}}}}"
                ),
                slots={"xx": S(_INT_POOL), "aa": X(_INT_POOL, ("xx",))},
            ),
            Template(
                name="cfrac_left_two_level",
                latex=(
                    r"{xx} = {aa}_0 + \cfrac[l]{{1}}"
                    r"{{{aa}_1 + \cfrac[l]{{1}}{{{aa}_2 + \cdots}}}}"
                ),
                slots={"xx": S(_INT_POOL), "aa": X(_INT_POOL, ("xx",))},
            ),
            Template(
                name="cfrac_right_two_level",
                latex=(
                    r"{xx} = {aa}_0 + \cfrac[r]{{1}}"
                    r"{{{aa}_1 + \cfrac[r]{{1}}{{{aa}_2 + \cdots}}}}"
                ),
                slots={"xx": S(_INT_POOL), "aa": X(_INT_POOL, ("xx",))},
            ),
        ],
    ),
    Template(
        name="cfrac_three_level",
        latex=(
            r"{xx} = {aa}_0 + \cfrac{{1}}"
            r"{{{aa}_1 + \cfrac{{1}}{{{aa}_2 + \cfrac{{1}}{{{aa}_3 + \cdots}}}}}}"
        ),
        slots={
            "xx": S(_INT_POOL),
            "aa": X(_INT_POOL, ("xx",)),
        },
    ),
    Template(
        name="cfrac_three_level_named",
        latex=(
            r"{aa}_0 + \cfrac{{1}}"
            r"{{{aa}_1 + \cfrac{{1}}{{{aa}_2 + \cfrac{{1}}{{{aa}_3 + \cdots}}}}}}"
        ),
        slots={
            "aa": S(_INT_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B6: Analytic number theory (8 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B6: list[Template] = [
    Template(
        name="dirichlet_series_def",
        latex=(
            r"F({ss}) = \sum{lim_mod}_{{{nn}=1}}^{{\infty}}"
            r" \frac{{{ff}({nn})}}{{{nn}^{{{ss}}}}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "ss": S(_ALPHA_POOL),
            "nn": S(_IDX_POOL),
            "ff": S(_FUNC_POOL),
        },
    ),
    Template(
        name="euler_product_general",
        latex=(
            r"F({ss}) = \prod{lim_mod}_{{{pp} \text{{ prime}}}}"
            r" \left(1 - \frac{{{ff}({pp})}}{{{pp}^{{{ss}}}}}\right)^{{-1}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "ss": S(_ALPHA_POOL),
            "pp": S(_PRIME_POOL),
            "ff": S(_FUNC_POOL),
        },
    ),
    Template(
        name="pnt_li",
        latex=(
            r"\pi({xx}) \sim \mathrm{{Li}}({xx})"
            r" = \int_2^{{{xx}}} \frac{{dt}}{{\ln t}}"
        ),
        slots={"xx": S(_INT_POOL)},
    ),
    Template(
        name="dirichlet_l_function",
        latex=(
            r"L({ss},\chi) = \sum{lim_mod}_{{{nn}=1}}^{{\infty}}"
            r" \frac{{\chi({nn})}}{{{nn}^{{{ss}}}}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "ss": S(_ALPHA_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="chebyshev_psi",
        latex=(r"\psi({xx}) = \sum{lim_mod}_{{{pp}^{{{kk}}} \leq {xx}}} \log {pp}"),
        slots={
            "lim_mod": S(("", r"\limits")),
            "xx": S(_INT_POOL),
            "pp": S(_PRIME_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="bertrand_postulate",
        latex=(
            r"\forall {nn} \geq 1,\;\exists \text{{ prime }} {pp}:"
            r"\; {nn} < {pp} \leq 2{nn}"
        ),
        slots={
            "nn": S(_IDX_POOL),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="functional_equation_zeta",
        latex=r"\xi({ss}) = \xi(1-{ss})",
        slots={"ss": S(_ALPHA_POOL)},
    ),
    Template(
        name="mertens_theorem",
        latex=(
            r"\prod{lim_mod}_{{{pp} \leq {xx}}}"
            r" \left(1 - \frac{{1}}{{{pp}}}\right)"
            r" \sim \frac{{e^{{-\gamma}}}}{{\ln {xx}}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "xx": S(_INT_POOL),
            "pp": S(_PRIME_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B7: Algebraic number theory (6 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B7: list[Template] = [
    Template(
        name="norm_def",
        latex=(
            r"N_{{{KK}/\mathbb{{Q}}}}({al})"
            r" = \prod{lim_mod}_{{i=1}}^{{{nn}}} \sigma_i({al})"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "KK": S(_FIELD_POOL),
            "al": S(_ALPHA_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="trace_def",
        latex=(
            r"\mathrm{{Tr}}_{{{KK}/\mathbb{{Q}}}}({al})"
            r" = \sum{lim_mod}_{{i=1}}^{{{nn}}} \sigma_i({al})"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "KK": S(_FIELD_POOL),
            "al": S(_ALPHA_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="dedekind_factorization",
        latex=(
            r"{pp}\,\mathcal{{O}}_K"
            r" = \prod{lim_mod}_{{i=1}}^{{{gg}}} \mathfrak{{P}}_i^{{e_i}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "pp": S(_PRIME_POOL),
            "gg": S(_IDX_POOL),
        },
    ),
    Template(
        name="ideal_norm_mult",
        latex=(
            r"N(\mathfrak{{a}}\mathfrak{{b}})"
            r" = N(\mathfrak{{a}})\,N(\mathfrak{{b}})"
            r"\quad \text{{in }} {RR}"
        ),
        slots={"RR": S(_RING_POOL)},
    ),
    Template(
        name="class_group_order",
        latex=(
            r"h_K = |\mathrm{{Cl}}(\mathcal{{O}}_{{{KK}}})|,"
            r"\quad {KK}/\mathbb{{Q}} \text{{ a number field}}"
        ),
        slots={"KK": S(_FIELD_POOL)},
    ),
    Template(
        name="minkowski_bound",
        latex=(
            r"M_K \leq \frac{{n!}}{{n^n}}"
            r"\left(\frac{{4}}{{\pi}}\right)^{{{rr}}}"
            r" \sqrt{{|\Delta_K|}}"
        ),
        slots={"rr": S(_IDX_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B8: p-adic valuations (6 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_B8: list[Template] = [
    Template(
        name="p_adic_valuation_def",
        latex=r"v_{{{pp}}}({nn}) = \max\!\{{k \geq 0 : {pp}^k \mid {nn}\}}",
        slots={
            "pp": S(_PRIME_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="p_adic_norm",
        latex=r"\|{nn}\|_{{{pp}}} = {pp}^{{-v_{{{pp}}}({nn})}}",
        slots={
            "pp": S(_PRIME_POOL),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="ultrametric_inequality",
        latex=(
            r"\|{aa}+{bb}\|_{{{pp}}}"
            r" \leq \max\!\left(\|{aa}\|_{{{pp}}},\,\|{bb}\|_{{{pp}}}\right)"
        ),
        slots={
            "aa": S(_INT_POOL),
            "bb": X(_INT_POOL, ("aa",)),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="p_adic_expansion",
        latex=(
            r"{aa} = \sum{lim_mod}_{{{kk}=0}}^{{\infty}} a_{{{kk}}}\,{pp}^{{{kk}}},"
            r"\quad a_{{{kk}}} \in \{{0,\ldots,{pp}-1\}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "aa": S(_INT_POOL),
            "pp": S(_PRIME_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="hensel_lemma",
        latex=(
            r"{ff}({aa}) \equiv 0 \pmod{{{pp}}},\;"
            r" {ff}'({aa}) \not\equiv 0"
            r" \implies \exists \tilde{{{aa}}},\;"
            r" {ff}(\tilde{{{aa}}}) \equiv 0 \pmod{{{pp}^{{{kk}}}}}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "aa": S(_INT_POOL),
            "pp": S(_PRIME_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="product_formula",
        latex=(r"\|{aa}\|_\infty \cdot \prod{lim_mod}_{{{pp}}} \|{aa}\|_{{{pp}}} = 1"),
        slots={
            "lim_mod": S(("", r"\limits")),
            "aa": S(_INT_POOL),
            "pp": S(_PRIME_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part C: High-n_eff function-pair templates (6 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_C: list[Template] = [
    Template(
        name="dirichlet_conv_pair",
        latex=(
            r"({fn1} * {fn2})({nn})"
            r" = \sum{lim_mod}_{{d \mid {nn}}} {fn1}(d)\,{fn2}({nn}/d)"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="multiplicative_pair",
        latex=(
            r"\gcd({mm},{nn})=1"
            r" \implies {fn1}({mm}\,{nn}) = {fn1}({mm})\,{fn1}({nn})"
        ),
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "mm": S(_MOD_POOL),
            "nn": X(_MOD_POOL, ("mm",)),
        },
    ),
    Template(
        name="dirichlet_series_pair",
        latex=(
            r"\sum{lim_mod}_{{{nn}=1}}^\infty \frac{{{fn1}({nn})}}{{{nn}^{{{ss}}}}}"
            r" \cdot \sum{lim_mod}_{{{nn}=1}}^\infty \frac{{{fn2}({nn})}}{{{nn}^{{{ss}}}}}"
            r" = \sum{lim_mod}_{{{nn}=1}}^\infty"
            r" \frac{{({fn1}*{fn2})({nn})}}{{{nn}^{{{ss}}}}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "nn": S(_IDX_POOL),
            "ss": S(_ALPHA_POOL),
        },
    ),
    Template(
        name="mobius_inv_pair",
        latex=(
            r"{fn1}({nn}) = \sum{lim_mod}_{{d \mid {nn}}} {fn2}(d)"
            r" \iff {fn2}({nn}) = \sum{lim_mod}_{{d \mid {nn}}} \mu(d)\,{fn1}({nn}/d)"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="sum_divisors_pair",
        latex=(
            r"\sum{lim_mod}_{{d \mid {nn}}} {fn1}(d)\,{fn2}({nn}/d)"
            r" = \sum{lim_mod}_{{d \mid {nn}}} {fn2}(d)\,{fn1}({nn}/d)"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="euler_product_pair",
        latex=(
            r"\prod{lim_mod}_{{{pp}}} \frac{{1}}{{1-{fn1}({pp})\,{pp}^{{-{ss}}}}}"
            r" = \sum{lim_mod}_{{{nn}=1}}^\infty \frac{{{fn1}({nn})}}{{{nn}^{{{ss}}}}}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "fn1": E(_fn_rich_nosub, n=100),
            "pp": S(_PRIME_POOL),
            "nn": S(_IDX_POOL),
            "ss": S(_ALPHA_POOL),
        },
    ),
]

# Part C additions
_TEMPLATES_C += [
    Template(
        name="fn_divisor_identity",
        latex=r"{fn1}(\sigma({nn})) = {fn2}\!\left(\sum{lim_mod}_{{d \mid {nn}}} d\right)",
        slots={
            "lim_mod": S(("", r"\limits")),
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="fn_prime_count",
        latex=r"{fn1}(\pi({nn})) = {fn2}\!\left(\sum{lim_mod}_{{{pp} \leq {nn},\,{pp}\text{{ prime}}}} 1\right)",
        slots={
            "lim_mod": S(("", r"\limits")),
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "nn": S(_IDX_POOL),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="fn_euler_phi_product",
        latex=r"{fn1}(\varphi({nn})) = {fn2}\!\left({nn} \prod{lim_mod}_{{{pp} \mid {nn}}} \!\!\left(1 - \frac{{1}}{{{pp}}}\right)\right)",
        slots={
            "lim_mod": S(("", r"\limits")),
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "nn": S(_IDX_POOL),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="fn_moebius_inversion",
        latex=r"{fn1}({nn}) = \sum{lim_mod}_{{d \mid {nn}}} {fn2}(d) \iff {fn2}({nn}) = \sum{lim_mod}_{{d \mid {nn}}} \mu(d)\,{fn1}({nn}/d)",
        slots={
            "lim_mod": S(("", r"\limits")),
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "nn": S(_IDX_POOL),
        },
    ),
    Template(
        name="fn_zeta_product",
        latex=r"{fn1}(\zeta({ss})) = {fn2}\!\left(\prod{lim_mod}_{{{pp}\text{{ prime}}}} \frac{{1}}{{1-{pp}^{{-{ss}}}}}\right)",
        slots={
            "lim_mod": S(("", r"\limits")),
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "ss": S(_ALPHA_POOL),
            "pp": S(_PRIME_POOL),
        },
    ),
    Template(
        name="fn_arithmetic_pair",
        latex=r"{fn1}({aa} \cdot {bb}) = {fn2}({aa}) \cdot {fn2}({bb}),\quad \gcd({aa},{bb})=1",
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "aa": S(_INT_POOL),
            "bb": X(_INT_POOL, ("aa",)),
        },
    ),
]

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

_TEMPLATES_D: list[Template] = [
    Template(
        name="nexists_quadratic_residue",
        latex=r"\nexists\, {aa} \in \mathbb{{Z}}:\; {aa}^2 \equiv {bb} \pmod{{{mm}}}",
        slots={"aa": S(_INT_POOL), "bb": X(_INT_POOL, ("aa",)), "mm": S(_PRIME_POOL)},
    ),
    Template(
        name="nexists_common_factor",
        latex=r"\nexists\, {pp} \text{{ prime}}:\; {pp} \mid {aa} \;\wedge\; {pp} \mid {bb}",
        slots={"aa": S(_INT_POOL), "bb": X(_INT_POOL, ("aa",)), "pp": S(_PRIME_POOL)},
    ),
]

_NUMBER_THEORY_TEMPLATES: list[Template] = (
    _TEMPLATES_A
    + _TEMPLATES_B1
    + _TEMPLATES_B2
    + _TEMPLATES_B3
    + _TEMPLATES_B4
    + _TEMPLATES_B5
    + _TEMPLATES_B5b
    + _TEMPLATES_B6
    + _TEMPLATES_B7
    + _TEMPLATES_B8
    + _TEMPLATES_C
    + _TEMPLATES_D
)

_W_NT: list[float] = compute_weights(_NUMBER_THEORY_TEMPLATES)
_number_theory = make_dispatcher(_NUMBER_THEORY_TEMPLATES, _W_NT)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "number_theory": _number_theory,
}

WEIGHTS: dict[str, float] = {
    "number_theory": 0.04,
}

TEMPLATES: dict[str, list[Template]] = {
    "number_theory": _NUMBER_THEORY_TEMPLATES,
}
