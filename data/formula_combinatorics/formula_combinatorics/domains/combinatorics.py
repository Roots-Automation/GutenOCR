"""Combinatorics domain generators."""

from __future__ import annotations

import random

from .._template_dsl import _FN_SLOT, _LIM_MOD, EP, P, S, Template, X
from .._vocab import (
    _COMB_K as _K_POOL,
)
from .._vocab import (
    _COMB_N as _N_POOL,
)
from .._vocab import (
    _SCALARS,
    _VARS,
)
from ._config import register_domain

# ---------------------------------------------------------------------------
# Shared pools
# ---------------------------------------------------------------------------
_SET_POOL: tuple[str, ...] = (
    "A",
    "B",
    "C",
    "D",
    "E",
    "S",
    "T",
    "U",
    r"\mathcal{A}",
    r"\mathcal{B}",
    r"\mathcal{C}",
    r"\mathcal{S}",
)

# ---------------------------------------------------------------------------
# Inline sub-generators
# ---------------------------------------------------------------------------


def _multinomial_full(rng: random.Random, n: str) -> str:
    """Draw two distinct scalars both excluding n; return the full multinomial formula."""
    pool = [s for s in _SCALARS if s != n]
    a, b = rng.sample(pool, 2)
    return (
        rf"\binom{{{n}}}{{{a},\,{b},\,{n}-{a}-{b}}} "
        rf"= \frac{{{n}!}}{{{a}!\,{b}!\,({n}-{a}-{b})!}}"
    )


def _recurrence_rhs(rng: random.Random, n: str) -> str:
    """Master-theorem RHS using the drawn n variable name."""
    return rng.choice([f"O({n})", "O(1)", f"O({n}^2)", rf"O(\log {n})", rf"O({n} \log {n})"])


def _lin_rec_a(rng: random.Random, n: str) -> str:
    """First recurrence coefficient, drawn from _SCALARS excluding n."""
    pool = [s for s in _SCALARS if s != n]
    return rng.choice(pool)


def _lin_rec_b(rng: random.Random, _param: str, exclude: frozenset[str]) -> str:
    """Second recurrence coefficient, drawn from _SCALARS excluding all values in exclude."""
    pool = [s for s in _SCALARS if s not in exclude]
    return rng.choice(pool)


# ---------------------------------------------------------------------------
# Part A — core combinatorics identities
# ---------------------------------------------------------------------------

_PART_A: list[Template] = [
    # c=0 — permutations P(n,k)
    Template(
        name="permutations",
        latex=r"P({n}, {k}) = \frac{{{n}!}}{{({n}-{k})!}}",
        slots={
            "n": S(_N_POOL),
            "k": S(_K_POOL),
        },
    ),
    # c=1 — derangements (inclusion-exclusion sum; ii is the summation index)
    Template(
        name="derangements_sum",
        latex=r"D_{{{n}}} = {n}!\sum{lim_mod}_{{{ii}=0}}^{{{n}}} \frac{{(-1)^{{{ii}}}}}{{{ii}!}}",
        slots={"lim_mod": _LIM_MOD, "n": S(_N_POOL), "ii": S(("j", "i", "k", "l"))},
    ),
    # c=2 — sum of all binomial coefficients
    Template(
        name="sum_binomials",
        latex=r"\sum{lim_mod}_{{{k}=0}}^{{{n}}} \binom{{{n}}}{{{k}}} = 2^{{{n}}}",
        slots={
            "lim_mod": _LIM_MOD,
            "n": S(_N_POOL),
            "k": S(_K_POOL),
        },
    ),
    # c=3 — binomial theorem (v is now a slot instead of hardcoded x)
    Template(
        name="binomial_theorem",
        latex=r"\sum{lim_mod}_{{{k}=0}}^{{{n}}} \binom{{{n}}}{{{k}}} {v}^{{{k}}} = (1+{v})^{{{n}}}",
        slots={
            "lim_mod": _LIM_MOD,
            "n": S(_N_POOL),
            "k": S(_K_POOL),
            "v": S(_VARS),
        },
    ),
    # c=4 — Pascal's rule
    Template(
        name="pascal_rule",
        latex=r"\binom{{{n}}}{{{k}}} = \binom{{{n}-1}}{{{k}-1}} + \binom{{{n}-1}}{{{k}}}",
        slots={
            "n": S(_N_POOL),
            "k": S(_K_POOL),
        },
    ),
    # c=5 — multinomial coefficient; a and b both exclude the drawn n
    Template(
        name="multinomial_coefficient",
        latex=r"{formula}",
        slots={
            "n": S(_N_POOL),
            "formula": P(_multinomial_full, param="n", n=72),
        },
    ),
    # c=6 — master theorem / divide-and-conquer recurrence (ff is the time function)
    Template(
        name="master_theorem_recurrence",
        latex=r"{ff}({n}) = {aa}\,{ff}\!\left(\frac{{{n}}}{{{bb}}}\right) + {rhs}",
        slots={
            "ff": _FN_SLOT,
            "n": S(_N_POOL),
            "aa": S(("2", "3", "4", "a", "b")),
            "bb": S(("2", "3", "4", "b", "c")),
            "rhs": P(_recurrence_rhs, param="n", n=5),
        },
    ),
    # c=7 — linear recurrence a_n = c1*a_{n-1} + c2*a_{n-2}
    # c1 excludes n; c2 excludes both n and c1.
    Template(
        name="linear_recurrence",
        latex=r"a_{{{n}}} = {c1}\,a_{{{n}-1}} + {c2}\,a_{{{n}-2}}",
        slots={
            "n": S(_N_POOL),
            "c1": P(_lin_rec_a, param="n", n=8),
            "c2": EP(_lin_rec_b, param="n", exclude=("n", "c1"), n=7),
        },
    ),
    # c=7b — named-sequence recurrence (rich function name for the sequence)
    Template(
        name="named_sequence_recurrence",
        latex=r"{ff}_{{{n}}} = {c1}\,{ff}_{{{n}-1}} + {c2}\,{ff}_{{{n}-2}}",
        slots={
            "ff": _FN_SLOT,
            "n": S(_N_POOL),
            "c1": P(_lin_rec_a, param="n", n=8),
            "c2": EP(_lin_rec_b, param="n", exclude=("n", "c1"), n=7),
        },
    ),
    # c=8 — ordinary generating function (gg uses rich function name)
    Template(
        name="ordinary_generating_function",
        latex=r"{gg}({v}) = \sum{lim_mod}_{{{n} \geq 0}} {sc}_{{{n}}}\, {v}^{{{n}}}",
        slots={
            "lim_mod": _LIM_MOD,
            "n": S(_N_POOL),
            "gg": _FN_SLOT,
            "v": S(_VARS),
            "sc": S(_SCALARS),
        },
    ),
    # c=9 — multiset (stars-and-bars) coefficient
    Template(
        name="multiset_coefficient",
        latex=r"\binom{{{n}+{k}-1}}{{{k}-1}}",
        slots={
            "n": S(_N_POOL),
            "k": S(_K_POOL),
        },
    ),
    # c=10 — derangement recurrence
    Template(
        name="derangement_recurrence",
        latex=r"D_{{{n}}} = ({n}-1)\!\left(D_{{{n}-1}} + D_{{{n}-2}}\right)",
        slots={"n": S(_N_POOL)},
    ),
    # c=11 — hockey stick / Christmas stocking identity (r2 slot, was hardcoded r)
    Template(
        name="hockey_stick_identity",
        latex=r"\sum{lim_mod}_{{{k}=0}}^{{{r2}}} \binom{{{n}+{k}}}{{{k}}} = \binom{{{n}+{r2}+1}}{{{r2}}}",
        slots={
            "lim_mod": _LIM_MOD,
            "n": S(_N_POOL),
            "k": S(_K_POOL),
            "r2": S(("r", "s", "R", "N")),
        },
    ),
    # c=12 — central binomial asymptotics
    Template(
        name="central_binomial_asymptotics",
        latex=r"\binom{{2{n}}}{{{n}}} \sim \frac{{4^{{{n}}}}}{{\sqrt{{\pi {n}}}}}",
        slots={"n": S(_N_POOL)},
    ),
    # c=13 — Stirling's approximation — 3 notation variants (was n_eff=1, no slots)
    Template(
        name="stirling_approximation",
        latex="",
        slots={},
        variants=[
            Template(
                name="stirling_standard",
                latex=r"{n}! \sim \sqrt{{2\pi {n}}} \left(\frac{{{n}}}{{e}}\right)^{{{n}}}",
                slots={"n": S(_N_POOL)},
            ),
            Template(
                name="stirling_log",
                latex=r"\ln({n}!) \approx {n}\ln {n} - {n} + \frac{{1}}{{2}}\ln(2\pi {n})",
                slots={"n": S(_N_POOL)},
            ),
            Template(
                name="stirling_gamma",
                latex=r"\Gamma({n}+1) \sim \sqrt{{2\pi {n}}} \left(\frac{{{n}}}{{e}}\right)^{{{n}}}",
                slots={"n": S(_N_POOL)},
            ),
        ],
    ),
    # c=14 — derangements alternating sum
    Template(
        name="derangement_alternating_sum",
        latex=(
            r"D_{{{n}}} = {n}! \left(1 - 1 + \frac{{1}}{{2!}} - \frac{{1}}{{3!}}"
            r" + \cdots + \frac{{(-1)^{{{n}}}}}{{{n}!}}\right)"
        ),
        slots={"n": S(_N_POOL)},
    ),
    # c=15 — Vandermonde's convolution
    Template(
        name="vandermonde_identity",
        latex=r"\binom{{{mm}+{n}}}{{{k}}} = \sum{lim_mod}_{{{j}=0}}^{{{k}}} \binom{{{mm}}}{{{j}}}\binom{{{n}}}{{{k}-{j}}}",
        slots={
            "lim_mod": _LIM_MOD,
            "mm": S(_N_POOL),
            "n": X(_N_POOL, ("mm",)),
            "k": S(_K_POOL),
            "j": X(_K_POOL, ("k",)),
        },
    ),
    # c=16 — Catalan numbers (3 equivalent forms)
    Template(
        name="catalan_numbers",
        latex="",
        slots={},
        variants=[
            Template(
                name="catalan_binomial",
                latex=r"C_{{{n}}} = \frac{{1}}{{{n}+1}}\binom{{2{n}}}{{{n}}}",
                slots={"n": S(_N_POOL)},
            ),
            Template(
                name="catalan_factorial",
                latex=r"C_{{{n}}} = \frac{{(2{n})!}}{{({n}+1)!\,{n}!}}",
                slots={"n": S(_N_POOL)},
            ),
            Template(
                name="catalan_recurrence",
                latex=r"C_{{{n}+1}} = \sum{lim_mod}_{{{k}=0}}^{{{n}}} C_{{{k}}} C_{{{n}-{k}}}",
                slots={"lim_mod": _LIM_MOD, "n": S(_N_POOL), "k": S(_K_POOL)},
            ),
        ],
    ),
    # c=17 — Stirling numbers of the second kind (two notation forms)
    Template(
        name="stirling_numbers_second",
        latex="",
        slots={},
        variants=[
            Template(
                name="stirling_second_s_notation",
                latex=r"S({n},{k}) = {k}\,S({n}-1,{k}) + S({n}-1,{k}-1)",
                slots={"n": S(_N_POOL), "k": S(_K_POOL)},
            ),
            Template(
                name="stirling_second_bracket",
                latex=(
                    r"\left\{{\begin{{matrix}}{n}\\{k}\end{{matrix}}\right\}}"
                    r" = {k}\left\{{\begin{{matrix}}{n}-1\\{k}\end{{matrix}}\right\}}"
                    r" + \left\{{\begin{{matrix}}{n}-1\\{k}-1\end{{matrix}}\right\}}"
                ),
                slots={"n": S(_N_POOL), "k": S(_K_POOL)},
            ),
        ],
    ),
    # c=18 — falling factorial / rising factorial (Pochhammer symbol)
    Template(
        name="falling_rising_factorial",
        latex="",
        slots={},
        variants=[
            Template(
                name="falling_factorial_descending",
                latex=r"({v})_{{{n}}} = {v}({v}-1)\cdots({v}-{n}+1)",
                slots={"v": S(_VARS), "n": S(("2", "3", "n", "m", "k"))},
            ),
            Template(
                name="falling_factorial_ratio",
                latex=r"({v})_{{{n}}} = \frac{{{v}!}}{{({v}-{n})!}}",
                slots={"v": S(_VARS), "n": S(_N_POOL)},
            ),
            Template(
                name="rising_factorial_pochhammer",
                latex=r"({sc})_{{{n}}} = {sc}({sc}+1)\cdots({sc}+{n}-1) = \frac{{\Gamma({sc}+{n})}}{{\Gamma({sc})}}",
                slots={"sc": S(_SCALARS), "n": S(_N_POOL)},
            ),
        ],
    ),
    # c=19 — absorption / extraction identity
    Template(
        name="absorption_identity",
        latex=r"{k}\binom{{{n}}}{{{k}}} = {n}\binom{{{n}-1}}{{{k}-1}}",
        slots={"n": S(_N_POOL), "k": S(_K_POOL)},
    ),
    # c=20 — Bell number recurrence (exponential formula)
    Template(
        name="bell_number_recurrence",
        latex=r"B_{{{n}+1}} = \sum{lim_mod}_{{{k}=0}}^{{{n}}} \binom{{{n}}}{{{k}}} B_{{{k}}}",
        slots={"lim_mod": _LIM_MOD, "n": S(_N_POOL), "k": S(_K_POOL)},
    ),
    # c=21 — inclusion-exclusion principle (2-set and 3-set forms)
    Template(
        name="inclusion_exclusion",
        latex="",
        slots={},
        variants=[
            Template(
                name="inclusion_exclusion_two",
                latex=r"|{ss1} \cup {ss2}| = |{ss1}| + |{ss2}| - |{ss1} \cap {ss2}|",
                slots={"ss1": S(_SET_POOL), "ss2": X(_SET_POOL, ("ss1",))},
            ),
            Template(
                name="inclusion_exclusion_three",
                latex=(
                    r"|{ss1} \cup {ss2} \cup {ss3}|"
                    r" = |{ss1}| + |{ss2}| + |{ss3}|"
                    r" - |{ss1} \cap {ss2}| - |{ss1} \cap {ss3}| - |{ss2} \cap {ss3}|"
                    r" + |{ss1} \cap {ss2} \cap {ss3}|"
                ),
                slots={
                    "ss1": S(_SET_POOL),
                    "ss2": X(_SET_POOL, ("ss1",)),
                    "ss3": X(_SET_POOL, ("ss1", "ss2")),
                },
            ),
        ],
    ),
    # c=22 — upper negation / negative binomial coefficient identity
    Template(
        name="negative_binomial_coeff",
        latex=r"\binom{{-{n}}}{{{k}}} = (-1)^{{{k}}} \binom{{{n}+{k}-1}}{{{k}}}",
        slots={"n": S(_N_POOL), "k": S(_K_POOL)},
    ),
    # c=23 — exponential generating function (complement to OGF; gg uses rich function name)
    Template(
        name="exponential_generating_function",
        latex=r"{gg}({v}) = \sum{lim_mod}_{{{n} \geq 0}} {sc}_{{{n}}}\, \frac{{{v}^{{{n}}}}}{{{n}!}}",
        slots={
            "lim_mod": _LIM_MOD,
            "n": S(_N_POOL),
            "gg": _FN_SLOT,
            "v": S(_VARS),
            "sc": S(_SCALARS),
        },
    ),
    # c=24 — Fibonacci identities (4 variants)
    Template(
        name="fibonacci_identities",
        latex="",
        slots={},
        variants=[
            Template(
                name="fibonacci_recurrence",
                latex=r"F_{{{n}}} = F_{{{n}-1}} + F_{{{n}-2}}",
                slots={"n": S(_N_POOL)},
            ),
            Template(
                name="fibonacci_addition",
                latex=r"F_{{{mm}+{n}}} = F_{{{mm}}} F_{{{n}+1}} + F_{{{mm}-1}} F_{{{n}}}",
                slots={"mm": S(_N_POOL), "n": X(_N_POOL, ("mm",))},
            ),
            Template(
                name="fibonacci_sum",
                latex=r"\sum{lim_mod}_{{{k}=1}}^{{{n}}} F_{{{k}}} = F_{{{n}+2}} - 1",
                slots={"lim_mod": _LIM_MOD, "n": S(_N_POOL), "k": S(_K_POOL)},
            ),
            Template(
                name="cassini_identity",
                latex=r"F_{{{n}-1}} F_{{{n}+1}} - F_{{{n}}}^2 = (-1)^{{{n}}}",
                slots={"n": S(_N_POOL)},
            ),
        ],
    ),
]

# ---------------------------------------------------------------------------
# Part B — high-n_eff function-pair templates
# ---------------------------------------------------------------------------

_PART_B: list[Template] = [
    Template(
        name="fn_binomial_symmetry",
        latex=r"{fn1}\!\binom{{{n}}}{{{k}}} = {fn2}\!\binom{{{n}}}{{{n}-{k}}}",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "n": S(_N_POOL),
            "k": S(_K_POOL),
        },
    ),
    Template(
        name="fn_generating_func",
        latex=r"{fn1}\!\left(\sum{lim_mod}_{{{n}\geq 0}} {sc}_{{{n}}}\,{v}^{{{n}}}\right) = {fn2}({v})",
        slots={
            "lim_mod": _LIM_MOD,
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "n": S(_N_POOL),
            "sc": S(_SCALARS),
            "v": S(_VARS),
        },
    ),
    Template(
        name="fn_recurrence_pair",
        latex=r"{fn1}(a_{{{n}}}) = {fn2}\!\left(a_{{{n}-1}} + a_{{{n}-2}}\right)",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "n": S(_N_POOL),
        },
    ),
    Template(
        name="fn_stirling_transform",
        latex=r"{fn1}(S({n},{k})) = {fn2}\!\left(\frac{{1}}{{{k}!}}\sum{lim_mod}_{{j=0}}^{{{k}}} (-1)^{{{k}-j}}\binom{{{k}}}{{j}} j^{{{n}}}\right)",
        slots={
            "lim_mod": _LIM_MOD,
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "n": S(_N_POOL),
            "k": S(_K_POOL),
        },
    ),
    Template(
        name="fn_bell_recurrence",
        latex=r"{fn1}(B_{{{n}+1}}) = {fn2}\!\left(\sum{lim_mod}_{{{k}=0}}^{{{n}}} \binom{{{n}}}{{{k}}} B_{{{k}}}\right)",
        slots={
            "lim_mod": _LIM_MOD,
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "n": S(_N_POOL),
            "k": S(_K_POOL),
        },
    ),
    Template(
        name="fn_catalan_formula",
        latex=r"{fn1}(C_{{{n}}}) = {fn2}\!\left(\frac{{1}}{{{n}+1}}\binom{{2{n}}}{{{n}}}\right)",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "n": S(_N_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_COMBINATORICS_TEMPLATES: list[Template] = _PART_A + _PART_B

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("combinatorics", _COMBINATORICS_TEMPLATES)
