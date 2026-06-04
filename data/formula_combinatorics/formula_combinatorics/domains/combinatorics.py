"""Combinatorics domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import E, P, S, Template, compute_weights, make_dispatcher
from .._vocab import _SCALARS

# ---------------------------------------------------------------------------
# Shared pools
# ---------------------------------------------------------------------------

_N_POOL: tuple[str, ...] = ("n", "m", "N")
_K_POOL: tuple[str, ...] = ("k", "r")

# ---------------------------------------------------------------------------
# Inline sub-generators
# ---------------------------------------------------------------------------


def _k2_sub(rng: random.Random) -> str:
    return rng.choice(["k", "m"])


def _aa_sub(rng: random.Random) -> str:
    return rng.choice(["2", "3", "a"])


def _bb_sub(rng: random.Random) -> str:
    return rng.choice(["2", "3", "b"])


def _multinomial_full(rng: random.Random, n: str) -> str:
    """Draw two distinct scalars both excluding n; return the full multinomial formula."""
    pool = [s for s in _SCALARS if s != n]
    a, b = rng.sample(pool, 2)
    return (
        rf"\binom{{{n}}}{{{a},\,{b},\,{n}-{a}-{b}}} "
        rf"= \frac{{{n}!}}{{{a}!\,{b}!\,({n}-{a}-{b})!}}"
    )


def _recurrence_rhs(rng: random.Random, n: str) -> str:
    """Master-theorem RHS: may echo the drawn n value."""
    return rng.choice([n, "O(1)", "O(n)"])


def _lin_rec_a(rng: random.Random, n: str) -> str:
    """First recurrence coefficient, drawn from _SCALARS excluding n."""
    pool = [s for s in _SCALARS if s != n]
    return rng.choice(pool)


def _lin_rec_b(rng: random.Random, na: str) -> str:
    """Second recurrence coefficient, drawn from _SCALARS excluding n and a (encoded as 'n|a')."""
    n, a = na.split("|", 1)
    pool = [s for s in _SCALARS if s != n and s != a]
    return rng.choice(pool)


# For c=7 (linear recurrence): a and b must exclude n AND each other.
# We handle a via ParamSub(_lin_rec_a, "n") and b via a second ParamSub that
# receives "n|a" encoded by a wrapper.  Because ParamSub only passes one slot
# value, we encode both exclusions into a single combined slot "_na" that carries
# the drawn n and a separated by "|".


def _lin_rec_na_combo(rng: random.Random, n: str) -> str:
    """Draw 'a' from _SCALARS excluding n; return 'a_value|n_value' for b to consume."""
    pool = [s for s in _SCALARS if s != n]
    a = rng.choice(pool)
    return f"{a}|{n}"


# ---------------------------------------------------------------------------
# Combinatorics templates
# ---------------------------------------------------------------------------

_COMBINATORICS_TEMPLATES: list[Template] = [
    # c=0 — permutations P(n,k)
    Template(
        name="permutations",
        latex=r"P({n}, {k}) = \frac{{{n}!}}{{({n}-{k})!}}",
        slots={
            "n": S(_N_POOL),
            "k": S(_K_POOL),
        },
    ),
    # c=1 — derangements (inclusion-exclusion sum)
    Template(
        name="derangements_sum",
        latex=r"D_{{{n}}} = {n}!\sum_{{j=0}}^{{{n}}} \frac{{(-1)^j}}{{j!}}",
        slots={"n": S(_N_POOL)},
    ),
    # c=2 — sum of all binomial coefficients
    Template(
        name="sum_binomials",
        latex=r"\sum_{{{k}=0}}^{{{n}}} \binom{{{n}}}{{{k}}} = 2^{{{n}}}",
        slots={
            "n": S(_N_POOL),
            "k": S(_K_POOL),
        },
    ),
    # c=3 — binomial theorem
    Template(
        name="binomial_theorem",
        latex=r"\sum_{{{k}=0}}^{{{n}}} \binom{{{n}}}{{{k}}} x^{{{k}}} = (1+x)^{{{n}}}",
        slots={
            "n": S(_N_POOL),
            "k": S(_K_POOL),
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
    # c=6 — master theorem / divide-and-conquer recurrence
    Template(
        name="master_theorem_recurrence",
        latex=r"T({n}) = {aa}\,T\!\left(\frac{{{n}}}{{{bb}}}\right) + {rhs}",
        slots={
            "n": S(_N_POOL),
            "aa": E(_aa_sub, n=3),
            "bb": E(_bb_sub, n=3),
            "rhs": P(_recurrence_rhs, param="n", n=3),
        },
    ),
    # c=7 — linear recurrence a_n = c1*a_{n-1} + c2*a_{n-2}
    # a and b must each exclude n; additionally b must exclude a.
    # We draw a via _lin_rec_a(rng, n), then encode "a|n" into a combined slot
    # so that _lin_rec_b can exclude both.
    Template(
        name="linear_recurrence",
        latex=r"a_{{{n}}} = {a}\,a_{{{n}-1}} + {b}\,a_{{{n}-2}}",
        slots={
            "n": S(_N_POOL),
            "a": P(_lin_rec_a, param="n", n=8),
            # _na carries the "a_value|n_value" string; b reads it to exclude both
            "_na": P(_lin_rec_na_combo, param="n", n=8),
            "b": P(_lin_rec_b, param="_na", n=7),
        },
    ),
    # c=8 — ordinary generating function
    Template(
        name="ordinary_generating_function",
        latex=r"G(x) = \sum_{{{n} \geq 0}} a_{{{n}}}\, x^{{{n}}}",
        slots={"n": S(_N_POOL)},
    ),
    # c=9 — multiset (stars-and-bars) coefficient
    Template(
        name="multiset_coefficient",
        latex=r"\binom{{{n}+{k2}-1}}{{{k2}-1}}",
        slots={
            "n": S(_N_POOL),
            "k2": E(_k2_sub, n=2),
        },
    ),
    # c=10 — derangement recurrence
    Template(
        name="derangement_recurrence",
        latex=r"D_{{{n}}} = ({n}-1)\!\left(D_{{{n}-1}} + D_{{{n}-2}}\right)",
        slots={"n": S(_N_POOL)},
    ),
    # c=11 — hockey stick / Christmas stocking identity
    Template(
        name="hockey_stick_identity",
        latex=r"\sum_{{{k}=0}}^{{r}} \binom{{{n}+{k}}}{{{k}}} = \binom{{{n}+r+1}}{{r}}",
        slots={
            "n": S(_N_POOL),
            "k": S(_K_POOL),
        },
    ),
    # c=12 — central binomial asymptotics
    Template(
        name="central_binomial_asymptotics",
        latex=r"\binom{{2{n}}}{{{n}}} \sim \frac{{4^{{{n}}}}}{{\sqrt{{\pi {n}}}}}",
        slots={"n": S(_N_POOL)},
    ),
    # c=13 — Stirling's approximation (fixed)
    Template(
        name="stirling_approximation",
        latex=r"n! \sim \sqrt{2\pi n} \left(\frac{n}{e}\right)^n",
        slots={},
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
]

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

_W_COMB: list[float] = compute_weights(_COMBINATORICS_TEMPLATES)

# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

_combinatorics = make_dispatcher(_COMBINATORICS_TEMPLATES, _W_COMB)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "combinatorics": _combinatorics,
}

WEIGHTS: dict[str, float] = {
    "combinatorics": 0.03,
}

TEMPLATES: dict[str, list[Template]] = {
    "combinatorics": _COMBINATORICS_TEMPLATES,
}
