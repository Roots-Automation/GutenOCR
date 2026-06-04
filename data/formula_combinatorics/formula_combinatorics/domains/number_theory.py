"""Number theory domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import S, Template, X, compute_weights, make_dispatcher
from .._vocab import _SCALARS

# ---------------------------------------------------------------------------
# Shared pools
# ---------------------------------------------------------------------------

_N_POOL: tuple[str, ...] = ("n", "m", "N")
_K_POOL: tuple[str, ...] = ("k", "r")

# ---------------------------------------------------------------------------
# Number theory templates
# ---------------------------------------------------------------------------

_NUMBER_THEORY_TEMPLATES: list[Template] = [
    # c=0 — binomial coefficient
    Template(
        name="binomial_coefficient",
        latex=r"\binom{{{n}}}{{{k}}}",
        slots={
            "n": S(_N_POOL),
            "k": S(_K_POOL),
        },
    ),
    # c=1 — binomial coefficient definition
    Template(
        name="binomial_coefficient_def",
        latex=r"\binom{{{n}}}{{{k}}} = \frac{{{n}!}}{{{k}!\,({n}-{k})!}}",
        slots={
            "n": S(_N_POOL),
            "k": S(_K_POOL),
        },
    ),
    # c=2 — modular congruence; m excludes a and b
    Template(
        name="modular_congruence",
        latex=r"{a} \equiv {b} \pmod{{{m}}}",
        slots={
            "a": S(_SCALARS),
            "b": X(_SCALARS, ("a",)),
            "m": X(("p", "q", "m"), ("a", "b")),
        },
    ),
    # c=3 — gcd * lcm identity
    Template(
        name="gcd_lcm_identity",
        latex=r"\gcd({a}, {b}) \cdot \operatorname{{lcm}}({a}, {b}) = {a} \cdot {b}",
        slots={
            "a": S(_SCALARS),
            "b": X(_SCALARS, ("a",)),
        },
    ),
    # c=4 — sum of first n integers
    Template(
        name="sum_first_n",
        latex=r"\sum_{{{k}=1}}^{{{n}}} {k} = \frac{{{n}({n}+1)}}{{2}}",
        slots={
            "n": S(_N_POOL),
            "k": S(_K_POOL),
        },
    ),
    # c=5 — sum of squares
    Template(
        name="sum_of_squares",
        latex=r"\sum_{{{k}=1}}^{{{n}}} {k}^2 = \frac{{{n}({n}+1)(2{n}+1)}}{{6}}",
        slots={
            "n": S(_N_POOL),
            "k": S(_K_POOL),
        },
    ),
    # c=6 — Fibonacci recurrence
    Template(
        name="fibonacci_recurrence",
        latex=r"F_{{{n}+2}} = F_{{{n}+1}} + F_{{{n}}}, \quad F_0 = 0, \; F_1 = 1",
        slots={"n": S(_N_POOL)},
    ),
    # c=7 — Euler's totient formula
    Template(
        name="euler_totient",
        latex=r"\phi({n}) = {n} \prod_{{p \mid {n}}} \left(1 - \frac{{1}}{{p}}\right)",
        slots={"n": S(_N_POOL)},
    ),
    # c=8 — Fermat's little theorem (fixed)
    Template(
        name="fermat_little_theorem",
        latex=r"a^{p-1} \equiv 1 \pmod{p}",
        slots={},
    ),
    # c=9 — Catalan numbers
    Template(
        name="catalan_numbers",
        latex=r"C_{{{n}}} = \frac{{1}}{{{n}+1}} \binom{{2{n}}}{{{n}}}",
        slots={"n": S(_N_POOL)},
    ),
    # c=10 — finite geometric series
    Template(
        name="geometric_series",
        latex=r"\sum_{{{k}=0}}^{{{n}}} {a}^{{{k}}} = \frac{{1 - {a}^{{{n}+1}}}}{{1 - {a}}}",
        slots={
            "n": S(_N_POOL),
            "k": S(_K_POOL),
            "a": S(_SCALARS),
        },
    ),
    # c=11 — factorial as product (fixed)
    Template(
        name="factorial_product",
        latex=r"n! = \prod_{k=1}^{n} k",
        slots={},
    ),
    # c=12 — Leibniz pi series (fixed)
    Template(
        name="leibniz_pi",
        latex=r"\sum_{k=0}^{\infty} \frac{(-1)^k}{2k+1} = \frac{\pi}{4}",
        slots={},
    ),
    # c=13 — Wilson's theorem (fixed)
    Template(
        name="wilson_theorem",
        latex=r"(p-1)! \equiv -1 \pmod{p}",
        slots={},
    ),
    # c=14 — sum of divisors function
    Template(
        name="sum_of_divisors",
        latex=r"\sigma({n}) = \sum_{{d \mid {n}}} d",
        slots={"n": S(_N_POOL)},
    ),
    # c=15 — number of divisors function
    Template(
        name="number_of_divisors",
        latex=r"\tau({n}) = \sum_{{d \mid {n}}} 1 = \prod_{{p^k \| {n}}} (k+1)",
        slots={"n": S(_N_POOL)},
    ),
    # c=16 — prime number theorem (fixed)
    Template(
        name="prime_number_theorem",
        latex=r"\pi(x) \sim \frac{x}{\ln x}",
        slots={},
    ),
    # c=17 — Möbius inversion formula
    Template(
        name="mobius_inversion",
        latex=(
            r"f({n}) = \sum_{{d \mid {n}}} g(d) "
            r"\implies g({n}) = \sum_{{d \mid {n}}} \mu(d) f({n}/d)"
        ),
        slots={"n": S(_N_POOL)},
    ),
    # c=18 — Riemann zeta / Euler product (fixed)
    Template(
        name="riemann_zeta_euler_product",
        latex=r"\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s} = \prod_{p \text{ prime}} \frac{1}{1-p^{-s}}",
        slots={},
    ),
    # c=19 — sum of cubes
    Template(
        name="sum_of_cubes",
        latex=r"\sum_{{{k}=1}}^{{{n}}} {k}^3 = \left(\frac{{{n}({n}+1)}}{{2}}\right)^2",
        slots={
            "n": S(_N_POOL),
            "k": S(_K_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

_W_NT: list[float] = compute_weights(_NUMBER_THEORY_TEMPLATES)

# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

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
