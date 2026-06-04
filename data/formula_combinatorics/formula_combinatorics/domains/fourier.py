"""Fourier analysis and signals domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import S, Template, compute_weights, make_dispatcher

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_XI_POOL: list[str] = [r"\xi", r"\omega", r"\nu"]
_T_PERIOD_POOL: list[str] = ["T", r"2\pi", "L"]
_TAU_POOL: list[str] = [r"\tau", "s", "u"]
_S_POOL: list[str] = ["s", r"\sigma + i\omega"]
_N_FFT_POOL: list[str] = ["N", "M"]
_DELTA_POOL: list[str] = ["a", "0", r"\tau"]

# ---------------------------------------------------------------------------
# Fourier / signals templates
# ---------------------------------------------------------------------------

_FOURIER_TEMPLATES: list[Template] = [
    Template(
        name="fourier_transform",
        latex=r"\hat{{f}}({xi}) = \int_{{-\infty}}^{{\infty}} f(x)\, e^{{-2\pi i x {xi}}} \, dx",
        slots={"xi": S(_XI_POOL)},
    ),
    Template(
        name="inverse_fourier_transform",
        latex=r"f(x) = \int_{{-\infty}}^{{\infty}} \hat{{f}}({xi})\, e^{{2\pi i x {xi}}} \, d{xi}",
        slots={"xi": S(_XI_POOL)},
    ),
    Template(
        name="fourier_series",
        latex=r"f(x) = \sum_{{n=-\infty}}^{{\infty}} c_n e^{{2\pi i n x / {T}}}",
        slots={"T": S(_T_PERIOD_POOL)},
    ),
    Template(
        name="fourier_coefficients",
        latex=r"c_n = \frac{{1}}{{{T}}} \int_0^{{{T}}} f(x)\, e^{{-2\pi i n x / {T}}} \, dx",
        slots={"T": S(_T_PERIOD_POOL)},
    ),
    Template(
        name="parseval",
        latex=(
            r"\int_{{-\infty}}^{{\infty}} |f(x)|^2 \, dx = "
            r"\int_{{-\infty}}^{{\infty}} |\hat{{f}}(\xi)|^2 \, d\xi"
        ),
        slots={},
    ),
    Template(
        name="convolution",
        latex=r"(f * g)(t) = \int_{{-\infty}}^{{\infty}} f({tau})\, g(t - {tau}) \, d{tau}",
        slots={"tau": S(_TAU_POOL)},
    ),
    Template(
        name="convolution_theorem",
        latex=r"\widehat{{f * g}} = \hat{{f}} \cdot \hat{{g}}",
        slots={},
    ),
    Template(
        name="laplace_transform",
        latex=r"\mathcal{{L}}\{{f(t)\}}({s}) = \int_0^{{\infty}} f(t)\, e^{{-{s} t}} \, dt",
        slots={"s": S(_S_POOL)},
    ),
    Template(
        name="z_transform",
        latex=r"X(z) = \sum_{{n=-\infty}}^{{\infty}} x[n]\, z^{{-n}}",
        slots={},
    ),
    Template(
        name="dft",
        latex=r"X[k] = \sum_{{n=0}}^{{{N}-1}} x[n]\, e^{{-2\pi i k n / {N}}}",
        slots={"N": S(_N_FFT_POOL)},
    ),
    Template(
        name="dirac_delta_sifting",
        latex=r"\int_{{-\infty}}^{{\infty}} \delta(x - {a})\, f(x) \, dx = f({a})",
        slots={"a": S(_DELTA_POOL)},
    ),
    Template(
        name="time_frequency_uncertainty",
        latex=r"\Delta t \, \Delta \omega \geq \frac{{1}}{{2}}",
        slots={},
    ),
    Template(
        name="poisson_summation",
        latex=r"\sum_{{n=-\infty}}^{{\infty}} f(n) = \sum_{{k=-\infty}}^{{\infty}} \hat{{f}}(k)",
        slots={},
    ),
    Template(
        name="transfer_function",
        latex=r"H(\omega) = \frac{{Y(\omega)}}{{X(\omega)}}",
        slots={},
    ),
]

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

_W_FOURIER: list[float] = compute_weights(_FOURIER_TEMPLATES)

# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

_fourier = make_dispatcher(_FOURIER_TEMPLATES, _W_FOURIER)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "fourier": _fourier,
}

WEIGHTS: dict[str, float] = {
    "fourier": 0.02,
}

TEMPLATES: dict[str, list[Template]] = {
    "fourier": _FOURIER_TEMPLATES,
}
