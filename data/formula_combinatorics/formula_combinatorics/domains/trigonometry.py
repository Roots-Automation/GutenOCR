"""Trigonometry domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import E, S, Template, X, compute_weights, make_dispatcher
from .._vocab import (
    _GEO_N,
    _SCALARS,
    _VARS,
    _arccos_nm,
    _arccosh_nm,
    _arccot_nm,
    _arccsc_nm,
    _arcsec_nm,
    _arcsin_nm,
    _arcsinh_nm,
    _arctan_nm,
    _arctanh_nm,
    _cos_nm,
    _cosh_nm,
    _cot_nm,
    _csc_nm,
    _fn_rich_nosub,
    _sec_nm,
    _sin_nm,
    _sinh_nm,
    _tan_nm,
    _tanh_nm,
)

# ---------------------------------------------------------------------------
# Shared pools
# ---------------------------------------------------------------------------

_SIDES_POOL: list[str] = ["a", "b", "c", "p", "q", "r"]
_ANGLES_POOL: list[str] = ["A", "B", "C", "P", "Q", "R"]
_TRIG_ARG_POOL: tuple[str, ...] = tuple(_SCALARS) + (
    r"\theta",
    r"\phi",
    r"\varphi",
    r"\psi",
    r"\omega",
    r"\alpha",
    r"\beta",
    r"\gamma",
    r"\delta",
)

# ---------------------------------------------------------------------------
# Inline sub-generators
# ---------------------------------------------------------------------------


def _fourier_n_sub(rng: random.Random) -> str:
    return rng.choice(["2", "3", "4", "5", "6", "n", "m", "N", "M", "K", "p"])


def _trig_fn_int_sub(rng: random.Random) -> str:
    return rng.choice(
        [
            r"\sin",
            r"\cos",
            r"\tan",
            r"\sinh",
            r"\cosh",
            r"\sec",
            r"\text{sine}",
            r"\text{cosine}",
            r"\text{tangent}",
            r"\text{sh}",
            r"\text{ch}",
        ]
    )


def _sin_dbl_coeff_sub(rng: random.Random) -> str:
    a = rng.choice(_TRIG_ARG_POOL)
    return rng.choice([f"2{a}", f"3{a}", a, f"{a}^2"])


def _cos_dbl_coeff_sub(rng: random.Random) -> str:
    a = rng.choice(_TRIG_ARG_POOL)
    return rng.choice(["2", "3", a, f"2{a}"])


def _trig_pyth_arg_sub(rng: random.Random) -> str:
    a = rng.choice(_TRIG_ARG_POOL)
    x = rng.choice(_VARS)
    return rng.choice([x, f"{a} {x}", f"{a}^2 {x}"])


def _euler_arg_sub(rng: random.Random) -> str:
    a = rng.choice(_TRIG_ARG_POOL)
    b = rng.choice([s for s in _TRIG_ARG_POOL if s != a])
    x = rng.choice(_VARS)
    return rng.choice([x, f"{a} {x}", f"{a}+{b}", a])


def _hyp_arg_sub(rng: random.Random) -> str:
    a = rng.choice(_TRIG_ARG_POOL)
    b = rng.choice([s for s in _TRIG_ARG_POOL if s != a])
    x = rng.choice(_VARS)
    return rng.choice([f"{a} {x}", f"{a}+{b}", f"{b} {x}+{a}"])


def _taylor_arg_sub(rng: random.Random) -> str:
    a = rng.choice(_TRIG_ARG_POOL)
    x = rng.choice(_VARS)
    return rng.choice([x, f"{a} {x}"])


# ---------------------------------------------------------------------------
# Trigonometry templates
# ---------------------------------------------------------------------------

_TRIG_TEMPLATES: list[Template] = [
    # c=0: Pythagorean identity — bare / scalar-coeff / scalar-squared variants
    Template(
        name="pythagorean_identity",
        latex="",
        slots={},
        variants=[
            Template(
                name="pythagorean_identity_bare",
                latex=r"{sn}^2 {x} + {cn}^2 {x} = 1",
                slots={
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="pythagorean_identity_coeff",
                latex=r"{sn}^2 {a} {x} + {cn}^2 {a} {x} = 1",
                slots={
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="pythagorean_identity_coeff_sq",
                latex=r"{sn}^2 {a}^2{x} + {cn}^2 {a}^2{x} = 1",
                slots={
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
        ],
    ),
    # c=1: double-angle sine
    Template(
        name="double_angle_sin",
        latex=(
            r"{sn}({coeff} {x}) = "
            r"2 {sn}\!\left(\frac{{{coeff} {x}}}{{2}}\right) {cn}\!\left(\frac{{{coeff} {x}}}{{2}}\right)"
        ),
        slots={
            "sn": E(_sin_nm, n=2),
            "cn": E(_cos_nm, n=2),
            "coeff": E(_sin_dbl_coeff_sub, n=72),
            "x": S(tuple(_VARS), idx=0.35),
        },
    ),
    # c=2: double-angle cosine
    Template(
        name="double_angle_cos",
        latex=(
            r"{cn}({coeff} {x}) = "
            r"{cn}^2\!\left(\frac{{{coeff} {x}}}{{2}}\right) - {sn}^2\!\left(\frac{{{coeff} {x}}}{{2}}\right)"
        ),
        slots={
            "sn": E(_sin_nm, n=2),
            "cn": E(_cos_nm, n=2),
            "coeff": E(_cos_dbl_coeff_sub, n=70),
            "x": S(tuple(_VARS), idx=0.35),
        },
    ),
    # c=3: sine addition formula
    Template(
        name="sin_addition",
        latex=r"{sn}({a} \pm {b}) = {sn} {a} {cn} {b} \pm {cn} {a} {sn} {b}",
        slots={
            "sn": E(_sin_nm, n=2),
            "cn": E(_cos_nm, n=2),
            "a": S(_TRIG_ARG_POOL, idx=0.35),
            "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
        },
    ),
    # c=4: cosine addition formula
    Template(
        name="cos_addition",
        latex=r"{cn}({a} + {b}) = {cn} {a} {cn} {b} - {sn} {a} {sn} {b}",
        slots={
            "sn": E(_sin_nm, n=2),
            "cn": E(_cos_nm, n=2),
            "a": S(_TRIG_ARG_POOL, idx=0.35),
            "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
        },
    ),
    # c=5: tan/cot as ratio — fn-pair × bare/coeff variants
    Template(
        name="tan_cot_ratio",
        latex="",
        slots={},
        variants=[
            Template(
                name="tan_ratio_bare",
                latex=r"{tn}\!\left({x}\right) = \frac{{{sn}\!\left({x}\right)}}{{{cn}\!\left({x}\right)}}",
                slots={
                    "tn": E(_tan_nm, n=2),
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="tan_ratio_coeff",
                latex=r"{tn}\!\left({a} {x}\right) = \frac{{{sn}\!\left({a} {x}\right)}}{{{cn}\!\left({a} {x}\right)}}",
                slots={
                    "tn": E(_tan_nm, n=2),
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="cot_ratio_bare",
                latex=r"{ctn}\!\left({x}\right) = \frac{{{cn}\!\left({x}\right)}}{{{sn}\!\left({x}\right)}}",
                slots={
                    "ctn": E(_cot_nm, n=2),
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="cot_ratio_coeff",
                latex=r"{ctn}\!\left({a} {x}\right) = \frac{{{cn}\!\left({a} {x}\right)}}{{{sn}\!\left({a} {x}\right)}}",
                slots={
                    "ctn": E(_cot_nm, n=2),
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
        ],
    ),
    # c=6: Euler's formula — positive and negative sign variants
    Template(
        name="euler_formula",
        latex="",
        slots={},
        variants=[
            Template(
                name="euler_formula_pos",
                latex=r"e^{{i {arg}}} = {cn} {arg} + i {sn} {arg}",
                slots={
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "arg": E(_euler_arg_sub, n=800),
                },
            ),
            Template(
                name="euler_formula_neg",
                latex=r"e^{{-i {arg}}} = {cn} {arg} - i {sn} {arg}",
                slots={
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "arg": E(_euler_arg_sub, n=800),
                },
            ),
        ],
    ),
    # c=7: law of cosines
    Template(
        name="law_of_cosines",
        latex=r"{s0}^2 = {s1}^2 + {s2}^2 - 2{s1}{s2} \cos {A}",
        slots={
            "s0": S(tuple(_SIDES_POOL), idx=0.35),
            "s1": X(tuple(_SIDES_POOL), ("s0",), idx=0.35),
            "s2": X(tuple(_SIDES_POOL), ("s0", "s1"), idx=0.35),
            "A": S(tuple(_ANGLES_POOL), idx=0.35),
        },
    ),
    # c=8: law of sines
    Template(
        name="law_of_sines",
        latex=(
            r"\frac{{\sin {A0}}}{{{s0}}} = "
            r"\frac{{\sin {A1}}}{{{s1}}} = "
            r"\frac{{\sin {A2}}}{{{s2}}}"
        ),
        slots={
            "s0": S(tuple(_SIDES_POOL), idx=0.35),
            "s1": X(tuple(_SIDES_POOL), ("s0",), idx=0.35),
            "s2": X(tuple(_SIDES_POOL), ("s0", "s1"), idx=0.35),
            "A0": S(tuple(_ANGLES_POOL), idx=0.35),
            "A1": X(tuple(_ANGLES_POOL), ("A0",), idx=0.35),
            "A2": X(tuple(_ANGLES_POOL), ("A0", "A1"), idx=0.35),
        },
    ),
    # c=9: trig integral
    Template(
        name="trig_integral",
        latex=r"\int {fn}\!\left({coeff} {x}\right) \, d{x}",
        slots={
            "fn": E(_trig_fn_int_sub, n=11),
            "coeff": S(_TRIG_ARG_POOL, idx=0.35),
            "x": S(tuple(_VARS), idx=0.35),
        },
    ),
    # c=10: arctan of ratio
    Template(
        name="arctan_ratio",
        latex=r"{atn}\!\left(\frac{{{a}}}{{{b}}}\right)",
        slots={
            "atn": E(_arctan_nm, n=4),
            "a": S(_TRIG_ARG_POOL, idx=0.35),
            "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
        },
    ),
    # c=11: Pythagorean identities for tan/cot — arg via Sub
    Template(
        name="pythagorean_tan_cot",
        latex="",
        slots={},
        variants=[
            Template(
                name="pythagorean_tan",
                latex=r"1 + {tn}^2({arg}) = {secn}^2({arg})",
                slots={
                    "tn": E(_tan_nm, n=2),
                    "secn": E(_sec_nm, n=2),
                    "arg": E(_trig_pyth_arg_sub, n=900),
                },
            ),
            Template(
                name="pythagorean_cot",
                latex=r"1 + {ctn}^2({arg}) = {cscn}^2({arg})",
                slots={
                    "ctn": E(_cot_nm, n=2),
                    "cscn": E(_csc_nm, n=2),
                    "arg": E(_trig_pyth_arg_sub, n=900),
                },
            ),
        ],
    ),
    # c=12: Euler's exponential form of sin/cos
    Template(
        name="euler_sin_cos_exponential",
        latex="",
        slots={},
        variants=[
            Template(
                name="euler_exp_sin_cos_bare",
                latex=(
                    r"{sn} {a} = \frac{{e^{{i{a}}} - e^{{-i{a}}}}}{{2i}}, "
                    r"\quad {cn} {a} = \frac{{e^{{i{a}}} + e^{{-i{a}}}}}{{2}}"
                ),
                slots={
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                },
            ),
            Template(
                name="euler_exp_sin_cos_coeff",
                latex=(
                    r"{sn}({c} {x}) = \frac{{e^{{i{c} {x}}} - e^{{-i{c} {x}}}}}{{2i}}, "
                    r"\quad {cn}({c} {x}) = \frac{{e^{{i{c} {x}}} + e^{{-i{c} {x}}}}}{{2}}"
                ),
                slots={
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "c": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
        ],
    ),
    # c=13: product-to-sum (sin cos)
    Template(
        name="product_to_sum_sin_cos",
        latex=r"{sn} {a} {cn} {b} = \tfrac{{1}}{{2}}\left[{sn}({a}+{b}) + {sn}({a}-{b})\right]",
        slots={
            "sn": E(_sin_nm, n=2),
            "cn": E(_cos_nm, n=2),
            "a": S(_TRIG_ARG_POOL, idx=0.35),
            "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
        },
    ),
    # c=14: half-angle formulas — bare and coeff variants
    Template(
        name="half_angle_formulas",
        latex="",
        slots={},
        variants=[
            Template(
                name="half_angle_bare",
                latex=(
                    r"{sn}^2 \frac{{{x}}}{{2}} = \frac{{1 - {cn} {x}}}{{2}}, "
                    r"\quad {cn}^2 \frac{{{x}}}{{2}} = \frac{{1 + {cn} {x}}}{{2}}"
                ),
                slots={
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="half_angle_coeff",
                latex=(
                    r"{sn}^2 \frac{{{a} {x}}}{{2}} = \frac{{1 - {cn} {a} {x}}}{{2}}, "
                    r"\quad {cn}^2 \frac{{{a} {x}}}{{2}} = \frac{{1 + {cn} {a} {x}}}{{2}}"
                ),
                slots={
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
        ],
    ),
    # c=15: sum-to-product
    Template(
        name="sum_to_product",
        latex=(
            r"{sn} {a} + {sn} {b} = "
            r"2 {sn}\!\left(\frac{{{a}+{b}}}{{2}}\right) {cn}\!\left(\frac{{{a}-{b}}}{{2}}\right)"
        ),
        slots={
            "sn": E(_sin_nm, n=2),
            "cn": E(_cos_nm, n=2),
            "a": S(_TRIG_ARG_POOL, idx=0.35),
            "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
        },
    ),
    # c=16: hyperbolic Pythagorean identity
    Template(
        name="hyperbolic_pythagorean",
        latex=r"{chn}^2({arg}) - {shn}^2({arg}) = 1",
        slots={
            "chn": E(_cosh_nm, n=2),
            "shn": E(_sinh_nm, n=2),
            "arg": E(_hyp_arg_sub, n=1200),
        },
    ),
    # c=17: inverse trig derivatives
    Template(
        name="inverse_trig_derivative",
        latex="",
        slots={},
        variants=[
            Template(
                name="arcsin_derivative",
                latex=r"\frac{{d}}{{d{x}}}{asn}\!\left({x}\right) = \frac{{1}}{{\sqrt{{1 - {x}^2}}}}",
                slots={"asn": E(_arcsin_nm, n=4), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="arcsin_derivative_coeff",
                latex=r"\frac{{d}}{{d{x}}}{asn}\!\left({c} {x}\right) = \frac{{{c}}}{{\sqrt{{1 - {c}^2{x}^2}}}}",
                slots={"asn": E(_arcsin_nm, n=4), "c": S(_TRIG_ARG_POOL, idx=0.35), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="arccos_derivative",
                latex=r"\frac{{d}}{{d{x}}}{acn}\!\left({x}\right) = \frac{{-1}}{{\sqrt{{1 - {x}^2}}}}",
                slots={"acn": E(_arccos_nm, n=4), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="arccos_derivative_coeff",
                latex=r"\frac{{d}}{{d{x}}}{acn}\!\left({c} {x}\right) = \frac{{-{c}}}{{\sqrt{{1 - {c}^2{x}^2}}}}",
                slots={"acn": E(_arccos_nm, n=4), "c": S(_TRIG_ARG_POOL, idx=0.35), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="arctan_derivative",
                latex=r"\frac{{d}}{{d{x}}}{atn}\!\left({x}\right) = \frac{{1}}{{1 + {x}^2}}",
                slots={"atn": E(_arctan_nm, n=4), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="arctan_derivative_coeff",
                latex=r"\frac{{d}}{{d{x}}}{atn}\!\left({c} {x}\right) = \frac{{{c}}}{{1 + {c}^2{x}^2}}",
                slots={"atn": E(_arctan_nm, n=4), "c": S(_TRIG_ARG_POOL, idx=0.35), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="arccot_derivative",
                latex=r"\frac{{d}}{{d{x}}}{actn}\!\left({x}\right) = \frac{{-1}}{{1 + {x}^2}}",
                slots={"actn": E(_arccot_nm, n=3), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="arcsec_derivative",
                latex=r"\frac{{d}}{{d{x}}}{asecn}\!\left({x}\right) = \frac{{1}}{{{x}\sqrt{{{x}^2 - 1}}}}",
                slots={"asecn": E(_arcsec_nm, n=3), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="arccsc_derivative",
                latex=r"\frac{{d}}{{d{x}}}{acscn}\!\left({x}\right) = \frac{{-1}}{{{x}\sqrt{{{x}^2 - 1}}}}",
                slots={"acscn": E(_arccsc_nm, n=3), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="arctanh_derivative",
                latex=r"\frac{{d}}{{d{x}}}{athn}\!\left({x}\right) = \frac{{1}}{{1 - {x}^2}}",
                slots={"athn": E(_arctanh_nm, n=3), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="arcsinh_derivative",
                latex=r"\frac{{d}}{{d{x}}}{ashn}\!\left({x}\right) = \frac{{1}}{{\sqrt{{1 + {x}^2}}}}",
                slots={"ashn": E(_arcsinh_nm, n=3), "x": S(tuple(_VARS), idx=0.35)},
            ),
        ],
    ),
    # c=18: hyperbolic addition formula
    Template(
        name="hyperbolic_addition",
        latex=r"{shn}({a} + {b}) = {shn} {a} {chn} {b} + {chn} {a} {shn} {b}",
        slots={
            "shn": E(_sinh_nm, n=2),
            "chn": E(_cosh_nm, n=2),
            "a": S(_TRIG_ARG_POOL, idx=0.35),
            "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
        },
    ),
    # c=19: Taylor series for trig functions
    Template(
        name="taylor_series_trig",
        latex="",
        slots={},
        variants=[
            Template(
                name="taylor_sin",
                latex=r"{sn}({arg}) = \sum{lim_mod}_{{n=0}}^{{\infty}} \frac{{(-1)^n ({arg})^{{2n+1}}}}{{(2n+1)!}}",
                slots={"lim_mod": S(("", r"\limits")), "sn": E(_sin_nm, n=2), "arg": E(_taylor_arg_sub, n=450)},
            ),
            Template(
                name="taylor_cos",
                latex=r"{cn}({arg}) = \sum{lim_mod}_{{n=0}}^{{\infty}} \frac{{(-1)^n ({arg})^{{2n}}}}{{(2n)!}}",
                slots={"lim_mod": S(("", r"\limits")), "cn": E(_cos_nm, n=2), "arg": E(_taylor_arg_sub, n=450)},
            ),
            Template(
                name="taylor_tan_approx",
                latex=r"{tn} {x} \approx {x} + \frac{{{x}^3}}{{3}} + \frac{{2{x}^5}}{{15}} + \cdots",
                slots={"tn": E(_tan_nm, n=2), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="taylor_tan_approx_coeff",
                latex=r"{tn}({c} {x}) \approx {c} {x} + \frac{{({c} {x})^3}}{{3}} + \frac{{2({c} {x})^5}}{{15}} + \cdots",
                slots={
                    "tn": E(_tan_nm, n=2),
                    "c": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
        ],
    ),
    # c=20: cofunction identities
    Template(
        name="cofunction_identity",
        latex="",
        slots={},
        variants=[
            Template(
                name="cofunction_sin_cos",
                latex=r"{sn} {x} = {cn}\!\left(\frac{{\pi}}{{2}} - {x}\right)",
                slots={"sn": E(_sin_nm, n=2), "cn": E(_cos_nm, n=2), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="cofunction_tan_cot",
                latex=r"{tn} {x} = {ctn}\!\left(\frac{{\pi}}{{2}} - {x}\right)",
                slots={"tn": E(_tan_nm, n=2), "ctn": E(_cot_nm, n=2), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="cofunction_sec_csc",
                latex=r"{secn} {x} = {cscn}\!\left(\frac{{\pi}}{{2}} - {x}\right)",
                slots={"secn": E(_sec_nm, n=2), "cscn": E(_csc_nm, n=2), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="cofunction_sin_cos_coeff",
                latex=r"{sn}({a} {x}) = {cn}\!\left(\frac{{\pi}}{{2}} - {a} {x}\right)",
                slots={
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="cofunction_tan_cot_coeff",
                latex=r"{tn}({a} {x}) = {ctn}\!\left(\frac{{\pi}}{{2}} - {a} {x}\right)",
                slots={
                    "tn": E(_tan_nm, n=2),
                    "ctn": E(_cot_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="cofunction_sec_csc_coeff",
                latex=r"{secn}({a} {x}) = {cscn}\!\left(\frac{{\pi}}{{2}} - {a} {x}\right)",
                slots={
                    "secn": E(_sec_nm, n=2),
                    "cscn": E(_csc_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
        ],
    ),
    # c=21: product-to-sum (cos cos)
    Template(
        name="product_to_sum_cos_cos",
        latex=r"{cn} {a} {cn} {b} = \tfrac{{1}}{{2}}\left[{cn}({a}-{b}) + {cn}({a}+{b})\right]",
        slots={
            "cn": E(_cos_nm, n=2),
            "a": S(_TRIG_ARG_POOL, idx=0.35),
            "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
        },
    ),
    # c=22: inverse trig identities
    Template(
        name="inverse_trig_identities",
        latex="",
        slots={},
        variants=[
            Template(
                name="arcsin_arccos_complement_bare",
                latex=r"{asn}({x}) + {acn}({x}) = \frac{{\pi}}{{2}}",
                slots={
                    "asn": E(_arcsin_nm, n=4),
                    "acn": E(_arccos_nm, n=4),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="arcsin_arccos_complement_coeff",
                latex=r"{asn}({a} {x}) + {acn}({a} {x}) = \frac{{\pi}}{{2}}",
                slots={
                    "asn": E(_arcsin_nm, n=4),
                    "acn": E(_arccos_nm, n=4),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="arctan_reciprocal_bare",
                latex=r"{atn}({x}) + {atn}\!\left(\frac{{1}}{{{x}}}\right) = \frac{{\pi}}{{2}}",
                slots={"atn": E(_arctan_nm, n=4), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="arctan_reciprocal_coeff",
                latex=r"{atn}({a} {x}) + {atn}\!\left(\frac{{1}}{{{a} {x}}}\right) = \frac{{\pi}}{{2}}",
                slots={
                    "atn": E(_arctan_nm, n=4),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="arctan_addition",
                latex=r"{atn}\!\left(\frac{{{a}+{b}}}{{1 - {a} {b}}}\right) = {atn} {a} + {atn} {b}",
                slots={
                    "atn": E(_arctan_nm, n=4),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
                },
            ),
        ],
    ),
    # c=23: parametric Pythagorean identity
    Template(
        name="parametric_pythagorean",
        latex=r"{sn}^2({a} {x}) + {cn}^2({a} {x}) = 1",
        slots={
            "sn": E(_sin_nm, n=2),
            "cn": E(_cos_nm, n=2),
            "a": S(_TRIG_ARG_POOL, idx=0.35),
            "x": S(tuple(_VARS), idx=0.35),
        },
    ),
    # c=24: Fourier/Euler roots of unity summation
    Template(
        name="roots_of_unity_sum",
        latex=r"\sum{lim_mod}_{{{k}=0}}^{{{n}-1}} e^{{2\pi i {k} {x} / {n}}} = 0",
        slots={
            "lim_mod": S(("", r"\limits")),
            "k": S(("j", "k", "l", "m", "r", "s", "t")),
            "n": E(_fourier_n_sub, n=11),
            "x": S(tuple(_VARS), idx=0.35),
        },
    ),
    # ── Triple-angle formulas ─────────────────────────────────────────────────
    Template(
        name="triple_angle_formulas",
        latex="",
        slots={},
        variants=[
            Template(
                name="triple_angle_sin",
                latex=r"{sn}(3{a}) = 3{sn} {a} - 4{sn}^3 {a}",
                slots={"sn": E(_sin_nm, n=2), "a": S(_TRIG_ARG_POOL, idx=0.35)},
            ),
            Template(
                name="triple_angle_cos",
                latex=r"{cn}(3{a}) = 4{cn}^3 {a} - 3{cn} {a}",
                slots={"cn": E(_cos_nm, n=2), "a": S(_TRIG_ARG_POOL, idx=0.35)},
            ),
            Template(
                name="triple_angle_tan",
                latex=r"{tn}(3{a}) = \frac{{3{tn} {a} - {tn}^3 {a}}}{{1 - 3{tn}^2 {a}}}",
                slots={"tn": E(_tan_nm, n=2), "a": S(_TRIG_ARG_POOL, idx=0.35)},
            ),
            Template(
                name="triple_angle_sin_coeff",
                latex=r"{sn}(3{c} {x}) = 3{sn}({c} {x}) - 4{sn}^3({c} {x})",
                slots={"sn": E(_sin_nm, n=2), "c": S(_TRIG_ARG_POOL, idx=0.35), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="triple_angle_cos_coeff",
                latex=r"{cn}(3{c} {x}) = 4{cn}^3({c} {x}) - 3{cn}({c} {x})",
                slots={"cn": E(_cos_nm, n=2), "c": S(_TRIG_ARG_POOL, idx=0.35), "x": S(tuple(_VARS), idx=0.35)},
            ),
        ],
    ),
    # ── Power reduction formulas ──────────────────────────────────────────────
    Template(
        name="power_reduction",
        latex="",
        slots={},
        variants=[
            Template(
                name="power_reduction_sin2",
                latex=r"{sn}^2 {a} = \frac{{1 - {cn}(2{a})}}{{2}}",
                slots={"sn": E(_sin_nm, n=2), "cn": E(_cos_nm, n=2), "a": S(_TRIG_ARG_POOL, idx=0.35)},
            ),
            Template(
                name="power_reduction_cos2",
                latex=r"{cn}^2 {a} = \frac{{1 + {cn}(2{a})}}{{2}}",
                slots={"cn": E(_cos_nm, n=2), "a": S(_TRIG_ARG_POOL, idx=0.35)},
            ),
            Template(
                name="power_reduction_sincos",
                latex=r"{sn} {a} {cn} {a} = \frac{{{sn}(2{a})}}{{2}}",
                slots={"sn": E(_sin_nm, n=2), "cn": E(_cos_nm, n=2), "a": S(_TRIG_ARG_POOL, idx=0.35)},
            ),
            Template(
                name="power_reduction_sin3",
                latex=r"{sn}^3 {a} = \frac{{3{sn} {a} - {sn}(3{a})}}{{4}}",
                slots={"sn": E(_sin_nm, n=2), "a": S(_TRIG_ARG_POOL, idx=0.35)},
            ),
            Template(
                name="power_reduction_cos3",
                latex=r"{cn}^3 {a} = \frac{{3{cn} {a} + {cn}(3{a})}}{{4}}",
                slots={"cn": E(_cos_nm, n=2), "a": S(_TRIG_ARG_POOL, idx=0.35)},
            ),
            Template(
                name="power_reduction_sin2_coeff",
                latex=r"{sn}^2({c} {x}) = \frac{{1 - {cn}(2{c} {x})}}{{2}}",
                slots={
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "c": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="power_reduction_cos2_coeff",
                latex=r"{cn}^2({c} {x}) = \frac{{1 + {cn}(2{c} {x})}}{{2}}",
                slots={"cn": E(_cos_nm, n=2), "c": S(_TRIG_ARG_POOL, idx=0.35), "x": S(tuple(_VARS), idx=0.35)},
            ),
        ],
    ),
    # ── Tangent addition and double-angle ─────────────────────────────────────
    Template(
        name="tangent_addition",
        latex="",
        slots={},
        variants=[
            Template(
                name="tan_addition_plus",
                latex=r"{tn}({a} + {b}) = \frac{{{tn} {a} + {tn} {b}}}{{1 - {tn} {a} {tn} {b}}}",
                slots={
                    "tn": E(_tan_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
                },
            ),
            Template(
                name="tan_addition_minus",
                latex=r"{tn}({a} - {b}) = \frac{{{tn} {a} - {tn} {b}}}{{1 + {tn} {a} {tn} {b}}}",
                slots={
                    "tn": E(_tan_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
                },
            ),
            Template(
                name="double_angle_tan",
                latex=r"{tn}(2{a}) = \frac{{2{tn} {a}}}{{1 - {tn}^2 {a}}}",
                slots={"tn": E(_tan_nm, n=2), "a": S(_TRIG_ARG_POOL, idx=0.35)},
            ),
        ],
    ),
    # ── Product-to-sum (sin sin) and sum-to-product (cosines) ─────────────────
    Template(
        name="product_sum_sin_sin",
        latex=r"{sn} {a} {sn} {b} = \tfrac{{1}}{{2}}\left[{cn}({a}-{b}) - {cn}({a}+{b})\right]",
        slots={
            "sn": E(_sin_nm, n=2),
            "cn": E(_cos_nm, n=2),
            "a": S(_TRIG_ARG_POOL, idx=0.35),
            "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
        },
    ),
    Template(
        name="sum_to_product_cos",
        latex="",
        slots={},
        variants=[
            Template(
                name="sum_to_product_cos_plus",
                latex=(
                    r"{cn} {a} + {cn} {b} = "
                    r"2{cn}\!\left(\frac{{{a}+{b}}}{{2}}\right){cn}\!\left(\frac{{{a}-{b}}}{{2}}\right)"
                ),
                slots={
                    "cn": E(_cos_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
                },
            ),
            Template(
                name="sum_to_product_cos_minus",
                latex=(
                    r"{cn} {a} - {cn} {b} = "
                    r"-2{sn}\!\left(\frac{{{a}+{b}}}{{2}}\right){sn}\!\left(\frac{{{a}-{b}}}{{2}}\right)"
                ),
                slots={
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
                },
            ),
            Template(
                name="sum_to_product_sin_minus",
                latex=(
                    r"{sn} {a} - {sn} {b} = "
                    r"2{cn}\!\left(\frac{{{a}+{b}}}{{2}}\right){sn}\!\left(\frac{{{a}-{b}}}{{2}}\right)"
                ),
                slots={
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
                },
            ),
        ],
    ),
    # ── Hyperbolic definitions and identities ─────────────────────────────────
    Template(
        name="hyperbolic_definitions",
        latex="",
        slots={},
        variants=[
            Template(
                name="sinh_definition",
                latex=r"{shn} {x} = \frac{{e^{{{x}}} - e^{{-{x}}}}}{{2}}",
                slots={"shn": E(_sinh_nm, n=2), "x": S(_TRIG_ARG_POOL, idx=0.35)},
            ),
            Template(
                name="cosh_definition",
                latex=r"{chn} {x} = \frac{{e^{{{x}}} + e^{{-{x}}}}}{{2}}",
                slots={"chn": E(_cosh_nm, n=2), "x": S(_TRIG_ARG_POOL, idx=0.35)},
            ),
            Template(
                name="tanh_definition",
                latex=r"{thn} {x} = \frac{{e^{{{x}}} - e^{{-{x}}}}}{{e^{{{x}}} + e^{{-{x}}}}}",
                slots={"thn": E(_tanh_nm, n=2), "x": S(_TRIG_ARG_POOL, idx=0.35)},
            ),
            Template(
                name="tanh_ratio",
                latex=r"{thn} {x} = \frac{{{shn} {x}}}{{{chn} {x}}}",
                slots={
                    "thn": E(_tanh_nm, n=2),
                    "shn": E(_sinh_nm, n=2),
                    "chn": E(_cosh_nm, n=2),
                    "x": S(_TRIG_ARG_POOL, idx=0.35),
                },
            ),
            Template(
                name="sinh_definition_coeff",
                latex=r"{shn}({c} {x}) = \frac{{e^{{{c} {x}}} - e^{{-{c} {x}}}}}{{2}}",
                slots={
                    "shn": E(_sinh_nm, n=2),
                    "c": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="cosh_definition_coeff",
                latex=r"{chn}({c} {x}) = \frac{{e^{{{c} {x}}} + e^{{-{c} {x}}}}}{{2}}",
                slots={
                    "chn": E(_cosh_nm, n=2),
                    "c": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
        ],
    ),
    Template(
        name="hyperbolic_cosh_addition",
        latex=r"{chn}({a} + {b}) = {chn} {a} {chn} {b} + {shn} {a} {shn} {b}",
        slots={
            "chn": E(_cosh_nm, n=2),
            "shn": E(_sinh_nm, n=2),
            "a": S(_TRIG_ARG_POOL, idx=0.35),
            "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
        },
    ),
    # ── Sine subtraction formula ──────────────────────────────────────────────
    Template(
        name="sin_subtraction",
        latex=r"{sn}({a} - {b}) = {sn} {a} {cn} {b} - {cn} {a} {sn} {b}",
        slots={
            "sn": E(_sin_nm, n=2),
            "cn": E(_cos_nm, n=2),
            "a": S(_TRIG_ARG_POOL, idx=0.35),
            "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
        },
    ),
    # ── Cosine subtraction formula ────────────────────────────────────────────
    Template(
        name="cos_subtraction",
        latex=r"{cn}({a} - {b}) = {cn} {a} {cn} {b} + {sn} {a} {sn} {b}",
        slots={
            "sn": E(_sin_nm, n=2),
            "cn": E(_cos_nm, n=2),
            "a": S(_TRIG_ARG_POOL, idx=0.35),
            "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
        },
    ),
    # ── Negative angle (odd/even) identities ──────────────────────────────────
    Template(
        name="negative_angle",
        latex="",
        slots={},
        variants=[
            Template(
                name="negative_angle_sin",
                latex=r"{sn}(-{x}) = -{sn} {x}",
                slots={"sn": E(_sin_nm, n=2), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="negative_angle_cos",
                latex=r"{cn}(-{x}) = {cn} {x}",
                slots={"cn": E(_cos_nm, n=2), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="negative_angle_tan",
                latex=r"{tn}(-{x}) = -{tn} {x}",
                slots={"tn": E(_tan_nm, n=2), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="negative_angle_sin_coeff",
                latex=r"{sn}(-{a} {x}) = -{sn}({a} {x})",
                slots={
                    "sn": E(_sin_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="negative_angle_cos_coeff",
                latex=r"{cn}(-{a} {x}) = {cn}({a} {x})",
                slots={
                    "cn": E(_cos_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="negative_angle_tan_coeff",
                latex=r"{tn}(-{a} {x}) = -{tn}({a} {x})",
                slots={
                    "tn": E(_tan_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
        ],
    ),
    # ── De Moivre's theorem ───────────────────────────────────────────────────
    Template(
        name="de_moivre",
        latex="",
        slots={},
        variants=[
            Template(
                name="de_moivre_bare",
                latex=(
                    r"\left({cn} {a} + i {sn} {a}\right)^{{{n}}} "
                    r"= {cn}({n} {a}) + i {sn}({n} {a})"
                ),
                slots={
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "n": S(_GEO_N),
                },
            ),
            Template(
                name="de_moivre_coeff",
                latex=(
                    r"\left({cn}({a} {x}) + i {sn}({a} {x})\right)^{{{n}}} "
                    r"= {cn}({n} {a} {x}) + i {sn}({n} {a} {x})"
                ),
                slots={
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                    "n": S(_GEO_N),
                },
            ),
        ],
    ),
    # ── Area of triangle ──────────────────────────────────────────────────────
    Template(
        name="triangle_area",
        latex="",
        slots={},
        variants=[
            Template(
                name="triangle_area_text",
                latex=r"\text{{Area}} = \tfrac{{1}}{{2}} {s0} \, {s1} \, {sn} {A}",
                slots={
                    "sn": E(_sin_nm, n=2),
                    "s0": S(tuple(_SIDES_POOL), idx=0.35),
                    "s1": X(tuple(_SIDES_POOL), ("s0",), idx=0.35),
                    "A": S(tuple(_ANGLES_POOL), idx=0.35),
                },
            ),
            Template(
                name="triangle_area_S",
                latex=r"S = \tfrac{{1}}{{2}} {s0} \, {s1} \, {sn} {A}",
                slots={
                    "sn": E(_sin_nm, n=2),
                    "s0": S(tuple(_SIDES_POOL), idx=0.35),
                    "s1": X(tuple(_SIDES_POOL), ("s0",), idx=0.35),
                    "A": S(tuple(_ANGLES_POOL), idx=0.35),
                },
            ),
        ],
    ),
    # ── Extended law of sines ─────────────────────────────────────────────────
    Template(
        name="law_of_sines_extended",
        latex=(
            r"\frac{{{s0}}}{{\sin {A0}}} = \frac{{{s1}}}{{\sin {A1}}} "
            r"= \frac{{{s2}}}{{\sin {A2}}} = 2R"
        ),
        slots={
            "s0": S(tuple(_SIDES_POOL), idx=0.35),
            "s1": X(tuple(_SIDES_POOL), ("s0",), idx=0.35),
            "s2": X(tuple(_SIDES_POOL), ("s0", "s1"), idx=0.35),
            "A0": S(tuple(_ANGLES_POOL), idx=0.35),
            "A1": X(tuple(_ANGLES_POOL), ("A0",), idx=0.35),
            "A2": X(tuple(_ANGLES_POOL), ("A0", "A1"), idx=0.35),
        },
    ),
    Template(
        name="law_of_sines_2R_ratio",
        latex=r"\frac{{{s0}}}{{\sin {A0}}} = 2R",
        slots={
            "s0": S(tuple(_SIDES_POOL), idx=0.35),
            "A0": S(tuple(_ANGLES_POOL), idx=0.35),
        },
    ),
    # ── Hyperbolic subtraction ────────────────────────────────────────────────
    Template(
        name="hyperbolic_subtraction",
        latex="",
        slots={},
        variants=[
            Template(
                name="hyperbolic_sinh_subtraction",
                latex=r"{shn}({a} - {b}) = {shn} {a} {chn} {b} - {chn} {a} {shn} {b}",
                slots={
                    "shn": E(_sinh_nm, n=2),
                    "chn": E(_cosh_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
                },
            ),
            Template(
                name="hyperbolic_cosh_subtraction",
                latex=r"{chn}({a} - {b}) = {chn} {a} {chn} {b} - {shn} {a} {shn} {b}",
                slots={
                    "chn": E(_cosh_nm, n=2),
                    "shn": E(_sinh_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
                },
            ),
        ],
    ),
    # ── Hyperbolic double-angle ───────────────────────────────────────────────
    Template(
        name="hyperbolic_double_angle",
        latex="",
        slots={},
        variants=[
            Template(
                name="hyperbolic_double_angle_sinh",
                latex=r"{shn}(2{a}) = 2 {shn} {a} {chn} {a}",
                slots={
                    "shn": E(_sinh_nm, n=2),
                    "chn": E(_cosh_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                },
            ),
            Template(
                name="hyperbolic_double_angle_cosh",
                latex=r"{chn}(2{a}) = {chn}^2 {a} + {shn}^2 {a}",
                slots={
                    "chn": E(_cosh_nm, n=2),
                    "shn": E(_sinh_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                },
            ),
            Template(
                name="hyperbolic_double_angle_sinh_coeff",
                latex=r"{shn}(2{c} {x}) = 2 {shn}({c} {x}) {chn}({c} {x})",
                slots={
                    "shn": E(_sinh_nm, n=2),
                    "chn": E(_cosh_nm, n=2),
                    "c": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="hyperbolic_double_angle_cosh_coeff",
                latex=r"{chn}(2{c} {x}) = {chn}^2({c} {x}) + {shn}^2({c} {x})",
                slots={
                    "chn": E(_cosh_nm, n=2),
                    "shn": E(_sinh_nm, n=2),
                    "c": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
        ],
    ),
    # ── Inverse hyperbolic in log form ────────────────────────────────────────
    Template(
        name="inverse_hyperbolic_log",
        latex="",
        slots={},
        variants=[
            Template(
                name="arcsinh_log",
                latex=r"{ashn}({x}) = \ln\!\left({x} + \sqrt{{{x}^2 + 1}}\right)",
                slots={"ashn": E(_arcsinh_nm, n=3), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="arccosh_log",
                latex=r"{achn}({x}) = \ln\!\left({x} + \sqrt{{{x}^2 - 1}}\right)",
                slots={"achn": E(_arccosh_nm, n=3), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="arctanh_log",
                latex=r"{athn}({x}) = \tfrac{{1}}{{2}} \ln \frac{{1 + {x}}}{{1 - {x}}}",
                slots={"athn": E(_arctanh_nm, n=3), "x": S(tuple(_VARS), idx=0.35)},
            ),
            Template(
                name="arcsinh_log_coeff",
                latex=r"{ashn}({c} {x}) = \ln\!\left({c} {x} + \sqrt{{{c}^2 {x}^2 + 1}}\right)",
                slots={
                    "ashn": E(_arcsinh_nm, n=3),
                    "c": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="arccosh_log_coeff",
                latex=r"{achn}({c} {x}) = \ln\!\left({c} {x} + \sqrt{{{c}^2 {x}^2 - 1}}\right)",
                slots={
                    "achn": E(_arccosh_nm, n=3),
                    "c": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="arctanh_log_coeff",
                latex=r"{athn}({c} {x}) = \tfrac{{1}}{{2}} \ln \frac{{1 + {c} {x}}}{{1 - {c} {x}}}",
                slots={
                    "athn": E(_arctanh_nm, n=3),
                    "c": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
        ],
    ),
    # ── Arctan difference ─────────────────────────────────────────────────────
    Template(
        name="arctan_difference",
        latex=r"{atn} {a} - {atn} {b} = {atn}\!\left(\frac{{{a} - {b}}}{{1 + {a} {b}}}\right)",
        slots={
            "atn": E(_arctan_nm, n=4),
            "a": S(_TRIG_ARG_POOL, idx=0.35),
            "b": X(_TRIG_ARG_POOL, ("a",), idx=0.35),
        },
    ),
    # ── Reciprocal identities (sec, csc) ──────────────────────────────────────
    Template(
        name="reciprocal_identities",
        latex="",
        slots={},
        variants=[
            Template(
                name="sec_reciprocal_bare",
                latex=r"{secn}\!\left({x}\right) = \frac{{1}}{{{cn}\!\left({x}\right)}}",
                slots={
                    "secn": E(_sec_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="csc_reciprocal_bare",
                latex=r"{cscn}\!\left({x}\right) = \frac{{1}}{{{sn}\!\left({x}\right)}}",
                slots={
                    "cscn": E(_csc_nm, n=2),
                    "sn": E(_sin_nm, n=2),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="sec_reciprocal_coeff",
                latex=r"{secn}\!\left({a} {x}\right) = \frac{{1}}{{{cn}\!\left({a} {x}\right)}}",
                slots={
                    "secn": E(_sec_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="csc_reciprocal_coeff",
                latex=r"{cscn}\!\left({a} {x}\right) = \frac{{1}}{{{sn}\!\left({a} {x}\right)}}",
                slots={
                    "cscn": E(_csc_nm, n=2),
                    "sn": E(_sin_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
        ],
    ),
    # ── Law of Tangents ───────────────────────────────────────────────────────
    Template(
        name="law_of_tangents",
        latex=(
            r"\frac{{{s0} - {s1}}}{{{s0} + {s1}}} = "
            r"\frac{{{tn}\!\left(\frac{{{A0} - {A1}}}{{2}}\right)}}"
            r"{{{tn}\!\left(\frac{{{A0} + {A1}}}{{2}}\right)}}"
        ),
        slots={
            "tn": E(_tan_nm, n=2),
            "s0": S(tuple(_SIDES_POOL), idx=0.35),
            "s1": X(tuple(_SIDES_POOL), ("s0",), idx=0.35),
            "A0": S(tuple(_ANGLES_POOL), idx=0.35),
            "A1": X(tuple(_ANGLES_POOL), ("A0",), idx=0.35),
        },
    ),
    # ── Weierstrass substitution (t = tan(x/2)) ───────────────────────────────
    Template(
        name="weierstrass_substitution",
        latex="",
        slots={},
        variants=[
            Template(
                name="weierstrass_def",
                latex=r"{t} = {tn}\!\frac{{{x}}}{{2}}",
                slots={
                    "tn": E(_tan_nm, n=2),
                    "x": S(tuple(_VARS), idx=0.35),
                    "t": X(tuple(_VARS), ("x",), idx=0.35),
                },
            ),
            Template(
                name="weierstrass_sin",
                latex=r"{sn} {x} = \frac{{2{t}}}{{1 + {t}^2}}",
                slots={
                    "sn": E(_sin_nm, n=2),
                    "x": S(tuple(_VARS), idx=0.35),
                    "t": X(tuple(_VARS), ("x",), idx=0.35),
                },
            ),
            Template(
                name="weierstrass_cos",
                latex=r"{cn} {x} = \frac{{1 - {t}^2}}{{1 + {t}^2}}",
                slots={
                    "cn": E(_cos_nm, n=2),
                    "x": S(tuple(_VARS), idx=0.35),
                    "t": X(tuple(_VARS), ("x",), idx=0.35),
                },
            ),
            Template(
                name="weierstrass_tan",
                latex=r"{tn} {x} = \frac{{2{t}}}{{1 - {t}^2}}",
                slots={
                    "tn": E(_tan_nm, n=2),
                    "x": S(tuple(_VARS), idx=0.35),
                    "t": X(tuple(_VARS), ("x",), idx=0.35),
                },
            ),
            Template(
                name="weierstrass_dx",
                latex=r"d{x} = \frac{{2}}{{1 + {t}^2}} \, d{t}",
                slots={
                    "x": S(tuple(_VARS), idx=0.35),
                    "t": X(tuple(_VARS), ("x",), idx=0.35),
                },
            ),
            Template(
                name="weierstrass_combined",
                latex=(
                    r"{t} = {tn}\!\frac{{{x}}}{{2}}, \quad "
                    r"{sn} {x} = \frac{{2{t}}}{{1 + {t}^2}}, \quad "
                    r"{cn} {x} = \frac{{1 - {t}^2}}{{1 + {t}^2}}"
                ),
                slots={
                    "tn": E(_tan_nm, n=2),
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "x": S(tuple(_VARS), idx=0.35),
                    "t": X(tuple(_VARS), ("x",), idx=0.35),
                },
            ),
        ],
    ),
    # ── Trig antiderivatives ──────────────────────────────────────────────────
    Template(
        name="trig_antiderivative",
        latex="",
        slots={},
        variants=[
            Template(
                name="antideriv_sin_bare",
                latex=r"\int {sn} {x}\,d{x} = -{cn} {x} + C",
                slots={
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="antideriv_cos_bare",
                latex=r"\int {cn} {x}\,d{x} = {sn} {x} + C",
                slots={
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="antideriv_sin_coeff",
                latex=r"\int {sn}({a} {x})\,d{x} = -\frac{{{cn}({a} {x})}}{{{a}}} + C",
                slots={
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="antideriv_cos_coeff",
                latex=r"\int {cn}({a} {x})\,d{x} = \frac{{{sn}({a} {x})}}{{{a}}} + C",
                slots={
                    "sn": E(_sin_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="antideriv_tan_cos",
                latex=r"\int {tn}({a} {x})\,d{x} = -\frac{{\ln\left|{cn}({a} {x})\right|}}{{{a}}} + C",
                slots={
                    "tn": E(_tan_nm, n=2),
                    "cn": E(_cos_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="antideriv_tan_sec",
                latex=r"\int {tn}({a} {x})\,d{x} = \frac{{\ln\left|{secn}({a} {x})\right|}}{{{a}}} + C",
                slots={
                    "tn": E(_tan_nm, n=2),
                    "secn": E(_sec_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="antideriv_sec2_coeff",
                latex=r"\int {secn}^2({a} {x})\,d{x} = \frac{{{tn}({a} {x})}}{{{a}}} + C",
                slots={
                    "secn": E(_sec_nm, n=2),
                    "tn": E(_tan_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="antideriv_csc2_coeff",
                latex=r"\int {cscn}^2({a} {x})\,d{x} = -\frac{{{ctn}({a} {x})}}{{{a}}} + C",
                slots={
                    "cscn": E(_csc_nm, n=2),
                    "ctn": E(_cot_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
            Template(
                name="antideriv_sec_coeff",
                latex=(
                    r"\int {secn}({a} {x})\,d{x} = "
                    r"\frac{{\ln\left|{secn}({a} {x}) + {tn}({a} {x})\right|}}{{{a}}} + C"
                ),
                slots={
                    "secn": E(_sec_nm, n=2),
                    "tn": E(_tan_nm, n=2),
                    "a": S(_TRIG_ARG_POOL, idx=0.35),
                    "x": S(tuple(_VARS), idx=0.35),
                },
            ),
        ],
    ),
    # ── Fourier series ────────────────────────────────────────────────────────
    Template(
        name="fourier_series",
        latex="",
        slots={},
        variants=[
            Template(
                name="fourier_series_full",
                latex=(
                    r"{fn}({v}) = \frac{{{ca}_0}}{{2}} + "
                    r"\sum{lim_mod}_{{k=1}}^{{\infty}}\!\left("
                    r"{ca}_k \cos\frac{{k\pi {v}}}{{{L}}} "
                    r"+ {cb}_k \sin\frac{{k\pi {v}}}{{{L}}}\right)"
                ),
                slots={
                    "lim_mod": S(("", r"\limits")),
                    "fn": E(_fn_rich_nosub, n=100),
                    "v": S(tuple(_VARS), idx=0.35),
                    "ca": S(tuple(_SCALARS)),
                    "cb": X(tuple(_SCALARS), ("ca",)),
                    "L": X(tuple(_SCALARS), ("ca", "cb")),
                },
            ),
            Template(
                name="fourier_coeff_an",
                latex=(
                    r"{ca}_n = \frac{{2}}{{{L}}} "
                    r"\int_0^{{{L}}} {fn}({v}) \cos\frac{{n\pi {v}}}{{{L}}} \, d{v}"
                ),
                slots={
                    "fn": E(_fn_rich_nosub, n=100),
                    "v": S(tuple(_VARS), idx=0.35),
                    "ca": S(tuple(_SCALARS)),
                    "L": X(tuple(_SCALARS), ("ca",)),
                },
            ),
            Template(
                name="fourier_coeff_bn",
                latex=(
                    r"{cb}_n = \frac{{2}}{{{L}}} "
                    r"\int_0^{{{L}}} {fn}({v}) \sin\frac{{n\pi {v}}}{{{L}}} \, d{v}"
                ),
                slots={
                    "fn": E(_fn_rich_nosub, n=100),
                    "v": S(tuple(_VARS), idx=0.35),
                    "cb": S(tuple(_SCALARS)),
                    "L": X(tuple(_SCALARS), ("cb",)),
                },
            ),
        ],
    ),
]

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

# cap=75M: prevents _expr-heavy branches (n_eff~10^14) from crowding out named identities
_W_TRIG: list[float] = compute_weights(_TRIG_TEMPLATES, cap=75_000_000)

# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

_trigonometry = make_dispatcher(_TRIG_TEMPLATES, _W_TRIG)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "trigonometry": _trigonometry,
}

WEIGHTS: dict[str, float] = {
    "trigonometry": 0.04,
}

TEMPLATES: dict[str, list[Template]] = {
    "trigonometry": _TRIG_TEMPLATES,
}
