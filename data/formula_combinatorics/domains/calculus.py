"""Calculus domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import E, S, Template, X, compute_weights, make_dispatcher
from .._templates import _def_integral, _indef_integral, _mixed_partial
from .._vocab import _VARS, _atom, _expr, _fn_rich, _fn_rich_nosub, _s

# ---------------------------------------------------------------------------
# Shared pools
# ---------------------------------------------------------------------------

_FUNC_Z_POOL: tuple[str, ...] = (r"\phi", r"\psi", r"\eta")
_VEC_FIELD_POOL: tuple[str, ...] = ("F", "G", "v")
_VOL_DOMAIN_POOL: tuple[str, ...] = ("V", r"\Omega", "D")
_SURFACE_POOL: tuple[str, ...] = ("S", r"\Sigma")
_CURVE_POOL: tuple[str, ...] = ("C", r"\partial S", r"\partial \Sigma")
_MFLD_POOL: tuple[str, ...] = ("M", r"\Omega", r"\Sigma")

# ---------------------------------------------------------------------------
# Inline sub-generators
# ---------------------------------------------------------------------------


def _order_n(rng: random.Random) -> str:
    return rng.choice(["2", "3", "n"])


def _pt_scalar_inf_zero(rng: random.Random) -> str:
    return rng.choice([_s(rng), r"\infty", "0"])


def _pt_inf_zero_scalar(rng: random.Random) -> str:
    return rng.choice([r"\infty", "0", _s(rng)])


# ---------------------------------------------------------------------------
# Calculus templates
# ---------------------------------------------------------------------------

_CALCULUS_TEMPLATES: list[Template] = [
    # c=0 — first derivative
    Template(
        name="first_derivative",
        latex=r"\frac{{d}}{{d{v}}}\left[{expr}\right]",
        slots={
            "v": S(_VARS),
            "expr": E(_expr, n=5000),
        },
    ),
    # c=1 — nth-order derivative
    Template(
        name="nth_derivative",
        latex=r"\frac{{d^{{{n}}}}}{{d{v}^{{{n}}}}}\left[{expr}\right]",
        slots={
            "v": S(_VARS),
            "n": E(_order_n, n=3),
            "expr": E(_expr, n=5000),
        },
    ),
    # c=2 — indefinite integral
    Template(
        name="indef_integral",
        latex=r"{result}",
        slots={"result": E(_indef_integral, n=5000)},
    ),
    # c=3 — definite integral
    Template(
        name="def_integral",
        latex=r"{result}",
        slots={"result": E(_def_integral, n=5000)},
    ),
    # c=4 — double integral over D
    Template(
        name="double_integral_domain",
        latex=r"\iint_{{\mathcal{{D}}}} {expr} \, d{v} \, d{v2}",
        slots={
            "v": S(_VARS),
            "v2": X(_VARS, ("v",)),
            "expr": E(_expr, n=5000),
        },
    ),
    # c=5 — limit to scalar/inf/0
    Template(
        name="limit_simple",
        latex=r"\lim_{{{v} \to {pt}}} {expr}",
        slots={
            "v": S(_VARS),
            "pt": E(_pt_scalar_inf_zero, n=11),
            "expr": E(_expr, n=5000),
        },
    ),
    # c=6 — limit of ratio
    Template(
        name="limit_ratio",
        latex=r"\lim_{{{v} \to {pt}}} \frac{{{num}}}{{{den}}}",
        slots={
            "v": S(_VARS),
            "pt": E(_pt_inf_zero_scalar, n=11),
            "num": E(_expr, n=5000),
            "den": E(_expr, n=5000),
        },
    ),
    # c=7 — mixed partial derivative
    Template(
        name="mixed_partial",
        latex=r"{result}",
        slots={"result": E(_mixed_partial, n=500)},
    ),
    # c=8 — Taylor series
    Template(
        name="taylor_series",
        latex=r"\sum_{{n=0}}^{{\infty}} \frac{{{f}^{{(n)}}({a})}}{{n!}} \left({v} - {a}\right)^n",
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "v": S(_VARS),
            "a": E(_atom, n=150),
        },
    ),
    # — Maclaurin series (Taylor at a=0)
    Template(
        name="maclaurin_series",
        latex=r"\sum_{{n=0}}^{{\infty}} \frac{{{f}^{{(n)}}(0)}}{{n!}} {v}^n",
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "v": S(_VARS),
        },
    ),
    # — Taylor polynomial (finite-terms form)
    Template(
        name="taylor_polynomial",
        latex=(
            r"{f}({a}) + {f}'({a})({v} - {a}) + "
            r"\frac{{{f}''({a})}}{{2!}}({v} - {a})^2 + \cdots"
        ),
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "v": S(_VARS),
            "a": E(_atom, n=150),
        },
    ),
    # c=9 — fundamental theorem of calculus
    Template(
        name="ftc",
        latex=r"\int_{{{a}}}^{{{b}}} {f}'({v}) \, d{v} = {f}({b}) - {f}({a})",
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "v": S(_VARS),
            "a": E(_atom, n=150),
            "b": E(_atom, n=150),
        },
    ),
    # c=10 — chain rule
    Template(
        name="chain_rule",
        latex=r"\frac{{d}}{{d{v}}}\left[{f}\!\left({g2}({v})\right)\right] = {f}'\!\left({g2}({v})\right) {g2}'({v})",
        slots={
            "v": S(_VARS),
            "f": E(_fn_rich_nosub, n=100),
            "g2": E(_fn_rich_nosub, n=100),
        },
    ),
    # — chain rule (Leibniz notation)
    Template(
        name="chain_rule_leibniz",
        latex=r"\frac{{d{z}}}{{d{v}}} = \frac{{d{z}}}{{d{u}}} \cdot \frac{{d{u}}}{{d{v}}}",
        slots={
            "z": S(_FUNC_Z_POOL),
            "v": S(_VARS),
            "u": X(_VARS, ("v",)),
        },
    ),
    # c=11 — gradient (2D)
    Template(
        name="gradient_2d",
        latex=(
            r"\nabla {f} = "
            r"\frac{{\partial {f}}}{{\partial {v}}} \mathbf{{e}}_1 + "
            r"\frac{{\partial {f}}}{{\partial {v2}}} \mathbf{{e}}_2"
        ),
        slots={
            "f": E(_fn_rich, n=272),
            "v": S(_VARS),
            "v2": X(_VARS, ("v",)),
        },
    ),
    # — gradient (2D, ij hat notation)
    Template(
        name="gradient_2d_ij",
        latex=(
            r"\nabla {f} = "
            r"\frac{{\partial {f}}}{{\partial {v}}} \hat{{i}} + "
            r"\frac{{\partial {f}}}{{\partial {v2}}} \hat{{j}}"
        ),
        slots={
            "f": E(_fn_rich, n=272),
            "v": S(_VARS),
            "v2": X(_VARS, ("v",)),
        },
    ),
    # — gradient (3D)
    Template(
        name="gradient_3d",
        latex=(
            r"\nabla {f} = "
            r"\frac{{\partial {f}}}{{\partial {v}}} \mathbf{{e}}_1 + "
            r"\frac{{\partial {f}}}{{\partial {v2}}} \mathbf{{e}}_2 + "
            r"\frac{{\partial {f}}}{{\partial {v3}}} \mathbf{{e}}_3"
        ),
        slots={
            "f": E(_fn_rich, n=272),
            "v": S(_VARS),
            "v2": X(_VARS, ("v",)),
            "v3": X(_VARS, ("v", "v2")),
        },
    ),
    # c=12 — divergence theorem
    Template(
        name="divergence_theorem",
        latex=(
            r"\iint_{{\partial {vol}}} \mathbf{{{fld}}} \cdot d\mathbf{{S}} = "
            r"\iiint_{{{vol}}} \nabla \cdot \mathbf{{{fld}}} \, dV"
        ),
        slots={
            "fld": S(_VEC_FIELD_POOL),
            "vol": S(_VOL_DOMAIN_POOL),
        },
    ),
    # c=13 — Laplacian (2D)
    Template(
        name="laplacian_2d",
        latex=(
            r"\nabla^2 {f} = "
            r"\frac{{\partial^2 {f}}}{{\partial {v}^2}} + "
            r"\frac{{\partial^2 {f}}}{{\partial {v2}^2}}"
        ),
        slots={
            "f": E(_fn_rich, n=272),
            "v": S(_VARS),
            "v2": X(_VARS, ("v",)),
        },
    ),
    # — Laplacian (2D, delta notation)
    Template(
        name="laplacian_delta",
        latex=(
            r"\Delta {f} = "
            r"\frac{{\partial^2 {f}}}{{\partial {v}^2}} + "
            r"\frac{{\partial^2 {f}}}{{\partial {v2}^2}}"
        ),
        slots={
            "f": E(_fn_rich, n=272),
            "v": S(_VARS),
            "v2": X(_VARS, ("v",)),
        },
    ),
    # — Laplacian (3D)
    Template(
        name="laplacian_3d",
        latex=(
            r"\nabla^2 {f} = "
            r"\frac{{\partial^2 {f}}}{{\partial {v}^2}} + "
            r"\frac{{\partial^2 {f}}}{{\partial {v2}^2}} + "
            r"\frac{{\partial^2 {f}}}{{\partial {v3}^2}}"
        ),
        slots={
            "f": E(_fn_rich, n=272),
            "v": S(_VARS),
            "v2": X(_VARS, ("v",)),
            "v3": X(_VARS, ("v", "v2")),
        },
    ),
    # c=14 — Stokes' theorem
    Template(
        name="stokes_theorem",
        latex=(
            r"\oint_{{{crv}}} \mathbf{{{fld}}} \cdot d\mathbf{{r}} = "
            r"\iint_{{{srf}}} \left(\nabla \times \mathbf{{{fld}}}\right) \cdot d\mathbf{{S}}"
        ),
        slots={
            "fld": S(_VEC_FIELD_POOL),
            "srf": S(_SURFACE_POOL),
            "crv": S(_CURVE_POOL),
        },
    ),
    # c=15 — Green's theorem (flux form)
    Template(
        name="greens_theorem_flux",
        latex=(
            r"\oint_{{{crv}}} \mathbf{{{fld}}} \cdot \hat{{n}} \, ds = "
            r"\iint_{{{dom}}} \nabla \cdot \mathbf{{{fld}}} \, dA"
        ),
        slots={
            "fld": S(_VEC_FIELD_POOL),
            "crv": S(("C", r"\partial D", r"\partial R")),
            "dom": S(("D", r"\Omega", "R")),
        },
    ),
    # c=16 — Stokes' theorem (differential forms)
    Template(
        name="stokes_differential_forms",
        latex=r"\int_{{\partial {mfld}}} \omega = \int_{{{mfld}}} d\omega",
        slots={
            "mfld": S(_MFLD_POOL),
        },
    ),
    # c=15 (orig) — iterated integral
    Template(
        name="iterated_integral",
        latex=(
            r"\int_{{{a}}}^{{{b}}} \int_{{g_1({v})}}^{{g_2({v})}} "
            r"{f}({v}, {v2}) \, d{v2} \, d{v}"
        ),
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "v": S(_VARS),
            "v2": X(_VARS, ("v",)),
            "a": E(_atom, n=150),
            "b": E(_atom, n=150),
        },
    ),
    # c=16 — L'Hôpital's rule
    Template(
        name="lhopital_rule",
        latex=r"\lim_{{{v} \to {a}}} \frac{{{f}({v})}}{{{g2}({v})}} = \lim_{{{v} \to {a}}} \frac{{{f}'({v})}}{{{g2}'({v})}}",
        slots={
            "v": S(_VARS),
            "f": E(_fn_rich_nosub, n=100),
            "g2": E(_fn_rich_nosub, n=100),
            "a": E(_atom, n=150),
        },
    ),
    # c=17 — integration by parts (symbolic)
    Template(
        name="integration_by_parts",
        latex=r"\int {f} \, d{g2} = {f} {g2} - \int {g2} \, d{f}",
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "g2": E(_fn_rich_nosub, n=100),
        },
    ),
    # — integration by parts (definite)
    Template(
        name="integration_by_parts_definite",
        latex=(
            r"\int_{{{a}}}^{{{b}}} {f} \, d{g2} = "
            r"\left[{f} \, {g2}\right]_{{{a}}}^{{{b}}} - \int_{{{a}}}^{{{b}}} {g2} \, d{f}"
        ),
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "g2": E(_fn_rich_nosub, n=100),
            "a": E(_atom, n=150),
            "b": E(_atom, n=150),
        },
    ),
    # c=18 — Leibniz integral rule
    Template(
        name="leibniz_integral_rule",
        latex=(
            r"\frac{{d}}{{d{v}}} \int_{{a({v})}}^{{b({v})}} {f}({v}, t) \, dt = "
            r"{f}({v}, b({v})) b'({v}) - {f}({v}, a({v})) a'({v}) + "
            r"\int_{{a({v})}}^{{b({v})}} \frac{{\partial {f}}}{{\partial {v}}} \, dt"
        ),
        slots={
            "v": S(_VARS),
            "f": E(_fn_rich_nosub, n=100),
        },
    ),
    # c=19 — mean value theorem
    Template(
        name="mean_value_theorem",
        latex=r"\exists c \in ({a}, {b}) : {f}'(c) = \frac{{{f}({b}) - {f}({a})}}{{{b} - {a}}}",
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "a": E(_atom, n=150),
            "b": E(_atom, n=150),
        },
    ),
    # — mean value theorem (integral form)
    Template(
        name="mean_value_theorem_integral",
        latex=r"\frac{{1}}{{{b} - {a}}} \int_{{{a}}}^{{{b}}} {f}({v}) \, d{v} = {f}(c)",
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "v": S(_VARS),
            "a": E(_atom, n=150),
            "b": E(_atom, n=150),
        },
    ),
]

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

_W_CALCULUS: list[float] = compute_weights(_CALCULUS_TEMPLATES)

# ---------------------------------------------------------------------------
# Dispatch functions
# ---------------------------------------------------------------------------

_calculus = make_dispatcher(_CALCULUS_TEMPLATES, _W_CALCULUS)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "calculus": _calculus,
}

WEIGHTS: dict[str, float] = {
    "calculus": 0.10,
}

TEMPLATES: dict[str, list[Template]] = {
    "calculus": _CALCULUS_TEMPLATES,
}
