"""Calculus domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import E, S, Template, X, compute_weights, make_dispatcher
from .._templates import _def_integral, _indef_integral, _interval, _substack_prod, _substack_sum
from .._vocab import _SCALARS, _VARS, _atom, _expr, _fn_rich, _fn_rich_nosub

# ---------------------------------------------------------------------------
# Shared pools
# ---------------------------------------------------------------------------

_VEC_FIELD_POOL: tuple[str, ...] = ("F", "G", "E", "B", "v", "u", "w", "H")
_VOL_DOMAIN_POOL: tuple[str, ...] = ("V", r"\Omega", "D", "U", r"\mathcal{V}", "R")
_SURFACE_POOL: tuple[str, ...] = ("S", r"\Sigma", r"\mathcal{S}", "A")
_CURVE_POOL: tuple[str, ...] = (
    "C",
    r"\partial S",
    r"\partial \Sigma",
    r"\Gamma",
    r"\partial D",
)
_MFLD_POOL: tuple[str, ...] = ("M", r"\Omega", r"\Sigma", r"\mathcal{M}", "N", "X")

# ---------------------------------------------------------------------------
# Calculus templates
# ---------------------------------------------------------------------------

_CALCULUS_TEMPLATES: list[Template] = [
    # first derivative
    Template(
        name="first_derivative",
        latex=r"\frac{{d}}{{d{v}}}\left[{expr}\right]",
        slots={
            "v": S(_VARS),
            "expr": E(_expr, n=5000),
        },
    ),
    # nth-order derivative (expanded pool, was 3-item _order_n)
    Template(
        name="nth_derivative",
        latex=r"\frac{{d^{{{n}}}}}{{d{v}^{{{n}}}}}\left[{expr}\right]",
        slots={
            "v": S(_VARS),
            "n": S(("2", "3", "4", "n", "m")),
            "expr": E(_expr, n=5000),
        },
    ),
    # indefinite integral
    Template(
        name="indef_integral",
        latex=r"{result}",
        slots={"result": E(_indef_integral, n=5000)},
    ),
    # definite integral
    Template(
        name="def_integral",
        latex=r"{result}",
        slots={"result": E(_def_integral, n=5000)},
    ),
    # double integral over D
    Template(
        name="double_integral_domain",
        latex=r"\iint_{{\mathcal{{D}}}} {expr} \, d{v} \, d{v2}",
        slots={
            "v": S(_VARS),
            "v2": X(_VARS, ("v",)),
            "expr": E(_expr, n=5000),
        },
    ),
    # limit to scalar/inf/0 (fixed sampling bug: was _pt_scalar_inf_zero)
    Template(
        name="limit_simple",
        latex=r"\lim_{{{v} \to {pt}}} {expr}",
        slots={
            "v": S(_VARS),
            "pt": S(tuple(_SCALARS) + (r"\infty", "0")),
            "expr": E(_expr, n=5000),
        },
    ),
    # limit of ratio (fixed sampling bug: was _pt_inf_zero_scalar)
    Template(
        name="limit_ratio",
        latex=r"\lim_{{{v} \to {pt}}} \frac{{{num}}}{{{den}}}",
        slots={
            "v": S(_VARS),
            "pt": S((r"\infty", "0") + tuple(_SCALARS)),
            "num": E(_expr, n=5000),
            "den": E(_expr, n=5000),
        },
    ),
    # mixed partial derivative (proper slotted template, was E(_mixed_partial, n=500))
    Template(
        name="mixed_partial",
        latex=r"\frac{{\partial^2 {f}}}{{\partial {v} \, \partial {v2}}}",
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "v": S(_VARS),
            "v2": X(_VARS, ("v",)),
        },
    ),
    # Taylor series
    Template(
        name="taylor_series",
        latex=r"\sum{lim_mod}_{{n=0}}^{{\infty}} \frac{{{f}^{{(n)}}({a})}}{{n!}} \left({v} - {a}\right)^n",
        slots={
            "lim_mod": S(("", r"\limits")),
            "f": E(_fn_rich_nosub, n=100),
            "v": S(_VARS),
            "a": E(_atom, n=150),
        },
    ),
    # Maclaurin series (Taylor at a=0)
    Template(
        name="maclaurin_series",
        latex=r"\sum{lim_mod}_{{n=0}}^{{\infty}} \frac{{{f}^{{(n)}}(0)}}{{n!}} {v}^n",
        slots={
            "lim_mod": S(("", r"\limits")),
            "f": E(_fn_rich_nosub, n=100),
            "v": S(_VARS),
        },
    ),
    # Taylor polynomial (finite-terms form)
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
    # fundamental theorem of calculus
    Template(
        name="ftc",
        latex=r"\int{lim_mod}_{{{a}}}^{{{b}}} {f}'({v}) \, d{v} = {f}({b}) - {f}({a})",
        slots={
            "lim_mod": S(("", r"\limits")),
            "f": E(_fn_rich_nosub, n=100),
            "v": S(_VARS),
            "a": E(_atom, n=150),
            "b": E(_atom, n=150),
        },
    ),
    # chain rule — prime notation (standalone; high n_eff, separate from Leibniz group)
    Template(
        name="chain_rule",
        latex=(
            r"\frac{{d}}{{d{v}}}\left[{f}\!\left({g2}({v})\right)\right] = "
            r"{f}'\!\left({g2}({v})\right) {g2}'({v})"
        ),
        slots={
            "v": S(_VARS),
            "f": E(_fn_rich_nosub, n=100),
            "g2": E(_fn_rich_nosub, n=100),
        },
    ),
    # chain rule — Leibniz notation (2 variants; expanded z pool, was 3-item _FUNC_Z_POOL)
    Template(
        name="chain_rule_leibniz",
        latex="",
        slots={},
        variants=[
            Template(
                name="chain_rule_leibniz_classic",
                latex=r"\frac{{d{z}}}{{d{v}}} = \frac{{d{z}}}{{d{u}}} \cdot \frac{{d{u}}}{{d{v}}}",
                slots={
                    "z": S((r"\phi", r"\psi", r"\eta", r"\zeta", r"\rho", "w")),
                    "v": S(_VARS),
                    "u": X(_VARS, ("v",)),
                },
            ),
            Template(
                name="chain_rule_leibniz_composed",
                latex=(r"\frac{{d{f}}}{{d{v}}} = \frac{{d{f}}}{{d{gg}}} \cdot \frac{{d{gg}}}{{d{v}}}"),
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "gg": E(_fn_rich_nosub, n=100),
                    "v": S(_VARS),
                },
            ),
        ],
    ),
    # gradient (2D e-basis)
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
    # gradient (2D ij-hat notation)
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
    # gradient (3D e-basis)
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
    # divergence theorem (3 notation variants: standard, hat-normal, div-operator)
    Template(
        name="divergence_theorem",
        latex="",
        slots={},
        variants=[
            Template(
                name="divergence_standard",
                latex=(
                    r"\iint_{{\partial {vol}}} \mathbf{{{fld}}} \cdot d\mathbf{{S}} = "
                    r"\iiint_{{{vol}}} \nabla \cdot \mathbf{{{fld}}} \, dV"
                ),
                slots={
                    "fld": S(_VEC_FIELD_POOL),
                    "vol": S(_VOL_DOMAIN_POOL),
                },
            ),
            Template(
                name="divergence_hat_normal",
                latex=(
                    r"\oiint_{{\partial {vol}}} \mathbf{{{fld}}} \cdot \hat{{n}} \, dS = "
                    r"\iiint_{{{vol}}} \nabla \cdot \mathbf{{{fld}}} \, dV"
                ),
                slots={
                    "fld": S(_VEC_FIELD_POOL),
                    "vol": S(_VOL_DOMAIN_POOL),
                },
            ),
            Template(
                name="divergence_div_operator",
                latex=(
                    r"\iint_{{\partial {vol}}} \mathbf{{{fld}}} \cdot d\mathbf{{S}} = "
                    r"\iiint_{{{vol}}} \operatorname{{div}} \mathbf{{{fld}}} \, dV"
                ),
                slots={
                    "fld": S(_VEC_FIELD_POOL),
                    "vol": S(_VOL_DOMAIN_POOL),
                },
            ),
        ],
    ),
    # Laplacian (2D ∇² notation)
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
    # Laplacian (2D Δ notation)
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
    # Laplacian (3D)
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
    # Stokes' theorem (3 variants: ∇×, curl operator, differential forms)
    Template(
        name="stokes_theorem",
        latex="",
        slots={},
        variants=[
            Template(
                name="stokes_standard",
                latex=(
                    r"\oint{lim_mod}_{{{crv}}} \mathbf{{{fld}}} \cdot d\mathbf{{r}} = "
                    r"\iint_{{{srf}}} \left(\nabla \times \mathbf{{{fld}}}\right) \cdot d\mathbf{{S}}"
                ),
                slots={
                    "lim_mod": S(("", r"\limits")),
                    "fld": S(_VEC_FIELD_POOL),
                    "srf": S(_SURFACE_POOL),
                    "crv": S(_CURVE_POOL),
                },
            ),
            Template(
                name="stokes_curl_operator",
                latex=(
                    r"\oint{lim_mod}_{{{crv}}} \mathbf{{{fld}}} \cdot d\mathbf{{r}} = "
                    r"\iint_{{{srf}}} \operatorname{{curl}}\!\left(\mathbf{{{fld}}}\right) \cdot d\mathbf{{S}}"
                ),
                slots={
                    "lim_mod": S(("", r"\limits")),
                    "fld": S(_VEC_FIELD_POOL),
                    "srf": S(_SURFACE_POOL),
                    "crv": S(_CURVE_POOL),
                },
            ),
            Template(
                name="stokes_differential_forms",
                latex=r"\int{lim_mod}_{{\partial {mfld}}} \omega = \int{lim_mod}_{{{mfld}}} d\omega",
                slots={
                    "lim_mod": S(("", r"\limits")),
                    "mfld": S(_MFLD_POOL),
                },
            ),
        ],
    ),
    # Green's theorem (flux and circulation forms)
    Template(
        name="greens_theorem",
        latex="",
        slots={},
        variants=[
            Template(
                name="greens_theorem_flux",
                latex=(
                    r"\oint{lim_mod}_{{{crv}}} \mathbf{{{fld}}} \cdot \hat{{n}} \, ds = "
                    r"\iint_{{{dom}}} \nabla \cdot \mathbf{{{fld}}} \, dA"
                ),
                slots={
                    "lim_mod": S(("", r"\limits")),
                    "fld": S(_VEC_FIELD_POOL),
                    "crv": S(("C", r"\partial D", r"\partial R")),
                    "dom": S(("D", r"\Omega", "R")),
                },
            ),
            Template(
                name="greens_theorem_circulation",
                latex=(
                    r"\oint{lim_mod}_{{{crv}}} \mathbf{{{fld}}} \cdot d\mathbf{{r}} = "
                    r"\iint_{{{dom}}} \left(\frac{{\partial Q}}{{\partial x}} - "
                    r"\frac{{\partial P}}{{\partial y}}\right) dA"
                ),
                slots={
                    "lim_mod": S(("", r"\limits")),
                    "fld": S(_VEC_FIELD_POOL),
                    "crv": S(("C", r"\partial D", r"\partial R")),
                    "dom": S(("D", r"\Omega", "R")),
                },
            ),
        ],
    ),
    # iterated integral
    Template(
        name="iterated_integral",
        latex=(
            r"\int{lim_mod}_{{{a}}}^{{{b}}} \int{lim_mod}_{{g_1({v})}}^{{g_2({v})}} "
            r"{f}({v}, {v2}) \, d{v2} \, d{v}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "f": E(_fn_rich_nosub, n=100),
            "v": S(_VARS),
            "v2": X(_VARS, ("v",)),
            "a": E(_atom, n=150),
            "b": E(_atom, n=150),
        },
    ),
    # L'Hôpital's rule
    Template(
        name="lhopital_rule",
        latex=(
            r"\lim_{{{v} \to {a}}} \frac{{{f}({v})}}{{{g2}({v})}} = "
            r"\lim_{{{v} \to {a}}} \frac{{{f}'({v})}}{{{g2}'({v})}}"
        ),
        slots={
            "v": S(_VARS),
            "f": E(_fn_rich_nosub, n=100),
            "g2": E(_fn_rich_nosub, n=100),
            "a": E(_atom, n=150),
        },
    ),
    # integration by parts (symbolic form)
    Template(
        name="integration_by_parts",
        latex=r"\int {f} \, d{g2} = {f} {g2} - \int {g2} \, d{f}",
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "g2": E(_fn_rich_nosub, n=100),
        },
    ),
    # integration by parts (definite form)
    Template(
        name="integration_by_parts_definite",
        latex=(
            r"\int{lim_mod}_{{{a}}}^{{{b}}} {f} \, d{g2} = "
            r"\left[{f} \, {g2}\right]_{{{a}}}^{{{b}}} - \int{lim_mod}_{{{a}}}^{{{b}}} {g2} \, d{f}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "f": E(_fn_rich_nosub, n=100),
            "g2": E(_fn_rich_nosub, n=100),
            "a": E(_atom, n=150),
            "b": E(_atom, n=150),
        },
    ),
    # Leibniz integral rule (integration variable now a slot)
    Template(
        name="leibniz_integral_rule",
        latex=(
            r"\frac{{d}}{{d{v}}} \int{lim_mod}_{{a({v})}}^{{b({v})}} {f}({v}, {it}) \, d{it} = "
            r"{f}({v}, b({v})) b'({v}) - {f}({v}, a({v})) a'({v}) + "
            r"\int{lim_mod}_{{a({v})}}^{{b({v})}} \frac{{\partial {f}}}{{\partial {v}}} \, d{it}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "v": S(_VARS),
            "f": E(_fn_rich_nosub, n=100),
            "it": S(("t", "s", r"\tau", r"\sigma")),
        },
    ),
    # mean value theorem (derivative form)
    Template(
        name="mean_value_theorem",
        latex=r"\exists c \in ({a}, {b}) : {f}'(c) = \frac{{{f}({b}) - {f}({a})}}{{{b} - {a}}}",
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "a": E(_atom, n=150),
            "b": E(_atom, n=150),
        },
    ),
    # mean value theorem (integral form)
    Template(
        name="mean_value_theorem_integral",
        latex=r"\frac{{1}}{{{b} - {a}}} \int{lim_mod}_{{{a}}}^{{{b}}} {f}({v}) \, d{v} = {f}(c)",
        slots={
            "lim_mod": S(("", r"\limits")),
            "f": E(_fn_rich_nosub, n=100),
            "v": S(_VARS),
            "a": E(_atom, n=150),
            "b": E(_atom, n=150),
        },
    ),
    # ---- New templates ----
    # B1: derivative rules (product, quotient, power, generalized power)
    Template(
        name="derivative_rules",
        latex="",
        slots={},
        variants=[
            Template(
                name="product_rule",
                latex=(
                    r"\frac{{d}}{{d{v}}}\left[{f}({v}) \cdot {g2}({v})\right] = "
                    r"{f}'({v}) {g2}({v}) + {f}({v}) {g2}'({v})"
                ),
                slots={
                    "v": S(_VARS),
                    "f": E(_fn_rich_nosub, n=100),
                    "g2": E(_fn_rich_nosub, n=100),
                },
            ),
            Template(
                name="quotient_rule",
                latex=(
                    r"\frac{{d}}{{d{v}}}\left[\frac{{{f}({v})}}{{{g2}({v})}}\right] = "
                    r"\frac{{{f}'({v}) {g2}({v}) - {f}({v}) {g2}'({v})}}{{{g2}({v})^2}}"
                ),
                slots={
                    "v": S(_VARS),
                    "f": E(_fn_rich_nosub, n=100),
                    "g2": E(_fn_rich_nosub, n=100),
                },
            ),
            Template(
                name="power_rule",
                latex=r"\frac{{d}}{{d{v}}} {v}^{{{n}}} = {n} {v}^{{{n}-1}}",
                slots={
                    "v": S(_VARS),
                    "n": S(("n", "m", "p", "k", r"\alpha")),
                },
            ),
            Template(
                name="generalized_power_rule",
                latex=(
                    r"\frac{{d}}{{d{v}}} \left[{f}({v})\right]^{{{n}}} = "
                    r"{n} \left[{f}({v})\right]^{{{n}-1}} {f}'({v})"
                ),
                slots={
                    "v": S(_VARS),
                    "f": E(_fn_rich_nosub, n=100),
                    "n": S(("n", "m", "p", "k", r"\alpha")),
                },
            ),
        ],
    ),
    # B2: geometric integrals (arc length, surface area of revolution, parametric arc length)
    Template(
        name="geometric_integrals",
        latex="",
        slots={},
        variants=[
            Template(
                name="arc_length",
                latex=(
                    r"L = \int{lim_mod}_{{{a}}}^{{{b}}} "
                    r"\sqrt{{1 + \left[{f}'({v})\right]^2}} \, d{v}"
                ),
                slots={
                    "lim_mod": S(("", r"\limits")),
                    "f": E(_fn_rich_nosub, n=100),
                    "v": S(_VARS),
                    "a": E(_atom, n=150),
                    "b": E(_atom, n=150),
                },
            ),
            Template(
                name="surface_area_revolution",
                latex=(
                    r"S = 2\pi \int{lim_mod}_{{{a}}}^{{{b}}} {f}({v}) "
                    r"\sqrt{{1 + \left[{f}'({v})\right]^2}} \, d{v}"
                ),
                slots={
                    "lim_mod": S(("", r"\limits")),
                    "f": E(_fn_rich_nosub, n=100),
                    "v": S(_VARS),
                    "a": E(_atom, n=150),
                    "b": E(_atom, n=150),
                },
            ),
            Template(
                name="parametric_arc_length",
                latex=(
                    r"L = \int{lim_mod}_{{{a}}}^{{{b}}} \sqrt{{"
                    r"\left(\frac{{d{px}}}{{d{v}}}\right)^2 + "
                    r"\left(\frac{{d{py}}}{{d{v}}}\right)^2}} \, d{v}"
                ),
                slots={
                    "lim_mod": S(("", r"\limits")),
                    "v": S(_VARS),
                    "px": S((r"\phi", r"\psi", r"\xi", "p", "q")),
                    "py": S((r"\phi", r"\psi", r"\xi", "p", "q")),
                    "a": E(_atom, n=150),
                    "b": E(_atom, n=150),
                },
                distinct=[["px", "py"]],
            ),
        ],
    ),
    # B3: directional derivative (dot-product form and limit-definition form)
    Template(
        name="directional_derivative",
        latex="",
        slots={},
        variants=[
            Template(
                name="dir_deriv_dot",
                latex=r"D_{{\mathbf{{{u}}}}} {f} = \nabla {f} \cdot \mathbf{{{u}}}",
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "u": S(("u", "v", "e", "n")),
                },
            ),
            Template(
                name="dir_deriv_limit",
                latex=(
                    r"D_{{\mathbf{{{u}}}}} {f}({x}) = "
                    r"\lim_{{h \to 0}} \frac{{{f}({x} + h\mathbf{{{u}}}) - {f}({x})}}{{h}}"
                ),
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "u": S(("u", "v", "e", "n")),
                    "x": S(_VARS),
                },
            ),
        ],
    ),
    # B4: Rolle's theorem
    Template(
        name="rolles_theorem",
        latex=r"{f}({a}) = {f}({b}) \implies \exists c \in ({a}, {b}) : {f}'(c) = 0",
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "a": E(_atom, n=150),
            "b": E(_atom, n=150),
        },
    ),
    # B5: change of variables (Jacobian substitution)
    Template(
        name="change_of_variables",
        latex=(
            r"\iint_{{R}} {f}(x, y) \, dA = "
            r"\iint_{{S}} {f}\!\left({g2}(u, v), {h}(u, v)\right) "
            r"\left|\frac{{\partial(x, y)}}{{\partial(u, v)}}\right| \, du \, dv"
        ),
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "g2": E(_fn_rich_nosub, n=100),
            "h": E(_fn_rich_nosub, n=100),
        },
    ),
    # B6: implicit differentiation (implicit function theorem and chain-rule form)
    Template(
        name="implicit_differentiation",
        latex="",
        slots={},
        variants=[
            Template(
                name="implicit_diff_formula",
                latex=(
                    r"{ff}({v}, {y}) = 0 \implies "
                    r"\frac{{d{y}}}{{d{v}}} = -\frac{{{ff}_{{{v}}}}}{{{ff}_{{{y}}}}}"
                ),
                slots={
                    "ff": E(_fn_rich_nosub, n=100),
                    "v": S(_VARS),
                    "y": X(_VARS, ("v",)),
                },
            ),
            Template(
                name="implicit_diff_chain",
                latex=(
                    r"\frac{{d}}{{d{v}}} {f}({v}, {g2}({v})) = "
                    r"\frac{{\partial {f}}}{{\partial {v}}} + "
                    r"\frac{{\partial {f}}}{{\partial {y}}} {g2}'({v})"
                ),
                slots={
                    "f": E(_fn_rich_nosub, n=100),
                    "g2": E(_fn_rich_nosub, n=100),
                    "v": S(_VARS),
                    "y": X(_VARS, ("v",)),
                },
            ),
        ],
    ),
]

_CALCULUS_TEMPLATES += [
    # C1: inverse trig derivatives and identity
    Template(
        name="deriv_arcsin",
        latex=r"\frac{{d}}{{d{vv}}} \arcsin({vv}) = \frac{{1}}{{\sqrt{{1 - {vv}^2}}}}",
        slots={"vv": S(_VARS)},
    ),
    Template(
        name="deriv_arccos",
        latex=r"\frac{{d}}{{d{vv}}} \arccos({vv}) = -\frac{{1}}{{\sqrt{{1 - {vv}^2}}}}",
        slots={"vv": S(_VARS)},
    ),
    Template(
        name="deriv_arctan",
        latex=r"\frac{{d}}{{d{vv}}} \arctan({vv}) = \frac{{1}}{{1 + {vv}^2}}",
        slots={"vv": S(_VARS)},
    ),
    Template(
        name="arctan_complement_identity",
        latex=r"\arctan({vv}) + \arctan\!\left(\frac{{1}}{{{vv}}}\right) = \frac{{\pi}}{{2}}",
        slots={"vv": S(_VARS)},
    ),
    # C2: hyperbolic and trig identities/derivatives
    Template(
        name="deriv_tanh",
        latex=r"\frac{{d}}{{d{vv}}} \tanh({vv}) = \operatorname{{sech}}^2({vv})",
        slots={"vv": S(_VARS)},
    ),
    Template(
        name="deriv_coth",
        latex=r"\frac{{d}}{{d{vv}}} \coth({vv}) = -\operatorname{{csch}}^2({vv})",
        slots={"vv": S(_VARS)},
    ),
    Template(
        name="csc_pythagorean_identity",
        latex=r"\csc^2({vv}) = 1 + \cot^2({vv})",
        slots={"vv": S(_VARS)},
    ),
    Template(
        name="csc_reciprocal",
        latex=r"\csc({vv}) = \frac{{1}}{{\sin({vv})}}",
        slots={"vv": S(_VARS)},
    ),
    # C3: vector displacement (overrightarrow)
    Template(
        name="vector_displacement",
        latex=r"\overrightarrow{{AB}} = {bb} - {aa}",
        slots={
            "aa": S(_VARS),
            "bb": X(_VARS, ("aa",)),
        },
    ),
    Template(
        name="vector_displacement_magnitude",
        latex=r"|\overrightarrow{{AB}}| = \sqrt{{(B_1 - A_1)^2 + (B_2 - A_2)^2}}",
        slots={},
    ),
    # C4: underbrace and overbrace annotations
    Template(
        name="underbrace_polynomial",
        latex=r"\underbrace{{a_0 + a_1 {vv} + \cdots + a_n {vv}^n}}_{{n+1 \text{{ terms}}}}",
        slots={"vv": S(_VARS)},
    ),
    Template(
        name="overbrace_binomial",
        latex=(
            r"\overbrace{{(1 + {vv})^n}}^{{n \text{{ factors}}}} = "
            r"\sum{lim_mod}_{{k=0}}^{{n}} \binom{{n}}{{k}} {vv}^k"
        ),
        slots={"lim_mod": S(("", r"\limits")), "vv": S(_VARS)},
    ),
    # C5: four-dimensional integral (iiiint)
    Template(
        name="iiiint_domain",
        latex=r"\iiiint_{{\Omega}} f(x,y,z,w)\, dx\, dy\, dz\, dw",
        slots={},
    ),
    # C6: k-fold integral (idotsint) — variable-arity multi-integral
    Template(
        name="idotsint_k_fold",
        latex=(
            r"\idotsint_{{{dom}}} {ff}({v}_1,\dots,{v}_{{{idx}}}) "
            r"\, d{v}_1 \dots d{v}_{{{idx}}}"
        ),
        slots={
            "dom": S(("V", r"\Omega", "D", r"\mathcal{D}")),
            "ff": S(("f", "g", "h", r"\mu", r"\rho", r"\phi")),
            "v": S(("u", "x", "t", "s")),
            "idx": S(("k", "n", "m")),
        },
    ),
    Template(
        name="idotsint_measure",
        latex=r"\idotsint_{{{dom}}} {ff}({v}_1,\dots,{v}_{{{idx}}}) \, d\mu",
        slots={
            "dom": S(("V", r"\Omega", "D", r"\mathbb{R}^n")),
            "ff": S(("f", "g", r"\phi", r"\psi")),
            "v": S(("x", "u", "t")),
            "idx": S(("n", "k", "m")),
        },
    ),
    # C7: bare scalar line integral (oint without theorem context)
    Template(
        name="line_integral_scalar",
        latex=r"\oint{lim_mod}_{{{C}}} {ff}({v}) \, d{v}",
        slots={
            "lim_mod": S(("", r"\limits")),
            "C": S(("C", r"\gamma", r"\partial D", "L", r"\Gamma")),
            "v": S(_VARS),
            "ff": E(_fn_rich_nosub, n=100),
        },
    ),
    Template(
        name="line_integral_scalar_equals",
        latex=r"\oint{lim_mod}_{{{C}}} {ff}({v}) \, d{v} = {val}",
        slots={
            "lim_mod": S(("", r"\limits")),
            "C": S(("C", r"\gamma", r"\partial D", "L")),
            "v": S(_VARS),
            "ff": E(_fn_rich_nosub, n=100),
            "val": E(_atom, n=150),
        },
    ),
]

_CF_POOL = ("f", "g", "h", r"\phi", r"\psi")
_CV_POOL = ("x", "t", "u", "s")
_CD_POOL = (r"\mathbb{R}", r"[a,b]", r"[0,1]", r"\mathbb{R}^n")

_PART_ARROWS: list[Template] = [
    Template(
        name="continuous_function_rightarrow",
        latex=r"{ff}: {dom} \rightarrow \mathbb{{R}} \text{{ continuous}}",
        slots={"ff": S(_CF_POOL), "dom": S(_CD_POOL)},
    ),
    Template(
        name="integral_operator_mapsto",
        latex=r"{ff}: {xx} \mapsto \int_a^{{{xx}}} g(t)\,dt",
        slots={"ff": S(_CF_POOL), "xx": S(_CV_POOL)},
    ),
    Template(
        name="exponential_longmapsto",
        latex=r"\exp: {xx} \longmapsto e^{{{xx}}}",
        slots={"xx": S(_CV_POOL)},
    ),
]
_CALCULUS_TEMPLATES += _PART_ARROWS

# ---------------------------------------------------------------------------
# Evaluation-bar templates
# ---------------------------------------------------------------------------

_EVAL_BAR: list[Template] = [
    # Antiderivative evaluated at bounds — F(x)\Big|_a^b
    Template(
        name="antiderivative_eval_bar",
        latex=r"{f}({v})\Big|_{{{a}}}^{{{b}}}",
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "v": S(_VARS),
            "a": E(_atom, n=150),
            "b": E(_atom, n=150),
        },
    ),
    # FTC with \left. ... \right| invisible-left evaluation bar
    Template(
        name="ftc_eval_bar",
        latex=r"\int{lim_mod}_{{{a}}}^{{{b}}} {f}({v})\,d{v} = \left.{g}({v})\right|_{{{a}}}^{{{b}}}",
        slots={
            "lim_mod": S(("", r"\limits")),
            "f": E(_fn_rich_nosub, n=100),
            "g": E(_fn_rich_nosub, n=100),
            "v": S(_VARS),
            "a": E(_atom, n=150),
            "b": E(_atom, n=150),
        },
    ),
    # FTC three-part: integral = antiderivative bar = F(b) - F(a)
    Template(
        name="ftc_difference_bar",
        latex=r"\int{lim_mod}_{{{a}}}^{{{b}}} {f}({v})\,d{v} = {g}({v})\bigg|_{{{a}}}^{{{b}}} = {g}({b}) - {g}({a})",
        slots={
            "lim_mod": S(("", r"\limits")),
            "f": E(_fn_rich_nosub, n=100),
            "g": E(_fn_rich_nosub, n=100),
            "v": S(_VARS),
            "a": E(_atom, n=150),
            "b": E(_atom, n=150),
        },
    ),
    # IBP using \Big| evaluation bar instead of square-bracket notation
    Template(
        name="ibp_eval_bar",
        latex=r"\int{lim_mod}_{{{a}}}^{{{b}}} {u}\,d{w} = {u}\,{w}\Big|_{{{a}}}^{{{b}}} - \int{lim_mod}_{{{a}}}^{{{b}}} {w}\,d{u}",
        slots={
            "lim_mod": S(("", r"\limits")),
            "u": E(_fn_rich_nosub, n=100),
            "w": E(_fn_rich_nosub, n=100),
            "a": E(_atom, n=150),
            "b": E(_atom, n=150),
        },
    ),
    # First derivative evaluated at a point: \left. d/dv [...] \right|_{v=a}
    Template(
        name="derivative_eval_point",
        latex=r"\left.\frac{{d}}{{d{v}}}\left[{expr}\right]\right|_{{{v}={a}}}",
        slots={
            "v": S(_VARS),
            "expr": E(_expr, n=5000),
            "a": E(_atom, n=150),
        },
    ),
    # Second derivative evaluated at a point
    Template(
        name="second_deriv_eval_point",
        latex=r"\left.\frac{{d^2}}{{d{v}^2}}\left[{expr}\right]\right|_{{{v}={a}}}",
        slots={
            "v": S(_VARS),
            "expr": E(_expr, n=5000),
            "a": E(_atom, n=150),
        },
    ),
    # Mixed partial evaluated at a point — uses \bigg| for tall fraction
    Template(
        name="mixed_partial_eval_point",
        latex=r"\frac{{\partial^2 {f}}}{{\partial {v}\,\partial {w}}}\bigg|_{{({a},{b})}}",
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "v": S(_VARS),
            "w": X(_VARS, ("v",)),
            "a": E(_atom, n=150),
            "b": E(_atom, n=150),
        },
    ),
    # Gradient evaluated at a point
    Template(
        name="gradient_eval_point",
        latex=r"\nabla {f}\bigg|_{{{v}={a}}}",
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "v": S(_VARS),
            "a": E(_atom, n=150),
        },
    ),
    # Partial derivative evaluated on a surface w = c
    Template(
        name="partial_eval_surface",
        latex=r"\left.\frac{{\partial {f}}}{{\partial {v}}}\right|_{{{w}={c}}}",
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "v": S(_VARS),
            "w": X(_VARS, ("v",)),
            "c": E(_atom, n=150),
        },
    ),
    # Lagrange remainder — common companion to Taylor templates
    Template(
        name="lagrange_remainder",
        latex=r"R_{{{n}}}({v}) = \frac{{{f}^{{({n}+1)}}(\xi)}}{{({n}+1)!}}\,({v}-{a})^{{{n}+1}},\quad \xi \in ({a},{v})",
        slots={
            "f": E(_fn_rich_nosub, n=100),
            "v": S(_VARS),
            "a": E(_atom, n=150),
            "n": S(("n", "N", "k")),
        },
    ),
]
_CALCULUS_TEMPLATES += _EVAL_BAR

# ---------------------------------------------------------------------------
# Medspace (\:) — differential and qualifier patterns
# ---------------------------------------------------------------------------

_PART_MEDSPACE: list[Template] = [
    Template(
        name="medspace_double_integral_diffs",
        latex=r"\iint_{{{dom}}} {ff}({v1},{v2}) \: d{v1} \: d{v2}",
        slots={
            "dom": S(("D", "R", r"\Omega", "S", "U")),
            "ff": E(_fn_rich_nosub, n=100),
            "v1": S(_VARS),
            "v2": X(_VARS, ("v1",)),
        },
    ),
    Template(
        name="medspace_triple_integral_diffs",
        latex=r"\iiint_{{{dom}}} {ff} \: d{v1} \: d{v2} \: d{v3}",
        slots={
            "dom": S(("V", r"\Omega", "D")),
            "ff": E(_fn_rich_nosub, n=100),
            "v1": S(_VARS),
            "v2": X(_VARS, ("v1",)),
            "v3": X(_VARS, ("v1", "v2")),
        },
    ),
    Template(
        name="medspace_forall_qualifier",
        latex=r"{ff}({vv}) = {expr}, \: \forall {vv} \in {dom}",
        slots={
            "ff": E(_fn_rich_nosub, n=100),
            "vv": S(_VARS),
            "expr": E(_expr, n=5000),
            "dom": S((r"\mathbb{R}", r"\mathbb{Z}", r"[a,b]", r"\mathbb{N}", r"(0,\infty)")),
        },
    ),
    Template(
        name="medspace_subset_qualifier",
        latex=r"{aa} \subseteq {bb}, \: {bb} \subseteq {cc} \implies {aa} \subseteq {cc}",
        slots={
            "aa": S(("A", "B", "C", "S", "T")),
            "bb": X(("A", "B", "C", "S", "T"), ("aa",)),
            "cc": X(("A", "B", "C", "S", "T"), ("aa", "bb")),
        },
    ),
    Template(
        name="medspace_integral_measure",
        latex=r"\int_{{{dom}}} {ff}({vv}) \: d\mu({vv})",
        slots={
            "dom": S(("E", "X", r"\Omega", "A")),
            "ff": E(_fn_rich_nosub, n=100),
            "vv": S(_VARS),
        },
    ),
    Template(
        name="medspace_differential_form",
        latex=r"{ff}({vv}) \: d{v1} \wedge d{v2}",
        slots={
            "ff": E(_fn_rich_nosub, n=100),
            "vv": S(_VARS),
            "v1": S(_VARS),
            "v2": X(_VARS, ("v1",)),
        },
    ),
    Template(
        name="medspace_lim_qualifier",
        latex=r"\lim_{{{vv} \to \infty}} {ff}({vv}) = {expr}, \: {ff} \text{{ monotone}}",
        slots={
            "vv": S(_VARS),
            "ff": E(_fn_rich_nosub, n=100),
            "expr": E(_atom, n=150),
        },
    ),
]
_CALCULUS_TEMPLATES += _PART_MEDSPACE

_CALCULUS_TEMPLATES += [
    Template(
        name="substack_sum",
        latex=r"{s}",
        slots={"s": E(_substack_sum, n=5_000_000)},
    ),
    Template(
        name="substack_prod",
        latex=r"{s}",
        slots={"s": E(_substack_prod, n=5_000_000)},
    ),
    Template(
        name="interval_notation",
        latex=r"{v} \in {ivl}",
        slots={
            "v": E(lambda rng: rng.choice(["x", "y", "z", "t", "u"]), n=5_000_000),
            "ivl": E(_interval, n=5_000_000),
        },
    ),
    Template(
        name="interval_bounds",
        latex=r"{f} : {ivl_a} \to {ivl_b}",
        slots={
            "f": E(lambda rng: rng.choice(["f", "g", "h", "F"]), n=4),
            "ivl_a": E(_interval, n=5_000_000),
            "ivl_b": E(_interval, n=5_000_000),
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
