"""Complex analysis domain generators."""

from __future__ import annotations

from .._template_dsl import _FN_SLOT, _LIM_MOD, S, Template, X
from ._config import register_domain

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_VAR_POOL = ("z", "w", "s", "t", "u", "v", r"\zeta", r"\xi", r"\eta", r"\omega")  # 10
_REAL_POOL = ("x", "y", "u", "v", "s", "t", "r")  # 7
_SCALAR_POOL = (
    "a",
    "b",
    "c",
    r"\alpha",
    r"\beta",
    r"\lambda",
    r"\mu",
    r"\rho",
    r"\kappa",
)  # 9
_ANGLE_POOL = (r"\theta", r"\phi", r"\varphi", r"\psi", r"\alpha", r"\beta")  # 6
_CENTER_POOL = ("a", r"z_0", r"\alpha", r"\beta", "0")  # 5
_CURVE_POOL = (r"\Gamma", "C", r"C_r", r"\partial D", r"\gamma")  # 5
_FUNC_POOL = ("f", "g", "h", "F", "G", r"\phi", r"\psi", r"\Phi")  # 8
_INT_POOL = ("m", "n", "k", "p", "q")  # 5
_COMP_POOL = ("u", "v", r"\varphi", r"\psi", "a", "b")  # 6 — component names
_MOBIUS_POOL = ("a", "b", "c", "d", r"\alpha", r"\beta", r"\gamma", r"\delta")  # 8
_NOME_POOL = ("q", r"q_0", "r")  # nome variable for Jacobi theta functions

# ---------------------------------------------------------------------------
# Part A: Reparameterized originals (14 templates)
# ---------------------------------------------------------------------------

_TEMPLATES_A: list[Template] = [
    Template(
        name="complex_cartesian",
        latex=r"z = {xx} + i{yy}",
        slots={"xx": S(_REAL_POOL), "yy": X(_REAL_POOL, ("xx",))},
    ),
    Template(
        name="complex_polar",
        latex=r"z = {rr} e^{{i{th}}}",
        slots={"rr": S(_SCALAR_POOL), "th": S(_ANGLE_POOL)},
    ),
    Template(
        name="modulus_squared",
        latex=r"|z|^2 = {xx}^2 + {yy}^2",
        slots={"xx": S(_REAL_POOL), "yy": X(_REAL_POOL, ("xx",))},
    ),
    Template(
        name="cauchy_riemann",
        latex=(
            r"\frac{{\partial {uu}}}{{\partial {xx}}} = \frac{{\partial {vv}}}{{\partial {yy}}},"
            r"\quad \frac{{\partial {uu}}}{{\partial {yy}}} = -\frac{{\partial {vv}}}{{\partial {xx}}}"
        ),
        slots={
            "uu": S(_COMP_POOL),
            "vv": X(_COMP_POOL, ("uu",)),
            "xx": S(_REAL_POOL),
            "yy": X(_REAL_POOL, ("xx",)),
        },
    ),
    Template(
        name="cauchy_integral_formula",
        latex=(
            r"{ff}({aa}) = \frac{{1}}{{2\pi i}}"
            r"\oint{lim_mod}_{{{CC}}} \frac{{{ff}(z)}}{{z - {aa}}} \, dz"
        ),
        slots={"lim_mod": _LIM_MOD, "ff": S(_FUNC_POOL), "aa": S(_CENTER_POOL), "CC": S(_CURVE_POOL)},
    ),
    Template(
        name="residue_theorem",
        latex=(r"\oint{lim_mod}_{{{CC}}} {ff}(z) \, dz = 2\pi i \sum_k \operatorname{{Res}}({ff}, z_k)"),
        slots={"lim_mod": _LIM_MOD, "ff": S(_FUNC_POOL), "CC": S(_CURVE_POOL)},
    ),
    Template(
        name="laurent_series",
        latex=r"{ff}(z) = \sum{lim_mod}_{{n=-\infty}}^{{\infty}} c_n (z - {aa})^n",
        slots={"lim_mod": _LIM_MOD, "ff": S(_FUNC_POOL), "aa": S(_CENTER_POOL)},
    ),
    Template(
        name="mobius_transformation",
        latex=(
            r"w = \frac{{{aa} z + {bb}}}{{{cc} z + {dd}}},"
            r"\quad {aa} {dd} - {bb} {cc} \neq 0"
        ),
        slots={
            "aa": S(_MOBIUS_POOL),
            "bb": X(_MOBIUS_POOL, ("aa",)),
            "cc": X(_MOBIUS_POOL, ("aa", "bb")),
            "dd": X(_MOBIUS_POOL, ("aa", "bb", "cc")),
        },
    ),
    Template(
        name="eulers_formula",
        latex=r"e^{{i{th}}} = \cos {th} + i \sin {th}",
        slots={"th": S(_ANGLE_POOL)},
    ),
    Template(
        name="complex_conjugate",
        latex=r"\bar{{{zz}}} = {xx} - i{yy}, \quad {zz}\bar{{{zz}}} = |{zz}|^2",
        slots={
            "zz": S(_VAR_POOL),
            "xx": S(_REAL_POOL),
            "yy": X(_REAL_POOL, ("xx",)),
        },
    ),
    Template(
        name="real_imag_parts",
        latex=(
            r"\operatorname{{Re}}({zz}) = \frac{{{zz} + \bar{{{zz}}}}}{{2}},"
            r"\quad \operatorname{{Im}}({zz}) = \frac{{{zz} - \bar{{{zz}}}}}{{2i}}"
        ),
        slots={"zz": S(_VAR_POOL)},
    ),
    Template(
        name="argument_principle",
        latex=(r"\frac{{1}}{{2\pi i}} \oint{lim_mod}_{{{CC}}} \frac{{{ff}'(z)}}{{{ff}(z)}} \, dz = N - P"),
        slots={"lim_mod": _LIM_MOD, "ff": S(_FUNC_POOL), "CC": S(_CURVE_POOL)},
    ),
    Template(
        name="rouche_theorem",
        latex=(
            r"|{ff}(z) - {gg}(z)| < |{gg}(z)|"
            r"\text{{ on }}{CC} \implies Z_{{{ff}}} = Z_{{{gg}}}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "CC": S(_CURVE_POOL),
        },
    ),
    Template(
        name="maximum_modulus",
        latex=(
            r"|{ff}(z)| \leq \max_{{|\zeta|={rr}}} |{ff}(\zeta)|"
            r"\text{{ for }} |z| \leq {rr}"
        ),
        slots={"ff": S(_FUNC_POOL), "rr": S(_SCALAR_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B1: Complex Arithmetic (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B1: list[Template] = [
    Template(
        name="de_moivre",
        latex=(r"(\cos {th} + i\sin {th})^{{{nn}}} = \cos({nn}{th}) + i\sin({nn}{th})"),
        slots={"th": S(_ANGLE_POOL), "nn": S(_INT_POOL)},
    ),
    Template(
        name="complex_logarithm",
        latex=r"\log {zz} = \ln|{zz}| + i\arg {zz}",
        slots={"zz": S(_VAR_POOL)},
    ),
    Template(
        name="argument_sum",
        latex=r"\arg({zz1} {zz2}) = \arg {zz1} + \arg {zz2} \pmod{{2\pi}}",
        slots={"zz1": S(_VAR_POOL), "zz2": X(_VAR_POOL, ("zz1",))},
    ),
    Template(
        name="complex_product_polar",
        latex=r"|{zz1} {zz2}| = |{zz1}|\,|{zz2}|",
        slots={"zz1": S(_VAR_POOL), "zz2": X(_VAR_POOL, ("zz1",))},
    ),
    Template(
        name="triangle_inequality",
        latex=r"|{zz1} + {zz2}| \leq |{zz1}| + |{zz2}|",
        slots={"zz1": S(_VAR_POOL), "zz2": X(_VAR_POOL, ("zz1",))},
    ),
    Template(
        name="inverse_euler",
        latex=r"\cos {th} = \frac{{e^{{i{th}}} + e^{{-i{th}}}}}{{2}}",
        slots={"th": S(_ANGLE_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B2: Analytic Functions (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B2: list[Template] = [
    Template(
        name="complex_derivative_def",
        latex=(
            r"{ff}'({zz}) = \lim_{{\Delta {zz}\to 0}}"
            r"\frac{{{ff}({zz}+\Delta {zz})-{ff}({zz})}}{{\Delta {zz}}}"
        ),
        slots={"ff": S(_FUNC_POOL), "zz": S(_VAR_POOL)},
    ),
    Template(
        name="harmonic_laplacian",
        latex=(
            r"\nabla^2 {uu} = "
            r"\frac{{\partial^2 {uu}}}{{\partial {xx}^2}} + "
            r"\frac{{\partial^2 {uu}}}{{\partial {yy}^2}} = 0"
        ),
        slots={
            "uu": S(_COMP_POOL),
            "xx": S(_REAL_POOL),
            "yy": X(_REAL_POOL, ("xx",)),
        },
    ),
    Template(
        name="liouville_bound",
        latex=(r"|{ff}(z)| \leq M \;\forall\,z \in \mathbb{{C}} \implies {ff} \equiv \text{{const}}"),
        slots={"ff": S(_FUNC_POOL)},
    ),
    Template(
        name="analytic_implies_cr",
        latex=(
            r"{ff} = {uu} + i{vv}\text{{ analytic}}"
            r"\Rightarrow \frac{{\partial {uu}}}{{\partial {xx}}} = \frac{{\partial {vv}}}{{\partial {yy}}}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "uu": S(_COMP_POOL),
            "vv": X(_COMP_POOL, ("uu",)),
            "xx": S(_REAL_POOL),
            "yy": X(_REAL_POOL, ("xx",)),
        },
    ),
    Template(
        name="power_function",
        latex=r"{ff}({zz}) = {zz}^{{{nn}}}",
        slots={"ff": S(_FUNC_POOL), "zz": S(_VAR_POOL), "nn": S(_INT_POOL)},
    ),
    Template(
        name="exp_function",
        latex=r"{ff}({zz}) = e^{{{zz}}}",
        slots={"ff": S(_FUNC_POOL), "zz": S(_VAR_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B3: Contour Integration (8)
# ---------------------------------------------------------------------------

_TEMPLATES_B3: list[Template] = [
    Template(
        name="cauchy_nth_derivative",
        latex=(
            r"{ff}^{{({nn})}}({aa}) = \frac{{{nn}!}}{{2\pi i}}"
            r"\oint{lim_mod}_{{{CC}}} \frac{{{ff}({zz})}}{{({zz}-{aa})^{{{nn}+1}}}} \, d{zz}"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "ff": S(_FUNC_POOL),
            "nn": S(_INT_POOL),
            "aa": S(_CENTER_POOL),
            "CC": S(_CURVE_POOL),
            "zz": S(_VAR_POOL),
        },
    ),
    Template(
        name="residue_simple_pole",
        latex=(
            r"\operatorname{{Res}}({ff}, {aa}) = "
            r"\lim_{{{zz} \to {aa}}} ({zz} - {aa}) {ff}({zz})"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "aa": S(_CENTER_POOL),
            "zz": S(_VAR_POOL),
        },
    ),
    Template(
        name="residue_higher_pole",
        latex=(
            r"\operatorname{{Res}}({ff}, {aa}) = \frac{{1}}{{({nn}-1)!}}"
            r"\lim_{{{zz} \to {aa}}} \frac{{d^{{{nn}-1}}}}{{d{zz}^{{{nn}-1}}}}"
            r"[({zz}-{aa})^{{{nn}}} {ff}({zz})]"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "aa": S(_CENTER_POOL),
            "nn": S(_INT_POOL),
            "zz": S(_VAR_POOL),
        },
    ),
    Template(
        name="winding_number",
        latex=(r"n(\gamma, {aa}) = \frac{{1}}{{2\pi i}} \oint_\gamma \frac{{d{zz}}}{{{zz} - {aa}}}"),
        slots={"aa": S(_CENTER_POOL), "zz": S(_VAR_POOL)},
    ),
    Template(
        name="ml_inequality",
        latex=(r"\left| \oint{lim_mod}_{{{CC}}} {ff}({zz}) \, d{zz} \right| \leq M \cdot L"),
        slots={"lim_mod": _LIM_MOD, "ff": S(_FUNC_POOL), "CC": S(_CURVE_POOL), "zz": S(_VAR_POOL)},
    ),
    Template(
        name="cauchy_goursat",
        latex=r"\oint{lim_mod}_{{{CC}}} {ff}({zz}) \, d{zz} = 0",
        slots={"lim_mod": _LIM_MOD, "ff": S(_FUNC_POOL), "CC": S(_CURVE_POOL), "zz": S(_VAR_POOL)},
    ),
    Template(
        name="residue_sum_formula",
        latex=(
            r"\oint{lim_mod}_{{{CC}}} {ff}({zz}) \, d{zz} = "
            r"2\pi i \sum_k \operatorname{{Res}}({ff}, a_k)"
        ),
        slots={"lim_mod": _LIM_MOD, "ff": S(_FUNC_POOL), "CC": S(_CURVE_POOL), "zz": S(_VAR_POOL)},
    ),
    Template(
        name="jordan_estimate",
        latex=(
            r"\left| \int{lim_mod}_{{C_R}} {ff}({zz}) e^{{i{aa} {zz}}} \, d{zz} \right| \to 0"
            r"\text{{ as }} R \to \infty"
        ),
        slots={"lim_mod": _LIM_MOD, "ff": S(_FUNC_POOL), "aa": S(_CENTER_POOL), "zz": S(_VAR_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B4: Series & Convergence (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B4: list[Template] = [
    Template(
        name="taylor_complex",
        latex=(
            r"{ff}({zz}) = \sum{lim_mod}_{{n=0}}^{{\infty}} "
            r"\frac{{{ff}^{{(n)}}({aa})}}{{n!}} ({zz} - {aa})^n"
        ),
        slots={"lim_mod": _LIM_MOD, "ff": S(_FUNC_POOL), "zz": S(_VAR_POOL), "aa": S(_CENTER_POOL)},
    ),
    Template(
        name="power_series_domain",
        latex=(
            r"{ff}({zz}) = \sum{lim_mod}_{{n=0}}^{{\infty}} c_n ({zz} - {aa})^n,"
            r"\quad |{zz} - {aa}| < R"
        ),
        slots={"lim_mod": _LIM_MOD, "ff": S(_FUNC_POOL), "zz": S(_VAR_POOL), "aa": S(_CENTER_POOL)},
    ),
    Template(
        name="laurent_annulus",
        latex=(
            r"{ff}({zz}) = \sum{lim_mod}_{{n=-\infty}}^{{\infty}} c_n ({zz} - {aa})^n,"
            r"\quad r < |{zz} - {aa}| < R"
        ),
        slots={"lim_mod": _LIM_MOD, "ff": S(_FUNC_POOL), "zz": S(_VAR_POOL), "aa": S(_CENTER_POOL)},
    ),
    Template(
        name="radius_limsup",
        latex=r"R = \frac{{1}}{{\limsup_{{n \to \infty}} |c_n|^{{1/n}}}}",
        slots={},
    ),
    Template(
        name="radius_ratio",
        latex=r"R = \lim_{{n \to \infty}} \left| \frac{{c_n}}{{c_{{n+1}}}} \right|",
        slots={},
    ),
    Template(
        name="weierstrass_product",
        latex=(
            r"{ff}({zz}) = {zz}^m e^{{{gg}({zz})}} "
            r"\prod_n \!\left(1 - \tfrac{{{zz}}}{{a_n}}\right) e^{{{zz}/a_n}}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "zz": S(_VAR_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B5: Conformal Mapping (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B5: list[Template] = [
    Template(
        name="joukowski_map",
        latex=r"{ww} = {zz} + \frac{{{rr}^2}}{{{zz}}}",
        slots={
            "ww": S(_VAR_POOL),
            "zz": X(_VAR_POOL, ("ww",)),
            "rr": S(_SCALAR_POOL),
        },
    ),
    Template(
        name="schwarz_lemma",
        latex=(
            r"|{ff}({zz})| \leq |{zz}|, \quad"
            r" {ff} : \mathbb{{D}} \to \mathbb{{D}},\; {ff}(0) = 0"
        ),
        slots={"ff": S(_FUNC_POOL), "zz": S(_VAR_POOL)},
    ),
    Template(
        name="conformal_composition",
        latex=r"({gg} \circ {ff})({zz}) = {gg}({ff}({zz}))\text{{ conformal}}",
        slots={
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "zz": S(_VAR_POOL),
        },
    ),
    Template(
        name="log_map",
        latex=r"{ww} = \log {zz} = \ln|{zz}| + i\arg {zz}",
        slots={"ww": S(_VAR_POOL), "zz": X(_VAR_POOL, ("ww",))},
    ),
    Template(
        name="cayley_map",
        latex=r"{ww} = \frac{{{zz} - i}}{{{zz} + i}},\quad \mathbb{{H}} \to \mathbb{{D}}",
        slots={"ww": S(_VAR_POOL), "zz": X(_VAR_POOL, ("ww",))},
    ),
    Template(
        name="exp_map",
        latex=(
            r"{ww} = e^{{{zz}}}, \quad"
            r" e^{{{xx} + i{yy}}} = e^{{{xx}}}(\cos {yy} + i\sin {yy})"
        ),
        slots={
            "ww": S(_VAR_POOL),
            "zz": X(_VAR_POOL, ("ww",)),
            "xx": S(_REAL_POOL),
            "yy": X(_REAL_POOL, ("xx",)),
        },
    ),
]

# ---------------------------------------------------------------------------
# Part B6: Special Theorems (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B6: list[Template] = [
    Template(
        name="open_mapping",
        latex=(
            r"{ff} \not\equiv \text{{const}},\;"
            r"{ff}\text{{ analytic}} \Rightarrow {ff}(\Omega)\text{{ open}}"
        ),
        slots={"ff": S(_FUNC_POOL)},
    ),
    Template(
        name="identity_theorem",
        latex=(
            r"{ff}\big|_E = {gg}\big|_E,\;"
            r"E' \cap \Omega \neq \emptyset"
            r"\Rightarrow {ff} \equiv {gg}"
        ),
        slots={"ff": S(_FUNC_POOL), "gg": X(_FUNC_POOL, ("ff",))},
    ),
    Template(
        name="schwarz_reflection",
        latex=r"{ff}(\bar{{{zz}}}) = \overline{{{ff}({zz})}}",
        slots={"ff": S(_FUNC_POOL), "zz": S(_VAR_POOL)},
    ),
    Template(
        name="casorati_weierstrass",
        latex=r"\overline{{{ff}(D^*({aa}))}} = \mathbb{{C}}",
        slots={"ff": S(_FUNC_POOL), "aa": S(_CENTER_POOL)},
    ),
    Template(
        name="montel_theorem",
        latex=(
            r"\sup_n |{ff}_n({zz})| \leq M"
            r"\Rightarrow ({ff}_n)\text{{ normal family}}"
        ),
        slots={"ff": S(_FUNC_POOL), "zz": S(_VAR_POOL)},
    ),
    Template(
        name="picards_little",
        latex=(
            r"{ff}\text{{ entire nonconstant}}"
            r"\Rightarrow \mathbb{{C}} \setminus {ff}(\mathbb{{C}})\text{{ finite}}"
        ),
        slots={"ff": S(_FUNC_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B7: Special Functions (6)
# ---------------------------------------------------------------------------

_TEMPLATES_B7: list[Template] = [
    Template(
        name="gamma_functional",
        latex=r"\Gamma({ss} + 1) = {ss}\,\Gamma({ss})",
        slots={"ss": S(_VAR_POOL)},
    ),
    Template(
        name="gamma_reflection",
        latex=r"\Gamma({ss})\,\Gamma(1 - {ss}) = \frac{{\pi}}{{\sin(\pi {ss})}}",
        slots={"ss": S(_VAR_POOL)},
    ),
    Template(
        name="riemann_zeta_def",
        latex=(
            r"\zeta({ss}) = \sum{lim_mod}_{{n=1}}^{{\infty}} \frac{{1}}{{n^{{{ss}}}}},"
            r"\quad \operatorname{{Re}}({ss}) > 1"
        ),
        slots={"lim_mod": _LIM_MOD, "ss": S(_VAR_POOL)},
    ),
    Template(
        name="zeta_functional_eq",
        latex=(
            r"\zeta({ss}) = 2^{{{ss}}} \pi^{{{ss}-1}}"
            r"\sin\!\tfrac{{\pi {ss}}}{{2}}\,\Gamma(1-{ss})\,\zeta(1-{ss})"
        ),
        slots={"ss": S(_VAR_POOL)},
    ),
    Template(
        name="mellin_transform",
        latex=(
            r"\mathcal{{M}}\{{{ff}\}}({ss}) = "
            r"\int_0^\infty {xx}^{{{ss}-1}} {ff}({xx})\,d{xx}"
        ),
        slots={
            "ff": S(_FUNC_POOL),
            "ss": S(_VAR_POOL),
            "xx": X(_VAR_POOL, ("ss",)),
        },
    ),
    Template(
        name="hurwitz_zeta",
        latex=(
            r"\zeta({ss}, {aa}) = "
            r"\sum{lim_mod}_{{n=0}}^{{\infty}} \frac{{1}}{{(n + {aa})^{{{ss}}}}}"
        ),
        slots={"lim_mod": _LIM_MOD, "ss": S(_VAR_POOL), "aa": X(_VAR_POOL, ("ss",))},
    ),
]

# ---------------------------------------------------------------------------
# Part C: High-n_eff function-pair templates (6)
# ---------------------------------------------------------------------------

_TEMPLATES_C: list[Template] = [
    Template(
        name="holomorphic_sum",
        latex=(
            r"({fn1} + {fn2})({zz}) = {fn1}({zz}) + {fn2}({zz})"
            r"\text{{ holomorphic}}"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "zz": S(_VAR_POOL),
        },
    ),
    Template(
        name="composition_analytic",
        latex=(r"({fn1} \circ {fn2})({zz}) = {fn1}({fn2}({zz}))\text{{ analytic}}"),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "zz": S(_VAR_POOL),
        },
    ),
    Template(
        name="product_rule_complex",
        latex=(
            r"({fn1} {fn2})'({zz}) = "
            r"{fn1}'({zz}){fn2}({zz}) + {fn1}({zz}){fn2}'({zz})"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "zz": S(_VAR_POOL),
        },
    ),
    Template(
        name="cauchy_inequality_fn",
        latex=(
            r"\left|{fn1}^{{(n)}}({aa})\right| \leq \frac{{n!\,M}}{{r^n}}"
            r"\text{{ on }} |{zz}-{aa}|=r"
        ),
        slots={
            "fn1": _FN_SLOT,
            "aa": S(_CENTER_POOL),
            "zz": S(_VAR_POOL),
        },
    ),
    Template(
        name="contour_fn_pair",
        latex=r"\oint{lim_mod}_{{{CC}}} {fn1}({zz}) {fn2}({zz}) \, d{zz}",
        slots={
            "lim_mod": _LIM_MOD,
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "CC": S(_CURVE_POOL),
            "zz": S(_VAR_POOL),
        },
    ),
    Template(
        name="fn_triple_composition",
        latex=r"{fn1}({fn2}({fn3}({zz})))",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "fn3": _FN_SLOT,
            "zz": S(_VAR_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Assemble all templates
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Part D: Jacobi theta functions (\vartheta)
# ---------------------------------------------------------------------------

_TEMPLATES_D: list[Template] = [
    Template(
        name="jacobi_theta_series",
        latex=(
            r"\vartheta_3({zz}, {qq}) = "
            r"\sum{lim_mod}_{{n=-\infty}}^{{\infty}} {qq}^{{n^2}} e^{{2\pi i n {zz}}}"
        ),
        slots={"lim_mod": _LIM_MOD, "zz": S(_VAR_POOL), "qq": S(_NOME_POOL)},
    ),
    Template(
        name="jacobi_theta_product",
        latex=(
            r"\vartheta_3(0, {qq}) = "
            r"\prod{lim_mod}_{{n=1}}^{{\infty}} (1-{qq}^{{2n}})(1+{qq}^{{2n-1}})^2"
        ),
        slots={"lim_mod": _LIM_MOD, "qq": S(_NOME_POOL)},
    ),
    Template(
        name="jacobi_theta_identity",
        latex=r"\vartheta_3(0,{qq})^4 = \vartheta_2(0,{qq})^4 + \vartheta_4(0,{qq})^4",
        slots={"qq": S(_NOME_POOL)},
    ),
    Template(
        name="jacobi_theta_symmetry",
        latex=r"\vartheta_3({zz}, {qq}) = \vartheta_3(-{zz}, {qq})",
        slots={"zz": S(_VAR_POOL), "qq": S(_NOME_POOL)},
    ),
    Template(
        name="jacobi_theta_modular",
        latex=(
            r"\vartheta_3\!\left(0,\, e^{{-\pi {tt}}}\right)"
            r" = \frac{{1}}{{\sqrt{{{tt}}}}}\; \vartheta_3\!\left(0,\, e^{{-\pi/{tt}}}\right)"
        ),
        slots={"tt": S(_REAL_POOL)},
    ),
]

_TEMPLATES_E: list[Template] = [
    Template(
        name="weierstrass_p_definition",
        latex=(
            r"\wp(z;\, g_2, g_3) = \frac{{1}}{{z^2}}"
            r" + \sum{lim_mod}_{{(m,n)\neq(0,0)}} \left("
            r"\frac{{1}}{{(z - \omega_{{mn}})^2}} - \frac{{1}}{{\omega_{{mn}}^2}}"
            r"\right)"
        ),
        slots={
            "lim_mod": _LIM_MOD,
        },
    ),
    Template(
        name="weierstrass_p_ode",
        latex=r"(\wp')^2 = 4\wp^3 - g_2 \wp - g_3",
        slots={},
    ),
    Template(
        name="weierstrass_p_periodicity",
        latex=r"\wp(z + \omega_1) = \wp(z + \omega_2) = \wp(z)",
        slots={},
    ),
    Template(
        name="weierstrass_p_even",
        latex=r"\wp(-z) = \wp(z)",
        slots={},
    ),
    Template(
        name="complex_cartesian_imath",
        latex=r"z = x + \imath\, y,\quad \bar{{z}} = x - \imath\, y",
        slots={},
    ),
]

_TEMPLATES_F: list[Template] = [
    Template(
        name="modulus_Re_Im",
        latex=r"|{zz}|^2 = \Re({zz})^2 + \Im({zz})^2",
        slots={"zz": S(_VAR_POOL)},
    ),
    Template(
        name="real_part_inequality",
        latex=r"\Re({zz}) \leq |{zz}|",
        slots={"zz": S(_VAR_POOL)},
    ),
    Template(
        name="complex_conjugate_Re_Im",
        latex=r"\overline{{{zz}}} = \Re({zz}) - i\,\Im({zz})",
        slots={"zz": S(_VAR_POOL)},
    ),
    Template(
        name="imaginary_part_bound",
        latex=r"|\Im({zz})| \leq |{zz}|",
        slots={"zz": S(_VAR_POOL)},
    ),
]

_COMPLEX_TEMPLATES: list[Template] = (
    _TEMPLATES_A
    + _TEMPLATES_B1
    + _TEMPLATES_B2
    + _TEMPLATES_B3
    + _TEMPLATES_B4
    + _TEMPLATES_B5
    + _TEMPLATES_B6
    + _TEMPLATES_B7
    + _TEMPLATES_C
    + _TEMPLATES_D
    + _TEMPLATES_E
    + _TEMPLATES_F
)

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("complex_analysis", _COMPLEX_TEMPLATES)
