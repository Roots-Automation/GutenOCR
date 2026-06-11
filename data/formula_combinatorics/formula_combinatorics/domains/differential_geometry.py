"""Differential geometry domain generators."""

from __future__ import annotations

from .._template_dsl import _FN_SLOT, _LIM_MOD, S, Template, X
from .._vocab import _INDICES as _IDX_POOL
from ._config import register_domain

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_VF_POOL = ("X", "Y", "Z", "V", "W", "U", "T", "S", r"\xi", r"\eta")
_FORM_POOL = (r"\omega", r"\alpha", r"\beta", r"\eta", r"\theta", r"\phi", r"\psi", r"\chi", r"\rho", r"\sigma")
_MFLD_POOL = (
    "M",
    "N",
    r"\Sigma",
    r"\mathcal{M}",
    r"\mathcal{N}",
    "P",
    r"\mathcal{P}",
    r"\mathcal{S}",
    r"\Gamma",
    r"\Lambda",
)
_METRIC_POOL = ("g", "h", "k", r"\gamma", r"\hat{g}", r"\tilde{g}", r"\bar{g}", r"\mathring{g}")
_PARAM_POOL = (r"\tau", r"\lambda", "s", "t", r"\sigma", "u", r"\mu", r"\rho")
_BUNDLE_POOL = (
    "E",
    "L",
    r"\mathcal{E}",
    r"\mathcal{L}",
    r"\mathcal{F}",
    "F",
    r"\mathcal{G}",
    r"\mathcal{H}",
    r"\mathcal{V}",
    "S",
)
_NVEC_POOL = ("N", r"\nu", r"\mathbf{n}", r"\hat{N}", r"\hat{n}", r"\mathbf{e}_n", "n", r"\bar{N}")
_KAPPA_POOL = (r"\kappa", "0", "1", "-1")

# ---------------------------------------------------------------------------
# Templates  (fully flattened — no variant groups — to avoid birthday paradox
# from unbalanced group sampling)
# ---------------------------------------------------------------------------

_DIFFGEOM_TEMPLATES: list[Template] = [
    # ---- A1: metric tensor -------------------------------------------------
    Template(
        name="metric_einstein_sum",
        latex=r"ds^2 = {met}_{{{ii}{jj}}} \, dx^{{{ii}}} dx^{{{jj}}}",
        slots={"met": S(_METRIC_POOL), "ii": S(_IDX_POOL), "jj": X(_IDX_POOL, ("ii",))},
    ),
    Template(
        name="metric_inner_product",
        latex=r"\langle u, v \rangle_p = {met}_p(u, v)",
        slots={"met": S(_METRIC_POOL)},
    ),
    Template(
        name="metric_positive_definite",
        latex=r"[{met}_{{{ii}{jj}}}] \text{{ is symmetric positive-definite}}",
        slots={"met": S(_METRIC_POOL), "ii": S(_IDX_POOL), "jj": X(_IDX_POOL, ("ii",))},
    ),
    # ---- A2: Christoffel symbols -------------------------------------------
    Template(
        name="christoffel_def",
        latex=(
            r"\Gamma^{{{kk}}}_{{{ii}{jj}}} = \tfrac{{1}}{{2}} {met}^{{{kk}{ll}}}"
            r" \!\left(\partial_{{{ii}}} {met}_{{{jj}{ll}}}"
            r" + \partial_{{{jj}}} {met}_{{{ii}{ll}}}"
            r" - \partial_{{{ll}}} {met}_{{{ii}{jj}}}\right)"
        ),
        slots={
            "met": S(_METRIC_POOL),
            "kk": S(_IDX_POOL),
            "ii": X(_IDX_POOL, ("kk",)),
            "jj": X(_IDX_POOL, ("kk", "ii")),
            "ll": X(_IDX_POOL, ("kk", "ii", "jj")),
        },
    ),
    Template(
        name="christoffel_symmetry",
        latex=r"\Gamma^{{{kk}}}_{{{ii}{jj}}} = \Gamma^{{{kk}}}_{{{jj}{ii}}}",
        slots={
            "kk": S(_IDX_POOL),
            "ii": X(_IDX_POOL, ("kk",)),
            "jj": X(_IDX_POOL, ("kk", "ii")),
        },
    ),
    # ---- A3: geodesic equation ---------------------------------------------
    Template(
        name="geodesic_affine",
        latex=(
            r"\frac{{d^2 x^{{{kk}}}}}{{d{par}^2}}"
            r" + \Gamma^{{{kk}}}_{{{ii}{jj}}}"
            r" \frac{{dx^{{{ii}}}}}{{d{par}}} \frac{{dx^{{{jj}}}}}{{d{par}}} = 0"
        ),
        slots={
            "par": S(_PARAM_POOL),
            "kk": S(_IDX_POOL),
            "ii": X(_IDX_POOL, ("kk",)),
            "jj": X(_IDX_POOL, ("kk", "ii")),
        },
    ),
    Template(
        name="geodesic_abstract",
        latex=r"\nabla_{{\dot{{\gamma}}}} \dot{{\gamma}} = 0",
        slots={},
    ),
    # ---- A4: Riemann curvature ---------------------------------------------
    Template(
        name="riemann_component",
        latex=(
            r"R^{{{ll}}}_{{{kk}{ii}{jj}}}"
            r" = \partial_{{{ii}}} \Gamma^{{{ll}}}_{{{jj}{kk}}}"
            r" - \partial_{{{jj}}} \Gamma^{{{ll}}}_{{{ii}{kk}}}"
            r" + \Gamma^{{{ll}}}_{{{ii}{mm}}} \Gamma^{{{mm}}}_{{{jj}{kk}}}"
            r" - \Gamma^{{{ll}}}_{{{jj}{mm}}} \Gamma^{{{mm}}}_{{{ii}{kk}}}"
        ),
        slots={
            "ll": S(_IDX_POOL),
            "kk": X(_IDX_POOL, ("ll",)),
            "ii": X(_IDX_POOL, ("ll", "kk")),
            "jj": X(_IDX_POOL, ("ll", "kk", "ii")),
            "mm": X(_IDX_POOL, ("ll", "kk", "ii", "jj")),
        },
    ),
    Template(
        name="riemann_operator",
        latex=(
            r"R({xx}, {yy}){zz}"
            r" = \nabla_{{{xx}}} \nabla_{{{yy}}} {zz}"
            r" - \nabla_{{{yy}}} \nabla_{{{xx}}} {zz}"
            r" - \nabla_{{[{xx},{yy}]}} {zz}"
        ),
        slots={
            "xx": S(_VF_POOL),
            "yy": X(_VF_POOL, ("xx",)),
            "zz": X(_VF_POOL, ("xx", "yy")),
        },
    ),
    Template(
        name="riemann_antisymmetry",
        latex=(
            r"R_{{{ii}{jj}{kk}{ll}}}"
            r" = -R_{{{jj}{ii}{kk}{ll}}}"
            r" = -R_{{{ii}{jj}{ll}{kk}}}"
        ),
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
            "ll": X(_IDX_POOL, ("ii", "jj", "kk")),
        },
    ),
    # ---- A5: torsion-free (balanced [30,30] — kept as group) ---------------
    Template(
        name="torsion_free",
        latex="",
        slots={},
        variants=[
            Template(
                name="torsion_zero",
                latex=(
                    r"T({xx}, {yy})"
                    r" = \nabla_{{{xx}}} {yy} - \nabla_{{{yy}}} {xx}"
                    r" - [{xx}, {yy}] = 0"
                ),
                slots={"xx": S(_VF_POOL), "yy": X(_VF_POOL, ("xx",))},
            ),
            Template(
                name="torsion_free_lc",
                latex=r"\nabla_{{{xx}}} {yy} - \nabla_{{{yy}}} {xx} = [{xx}, {yy}]",
                slots={"xx": S(_VF_POOL), "yy": X(_VF_POOL, ("xx",))},
            ),
        ],
    ),
    # ---- A6: exterior derivative squared -----------------------------------
    Template(
        name="d_squared_zero",
        latex=r"d(d{om}) = 0",
        slots={"om": S(_FORM_POOL)},
    ),
    Template(
        name="d_squared_operator",
        latex=r"d^2 = 0",
        slots={},
    ),
    # ---- A7: Stokes theorem (balanced [30,30,30] — kept as group) ----------
    Template(
        name="stokes_theorem",
        latex="",
        slots={},
        variants=[
            Template(
                name="stokes_general",
                latex=r"\int{lim_mod}_{{{mfld}}} d{om} = \int{lim_mod}_{{\partial {mfld}}} {om}",
                slots={"lim_mod": _LIM_MOD, "mfld": S(_MFLD_POOL), "om": S(_FORM_POOL)},
            ),
            Template(
                name="stokes_oint",
                latex=r"\int{lim_mod}_{{{mfld}}} d{om} = \oint{lim_mod}_{{\partial {mfld}}} {om}",
                slots={"lim_mod": _LIM_MOD, "mfld": S(_MFLD_POOL), "om": S(_FORM_POOL)},
            ),
            Template(
                name="stokes_no_boundary",
                latex=r"\partial {mfld} = \emptyset \implies \int{lim_mod}_{{{mfld}}} d{om} = 0",
                slots={"lim_mod": _LIM_MOD, "mfld": S(_MFLD_POOL), "om": S(_FORM_POOL)},
            ),
        ],
    ),
    # ---- A8: Cartan magic formula ------------------------------------------
    Template(
        name="cartan_lie_deriv",
        latex=(
            r"\mathcal{{L}}_{{{xx}}} {om}"
            r" = d(\iota_{{{xx}}} {om}) + \iota_{{{xx}}} d{om}"
        ),
        slots={"xx": S(_VF_POOL), "om": S(_FORM_POOL)},
    ),
    Template(
        name="cartan_commutator",
        latex=r"[\mathcal{{L}}_{{{xx}}}, d] = 0",
        slots={"xx": S(_VF_POOL)},
    ),
    # ---- A9: Gaussian curvature --------------------------------------------
    Template(
        name="gaussian_component",
        latex=(
            r"K = \frac{{R_{{1212}}}}"
            r"{{{{{met}}}_{{11}} {met}_{{22}} - {met}_{{12}}^2}}"
        ),
        slots={"met": S(_METRIC_POOL)},
    ),
    Template(
        name="gaussian_principal",
        latex=r"K = \kappa_1 \kappa_2",
        slots={},
    ),
    Template(
        name="gaussian_ricci_surface",
        latex=r"K = \tfrac{{1}}{{2}} R",
        slots={},
    ),
    # ---- A10: Ricci tensor -------------------------------------------------
    Template(
        name="ricci_contraction",
        latex=r"R_{{{ii}{jj}}} = R^{{{kk}}}_{{{ii}{kk}{jj}}}",
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
        },
    ),
    Template(
        name="ricci_symmetric",
        latex=r"R_{{{ii}{jj}}} = R_{{{jj}{ii}}}",
        slots={"ii": S(_IDX_POOL), "jj": X(_IDX_POOL, ("ii",))},
    ),
    Template(
        name="ricci_abstract",
        latex=(
            r"\operatorname{{Ric}}({xx}, {yy})"
            r" = \operatorname{{tr}}\!\bigl(Z \mapsto R(Z, {xx})\,{yy}\bigr)"
        ),
        slots={"xx": S(_VF_POOL), "yy": X(_VF_POOL, ("xx",))},
    ),
    # ---- A11: Einstein field equations -------------------------------------
    Template(
        name="einstein_index",
        latex=(
            r"G_{{{ii}{jj}}} = R_{{{ii}{jj}}}"
            r" - \tfrac{{1}}{{2}} {met}_{{{ii}{jj}}} R"
            r" = \frac{{8\pi G}}{{c^4}} T_{{{ii}{jj}}}"
        ),
        slots={
            "met": S(_METRIC_POOL),
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
    Template(
        name="einstein_tensor_form",
        latex=(
            r"\mathbf{{G}} = \mathbf{{R}}"
            r" - \tfrac{{1}}{{2}} R\,\mathbf{{g}}"
            r" = \frac{{8\pi G}}{{c^4}} \mathbf{{T}}"
        ),
        slots={},
    ),
    Template(
        name="einstein_trace_reversed",
        latex=(
            r"R_{{{ii}{jj}}} = \frac{{8\pi G}}{{c^4}}"
            r"\!\left(T_{{{ii}{jj}}}"
            r" - \tfrac{{1}}{{2}} {met}_{{{ii}{jj}}} T\right)"
        ),
        slots={
            "met": S(_METRIC_POOL),
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
    # ---- A12: covariant derivative of tensor --------------------------------
    Template(
        name="cov_deriv_contravariant",
        latex=(
            r"\nabla_{{{kk}}} T^{{{ii}{jj}}}"
            r" = \partial_{{{kk}}} T^{{{ii}{jj}}}"
            r" + \Gamma^{{{ii}}}_{{{kk}{ll}}} T^{{{ll}{jj}}}"
            r" + \Gamma^{{{jj}}}_{{{kk}{ll}}} T^{{{ii}{ll}}}"
        ),
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
            "ll": X(_IDX_POOL, ("ii", "jj", "kk")),
        },
    ),
    Template(
        name="cov_deriv_covariant",
        latex=(
            r"\nabla_{{{kk}}} T_{{{ii}{jj}}}"
            r" = \partial_{{{kk}}} T_{{{ii}{jj}}}"
            r" - \Gamma^{{{ll}}}_{{{kk}{ii}}} T_{{{ll}{jj}}}"
            r" - \Gamma^{{{ll}}}_{{{kk}{jj}}} T_{{{ii}{ll}}}"
        ),
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
            "ll": X(_IDX_POOL, ("ii", "jj", "kk")),
        },
    ),
    Template(
        name="cov_deriv_vector",
        latex=(
            r"\nabla_{{{kk}}} V^{{{ii}}}"
            r" = \partial_{{{kk}}} V^{{{ii}}}"
            r" + \Gamma^{{{ii}}}_{{{kk}{jj}}} V^{{{jj}}}"
        ),
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
        },
    ),
    # ---- A13: Lie bracket --------------------------------------------------
    Template(
        name="lie_bracket_component",
        latex=(
            r"[{xx}, {yy}]^{{{ii}}}"
            r" = {xx}^{{{jj}}} \partial_{{{jj}}} {yy}^{{{ii}}}"
            r" - {yy}^{{{jj}}} \partial_{{{jj}}} {xx}^{{{ii}}}"
        ),
        slots={
            "xx": S(_VF_POOL),
            "yy": X(_VF_POOL, ("xx",)),
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
    Template(
        name="lie_bracket_functional",
        latex=r"[{xx}, {yy}] f = {xx}({yy} f) - {yy}({xx} f)",
        slots={"xx": S(_VF_POOL), "yy": X(_VF_POOL, ("xx",))},
    ),
    # ---- A14: Gauss-Bonnet (balanced [5,5] — kept as group) ----------------
    Template(
        name="gauss_bonnet",
        latex="",
        slots={},
        variants=[
            Template(
                name="gauss_bonnet_boundary",
                latex=(
                    r"\int{lim_mod}_{{{mfld}}} K \, dA"
                    r" + \int{lim_mod}_{{\partial {mfld}}} \kappa_g \, ds"
                    r" = 2\pi \chi({mfld})"
                ),
                slots={"lim_mod": _LIM_MOD, "mfld": S(_MFLD_POOL)},
            ),
            Template(
                name="gauss_bonnet_closed",
                latex=r"\int{lim_mod}_{{{mfld}}} K \, dA = 2\pi \chi({mfld})",
                slots={"lim_mod": _LIM_MOD, "mfld": S(_MFLD_POOL)},
            ),
        ],
    ),
    # ---- B1: differential forms algebra ------------------------------------
    Template(
        name="wedge_anticommutativity",
        latex=r"{om} \wedge {al} = (-1)^{{pq}} {al} \wedge {om}",
        slots={"om": S(_FORM_POOL), "al": X(_FORM_POOL, ("om",))},
    ),
    Template(
        name="wedge_associativity",
        latex=(
            r"({om} \wedge {al}) \wedge {bt}"
            r" = {om} \wedge ({al} \wedge {bt})"
        ),
        slots={
            "om": S(_FORM_POOL),
            "al": X(_FORM_POOL, ("om",)),
            "bt": X(_FORM_POOL, ("om", "al")),
        },
    ),
    Template(
        name="wedge_distributivity",
        latex=(
            r"({om} + {al}) \wedge {bt}"
            r" = {om} \wedge {bt} + {al} \wedge {bt}"
        ),
        slots={
            "om": S(_FORM_POOL),
            "al": X(_FORM_POOL, ("om",)),
            "bt": X(_FORM_POOL, ("om", "al")),
        },
    ),
    Template(
        name="exterior_derivative_product",
        latex=(
            r"d({om} \wedge {al})"
            r" = d{om} \wedge {al} + (-1)^k {om} \wedge d{al}"
        ),
        slots={"om": S(_FORM_POOL), "al": X(_FORM_POOL, ("om",))},
    ),
    # ---- B2: pullback / pushforward ----------------------------------------
    Template(
        name="pullback_product",
        latex=(
            r"{fn1}^*({om} \wedge {al})"
            r" = {fn1}^* {om} \wedge {fn1}^* {al}"
        ),
        slots={
            "fn1": _FN_SLOT,
            "om": S(_FORM_POOL),
            "al": X(_FORM_POOL, ("om",)),
        },
    ),
    Template(
        name="pullback_form",
        latex=r"({fn1}^* {om})_p(v) = {om}_{{{fn1}(p)}}(d{fn1}_p \cdot v)",
        slots={"fn1": _FN_SLOT, "om": S(_FORM_POOL)},
    ),
    Template(
        name="pushforward_vector",
        latex=r"({fn1}_* {xx})_q = d{fn1}_p \cdot {xx}_p",
        slots={"fn1": _FN_SLOT, "xx": S(_VF_POOL)},
    ),
    # ---- B3: covariant derivative identities --------------------------------
    Template(
        name="cov_deriv_product_rule",
        latex=(
            r"\nabla_{{{xx}}} ({fn1} {yy})"
            r" = ({xx}\,{fn1})\,{yy} + {fn1}\,\nabla_{{{xx}}} {yy}"
        ),
        slots={
            "xx": S(_VF_POOL),
            "yy": X(_VF_POOL, ("xx",)),
            "fn1": _FN_SLOT,
        },
    ),
    Template(
        name="metric_compatibility",
        latex=(
            r"\nabla_{{{zz}}} \langle {xx}, {yy} \rangle"
            r" = \langle \nabla_{{{zz}}} {xx}, {yy} \rangle"
            r" + \langle {xx}, \nabla_{{{zz}}} {yy} \rangle"
        ),
        slots={
            "xx": S(_VF_POOL),
            "yy": X(_VF_POOL, ("xx",)),
            "zz": X(_VF_POOL, ("xx", "yy")),
        },
    ),
    Template(
        name="curvature_operator_defn",
        latex=(
            r"R({xx}, {yy}){zz}"
            r" = \nabla_{{{xx}}} \nabla_{{{yy}}} {zz}"
            r" - \nabla_{{{yy}}} \nabla_{{{xx}}} {zz}"
            r" - \nabla_{{[{xx},{yy}]}} {zz}"
        ),
        slots={
            "xx": S(_VF_POOL),
            "yy": X(_VF_POOL, ("xx",)),
            "zz": X(_VF_POOL, ("xx", "yy")),
        },
    ),
    # ---- B4: parallel transport --------------------------------------------
    Template(
        name="parallel_abstract",
        latex=r"\frac{{D{vv}}}{{d{par}}} = 0",
        slots={"vv": S(_VF_POOL), "par": S(_PARAM_POOL)},
    ),
    Template(
        name="parallel_component",
        latex=(
            r"\frac{{dV^{{{kk}}}}}{{d{par}}}"
            r" + \Gamma^{{{kk}}}_{{{ii}{jj}}}"
            r" \frac{{dx^{{{ii}}}}}{{d{par}}} V^{{{jj}}} = 0"
        ),
        slots={
            "par": S(_PARAM_POOL),
            "kk": S(_IDX_POOL),
            "ii": X(_IDX_POOL, ("kk",)),
            "jj": X(_IDX_POOL, ("kk", "ii")),
        },
    ),
    # ---- B5: sectional curvature -------------------------------------------
    Template(
        name="sectional_ratio",
        latex=(
            r"K(\sigma) = \frac{{R({xx}, {yy}, {yy}, {xx})}}"
            r"{{|{xx}|^2 |{yy}|^2 - \langle {xx},{yy} \rangle^2}}"
        ),
        slots={"xx": S(_VF_POOL), "yy": X(_VF_POOL, ("xx",))},
    ),
    Template(
        name="sectional_constant",
        latex=r"K \equiv {kap}",
        slots={"kap": S(_KAPPA_POOL)},
    ),
    Template(
        name="sectional_model_spaces",
        latex=(
            r"K_{{S^n}} = +1,"
            r" \quad K_{{\mathbb{{R}}^n}} = 0,"
            r" \quad K_{{\mathbb{{H}}^n}} = -1"
        ),
        slots={},
    ),
    # ---- B6: Ricci scalar --------------------------------------------------
    Template(
        name="ricci_scalar_trace",
        latex=r"R = {met}^{{{ii}{jj}}} R_{{{ii}{jj}}}",
        slots={
            "met": S(_METRIC_POOL),
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
    Template(
        name="ricci_scalar_sectional",
        latex=r"R = \sum{lim_mod}_{{{ii} < {jj}}} 2\, K(e_{{{ii}}}, e_{{{jj}}})",
        slots={"lim_mod": _LIM_MOD, "ii": S(_IDX_POOL), "jj": X(_IDX_POOL, ("ii",))},
    ),
    # ---- B7: Hodge star ----------------------------------------------------
    Template(
        name="hodge_pairing",
        latex=(
            r"{om} \wedge \star {al}"
            r" = \langle {om}, {al} \rangle \, \operatorname{{vol}}"
        ),
        slots={"om": S(_FORM_POOL), "al": X(_FORM_POOL, ("om",))},
    ),
    Template(
        name="hodge_involution",
        latex=r"\star \star {om} = (-1)^{{k(n-k)}} {om}",
        slots={"om": S(_FORM_POOL)},
    ),
    Template(
        name="codifferential_def",
        latex=r"\delta {om} = (-1)^{{n(k+1)+1}} \star d \star {om}",
        slots={"om": S(_FORM_POOL)},
    ),
    # ---- B8: Hodge decomposition / Laplace-Beltrami ------------------------
    Template(
        name="hodge_decomposition",
        latex=r"{om} = d{al} + \delta {bt} + {et}",
        slots={
            "om": S(_FORM_POOL),
            "al": X(_FORM_POOL, ("om",)),
            "bt": X(_FORM_POOL, ("om", "al")),
            "et": X(_FORM_POOL, ("om", "al", "bt")),
        },
    ),
    Template(
        name="laplace_beltrami_def",
        latex=r"\Delta {om} = (d\delta + \delta d){om}",
        slots={"om": S(_FORM_POOL)},
    ),
    # ---- B9: exponential map -----------------------------------------------
    Template(
        name="exp_map_def",
        latex=(
            r"\exp_p \colon T_p M \to M,"
            r" \quad {vv} \mapsto \gamma_{{{vv}}}(1)"
        ),
        slots={"vv": S(_VF_POOL)},
    ),
    Template(
        name="exp_map_geodesic",
        latex=(
            r"\exp_p({par}\,{vv}) = \gamma({par}),"
            r" \quad \gamma'(0) = {vv}"
        ),
        slots={"vv": S(_VF_POOL), "par": S(_PARAM_POOL)},
    ),
    # ---- B10b: high-n_eff function-pair templates ---------------------------
    Template(
        name="green_second_identity",
        latex=(
            r"\int{lim_mod}_{{{mfld}}} ({fn1} \Delta {fn2} - {fn2} \Delta {fn1})"
            r" \, \operatorname{{vol}} = 0"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "mfld": S(_MFLD_POOL),
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
        },
    ),
    Template(
        name="laplacian_leibniz",
        latex=(
            r"\Delta({fn1} {fn2})"
            r" = {fn1} \Delta {fn2} + {fn2} \Delta {fn1}"
            r" + 2 \langle \nabla {fn1}, \nabla {fn2} \rangle_{{{met}}}"
        ),
        slots={
            "met": S(_METRIC_POOL),
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
        },
    ),
    Template(
        name="integration_by_parts_manifold",
        latex=(
            r"\int{lim_mod}_{{{mfld}}} {fn1}\, \operatorname{{div}}({xx})\, \operatorname{{vol}}"
            r" = -\int{lim_mod}_{{{mfld}}} {met}(\operatorname{{grad}} {fn1}, {xx})\, \operatorname{{vol}}"
            r" + \int{lim_mod}_{{\partial {mfld}}} {fn1} \langle {xx}, \nu \rangle \, dA"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "mfld": S(_MFLD_POOL),
            "fn1": _FN_SLOT,
            "xx": S(_VF_POOL),
            "met": S(_METRIC_POOL),
        },
    ),
    Template(
        name="composition_pullback",
        latex=(
            r"({fn1} \circ {fn2})^* {om}"
            r" = {fn2}^*\!\left({fn1}^* {om}\right)"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "om": S(_FORM_POOL),
        },
    ),
    Template(
        name="green_formula_closed",
        latex=(
            r"\int{lim_mod}_{{{mfld}}} {fn1} \Delta {fn2} \, \operatorname{{vol}}"
            r" = -\int{lim_mod}_{{{mfld}}} \langle \nabla {fn1}, \nabla {fn2}"
            r" \rangle_{{{met}}} \, \operatorname{{vol}}"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "mfld": S(_MFLD_POOL),
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "met": S(_METRIC_POOL),
        },
    ),
    Template(
        name="exterior_deriv_product_fn",
        latex=r"d({fn1} \cdot {fn2}) = {fn2} \, d{fn1} + {fn1} \, d{fn2}",
        slots={"fn1": _FN_SLOT, "fn2": _FN_SLOT},
    ),
    Template(
        name="gradient_product_rule",
        latex=(
            r"\operatorname{{grad}}_{{{met}}}({fn1} {fn2})"
            r" = {fn1}\, \operatorname{{grad}}_{{{met}}} {fn2}"
            r" + {fn2}\, \operatorname{{grad}}_{{{met}}} {fn1}"
        ),
        slots={
            "met": S(_METRIC_POOL),
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
        },
    ),
    # ---- B10: Lie derivative (split: fn×fn variant dominates) --------------
    Template(
        name="lie_deriv_function",
        latex=(
            r"\mathcal{{L}}_{{{xx}}} ({fn1} \cdot {fn2})"
            r" = (\mathcal{{L}}_{{{xx}}} {fn1})\,{fn2}"
            r" + {fn1}\,(\mathcal{{L}}_{{{xx}}} {fn2})"
        ),
        slots={
            "xx": S(_VF_POOL),
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
        },
    ),
    Template(
        name="lie_deriv_bracket",
        latex=r"\mathcal{{L}}_{{{xx}}} {yy} = [{xx}, {yy}]",
        slots={"xx": S(_VF_POOL), "yy": X(_VF_POOL, ("xx",))},
    ),
    Template(
        name="lie_deriv_form",
        latex=(
            r"\mathcal{{L}}_{{{xx}}} {om}"
            r" = d(\iota_{{{xx}}} {om}) + \iota_{{{xx}}} d{om}"
        ),
        slots={"xx": S(_VF_POOL), "om": S(_FORM_POOL)},
    ),
    # ---- B11: symplectic form ----------------------------------------------
    Template(
        name="symplectic_closed",
        latex=r"d{om} = 0",
        slots={"om": S(_FORM_POOL)},
    ),
    Template(
        name="symplectic_nondegenerate",
        latex=r"{om}({xx}, {yy}) = 0 \;\forall {yy} \implies {xx} = 0",
        slots={
            "om": S(_FORM_POOL),
            "xx": S(_VF_POOL),
            "yy": X(_VF_POOL, ("xx",)),
        },
    ),
    Template(
        name="darboux_theorem",
        latex=r"{om} = \sum{lim_mod}_{{i=1}}^n dp_i \wedge dq_i",
        slots={"lim_mod": _LIM_MOD, "om": S(_FORM_POOL)},
    ),
    # ---- B12: de Rham cohomology -------------------------------------------
    Template(
        name="closed_form_defn",
        latex=r"d{om} = 0 \quad ({om} \text{{ is closed}})",
        slots={"om": S(_FORM_POOL)},
    ),
    Template(
        name="exact_form_defn",
        latex=r"{om} = d{al} \quad ({om} \text{{ is exact}})",
        slots={"om": S(_FORM_POOL), "al": X(_FORM_POOL, ("om",))},
    ),
    Template(
        name="poincare_lemma",
        latex=r"H^k_{{\mathrm{{dR}}}}(\mathbb{{R}}^n) = 0 \text{{ for }} k > 0",
        slots={},
    ),
    # ---- B13: second fundamental form -------------------------------------
    Template(
        name="second_ff_def",
        latex=(
            r"II({xx}, {yy})"
            r" = -\langle \nabla_{{{xx}}} {yy}, {nvec} \rangle"
        ),
        slots={
            "xx": S(_VF_POOL),
            "yy": X(_VF_POOL, ("xx",)),
            "nvec": S(_NVEC_POOL),
        },
    ),
    Template(
        name="shape_operator",
        latex=r"II({xx}, {yy}) = \langle S({xx}), {yy} \rangle",
        slots={"xx": S(_VF_POOL), "yy": X(_VF_POOL, ("xx",))},
    ),
    Template(
        name="weingarten_map",
        latex=r"\nabla_{{{xx}}} {nvec} = -S({xx})",
        slots={"xx": S(_VF_POOL), "nvec": S(_NVEC_POOL)},
    ),
    # ---- B14: mean / Gaussian curvature (both n_eff=1, balanced) -----------
    Template(
        name="mean_and_principal_curvature",
        latex="",
        slots={},
        variants=[
            Template(
                name="mean_curvature",
                latex=(
                    r"H = \tfrac{{1}}{{2}}(\kappa_1 + \kappa_2)"
                    r" = \tfrac{{1}}{{2}} \operatorname{{tr}}(S)"
                ),
                slots={},
            ),
            Template(
                name="gaussian_via_shape",
                latex=r"K = \kappa_1 \kappa_2 = \det(S)",
                slots={},
            ),
        ],
    ),
    # ---- B15: Bianchi identities (balanced [360,360]) ----------------------
    Template(
        name="bianchi_identities",
        latex="",
        slots={},
        variants=[
            Template(
                name="first_bianchi",
                latex=(
                    r"R^{{{ll}}}_{{{kk}{ii}{jj}}}"
                    r" + R^{{{ll}}}_{{{jj}{kk}{ii}}}"
                    r" + R^{{{ll}}}_{{{ii}{jj}{kk}}} = 0"
                ),
                slots={
                    "ll": S(_IDX_POOL),
                    "kk": X(_IDX_POOL, ("ll",)),
                    "ii": X(_IDX_POOL, ("ll", "kk")),
                    "jj": X(_IDX_POOL, ("ll", "kk", "ii")),
                },
            ),
            Template(
                name="second_bianchi",
                latex=(
                    r"\nabla_{{{ll}}} R_{{{kk}{ii}{jj}}}"
                    r" + \nabla_{{{ii}}} R_{{{kk}{jj}{ll}}}"
                    r" + \nabla_{{{jj}}} R_{{{kk}{ll}{ii}}} = 0"
                ),
                slots={
                    "ll": S(_IDX_POOL),
                    "kk": X(_IDX_POOL, ("ll",)),
                    "ii": X(_IDX_POOL, ("ll", "kk")),
                    "jj": X(_IDX_POOL, ("ll", "kk", "ii")),
                },
            ),
        ],
    ),
    # ---- B16: Ricci flow (balanced [90,90]) --------------------------------
    Template(
        name="ricci_flow",
        latex="",
        slots={},
        variants=[
            Template(
                name="ricci_flow_eq",
                latex=(
                    r"\frac{{\partial {met}_{{{ii}{jj}}}}}{{\partial t}}"
                    r" = -2 R_{{{ii}{jj}}}"
                ),
                slots={
                    "met": S(_METRIC_POOL),
                    "ii": S(_IDX_POOL),
                    "jj": X(_IDX_POOL, ("ii",)),
                },
            ),
            Template(
                name="ricci_flow_normalized",
                latex=(
                    r"\frac{{\partial {met}_{{{ii}{jj}}}}}{{\partial t}}"
                    r" = -2 R_{{{ii}{jj}}}"
                    r" + \frac{{2}}{{n}} R\,{met}_{{{ii}{jj}}}"
                ),
                slots={
                    "met": S(_METRIC_POOL),
                    "ii": S(_IDX_POOL),
                    "jj": X(_IDX_POOL, ("ii",)),
                },
            ),
        ],
    ),
    # ---- B17: volume form --------------------------------------------------
    Template(
        name="volume_riemannian",
        latex=(
            r"\operatorname{{vol}}_{{{met}}}"
            r" = \sqrt{{\lvert \det [{met}_{{{ii}{jj}}}] \rvert}}"
            r" \, dx^1 \wedge \cdots \wedge dx^n"
        ),
        slots={
            "met": S(_METRIC_POOL),
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
    Template(
        name="volume_abstract",
        latex=(
            r"\omega_{{{mfld}}} \in \Omega^n({mfld})"
            r" \text{{ is a nowhere-vanishing top form}}"
        ),
        slots={"mfld": S(_MFLD_POOL)},
    ),
    # ---- B18: fiber bundle curvature (balanced [30,25]) --------------------
    Template(
        name="fiber_bundle_curvature",
        latex="",
        slots={},
        variants=[
            Template(
                name="curvature_2form",
                latex=(
                    r"F_\nabla = d{om} + {om} \wedge {om}"
                    r" \in \Omega^2(\operatorname{{End}}({bun}))"
                ),
                slots={"om": S(_FORM_POOL), "bun": S(_BUNDLE_POOL)},
            ),
            Template(
                name="chern_weil",
                latex=(
                    r"c_1({bun}) = \left[\frac{{i}}{{2\pi}} F_\nabla\right]"
                    r" \in H^2({mfld};\, \mathbb{{Z}})"
                ),
                slots={"bun": S(_BUNDLE_POOL), "mfld": S(_MFLD_POOL)},
            ),
        ],
    ),
    # ---- B19: Gauss-Codazzi ------------------------------------------------
    Template(
        name="gauss_equation",
        latex=(
            r"R({xx},{yy},{zz},{ww})"
            r" = \bar{{R}}({xx},{yy},{zz},{ww})"
            r" + II({xx},{ww})\,II({yy},{zz})"
            r" - II({xx},{zz})\,II({yy},{ww})"
        ),
        slots={
            "xx": S(_VF_POOL),
            "yy": X(_VF_POOL, ("xx",)),
            "zz": X(_VF_POOL, ("xx", "yy")),
            "ww": X(_VF_POOL, ("xx", "yy", "zz")),
        },
    ),
    Template(
        name="codazzi_mainardi",
        latex=(
            r"(\nabla_{{{xx}}} II)({yy},{zz})"
            r" = (\nabla_{{{yy}}} II)({xx},{zz})"
        ),
        slots={
            "xx": S(_VF_POOL),
            "yy": X(_VF_POOL, ("xx",)),
            "zz": X(_VF_POOL, ("xx", "yy")),
        },
    ),
    # ---- B20: connection forms ---------------------------------------------
    Template(
        name="cartan_structure_eq",
        latex=(
            r"d\omega^{{{ii}}}"
            r" = -\omega^{{{ii}}}{{}}_{{{jj}}} \wedge \omega^{{{jj}}}"
        ),
        slots={"ii": S(_IDX_POOL), "jj": X(_IDX_POOL, ("ii",))},
    ),
    Template(
        name="curvature_form_eq",
        latex=(
            r"\Omega^{{{ii}}}{{}}_{{{jj}}}"
            r" = d\omega^{{{ii}}}{{}}_{{{jj}}}"
            r" + \omega^{{{ii}}}{{}}_{{{kk}}} \wedge \omega^{{{kk}}}{{}}_{{{jj}}}"
        ),
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
        },
    ),
]

# Part C additions — 8 fn-pair templates
_DIFFGEOM_TEMPLATES += [
    Template(
        name="fn_metric_contraction",
        latex=(
            r"{fn1}\!\left({met}^{{{ii}{jj}}} T_{{{ii}{jj}}}\right)"
            r" = {fn2}\!\left(\operatorname{{tr}}_{{{met}}} T\right)"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "met": S(_METRIC_POOL),
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
    Template(
        name="fn_curvature_trace",
        latex=(
            r"{fn1}(R_{{{ii}{jj}}})"
            r" = {fn2}\!\left({met}^{{{kk}{ll}}} R_{{{kk}{ii}{ll}{jj}}}\right)"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "met": S(_METRIC_POOL),
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
            "ll": X(_IDX_POOL, ("ii", "jj", "kk")),
        },
    ),
    Template(
        name="fn_lie_bracket",
        latex=r"{fn1}([{xx},{yy}]) = {fn2}({xx} {yy} - {yy} {xx})",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "xx": S(_VF_POOL),
            "yy": X(_VF_POOL, ("xx",)),
        },
    ),
    Template(
        name="fn_exterior_product",
        latex=r"{fn1}({om} \wedge {al}) = {fn2}((-1)^{{pq}}\,{al} \wedge {om})",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "om": S(_FORM_POOL),
            "al": X(_FORM_POOL, ("om",)),
        },
    ),
    Template(
        name="fn_hodge_dual",
        latex=r"{fn1}(\star {om}) = {fn2}\!\left(\langle {om}, {al} \rangle\,\mathrm{{vol}}\right)",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "om": S(_FORM_POOL),
            "al": X(_FORM_POOL, ("om",)),
        },
    ),
    Template(
        name="fn_geodesic_deviation",
        latex=(
            r"{fn1}\!\left(\frac{{D^2 {xx}}}{{d{par}^2}}\right)"
            r" = {fn2}(R({xx}, \dot{{\gamma}})\dot{{\gamma}})"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "xx": S(_VF_POOL),
            "par": S(_PARAM_POOL),
        },
    ),
    Template(
        name="fn_connection_form",
        latex=(
            r"{fn1}(\nabla_{{{xx}}} {yy})"
            r" = {fn2}\!\left({xx}({yy}) + \omega({xx})\,{yy}\right)"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "xx": S(_VF_POOL),
            "yy": X(_VF_POOL, ("xx",)),
        },
    ),
    Template(
        name="fn_holonomy",
        latex=(
            r"{fn1}(\operatorname{{Hol}}_p(\nabla))"
            r" = {fn2}\!\left(\bigl\{{P_\gamma : \gamma \in \Omega_p({mfld})\bigr\}}\right)"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "mfld": S(_MFLD_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Sampling weights
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Geodesic curvature templates (\varkappa)
# ---------------------------------------------------------------------------

_DIFFGEOM_TEMPLATES += [
    Template(
        name="geodesic_curvature_def",
        latex=(
            r"\varkappa_g(\gamma)"
            r" = \left\langle \nabla_{{\dot{{\gamma}}}} \dot{{\gamma}},\, \mathbf{{n}} \right\rangle"
        ),
        slots={},
    ),
    Template(
        name="gauss_bonnet_boundary_iint",
        latex=(
            r"\iint_{{{mm}}} K \, dA"
            r" + \int{lim_mod}_{{\partial {mm}}} \varkappa_g \, ds"
            r" = 2\pi \chi({mm})"
        ),
        slots={"lim_mod": _LIM_MOD, "mm": S(_MFLD_POOL)},
    ),
    Template(
        name="geodesic_curvature_signed",
        latex=r"\varkappa_g = \frac{{d\theta}}{{ds}} + \frac{{d\phi}}{{ds}}",
        slots={},
    ),
    Template(
        name="geodesic_curvature_covariant",
        latex=(
            r"\varkappa_g = {met}_{{{ii}{jj}}}\,"
            r"\dot{{\gamma}}^{{{ii}}} \nabla_{{\dot{{\gamma}}}} \dot{{\gamma}}^{{{jj}}}"
        ),
        slots={
            "met": S(_METRIC_POOL),
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
]

_DIFFGEOM_TEMPLATES += [
    # \pitchfork — transversality
    Template(
        name="transversality_notation",
        latex=r"{mm} \pitchfork {nn}",
        slots={"mm": S(_MFLD_POOL), "nn": S(_MFLD_POOL)},
        distinct=[["mm", "nn"]],
    ),
    Template(
        name="transversality_preimage",
        latex=r"f \pitchfork {nn} \Rightarrow f^{{-1}}({nn}) \text{{ is a submanifold}}",
        slots={"nn": S(_MFLD_POOL)},
    ),
    # \sharp, \flat — musical isomorphisms (metric-induced index raising/lowering)
    Template(
        name="sharp_index_raising",
        latex=r"{al}^{{\sharp}} = {met}^{{{ii}{jj}}} {al}_{{{jj}}} \partial_{{{ii}}}",
        slots={
            "al": S(_FORM_POOL),
            "met": S(_METRIC_POOL),
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
    Template(
        name="flat_index_lowering",
        latex=r"{vf}^{{\flat}} = {met}_{{{ii}{jj}}} {vf}^{{{ii}}} dx^{{{jj}}}",
        slots={
            "vf": S(_VF_POOL),
            "met": S(_METRIC_POOL),
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
    Template(
        name="sharp_flat_inverse",
        latex=r"({al}^{{\sharp}})^{{\flat}} = {al},\quad ({vf}^{{\flat}})^{{\sharp}} = {vf}",
        slots={"al": S(_FORM_POOL), "vf": S(_VF_POOL)},
    ),
    # \natural — natural map / canonical projection
    Template(
        name="natural_map_projection",
        latex=r"\pi^{{\natural}} : T^*{mm} \to {mm}",
        slots={"mm": S(_MFLD_POOL)},
    ),
]

# ---- Tensor/index notation completeness additions ----------------------------
_DIFFGEOM_TEMPLATES += [
    # -- Missing covariant derivative cases --
    Template(
        name="cov_deriv_covector",
        latex=(
            r"\nabla_{{{kk}}} V_{{{ii}}}"
            r" = \partial_{{{kk}}} V_{{{ii}}}"
            r" - \Gamma^{{{ll}}}_{{{kk}{ii}}} V_{{{ll}}}"
        ),
        slots={
            "ii": S(_IDX_POOL),
            "kk": X(_IDX_POOL, ("ii",)),
            "ll": X(_IDX_POOL, ("ii", "kk")),
        },
    ),
    Template(
        name="cov_deriv_mixed",
        latex=(
            r"\nabla_{{{kk}}} T^{{{ii}}}{{}}_{{{jj}}}"
            r" = \partial_{{{kk}}} T^{{{ii}}}{{}}_{{{jj}}}"
            r" + \Gamma^{{{ii}}}_{{{kk}{ll}}} T^{{{ll}}}{{}}_{{{jj}}}"
            r" - \Gamma^{{{ll}}}_{{{kk}{jj}}} T^{{{ii}}}{{}}_{{{ll}}}"
        ),
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
            "ll": X(_IDX_POOL, ("ii", "jj", "kk")),
        },
    ),
    # -- Index raising/lowering (component form) --
    Template(
        name="index_raising_vector",
        latex=r"V^{{{ii}}} = {met}^{{{ii}{jj}}} V_{{{jj}}}",
        slots={
            "met": S(_METRIC_POOL),
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
    Template(
        name="index_lowering_vector",
        latex=r"V_{{{ii}}} = {met}_{{{ii}{jj}}} V^{{{jj}}}",
        slots={
            "met": S(_METRIC_POOL),
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
    Template(
        name="index_raising_tensor",
        latex=r"T^{{{ii}{jj}}} = {met}^{{{ii}{kk}}} {met}^{{{jj}{ll}}} T_{{{kk}{ll}}}",
        slots={
            "met": S(_METRIC_POOL),
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
            "ll": X(_IDX_POOL, ("ii", "jj", "kk")),
        },
    ),
    Template(
        name="index_lowering_tensor",
        latex=r"T_{{{ii}{jj}}} = {met}_{{{ii}{kk}}} {met}_{{{jj}{ll}}} T^{{{kk}{ll}}}",
        slots={
            "met": S(_METRIC_POOL),
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
            "ll": X(_IDX_POOL, ("ii", "jj", "kk")),
        },
    ),
    # -- Einstein summation convention (explicit statement) --
    Template(
        name="einstein_convention_def",
        latex=r"A^{{{ii}}} B_{{{ii}}} \equiv \sum_{{{ii}}} A^{{{ii}}} B_{{{ii}}}",
        slots={"ii": S(_IDX_POOL)},
    ),
    Template(
        name="einstein_trace_def",
        latex=(
            r"T^{{{ii}}}{{}}_{{{ii}}}"
            r" \equiv \sum_{{{ii}}} T^{{{ii}}}{{}}_{{{ii}}}"
            r" = \operatorname{{tr}} T"
        ),
        slots={"ii": S(_IDX_POOL)},
    ),
    # -- Symmetrization and antisymmetrization --
    Template(
        name="tensor_symmetrization",
        latex=(r"T_{{({ii}{jj})}} = \tfrac{{1}}{{2}}\!\left(T_{{{ii}{jj}}} + T_{{{jj}{ii}}}\right)"),
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
    Template(
        name="tensor_antisymmetrization",
        latex=(r"T_{{[{ii}{jj}]}} = \tfrac{{1}}{{2}}\!\left(T_{{{ii}{jj}}} - T_{{{jj}{ii}}}\right)"),
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
    Template(
        name="tensor_sym_decomp",
        latex=r"T_{{{ii}{jj}}} = T_{{({ii}{jj})}} + T_{{[{ii}{jj}]}}",
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
    # -- Torsion tensor components --
    Template(
        name="torsion_components",
        latex=(
            r"T^{{{kk}}}{{}}_{{{ii}{jj}}}"
            r" = \Gamma^{{{kk}}}_{{{ii}{jj}}} - \Gamma^{{{kk}}}_{{{jj}{ii}}}"
        ),
        slots={
            "kk": S(_IDX_POOL),
            "ii": X(_IDX_POOL, ("kk",)),
            "jj": X(_IDX_POOL, ("kk", "ii")),
        },
    ),
    # -- Riemann pair symmetry --
    Template(
        name="riemann_pair_symmetry",
        latex=r"R_{{{ii}{jj}{kk}{ll}}} = R_{{{kk}{ll}{ii}{jj}}}",
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
            "ll": X(_IDX_POOL, ("ii", "jj", "kk")),
        },
    ),
    # -- Weyl tensor --
    Template(
        name="weyl_def",
        latex=(
            r"C_{{{ii}{jj}{kk}{ll}}} = R_{{{ii}{jj}{kk}{ll}}}"
            r" - \tfrac{{1}}{{n-2}}\!\left("
            r"{met}_{{{ii}{kk}}} R_{{{jj}{ll}}}"
            r" - {met}_{{{ii}{ll}}} R_{{{jj}{kk}}}"
            r" - {met}_{{{jj}{kk}}} R_{{{ii}{ll}}}"
            r" + {met}_{{{jj}{ll}}} R_{{{ii}{kk}}}\right)"
            r" + \tfrac{{R}}{{(n-1)(n-2)}}\!\left("
            r"{met}_{{{ii}{kk}}} {met}_{{{jj}{ll}}}"
            r" - {met}_{{{ii}{ll}}} {met}_{{{jj}{kk}}}\right)"
        ),
        slots={
            "met": S(_METRIC_POOL),
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
            "ll": X(_IDX_POOL, ("ii", "jj", "kk")),
        },
    ),
    Template(
        name="weyl_traceless",
        latex=r"{met}^{{{ii}{kk}}} C_{{{ii}{jj}{kk}{ll}}} = 0",
        slots={
            "met": S(_METRIC_POOL),
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
            "ll": X(_IDX_POOL, ("ii", "jj", "kk")),
        },
    ),
    Template(
        name="weyl_antisymmetry",
        latex=(
            r"C_{{{ii}{jj}{kk}{ll}}}"
            r" = -C_{{{jj}{ii}{kk}{ll}}}"
            r" = -C_{{{ii}{jj}{ll}{kk}}}"
        ),
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
            "ll": X(_IDX_POOL, ("ii", "jj", "kk")),
        },
    ),
    # -- Kronecker delta (geometry side) --
    Template(
        name="kronecker_mixed_geom",
        latex=r"\delta^{{{ii}}}_{{{jj}}}",
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
    Template(
        name="kronecker_flat_trace",
        latex=r"\delta^{{{ii}}}_{{{ii}}} = \dim M",
        slots={"ii": S(_IDX_POOL)},
    ),
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("differential_geometry", _DIFFGEOM_TEMPLATES)
