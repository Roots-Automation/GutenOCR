"""Optimization domain generator — radically expanded for OCR pretraining diversity."""

from __future__ import annotations

from .._template_dsl import _FN_SLOT, _LIM_MOD, S, Template, X
from .._vocab import _CALLIGRAPHIC
from ._config import register_domain

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

# Notation-style pools (primary OCR diversity levers)
_X_POOL = ("x", "w", r"\theta", r"\mathbf{x}", r"\mathbf{w}")  # 5 — variable styles
_F_POOL = ("f", "F", r"\mathcal{F}", r"\varphi", "g", "h")  # 6 — objective fn styles
_G_POOL = ("g", "h", r"\psi", r"\phi", "c")  # 5 — constraint fn styles

# Expanded parameter pools
_ETA_POOL = (r"\eta", r"\alpha", r"\gamma", r"\tau", r"\rho")  # 5 — step sizes
_LAM_POOL = (r"\lambda", r"\mu", r"\nu", r"\rho", r"\sigma")  # 5 — Lagrange multipliers
_BETA_POOL = (r"\beta", r"\beta_1", r"\beta_2", r"\rho", r"\beta_0", r"\beta_k")  # 6 — momentum params
_IDX_POOL = ("k", "t", "n", "s", "j", "r", "l", "m")  # 8 — iteration indices
_MU_POOL = (r"\mu", r"\mu_k", r"\mu_0", r"\nu", r"\rho", r"\kappa")  # 6 — penalty / env params
_H_POOL = ("h", r"\phi", r"\psi", r"\omega")  # 4 — mirror-map / Bregman fn
_T_POOL = ("T", "K", "N", r"\tau", r"T_0", r"N_0")  # 6 — horizon / total steps
_EPS_POOL = (r"\varepsilon", r"\epsilon", r"\delta", r"\kappa", r"\eta_0", r"\nu")  # 6 — tolerance styles

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_OPTIMIZATION_TEMPLATES: list[Template] = [
    # ---- Part A: Reparameterized originals (15) ----------------------------
    Template(
        name="gradient_descent_step",
        latex=(
            r"{xx}_{{{kk}+1}} = {xx}_{{{kk}}}"
            r" - {eta}\,\nabla {ff}({xx}_{{{kk}}})"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "eta": S(_ETA_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="lagrangian_def",
        latex=(
            r"\mathcal{{L}}({xx},\,{lam})"
            r" = {ff}({xx}) + {lam}^\top {gg}({xx})"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "lam": S(_LAM_POOL),
            "gg": S(_G_POOL),
        },
    ),
    Template(
        name="kkt_stationarity",
        latex=r"\nabla {ff}({xx}^*) = {lam}\,\nabla {gg}({xx}^*)",
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "lam": S(_LAM_POOL),
            "gg": S(_G_POOL),
        },
    ),
    Template(
        name="convexity_definition",
        latex=(
            r"{ff}({lam}\,{xx} + (1-{lam})\,y)"
            r" \leq {lam}\,{ff}({xx}) + (1-{lam})\,{ff}(y)"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL), "lam": S(_LAM_POOL)},
    ),
    Template(
        name="conjugate_function",
        latex=(
            r"{ff}^*(y) = \sup_{{{xx}}}"
            r" \bigl\langle y,\,{xx}\bigr\rangle - {ff}({xx})"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL)},
    ),
    Template(
        name="proximal_operator",
        latex=(
            r"\operatorname{{prox}}_{{{ff}}}({xx})"
            r" = \arg\min_{{u}}\left\{{{ff}(u)"
            r" + \tfrac{{1}}{{2{eta}}}\|u-{xx}\|^2\right\}}"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL), "eta": S(_ETA_POOL)},
    ),
    Template(
        name="newtons_method_step",
        latex=(
            r"{xx}_{{{kk}+1}} = {xx}_{{{kk}}}"
            r" - \bigl[\nabla^2 {ff}({xx}_{{{kk}}})\bigr]^{{-1}}"
            r"\,\nabla {ff}({xx}_{{{kk}}})"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL), "kk": S(_IDX_POOL)},
    ),
    Template(
        name="constrained_minimization",
        latex=(
            r"\min_{{{xx}}}\;{ff}({xx})"
            r"\quad\text{{s.t.}}\quad {gg}_i({xx}) \leq 0"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL), "gg": S(_G_POOL)},
    ),
    Template(
        name="lipschitz_gradient",
        latex=(
            r"\|\nabla {ff}({xx}) - \nabla {ff}(y)\|"
            r" \leq L\,\|{xx}-y\|"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL)},
    ),
    Template(
        name="first_order_convexity",
        latex=(
            r"{ff}(y) \geq {ff}({xx})"
            r" + \nabla {ff}({xx})^\top(y-{xx})"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL)},
    ),
    Template(
        name="sublinear_convergence_rate",
        latex=(
            r"\frac{{1}}{{{tt}}}\sum{lim_mod}_{{t=1}}^{{{tt}}}"
            r" {ff}({xx}_t) - {ff}({xx}^*)"
            r" \leq O\!\left(\frac{{1}}{{\sqrt{{{tt}}}}}\right)"
        ),
        slots={"lim_mod": _LIM_MOD, "xx": S(_X_POOL), "ff": S(_F_POOL), "tt": S(_T_POOL)},
    ),
    Template(
        name="linear_convergence",
        latex=(
            r"\|{xx}_{{{kk}+1}} - {xx}^*\|"
            r" \leq {lam}\,\|{xx}_{{{kk}}} - {xx}^*\|"
        ),
        slots={"xx": S(_X_POOL), "lam": S(_LAM_POOL), "kk": S(_IDX_POOL)},
    ),
    Template(
        name="strong_convexity",
        latex=(
            r"{ff}(y) \geq {ff}({xx})"
            r" + \nabla {ff}({xx})^\top(y-{xx})"
            r" + \frac{{{lam}}}{{2}}\|y-{xx}\|^2"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL), "lam": S(_LAM_POOL)},
    ),
    Template(
        name="proximal_point_step",
        latex=(
            r"{xx}^{{{kk}+1}} = \arg\min_{{{xx}}}"
            r"\left\{{{ff}({xx})"
            r" + \frac{{{lam}}}{{2}}\|{xx}-{xx}^{{{kk}}}\|^2\right\}}"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "lam": S(_LAM_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="subgradient_step",
        latex=(
            r"{xx}_{{{kk}+1}} = {xx}_{{{kk}}}"
            r" - {eta}_{{{kk}}}\,g_{{{kk}}},"
            r"\quad g_{{{kk}}} \in \partial {ff}({xx}_{{{kk}}})"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "eta": S(_ETA_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    # ---- Part B: Gradient Methods (10) -------------------------------------
    Template(
        name="gradient_descent_convergence",
        latex=(
            r"{ff}({xx}_k) - {ff}({xx}^*)"
            r" \leq \frac{{L\,\|{xx}_0-{xx}^*\|^2}}{{2k}}"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL)},
    ),
    Template(
        name="projected_gradient_step",
        latex=(
            r"{xx}_{{{kk}+1}}"
            r" = \Pi_C\!\left({xx}_{{{kk}}} - {eta}\,\nabla {ff}({xx}_{{{kk}}})\right)"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "eta": S(_ETA_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="accelerated_gradient_y",
        latex=(
            r"y_{{{kk}}} = {xx}_{{{kk}}}"
            r" + \frac{{{kk}-1}}{{{kk}+2}}"
            r"\bigl({xx}_{{{kk}}}-{xx}_{{{kk}-1}}\bigr)"
        ),
        slots={"xx": S(_X_POOL), "kk": S(_IDX_POOL)},
    ),
    Template(
        name="accelerated_gradient_x",
        latex=(
            r"{xx}_{{{kk}+1}}"
            r" = y_{{{kk}}} - {eta}\,\nabla {ff}(y_{{{kk}}})"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "eta": S(_ETA_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="momentum_step",
        latex=(
            r"{xx}_{{{kk}+1}} = {xx}_{{{kk}}}"
            r" - {eta}\,\nabla {ff}({xx}_{{{kk}}})"
            r" + {bt}\,\bigl({xx}_{{{kk}}}-{xx}_{{{kk}-1}}\bigr)"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "eta": S(_ETA_POOL),
            "bt": S(_BETA_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="adam_m_update",
        latex=(
            r"m_{{{kk}+1}} = {bt}\,m_{{{kk}}}"
            r" + (1-{bt})\,\nabla {ff}({xx}_{{{kk}}})"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "bt": S(_BETA_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="adam_v_update",
        latex=(
            r"v_{{{kk}+1}} = {bt}\,v_{{{kk}}}"
            r" + (1-{bt})\,\bigl(\nabla {ff}({xx}_{{{kk}}})\bigr)^2"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "bt": S(_BETA_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="adam_x_update",
        latex=(
            r"{xx}_{{{kk}+1}} = {xx}_{{{kk}}}"
            r" - \frac{{\hat{{m}}_{{{kk}}}}}{{\sqrt{{\hat{{v}}_{{{kk}}}}}+{eps}}}"
        ),
        slots={"xx": S(_X_POOL), "kk": S(_IDX_POOL), "eps": S(_EPS_POOL)},
    ),
    Template(
        name="gradient_norm_convergence",
        latex=(
            r"\min_{{{kk}\leq {tt}}}\|\nabla {ff}({xx}_{{{kk}}})\|"
            r" \leq \frac{{C}}{{\sqrt{{{tt}}}}}"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "kk": S(_IDX_POOL),
            "tt": S(_T_POOL),
        },
    ),
    Template(
        name="wolfe_condition_sufficient",
        latex=(
            r"{ff}({xx}+{eta}\,d)"
            r" \leq {ff}({xx}) + c_1\,{eta}\,\nabla {ff}({xx})^\top d"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL), "eta": S(_ETA_POOL)},
    ),
    # ---- Part B: Stochastic Methods (8) ------------------------------------
    Template(
        name="sgd_step",
        latex=(
            r"{xx}_{{{kk}+1}} = {xx}_{{{kk}}}"
            r" - {eta}_{{{kk}}}\,\nabla {ff}_{{{ii}}}({xx}_{{{kk}}})"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "eta": S(_ETA_POOL),
            "kk": S(_IDX_POOL),
            "ii": X(_IDX_POOL, ("kk",)),
        },
    ),
    Template(
        name="sgd_convergence_rate",
        latex=(
            r"\mathbb{{E}}\bigl[{ff}({xx}_{{{kk}}})\bigr] - {ff}({xx}^*)"
            r" \leq O\!\left(\frac{{1}}{{\sqrt{{{kk}}}}}\right)"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL), "kk": S(_IDX_POOL)},
    ),
    Template(
        name="variance_reduction_update",
        latex=(
            r"{xx}_{{{kk}+1}} = {xx}_{{{kk}}} - {eta}\!\left("
            r"\nabla {ff}_{{{ii}}}({xx}_{{{kk}}})"
            r" - \nabla {ff}_{{{ii}}}(\tilde{{{xx}}})"
            r" + \nabla {ff}(\tilde{{{xx}}})\right)"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "eta": S(_ETA_POOL),
            "kk": S(_IDX_POOL),
            "ii": X(_IDX_POOL, ("kk",)),
        },
    ),
    Template(
        name="minibatch_gradient",
        latex=(
            r"{xx}_{{{kk}+1}} = {xx}_{{{kk}}}"
            r" - \frac{{{eta}}}{{|B|}}\sum{lim_mod}_{{i\in B}}"
            r"\nabla {ff}_i({xx}_{{{kk}}})"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "eta": S(_ETA_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="online_regret_def",
        latex=(
            r"R_{{{tt}}} = \sum{lim_mod}_{{t=1}}^{{{tt}}}"
            r" {ff}_t({xx}_t)"
            r" - \min_{{{xx}}}\sum{lim_mod}_{{t=1}}^{{{tt}}} {ff}_t({xx})"
        ),
        slots={"lim_mod": _LIM_MOD, "xx": S(_X_POOL), "ff": S(_F_POOL), "tt": S(_T_POOL)},
    ),
    Template(
        name="regret_bound_ogd",
        latex=(
            r"R_{{{tt}}} \leq \frac{{\|{xx}_1-{xx}^*\|^2}}{{2{eta}}}"
            r" + \frac{{{eta}}}{{2}}\sum{lim_mod}_{{t=1}}^{{{tt}}}"
            r"\|\nabla {ff}_t({xx}_t)\|^2"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "eta": S(_ETA_POOL),
            "tt": S(_T_POOL),
        },
    ),
    Template(
        name="sag_update",
        latex=(
            r"{xx}_{{{kk}+1}} = {xx}_{{{kk}}}"
            r" - \frac{{{eta}}}{{n}}\sum{lim_mod}_{{i=1}}^{{n}}"
            r"\nabla {ff}_i^{{{kk}}}"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "eta": S(_ETA_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="coordinate_descent_step",
        latex=(
            r"{xx}_{{{ii}}}^{{{kk}+1}}"
            r" = \arg\min_{{s}}\,{ff}(s\,e_{{{ii}}}"
            r" + {xx}^{{{kk}}}_{{-{ii}}})"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "ii": S(_IDX_POOL),
            "kk": X(_IDX_POOL, ("ii",)),
        },
    ),
    # ---- Part B: Convex Analysis (10) --------------------------------------
    Template(
        name="fenchel_inequality",
        latex=(
            r"{ff}({xx}) + {ff}^*(y)"
            r" \geq \langle {xx},\,y\rangle"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL)},
    ),
    Template(
        name="fenchel_duality_theorem",
        latex=r"{ff}^{{**}}({xx}) = {ff}({xx})",
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL)},
    ),
    Template(
        name="moreau_envelope",
        latex=(
            r"{ff}^{{{mu}}}({xx})"
            r" = \min_y\left\{{{ff}(y)"
            r" + \frac{{1}}{{2\,{mu}}}\|{xx}-y\|^2\right\}}"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL), "mu": S(_MU_POOL)},
    ),
    Template(
        name="subgradient_condition",
        latex=(
            r"g \in \partial {ff}({xx})"
            r" \iff {ff}(y) \geq {ff}({xx})"
            r" + \langle g,\,y-{xx}\rangle\;\forall y"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL)},
    ),
    Template(
        name="subgradient_update_rule",
        latex=(
            r"{xx}_{{{kk}+1}} = {xx}_{{{kk}}}"
            r" - {eta}_{{{kk}}}\,g_{{{kk}}},"
            r"\quad g_{{{kk}}} \in \partial {ff}({xx}_{{{kk}}})"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "eta": S(_ETA_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="epigraph_def",
        latex=(
            r"\mathrm{{epi}}({ff})"
            r" = \{{({xx},\,t) \in \mathbb{{R}}^{{n+1}}"
            r" : {ff}({xx}) \leq t\}}"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL)},
    ),
    Template(
        name="hessian_positive_def",
        latex=r"\nabla^2 {ff}({xx}) \succeq {lam}\,I",
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL), "lam": S(_LAM_POOL)},
    ),
    Template(
        name="lipschitz_smooth_bound",
        latex=(
            r"{ff}(y) \leq {ff}({xx})"
            r" + \nabla {ff}({xx})^\top(y-{xx})"
            r" + \frac{{L}}{{2}}\|y-{xx}\|^2"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL)},
    ),
    Template(
        name="subdifferential_sum",
        latex=(
            r"\partial({ff}+{gg})({xx})"
            r" \supseteq \partial {ff}({xx}) + \partial {gg}({xx})"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL), "gg": S(_G_POOL)},
    ),
    Template(
        name="indicator_proximal",
        latex=(
            r"\operatorname{{prox}}_{{{eta}\iota_C}}({xx})"
            r" = \Pi_C({xx})"
        ),
        slots={"xx": S(_X_POOL), "eta": S(_ETA_POOL)},
    ),
    # ---- Part B: KKT and Duality (8) ---------------------------------------
    Template(
        name="kkt_primal_feasibility",
        latex=r"{gg}_i({xx}^*) \leq 0,\quad i=1,\ldots,m",
        slots={"xx": S(_X_POOL), "gg": S(_G_POOL)},
    ),
    Template(
        name="kkt_dual_feasibility",
        latex=r"{lam}_i \geq 0,\quad i=1,\ldots,m",
        slots={"lam": S(_LAM_POOL)},
    ),
    Template(
        name="kkt_complementary_slackness",
        latex=r"{lam}_i\,{gg}_i({xx}^*) = 0",
        slots={"xx": S(_X_POOL), "lam": S(_LAM_POOL), "gg": S(_G_POOL)},
    ),
    Template(
        name="lagrangian_dual_fn",
        latex=(
            r"d({lam}) = \inf_{{{xx}}}\,"
            r"\mathcal{{L}}({xx},\,{lam})"
        ),
        slots={"xx": S(_X_POOL), "lam": S(_LAM_POOL)},
    ),
    Template(
        name="weak_duality",
        latex=r"d({lam}) \leq p^*\quad\forall\,{lam} \geq 0",
        slots={"lam": S(_LAM_POOL)},
    ),
    Template(
        name="slater_condition",
        latex=(
            r"\exists\,{xx}_0 : {gg}_i({xx}_0) < 0"
            r" \;\Rightarrow\; d^* = p^*"
        ),
        slots={"xx": S(_X_POOL), "gg": S(_G_POOL)},
    ),
    Template(
        name="duality_gap",
        latex=r"p^* - d^* \geq 0",
        slots={},
    ),
    Template(
        name="penalty_augmented_lagrangian",
        latex=(
            r"\mathcal{{L}}_\rho({xx},{lam})"
            r" = {ff}({xx}) + {lam}^\top {gg}({xx})"
            r" + \frac{{\rho}}{{2}}\|{gg}({xx})\|^2"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "lam": S(_LAM_POOL),
            "gg": S(_G_POOL),
        },
    ),
    # ---- Part B: Proximal and ADMM (7) -------------------------------------
    Template(
        name="proximal_gradient_step",
        latex=(
            r"{xx}_{{{kk}+1}}"
            r" = \operatorname{{prox}}_{{{eta}\,{gg}}}\!"
            r"\left({xx}_{{{kk}}} - {eta}\,\nabla {ff}({xx}_{{{kk}}})\right)"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "gg": S(_G_POOL),
            "eta": S(_ETA_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="admm_x_update",
        latex=(
            r"{xx}^{{{kk}+1}} = \arg\min_{{{xx}}}\,"
            r"{ff}({xx}) + \frac{{\rho}}{{2}}"
            r"\|A{xx} + B{zz}^{{{kk}}} - c + u^{{{kk}}}\|^2"
        ),
        slots={
            "xx": S(_X_POOL),
            "zz": X(_X_POOL, ("xx",)),
            "ff": S(_F_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="admm_z_update",
        latex=(
            r"{zz}^{{{kk}+1}} = \arg\min_{{{zz}}}\,"
            r"{gg}({zz}) + \frac{{\rho}}{{2}}"
            r"\|A{xx}^{{{kk}+1}} + B{zz} - c + u^{{{kk}}}\|^2"
        ),
        slots={
            "xx": S(_X_POOL),
            "zz": X(_X_POOL, ("xx",)),
            "gg": S(_G_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="admm_u_update",
        latex=(
            r"u^{{{kk}+1}} = u^{{{kk}}}"
            r" + A{xx}^{{{kk}+1}} + B{zz}^{{{kk}+1}} - c"
        ),
        slots={
            "xx": S(_X_POOL),
            "zz": X(_X_POOL, ("xx",)),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="proximal_point_iteration",
        latex=(
            r"{xx}^{{{kk}+1}}"
            r" = \operatorname{{prox}}_{{{lam}\,{ff}}}({xx}^{{{kk}}})"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "lam": S(_LAM_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="moreau_decomposition",
        latex=(
            r"{xx} = \operatorname{{prox}}_{{{eta}\,{ff}}}({xx})"
            r" + {eta}\,\operatorname{{prox}}_{{(1/{eta}){ff}^*}}({xx}/{eta})"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL), "eta": S(_ETA_POOL)},
    ),
    Template(
        name="douglas_rachford",
        latex=(
            r"{xx}^{{{kk}+1}} = \tfrac{{1}}{{2}}\!\left("
            r"\operatorname{{prox}}_{{{eta}\,{ff}}}(z^{{{kk}}})"
            r" + z^{{{kk}}}\right)"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "eta": S(_ETA_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    # ---- Part B: Second-Order Methods (6) ----------------------------------
    Template(
        name="bfgs_update",
        latex=(
            r"H_{{{kk}+1}} = H_{{{kk}}}"
            r" + \frac{{y_{{{kk}}}\,y_{{{kk}}}^\top}}{{y_{{{kk}}}^\top s_{{{kk}}}}}"
            r" - \frac{{H_{{{kk}}}\,s_{{{kk}}}\,s_{{{kk}}}^\top H_{{{kk}}}}}{{s_{{{kk}}}^\top H_{{{kk}}}\,s_{{{kk}}}}}"
        ),
        slots={"kk": S(_IDX_POOL)},
    ),
    Template(
        name="secant_condition",
        latex=r"H_{{{kk}+1}}\,s_{{{kk}}} = y_{{{kk}}}",
        slots={"kk": S(_IDX_POOL)},
    ),
    Template(
        name="newton_decrement",
        latex=(
            r"\lambda({xx})^2"
            r" = \nabla {ff}({xx})^\top"
            r"\bigl[\nabla^2 {ff}({xx})\bigr]^{{-1}}"
            r"\nabla {ff}({xx})"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL)},
    ),
    Template(
        name="quadratic_convergence_newton",
        latex=(
            r"\|{xx}_{{{kk}+1}} - {xx}^*\|"
            r" \leq c\,\|{xx}_{{{kk}}} - {xx}^*\|^2"
        ),
        slots={"xx": S(_X_POOL), "kk": S(_IDX_POOL)},
    ),
    Template(
        name="inexact_newton",
        latex=(
            r"\|{ff}({xx}_{{{kk}}}) + \nabla {ff}({xx}_{{{kk}}})\,d_{{{kk}}}\|"
            r" \leq {eta}_{{{kk}}}\,\|{ff}({xx}_{{{kk}}})\|"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "eta": S(_ETA_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="gauss_newton_step",
        latex=(
            r"{xx}_{{{kk}+1}} = {xx}_{{{kk}}}"
            r" - (J^\top J)^{{-1}} J^\top {ff}({xx}_{{{kk}}})"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL), "kk": S(_IDX_POOL)},
    ),
    # ---- Part B: Mirror Descent and Frank-Wolfe (5) ------------------------
    Template(
        name="bregman_divergence",
        latex=(
            r"D_{{{hh}}}({xx},\,y)"
            r" = {hh}({xx}) - {hh}(y)"
            r" - \nabla {hh}(y)^\top({xx}-y)"
        ),
        slots={"xx": S(_X_POOL), "hh": S(_H_POOL)},
    ),
    Template(
        name="mirror_descent_update",
        latex=(
            r"\nabla {hh}({xx}_{{{kk}+1}})"
            r" = \nabla {hh}({xx}_{{{kk}}})"
            r" - {eta}_{{{kk}}}\,g_{{{kk}}}"
        ),
        slots={
            "xx": S(_X_POOL),
            "hh": S(_H_POOL),
            "eta": S(_ETA_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="mirror_descent_regret",
        latex=(
            r"\sum{lim_mod}_{{t=1}}^{{{tt}}}\langle g_t,\,{xx}_t-u\rangle"
            r" \leq \frac{{D_{{{hh}}}(u,\,{xx}_1)}}{{{eta}}}"
            r" + {eta}\sum{lim_mod}_{{t=1}}^{{{tt}}}\|g_t\|_*^2"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "xx": S(_X_POOL),
            "hh": S(_H_POOL),
            "eta": S(_ETA_POOL),
            "tt": S(_T_POOL),
        },
    ),
    Template(
        name="frank_wolfe_lmo",
        latex=(
            r"s_{{{kk}}} = \arg\min_{{s\in C}}"
            r"\langle\nabla {ff}({xx}_{{{kk}}}),\,s\rangle"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL), "kk": S(_IDX_POOL)},
    ),
    Template(
        name="frank_wolfe_update",
        latex=(
            r"{xx}_{{{kk}+1}}"
            r" = (1-{gm}_{{{kk}}})\,{xx}_{{{kk}}}"
            r" + {gm}_{{{kk}}}\,s_{{{kk}}}"
        ),
        slots={"xx": S(_X_POOL), "gm": S(_ETA_POOL), "kk": S(_IDX_POOL)},
    ),
    # ---- Part B: Norms and Regularization (5) ------------------------------
    Template(
        name="l1_regularization",
        latex=r"\min_{{{xx}}}\;{ff}({xx}) + {lam}\|{xx}\|_1",
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL), "lam": S(_LAM_POOL)},
    ),
    Template(
        name="l2_regularization",
        latex=(
            r"\min_{{{xx}}}\;{ff}({xx})"
            r" + \frac{{{lam}}}{{2}}\|{xx}\|^2"
        ),
        slots={"xx": S(_X_POOL), "ff": S(_F_POOL), "lam": S(_LAM_POOL)},
    ),
    Template(
        name="elastic_net",
        latex=(
            r"\min_{{{xx}}}\;{ff}({xx})"
            r" + {la}\|{xx}\|_1"
            r" + \frac{{{lb}}}{{2}}\|{xx}\|^2"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "la": S(_LAM_POOL),
            "lb": X(_LAM_POOL, ("la",)),
        },
    ),
    Template(
        name="nuclear_norm_regularization",
        latex=r"\min_{{W}}\;{ff}(W) + {lam}\|W\|_*",
        slots={"ff": S(_F_POOL), "lam": S(_LAM_POOL)},
    ),
    Template(
        name="group_lasso",
        latex=(
            r"\min_{{{xx}}}\;{ff}({xx})"
            r" + {lam}\sum{lim_mod}_{{g}}\|{xx}_g\|"
        ),
        slots={"lim_mod": _LIM_MOD, "xx": S(_X_POOL), "ff": S(_F_POOL), "lam": S(_LAM_POOL)},
    ),
    # ---- Part C: High-n_eff function-pair templates (8) --------------------
    Template(
        name="gradient_flow_ode",
        latex=r"\dot{{{xx}}}(t) = -\nabla {fn}({xx}(t))",
        slots={"xx": S(_X_POOL), "fn": _FN_SLOT},
    ),
    Template(
        name="envelope_theorem",
        latex=(
            r"\frac{{d}}{{d\theta}}\,V(\theta)"
            r" = \frac{{\partial {fn}({xx}^*(\theta),\,\theta)}}{{\partial\theta}}"
        ),
        slots={"xx": S(_X_POOL), "fn": _FN_SLOT},
    ),
    Template(
        name="variational_inequality",
        latex=(
            r"\bigl\langle {fn}({xx}^*),\,{xx}-{xx}^*\bigr\rangle"
            r" \geq 0\;\forall\,{xx}\in C"
        ),
        slots={"xx": S(_X_POOL), "fn": _FN_SLOT},
    ),
    Template(
        name="saddle_point_lagrangian",
        latex=(
            r"{fn}({xx}^*,{lam})"
            r" \leq {fn}({xx}^*,{lam}^*)"
            r" \leq {fn}({xx},{lam}^*)"
        ),
        slots={"xx": S(_X_POOL), "lam": S(_LAM_POOL), "fn": _FN_SLOT},
    ),
    Template(
        name="fixed_point_iteration",
        latex=(
            r"{xx}^{{k+1}} = {fn}({xx}^k),"
            r"\quad {xx}^* = {fn}({xx}^*)"
        ),
        slots={"xx": S(_X_POOL), "fn": _FN_SLOT},
    ),
    Template(
        name="operator_splitting",
        latex=r"{fn1}({xx}) + {fn2}({xx}) = 0",
        slots={
            "xx": S(_X_POOL),
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
        },
    ),
    Template(
        name="composite_gradient_map",
        latex=(
            r"G_{{{eta}}}({xx})"
            r" = \frac{{1}}{{{eta}}}\bigl({xx}"
            r" - \operatorname{{prox}}_{{{eta}\,{fn1}}}\!"
            r"\left({xx} - {eta}\,\nabla {fn2}({xx})\right)\bigr)"
        ),
        slots={
            "xx": S(_X_POOL),
            "eta": S(_ETA_POOL),
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
        },
    ),
    Template(
        name="proximal_point_contraction",
        latex=(
            r"\|\operatorname{{prox}}_{{{eta}\,{fn}}}({xx})"
            r" - \operatorname{{prox}}_{{{eta}\,{fn}}}(y)\|"
            r" \leq \|{xx}-y\|"
        ),
        slots={
            "xx": S(_X_POOL),
            "eta": S(_ETA_POOL),
            "fn": _FN_SLOT,
        },
    ),
    Template(
        name="descent_lemma",
        latex=(
            r"{fn1}({xx}_{{{kk}+1}})"
            r" \leq {fn1}({xx}_{{{kk}}})"
            r" - \frac{{1}}{{2{eta}}}\|{xx}_{{{kk}+1}}-{xx}_{{{kk}}}\|^2"
            r" - \frac{{1}}{{2{eta}}}\|\nabla {fn2}({xx}_{{{kk}}})\|^2"
        ),
        slots={
            "xx": S(_X_POOL),
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "eta": S(_ETA_POOL),
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="stationarity_gap",
        latex=(
            r"\|{fn1}({xx}) - {fn2}({xx})\|"
            r" \leq {lam}\,\|{xx} - {xx}^*\|"
        ),
        slots={
            "xx": S(_X_POOL),
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "lam": S(_LAM_POOL),
        },
    ),
    Template(
        name="lyapunov_decrease",
        latex=(
            r"V_{{{kk}+1}} \leq {fn1}({xx}_{{{kk}}})\,V_{{{kk}}}"
            r" + {fn2}({xx}_{{{kk}}})"
        ),
        slots={
            "xx": S(_X_POOL),
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "kk": S(_IDX_POOL),
        },
    ),
    Template(
        name="proximal_subgradient_bound",
        latex=(
            r"\|{fn1}({xx}) - {fn2}({xx}^*)\|"
            r" \leq \frac{{\|{xx}_0-{xx}^*\|^2}}{{2{eta}\,{tt}}}"
            r" + \frac{{{eta}}}{{2}}\|{fn2}({xx}^*)\|^2"
        ),
        slots={
            "xx": S(_X_POOL),
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "eta": S(_ETA_POOL),
            "tt": S(_T_POOL),
        },
    ),
    # ---- Argmin / Argmax (starred operator form, \operatorname*) -----------
    Template(
        name="argmin_basic",
        latex=r"\operatorname*{{argmin}}_{{{xx}}} {ff}({xx})",
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
        },
    ),
    Template(
        name="argmax_basic",
        latex=r"\operatorname*{{argmax}}_{{{xx}}} {ff}({xx})",
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
        },
    ),
    Template(
        name="argmin_constrained",
        latex=r"\operatorname*{{argmin}}_{{{xx} \in {CC}}} {ff}({xx})",
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "CC": S(tuple(_CALLIGRAPHIC)),
        },
    ),
    Template(
        name="argmax_constrained",
        latex=r"\operatorname*{{argmax}}_{{{xx} \in {CC}}} {ff}({xx})",
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "CC": S(tuple(_CALLIGRAPHIC)),
        },
    ),
    Template(
        name="argmin_assignment",
        latex=r"{xx}^* = \operatorname*{{argmin}}_{{{xx}}} {ff}({xx})",
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
        },
    ),
    Template(
        name="argmax_assignment",
        latex=r"\hat{{{par}}} = \operatorname*{{argmax}}_{{{par}}} {ll}({par})",
        slots={
            "par": S((_X_POOL[2],) + (_X_POOL[0],) + tuple(_ETA_POOL[:3])),
            "ll": S((_F_POOL[0],) + (_F_POOL[1],) + (r"\mathcal{L}", r"\ell")),
        },
    ),
    Template(
        name="argmin_norm_sq",
        latex=r"\operatorname*{{argmin}}_{{{xx}}} \|{AA}{xx} - {bb}\|^2",
        slots={
            "xx": S(_X_POOL),
            "AA": S(("A", "B", "M", "W", "H")),
            "bb": S(("b", "c", "d", "y", r"\mathbf{b}", r"\mathbf{y}")),
        },
    ),
    Template(
        name="argmin_sum",
        latex=(
            r"\operatorname*{{argmin}}_{{{xx}}}"
            r" \sum_{{i=1}}^{{{nn}}} {ff}_i({xx})"
        ),
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "nn": S(("n", "N", "T", "m", "M")),
        },
    ),
    Template(
        name="argmin_limits_variant",
        latex=r"\operatorname*{{argmin}}\limits_{{{xx} \in {CC}}} {ff}({xx})",
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "CC": S(tuple(_CALLIGRAPHIC)),
        },
    ),
    Template(
        name="argmax_limits_variant",
        latex=r"\operatorname*{{argmax}}\limits_{{{xx} \in {CC}}} {ff}({xx})",
        slots={
            "xx": S(_X_POOL),
            "ff": S(_F_POOL),
            "CC": S(tuple(_CALLIGRAPHIC)),
        },
    ),
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

GENERATORS, WEIGHTS, TEMPLATES = register_domain("optimization", _OPTIMIZATION_TEMPLATES)
