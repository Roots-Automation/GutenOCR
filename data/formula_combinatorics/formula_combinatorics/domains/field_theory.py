"""Field theory domain — variational principles, tensor notation, particle physics."""

from __future__ import annotations

from ..engine._template_dsl import _FN_SLOT, _LIM_MOD, S, Template, X
from ._config import register_domain
from ._physics_vocab import (
    _COORD_POOL,
    _ENERGY_POOL,
    _HAM_POOL,
    _IDX_POOL,
    _K_POOL,
    _MASS_POOL,
    _OMEGA_POOL,
    _PSI_POOL,
    _Q_POOL,
    _SPACETIME_IDX_POOL,
)

_TEMPLATES: list[Template] = [
    # ── Part C1: Variational and field equations ─────────────────────────────
    Template(
        name="variational_action",
        latex=(
            r"\delta\int{lim_mod}_{{t_1}}^{{t_2}}"
            r" {fn}\!\left({qq},\,\dot{{{qq}}},\,t\right)\,dt = 0"
        ),
        slots={"lim_mod": _LIM_MOD, "fn": _FN_SLOT, "qq": S(_Q_POOL)},
    ),
    Template(
        name="euler_lagrange_functional",
        latex=(
            r"\frac{{d}}{{dt}}\frac{{\partial {fn}}}{{\partial \dot{{{qq}}}}}"
            r" - \frac{{\partial {fn}}}{{\partial {qq}}} = 0"
        ),
        slots={"fn": _FN_SLOT, "qq": S(_Q_POOL)},
    ),
    Template(
        name="greens_function_equation",
        latex=r"{fn1}({cc})\,{fn2}({cc}') = -4\pi\,\delta({cc}-{cc}')",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "cc": S(_COORD_POOL),
        },
    ),
    Template(
        name="action_functional",
        latex=(
            r"S[{fn1}] = \int{lim_mod}_{{t_0}}^{{t_1}}"
            r" {fn2}\!\left({qq},\,\dot{{{qq}}},\,t\right)\,dt"
        ),
        slots={
            "lim_mod": _LIM_MOD,
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "qq": S(_Q_POOL),
        },
    ),
    Template(
        name="lagrangian_density_field",
        latex=(
            r"\mathcal{{L}}\bigl({fn1},\,\partial_{{{mu}}}{fn1}\bigr)"
            r" = {fn2}({fn1}) - V({fn1})"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "mu": S(_SPACETIME_IDX_POOL),
        },
    ),
    Template(
        name="noether_current",
        latex=(
            r"j^{{{mu}}} ="
            r" \frac{{\partial \mathcal{{L}}}}{{\partial(\partial_{{{mu}}}{fn1})}}"
            r"\,\delta {fn2}"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "mu": S(_SPACETIME_IDX_POOL),
        },
    ),
    Template(
        name="scattering_amplitude",
        latex=(
            r"\mathcal{{M}} = \langle {fn1} | T | {fn2} \rangle"
            r" = \langle {fn1} | {ham} | {fn2} \rangle"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "ham": S(_HAM_POOL),
        },
    ),
    Template(
        name="field_equation_general",
        latex=(
            r"\partial_{{{mu}}}\frac{{\partial \mathcal{{L}}}}{{\partial(\partial_{{{mu}}}{fn1})}}"
            r" - \frac{{\partial \mathcal{{L}}}}{{\partial {fn2}}} = 0"
        ),
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "mu": S(_SPACETIME_IDX_POOL),
        },
    ),
    # ── Part C2: Function-pair (field theory context) ────────────────────────
    Template(
        name="fn_dispersion_relation",
        latex=r"{fn1}({om}) = {fn2}({kk}\,c)",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "om": S(_OMEGA_POOL),
            "kk": S(_K_POOL),
        },
    ),
    Template(
        name="fn_correlation_fn",
        latex=r"{fn1}(r) = {fn2}\!\left(\langle {psi}(0)\,{psi}(r)\rangle\right)",
        slots={
            "fn1": _FN_SLOT,
            "fn2": _FN_SLOT,
            "psi": S(_PSI_POOL),
        },
    ),
    # ── Appendix C3: Upsilon meson / particle physics ────────────────────────
    Template(
        name="upsilon_meson_mass",
        latex=r"m_\Upsilon \approx 9.460\,\frac{{\mathrm{{GeV}}}}{{c^2}}",
        slots={},
    ),
    Template(
        name="upsilon_leptonic_width",
        latex=(
            r"\Gamma(\Upsilon \to \ell^+ \ell^-)"
            r" = \frac{{16\pi \alpha^2 e_b^2}}{{3\, {mm}^2}} |\psi(0)|^2"
        ),
        slots={"mm": S(_MASS_POOL)},
    ),
    # ── Appendix C4: Dirac notation and conserved currents ───────────────────
    Template(
        name="dirac_adjoint_derivative",
        latex=r"\bar{{\psi}}\,\overleftarrow{{\partial}}_{{\mu}} = -\partial_{{\mu}}\bar{{\psi}}",
        slots={},
    ),
    Template(
        name="conserved_current_lr_arrows",
        latex=(
            r"j^{{\mu}} = \bar{{\psi}}\,\gamma^{{\mu}}\,\psi,"
            r"\quad j^{{\mu}} = \bar{{\psi}}\,\overrightarrow{{\partial}}^{{\mu}}\psi"
            r" - \bar{{\psi}}\,\overleftarrow{{\partial}}^{{\mu}}\psi"
        ),
        slots={},
    ),
    # ── Part C5: 4-vectors and relativistic tensors ──────────────────────────
    Template(
        name="four_momentum_def",
        latex=r"p^{{{mu}}} = ({ee}/c,\, \mathbf{{p}})",
        slots={
            "mu": S(_SPACETIME_IDX_POOL),
            "ee": S(_ENERGY_POOL),
        },
    ),
    Template(
        name="four_vector_norm",
        latex=r"p_{{{mu}}} p^{{{mu}}} = -{mm}^2 c^2",
        slots={
            "mu": S(_SPACETIME_IDX_POOL),
            "mm": S(_MASS_POOL),
        },
    ),
    Template(
        name="minkowski_metric_idx",
        latex=r"ds^2 = \eta_{{{mu}{nu}}} dx^{{{mu}}} dx^{{{nu}}}",
        slots={
            "mu": S(_SPACETIME_IDX_POOL),
            "nu": X(_SPACETIME_IDX_POOL, ("mu",)),
        },
    ),
    Template(
        name="stress_energy_conservation",
        latex=r"\partial_{{{mu}}} T^{{{mu}{nu}}} = 0",
        slots={
            "mu": S(_SPACETIME_IDX_POOL),
            "nu": X(_SPACETIME_IDX_POOL, ("mu",)),
        },
    ),
    # ── Part C5: Levi-Civita symbol ───────────────────────────────────────────
    Template(
        name="levi_civita_3d_cross",
        latex=(
            r"(\mathbf{{A}}\times\mathbf{{B}})^{{{ii}}}"
            r" = \varepsilon^{{{ii}{jj}{kk}}} A_{{{jj}}} B_{{{kk}}}"
        ),
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
        },
    ),
    Template(
        name="levi_civita_3d_curl",
        latex=(
            r"(\nabla\times\mathbf{{A}})^{{{ii}}}"
            r" = \varepsilon^{{{ii}{jj}{kk}}} \partial_{{{jj}}} A_{{{kk}}}"
        ),
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
        },
    ),
    Template(
        name="levi_civita_contraction",
        latex=(
            r"\varepsilon_{{{ii}{jj}{kk}}} \varepsilon^{{{ii}{ll}{mm}}}"
            r" = \delta^{{{ll}}}_{{{jj}}} \delta^{{{mm}}}_{{{kk}}}"
            r" - \delta^{{{ll}}}_{{{kk}}} \delta^{{{mm}}}_{{{jj}}}"
        ),
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
            "kk": X(_IDX_POOL, ("ii", "jj")),
            "ll": X(_IDX_POOL, ("ii", "jj", "kk")),
            "mm": X(_IDX_POOL, ("ii", "jj", "kk", "ll")),
        },
    ),
    Template(
        name="levi_civita_4d_dual",
        latex=(
            r"\tilde{{F}}^{{{mu}{nu}}}"
            r" = \tfrac{{1}}{{2}} \varepsilon^{{{mu}{nu}{rho}{sig}}} F_{{{rho}{sig}}}"
        ),
        slots={
            "mu": S(_SPACETIME_IDX_POOL),
            "nu": X(_SPACETIME_IDX_POOL, ("mu",)),
            "rho": X(_SPACETIME_IDX_POOL, ("mu", "nu")),
            "sig": X(_SPACETIME_IDX_POOL, ("mu", "nu", "rho")),
        },
    ),
    # ── Part C5: Kronecker delta ──────────────────────────────────────────────
    Template(
        name="kronecker_mixed",
        latex=r"\delta^{{{ii}}}_{{{jj}}}",
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
    Template(
        name="kronecker_trace",
        latex=r"\delta^{{{ii}}}_{{{ii}}} = n",
        slots={"ii": S(_IDX_POOL)},
    ),
    Template(
        name="kronecker_contraction",
        latex=r"\delta^{{{ii}}}_{{{jj}}} T^{{{jj}}} = T^{{{ii}}}",
        slots={
            "ii": S(_IDX_POOL),
            "jj": X(_IDX_POOL, ("ii",)),
        },
    ),
]

GENERATORS, WEIGHTS, TEMPLATES = register_domain("field_theory", _TEMPLATES)
