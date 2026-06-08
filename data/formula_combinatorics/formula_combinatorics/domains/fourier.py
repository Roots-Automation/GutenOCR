"""Fourier analysis and signals domain generators."""

from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import E, S, Template, X, compute_weights, make_dispatcher
from .._vocab import _fn_rich_nosub

# ---------------------------------------------------------------------------
# Slot pools
# ---------------------------------------------------------------------------

_FUNC_POOL = ("f", "g", "h", r"\varphi", r"\psi", "F", "G")
_FREQ_POOL = (r"\xi", r"\omega", r"\nu", "k", r"\zeta", r"\lambda")
_TIME_POOL = ("t", r"\tau", "s", "u", "x")
_PERIOD_POOL = ("T", r"2\pi", "L", "P")
_IDX_POOL = ("n", "m", "k", "p", "q")
_SIZE_POOL = ("N", "M", "K")
_SIGNAL_POOL = ("x", "y", "s", "u", "v", "X", "Y")
_KERNEL_POOL = ("h", "g", r"\phi", r"\psi", "w")
_COEFF_POOL = ("a", "b", "c", r"\alpha", r"\beta")
_TVAR_POOL = (r"\tau", "s", "u", "v")
_S_POOL = ("s", r"s + j\omega", r"\sigma + i\omega", r"\lambda")

# ---------------------------------------------------------------------------
# Part A: Reparameterized originals (14)
# ---------------------------------------------------------------------------

_PART_A: list[Template] = [
    Template(
        name="fourier_transform",
        latex=r"\hat{{{ff}}}({xi}) = \int{lim_mod}_{{-\infty}}^{{\infty}} {ff}({tt})\, e^{{-2\pi i {tt} {xi}}} \, d{tt}",
        slots={"lim_mod": S(("", r"\limits")), "ff": S(_FUNC_POOL), "xi": S(_FREQ_POOL), "tt": S(_TIME_POOL)},
    ),
    Template(
        name="inverse_fourier_transform",
        latex=r"{ff}({tt}) = \int{lim_mod}_{{-\infty}}^{{\infty}} \hat{{{ff}}}({xi})\, e^{{2\pi i {tt} {xi}}} \, d{xi}",
        slots={"lim_mod": S(("", r"\limits")), "ff": S(_FUNC_POOL), "xi": S(_FREQ_POOL), "tt": S(_TIME_POOL)},
    ),
    Template(
        name="fourier_series",
        latex=r"{ff}({vv}) = \sum{lim_mod}_{{n=-\infty}}^{{\infty}} c_n\, e^{{2\pi i n {vv} / {T}}}",
        slots={"lim_mod": S(("", r"\limits")), "ff": S(_FUNC_POOL), "T": S(_PERIOD_POOL), "vv": S(_TIME_POOL)},
    ),
    Template(
        name="fourier_coefficients",
        latex=r"c_n = \frac{{1}}{{{T}}} \int_0^{{{T}}} {ff}({vv})\, e^{{-2\pi i n {vv} / {T}}} \, d{vv}",
        slots={"ff": S(_FUNC_POOL), "T": S(_PERIOD_POOL), "vv": S(_TIME_POOL)},
    ),
    Template(
        name="parseval",
        latex=r"\int{lim_mod}_{{-\infty}}^{{\infty}} |{ff}({tt})|^2 \, d{tt} = \int{lim_mod}_{{-\infty}}^{{\infty}} |\hat{{{ff}}}({xi})|^2 \, d{xi}",
        slots={"lim_mod": S(("", r"\limits")), "ff": S(_FUNC_POOL), "xi": S(_FREQ_POOL), "tt": S(_TIME_POOL)},
    ),
    Template(
        name="convolution",
        latex=r"({ff} * {gg})({tt}) = \int{lim_mod}_{{-\infty}}^{{\infty}} {ff}({tau})\, {gg}({tt} - {tau}) \, d{tau}",
        slots={
            "lim_mod": S(("", r"\limits")),
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "tt": S(_TIME_POOL),
            "tau": S(_TVAR_POOL),
        },
    ),
    Template(
        name="convolution_theorem",
        latex=r"\widehat{{{ff} * {gg}}} = \hat{{{ff}}} \cdot \hat{{{gg}}}",
        slots={"ff": S(_FUNC_POOL), "gg": X(_FUNC_POOL, ("ff",))},
    ),
    Template(
        name="laplace_transform",
        latex=r"\mathcal{{L}}\left\{{{ff}(t)\right\}}({ss}) = \int_0^{{\infty}} {ff}(t)\, e^{{-{ss}\, t}} \, dt",
        slots={"ff": S(_FUNC_POOL), "ss": S(_S_POOL)},
    ),
    Template(
        name="z_transform",
        latex=r"X(z) = \sum{lim_mod}_{{{nn}=-\infty}}^{{\infty}} {xx}[{nn}]\, z^{{-{nn}}}",
        slots={"lim_mod": S(("", r"\limits")), "xx": S(_SIGNAL_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="dft",
        latex=r"X[k] = \sum{lim_mod}_{{{nn}=0}}^{{{N}-1}} {xx}[{nn}]\, e^{{-2\pi i k {nn} / {N}}}",
        slots={"lim_mod": S(("", r"\limits")), "xx": S(_SIGNAL_POOL), "N": S(_SIZE_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="dirac_delta_sifting",
        latex=r"\int{lim_mod}_{{-\infty}}^{{\infty}} \delta({tt} - {aa})\, {ff}({tt}) \, d{tt} = {ff}({aa})",
        slots={"lim_mod": S(("", r"\limits")), "ff": S(_FUNC_POOL), "tt": S(_TIME_POOL), "aa": S(_COEFF_POOL)},
    ),
    Template(
        name="time_frequency_uncertainty",
        latex=r"\Delta {Dt} \cdot \Delta {Do} \geq \tfrac{{1}}{{2}}",
        slots={"Dt": S(_TIME_POOL), "Do": S(_FREQ_POOL)},
    ),
    Template(
        name="poisson_summation",
        latex=r"\sum{lim_mod}_{{{nn}=-\infty}}^{{\infty}} {ff}({nn}) = \sum{lim_mod}_{{k=-\infty}}^{{\infty}} \hat{{{ff}}}(k)",
        slots={"lim_mod": S(("", r"\limits")), "ff": S(_FUNC_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="transfer_function",
        latex=r"H({fo}) = \frac{{{yy}({fo})}}{{{xx}({fo})}}",
        slots={"xx": S(_SIGNAL_POOL), "yy": X(_SIGNAL_POOL, ("xx",)), "fo": S(_FREQ_POOL)},
    ),
]

# ---------------------------------------------------------------------------
# Part B: New flat templates
# ---------------------------------------------------------------------------

# Continuous FT properties (8)
_PART_B_CFT: list[Template] = [
    Template(
        name="ft_linearity",
        latex=r"\widehat{{a\,{ff}+b\,{gg}}}({xi}) = a\,\hat{{{ff}}}({xi}) + b\,\hat{{{gg}}}({xi})",
        slots={"ff": S(_FUNC_POOL), "gg": X(_FUNC_POOL, ("ff",)), "xi": S(_FREQ_POOL)},
    ),
    Template(
        name="ft_time_shift",
        latex=r"\widehat{{{ff}(\cdot - {aa})}}({xi}) = e^{{-2\pi i\,{aa}\,{xi}}}\,\hat{{{ff}}}({xi})",
        slots={"ff": S(_FUNC_POOL), "xi": S(_FREQ_POOL), "aa": S(_COEFF_POOL)},
    ),
    Template(
        name="ft_modulation",
        latex=r"\widehat{{e^{{2\pi i\,{aa}\,{tt}}}\,{ff}}}({xi}) = \hat{{{ff}}}({xi} - {aa})",
        slots={"ff": S(_FUNC_POOL), "xi": S(_FREQ_POOL), "aa": S(_COEFF_POOL), "tt": S(_TIME_POOL)},
    ),
    Template(
        name="ft_scaling",
        latex=r"\widehat{{{ff}({aa}\,{tt})}}({xi}) = \frac{{1}}{{|{aa}|}}\,\hat{{{ff}}}\!\left(\frac{{{xi}}}{{{aa}}}\right)",
        slots={"ff": S(_FUNC_POOL), "xi": S(_FREQ_POOL), "aa": S(_COEFF_POOL), "tt": S(_TIME_POOL)},
    ),
    Template(
        name="ft_derivative",
        latex=r"\widehat{{{ff}^{{({nn})}}}}({xi}) = (2\pi i\,{xi})^{{{nn}}}\,\hat{{{ff}}}({xi})",
        slots={"ff": S(_FUNC_POOL), "xi": S(_FREQ_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="ft_duality",
        latex=r"\widehat{{\hat{{{ff}}}}}({tt}) = {ff}(-{tt})",
        slots={"ff": S(_FUNC_POOL), "tt": S(_TIME_POOL)},
    ),
    Template(
        name="riemann_lebesgue",
        latex=r"\lim_{{|{xi}|\to\infty}} \hat{{{ff}}}({xi}) = 0",
        slots={"ff": S(_FUNC_POOL), "xi": S(_FREQ_POOL)},
    ),
    Template(
        name="plancherel_theorem",
        latex=r"\langle \hat{{{ff}}},\, \hat{{{gg}}} \rangle = \langle {ff},\, {gg} \rangle_{{L^2}}",
        slots={"ff": S(_FUNC_POOL), "gg": X(_FUNC_POOL, ("ff",))},
    ),
]

# Fourier series (8)
_PART_B_FS: list[Template] = [
    Template(
        name="fourier_series_real",
        latex=(
            r"{ff}({vv}) = \frac{{a_0}}{{2}}"
            r" + \sum{lim_mod}_{{n=1}}^\infty \Bigl(a_n\cos\tfrac{{2\pi n\,{vv}}}{{{T}}}"
            r" + b_n\sin\tfrac{{2\pi n\,{vv}}}{{{T}}}\Bigr)"
        ),
        slots={"lim_mod": S(("", r"\limits")), "ff": S(_FUNC_POOL), "T": S(_PERIOD_POOL), "vv": S(_TIME_POOL)},
    ),
    Template(
        name="fourier_coeff_cosine",
        latex=r"a_{{{nn}}} = \frac{{2}}{{{T}}}\int_0^{{{T}}} {ff}({vv})\cos\tfrac{{2\pi\,{nn}\,{vv}}}{{{T}}}\,d{vv}",
        slots={"ff": S(_FUNC_POOL), "T": S(_PERIOD_POOL), "nn": S(_IDX_POOL), "vv": S(_TIME_POOL)},
    ),
    Template(
        name="fourier_coeff_sine",
        latex=r"b_{{{nn}}} = \frac{{2}}{{{T}}}\int_0^{{{T}}} {ff}({vv})\sin\tfrac{{2\pi\,{nn}\,{vv}}}{{{T}}}\,d{vv}",
        slots={"ff": S(_FUNC_POOL), "T": S(_PERIOD_POOL), "nn": S(_IDX_POOL), "vv": S(_TIME_POOL)},
    ),
    Template(
        name="dirichlet_kernel",
        latex=r"D_{{{nn}}}({vv}) = \sum{lim_mod}_{{k=-{nn}}}^{{{nn}}} e^{{ik{vv}}} = \frac{{\sin\bigl(({nn}+\tfrac{{1}}{{2}}){vv}\bigr)}}{{\sin({vv}/2)}}",
        slots={"lim_mod": S(("", r"\limits")), "nn": S(_IDX_POOL), "vv": S(_TIME_POOL)},
    ),
    Template(
        name="fejer_kernel",
        latex=r"F_{{{nn}}}({vv}) = \frac{{1}}{{{nn}}}\sum{lim_mod}_{{k=0}}^{{{nn}-1}} D_k({vv})",
        slots={"lim_mod": S(("", r"\limits")), "nn": S(_IDX_POOL), "vv": S(_TIME_POOL)},
    ),
    Template(
        name="parseval_series",
        latex=r"\sum{lim_mod}_{{n=-\infty}}^\infty |c_n|^2 = \frac{{1}}{{{T}}}\int_0^{{{T}}} |{ff}({vv})|^2\,d{vv}",
        slots={"lim_mod": S(("", r"\limits")), "ff": S(_FUNC_POOL), "T": S(_PERIOD_POOL), "vv": S(_TIME_POOL)},
    ),
    Template(
        name="bessel_inequality",
        latex=r"\sum{lim_mod}_{{n=0}}^\infty \bigl|\langle {ff},\, e_n\rangle\bigr|^2 \leq \|{ff}\|^2",
        slots={"lim_mod": S(("", r"\limits")), "ff": S(_FUNC_POOL)},
    ),
    Template(
        name="orthonormality_trig",
        latex=(
            r"\frac{{1}}{{{T}}}\int_0^{{{T}}} e^{{2\pi i\,{nn}\,{vv}/{T}}}"
            r"\,e^{{-2\pi i\,{mm}\,{vv}/{T}}}\,d{vv} = \delta_{{{nn}{mm}}}"
        ),
        slots={
            "T": S(_PERIOD_POOL),
            "nn": S(_IDX_POOL),
            "mm": X(_IDX_POOL, ("nn",)),
            "vv": S(_TIME_POOL),
        },
    ),
]

# DFT / FFT (6)
_PART_B_DFT: list[Template] = [
    Template(
        name="idft",
        latex=r"{xx}[{nn}] = \frac{{1}}{{{N}}}\sum{lim_mod}_{{k=0}}^{{{N}-1}} X[k]\,e^{{2\pi i k\,{nn}/{N}}}",
        slots={"lim_mod": S(("", r"\limits")), "xx": S(_SIGNAL_POOL), "N": S(_SIZE_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="dft_shift",
        latex=r"X[k]\,e^{{2\pi i k\,{nn}/{N}}} \;\longleftrightarrow\; {xx}[{nn} - n_0]",
        slots={"xx": S(_SIGNAL_POOL), "N": S(_SIZE_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="dft_parseval",
        latex=r"\sum{lim_mod}_{{{nn}=0}}^{{{N}-1}}|{xx}[{nn}]|^2 = \frac{{1}}{{{N}}}\sum{lim_mod}_{{k=0}}^{{{N}-1}}|X[k]|^2",
        slots={"lim_mod": S(("", r"\limits")), "xx": S(_SIGNAL_POOL), "N": S(_SIZE_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="dft_convolution",
        latex=r"{xx}[n] * {hh}[n] \;\longleftrightarrow\; X[k]\,H[k]",
        slots={"xx": S(_SIGNAL_POOL), "hh": S(_KERNEL_POOL)},
    ),
    Template(
        name="twiddle_factor",
        latex=r"W_{{{N}}} = e^{{-2\pi i/{N}}},\quad X[k] = \sum{lim_mod}_{{{nn}=0}}^{{{N}-1}} {xx}[{nn}]\,W_{{{N}}}^{{{nn} k}}",
        slots={"lim_mod": S(("", r"\limits")), "xx": S(_SIGNAL_POOL), "N": S(_SIZE_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="nyquist_sampling",
        latex=r"{fs} \geq 2\,{fb}",
        slots={"fs": S(_FREQ_POOL), "fb": X(_FREQ_POOL, ("fs",))},
    ),
]

# Laplace transform (6)
_PART_B_LAP: list[Template] = [
    Template(
        name="laplace_derivative",
        latex=r"\mathcal{{L}}\left\{{{ff}'(t)\right\}}({ss}) = {ss}\,F({ss}) - {ff}(0)",
        slots={"ff": S(_FUNC_POOL), "ss": S(_S_POOL)},
    ),
    Template(
        name="laplace_convolution",
        latex=r"\mathcal{{L}}\left\{{{ff} * {gg}\right\}}({ss}) = F({ss})\cdot G({ss})",
        slots={"ff": S(_FUNC_POOL), "gg": X(_FUNC_POOL, ("ff",)), "ss": S(_S_POOL)},
    ),
    Template(
        name="laplace_shift",
        latex=r"\mathcal{{L}}\left\{{e^{{{aa}\,t}}\,{ff}(t)\right\}}({ss}) = F({ss} - {aa})",
        slots={"ff": S(_FUNC_POOL), "ss": S(_S_POOL), "aa": S(_COEFF_POOL)},
    ),
    Template(
        name="laplace_scale",
        latex=r"\mathcal{{L}}\left\{{{ff}({aa}\,t)\right\}}({ss}) = \frac{{1}}{{|{aa}|}}\,F\!\left(\frac{{{ss}}}{{{aa}}}\right)",
        slots={"ff": S(_FUNC_POOL), "ss": S(_S_POOL), "aa": S(_COEFF_POOL)},
    ),
    Template(
        name="laplace_initial_value",
        latex=r"\lim_{{t\to 0^+}} {ff}(t) = \lim_{{{ss}\to\infty}} {ss}\,F({ss})",
        slots={"ff": S(_FUNC_POOL), "ss": S(_S_POOL)},
    ),
    Template(
        name="laplace_final_value",
        latex=r"\lim_{{t\to\infty}} {ff}(t) = \lim_{{{ss}\to 0}} {ss}\,F({ss})",
        slots={"ff": S(_FUNC_POOL), "ss": S(_S_POOL)},
    ),
]

# Z-transform (4)
_PART_B_ZT: list[Template] = [
    Template(
        name="z_transform_shift",
        latex=r"\mathcal{{Z}}\left\{{{xx}[n-{nn}]\right\}}(z) = z^{{-{nn}}}\,X(z)",
        slots={"xx": S(_SIGNAL_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="z_transform_convolution",
        latex=r"\mathcal{{Z}}\left\{{{xx} * {hh}\right\}}(z) = X(z)\,H(z)",
        slots={"xx": S(_SIGNAL_POOL), "hh": S(_KERNEL_POOL)},
    ),
    Template(
        name="z_transform_final_value",
        latex=r"\lim_{{n\to\infty}} {xx}[n] = \lim_{{z\to 1}}(z-1)\,X(z)",
        slots={"xx": S(_SIGNAL_POOL)},
    ),
    Template(
        name="z_transform_poles",
        latex=r"X(z) = \frac{{B(z)}}{{A(z)}},\quad \text{{poles: }} A(z) = 0",
        slots={"xx": S(_SIGNAL_POOL)},
    ),
]

# Harmonic analysis / functional analysis (6)
_PART_B_HA: list[Template] = [
    Template(
        name="l2_norm_def",
        latex=r"\|{ff}\|_{{L^2}} = \left(\int{lim_mod}_{{-\infty}}^{{\infty}} |{ff}({tt})|^2\,d{tt}\right)^{{1/2}}",
        slots={"lim_mod": S(("", r"\limits")), "ff": S(_FUNC_POOL), "tt": S(_TIME_POOL)},
    ),
    Template(
        name="inner_product_L2",
        latex=r"\langle {ff},\, {gg}\rangle = \int{lim_mod}_{{-\infty}}^{{\infty}} {ff}({tt})\,\overline{{{gg}({tt})}}\,d{tt}",
        slots={"lim_mod": S(("", r"\limits")), "ff": S(_FUNC_POOL), "gg": X(_FUNC_POOL, ("ff",)), "tt": S(_TIME_POOL)},
    ),
    Template(
        name="hilbert_transform",
        latex=(
            r"\mathcal{{H}}\left\{{{ff}\right\}}({tt})"
            r" = \frac{{1}}{{\pi}}\,\mathrm{{p.v.}}"
            r"\int{lim_mod}_{{-\infty}}^\infty \frac{{{ff}({tau})}}{{{tt} - {tau}}}\,d{tau}"
        ),
        slots={"lim_mod": S(("", r"\limits")), "ff": S(_FUNC_POOL), "tt": S(_TIME_POOL), "tau": S(_TVAR_POOL)},
    ),
    Template(
        name="autocorrelation",
        latex=r"R_{{{ff}{ff}}}({tau}) = \int{lim_mod}_{{-\infty}}^\infty {ff}({tt})\,\overline{{{ff}({tt}-{tau})}}\,d{tt}",
        slots={"lim_mod": S(("", r"\limits")), "ff": S(_FUNC_POOL), "tt": S(_TIME_POOL), "tau": S(_TVAR_POOL)},
    ),
    Template(
        name="cross_correlation",
        latex=r"R_{{{ff} {gg}}}({tau}) = \int{lim_mod}_{{-\infty}}^\infty {ff}({tt})\,\overline{{{gg}({tt}-{tau})}}\,d{tt}",
        slots={
            "lim_mod": S(("", r"\limits")),
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "tt": S(_TIME_POOL),
            "tau": S(_TVAR_POOL),
        },
    ),
    Template(
        name="power_spectral_density",
        latex=r"S_{{{ff}}}({xi}) = \bigl|\hat{{{ff}}}({xi})\bigr|^2",
        slots={"ff": S(_FUNC_POOL), "xi": S(_FREQ_POOL)},
    ),
]

# Wavelet / STFT (4)
_PART_B_WAV: list[Template] = [
    Template(
        name="stft",
        latex=(
            r"V_{{{ff}}}({xi},\,{tau})"
            r" = \int{lim_mod}_{{-\infty}}^\infty {ff}({tt})\,\overline{{{gg}({tt}-{tau})}}"
            r"\,e^{{-2\pi i\,{xi}\,{tt}}}\,d{tt}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "ff": S(_FUNC_POOL),
            "gg": X(_FUNC_POOL, ("ff",)),
            "xi": S(_FREQ_POOL),
            "tau": S(_TVAR_POOL),
            "tt": S(_TIME_POOL),
        },
    ),
    Template(
        name="cwt",
        latex=(
            r"W_{{{ff}}}(a,b)"
            r" = \frac{{1}}{{\sqrt{{|a|}}}}"
            r"\int{lim_mod}_{{-\infty}}^\infty {ff}({tt})\,\overline{{\psi\!\left(\frac{{{tt}-b}}{{a}}\right)}}\,d{tt}"
        ),
        slots={"lim_mod": S(("", r"\limits")), "ff": S(_FUNC_POOL), "tt": S(_TIME_POOL)},
    ),
    Template(
        name="admissibility_condition",
        latex=r"C_\psi = \int_0^\infty \frac{{|\hat{{\psi}}({xi})|^2}}{{{xi}}}\,d{xi} < \infty",
        slots={"xi": S(_FREQ_POOL)},
    ),
    Template(
        name="wavelet_reconstruction",
        latex=(
            r"{ff}({tt}) = \frac{{1}}{{C_\psi}}"
            r"\int_0^\infty\!\int{lim_mod}_{{-\infty}}^\infty W_{{{ff}}}(a,b)\,\psi_{{a,b}}({tt})"
            r"\,\frac{{db\,da}}{{a^2}}"
        ),
        slots={"lim_mod": S(("", r"\limits")), "ff": S(_FUNC_POOL), "tt": S(_TIME_POOL)},
    ),
]

# Other transforms (4)
_PART_B_OTH: list[Template] = [
    Template(
        name="dct_type2",
        latex=r"X[k] = \sum{lim_mod}_{{{nn}=0}}^{{{N}-1}} {xx}[{nn}]\cos\!\left(\frac{{\pi(2{nn}+1)k}}{{2{N}}}\right)",
        slots={"lim_mod": S(("", r"\limits")), "xx": S(_SIGNAL_POOL), "N": S(_SIZE_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="dst_type2",
        latex=r"X[k] = \sum{lim_mod}_{{{nn}=0}}^{{{N}-1}} {xx}[{nn}]\sin\!\left(\frac{{\pi(2{nn}+1)k}}{{2{N}}}\right)",
        slots={"lim_mod": S(("", r"\limits")), "xx": S(_SIGNAL_POOL), "N": S(_SIZE_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="hadamard_transform",
        latex=(
            r"H_{{{nn}}} = H_1^{{\otimes {nn}}},\quad"
            r" H_1 = \tfrac{{1}}{{\sqrt{{2}}}}\begin{{pmatrix}}1&1\\1&-1\end{{pmatrix}}"
        ),
        slots={"nn": S(_IDX_POOL)},
    ),
    Template(
        name="mellin_transform",
        latex=r"\mathcal{{M}}\left\{{{ff}\right\}}({ss}) = \int_0^\infty {ff}({tt})\,{tt}^{{{ss}-1}}\,d{tt}",
        slots={"ff": S(_FUNC_POOL), "ss": S(_S_POOL), "tt": S(_TIME_POOL)},
    ),
]

# Filter theory (4)
_PART_B_FILT: list[Template] = [
    Template(
        name="frequency_response",
        latex=r"H({fo}) = \sum{lim_mod}_{{n=-\infty}}^\infty {hh}[n]\,e^{{-i\,{fo}\,n}}",
        slots={"lim_mod": S(("", r"\limits")), "hh": S(_KERNEL_POOL), "fo": S(_FREQ_POOL)},
    ),
    Template(
        name="impulse_response_convolution",
        latex=r"{yy}[n] = \sum{lim_mod}_{{k=-\infty}}^\infty {hh}[k]\,{xx}[n-k]",
        slots={
            "lim_mod": S(("", r"\limits")),
            "xx": S(_SIGNAL_POOL),
            "yy": X(_SIGNAL_POOL, ("xx",)),
            "hh": S(_KERNEL_POOL),
        },
    ),
    Template(
        name="group_delay",
        latex=r"\tau_{{{hh}}}({fo}) = -\frac{{d}}{{d{fo}}}\angle H_{{{hh}}}({fo})",
        slots={"hh": S(_KERNEL_POOL), "fo": S(_FREQ_POOL)},
    ),
    Template(
        name="bode_magnitude",
        latex=r"|H_{{{hh}}}(j{fo})| = \frac{{\prod_k |{fo} - z_k|}}{{\prod_k |{fo} - p_k|}}",
        slots={"hh": S(_KERNEL_POOL), "fo": S(_FREQ_POOL)},
    ),
]

_PART_B: list[Template] = (
    _PART_B_CFT
    + _PART_B_FS
    + _PART_B_DFT
    + _PART_B_LAP
    + _PART_B_ZT
    + _PART_B_HA
    + _PART_B_WAV
    + _PART_B_OTH
    + _PART_B_FILT
)

# ---------------------------------------------------------------------------
# Part C: High-n_eff function-pair templates (6)
# ---------------------------------------------------------------------------

_PART_C: list[Template] = [
    Template(
        name="plancherel_pair",
        latex=r"\langle {fn1},\, {fn2} \rangle_{{L^2}} = \langle \hat{{{fn1}}},\, \hat{{{fn2}}} \rangle_{{L^2}}",
        slots={"fn1": E(_fn_rich_nosub, n=100), "fn2": E(_fn_rich_nosub, n=100)},
    ),
    Template(
        name="convolution_pair_ft",
        latex=r"\widehat{{{fn1} * {fn2}}}({xi}) = \hat{{{fn1}}}({xi})\cdot\hat{{{fn2}}}({xi})",
        slots={"fn1": E(_fn_rich_nosub, n=100), "fn2": E(_fn_rich_nosub, n=100), "xi": S(_FREQ_POOL)},
    ),
    Template(
        name="laplace_superposition",
        latex=(
            r"\mathcal{{L}}\left\{{{fn1} + {fn2}\right\}}({ss})"
            r" = \mathcal{{L}}\left\{{{fn1}\right\}}({ss})"
            r" + \mathcal{{L}}\left\{{{fn2}\right\}}({ss})"
        ),
        slots={"fn1": E(_fn_rich_nosub, n=100), "fn2": E(_fn_rich_nosub, n=100), "ss": S(_S_POOL)},
    ),
    Template(
        name="ft_product_convolution",
        latex=r"\widehat{{{fn1} \cdot {fn2}}}({xi}) = \hat{{{fn1}}}({xi}) * \hat{{{fn2}}}({xi})",
        slots={"fn1": E(_fn_rich_nosub, n=100), "fn2": E(_fn_rich_nosub, n=100), "xi": S(_FREQ_POOL)},
    ),
    Template(
        name="cross_correlation_ft",
        latex=r"\widehat{{R_{{{fn1} {fn2}}}}}({xi}) = \overline{{\hat{{{fn1}}}({xi})}}\,\hat{{{fn2}}}({xi})",
        slots={"fn1": E(_fn_rich_nosub, n=100), "fn2": E(_fn_rich_nosub, n=100), "xi": S(_FREQ_POOL)},
    ),
    Template(
        name="z_superposition",
        latex=(
            r"\mathcal{{Z}}\left\{{{fn1}[n] + {fn2}[n]\right\}}(z)"
            r" = X_1(z) + X_2(z)"
        ),
        slots={"fn1": E(_fn_rich_nosub, n=100), "fn2": E(_fn_rich_nosub, n=100)},
    ),
    Template(
        name="ft_3way_convolution",
        latex=r"\widehat{{{fn1} * {fn2} * {fn3}}}({xi}) = \hat{{{fn1}}}({xi})\cdot\hat{{{fn2}}}({xi})\cdot\hat{{{fn3}}}({xi})",
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "fn3": E(_fn_rich_nosub, n=100),
            "xi": S(_FREQ_POOL),
        },
    ),
    Template(
        name="ft_inversion_pair",
        latex=r"\widehat{{\hat{{{fn1}}}}}({tt}) = {fn2}(-{tt})",
        slots={"fn1": E(_fn_rich_nosub, n=100), "fn2": E(_fn_rich_nosub, n=100), "tt": S(_TIME_POOL)},
    ),
    Template(
        name="autocorrelation_pair",
        latex=(
            r"R_{{{fn1} {fn2}}}({tau})"
            r" = \int{lim_mod}_{{-\infty}}^\infty {fn1}({tt})\,\overline{{{fn2}({tt}-{tau})}}\,d{tt}"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "tau": S(_TVAR_POOL),
            "tt": S(_TIME_POOL),
        },
    ),
    Template(
        name="l2_triangle_pair",
        latex=r"\|{fn1} + {fn2}\|_{{L^2}}^2 = \|{fn1}\|^2 + 2\,\mathrm{{Re}}\langle {fn1},{fn2}\rangle + \|{fn2}\|^2",
        slots={"fn1": E(_fn_rich_nosub, n=100), "fn2": E(_fn_rich_nosub, n=100)},
    ),
    Template(
        name="laplace_product_pair",
        latex=(
            r"\mathcal{{L}}\left\{{{fn1}(t) \cdot {fn2}(t)\right\}}({ss})"
            r" = \frac{{1}}{{2\pi i}}\int{lim_mod}_{{{cc}-i\infty}}^{{{cc}+i\infty}} F_1(\sigma)\,F_2({ss}-\sigma)\,d\sigma"
        ),
        slots={
            "lim_mod": S(("", r"\limits")),
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "ss": S(_S_POOL),
            "cc": S(_COEFF_POOL),
        },
    ),
    Template(
        name="ft_modulation_pair",
        latex=r"\widehat{{e^{{2\pi i\,{aa}\,{tt}}}\,{fn1}}}({xi}) = \hat{{{fn2}}}({xi} - {aa})",
        slots={
            "fn1": E(_fn_rich_nosub, n=100),
            "fn2": E(_fn_rich_nosub, n=100),
            "xi": S(_FREQ_POOL),
            "aa": S(_COEFF_POOL),
            "tt": S(_TIME_POOL),
        },
    ),
]

# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_PART_D: list[Template] = [
    Template(
        name="fourier_basis_orthogonality",
        latex=r"\langle e_{{{nn}}}, e_{{{mm}}} \rangle_{{L^2[0,{T}]}} = 0, \quad {nn} \neq {mm}",
        slots={"nn": S(_IDX_POOL), "mm": X(_IDX_POOL, ("nn",)), "T": S(_PERIOD_POOL)},
    ),
    Template(
        name="sin_cos_orthogonality",
        latex=r"\cos\!\tfrac{{2\pi\,{nn}\,{tt}}}{{{T}}} \perp \sin\!\tfrac{{2\pi\,{mm}\,{tt}}}{{{T}}} \;\text{{in }}\; L^2[0,{T}]",
        slots={"nn": S(_IDX_POOL), "mm": X(_IDX_POOL, ("nn",)), "T": S(_PERIOD_POOL), "tt": S(_TIME_POOL)},
    ),
    Template(
        name="hilbert_basis_perp",
        latex=r"\langle {ff}, e_{{{nn}}} \rangle = 0 \;\forall\, {nn} \implies {ff} \perp \overline{{\mathrm{{span}}}}\{{e_n\}}",
        slots={"ff": S(_FUNC_POOL), "nn": S(_IDX_POOL)},
    ),
    Template(
        name="fourier_truncation_approx",
        latex=r"{ff}({tt}) \approx \sum{lim_mod}_{{|k| \leq {N}}} \hat{{{ff}}}(k)\, e^{{2\pi i\,k\,{tt}/{T}}}",
        slots={
            "lim_mod": S(("", r"\limits")),
            "ff": S(_FUNC_POOL),
            "tt": S(_TIME_POOL),
            "T": S(_PERIOD_POOL),
            "N": S(_SIZE_POOL),
        },
    ),
    Template(
        name="dft_cft_approx",
        latex=r"X[{nn}] \approx \hat{{x}}\!\left(\frac{{{nn}}}{{{N}\,\Delta t}}\right), \quad {nn} = 0,\ldots,{N}-1",
        slots={"nn": S(_IDX_POOL), "N": S(_SIZE_POOL)},
    ),
]

_PART_E: list[Template] = [
    Template(
        name="fourier_real_part",
        latex=r"\Re(\hat{{{ff}}}({xi})) = \int{lim_mod}_{{-\infty}}^{{\infty}} {ff}({tt}) \cos(2\pi {xi} {tt})\,d{tt}",
        slots={"lim_mod": S(("", r"\limits")), "ff": S(_FUNC_POOL), "xi": S(_FREQ_POOL), "tt": S(_TIME_POOL)},
    ),
    Template(
        name="fourier_imaginary_part",
        latex=r"\Im(\hat{{{ff}}}({xi})) = -\int{lim_mod}_{{-\infty}}^{{\infty}} {ff}({tt}) \sin(2\pi {xi} {tt})\,d{tt}",
        slots={"lim_mod": S(("", r"\limits")), "ff": S(_FUNC_POOL), "xi": S(_FREQ_POOL), "tt": S(_TIME_POOL)},
    ),
    Template(
        name="power_spectral_density_Re",
        latex=r"S_{{{ff}}}({xi}) = \Re\!\left(\int{lim_mod}_{{-\infty}}^{{\infty}} R_{{{ff}}}(\tau)\,e^{{-2\pi i {xi} \tau}}\,d\tau\right)",
        slots={"lim_mod": S(("", r"\limits")), "ff": S(_FUNC_POOL), "xi": S(_FREQ_POOL)},
    ),
]

_FOURIER_TEMPLATES: list[Template] = _PART_A + _PART_B + _PART_C + _PART_D + _PART_E

_W_FOURIER: list[float] = compute_weights(_FOURIER_TEMPLATES)

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
