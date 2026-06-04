# Trigonometry Domain — Template Reference

Generated from `formula_combinatorics/domains/algebra.py` (`_TRIG_TEMPLATES`).

## Slot Pools

| Pool | Symbols | Size |
|---|---|---|
| `_VARS` | x y z t u v r s | 8 |
| `_SCALARS` | a b c d k m n p q | 9 |
| `_TRIG_ARG_POOL` | `_SCALARS` + θ φ ϕ ψ ω α β γ δ | 18 |
| `_SIDES_POOL` | a b c p q r | 6 |
| `_ANGLES_POOL` | A B C P Q R | 6 |
| `_GEO_N` | n m N M K p r | 7 |

`idx=0.35` decoration multiplies effective pool size by `1 + 0.35×7 = 3.45`.
So `_TRIG_ARG_POOL` with `idx=0.35` → eff\_size ≈ 62; `_VARS` with `idx=0.35` → eff\_size ≈ 28.

**n\_eff and function-name selectors**: each `E(_sin_nm, n=2)` / `E(_cos_nm, n=2)` slot contributes
a factor of 2 to n\_eff (two notation forms: `\sin` / `\text{sine}`, etc.). Templates with two such
selectors (e.g. sin + cos) get a 4× multiplier over the slot-only product; templates with one get 2×.

## Summary Statistics

| Stat | Value |
|---|---|
| Templates | 48 |
| Total n\_eff | 51,390,653 |
| Median birthday horizon | ~939 |
| Batch uniqueness (1k) | 99.9% |
| Domain weight | 4% |

---

## Templates

### Pythagorean Identity (`pythagorean_identity`)

**n\_eff: 13,822**

| Variant | Slots | n\_eff |
|---|---|---|
| `pythagorean_identity_bare` | x∈VARS idx=0.35 | 110 |
| `pythagorean_identity_coeff` | a∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 6,856 |
| `pythagorean_identity_coeff_sq` | a∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 6,856 |

---

### Double-Angle Sine (`double_angle_sin`)

**n\_eff: 7,949**
Slots: `coeff` ← `_sin_dbl_coeff_sub` (n≈72), `x`∈VARS idx=0.35 (≈28).
Sub-generator: draws one of `{2a, 3a, a, a²}` with `a ∈ _TRIG_ARG_POOL`.

---

### Double-Angle Cosine (`double_angle_cos`)

**n\_eff: 7,728**
Slots: `coeff` ← `_cos_dbl_coeff_sub` (n≈70), `x`∈VARS idx=0.35.
Sub-generator: draws one of `{2, 3, a, 2a}` with `a ∈ _TRIG_ARG_POOL`.

---

### Sine Addition (`sin_addition`)

**n\_eff: 14,569**
`a`∈TRIG\_ARG idx=0.35, `b`∈TRIG\_ARG∖{a} idx=0.35. ExcludeSlot.
Note: template uses `\pm` so it renders both addition and subtraction cases in a single expression.

---

### Cosine Addition (`cos_addition`)

**n\_eff: 14,569**
`a`∈TRIG\_ARG idx=0.35, `b`∈TRIG\_ARG∖{a} idx=0.35.

---

### Sine Subtraction (`sin_subtraction`)

**n\_eff: 14,569**
`a`∈TRIG\_ARG idx=0.35, `b`∈TRIG\_ARG∖{a} idx=0.35.

---

### Cosine Subtraction (`cos_subtraction`)

**n\_eff: 14,569**
`a`∈TRIG\_ARG idx=0.35, `b`∈TRIG\_ARG∖{a} idx=0.35.

---

### Tan/Cot Ratio (`tan_cot_ratio`)

**n\_eff: 27,865**

| Variant | Slots | n\_eff |
|---|---|---|
| `tan_ratio_bare` | x∈VARS idx=0.35 | 221 |
| `tan_ratio_coeff` | a∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 13,712 |
| `cot_ratio_bare` | x∈VARS idx=0.35 | 221 |
| `cot_ratio_coeff` | a∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 13,712 |

---

### Euler Formula (`euler_formula`)

**n\_eff: 6,400**
Sub-generator `_euler_arg_sub` (n≈800): draws one of `{x, ax, a+b, a}` with a,b∈TRIG\_ARG\_POOL, x∈VARS.

| Variant | n\_eff |
|---|---|
| `euler_formula_pos` | 3,200 |
| `euler_formula_neg` | 3,200 |

---

### Law of Cosines (`law_of_cosines`)

**n\_eff: 102,002**
`s0`∈SIDES idx=0.35, `s1`∈SIDES∖{s0} idx=0.35, `s2`∈SIDES∖{s0,s1} idx=0.35, `A`∈ANGLES idx=0.35.

---

### Law of Sines (`law_of_sines`)

**n\_eff: 24,281,587** ← dominant template
`s0,s1,s2`∈SIDES (ExcludeSlot chain, idx=0.35), `A0,A1,A2`∈ANGLES (ExcludeSlot chain, idx=0.35).

---

### Trig Integral (`trig_integral`)

**n\_eff: 18,854**
`fn`∈{sin,cos,tan,sinh,cosh,sec,sine,cosine,tangent,sh,ch} (11, via `_trig_fn_int_sub`), `coeff`∈TRIG\_ARG idx=0.35 (≈62), `x`∈VARS idx=0.35 (≈28).

---

### Arctan Ratio (`arctan_ratio`)

**n\_eff: 14,569**
`a`∈TRIG\_ARG idx=0.35, `b`∈TRIG\_ARG∖{a} idx=0.35.

---

### Pythagorean Tan/Cot (`pythagorean_tan_cot`)

**n\_eff: 7,200**
Sub-generator `_trig_pyth_arg_sub` (n≈900): draws one of `{x, ax, a²x}` with a∈TRIG\_ARG\_POOL, x∈VARS.

| Variant | n\_eff |
|---|---|
| `pythagorean_tan` | 3,600 |
| `pythagorean_cot` | 3,600 |

---

### Euler Sin/Cos Exponential (`euler_sin_cos_exponential`)

**n\_eff: 7,104**

| Variant | Slots | n\_eff |
|---|---|---|
| `euler_exp_sin_cos_bare` | a∈TRIG\_ARG idx=0.35 | 248 |
| `euler_exp_sin_cos_coeff` | c∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 6,856 |

---

### Product-to-Sum: sin·cos (`product_to_sum_sin_cos`)

**n\_eff: 14,569**
`a`∈TRIG\_ARG idx=0.35, `b`∈TRIG\_ARG∖{a} idx=0.35.

---

### Product-to-Sum: sin·sin (`product_sum_sin_sin`)

**n\_eff: 14,569**
`a`∈TRIG\_ARG idx=0.35, `b`∈TRIG\_ARG∖{a} idx=0.35.

---

### Product-to-Sum: cos·cos (`product_to_sum_cos_cos`)

**n\_eff: 7,284**
`a`∈TRIG\_ARG idx=0.35, `b`∈TRIG\_ARG∖{a} idx=0.35.
Lower than sin·cos and sin·sin (14,569 each) because this template has one function-name selector (`cn`) vs two (`sn`+`cn`).

---

### Half-Angle Formulas (`half_angle_formulas`)

**n\_eff: 6,966**

| Variant | Slots | n\_eff |
|---|---|---|
| `half_angle_bare` | x∈VARS idx=0.35 | 110 |
| `half_angle_coeff` | a∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 6,856 |

---

### Sum-to-Product: sin (`sum_to_product`)

**n\_eff: 14,569**
`a`∈TRIG\_ARG idx=0.35, `b`∈TRIG\_ARG∖{a} idx=0.35.

---

### Sum-to-Product: cos (`sum_to_product_cos`)

**n\_eff: 36,422**

| Variant | Slots | n\_eff |
|---|---|---|
| `sum_to_product_cos_plus` | a∈TRIG\_ARG idx=0.35, b∈TRIG\_ARG∖{a} idx=0.35 | 7,284 |
| `sum_to_product_cos_minus` | a∈TRIG\_ARG idx=0.35, b∈TRIG\_ARG∖{a} idx=0.35 | 14,569 |
| `sum_to_product_sin_minus` | a∈TRIG\_ARG idx=0.35, b∈TRIG\_ARG∖{a} idx=0.35 | 14,569 |

`cos_plus` has one name selector (`cn`) while `cos_minus`/`sin_minus` have two (`sn`+`cn`), hence the 2× difference.

---

### Hyperbolic Pythagorean (`hyperbolic_pythagorean`)

**n\_eff: 4,800**
Sub-generator `_hyp_arg_sub` (n≈1200): draws one of `{ax, a+b, bx+a}` with a,b∈TRIG\_ARG\_POOL, x∈VARS.

---

### Hyperbolic Addition: sinh (`hyperbolic_addition`)

**n\_eff: 14,569**
`a`∈TRIG\_ARG idx=0.35, `b`∈TRIG\_ARG∖{a} idx=0.35.

---

### Hyperbolic Addition: cosh (`hyperbolic_cosh_addition`)

**n\_eff: 14,569**
`a`∈TRIG\_ARG idx=0.35, `b`∈TRIG\_ARG∖{a} idx=0.35.

---

### Hyperbolic Subtraction (`hyperbolic_subtraction`)

**n\_eff: 29,137**

| Variant | n\_eff |
|---|---|
| `hyperbolic_sinh_subtraction` | 14,569 |
| `hyperbolic_cosh_subtraction` | 14,569 |

---

### Hyperbolic Double-Angle (`hyperbolic_double_angle`)

**n\_eff: 14,208**

| Variant | Slots | n\_eff |
|---|---|---|
| `hyperbolic_double_angle_sinh` | a∈TRIG\_ARG idx=0.35 | 248 |
| `hyperbolic_double_angle_cosh` | a∈TRIG\_ARG idx=0.35 | 248 |
| `hyperbolic_double_angle_sinh_coeff` | c∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 6,856 |
| `hyperbolic_double_angle_cosh_coeff` | c∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 6,856 |

---

### Hyperbolic Definitions (`hyperbolic_definitions`)

**n\_eff: 7,725**

| Variant | Slots | n\_eff |
|---|---|---|
| `sinh_definition` | x∈TRIG\_ARG idx=0.35 | 124 |
| `cosh_definition` | x∈TRIG\_ARG idx=0.35 | 124 |
| `tanh_definition` | x∈TRIG\_ARG idx=0.35 | 124 |
| `tanh_ratio` | x∈TRIG\_ARG idx=0.35 | 497 |
| `sinh_definition_coeff` | c∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 3,428 |
| `cosh_definition_coeff` | c∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 3,428 |

`tanh_ratio` is higher (497) because it uses three function-name selectors (thn, shn, chn).

---

### Inverse Hyperbolic in Log Form (`inverse_hyperbolic_log`)

**n\_eff: 15,674**
Mixed logarithm + radical notation. High OCR training value.

| Variant | Slots | n\_eff |
|---|---|---|
| `arcsinh_log` | x∈VARS idx=0.35 | 83 |
| `arccosh_log` | x∈VARS idx=0.35 | 83 |
| `arctanh_log` | x∈VARS idx=0.35 | 83 |
| `arcsinh_log_coeff` | c∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 5,142 |
| `arccosh_log_coeff` | c∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 5,142 |
| `arctanh_log_coeff` | c∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 5,142 |

Bare variants use 3-option name pools (arcsinh/arccosh/arctanh each have 3 forms), giving n\_eff = 3 × 28 = 83.

---

### Inverse Trig Derivative (`inverse_trig_derivative`)

**n\_eff: 21,313**

| Variant | Slots | n\_eff |
|---|---|---|
| `arcsin_derivative` | x∈VARS idx=0.35 | 110 |
| `arcsin_derivative_coeff` | c∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 6,856 |
| `arccos_derivative` | x∈VARS idx=0.35 | 110 |
| `arccos_derivative_coeff` | c∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 6,856 |
| `arctan_derivative` | x∈VARS idx=0.35 | 110 |
| `arctan_derivative_coeff` | c∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 6,856 |
| `arccot_derivative` | x∈VARS idx=0.35 | 83 |
| `arcsec_derivative` | x∈VARS idx=0.35 | 83 |
| `arccsc_derivative` | x∈VARS idx=0.35 | 83 |
| `arctanh_derivative` | x∈VARS idx=0.35 | 83 |
| `arcsinh_derivative` | x∈VARS idx=0.35 | 83 |

arcsin/arccos/arctan use 4-form name pools (n\_eff = 4×28 = 110); arccot/arcsec/arccsc/arctanh/arcsinh use 3-form pools (n\_eff = 3×28 = 83).

---

### Taylor Series: Trig (`taylor_series_trig`)

**n\_eff: 5,283**
Sub-generator `_taylor_arg_sub` (n≈450): draws one of `{x, ax}` with a∈TRIG\_ARG\_POOL, x∈VARS.

| Variant | Slots | n\_eff |
|---|---|---|
| `taylor_sin` | arg∈`_taylor_arg_sub` | 900 |
| `taylor_cos` | arg∈`_taylor_arg_sub` | 900 |
| `taylor_tan_approx` | x∈VARS idx=0.35 | 55 |
| `taylor_tan_approx_coeff` | c∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 3,428 |

`taylor_sin`/`taylor_cos` n\_eff ≈ 2 × 450 = 900 (name selector × sub-generator).

---

### Cofunction Identity (`cofunction_identity`)

**n\_eff: 20,899**

| Variant | Slots | n\_eff |
|---|---|---|
| `cofunction_sin_cos` | x∈VARS idx=0.35 | 110 |
| `cofunction_tan_cot` | x∈VARS idx=0.35 | 110 |
| `cofunction_sec_csc` | x∈VARS idx=0.35 | 110 |
| `cofunction_sin_cos_coeff` | a∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 6,856 |
| `cofunction_tan_cot_coeff` | a∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 6,856 |
| `cofunction_sec_csc_coeff` | a∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 6,856 |

---

### Inverse Trig Identities (`inverse_trig_identities`)

**n\_eff: 49,400**

| Variant | Slots | n\_eff |
|---|---|---|
| `arcsin_arccos_complement_bare` | x∈VARS idx=0.35 | 442 |
| `arcsin_arccos_complement_coeff` | a∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 27,423 |
| `arctan_reciprocal_bare` | x∈VARS idx=0.35 | 110 |
| `arctan_reciprocal_coeff` | a∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 6,856 |
| `arctan_addition` | a∈TRIG\_ARG idx=0.35, b∈TRIG\_ARG∖{a} idx=0.35 | 14,569 |

`arcsin_arccos_complement_bare` is higher (442) because it uses two 4-form name pools (arcsin + arccos): 4×4×28 = 448 ≈ 442.

---

### Parametric Pythagorean (`parametric_pythagorean`)

**n\_eff: 6,856**
`a`∈TRIG\_ARG idx=0.35, `x`∈VARS idx=0.35.

---

### Roots of Unity Sum (`roots_of_unity_sum`)

**n\_eff: 2,125**
`k`∈{j,k,l,m,r,s,t} (7), `n`∈{2,3,4,5,6,n,m,N,M,K,p} (11, via `_fourier_n_sub`), `x`∈VARS idx=0.35 (≈28).

---

### Triple-Angle Formulas (`triple_angle_formulas`)

**n\_eff: 7,228**

| Variant | Slots | n\_eff |
|---|---|---|
| `triple_angle_sin` | a∈TRIG\_ARG idx=0.35 | 124 |
| `triple_angle_cos` | a∈TRIG\_ARG idx=0.35 | 124 |
| `triple_angle_tan` | a∈TRIG\_ARG idx=0.35 | 124 |
| `triple_angle_sin_coeff` | c∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 3,428 |
| `triple_angle_cos_coeff` | c∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 3,428 |

---

### Power Reduction (`power_reduction`)

**n\_eff: 11,153**

| Variant | Slots | n\_eff |
|---|---|---|
| `power_reduction_sin2` | a∈TRIG\_ARG idx=0.35 | 248 |
| `power_reduction_cos2` | a∈TRIG\_ARG idx=0.35 | 124 |
| `power_reduction_sincos` | a∈TRIG\_ARG idx=0.35 | 248 |
| `power_reduction_sin3` | a∈TRIG\_ARG idx=0.35 | 124 |
| `power_reduction_cos3` | a∈TRIG\_ARG idx=0.35 | 124 |
| `power_reduction_sin2_coeff` | c∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 6,856 |
| `power_reduction_cos2_coeff` | c∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 3,428 |

sin² and sincos variants have two name selectors (sn+cn → 248); cos² and sin³/cos³ have one (→ 124).

---

### Tangent Addition (`tangent_addition`)

**n\_eff: 14,693**

| Variant | Slots | n\_eff |
|---|---|---|
| `tan_addition_plus` | a∈TRIG\_ARG idx=0.35, b∈TRIG\_ARG∖{a} idx=0.35 | 7,284 |
| `tan_addition_minus` | a∈TRIG\_ARG idx=0.35, b∈TRIG\_ARG∖{a} idx=0.35 | 7,284 |
| `double_angle_tan` | a∈TRIG\_ARG idx=0.35 | 124 |

---

### Negative Angle Identities (`negative_angle`)

**n\_eff: 10,449**

| Variant | Slots | n\_eff |
|---|---|---|
| `negative_angle_sin` | x∈VARS idx=0.35 | 55 |
| `negative_angle_cos` | x∈VARS idx=0.35 | 55 |
| `negative_angle_tan` | x∈VARS idx=0.35 | 55 |
| `negative_angle_sin_coeff` | a∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 3,428 |
| `negative_angle_cos_coeff` | a∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 3,428 |
| `negative_angle_tan_coeff` | a∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 3,428 |

---

### De Moivre's Theorem (`de_moivre`)

**n\_eff: 49,730**

| Variant | Slots | n\_eff |
|---|---|---|
| `de_moivre_bare` | a∈TRIG\_ARG idx=0.35, n∈`_GEO_N` (7) | 1,739 |
| `de_moivre_coeff` | a∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35, n∈`_GEO_N` (7) | 47,991 |

---

### Area of Triangle (`triangle_area`)

**n\_eff: 29,566**
`s0`∈SIDES idx=0.35, `s1`∈SIDES∖{s0} idx=0.35, `A`∈ANGLES idx=0.35. Two label variants: `\text{Area}=` and `S=`.

| Variant | n\_eff |
|---|---|
| `triangle_area_text` | 14,783 |
| `triangle_area_S` | 14,783 |

---

### Law of Sines Extended (`law_of_sines_extended`)

**n\_eff: 24,281,587** ← second dominant template
`s0,s1,s2`∈SIDES (ExcludeSlot chain, idx=0.35), `A0,A1,A2`∈ANGLES (ExcludeSlot chain, idx=0.35). Appends `= 2R`.

---

### Law of Sines 2R Ratio (`law_of_sines_2R_ratio`)

**n\_eff: 428**
`s0`∈SIDES idx=0.35, `A0`∈ANGLES idx=0.35. Single ratio form `a / sin A = 2R`. Low n\_eff → low weight; kept as standalone to avoid infecting any high-n\_eff group.

---

### Reciprocal Identities (`reciprocal_identities`)

**n\_eff: 13,932**

| Variant | Slots | n\_eff |
|---|---|---|
| `sec_reciprocal_bare` | x∈VARS idx=0.35 | 110 |
| `csc_reciprocal_bare` | x∈VARS idx=0.35 | 110 |
| `sec_reciprocal_coeff` | a∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 6,856 |
| `csc_reciprocal_coeff` | a∈TRIG\_ARG idx=0.35, x∈VARS idx=0.35 | 6,856 |

---

### Arctan Difference (`arctan_difference`)

**n\_eff: 14,569**
`a`∈TRIG\_ARG idx=0.35, `b`∈TRIG\_ARG∖{a} idx=0.35. Complements `arctan_addition` (variant inside `inverse_trig_identities`).

---

### Law of Tangents (`law_of_tangents`)

**n\_eff: 255,005**
`s0`∈SIDES idx=0.35, `s1`∈SIDES∖{s0} idx=0.35, `A0`∈ANGLES idx=0.35, `A1`∈ANGLES∖{A0} idx=0.35.
Formula: `(a−b)/(a+b) = tan((A−B)/2) / tan((A+B)/2)`. The third triangle law alongside law of sines and cosines.

---

### Fourier Series (`fourier_series`)

**n\_eff: 1,788,480**
`fn`∈`_fn_rich_nosub` (100 notation forms), `v`∈VARS idx=0.35, `ca`∈SCALARS (9), `cb`∈SCALARS∖{ca} (8), `L`∈SCALARS∖{ca,cb} (7).

| Variant | n\_eff | Description |
|---|---|---|
| `fourier_series_full` | 1,391,040 | `f(x) = a₀/2 + Σ(aₖ cos(kπx/L) + bₖ sin(kπx/L))` |
| `fourier_coeff_an` | 198,720 | `aₙ = (2/L) ∫₀ᴸ f(v) cos(nπv/L) dv` |
| `fourier_coeff_bn` | 198,720 | `bₙ = (2/L) ∫₀ᴸ f(v) sin(nπv/L) dv` |

---

### Weierstrass Substitution (`weierstrass_substitution`)

**n\_eff: 11,331**
`x`∈VARS idx=0.35, `t`∈VARS∖{x} idx=0.35 (eff≈28×24≈667 per variant for slots only).

| Variant | n\_eff | Description |
|---|---|---|
| `weierstrass_def` | 1,333 | `t = tan(x/2)` — the substitution |
| `weierstrass_sin` | 1,333 | `sin x = 2t / (1 + t²)` |
| `weierstrass_cos` | 1,333 | `cos x = (1 − t²) / (1 + t²)` |
| `weierstrass_tan` | 1,333 | `tan x = 2t / (1 − t²)` |
| `weierstrass_dx` | 667 | `dx = 2/(1+t²) dt` — differential form (no function-name selector) |
| `weierstrass_combined` | 5,332 | All three identities on one line (three name selectors → 2³ × 667) |

---

### Trig Antiderivatives (`trig_antiderivative`)

**n\_eff: 48,212**
9 variants: bare variants use x∈VARS idx=0.35; coeff variants add a∈TRIG\_ARG idx=0.35.

| Variant | n\_eff | Formula |
|---|---|---|
| `antideriv_sin_bare` | 110 | `∫sin x dx = −cos x + C` |
| `antideriv_cos_bare` | 110 | `∫cos x dx = sin x + C` |
| `antideriv_sin_coeff` | 6,856 | `∫sin(ax) dx = −cos(ax)/a + C` |
| `antideriv_cos_coeff` | 6,856 | `∫cos(ax) dx = sin(ax)/a + C` |
| `antideriv_tan_cos` | 6,856 | `∫tan(ax) dx = −ln\|cos(ax)\|/a + C` |
| `antideriv_tan_sec` | 6,856 | `∫tan(ax) dx = ln\|sec(ax)\|/a + C` |
| `antideriv_sec2_coeff` | 6,856 | `∫sec²(ax) dx = tan(ax)/a + C` |
| `antideriv_csc2_coeff` | 6,856 | `∫csc²(ax) dx = −cot(ax)/a + C` |
| `antideriv_sec_coeff` | 6,856 | `∫sec(ax) dx = ln\|sec(ax)+tan(ax)\|/a + C` |

---

## Key Design Decisions

- **`_TRIG_ARG_POOL`** (18 symbols): `_SCALARS` + θ φ ϕ ψ ω α β γ δ (`\phi` and `\varphi` are distinct).
  Used with `idx=0.35` for all angle/coefficient slots, multiplying effective pool to ≈62.

- **Function-name selector multiplier**: each `E(_sin_nm, n=2)` or similar slot contributes a factor
  of 2 to n\_eff (e.g. `\sin` vs `\text{sine}`). A template with two such selectors (sin + cos) gets
  4× the slot-product n\_eff vs the same template with no selectors. This explains why `product_to_sum_cos_cos` (one selector) has half the n\_eff of `product_to_sum_sin_cos` (two selectors).

- **Dual dominant templates**: `law_of_sines` and `law_of_sines_extended` each have 24.3M n\_eff
  and together account for ~82% of sampling weight. The combined 48.6M-combination space drives the
  birthday horizon to ~939.

- **Coeff variants for single-arg templates**: `triple_angle`, `power_reduction`, `hyperbolic_definitions`,
  `hyperbolic_double_angle`, `negative_angle`, `inverse_hyperbolic_log`, `reciprocal_identities`, and
  `trig_antiderivative` all add coeff variants (c·x form) that jump n\_eff by ~28×. Without these,
  bare variants would be drawn more frequently than their small pool warrants.

- **`_N_EFF_CAP = 75_000_000`**: both law\_of\_sines templates (24M each) are below the cap and
  use their full weight.

- **Variant n\_eff uniformity rule**: `rng.choice(variants)` is uniform — mixing a tiny-n\_eff variant
  with a large-n\_eff variant in the same group causes the large variant's weight to drive frequent
  draws from the tiny pool. `law_of_sines_2R_ratio` (n\_eff=428) is therefore a standalone template
  rather than a variant of `law_of_sines_extended` (n\_eff=24.3M). The threshold for concern is
  roughly: group weight >10% **and** n\_eff ratio >1,000× between variants.

- **Weierstrass substitution uses `_VARS × _VARS∖{x}`**: both `x` and `t` draw from VARS (8 items,
  eff≈28/24 with exclusion). The six variants sum to n\_eff≈11,331, giving modest weight appropriate
  for a compact identity group.

- **Antiderivative `+ C` constant**: the literal `C` is not a slot — Python's `.format()` only
  substitutes `{name}` patterns. Writing bare `C` is safe as long as no slot is named `C`.

- **Greek adjacency in LaTeX**: slots from `_TRIG_ARG_POOL` or `_COEFF_POOL` must not be written
  adjacent to alphanumeric characters without a space in the template string (e.g. `{a}{b}` when
  `a=\psi` produces `\psid` — an undefined LaTeX command). Always write `{a} {b}`.
