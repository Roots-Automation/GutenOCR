# Algebra Domain — Template Reference

Templates for the `algebra` domain generator. All templates use the
[Template DSL](../_template_dsl.py): named `Slot` / `ExcludeSlot` / `Sub` /
`ParamSub` declarations with analytical `n_eff` computation and
`sqrt(min(n_eff, CAP))` sampling weights (`CAP = 75_000_000`).

**Summary:** 39 templates · total n_eff ≈ 6.25 × 10¹⁴ · median birthday
horizon ≈ 2,149 · batch uniqueness 99.9% at 1,000 samples ·
rejection rate ~5.8% at 1,000,000 unique samples.

---

## Quadratic Family

Eight templates sharing the same slot set (`v0`, `p`, `q`, `r` drawn
distinctly from `_UNION` with `idx=0.35`).

| Template | n_eff |
|---|---|
| `quadratic_formula_pm` | 10,923,024 |
| `quadratic_formula_pos` | 10,923,024 |
| `quadratic_formula_neg` | 10,923,024 |
| `discriminant_def` | 10,923,024 |
| `discriminant_condition` | 32,769,072 |
| `quadratic_factored` | 10,923,024 |
| `quadratic_completed_square` | 10,923,024 |
| `quadratic_monic` | 10,923,024 |

**Sample:** `d_{2} = \frac{-t \pm \sqrt{t^2 - 4 a k}}{2a}`

---

## Polynomial Forms

| Template | n_eff | Variants | Variant n_eff |
|---|---|---|---|
| `expanded_polynomial` | 50,534,397 | `expanded_polynomial_deg3` | 555,323 |
| | | `expanded_polynomial_deg4` | 4,997,907 |
| | | `expanded_polynomial_deg5` | 44,981,166 |
| `rational_fraction` | 6,900,000 | — | — |
| `polynomial_nth_root` | 1,242,000 | — | — |

**Sample:** `a_{0} t^{5} + n t^{4} + k t^{3} + b t^{2} + d t + n`

---

## Factoring Identities

| Template | n_eff | Variants | Variant n_eff |
|---|---|---|---|
| `difference_of_squares` | 138,000 | — | — |
| `sum_of_cubes` | 25,000,000 | — | — |
| `difference_of_cubes` | 25,000,000 | — | — |
| `perfect_square` | 50,000,000 | `perfect_square_plus` | 25,000,000 |
| | | `perfect_square_minus` | 25,000,000 |
| `general_factored` | 216,360,617 | `general_factored_2roots` | 571,211 |
| | | `general_factored_3roots` | 11,824,077 |
| | | `general_factored_4roots` | 203,965,328 |
| `sophie_germain` | 10,266 | `sophie_germain_standard` | 3,422 |
| | | `sophie_germain_sum_of_squares` | 3,422 |
| | | `sophie_germain_lhs_only` | 3,422 |
| `nth_power_factoring` | 88,972 | `difference_nth_powers_ellipsis2` | 23,954 |
| | | `difference_nth_powers_ellipsis3` | 23,954 |
| | | `sum_odd_powers_ellipsis2` | 20,532 |
| | | `sum_odd_powers_ellipsis3` | 20,532 |

**Sample (sophie_germain):** `\kappa^4 + 4\gamma^4 = \left((\kappa+\gamma)^2 + \gamma^2\right)\left((\kappa-\gamma)^2 + \gamma^2\right)`

**Sample (nth_power):** `\mu^{2p+1} + b^{2p+1} = (\mu+b)\left(\mu^{2p} - \mu^{2p-1}b + \cdots + b^{2p}\right)`

---

## Polynomial Theorems

| Template | n_eff | Variants | Variant n_eff |
|---|---|---|---|
| `remainder_factor_theorem` | 51,336,000 | `remainder_theorem` | 13,800,000 |
| | | `factor_theorem` | 37,536,000 |
| `polynomial_division` | 5,636,187,400 | `polynomial_division_abstract` | 2,760,000,000 |
| | | `polynomial_division_remainder_form` | 2,760,000,000 |
| | | `polynomial_division_linear_divisor` | 16,187,400 |
| | | `polynomial_division_uniqueness` | 100,000,000 |
| `polynomial_root_form` | 31,251,408,240 | `polynomial_root_form_general` | 30,147,546,240 |
| | | `polynomial_root_form_monic` | 538,349,040 |
| | | `polynomial_root_form_quadratic` | 538,349,040 |
| | | `polynomial_root_form_subscript` | 27,163,920 |

**Sample (division):** `\tilde{Q}(v) = \tilde{\chi}(v) \cdot \tilde{H}(v) + \tilde{F}(v), \quad \deg \tilde{F} < \deg \tilde{H}`

**Sample (root form):** `\bar{P}(z) = \lambda_{n}(z - \alpha)(z - q)`

---

## Binomial Theorem

| Template | n_eff |
|---|---|
| `binomial_theorem` | 250,000,000 |

`u`, `w` from `_expr` (n=5,000); `n` from `_exp_sym_sub` (10 options).

**Sample:** `\left(u + w\right)^{n} = \sum_{k=0}^{n} \binom{n}{k} \left(u\right)^k \left(w\right)^{n-k}`

---

## Logarithm Identities

| Template | n_eff | Variants | Variant n_eff |
|---|---|---|---|
| `log_identities` | 301,522,270 | `log_quotient_rule` | 82,270 |
| | | `log_power_rule` | 780,000 |
| | | `log_change_of_base` | 660,000 |
| | | `log_product_rule` | 300,000,000 |

`base` from `_LOG_BASES` (12 options); arguments from `_expr` (n=5,000).

**Sample:** `\log_{d}\!\left(\frac{c u_{k}}{m}\right) = \log_{d} c + \log_{d} u_{k} - \log_{d} m`

---

## Epsilon-Delta

| Template | n_eff | Variants | Variant n_eff |
|---|---|---|---|
| `epsilon_delta` | 93,733,221 | `epsilon_delta_nearness` | 16,560 |
| | | `epsilon_delta_full` | 93,239,424 |
| | | `epsilon_delta_fn_nearness` | 466,197 |
| | | `epsilon_delta_interval` | 11,040 |

Variable from `_VARS` with `idx=0.35`; functions from `_fn_rich` (n=272).

**Sample:** `0 < \left|v - p\right| < \delta \implies \left|\hat{f}(v) - m\right| < \varepsilon`

---

## Completing the Square

| Template | n_eff | Variants | Variant n_eff |
|---|---|---|---|
| `completing_the_square` | 272,898 | `completing_square_monic` | 171,396 |
| | | `completing_square_nonmonic` | 47,991 |
| | | `completing_square_partial` | 5,520 |
| | | `completing_square_vertex` | 47,991 |

**Sample:** `z^2 + k z + u_{n} = \left(z + \frac{k}{2}\right)^2 + u_{n} - \frac{k^2}{4}`

---

## Exponential Growth

| Template | n_eff | Variants | Variant n_eff |
|---|---|---|---|
| `exponential_growth` | 686,869 | `exponential_growth_pos` | 68,001 |
| | | `exponential_decay` | 68,001 |
| | | `exponential_general_base` | 6,856 |
| | | `logistic_growth` | 544,011 |

Variable `v`, scalar `a` from `_VARS`/`_SCALARS`; growth rate `g` from `_GREEK` with `idx=0.35`.

**Sample:** `s(t) = \frac{p}{1 + d e^{-\varepsilon_{m} t}}`

---

## Miscellaneous Algebra

| Template | n_eff |
|---|---|
| `floor_fraction` | 750,000 |
| `product_formula` | 231,840 |
| `partial_fraction` | 555,323 |
| `proportion_identity` | 625,000,000,000,000 |
| `sum_of_squares` | 3,750,000,000 |

**Notes:**
- `floor_fraction`: `⌊num/den⌋`, `num` from `_expr`, `den` from `_atom`
- `product_formula`: `∏_{k=start}^{n} (1 + a/(k+v))`
- `partial_fraction`: `a / v(v−b) = s₁/v + s₂/(v−b)`
- `proportion_identity`: `e₁/e₂ = e₃/e₄`, all four slots from `_expr` (n=5,000)
- `sum_of_squares`: `(e₁)² + (e₂)² = at²`

---

## Vieta's Formulas

| Template | n_eff | Variants | Variant n_eff |
|---|---|---|---|
| `vieta_formulas` | 97,373 | `vieta_quadratic` | 13,910 |
| | | `vieta_cubic` | 83,462 |

Root variable from `_VARS`; coefficients from `_SCALARS` via exclusive chain.

**Sample:** `z_1 + z_2 + z_3 = -\frac{d}{p}, \quad z_1 z_2 + z_1 z_3 + z_2 z_3 = \frac{k}{p}, \quad z_1 z_2 z_3 = -\frac{q}{p}`

---

## Inequalities

### AM-GM and Mean Inequalities

| Template | n_eff | Variants | Variant n_eff |
|---|---|---|---|
| `am_gm` | 300,124 | `am_gm_two` | 3,422 |
| | | `am_gm_three` | 195,054 |
| | | `am_gm_power_mean` | 94,447 |
| | | `am_gm_nvar` | 119 |
| | | `hm_gm_two` | 3,422 |
| | | `hm_gm_am_chain` | 3,422 |
| | | `hm_def` | 119 |
| | | `am_gm_hm_inequality` | 119 |

Slots from `_COEFF_POOL` (9 scalars + 8 Greek), distinct pairs.

**Sample:** `\frac{2\mu\,\kappa}{\mu+\kappa} \leq \sqrt{\mu\,\kappa} \leq \frac{\mu+\kappa}{2}`

### Cauchy-Schwarz

| Template | n_eff | Variants | Variant n_eff |
|---|---|---|---|
| `cauchy_schwarz` | 45,944,074,761 | `cauchy_schwarz_sum` | 3,024 |
| | | `cauchy_schwarz_integral` | 45,944,064,000 |
| | | `cauchy_schwarz_vector` | 7,737 |

Sum: index from `_cs_idx_sub` (6), upper bound from `_cs_ub_sub` (7), coefficients from `_SCALARS`.
Integral: variable from `_VARS`, bounds from `_atom`, functions from `_fn_rich` (n=272).
Vector: bold vectors from `_VEC_POOL` (26 letters) with `idx=0.35`.

**Sample:** `\left(\int_{a}^{b} \hat{f}(x) \, g(x) \, dx\right)^2 \leq \int_{a}^{b} \hat{f}(x)^2 \, dx \cdot \int_{a}^{b} g(x)^2 \, dx`

### Triangle Inequality

| Template | n_eff | Variants | Variant n_eff |
|---|---|---|---|
| `triangle_inequality` | 18,122 | `triangle_inequality_basic` | 3,422 |
| | | `triangle_inequality_difference` | 3,422 |
| | | `triangle_inequality_reverse` | 3,422 |
| | | `triangle_inequality_norm` | 7,737 |
| | | `triangle_inequality_nterms` | 119 |

Slots from `_COEFF_POOL` with `idx=0.35` (basic/difference/reverse);
`_VEC_POOL` with `idx=0.35` (norm); `_COEFF_POOL` + `_GEO_N` (n-terms).

**Sample:** `\left|c_{j} + b_{i}\right| \leq \left|c_{j}\right| + \left|b_{i}\right|`

---

## Series and Sums

### Power Sums

| Template | n_eff | Variants | Variant n_eff |
|---|---|---|---|
| `power_sums` | 96,558 | `sum_of_integers` | 7 |
| | | `sum_of_integers_ellipsis` | 21 |
| | | → `sum_of_integers_ellipsis_2` | 7 |
| | | → `sum_of_integers_ellipsis_3` | 7 |
| | | → `sum_of_integers_ellipsis_4` | 7 |
| | | `sum_of_squares` | 7 |
| | | `sum_of_cubes` | 7 |
| | | `arithmetic_progression_sum` | 23,954 |
| | | `arithmetic_progression_ellipsis` | 71,862 |
| | | → `arithmetic_progression_ellipsis_2` | 23,954 |
| | | → `arithmetic_progression_ellipsis_3` | 23,954 |
| | | → `arithmetic_progression_ellipsis_4` | 23,954 |
| | | `telescoping_sum` | 700 |

`n` from `_GEO_N` (7 options); `a`/`d` from `_COEFF_POOL` with `idx=0.35` (distinct pair); function `f` from `_fn_rich_nosub` (n=100) for telescoping.

**Sample:** `\sum_{k=1}^{K} k^2 = \frac{K(K+1)(2K+1)}{6}`

**Sample:** `\alpha_{j} + (\alpha_{j}+\rho) + (\alpha_{j}+2\rho) + (\alpha_{j}+3\rho) + \cdots + (\alpha_{j}+N\rho) = \frac{(N+1)(2\alpha_{j}+N\rho)}{2}`

### Geometric Series

| Template | n_eff | Variants | Variant n_eff |
|---|---|---|---|
| `geometric_series` | 51,741 | `geometric_series_finite` | 23,954 |
| | | `geometric_series_finite_unit` | 411 |
| | | `geometric_series_infinite` | 3,422 |
| | | `geometric_series_closed_form` | 23,954 |

`a`/`r` drawn distinctly from `_COEFF_POOL` with `idx=0.35`; `n` from `_GEO_N`.

**Sample:** `\sum_{k=0}^{K} c\, p^k = c \, \frac{1 - p^{K+1}}{1 - p}`

---

## Conjugate and Radical Forms

| Template | n_eff | Variants | Variant n_eff |
|---|---|---|---|
| `conjugate_pairs` | 94,796 | `conjugate_radical_basic` | 1,406 |
| | | `conjugate_radical_both` | 3,422 |
| | | `conjugate_radical_coeff` | 85,140 |
| | | `conjugate_complex` | 3,422 |
| | | `conjugate_rationalise` | 1,406 |

Slots from `_COEFF_POOL` with distinct groups; radical argument without `idx` (subscripts look odd under `√`).

**Sample:** `(\alpha + \mu\sqrt{\theta})(\alpha - \mu\sqrt{\theta}) = \alpha^2 - \mu^2 \theta`

---

## Exponent Rules

| Template | n_eff | Variants | Variant n_eff |
|---|---|---|---|
| `rational_exponent_rules` | 9,523 | `rational_exp_radical` | 1,159 |
| | | `negative_exp` | 193 |
| | | `product_of_powers` | 1,159 |
| | | `power_of_power` | 1,159 |
| | | `power_of_product` | 4,666 |
| | | `quotient_of_powers` | 1,159 |
| | | `zero_exponent` | 28 |

`x`/`y` from `_VARS` with `idx=0.35`; exponents from `_EXP_POOL` = `{2,3,4,m,n,p,q}` (exclusive pairs where applicable).

**Sample:** `\left(z^{2}\right)^{4} = z^{2 \cdot 4}`

---

## Slot Pools Reference

| Pool | Contents | Size |
|---|---|---|
| `_VARS` | `x y z w v u t s` | 8 |
| `_SCALARS` | `a b c d k m n p q` | 9 |
| `_UNION` | `_VARS ∪ _SCALARS` | 17 |
| `_GREEK` | α β γ δ ε ζ η θ κ λ μ ν ξ ρ σ τ φ χ ψ ω ε φ ϑ | 23 |
| `_COEFF_POOL` | `_SCALARS` + α β γ λ μ ρ κ θ | 17 |
| `_GEO_N` | `n m N M K p r` | 7 |
| `_EXP_POOL` | `2 3 4 m n p q` | 7 |
| `_VEC_POOL` | a–z (all 26 lowercase) | 26 |
| `_LOG_BASES` | `2 10 e` + `_SCALARS` | 12 |

### Index decoration (`idx`)

Slots with `idx=0.35` apply `_maybe_idx` with probability 0.35, appending
a subscript from `{0,1,2,i,j,k,n,m}`. This multiplies the effective pool
size by `1 + 0.35 × 7 = 3.45`.

### Sub-generators

| Generator | Approx. options | Used in |
|---|---|---|
| `_fn_rich` | ~272 | epsilon-delta, Cauchy-Schwarz integral |
| `_fn_rich_nosub` | ~100 | remainder theorem, polynomial division, root form, telescoping |
| `_expr` | ~5,000 | difference of squares, logs, binomial, proportion |
| `_idx_atom` | ~200 | product formula |
| `_poly(rng, v)` | ~5,000 | expanded polynomial, nth root |

---

## Diversity Statistics

| Metric | Value |
|---|---|
| Templates | 39 |
| Total n_eff | ≈ 6.25 × 10¹⁴ |
| Sampling cap (`_N_EFF_CAP`) | 75,000,000 |
| Median birthday horizon (50 trials, max 5k) | 2,149 |
| Min birthday horizon | 245 |
| Max birthday horizon | 4,114 |
| No-collision trials / 50 | 0 |
| Batch uniqueness at 1,000 samples | 99.9% |
| Rejection rate at 1,000,000 unique samples | ~5.8% |
| Time to generate 1M unique samples | ~11.4s |
