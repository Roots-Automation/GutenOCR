# WU3: Calibration & coverage instrumentation

**Status:** complete | **Effort:** M (1-2 days) | **Depends on:** WU2 (render signal, soft); collision_probe already exists | **Unblocks:** trustworthy weighting; the agenda's H3 (coverage mechanism)
**Repo:** `Roots-Automation/GutenOCR`, `data/formula_combinatorics/` (PR #24).

## Intent

Turn the two headline claims, "analytic combinatorics" and "balanced global symbol coverage", from intentions into CI-enforced, tested guarantees.

## Problem & motivation

The generator's value proposition rests on two properties that are currently asserted in prose but not enforced by tests:

1. **n_eff is only partly analytic.** It is exact for pure-`Slot` templates, but any template using a sub-generator (`Sub`/`ParamSub`/`ExcludeParamSub`) inherits a hand-set `n_eff_estimate` constant (`_template_dsl.py:209,229,256`; defaults like `1e4`, and the shared slots `_EXPR_SLOT=5000`, `_ATOM_SLOT=150`, `_FN_SLOT=100`, `_FN_RICH_SLOT=272` at `_template_dsl.py:539-542`). The sqrt(n_eff) weighting (`compute_weights`, `_template_dsl.py:469-474`) is therefore only as correct as these guesses, and nothing checks them.
2. **Coverage is claimed but never measured.** The "enforced global symbol coverage / balanced" property has no global symbol set declared and no test that the union of outputs covers it. The closest existing checks are a per-sampler "at least 50% of pool seen in 200 draws" floor (`tests/test_vocab_unit.py:297-301`) and per-domain keyword-presence tests. This directly undercuts the agenda's H3: you cannot ablate coverage-on vs coverage-off if coverage is not instrumented.

## Current state (evidence)

- `n_eff()` analytic for slots, estimate for subs (`_template_dsl.py:371-412`). Tests only assert `n_eff > 0` (`tests/test_all_domains.py:71-79`) and slot-level correctness on toy pools (`tests/test_dsl_unit.py:35+`).
- `tools/collision_probe.py` measures the empirical birthday-horizon per domain (first-collision sample count) and can print analytic n_eff (`--n-eff`), but it is a standalone dev script, never wired into CI as an assertion.
- No global symbol-set declaration anywhere; coverage tests are per-pool floors only.

## Target state

- **n_eff calibration in CI.** A test that runs the collision probe (bounded sample budget) and asserts the empirical horizon is within a factor of the analytic n_eff per domain; auto-flag any `n_eff_estimate` that is off by more than ~10x so the sub-generator estimates get tuned to reality.
- **Declared global symbol set + coverage assertion.** A canonical inventory of the math symbols/commands the corpus claims to cover, and a CI test that a fixed-budget corpus draw exercises every entry (or every entry at the appropriate stratum). The "balanced" sampler gets a verifiable definition.
- **Coverage as a measurable knob.** Expose coverage as something the corpus can report and (for WU5) target, so H3's coverage-on/off ablation is a real configuration, not a hand-wave.

## Design sketch

- Promote `collision_probe.py` logic into a tested module; add a `calibration` test that asserts `empirical_horizon ~ analytic_n_eff` within bounds and emits a tuning report for off-estimates.
- Declare the global symbol set as data (a list/JSON of commands and symbols, ideally derived from the union of domain pools plus a curated "must-cover" set). Add a coverage report: given a corpus, which declared symbols appeared, at what frequency, in which strata.
- A coverage test: a corpus of size N (fixed seed) covers 100% of the "must-cover" set and at least a floor of the long tail.
- If WU2 has landed, fold render-success rate into the same per-template report so calibration, coverage, and validity are one dashboard.

## Acceptance criteria

- CI fails if any domain's empirical diversity diverges from analytic n_eff beyond tolerance.
- CI fails if a fixed-seed corpus does not cover the declared must-cover symbol set.
- A tuning report lists sub-generator `n_eff_estimate` values that need adjustment, with the empirical target.
- Coverage is reportable per corpus (symbol -> count, by stratum).

## Dependencies & ordering

- **Inbound (soft):** WU2's render signal makes the dashboard complete but is not strictly required; collision-probe and n_eff already exist, so WU3 can begin in parallel with WU2.
- **Outbound:** WU5's coverage-targeting knob and WU6's stratified records both rely on the declared symbol set landing here.

## Risks & open questions (for fleshing)

- Defining "the global symbol set" is a judgment call: union of all pools (descriptive) vs a curated must-cover list grounded in real math-OCR symbol frequency (prescriptive). The agenda argues for the prescriptive, frequency-tiered version (head/body/tail), which also seeds WU5's symbol-tier knob.
- Calibration tolerance and sample budget: tight enough to catch real miscalibration, loose enough to avoid flaky CI. The birthday-horizon is stochastic; use multiple seeds.
- Sub-generator n_eff is genuinely hard to make exact (recursive `_expr` can be unbounded); decide whether to keep calibrated estimates or compute true analytic bounds for the recursive generators.
