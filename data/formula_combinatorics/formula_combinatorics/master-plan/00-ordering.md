# formula-combinatorics: expansion plan and ordering

Master sequencing doc for the seven units of work that take the synthetic LaTeX generator from "good generator" (GutenOCR PR #24, branch `feat/domain-splits`) to "benchmark and training instrument" (the vision in the Brain agenda `synthetic-math-recognition-benchmark`).

**Repo:** `Roots-Automation/GutenOCR`, package at `data/formula_combinatorics/`.
**Authored:** 2026-06-12, from a direct read of PR #24. File-line references in the unit docs are against that branch.
**How to use:** each `0N-*.md` is a self-contained RFC. Hand any one to a work agent; it carries its own problem statement, current-state evidence, target state, acceptance criteria, and open questions. This doc holds the order and the dependency map.

## The units

| # | Unit | Effort | One-line scope |
|---|------|--------|----------------|
| 1 | [Truth-up & hygiene](01-truth-up.md) | S | Fix README/code drift, surface swallowed errors, weight variants, version bump. |
| 2 | [Render-in-the-loop validity gate](02-render-gate.md) | L | Compile every formula, keep-if-renders, emit images, per-template error rates. |
| 3 | [Calibration & coverage instrumentation](03-calibration-coverage.md) | M | n_eff vs empirical in CI; declare and assert global symbol coverage. |
| 4 | [Engine/content split + content-as-data](04-engine-content-split.md) | L | Separate engine from the 33 domains; move templates toward declarative data. |
| 5 | [Parameterization expansion](05-parameterization.md) | M | ✅ Difficulty, depth, length, symbol-tier, weight-override, split config knobs. |
| 6 | [Benchmark-native output + splits + semantic key](06-benchmark-output.md) | M-L | Rich per-sample records, held-out split manifests, semantic-equivalence key. |
| 7 | [Domain & coverage expansion](07-domain-expansion.md) | L | Physics sub-split, proof-theory layout, remaining notational gaps. |

Effort legend: S = hours, M = 1-2 days, L = 3-5 days, XL = week+. These are pre-fleshing estimates; each unit doc has a finer breakdown.

## Recommended order

```
1  truth-up ───────────────────────────────► (unblocks trust; no deps)
        │
2  render gate ──────────┬──────────────────► (keystone: validity + images + signal)
        │                │
3  calibration/coverage ─┘ (consumes 2's empirical signal)
        │
4  engine/content split ─────────────────────► (refactor once engine behavior is settled)
        │
5  parameterization ─────────► (easier on the data-driven base from 4)
        │
6  benchmark output ────────► (needs metadata from 2 + 4 + 5; the benchmark bridge)

7  domain expansion ──── runs in parallel throughout; "completes" after 3 (verify) + 4 (cheap authoring)
```

## Rationale and dependency notes

- **1 first** because it is cheap, removes the drift that would mislead a work agent (the README documents a `--align-fraction` parameter that no longer exists in the code), and has no dependencies.
- **2 is the keystone and goes early** because almost every quality claim downstream rests on validity, it produces the rendered images the OCR training pipeline needs anyway, and it generates the empirical render-success signal that unit 3 consumes. Its only input is the Phase-0 font-stack decision (a small spec choice, see the agenda's Phase 0).
- **3 follows or parallels 2.** It converts the two headline claims (analytic n_eff, balanced coverage) from aspiration into CI-enforced guarantees, using the render and collision signal.
- **4 is placed mid, not early.** Refactoring the engine/content boundary while 2 and 3 are still changing engine behavior would migrate the 29k lines of template content twice. Settle behavior first, then split.
- **5 after 4** because the new knobs (difficulty, depth, length, symbol-tier) are far cheaper to add on a data-driven template base than on Python literals.
- **6 last of the build units** because it is the generator-to-benchmark bridge and depends on metadata produced by 2 (render status), 4 (stable template ids as data), and 5 (stratification tags).
- **7 is continuous.** Content growth can proceed any time, but new strata are only verifiable once 3's coverage instrumentation exists, and only cheap to author once 4's data format lands, so it nominally trails.

## The one sequencing tradeoff to decide

**Unit 4 (the split) early vs mid.** Mid (the recommendation) avoids migrating template content twice. Early gives every later unit a cleaner base to build on, at the cost of refactoring against a still-moving engine. If the team would rather pay the churn for a clean foundation, swap 4 ahead of 2. Flag this for the work-side lead.

## What this plan does not cover

These units harden and generalize the generator and stand up the benchmark substrate. They stop at the generator boundary. The downstream research program (the C0-C5 training ladder, the held-out real anchor, CDM evaluation, the SynthDoG-Grounding Phase 2 retrain) lives in the Brain agenda `synthetic-math-recognition-benchmark` and is gated on this substrate, not the reverse.
