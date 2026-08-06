# WU7: Domain & coverage expansion

**Status:** complete (2026-06-17) | **Effort:** L (3-5 days, ongoing) | **Depends on:** WU3 (verify new strata), WU4 (cheap authoring) | **Unblocks:** the agenda's coverage-completeness claim
**Repo:** `Roots-Automation/GutenOCR`, `data/formula_combinatorics/` (PR #24).

## Completion summary (2026-06-17)

Delivered on branch `feat/domain-splits`, commits `abafe8b` through `ad5f153`.

**Delivered:**
- TOML migration: all 40 domains migrated; zero Python domain files remain
- Physics sub-split: `classical_mechanics`, `electromagnetism`, `statistical_mechanics`, `quantum_mechanics`, `field_theory` as separate registered domains; old `physics.py` (1075L) deleted
- New domains: `proof_theory`, `asymptotics`, `topology`, `ring_field_theory`, `measure_theory`, `representation_theory`, `stochastic_processes`, `p_adic` and others
- Per-sample metadata carries sub-domain name directly (physics sub-strata addressable via domain field)

**Gap table:**
- Physics sub-regimes: ✅ Addressed via separate domains
- Proof-theory inference-rule layout: ✅ `proof_theory.toml` domain with `\dfrac`/`\vdash` patterns; structural test added
- Asymptotics / numerical analysis: ✅ `asymptotics.toml` domain
- Order / lattice theory: ✗ Consciously deferred — insufficient distinct notation to earn a stratum under the design principle
- Game theory / mathematical finance: ✗ Consciously deferred — notation reuse; no new stratum

**Deferred (separate installable packages):** Two-`pyproject.toml` split (formula-engine / formula-content) remains unstarted. Tracked as a standalone future item.

## Intent

Close the remaining notational-regime gaps so the corpus's coverage claim is genuinely complete, and finish the partial splits. Content growth, guided by the "strata are notational regimes, not academic fields" principle.

## Problem & motivation

PR #24 already closed most of the agenda's original gap list (quantum bra-ket, statistics, chemistry are now first-class domains; tensor/index is largely covered in `differential_geometry`). What remains are a partial split and a short tail of genuinely-distinct notational regimes the corpus does not yet exercise. A coverage benchmark's whole value is breadth, so the tail matters more here than in a training-only generator.

## Current state (evidence)

- 33 domains registered (`domains/__init__.py:12-46`), tagged foundational/advanced/applied/structural in `_config.py:41-80`.
- **Physics is still a mega-regime, only partially split.** Quantum bra-ket was peeled into `quantum_notation` (`domains/quantum_notation.py`, 291 lines), but `physics.py` (1075 lines) still conflates Lagrangian/Hamiltonian mechanics, vector-calculus electromagnetism, tensor relativity, and statistical mechanics.
- **Proof-theory inference-rule layout is light.** `logic.py` (1441 lines) covers turnstiles and sequents well (`\vdash` is frequent), but the premises-over-conclusion 2D inference-rule layout (a distinct fraction-like structure) is sparse (~a handful of templates).
- **Open tail from the agenda's gap table:** asymptotics / numerical analysis (Landau notation, error bounds), order / lattice theory, game theory / mathematical finance. All "mostly notation reuse" or "partly covered" today, none with a dedicated stratum.

## Target state

- `physics` split into labeled sub-regimes (or sampled with labeled sub-strata) so per-stratum diagnosis is possible, consistent with the design principle.
- Proof-theory / type-theory inference-rule layout developed into a real stratum (premises-over-conclusion, typing judgments, sequent calculus rule shapes), since it is a distinct 2D structure and a known OCR hard case.
- The open-tail regimes added where they introduce genuinely distinct notation (asymptotics is the strongest candidate; order/lattice and game theory are lower-value, mostly notation reuse, add only if breadth-for-breadth is wanted).
- Every new stratum verified by WU3's coverage instrumentation and authored via WU4's data packs.

## Design sketch

- Treat each addition as: define the distinct symbol set and 2D structures the stratum introduces (the inclusion test from the design principle), author templates, set weight/cap/tag/difficulty in config, add structural + frequency-threshold tests in the per-domain test pattern (`tests/test_domain_structural.py`, `tests/test_domain_content.py`).
- For the physics split, decide split-into-domains vs labeled-sub-strata-within-physics; the latter is less disruptive and still gives per-stratum diagnosis if the sub-stratum label rides in the per-sample record (WU6).
- Inference-rule layout likely needs a small DSL extension or shared template helper for the premises-over-conclusion structure (a `\frac`-like or `prooftree`/`mathpartir` construct), which the render gate (WU2) must support in the font/package set.

## Deferred from WU4 (also tracked here)

Two items were explicitly deferred from WU4 because WU7 is their natural completion point:

**Monster domain migration to TOML.** WU4 migrated only the three smallest domains (math_fonts, quantum_notation, linear_algebra) to prove the TOML schema. The remaining 30 Python domain files — including the monoliths (algebra 1879L, trigonometry 1534L, logic 1441L, differential_geometry 1362L, physics 1075L) — still live as Python `Template` literals. Migrate all remaining domains to `.toml` packs as part of WU7's content work. Every new domain authored in WU7 should be TOML-first; backfilling the existing ones can proceed in parallel. A code-to-data AST extractor may be worth building for the monoliths rather than hand-porting.

**Separate installable packages.** WU4's target state specifies `formula-engine` and `formula-content` as separately installable, versioned units. WU4 delivered a logical in-repo split (the `engine/` subdirectory) but deferred the `pyproject.toml` split. Split into two packages as part of WU7 once enough domains are TOML packs — at that point content truly has no engine code and can live in a separate distribution, enabling external contribution and per-pack versioning without a full package release.

## Acceptance criteria

- Physics sub-regimes are individually addressable (as domains or labeled sub-strata) and appear in per-sample metadata.
- A proof-theory inference-rule stratum exists with structural tests asserting the premises-over-conclusion layout.
- Any added regime passes WU3 coverage and WU2 render certification.
- The agenda's gap table can be marked closed or consciously deferred per regime, with rationale.

## Dependencies & ordering

- **Inbound:** WU3 (so new strata are coverage-verified), WU4 (so authoring is cheap and the new content is data, not a new 1k-line Python file). Can proceed in parallel throughout but "completes" after those.
- **Outbound:** the agenda's coverage-completeness and the per-stratum diagnosis story.

## Risks & open questions (for fleshing)

- The inclusion bar: order/lattice and game-theory may not introduce enough distinct notation to earn a stratum under the design principle. Decide per regime; do not pad for count.
- Inference-rule layout requires LaTeX packages (e.g. `mathpartir`, `bussproofs`) that must be in the render gate's allowed set and ideally OFL-clean; coordinate with WU2's Phase-0 font/package decision.
- Physics split vs sub-strata is a modeling choice with downstream effects on weights and benchmark stratification; settle before authoring.
- This unit is genuinely open-ended; cap it to the agenda-relevant regimes rather than pursuing all of mathematics.
