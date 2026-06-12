# WU6: Benchmark-native output + held-out splits + semantic key

**Status:** proposed | **Effort:** M-L (3-5 days) | **Depends on:** WU2 (render status), WU4 (stable template ids), WU5 (stratification features) | **Unblocks:** the agenda's entire benchmark layer (held-out strata, CDM eval, the C0-C5 ladder)
**Repo:** `Roots-Automation/GutenOCR`, `data/formula_combinatorics/` (PR #24).

## Intent

Make the generator emit a benchmark, not just a training corpus: every sample carries full provenance, the corpus can be partitioned into principled held-out splits, and each sample exposes a semantic key for equivalence evaluation. This is the bridge from PR #24 (generator) to the agenda's frontier (benchmark instrument).

## Problem & motivation

The agenda's benchmark needs three things the generator cannot currently produce:

1. **Per-sample provenance** to build splits and analyze results by stratum. Output is domain-only.
2. **Held-out splits** (hold out 6 of 33 domains, a fraction of templates within trained domains, ~15% of symbols) to measure cross-regime / compositional / rare-symbol generalization. There is no split machinery.
3. **A semantic key** to compute the semantic-equivalence metric the agenda calls an "unoccupied square", without parsing output LaTeX (which has the many-to-one pathology).

## Current state (evidence)

- Output is `{"0": "<latex>", ...}` or, with `--metadata`, `{"formula": ..., "domain": ...}` (`corpus.py:104-107`). No template id, difficulty, structural features, symbol set, n_eff, or render status per sample.
- Dedup is exact-string (`corpus.py:101-102`); uniqueness is string-level, not semantic.
- No notion of train vs held-out; the generator emits one undifferentiated dict.
- The DSL already holds the structured ingredients of a semantic key: `sample()` computes a `draws` dict of slot-name to value (`_template_dsl.py:321-360`) for a named `Template`. That `(template id, draws)` pair is the semantic identity, currently discarded.

## Target state

- **Rich per-sample records** in a benchmark-grade format (JSONL or parquet): formula, domain, template id, difficulty, structural features (depth, flags, length), symbol set used, n_eff of the source template, render status + image path (from WU2), and the semantic key.
- **Held-out split config and manifests:** request held-out domains, templates, and symbol fractions by name or proportion; the generator emits disjoint `train` / `held_out_*` partitions plus a manifest recording exactly what was held out and the seed, so splits are reproducible and contamination-free.
- **Semantic key per sample:** the `(template id, slot draws)` tuple (and/or a canonicalized AST) exposed as a stable identifier, enabling a semantic-equivalence metric and semantic-level dedup as an option alongside string dedup.
- Backward-compatible: the GutenOCR-style `{"i": "<latex>"}` output (`generate.py:3-4`, consumed by `data/grounded_latex/generate_equations.py`) remains available as a projection of the rich records.

## Design sketch

- Thread template identity through `sample()`: return or attach `(template.name, draws)` rather than discarding it; the dispatcher (`make_dispatcher`, `_template_dsl.py:477-487`) and `generate()` propagate it into the record.
- Define a `Sample` record schema; add an output writer for JSONL/parquet plus the legacy dict projection.
- Split engine: given hold-out specs, partition the *template/domain/symbol space first*, then generate each partition independently with disjoint constraints, guaranteeing no leakage (rather than generating one corpus and splitting after, which risks near-duplicates across splits).
- Semantic key: start with `(template id, normalized draws)`; optionally add a canonical AST if WU4's declarative templates make one cheap.

## Acceptance criteria

- A run emits JSONL/parquet records with the full field set; the legacy dict projection is byte-identical to today's output for the same seed/config.
- A held-out run produces disjoint partitions with a manifest; a test asserts zero formula overlap and zero held-out template/domain/symbol leakage into train.
- The semantic key is stable across runs (same template+draws -> same key) and supports semantic dedup as an option.
- Results can be sliced by every stratum (domain, difficulty, depth, length, symbol-tier) from the records alone.

## Dependencies & ordering

- **Inbound:** WU2 (render status/image path), WU4 (stable template ids from versioned packs), WU5 (structural features and symbol tiers). Last of the build units for that reason.
- **Outbound:** the agenda's Phase 1 (the C0-C5 ladder, held-out evaluation, CDM, the semantic metric) consumes these records and splits directly.

## Deferred from WU5 (must be completed here)

**`--hold-out-templates` and `--hold-out-symbols` CLI flags.** WU5 implemented `--hold-out-domains` (domain-level hold-out, excludes entire domains from generation). The template-level and symbol-level hold-out knobs — `--hold-out-templates` (hold out a named list or fraction of templates within active domains) and `--hold-out-symbols` (hold out a named list or fraction of symbols so they don't appear in the training partition) — were explicitly deferred to this unit. Both must be implemented here as part of the split engine: they are the template-level and symbol-level analogs of `--hold-out-domains`, and the split manifests WU6 emits must record exactly which templates and symbols were held out (with the seed) for reproducibility and contamination verification. The `filter_templates()` helper and `Template.strata` / pool introspection infrastructure from WU5 provide the O(1) pre-filtering primitives needed.

**Frequency-tiered symbol set (head / body / tail) and `--symbol-tier` extension.** WU3 is complete and delivered type-based `SYMBOL_STRATA` (greek, calligraphic, blackboard_bold, functions, bold_vectors, bold_greek) but not a prescriptive frequency-ordered partition. WU5's `--symbol-tier` knob uses the type-based strata as a substitute. The frequency-tiered partition — head (high-frequency in real math-OCR), body (mid), tail (rare), grounded in real or principled-proxy symbol frequency data — must be defined here, because WU6's rich per-sample records are supposed to carry a symbol-tier label per sample for the benchmark's stratification axes. Once defined: (a) add the tier labels to the per-sample record schema, and (b) extend `--symbol-tier` in `generate.py` / `corpus.py` to accept `head`/`body`/`tail` as additional choices alongside the existing SYMBOL_STRATA names.

**`--coverage-mode` CLI flag.** WU3 delivered the coverage measurement infrastructure (`engine/_coverage.py`: `CoverageReport`, `measure_coverage()`, `measure_coverage_by_domain()`) and `MUST_COVER` / `SYMBOL_STRATA`. WU5's target state included a `--coverage-mode` knob (keep sampling until every declared symbol / template-class is hit N times — the "balanced" sampler made explicit) but it was not implemented. It belongs here because balanced coverage is a benchmark-corpus quality concern: a benchmark needs every must-cover symbol to appear with adequate frequency, not just once. Implement as `coverage_mode: bool` / `coverage_n: int` parameters to `corpus.generate()` with a `--coverage-mode [N]` CLI flag, building on the existing `measure_coverage()` infrastructure.

## Deferred from WU4 (also tracked here)

**Full pack provenance in per-sample records.** WU4 embeds `content_pack_hash` (SHA-256 of the TOML file) in per-sample metadata when `include_metadata=True`. The remaining `PackMeta` fields — `name`, `version`, `author`, `license`, `description` — are loaded into memory but not yet surfaced in per-sample records. Include them in the rich per-sample record schema defined here (e.g. as a nested `content_pack` object or flattened `content_pack_version`, `content_pack_author` fields), so a corpus snapshot carries full license-traceable provenance per sample without needing to re-load the pack file.

## Risks & open questions (for fleshing)

- Split granularity for symbols: "hold out 15% of the symbol set" needs a precise definition of "uses a held-out symbol" at the template level, and templates often mix symbols; decide whether a single held-out symbol disqualifies a template from train.
- Semantic key vs full AST: the `(template id, draws)` key is cheap and exact but tied to template identity; a canonical AST is more general but costs a parser. The agenda only needs equivalence, so the key likely suffices; confirm against the metric design.
- Format choice (JSONL vs parquet) and how image artifacts are referenced (inline path vs content-addressed store) at 1M scale.
- Whether semantic dedup should be the default for benchmark builds (visual diversity wants string dedup; a clean benchmark may want semantic dedup to avoid trivially-equivalent items across splits).
