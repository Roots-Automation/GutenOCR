# WU4: Engine/content split + content-as-data

**Status:** complete (2026-06-12) | **Effort:** L (3-5 days) | **Depends on:** WU2 + WU3 (engine behavior settled) | **Unblocks:** cheaper WU5, WU6, WU7
**Repo:** `Roots-Automation/GutenOCR`, `data/formula_combinatorics/` (PR #24).

## Completion summary (2026-06-12)

Delivered on branch `feat/domain-splits`, commit `08f2f7b`.

**Delivered:**
- `formula_combinatorics/engine/` — `_template_dsl`, `_templates`, `_vocab`, `render`, `_coverage`, `symbol_inventory` moved here; `engine/__init__.py` re-exports full public API unchanged.
- All 33 domain `.py` files updated to `..engine.*` imports.
- `engine/_pack_loader.py` — `load_pack(path) → PackResult(templates, meta)` with SHA-256 provenance hash.
- `engine/_sub_registry.py` — 30+ named callables (`_atom`, `_expr`, matrix variants, smallmatrix variants).
- Three small domains migrated to TOML and byte-identical verified: `math_fonts.toml` (23 templates), `quantum_notation.toml` (29 templates), `linear_algebra.toml` (51 templates); Python modules deleted.
- `PACK_HASHES` dict, `content_pack_hash` per-sample in corpus metadata, `--content-hash` CLI flag.
- `tests/test_pack_loader.py` (37 test cases).

**Deferred (tracked in later WUs):**
- Monster domain migration to TOML (algebra, trigonometry, logic, differential_geometry, and the remaining 27 `.py` domains) → **WU7**.
- Separate installable packages (`formula-engine` / `formula-content` split pyproject.toml) → **WU7**.
- Full provenance fields (author, license, version from `PackMeta`) surfaced in per-sample corpus records, not just the hash → **WU6**.

## Intent

Separate the small, stable **engine** from the large, growing **content**, and move template content from Python literals toward a declarative data representation. This makes content versionable, externally contributable, license-traceable, and freezeable as a benchmark snapshot.

## Problem & motivation

Engine and content are fused. The engine (DSL, sampling, config, corpus loop) is ~1.6k lines; the 33 domains are ~29k lines of Python `Template(...)` literals (`wc -l formula_combinatorics/domains/*.py` totals 29,031). The biggest domains are monoliths: `algebra.py` 1879, `trigonometry.py` 1534, `logic.py` 1441, `differential_geometry.py` 1362. Consequences:

- **Content cannot evolve independently of code.** A new template pack means a code change and a package release.
- **The corpus is not a freezeable artifact.** A benchmark needs a versioned, hash-pinned content snapshot; right now content version == package version.
- **Review and contribution are hard.** Adding or auditing a domain means reading 1-2k lines of nested Python literals (`algebra.py:92+` is a flat list of `Template` objects with inline pool tuples).
- **Licensing provenance is coarse.** The clean-room claim is per-package; per-pack provenance (who authored which templates, under what process) is not represented.

## Current state (evidence)

- Domain modules import engine helpers and emit Python literals: `algebra.py:1-22` imports from `_template_dsl`, `_templates`, `_vocab`; `algebra.py:92-...` is `_ALGEBRA_TEMPLATES: list[Template] = [...]`.
- Central config already exists and is good (`domains/_config.py:41-80`, `register_domain` wrapper). The weight/cap/tag/difficulty metadata is already data, only the templates are code.
- Registry merges modules dynamically (`domains/__init__.py:52-56`), so the loading seam already exists.

## Target state

- Two installable units: **`formula-engine`** (DSL, `sample`, `n_eff`, `compute_weights`, corpus loop, render gate, metadata, config schema) and **`formula-content`** (the domains). Clear, versioned dependency.
- Template content expressed declaratively (TOML/YAML/JSON template packs) loaded by the engine, with Python escape hatches only where the combinatorics genuinely need code (recursive sub-generators like `_expr`). The DSL's `Template`/`Slot` dataclasses are already a near-serializable schema.
- Per-pack metadata: provenance, license, version, author process. A frozen content snapshot is hash-addressable for benchmark reproducibility.
- Monster domains decomposed into sub-regime packs (e.g. `algebra/` split into quadratics, polynomials, logs, fractions, ...).

## Design sketch

- Define a serialization schema for `Template` (name, latex, slots-by-type, distinct, variants). Slots map cleanly: `Slot` = pool + idx; `ExcludeSlot` = pool + exclude_from + idx; the `Sub` family references a named generator from an engine-provided registry (the one irreducible code dependency).
- Engine gains a loader: read template packs from a content directory, resolve sub-generator references against a registered callable table, build the same in-memory `Template` objects the code path builds today. Keep the Python-literal path working during migration (dual-source).
- Migrate domain-by-domain, smallest first (`math_fonts.py` 275, `quantum_notation.py` 291, `linear_algebra.py` 344) to prove the schema before the monoliths.
- Decompose the large domains into sub-regime files/packs as part of migration.

## Acceptance criteria

- Engine and content are separately installable and versioned; engine has no domain content.
- At least the small domains load from declarative packs and produce byte-identical corpora (fixed seed) to the Python-literal path.
- A frozen content snapshot is hash-addressable and recorded in corpus metadata.
- Per-pack license/provenance metadata exists and is surfaced in the corpus manifest.
- The full test suite passes against the data-driven path.

## Dependencies & ordering

- **Inbound:** best after WU2 + WU3 so the engine's behavior (render gate, weighting, coverage) is stable and content migrates once. (See the ordering doc's tradeoff note if the team prefers split-first.)
- **Outbound:** WU5's new knobs and WU7's new domains are markedly cheaper to author on the data-driven base; WU6's stable template ids come from versioned packs.

## Risks & open questions (for fleshing)

- The sub-generator escape hatch: recursive generators (`_expr`, `_atom`, `_poly`) cannot be pure data. Decide the boundary, a declarative template referencing a fixed library of named code generators, and how that interacts with the clean-room/provenance story.
- Migration cost vs payoff for the monoliths: a 1879-line domain is a real port. Consider a code-to-data extractor that reads the existing `Template` literals via AST and emits packs, rather than hand-porting.
- Whether to keep one repo with two packages or split repos; bears on how work agents and external contributors interact with content.
- Serialization format choice (TOML vs YAML vs JSON) and how `r"..."` LaTeX with `{{`/`}}` escaping survives the round-trip.
