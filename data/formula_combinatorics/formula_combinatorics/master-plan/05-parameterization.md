# WU5: Parameterization expansion

**Status:** complete | **Effort:** M (1-2 days) | **Depends on:** WU4 (cheaper on data base), WU3 (symbol set) | **Unblocks:** WU6 stratified splits; the benchmark's stratification axes
**Repo:** `Roots-Automation/GutenOCR`, `data/formula_combinatorics/` (PR #24).

## Intent

Expose, as first-class knobs, the stratification axes the benchmark is defined over. Today only one of the four axes is controllable.

## Problem & motivation

The agenda stratifies the benchmark by four axes: notational regime, symbol-frequency tier, structural complexity, and expression length. Only the first (regime, via `--domains`/`--tags`) is a knob. The others require post-hoc filtering of an undifferentiated corpus, which is wasteful and cannot guarantee a target distribution. Several controls that the code already has data for are simply unwired.

## Current state (evidence)

- Available knobs: `--count`, `--seed`, `--domains`, `--display-fraction`, `--inline-fraction`, `--tags`, `--exclude-tags`, `--metadata` (`generate.py:42-104`).
- **Difficulty exists but is unwired.** `DOMAIN_DIFFICULTY` is built and exported (`domains/__init__.py:67-68`; `DomainMeta.difficulty` at `_config.py:30`), but there is no `--difficulty` flag.
- **Weights are not overridable at call time.** `generate()` takes a `weights` dict (`corpus.py:39`) but the CLI always passes `DEFAULT_WEIGHTS` (`generate.py:127`); changing the distribution means editing `_config.py`.
- **No structural/length/symbol-tier control.** Templates carry no structural-complexity or length tags; there is no symbol-frequency-tier concept (depends on WU3's declared symbol set).

## Target state

New knobs, all composable with the existing ones:

- `--difficulty {elementary,undergraduate,graduate,research}` (filter via the existing metadata).
- `--max-depth` / `--length-range` for structural complexity and expression length (requires per-template structural features, computed once and stored).
- `--symbol-tier {head,body,tail}` selecting strata defined by WU3's frequency-tiered symbol set.
- `--weights PATH` (or `--weight domain=value`) to override the per-domain distribution from a config file, no source edit.
- `--coverage-mode` (from WU3): keep sampling until every declared symbol/template-class is hit N times (the "balanced" sampler made explicit).
- The beginning of held-out-split config (`--hold-out-domains`, `--hold-out-templates`, `--hold-out-symbols`), elaborated fully in WU6.

## Design sketch

- Structural features: at registration (or pack-load, post-WU4), compute per-template depth (brace/fraction nesting), presence flags (fraction, matrix, integral, sub/superscript chain), and a length proxy; store on the `Template` or in a sidecar so filtering is O(1) at sample time.
- Filtering composes at the domain and template level: domain-level filters (difficulty, tags) prune the active set; template-level filters (depth, length, symbol-tier) prune within a domain's template list before dispatch.
- Weight override merges a user dict over `DEFAULT_WEIGHTS`; normalization already handles arbitrary positive weights (`corpus.py:77-79`).

## Acceptance criteria

- Each new flag is honored and tested: a `--difficulty graduate` run contains only graduate domains; a `--length-range` run respects bounds; `--weights` shifts the empirical domain distribution to the requested one.
- Filters compose correctly (e.g. `--tags advanced --max-depth 3` intersects).
- A run that over-constrains (empty result set) fails loudly with a clear message, not a silent short corpus (ties to WU1).

## Implementation notes (post-completion)

- `--symbol-tier` was implemented using `SYMBOL_STRATA` type-based tier names (`greek`, `calligraphic`, `blackboard_bold`, `functions`, `bold_vectors`, `bold_greek`) rather than the frequency-based `head`/`body`/`tail` partition the spec described. WU3 (complete) did not deliver a frequency-ordered partition. The frequency-tier partition and the `--symbol-tier head/body/tail` extension are deferred to WU6. See WU6's deferred section.
- `--coverage-mode` was not implemented; deferred to WU6 (benchmark-corpus quality concern). See WU6's deferred section.
- `--hold-out-templates` and `--hold-out-symbols` were not implemented; deferred to WU6 (see that unit's deferred section). `--hold-out-domains` was implemented here.
- Presence flags (`has_fraction`, `has_matrix`, `has_integral`, `has_script_chain`) were added to `Template` as stored structural metadata, available to WU6's rich per-sample records and any future presence-flag filter knobs.

## Dependencies & ordering

- **Inbound:** WU4 makes per-template feature tagging cheap; WU3 defines the symbol tiers that `--symbol-tier` selects.
- **Outbound:** WU6's held-out splits build directly on these filters and the structural features.

## Risks & open questions (for fleshing)

- Structural-complexity metric definition: what counts as "depth", how length is measured (token count vs rendered width vs character count). Rendered width (from WU2) may be the most OCR-relevant.
- Over-constraint behavior and the rejection-loop budget (`max_attempts` at `corpus.py:85`): tight template-level filters can starve the loop; consider pre-filtering the template list rather than rejection sampling.
- Whether symbol-tier filtering operates on templates (does this template's pool include a tail symbol?) or on draws (post-hoc), and how that interacts with coverage-mode.
