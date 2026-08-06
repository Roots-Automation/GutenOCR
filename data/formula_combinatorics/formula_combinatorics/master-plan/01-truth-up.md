# WU1: Truth-up & hygiene

**Status:** complete (185d3db) | **Effort:** S (hours) | **Depends on:** nothing | **Unblocks:** clean base for all later units
**Repo:** `Roots-Automation/GutenOCR`, `data/formula_combinatorics/` (PR #24).

## Intent

Make the codebase honest about what it does before any agent builds on it. Small, mechanical, high-trust-per-line.

## Problem & motivation

Several places where the documentation or behavior misrepresents the code. Each is a future-agent landmine: a doc that describes a prior architecture, a failure mode that hides itself, a weighting inconsistency that quietly violates the stated design.

## Current state (evidence)

1. **README documents a parameter that does not exist.** `README.md` advertises `--align-fraction` (default 0.15), an `align_fraction=0.15` Python API arg, and "an additional align_fraction of output uses multi-line environments" (`README.md:38,55,86,138`). The real `corpus.generate()` signature has no such parameter (`formula_combinatorics/corpus.py:35-46`); the actual knobs are `--display-fraction` (0.20) and `--inline-fraction` (0.10), and `align` is now a *domain* weighted 0.15 in `domains/_config.py:79`. The README describes the pre-split architecture.
2. **README says 24 domains.** `README.md:3,107` and its domain table list 24; the package registers 33 (`domains/__init__.py:12-46`). The PR body says 33. The README is stale by two architectural generations.
3. **Generator errors are swallowed silently.** The generate loop wraps each draw in `except Exception: logger.warning(...); continue` (`corpus.py:108-109`). A systematically broken template or domain under-contributes to the corpus invisibly; the caller gets a short corpus with at most a generic "generated N/M" warning (`corpus.py:111-117`), never a per-domain error rate.
4. **`variants` are sampled uniformly, not by n_eff.** Template-level sampling is sqrt(n_eff)-weighted (`_template_dsl.py:469-487`), but a template with `variants` picks among them with `rng.choice(t.variants)` (`_template_dsl.py:314`), uniform. A high-combinatorics variant and a trivial one are equally likely, inconsistent with the stated weighting philosophy.
5. **Version is `0.1.0`** (`pyproject.toml`) with no changelog; the package has had multiple architectural shifts (align-as-fraction to align-as-domain, 24 to 33 domains, central config).

## Target state

- README regenerated from the actual CLI surface (or, better, a `--help`-derived section so it cannot drift again). Domain count and table auto-derived from `domains/__init__.py`.
- `generate()` returns or logs a per-domain attempt/success/error tally so a broken domain is visible, not silent. Decide: structured return field vs a summary log line at WARNING when any domain's error rate exceeds a threshold.
- `variants` sampled by sqrt(n_eff) weight, consistent with template-level dispatch.
- Version bumped (0.2.0) with a short CHANGELOG capturing the align and 33-domain shifts.

## Design sketch

- For the README: add a small `tools/gen_readme_tables.py` (or a doctest-style test) that renders the domain table and CLI options from the live registry, so drift becomes a test failure.
- For error surfacing: accumulate a `Counter` of `(domain -> attempts)` and `(domain -> errors)` in the generate loop; emit a one-line summary; optionally raise if any domain's error rate is above a configurable ceiling (default off, to preserve robustness).
- For variants: give `Template` a cached per-variant weight, or have `sample()` choose variants via `rng.choices(t.variants, weights=[sqrt(n_eff(v)) for v in t.variants])`. Precompute at registration to avoid per-draw cost.

## Acceptance criteria

- A test asserts the README domain count and CLI option set match the live registry.
- A unit test injects a deliberately-throwing domain and asserts the per-domain error tally reflects it.
- A statistical test asserts variant selection frequency tracks sqrt(n_eff) within tolerance.
- `pyproject.toml` version bumped; CHANGELOG present.

## Dependencies & ordering

None inbound. Should land first so later agents read accurate docs and get real error signal while building units 2-7.

## Risks & open questions (for fleshing)

- Should the error ceiling default to raising or warning? Raising is safer for benchmark builds, warning is safer for exploratory runs. Possibly a `strict: bool` flag.
- Variant reweighting changes the output distribution of every domain that uses variants; confirm with the team whether any frequency-threshold tests in `tests/` encode the current uniform behavior and will need rebaselining.
