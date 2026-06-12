# WU2: Render-in-the-loop validity gate

**Status:** complete | **Effort:** L (3-5 days; XL with full font infra) | **Depends on:** Phase-0 font-stack decision | **Unblocks:** WU3 (empirical signal), WU6 (image paths in records), the agenda's Phase 1 (images)
**Repo:** `Roots-Automation/GutenOCR`, `data/formula_combinatorics/` (PR #24).

## Intent

Every formula the corpus ships is certified to compile and render to an image under a known font stack. This is the single feature that turns a string generator into a training-data instrument.

## Problem & motivation

The package emits LaTeX strings validated by two regexes (`_GREEK_CONCAT_RE`, `_DOUBLE_SUB_RE` in `_template_dsl.py:97-132`) plus per-domain brace-balance assertions in the test suite. Nothing compiles or renders the output. A string that passes the regexes but fails under a real TeX engine (undefined control sequence, bad math-mode nesting, a package not loaded) enters the corpus silently. Three consequences:

1. **The corpus is not certified trainable.** You cannot assert the OCR model is learning from valid images.
2. **No images are produced.** The OCR pipeline needs rendered images; rendering currently lives only in a manual dev tool (`tools/domain_inspector.py`, MathJax to HTML).
3. **No real per-template error rate exists.** Validity is assumed, never measured.

This is also where the clean-provenance story becomes concrete: rendering is the step where fonts actually enter, so the Phase-0 OFL-only font allowlist is enforced here, not in the abstract.

## Current state (evidence)

- Validity = two regexes (`_template_dsl.py:121-132`), invoked at the end of `sample()` (`_template_dsl.py:362`).
- No `subprocess`, no TeX engine, no headless renderer anywhere in `formula_combinatorics/` or `tests/`. The only renderer is `tools/domain_inspector.py` (MathJax to HTML, manual, not a gate).
- `corpus.generate()` (`corpus.py:35-118`) produces strings only; the wrap helpers (`corpus.py:21-32`) add display/inline delimiters but never render.

## Target state

A rendering stage, runnable inline in generation or as a post-pass, that for each formula:

- attempts compilation/render under the target engine with the OFL-only font allowlist (Phase 0: Latin Modern Math, TeX Gyre, STIX Two, XITS, Asana Math, Fira Math, Libertinus Math, Neo Euler), in a sandboxed environment where proprietary fonts are physically unreachable;
- keeps the formula only if it renders cleanly; routes failures to a reject log keyed by `(domain, template name)`;
- emits the rendered image (and its dimensions/bbox if cheap) alongside the formula;
- reports per-template and per-domain success rates.

## Design sketch

Two viable engines, likely both behind one interface:

- **TeX path (authoritative):** `lualatex`/`xelatex` in a minimal container with the font allowlist; render to PDF, crop to image. Highest fidelity to "real document math", and the only path that exercises the actual font stack. Slow; batch and parallelize.
- **Web path (fast triage):** headless KaTeX/MathJax (node or a Python binding) as a cheap pre-filter to catch the bulk of syntax failures before paying for TeX. Note KaTeX supports a subset of LaTeX, so a KaTeX failure is not necessarily a TeX failure; use it as a fast negative filter, not the authority.

Recommended shape: a `render.py` module exposing `render(formula, engine, fontset) -> RenderResult{ok, image_path, error, engine}`; a corpus integration flag (`--render`, `--render-engine`, `--reject-log`) that runs the gate during generation; reject logs aggregated into a per-template report. Keep the renderer optional so string-only fast runs still work.

Performance: rendering 100k-1M formulas is the cost center. Parallelize across workers, cache by formula hash, and consider a two-stage (KaTeX filter, then TeX certify on the survivors) pipeline.

## Acceptance criteria

- A generation run with `--render` produces, for each kept formula, a rendered image and a record of the engine/fontset used.
- A reject log lists every failed formula with its `(domain, template)` and the engine error.
- A CI test renders a fixed-seed sample from every domain and asserts a success-rate floor per domain (threshold TBD, likely high, e.g. >0.99 after template fixes).
- The font sandbox is verified: a test asserts a proprietary font is unreachable and only allowlisted fonts resolve.
- Per-template success-rate report is emitted and consumable by WU3.

## Dependencies & ordering

- **Inbound:** the Phase-0 font-stack decision (which engine, which allowlist). Small; can be made at kickoff.
- **Outbound:** WU3 consumes the per-template success rates; WU6 consumes the image paths and render status in its per-sample records.

## Risks & open questions (for fleshing)

- Engine choice and the KaTeX-subset caveat: decide whether TeX is the sole authority or KaTeX is an acceptable certify target for the web-deploy story. The agenda's CDM evaluation is image-space, which argues for TeX-rendered images as ground truth.
- Throughput at 1M scale: container startup cost, batching strategy, whether to render at generation time or as a separate certified-corpus pass.
- What to do with the ~small tail of legitimate-but-TeX-finicky formulas: fix the template, drop the formula, or quarantine. Tie to WU1's error surfacing.
- Image rendering parameters (resolution, padding, DPI, background) are themselves a benchmark axis (real photos vs clean renders); decide how much of that variation lives here vs in the downstream grounded pipeline (`data/grounded_latex/`).
