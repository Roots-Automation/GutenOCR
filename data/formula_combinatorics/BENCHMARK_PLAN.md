# formula-combinatorics: benchmarking & demonstration plan

Drafted 2026-06-24. Two tracks: **A — demonstrate the corpus** (benchmark the *data*; cheap, mostly run-the-pipeline) and **B — demonstrate it works** (benchmark a *model* with the data; the real payoff, training-bound). Do A first; A's rendered corpus + report is an input to B and is independently showable.

The generation engine, the benchmark-native JSONL output (`--metadata` rich Sample records), the two-stage render gate, the hold-out split machinery, and the diagnostic tools all already exist. Neither track is "build infrastructure"; both are "execute, validate, report."

CLI entrypoint is the `formula-generate` console script (prefix with `uv run` if the venv is not active). 39 domains across the foundational→research taxonomy.

---

## Step 0 (prerequisite for both): verify the upgrade wave is actually complete

The recent commit wave was foundational-heavy (algebra, geometry, trigonometry, calculus). That is not the same as "all 39 domains audited." Confirm completion objectively before declaring it done; expect any gaps in the advanced/research tail (category_theory, p_adic, representation_theory, the graduate domains).

```bash
# Symbol coverage per domain (MUST_COVER strata)
uv run python tools/coverage_report.py            # see --help for args

# Realized diversity / birthday-horizon per domain (is n_eff real, not just declared)
uv run python tools/collision_probe.py            # supports --batch-per-template N

# Visual spot-check per domain
uv run python tools/domain_inspector.py
```

**The single best objective completion signal is per-domain render-reject rate** (produced in track A below). A domain that renders dirty or shows low realized diversity is not finished regardless of commit count.

**Acceptance for Step 0:** every domain hits its MUST_COVER symbol coverage, realized diversity is adequate (collision_probe), no dead pools, and (after A2) per-domain render pass-rate clears the bar. Record any under-built domains as remaining work.

---

## Track A — Demonstrate the corpus (benchmark the data)

**Goal:** a rendered, benchmark-native corpus plus a quality/coverage report and a sample gallery, proving the synthetic data is diverse, clean-rendering, and IP-clean. ~Half-day to a day given what is built.

### A0. Pin the snapshot (provenance)

```bash
uv run formula-generate --content-hash      # aggregate SHA-256 of all TOML packs
```
Record this hash in the report. Every rich record also carries `content_pack_hash / name / version / author / license`, so the corpus is self-describing and reproducible.

### A1. Run the Step-0 diagnostics (above)

### A2. Generate the certified, rendered, benchmark-native corpus

```bash
uv run formula-generate \
  --output corpus/formulas.jsonl --output-format jsonl --metadata \
  --count 50000 --seed 42 \
  --render --render-engine two-stage \
  --render-output corpus/images/ \
  --render-reject-log corpus/rejects.jsonl
```
Yields: `corpus/formulas.jsonl` (rich records: formula, domain, template_name, semantic_key, split, depth, char_length, has_fraction/matrix/integral/script_chain, strata, n_eff, symbol_tier, render_ok, image_path, difficulty, provenance), `corpus/images/<hash>.png` (one per certified formula), `corpus/rejects.jsonl` (failures with domain/template/engine/error).

Prereqs: `node` + `katex` npm package (KaTeX pre-filter); `lualatex` + `pdftoppm`/poppler (TeX stage). Render uses the OFL-only font allowlist.

### A3. Build the quality report (the deliverable)

Compute from `formulas.jsonl` + `rejects.jsonl`:

- **Render pass-rate**, overall and per-domain (`1 - rejects/attempts`). Bar: e.g. ≥99% two-stage; investigate any domain below.
- **Distribution**: per-domain realized counts vs target weights; difficulty and symbol_tier breakdowns.
- **Diversity**: realized unique formulas vs declared n_eff; collision rate at this count (semantic_key dedup / collision_probe).
- **Coverage**: MUST_COVER symbol coverage (coverage_report).
- **Structural mix**: has_fraction/matrix/integral/script_chain rates; depth and char_length histograms.
- **Provenance / IP**: the pinned pack hash + per-pack license + OFL-font allowlist = the clean-room, no-third-party-content story.

### A4. Sample gallery + write-up

Render a grid of N images per domain/difficulty (sample from `corpus/images/` grouped by `domain`/`difficulty` in the JSONL; `domain_inspector.py` for interactive browsing). Write a short `REPORT.md` with the tables + gallery + pinned hash.

**Acceptance for A:** full corpus generated and rendered at/above the pass-rate bar; report shows per-domain coverage, diversity, structural mix, and provenance; gallery renders; snapshot hash pinned.

---

## Track B — Demonstrate it works (benchmark a model with the data)

**Goal:** show the synthetic corpus measurably improves formula recognition. This is the demonstration that earns the project's keep. Multi-day, training-bound.

### Design decisions

- **Generalization via hold-out (the core design).** Build train + three held-out eval sets, each isolating a different generalization claim:
  - `--hold-out-domains` → can the model handle *unseen math domains*?
  - `--hold-out-templates` → *unseen templates within seen domains*?
  - `--hold-out-symbols` → *unseen MUST_COVER symbols*?
  This is exactly what the hold-out machinery + split manifest were built for.
- **External transfer benchmark.** Evaluate on a real formula-OCR benchmark to prove transfer beyond synthetic. Primary: **im2latex-100k** (printed LaTeX, closest to this distribution). Stretch / negative control: **CROHME** (handwritten; expect a distribution gap).
- **Metrics:** CER (matches GutenOCR's existing Fox reporting, 0.053 CER), exact-match accuracy, token-level edit distance on the LaTeX stream. Optional render-and-compare (compile prediction, image-diff) for a normalization-robust score.
- **The claim is an ablation:** baseline (GutenOCR as-is / without formula synthetic) vs +synthetic, everything else fixed.

### B1. Build the splits

```bash
uv run formula-generate \
  --output splits/train.jsonl --output-format jsonl --metadata \
  --count 200000 --seed 42 \
  --hold-out-domains number_theory topology \
  --hold-out-templates 0.10 \
  --hold-out-symbols 0.05 \
  --render --render-engine two-stage --render-output splits/train_images/ \
  --manifest splits/split.manifest.json
```
Writes `train.jsonl` + companion held-out files (`*.held_out_templates.*`, held-out-symbols companion) + `split.manifest.json`. Then materialize the eval sets:
- `eval_unseen_templates` from the held-out-templates companion file,
- `eval_unseen_domains` via `--domains number_theory topology`,
- `eval_unseen_symbols` from the held-out-symbols companion.
Render each (`--render`). Confirm the exact companion-file → eval-set mechanics against the manifest / `--help`.

### B2. Baseline

Evaluate current released GutenOCR (3B/7B) on im2latex-100k and on the three synthetic held-out eval sets → baseline CER / exact-match. This is the "before."

### B3. Train / finetune on the synthetic train corpus

**Integration point owned by the main GutenOCR training pipeline** (DeepSpeed ZeRO on DGX H100), not by formula_combinatorics. The data contract this track provides: rendered `image_path` → `formula` (LaTeX target) pairs in JSONL, with per-sample metadata for stratified eval. Add as a training source in the existing trainer. (This is the one piece not verified from inside this package; you own that pipeline.)

**Data contract:** each record in `splits/train.jsonl` provides `image_path` (absolute path to a PNG rendered at `--render-dpi`, default 150 DPI) and `formula` (raw LaTeX string, no normalization). The trainer consumes these two fields; all other metadata fields (`domain`, `template_name`, `difficulty`, `split`, etc.) are available for stratified sampling or curriculum ordering but are not required.

### B4. Evaluate the finetuned model

Same eval sets as B2 (im2latex-100k + the three synthetic held-out sets) → "after."

### B5. Report

Ablation table: (baseline vs +synthetic) × (im2latex-100k, unseen-templates, unseen-domains, unseen-symbols) on CER / exact-match. The generalization axes say *what* the synthetic data teaches (symbols vs templates vs domains); im2latex transfer says it is not just memorizing the synthetic distribution.

**Acceptance for B:** a clean ablation showing +synthetic improves (or characterizing where it does and does not) on real im2latex plus the generalization splits.

### B risks / gotchas

- **LaTeX non-uniqueness.** im2latex CER needs a normalizer or render-and-compare for a fair score; do not compare raw token streams naively.
- **Distribution gap.** Synthetic is clean printed; im2latex (printed) is a good match, CROHME (handwritten) is a gap, treat as stretch.
- **Confounds.** Hold model size, steps, and non-formula data fixed across baseline vs +synthetic so the delta is attributable to the synthetic corpus.

---

## Sequencing

1. Step 0 diagnostics (fold the render-reject signal in from A2).
2. Track A: corpus + report + gallery. Showable on its own; also produces the rendered data B consumes.
3. Track B: splits → baseline → finetune → eval → ablation.

## Note

There is no written roadmap in the repo today; the plan has lived in the commit stream. Consider committing this file (or a `## Next` section in CHANGELOG) so pieces are hand-off-able to the juniors on the VLM line.
