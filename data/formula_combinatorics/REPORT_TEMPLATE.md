# formula-combinatorics corpus report

Generated: <!-- date -->
Content pack hash: <!-- uv run formula-generate --content-hash -->
Corpus: `corpus/formulas.jsonl` · `corpus/images/` · `corpus/rejects.jsonl`

---

## 1. Render pass-rate

| Domain | Attempts | Rejects | Pass-rate |
|--------|----------|---------|-----------|
| _all_ | | | |
<!-- one row per domain; flag any domain below 99% -->

**Overall:** X / Y (Z%)
**Bar:** ≥99% two-stage. Domains below bar: <!-- list or "none" -->

---

## 2. Distribution

Per-domain realized counts vs target weights.

| Domain | Target weight | Realized count | Realized % |
|--------|--------------|----------------|------------|
<!-- one row per domain -->

Difficulty breakdown:

| Difficulty | Count | % |
|------------|-------|---|
<!-- easy / medium / hard / research -->

Symbol tier breakdown:

| Tier | Count | % |
|------|-------|---|

---

## 3. Diversity

| Domain | Declared n_eff (min template) | Realized unique | Collision rate |
|--------|-------------------------------|-----------------|----------------|
<!-- collision rate = duplicate semantic_keys / total draws for that domain -->

Notes on any domain where realized unique falls short of declared n_eff.

---

## 4. Symbol coverage (MUST_COVER)

Output of `uv run python tools/coverage_report.py` summarized here.

| Domain | MUST_COVER symbols | Covered | Missing |
|--------|--------------------|---------|---------|
<!-- flag any domain with missing symbols -->

---

## 5. Structural mix

| Feature | Count | Rate |
|---------|-------|------|
| has_fraction | | |
| has_matrix | | |
| has_integral | | |
| has_script_chain | | |

Depth histogram: <!-- min / p25 / median / p75 / max -->
Char length histogram: <!-- min / p25 / median / p75 / max -->

---

## 6. Provenance / IP

- Content pack hash: <!-- same as header -->
- Licenses: <!-- list per-pack licenses from formulas.jsonl provenance fields -->
- Font allowlist: OFL-only (enforced via `--render-engine two-stage`)
- No third-party formula content: all templates hand-authored in TOML

---

## 7. Sample gallery

One representative render per domain (highest-difficulty available).
Full interactive grid: `uv run python tools/domain_inspector.py`

| Domain | Sample |
|--------|--------|
<!-- | algebra | ![](corpus/images/<hash>.png) | -->
<!-- one row per domain, image path relative to repo root -->

---

## Acceptance checklist

- [ ] Overall render pass-rate ≥99%
- [ ] No domain below 99% pass-rate (or documented exception)
- [ ] All MUST_COVER symbols covered in every domain
- [ ] Realized unique consistent with declared n_eff (no template saturating far below)
- [ ] Content pack hash pinned above
- [ ] Gallery renders for all domains
