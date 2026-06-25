# Domain certification rubric

Applied to every domain TOML before it enters the certified corpus.
A domain **passes** when all ten criteria clear their bar.
Any failing criterion blocks Track A corpus generation for that domain.

---

## Quick-reference checklist

- [ ] **Scope** — content matches the domain's stated level; nothing out of scope
- [ ] **Coverage** — no obvious sub-topic gaps at this level
- [ ] **Grouping** — no cross-template duplication; variants nested within the correct template
- [ ] **TOML hygiene** — 2-space indentation, section headers, named pool constants
- [ ] **Slot types** — S / X / E / EP / P assigned correctly for each slot's role
- [ ] **Pool hygiene** — pools sized and populated for their semantic role; no dead pools
- [ ] **idx discipline** — idx present only where it adds visual diversity; consistent across variants
- [ ] **n_eff floor** — every leaf template n_eff ≥ 10,000 (or documented exception)
- [ ] **Collision certification** — `collision_probe` at 10k draws shows no unexpected saturation
- [ ] **Render certification** — zero rejects through the two-stage pipeline

---

## Criteria detail

### 1. Scope

**What:** Every template belongs to this domain at its declared difficulty level. Nothing is out of scope (too advanced, wrong sub-field) or trivially misclassified.

**Bar:** A reviewer familiar with the subject can confirm every template name without raising an objection.

**How:** Human review. Cross-check template names against the domain's stated level (`difficulty` field and domain header comment).

**Common failures:**
- Graduate-level content in an undergraduate domain (e.g. binomial theorem in elementary algebra)
- Templates copied from another domain without renaming or rechecking scope

---

### 2. Coverage

**What:** The domain covers its sub-topic space without obvious gaps. A student or practitioner would not expect a formula to be present and find it missing.

**Bar:** Reviewer can identify no more than one or two plausible additions, and those additions are documented as intentional omissions (too advanced, covered elsewhere, or out of scope).

**How:** Human review. List sub-topics for the domain; verify each is represented by at least one template.

**Common failures:**
- Entire sub-topic class missing (e.g. no logarithm identities in an algebra domain)
- Only the simplest form of a concept covered, none of its standard variants

---

### 3. Grouping

**What:** Semantically related formula variants are grouped as `[[template.variants]]` under a shared `[[template]]`. No variant duplicates content already covered by another template in the same domain.

**Bar:** Zero cross-template duplicates. Variants that share slots and structure are grouped together, not scattered as separate top-level templates.

**How:** Human review. For each template, verify its variants are thematically coherent. For each variant, confirm no other template produces the same formula class.

**Common failures:**
- A slope-intercept form appearing both as a standalone template and as a variant under `linear_system`
- Variants with unrelated structures grouped under the same template because they were added opportunistically

---

### 4. TOML hygiene

**What:** The TOML file follows project conventions: 2-space indentation at every nesting level, section header comments (`# ---`) separating logical groups, pool references use named constants (`pool = "VARS_POOL"`) rather than duplicate inline literals, no empty section headers.

**Bar:** `tomllib.load()` parses without error; no inline pool literal appears more than once that could be a named constant; no dead section headers.

**How:** `python -c "import tomllib; tomllib.load(open('domain.toml','rb'))"`. Human review for style.

**Common failures:**
- Empty section header left over from a removed group of templates
- Inline pool literal `['x','y','z']` repeated across multiple slots instead of factored into a named constant
- Inconsistent indentation (4-space mixed with 2-space)

---

### 5. Slot types

**What:** Each slot uses the correct DSL type for its role:
- `S` — independent draw from a pool
- `X` — draw from a pool excluding one or more previously drawn slots
- `E` — expression sampled by a generator function (opaque, declares `n` for n_eff)
- `EP` — expression generator parameterized by another slot's value
- `P` — parameterized sub-expression (draws from another slot's value)

**Bar:** No slot uses `S` where exclusion is semantically required (would produce `x + x`), no slot uses `E` where a pool would suffice, no slot uses `X` without a matching `exclude_from`.

**How:** Human review. For each slot, ask: can two independently drawn values from this slot be equal in a way that produces a degenerate formula?

**Common failures:**
- Two variable slots both `S` from the same pool in a formula like `{a}/{a}` — should be `X`
- `X` slot with `exclude_from` pointing to a slot that does not exist in this variant
- `E` used where a small named pool would give better n_eff transparency

---

### 6. Pool hygiene

**What:** Pools are sized appropriately for their semantic role. Entries are correct LaTeX symbols for that role. No entry is duplicated within a pool. Pools that are reused across templates are declared once as named constants.

**Bar:** No duplicate entries within a pool. Every entry renders as a valid LaTeX token in context. Pool size is justified by the slot's semantic breadth (a "variable" pool should not have 2 entries; a "constant" pool should not have 100).

**How:** Human review of pool declarations. `collision_probe` will surface dead or tiny pools indirectly (low n_eff).

**Common failures:**
- Pool with only 1–2 entries when 8–16 would be natural (limits n_eff severely)
- Pool containing a symbol that produces a LaTeX render error in context (e.g. a command that requires a package not loaded)
- Same 8-symbol pool copy-pasted inline across 12 templates instead of declared once

---

### 7. idx discipline

**What:** The `idx` decoration (subscript with probability `idx`, default pool of 8 subscripts) is applied only where a subscripted variable is natural and visually uncluttered. The same slot role uses the same `idx` value across all variants of a template.

**Bar:** No variant has `idx` on a slot where sibling variants lack it for the same slot role without documented reason. `idx` is absent on base/subscript slots where the base symbol is already inherently subscripted (e.g. a log base that would become `\log_{a_{1}}`). `idx` is absent on slots that appear many times in a single formula where repeated subscripted decoration would be cluttered.

**How:** Human review. For each template, list all slot names and their `idx` values across variants; verify consistency.

**Common failures:**
- Four variants of a template have `idx=0.35` on a slot; four sibling variants have none — inconsistency without documented reason
- `idx` applied to a log base slot that already draws from a symbol pool like `LOG_BASES_POOL`, producing doubly-subscripted output (`\log_{a_{1}}`)
- `idx` applied to a slot that appears 3–4 times in a single formula, making it visually cluttered

---

### 8. n_eff floor

**What:** Every leaf template (a template with no variants, or each variant individually) has a declared n_eff of at least 10,000. Templates below this floor cannot survive 10k collision-probe draws without saturating, which means they cannot be certified.

**Bar:** `min(n_eff(t) for t in leaf_templates) ≥ 10,000` for parametric templates.

**Notation-only exception:** A leaf variant may be certified with n_eff ≥ 100 if ALL of the following hold:
1. Its only slot variation is a function-name pool (notation choice) and a single argument variable — the formula's semantic content does not change with the choice of argument name.
2. All NAME-type pools are at their maximum realistic size — no entries fabricated solely to raise n_eff.
3. A coeff variant of the same identity exists in the same template and clears the 10,000 floor.
4. The variant is suffixed `_bare` or `_pure` in its name, making its status identifiable in tooling output.

These variants represent how a formula appears in simple textbooks alongside more diverse coeff variants. They are expected to saturate in `collision_probe` at 10k draws; this saturation is documented and acceptable.

**How:**
```bash
uv run python tools/collision_probe.py --domain <name> --batch-per-template 10000
```
Check the `n_eff` column in output. Any template below 10,000 is flagged.

**Common failures:**
- Small pool (3–4 entries) with no `idx` decoration → n_eff = 12–64
- `distinct` constraint applied so aggressively that effective combinations collapse (e.g. `perm(4,4) = 24`)
- `E` slot declaring `n=10` instead of an accurate estimate of its generator's output space

---

### 9. Collision certification

**What:** Running `collision_probe` at 10,000 draws per template produces a realized-unique count consistent with the birthday-paradox expectation for that template's declared n_eff. No template saturates (observed unique ≈ draw count) when n_eff >> 10,000.

**Bar:**
- For n_eff ≥ 100,000: observed unique ≥ 9,500 (no saturation signal)
- For 10,000 ≤ n_eff < 100,000: observed unique consistent with birthday-paradox formula `n_eff × (1 - e^{-k/n_eff})` within ±10%
- No template with n_eff < 10,000 (see criterion 8)

**How:**
```bash
uv run python tools/collision_probe.py --domain <name> --batch-per-template 10000
```
Compare `unique` column against expected birthday-paradox value.

**Common failures:**
- Observed unique far below n_eff prediction — indicates n_eff is over-declared (pool sizes wrong, exclusions not counted, idx multiplier misapplied)
- Observed unique = draw count — template is saturating; n_eff is effectively below 10k
- Observed unique >> n_eff prediction — indicates n_eff is under-declared (missed a slot, idx not counted)

---

### 10. Render certification

**What:** Every template in the domain renders successfully through the two-stage pipeline (KaTeX pre-filter → lualatex). Zero rejects.

**Bar:** Reject rate = 0.0% for the domain. Any rejection is a blocking defect.

**How:**
```bash
uv run formula-generate --domains <name> --count 5000 --seed 42 \
  --render --render-engine two-stage \
  --render-reject-log /tmp/<name>_rejects.jsonl
wc -l /tmp/<name>_rejects.jsonl   # must be 0
```

**Common failures:**
- Display environment (`align*`, `multline`, `gather*`, `equation+split`) wrapped in inline-math template — use bare document template
- LaTeX command requiring a package not in the render template (e.g. `\ce{}` without mhchem, `\mathbb{}` without amsfonts)
- Malformed pool entry that produces invalid LaTeX in context (unclosed brace, unknown command)

---

## Applying the rubric

For each domain:

1. **Human review** — criteria 1–7 require reading the TOML. Flag each failure with the criterion number and a one-line description.
2. **Automated checks** — run collision_probe (criteria 8–9) and the render smoke-test (criterion 10).
3. **Record outcome** — in the domain's section of `REPORT.md`, mark each criterion Pass or Flag. A domain with any Flag is not certified for the corpus.
4. **Fix and re-run** — address flagged criteria, re-run automated checks, update the record.

A domain is **certified** when all ten criteria are marked Pass.
