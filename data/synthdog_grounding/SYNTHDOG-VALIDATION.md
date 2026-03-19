# SynthDoG Grounding — Validation Threads

Open questions and known issues surfaced during code review. Each thread has enough
context to investigate and resolve independently.

---

## Thread 1 · Luminance threshold is wrong (quality impact) — RESOLVED

**Status: RESOLVED** — threshold corrected in `content.py:28` and `content.py:75`;
see fix details below.

### Original problem statement

**File:** `elements/content.py` (formerly line 73, now line 75)

```python
# OLD (wrong):
gray_range = [0, 64] if lum > 0.5 else [191, 255]
```

The threshold 0.5 is not the WCAG crossover point. The crossover — where white and
black text have equal contrast against a background — is at luminance ≈ **0.179**:

```
contrast_white = 1.05 / (L + 0.05)
contrast_black = (L + 0.05) / 0.05
equal when (L + 0.05)² = 0.0525  →  L ≈ 0.179
```

With threshold = 0.5, backgrounds in luminance [0.179, 0.5] (muted colors, medium
grays) are classified as "dark" and receive light text [191, 255]. But dark text
[0, 64] would give significantly better contrast over this range.

### Empirical findings

Analysis script: `analyze_luminance_threshold.py` (`uv run python analyze_luminance_threshold.py`)

| Metric | Value |
|--------|-------|
| Affected fraction of paper samples | **21.19%** |
| Worst-case WCAG contrast (old threshold, lum ≈ 0.499) | **1.04 : 1** (near-invisible) |
| Worst-case WCAG contrast (corrected threshold, lum ≈ 0.180) | **2.27 : 1** |
| Samples in affected zone with contrast < 3:1 | **100%** |

Post-processing (brightness ±48, contrast ×1–1.5) is not a reliable mitigating
factor — it can worsen contrast on already-poor samples.

### Known edge case

At lum ≈ 0.180–0.184 the corrected threshold (0.179) produces marginally worse
worst-case contrast (~2.27:1 vs ~2.48:1 under the old threshold). This is because
both text ranges are nearly equidistant from that narrow band. This is expected
behavior and a massive improvement over the global worst case of 1.04:1 under the
old threshold. It is not a regression.

### Fix applied

`elements/content.py:28` and `elements/content.py:75` — changed threshold from `0.5` to `0.179`:

```python
# NEW (correct):
gray_range = [0, 64] if lum > 0.179 else [191, 255]
```

Same correction applied to the `_make_adaptive_color` helper (`lum < 0.179`).

### Reproduction

```bash
cd data/synthdog_grounding
uv run python analyze_luminance_threshold.py
```

---

## Thread 2 · `build_block_annotations` is called twice; first result is discarded — RESOLVED

**Status: RESOLVED** — `filter_degenerate` no longer builds or accepts blocks;
`build_annotations` calls `build_block_annotations` once, after filtering, on surviving lines only.

### Original problem statement

**File:** `annotations.py:255` and `annotations.py:172`

```python
# build_annotations():
blocks = build_block_annotations(block_ids, line_bboxes)   # ← result immediately overwritten

lines, words, blocks, deg_line_ct, deg_word_ct = filter_degenerate(
    lines, words, blocks, ...
)
# filter_degenerate() calls build_block_annotations() internally and returns new blocks
```

The first `build_block_annotations` call is dead work. Its output is passed into
`filter_degenerate` only to be discarded when blocks are rebuilt from the surviving
lines.

### Fix applied

- `filter_degenerate` signature changed: `blocks` parameter removed, return type narrowed from
  5-tuple to 4-tuple (no `list[BlockAnnotation]`). Internal `build_block_annotations` call removed.
- `build_annotations` updated: dead `blocks = build_block_annotations(...)` line removed;
  `build_block_annotations` now called once after `filter_degenerate`, using surviving lines only.

---

## Thread 3 · `filter_degenerate` mutates annotation objects in-place — RESOLVED

**Status: RESOLVED** — `filter_degenerate` now uses `dataclasses.replace()` to
produce new `LineAnnotation` / `WordAnnotation` instances with updated IDs; input
objects are never mutated.

**File:** `annotations.py:156-167`

```python
ln.line_id = new_idx
wd.line_id = old_to_new[wd.line_id]
wd.word_id = new_word_id
```

These are the same `LineAnnotation` / `WordAnnotation` dataclass instances stored
in the `data` dict returned by `generate()`. In-place mutation through shared
references means any code holding a reference to those objects before `filter_degenerate`
runs sees silently updated IDs.

Currently harmless because nothing else holds references across this point, but it
is a latent bug source if the call order or data flow changes.

**Resolution:** Build new annotation instances with updated IDs rather than
mutating existing ones.

---

## Thread 4 · `document.py` pokes SynthTiger's private `_init` method

**File:** `elements/document.py:69`

```python
self.elastic_distortion._init(**elastic_config)
```

`_init` is a private SynthTiger method. If SynthTiger renames or changes it,
this line silently no-ops — the elastic distortion config stops being applied and
there is no error or warning.

**Questions to resolve:**
- What is the correct public API to configure a SynthTiger component post-construction?
- Can the component be constructed with the config in the first place to avoid this?

---

## Thread 5 · `TextReader` file handle is never explicitly closed

**File:** `elements/readers.py:30`

```python
self.fp = open(path, encoding="utf-8")  # noqa: SIM115
```

The file is opened in `__init__` and the only guaranteed close path is `__del__`,
which is unreliable under CPython and essentially never called during a long
generation run. The `noqa: SIM115` suppresses the linter hint to use a context manager.

One open file handle per worker process is not a crisis, but it is a correctness
issue, and it is what the linter is flagging.

**Resolution options:**
- Pass the file path to `Content` and open it inside a `with` block for the
  duration of `generate()`.
- Make `TextReader` a proper context manager and ensure it is entered/exited at
  the `Document` level.
- Alternatively: accept the leak and document it, since the OS reclaims handles
  on process exit and workers are short-lived.

---

## Thread 6 · Leading-space textbox artifact causes bbox offset

**File:** `elements/textbox.py:119-124`

```python
if len(chars):
    text.prev()   # backs reader up to point AT the space
    chars.pop()
    char_layers.pop()
```

After this, the reader cursor sits on the boundary space character. The next call
to `generate()` reads that space first. Since spaces are not `\r\n` they are not
skipped (line 101); a `TextLayer` is created for the space and placed at `left=0`.
The line's bounding box left edge is therefore the space character's left edge,
not the first visible character's left edge.

The text string (`.strip()`'d at line 126) has no leading space, so the stored
text and its bbox are slightly misaligned horizontally.

**Questions to resolve:**
- How wide is a space character in practice relative to line height? Is the offset
  measurable?
- Should the reader be advanced past the space so the next textbox starts at the
  first non-space character?

---

## Thread 7 · `_package_data` serializes blocks/words but not lines (inconsistency)

**File:** `template.py:102-116` vs `template.py:310`

```python
# _package_data():
text_blocks_dicts = [block_annotation_to_dict(b) for b in blocks]
text_words_dicts  = [word_annotation_to_dict(wd) for wd in words]
# lines are NOT serialized here

# save():
text_lines_data = [line_annotation_to_dict(ln) for ln in lines]  # done here instead
```

`data["text_blocks"]` and `data["text_words"]` are already plain dicts when they
leave `generate()`. `data["lines"]` is still a list of `LineAnnotation` dataclasses.
Anything consuming the raw data dict sees an inconsistent mix of types.

**Resolution:** Serialize all three annotation types in the same place — either
all in `_package_data`, or all in `save()`.

---

## Thread 8 · `content_color` silently overwrites `textbox_color` (undocumented interaction)

**File:** `elements/content.py:112-118`

```python
for ...:
    textbox_color.apply([text_layer])   # fires per-line (prob 0.2, or 1.0 on dark bg)

content_color.apply(text_layers)        # fires once for ALL lines (prob 0.2, or 1.0 on dark bg)
```

When `content_color` fires it overwrites whatever `textbox_color` set on individual
lines. On dark backgrounds both are forced to `prob=1.0`, so `content_color` always
wins and all per-line color variation from `textbox_color` is discarded.

The net behavior is correct (unified document color), but it is not obvious from the
config that `textbox_color` only has visible effect when `content_color` does not fire.

**Resolution options:**
- Document the interaction explicitly in the config and code.
- Rethink the two-stage color model: does `textbox_color` serve a purpose distinct
  from `content_color`, or can one be removed?

---

## Thread 9 · Post-processing shadow renders text unreadable regardless of text color (quality impact)

**Files:** `config/config_base.yaml:97-100`, `elements/content.py:75-82`

### Problem statement

Text color is chosen at generation time against the raw paper color. Post-processing
effects — specifically the shadow and brightness steps — are applied afterwards with
no awareness of text readability.

The shadow effect is configured as:

```yaml
- prob: 1
  args:
    intensity: [0, 160]
    amount:    [0, 1]
    bidirectional: 0
```

At high intensity + high amount, a unidirectional shadow darkens large regions of
the image close to pure black. Dark text that was correctly chosen for a light
background becomes dark-on-dark and near-invisible in those regions. The brightness
step (`beta: [-48, 48]`) can compound this further.

The luminance threshold fix (Thread 1) provides no protection here: it reasons about
the original paper color, not the final composited scene.

### Observed failure

A muted pink background (lum > 0.179) correctly received dark text [0, 64]. After
compositing, the shadow reduced the left half and bottom third of the image to
near-black, making the dark text unreadable in those regions.

### Resolution options

- **Clamp shadow parameters** — reduce the upper bounds (e.g. `intensity: [0, 80]`,
  `amount: [0, 0.5]`) so the worst-case shadow cannot obscure text.
- **Post-hoc contrast check** — after compositing, sample contrast in the text
  regions and reject (regenerate) samples below a threshold.
- **Shadow-aware color selection** — estimate the effective background luminance
  after shadow and choose text color against that instead of the raw paper color.
