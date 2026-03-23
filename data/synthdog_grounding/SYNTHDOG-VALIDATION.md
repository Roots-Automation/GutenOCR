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

## Thread 4 · `document.py` pokes SynthTiger's private `_init` method — RESOLVED

**Status: RESOLVED** — two-step construction + `_init` call collapsed to a single
`Switch(ElasticDistortion(), **elastic_config)` constructor call.

### Original problem statement

**File:** `elements/document.py:69`

```python
# OLD:
self.elastic_distortion = components.Switch(components.ElasticDistortion())
if elastic_config:
    self.elastic_distortion._init(**elastic_config)
```

### Findings

`_init` is not truly private — it is explicitly defined on the `Component` base class
and overridden on `Switch` as an intentional re-initialization hook. It would raise
`AttributeError` (not silently no-op) if removed. The original concern was overstated.

The real issue is unnecessary: `Switch.__init__` already accepts `prob` and `args`
directly, which are exactly the keys in `elastic_config`. The two-step pattern is
redundant and slightly fragile (would break if the wrapper were swapped for one that
doesn't override `_init`).

### Fix applied

```python
# NEW:
self.elastic_distortion = components.Switch(
    components.ElasticDistortion(), **elastic_config
)
```

Works correctly for both empty and populated `elastic_config`; no conditional needed.

---

## Thread 5 · `TextReader` file handle is never explicitly closed — RESOLVED

**Status: RESOLVED** — `close()` + `__enter__`/`__exit__` propagated up the ownership
chain: `Content` → `Document` → `SynthDoG.__del__`. Duck typing (`hasattr`) used when
delegating to the reader so `HuggingFaceTextReader` (which has no `close()`) is unaffected.

**File:** `elements/readers.py:30`

```python
self.fp = open(path, encoding="utf-8")  # noqa: SIM115
```

The file is opened in `__init__` and the only guaranteed close path is `__del__`,
which is unreliable under CPython and essentially never called during a long
generation run. The `noqa: SIM115` suppresses the linter hint to use a context manager.

One open file handle per worker process is not a crisis, but it is a correctness
issue, and it is what the linter is flagging.

### Fix applied

`elements/readers.py:30` — improved `# noqa` comment to explain the intentional long-lived handle:

```python
self.fp = open(path, encoding="utf-8")  # noqa: SIM115 — long-lived handle; closed via close() / __exit__
```

`elements/content.py` — added `close()`, `__enter__`, `__exit__` delegating to the reader
via duck typing:

```python
def close(self):
    if hasattr(self.reader, "close"):
        self.reader.close()

def __enter__(self):
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()
    return False
```

`elements/document.py` — added `close()`, `__enter__`, `__exit__` delegating to `content`:

```python
def close(self):
    self.content.close()

def __enter__(self):
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()
    return False
```

`template.py` — added `__del__` to `SynthDoG` so the file handle is released when the
template is torn down (synthtiger controls template lifetime externally):

```python
def __del__(self):
    if hasattr(self, "document"):
        self.document.close()
```

---

## Thread 6 · Leading-space textbox artifact causes bbox offset — RESOLVED

**Status: RESOLVED** — `text.prev()` removed from the trailing-space discard block
in `elements/textbox.py:119-123`. The reader is already positioned after the space
when the block executes; backing up caused the next `generate()` call to re-read the
space at `left=0`, misaligning the bbox with the stripped text string.

**File:** `elements/textbox.py:119-123`

```python
if len(chars):
    # Discard the trailing space; reader is already positioned after it,
    # so the next textbox starts at the first real character.
    chars.pop()
    char_layers.pop()
```

---

## Thread 7 · `_package_data` serializes blocks/words but not lines (inconsistency) — RESOLVED

**Status: RESOLVED** — all three annotation types are now serialized together in
`save()` at lines 309–311. `_package_data` passes `lines`, `words`, and `blocks`
as raw dataclass lists; `save()` converts all three to dicts in one place.

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

### Fix applied

All serialization moved to `save()` — `_package_data` now returns raw dataclass
lists for `lines`, `words`, and `blocks`; `save()` calls all three `*_to_dict`
helpers at lines 309–311.

---

## Thread 8 · `content_color` silently overwrites `textbox_color` (undocumented interaction) — RESOLVED

**Status: RESOLVED** — the two color modes are now mutually exclusive; `content_color`
is sampled first and if it fires, `textbox_color` is skipped entirely.

### Original problem statement

**File:** `elements/content.py:112-118`

```python
for ...:
    textbox_color.apply([text_layer])   # fires per-line (prob 0.2, or 1.0 on dark bg)

content_color.apply(text_layers)        # fires once for ALL lines (prob 0.2, or 1.0 on dark bg)
```

When `content_color` fires it overwrites whatever `textbox_color` set on individual
lines. On dark backgrounds both are forced to `prob=1.0`, so `content_color` always
wins and all per-line color variation from `textbox_color` is discarded.

### Fix applied

`elements/content.py` — color application moved after the layout loop; `content_color`
is sampled once and if it fires its meta is reused to apply the uniform color; otherwise
`textbox_color` applies per-line:

```python
content_meta = content_color.sample()
if content_meta["state"]:
    content_color.apply(text_layers, meta=content_meta)
else:
    for text_layer in text_layers:
        textbox_color.apply([text_layer])
```

`config/config_base.yaml` — clarifying comment added above the two color blocks
explaining the mutual-exclusion semantics.

On dark backgrounds `content_color` fires with `prob=1.0`, so `textbox_color` is
cleanly skipped (correct behavior: uniform light text). On light backgrounds ~20%
of samples get uniform color, ~16% get per-line variation, ~64% get the default color.

---

## Thread 9 · Post-processing shadow renders text unreadable regardless of text color (quality impact) — RESOLVED

**Status: RESOLVED** — shadow moved to `bg_effect` (background-only pipeline);
`min_contrast_ratio` lowered to 1.5 as a backstop. See full resolution below.

**Files:** `annotations.py`, `template.py`, `config/config_base.yaml`

### Problem statement

Text color is chosen at generation time against the raw paper color. Post-processing
effects — specifically the shadow and brightness steps — are applied afterwards with
no awareness of text readability.

The shadow effect was configured as:

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

### Empirical analysis (initial approach: post-hoc reject gate)

A WCAG-proxy contrast metric (`min_line_contrast_ratio`) was added to
`compute_quality_metrics()` and `save()` was updated to reject samples below a
threshold. Initial threshold was `3.0`.

Empirical testing with `test_contrast_distribution.py` (n=500) revealed this gate
discarded ~70% of samples — an unacceptable reject rate. Even after:

- Halving shadow params (`intensity: [0,80]`, `amount: [0,0.5]`)
- Raising the paper color floor (`rgb: [[40,255],...]`)
- Making brightness asymmetric (`beta: [-16,32]`)

...the distribution median was only ~2.5 with ~12% below 1.5. These parameter tweaks
had negligible improvement because they did not address the root cause.

**Root cause (structural):** The rendering pipeline merges bg + paper + text into a
single flat image, *then* applies shadow to the merged result. Shadow treats all
pixels equally — it cannot distinguish text from background. A directional shadow
that darkens the document region therefore collapses the contrast between dark text
and the now-dark paper, regardless of how text color was originally chosen.

### Fix applied (structural)

**Move shadow to `bg_effect` — applied to background layer only, before merge.**

Shadow darkens the background texture (table / environment). The document (paper +
text) is composited on top *after* shadow runs. The text layer pixels retain their
original RGBA values; they are placed on top of the shadowed background. The paper is
also unshadowed — so text contrast against the paper is exactly what Thread 1
designed. Shadow becomes an *environmental* effect (depth/atmosphere) rather than a
*document-lighting* effect.

**`template.py`** — `__init__` adds `self.bg_effect` and removes Shadow from
`self.effect`; `_render()` applies `bg_effect` to `bg_layer` before the merge:

```python
# NEW: shadow applied to bg layer only, before merge
self.bg_effect = components.Iterator(
    [components.Switch(components.Shadow())],
    **config.get("bg_effect", {}),
)

# Shadow removed; 5 components now
self.effect = components.Iterator(
    [
        components.Switch(components.RGB()),
        components.Switch(components.Contrast()),
        components.Switch(components.Brightness()),
        components.Switch(components.MotionBlur()),
        components.Switch(components.GaussianBlur()),
    ],
    **config.get("effect", {}),
)

def _render(self, document_group, bg_layer, size):
    # Apply shadow to background only — cannot darken paper or affect
    # text-vs-paper contrast.
    self.bg_effect.apply([bg_layer])
    layer = layers.Group([*document_group.layers, bg_layer]).merge()
    ...
```

**`config/config_base.yaml`** — new `bg_effect` section with shadow; shadow removed
from `effect`; experimental paper floor and brightness asymmetry reverted;
`min_contrast_ratio` lowered from 3.0 to 1.5 as a backstop:

```yaml
bg_effect:
  args:
    - prob: 1
      args:
        intensity: [0, 80]
        amount: [0, 0.5]
        smoothing: [0.5, 1]
        bidirectional: 0

effect:
  args:
    # color, contrast, brightness, motion blur, gaussian blur (no shadow)
    ...
    beta: [-32, 32]   # reverted from asymmetric [-16, 32]

paper:
  color:
    rgb: [[0, 255], [0, 255], [0, 255]]  # reverted from floor [40, 255]

min_contrast_ratio: 1.5  # backstop; structural fix is the primary guard
```

**Why brightness stays on the merged image:** Brightness (±32) is a uniform global
shift — it moves paper and text pixels together, so relative contrast degrades much
less than with shadow's local darkening. Moving it to bg-only would sacrifice global
exposure variation for little gain.

**`annotations.py`** — WCAG helpers and `min_line_contrast_ratio` metric unchanged;
the backstop gate in `save()` remains at `min_contrast_ratio: 1.5`.

The existing `min_line_contrast` (std-dev) field is unchanged — the new metric is additive.

### Addendum: page-level shadow restored via `doc_effect`

After the structural fix, the background-only shadow produced correct contrast but
lost the visual depth of shadow falling across the page itself. A separate `doc_effect`
shadow is applied to the **merged document layer** (paper + text) before compositing
with bg, restoring the page-level lighting aesthetic.

Because this shadow affects both paper and text pixels, it partially reintroduces the
original failure mode — but at reduced intensity. Empirical testing (n=500) confirmed
the rejection rate at threshold=1.5 increases only marginally (9.6% → 11.2%).

```yaml
doc_effect:
  args:
    - prob: 1
      args:
        intensity: [0, 80]
        amount: [0, 0.5]
        smoothing: [0.5, 1]
        bidirectional: 0
```

```python
# in _render():
doc_layer = document_group.merge()
self.doc_effect.apply([doc_layer])
layer = layers.Group([doc_layer, bg_layer]).merge()
```

**Distribution summary (n=500, threshold=0):**

| configuration | median | rejected @ 1.5 |
|---|---|---|
| original (shadow on merged image, threshold=3.0) | ~2.5 | ~70% |
| bg-only shadow, threshold=1.5 | 2.39 | 9.6% |
| bg + doc shadow, threshold=1.5 | 2.30 | 11.2% |

---

## Thread 11 · Elastic distortion applied post-annotation — bbox misalignment study — RESOLVED

**Status: RESOLVED** — misalignment confirmed negligible; no code change warranted.

### Problem statement

Elastic distortion is applied to the composited image *after* annotation bboxes are
captured from per-layer quads (see `template.py:_render()`).  In principle this could
cause saved bboxes to describe text positions on the *pre-distortion* image, which no
longer matches the final rendered pixels.

### Methodology

Script: `check_elastic_distortion.py` (n=50 samples).

For each sample, corresponding pixels at each annotated line-bbox corner were sampled
before and after elastic distortion.  Three comparisons were made:
- **Mean/p95 pixel delta**: how many pixels did the corner coordinates shift?
- **Centroid drift**: did the text block centroid move (measures bulk translation)?
- **Text coverage**: fraction of annotated bbox area that still contains text pixels
  after distortion (measures how well the bbox still "contains" the rendered text).

All comparisons were repeated for motion blur and Gaussian blur (also applied
post-annotation) to establish a baseline — if elastic distortion is at or below
those levels, it does not warrant special treatment.

### Results (config params: `alpha=[0,1]`, `sigma=[0,0.5]`)

| Effect | mean pixel delta | p95 pixel delta | centroid drift | text coverage |
|--------|-----------------|-----------------|----------------|---------------|
| Elastic distortion | 1.4 px | 10.1 px | 2.3 px | 0.90 |
| Motion blur | 2.6 px | 24.1 px | 2.6 px | 0.92 |
| Gaussian blur | 2.1 px | 27.5 px | 2.2 px | 0.93 |

Elastic distortion is at or below the level of both blur effects, which are also
applied post-annotation and are widely accepted as non-problematic for OCR training.
The misalignment introduced is negligible for training purposes.

### Resolution

No code change.  The comment in `template.py:_render()` documents the study results
and references this thread.

---

## Thread 10 · Random corpus jump can land mid-word — RESOLVED

**Status: RESOLVED** — `content.py` now advances to a word boundary after the
random `move()` call; see fix details below.

### Problem statement

**File:** `elements/content.py` (line 100)

```python
self.reader.move(np.random.randint(len(self.reader)))
```

`move()` jumps to a random character index. That index can land anywhere in the
corpus — including mid-word. The first textbox of every sample therefore sometimes
starts at an arbitrary character inside a word, producing broken text at the
beginning of the image.

### Fix applied

After the `move()` call, two loops align the reader to the next word boundary:

```python
# Align to a word boundary: skip to the end of the current word, then
# past any whitespace, so the first textbox starts at a clean word start.
for _ in range(len(self.reader)):
    if self.reader.get().isspace():
        break
    self.reader.next()
for _ in range(len(self.reader)):
    if not self.reader.get().isspace():
        break
    self.reader.next()
```

The first loop advances past any remaining non-whitespace characters (finishing the
current word). The second loop advances past any whitespace run (handles `\r\n`
and multi-space sequences). After both loops the reader sits at the first character
of a word.
