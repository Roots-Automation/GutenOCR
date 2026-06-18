# formula-combinatorics

Synthetic LaTeX mathematical formula generator for OCR training data. Produces mathematically realistic LaTeX strings across 39 math domains using a declarative template DSL with hand-crafted, domain-specific generators. All output is synthetic — no third-party content is used.

Output is a JSON object compatible with `GutenOCR/data/grounded_latex/generate_equations.py`.

---

## Installation

From within this directory:

```bash
uv sync
# or with pip:
pip install -e .
```

This installs the `formula-combinatorics` package and the `formula-generate` CLI command.

The package has no required third-party dependencies — it uses only the stdlib. If `roots-ocr` is already installed in your environment it will be used for richer CLI logging; otherwise a stdlib `basicConfig` fallback is used transparently.

---

## CLI Usage

```bash
formula-generate --output formulas.json --count 50000 --seed 42
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--output PATH` | *(required)* | Output file path (extension determined by `--output-format`) |
| `--output-format FORMAT` | `json` | Output format: `json` (legacy dict) or `jsonl` (one Sample record per line) |
| `--count N` | `100000` | Number of unique formulas to generate |
| `--seed INT` | `None` | Random seed for reproducibility |
| `--display-fraction F` | `0.20` | Fraction of bare formulas wrapped in display-math environments (`\[...\]`, `equation`) |
| `--inline-fraction F` | `0.10` | Fraction of bare formulas wrapped in inline `$...$` delimiters |
| `--domains D [D ...]` | all | Restrict to specific domains (see list below) |
| `--tags TAG [TAG ...]` | `None` | Include only domains with any of these tags (`foundational`, `advanced`, `applied`, `structural`) |
| `--exclude-tags TAG [TAG ...]` | `None` | Exclude domains with any of these tags |
| `--difficulty LEVEL [LEVEL ...]` | `None` | Include only domains at the given difficulty level(s): `elementary`, `undergraduate`, `graduate`, `research` |
| `--max-depth N` | `None` | Keep only templates whose LaTeX brace-nesting depth is ≤ N |
| `--length-range MIN MAX` | `None` | Keep only templates whose character-length proxy falls in `[MIN, MAX]` |
| `--symbol-tier TIER [TIER ...]` | `None` | Keep only templates exercising the named symbol strata (`greek`, `calligraphic`, etc.) or frequency tiers (`head`, `body`, `tail`) |
| `--hold-out-domains DOMAIN [DOMAIN ...]` | `None` | Exclude these domains from generation (domain-level hold-out for split construction) |
| `--hold-out-templates FRAC_OR_NAME [...]` | `None` | Hold out a fraction (e.g. `0.10`) or named list of templates; writes a companion `*.held_out_templates.*` file |
| `--hold-out-symbols FRAC_OR_SYMBOL [...]` | `None` | Hold out a fraction or named list of MUST_COVER symbols; templates exercising any held-out symbol are excluded from train |
| `--include-draws` | off | Attach slot-name→value `draws` dict to each record (increases output size) |
| `--coverage-mode [N]` | off | Keep sampling until every MUST_COVER symbol appears ≥ N times (default N=5) |
| `--manifest PATH` | `<output>.manifest.json` | Path for the split manifest JSON file (written when any hold-out is active) |
| `--weights PATH` | `None` | JSON file mapping domain names to sampling weights, merged over defaults |
| `--weight DOMAIN=VALUE [...]` | `None` | Inline per-domain weight override(s) in `domain=value` format (e.g. `--weight algebra=10.0`) |
| `--metadata` | off | Output rich Sample dicts (formula + domain + template_name + provenance + structural metrics) instead of bare strings |

#### Render gate

| Flag | Default | Description |
|---|---|---|
| `--render` | off | Run the render gate: compile every formula and keep only those that render cleanly |
| `--render-engine ENGINE` | `two-stage` | `katex` (fast pre-filter), `tex` (lualatex, authoritative), or `two-stage` (KaTeX → lualatex) |
| `--render-output DIR` | `<output_stem>_images/` | Directory for rendered PNG images |
| `--render-reject-log PATH` | `<output_stem>_rejects.jsonl` | JSONL file listing failed formulas with `(domain, template_name, engine, error)` |
| `--render-workers N` | `4` | Parallel workers for the TeX stage |
| `--render-dpi N` | `150` | PNG resolution in DPI for the TeX stage |
| `--katex-node-bin PATH` | `node` | Path to the `node` executable |
| `--tex-bin PATH` | `lualatex` | Path to `lualatex` or `xelatex` |
| `--content-hash` | `false` | Print the aggregate SHA-256 content hash of all loaded TOML template packs and exit; useful for pinning corpus snapshots |

**System prerequisites for `--render`:**

- `katex` / `two-stage`: `node` on `PATH`; `katex` npm package installed (`npm install katex` in the package directory)
- `tex` / `two-stage`: `lualatex` on `PATH` (TeX Live or MiKTeX); `pdftoppm` on `PATH` (poppler-utils)

**OFL-only font allowlist** (Latin Modern Math, TeX Gyre, STIX Two, XITS, Asana Math, Fira Math, Libertinus Math, Neo Euler): the TeX stage sets `OSFONTDIR=""` to prevent system font resolution outside the allowlist.

**Example with render gate:**
```bash
formula-generate --output formulas.json --count 10000 --seed 42 \
  --render --render-engine two-stage --render-output images/ \
  --metadata
```
Output: `formulas.json` (certified formulas), `images/<hash>.png` (one per formula), `formulas_rejects.jsonl` (failures).

### Examples

Generate 10,000 formulas from all domains with a fixed seed:
```bash
formula-generate --output train.json --count 10000 --seed 0
```

Generate only calculus and probability formulas:
```bash
formula-generate --output calc_prob.json --count 5000 --domains calculus probability
```

Generate only foundational-tagged domains:
```bash
formula-generate --output foundational.json --count 10000 --tags foundational
```

Generate with a higher proportion of display-math wrapping:
```bash
formula-generate --output wide.json --count 10000 --display-fraction 0.40
```

---

## Output Format

```json
{
  "0": "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}",
  "1": "P(X = k) = \\frac{\\lambda^k e^{-\\lambda}}{k!}",
  "2": "\\begin{align*}f(x) &= \\frac{\\alpha}{\\beta} + x^2 \\\\&= \\sqrt{x + 1}\\end{align*}"
}
```

Keys are sequential string integers starting at `"0"`. Values are valid LaTeX strings, either single-line inline expressions or multi-line `align*` / `cases` environments.

---

## Python API

```python
from formula_combinatorics import generate, GENERATORS, DEFAULT_WEIGHTS

# Generate 1000 formulas across all domains
corpus = generate(
    count=1000,
    domains=list(GENERATORS.keys()),
    generators=GENERATORS,
    weights=DEFAULT_WEIGHTS,
    seed=42,
    display_fraction=0.20,
    inline_fraction=0.10,
)
# corpus == {"0": "...", "1": "...", ...}
```

Pass `strict=True` to raise `RuntimeError` if any domain's error rate exceeds 1% (useful in benchmark build pipelines):

```python
corpus = generate(..., strict=True)
```

### Calling individual domain generators

Each domain generator is a plain function `(rng: random.Random) -> str`:

```python
import random
from formula_combinatorics.domains import GENERATORS

rng = random.Random(0)
formula = GENERATORS["calculus"](rng)
```

---

## Domains

39 domains are available, each as a declarative TOML pack. Default sampling weights are unnormalized; `generate()` renormalizes at call time. Weights reflect approximate prevalence in mathematical OCR corpora.

| Domain | File | Weight | Tags | Difficulty |
|---|---|---|---|---|
| **foundational · elementary** | | | | |
| `algebra` | `domains/algebra.toml` | 0.09 | foundational | elementary |
| `geometry` | `domains/geometry.toml` | 0.07 | foundational | elementary |
| `trigonometry` | `domains/trigonometry.toml` | 0.04 | foundational | elementary |
| **foundational · undergraduate** | | | | |
| `calculus` | `domains/calculus.toml` | 0.10 | foundational | undergraduate |
| `linear_algebra` | `domains/linear_algebra.toml` | 0.08 | foundational | undergraduate |
| `probability` | `domains/probability.toml` | 0.07 | foundational | undergraduate |
| `set_theory` | `domains/set_theory.toml` | 0.05 | foundational | undergraduate |
| `logic` | `domains/logic.toml` | 0.05 | foundational | undergraduate |
| `statistics` | `domains/statistics.toml` | 0.04 | foundational | undergraduate |
| `combinatorics` | `domains/combinatorics.toml` | 0.03 | foundational | undergraduate |
| `differential_equations` | `domains/differential_equations.toml` | 0.03 | foundational | undergraduate |
| **applied · undergraduate** | | | | |
| `chemistry` | `domains/chemistry.toml` | 0.05 | applied | undergraduate |
| `optimization` | `domains/optimization.toml` | 0.05 | applied | undergraduate |
| `classical_mechanics` | `domains/classical_mechanics.toml` | 0.03 | applied | undergraduate |
| `electromagnetism` | `domains/electromagnetism.toml` | 0.02 | applied | undergraduate |
| `statistical_mechanics` | `domains/statistical_mechanics.toml` | 0.02 | applied | undergraduate |
| **structural · undergraduate** | | | | |
| `align` | `domains/align.toml` | 0.15 | structural | undergraduate |
| `math_fonts` | `domains/math_fonts.toml` | 0.03 | structural | undergraduate |
| **applied · graduate** | | | | |
| `quantum_notation` | `domains/quantum_notation.toml` | 0.04 | applied | graduate |
| `asymptotics` | `domains/asymptotics.toml` | 0.03 | applied | graduate |
| `information_theory` | `domains/information_theory.toml` | 0.03 | applied | graduate |
| `quantum_mechanics` | `domains/quantum_mechanics.toml` | 0.02 | applied | graduate |
| **advanced · graduate** | | | | |
| `analysis` | `domains/analysis.toml` | 0.04 | advanced | graduate |
| `group_theory` | `domains/group_theory.toml` | 0.04 | advanced | graduate |
| `number_theory` | `domains/number_theory.toml` | 0.04 | advanced | graduate |
| `topology` | `domains/topology.toml` | 0.04 | advanced | graduate |
| `complex_analysis` | `domains/complex_analysis.toml` | 0.03 | advanced | graduate |
| `proof_theory` | `domains/proof_theory.toml` | 0.02 | advanced | graduate |
| `differential_geometry` | `domains/differential_geometry.toml` | 0.02 | advanced | graduate |
| `fourier` | `domains/fourier.toml` | 0.02 | advanced | graduate |
| `graph_theory` | `domains/graph_theory.toml` | 0.02 | advanced | graduate |
| `measure_theory` | `domains/measure_theory.toml` | 0.02 | advanced | graduate |
| `stochastic_processes` | `domains/stochastic_processes.toml` | 0.02 | advanced | graduate |
| `ring_field_theory` | `domains/ring_field_theory.toml` | 0.01 | advanced | graduate |
| **structural · graduate** | | | | |
| `custom_operators` | `domains/custom_operators.toml` | 0.03 | structural | graduate |
| **advanced · research** | | | | |
| `category_theory` | `domains/category_theory.toml` | 0.02 | advanced | research |
| `p_adic` | `domains/p_adic.toml` | 0.02 | advanced | research |
| `field_theory` | `domains/field_theory.toml` | 0.01 | advanced | research |
| `representation_theory` | `domains/representation_theory.toml` | 0.01 | advanced | research |

Sampling weights are renormalized automatically when `--domains` restricts the active set, so partial runs produce the correct relative distribution.

The `align` domain (weight 0.15) generates multi-line `align*` and `cases` environments as a first-class domain, not a separate fraction parameter.

---

## Adding a New Domain

The canonical path is TOML-first. A domain pack is a single `.toml` file; the engine's pack loader wires it to `GENERATORS`, `DEFAULT_WEIGHTS`, and `TEMPLATES` automatically.

### 1. Create a TOML pack

Create `formula_combinatorics/domains/my_domain.toml`:

```toml
[[templates]]
name = "example"
latex = "\\pi_1(S^1) \\cong \\mathbb{Z}"

[[templates]]
name = "example_with_slot"
latex = "H^n({X}; \\mathbb{Z}) \\cong H_n({X}; \\mathbb{Z})"

[[templates.slots]]
name = "X"
choices = ["M", "S", "X"]
```

See `engine/_pack_loader.py` for the full TOML schema (slots, sub-templates, weights, strata tags).

### 2. Register in `_config.py`

Add an entry under `DOMAIN_CONFIG` in `formula_combinatorics/domains/_config.py`:

```python
"my_domain": DomainMeta(weight=0.02, tags=("advanced",), difficulty="graduate"),
```

No `__init__.py` edit required — the pack loader discovers the `.toml` file by matching the domain name.

### 3. Verify

```bash
uv run formula-generate --domains my_domain --count 10 --metadata
```

The import-time assertion `assert set(DEFAULT_WEIGHTS) == set(GENERATORS)` will catch any mismatch immediately.

### Python domain authoring (legacy)

The Python-module path still works for domains that need imperative logic beyond the TOML DSL. See `engine/_template_dsl.py` for the `Template`, `Slot`, `Sub`, and `ParamSub` primitives. New domains should prefer TOML.

---

## Module Layout

```
formula_combinatorics/          ← project root
├── pyproject.toml
├── uv.lock
├── README.md
├── CHANGELOG.md
├── tools/                      ← dev utilities (not part of the package)
│   ├── domain_inspector.py     # renders sample formulas to self-contained MathJax HTML
│   ├── coverage_report.py      # symbol coverage analysis (by-symbol, by-stratum, by-domain)
│   ├── collision_probe.py      # birthday-problem diversity analysis and n_eff inspection
│   └── gen_readme_tables.py    # regenerates domain table from live registry
└── formula_combinatorics/      ← installable package
    ├── __init__.py             # public API: generate, GENERATORS, DEFAULT_WEIGHTS
    ├── generate.py             # CLI entry point (formula-generate)
    ├── corpus.py               # generation engine: dedup loop, split helpers, JSONL writer
    ├── engine/                 # core DSL, coverage, render, and sample infrastructure
    │   ├── _template_dsl.py    # Template DSL: Slot, Sub, ParamSub, n_eff, make_dispatcher
    │   ├── _coverage.py        # CoverageReport, measure_coverage, measure_coverage_by_domain
    │   ├── _sample.py          # Sample TypedDict (22 fields), make_semantic_key
    │   ├── _pack_loader.py     # TOML pack loader: PackMeta, load_pack
    │   ├── _calibration.py     # n_eff calibration probes
    │   ├── _vocab.py           # shared constants and atomic sampling helpers
    │   ├── _templates.py       # shared LaTeX fragment builders
    │   ├── render.py           # KaTeX + lualatex render gate
    │   └── symbol_inventory.py # MUST_COVER, SYMBOL_STRATA, SYMBOL_FREQUENCY_TIERS
    └── domains/
        ├── __init__.py         # merged GENERATORS, DEFAULT_WEIGHTS, TEMPLATES registry
        ├── _config.py          # per-domain weight, tags, difficulty configuration
        └── *.toml              # 39 domain packs (one per domain, all declarative TOML)
```

---

## Dev Tools

All tools live in `tools/` and are run directly with `uv run python tools/<name>.py`.
They are not part of the installable package.

### domain_inspector.py — visual formula browser

Samples N formulas per domain and renders them to a self-contained MathJax HTML file.
Use this to spot diversity gaps, rendering issues, and notation problems.

```bash
uv run python tools/domain_inspector.py                        # 8 samples/domain, opens browser
uv run python tools/domain_inspector.py --samples 20           # more samples
uv run python tools/domain_inspector.py --domains geometry calculus
uv run python tools/domain_inspector.py --seed 42 --no-open   # reproducible, don't open browser
uv run python tools/domain_inspector.py --output report.html
```

### collision_probe.py — diversity / birthday-horizon analysis

Samples from a domain until a repeated formula is found. The sample count at first
collision is the "birthday horizon" — a proxy for output diversity. High counts (or no
collision within the cap) indicate a rich combinatorial space.

```bash
uv run python tools/collision_probe.py                         # all domains, cap=50k
uv run python tools/collision_probe.py --domains algebra --max-samples 200000
uv run python tools/collision_probe.py --trials 20             # distribution over 20 seeds
uv run python tools/collision_probe.py --show-collision        # print the duplicate pair
uv run python tools/collision_probe.py --batch 10000           # % unique in a fixed batch
uv run python tools/collision_probe.py --n-eff                 # analytical n_eff per template (no sampling)
```

### coverage_report.py — symbol coverage analysis

Generates a corpus and reports which `MUST_COVER` symbols appear, at what frequency,
optionally broken down by stratum or domain.

```bash
uv run python tools/coverage_report.py                         # n=5000, flat symbol table
uv run python tools/coverage_report.py --n 20000 --seed 42
uv run python tools/coverage_report.py --by-stratum            # greek / calligraphic / … breakdown
uv run python tools/coverage_report.py --by-domain             # per-domain coverage fraction
uv run python tools/coverage_report.py --by-stratum --by-domain
```

### gen_readme_tables.py — domain table + CLI flag reference

Prints the domain table (from the live registry) and `formula-generate --help` output.
Run this after adding a domain or changing weights/tags to get copy-pasteable README content.

```bash
uv run python tools/gen_readme_tables.py
```
