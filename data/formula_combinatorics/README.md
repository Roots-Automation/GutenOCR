# formula-combinatorics

Synthetic LaTeX mathematical formula generator for OCR training data. Produces mathematically realistic LaTeX strings across 33 math domains using a declarative template DSL with hand-crafted, domain-specific generators. All output is synthetic — no third-party content is used.

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
| `--output PATH` | *(required)* | Output JSON file path |
| `--count N` | `100000` | Number of unique formulas to generate |
| `--seed INT` | `None` | Random seed for reproducibility |
| `--display-fraction F` | `0.20` | Fraction of bare formulas wrapped in display-math environments (`\[...\]`, `equation`) |
| `--inline-fraction F` | `0.10` | Fraction of bare formulas wrapped in inline `$...$` delimiters |
| `--domains D [D ...]` | all | Restrict to specific domains (see list below) |
| `--tags TAG [TAG ...]` | `None` | Include only domains with any of these tags (`foundational`, `advanced`, `applied`, `structural`) |
| `--exclude-tags TAG [TAG ...]` | `None` | Exclude domains with any of these tags |
| `--metadata` | off | Output `{"formula": ..., "domain": ..., "template_name": ...}` dicts instead of bare strings |

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

33 domains are available, each in its own file. Default sampling weights are unnormalized; `generate()` renormalizes at call time. Weights reflect approximate prevalence in mathematical OCR corpora.

| Domain | File | Weight | Tags | Difficulty |
|---|---|---|---|---|
| `algebra` | `domains/algebra.py` | 0.09 | foundational | elementary |
| `trigonometry` | `domains/trigonometry.py` | 0.04 | foundational | elementary |
| `calculus` | `domains/calculus.py` | 0.10 | foundational | undergraduate |
| `linear_algebra` | `domains/linear_algebra.py` | 0.08 | foundational | undergraduate |
| `geometry` | `domains/geometry.py` | 0.07 | foundational | elementary |
| `probability` | `domains/probability.py` | 0.07 | foundational | undergraduate |
| `physics` | `domains/physics.py` | 0.06 | applied | undergraduate |
| `chemistry` | `domains/chemistry.py` | 0.05 | applied | undergraduate |
| `optimization` | `domains/optimization.py` | 0.05 | applied | undergraduate |
| `set_theory` | `domains/set_theory.py` | 0.05 | foundational | undergraduate |
| `logic` | `domains/logic.py` | 0.05 | foundational | undergraduate |
| `group_theory` | `domains/group_theory.py` | 0.04 | advanced | graduate |
| `analysis` | `domains/analysis.py` | 0.04 | advanced | graduate |
| `number_theory` | `domains/number_theory.py` | 0.04 | advanced | graduate |
| `quantum_notation` | `domains/quantum_notation.py` | 0.04 | applied | graduate |
| `statistics` | `domains/statistics.py` | 0.04 | foundational | undergraduate |
| `topology` | `domains/topology.py` | 0.04 | advanced | graduate |
| `complex_analysis` | `domains/complex_analysis.py` | 0.03 | advanced | graduate |
| `combinatorics` | `domains/combinatorics.py` | 0.03 | foundational | undergraduate |
| `custom_operators` | `domains/custom_operators.py` | 0.03 | structural | graduate |
| `differential_equations` | `domains/differential_equations.py` | 0.03 | foundational | undergraduate |
| `information_theory` | `domains/information_theory.py` | 0.03 | applied | graduate |
| `math_fonts` | `domains/math_fonts.py` | 0.03 | structural | undergraduate |
| `category_theory` | `domains/category_theory.py` | 0.02 | advanced | research |
| `differential_geometry` | `domains/differential_geometry.py` | 0.02 | advanced | graduate |
| `fourier` | `domains/fourier.py` | 0.02 | advanced | graduate |
| `graph_theory` | `domains/graph_theory.py` | 0.02 | advanced | graduate |
| `measure_theory` | `domains/measure_theory.py` | 0.02 | advanced | graduate |
| `p_adic` | `domains/p_adic.py` | 0.02 | advanced | research |
| `stochastic_processes` | `domains/stochastic_processes.py` | 0.02 | advanced | graduate |
| `representation_theory` | `domains/representation_theory.py` | 0.01 | advanced | research |
| `ring_field_theory` | `domains/ring_field_theory.py` | 0.01 | advanced | graduate |
| `align` | `domains/align.py` | 0.15 | structural | undergraduate |

Sampling weights are renormalized automatically when `--domains` restricts the active set, so partial runs produce the correct relative distribution.

The `align` domain (weight 0.15) generates multi-line `align*` and `cases` environments as a first-class domain, not a separate fraction parameter.

---

## Adding a New Domain

1. Create a new file in `formula_combinatorics/domains/`, e.g. `domains/my_domain.py`, following the standard module structure:

```python
# domains/my_domain.py
from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import S, Template
from ._config import register_domain

_MY_TEMPLATES: list[Template] = [
    Template(
        name="example",
        latex=r"\pi_1(S^1) \cong \mathbb{{Z}}",
        slots={},
    ),
    Template(
        name="example_with_slot",
        latex=r"H^n({X}; \mathbb{{Z}}) \cong H_n({X}; \mathbb{{Z}})",
        slots={"X": S(["M", "S", "X"])},
    ),
]

GENERATORS, WEIGHTS, TEMPLATES = register_domain("my_domain", _MY_TEMPLATES)
```

2. Add an entry in `domains/_config.py` under `DOMAIN_CONFIG`:

```python
"my_domain": DomainMeta(weight=0.02, tags=("advanced",), difficulty="graduate"),
```

3. Add `"my_domain"` to `_DOMAIN_NAMES` in `domains/__init__.py`.

The import-time assertion `assert set(DEFAULT_WEIGHTS) == set(GENERATORS)` will catch any mismatch immediately.

### Using shared vocabulary helpers

Pull from `_vocab.py` and `_templates.py` to avoid reinventing common patterns:

```python
from .._vocab import _VARS, _SCALARS, _atom, _expr, _s, _v
from .._templates import _def_integral, _indef_integral, _mixed_partial
```

---

## Module Layout

```
formula_combinatorics/          ← project root
├── pyproject.toml
├── uv.lock
├── README.md
├── CHANGELOG.md
├── tools/                      ← dev utilities (not part of the package)
│   ├── domain_inspector.py     # renders sample formulas to HTML via MathJax
│   ├── collision_probe.py      # birthday-problem diversity analysis
│   └── gen_readme_tables.py    # prints domain table + CLI flags from live registry
└── formula_combinatorics/      ← installable package
    ├── __init__.py             # public API: generate, GENERATORS, DEFAULT_WEIGHTS
    ├── generate.py             # CLI entry point (formula-generate)
    ├── corpus.py               # generation engine: dedup loop
    ├── _vocab.py               # shared constants and atomic sampling helpers
    ├── _templates.py           # shared LaTeX fragment builders
    ├── _template_dsl.py        # Template DSL: Slot, Sub, compute_weights, make_dispatcher
    └── domains/
        ├── __init__.py         # merged GENERATORS, DEFAULT_WEIGHTS, TEMPLATES registry
        ├── _config.py          # per-domain weight, cap, tags, difficulty configuration
        ├── algebra.py
        ├── trigonometry.py
        ├── calculus.py
        ├── linear_algebra.py
        ├── geometry.py
        ├── probability.py
        ├── physics.py
        ├── chemistry.py
        ├── optimization.py
        ├── set_theory.py
        ├── logic.py
        ├── group_theory.py
        ├── analysis.py
        ├── number_theory.py
        ├── quantum_notation.py
        ├── statistics.py
        ├── topology.py
        ├── complex_analysis.py
        ├── combinatorics.py
        ├── custom_operators.py
        ├── differential_equations.py
        ├── information_theory.py
        ├── math_fonts.py
        ├── category_theory.py
        ├── differential_geometry.py
        ├── fourier.py
        ├── graph_theory.py
        ├── measure_theory.py
        ├── p_adic.py
        ├── stochastic_processes.py
        ├── representation_theory.py
        ├── ring_field_theory.py
        └── align.py
```
