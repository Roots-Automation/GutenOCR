# formula-combinatorics

Synthetic LaTeX mathematical formula generator for OCR training data. Produces mathematically realistic LaTeX strings across 24 math domains using a declarative template DSL with hand-crafted, domain-specific generators. All output is synthetic — no third-party content is used.

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
| `--align-fraction F` | `0.15` | Fraction of output using multi-line `align*` environments |
| `--domains D [D ...]` | all | Restrict to specific domains (see list below) |

### Examples

Generate 10,000 formulas from all domains with a fixed seed:
```bash
formula-generate --output train.json --count 10000 --seed 0
```

Generate only calculus and probability formulas:
```bash
formula-generate --output calc_prob.json --count 5000 --domains calculus probability
```

Generate with a higher proportion of multi-line expressions:
```bash
formula-generate --output wide.json --count 10000 --align-fraction 0.30
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
    align_fraction=0.15,
)
# corpus == {"0": "...", "1": "...", ...}
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

24 domains are available, each in its own file. Default sampling weights reflect approximate prevalence in mathematical OCR corpora and sum to 100%.

| Domain | File | Default weight |
|---|---|---|
| `algebra` | `domains/algebra.py` | 9% |
| `trigonometry` | `domains/trigonometry.py` | 4% |
| `calculus` | `domains/calculus.py` | 10% |
| `analysis` | `domains/analysis.py` | 4% |
| `differential_equations` | `domains/differential_equations.py` | 3% |
| `linear_algebra` | `domains/linear_algebra.py` | 8% |
| `probability` | `domains/probability.py` | 9% |
| `information_theory` | `domains/information_theory.py` | 3% |
| `number_theory` | `domains/number_theory.py` | 4% |
| `combinatorics` | `domains/combinatorics.py` | 3% |
| `graph_theory` | `domains/graph_theory.py` | 2% |
| `group_theory` | `domains/group_theory.py` | 4% |
| `ring_field_theory` | `domains/ring_theory.py` | 1% |
| `representation_theory` | `domains/representation.py` | 1% |
| `differential_geometry` | `domains/differential_geometry.py` | 2% |
| `topology` | `domains/topology.py` | 4% |
| `complex_analysis` | `domains/complex_analysis.py` | 3% |
| `fourier` | `domains/fourier.py` | 2% |
| `physics` | `domains/physics.py` | 6% |
| `measure_theory` | `domains/measure_theory.py` | 2% |
| `p_adic` | `domains/p_adic.py` | 2% |
| `optimization` | `domains/optimization.py` | 5% |
| `set_theory` | `domains/set_theory.py` | 5% |
| `logic` | `domains/logic.py` | 4% |

Sampling weights are renormalized automatically when `--domains` restricts the active set, so partial runs produce the correct relative distribution.

An additional `align_fraction` (default 15%) of output uses multi-line `align*` or `cases` environments drawn from a separate pool of structural templates, independent of domain.

---

## Adding a New Domain

1. Create a new file in `formula_combinatorics/domains/`, e.g. `domains/my_domain.py`, following the standard module structure:

```python
# domains/my_domain.py
from __future__ import annotations

import random
from collections.abc import Callable

from .._template_dsl import S, Template, compute_weights, make_dispatcher

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

_W = compute_weights(_MY_TEMPLATES)
_my_domain = make_dispatcher(_MY_TEMPLATES, _W)

GENERATORS: dict[str, Callable[[random.Random], str]] = {"my_domain": _my_domain}
WEIGHTS: dict[str, float] = {"my_domain": 0.02}
TEMPLATES: dict[str, list[Template]] = {"my_domain": _MY_TEMPLATES}
```

2. Import and merge in `domains/__init__.py`:

```python
from .my_domain import GENERATORS as _G_MY, WEIGHTS as _W_MY, TEMPLATES as _T_MY

GENERATORS = {**existing..., **_G_MY}
DEFAULT_WEIGHTS = {**existing..., **_W_MY}
TEMPLATES = {**existing..., **_T_MY}
```

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
├── tools/                      ← dev utilities (not part of the package)
│   ├── domain_inspector.py     # renders sample formulas to HTML via MathJax
│   └── collision_probe.py      # birthday-problem diversity analysis
└── formula_combinatorics/      ← installable package
    ├── __init__.py             # public API: generate, GENERATORS, DEFAULT_WEIGHTS
    ├── generate.py             # CLI entry point (formula-generate)
    ├── corpus.py               # generation engine: dedup loop
    ├── align.py                # multi-line align* / cases environment builder
    ├── _vocab.py               # shared constants and atomic sampling helpers
    ├── _templates.py           # shared LaTeX fragment builders
    ├── _template_dsl.py        # Template DSL: Slot, Sub, compute_weights, make_dispatcher
    └── domains/
        ├── __init__.py         # merged GENERATORS, DEFAULT_WEIGHTS, TEMPLATES registry
        ├── algebra.py
        ├── trigonometry.py
        ├── calculus.py
        ├── analysis.py
        ├── differential_equations.py
        ├── linear_algebra.py
        ├── probability.py
        ├── information_theory.py
        ├── number_theory.py
        ├── combinatorics.py
        ├── graph_theory.py
        ├── group_theory.py
        ├── ring_theory.py
        ├── representation.py
        ├── differential_geometry.py
        ├── topology.py
        ├── complex_analysis.py
        ├── fourier.py
        ├── physics.py
        ├── measure_theory.py
        ├── p_adic.py
        ├── optimization.py
        ├── set_theory.py
        └── logic.py
```
