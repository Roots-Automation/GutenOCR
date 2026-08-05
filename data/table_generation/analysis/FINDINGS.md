# Table Generator Distribution Findings

Comparison of the synthetic generator against 8 real TSR datasets using `analysis/compare.py`.
All real dataset stats are cached in `analysis/cache/`. Run `uv run python -m analysis.compare` to reproduce.

## Datasets compared

| Dataset | n | Source | Notes |
|---|---|---|---|
| **synthetic** | 10k | this generator | flavor=semantic, seed=42 |
| FinTabNet | 88k | docling-project/FinTabNet_OTSL | financial tables, base OTSL |
| PubTabNet | 388k | docling-project/PubTabNet_OTSL | scientific tables, base OTSL |
| SynthTabNet | 600k | docling-project/SynthTabNet_OTSL | synthetic, base OTSL |
| PubTables-1M v1.1 | 523k | docling-project/PubTables-1M_OTSL-v1.1 | semantic OTSL (ched/srow) |
| MUSTARD | 1.4k | bevaya/MUSTARD | 13 languages/scripts, test-only |
| MultiHiertt | 32k | bevaya/MultiHiertt | hierarchical financial, HTML→OTSL |
| HiTab | 6k | bevaya/HiTab-StatCan-NSF | gov statistics, HTML→OTSL |
| ENTRANT | 630k | bevaya/ENTRANT | SEC filings, 10 configs capped at 100k rows each |

---

## Key findings

### 1. Tables are too short

The synthetic generator hard-caps at 12 rows and 8 cols. Real datasets have much longer tails:

| | synthetic | FinTabNet | PubTabNet | SynthTabNet | HiTab | ENTRANT |
|---|---|---|---|---|---|---|
| mean rows | **6.98** | 12.07 | 13.61 | 13.47 | 22.49 | 28.23 |
| p90 rows | **11** | 25 | 28 | 22 | 42 | 54 |
| mean cols | 4.99 | 4.37 | 5.14 | 8.43 | 8.87 | 5.46 |

**Recommendation**: raise `max_rows` to 30+, skew the distribution toward the longer tail rather than staying uniform. Column counts are roughly OK for the document-domain datasets (FinTabNet, PubTabNet, ENTRANT); SynthTabNet and HiTab are wider but are less representative targets.

### 2. Span rate is far too high, especially 2D spans

| | synthetic | FinTabNet | PubTabNet | PubTables-1M | MUSTARD | HiTab |
|---|---|---|---|---|---|---|
| any span (%) | **92.9** | 50.1 | 30.5 | 52.2 | 68.6 | 97.3 |
| 2D span (%) | **64.3** | 0.0 | 0.0 | 0.2 | 39.6 | 59.3 |
| ext cells (%) | **24.2** | 3.7 | 2.7 | 7.2 | 11.8 | 14.9 |

The current `span_prob=0.2` per cell, with no cap on rowspan×colspan combinations, produces 2D spans (xcel) at a rate that matches only HiTab (government statistics) and MUSTARD (scene text). For document-domain tables these are extremely rare.

**Recommendation**:
- Lower `span_prob` to ~0.08–0.10
- Either prohibit 2D spans (`rowspan > 1 AND colspan > 1` simultaneously) or drastically reduce their probability

### 3. Semantic token rates are mostly reasonable, but rhed has no real-world evidence

| | synthetic | PubTables-1M v1.1 | HiTab | ENTRANT |
|---|---|---|---|---|
| tables with ched (%) | 69.7 | 92.4 | 100.0 | 100.0 |
| tables with rhed (%) | **34.1** | 0.0 | 0.0 | 0.0 |
| tables with srow (%) | 22.9 | 19.8 | 0.0 | 0.0 |

- **ched**: our 69.7% is directionally right vs. the 92.4% in PubTables-1M. Consider raising `header_prob` slightly.
- **rhed**: no dataset we examined labels a row-header column with a dedicated token. Generating `rhed` at 34% adds a token class with no supervision signal in any available training data. Consider setting `row_header_prob=0` until a dataset with `rhed` annotations is found.
- **srow**: our 22.9% matches PubTables-1M v1.1's 19.8% well.

### 4. Token mix summary

The synthetic generator's token distribution is dominated by span extension tokens relative to real data:

| token | synthetic | FinTabNet | PubTabNet | PubTables-1M |
|---|---|---|---|---|
| fcel | 47.3% | 82.9% | 86.4% | 77.5% |
| ecel | 8.4% | 13.4% | 10.2% | 4.8% |
| lcel | 8.7% | 3.6% | 3.4% | 5.7% |
| ucel | 10.7% | 0.0% | 0.0% | 3.3% |
| xcel | 6.8% | 0.0% | 0.0% | 0.0% |
| ched | 8.8% | — | — | 7.8% |
| rhed | 5.0% | — | — | 0.0% |
| srow | 4.3% | — | — | 1.0% |

`fcel` should be ~75–85% of non-nl tokens in document tables; we're at 47% because spans and semantic tokens crowd it out.

---

## Recommended parameter changes (not yet applied)

```python
generate_table_structure(
    min_rows=2,       max_rows=30,      # was max_rows=12
    min_cols=2,       max_cols=10,      # was max_cols=8
    span_prob=0.08,                     # was 0.20
    header_prob=0.85,                   # was 0.70
    row_header_prob=0.0,                # was 0.35 — no dataset evidence
    section_row_prob=0.20,              # was 0.25, minor adjustment
    # + suppress 2D spans (to implement in generate_table_structure)
)
```

Row distribution should be non-uniform — real tables peak around 5–10 rows and tail off; the synthetic uniform distribution over 2–12 is immediately visible in the line chart.

---

## What was added to support this analysis

- `otsl.py`: `OTSL_VOCAB_BASE` (6 tokens) and `OTSL_VOCAB_SEMANTIC` (9 tokens); `structure_to_otsl(flavor=)` and `validate_otsl(flavor=)`
- `table_structure.py`: `has_row_header`, `section_rows` on `TableStructure`; `row_header_prob`, `section_row_prob` on `generate_table_structure()`
- `generate.py`: `--flavor base|semantic` CLI flag
- `analysis/stats.py`: per-table stat extraction and dataset summarization
- `analysis/compare.py`: streaming loaders + JSON cache for all 8 datasets
- `analysis/html_to_otsl.py`: HTML→OTSL converter (rowspan/colspan-aware) + MUSTARD char remap
