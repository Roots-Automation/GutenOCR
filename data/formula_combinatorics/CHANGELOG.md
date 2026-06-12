# Changelog

## 0.2.0 (2026-06-12)

### Changed
- `align` is now a first-class domain (weight 0.15) rather than a separate fraction parameter. The `align_fraction` API argument and `--align-fraction` CLI flag have been removed; use `--domains` or `--tags structural` to control structural coverage.
- Domain count expanded from 24 to 33: added `category_theory`, `chemistry`, `quantum_notation`, `statistics`, `stochastic_processes`, `custom_operators`, `math_fonts`, `geometry`, `align`.
- `variants` within a template are now sampled proportional to `sqrt(n_eff)`, consistent with template-level weighted dispatch. Previously variants were sampled uniformly.
- `generate()` now accumulates per-domain attempt and error counts. If any domain reaches ≥1% error rate with ≥10 attempts, a `WARNING` is emitted. Pass `strict=True` to raise `RuntimeError` instead (useful in benchmark pipelines).
- README updated to match the live CLI surface, domain registry, and Python API.

## 0.1.0

Initial release.
