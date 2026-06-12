"""Render-gate: compile and render LaTeX formulas, keeping only those that succeed.

Engines
-------
katex      Fast pre-filter via Node.js + KaTeX (subset of LaTeX).
tex        Authoritative renderer via lualatex + pdftoppm (full LaTeX, produces PNG).
two-stage  KaTeX pre-filter followed by lualatex certify on survivors (default).

Public API
----------
render_corpus(formulas, output_dir, ...)  -> (filtered_formulas, RenderReport)

System requirements (engine-dependent)
---------------------------------------
katex / two-stage:  node on PATH; katex npm package installed
tex / two-stage:    lualatex on PATH; pdftoppm on PATH (poppler-utils)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

_KATEX_RUNNER = Path(__file__).with_name("_katex_runner.js")

# Minimal standalone LaTeX document for a display-mode formula.
_TEX_TEMPLATE = r"""\documentclass{{article}}
\usepackage[margin=4pt,paperwidth=30cm,paperheight=6cm]{{geometry}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{mathtools}}
\pagestyle{{empty}}
\begin{{document}}
$\displaystyle {formula}$
\end{{document}}
"""


@dataclass
class RenderResult:
    ok: bool
    engine: str
    formula: str
    image_path: Path | None = None
    error: str | None = None
    domain: str | None = None
    template_name: str | None = None


@dataclass
class RenderReport:
    """Aggregated render-gate statistics."""

    # per_template[domain][template_name] = {"ok": int, "fail": int}
    per_template: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)
    total_ok: int = 0
    total_fail: int = 0
    reject_log_path: Path | None = None

    def record(self, result: RenderResult) -> None:
        domain = result.domain or "<unknown>"
        tname = result.template_name or "<unknown>"
        bucket = self.per_template.setdefault(domain, {}).setdefault(tname, {"ok": 0, "fail": 0})
        if result.ok:
            bucket["ok"] += 1
            self.total_ok += 1
        else:
            bucket["fail"] += 1
            self.total_fail += 1

    def success_rate(self) -> float:
        total = self.total_ok + self.total_fail
        return self.total_ok / total if total else 1.0

    def domain_rates(self) -> dict[str, float]:
        rates: dict[str, float] = {}
        for domain, templates in self.per_template.items():
            ok = sum(v["ok"] for v in templates.values())
            fail = sum(v["fail"] for v in templates.values())
            total = ok + fail
            rates[domain] = ok / total if total else 1.0
        return rates


# ---------------------------------------------------------------------------
# KaTeX renderer (fast pre-filter)
# ---------------------------------------------------------------------------


def _check_node(node_bin: str) -> None:
    if not shutil.which(node_bin):
        raise RuntimeError(f"'{node_bin}' not found on PATH. Install Node.js to use the KaTeX render engine.")


def _check_katex(node_bin: str) -> bool:
    """Return True if the katex npm package is resolvable from the runner's directory."""
    check_script = "try{require('katex');process.exit(0);}catch(e){process.exit(1);}"
    result = subprocess.run(
        [node_bin, "-e", check_script],
        capture_output=True,
        cwd=str(_KATEX_RUNNER.parent),
    )
    return result.returncode == 0


class _KatexRenderer:
    """Validate formulas in batches via a long-lived Node.js subprocess."""

    def __init__(self, node_bin: str = "node", batch_size: int = 500) -> None:
        _check_node(node_bin)
        if not _check_katex(node_bin):
            raise RuntimeError(
                f"katex npm package not found. Install it with: npm install katex\n(Run from: {_KATEX_RUNNER.parent})"
            )
        self._node_bin = node_bin
        self._batch_size = batch_size

    def validate_batch(self, formulas: list[str]) -> list[tuple[bool, str | None]]:
        """Return (ok, error_or_None) for each formula in *formulas*."""
        if not formulas:
            return []

        results: list[tuple[bool, str | None]] = [None] * len(formulas)  # type: ignore[list-item]
        for batch_start in range(0, len(formulas), self._batch_size):
            batch = formulas[batch_start : batch_start + self._batch_size]
            lines = [json.dumps({"idx": str(batch_start + i), "formula": f}) for i, f in enumerate(batch)]
            payload = "\n".join(lines) + "\n"
            proc = subprocess.run(
                [self._node_bin, str(_KATEX_RUNNER)],
                input=payload,
                capture_output=True,
                text=True,
                cwd=str(_KATEX_RUNNER.parent),
            )
            for out_line in proc.stdout.splitlines():
                out_line = out_line.strip()
                if not out_line:
                    continue
                try:
                    obj = json.loads(out_line)
                    idx = int(obj["idx"])
                    if obj.get("ok"):
                        results[idx] = (True, None)
                    else:
                        results[idx] = (False, obj.get("error", "katex error"))
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
            # Fill any missing (e.g. if node crashed mid-batch)
            for i in range(len(batch)):
                if results[batch_start + i] is None:
                    results[batch_start + i] = (False, "no response from katex runner")

        return results


# ---------------------------------------------------------------------------
# TeX renderer (authoritative, produces PNG)
# ---------------------------------------------------------------------------


def _check_lualatex(tex_bin: str) -> None:
    if not shutil.which(tex_bin):
        raise RuntimeError(f"'{tex_bin}' not found on PATH. Install TeX Live or MiKTeX to use the TeX render engine.")


def _check_pdftoppm() -> str:
    """Return path to pdftoppm, or raise if not found."""
    path = shutil.which("pdftoppm")
    if path:
        return path
    # Try ImageMagick as fallback
    for alt in ("magick", "convert"):
        if shutil.which(alt):
            return alt
    raise RuntimeError("pdftoppm (poppler-utils) not found on PATH. Install poppler-utils to convert PDF to PNG.")


def _formula_hash(formula: str) -> str:
    return hashlib.sha256(formula.encode()).hexdigest()[:16]


def _strip_display_delimiters(formula: str) -> str:
    """Strip outer display-math delimiters for standalone rendering."""
    formula = formula.strip()
    for start, end in [
        (r"\[", r"\]"),
        (r"\begin{equation}", r"\end{equation}"),
    ]:
        if formula.startswith(start) and formula.endswith(end):
            formula = formula[len(start) : len(formula) - len(end)].strip()
    if formula.startswith("$") and formula.endswith("$") and len(formula) > 1:
        formula = formula[1:-1].strip()
    # Remove \tag{...} that standalone can't handle in displaystyle
    formula = re.sub(r"\\tag\{[^}]*\}", "", formula).strip()
    return formula


class _TexRenderer:
    """Render formulas to PNG via lualatex + pdftoppm."""

    def __init__(
        self,
        tex_bin: str = "lualatex",
        ofl_font_dir: Path | None = None,
        dpi: int = 150,
        workers: int = 4,
    ) -> None:
        _check_lualatex(tex_bin)
        self._tex_bin = tex_bin
        self._ofl_font_dir = ofl_font_dir
        self._dpi = dpi
        self._workers = workers
        self._pdftoppm = _check_pdftoppm()

    def _render_one(self, formula: str, output_dir: Path) -> tuple[bool, Path | None, str | None]:
        """Compile one formula.  Returns (ok, png_path, error)."""
        raw = _strip_display_delimiters(formula)
        tex_src = _TEX_TEMPLATE.format(formula=raw)
        fhash = _formula_hash(formula)

        with tempfile.TemporaryDirectory(prefix="fcrender_") as tmpdir:
            tex_file = Path(tmpdir) / "formula.tex"
            tex_file.write_text(tex_src, encoding="utf-8")

            env = os.environ.copy()
            # Font sandbox: restrict OS font discovery to OFL dir only (or empty)
            env["OSFONTDIR"] = str(self._ofl_font_dir) if self._ofl_font_dir else ""
            env["TEXMFVAR"] = tmpdir
            env["TEXMFCONFIG"] = tmpdir

            proc = subprocess.run(
                [
                    self._tex_bin,
                    "--interaction=nonstopmode",
                    "--halt-on-error",
                    f"--output-directory={tmpdir}",
                    str(tex_file),
                ],
                capture_output=True,
                text=True,
                env=env,
                cwd=tmpdir,
            )

            if proc.returncode != 0:
                # Extract first error line from log
                error_lines = [line for line in proc.stdout.splitlines() if line.startswith("!")]
                error = error_lines[0] if error_lines else (proc.stdout[-300:] or proc.stderr[-300:])
                return False, None, error.strip()

            pdf_path = Path(tmpdir) / "formula.pdf"
            if not pdf_path.exists():
                return False, None, "lualatex produced no PDF"

            png_stem = output_dir / fhash
            pdftoppm_bin = self._pdftoppm

            if pdftoppm_bin.endswith("pdftoppm"):
                conv = subprocess.run(
                    [
                        pdftoppm_bin,
                        "-r",
                        str(self._dpi),
                        "-png",
                        "-singlefile",
                        str(pdf_path),
                        str(png_stem),
                    ],
                    capture_output=True,
                )
                png_path = png_stem.with_suffix(".png")
            else:
                # ImageMagick fallback
                png_path = png_stem.with_suffix(".png")
                conv = subprocess.run(
                    [
                        pdftoppm_bin,
                        "-density",
                        str(self._dpi),
                        str(pdf_path),
                        str(png_path),
                    ],
                    capture_output=True,
                )

            if conv.returncode != 0 or not png_path.exists():
                return False, None, f"PDF→PNG conversion failed (exit {conv.returncode})"

            return True, png_path, None

    def render_batch(
        self,
        items: list[tuple[str, str | None, str | None]],
        output_dir: Path,
    ) -> list[RenderResult]:
        """Render a list of (formula, domain, template_name) tuples in parallel."""
        output_dir.mkdir(parents=True, exist_ok=True)

        def _task(item: tuple[str, str | None, str | None]) -> RenderResult:
            formula, domain, tname = item
            ok, img, err = self._render_one(formula, output_dir)
            return RenderResult(
                ok=ok,
                engine="tex",
                formula=formula,
                image_path=img,
                error=err,
                domain=domain,
                template_name=tname,
            )

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futures = {pool.submit(_task, item): item for item in items}
            results = []
            for fut in as_completed(futures):
                results.append(fut.result())
        return results


# ---------------------------------------------------------------------------
# Two-stage renderer
# ---------------------------------------------------------------------------


class _TwoStageRenderer:
    def __init__(
        self,
        node_bin: str = "node",
        tex_bin: str = "lualatex",
        ofl_font_dir: Path | None = None,
        dpi: int = 150,
        workers: int = 4,
        batch_size: int = 500,
    ) -> None:
        self._katex = _KatexRenderer(node_bin=node_bin, batch_size=batch_size)
        self._tex = _TexRenderer(
            tex_bin=tex_bin,
            ofl_font_dir=ofl_font_dir,
            dpi=dpi,
            workers=workers,
        )

    def render(
        self,
        items: list[tuple[str, str | None, str | None]],
        output_dir: Path,
        reject_sink: list[RenderResult],
    ) -> list[RenderResult]:
        """Run two-stage render.  Katex failures go to reject_sink; TeX survivors are returned."""
        formulas = [it[0] for it in items]

        logger.info("KaTeX pre-filter: validating %d formulas …", len(formulas))
        katex_results = self._katex.validate_batch(formulas)

        katex_pass: list[tuple[str, str | None, str | None]] = []
        for (ok, error), item in zip(katex_results, items):
            formula, domain, tname = item
            if not ok:
                reject_sink.append(
                    RenderResult(
                        ok=False,
                        engine="katex",
                        formula=formula,
                        error=error,
                        domain=domain,
                        template_name=tname,
                    )
                )
            else:
                katex_pass.append(item)

        logger.info(
            "KaTeX: %d pass, %d fail → sending %d to lualatex",
            len(katex_pass),
            len(items) - len(katex_pass),
            len(katex_pass),
        )
        return self._tex.render_batch(katex_pass, output_dir)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_corpus(
    formulas: dict[str, dict],
    output_dir: Path,
    engine: str = "two-stage",
    reject_log: Path | None = None,
    workers: int = 4,
    dpi: int = 150,
    katex_node_bin: str = "node",
    tex_bin: str = "lualatex",
    ofl_font_dir: Path | None = None,
) -> tuple[dict[str, dict], RenderReport]:
    """Run the render gate over a metadata corpus.

    Args:
        formulas: Output of ``generate(include_metadata=True)``.  Each value must
            be a dict with at least a ``"formula"`` key; ``"domain"`` and
            ``"template_name"`` are used for the reject log if present.
        output_dir: Directory where PNG images are written.
        engine: ``"katex"``, ``"tex"``, or ``"two-stage"`` (default).
        reject_log: Path to a JSONL file for failed formulas.  Written only if
            provided; created/overwritten each run.
        workers: Thread-pool size for the TeX stage.
        dpi: PNG resolution for the TeX stage.
        katex_node_bin: Path to the ``node`` executable.
        tex_bin: Path to ``lualatex`` (or ``xelatex``).
        ofl_font_dir: If given, only this directory is searched for OS fonts
            during lualatex runs (font sandbox).

    Returns:
        ``(filtered_formulas, report)`` — filtered_formulas contains only the
        entries that rendered successfully, with ``"image_path"`` added to each
        metadata dict.  ``report`` carries per-template statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items: list[tuple[str, str | None, str | None]] = []
    keys: list[str] = []
    for key, val in formulas.items():
        if not isinstance(val, dict):
            raise TypeError(f"render_corpus requires include_metadata=True output; got {type(val)} for key {key!r}")
        keys.append(key)
        items.append((val["formula"], val.get("domain"), val.get("template_name")))

    rejects: list[RenderResult] = []
    successes: list[RenderResult] = []

    if engine == "katex":
        renderer = _KatexRenderer(node_bin=katex_node_bin)
        raw = renderer.validate_batch([it[0] for it in items])
        for (ok, error), item, key in zip(raw, items, keys):
            formula, domain, tname = item
            r = RenderResult(ok=ok, engine="katex", formula=formula, error=error, domain=domain, template_name=tname)
            (successes if ok else rejects).append(r)

    elif engine == "tex":
        tex_renderer = _TexRenderer(tex_bin=tex_bin, ofl_font_dir=ofl_font_dir, dpi=dpi, workers=workers)
        results = tex_renderer.render_batch(items, output_dir)
        for r in results:
            (successes if r.ok else rejects).append(r)

    elif engine == "two-stage":
        renderer_2s = _TwoStageRenderer(
            node_bin=katex_node_bin,
            tex_bin=tex_bin,
            ofl_font_dir=ofl_font_dir,
            dpi=dpi,
            workers=workers,
        )
        tex_results = renderer_2s.render(items, output_dir, rejects)
        for r in tex_results:
            (successes if r.ok else rejects).append(r)

    else:
        raise ValueError(f"Unknown engine {engine!r}; choose 'katex', 'tex', or 'two-stage'")

    # Build report
    report = RenderReport(reject_log_path=reject_log)
    for r in successes + rejects:
        report.record(r)

    # Write reject log
    if reject_log and rejects:
        reject_log = Path(reject_log)
        reject_log.parent.mkdir(parents=True, exist_ok=True)
        with open(reject_log, "w", encoding="utf-8") as fh:
            for r in rejects:
                fh.write(
                    json.dumps(
                        {
                            "formula": r.formula,
                            "domain": r.domain,
                            "template_name": r.template_name,
                            "engine": r.engine,
                            "error": r.error,
                        }
                    )
                    + "\n"
                )

    # Build filtered output dict (keys preserved in original order)
    success_formulas = {r.formula for r in successes}
    filtered: dict[str, dict] = {}
    new_key = 0
    for key, val in formulas.items():
        formula = val["formula"]
        if formula in success_formulas:
            entry = dict(val)
            # Attach image path from the matching RenderResult
            matching = next((r for r in successes if r.formula == formula), None)
            if matching and matching.image_path:
                entry["image_path"] = str(matching.image_path)
            filtered[str(new_key)] = entry
            new_key += 1

    logger.info(
        "Render gate complete: %d/%d kept (%.1f%%) | engine=%s",
        report.total_ok,
        report.total_ok + report.total_fail,
        report.success_rate() * 100,
        engine,
    )

    for domain, rate in sorted(report.domain_rates().items()):
        logger.info("  %-30s %.1f%%", domain, rate * 100)

    return filtered, report
