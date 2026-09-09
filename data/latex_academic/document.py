"""LaTeXDocument: Jinja2 render → lualatex compile → rasterize."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import jinja2
from PIL import Image

_TEMPLATES_DIR = Path(__file__).parent / "resources" / "templates"

# Non-standard Jinja2 delimiters to avoid collision with LaTeX { } and %
_JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
    block_start_string="<<%",
    block_end_string="%>>",
    variable_start_string="<<",
    variable_end_string=">>",
    comment_start_string="<<#",
    comment_end_string="#>>",
    keep_trailing_newline=True,
)


class CompilationError(RuntimeError):
    """Raised when lualatex exits non-zero or emits a fatal error."""

    def __init__(self, message: str, log: str = "") -> None:
        super().__init__(message)
        self.log = log


def _check_dependencies() -> None:
    """Raise RuntimeError early if system tools are missing."""
    missing = []
    if not shutil.which("lualatex"):
        missing.append("lualatex (install texlive-full or use tlmgr)")
    if not shutil.which("pdftoppm"):
        missing.append("pdftoppm / poppler-utils (install poppler-utils)")
    if missing:
        raise RuntimeError("Missing required system tools:\n" + "\n".join(f"  - {m}" for m in missing))


def _parse_log_for_fatal(log_text: str) -> list[str]:
    """Extract lines that begin with '!' (LaTeX fatal errors)."""
    return [line for line in log_text.splitlines() if line.startswith("!")]


@dataclass
class PageImages:
    """Rasterized pages for one compiled PDF.

    The caller is responsible for cleaning up tmpdir after extraction.
    Call cleanup() when the pdf_path is no longer needed.
    """

    images: list[Image.Image]
    pdf_path: Path
    page_width_pts: list[float]
    page_height_pts: list[float]
    _tmpdir: Path | None = None

    def cleanup(self) -> None:
        """Remove the temp directory containing the compiled PDF."""
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None


class LaTeXDocument:
    """Manages the full render → compile → rasterize pipeline for one document."""

    def __init__(self, template_name: str = "document.tex.j2") -> None:
        _check_dependencies()
        self._template = _JINJA_ENV.get_template(template_name)

    def render_tex(self, context: dict, tmpdir: Path) -> Path:
        """Render the Jinja2 template to a .tex file in tmpdir."""
        tex_source = self._template.render(**context)
        tex_path = tmpdir / "document.tex"
        tex_path.write_text(tex_source, encoding="utf-8")
        return tex_path

    def compile(self, tex_path: Path) -> Path:
        r"""Run lualatex twice and return the .pdf path.

        Two passes are needed so \ref / \label cross-references resolve.
        Raises CompilationError on non-zero exit or fatal LaTeX error.
        """
        cmd = [
            "lualatex",
            "--interaction=nonstopmode",
            "--output-directory",
            str(tex_path.parent),
            str(tex_path),
        ]
        for _pass in range(2):
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            log_path = tex_path.with_suffix(".log")
            log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else result.stdout
            fatal_lines = _parse_log_for_fatal(log_text)
            if result.returncode != 0 or fatal_lines:
                summary = "\n".join(fatal_lines[:10]) if fatal_lines else f"exit code {result.returncode}"
                raise CompilationError(f"lualatex failed: {summary}", log=log_text)

        pdf_path = tex_path.with_suffix(".pdf")
        if not pdf_path.exists():
            raise CompilationError("lualatex exited successfully but no PDF was produced.")
        return pdf_path

    def rasterize(self, pdf_path: Path, dpi: int = 150) -> PageImages:
        """Convert PDF pages to PIL Images via pdf2image (poppler).

        thread_count=1 prevents nested parallelism when called from
        within a ProcessPoolExecutor.
        """
        import pdfplumber
        from pdf2image import convert_from_path

        images = convert_from_path(str(pdf_path), dpi=dpi, fmt="JPEG", thread_count=1)
        images = [img.convert("RGB") for img in images]

        # Record per-page PDF dimensions (in points) for bbox scaling.
        widths_pts: list[float] = []
        heights_pts: list[float] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                widths_pts.append(float(page.width))
                heights_pts.append(float(page.height))

        return PageImages(
            images=images,
            pdf_path=pdf_path,
            page_width_pts=widths_pts,
            page_height_pts=heights_pts,
        )

    def generate(self, context: dict, dpi: int = 150) -> PageImages:
        """Full pipeline: Jinja2 → .tex → lualatex (×2) → raster images.

        The caller MUST call result.cleanup() after extraction to remove
        the tmpdir. This allows extract_pages() to read the PDF after
        generate() returns.
        """
        tmpdir = Path(tempfile.mkdtemp(prefix="latexacademic_"))
        try:
            tex_path = self.render_tex(context, tmpdir)
            pdf_path = self.compile(tex_path)
            result = self.rasterize(pdf_path, dpi=dpi)
            result._tmpdir = tmpdir
            return result
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise
