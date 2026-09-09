"""Tests for document.py — slow tests require lualatex + poppler."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_minimal_context() -> dict:
    from content import SyntheticContent
    from layout import LayoutConfig

    rng = random.Random(42)
    layout = LayoutConfig.sample(rng)
    content = SyntheticContent.generate(layout, rng)
    return content.to_template_context()


class TestJinja2Rendering:
    def test_renders_to_tex_string(self):
        from document import _JINJA_ENV

        ctx = _make_minimal_context()
        template = _JINJA_ENV.get_template("document.tex.j2")
        tex = template.render(**ctx)
        assert r"\documentclass" in tex
        assert r"\begin{document}" in tex
        assert r"\end{document}" in tex

    def test_two_col_renders_multicols(self):
        from document import _JINJA_ENV

        ctx = _make_minimal_context()
        ctx["n_cols"] = 2
        template = _JINJA_ENV.get_template("document.tex.j2")
        tex = template.render(**ctx)
        assert r"\begin{multicols}" in tex

    def test_one_col_no_multicols(self):
        from document import _JINJA_ENV

        ctx = _make_minimal_context()
        ctx["n_cols"] = 1
        template = _JINJA_ENV.get_template("document.tex.j2")
        tex = template.render(**ctx)
        assert r"\begin{multicols}" not in tex

    def test_abstract_block_present_when_requested(self):
        from document import _JINJA_ENV

        ctx = _make_minimal_context()
        ctx["has_abstract"] = True
        ctx["abstract"] = "This is the abstract."
        template = _JINJA_ENV.get_template("document.tex.j2")
        tex = template.render(**ctx)
        assert "Abstract." in tex

    def test_no_abstract_block_when_not_requested(self):
        from document import _JINJA_ENV

        ctx = _make_minimal_context()
        ctx["has_abstract"] = False
        ctx["abstract"] = ""
        template = _JINJA_ENV.get_template("document.tex.j2")
        tex = template.render(**ctx)
        assert "textbf{Abstract." not in tex


@pytest.mark.slow
class TestLualatexCompilation:
    def test_compiles_minimal_document(self, tmp_path):
        import shutil

        if not shutil.which("lualatex"):
            pytest.skip("lualatex not available")

        from document import LaTeXDocument

        doc = LaTeXDocument()
        ctx = _make_minimal_context()
        ctx["n_cols"] = 1  # simpler for smoke test
        page_images = doc.generate(ctx, dpi=72)
        assert len(page_images.images) >= 1
        assert page_images.images[0].width > 0
        assert page_images.images[0].height > 0

    def test_rasterizes_to_pil_images(self, tmp_path):
        import shutil

        from PIL import Image

        if not shutil.which("lualatex"):
            pytest.skip("lualatex not available")

        from document import LaTeXDocument

        doc = LaTeXDocument()
        ctx = _make_minimal_context()
        page_images = doc.generate(ctx, dpi=72)
        for img in page_images.images:
            assert isinstance(img, Image.Image)
            assert img.mode == "RGB"
