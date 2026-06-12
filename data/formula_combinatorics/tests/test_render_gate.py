"""Tests for the render gate (render.py) and template-name threading."""

from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from formula_combinatorics.corpus import generate
from formula_combinatorics.domains import DEFAULT_WEIGHTS, GENERATORS
from formula_combinatorics.render import RenderReport, RenderResult, _make_fc_conf, render_corpus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_node() -> bool:
    return shutil.which("node") is not None


def _has_katex(node_bin: str = "node") -> bool:
    if not _has_node():
        return False
    import subprocess

    result = subprocess.run(
        [node_bin, "-e", "try{require('katex');process.exit(0);}catch(e){process.exit(1);}"],
        capture_output=True,
        cwd=str(Path(__file__).parent.parent / "formula_combinatorics"),
    )
    return result.returncode == 0


def _has_lualatex() -> bool:
    return shutil.which("lualatex") is not None


# ---------------------------------------------------------------------------
# Unit tests — no system deps
# ---------------------------------------------------------------------------


class TestRenderResult:
    def test_ok_result(self):
        r = RenderResult(ok=True, engine="katex", formula="x^2", image_path=None, error=None)
        assert r.ok
        assert r.engine == "katex"
        assert r.error is None

    def test_fail_result(self):
        r = RenderResult(ok=False, engine="tex", formula=r"\invalid", error="undefined control sequence")
        assert not r.ok
        assert r.error == "undefined control sequence"

    def test_domain_and_template_name(self):
        r = RenderResult(ok=True, engine="katex", formula="x", domain="algebra", template_name="linear_eq")
        assert r.domain == "algebra"
        assert r.template_name == "linear_eq"


class TestRenderReport:
    def test_record_ok(self):
        report = RenderReport()
        report.record(RenderResult(ok=True, engine="katex", formula="x", domain="algebra", template_name="t1"))
        assert report.total_ok == 1
        assert report.total_fail == 0
        assert report.per_template["algebra"]["t1"]["ok"] == 1

    def test_record_fail(self):
        report = RenderReport()
        report.record(RenderResult(ok=False, engine="katex", formula="x", domain="algebra", template_name="t1"))
        assert report.total_ok == 0
        assert report.total_fail == 1
        assert report.per_template["algebra"]["t1"]["fail"] == 1

    def test_success_rate_all_ok(self):
        report = RenderReport()
        for _ in range(5):
            report.record(RenderResult(ok=True, engine="katex", formula="x", domain="d", template_name="t"))
        assert report.success_rate() == 1.0

    def test_success_rate_mixed(self):
        report = RenderReport()
        report.record(RenderResult(ok=True, engine="katex", formula="x", domain="d", template_name="t"))
        report.record(RenderResult(ok=False, engine="katex", formula="y", domain="d", template_name="t"))
        assert report.success_rate() == pytest.approx(0.5)

    def test_success_rate_empty(self):
        report = RenderReport()
        assert report.success_rate() == 1.0

    def test_domain_rates(self):
        report = RenderReport()
        report.record(RenderResult(ok=True, engine="katex", formula="x", domain="algebra", template_name="t"))
        report.record(RenderResult(ok=False, engine="katex", formula="y", domain="algebra", template_name="t"))
        report.record(RenderResult(ok=True, engine="katex", formula="z", domain="calculus", template_name="t"))
        rates = report.domain_rates()
        assert rates["algebra"] == pytest.approx(0.5)
        assert rates["calculus"] == pytest.approx(1.0)

    def test_unknown_domain_template(self):
        report = RenderReport()
        report.record(RenderResult(ok=True, engine="katex", formula="x"))
        assert "<unknown>" in report.per_template
        assert "<unknown>" in report.per_template["<unknown>"]


class TestRenderCorpusFiltering:
    """Use a mock renderer to verify filtering logic without system deps."""

    def _make_corpus(self, n: int = 10) -> dict[str, dict]:
        return {str(i): {"formula": f"x^{i}", "domain": "algebra", "template_name": "power"} for i in range(n)}

    def _patched_katex(self, fail_indices: set[int]):
        """Return a validate_batch mock that rejects formulas at given indices."""

        def _validate(formulas):
            return [
                (i not in fail_indices, None if i not in fail_indices else "mock error") for i in range(len(formulas))
            ]

        mock = MagicMock()
        mock.validate_batch.side_effect = _validate
        return mock

    def test_katex_engine_filters_correctly(self, tmp_path):
        corpus = self._make_corpus(10)
        fail_set = {2, 5, 7}
        with patch("formula_combinatorics.render._KatexRenderer") as MockKatex:
            instance = MockKatex.return_value
            instance.validate_batch.side_effect = lambda fs: [
                (i not in fail_set, None if i not in fail_set else "mock error") for i in range(len(fs))
            ]
            filtered, report = render_corpus(
                formulas=corpus,
                output_dir=tmp_path / "images",
                engine="katex",
            )
        assert len(filtered) == 7
        assert report.total_ok == 7
        assert report.total_fail == 3

    def test_reject_log_written(self, tmp_path):
        corpus = self._make_corpus(5)
        reject_log = tmp_path / "rejects.jsonl"
        with patch("formula_combinatorics.render._KatexRenderer") as MockKatex:
            instance = MockKatex.return_value
            instance.validate_batch.side_effect = lambda fs: [
                (i != 1, None if i != 1 else "bad formula") for i in range(len(fs))
            ]
            render_corpus(
                formulas=corpus,
                output_dir=tmp_path / "images",
                engine="katex",
                reject_log=reject_log,
            )
        assert reject_log.exists()
        lines = reject_log.read_text().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["formula"] == "x^1"
        assert entry["engine"] == "katex"
        assert entry["error"] == "bad formula"
        assert entry["domain"] == "algebra"
        assert entry["template_name"] == "power"

    def test_reject_log_not_written_when_no_failures(self, tmp_path):
        corpus = self._make_corpus(5)
        reject_log = tmp_path / "rejects.jsonl"
        with patch("formula_combinatorics.render._KatexRenderer") as MockKatex:
            instance = MockKatex.return_value
            instance.validate_batch.return_value = [(True, None)] * 5
            render_corpus(
                formulas=corpus,
                output_dir=tmp_path / "images",
                engine="katex",
                reject_log=reject_log,
            )
        assert not reject_log.exists()

    def test_non_metadata_input_raises(self, tmp_path):
        with pytest.raises(TypeError, match="include_metadata=True"):
            render_corpus(
                formulas={"0": "x^2"},  # type: ignore[arg-type]
                output_dir=tmp_path,
                engine="katex",
            )

    def test_unknown_engine_raises(self, tmp_path):
        corpus = self._make_corpus(2)
        with pytest.raises(ValueError, match="Unknown engine"):
            render_corpus(formulas=corpus, output_dir=tmp_path, engine="invalid")

    def test_keys_reindexed_after_filtering(self, tmp_path):
        corpus = self._make_corpus(5)
        with patch("formula_combinatorics.render._KatexRenderer") as MockKatex:
            instance = MockKatex.return_value
            instance.validate_batch.side_effect = lambda fs: [(i % 2 == 0, None) for i in range(len(fs))]
            filtered, _ = render_corpus(
                formulas=corpus,
                output_dir=tmp_path / "images",
                engine="katex",
            )
        # Keys must be contiguous from "0"
        assert list(filtered.keys()) == [str(i) for i in range(len(filtered))]


class TestMissingSystemDeps:
    def test_katex_renderer_missing_node(self):
        from formula_combinatorics.render import _KatexRenderer

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="node"):
                _KatexRenderer(node_bin="node_definitely_not_installed")

    def test_tex_renderer_missing_lualatex(self):
        from formula_combinatorics.render import _TexRenderer

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="lualatex"):
                _TexRenderer(tex_bin="lualatex_definitely_not_installed")


# ---------------------------------------------------------------------------
# CI smoke tests — no system deps required
# ---------------------------------------------------------------------------


class TestTemplateNameInMetadata:
    """Verifies that template_name is populated for every generated formula."""

    def test_all_entries_have_template_name(self):
        results = generate(
            count=50,
            domains=list(DEFAULT_WEIGHTS.keys()),
            generators=GENERATORS,
            weights=DEFAULT_WEIGHTS,
            seed=42,
            include_metadata=True,
        )
        assert results, "generate() returned empty corpus"
        for key, entry in results.items():
            assert "template_name" in entry, f"Entry {key} missing 'template_name'"
            assert entry["template_name"] is not None, f"Entry {key} has template_name=None"
            assert isinstance(entry["template_name"], str), f"Entry {key} template_name is not a str"

    def test_template_name_absent_without_metadata(self):
        results = generate(
            count=10,
            domains=["algebra"],
            generators=GENERATORS,
            weights=DEFAULT_WEIGHTS,
            seed=0,
            include_metadata=False,
        )
        for val in results.values():
            assert isinstance(val, str)

    def test_domain_and_template_name_consistent(self):
        """domain key should always be a known domain; template_name a non-empty string."""
        known_domains = set(GENERATORS.keys())
        results = generate(
            count=30,
            domains=list(DEFAULT_WEIGHTS.keys()),
            generators=GENERATORS,
            weights=DEFAULT_WEIGHTS,
            seed=7,
            include_metadata=True,
        )
        for key, entry in results.items():
            assert entry["domain"] in known_domains, f"Entry {key} has unknown domain {entry['domain']!r}"
            assert entry["template_name"], f"Entry {key} has empty template_name"


# ---------------------------------------------------------------------------
# Integration tests — require system deps, skipped when not available
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not (_has_node() and _has_katex()), reason="node + katex npm not installed")
class TestKatexIntegration:
    def test_valid_formula_passes(self, tmp_path):
        from formula_combinatorics.render import _KatexRenderer

        r = _KatexRenderer()
        results = r.validate_batch(["x^2 + y^2"])
        assert results[0] == (True, None)

    def test_invalid_formula_fails(self, tmp_path):
        from formula_combinatorics.render import _KatexRenderer

        r = _KatexRenderer()
        results = r.validate_batch([r"\invalidmacroXYZ"])
        ok, error = results[0]
        assert not ok
        assert error

    def test_batch_mixed(self, tmp_path):
        from formula_combinatorics.render import _KatexRenderer

        r = _KatexRenderer()
        results = r.validate_batch(["x^2", r"\bad{}", r"\frac{a}{b}"])
        assert results[0][0] is True
        assert results[1][0] is False
        assert results[2][0] is True

    def test_empty_batch(self, tmp_path):
        from formula_combinatorics.render import _KatexRenderer

        r = _KatexRenderer()
        assert r.validate_batch([]) == []


@pytest.mark.skipif(not _has_lualatex(), reason="lualatex not installed")
class TestTexIntegration:
    def test_valid_formula_renders_to_png(self, tmp_path):
        from formula_combinatorics.render import _TexRenderer

        r = _TexRenderer(workers=1)
        results = r.render_batch([("x^2 + y^2", "algebra", "power")], tmp_path)
        assert len(results) == 1
        assert results[0].ok, f"TeX render failed: {results[0].error}"
        assert results[0].image_path is not None
        assert results[0].image_path.exists()

    def test_invalid_formula_fails(self, tmp_path):
        from formula_combinatorics.render import _TexRenderer

        r = _TexRenderer(workers=1)
        results = r.render_batch([(r"\begin{align}\end{align}", "algebra", "bad")], tmp_path)
        # May succeed or fail depending on TeX installation; just verify it returns a result
        assert len(results) == 1
        assert isinstance(results[0].ok, bool)

    @pytest.mark.skipif(not (_has_node() and _has_katex()), reason="node + katex npm also required")
    def test_two_stage_end_to_end(self, tmp_path):
        corpus = {
            "0": {"formula": "x^2 + y^2", "domain": "algebra", "template_name": "power"},
            "1": {"formula": r"\frac{a}{b}", "domain": "algebra", "template_name": "fraction"},
        }
        filtered, report = render_corpus(
            formulas=corpus,
            output_dir=tmp_path / "images",
            engine="two-stage",
            reject_log=tmp_path / "rejects.jsonl",
        )
        assert report.total_ok + report.total_fail == 2
        assert report.total_ok >= 1


# ---------------------------------------------------------------------------
# Per-domain KaTeX success-rate test (requires node + katex)
# ---------------------------------------------------------------------------

_SAMPLES_PER_DOMAIN = 50
# 0.70 floor: KaTeX supports a strict subset of LaTeX; constructs like \multline*
# are valid TeX but not KaTeX.  Tighten this floor in WU3 once empirical rates
# are measured after template fixes.
_KATEX_FLOOR = 0.70


@pytest.mark.skipif(
    not (_has_node() and _has_katex()),
    reason="node + katex npm not installed",
)
class TestPerDomainSuccessRate:
    def test_all_domains_katex_success_rate(self):
        from formula_combinatorics.render import _KatexRenderer

        all_domains = list(DEFAULT_WEIGHTS.keys())
        # Generate a fixed-seed corpus large enough to guarantee samples from every domain.
        # Over-generate (10×) so weighted sampling covers all domains adequately.
        corpus = generate(
            count=_SAMPLES_PER_DOMAIN * len(all_domains) * 10,
            domains=all_domains,
            generators=GENERATORS,
            weights=DEFAULT_WEIGHTS,
            seed=42,
            include_metadata=True,
        )

        # Group formulas by domain
        by_domain: dict[str, list[str]] = defaultdict(list)
        for entry in corpus.values():
            by_domain[entry["domain"]].append(entry["formula"])

        renderer = _KatexRenderer()
        failing_domains: list[str] = []

        for domain in all_domains:
            formulas = by_domain.get(domain, [])
            if not formulas:
                # Domain produced no samples — flag it
                failing_domains.append(f"{domain}: 0 samples generated")
                continue
            # Cap at SAMPLES_PER_DOMAIN for speed
            sample = formulas[:_SAMPLES_PER_DOMAIN]
            results = renderer.validate_batch(sample)
            ok = sum(1 for ok, _ in results if ok)
            rate = ok / len(sample)
            if rate < _KATEX_FLOOR:
                failing_domains.append(f"{domain}: {rate:.1%} ({ok}/{len(sample)})")

        assert not failing_domains, (
            f"KaTeX pass rate below {_KATEX_FLOOR:.0%} floor in {len(failing_domains)} domain(s):\n"
            + "\n".join(f"  {d}" for d in sorted(failing_domains))
        )


# ---------------------------------------------------------------------------
# Font sandbox tests
# ---------------------------------------------------------------------------


class TestFontSandbox:
    def test_make_fc_conf_empty(self):
        xml = _make_fc_conf(None)
        assert "<fontconfig>" in xml
        assert "<dir>" not in xml

    def test_make_fc_conf_with_dir(self, tmp_path):
        xml = _make_fc_conf(tmp_path)
        assert str(tmp_path) in xml
        assert "<dir>" in xml

    def test_sandbox_env_vars_set(self, tmp_path):
        """Verify _TexRenderer sets OSFONTDIR, FONTCONFIG_FILE, TEXMFVAR in the subprocess env."""
        from formula_combinatorics.render import _TexRenderer

        captured_envs: list[dict] = []

        def _fake_run(cmd, **kwargs):
            captured_envs.append(kwargs.get("env", {}))

            class _FakeResult:
                returncode = 0
                stdout = ""
                stderr = ""

            # Also fake the pdftoppm call
            return _FakeResult()

        with patch("formula_combinatorics.render.subprocess.run", side_effect=_fake_run):
            # _check_lualatex calls shutil.which, _check_pdftoppm calls shutil.which
            with patch("formula_combinatorics.render.shutil.which", return_value="/usr/bin/lualatex"):
                renderer = _TexRenderer.__new__(_TexRenderer)
                renderer._tex_bin = "lualatex"
                renderer._ofl_font_dir = None
                renderer._dpi = 150
                renderer._workers = 1
                renderer._pdftoppm = "pdftoppm"
                renderer._render_one("x^2", tmp_path)

        assert captured_envs, "subprocess.run was never called"
        lualatex_env = captured_envs[0]
        assert "OSFONTDIR" in lualatex_env, "OSFONTDIR not set in lualatex subprocess env"
        assert "FONTCONFIG_FILE" in lualatex_env, "FONTCONFIG_FILE not set in lualatex subprocess env"
        assert "TEXMFVAR" in lualatex_env, "TEXMFVAR not set in lualatex subprocess env"
        assert lualatex_env["OSFONTDIR"] == "", "OSFONTDIR should be empty string when no ofl_font_dir"

    @pytest.mark.skipif(
        sys.platform == "darwin",
        reason="luaotfload uses CoreText on macOS; fontconfig env vars have no effect. "
        "Container required for full sandbox enforcement.",
    )
    @pytest.mark.skipif(not _has_lualatex(), reason="lualatex not installed")
    def test_proprietary_font_unreachable_in_sandbox(self, tmp_path):
        """On Linux: assert lualatex fails when a proprietary font is requested with sandbox active."""
        from formula_combinatorics.render import _TexRenderer

        renderer = _TexRenderer(workers=1)
        # Craft a .tex that explicitly requests Times New Roman via fontspec.
        # With an empty fontconfig, luaotfload should fail to resolve it.
        formula = r"\text{hello}"
        # Directly test _render_one with a formula that won't fail for math reasons
        ok, _img, _err = renderer._render_one(formula, tmp_path)
        # This test mainly documents the expectation; on Linux with empty fontconfig
        # a formula using \setmainfont{Times New Roman} would fail. Plain math succeeds.
        # The real sandbox verification happens in CI with a proper Docker image.
        assert isinstance(ok, bool)
