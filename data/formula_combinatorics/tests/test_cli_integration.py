"""Integration tests for the formula-generate CLI (generate.py)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from formula_combinatorics.domains import GENERATORS
from formula_combinatorics.generate import main


def _run(argv: list[str]) -> None:
    with patch("sys.argv", argv):
        main()


# ---------------------------------------------------------------------------
# Basic invocation
# ---------------------------------------------------------------------------


def test_basic_invocation_creates_valid_json(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(["formula-generate", "--output", str(out), "--count", "50", "--seed", "0"])
    assert out.exists()
    with open(out) as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_output_keys_are_sequential_string_ints(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(["formula-generate", "--output", str(out), "--count", "30", "--seed", "1"])
    data = json.load(open(out))
    assert set(data.keys()) == {str(i) for i in range(30)}


def test_output_values_are_nonempty_strings(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(["formula-generate", "--output", str(out), "--count", "20", "--seed", "2"])
    data = json.load(open(out))
    for k, v in data.items():
        assert isinstance(v, str) and len(v) > 0, f"Empty or non-string value at key {k!r}"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_seed_reproducibility(tmp_path: Path) -> None:
    out1 = tmp_path / "run1.json"
    out2 = tmp_path / "run2.json"
    _run(["formula-generate", "--output", str(out1), "--count", "40", "--seed", "7"])
    _run(["formula-generate", "--output", str(out2), "--count", "40", "--seed", "7"])
    assert json.load(open(out1)) == json.load(open(out2))


def test_different_seeds_differ(tmp_path: Path) -> None:
    out1 = tmp_path / "seed0.json"
    out2 = tmp_path / "seed1.json"
    _run(["formula-generate", "--output", str(out1), "--count", "40", "--seed", "0"])
    _run(["formula-generate", "--output", str(out2), "--count", "40", "--seed", "99"])
    assert json.load(open(out1)) != json.load(open(out2))


# ---------------------------------------------------------------------------
# --domains filtering
# ---------------------------------------------------------------------------


def test_domains_flag_restricts_output(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "30",
            "--seed",
            "0",
            "--domains",
            "algebra",
            "calculus",
            "--metadata",
        ]
    )
    data = json.load(open(out))
    used_domains = {v["domain"] for v in data.values()}
    assert used_domains <= {"algebra", "calculus"}


def test_unknown_domain_causes_argparse_exit(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    with pytest.raises(SystemExit) as exc_info:
        _run(
            [
                "formula-generate",
                "--output",
                str(out),
                "--count",
                "10",
                "--domains",
                "algebra",
                "definitely_not_a_real_domain_xyz",
            ]
        )
    assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# --tags and --exclude-tags
# ---------------------------------------------------------------------------


def test_tags_flag_runs_without_error(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(["formula-generate", "--output", str(out), "--count", "20", "--seed", "0", "--tags", "foundational"])
    data = json.load(open(out))
    assert len(data) == 20


def test_exclude_tags_flag_runs_without_error(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "20",
            "--seed",
            "0",
            "--exclude-tags",
            "structural",
        ]
    )
    data = json.load(open(out))
    assert len(data) == 20


# ---------------------------------------------------------------------------
# --metadata flag
# ---------------------------------------------------------------------------


def test_metadata_flag_produces_formula_domain_dicts(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(["formula-generate", "--output", str(out), "--count", "30", "--seed", "0", "--metadata"])
    data = json.load(open(out))
    for k, v in data.items():
        assert isinstance(v, dict), f"Key {k!r}: expected dict, got {type(v)}"
        assert "formula" in v and "domain" in v
        assert isinstance(v["formula"], str) and len(v["formula"]) > 0
        assert v["domain"] in GENERATORS


# ---------------------------------------------------------------------------
# --display-fraction / --inline-fraction
# ---------------------------------------------------------------------------


def test_no_wrapping_when_fractions_zero(tmp_path: Path) -> None:
    """With fractions=0, the wrapping logic must NOT add any display/inline
    delimiters.  Use non-structural domains so no formula is pre-wrapped
    by the domain itself (structural domains like 'align' produce \begin{} natively)."""
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "50",
            "--seed",
            "0",
            "--display-fraction",
            "0.0",
            "--inline-fraction",
            "0.0",
            "--domains",
            "algebra",
            "calculus",
            "trigonometry",
        ]
    )
    data = json.load(open(out))
    for v in data.values():
        assert not v.startswith(r"\["), f"Unexpected display wrap: {v[:30]!r}"
        assert not v.startswith(r"\begin{"), f"Unexpected begin env: {v[:30]!r}"
        assert not v.startswith("$"), f"Unexpected inline wrap: {v[:30]!r}"


def test_display_fraction_one_wraps_all(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "30",
            "--seed",
            "0",
            "--display-fraction",
            "1.0",
            "--inline-fraction",
            "0.0",
        ]
    )
    data = json.load(open(out))
    for v in data.values():
        is_wrapped = v.startswith(r"\[") or v.startswith(r"\begin{")
        assert is_wrapped, f"Expected display wrap but got: {v[:40]!r}"


# ---------------------------------------------------------------------------
# Fraction out-of-range validation → sys.exit(1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--display-fraction", "-0.1"),
        ("--display-fraction", "1.1"),
        ("--inline-fraction", "-0.1"),
        ("--inline-fraction", "1.1"),
    ],
)
def test_fraction_out_of_range_exits_nonzero(flag: str, value: str, tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    with pytest.raises(SystemExit) as exc_info:
        _run(["formula-generate", "--output", str(out), "--count", "10", flag, value])
    assert exc_info.value.code != 0


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--display-fraction", "0.0"),
        ("--display-fraction", "1.0"),
        ("--inline-fraction", "0.0"),
        ("--inline-fraction", "1.0"),
    ],
)
def test_fraction_boundary_values_succeed(flag: str, value: str, tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    # Other fraction must be set to 0 to avoid sum > 1 generating an error or weird behaviour
    other_flag = "--inline-fraction" if flag == "--display-fraction" else "--display-fraction"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "5",
            "--seed",
            "0",
            flag,
            value,
            other_flag,
            "0.0",
        ]
    )
    assert out.exists()


# ---------------------------------------------------------------------------
# --count 0 edge case
# ---------------------------------------------------------------------------


def test_count_zero_produces_empty_output(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(["formula-generate", "--output", str(out), "--count", "0", "--seed", "0"])
    assert out.exists()
    data = json.load(open(out))
    assert isinstance(data, dict)
    assert len(data) == 0


# ---------------------------------------------------------------------------
# Parent directory auto-creation
# ---------------------------------------------------------------------------


def test_output_parent_dirs_autocreated(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "out.json"
    assert not nested.parent.exists()
    _run(["formula-generate", "--output", str(nested), "--count", "5", "--seed", "0"])
    assert nested.exists(), f"Output file not created at {nested}"


# ---------------------------------------------------------------------------
# --output-format jsonl
# ---------------------------------------------------------------------------


def test_output_format_jsonl_creates_jsonl_file(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    _run(["formula-generate", "--output", str(out), "--count", "15", "--seed", "0", "--output-format", "jsonl"])
    assert out.exists()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 15
    for line in lines:
        obj = json.loads(line)
        assert "index" in obj and "formula" in obj
        assert isinstance(obj["formula"], str) and len(obj["formula"]) > 0


def test_output_format_jsonl_extension_inferred(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(["formula-generate", "--output", str(out), "--count", "5", "--seed", "0", "--output-format", "jsonl"])
    # When format is jsonl, the writer uses write_jsonl regardless of the provided extension
    assert out.exists()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for line in lines:
        json.loads(line)  # must not raise


# ---------------------------------------------------------------------------
# --include-draws
# ---------------------------------------------------------------------------


def test_include_draws_attaches_draw_dict(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "10",
            "--seed",
            "0",
            "--metadata",
            "--include-draws",
        ]
    )
    data = json.load(open(out))
    for record in data.values():
        assert "draws" in record, "Expected 'draws' key in record"
        assert isinstance(record["draws"], dict)


def test_metadata_without_include_draws_omits_draws(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(["formula-generate", "--output", str(out), "--count", "10", "--seed", "0", "--metadata"])
    data = json.load(open(out))
    for record in data.values():
        assert "draws" not in record or record["draws"] is None


# ---------------------------------------------------------------------------
# --weight / --weights
# ---------------------------------------------------------------------------


def test_weight_inline_skews_domain_distribution(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "200",
            "--seed",
            "0",
            "--metadata",
            "--weight",
            "algebra=100.0",
            "--weight",
            "calculus=0.01",
        ]
    )
    data = json.load(open(out))
    domains = [v["domain"] for v in data.values()]
    assert domains.count("algebra") > domains.count("calculus") * 5


def test_weights_file_skews_domain_distribution(tmp_path: Path) -> None:
    weights_file = tmp_path / "weights.json"
    weights_file.write_text(json.dumps({"algebra": 100.0, "calculus": 0.01}))
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "200",
            "--seed",
            "0",
            "--metadata",
            "--weights",
            str(weights_file),
        ]
    )
    data = json.load(open(out))
    domains = [v["domain"] for v in data.values()]
    assert domains.count("algebra") > domains.count("calculus") * 5


def test_weight_inline_overrides_file_weight(tmp_path: Path) -> None:
    weights_file = tmp_path / "weights.json"
    weights_file.write_text(json.dumps({"algebra": 100.0}))
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "100",
            "--seed",
            "0",
            "--domains",
            "algebra",
            "calculus",
            "--metadata",
            "--weights",
            str(weights_file),
            "--weight",
            "algebra=0.01",
            "calculus=100.0",  # nargs="+": both in one invocation
        ]
    )
    data = json.load(open(out))
    domains = [v["domain"] for v in data.values()]
    # Inline --weight algebra=0.01 overrides the file's algebra=100.0; calculus should dominate
    assert domains.count("calculus") > domains.count("algebra") * 5


def test_weight_missing_equals_exits_nonzero(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    with pytest.raises(SystemExit) as exc_info:
        _run(["formula-generate", "--output", str(out), "--count", "5", "--weight", "badformat"])
    assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# --hold-out-templates
# ---------------------------------------------------------------------------


def test_hold_out_templates_fraction_creates_both_files(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "60",
            "--seed",
            "0",
            "--hold-out-templates",
            "0.2",
        ]
    )
    assert out.exists()
    held = tmp_path / "out.held_out_templates.json"
    assert held.exists(), "Expected held_out_templates output file"
    manifest = tmp_path / "out.manifest.json"
    assert manifest.exists(), "Expected manifest file"
    manifest_data = json.load(open(manifest))
    assert "held_out_template_names" in manifest_data


def test_hold_out_templates_fraction_out_of_range_exits(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    with pytest.raises(SystemExit) as exc_info:
        _run(
            [
                "formula-generate",
                "--output",
                str(out),
                "--count",
                "10",
                "--hold-out-templates",
                "1.5",
            ]
        )
    assert exc_info.value.code != 0


@pytest.mark.slow
def test_hold_out_templates_template_names_disjoint(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "200",
            "--seed",
            "42",
            "--hold-out-templates",
            "0.2",
            "--metadata",
        ]
    )
    train_data = json.load(open(out))
    held_data = json.load(open(tmp_path / "out.held_out_templates.json"))
    train_names = {v["template_name"] for v in train_data.values() if v.get("template_name")}
    held_names = {v["template_name"] for v in held_data.values() if v.get("template_name")}
    overlap = train_names & held_names
    assert not overlap, f"Template name contamination: {len(overlap)} names in both partitions"


# ---------------------------------------------------------------------------
# --hold-out-symbols
# ---------------------------------------------------------------------------


def test_hold_out_symbols_fraction_creates_both_files(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "60",
            "--seed",
            "0",
            "--hold-out-symbols",
            "0.15",
        ]
    )
    assert out.exists()
    held = tmp_path / "out.held_out_symbols.json"
    assert held.exists(), "Expected held_out_symbols output file"
    manifest = tmp_path / "out.manifest.json"
    assert manifest.exists(), "Expected manifest file"
    manifest_data = json.load(open(manifest))
    assert "held_out_symbols" in manifest_data
    assert isinstance(manifest_data["held_out_symbols"], list)


# ---------------------------------------------------------------------------
# --manifest (custom path)
# ---------------------------------------------------------------------------


def test_manifest_custom_path_used(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    custom_manifest = tmp_path / "my_manifest.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "30",
            "--seed",
            "0",
            "--hold-out-templates",
            "0.1",
            "--manifest",
            str(custom_manifest),
        ]
    )
    assert custom_manifest.exists(), "Expected custom manifest path to be used"
    default_manifest = tmp_path / "out.manifest.json"
    assert not default_manifest.exists(), "Default manifest path should not be created"
    manifest_data = json.load(open(custom_manifest))
    assert "seed" in manifest_data


# ---------------------------------------------------------------------------
# --hold-out-domains (manifest)
# ---------------------------------------------------------------------------


def test_hold_out_domains_excludes_domain_and_writes_manifest(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "50",
            "--seed",
            "0",
            "--hold-out-domains",
            "algebra",
            "--metadata",
        ]
    )
    data = json.load(open(out))
    assert all(v["domain"] != "algebra" for v in data.values()), "algebra should be excluded from train"
    manifest = tmp_path / "out.manifest.json"
    assert manifest.exists()
    manifest_data = json.load(open(manifest))
    assert "algebra" in manifest_data.get("held_out_domains", [])


# ---------------------------------------------------------------------------
# Parameterization flags (CLI smoke tests)
# ---------------------------------------------------------------------------


def test_difficulty_flag_runs_without_error(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "20",
            "--seed",
            "0",
            "--difficulty",
            "elementary",
            "--metadata",
        ]
    )
    data = json.load(open(out))
    assert len(data) > 0


def test_max_depth_flag_runs_without_error(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "20",
            "--seed",
            "0",
            "--max-depth",
            "3",
            "--metadata",
        ]
    )
    data = json.load(open(out))
    assert len(data) > 0


def test_length_range_flag_runs_without_error(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "20",
            "--seed",
            "0",
            "--length-range",
            "5",
            "100",
            "--metadata",
        ]
    )
    data = json.load(open(out))
    assert len(data) > 0


def test_length_range_min_gt_max_exits_nonzero(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    with pytest.raises(SystemExit) as exc_info:
        _run(["formula-generate", "--output", str(out), "--count", "5", "--length-range", "100", "5"])
    assert exc_info.value.code != 0


def test_symbol_tier_head_runs_without_error(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "20",
            "--seed",
            "0",
            "--symbol-tier",
            "head",
            "--metadata",
        ]
    )
    data = json.load(open(out))
    assert len(data) > 0


# ---------------------------------------------------------------------------
# --content-hash (early exit)
# ---------------------------------------------------------------------------


def test_content_hash_exits_zero_and_prints_aggregate(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _run(["formula-generate", "--content-hash"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "aggregate:" in captured.out


# ---------------------------------------------------------------------------
# --coverage-mode (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_coverage_mode_cli_runs_without_error(tmp_path: Path) -> None:
    out = tmp_path / "out.json"
    _run(
        [
            "formula-generate",
            "--output",
            str(out),
            "--count",
            "100",
            "--seed",
            "0",
            "--coverage-mode",
            "2",
            "--domains",
            "algebra",
            "calculus",
        ]
    )
    data = json.load(open(out))
    assert len(data) > 0
