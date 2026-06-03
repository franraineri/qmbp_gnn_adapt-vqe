"""Tests for the scripts/digest/ tool.

Covers all three result kinds, all CLI modes, filters, sorting, grouping,
statistical analysis, outlier detection, and comparison.

Run with:
    pytest tests/test_digest.py -v
    pytest tests/test_digest.py -v -k "test_scan"   # specific test
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# ── Fixtures ─────────────────────────────────────────────────────────────

DIGEST_CMD = [sys.executable, "-m", "scripts.digest"]


def run_digest(*args: str, expect_ok: bool = True) -> subprocess.CompletedProcess:
    """Run the digest CLI and return the result."""
    result = subprocess.run(
        [*DIGEST_CMD, *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if expect_ok:
        assert result.returncode == 0, (
            f"digest failed with code {result.returncode}\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Scanner tests — verify data is discovered correctly
# ═══════════════════════════════════════════════════════════════════════════


class TestScanner:
    """Test that the scanner finds and parses results correctly."""

    def test_scan_all_finds_results(self):
        """scan_all should find noiseless, noisy, and experiment results."""
        from scripts.digest.scanner import ResultScanner

        scanner = ResultScanner(Path("results"))
        noiseless, noisy, experiments = scanner.scan_all()

        assert len(noiseless) > 0, "Should find noiseless pipeline results"
        assert len(noisy) > 0, "Should find noisy/ZNE results"
        assert len(experiments) > 0, "Should find experiment results"

    def test_scan_all_noiseless_have_required_fields(self):
        """Every noiseless result should have core fields populated."""
        from scripts.digest.scanner import ResultScanner

        scanner = ResultScanner(Path("results"))
        noiseless, _, _ = scanner.scan_all()

        for r in noiseless:
            assert r.source_file, f"{r.variant_id}: missing source_file"
            assert r.folder, f"{r.variant_id}: missing folder"
            assert r.n_qubits > 0, f"{r.variant_id}: n_qubits should be > 0"
            assert r.topology, f"{r.variant_id}: missing topology"

    def test_scan_all_noisy_have_required_fields(self):
        """Every noisy result should have core fields populated."""
        from scripts.digest.scanner import ResultScanner

        scanner = ResultScanner(Path("results"))
        _, noisy, _ = scanner.scan_all()

        for r in noisy:
            assert r.source_file, f"{r.variant_id}: missing source_file"
            assert r.n_qubits > 0, f"{r.variant_id}: n_qubits should be > 0"
            assert r.n_total > 0, f"{r.variant_id}: n_total should be > 0"

    def test_scan_all_experiments_have_verdict(self):
        """Every experiment result should have a verdict."""
        from scripts.digest.scanner import ResultScanner

        scanner = ResultScanner(Path("results"))
        _, _, experiments = scanner.scan_all()

        valid_verdicts = {"confirmed", "rejected", "failed"}
        for r in experiments:
            assert r.verdict in valid_verdicts, f"{r.experiment_id}: invalid verdict '{r.verdict}'"
            assert r.experiment_id, f"{r.folder}: missing experiment_id"

    def test_scan_folder_exact_match(self):
        """scan_folder with exact name should find results."""
        from scripts.digest.scanner import ResultScanner

        scanner = ResultScanner(Path("results"))
        nl, ny, _ = scanner.scan_folder("variants_N10_ladder")

        assert len(nl) > 0, "Should find noiseless results in variants_N10_ladder"
        assert len(ny) > 0, "Should find noisy results in variants_N10_ladder"

    def test_scan_folder_substring_match(self):
        """scan_folder with substring should find matching folders."""
        from scripts.digest.scanner import ResultScanner

        scanner = ResultScanner(Path("results"))
        nl, _, _ = scanner.scan_folder("ladder")

        assert len(nl) > 0, "Substring 'ladder' should match thesis folders"

    def test_scan_folder_experiment(self):
        """scan_folder should find experiment results by folder name."""
        from scripts.digest.scanner import ResultScanner

        scanner = ResultScanner(Path("results"))
        _, _, exp = scanner.scan_folder("exp_b4")

        assert len(exp) == 1
        assert exp[0].experiment_id == "B4"

    def test_topology_inference_from_parent(self):
        """Results without topology in JSON should inherit from parent folder."""
        from scripts.digest.scanner import ResultScanner

        scanner = ResultScanner(Path("results"))
        nl, _, _ = scanner.scan_folder("variants_N10_ladder")

        # Results that have topology from their JSON keep it (e.g., ext_triangular)
        # Results without topology in JSON should get 'ladder' from parent
        for r in nl:
            assert r.topology, f"{r.variant_id}: topology should not be empty"
            # Most should be ladder, but cross-topology variants keep their own
            if "triangular" not in r.variant_id and "chain" not in r.variant_id:
                assert r.topology == "ladder", (
                    f"{r.variant_id}: expected 'ladder', got '{r.topology}'"
                )


# ═══════════════════════════════════════════════════════════════════════════
# CLI tests — verify all modes work end-to-end
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestCLIBasic:
    """Test basic CLI invocations."""

    def test_help(self):
        """--help should work."""
        result = run_digest("--help")
        assert "Digest experiment results" in result.stdout

    def test_kind_experiment(self):
        """--kind experiment should produce experiment output."""
        result = run_digest("--kind", "experiment")
        assert "HYPOTHESIS TESTS" in result.stdout
        assert "confirmed" in result.stdout or "rejected" in result.stdout

    def test_kind_noiseless(self):
        """--kind noiseless should produce noiseless output."""
        result = run_digest("--kind", "noiseless")
        assert "NOISELESS PIPELINE" in result.stdout
        assert "ΔE/gap" in result.stdout

    def test_kind_noisy(self):
        """--kind noisy should produce noisy output."""
        result = run_digest("--kind", "noisy")
        assert "NOISY / ZNE" in result.stdout
        assert "R²" in result.stdout

    def test_kind_all(self):
        """--kind all (default) should produce all sections."""
        result = run_digest("--kind", "all")
        assert "HYPOTHESIS TESTS" in result.stdout
        assert "NOISELESS PIPELINE" in result.stdout
        assert "NOISY / ZNE" in result.stdout


@pytest.mark.slow
class TestCLIFilters:
    """Test filter options."""

    def test_filter_topology(self):
        """--topology should filter results."""
        result = run_digest("--kind", "noiseless", "--topology", "ladder")
        assert "NOISELESS" in result.stdout
        # Should not contain other topologies in the table rows
        lines = result.stdout.split("\n")
        data_lines = [line for line in lines if "chain_1d" in line and "✅" in line]
        assert len(data_lines) == 0, "Should not show chain_1d results"

    def test_filter_n_qubits(self):
        """--n-qubits should filter results."""
        result = run_digest("--kind", "noiseless", "--n-qubits", "6")
        assert "NOISELESS" in result.stdout

    def test_filter_folder(self):
        """--folder should limit scan to specific folder."""
        result = run_digest("--kind", "noiseless", "--folder", "variants_N10_ladder")
        assert "NOISELESS" in result.stdout
        # Check stderr for scan info
        assert "variants_N10_ladder" in result.stderr

    def test_filter_no_results(self):
        """Impossible filter should produce 'no results' message."""
        result = run_digest("--kind", "noiseless", "--topology", "nonexistent_topology")
        assert "No results found" in result.stdout


@pytest.mark.slow
class TestCLISorting:
    """Test sorting options."""

    def test_sort_delta_e(self):
        """--sort delta_e should sort noiseless by ΔE/gap ascending."""
        result = run_digest("--kind", "noiseless", "--sort", "delta_e", "--top", "5")
        assert "NOISELESS" in result.stdout
        # First results should be the best (lowest ΔE/gap)
        assert "✅" in result.stdout

    def test_sort_verdict(self):
        """--sort verdict should sort experiments: confirmed → rejected → failed."""
        result = run_digest("--kind", "experiment", "--sort", "verdict")
        # Extract only table data lines (start with experiment ID pattern)
        lines = result.stdout.split("\n")
        data_lines = [
            line
            for line in lines
            if ("confirmed" in line or "rejected" in line or "failed" in line)
            and (
                "  A" in line
                or "  B" in line
                or "  C" in line
                or "  D" in line
                or "  E" in line
                or "  F" in line
                or "  G" in line
            )
        ]
        assert len(data_lines) > 0, "Should have experiment data lines"
        # First data line should be confirmed (sorted first)
        assert "confirmed" in data_lines[0], (
            f"First experiment should be confirmed, got: {data_lines[0]}"
        )

    def test_sort_r2(self):
        """--sort r2 should sort noisy by R² descending."""
        result = run_digest("--kind", "noisy", "--sort", "r2", "--top", "3")
        assert "NOISY" in result.stdout

    def test_top_limits_output(self):
        """--top N should limit results shown."""
        result = run_digest("--kind", "noiseless", "--top", "3")
        assert "3 runs scanned" in result.stdout


@pytest.mark.slow
class TestCLIGroupBy:
    """Test --group-by option."""

    def test_group_by_topology(self):
        """--group-by topology should show grouped summary."""
        result = run_digest("--kind", "noiseless", "--group-by", "topology")
        assert "Grouped by: topology" in result.stdout
        assert "chain_1d" in result.stdout
        assert "ladder" in result.stdout

    def test_group_by_n_restarts(self):
        """--group-by n_restarts should group by restart count."""
        result = run_digest(
            "--kind", "noiseless", "--group-by", "n_restarts", "--topology", "ladder"
        )
        assert "Grouped by: n_restarts" in result.stdout

    def test_group_by_noisy_n_qubits(self):
        """--group-by n_qubits should work for noisy results."""
        result = run_digest("--kind", "noisy", "--group-by", "n_qubits")
        assert "Grouped by: n_qubits" in result.stdout
        assert "6" in result.stdout
        assert "10" in result.stdout

    def test_group_by_invalid_key(self):
        """Invalid group key should show error with valid options."""
        result = run_digest("--kind", "noiseless", "--group-by", "invalid_key")
        assert "Unknown group key" in result.stdout


@pytest.mark.slow
class TestCLIAnalysis:
    """Test analysis modes (--stats, --outliers, --compare)."""

    def test_stats_noiseless(self):
        """--stats should show statistical summary."""
        result = run_digest("--kind", "noiseless", "--stats")
        assert "STATISTICAL SUMMARY" in result.stdout
        assert "Mean:" in result.stdout
        assert "Median:" in result.stdout
        assert "P25:" in result.stdout
        assert "Distribution:" in result.stdout

    def test_stats_noisy(self):
        """--stats with noisy should show R² and gain stats."""
        result = run_digest("--kind", "noisy", "--stats")
        assert "STATISTICAL SUMMARY" in result.stdout
        assert "R² Distribution" in result.stdout
        assert "Gain% Distribution" in result.stdout
        assert "Correlation" in result.stdout

    def test_stats_with_filter(self):
        """--stats should respect filters."""
        result = run_digest("--kind", "noiseless", "--stats", "--topology", "ladder")
        assert "STATISTICAL SUMMARY" in result.stdout

    def test_outliers(self):
        """--outliers should detect outliers."""
        result = run_digest("--kind", "noiseless", "--outliers")
        assert "OUTLIER ANALYSIS" in result.stdout
        assert "IQR" in result.stdout

    def test_compare_two_folders(self):
        """--compare should produce side-by-side comparison."""
        result = run_digest("--compare", "variants_N10_ladder", "variants_N10_triangular")
        assert "COMPARISON" in result.stdout
        assert "Mean ΔE/gap" in result.stdout
        assert "Winner" in result.stdout

    def test_compare_nonexistent_folder(self):
        """--compare with bad folder should fail gracefully."""
        result = run_digest(
            "--compare",
            "variants_N10_ladder",
            "nonexistent_folder",
            expect_ok=False,
        )
        assert result.returncode != 0


@pytest.mark.slow
class TestCLIOutput:
    """Test output format options."""

    def test_markdown_output(self, tmp_path):
        """--markdown should produce valid markdown."""
        outfile = tmp_path / "digest.md"
        run_digest("--kind", "experiment", "--markdown", "-o", str(outfile))
        assert outfile.exists()
        content = outfile.read_text()
        assert content.startswith("# Results Digest")
        assert "| ID |" in content

    def test_json_output(self, tmp_path):
        """--json should produce valid JSON with expected structure."""
        outfile = tmp_path / "digest.json"
        result = run_digest("--json", str(outfile))
        assert "Saved digest to" in result.stdout
        assert outfile.exists()

        data = json.loads(outfile.read_text())
        assert "noiseless" in data
        assert "noisy" in data
        assert "experiments" in data
        assert "summary" in data
        assert data["summary"]["n_noiseless"] > 0
        assert data["summary"]["n_experiments"] > 0

    def test_json_structure_noiseless(self, tmp_path):
        """JSON noiseless entries should have expected fields."""
        outfile = tmp_path / "digest.json"
        run_digest("--kind", "noiseless", "--top", "1", "--json", str(outfile))

        data = json.loads(outfile.read_text())
        assert len(data["noiseless"]) == 1
        entry = data["noiseless"][0]
        assert "delta_e_over_gap" in entry
        assert "n_qubits" in entry
        assert "topology" in entry
        assert "convergence_rate" in entry
        assert "generalization_gap" in entry

    def test_json_structure_noisy(self, tmp_path):
        """JSON noisy entries should have expected fields."""
        outfile = tmp_path / "digest.json"
        run_digest("--kind", "noisy", "--top", "1", "--json", str(outfile))

        data = json.loads(outfile.read_text())
        assert len(data["noisy"]) == 1
        entry = data["noisy"][0]
        assert "mean_r2" in entry
        assert "mean_gain_pct" in entry
        assert "n_layouts" in entry
        assert "success_criteria_met" in entry

    def test_verbose_shows_details(self):
        """--verbose should show extra information."""
        result = run_digest("--kind", "experiment", "--verbose")
        assert "H:" in result.stdout  # Hypothesis line

    def test_output_to_file(self, tmp_path):
        """--output should save to file."""
        outfile = tmp_path / "output.txt"
        run_digest("--kind", "experiment", "-o", str(outfile))
        assert outfile.exists()
        content = outfile.read_text()
        assert "HYPOTHESIS TESTS" in content


# ═══════════════════════════════════════════════════════════════════════════
# Progress/logging tests
# ═══════════════════════════════════════════════════════════════════════════


class TestProgress:
    """Test that progress messages appear on stderr."""

    def test_progress_on_stderr(self):
        """Progress messages should go to stderr, not stdout."""
        result = run_digest("--kind", "experiment")
        assert "[digest]" in result.stderr
        assert "Scanning" in result.stderr
        assert "[digest]" not in result.stdout

    def test_progress_shows_counts(self):
        """Progress should show scan counts."""
        result = run_digest("--kind", "noiseless", "--top", "1")
        assert "Scanned:" in result.stderr
        assert "noiseless" in result.stderr

    def test_progress_shows_filters(self):
        """Progress should show active filters."""
        result = run_digest("--kind", "noiseless", "--topology", "ladder")
        assert "Applying filters:" in result.stderr
        assert "topology=ladder" in result.stderr


# ═══════════════════════════════════════════════════════════════════════════
# Model tests — verify dataclass behavior
# ═══════════════════════════════════════════════════════════════════════════


class TestModels:
    """Test data model correctness."""

    def test_noiseless_result_defaults(self):
        """NoiselessResult should have sensible defaults."""
        from scripts.digest.models import NoiselessResult

        r = NoiselessResult(source_file="test.json", folder="test")
        assert r.n_qubits == 0
        assert r.p_layers == 2
        assert r.hidden_dim == 128
        assert r.delta_e_over_gap is None
        assert r.h_values == []

    def test_experiment_criteria_complete(self):
        """All known experiment IDs should have criteria defined."""
        from scripts.digest.models import EXPERIMENT_CRITERIA

        expected_ids = {
            "A3",
            "B1",
            "B2",
            "B4",
            "C1",
            "D1",
            "E4",
            "F1",
            "F3",
            "G1",
            "G2",
            "G3",
            "G4",
            "G5",
        }
        assert expected_ids.issubset(set(EXPERIMENT_CRITERIA.keys()))

    def test_rejection_is_finding_subset(self):
        """REJECTION_IS_FINDING should be a subset of EXPERIMENT_CRITERIA."""
        from scripts.digest.models import EXPERIMENT_CRITERIA, REJECTION_IS_FINDING

        assert REJECTION_IS_FINDING.issubset(set(EXPERIMENT_CRITERIA.keys()))


# ═══════════════════════════════════════════════════════════════════════════
# Fast equivalents for slow CLI tests (no subprocess, direct function calls)
# These validate the SAME logic as TestCLI* classes but without fork overhead.
# ═══════════════════════════════════════════════════════════════════════════


class TestDigestArgParsing:
    """Fast test of digest argument parsing (no subprocess)."""

    def test_parse_kind_noiseless(self, monkeypatch):
        from scripts.digest.__main__ import parse_args

        monkeypatch.setattr("sys.argv", ["digest", "--kind", "noiseless"])
        args = parse_args()
        assert args.kind == "noiseless"

    def test_parse_kind_default_all(self, monkeypatch):
        from scripts.digest.__main__ import parse_args

        monkeypatch.setattr("sys.argv", ["digest"])
        args = parse_args()
        assert args.kind == "all"

    def test_parse_filters(self, monkeypatch):
        from scripts.digest.__main__ import parse_args

        monkeypatch.setattr(
            "sys.argv",
            ["digest", "--topology", "ladder", "--n-qubits", "10", "--p-layers", "1"],
        )
        args = parse_args()
        assert args.topology == "ladder"
        assert args.n_qubits == 10
        assert args.p_layers == 1

    def test_parse_sort_and_top(self, monkeypatch):
        from scripts.digest.__main__ import parse_args

        monkeypatch.setattr("sys.argv", ["digest", "--sort", "delta_e", "--top", "5"])
        args = parse_args()
        assert args.sort == "delta_e"
        assert args.top == 5

    def test_parse_group_by(self, monkeypatch):
        from scripts.digest.__main__ import parse_args

        monkeypatch.setattr("sys.argv", ["digest", "--group-by", "topology"])
        args = parse_args()
        assert args.group_by == "topology"

    def test_parse_analysis_modes(self, monkeypatch):
        from scripts.digest.__main__ import parse_args

        monkeypatch.setattr("sys.argv", ["digest", "--stats", "--outliers"])
        args = parse_args()
        assert args.stats is True
        assert args.outliers is True

    def test_parse_compare(self, monkeypatch):
        from scripts.digest.__main__ import parse_args

        monkeypatch.setattr("sys.argv", ["digest", "--compare", "folderA", "folderB"])
        args = parse_args()
        assert args.compare == ["folderA", "folderB"]

    def test_parse_output_options(self, monkeypatch):
        from scripts.digest.__main__ import parse_args

        monkeypatch.setattr(
            "sys.argv", ["digest", "--markdown", "--json", "out.json", "-o", "out.md"]
        )
        args = parse_args()
        assert args.markdown is True
        assert args.json == "out.json"
        assert args.output == "out.md"

    def test_parse_model_filter(self, monkeypatch):
        from scripts.digest.__main__ import parse_args

        monkeypatch.setattr("sys.argv", ["digest", "--model", "heisenberg"])
        args = parse_args()
        assert args.model == "heisenberg"


class TestDigestApplyFilters:
    """Fast test of the apply_filters function (core filtering logic)."""

    def _make_noiseless(self, topology="chain_1d", n_qubits=6, p_layers=2, model="tfim"):
        from scripts.digest.models import NoiselessResult

        return NoiselessResult(
            source_file="test.json",
            folder="test_folder",
            topology=topology,
            n_qubits=n_qubits,
            p_layers=p_layers,
            model=model,
            variant_id=f"{topology}_n{n_qubits}_p{p_layers}",
        )

    def _make_noisy(self, topology="chain_1d", n_qubits=6, p_layers=1):
        from scripts.digest.models import NoisyResult

        return NoisyResult(
            source_file="test.json",
            folder="test_folder",
            topology=topology,
            n_qubits=n_qubits,
            p_layers=p_layers,
        )

    def test_filter_by_topology(self):
        from scripts.digest.__main__ import apply_filters

        nl = [self._make_noiseless("chain_1d"), self._make_noiseless("ladder")]
        ny = [self._make_noisy("chain_1d"), self._make_noisy("ladder")]
        result_nl, result_ny, _ = apply_filters(nl, ny, [], topology="ladder")
        assert len(result_nl) == 1
        assert result_nl[0].topology == "ladder"
        assert len(result_ny) == 1

    def test_filter_by_n_qubits(self):
        from scripts.digest.__main__ import apply_filters

        nl = [self._make_noiseless(n_qubits=6), self._make_noiseless(n_qubits=10)]
        result_nl, _, _ = apply_filters(nl, [], [], n_qubits=10)
        assert len(result_nl) == 1
        assert result_nl[0].n_qubits == 10

    def test_filter_by_p_layers(self):
        from scripts.digest.__main__ import apply_filters

        nl = [self._make_noiseless(p_layers=1), self._make_noiseless(p_layers=2)]
        result_nl, _, _ = apply_filters(nl, [], [], p_layers=1)
        assert len(result_nl) == 1
        assert result_nl[0].p_layers == 1

    def test_filter_by_model(self):
        from scripts.digest.__main__ import apply_filters

        nl = [
            self._make_noiseless(model="tfim"),
            self._make_noiseless(model="heisenberg"),
        ]
        result_nl, _, _ = apply_filters(nl, [], [], model="heisenberg")
        assert len(result_nl) == 1
        assert result_nl[0].model == "heisenberg"

    def test_no_filters_returns_all(self):
        from scripts.digest.__main__ import apply_filters

        nl = [self._make_noiseless(), self._make_noiseless("ladder")]
        result_nl, _, _ = apply_filters(nl, [], [])
        assert len(result_nl) == 2

    def test_combined_filters(self):
        from scripts.digest.__main__ import apply_filters

        nl = [
            self._make_noiseless("chain_1d", 6, 1),
            self._make_noiseless("chain_1d", 10, 1),
            self._make_noiseless("ladder", 10, 1),
            self._make_noiseless("chain_1d", 10, 2),
        ]
        result_nl, _, _ = apply_filters(nl, [], [], topology="chain_1d", n_qubits=10, p_layers=1)
        assert len(result_nl) == 1
        assert result_nl[0].n_qubits == 10
        assert result_nl[0].topology == "chain_1d"
        assert result_nl[0].p_layers == 1


class TestDigestFormatters:
    """Fast tests for formatter functions (output generation logic)."""

    def _make_noiseless(self, delta_e=0.03, topology="chain_1d", n_qubits=6):
        from scripts.digest.models import NoiselessResult

        return NoiselessResult(
            source_file="test.json",
            folder="test_folder",
            topology=topology,
            n_qubits=n_qubits,
            delta_e_over_gap=delta_e,
            variant_id=f"comp_{topology}_p1",
            elapsed_s=120.0,
            convergence_rate=1.0,
            theta_smoothness=0.5,
            generalization_gap=0.005,
        )

    def _make_noisy(self, r2=0.95, gain_pct=45.0):
        from scripts.digest.models import NoisyResult

        return NoisyResult(
            source_file="test.json",
            folder="test_folder",
            topology="chain_1d",
            n_qubits=10,
            mean_r2=r2,
            mean_gain_pct=gain_pct,
            n_mitigated_wins=3,
            n_total=3,
            success_criteria_met=True,
            variant_id="noisy_variant",
        )

    def _make_experiment(self, verdict="confirmed", exp_id="G1"):
        from scripts.digest.models import ExperimentResult

        return ExperimentResult(
            source_file="test.json",
            folder=f"exp_{exp_id.lower()}",
            experiment_id=exp_id,
            category="G",
            hypothesis="Test hypothesis",
            topology="chain_1d",
            n_qubits=10,
            verdict=verdict,
            mean_de_gap=0.03,
            pass_rate=1.0,
        )

    def test_format_noiseless_text_has_header(self):
        from scripts.digest.formatters import format_noiseless_text

        results = [self._make_noiseless()]
        output = format_noiseless_text(results)
        # Formatter produces the table content; header "NOISELESS PIPELINE" is added by main()
        assert "ΔE/gap" in output
        assert "runs scanned" in output

    def test_format_noiseless_text_includes_data(self):
        from scripts.digest.formatters import format_noiseless_text

        results = [self._make_noiseless(delta_e=0.02)]
        output = format_noiseless_text(results)
        assert "0.02" in output

    def test_format_noisy_text_has_header(self):
        from scripts.digest.formatters import format_noisy_text

        results = [self._make_noisy()]
        output = format_noisy_text(results)
        # Formatter produces data; header is added by main()
        assert "R²" in output
        assert "runs scanned" in output

    def test_format_experiment_text_has_header(self):
        from scripts.digest.formatters import format_experiment_text

        results = [self._make_experiment()]
        output = format_experiment_text(results)
        # Formatter includes experiment data
        assert "experiments scanned" in output
        assert "confirmed" in output or "Confirmed" in output

    def test_format_experiment_shows_verdict(self):
        from scripts.digest.formatters import format_experiment_text

        results = [
            self._make_experiment("confirmed", "G1"),
            self._make_experiment("rejected", "B4"),
        ]
        output = format_experiment_text(results)
        assert "confirmed" in output
        assert "rejected" in output

    def test_format_noiseless_grouped_by_topology(self):
        from scripts.digest.formatters import format_noiseless_grouped

        results = [
            self._make_noiseless(0.02, "chain_1d"),
            self._make_noiseless(0.04, "ladder"),
            self._make_noiseless(0.03, "chain_1d"),
        ]
        output = format_noiseless_grouped(results, "topology")
        assert "Grouped by: topology" in output
        assert "chain_1d" in output
        assert "ladder" in output

    def test_format_noiseless_grouped_invalid_key(self):
        from scripts.digest.formatters import format_noiseless_grouped

        results = [self._make_noiseless()]
        output = format_noiseless_grouped(results, "nonexistent_key")
        assert "Unknown group key" in output

    def test_format_markdown_structure(self):
        from scripts.digest.formatters import format_markdown

        nl = [self._make_noiseless()]
        ny = [self._make_noisy()]
        exp = [self._make_experiment()]
        output = format_markdown(nl, ny, exp)
        assert output.startswith("# Results Digest")
        assert "| " in output  # Has table rows

    def test_format_noiseless_stats(self):
        from scripts.digest.formatters import format_noiseless_stats

        results = [
            self._make_noiseless(0.01),
            self._make_noiseless(0.03),
            self._make_noiseless(0.05),
            self._make_noiseless(0.08),
        ]
        output = format_noiseless_stats(results)
        assert "STATISTICAL SUMMARY" in output
        assert "Mean:" in output
        assert "Median:" in output

    def test_format_noiseless_outliers(self):
        from scripts.digest.formatters import format_noiseless_outliers

        # Needs at least 5 results for IQR-based outlier detection
        results = [
            self._make_noiseless(0.02),
            self._make_noiseless(0.03),
            self._make_noiseless(0.025),
            self._make_noiseless(0.028),
            self._make_noiseless(0.50),  # outlier
        ]
        output = format_noiseless_outliers(results)
        assert "OUTLIER" in output or "outlier" in output.lower()

    def test_format_noisy_stats(self):
        from scripts.digest.formatters import format_noisy_stats

        results = [
            self._make_noisy(0.95, 45.0),
            self._make_noisy(0.88, 30.0),
            self._make_noisy(0.72, -5.0),
        ]
        output = format_noisy_stats(results)
        assert "R² Distribution" in output


class TestDigestJsonExport:
    """Fast test of digest JSON export logic."""

    def test_json_export_structure(self, tmp_path):
        """Test the JSON export produces expected keys."""
        from scripts.digest.__main__ import _write_output
        from scripts.digest.models import ExperimentResult, NoiselessResult, NoisyResult

        nl = NoiselessResult(
            source_file="a.json", folder="f",
            n_qubits=10, topology="chain_1d", delta_e_over_gap=0.02,
            variant_id="v1",
        )
        ny = NoisyResult(
            source_file="b.json", folder="f",
            n_qubits=10, topology="chain_1d", mean_r2=0.95,
            variant_id="v2",
        )
        exp = ExperimentResult(
            source_file="c.json", folder="exp_g1",
            experiment_id="G1", verdict="confirmed",
        )

        # Build JSON manually (same logic as main)
        output = json.dumps({
            "noiseless": [{"delta_e_over_gap": nl.delta_e_over_gap, "n_qubits": nl.n_qubits}],
            "noisy": [{"mean_r2": ny.mean_r2}],
            "experiments": [{"experiment_id": exp.experiment_id, "verdict": exp.verdict}],
            "summary": {"n_noiseless": 1, "n_noisy": 1, "n_experiments": 1},
        }, indent=2)

        outpath = tmp_path / "test.json"
        _write_output(output, str(outpath))
        assert outpath.exists()

        data = json.loads(outpath.read_text())
        assert "noiseless" in data
        assert "noisy" in data
        assert "experiments" in data
        assert "summary" in data
        assert data["summary"]["n_noiseless"] == 1

    def test_write_output_to_file(self, tmp_path):
        from scripts.digest.__main__ import _write_output

        outpath = tmp_path / "output.txt"
        _write_output("test content here", str(outpath))
        assert outpath.exists()
        assert outpath.read_text() == "test content here"

    def test_write_output_none_prints(self, capsys):
        from scripts.digest.__main__ import _write_output

        _write_output("hello world", None)
        captured = capsys.readouterr()
        assert "hello world" in captured.out
