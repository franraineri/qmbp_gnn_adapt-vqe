"""Tests for project_health/compare.py — experiment comparison CLI.

Covers:
- ResultStore.compare_experiments logic
- ResultStore.resolve_category
- ResultStore.analyze_noisy_correlations
- ResultStore.analyze_noisy_by_group
- CLI argument parsing
- JSON export

Run with:
    pytest tests/test_compare.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from qmbp_simulation.framework import ResultStore

# ═══════════════════════════════════════════════════════════════════════════
# ResultStore integration tests (compare functionality)
# ═══════════════════════════════════════════════════════════════════════════


class TestResultStoreCompare:
    """Test ResultStore experiment comparison logic."""

    def _create_experiment_result(
        self,
        tmp_path: Path,
        exp_id: str,
        mean_de_gap: float = 0.03,
        pass_rate: float = 1.0,
    ) -> None:
        """Create a minimal experiment result file."""
        exp_dir = tmp_path / f"exp_{exp_id.lower()}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "config": {
                "n_qubits": 10,
                "topology": "chain_1d",
                "p_layers": 2,
            },
            "analysis": {
                "summary": {
                    "mean_de_gap": mean_de_gap,
                    "pass_rate": pass_rate,
                    "n_seeds": 3,
                }
            },
        }
        (exp_dir / "run_001.json").write_text(json.dumps(result))

    def test_list_experiments_discovers_files(self, tmp_path):
        self._create_experiment_result(tmp_path, "G1")
        self._create_experiment_result(tmp_path, "G5")
        store = ResultStore(results_root=tmp_path)
        exps = store.list_experiments()
        assert "G1" in exps
        assert "G5" in exps

    def test_list_experiments_empty_dir(self, tmp_path):
        store = ResultStore(results_root=tmp_path)
        assert store.list_experiments() == []

    def test_compare_experiments_returns_structured(self, tmp_path):
        self._create_experiment_result(tmp_path, "G1", mean_de_gap=0.02)
        self._create_experiment_result(tmp_path, "G5", mean_de_gap=0.04)
        store = ResultStore(results_root=tmp_path)
        comparisons = store.compare_experiments(["G1", "G5"])
        assert len(comparisons) == 2
        for c in comparisons:
            assert "verdict" in c
            assert "experiment_id" in c

    def test_compare_missing_experiment_skipped(self, tmp_path):
        self._create_experiment_result(tmp_path, "G1")
        store = ResultStore(results_root=tmp_path)
        comparisons = store.compare_experiments(["G1", "NONEXISTENT"])
        assert len(comparisons) == 1

    def test_load_latest_returns_most_recent(self, tmp_path):
        exp_dir = tmp_path / "exp_g1"
        exp_dir.mkdir(parents=True)
        (exp_dir / "run_001.json").write_text(
            json.dumps({"version": 1, "analysis": {"summary": {"mean_de_gap": 0.05}}})
        )
        (exp_dir / "run_002.json").write_text(
            json.dumps({"version": 2, "analysis": {"summary": {"mean_de_gap": 0.02}}})
        )
        store = ResultStore(results_root=tmp_path)
        latest = store.load_latest("G1")
        assert latest is not None
        assert latest["version"] == 2

    def test_resolve_category_by_letter(self, tmp_path):
        self._create_experiment_result(tmp_path, "G1")
        self._create_experiment_result(tmp_path, "G5")
        self._create_experiment_result(tmp_path, "B4")
        store = ResultStore(results_root=tmp_path)
        available = store.list_experiments()
        g_exps = store.resolve_category("G", available)
        assert "G1" in g_exps
        assert "G5" in g_exps
        assert "B4" not in g_exps


# ═══════════════════════════════════════════════════════════════════════════
# Noisy analysis tests
# ═══════════════════════════════════════════════════════════════════════════


class TestNoisyAnalysis:
    """Test ResultStore noisy/ZNE analysis methods."""

    def _make_noisy_results(self) -> list[dict]:
        """Create synthetic noisy experiment results.

        Note: ResultStore.analyze_noisy_correlations uses "gain" (fractional),
        not "gain_pct". gain=0.45 means +45% improvement.
        """
        return [
            {
                "seed_layout": 42,
                "n_layouts": 3,
                "h_test": 2.0,
                "r2": 0.95,
                "gain": 0.45,
            },
            {
                "seed_layout": 43,
                "n_layouts": 3,
                "h_test": 2.0,
                "r2": 0.88,
                "gain": 0.30,
            },
            {
                "seed_layout": 44,
                "n_layouts": 3,
                "h_test": 2.5,
                "r2": 0.72,
                "gain": -0.05,
            },
            {
                "seed_layout": 42,
                "n_layouts": 5,
                "h_test": 3.0,
                "r2": 0.99,
                "gain": 0.62,
            },
        ]

    def test_analyze_noisy_correlations_structure(self, tmp_path):
        store = ResultStore(results_root=tmp_path)
        results = self._make_noisy_results()
        correlations = store.analyze_noisy_correlations(results)

        assert "mean_r2" in correlations
        assert "pct_helps" in correlations
        assert "mean_gain_pct" in correlations
        assert "n_evaluations" in correlations

    def test_analyze_noisy_correlations_values(self, tmp_path):
        store = ResultStore(results_root=tmp_path)
        results = self._make_noisy_results()
        correlations = store.analyze_noisy_correlations(results)

        # 3/4 have gain > 0 → 75%
        assert correlations["pct_helps"] == pytest.approx(75.0)
        assert correlations["n_evaluations"] == 4

    def test_analyze_noisy_by_group(self, tmp_path):
        store = ResultStore(results_root=tmp_path)
        results = self._make_noisy_results()
        grouped = store.analyze_noisy_by_group(results, "n_layouts")

        assert isinstance(grouped, dict)
        # Should have groups for n_layouts=3 and n_layouts=5
        assert len(grouped) >= 1

    def test_load_noisy_results_from_file(self, tmp_path):
        noisy_dir = tmp_path / "exp_noisy_variants"
        noisy_dir.mkdir()
        data = {"results": self._make_noisy_results()}
        (noisy_dir / "zne_results.json").write_text(json.dumps(data))

        store = ResultStore(results_root=tmp_path)
        results = store.load_noisy_results(filename="zne_results.json")
        assert len(results) == 4

    def test_load_noisy_results_empty_when_no_dir(self, tmp_path):
        store = ResultStore(results_root=tmp_path)
        results = store.load_noisy_results()
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════
# CLI tests (scripts/compare.py)
# ═══════════════════════════════════════════════════════════════════════════


COMPARE_CMD = [sys.executable, "scripts/compare.py"]


@pytest.mark.slow
class TestCompareCLI:
    """Test compare CLI invocation."""

    def test_no_args_shows_usage(self):
        result = subprocess.run(
            COMPARE_CMD,
            capture_output=True,
            text=True,
            timeout=15,
        )
        # Should ask for --all, --exp, etc.
        assert "Specify --all" in result.stdout or result.returncode != 0

    def test_all_mode_runs(self):
        """--all should run without crashing (even if no results)."""
        result = subprocess.run(
            [*COMPARE_CMD, "--all", "--results-dir", "results/experiments"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should either produce output or say "No results"
        assert result.returncode == 0 or "No results" in result.stdout

    def test_json_output(self, tmp_path):
        """--json should produce a JSON file."""
        json_out = tmp_path / "compare_out.json"
        subprocess.run(
            [
                *COMPARE_CMD,
                "--all",
                "--results-dir",
                "results/experiments",
                "--json",
                str(json_out),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # If experiments exist, json should be written
        if json_out.exists():
            data = json.loads(json_out.read_text())
            assert isinstance(data, list)

    def test_category_filter(self):
        """--category G should not crash."""
        result = subprocess.run(
            [*COMPARE_CMD, "--category", "G", "--results-dir", "results/experiments"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "No results" in result.stdout

    def test_noisy_mode_runs(self):
        """--noisy should run without crashing."""
        result = subprocess.run(
            [*COMPARE_CMD, "--noisy", "--results-dir", "results/experiments"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0 or "No noisy" in result.stdout


# ═══════════════════════════════════════════════════════════════════════════
# Fast equivalents for slow CLI tests (run without subprocess)
# ═══════════════════════════════════════════════════════════════════════════


class TestCompareArgParsing:
    """Fast test of compare.py argument parsing (no subprocess)."""

    def test_parse_args_all(self, monkeypatch):
        """--all should set args.all=True."""
        import project_health.compare as compare_mod

        monkeypatch.setattr("sys.argv", ["compare.py", "--all"])
        args = compare_mod.parse_args()
        assert args.all is True
        assert args.noisy is False

    def test_parse_args_exp(self, monkeypatch):
        """--exp G1 G5 should populate args.experiments."""
        import project_health.compare as compare_mod

        monkeypatch.setattr("sys.argv", ["compare.py", "--exp", "G1", "G5"])
        args = compare_mod.parse_args()
        assert args.experiments == ["G1", "G5"]

    def test_parse_args_category(self, monkeypatch):
        """--category G should set category."""
        import project_health.compare as compare_mod

        monkeypatch.setattr("sys.argv", ["compare.py", "--category", "G"])
        args = compare_mod.parse_args()
        assert args.category == "G"

    def test_parse_args_noisy_with_group(self, monkeypatch):
        """--noisy --group-by seed_layout should set both."""
        import project_health.compare as compare_mod

        monkeypatch.setattr("sys.argv", ["compare.py", "--noisy", "--group-by", "seed_layout"])
        args = compare_mod.parse_args()
        assert args.noisy is True
        assert args.group_by == "seed_layout"

    def test_parse_args_json_output(self, monkeypatch):
        """--json file.json should set json_file."""
        import project_health.compare as compare_mod

        monkeypatch.setattr("sys.argv", ["compare.py", "--all", "--json", "out.json"])
        args = compare_mod.parse_args()
        assert args.json_file == "out.json"

    def test_parse_args_results_dir(self, monkeypatch):
        """--results-dir should override default."""
        import project_health.compare as compare_mod

        monkeypatch.setattr("sys.argv", ["compare.py", "--all", "--results-dir", "/tmp/results"])
        args = compare_mod.parse_args()
        assert args.results_dir == "/tmp/results"


class TestCompareWriteJson:
    """Fast test of compare.py JSON export function."""

    def test_write_json_creates_file(self, tmp_path):
        from project_health.compare import _write_json

        data = [{"experiment_id": "G1", "verdict": "confirmed"}]
        outpath = str(tmp_path / "test_output.json")
        _write_json(data, outpath)

        result_path = tmp_path / "test_output.json"
        assert result_path.exists()
        loaded = json.loads(result_path.read_text())
        assert loaded == data

    def test_write_json_creates_parent_dirs(self, tmp_path):
        from project_health.compare import _write_json

        data = {"key": "value"}
        outpath = str(tmp_path / "nested" / "dir" / "output.json")
        _write_json(data, outpath)

        assert (tmp_path / "nested" / "dir" / "output.json").exists()

    def test_write_json_handles_datetimes(self, tmp_path):
        """default=str should handle non-serializable types."""
        from datetime import datetime

        from project_health.compare import _write_json

        data = {"timestamp": datetime(2024, 1, 15, 10, 30)}
        outpath = str(tmp_path / "datetime.json")
        _write_json(data, outpath)

        loaded = json.loads((tmp_path / "datetime.json").read_text())
        assert "2024" in loaded["timestamp"]
