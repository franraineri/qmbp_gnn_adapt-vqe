"""Test MitigationBenchmarkAnalyzer derived metrics, ranking, and export (task 8.2)."""

import json
import tempfile
from pathlib import Path

import pytest

from project_health.analysis.mitigation_benchmark_analyzer import (
    MitigationBenchmarkAnalyzer,
)


def _make_entry(
    config_id="C0_raw",
    h_value=3.5,
    e_raw=-12.5,
    e_mitigated=-12.7,
    e_exact=-12.8,
    delta_e_gap=0.035,
    shots=16384,
    fidelity_estimate=0.85,
    mode="fake_backend",
):
    """Create a valid result envelope for testing."""
    return {
        "benchmark_metadata": {
            "config_id": config_id,
            "h_value": h_value,
            "execution_mode": mode,
            "timestamp": "2026-06-18T10:30:00",
            "seed": 42,
        },
        "circuit_stats": {
            "depth_logical": 10,
            "depth_transpiled": 25,
            "fidelity_estimate": fidelity_estimate,
        },
        "results": {
            "e_raw": e_raw,
            "e_mitigated": e_mitigated,
            "e_exact": e_exact,
            "delta_e_gap": delta_e_gap,
        },
        "shots": shots,
    }


def _setup_analyzer(entries: list[dict]) -> MitigationBenchmarkAnalyzer:
    """Create analyzer with pre-loaded entries (bypasses scan)."""
    a = MitigationBenchmarkAnalyzer(results_dir=Path("/tmp/fake"))
    a.entries = entries
    return a


class TestComputeDerivedMetrics:
    """Tests for compute_derived_metrics()."""

    def test_basic_metrics_computation(self):
        entries = [
            _make_entry("C0_raw", h_value=3.25, delta_e_gap=0.04, shots=16384),
            _make_entry("C0_raw", h_value=3.5, delta_e_gap=0.03, shots=16384),
            _make_entry("C5_full", h_value=3.25, delta_e_gap=0.02, shots=32768),
            _make_entry("C5_full", h_value=3.5, delta_e_gap=0.01, shots=32768),
        ]
        a = _setup_analyzer(entries)
        metrics = a.compute_derived_metrics()

        assert "C0_raw" in metrics
        assert "C5_full" in metrics
        assert metrics["C0_raw"]["mean_delta_e_gap"] == pytest.approx(0.035)
        assert metrics["C5_full"]["mean_delta_e_gap"] == pytest.approx(0.015)

    def test_mean_delta_e_gap_formula(self):
        entries = [
            _make_entry("C0_raw", delta_e_gap=0.10),
            _make_entry("C0_raw", delta_e_gap=0.20),
            _make_entry("C0_raw", delta_e_gap=0.30),
        ]
        a = _setup_analyzer(entries)
        metrics = a.compute_derived_metrics()
        assert metrics["C0_raw"]["mean_delta_e_gap"] == pytest.approx(0.20)

    def test_improvement_vs_raw_formula(self):
        # improvement_vs_raw = (E_raw - E_mitigated) / (E_raw - E_exact)
        # E_raw=-12.5, E_mitigated=-12.7, E_exact=-12.8
        # = (-12.5 - (-12.7)) / (-12.5 - (-12.8)) = 0.2 / 0.3 = 0.6667
        entries = [
            _make_entry("C0_raw", shots=16384),
            _make_entry("C5_full", e_raw=-12.5, e_mitigated=-12.7, e_exact=-12.8, shots=16384),
        ]
        a = _setup_analyzer(entries)
        metrics = a.compute_derived_metrics()
        expected = (-12.5 - (-12.7)) / (-12.5 - (-12.8))
        assert metrics["C5_full"]["improvement_vs_raw"] == pytest.approx(expected, rel=1e-6)

    def test_overhead_factor_formula(self):
        # overhead_factor = shots_config / shots_C0_raw
        entries = [
            _make_entry("C0_raw", shots=16384),
            _make_entry("C5_full", shots=32768),
        ]
        a = _setup_analyzer(entries)
        metrics = a.compute_derived_metrics()
        assert metrics["C5_full"]["overhead_factor"] == pytest.approx(2.0)
        assert metrics["C0_raw"]["overhead_factor"] == pytest.approx(1.0)

    def test_precision_per_shot_formula(self):
        # precision_per_shot = (1 - delta_e_gap) / total_shots
        entries = [
            _make_entry("C0_raw", delta_e_gap=0.04, shots=16384),
        ]
        a = _setup_analyzer(entries)
        metrics = a.compute_derived_metrics()
        expected = (1.0 - 0.04) / 16384
        assert metrics["C0_raw"]["precision_per_shot"] == pytest.approx(expected, rel=1e-6)

    def test_net_benefit_formula(self):
        # net_benefit = fidelity_estimate * (1 - delta_e_gap)
        entries = [
            _make_entry("C0_raw", delta_e_gap=0.04, fidelity_estimate=0.85),
        ]
        a = _setup_analyzer(entries)
        metrics = a.compute_derived_metrics()
        expected = 0.85 * (1.0 - 0.04)
        assert metrics["C0_raw"]["net_benefit"] == pytest.approx(expected, rel=1e-6)

    def test_missing_c0_raw_baseline_warns(self, caplog):
        """When C0_raw is missing, improvement_vs_raw and overhead_factor are None."""
        entries = [
            _make_entry("C5_full", delta_e_gap=0.02, shots=32768),
        ]
        a = _setup_analyzer(entries)
        import logging

        with caplog.at_level(logging.WARNING):
            metrics = a.compute_derived_metrics()
        assert "C0_raw baseline not found" in caplog.text
        assert metrics["C5_full"]["improvement_vs_raw"] is None
        assert metrics["C5_full"]["overhead_factor"] is None

    def test_e_raw_equals_e_exact_skips_improvement(self):
        """When E_raw == E_exact, improvement_vs_raw is None (division by zero avoided)."""
        entries = [
            _make_entry("C0_raw", shots=16384),
            _make_entry("C5_full", e_raw=-12.8, e_mitigated=-12.75, e_exact=-12.8, shots=16384),
        ]
        a = _setup_analyzer(entries)
        metrics = a.compute_derived_metrics()
        # C5_full should have None improvement (denominator is zero)
        assert metrics["C5_full"]["improvement_vs_raw"] is None

    def test_no_entries_returns_empty(self):
        a = _setup_analyzer([])
        metrics = a.compute_derived_metrics()
        assert metrics == {}

    def test_stores_in_derived_metrics_attribute(self):
        entries = [_make_entry("C0_raw", delta_e_gap=0.04)]
        a = _setup_analyzer(entries)
        result = a.compute_derived_metrics()
        assert a.derived_metrics == result
        assert "C0_raw" in a.derived_metrics


class TestRankConfigs:
    """Tests for rank_configs()."""

    def test_ranking_by_mean_delta_e_gap_ascending(self):
        entries = [
            _make_entry("C0_raw", delta_e_gap=0.10),
            _make_entry("C3_gf", delta_e_gap=0.05),
            _make_entry("C5_full", delta_e_gap=0.02),
        ]
        a = _setup_analyzer(entries)
        a.compute_derived_metrics()
        ranking = a.rank_configs()

        assert len(ranking) == 3
        assert ranking[0]["config_id"] == "C5_full"
        assert ranking[0]["rank"] == 1
        assert ranking[1]["config_id"] == "C3_gf"
        assert ranking[1]["rank"] == 2
        assert ranking[2]["config_id"] == "C0_raw"
        assert ranking[2]["rank"] == 3

    def test_ranking_includes_all_metrics(self):
        entries = [
            _make_entry("C0_raw", delta_e_gap=0.04, shots=16384, fidelity_estimate=0.85),
        ]
        a = _setup_analyzer(entries)
        a.compute_derived_metrics()
        ranking = a.rank_configs()

        assert len(ranking) == 1
        entry = ranking[0]
        assert "rank" in entry
        assert "config_id" in entry
        assert "mean_delta_e_gap" in entry
        assert "improvement_vs_raw" in entry
        assert "overhead_factor" in entry
        assert "precision_per_shot" in entry
        assert "net_benefit" in entry

    def test_empty_derived_metrics_returns_empty(self):
        a = _setup_analyzer([])
        ranking = a.rank_configs()
        assert ranking == []

    def test_configs_without_delta_excluded_from_ranking(self):
        """Configs with None mean_delta_e_gap should not appear in ranking."""
        entries = [
            _make_entry("C0_raw", delta_e_gap=0.04),
        ]
        a = _setup_analyzer(entries)
        a.compute_derived_metrics()
        # Manually set one config to None delta
        a.derived_metrics["C_broken"] = {
            "config_id": "C_broken",
            "n_entries": 1,
            "mean_delta_e_gap": None,
            "improvement_vs_raw": None,
            "overhead_factor": None,
            "precision_per_shot": None,
            "net_benefit": None,
        }
        ranking = a.rank_configs()
        config_ids = [r["config_id"] for r in ranking]
        assert "C_broken" not in config_ids
        assert "C0_raw" in config_ids


class TestExportComparisonTable:
    """Tests for export_comparison_table()."""

    def test_exports_json_file(self, tmp_path):
        entries = [
            _make_entry("C0_raw", delta_e_gap=0.04, shots=16384),
            _make_entry("C5_full", delta_e_gap=0.02, shots=32768),
        ]
        a = _setup_analyzer(entries)
        a.compute_derived_metrics()

        output_dir = tmp_path / "analysis"
        a.export_comparison_table(output_dir=output_dir)

        out_file = output_dir / "comparison_table.json"
        assert out_file.exists()

        data = json.loads(out_file.read_text())
        assert "ranking" in data
        assert "metadata" in data

    def test_export_metadata_fields(self, tmp_path):
        entries = [
            _make_entry("C0_raw", delta_e_gap=0.04),
            _make_entry("C0_raw", delta_e_gap=0.05),
            _make_entry("C5_full", delta_e_gap=0.02),
        ]
        a = _setup_analyzer(entries)
        a.compute_derived_metrics()
        a.export_comparison_table(output_dir=tmp_path)

        data = json.loads((tmp_path / "comparison_table.json").read_text())
        meta = data["metadata"]
        assert meta["n_configs"] == 2
        assert meta["n_entries"] == 3
        assert meta["baseline_config"] == "C0_raw"

    def test_export_without_c0_raw_baseline(self, tmp_path):
        entries = [_make_entry("C5_full", delta_e_gap=0.02)]
        a = _setup_analyzer(entries)
        a.compute_derived_metrics()
        a.export_comparison_table(output_dir=tmp_path)

        data = json.loads((tmp_path / "comparison_table.json").read_text())
        assert data["metadata"]["baseline_config"] is None

    def test_export_ranking_order_preserved(self, tmp_path):
        entries = [
            _make_entry("C0_raw", delta_e_gap=0.10),
            _make_entry("C3_gf", delta_e_gap=0.05),
            _make_entry("C5_full", delta_e_gap=0.01),
        ]
        a = _setup_analyzer(entries)
        a.compute_derived_metrics()
        a.export_comparison_table(output_dir=tmp_path)

        data = json.loads((tmp_path / "comparison_table.json").read_text())
        ranking = data["ranking"]
        assert ranking[0]["config_id"] == "C5_full"
        assert ranking[1]["config_id"] == "C3_gf"
        assert ranking[2]["config_id"] == "C0_raw"

    def test_default_output_dir_uses_results_dir(self):
        """When output_dir is None, uses self.results_dir / 'analysis'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            entries = [_make_entry("C0_raw", delta_e_gap=0.04)]
            a = MitigationBenchmarkAnalyzer(results_dir=results_dir)
            a.entries = entries
            a.compute_derived_metrics()
            a.export_comparison_table()

            out_file = results_dir / "analysis" / "comparison_table.json"
            assert out_file.exists()
