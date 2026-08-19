"""Unit tests for compute_pareto_frontier and compute_ablation.

Tests the Pareto frontier and ablation study implementations in
MitigationBenchmarkAnalyzer with synthetic data.
"""

import json
import tempfile
from pathlib import Path

from project_health.analysis.hardware.mitigation_benchmark_analyzer import (
    MitigationBenchmarkAnalyzer,
)


def _make_entry(
    config_id, delta_e_gap, shots=16384, e_raw=-8.0, e_mitigated=-9.0, e_exact=-10.0, fidelity=0.95
):
    """Create a minimal valid ResultEnvelope for testing."""
    return {
        "benchmark_metadata": {
            "config_id": config_id,
            "h_value": 3.5,
            "execution_mode": "fake_backend",
        },
        "circuit_stats": {"fidelity_estimate": fidelity},
        "results": {
            "delta_e_gap": delta_e_gap,
            "e_raw": e_raw,
            "e_mitigated": e_mitigated,
            "e_exact": e_exact,
        },
        "shots": shots,
    }


class TestParetoFrontier:
    """Tests for compute_pareto_frontier()."""

    def test_dominated_config_excluded(self):
        """Config dominated in both objectives is excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
            a.entries = [
                _make_entry("C0_raw", 0.10, shots=16384),
                _make_entry("C1_dd_only", 0.08, shots=16384),
                _make_entry("C5_full_pea_balanced", 0.02, shots=65536),
            ]
            frontier = a.compute_pareto_frontier()
            assert "C0_raw" not in frontier
            assert "C1_dd_only" in frontier
            assert "C5_full_pea_balanced" in frontier

    def test_single_config_on_frontier(self):
        """A single config is always on the Pareto frontier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
            a.entries = [_make_entry("C0_raw", 0.10)]
            frontier = a.compute_pareto_frontier()
            assert frontier == ["C0_raw"]

    def test_empty_entries_returns_empty(self):
        """No entries yields empty frontier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
            frontier = a.compute_pareto_frontier()
            assert frontier == []

    def test_frontier_sorted_by_overhead(self):
        """Frontier configs are sorted by overhead_factor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
            a.entries = [
                _make_entry("C0_raw", 0.10, shots=16384),
                _make_entry("C1_dd_only", 0.08, shots=16384),
                _make_entry("C3_full_gf", 0.04, shots=49152),
                _make_entry("C5_full_pea_balanced", 0.02, shots=65536),
            ]
            frontier = a.compute_pareto_frontier()
            overheads = [a.derived_metrics[c]["overhead_factor"] for c in frontier]
            assert overheads == sorted(overheads)

    def test_equal_overhead_only_best_gap(self):
        """Same overhead -> only lowest gap on frontier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
            a.entries = [
                _make_entry("C0_raw", 0.10, shots=16384),
                _make_entry("C1_dd_only", 0.05, shots=16384),
                _make_entry("C2_dd_tw", 0.03, shots=16384),
            ]
            frontier = a.compute_pareto_frontier()
            assert frontier == ["C2_dd_tw"]


class TestAblationStudy:
    """Tests for compute_ablation()."""

    def test_all_techniques_computed(self):
        """All 7 ablation pairs computed when data present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
            a.entries = [
                _make_entry("C0_raw", 0.10),
                _make_entry("C1_dd_only", 0.08),
                _make_entry("C2_dd_tw", 0.06),
                _make_entry("C3_full_gf", 0.04),
                _make_entry("C5_full_pea_balanced", 0.02),
                _make_entry("C10_kitchen_sink", 0.025),
                _make_entry("C15_pea_no_affine", 0.03),
                _make_entry("C16_aqc_pea", 0.015),
            ]
            ablation = a.compute_ablation()
            for tech, data in ablation.items():
                assert data["status"] == "computed", f"{tech}"
                assert data["contribution"] is not None

    def test_contribution_values(self):
        """Ablation contributions match formulas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
            a.entries = [
                _make_entry("C0_raw", 0.10),
                _make_entry("C1_dd_only", 0.08),
                _make_entry("C2_dd_tw", 0.06),
                _make_entry("C3_full_gf", 0.04),
                _make_entry("C5_full_pea_balanced", 0.02),
                _make_entry("C10_kitchen_sink", 0.025),
                _make_entry("C15_pea_no_affine", 0.03),
                _make_entry("C16_aqc_pea", 0.015),
            ]
            ab = a.compute_ablation()
            assert abs(ab["DD"]["contribution"] - 0.02) < 1e-10
            assert abs(ab["Twirling+TREX"]["contribution"] - 0.02) < 1e-10
            assert abs(ab["ZNE (GF)"]["contribution"] - 0.02) < 1e-10
            assert abs(ab["ZNE (PEA)"]["contribution"] - 0.04) < 1e-10
            assert abs(ab["Affine"]["contribution"] - 0.01) < 1e-10
            assert abs(ab["GNN-QEM"]["contribution"] + 0.005) < 1e-10
            assert abs(ab["AQC"]["contribution"] - 0.005) < 1e-10

    def test_missing_configs_graceful(self):
        """Missing config pairs -> status='missing_data'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
            a.entries = [
                _make_entry("C0_raw", 0.10),
                _make_entry("C1_dd_only", 0.08),
            ]
            ab = a.compute_ablation()
            assert ab["DD"]["status"] == "computed"
            assert ab["ZNE (GF)"]["status"] == "missing_data"
            assert ab["GNN-QEM"]["status"] == "missing_data"

    def test_export_json_created(self):
        """ablation_study.json exported to analysis/ dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
            a.entries = [
                _make_entry("C0_raw", 0.10),
                _make_entry("C1_dd_only", 0.08),
            ]
            a.compute_ablation()
            path = Path(tmpdir) / "analysis" / "ablation_study.json"
            assert path.exists()
            data = json.loads(path.read_text())
            assert "description" in data
            assert "techniques" in data
            assert "DD" in data["techniques"]


class TestDerivedMetricsCaching:
    """Tests for compute_derived_metrics caching."""

    def test_returns_cached_on_second_call(self):
        """Second call returns same object (cached)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
            a.entries = [_make_entry("C0_raw", 0.10)]
            m1 = a.compute_derived_metrics()
            m2 = a.compute_derived_metrics()
            assert m1 is m2

    def test_overhead_without_c0_baseline(self):
        """Without C0_raw, overhead is None (omitted per spec)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
            a.entries = [
                _make_entry("C5_full_pea_balanced", 0.02, shots=65536),
            ]
            metrics = a.compute_derived_metrics()
            assert metrics["C5_full_pea_balanced"]["overhead_factor"] is None
