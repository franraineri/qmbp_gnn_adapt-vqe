"""Tests for MitigationBenchmarkAnalyzer.compute_sensitivity_curves() (task 8.5)."""

import json
import logging
import tempfile
from pathlib import Path

from project_health.analysis.mitigation_benchmark_analyzer import (
    MitigationBenchmarkAnalyzer,
)


def _entry(config_id: str, delta_e_gap: float, h_value: float = 3.5, mode: str = "fake_backend"):
    """Create a valid result envelope entry."""
    return {
        "benchmark_metadata": {
            "config_id": config_id,
            "h_value": h_value,
            "execution_mode": mode,
            "timestamp": "2026-06-18T10:30:00",
            "seed": 42,
        },
        "circuit_stats": {"depth_logical": 10, "depth_transpiled": 25, "n_2q_gates": 18},
        "results": {"e_raw": -12.5, "e_exact": -12.8, "delta_e_gap": delta_e_gap},
    }


def _setup_analyzer(entries: list[dict]) -> MitigationBenchmarkAnalyzer:
    """Create analyzer with entries written to a temp dir and scanned."""
    tmpdir = tempfile.mkdtemp()
    for i, entry in enumerate(entries):
        mode = entry["benchmark_metadata"]["execution_mode"]
        config_id = entry["benchmark_metadata"]["config_id"]
        d = Path(tmpdir) / mode / config_id
        d.mkdir(parents=True, exist_ok=True)
        (d / f"h350_run_{i:04d}.json").write_text(json.dumps(entry))
    analyzer = MitigationBenchmarkAnalyzer(results_dir=Path(tmpdir))
    analyzer.scan()
    return analyzer


# ═══════════════════════════════════════════════════════════════════════════════
# PEA budget curve tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestPEABudgetCurve:
    """Tests for PEA budget sensitivity curve (C4/C5/C6)."""

    def test_basic_three_config_curve(self):
        """Three PEA configs produce sorted data points by budget."""
        entries = [
            _entry("C4_full_pea_light", 0.05, h_value=3.25),
            _entry("C4_full_pea_light", 0.06, h_value=3.5),
            _entry("C5_full_pea_balanced", 0.03, h_value=3.25),
            _entry("C5_full_pea_balanced", 0.04, h_value=3.5),
            _entry("C6_full_pea_heavy", 0.02, h_value=3.25),
            _entry("C6_full_pea_heavy", 0.025, h_value=3.5),
        ]
        analyzer = _setup_analyzer(entries)
        result = analyzer.compute_sensitivity_curves()

        pea = result["pea_budget"]
        assert pea["sufficient_data"] is True
        assert len(pea["data_points"]) == 3
        # Sorted by budget: 4096 < 9216 < 16384
        budgets = [p[0] for p in pea["data_points"]]
        assert budgets == [4096.0, 9216.0, 16384.0]

    def test_mean_delta_computed_correctly(self):
        """Mean delta_e_gap is averaged across h-values per budget."""
        entries = [
            _entry("C4_full_pea_light", 0.04, h_value=3.25),
            _entry("C4_full_pea_light", 0.06, h_value=3.5),
            _entry("C5_full_pea_balanced", 0.02, h_value=3.25),
            _entry("C5_full_pea_balanced", 0.04, h_value=3.5),
            _entry("C6_full_pea_heavy", 0.01, h_value=3.25),
            _entry("C6_full_pea_heavy", 0.03, h_value=3.5),
        ]
        analyzer = _setup_analyzer(entries)
        result = analyzer.compute_sensitivity_curves()

        pea = result["pea_budget"]
        # C4: mean(0.04, 0.06) = 0.05
        # C5: mean(0.02, 0.04) = 0.03
        # C6: mean(0.01, 0.03) = 0.02
        assert abs(pea["data_points"][0][1] - 0.05) < 1e-10
        assert abs(pea["data_points"][1][1] - 0.03) < 1e-10
        assert abs(pea["data_points"][2][1] - 0.02) < 1e-10

    def test_insufficient_data_warns(self, caplog):
        """Warns when fewer than 3 PEA budget data points available."""
        entries = [
            _entry("C4_full_pea_light", 0.05),
            _entry("C5_full_pea_balanced", 0.03),
        ]
        analyzer = _setup_analyzer(entries)
        with caplog.at_level(logging.WARNING):
            result = analyzer.compute_sensitivity_curves()

        pea = result["pea_budget"]
        assert pea["sufficient_data"] is False
        assert len(pea["data_points"]) == 2
        assert "PEA budget sensitivity" in caplog.text

    def test_no_pea_entries_empty(self, caplog):
        """Returns empty curve with insufficient_data when no PEA entries."""
        entries = [
            _entry("C0_raw", 0.10),
            _entry("C1_dd_only", 0.08),
        ]
        analyzer = _setup_analyzer(entries)
        with caplog.at_level(logging.WARNING):
            result = analyzer.compute_sensitivity_curves()

        pea = result["pea_budget"]
        assert pea["sufficient_data"] is False
        assert pea["data_points"] == []

    def test_saturation_detected(self):
        """Saturation point detected when improvement/cost_delta < 0.1."""
        # Manufacture entries where C5→C6 improvement is tiny relative to cost
        # C4 budget=4096, delta=0.10
        # C5 budget=9216, delta=0.04 → improvement=0.06, cost=5120, ratio=0.0000117
        # C6 budget=16384, delta=0.039 → improvement=0.001, cost=7168, ratio=0.000000139
        # Both transitions have ratio < 0.1, so saturation at budget=9216
        entries = [
            _entry("C4_full_pea_light", 0.10),
            _entry("C5_full_pea_balanced", 0.04),
            _entry("C6_full_pea_heavy", 0.039),
        ]
        analyzer = _setup_analyzer(entries)
        result = analyzer.compute_sensitivity_curves()

        pea = result["pea_budget"]
        # First transition: |0.10-0.04|/5120 = 0.0000117 < 0.1 → saturation at 9216
        assert pea["saturation_point"] == 9216.0

    def test_no_saturation_when_large_improvement(self):
        """No saturation when all transitions have large improvement/cost ratio."""
        # Need |improvement|/cost_delta >= 0.1
        # C4 budget=4096, delta=1.0
        # C5 budget=9216, delta=0.0 → improvement=1.0, cost=5120, ratio=0.000195
        # Actually this is still < 0.1. Let's use extreme values.
        # To get ratio >= 0.1, need |improvement| >= 0.1 * cost_delta
        # C4→C5: cost=5120, need |imp| >= 512. Use delta=600, 0
        entries = [
            _entry("C4_full_pea_light", 600.0),
            _entry("C5_full_pea_balanced", 88.0),
            # improvement = 600-88=512, cost=5120, ratio=0.1 (exactly at threshold)
            _entry("C6_full_pea_heavy", 0.0),
            # improvement = 88-0=88, cost=7168, ratio ≈ 0.012 < 0.1
        ]
        analyzer = _setup_analyzer(entries)
        result = analyzer.compute_sensitivity_curves()

        pea = result["pea_budget"]
        # C4→C5: 512/5120 = 0.1 (not < 0.1, so NOT saturation)
        # C5→C6: 88/7168 ≈ 0.0123 < 0.1 → saturation at 16384
        assert pea["saturation_point"] == 16384.0


# ═══════════════════════════════════════════════════════════════════════════════
# Twirling curve tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTwirlingCurve:
    """Tests for twirling sensitivity curve."""

    def test_basic_twirling_curve(self):
        """Configs with different twirling values produce sorted curve."""
        entries = [
            _entry("C2_dd_tw", 0.06),  # twirling=32
            _entry("C5_full_pea_balanced", 0.03),  # twirling=48
            _entry("C6_full_pea_heavy", 0.025),  # twirling=64
        ]
        analyzer = _setup_analyzer(entries)
        result = analyzer.compute_sensitivity_curves()

        tw = result["twirling"]
        assert tw["sufficient_data"] is True
        assert len(tw["data_points"]) == 3
        # Sorted by twirling_num_randomizations: 32 < 48 < 64
        nums = [p[0] for p in tw["data_points"]]
        assert nums == [32.0, 48.0, 64.0]

    def test_twirling_mean_across_h_values(self):
        """Mean delta computed from multiple h-value entries per config."""
        entries = [
            _entry("C2_dd_tw", 0.05, h_value=3.25),
            _entry("C2_dd_tw", 0.07, h_value=3.5),
            _entry("C5_full_pea_balanced", 0.02, h_value=3.25),
            _entry("C5_full_pea_balanced", 0.04, h_value=3.5),
            _entry("C6_full_pea_heavy", 0.01, h_value=3.25),
            _entry("C6_full_pea_heavy", 0.03, h_value=3.5),
        ]
        analyzer = _setup_analyzer(entries)
        result = analyzer.compute_sensitivity_curves()

        tw = result["twirling"]
        # C2 (tw=32): mean(0.05, 0.07) = 0.06
        # C5 (tw=48): mean(0.02, 0.04) = 0.03
        # C6 (tw=64): mean(0.01, 0.03) = 0.02
        assert abs(tw["data_points"][0][1] - 0.06) < 1e-10
        assert abs(tw["data_points"][1][1] - 0.03) < 1e-10
        assert abs(tw["data_points"][2][1] - 0.02) < 1e-10

    def test_excludes_configs_without_twirling(self):
        """Configs with twirling=None are excluded from the curve."""
        entries = [
            _entry("C0_raw", 0.10),  # twirling=None (excluded)
            _entry("C1_dd_only", 0.08),  # twirling=None (excluded)
            _entry("C2_dd_tw", 0.06),  # twirling=32
            _entry("C5_full_pea_balanced", 0.03),  # twirling=48
            _entry("C6_full_pea_heavy", 0.025),  # twirling=64
        ]
        analyzer = _setup_analyzer(entries)
        result = analyzer.compute_sensitivity_curves()

        tw = result["twirling"]
        assert tw["sufficient_data"] is True
        assert len(tw["data_points"]) == 3
        # Only twirling values 32, 48, 64 should appear
        nums = [p[0] for p in tw["data_points"]]
        assert 0.0 not in nums  # None-twirling configs excluded

    def test_insufficient_twirling_data(self, caplog):
        """Warns when fewer than 3 twirling data points."""
        entries = [
            _entry("C2_dd_tw", 0.06),  # twirling=32
        ]
        analyzer = _setup_analyzer(entries)
        with caplog.at_level(logging.WARNING):
            result = analyzer.compute_sensitivity_curves()

        tw = result["twirling"]
        assert tw["sufficient_data"] is False
        assert "Twirling sensitivity" in caplog.text

    def test_aggregates_same_twirling_value(self):
        """Multiple configs with same twirling value are merged."""
        # C2_dd_tw and C3_full_gf both have twirling=32
        entries = [
            _entry("C2_dd_tw", 0.06),
            _entry("C3_full_gf", 0.04),
            _entry("C5_full_pea_balanced", 0.02),
            _entry("C6_full_pea_heavy", 0.01),
        ]
        analyzer = _setup_analyzer(entries)
        result = analyzer.compute_sensitivity_curves()

        tw = result["twirling"]
        # tw=32: mean(0.06, 0.04) = 0.05, tw=48: 0.02, tw=64: 0.01
        assert abs(tw["data_points"][0][1] - 0.05) < 1e-10
        assert tw["data_points"][0][0] == 32.0


# ═══════════════════════════════════════════════════════════════════════════════
# Saturation point logic tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSaturationPoint:
    """Tests for _find_saturation_point static method."""

    def test_no_saturation_with_empty_list(self):
        """Returns None for empty data points."""
        result = MitigationBenchmarkAnalyzer._find_saturation_point([])
        assert result is None

    def test_no_saturation_with_single_point(self):
        """Returns None for a single data point."""
        result = MitigationBenchmarkAnalyzer._find_saturation_point([(100.0, 0.05)])
        assert result is None

    def test_saturation_at_second_point(self):
        """Saturation detected at second point when ratio < 0.1."""
        # (100, 0.5) → (200, 0.49): improvement=0.01, cost=100, ratio=0.0001
        pts = [(100.0, 0.5), (200.0, 0.49)]
        result = MitigationBenchmarkAnalyzer._find_saturation_point(pts)
        assert result == 200.0

    def test_no_saturation_when_ratio_above_threshold(self):
        """No saturation when all ratios >= 0.1."""
        # (10, 100), (20, 99): improvement=1, cost=10, ratio=0.1 (not <0.1)
        pts = [(10.0, 100.0), (20.0, 99.0)]
        result = MitigationBenchmarkAnalyzer._find_saturation_point(pts)
        assert result is None

    def test_saturation_at_correct_transition(self):
        """Saturation occurs at the first point exceeding threshold."""
        # (10, 1.5), (20, 0.5): imp=1.0, cost=10, ratio=0.1 (NOT saturated)
        # (20, 0.5), (30, 0.49): imp=0.01, cost=10, ratio=0.001 (SATURATED)
        pts = [(10.0, 1.5), (20.0, 0.5), (30.0, 0.49)]
        result = MitigationBenchmarkAnalyzer._find_saturation_point(pts)
        assert result == 30.0


# ═══════════════════════════════════════════════════════════════════════════════
# Return structure tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSensitivityReturnStructure:
    """Tests for the overall return structure of compute_sensitivity_curves."""

    def test_returns_both_keys(self):
        """Result always has 'pea_budget' and 'twirling' keys."""
        analyzer = _setup_analyzer([])
        result = analyzer.compute_sensitivity_curves()
        assert "pea_budget" in result
        assert "twirling" in result

    def test_each_key_has_required_fields(self):
        """Each curve dict has data_points, saturation_point, sufficient_data."""
        analyzer = _setup_analyzer([])
        result = analyzer.compute_sensitivity_curves()

        for key in ("pea_budget", "twirling"):
            curve = result[key]
            assert "data_points" in curve
            assert "saturation_point" in curve
            assert "sufficient_data" in curve

    def test_empty_entries_gives_empty_curves(self):
        """With no entries, both curves are empty with sufficient_data=False."""
        analyzer = _setup_analyzer([])
        result = analyzer.compute_sensitivity_curves()

        assert result["pea_budget"]["data_points"] == []
        assert result["pea_budget"]["sufficient_data"] is False
        assert result["pea_budget"]["saturation_point"] is None
        assert result["twirling"]["data_points"] == []
        assert result["twirling"]["sufficient_data"] is False
        assert result["twirling"]["saturation_point"] is None
