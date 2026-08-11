"""Unit tests for qmbp_simulation.analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.analysis import (
    compute_energy_decomposition,
    compute_snr,
    compute_theta_smoothness,
)


class TestComputeSNR:
    """Test compute_snr returns ≥ 0."""

    def test_positive_observable(self):
        snr = compute_snr(0.5, 1000)
        assert snr >= 0.0
        assert np.isclose(snr, 0.5 * np.sqrt(1000))

    def test_negative_observable(self):
        snr = compute_snr(-0.3, 8192)
        assert snr >= 0.0

    def test_zero_observable(self):
        snr = compute_snr(0.0, 1000)
        assert snr == 0.0

    def test_invalid_shots_raises(self):
        with pytest.raises(ValueError):
            compute_snr(0.5, 0)
        with pytest.raises(ValueError):
            compute_snr(0.5, -10)


class TestComputeThetaSmoothness:
    """Test compute_theta_smoothness returns ≥ 0 or None."""

    def test_smooth_parameters(self):
        # Slowly varying parameters
        theta = np.array([[0.1, 0.2], [0.12, 0.21], [0.14, 0.22]])
        result = compute_theta_smoothness(theta)
        assert result is not None
        assert result >= 0.0

    def test_single_point_returns_none(self):
        theta = np.array([[0.1, 0.2]])
        result = compute_theta_smoothness(theta)
        assert result is None

    def test_discontinuous_parameters(self):
        # Large jump between points
        theta = np.array([[0.1, 0.2], [3.0, -2.0], [0.15, 0.25]])
        result = compute_theta_smoothness(theta)
        assert result is not None
        assert result > 2.0  # Should detect the large jump


class TestComputeEnergyDecomposition:
    """Test compute_energy_decomposition components sum correctly."""

    def test_components_sum_to_total_error(self):
        result = compute_energy_decomposition(e_exact=-5.0, e_vqe_ceiling=-4.8, e_predicted=-4.5)
        total_error = abs(result["e_mpnn_predicted"] - result["e_exact"])
        component_sum = result["error_from_circuit"] + result["error_from_mpnn"]
        np.testing.assert_allclose(total_error, component_sum, atol=1e-12)

    def test_all_values_nonnegative(self):
        result = compute_energy_decomposition(e_exact=-5.0, e_vqe_ceiling=-4.9, e_predicted=-4.7)
        assert result["error_from_circuit"] >= 0.0
        assert result["error_from_mpnn"] >= 0.0

    def test_perfect_prediction(self):
        result = compute_energy_decomposition(e_exact=-5.0, e_vqe_ceiling=-5.0, e_predicted=-5.0)
        assert result["error_from_circuit"] == 0.0
        assert result["error_from_mpnn"] == 0.0

    def test_circuit_error_dominates_when_vqe_far(self):
        result = compute_energy_decomposition(e_exact=-5.0, e_vqe_ceiling=-3.0, e_predicted=-3.0)
        # If MPNN perfectly predicts VQE ceiling, all error is from circuit
        assert result["error_from_circuit"] > 0.0
        assert result["error_from_mpnn"] == 0.0


class TestDiagnosticCollector:
    """Test DiagnosticCollector basic functionality."""

    def test_collector_initializes_empty(self, tmp_path):
        from qmbp_simulation.analysis import DiagnosticCollector

        collector = DiagnosticCollector(save_dir=tmp_path)
        data = collector.to_dict()
        assert isinstance(data, dict)
        assert "phase1" in data
        assert "phase2" in data

    def test_record_phase1_stores_data(self, tmp_path):
        from qmbp_simulation.analysis import DiagnosticCollector

        collector = DiagnosticCollector(save_dir=tmp_path)
        collector.record_phase1(
            n_points=5,
            elapsed_s=1.2,
            gap_min=0.3,
        )
        data = collector.to_dict()
        assert data["phase1"]["n_points"] == 5
        assert data["phase1"]["elapsed_s"] == 1.2
        assert data["phase1"]["gap_min"] == 0.3

    def test_record_vqe_point_stores_data(self, tmp_path):
        from qmbp_simulation.analysis import DiagnosticCollector

        collector = DiagnosticCollector(save_dir=tmp_path)
        collector.record_vqe_point(
            h=1.5,
            n_iters=50,
            restart_energies=[-4.5, -4.3, -4.4],
            theta_opt=np.array([0.1, 0.2]),
            elapsed_s=2.5,
        )
        data = collector.to_dict()
        assert len(data["phase2"]["per_h_iterations"]) == 1
        assert data["phase2"]["per_h_iterations"][0] == 50


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Integration Tests — Unified Scaling Analysis
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeScalabilityScore:
    """Test compute_scalability_score for unified topology scoring."""

    def test_excellent_scaling(self):
        from qmbp_simulation.analysis.metrics import compute_scalability_score

        score, reason = compute_scalability_score(
            'chain_1d', n_max_viable=20, pass_rate_dual=0.95, h_frontier=2.5
        )
        assert 0.8 <= score <= 1.0
        assert reason == 'excellent_scaling'

    def test_poor_scaling_limited_n(self):
        from qmbp_simulation.analysis.metrics import compute_scalability_score

        score, reason = compute_scalability_score(
            'triangular', n_max_viable=4, pass_rate_dual=0.30, h_frontier=4.0
        )
        assert score < 0.4
        assert 'limited' in reason

    def test_moderate_scaling(self):
        from qmbp_simulation.analysis.metrics import compute_scalability_score

        score, reason = compute_scalability_score(
            'ladder', n_max_viable=12, pass_rate_dual=0.70, h_frontier=3.0
        )
        assert 0.5 <= score <= 0.85
        assert reason in ('moderate_scaling', 'excellent_scaling')

    def test_none_n_max_viable(self):
        from qmbp_simulation.analysis.metrics import compute_scalability_score

        score, reason = compute_scalability_score(
            'unknown', n_max_viable=None, pass_rate_dual=0.50, h_frontier=3.0
        )
        # Score should be low when n_max_viable is unknown
        assert score < 0.5

    def test_none_h_frontier(self):
        from qmbp_simulation.analysis.metrics import compute_scalability_score

        score, reason = compute_scalability_score(
            'chain_1d', n_max_viable=16, pass_rate_dual=0.80, h_frontier=None
        )
        # Should still compute with default h_factor=0.5
        assert 0.4 <= score <= 0.9


class TestComputeTrainingReadiness:
    """Test compute_training_readiness for training data quality assessment."""

    def test_not_ready_low_verified(self):
        from qmbp_simulation.analysis.metrics import compute_training_readiness

        tier_breakdown = {
            'file1.npz': {'verified': 0, 'approximate': 10, 'unverified': 90, 'total': 100},
            'file2.npz': {'verified': 5, 'approximate': 15, 'unverified': 80, 'total': 100},
        }
        utility = {'useful': [1, 2, 3], 'insufficient_signal': [], 'not_useful': []}
        ready, reason, stats = compute_training_readiness(tier_breakdown, utility)
        assert ready is False
        assert 'verified_ratio_too_low' in reason
        assert stats['verified_ratio'] < 0.30

    def test_ready_high_verified(self):
        from qmbp_simulation.analysis.metrics import compute_training_readiness

        tier_breakdown = {
            'file1.npz': {'verified': 60, 'approximate': 30, 'unverified': 10, 'total': 100},
        }
        utility = {'useful': [1, 2, 3, 4], 'insufficient_signal': [], 'not_useful': []}
        ready, reason, stats = compute_training_readiness(tier_breakdown, utility)
        assert ready is True
        assert reason == 'ready'

    def test_not_ready_more_not_useful(self):
        from qmbp_simulation.analysis.metrics import compute_training_readiness

        tier_breakdown = {
            'file1.npz': {'verified': 50, 'approximate': 30, 'unverified': 20, 'total': 100},
        }
        utility = {'useful': [1], 'insufficient_signal': [], 'not_useful': [2, 3, 4]}
        ready, reason, stats = compute_training_readiness(tier_breakdown, utility)
        assert ready is False
        assert 'more_not_useful' in reason

    def test_no_data_available(self):
        from qmbp_simulation.analysis.metrics import compute_training_readiness

        ready, reason, stats = compute_training_readiness(None, None)
        assert ready is False
        assert 'no_quality_data' in reason

    def test_legacy_npz_warning(self):
        from qmbp_simulation.analysis.metrics import compute_training_readiness

        tier_breakdown = {
            'file1.npz': {'verified': 0, 'approximate': 0, 'unverified': 20, 'total': 20, 'legacy': True},
            'file2.npz': {'verified': 40, 'approximate': 10, 'unverified': 0, 'total': 50},
        }
        utility = {'useful': [1, 2, 3, 4], 'insufficient_signal': [], 'not_useful': []}
        ready, reason, stats = compute_training_readiness(tier_breakdown, utility)
        assert ready is True
        assert 'legacy' in reason


class TestComputeExtrapolationViability:
    """Test compute_extrapolation_viability for large-N prediction assessment."""

    def test_within_viable_range(self):
        from qmbp_simulation.analysis.metrics import compute_extrapolation_viability

        viable, reason, pred = compute_extrapolation_viability(
            'chain_1d', n_max_viable=20, mean_de_gap_per_n=None, target_n=15
        )
        assert viable is True
        assert 'within_viable' in reason

    def test_far_beyond_n_max(self):
        from qmbp_simulation.analysis.metrics import compute_extrapolation_viability

        viable, reason, pred = compute_extrapolation_viability(
            'chain_1d', n_max_viable=10, mean_de_gap_per_n=None, target_n=50
        )
        assert viable is False
        assert 'far_beyond' in reason

    def test_no_cross_n_data(self):
        from qmbp_simulation.analysis.metrics import compute_extrapolation_viability

        viable, reason, pred = compute_extrapolation_viability(
            'unknown', n_max_viable=None, mean_de_gap_per_n=None, target_n=30
        )
        assert viable is False
        assert 'no_cross_n_data' in reason

    def test_moderately_beyond(self):
        from qmbp_simulation.analysis.metrics import compute_extrapolation_viability

        viable, reason, pred = compute_extrapolation_viability(
            'chain_1d', n_max_viable=20, mean_de_gap_per_n=None, target_n=28
        )
        # 28 <= 1.5 * 20 = 30, so should be viable
        assert viable is True
        assert 'moderately_beyond' in reason

    def test_with_trend_data_viable(self):
        from qmbp_simulation.analysis.metrics import compute_extrapolation_viability

        mean_dg = {6: 0.08, 10: 0.06, 16: 0.05, 20: 0.04}
        viable, reason, pred = compute_extrapolation_viability(
            'chain_1d', n_max_viable=20, mean_de_gap_per_n=mean_dg, target_n=30
        )
        # Trend suggests decreasing de_gap, extrapolation should be favorable
        assert viable is True
        assert 'extrapolated_de_gap' in pred

    def test_with_trend_data_not_viable(self):
        from qmbp_simulation.analysis.metrics import compute_extrapolation_viability

        mean_dg = {6: 0.05, 10: 0.10, 16: 0.15, 20: 0.20}
        viable, reason, pred = compute_extrapolation_viability(
            'chain_1d', n_max_viable=20, mean_de_gap_per_n=mean_dg, target_n=40
        )
        # Trend suggests increasing de_gap, extrapolation unlikely to work
        assert viable is False
        assert 'above_threshold' in reason


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for unified dual criterion (pass_rate_dual)
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeDeploySummaryDual:
    """Test compute_deploy_summary returns correct pass_rate_dual."""

    def test_dual_with_abs_error_available(self):
        from qmbp_simulation.analysis.metrics import compute_deploy_summary

        results = [
            {"de_gap": 0.03, "abs_error": 0.05},   # pass both
            {"de_gap": 0.04, "abs_error": 0.08},   # pass both
            {"de_gap": 0.03, "abs_error": 0.15},   # pass 5pct, FAIL dual
            {"de_gap": 0.06, "abs_error": 0.03},   # FAIL 5pct, pass abs
            {"de_gap": 0.02, "abs_error": 0.02},   # pass both
        ]
        s = compute_deploy_summary(results)

        assert s["pass_rate_5pct"] == pytest.approx(0.8)
        assert s["pass_rate_dual"] == pytest.approx(0.6)
        assert s["n_pass_5pct"] == 4
        assert s["n_pass_dual"] == 3

    def test_dual_fallback_when_no_abs_error(self):
        from qmbp_simulation.analysis.metrics import compute_deploy_summary

        results = [
            {"de_gap": 0.03},
            {"de_gap": 0.06},
        ]
        s = compute_deploy_summary(results)

        # Without abs_error, dual falls back to single criterion
        assert s["pass_rate_dual"] == s["pass_rate_5pct"]
        assert s["n_pass_dual"] == s["n_pass_5pct"]

    def test_dual_gap_masking_detected(self):
        """Gap masking: small de_gap but large abs_error due to big gap."""
        from qmbp_simulation.analysis.metrics import compute_deploy_summary

        # Simulate h >> h_c: gap = 10, |ΔE| = 0.4, so de_gap = 0.04 < 5%
        # But |ΔE| = 0.4 > 0.10 → should FAIL dual
        results = [
            {"de_gap": 0.04, "abs_error": 0.40},  # gap masked
            {"de_gap": 0.03, "abs_error": 0.30},  # gap masked
            {"de_gap": 0.02, "abs_error": 0.20},  # gap masked
        ]
        s = compute_deploy_summary(results)

        assert s["pass_rate_5pct"] == pytest.approx(1.0)  # all pass single
        assert s["pass_rate_dual"] == pytest.approx(0.0)  # all fail dual
        assert s["n_pass_dual"] == 0

    def test_empty_results(self):
        from qmbp_simulation.analysis.metrics import compute_deploy_summary

        s = compute_deploy_summary([])
        assert s["n_points"] == 0
        assert s["pass_rate_5pct"] == 0.0

    def test_single_point_pass(self):
        from qmbp_simulation.analysis.metrics import compute_deploy_summary

        s = compute_deploy_summary([{"de_gap": 0.01, "abs_error": 0.005}])
        assert s["pass_rate_dual"] == pytest.approx(1.0)
        assert s["n_pass_dual"] == 1

    def test_single_point_fail_de_gap(self):
        from qmbp_simulation.analysis.metrics import compute_deploy_summary

        s = compute_deploy_summary([{"de_gap": 0.06, "abs_error": 0.005}])
        assert s["pass_rate_dual"] == pytest.approx(0.0)

    def test_single_point_fail_abs_error(self):
        from qmbp_simulation.analysis.metrics import compute_deploy_summary

        s = compute_deploy_summary([{"de_gap": 0.01, "abs_error": 0.15}])
        assert s["pass_rate_dual"] == pytest.approx(0.0)

    def test_boundary_values_exactly_at_threshold(self):
        """Points exactly at threshold boundaries."""
        from qmbp_simulation.analysis.metrics import compute_deploy_summary

        results = [
            {"de_gap": 0.05, "abs_error": 0.10},   # at boundary → FAIL (>=)
            {"de_gap": 0.049, "abs_error": 0.099},  # just below → PASS
        ]
        s = compute_deploy_summary(results)
        assert s["n_pass_dual"] == 1
        assert s["pass_rate_dual"] == pytest.approx(0.5)

    def test_partial_abs_error_data(self):
        """When only some results have abs_error."""
        from qmbp_simulation.analysis.metrics import compute_deploy_summary

        results = [
            {"de_gap": 0.03, "abs_error": 0.05},
            {"de_gap": 0.03},  # no abs_error
            {"de_gap": 0.03, "abs_error": 0.08},
        ]
        s = compute_deploy_summary(results)

        # Only 2 of 3 have abs_error → fallback (abs_errors length != n)
        assert s["n_pass_dual"] == s["n_pass_5pct"]


class TestIsPointFailureDual:
    """Test is_point_failure implements dual criterion correctly."""

    def test_pass_both_criteria(self):
        from qmbp_simulation.analysis.metrics import is_point_failure

        assert is_point_failure(de_gap=0.03, abs_error=0.05) is False

    def test_fail_de_gap_only(self):
        from qmbp_simulation.analysis.metrics import is_point_failure

        assert is_point_failure(de_gap=0.06, abs_error=0.05) is True

    def test_fail_abs_error_only(self):
        from qmbp_simulation.analysis.metrics import is_point_failure

        assert is_point_failure(de_gap=0.03, abs_error=0.15) is True

    def test_fail_both(self):
        from qmbp_simulation.analysis.metrics import is_point_failure

        assert is_point_failure(de_gap=0.10, abs_error=0.50) is True

    def test_no_abs_error_passes_if_de_gap_ok(self):
        """Without abs_error, only de_gap is checked."""
        from qmbp_simulation.analysis.metrics import is_point_failure

        assert is_point_failure(de_gap=0.03, abs_error=None) is False

    def test_no_abs_error_fails_if_de_gap_bad(self):
        from qmbp_simulation.analysis.metrics import is_point_failure

        assert is_point_failure(de_gap=0.06, abs_error=None) is True

    def test_nan_abs_error_is_failure(self):
        from qmbp_simulation.analysis.metrics import is_point_failure

        assert is_point_failure(de_gap=0.03, abs_error=float("nan")) is True

    def test_zero_de_gap_and_zero_abs_error(self):
        """Perfect result should pass."""
        from qmbp_simulation.analysis.metrics import is_point_failure

        assert is_point_failure(de_gap=0.0, abs_error=0.0) is False


class TestAcceleratedResultDualPassRate:
    """Test AcceleratedResult.pass_rate uses dual criterion."""

    def test_pass_rate_uses_dual(self):
        from qmbp_simulation.pipeline.accelerated import AcceleratedResult

        ar = AcceleratedResult(
            h_values=np.array([1.0, 2.0, 3.0]),
            theta_opt=np.zeros((3, 5)),
            energies=np.array([-10.0, -8.0, -6.0]),
            de_gaps=np.array([0.02, 0.04, 0.06]),
            gaps=np.array([1.0, 1.0, 1.0]),
            e_exact=np.array([-10.05, -8.05, -6.15]),
            method=["vqe", "mpnn", "mpnn"],
        )
        # pt1: de_gap=0.02<0.05, |ΔE|=0.05<0.10 → pass
        # pt2: de_gap=0.04<0.05, |ΔE|=0.05<0.10 → pass
        # pt3: de_gap=0.06>=0.05 → fail
        assert ar.pass_rate == pytest.approx(2 / 3, abs=0.01)

    def test_gap_masking_detected(self):
        """Large gap artificially reduces de_gap but abs_error is still big."""
        from qmbp_simulation.pipeline.accelerated import AcceleratedResult

        ar = AcceleratedResult(
            h_values=np.array([3.0, 4.0, 5.0]),
            theta_opt=np.zeros((3, 5)),
            energies=np.array([-5.8, -7.6, -9.5]),
            de_gaps=np.array([0.03, 0.02, 0.04]),  # all < 5%
            gaps=np.array([5.0, 7.0, 9.0]),
            e_exact=np.array([-6.0, -7.8, -9.9]),  # |ΔE| = 0.2, 0.2, 0.4 > 0.10
            method=["mpnn", "mpnn", "mpnn"],
        )
        # Old metric would say 100%, dual says 0%
        assert ar.pass_rate == pytest.approx(0.0)

    def test_empty_result(self):
        from qmbp_simulation.pipeline.accelerated import AcceleratedResult

        ar = AcceleratedResult(
            h_values=np.array([]),
            theta_opt=np.zeros((0, 5)),
            energies=np.array([]),
            de_gaps=np.array([]),
            gaps=np.array([]),
            e_exact=np.array([]),
            method=[],
        )
        assert ar.pass_rate == 0.0


class TestDashboardHasDualField:
    """Verify generate_model_quality_dashboard includes pass_rate_dual_criterion."""

    def test_dashboard_config_has_dual_field(self):
        """Dashboard configs must include pass_rate_dual_criterion."""
        from qmbp_simulation.analysis.metrics import generate_model_quality_dashboard
        from pathlib import Path
        import json

        # Read the existing dashboard (don't regenerate — slow + needs NPZ)
        root = Path(__file__).resolve().parents[2]
        dashboard_path = root / "data" / "model_quality_dashboard.json"
        if not dashboard_path.exists():
            pytest.skip("Dashboard not found (needs NPZ data)")

        dashboard = json.load(open(dashboard_path))
        configs = dashboard.get("configs", [])
        assert len(configs) > 0, "Dashboard has no configs"

        # Every config must have pass_rate_dual_criterion
        for c in configs:
            assert "pass_rate_dual_criterion" in c, (
                f"Config {c.get('topology')} N={c.get('n_qubits')} "
                f"missing pass_rate_dual_criterion"
            )
            # And it must be <= pass_rate_5pct (dual is always stricter)
            dual = c["pass_rate_dual_criterion"]
            single = c.get("pass_rate_5pct", 0)
            assert dual <= single + 1e-10, (
                f"Config {c.get('topology')} N={c.get('n_qubits')}: "
                f"dual={dual:.3f} > single={single:.3f} (impossible)"
            )

    def test_topology_summary_has_dual_field(self):
        """Topology summary must include best_pass_rate_dual."""
        from pathlib import Path
        import json

        root = Path(__file__).resolve().parents[2]
        dashboard_path = root / "data" / "model_quality_dashboard.json"
        if not dashboard_path.exists():
            pytest.skip("Dashboard not found")

        dashboard = json.load(open(dashboard_path))
        topo_sum = dashboard.get("topology_summary", {})
        assert len(topo_sum) > 0

        for topo, info in topo_sum.items():
            assert "best_pass_rate_dual" in info, (
                f"topology_summary[{topo}] missing best_pass_rate_dual"
            )


class TestMetricVersionInEnvelope:
    """Verify result envelopes carry metric_version for traceability."""

    def test_metric_version_present(self):
        """New runs must include metric_version in summary."""
        # We can't easily run a full runner in a unit test, but we can
        # check that the field is set in the code by inspecting the dict
        # construction.
        import ast
        from pathlib import Path

        runner_base = Path("src/qmbp_simulation/framework/runner_base.py")
        source = runner_base.read_text()
        assert '"metric_version": "dual_v1"' in source, (
            "runner_base.py must include metric_version in summary dicts"
        )
