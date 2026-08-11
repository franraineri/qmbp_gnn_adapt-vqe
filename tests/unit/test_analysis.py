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
