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
