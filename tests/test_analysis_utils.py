"""Unit tests for pure diagnostic computation functions in analysis_utils.py.

Tests are fast (no FakeTorino, no heavy imports). Covers:
- compute_snr
- compute_theta_smoothness
- compute_classification_confidence
- compute_energy_decomposition

**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7**
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.poc.v6.analysis_utils import (  # noqa: E402
    compute_classification_confidence,
    compute_energy_decomposition,
    compute_snr,
    compute_theta_smoothness,
)

# ─────────────────────────────────────────────────────────────────────────────
# TestComputeSNR
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeSNR:
    """Tests for compute_snr: |observable_value| * sqrt(shots)."""

    def test_snr_basic_computation(self):
        """compute_snr(1.0, 100) == 10.0."""
        assert compute_snr(1.0, 100) == pytest.approx(10.0, abs=1e-12)

    def test_snr_zero_observable(self):
        """compute_snr(0.0, 100) == 0.0."""
        assert compute_snr(0.0, 100) == pytest.approx(0.0, abs=1e-12)

    def test_snr_negative_observable(self):
        """compute_snr(-0.5, 4) == 1.0 (absolute value used)."""
        # |-0.5| * sqrt(4) = 0.5 * 2 = 1.0
        assert compute_snr(-0.5, 4) == pytest.approx(1.0, abs=1e-12)

    def test_snr_shots_1(self):
        """compute_snr(1.0, 1) == 1.0 (minimum valid shots)."""
        assert compute_snr(1.0, 1) == pytest.approx(1.0, abs=1e-12)

    def test_snr_large_shots(self):
        """compute_snr(0.01, 1_000_000) is finite and correct."""
        expected = 0.01 * np.sqrt(1_000_000)  # 0.01 * 1000 = 10.0
        result = compute_snr(0.01, 1_000_000)
        assert np.isfinite(result)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_snr_very_small_observable(self):
        """compute_snr(1e-10, 4096) is non-negative."""
        result = compute_snr(1e-10, 4096)
        assert result >= 0.0
        assert result == pytest.approx(1e-10 * np.sqrt(4096), abs=1e-20)

    def test_snr_monotonicity(self):
        """For fixed v≠0, shots_1 < shots_2 implies snr_1 < snr_2."""
        v = 0.7
        shots_values = [10, 100, 1000, 10000]
        snr_values = [compute_snr(v, s) for s in shots_values]
        for i in range(len(snr_values) - 1):
            assert snr_values[i] < snr_values[i + 1]

    def test_snr_raises_on_zero_shots(self):
        """shots=0 raises ValueError."""
        with pytest.raises(ValueError):
            compute_snr(1.0, 0)

    def test_snr_raises_on_negative_shots(self):
        """shots=-1 raises ValueError."""
        with pytest.raises(ValueError):
            compute_snr(1.0, -1)

    def test_snr_raises_on_float_shots(self):
        """shots=3.14 raises ValueError (not int)."""
        with pytest.raises(ValueError):
            compute_snr(1.0, 3.14)

    def test_snr_raises_on_none_shots(self):
        """shots=None raises TypeError or ValueError."""
        with pytest.raises((TypeError, ValueError)):
            compute_snr(1.0, None)

    def test_snr_numpy_int_shots(self):
        """shots=np.int64(100) should work (numpy integer)."""
        result = compute_snr(1.0, np.int64(100))
        assert result == pytest.approx(10.0, abs=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeThetaSmoothness
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeThetaSmoothness:
    """Tests for compute_theta_smoothness: max_i ||θ_i - θ_{i-1}||_∞."""

    def test_smoothness_basic(self):
        """Known array → known result."""
        # Three vectors: diffs are [0.1, 0.1, 0.1, 0.1] and [1.0, 0.0, 0.0, 0.0]
        theta = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.1, 0.1, 0.1, 0.1],  # max diff from prev = 0.1
                [1.1, 0.1, 0.1, 0.1],  # max diff from prev = 1.0
            ]
        )
        result = compute_theta_smoothness(theta)
        assert result == pytest.approx(1.0, abs=1e-12)

    def test_smoothness_single_point_returns_none(self):
        """Shape (1, 4) → None."""
        theta = np.array([[0.1, 0.2, 0.3, 0.4]])
        result = compute_theta_smoothness(theta)
        assert result is None

    def test_smoothness_empty_returns_none(self):
        """Shape (0, 4) → None (or handle gracefully)."""
        theta = np.empty((0, 4))
        result = compute_theta_smoothness(theta)
        assert result is None

    def test_smoothness_identical_vectors(self):
        """All rows same → 0.0."""
        theta = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0],
            ]
        )
        result = compute_theta_smoothness(theta)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_smoothness_two_points(self):
        """Shape (2, 4) → single diff."""
        theta = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.3, 0.7, 0.1],
            ]
        )
        # max(|0.5|, |0.3|, |0.7|, |0.1|) = 0.7
        result = compute_theta_smoothness(theta)
        assert result == pytest.approx(0.7, abs=1e-12)

    def test_smoothness_non_negative(self):
        """Result is always ≥ 0."""
        rng = np.random.default_rng(42)
        theta = rng.standard_normal((10, 8))
        result = compute_theta_smoothness(theta)
        assert result >= 0.0

    def test_smoothness_uses_infinity_norm(self):
        """Verify it's max(abs(diff)) not L2."""
        # If L2 were used, result would be sqrt(0.01 + 0.01 + 0.01 + 1.0) ≈ 1.015
        # With infinity norm, result is 1.0
        theta = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.1, 0.1, 0.1, 1.0],
            ]
        )
        result = compute_theta_smoothness(theta)
        assert result == pytest.approx(1.0, abs=1e-12)
        # Confirm it's NOT the L2 norm
        l2_norm = np.sqrt(0.01 + 0.01 + 0.01 + 1.0)
        assert result != pytest.approx(l2_norm, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeClassificationConfidence
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeClassificationConfidence:
    """Tests for compute_classification_confidence: |mag_x - corr_zz| * sqrt(shots)."""

    def test_confidence_basic(self):
        """compute_classification_confidence(0.8, 0.2, 100) == 6.0."""
        # |0.8 - 0.2| * sqrt(100) = 0.6 * 10 = 6.0
        result = compute_classification_confidence(0.8, 0.2, 100)
        assert result == pytest.approx(6.0, abs=1e-12)

    def test_confidence_zero_difference(self):
        """mag_x == corr_zz → 0.0."""
        result = compute_classification_confidence(0.5, 0.5, 4096)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_confidence_negative_values(self):
        """Works with negative corr_zz."""
        # |0.3 - (-0.4)| * sqrt(16) = 0.7 * 4 = 2.8
        result = compute_classification_confidence(0.3, -0.4, 16)
        assert result == pytest.approx(2.8, abs=1e-12)

    def test_confidence_raises_on_zero_shots(self):
        """ValueError on shots=0."""
        with pytest.raises(ValueError):
            compute_classification_confidence(0.5, 0.3, 0)

    def test_confidence_raises_on_negative_shots(self):
        """ValueError on shots=-1."""
        with pytest.raises(ValueError):
            compute_classification_confidence(0.5, 0.3, -1)

    def test_confidence_raises_on_float_shots(self):
        """ValueError on shots=3.14."""
        with pytest.raises(ValueError):
            compute_classification_confidence(0.5, 0.3, 3.14)

    def test_confidence_symmetry(self):
        """|mag_x - corr_zz| is symmetric in sign swap."""
        # |a - b| == |b - a|
        r1 = compute_classification_confidence(0.8, 0.2, 100)
        r2 = compute_classification_confidence(0.2, 0.8, 100)
        assert r1 == pytest.approx(r2, abs=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# TestComputeEnergyDecomposition
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeEnergyDecomposition:
    """Tests for compute_energy_decomposition: error attribution."""

    def test_decomposition_basic(self):
        """Known values → known result."""
        result = compute_energy_decomposition(
            e_exact=-9.25,
            e_vqe_ceiling=-9.22,
            e_predicted=-9.15,
        )
        assert result["e_exact"] == pytest.approx(-9.25, abs=1e-12)
        assert result["e_vqe_ceiling"] == pytest.approx(-9.22, abs=1e-12)
        assert result["e_mpnn_predicted"] == pytest.approx(-9.15, abs=1e-12)
        assert result["error_from_circuit"] == pytest.approx(0.03, abs=1e-12)
        assert result["error_from_mpnn"] == pytest.approx(0.07, abs=1e-12)

    def test_decomposition_all_equal(self):
        """e_exact == e_vqe == e_pred → all errors 0."""
        result = compute_energy_decomposition(-5.0, -5.0, -5.0)
        assert result["error_from_circuit"] == pytest.approx(0.0, abs=1e-12)
        assert result["error_from_mpnn"] == pytest.approx(0.0, abs=1e-12)

    def test_decomposition_additivity_invariant(self):
        """error_from_circuit + error_from_mpnn == |e_pred - e_exact| within 1e-12."""
        result = compute_energy_decomposition(
            e_exact=-10.0,
            e_vqe_ceiling=-9.8,
            e_predicted=-9.5,
        )
        total_error = abs(result["e_mpnn_predicted"] - result["e_exact"])
        sum_errors = result["error_from_circuit"] + result["error_from_mpnn"]
        assert sum_errors == pytest.approx(total_error, abs=1e-12)

    def test_decomposition_keys(self):
        """Result has all 5 expected keys."""
        result = compute_energy_decomposition(-5.0, -4.9, -4.8)
        expected_keys = {
            "e_exact",
            "e_vqe_ceiling",
            "e_mpnn_predicted",
            "error_from_circuit",
            "error_from_mpnn",
        }
        assert set(result.keys()) == expected_keys

    def test_decomposition_non_negative_errors(self):
        """Both error components ≥ 0."""
        result = compute_energy_decomposition(-9.25, -9.22, -9.15)
        assert result["error_from_circuit"] >= 0.0
        assert result["error_from_mpnn"] >= 0.0

    def test_decomposition_large_values(self):
        """Works with energies like -1000.0."""
        result = compute_energy_decomposition(-1000.0, -999.5, -998.0)
        assert result["error_from_circuit"] == pytest.approx(0.5, abs=1e-12)
        assert result["error_from_mpnn"] == pytest.approx(1.5, abs=1e-12)
        # Additivity
        total = abs(result["e_mpnn_predicted"] - result["e_exact"])
        assert result["error_from_circuit"] + result["error_from_mpnn"] == pytest.approx(
            total, abs=1e-12
        )

    def test_decomposition_small_differences(self):
        """Works with differences < 1e-10."""
        result = compute_energy_decomposition(-5.0, -5.0 + 1e-11, -5.0 + 2e-11)
        assert result["error_from_circuit"] >= 0.0
        assert result["error_from_mpnn"] >= 0.0
        total = abs(result["e_mpnn_predicted"] - result["e_exact"])
        sum_errors = result["error_from_circuit"] + result["error_from_mpnn"]
        assert sum_errors == pytest.approx(total, abs=1e-12)
