"""Unit tests for ThetaValidator — θ_pred quality assurance module.

Tests each validation level independently and the orchestrator together.
Lightweight: uses synthetic data (no VQE/MPNN execution needed).
"""

from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.analysis.theta_validator import (
    ThetaValidator,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def training_data():
    """Synthetic training θ_opt: smooth linear variation over 10 h-points."""
    rng = np.random.default_rng(42)
    n_points = 10
    n_params = 4
    h_values = np.linspace(2.0, 1.0, n_points)  # descending

    # Smooth θ_opt: linear in h with small noise
    theta_opt = np.zeros((n_points, n_params))
    for i in range(n_params):
        theta_opt[:, i] = 0.5 * h_values + 0.1 * i + rng.normal(0, 0.01, n_points)

    return theta_opt, h_values


@pytest.fixture
def validator(training_data):
    theta_opt, h_values = training_data
    return ThetaValidator.from_training_data(theta_opt, h_values)


# ── L1: Bound Check Tests ────────────────────────────────────────────────────


class TestBoundCheck:
    def test_good_prediction_passes(self, validator, training_data):
        """θ_pred within training range should pass."""
        theta_opt, h_values = training_data
        # Predict something within the training distribution
        theta_pred = theta_opt[5] + np.array([0.01, -0.01, 0.02, -0.005])
        result = validator.check_bounds(theta_pred)
        assert result.passed is True
        assert result.n_out_of_bounds == 0

    def test_extreme_prediction_fails(self, validator):
        """θ_pred far outside training range should fail."""
        theta_pred = np.array([100.0, -100.0, 50.0, -50.0])
        result = validator.check_bounds(theta_pred)
        assert result.passed is False
        assert result.n_out_of_bounds == 4

    def test_partial_oob(self, validator, training_data):
        """Only some params out of bounds."""
        theta_opt, _ = training_data
        theta_pred = theta_opt[3].copy()
        theta_pred[0] = 999.0  # Only first param OOB
        result = validator.check_bounds(theta_pred)
        assert result.passed is False
        assert result.n_out_of_bounds == 1
        assert 0 in result.out_of_bounds_indices


# ── L2: Numerical Sanity Tests ───────────────────────────────────────────────


class TestNumericalSanity:
    def test_clean_array_passes(self, validator):
        theta_pred = np.array([0.5, -0.3, 0.7, 0.1])
        result = validator.check_numerical_sanity(theta_pred)
        assert result.passed is True
        assert not result.has_nan
        assert not result.has_inf

    def test_nan_detected(self, validator):
        theta_pred = np.array([0.5, np.nan, 0.7, 0.1])
        result = validator.check_numerical_sanity(theta_pred)
        assert result.passed is False
        assert result.has_nan
        assert 1 in result.nan_indices

    def test_inf_detected(self, validator):
        theta_pred = np.array([0.5, 0.3, np.inf, 0.1])
        result = validator.check_numerical_sanity(theta_pred)
        assert result.passed is False
        assert result.has_inf
        assert 2 in result.inf_indices


# ── L3: Interpolation Consistency Tests ──────────────────────────────────────


class TestInterpolation:
    def test_nearby_prediction_passes(self, validator, training_data):
        """θ_pred close to training manifold should pass."""
        theta_opt, _ = training_data
        theta_pred = theta_opt[4] + np.random.default_rng(0).normal(0, 0.01, 4)
        result = validator.check_interpolation(theta_pred)
        assert result.passed is True
        assert result.ratio < 2.0

    def test_distant_prediction_fails(self, validator, training_data):
        """θ_pred far from training manifold should fail."""
        theta_opt, _ = training_data
        theta_pred = theta_opt[4] + 10.0  # Large offset
        result = validator.check_interpolation(theta_pred, threshold=2.0)
        assert result.passed is False
        assert result.ratio > 2.0

    def test_no_training_data_passes(self):
        """Without training data, interpolation check is skipped (passes)."""
        validator = ThetaValidator(
            theta_mean=np.array([0.5, 0.5]),
            theta_std=np.array([0.1, 0.1]),
            theta_min=np.array([0.0, 0.0]),
            theta_max=np.array([1.0, 1.0]),
            training_thetas=None,
        )
        result = validator.check_interpolation(np.array([0.5, 0.5]))
        assert result.passed is True


# ── L5: Gradient Norm Tests ──────────────────────────────────────────────────


class TestGradientNorm:
    def test_at_minimum_passes(self, validator):
        """At the minimum of a quadratic, gradient should be ~0."""

        # E(θ) = sum(θᵢ²) → minimum at origin
        def energy_fn(theta):
            return float(np.sum(theta**2))

        theta_pred = np.array([0.001, -0.001, 0.0005, 0.0])
        result = validator.check_gradient_norm(theta_pred, energy_fn, threshold=0.1)
        assert result.passed is True
        assert result.gradient_norm < 0.1

    def test_far_from_minimum_fails(self, validator):
        """Far from minimum, gradient should be large."""

        def energy_fn(theta):
            return float(np.sum(theta**2))

        theta_pred = np.array([5.0, 5.0, 5.0, 5.0])
        result = validator.check_gradient_norm(theta_pred, energy_fn, threshold=0.1)
        assert result.passed is False
        assert result.gradient_norm > 1.0


# ── L7: Sensitivity Tests ────────────────────────────────────────────────────


class TestSensitivity:
    def test_insensitive_params_pass(self, validator):
        """When E is flat, all params should be insensitive."""

        def energy_fn(theta):
            return 0.0  # Constant energy

        theta_pred = np.array([1.0, 2.0, 3.0, 4.0])
        result = validator.check_sensitivity(theta_pred, energy_fn)
        assert result.passed is True
        assert result.max_sensitivity < 1e-10

    def test_sensitive_param_detected(self, validator):
        """When E depends strongly on one param, flag it."""

        # E = 100 * θ₀² + 0 * rest
        def energy_fn(theta):
            return 100.0 * theta[0] ** 2

        theta_pred = np.array([1.0, 0.0, 0.0, 0.0])
        result = validator.check_sensitivity(theta_pred, energy_fn, sensitivity_threshold=1.0)
        assert result.passed is False
        assert 0 in result.fragile_indices
        assert result.max_sensitivity > 100.0


# ── Orchestrator Tests ───────────────────────────────────────────────────────


class TestValidateOrchestrator:
    def test_level1_only(self, validator, training_data):
        theta_opt, _ = training_data
        theta_pred = theta_opt[5]
        report = validator.validate(theta_pred, level=1)
        assert report.level_executed == 1
        assert report.bound_check is not None
        assert report.numerical_sanity is None
        assert report.passes() is True

    def test_level3_full(self, validator, training_data):
        theta_opt, _ = training_data
        theta_pred = theta_opt[5]
        report = validator.validate(theta_pred, level=3, h_test=1.5)
        assert report.level_executed == 3
        assert report.bound_check is not None
        assert report.numerical_sanity is not None
        assert report.interpolation is not None
        assert report.fidelity is None  # Not executed at L3
        assert report.passes() is True

    def test_nan_aborts_early(self, validator):
        """NaN in θ_pred should abort further checks."""
        theta_pred = np.array([0.5, np.nan, 0.7, 0.1])
        report = validator.validate(theta_pred, level=7)
        # L2 fails → should return early, no L3+ executed
        assert report.numerical_sanity is not None
        assert report.numerical_sanity.passed is False
        assert report.interpolation is None
        assert report.passes() is False

    def test_confidence_score(self, validator, training_data):
        theta_opt, _ = training_data
        theta_pred = theta_opt[5]
        report = validator.validate(theta_pred, level=3)
        assert 0.0 <= report.confidence_score <= 1.0
        # All pass → confidence should be 1.0
        assert report.confidence_score == 1.0

    def test_report_to_dict(self, validator, training_data):
        theta_opt, _ = training_data
        theta_pred = theta_opt[5]
        report = validator.validate(theta_pred, level=3)
        d = report.to_dict()
        assert "overall_pass" in d
        assert "confidence_score" in d
        assert "L1_bound_check" in d
        assert "L2_numerical_sanity" in d
        assert "L3_interpolation" in d
        assert d["overall_pass"] is True


# ── from_training_data Factory Tests ─────────────────────────────────────────


class TestFactory:
    def test_single_point(self):
        """Single training point should still work (σ→0 handled)."""
        theta_opt = np.array([[0.5, 0.3, 0.7, 0.1]])
        validator = ThetaValidator.from_training_data(theta_opt)
        result = validator.check_bounds(np.array([0.5, 0.3, 0.7, 0.1]))
        assert result.passed is True

    def test_1d_input_reshaped(self):
        """1D input should be reshaped to (1, n_params)."""
        theta_opt = np.array([0.5, 0.3, 0.7, 0.1])
        validator = ThetaValidator.from_training_data(theta_opt)
        assert validator.n_params == 4
