"""Tests for theta outlier detection and filtering.

Validates the pre-MPNN outlier filter that detects and corrects
VQE local-minimum traps before MPNN training. These traps appear
as isolated spikes in the θ(h) curve that would corrupt MPNN learning.

Run:
    python -m pytest tests/unit/test_theta_outlier_filter.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.analysis.theta_alignment import (
    OutlierReport,
    detect_theta_outliers,
    filter_theta_outliers,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def smooth_theta():
    """A perfectly smooth θ(h) curve (linear, no outliers)."""
    n_points = 20
    h = np.linspace(5.0, 1.0, n_points)
    theta = np.column_stack(
        [
            np.linspace(0.1, 1.0, n_points),
            np.linspace(0.5, 1.5, n_points),
        ]
    )
    e_exact = np.linspace(-25.0, -10.0, n_points)
    return theta, h, e_exact


@pytest.fixture
def single_outlier_theta():
    """Smooth curve with one isolated outlier at index 5."""
    n_points = 15
    h = np.linspace(5.0, 1.0, n_points)
    theta = np.column_stack(
        [
            np.linspace(0.1, 1.0, n_points),
            np.linspace(0.5, 1.5, n_points),
        ]
    )
    e_exact = np.linspace(-25.0, -10.0, n_points)
    # Inject outlier at index 5
    theta[5] = [3.0, -2.5]
    return theta, h, e_exact


@pytest.fixture
def multi_outlier_theta():
    """Smooth curve with two outliers at indices 4 and 10."""
    n_points = 20
    h = np.linspace(5.0, 1.0, n_points)
    theta = np.column_stack(
        [
            np.linspace(0.1, 2.0, n_points),
            np.linspace(0.5, 2.5, n_points),
            np.linspace(-0.3, 0.7, n_points),
        ]
    )
    e_exact = np.linspace(-30.0, -12.0, n_points)
    # Inject outliers
    theta[4] = [5.0, -3.0, 2.8]
    theta[10] = [-4.0, 4.5, -3.1]
    return theta, h, e_exact


# ═══════════════════════════════════════════════════════════════════════════════
# Detection Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDetectThetaOutliers:
    """Test outlier detection logic."""

    def test_no_outliers_in_smooth_curve(self, smooth_theta):
        """Smooth linear θ(h) should produce zero outliers."""
        theta, h, _ = smooth_theta
        report = detect_theta_outliers(theta, h)
        assert report.n_outliers == 0
        assert report.outlier_indices == []

    def test_detects_single_outlier(self, single_outlier_theta):
        """Single spike at index 5 should be detected."""
        theta, h, _ = single_outlier_theta
        report = detect_theta_outliers(theta, h)
        assert report.n_outliers >= 1
        assert 5 in report.outlier_indices

    def test_detects_multiple_outliers(self, multi_outlier_theta):
        """Two outliers at indices 4 and 10 should both be detected."""
        theta, h, _ = multi_outlier_theta
        report = detect_theta_outliers(theta, h)
        assert 4 in report.outlier_indices
        assert 10 in report.outlier_indices

    def test_returns_outlier_report_dataclass(self, single_outlier_theta):
        """Return type is OutlierReport with correct fields."""
        theta, h, _ = single_outlier_theta
        report = detect_theta_outliers(theta, h)
        assert isinstance(report, OutlierReport)
        assert report.n_points == len(h)
        assert len(report.outlier_h_values) == report.n_outliers

    def test_fidelity_based_detection(self, smooth_theta):
        """Low-fidelity point with high-fidelity neighbors is an outlier."""
        theta, h, _ = smooth_theta
        fids = np.ones(len(h)) * 0.95
        fids[7] = 0.01  # catastrophic fidelity drop
        report = detect_theta_outliers(theta, h, fidelities=fids)
        assert 7 in report.outlier_indices

    def test_threshold_sensitivity(self, single_outlier_theta):
        """Lower threshold should detect more aggressively."""
        theta, h, _ = single_outlier_theta
        report_strict = detect_theta_outliers(theta, h, threshold=5.0)
        report_loose = detect_theta_outliers(theta, h, threshold=1.0)
        assert report_loose.n_outliers >= report_strict.n_outliers

    def test_endpoints_never_flagged(self, single_outlier_theta):
        """First and last points should never be outliers."""
        theta, h, _ = single_outlier_theta
        # Put outlier at endpoints
        theta_copy = theta.copy()
        theta_copy[0] = [99.0, -99.0]
        theta_copy[-1] = [-99.0, 99.0]
        report = detect_theta_outliers(theta_copy, h)
        assert 0 not in report.outlier_indices
        assert len(h) - 1 not in report.outlier_indices

    def test_minimum_points_required(self):
        """With fewer than 3 points, no outliers can be detected."""
        theta = np.array([[0.1, 0.2], [3.0, -2.5]])
        h = np.array([5.0, 3.0])
        report = detect_theta_outliers(theta, h)
        assert report.n_outliers == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Filtering Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFilterThetaOutliers:
    """Test outlier filtering (interpolation and removal)."""

    def test_interpolate_replaces_outlier_with_neighbor_average(self, single_outlier_theta):
        """Interpolation should replace outlier with mean of neighbors."""
        theta, h, e = single_outlier_theta
        theta_clean, h_c, e_c, _, report = filter_theta_outliers(
            theta, h, e, replace_strategy="interpolate"
        )
        if report.n_outliers > 0 and 5 in report.outlier_indices:
            expected = (theta[4] + theta[6]) / 2.0
            np.testing.assert_allclose(theta_clean[5], expected, atol=0.01)

    def test_interpolate_preserves_array_shape(self, single_outlier_theta):
        """Interpolation keeps the same number of points."""
        theta, h, e = single_outlier_theta
        theta_c, h_c, e_c, _, _ = filter_theta_outliers(theta, h, e, replace_strategy="interpolate")
        assert theta_c.shape == theta.shape
        assert len(h_c) == len(h)
        assert len(e_c) == len(e)

    def test_remove_reduces_array_size(self, single_outlier_theta):
        """Remove strategy should reduce dataset size by n_outliers."""
        theta, h, e = single_outlier_theta
        theta_c, h_c, e_c, _, report = filter_theta_outliers(theta, h, e, replace_strategy="remove")
        assert len(h_c) == len(h) - report.n_outliers

    def test_no_outliers_returns_unchanged(self, smooth_theta):
        """With no outliers, arrays are returned unchanged."""
        theta, h, e = smooth_theta
        theta_c, h_c, e_c, _, report = filter_theta_outliers(theta, h, e)
        assert report.n_outliers == 0
        np.testing.assert_array_equal(theta_c, theta)
        np.testing.assert_array_equal(h_c, h)

    def test_fidelities_passed_through(self, single_outlier_theta):
        """Fidelity array is correctly filtered/preserved."""
        theta, h, e = single_outlier_theta
        fids = np.ones(len(h)) * 0.98
        _, _, _, fid_c, _ = filter_theta_outliers(
            theta, h, e, fidelities=fids, replace_strategy="interpolate"
        )
        assert fid_c is not None
        assert len(fid_c) == len(h)

    def test_remove_with_fidelities(self, single_outlier_theta):
        """Remove strategy also filters the fidelity array."""
        theta, h, e = single_outlier_theta
        fids = np.ones(len(h)) * 0.98
        _, _, _, fid_c, report = filter_theta_outliers(
            theta, h, e, fidelities=fids, replace_strategy="remove"
        )
        if report.n_outliers > 0:
            assert fid_c is not None
            assert len(fid_c) == len(h) - report.n_outliers

    def test_invalid_strategy_raises(self, single_outlier_theta):
        """Unknown replace_strategy should raise ValueError."""
        theta, h, e = single_outlier_theta
        with pytest.raises(ValueError, match="Unknown replace_strategy"):
            filter_theta_outliers(theta, h, e, replace_strategy="magic")

    def test_non_outlier_points_unchanged(self, single_outlier_theta):
        """Points that are not outliers should not be modified."""
        theta, h, e = single_outlier_theta
        theta_c, _, _, _, report = filter_theta_outliers(
            theta, h, e, replace_strategy="interpolate"
        )
        for i in range(len(h)):
            if i not in report.outlier_indices:
                np.testing.assert_array_equal(theta_c[i], theta[i])
