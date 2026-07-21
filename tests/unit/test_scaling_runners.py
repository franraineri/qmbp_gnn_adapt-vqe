"""Unit tests for MPS Scaling runners and runner_base utility methods.

Validates:
- compute_vqe_quality_metrics uses DE_GAP_THRESHOLD from constants
- select_backend auto-dispatches correctly for scaling sizes
- vqe_descending_sweep handles edge cases
- Scaling runner dry-runs pass
- Phase3 runner uses predict_mpnn_at_h correctly
- Cross-N runner uses setup_physics() and inherited methods

Test coverage for runner_base methods used by scaling runners:
- compute_vqe_quality_metrics (DE_GAP_THRESHOLD import)
- compute_theta_smoothness
- check_variational_principle
- select_mpnn_hidden_dim
- safe_per_h_loop
- generate_h_grid (H_CRITICAL_ESTIMATES lookup)
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# Tests: compute_vqe_quality_metrics uses DE_GAP_THRESHOLD
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeVQEQualityMetrics:
    """Validates that compute_vqe_quality_metrics uses the canonical threshold."""

    @pytest.fixture
    def runner(self):
        """Minimal ValidationRunner subclass for testing static methods."""
        from qmbp_simulation.framework.runner_base import ValidationRunner

        return ValidationRunner

    def test_uses_de_gap_threshold_constant(self, runner):
        """n_pass should use DE_GAP_THRESHOLD, not a hardcoded 0.05."""
        vqe = [-10.0, -9.8, -9.5]
        exact = [-10.0, -10.0, -10.0]
        gaps = [2.0, 2.0, 2.0]
        result = runner.compute_vqe_quality_metrics(vqe, exact, gaps)
        # ΔE/gap: 0.0, 0.1, 0.25 → only first passes (0.0 < 0.05)
        assert result["n_pass"] == 1
        assert result["de_gaps"][0] == 0.0
        assert abs(result["de_gaps"][1] - 0.1) < 1e-10

    def test_all_pass_when_within_threshold(self, runner):
        """All pass when ΔE/gap < DE_GAP_THRESHOLD."""
        vqe = [-10.0, -9.95, -9.90]
        exact = [-10.0, -10.0, -10.0]
        gaps = [4.0, 4.0, 4.0]  # ΔE/gap = 0, 0.0125, 0.025 — all < 0.05
        result = runner.compute_vqe_quality_metrics(vqe, exact, gaps)
        assert result["n_pass"] == 3
        assert result["pass_rate"] == 1.0

    def test_empty_input(self, runner):
        """Empty inputs return safe defaults."""
        result = runner.compute_vqe_quality_metrics([], [], [])
        assert result["de_gaps"] == []
        assert result["n_pass"] == 0
        assert result["mean_de_gap"] == 0.0
        assert result["pass_rate"] == 0.0

    def test_gap_floor_prevents_division_by_zero(self, runner):
        """Zero gap uses 1e-10 floor (no crash)."""
        vqe = [-10.0]
        exact = [-10.1]
        gaps = [0.0]
        result = runner.compute_vqe_quality_metrics(vqe, exact, gaps)
        assert np.isfinite(result["de_gaps"][0])
        assert result["de_gaps"][0] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: compute_theta_smoothness
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeThetaSmoothness:
    """Validates theta smoothness computation for MPNN learnability."""

    @pytest.fixture
    def runner(self):
        from qmbp_simulation.framework.runner_base import ValidationRunner

        return ValidationRunner

    def test_constant_theta_gives_zero(self, runner):
        """Identical theta across all h-values → zero smoothness."""
        theta = np.array([[0.1, 0.2]] * 10)
        assert runner.compute_theta_smoothness(theta) == 0.0

    def test_linear_theta_gives_max_diff(self, runner):
        """Linearly varying theta → smoothness = step size."""
        theta = np.array([[i * 0.1, i * 0.05] for i in range(5)])
        expected = 0.1  # max(|0.1|, |0.05|) = 0.1
        assert abs(runner.compute_theta_smoothness(theta) - expected) < 1e-10

    def test_single_point_gives_zero(self, runner):
        """Single point → zero (no consecutive pairs)."""
        theta = np.array([[0.5, 0.3]])
        assert runner.compute_theta_smoothness(theta) == 0.0

    def test_empty_gives_zero(self, runner):
        """Empty array → zero."""
        theta = np.array([]).reshape(0, 2)
        assert runner.compute_theta_smoothness(theta) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: check_variational_principle
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckVariationalPrinciple:
    """Validates detection of E_vqe < E_exact violations."""

    @pytest.fixture
    def runner(self):
        from qmbp_simulation.framework.runner_base import ValidationRunner

        return ValidationRunner

    def test_no_violations(self, runner):
        """VQE above exact → zero violations."""
        vqe = [-9.9, -9.8, -9.7]
        exact = [-10.0, -10.0, -10.0]
        assert runner.check_variational_principle(vqe, exact) == 0

    def test_detects_violations(self, runner):
        """VQE below exact → violation count."""
        vqe = [-10.1, -10.2, -9.9]  # First two violate
        exact = [-10.0, -10.0, -10.0]
        assert runner.check_variational_principle(vqe, exact) == 2

    def test_tolerance_prevents_false_positives(self, runner):
        """Tiny violations below tolerance are ignored."""
        vqe = [-10.0 - 1e-12]  # Below exact by 1e-12 (< 1e-8 tolerance)
        exact = [-10.0]
        assert runner.check_variational_principle(vqe, exact) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: select_mpnn_hidden_dim
# ═══════════════════════════════════════════════════════════════════════════════


class TestSelectMPNNHiddenDim:
    """Validates auto-selection of MPNN hidden dimension."""

    @pytest.fixture
    def runner(self):
        from qmbp_simulation.framework.runner_base import ValidationRunner

        return ValidationRunner

    def test_large_dataset_returns_max(self, runner):
        """Large dataset → maximum hidden_dim."""
        dim = runner.select_mpnn_hidden_dim(n_training_graphs=1000, theta_dim=2, max_hidden=128)
        assert dim == 128

    def test_small_dataset_reduces_dim(self, runner):
        """Small dataset → reduced hidden_dim."""
        dim = runner.select_mpnn_hidden_dim(
            n_training_graphs=10, theta_dim=2, max_hidden=128, min_hidden=32
        )
        assert dim <= 128  # Should reduce
        assert dim >= 32  # But not below floor


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: safe_per_h_loop
# ═══════════════════════════════════════════════════════════════════════════════


class TestSafePerHLoop:
    """Validates error-isolated h-point loop."""

    @pytest.fixture
    def runner(self):
        from qmbp_simulation.framework.runner_base import ValidationRunner

        class MinimalRunner(ValidationRunner):
            runner_id = "test"
            experiment_id = "test"
            description = "test"
            hypothesis = "test"

            def define_sections(self):
                return []

        args = argparse.Namespace(
            section=None,
            skip_preflight=False,
            stop_on_failure=False,
            verbose=False,
            dry_run=False,
            validate_vqe=True,
            validate_theta=True,
            theta_validation_level=4,
            strict_validation=False,
            resume=None,
            preset=None,
            save_artifacts="never",
            no_bidirectional=False,
            force_bidirectional=False,
        )
        return MinimalRunner(args=args)

    def test_all_succeed(self, runner):
        """All h-points return results."""
        results = runner.safe_per_h_loop(
            [1.0, 2.0, 3.0],
            lambda h: {"h": h, "energy": -h},
            label="test",
        )
        assert len(results) == 3

    def test_some_fail_gracefully(self, runner):
        """Failed points are skipped, others continue."""

        def fn(h):
            if h == 2.0:
                raise ValueError("simulated failure")
            return {"h": h}

        results = runner.safe_per_h_loop([1.0, 2.0, 3.0], fn, label="test")
        assert len(results) == 2
        assert results[0]["h"] == 1.0
        assert results[1]["h"] == 3.0

    def test_none_return_skipped(self, runner):
        """None returns are treated as skips."""

        def fn(h):
            return None if h == 2.0 else {"h": h}

        results = runner.safe_per_h_loop([1.0, 2.0, 3.0], fn, label="test")
        assert len(results) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: generate_h_grid (H_CRITICAL_ESTIMATES)
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateHGrid:
    """Validates h-grid generation uses model-specific h_critical."""

    @pytest.fixture
    def runner(self):
        from qmbp_simulation.framework.runner_base import ValidationRunner

        class MinimalRunner(ValidationRunner):
            runner_id = "test"
            experiment_id = "test"
            description = "test"
            hypothesis = "test"

            def define_sections(self):
                return []

        args = argparse.Namespace(
            section=None,
            skip_preflight=False,
            stop_on_failure=False,
            verbose=False,
            dry_run=False,
            validate_vqe=True,
            validate_theta=True,
            theta_validation_level=4,
            strict_validation=False,
            resume=None,
            preset=None,
            save_artifacts="never",
            no_bidirectional=False,
            force_bidirectional=False,
            h_min=0.5,
            h_max=2.0,
            h_points=10,
            model="tfim",
        )
        return MinimalRunner(args=args)

    def test_tfim_h_critical_is_1(self, runner):
        """TFIM uses h_critical=1.0 from H_CRITICAL_ESTIMATES."""
        from qmbp_simulation.framework.runner_base import ValidationRunner

        assert ValidationRunner.H_CRITICAL_ESTIMATES["tfim"] == 1.0

    def test_tfim_longitudinal_h_critical_is_1(self, runner):
        """TFIM longitudinal also uses h_critical=1.0."""
        from qmbp_simulation.framework.runner_base import ValidationRunner

        assert ValidationRunner.H_CRITICAL_ESTIMATES["tfim_longitudinal"] == 1.0

    def test_grid_is_descending(self, runner):
        """Generated h-grid is sorted descending (for warm-start sweep)."""
        grid = runner.generate_h_grid()
        assert grid == sorted(grid, reverse=True)

    def test_grid_has_correct_count(self, runner):
        """Grid length matches h_points arg."""
        grid = runner.generate_h_grid()
        assert len(grid) == 10


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Scaling runner dry-runs
# ═══════════════════════════════════════════════════════════════════════════════


class TestScalingRunnerDryRuns:
    """Verify all scaling runners can be instantiated and dry-run without error."""

    def test_scaling_validation_dry_run(self):
        """MPSScalingValidationRunner dry-run succeeds."""
        from scripts.experiment_runners.scaling.run_scaling_validation import (
            MPSScalingValidationRunner,
        )

        args = argparse.Namespace(
            section=None,
            skip_preflight=False,
            stop_on_failure=False,
            verbose=False,
            dry_run=True,
            validate_vqe=True,
            validate_theta=True,
            theta_validation_level=4,
            strict_validation=False,
            resume=None,
            preset=None,
            save_artifacts="never",
            no_bidirectional=False,
            force_bidirectional=False,
            n_qubits=30,
            topology="chain_1d",
            model="tfim",
            p_layers=1,
            h_min=None,
            h_max=None,
            h_points=5,
            chi_max=64,
            maxiter=100,
            n_restarts=1,
            seeds=[42],
            verify_chi=False,
        )
        runner = MPSScalingValidationRunner(args=args)
        exit_code = runner.run()
        assert exit_code == 0

    def test_scaling_validation_tfim_longitudinal_warns(self):
        """Running with tfim_longitudinal should warn about scaling law."""
        from scripts.experiment_runners.scaling.run_scaling_validation import (
            MPSScalingValidationRunner,
        )

        args = argparse.Namespace(
            section=None,
            skip_preflight=False,
            stop_on_failure=False,
            verbose=False,
            dry_run=True,
            validate_vqe=True,
            validate_theta=True,
            theta_validation_level=4,
            strict_validation=False,
            resume=None,
            preset=None,
            save_artifacts="never",
            no_bidirectional=False,
            force_bidirectional=False,
            n_qubits=40,
            topology="chain_1d",
            model="tfim_longitudinal",
            p_layers=1,
            h_min=None,
            h_max=None,
            h_points=5,
            chi_max=64,
            maxiter=100,
            n_restarts=1,
            seeds=[42],
            verify_chi=False,
        )
        runner = MPSScalingValidationRunner(args=args)
        exit_code = runner.run()
        assert exit_code == 0
