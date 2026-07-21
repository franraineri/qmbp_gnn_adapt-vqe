"""Tests for h-grid generation, validation, and runner integration.

Covers:
- generate_nonuniform_h_grid behavior and edge cases
- ValidationRunner.generate_h_grid method
- Preflight validation of h_min, h_max, h_points
- Grid properties (descending order, correct count, valid range)
"""

import argparse

import numpy as np
import pytest

from qmbp_simulation.pipeline.dataset_io import generate_nonuniform_h_grid

# ═══════════════════════════════════════════════════════════════════════════════
# Tests: generate_nonuniform_h_grid (low-level function)
# ═══════════════════════════════════════════════════════════════════════════════


class TestNonuniformHGrid:
    """Test the core grid generation function."""

    def test_returns_correct_count(self):
        grid = generate_nonuniform_h_grid(0.5, 2.0, 15, h_critical=1.0)
        assert len(grid) == 15

    def test_returns_descending_order(self):
        grid = generate_nonuniform_h_grid(0.5, 2.0, 15, h_critical=1.0)
        assert np.all(np.diff(grid) <= 0), "Grid must be descending"

    def test_range_matches_hmin_hmax(self):
        grid = generate_nonuniform_h_grid(1.0, 3.0, 20, h_critical=1.5)
        assert np.isclose(grid.max(), 3.0, atol=1e-10)
        assert np.isclose(grid.min(), 1.0, atol=1e-10)

    def test_denser_near_critical(self):
        """Points near h_critical should have smaller spacing."""
        grid = generate_nonuniform_h_grid(0.5, 3.0, 30, h_critical=1.5)
        sorted_grid = np.sort(grid)
        spacings = np.diff(sorted_grid)

        # Find spacing near h_crit vs far from it
        near_mask = (sorted_grid[:-1] > 1.0) & (sorted_grid[:-1] < 2.0)
        far_mask = sorted_grid[:-1] > 2.5
        if near_mask.any() and far_mask.any():
            avg_near = spacings[near_mask].mean()
            avg_far = spacings[far_mask].mean()
            assert avg_near < avg_far, (
                f"Spacing near h_crit ({avg_near:.4f}) should be less than "
                f"far from it ({avg_far:.4f})"
            )

    def test_h_critical_none_uses_midpoint(self):
        """h_critical=None should use (h_min+h_max)/2."""
        grid = generate_nonuniform_h_grid(1.0, 3.0, 20, h_critical=None)
        assert len(grid) == 20
        # Dense region should be around midpoint (2.0)
        sorted_grid = np.sort(grid)
        mid_count = np.sum((sorted_grid > 1.5) & (sorted_grid < 2.5))
        assert mid_count >= 5, f"Expected >=5 points near midpoint, got {mid_count}"

    def test_h_critical_outside_range_degrades_to_uniform(self):
        """If h_critical is far outside [h_min, h_max], grid is ~uniform."""
        grid = generate_nonuniform_h_grid(2.0, 4.0, 10, h_critical=0.5)
        sorted_grid = np.sort(grid)
        spacings = np.diff(sorted_grid)
        # Ratio of max/min spacing should be low (near uniform)
        ratio = spacings.max() / spacings.min()
        assert ratio < 3.0, f"Expected near-uniform (ratio<3), got {ratio:.1f}"

    def test_small_n_points(self):
        """n_points=3 should still produce a valid grid."""
        grid = generate_nonuniform_h_grid(0.5, 2.0, 3, h_critical=1.0)
        assert len(grid) == 3
        assert grid[0] > grid[-1]

    def test_large_n_points(self):
        """n_points=100 should produce exactly 100 unique points."""
        grid = generate_nonuniform_h_grid(0.5, 5.0, 100, h_critical=1.0)
        assert len(grid) == 100
        assert len(np.unique(grid)) == 100

    def test_narrow_range(self):
        """Very narrow range should still produce valid grid."""
        grid = generate_nonuniform_h_grid(1.0, 1.1, 5, h_critical=1.05)
        assert len(grid) == 5
        assert np.isclose(grid.max(), 1.1, atol=1e-10)
        assert np.isclose(grid.min(), 1.0, atol=1e-10)

    def test_all_points_within_range(self):
        """No point should exceed [h_min, h_max]."""
        for h_min, h_max, h_crit in [(0.5, 2.0, 1.0), (1.3, 3.0, 1.0), (0.1, 5.0, 2.5)]:
            grid = generate_nonuniform_h_grid(h_min, h_max, 20, h_critical=h_crit)
            assert grid.min() >= h_min - 1e-10, f"Point below h_min: {grid.min()} < {h_min}"
            assert grid.max() <= h_max + 1e-10, f"Point above h_max: {grid.max()} > {h_max}"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: ValidationRunner.generate_h_grid (base-class method)
# ═══════════════════════════════════════════════════════════════════════════════

from qmbp_simulation.framework.runner_base import Section, ValidationRunner


class _HGridTestRunner(ValidationRunner):
    """Minimal runner for testing generate_h_grid."""

    runner_id = "hgrid_test"
    experiment_id = "HGRID_TEST"
    description = "H-grid test runner"
    hypothesis = "H-grid generation works"

    def define_sections(self):
        return [Section(id=1, name="Dummy", fn=lambda: {"pass": True}, hypothesis="")]


class TestRunnerGenerateHGrid:
    """Test the ValidationRunner.generate_h_grid method."""

    @pytest.fixture
    def runner(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "qmbp_simulation.framework.result_io._DEFAULT_RESULTS_ROOT",
            tmp_path / "experiments",
        )
        args = argparse.Namespace(
            section=None,
            skip_preflight=True,
            stop_on_failure=False,
            verbose=False,
            dry_run=True,
            validate_vqe=False,
            validate_theta=False,
            theta_validation_level=4,
            strict_validation=False,
            resume=None,
            preset=None,
            h_min=0.5,
            h_max=2.0,
            h_points=15,
            model="tfim",
        )
        return _HGridTestRunner(args=args)

    def test_uses_args_defaults(self, runner):
        grid = runner.generate_h_grid()
        assert len(grid) == 15
        assert grid[0] > grid[-1]  # descending
        assert max(grid) <= 2.0 + 1e-10
        assert min(grid) >= 0.5 - 1e-10

    def test_override_params(self, runner):
        grid = runner.generate_h_grid(h_min=1.0, h_max=3.0, h_points=20)
        assert len(grid) == 20
        assert max(grid) <= 3.0 + 1e-10
        assert min(grid) >= 1.0 - 1e-10

    def test_model_specific_h_critical(self, runner):
        """Different models should use different h_critical values."""
        grid_tfim = runner.generate_h_grid(model="tfim")
        grid_frust = runner.generate_h_grid(model="tfim_frustrated")
        # Both valid, but different density patterns
        assert len(grid_tfim) == 15
        assert len(grid_frust) == 15

    def test_unknown_model_uses_midpoint(self, runner):
        grid = runner.generate_h_grid(model="unknown_model")
        assert len(grid) == 15

    def test_returns_list_not_array(self, runner):
        grid = runner.generate_h_grid()
        assert isinstance(grid, list)
        assert all(isinstance(x, float) for x in grid)


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Preflight validation (h_min, h_max, h_points)
# ═══════════════════════════════════════════════════════════════════════════════

from scripts.experiment_runners.noiseless.run_noiseless_pipeline import NoiselessPipelineRunner


class TestPreflightHValidation:
    """Test that run_preflight catches invalid h-grid configurations."""

    @pytest.fixture(autouse=True)
    def _isolate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "qmbp_simulation.framework.result_io._DEFAULT_RESULTS_ROOT",
            tmp_path / "experiments",
        )

    def _make_runner(self, **kwargs):
        defaults = {
            "section": None,
            "skip_preflight": False,
            "stop_on_failure": False,
            "verbose": False,
            "dry_run": False,
            "validate_vqe": True,
            "validate_theta": True,
            "theta_validation_level": 4,
            "strict_validation": False,
            "resume": None,
            "preset": None,
            "n_qubits": 4,
            "p_layers": 2,
            "topology": ["chain_1d"],
            "model": "tfim",
            "model_params": None,
            "h_min": 0.5,
            "h_max": 2.0,
            "h_points": 15,
            "seeds": [42],
            "maxiter": 50,
            "n_restarts": 1,
            "output": None,
            "no_bidirectional": False,
            "force_bidirectional": False,
            "save_artifacts": "never",
            "no_physics_loss": False,
            "physics_loss_weight": 0.2,
            "physics_loss_start": 800,
        }
        defaults.update(kwargs)
        args = argparse.Namespace(**defaults)
        return NoiselessPipelineRunner(args=args)

    def test_valid_config_passes(self):
        runner = self._make_runner()
        assert runner.run_preflight() is True

    def test_h_min_greater_than_h_max_fails(self):
        runner = self._make_runner(h_min=2.0, h_max=1.0)
        assert runner.run_preflight() is False

    def test_h_min_equals_h_max_fails(self):
        runner = self._make_runner(h_min=1.5, h_max=1.5)
        assert runner.run_preflight() is False

    def test_h_points_too_few_fails(self):
        runner = self._make_runner(h_points=2)
        assert runner.run_preflight() is False

    def test_h_points_one_fails(self):
        runner = self._make_runner(h_points=1)
        assert runner.run_preflight() is False

    def test_h_points_three_passes_with_warning(self):
        """3 points passes (minimum) but should generate a warning."""
        runner = self._make_runner(h_points=3)
        assert runner.run_preflight() is True

    def test_negative_h_min_fails(self):
        runner = self._make_runner(h_min=-0.5)
        assert runner.run_preflight() is False

    def test_zero_h_max_fails(self):
        runner = self._make_runner(h_max=0.0)
        assert runner.run_preflight() is False

    def test_n_qubits_too_large_fails(self):
        runner = self._make_runner(n_qubits=30)
        assert runner.run_preflight() is False

    def test_n_qubits_too_small_fails(self):
        runner = self._make_runner(n_qubits=1)
        assert runner.run_preflight() is False


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Input validation in generate_nonuniform_h_grid
# ═══════════════════════════════════════════════════════════════════════════════


class TestHGridInputValidation:
    """Test that generate_nonuniform_h_grid rejects invalid inputs."""

    def test_h_min_equals_h_max_raises(self):
        with pytest.raises(ValueError, match="h_min.*must be < h_max"):
            generate_nonuniform_h_grid(1.5, 1.5, 10, h_critical=1.5)

    def test_h_min_greater_than_h_max_raises(self):
        with pytest.raises(ValueError, match="h_min.*must be < h_max"):
            generate_nonuniform_h_grid(2.0, 1.0, 10, h_critical=1.5)

    def test_n_points_zero_raises(self):
        with pytest.raises(ValueError, match="n_points must be >= 1"):
            generate_nonuniform_h_grid(0.5, 2.0, 0, h_critical=1.0)

    def test_n_points_negative_raises(self):
        with pytest.raises(ValueError, match="n_points must be >= 1"):
            generate_nonuniform_h_grid(0.5, 2.0, -5, h_critical=1.0)
