"""Unit tests for qmbp_simulation.optimizers module."""

from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.models import (
    HamiltonianBuilder,
    VQEConfig,
)
from qmbp_simulation.optimizers import VQEOptimizer


class TestDescendingSweepEnforcement:
    """Test descending sweep enforcement (ValueError on ascending)."""

    def test_ascending_h_values_raises(self, small_lattice, small_circuit):
        config = VQEConfig(p_layers=1, n_restarts=1, maxiter=10)
        optimizer = VQEOptimizer(config)
        qc, _ = small_circuit
        h_ascending = np.array([0.5, 1.0, 1.5, 2.0])
        with pytest.raises(ValueError, match="descending order"):
            optimizer.descending_sweep(h_ascending, qc, small_lattice)

    def test_descending_h_values_accepted(self, small_lattice, small_circuit):
        config = VQEConfig(p_layers=1, n_restarts=1, maxiter=5)
        optimizer = VQEOptimizer(config)
        qc, _ = small_circuit
        h_descending = np.array([2.0, 1.5])
        # Should not raise
        results = optimizer.descending_sweep(h_descending, qc, small_lattice)
        assert len(results) == 2


class TestWarmStartPropagation:
    """Test warm-start propagation between h-points."""

    def test_previous_theta_used_as_initial_guess(self):
        config = VQEConfig(p_layers=1)
        previous = np.array([0.5, -0.3])
        guess = VQEOptimizer.get_initial_guess(
            n_params=2, h_value=1.5, config=config, previous_theta=previous
        )
        # Warm-start should return a copy of previous_theta
        np.testing.assert_array_equal(guess, previous)
        # Must be a copy, not the same object
        assert guess is not previous

    def test_no_previous_theta_uses_zeros_or_random(self):
        config = VQEConfig(p_layers=1, warm_start_seed_zeros=True)
        guess = VQEOptimizer.get_initial_guess(
            n_params=2, h_value=1.5, config=config, previous_theta=None
        )
        assert guess.shape == (2,)


class TestTrajectoryRecording:
    """Test trajectory recording with callbacks."""

    def test_trajectory_recorded_when_callbacks_enabled(
        self, small_lattice, small_circuit, noiseless_backend
    ):
        config = VQEConfig(p_layers=1, n_restarts=1, maxiter=10, enable_callbacks=True)
        optimizer = VQEOptimizer(config, backend=noiseless_backend)
        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        qc, _ = small_circuit
        initial = np.array([0.1, 0.2])
        result = optimizer.optimize(H, qc, initial)
        # Trajectory should be recorded
        assert result.trajectory is not None
        assert len(result.trajectory.energies) > 0


class TestVQEResultStructure:
    """Test VQE result structure and properties."""

    def test_optimize_returns_vqe_result(self, small_lattice, small_circuit, noiseless_backend):
        from qmbp_simulation.models import VQEResult

        config = VQEConfig(p_layers=1, n_restarts=1, maxiter=10)
        optimizer = VQEOptimizer(config, backend=noiseless_backend)
        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        qc, _ = small_circuit
        initial = np.array([0.1, 0.2])
        result = optimizer.optimize(H, qc, initial)
        assert isinstance(result, VQEResult)
        assert np.isfinite(result.energy)
        assert result.theta_opt.shape == (2,)
        assert result.n_iterations > 0

    def test_sweep_results_ordered_by_h(self, small_lattice, small_circuit, noiseless_backend):
        config = VQEConfig(p_layers=1, n_restarts=1, maxiter=5)
        optimizer = VQEOptimizer(config, backend=noiseless_backend)
        qc, _ = small_circuit
        h_values = np.array([2.0, 1.5, 1.0])
        results = optimizer.descending_sweep(h_values, qc, small_lattice)
        # Results should be in same order as h_values
        for i, r in enumerate(results):
            assert r.h_value == h_values[i]

    def test_energy_is_finite_for_all_h_points(
        self, small_lattice, small_circuit, noiseless_backend
    ):
        config = VQEConfig(p_layers=1, n_restarts=1, maxiter=10)
        optimizer = VQEOptimizer(config, backend=noiseless_backend)
        qc, _ = small_circuit
        h_values = np.array([2.0, 1.5])
        results = optimizer.descending_sweep(h_values, qc, small_lattice)
        for r in results:
            assert np.isfinite(r.energy)


class TestSPSAOptimizer:
    """Test SPSAOptimizer basic behavior."""

    def test_spsa_returns_vqe_result(self, small_lattice, small_circuit, noiseless_backend):
        from qmbp_simulation.models import VQEResult
        from qmbp_simulation.optimizers import SPSAOptimizer

        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        qc, _ = small_circuit
        initial = np.array([0.1, 0.2])

        spsa = SPSAOptimizer(backend=noiseless_backend, a=0.1, c=0.05)
        result = spsa.optimize(qc, H, initial, n_iterations=10)
        assert isinstance(result, VQEResult)
        assert np.isfinite(result.energy)
        assert result.theta_opt.shape == (2,)

    def test_spsa_energy_improves_or_stays(self, small_lattice, small_circuit, noiseless_backend):
        from qmbp_simulation.optimizers import SPSAOptimizer

        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        qc, _ = small_circuit
        initial = np.zeros(2)

        # Initial energy
        e_init = noiseless_backend.evaluate(qc, H, initial)

        spsa = SPSAOptimizer(backend=noiseless_backend, a=0.1, c=0.05)
        result = spsa.optimize(qc, H, initial, n_iterations=50)
        # SPSA should find energy ≤ initial (or very close)
        assert result.energy <= e_init + 0.1

    def test_spsa_parameters_within_bounds(self, small_lattice, small_circuit, noiseless_backend):
        from qmbp_simulation.optimizers import SPSAOptimizer

        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        qc, _ = small_circuit
        initial = np.array([0.5, -0.5])

        spsa = SPSAOptimizer(backend=noiseless_backend, a=0.1, c=0.05)
        result = spsa.optimize(qc, H, initial, n_iterations=20)
        # Parameters should be clipped to [-π, π]
        assert np.all(result.theta_opt >= -np.pi)
        assert np.all(result.theta_opt <= np.pi)
