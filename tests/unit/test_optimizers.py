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

class TestVQETerminationGuards:
    """Test VQE termination guards: maxfev cap, stagnation detection, wall-clock timing."""

    def test_cobyla_auto_switch_terminates(self, noiseless_backend):
        """COBYLA auto-switch (n_params > 8) terminates via maxfev cap."""
        import time

        from qmbp_simulation import make_lattice
        from qmbp_simulation.circuits import HVACircuitBuilder

        lattice = make_lattice("chain_1d", 10, J=1.0, h=1.0)
        builder = HamiltonianBuilder()
        H = builder.build(lattice)
        hva = HVACircuitBuilder()
        circuit, _ = hva.create(10, 5, lattice)
        n_params = circuit.num_parameters
        assert n_params > 8  # Triggers COBYLA auto-switch

        config = VQEConfig(
            p_layers=5,
            n_restarts=1,
            maxiter=200,
            method="L-BFGS-B",
            enable_callbacks=False,
        )
        optimizer = VQEOptimizer(config=config, backend=noiseless_backend, seed=42)

        t0 = time.perf_counter()
        result = optimizer.optimize(H, circuit, np.random.default_rng(42).uniform(-0.01, 0.01, n_params))
        elapsed = time.perf_counter() - t0

        assert np.isfinite(result.energy)
        assert elapsed < 30.0, f"COBYLA took {elapsed:.1f}s (expected < 30s)"

    def test_stagnation_early_stop(self, small_lattice, small_circuit, noiseless_backend):
        """Restarts that don't improve are detected and remaining restarts skipped."""
        import time

        config = VQEConfig(
            p_layers=1,
            n_restarts=20,  # Many restarts — stagnation should fire early
            maxiter=500,
            method="L-BFGS-B",
            enable_callbacks=False,
        )
        optimizer = VQEOptimizer(config=config, backend=noiseless_backend, seed=42)
        builder = HamiltonianBuilder()
        # h=2.0 is the paramagnetic limit — easy landscape, warm-start converges immediately
        from qmbp_simulation import make_lattice

        lattice_easy = make_lattice("chain_1d", 4, J=1.0, h=2.0)
        H = builder.build(lattice_easy)
        qc, _ = small_circuit

        t0 = time.perf_counter()
        result = optimizer.optimize(H, qc, np.array([0.1, 0.2]))
        elapsed = time.perf_counter() - t0

        assert np.isfinite(result.energy)
        # With 20 restarts but stagnation threshold=3, should terminate much faster
        # than 20 full restarts. At 500 maxiter each, 20 restarts would take ~5s+.
        # With early-stop it should be < 2s.
        assert elapsed < 5.0, f"Stagnation detection failed: took {elapsed:.1f}s with 20 restarts"

    def test_nelder_mead_maxfev_cap(self, small_lattice, small_circuit, noiseless_backend):
        """Nelder-Mead terminates via maxfev cap even with tight ftol."""
        import time

        config = VQEConfig(
            p_layers=1,
            n_restarts=1,
            maxiter=100,
            method="Nelder-Mead",
            enable_callbacks=False,
        )
        optimizer = VQEOptimizer(config=config, backend=noiseless_backend, seed=42)
        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        qc, _ = small_circuit

        t0 = time.perf_counter()
        result = optimizer.optimize(H, qc, np.array([0.1, 0.2]))
        elapsed = time.perf_counter() - t0

        assert np.isfinite(result.energy)
        assert elapsed < 5.0, f"Nelder-Mead took {elapsed:.1f}s (expected < 5s)"

    def test_cobyla_eval_count_bounded(self, noiseless_backend):
        """COBYLA function evaluations are capped by maxfun formula."""
        from qmbp_simulation import make_lattice
        from qmbp_simulation.circuits import HVACircuitBuilder

        lattice = make_lattice("chain_1d", 10, J=1.0, h=1.0)
        builder = HamiltonianBuilder()
        H = builder.build(lattice)
        hva = HVACircuitBuilder()
        circuit, _ = hva.create(10, 5, lattice)
        n_params = circuit.num_parameters

        # Track actual function evaluations
        eval_count = [0]
        original_evaluate = noiseless_backend.evaluate

        def counting_evaluate(qc, H, params):
            eval_count[0] += 1
            return original_evaluate(qc, H, params)

        noiseless_backend.evaluate = counting_evaluate

        config = VQEConfig(
            p_layers=5,
            n_restarts=1,  # Minimum allowed
            maxiter=100,
            method="COBYLA",
            enable_callbacks=False,
        )
        optimizer = VQEOptimizer(config=config, backend=noiseless_backend, seed=42)
        optimizer.optimize(H, circuit, np.random.default_rng(42).uniform(-0.01, 0.01, n_params))

        # maxfun = maxiter * min(n_params + 5, 50) = 100 * 15 = 1500
        # With 1 restart: 2 COBYLA calls, each capped at maxfun
        maxfun = 100 * min(n_params + 5, 50)
        # Allow 2x for warm-start + 1 restart, plus some overhead
        assert eval_count[0] <= 2 * maxfun + 200, (
            f"COBYLA exceeded eval cap: {eval_count[0]} > {2 * maxfun + 200}"
        )

        # Restore original
        noiseless_backend.evaluate = original_evaluate
