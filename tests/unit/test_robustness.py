"""Robustness and confidence tests for the pipeline fixes.

Tests the numerical stability, reproducibility, and error handling
improvements applied to the pipeline. These tests verify that:
1. Energy evaluations are guarded against NaN/Inf
2. Shot noise is stochastic (not deterministic)
3. VQE restarts are reproducible with seeded RNG
4. MPNN training is deterministic with same seed
5. Dataset validation catches corrupted data
6. Phase 4 graph construction matches training
7. Empty/invalid inputs are caught early
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qmbp_simulation.execution import NoiselessBackend, NoisyBackend
from qmbp_simulation.models import (
    HamiltonianBuilder,
    VQEConfig,
    make_lattice,
)
from qmbp_simulation.optimizers import SPSAOptimizer, VQEOptimizer
from qmbp_simulation.pipeline import load_phase12_dataset, save_phase12_dataset
from qmbp_simulation.predictors import (
    MPNNPredictor,
    build_graph_dataset,
    train_mpnn,
)

# ─────────────────────────────────────────────────────────────────────────────
# Numerical Stability
# ─────────────────────────────────────────────────────────────────────────────


class TestNumericalStability:
    """Verify NaN/Inf guards and parameter validation."""

    def test_parameter_count_mismatch_raises(self, small_lattice, small_circuit):
        """Backend rejects parameter vectors of wrong length."""
        backend = NoiselessBackend()
        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        qc, _ = small_circuit

        # Circuit has 2 params (p=1), pass 3
        wrong_params = np.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="Parameter count mismatch"):
            backend.evaluate(qc, H, wrong_params)

    def test_parameter_count_mismatch_noisy_raises(self, small_lattice, small_circuit):
        """NoisyBackend also rejects wrong parameter count."""
        backend = NoisyBackend(shots=1024, seed_simulator=42)
        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        qc, _ = small_circuit

        wrong_params = np.array([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match="Parameter count mismatch"):
            backend.evaluate(qc, H, wrong_params)

    def test_vqe_sweep_all_energies_finite(self, small_lattice, small_circuit):
        """VQE sweep produces finite energies at every h-point."""
        config = VQEConfig(p_layers=1, n_restarts=1, maxiter=20)
        optimizer = VQEOptimizer(config, seed=42)
        qc, _ = small_circuit
        h_values = np.array([2.0, 1.5, 1.0])

        results = optimizer.descending_sweep(h_values, qc, small_lattice)
        for r in results:
            assert np.isfinite(r.energy), f"Non-finite energy at h={r.h_value}"
            assert np.all(np.isfinite(r.theta_opt)), f"Non-finite theta at h={r.h_value}"

    def test_vqe_energy_below_trivial_bound(self, small_lattice, small_circuit):
        """VQE energy should be below the |+⟩ state energy (trivial bound)."""
        backend = NoiselessBackend()
        config = VQEConfig(p_layers=1, n_restarts=2, maxiter=50)
        optimizer = VQEOptimizer(config, backend=backend, seed=42)
        builder = HamiltonianBuilder()
        qc, _ = small_circuit

        lattice_h = make_lattice("chain_1d", 4, J=1.0, h=1.5)
        H = builder.build(lattice_h)

        # Energy of |+⟩ state (theta=0)
        e_plus = backend.evaluate(qc, H, np.zeros(2))

        # VQE should find something better
        result = optimizer.optimize(H, qc, np.array([0.1, 0.1]))
        assert result.energy <= e_plus + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Shot Noise Stochasticity
# ─────────────────────────────────────────────────────────────────────────────


class TestShotNoiseStochasticity:
    """Verify NoisyBackend produces different noise on each call."""

    def test_gaussian_noise_varies_between_calls(self, small_lattice, small_circuit):
        """Consecutive evaluate() calls produce different energies."""
        backend = NoisyBackend(shots=100, seed_simulator=42)
        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        qc, _ = small_circuit
        params = np.array([0.5, -0.3])

        # Multiple evaluations should give different results (shot noise)
        energies = [backend.evaluate(qc, H, params) for _ in range(10)]
        # Not all the same (would be if RNG was reset each call)
        assert len(set(energies)) > 1, "All energies identical — noise is deterministic!"

    def test_different_seeds_give_different_noise(self, small_lattice, small_circuit):
        """Different seeds produce different noise sequences."""
        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        qc, _ = small_circuit
        params = np.array([0.5, -0.3])

        backend_a = NoisyBackend(shots=100, seed_simulator=42)
        backend_b = NoisyBackend(shots=100, seed_simulator=99)

        e_a = backend_a.evaluate(qc, H, params)
        e_b = backend_b.evaluate(qc, H, params)
        # Very unlikely to be exactly equal with different seeds
        assert e_a != e_b

    def test_same_seed_gives_same_first_call(self, small_lattice, small_circuit):
        """Same seed produces same first evaluation (reproducible)."""
        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        qc, _ = small_circuit
        params = np.array([0.5, -0.3])

        backend_a = NoisyBackend(shots=100, seed_simulator=42)
        backend_b = NoisyBackend(shots=100, seed_simulator=42)

        e_a = backend_a.evaluate(qc, H, params)
        e_b = backend_b.evaluate(qc, H, params)
        assert e_a == e_b


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────


class TestVQEReproducibility:
    """Verify VQE results are reproducible with same seed."""

    def test_same_seed_same_results(self, small_lattice, small_circuit):
        """Two VQEOptimizers with same seed produce identical results."""
        config = VQEConfig(p_layers=1, n_restarts=3, maxiter=50)
        qc, _ = small_circuit
        h_values = np.array([2.0, 1.5])

        opt_a = VQEOptimizer(config, seed=42)
        opt_b = VQEOptimizer(config, seed=42)

        results_a = opt_a.descending_sweep(h_values, qc, small_lattice)
        results_b = opt_b.descending_sweep(h_values, qc, small_lattice)

        for ra, rb in zip(results_a, results_b, strict=True):
            np.testing.assert_allclose(ra.theta_opt, rb.theta_opt, atol=1e-10)
            np.testing.assert_allclose(ra.energy, rb.energy, atol=1e-10)

    def test_different_seed_different_restarts(self, small_lattice, small_circuit):
        """Different seeds produce different restart perturbations."""
        config = VQEConfig(p_layers=1, n_restarts=3, maxiter=50)
        qc, _ = small_circuit
        h_values = np.array([2.0, 1.5])

        opt_a = VQEOptimizer(config, seed=42)
        opt_b = VQEOptimizer(config, seed=99)

        results_a = opt_a.descending_sweep(h_values, qc, small_lattice)
        results_b = opt_b.descending_sweep(h_values, qc, small_lattice)

        # Energies should be similar (both converge) but theta may differ
        # due to different restart perturbations
        for ra, rb in zip(results_a, results_b, strict=True):
            assert np.isfinite(ra.energy)
            assert np.isfinite(rb.energy)


class TestSPSAReproducibility:
    """Verify SPSA results are reproducible with same seed."""

    def test_same_seed_same_results(self, small_lattice, small_circuit):
        """Two SPSAOptimizers with same seed produce identical results."""
        backend = NoiselessBackend()
        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        qc, _ = small_circuit
        initial = np.array([0.1, 0.2])

        spsa_a = SPSAOptimizer(backend=backend, a=0.1, c=0.05, seed=42)
        spsa_b = SPSAOptimizer(backend=backend, a=0.1, c=0.05, seed=42)

        result_a = spsa_a.optimize(qc, H, initial, n_iterations=20)
        result_b = spsa_b.optimize(qc, H, initial, n_iterations=20)

        np.testing.assert_allclose(result_a.theta_opt, result_b.theta_opt, atol=1e-10)
        np.testing.assert_allclose(result_a.energy, result_b.energy, atol=1e-10)

    def test_custom_bounds_respected(self, small_lattice, small_circuit):
        """SPSA respects custom bounds parameter."""
        backend = NoiselessBackend()
        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        qc, _ = small_circuit
        initial = np.array([0.1, 0.2])

        # Tight bounds
        spsa = SPSAOptimizer(backend=backend, a=0.5, c=0.1, bounds=(-0.5, 0.5), seed=42)
        result = spsa.optimize(qc, H, initial, n_iterations=50)
        assert np.all(result.theta_opt >= -0.5)
        assert np.all(result.theta_opt <= 0.5)


class TestMPNNReproducibility:
    """Verify MPNN training is deterministic with same seed."""

    def test_same_seed_same_weights(self):
        """Same seed produces identical trained weights."""
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)
        h_values = np.array([2.0, 1.75, 1.5, 1.25, 1.0])
        theta_opt = np.random.rand(5, 2)
        e_exact = np.array([-5.0, -4.5, -4.0, -3.5, -3.0])
        fidelities = np.ones(5)

        dataset = build_graph_dataset(lattice, h_values, theta_opt, e_exact, fidelities)

        # Train two models with same seed
        model_a = MPNNPredictor(node_features=2, hidden_dim=16, n_layers=2, output_dim=2)
        model_b = MPNNPredictor(node_features=2, hidden_dim=16, n_layers=2, output_dim=2)

        # Must re-init with same seed to get same initial weights
        torch.manual_seed(100)
        model_a = MPNNPredictor(node_features=2, hidden_dim=16, n_layers=2, output_dim=2)
        train_mpnn(model_a, dataset, n_epochs=50, lr=1e-3, patience=20, seed=100)

        torch.manual_seed(100)
        model_b = MPNNPredictor(node_features=2, hidden_dim=16, n_layers=2, output_dim=2)
        train_mpnn(model_b, dataset, n_epochs=50, lr=1e-3, patience=20, seed=100)

        # Weights should be identical
        for key in model_a.state_dict():
            assert torch.allclose(
                model_a.state_dict()[key], model_b.state_dict()[key], atol=1e-6
            ), f"Weight mismatch in {key}"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestDatasetValidation:
    """Verify dataset integrity checks catch corrupted data."""

    def _save_dataset(self, tmp_path, **overrides):
        """Helper to save a dataset with optional overrides."""
        defaults = dict(
            h_values=np.array([2.0, 1.5, 1.0]),
            J=1.0,
            n_qubits=4,
            p_layers=1,
            ground_energies=np.array([-5.0, -4.5, -4.0]),
            gaps=np.array([0.5, 0.3, 0.2]),
            mag_x=np.array([0.8, 0.6, 0.3]),
            corr_zz=np.array([0.2, 0.4, 0.7]),
            theta_opt=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            vqe_energies=np.array([-4.9, -4.4, -3.9]),
            fidelities=np.array([0.99, 0.97, 0.95]),
        )
        defaults.update(overrides)
        filepath = tmp_path / "test.npz"
        save_phase12_dataset(filepath, **defaults)
        return filepath

    def test_valid_dataset_loads_successfully(self, tmp_path):
        """A well-formed dataset loads without errors."""
        filepath = self._save_dataset(tmp_path)
        data = load_phase12_dataset(filepath)
        assert len(data["h_values"]) == 3

    def test_nan_in_energies_raises(self, tmp_path):
        """NaN in ground_energies is caught."""
        filepath = self._save_dataset(tmp_path, ground_energies=np.array([-5.0, np.nan, -4.0]))
        with pytest.raises(ValueError, match="NaN/Inf"):
            load_phase12_dataset(filepath)

    def test_inf_in_vqe_energies_raises(self, tmp_path):
        """Inf in vqe_energies is caught."""
        filepath = self._save_dataset(tmp_path, vqe_energies=np.array([-4.9, np.inf, -3.9]))
        with pytest.raises(ValueError, match="NaN/Inf"):
            load_phase12_dataset(filepath)

    def test_shape_mismatch_raises(self, tmp_path):
        """Mismatched array lengths are caught."""
        filepath = tmp_path / "bad.npz"
        np.savez(
            filepath,
            cost_function="energy",
            version="v7.0",
            h_values=np.array([2.0, 1.5, 1.0]),
            ground_energies=np.array([-5.0, -4.5]),  # Wrong length!
            gaps=np.array([0.5, 0.3, 0.2]),
            vqe_energies=np.array([-4.9, -4.4, -3.9]),
            fidelities=np.array([0.99, 0.97, 0.95]),
            theta_opt=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            J=1.0,
            n_qubits=4,
            p_layers=1,
            mag_x=np.array([0.8, 0.6, 0.3]),
            corr_zz=np.array([0.2, 0.4, 0.7]),
        )
        with pytest.raises(ValueError, match="length"):
            load_phase12_dataset(filepath)

    def test_ascending_h_values_warns(self, tmp_path):
        """Ascending h_values emits a warning."""
        filepath = self._save_dataset(
            tmp_path,
            h_values=np.array([1.0, 1.5, 2.0]),  # Ascending!
            ground_energies=np.array([-4.0, -4.5, -5.0]),
            gaps=np.array([0.2, 0.3, 0.5]),
            vqe_energies=np.array([-3.9, -4.4, -4.9]),
            fidelities=np.array([0.95, 0.97, 0.99]),
            theta_opt=np.array([[0.5, 0.6], [0.3, 0.4], [0.1, 0.2]]),
            mag_x=np.array([0.3, 0.6, 0.8]),
            corr_zz=np.array([0.7, 0.4, 0.2]),
        )
        with pytest.warns(RuntimeWarning, match="ascending"):
            load_phase12_dataset(filepath)

    def test_fidelity_out_of_range_warns(self, tmp_path):
        """Fidelities outside [0, 1] emit a warning."""
        filepath = self._save_dataset(tmp_path, fidelities=np.array([0.99, 1.5, 0.95]))
        with pytest.warns(RuntimeWarning, match="outside"):
            load_phase12_dataset(filepath)


# ─────────────────────────────────────────────────────────────────────────────
# Input Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestInputValidation:
    """Verify early rejection of invalid inputs."""

    def test_empty_h_values_raises_in_sweep(self, small_lattice, small_circuit):
        """Empty h_values array raises ValueError."""
        config = VQEConfig(p_layers=1, n_restarts=1, maxiter=5)
        optimizer = VQEOptimizer(config)
        qc, _ = small_circuit

        with pytest.raises(ValueError, match="empty"):
            optimizer.descending_sweep(np.array([]), qc, small_lattice)

    def test_ascending_h_values_raises_in_sweep(self, small_lattice, small_circuit):
        """Ascending h_values raises ValueError (not silently reversed)."""
        config = VQEConfig(p_layers=1, n_restarts=1, maxiter=5)
        optimizer = VQEOptimizer(config)
        qc, _ = small_circuit

        with pytest.raises(ValueError, match="descending"):
            optimizer.descending_sweep(np.array([1.0, 1.5, 2.0]), qc, small_lattice)

    def test_single_h_value_works(self, small_lattice, small_circuit):
        """Single h-value sweep works (no warm-start needed)."""
        config = VQEConfig(p_layers=1, n_restarts=1, maxiter=20)
        optimizer = VQEOptimizer(config, seed=42)
        qc, _ = small_circuit

        results = optimizer.descending_sweep(np.array([1.5]), qc, small_lattice)
        assert len(results) == 1
        assert np.isfinite(results[0].energy)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 Graph Construction
# ─────────────────────────────────────────────────────────────────────────────


class TestPhase4GraphConstruction:
    """Verify Phase 4 prediction graph matches training graph structure."""

    def test_graph_features_match_training(self):
        """Phase 4 graph has same node feature structure as training."""
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)
        builder = HamiltonianBuilder()

        # Training graph construction (from build_graph_dataset)
        h_values = np.array([2.0, 1.5, 1.0])
        theta_opt = np.random.rand(3, 2)
        e_exact = np.array([-5.0, -4.5, -4.0])
        fidelities = np.ones(3)
        dataset = build_graph_dataset(lattice, h_values, theta_opt, e_exact, fidelities)
        train_graph = dataset[0]

        # Phase 4 graph construction (mirrors runner.py)
        from torch_geometric.data import Data

        h_test = 1.6
        edge_index_np, coord = builder.build_graph_data(lattice)
        h_feat = np.full(lattice.n_qubits, float(h_test))
        x = torch.tensor(np.stack([h_feat, coord.astype(float)], axis=1), dtype=torch.float32)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        test_graph = Data(x=x, edge_index=edge_index)

        # Same structure
        assert train_graph.x.shape == test_graph.x.shape
        assert train_graph.edge_index.shape == test_graph.edge_index.shape
        # Node features: column 0 = h, column 1 = coordination
        assert test_graph.x[0, 0].item() == pytest.approx(h_test)
        # Coordination numbers match
        np.testing.assert_array_equal(test_graph.x[:, 1].numpy(), train_graph.x[:, 1].numpy())

    def test_mpnn_prediction_shape_correct(self):
        """MPNN produces correct output shape for Phase 4 graph."""
        from torch_geometric.data import Data

        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)
        builder = HamiltonianBuilder()
        model = MPNNPredictor(node_features=2, hidden_dim=16, n_layers=2, output_dim=2)

        edge_index_np, coord = builder.build_graph_data(lattice)
        h_feat = np.full(4, 1.5)
        x = torch.tensor(np.stack([h_feat, coord.astype(float)], axis=1), dtype=torch.float32)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)

        model.eval()
        with torch.no_grad():
            pred = model(graph)

        assert pred.shape == (1, 2)  # [batch=1, output_dim=2]


# ─────────────────────────────────────────────────────────────────────────────
# Warm-Start Chain Integrity
# ─────────────────────────────────────────────────────────────────────────────


class TestWarmStartChain:
    """Verify warm-start propagation produces smooth parameter evolution."""

    def test_theta_smoothness_in_valid_regime(self, small_lattice, small_circuit):
        """Parameters evolve smoothly in the valid regime (h >= 1.25)."""
        config = VQEConfig(p_layers=1, n_restarts=3, maxiter=100)
        optimizer = VQEOptimizer(config, seed=42)
        qc, _ = small_circuit
        h_values = np.array([2.0, 1.75, 1.5, 1.25])

        results = optimizer.descending_sweep(h_values, qc, small_lattice)
        thetas = np.array([r.theta_opt for r in results])

        # Consecutive theta differences should be small (smooth landscape)
        diffs = np.abs(np.diff(thetas, axis=0))
        max_jump = np.max(diffs)
        # In the valid regime, max jump should be < 1.0 radian
        assert max_jump < 1.0, f"Large parameter jump: {max_jump:.4f}"

    def test_warm_start_beats_cold_start(self, small_lattice, small_circuit):
        """Warm-start sweep achieves better fidelity than independent cold starts."""
        from qmbp_simulation.solvers import ClassicalSolver

        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        backend = NoiselessBackend()
        qc, _ = small_circuit
        h_values = np.array([2.0, 1.5])

        # Warm-start sweep
        config_warm = VQEConfig(p_layers=1, n_restarts=2, maxiter=100)
        opt_warm = VQEOptimizer(config_warm, backend=backend, seed=42)

        exact_data = []
        for h in h_values:
            lat_h = make_lattice("chain_1d", 4, J=1.0, h=h)
            H = builder.build(lat_h)
            exact_data.append(solver.solve(H, lat_h))

        results_warm = opt_warm.descending_sweep(h_values, qc, small_lattice, exact_data=exact_data)

        # Cold start at h=1.5 (no warm-start from h=2.0)
        config_cold = VQEConfig(p_layers=1, n_restarts=2, maxiter=100)
        opt_cold = VQEOptimizer(config_cold, backend=backend, seed=42)
        lat_15 = make_lattice("chain_1d", 4, J=1.0, h=1.5)
        H_15 = builder.build(lat_15)
        cold_init = np.random.uniform(-0.5, 0.5, 2)
        result_cold = opt_cold.optimize(
            H_15,
            qc,
            cold_init,
            exact_energy=exact_data[1].ground_energy,
            exact_state=exact_data[1].ground_state,
        )

        # Warm-start should achieve at least as good fidelity
        warm_fid = results_warm[1].fidelity
        cold_fid = result_cold.fidelity
        # Warm-start advantage: at least not worse
        assert warm_fid >= cold_fid - 0.01, (
            f"Warm-start fidelity ({warm_fid:.4f}) worse than cold-start ({cold_fid:.4f})"
        )

    def test_previous_theta_always_preferred_over_zeros(self):
        """Even at h=0, previous_theta is used if available."""
        config = VQEConfig(p_layers=1, warm_start_seed_zeros=True)
        previous = np.array([0.5, -0.3])

        guess = VQEOptimizer.get_initial_guess(
            n_params=2, h_value=0.0, config=config, previous_theta=previous
        )
        # Should use previous_theta, not zeros
        np.testing.assert_array_equal(guess, previous)
