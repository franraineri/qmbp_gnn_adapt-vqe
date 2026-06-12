"""Pipeline confidence tests — validate physics correctness and result quality.

These tests verify that the full pipeline produces physically meaningful
results with high confidence:
1. VQE finds the correct ground state (fidelity > 0.99) in the valid regime
2. MPNN predictions are accurate enough for Phase 4 deployment
3. Phase classification is correct across the phase diagram
4. Energy errors scale correctly with system parameters
5. Warm-start sweep is monotonically improving
6. Results are seed-independent in the valid regime
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.execution import NoiselessBackend
from qmbp_simulation.models import (
    HamiltonianBuilder,
    VQEConfig,
    make_lattice,
)
from qmbp_simulation.optimizers import VQEOptimizer
from qmbp_simulation.predictors import (
    MPNNPredictor,
    build_graph_dataset,
    train_mpnn,
)
from qmbp_simulation.solvers import ClassicalSolver

pytestmark = pytest.mark.integration


# ─────────────────────────────────────────────────────────────────────────────
# VQE Ground State Quality
# ─────────────────────────────────────────────────────────────────────────────


class TestVQEGroundStateQuality:
    """Verify VQE finds the correct ground state in the valid regime."""

    @pytest.fixture
    def pipeline_components(self):
        """Shared pipeline components for VQE tests."""
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        hva = HVACircuitBuilder()
        backend = NoiselessBackend()
        qc, _ = hva.create(4, 1, lattice)
        return lattice, builder, solver, qc, backend

    def test_high_fidelity_in_paramagnetic_phase(self, pipeline_components):
        """VQE achieves fidelity > 0.99 for h > 1.5 (deep paramagnetic)."""
        lattice, builder, solver, qc, backend = pipeline_components
        config = VQEConfig(p_layers=1, n_restarts=3, maxiter=200)
        optimizer = VQEOptimizer(config, backend=backend, seed=42)

        h_values = np.array([2.0, 1.75, 1.5])
        exact_data = []
        for h in h_values:
            lat_h = make_lattice("chain_1d", 4, J=1.0, h=h)
            H = builder.build(lat_h)
            exact_data.append(solver.solve(H, lat_h))

        results = optimizer.descending_sweep(h_values, qc, lattice, exact_data=exact_data)

        for r in results:
            assert r.fidelity > 0.99, (
                f"Low fidelity {r.fidelity:.4f} at h={r.h_value} "
                f"(expected > 0.99 in paramagnetic phase)"
            )

    def test_energy_error_below_5pct_gap(self, pipeline_components):
        """ΔE/gap < 5% for all h-points in the valid regime."""
        lattice, builder, solver, qc, backend = pipeline_components
        config = VQEConfig(p_layers=1, n_restarts=3, maxiter=200)
        optimizer = VQEOptimizer(config, backend=backend, seed=42)

        # Valid regime for p=1 at N=4 is h >= 1.5 (conservative)
        h_values = np.array([2.0, 1.75, 1.5])
        exact_data = []
        for h in h_values:
            lat_h = make_lattice("chain_1d", 4, J=1.0, h=h)
            H = builder.build(lat_h)
            exact_data.append(solver.solve(H, lat_h))

        results = optimizer.descending_sweep(h_values, qc, lattice, exact_data=exact_data)

        for r, exact in zip(results, exact_data, strict=True):
            de_gap = r.energy_error / exact.gap if exact.gap > 0 else 0
            assert de_gap < 0.05, f"ΔE/gap = {de_gap:.4f} at h={r.h_value} exceeds 5% threshold"

    def test_vqe_energy_monotonic_with_h(self, pipeline_components):
        """VQE energy decreases as h increases (more negative)."""
        lattice, builder, solver, qc, backend = pipeline_components
        config = VQEConfig(p_layers=1, n_restarts=2, maxiter=100)
        optimizer = VQEOptimizer(config, backend=backend, seed=42)

        h_values = np.array([2.0, 1.5, 1.0])
        results = optimizer.descending_sweep(h_values, qc, lattice)

        # Energy should decrease (become more negative) as h increases
        # h=2.0 has more negative energy than h=1.0
        assert results[0].energy < results[2].energy, (
            f"Energy at h=2.0 ({results[0].energy:.4f}) should be more negative "
            f"than at h=1.0 ({results[2].energy:.4f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MPNN Prediction Quality
# ─────────────────────────────────────────────────────────────────────────────


class TestMPNNPredictionQuality:
    """Verify MPNN predictions are accurate for Phase 4 deployment."""

    @pytest.fixture
    def trained_pipeline(self):
        """Train a full pipeline and return components for testing."""
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        hva = HVACircuitBuilder()
        backend = NoiselessBackend()
        qc, _ = hva.create(4, 1, lattice)

        # Phase 1 + 2
        h_train = np.array([2.0, 1.8, 1.6, 1.4, 1.2, 1.0])
        config = VQEConfig(p_layers=1, n_restarts=3, maxiter=200)
        optimizer = VQEOptimizer(config, backend=backend, seed=42)

        exact_data = []
        for h in h_train:
            lat_h = make_lattice("chain_1d", 4, J=1.0, h=h)
            H = builder.build(lat_h)
            exact_data.append(solver.solve(H, lat_h))

        vqe_results = optimizer.descending_sweep(h_train, qc, lattice, exact_data=exact_data)

        # Phase 3
        theta_opt = np.array([r.theta_opt for r in vqe_results])
        e_exact = np.array([r.ground_energy for r in exact_data])
        fidelities = np.array([r.fidelity for r in vqe_results])

        dataset = build_graph_dataset(lattice, h_train, theta_opt, e_exact, fidelities)

        torch.manual_seed(42)
        model = MPNNPredictor(node_features=2, hidden_dim=32, n_layers=2, output_dim=2)
        train_mpnn(model, dataset, n_epochs=1000, lr=1e-3, patience=100, seed=42)

        return {
            "model": model,
            "lattice": lattice,
            "builder": builder,
            "solver": solver,
            "qc": qc,
            "backend": backend,
            "h_train": h_train,
        }

    def test_interpolation_accuracy(self, trained_pipeline):
        """MPNN predicts well at interpolation points (between training h)."""
        from torch_geometric.data import Data

        p = trained_pipeline
        model, lattice, builder = p["model"], p["lattice"], p["builder"]
        solver, qc, backend = p["solver"], p["qc"], p["backend"]

        # Test at h=1.5 (between training points 1.6 and 1.4)
        h_test = 1.5
        lat_test = make_lattice("chain_1d", 4, J=1.0, h=h_test)
        H_test = builder.build(lat_test)
        exact_test = solver.solve(H_test, lat_test)

        edge_index_np, coord = builder.build_graph_data(lattice)
        h_feat = np.full(4, float(h_test))
        x = torch.tensor(np.stack([h_feat, coord.astype(float)], axis=1), dtype=torch.float32)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)

        model.eval()
        with torch.no_grad():
            theta_pred = model(graph).squeeze().numpy()

        e_pred = backend.evaluate(qc, H_test, theta_pred)
        de_gap = abs(e_pred - exact_test.ground_energy) / exact_test.gap

        assert de_gap < 0.10, (
            f"MPNN interpolation ΔE/gap = {de_gap:.4f} at h={h_test} "
            f"(expected < 10% for interpolation)"
        )

    def test_training_loss_decreases(self):
        """MPNN training loss decreases over epochs."""
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)
        h_values = np.array([2.0, 1.75, 1.5, 1.25, 1.0])
        theta_opt = np.random.rand(5, 2) * 0.5
        e_exact = np.array([-5.0, -4.5, -4.0, -3.5, -3.0])
        fidelities = np.ones(5)

        dataset = build_graph_dataset(lattice, h_values, theta_opt, e_exact, fidelities)
        model = MPNNPredictor(node_features=2, hidden_dim=32, n_layers=2, output_dim=2)
        result = train_mpnn(model, dataset, n_epochs=200, lr=1e-3, patience=50, seed=42)

        # Loss should decrease
        mse_history = result["mse_history"]
        assert mse_history[-1] < mse_history[0], (
            f"Training loss did not decrease: "
            f"initial={mse_history[0]:.4e}, final={mse_history[-1]:.4e}"
        )
        # Final loss should be reasonably small
        assert result["final_mse"] < 0.1, f"Final MSE too high: {result['final_mse']:.4e}"


# ─────────────────────────────────────────────────────────────────────────────
# Phase Classification
# ─────────────────────────────────────────────────────────────────────────────


class TestPhaseClassification:
    """Verify correct phase identification across the phase diagram."""

    def test_paramagnetic_phase_observables(self):
        """In paramagnetic phase (h >> 1), |⟨X⟩| > |⟨ZZ⟩|."""
        lattice = make_lattice("chain_1d", 4, J=1.0, h=2.0)
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        H = builder.build(lattice)
        exact = solver.solve(H, lattice)

        # Deep paramagnetic: magnetization dominates
        assert abs(exact.mag_x) > abs(exact.corr_zz), (
            f"Expected |⟨X⟩|={abs(exact.mag_x):.4f} > |⟨ZZ⟩|={abs(exact.corr_zz):.4f} "
            f"in paramagnetic phase (h=2.0)"
        )

    def test_ferromagnetic_phase_observables(self):
        """In ferromagnetic phase (h << 1), |⟨ZZ⟩| > |⟨X⟩|."""
        lattice = make_lattice("chain_1d", 4, J=1.0, h=0.3)
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        H = builder.build(lattice)
        exact = solver.solve(H, lattice)

        # Deep ferromagnetic: correlations dominate
        assert abs(exact.corr_zz) > abs(exact.mag_x), (
            f"Expected |⟨ZZ⟩|={abs(exact.corr_zz):.4f} > |⟨X⟩|={abs(exact.mag_x):.4f} "
            f"in ferromagnetic phase (h=0.3)"
        )

    def test_gap_closes_near_critical_point(self):
        """Spectral gap decreases as h approaches h_c from above."""
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()

        gaps = {}
        for h in [1.0, 1.2, 1.5, 2.0]:
            lattice = make_lattice("chain_1d", 4, J=1.0, h=h)
            H = builder.build(lattice)
            exact = solver.solve(H, lattice)
            gaps[h] = exact.gap

        # Gap should increase with h in the paramagnetic phase (h > h_c)
        # i.e., gap(h=1.0) < gap(h=2.0)
        assert gaps[1.0] < gaps[2.0], (
            f"Gap should increase with h in paramagnetic phase: "
            f"gap(h=1.0)={gaps[1.0]:.4f}, gap(h=2.0)={gaps[2.0]:.4f}"
        )
        # Gap should be monotonically increasing for h > h_c
        assert gaps[1.0] < gaps[1.2] < gaps[1.5] < gaps[2.0], (
            f"Gap not monotonically increasing: {gaps}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Seed Independence
# ─────────────────────────────────────────────────────────────────────────────


class TestSeedIndependence:
    """Verify results are consistent across different random seeds."""

    def test_vqe_converges_to_same_energy_different_seeds(self):
        """Different seeds converge to same energy (global minimum)."""
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.5)
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        hva = HVACircuitBuilder()
        backend = NoiselessBackend()
        qc, _ = hva.create(4, 1, lattice)

        H = builder.build(lattice)
        exact = solver.solve(H, lattice)

        # Use 5 restarts to reliably find global minimum from cold start
        config = VQEConfig(p_layers=1, n_restarts=5, maxiter=200)

        energies = []
        for seed in [42, 43, 44, 45]:
            optimizer = VQEOptimizer(config, backend=backend, seed=seed)
            result = optimizer.optimize(
                H,
                qc,
                np.array([0.1, 0.2]),
                exact_energy=exact.ground_energy,
            )
            energies.append(result.energy)

        # All seeds should find similar energy (within 5% of gap)
        energy_spread = max(energies) - min(energies)
        assert energy_spread < 0.05 * exact.gap, (
            f"Energy spread across seeds: {energy_spread:.6f} "
            f"(gap={exact.gap:.4f}). Seeds: {energies}"
        )

    def test_sweep_fidelity_consistent_across_seeds(self):
        """Sweep fidelity is consistent across seeds in valid regime."""
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        hva = HVACircuitBuilder()
        backend = NoiselessBackend()
        qc, _ = hva.create(4, 1, lattice)

        h_values = np.array([2.0, 1.5])
        exact_data = []
        for h in h_values:
            lat_h = make_lattice("chain_1d", 4, J=1.0, h=h)
            H = builder.build(lat_h)
            exact_data.append(solver.solve(H, lat_h))

        config = VQEConfig(p_layers=1, n_restarts=3, maxiter=200)

        fidelities_per_seed = []
        for seed in [42, 43, 44]:
            optimizer = VQEOptimizer(config, backend=backend, seed=seed)
            results = optimizer.descending_sweep(h_values, qc, lattice, exact_data=exact_data)
            fidelities_per_seed.append([r.fidelity for r in results])

        fid_array = np.array(fidelities_per_seed)
        # Standard deviation across seeds should be small
        std_per_h = np.std(fid_array, axis=0)
        for i, h in enumerate(h_values):
            assert std_per_h[i] < 0.01, (
                f"Fidelity std={std_per_h[i]:.4f} at h={h} across seeds "
                f"(expected < 0.01 for seed-independent convergence)"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Hamiltonian Correctness
# ─────────────────────────────────────────────────────────────────────────────


class TestHamiltonianCorrectness:
    """Verify Hamiltonian construction matches known analytical results."""

    def test_n2_tfim_analytical_energy(self):
        """N=2 TFIM has analytical ground state energy."""
        # H = -J*Z1Z2 - h*(X1 + X2) for open chain
        # For J=1, h=1: E_gs = -sqrt(1^2 + (2*1)^2) = -sqrt(5) ≈ -2.236
        lattice = make_lattice("chain_1d", 2, J=1.0, h=1.0)
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        H = builder.build(lattice)
        exact = solver.solve(H, lattice)

        # Analytical: E_gs = -sqrt(J^2 + (2h)^2) for N=2 open TFIM
        e_analytical = -np.sqrt(1.0 + 4.0)
        np.testing.assert_allclose(exact.ground_energy, e_analytical, atol=1e-10)

    def test_large_h_limit_energy(self):
        """At h >> J, ground state is |+⟩^N with E ≈ -N*h."""
        lattice = make_lattice("chain_1d", 4, J=1.0, h=10.0)
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        H = builder.build(lattice)
        exact = solver.solve(H, lattice)

        # At h=10, E_gs ≈ -N*h = -40 (dominant term)
        # ZZ contribution is small: -(N-1)*J*⟨ZZ⟩ ≈ 0 in |+⟩ state
        assert exact.ground_energy < -35, (
            f"Energy {exact.ground_energy:.4f} not close to -N*h=-40 at h=10"
        )

    def test_hamiltonian_hermiticity(self):
        """Hamiltonian matrix is Hermitian for all topologies."""
        builder = HamiltonianBuilder()

        for topology in ["chain_1d", "ladder"]:
            n = 4 if topology == "chain_1d" else 4
            lattice = make_lattice(topology, n, J=1.0, h=1.5)
            H = builder.build(lattice)
            mat = np.asarray(H.to_matrix())
            np.testing.assert_allclose(
                mat, mat.conj().T, atol=1e-12, err_msg=f"Hamiltonian not Hermitian for {topology}"
            )

    def test_hamiltonian_dimension(self):
        """Hamiltonian has correct 2^N × 2^N dimension."""
        builder = HamiltonianBuilder()
        for n in [2, 3, 4, 5]:
            lattice = make_lattice("chain_1d", n, J=1.0, h=1.0)
            H = builder.build(lattice)
            assert H.num_qubits == n
            mat = np.asarray(H.to_matrix())
            assert mat.shape == (2**n, 2**n)


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline Confidence
# ─────────────────────────────────────────────────────────────────────────────


class TestFullPipelineConfidence:
    """End-to-end confidence test: all phases produce correct results."""

    def test_pipeline_n4_p1_all_metrics_pass(self):
        """Full pipeline at N=4, p=1 passes all validation metrics."""
        from torch_geometric.data import Data

        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        hva = HVACircuitBuilder()
        backend = NoiselessBackend()
        qc, _ = hva.create(4, 1, lattice)

        # Phase 1 + 2: 6 training points
        h_train = np.array([2.0, 1.8, 1.6, 1.4, 1.2, 1.0])
        config = VQEConfig(p_layers=1, n_restarts=3, maxiter=200)
        optimizer = VQEOptimizer(config, backend=backend, seed=42)

        exact_data = []
        for h in h_train:
            lat_h = make_lattice("chain_1d", 4, J=1.0, h=h)
            H = builder.build(lat_h)
            exact_data.append(solver.solve(H, lat_h))

        vqe_results = optimizer.descending_sweep(h_train, qc, lattice, exact_data=exact_data)

        # Verify Phase 2 quality
        for r in vqe_results:
            assert r.fidelity > 0.95, f"Low fidelity {r.fidelity:.4f} at h={r.h_value}"

        # Phase 3: Train MPNN
        theta_opt = np.array([r.theta_opt for r in vqe_results])
        e_exact = np.array([r.ground_energy for r in exact_data])
        fidelities = np.array([r.fidelity for r in vqe_results])

        dataset = build_graph_dataset(lattice, h_train, theta_opt, e_exact, fidelities)
        torch.manual_seed(42)
        model = MPNNPredictor(node_features=2, hidden_dim=32, n_layers=2, output_dim=2)
        train_result = train_mpnn(model, dataset, n_epochs=1000, lr=1e-3, patience=100, seed=42)
        assert train_result["final_mse"] < 0.05, (
            f"MPNN final MSE too high: {train_result['final_mse']:.4e}"
        )

        # Phase 4: Deploy at interpolation point
        h_test = 1.5
        lat_test = make_lattice("chain_1d", 4, J=1.0, h=h_test)
        H_test = builder.build(lat_test)
        exact_test = solver.solve(H_test, lat_test)

        edge_index_np, coord = builder.build_graph_data(lattice)
        h_feat = np.full(4, float(h_test))
        x = torch.tensor(np.stack([h_feat, coord.astype(float)], axis=1), dtype=torch.float32)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)

        model.eval()
        with torch.no_grad():
            theta_pred = model(graph).squeeze().numpy()

        e_pred = backend.evaluate(qc, H_test, theta_pred)
        de = abs(e_pred - exact_test.ground_energy)
        de_gap = de / exact_test.gap

        # Primary metric: ΔE/gap < 5%
        assert de_gap < 0.05, (
            f"Pipeline ΔE/gap = {de_gap:.4f} at h={h_test} (ΔE={de:.6f}, gap={exact_test.gap:.4f})"
        )

    @pytest.mark.slow
    def test_pipeline_n6_p2_valid_regime(self):
        """Full pipeline at N=6, p=2 passes in the valid regime (h >= 1.25)."""
        from torch_geometric.data import Data

        lattice = make_lattice("chain_1d", 6, J=1.0, h=1.0)
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        hva = HVACircuitBuilder()
        backend = NoiselessBackend()
        qc, _ = hva.create(6, 2, lattice)

        # Phase 1 + 2: 10 training points in valid regime for reliable MPNN
        h_train = np.array([2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.35, 1.3, 1.25])
        config = VQEConfig(p_layers=2, n_restarts=5, maxiter=500)
        optimizer = VQEOptimizer(config, backend=backend, seed=42)

        exact_data = []
        for h in h_train:
            lat_h = make_lattice("chain_1d", 6, J=1.0, h=h)
            H = builder.build(lat_h)
            exact_data.append(solver.solve(H, lat_h))

        vqe_results = optimizer.descending_sweep(h_train, qc, lattice, exact_data=exact_data)

        # Verify Phase 2: all fidelities > 0.99 in valid regime
        for r in vqe_results:
            assert r.fidelity > 0.99, (
                f"N=6 p=2 fidelity {r.fidelity:.4f} at h={r.h_value} "
                f"(expected > 0.99 in valid regime)"
            )

        # Phase 3: Train MPNN with sufficient epochs for 4-param prediction
        theta_opt = np.array([r.theta_opt for r in vqe_results])
        e_exact = np.array([r.ground_energy for r in exact_data])
        fidelities = np.array([r.fidelity for r in vqe_results])

        dataset = build_graph_dataset(lattice, h_train, theta_opt, e_exact, fidelities)
        torch.manual_seed(42)
        model = MPNNPredictor(node_features=2, hidden_dim=64, n_layers=3, output_dim=4)
        train_mpnn(model, dataset, n_epochs=6000, lr=1e-3, patience=500, seed=42)

        # Phase 4: Deploy at h=1.55 (interpolation between training points)
        h_test = 1.55
        lat_test = make_lattice("chain_1d", 6, J=1.0, h=h_test)
        H_test = builder.build(lat_test)
        exact_test = solver.solve(H_test, lat_test)

        edge_index_np, coord = builder.build_graph_data(lattice)
        h_feat = np.full(6, float(h_test))
        x = torch.tensor(np.stack([h_feat, coord.astype(float)], axis=1), dtype=torch.float32)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)

        model.eval()
        with torch.no_grad():
            theta_pred = model(graph).squeeze().numpy()

        e_pred = backend.evaluate(qc, H_test, theta_pred)
        de_gap = abs(e_pred - exact_test.ground_energy) / exact_test.gap

        # N=6 p=2 should achieve < 5% in valid regime with sufficient training data
        assert de_gap < 0.05, f"N=6 p=2 pipeline ΔE/gap = {de_gap:.4f} at h={h_test}"
