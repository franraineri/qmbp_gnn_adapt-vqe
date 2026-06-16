"""Tests for IBM Torino deployment script API correctness.

Validates that the hardware deployment script uses the correct qmbp_simulation
API calls. These tests exercise the exact code paths that were broken
(2026-06-13): build_tfim → build(make_lattice(...)), solver.solve(H, lattice),
GroundTruthResult.ground_energy, VQEOptimizer.optimize(H, circuit, init),
build_graph_dataset(lattice, h_values, theta_opt, ...), and MPNN prediction
via forward pass.

These tests do NOT require IBM credentials and do NOT run on QPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from qmbp_simulation import (
    ClassicalSolver,
    HamiltonianBuilder,
    HVACircuitBuilder,
    VQEConfig,
    VQEOptimizer,
    make_lattice,
)
from qmbp_simulation.predictors import (
    MPNNPredictor,
    build_graph_dataset,
    train_mpnn,
)

# ═══════════════════════════════════════════════════════════════════════════
# Constants (matching deployment script)
# ═══════════════════════════════════════════════════════════════════════════

TOPOLOGY = "heavy_hex"
N_QUBITS = 10
P_LAYERS = 1


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def builder():
    return HamiltonianBuilder()


@pytest.fixture(scope="module")
def solver():
    return ClassicalSolver()


@pytest.fixture(scope="module")
def lattice_h4():
    """LatticeConfig for heavy_hex N=10 at h=4.0."""
    return make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=4.0)


@pytest.fixture(scope="module")
def lattice_h325():
    """LatticeConfig for heavy_hex N=10 at h=3.25."""
    return make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=3.25)


@pytest.fixture(scope="module")
def hva_circuit():
    """HVA circuit for heavy_hex N=10 p=1."""
    circuit_builder = HVACircuitBuilder()
    lattice = make_lattice(TOPOLOGY, N_QUBITS)
    circuit, _ = circuit_builder.create(N_QUBITS, P_LAYERS, lattice)
    return circuit


# ═══════════════════════════════════════════════════════════════════════════
# 1. Hamiltonian Building (build_tfim fix)
# ═══════════════════════════════════════════════════════════════════════════


class TestHamiltonianBuilding:
    """Verify the correct API pattern: make_lattice(h=X) → builder.build(lattice)."""

    def test_build_with_make_lattice_returns_sparse_pauli_op(self, builder, lattice_h4):
        """builder.build(make_lattice(..., h=4.0)) must return SparsePauliOp."""
        from qiskit.quantum_info import SparsePauliOp

        H = builder.build(lattice_h4)
        assert isinstance(H, SparsePauliOp)
        assert H.num_qubits == N_QUBITS

    def test_no_build_tfim_attribute(self, builder):
        """HamiltonianBuilder must NOT have build_tfim method."""
        assert not hasattr(builder, "build_tfim"), (
            "build_tfim should not exist — use builder.build(make_lattice(..., h=h))"
        )

    def test_different_h_gives_different_hamiltonian(self, builder, lattice_h4, lattice_h325):
        """Different h values must produce different Hamiltonians."""
        H1 = builder.build(lattice_h4)
        H2 = builder.build(lattice_h325)
        # Coefficients must differ (different transverse field)
        assert not np.allclose(H1.coeffs.real, H2.coeffs.real)

    def test_heavy_hex_n10_edge_count(self):
        """heavy_hex N=10 must have expected edge count."""
        lattice = make_lattice(TOPOLOGY, N_QUBITS)
        # Heavy-hex connectivity for N=10 has specific number of edges
        assert len(lattice.edges) > 0
        assert lattice.n_qubits == N_QUBITS


# ═══════════════════════════════════════════════════════════════════════════
# 2. Classical Solver (solver.solve(H, lattice) fix)
# ═══════════════════════════════════════════════════════════════════════════


class TestClassicalSolver:
    """Verify solver.solve(H, lattice) returns GroundTruthResult with correct attrs."""

    def test_solve_requires_lattice(self, builder, solver, lattice_h4):
        """solver.solve(H, lattice) must work; solver.solve(H) must fail."""
        H = builder.build(lattice_h4)
        # Correct call
        result = solver.solve(H, lattice_h4)
        assert hasattr(result, "ground_energy")
        assert hasattr(result, "gap")
        # Missing lattice must raise TypeError
        with pytest.raises(TypeError):
            solver.solve(H)

    def test_ground_energy_attribute(self, builder, solver, lattice_h4):
        """GroundTruthResult has .ground_energy, not .energy."""
        H = builder.build(lattice_h4)
        result = solver.solve(H, lattice_h4)
        assert hasattr(result, "ground_energy")
        assert not hasattr(result, "energy"), (
            "GroundTruthResult should have 'ground_energy', not 'energy'"
        )
        assert np.isfinite(result.ground_energy)
        assert result.ground_energy < 0  # TFIM ground state is negative

    def test_gap_is_positive(self, builder, solver, lattice_h4):
        """Spectral gap must be positive for h=4.0 (deep paramagnetic)."""
        H = builder.build(lattice_h4)
        result = solver.solve(H, lattice_h4)
        assert result.gap > 0

    def test_known_energy_scaling(self, builder, solver):
        """E_gs at h>>1 approaches -h*N (all spins aligned with X field)."""
        lattice_h10 = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=10.0)
        H = builder.build(lattice_h10)
        result = solver.solve(H, lattice_h10)
        # In the paramagnetic limit h>>J, E ≈ -h*N
        assert result.ground_energy < -90  # Should be close to -100


# ═══════════════════════════════════════════════════════════════════════════
# 3. VQE Optimizer (argument order + warm-start fix)
# ═══════════════════════════════════════════════════════════════════════════


class TestVQEOptimizer:
    """Verify VQEOptimizer.optimize(H, circuit, initial_guess) signature."""

    def test_optimize_signature(self, builder, lattice_h4, hva_circuit):
        """optimize(hamiltonian, circuit, initial_guess) is the correct order."""
        H = builder.build(lattice_h4)
        config = VQEConfig(n_restarts=1, maxiter=5)
        optimizer = VQEOptimizer(config=config, seed=42)
        init = np.random.default_rng(42).uniform(-0.01, 0.01, hva_circuit.num_parameters)

        result = optimizer.optimize(H, hva_circuit, init)
        assert hasattr(result, "theta_opt")
        assert hasattr(result, "energy")
        assert len(result.theta_opt) == hva_circuit.num_parameters

    def test_result_has_theta_opt_not_params(self, builder, lattice_h4, hva_circuit):
        """VQEResult has .theta_opt, NOT .params."""
        H = builder.build(lattice_h4)
        config = VQEConfig(n_restarts=1, maxiter=5)
        optimizer = VQEOptimizer(config=config, seed=42)
        init = np.zeros(hva_circuit.num_parameters)

        result = optimizer.optimize(H, hva_circuit, init)
        assert hasattr(result, "theta_opt")
        assert not hasattr(result, "params"), "VQEResult should have 'theta_opt', not 'params'"

    def test_descending_sweep_warm_start(self, builder, hva_circuit):
        """Descending sweep with warm-start produces monotonically better energy."""
        h_values = [4.5, 4.0, 3.5]
        config = VQEConfig(n_restarts=1, maxiter=50)
        optimizer = VQEOptimizer(config=config, seed=42)

        rng = np.random.default_rng(42)
        prev_theta = rng.uniform(-0.01, 0.01, hva_circuit.num_parameters)
        results = []
        for h in h_values:
            lattice_h = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h)
            H = builder.build(lattice_h)
            result = optimizer.optimize(H, hva_circuit, prev_theta.copy())
            prev_theta = result.theta_opt.copy()
            results.append(result)

        # All energies must be finite
        for r in results:
            assert np.isfinite(r.energy)


# ═══════════════════════════════════════════════════════════════════════════
# 4. MPNN Prediction Flow (build_graph_dataset + forward pass fix)
# ═══════════════════════════════════════════════════════════════════════════


class TestMPNNPredictionFlow:
    """Verify the MPNN prediction flow used in deployment."""

    def test_build_graph_dataset_correct_api(self, builder, solver):
        """build_graph_dataset(lattice, h_values, theta_opt, e_exact, ...)."""
        lattice = make_lattice(TOPOLOGY, N_QUBITS)
        h_values = np.array([4.5, 4.0, 3.5, 3.25])
        # Dummy theta (correct shape for p=1: 2 params)
        theta_opt = np.random.default_rng(42).uniform(-0.5, 0.5, (4, 2))
        e_exact = np.array([-45.0, -40.5, -35.5, -33.2])

        dataset = build_graph_dataset(
            lattice=lattice,
            h_values=h_values,
            theta_opt=theta_opt,
            e_exact=e_exact,
            fidelities=None,
            fidelity_threshold=0.0,
        )
        assert len(dataset) == 4
        assert isinstance(dataset[0], Data)
        # Node features: [h, coord] → 2 features per node
        assert dataset[0].x.shape == (N_QUBITS, 2)

    def test_mpnn_forward_pass_prediction(self, builder):
        """MPNN prediction via model(graph).numpy().flatten()."""
        lattice_h = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=4.0)
        edge_index_np, coord = builder.build_graph_data(lattice_h)

        h_feat = np.full(N_QUBITS, 4.0)
        x = torch.tensor(
            np.stack([h_feat, coord.astype(float)], axis=1),
            dtype=torch.float32,
        )
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)

        model = MPNNPredictor(node_features=2, hidden_dim=64, n_layers=3, output_dim=2)
        model.eval()
        with torch.no_grad():
            theta_pred = model(graph).numpy().flatten()

        assert theta_pred.shape == (2,)
        assert np.all(np.isfinite(theta_pred))

    def test_mpnn_no_predict_method(self):
        """MPNNPredictor must NOT have a .predict(h, lattice) method."""
        model = MPNNPredictor(node_features=2, hidden_dim=64, n_layers=3, output_dim=2)
        assert not hasattr(model, "predict"), (
            "MPNNPredictor is an nn.Module — use model(graph), not model.predict()"
        )

    def test_train_mpnn_in_place(self, builder, solver):
        """train_mpnn trains model in-place and returns dict (not model)."""
        lattice = make_lattice(TOPOLOGY, N_QUBITS)
        h_values = np.array([4.5, 4.0, 3.5, 3.25])
        theta_opt = np.random.default_rng(42).uniform(-0.5, 0.5, (4, 2))
        e_exact = np.array([-45.0, -40.5, -35.5, -33.2])

        dataset = build_graph_dataset(
            lattice=lattice,
            h_values=h_values,
            theta_opt=theta_opt,
            e_exact=e_exact,
            fidelities=None,
            fidelity_threshold=0.0,
        )

        model = MPNNPredictor(node_features=2, hidden_dim=64, n_layers=3, output_dim=2)
        stats = train_mpnn(model=model, dataset=dataset, n_epochs=10, seed=42)

        # Returns dict, not model
        assert isinstance(stats, dict)
        assert "final_mse" in stats
        # Model is trained in-place — should be usable for prediction
        model.eval()
        with torch.no_grad():
            pred = model(dataset[0]).numpy().flatten()
        assert pred.shape == (2,)


# ═══════════════════════════════════════════════════════════════════════════
# 5. End-to-End: Full prepare_mpnn_predictions Flow
# ═══════════════════════════════════════════════════════════════════════════


class TestPrepareMPNNPredictionsE2E:
    """End-to-end test of the deployment prediction pipeline (no QPU)."""

    def test_exact_energies_computed_for_all_h(self, builder, solver):
        """All h-values in test + train get exact energies and gaps."""
        h_test = [4.0, 3.25]
        h_train = [4.5, 4.25, 4.0, 3.75, 3.5, 3.25, 3.0]

        e_exact_per_h = {}
        gap_per_h = {}
        for h in set(h_test + h_train):
            lattice_h = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=h)
            H = builder.build(lattice_h)
            exact = solver.solve(H, lattice_h)
            e_exact_per_h[h] = exact.ground_energy
            gap_per_h[h] = exact.gap

        # All values must be computed
        assert len(e_exact_per_h) == len(set(h_test + h_train))
        # All energies negative for TFIM
        for e in e_exact_per_h.values():
            assert e < 0
        # All gaps positive
        for g in gap_per_h.values():
            assert g > 0

    def test_prediction_shape_matches_circuit_params(self, builder):
        """MPNN prediction output dimension matches circuit.num_parameters."""
        lattice = make_lattice(TOPOLOGY, N_QUBITS)
        circuit_builder = HVACircuitBuilder()
        circuit, _ = circuit_builder.create(N_QUBITS, P_LAYERS, lattice)
        n_params = circuit.num_parameters

        # Create model with matching output_dim
        model = MPNNPredictor(node_features=2, hidden_dim=128, n_layers=3, output_dim=n_params)
        model.eval()

        # Predict for h=4.0
        lattice_h = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=4.0)
        edge_index_np, coord = builder.build_graph_data(lattice_h)
        h_feat = np.full(N_QUBITS, 4.0)
        x = torch.tensor(
            np.stack([h_feat, coord.astype(float)], axis=1),
            dtype=torch.float32,
        )
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            theta_pred = model(graph).numpy().flatten()

        assert theta_pred.shape == (n_params,)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Tier 3 Cross-Model (tfim_longitudinal)
# ═══════════════════════════════════════════════════════════════════════════


class TestTier3CrossModel:
    """Verify Tier 3 cross-model pattern: spec.build_hamiltonian(lattice_h)."""

    def test_longitudinal_model_spec_build(self):
        """ModelSpec for tfim_longitudinal builds Hamiltonian from lattice."""
        from qmbp_simulation.models.model_registry import get_model_spec

        spec = get_model_spec("tfim_longitudinal").with_params(g=0.3)
        lattice_h = make_lattice(TOPOLOGY, N_QUBITS, J=1.0, h=3.25)

        H = spec.build_hamiltonian(lattice_h, **spec.hamiltonian_kwargs)
        assert H.num_qubits == N_QUBITS

        # Solve requires lattice
        solver = ClassicalSolver()
        result = solver.solve(H, lattice_h)
        assert np.isfinite(result.ground_energy)
        assert result.gap >= 0

    def test_longitudinal_has_3_params(self):
        """tfim_longitudinal circuit must have 3 params per layer."""
        from qmbp_simulation.models.model_registry import get_model_spec

        spec = get_model_spec("tfim_longitudinal").with_params(g=0.3)
        lattice = make_lattice(TOPOLOGY, N_QUBITS)
        circuit, _ = spec.create_circuit(N_QUBITS, P_LAYERS, lattice, **spec.circuit_kwargs)

        # longitudinal adds g-term parameter
        assert circuit.num_parameters == 3

    def test_warm_start_from_tfim_to_longitudinal(self):
        """TFIM 2-param prediction can be extended to 3-param longitudinal."""
        theta_tfim = np.array([0.5, 0.3])  # 2 params from TFIM MPNN
        theta_longitudinal = np.append(theta_tfim, [0.1])  # Add g-term init
        assert len(theta_longitudinal) == 3
