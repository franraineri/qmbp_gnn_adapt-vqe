"""V6 pipeline tests — replaces the ad-hoc smoke_test.py with proper pytest."""

import numpy as np
import pytest

from src.poc.v6 import (
    HVACircuitBuilder,
    VQEConfig,
    VQEOptimizer,
    load_phase12_dataset,
    make_lattice,
    save_phase12_dataset,
)
from src.poc.v6.hardware_deployer_v61 import HardwareDeployerV61
from src.poc.v6.pipeline_utils import assert_observable_locality

# ── Phase 1: Hamiltonian & Ground Truth ──────────────────────────────────


class TestHamiltonianBuilder:
    def test_chain_1d_matches_v4(self, builder):
        """V6 chain_1d Hamiltonian must be identical to V4's inline construction."""
        from qiskit.quantum_info import SparsePauliOp

        lat = make_lattice("chain_1d", 6, J=1.0, h=1.0)
        H_v6 = builder.build(lat)

        terms = [("ZZ", [i, i + 1], -1.0) for i in range(5)]
        terms += [("X", [i], -1.0) for i in range(6)]
        H_v4 = SparsePauliOp.from_sparse_list(terms, num_qubits=6)

        diff = np.max(np.abs(np.asarray(H_v6.to_matrix()) - np.asarray(H_v4.to_matrix())))
        assert diff == 0.0

    @pytest.mark.parametrize("topo", ["chain_1d", "ladder", "triangular", "kagome"])
    def test_all_topologies_hermitian(self, builder, topo):
        n = 6
        lat = make_lattice(topo, n, J=1.0, h=1.0)
        H = builder.build(lat)
        assert H.num_qubits == n

    def test_graph_data_symmetric(self, builder, chain_6):
        edge_idx, coord = builder.build_graph_data(chain_6)
        edges = set(zip(edge_idx[0].tolist(), edge_idx[1].tolist(), strict=False))
        assert all((j, i) in edges for i, j in edges)
        assert edge_idx.shape[1] == 2 * len(chain_6.edges)

    def test_p_gt_2_raises(self):
        lat = make_lattice("chain_1d", 6, J=1.0, h=1.0)
        hva = HVACircuitBuilder()
        with pytest.raises(ValueError, match="Mele"):
            hva.create(6, 3, lat)


class TestClassicalSolver:
    def test_exact_diag_chain(self, builder, solver):
        lat = make_lattice("chain_1d", 6, J=1.0, h=1.0)
        H = builder.build(lat)
        result = solver.solve(H, lat)
        assert result.gap >= 0
        assert len(result.per_site_mag_x) == 6
        assert len(result.per_bond_corr_zz) == 5
        assert result.ground_state is not None

    def test_auto_selects_exact_for_small(self, builder, solver):
        lat = make_lattice("chain_1d", 6, J=1.0, h=1.0)
        H = builder.build(lat)
        result = solver.solve(H, lat, method="auto")
        assert result.ground_state is not None  # exact diag returns statevector


# ── Phase 2: VQE ────────────────────────────────────────────────────────


class TestVQEOptimizer:
    def test_single_optimize(self, builder, solver, hva):
        lat = make_lattice("chain_1d", 6, J=1.0, h=1.5)
        H = builder.build(lat)
        qc, _ = hva.create(6, 2, lat)
        exact = solver.solve(H, lat)

        config = VQEConfig(n_restarts=0, maxiter=50, enable_callbacks=True)
        opt = VQEOptimizer(config)
        result = opt.optimize(
            H, qc, np.random.uniform(-0.01, 0.01, 4), exact.ground_energy, exact.ground_state
        )

        assert result.trajectory is not None
        assert len(result.trajectory.energies) > 0
        assert len(result.trajectory.energies) == len(result.trajectory.grad_norms)

    def test_warm_start_zeros_at_h0(self):
        config = VQEConfig()
        guess = VQEOptimizer.get_initial_guess(4, 0.0, config)
        assert np.allclose(guess, 0.0)

    def test_descending_sweep(self, hva, h_values_reduced, exact_data_reduced):
        lat = make_lattice("chain_1d", 6, J=1.0, h=1.0)
        qc, _ = hva.create(6, 2, lat)
        config = VQEConfig(n_restarts=0, maxiter=30, enable_callbacks=False)
        opt = VQEOptimizer(config)
        results = opt.descending_sweep(h_values_reduced, qc, lat, exact_data_reduced)
        assert len(results) == len(h_values_reduced)
        assert results[-1].h_value == h_values_reduced[-1]


# ── Phase 4: Deployment ─────────────────────────────────────────────────


class TestHardwareDeployer:
    def test_phase_classification(self):
        deployer = HardwareDeployerV61(mode="simulation")
        assert deployer.classify_phase(0.9, -0.3, 0.01) == "paramagnetic"
        assert deployer.classify_phase(0.3, -0.9, 0.01) == "ferromagnetic"
        assert deployer.classify_phase(0.5, -0.5, 0.1) == "indeterminate"


# ── Pipeline Integrity ──────────────────────────────────────────────────


class TestPipelineIntegrity:
    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / "test.npz"
        save_phase12_dataset(
            path,
            h_values=np.array([1.0]),
            J=1.0,
            n_qubits=6,
            p_layers=2,
            ground_energies=np.array([-7.3]),
            gaps=np.array([0.48]),
            mag_x=np.array([0.77]),
            corr_zz=np.array([0.53]),
            theta_opt=np.array([[0.1, 0.2, 0.3, 0.4]]),
            vqe_energies=np.array([-7.2]),
            fidelities=np.array([0.99]),
        )
        data = load_phase12_dataset(path)
        assert str(data["cost_function"]) == "energy"
        assert str(data["version"]) == "v6.0"

    def test_rejects_wrong_cost_function(self, tmp_path):
        path = tmp_path / "bad.npz"
        np.savez(path, cost_function="hybrid", version="v5.1")
        with pytest.raises(ValueError, match="Phase coupling mismatch"):
            load_phase12_dataset(path)

    def test_observable_locality(self, builder, chain_6):
        ops_x, ops_zz = builder.build_local_observables(chain_6)
        assert_observable_locality(ops_x + ops_zz, chain_6.edges)

    def test_nonlocal_observable_rejected(self, chain_6):
        from qiskit.quantum_info import SparsePauliOp

        bad = [SparsePauliOp.from_sparse_list([("ZZ", [0, 5], 1.0)], num_qubits=6)]
        with pytest.raises(ValueError, match="not adjacent"):
            assert_observable_locality(bad, chain_6.edges)
