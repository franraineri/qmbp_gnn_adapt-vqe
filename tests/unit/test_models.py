"""Unit tests for qmbp_simulation.models module."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from qmbp_simulation.models import (
    SUPPORTED_TOPOLOGIES,
    HamiltonianBuilder,
    LatticeConfig,
    make_lattice,
)


class TestMakeLattice:
    """Test make_lattice for all supported topologies."""

    @pytest.mark.parametrize("topology", SUPPORTED_TOPOLOGIES)
    def test_creates_valid_lattice(self, topology):
        # kagome and triangular need specific qubit counts
        n = 6 if topology in ("kagome", "triangular", "ladder") else 4
        lattice = make_lattice(topology, n, J=1.0, h=1.5)
        assert lattice.topology == topology
        assert lattice.n_qubits == n
        assert len(lattice.edges) > 0
        assert lattice.coordination_numbers.shape == (n,)

    def test_chain_1d_edges(self):
        lattice = make_lattice("chain_1d", 4)
        # Open chain: 3 edges for 4 qubits
        assert len(lattice.edges) == 3

    def test_periodic_chain_has_extra_edge(self):
        lattice = make_lattice("chain_1d", 4, periodic=True)
        # Periodic chain: 4 edges for 4 qubits
        assert len(lattice.edges) == 4

    def test_ladder_topology_edges(self):
        lattice = make_lattice("ladder", 6)
        # Ladder with 6 qubits: 2 legs of 3 + 3 rungs = 7 edges
        assert len(lattice.edges) == 7
        assert lattice.n_qubits == 6

    def test_kagome_topology_coordination(self):
        lattice = make_lattice("kagome", 6)
        # Kagome lattice has coordination number 4
        assert lattice.n_qubits == 6
        assert len(lattice.edges) > 0

    def test_make_lattice_with_custom_J_and_h(self):
        lattice = make_lattice("chain_1d", 4, J=2.0, h=0.5)
        assert lattice.J == 2.0
        assert lattice.h == 0.5

    def test_invalid_topology_raises(self):
        with pytest.raises(ValueError, match="Unknown topology"):
            make_lattice("hexagonal", 6)


class TestHamiltonianBuilder:
    """Test HamiltonianBuilder.build() produces valid SparsePauliOp."""

    def test_build_returns_sparse_pauli_op(self, small_lattice):
        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        assert isinstance(H, SparsePauliOp)

    def test_hamiltonian_correct_qubit_count(self, small_lattice):
        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        assert H.num_qubits == small_lattice.n_qubits

    def test_hamiltonian_is_hermitian(self, small_lattice):
        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        mat = H.to_matrix()
        np.testing.assert_allclose(
            mat, mat.conj().T, atol=1e-12, err_msg="Hamiltonian must be Hermitian"
        )

    def test_validate_rejects_wrong_qubit_count(self, small_lattice):
        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        with pytest.raises(ValueError):
            builder.validate(H, n_qubits=small_lattice.n_qubits + 1)

    def test_hamiltonian_dimension_is_2_to_n(self, small_lattice):
        builder = HamiltonianBuilder()
        H = builder.build(small_lattice)
        mat = H.to_matrix()
        expected_dim = 2**small_lattice.n_qubits
        assert mat.shape == (expected_dim, expected_dim)

    def test_build_graph_data_returns_valid_structure(self, small_lattice):
        builder = HamiltonianBuilder()
        edge_index, edge_attr = builder.build_graph_data(small_lattice)
        # edge_index should be 2 x num_edges
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0
        # All indices should be valid qubit indices
        assert np.all(edge_index >= 0)
        assert np.all(edge_index < small_lattice.n_qubits)

    def test_build_local_observables(self, small_lattice):
        builder = HamiltonianBuilder()
        ops_x, ops_zz = builder.build_local_observables(small_lattice)
        # Should return per-site X and per-bond ZZ operators
        assert len(ops_x) == small_lattice.n_qubits
        assert len(ops_zz) == len(small_lattice.edges)
        # Each observable should be a SparsePauliOp
        for op in ops_x:
            assert isinstance(op, SparsePauliOp)
            assert op.num_qubits == small_lattice.n_qubits
        for op in ops_zz:
            assert isinstance(op, SparsePauliOp)
            assert op.num_qubits == small_lattice.n_qubits

    def test_hamiltonian_known_energy_at_h0(self):
        """At h=0, ground energy of N-qubit chain is -(N-1)*J."""
        builder = HamiltonianBuilder()
        lattice = make_lattice("chain_1d", 4, J=1.0, h=0.0)
        H = builder.build(lattice)
        mat = H.to_matrix()
        eigenvalues = np.linalg.eigvalsh(mat)
        # Ground energy for h=0 TFIM chain: -(N-1)*J = -3.0
        np.testing.assert_allclose(eigenvalues[0], -3.0, atol=1e-10)


class TestDataModelValidation:
    """Test dataclass validation in data_models.py."""

    def test_lattice_config_rejects_n_qubits_less_than_2(self):
        with pytest.raises(ValueError, match="n_qubits must be ≥ 2"):
            LatticeConfig(
                topology="chain_1d",
                n_qubits=1,
                J=1.0,
                h=1.0,
                edges=[(0, 1)],
                coordination_numbers=np.array([1]),
            )

    def test_lattice_config_rejects_unsupported_topology(self):
        with pytest.raises(ValueError, match="Unsupported topology"):
            LatticeConfig(
                topology="honeycomb",
                n_qubits=4,
                J=1.0,
                h=1.0,
                edges=[(0, 1), (1, 2), (2, 3)],
                coordination_numbers=np.array([1, 2, 2, 1]),
            )

    def test_vqe_config_rejects_p_layers_above_max(self):
        from qmbp_simulation.models import VQEConfig
        from qmbp_simulation.models.constants import MAX_P_LAYERS

        with pytest.raises(ValueError, match="p_layers must be ≤"):
            VQEConfig(p_layers=MAX_P_LAYERS + 1)

    def test_vqe_config_rejects_ascending_sweep(self):
        from qmbp_simulation.models import VQEConfig

        with pytest.raises(ValueError, match="sweep_direction must be"):
            VQEConfig(sweep_direction="ascending")

    def test_ground_truth_result_rejects_negative_gap(self):
        from qmbp_simulation.models import GroundTruthResult

        with pytest.raises(ValueError, match="gap must be ≥ 0"):
            GroundTruthResult(
                h_value=1.0,
                ground_energy=-5.0,
                gap=-0.1,
                ground_state=None,
                mag_x=0.5,
                corr_zz=0.3,
                per_site_mag_x=np.array([0.5, 0.5]),
                per_bond_corr_zz=np.array([0.3]),
            )
