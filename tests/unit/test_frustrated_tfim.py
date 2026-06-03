"""Unit tests for Frustrated TFIM (J1-J2) model.

Tests the full stack: Hamiltonian, circuit, model registry, and basic
physics for H = -J₁·ZZ_nn + J₂·ZZ_nnn - h·X.

Extracted from scripts/verify_frustrated_tfim.py and formalized.

Run with:
    pytest tests/unit/test_frustrated_tfim.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.models import HamiltonianBuilder, make_lattice
from qmbp_simulation.models.model_registry import get_model_spec, list_models

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def builder():
    return HamiltonianBuilder()


@pytest.fixture
def hva():
    return HVACircuitBuilder()


@pytest.fixture
def chain4():
    return make_lattice("chain_1d", 4, J=1.0, h=1.5)


@pytest.fixture
def chain6():
    return make_lattice("chain_1d", 6, J=1.0, h=1.5)


def _to_dense(matrix):
    """Convert to dense numpy array (handles sparse and ndarray)."""
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


# ═══════════════════════════════════════════════════════════════════════════
# Hamiltonian tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFrustratedTFIMHamiltonian:
    """Tests for HamiltonianBuilder.build_frustrated_tfim()."""

    def test_returns_sparse_pauli_op(self, builder, chain4):
        H = builder.build_frustrated_tfim(chain4, J2=0.3)
        assert isinstance(H, SparsePauliOp)

    def test_correct_qubit_count(self, builder, chain4):
        H = builder.build_frustrated_tfim(chain4, J2=0.3)
        assert H.num_qubits == 4

    def test_correct_dimension(self, builder, chain4):
        H = builder.build_frustrated_tfim(chain4, J2=0.3)
        mat = _to_dense(H.to_matrix())
        assert mat.shape == (16, 16)

    @pytest.mark.parametrize("J2", [0.0, 0.1, 0.3, 0.5, 1.0])
    def test_hermitian_for_all_j2(self, builder, chain6, J2):
        H = builder.build_frustrated_tfim(chain6, J2=J2)
        mat = _to_dense(H.to_matrix())
        np.testing.assert_allclose(
            mat,
            mat.conj().T,
            atol=1e-12,
            err_msg=f"Not Hermitian at J2={J2}",
        )

    def test_j2_zero_reduces_to_standard_tfim(self, builder, chain6):
        """At J2=0, must be identical to build() (standard TFIM)."""
        H_std = builder.build(chain6)
        H_frust = builder.build_frustrated_tfim(chain6, J2=0.0)
        assert H_frust.equiv(H_std), "J2=0 frustrated TFIM != standard TFIM"

    def test_spectrum_matches_manual_n4(self, builder):
        """Verify spectrum against manually constructed Hamiltonian (N=4)."""
        N, J1, J2, h = 4, 1.0, 0.3, 1.5
        lattice = make_lattice("chain_1d", N, J=J1, h=h)
        H = builder.build_frustrated_tfim(lattice, J2=J2)
        evals_api = np.sort(np.linalg.eigvalsh(_to_dense(H.to_matrix())))

        # Manual: NN edges (0,1),(1,2),(2,3); NNN edges (0,2),(1,3)
        terms = []
        for i, j in [(0, 1), (1, 2), (2, 3)]:
            terms.append(("ZZ", [i, j], -J1))
        for i, j in [(0, 2), (1, 3)]:
            terms.append(("ZZ", [i, j], J2))
        for i in range(N):
            terms.append(("X", [i], -h))
        H_manual = SparsePauliOp.from_sparse_list(terms, num_qubits=N)
        evals_manual = np.sort(np.linalg.eigvalsh(_to_dense(H_manual.to_matrix())))

        np.testing.assert_allclose(evals_api, evals_manual, atol=1e-12)

    def test_j2_changes_spectrum(self, builder, chain6):
        """J2>0 should change the energy spectrum relative to J2=0."""
        H0 = builder.build_frustrated_tfim(chain6, J2=0.0)
        H_frust = builder.build_frustrated_tfim(chain6, J2=0.5)
        evals_0 = np.sort(np.linalg.eigvalsh(_to_dense(H0.to_matrix())))
        evals_f = np.sort(np.linalg.eigvalsh(_to_dense(H_frust.to_matrix())))
        # Spectra should differ
        assert not np.allclose(evals_0, evals_f, atol=1e-8)

    def test_nnn_edges_computed_correctly(self, builder):
        """NNN edges for chain_1d N=6 should be (0,2),(1,3),(2,4),(3,5)."""
        lattice = make_lattice("chain_1d", 6, J=1.0, h=1.0)
        nnn = HamiltonianBuilder._generate_nnn_edges(lattice)
        expected = [(0, 2), (1, 3), (2, 4), (3, 5)]
        assert nnn == expected

    def test_nnn_edges_n4(self, builder):
        """NNN edges for chain N=4: (0,2),(1,3)."""
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)
        nnn = HamiltonianBuilder._generate_nnn_edges(lattice)
        expected = [(0, 2), (1, 3)]
        assert nnn == expected

    def test_ground_energy_at_h0_j2_zero(self, builder):
        """At h=0, J2=0: ground energy = -(N-1)*J (same as TFIM)."""
        lattice = make_lattice("chain_1d", 4, J=1.0, h=0.0)
        H = builder.build_frustrated_tfim(lattice, J2=0.0)
        evals = np.linalg.eigvalsh(_to_dense(H.to_matrix()))
        np.testing.assert_allclose(min(evals), -3.0, atol=1e-10)

    def test_frustration_raises_ground_energy(self, builder):
        """At h=0, adding J2>0 frustration should RAISE ground energy.

        Competing NN (ferro) and NNN (antiferro) interactions prevent
        the system from simultaneously satisfying all bonds.
        """
        lattice = make_lattice("chain_1d", 6, J=1.0, h=0.0)
        H0 = builder.build_frustrated_tfim(lattice, J2=0.0)
        H_frust = builder.build_frustrated_tfim(lattice, J2=0.5)
        e0 = min(np.linalg.eigvalsh(_to_dense(H0.to_matrix())))
        e_frust = min(np.linalg.eigvalsh(_to_dense(H_frust.to_matrix())))
        assert e_frust > e0, "Frustration should raise ground energy"


# ═══════════════════════════════════════════════════════════════════════════
# Circuit tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFrustratedTFIMCircuit:
    """Tests for HVACircuitBuilder.create_frustrated_tfim()."""

    def test_parameter_count_p1(self, hva, chain4):
        qc, theta = hva.create_frustrated_tfim(4, 1, chain4)
        assert len(theta) == 3  # 3 params/layer: θ_nn, θ_nnn, θ_x
        assert qc.num_parameters == 3

    def test_parameter_count_p2(self, hva, chain4):
        qc, theta = hva.create_frustrated_tfim(4, 2, chain4)
        assert len(theta) == 6  # 3 params/layer × 2
        assert qc.num_parameters == 6

    def test_correct_qubit_count(self, hva, chain6):
        qc, _ = hva.create_frustrated_tfim(6, 2, chain6)
        assert qc.num_qubits == 6

    def test_p_greater_than_2_raises(self, hva, chain4):
        with pytest.raises(ValueError):
            hva.create_frustrated_tfim(4, 3, chain4)

    def test_qubit_mismatch_raises(self, hva, chain4):
        with pytest.raises(ValueError):
            hva.create_frustrated_tfim(6, 1, chain4)  # lattice has 4, requesting 6

    def test_initial_state_is_plus(self, hva, chain4):
        """Circuit starts with H gates (|+⟩^N initial state)."""
        qc, theta = hva.create_frustrated_tfim(4, 1, chain4)
        from qiskit.quantum_info import Statevector

        bound = qc.assign_parameters(np.zeros(3))
        sv = Statevector(bound)
        # |+⟩^4 = (1/√16) * [1,1,1,...,1]
        expected = np.ones(16) / 4.0
        np.testing.assert_allclose(np.abs(sv.data), np.abs(expected), atol=1e-10)

    def test_rzz_gates_present(self, hva, chain6):
        """Circuit should contain RZZ gates for both NN and NNN bonds."""
        qc, _ = hva.create_frustrated_tfim(6, 1, chain6)
        ops = qc.count_ops()
        assert "rzz" in ops
        # N=6 chain: 5 NN + 4 NNN = 9 RZZ gates for p=1
        assert ops["rzz"] == 9

    def test_rx_gates_present(self, hva, chain6):
        """Circuit should have RX on all qubits per layer."""
        qc, _ = hva.create_frustrated_tfim(6, 1, chain6)
        ops = qc.count_ops()
        assert "rx" in ops
        assert ops["rx"] == 6  # one per qubit per layer

    @pytest.mark.parametrize("topology", ["chain_1d", "ladder"])
    def test_works_with_different_topologies(self, hva, topology):
        lattice = make_lattice(topology, 6, J=1.0, h=1.0)
        qc, theta = hva.create_frustrated_tfim(6, 1, lattice)
        assert qc.num_qubits == 6
        assert qc.num_parameters == 3


# ═══════════════════════════════════════════════════════════════════════════
# Model Registry tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFrustratedTFIMRegistry:
    """Tests for the tfim_frustrated model in the registry."""

    def test_registered_in_list(self):
        assert "tfim_frustrated" in list_models()

    def test_spec_params_per_layer(self):
        spec = get_model_spec("tfim_frustrated")
        assert spec.params_per_layer == 3

    def test_spec_initial_state(self):
        spec = get_model_spec("tfim_frustrated")
        assert spec.initial_state == "plus"

    def test_spec_hamiltonian_kwargs_default_j2(self):
        spec = get_model_spec("tfim_frustrated")
        assert spec.hamiltonian_kwargs["J2"] == 0.0

    def test_build_hamiltonian_callable(self):
        """Registry build_hamiltonian produces valid Hamiltonian."""
        spec = get_model_spec("tfim_frustrated")
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.5)
        H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
        assert isinstance(H, SparsePauliOp)
        assert H.num_qubits == 4

    def test_create_circuit_callable(self):
        """Registry create_circuit produces valid circuit."""
        spec = get_model_spec("tfim_frustrated")
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.5)
        qc, theta = spec.create_circuit(4, 2, lattice)
        assert qc.num_qubits == 4
        assert qc.num_parameters == 6  # 3/layer × 2


# ═══════════════════════════════════════════════════════════════════════════
# Physics / Frustration signature tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFrustrationPhysics:
    """Test that frustration introduces expected physical signatures."""

    def test_nn_correlator_sign(self, builder):
        """In ground state, NN ⟨ZZ⟩ > 0 (ferromagnetic tendency) at h=0."""
        lattice = make_lattice("chain_1d", 6, J=1.0, h=0.5)
        H = builder.build_frustrated_tfim(lattice, J2=0.3)
        mat = _to_dense(H.to_matrix())
        _, evecs = np.linalg.eigh(mat)
        gs = evecs[:, 0]

        # <ZZ> on bond (0,1)
        zz01 = SparsePauliOp.from_sparse_list([("ZZ", [0, 1], 1.0)], num_qubits=6)
        val = float(np.real(gs.conj() @ _to_dense(zz01.to_matrix()) @ gs))
        # At low h with J1>0, NN correlator should be positive (aligned spins)
        assert val > 0, f"Expected NN ⟨ZZ⟩ > 0, got {val}"

    def test_frustration_detectable_from_nnn(self, builder):
        """At J2>0 and h=0, NNN ⟨ZZ⟩ should differ from the unfrustrated case.

        Frustration competes with NN alignment. At h=0 with J2>0, the ground
        state compromises between NN alignment and NNN anti-alignment, so the
        NNN correlator should be LOWER than in the unfrustrated case (J2=0).
        """
        # Unfrustrated (J2=0) at h=0: all spins aligned → NNN ⟨ZZ⟩ = +1
        lattice = make_lattice("chain_1d", 6, J=1.0, h=0.01)
        H0 = builder.build_frustrated_tfim(lattice, J2=0.0)
        mat0 = _to_dense(H0.to_matrix())
        _, evecs0 = np.linalg.eigh(mat0)
        gs0 = evecs0[:, 0]

        zz02 = SparsePauliOp.from_sparse_list([("ZZ", [0, 2], 1.0)], num_qubits=6)
        val_unfrust = float(np.real(gs0.conj() @ _to_dense(zz02.to_matrix()) @ gs0))

        # Frustrated (J2=0.5) at same h: NNN correlator should decrease
        H_f = builder.build_frustrated_tfim(lattice, J2=0.5)
        mat_f = _to_dense(H_f.to_matrix())
        _, evecs_f = np.linalg.eigh(mat_f)
        gs_f = evecs_f[:, 0]
        val_frust = float(np.real(gs_f.conj() @ _to_dense(zz02.to_matrix()) @ gs_f))

        # Frustration should reduce NNN correlation relative to unfrustrated
        assert val_frust < val_unfrust, (
            f"Frustration should reduce NNN ⟨ZZ⟩: unfrust={val_unfrust:.4f}, frust={val_frust:.4f}"
        )

    def test_gap_stays_open_with_frustration(self, builder):
        """At moderate h and J2, the gap should remain finite (no gap closing)."""
        lattice = make_lattice("chain_1d", 6, J=1.0, h=1.5)
        H = builder.build_frustrated_tfim(lattice, J2=0.3)
        evals = np.sort(np.linalg.eigvalsh(_to_dense(H.to_matrix())))
        gap = evals[1] - evals[0]
        assert gap > 0.01, f"Gap unexpectedly small: {gap}"
