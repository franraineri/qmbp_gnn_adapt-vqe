"""Unit tests for TFIM + Longitudinal Field extension.

Tests the full stack: Hamiltonian, circuit, model registry, and VQE
correctness for H = -J·ZZ - h·X - g·Z.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity

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


# ─────────────────────────────────────────────────────────────────────────────
# Hamiltonian Builder Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTFIMLongitudinalHamiltonian:
    """Tests for HamiltonianBuilder.build_tfim_longitudinal()."""

    def test_returns_sparse_pauli_op(self, builder, chain4):
        H = builder.build_tfim_longitudinal(chain4, g=0.3)
        assert isinstance(H, SparsePauliOp)

    def test_correct_qubit_count(self, builder, chain4):
        H = builder.build_tfim_longitudinal(chain4, g=0.3)
        assert H.num_qubits == 4

    def test_correct_dimension(self, builder, chain4):
        H = builder.build_tfim_longitudinal(chain4, g=0.3)
        mat = _to_dense(H.to_matrix())
        assert mat.shape == (16, 16)

    @pytest.mark.parametrize("g", [0.0, 0.1, 0.3, 0.5, 1.0, -0.5])
    def test_hermitian_for_all_g(self, builder, chain4, g):
        H = builder.build_tfim_longitudinal(chain4, g=g)
        mat = _to_dense(H.to_matrix())
        np.testing.assert_allclose(
            mat,
            mat.conj().T,
            atol=1e-12,
            err_msg=f"Hamiltonian not Hermitian at g={g}",
        )

    def test_reduces_to_standard_tfim_at_g_zero(self, builder, chain6):
        """At g=0, must be identical to build() (standard TFIM)."""
        H_std = builder.build(chain6)
        H_long = builder.build_tfim_longitudinal(chain6, g=0.0)
        assert H_long.equiv(H_std)

    def test_spectrum_matches_manual_construction(self, builder):
        """Verify spectrum against a hand-built Hamiltonian."""
        N, J, h, g = 4, 1.0, 1.5, 0.3
        lattice = make_lattice("chain_1d", N, J=J, h=h)
        H = builder.build_tfim_longitudinal(lattice, g=g)
        evals_api = np.sort(np.linalg.eigvalsh(_to_dense(H.to_matrix())))

        # Manual: -J*ZZ(edges) - h*X(sites) - g*Z(sites)
        terms = []
        for i in range(N - 1):
            terms.append(("ZZ", [i, i + 1], -J))
        for i in range(N):
            terms.append(("X", [i], -h))
        for i in range(N):
            terms.append(("Z", [i], -g))
        H_manual = SparsePauliOp.from_sparse_list(terms, num_qubits=N)
        evals_manual = np.sort(np.linalg.eigvalsh(_to_dense(H_manual.to_matrix())))

        np.testing.assert_allclose(evals_api, evals_manual, atol=1e-12)

    def test_known_energy_at_h_zero_g_zero(self, builder):
        """At h=0, g=0: ground energy = -(N-1)*J for ferromagnetic chain."""
        lattice = make_lattice("chain_1d", 4, J=1.0, h=0.0)
        H = builder.build_tfim_longitudinal(lattice, g=0.0)
        evals = np.linalg.eigvalsh(_to_dense(H.to_matrix()))
        np.testing.assert_allclose(evals[0], -3.0, atol=1e-10)

    def test_known_energy_at_h_zero_g_positive(self, builder):
        """At h=0, g>0: ground energy = -(N-1)*J - N*g (all spins aligned with g)."""
        N, g = 4, 0.5
        lattice = make_lattice("chain_1d", N, J=1.0, h=0.0)
        H = builder.build_tfim_longitudinal(lattice, g=g)
        evals = np.linalg.eigvalsh(_to_dense(H.to_matrix()))
        # All spins in |1⟩ (Z eigenvalue = -1): E = -(N-1)*J - N*(-g)*(-1) = -(N-1) - N*g
        # Wait: H = -J*ZZ - g*Z. For |111...1⟩: ZZ = +1 for all pairs, Z_i = -1
        # E = -J*(N-1)*(+1) - g*N*(-1) = -(N-1) + N*g... That's higher.
        # For |000...0⟩: ZZ = +1, Z_i = +1 → E = -(N-1) - g*N = -(N-1)-N*g
        expected = -(N - 1) - N * g
        np.testing.assert_allclose(evals[0], expected, atol=1e-10)

    def test_z2_symmetry_preserved_at_g_zero(self, builder):
        """At g=0, <Z_i>=0 in ground state (Z2 symmetry unbroken)."""
        lattice = make_lattice("chain_1d", 6, J=1.0, h=1.5)
        H = builder.build_tfim_longitudinal(lattice, g=0.0)
        mat = _to_dense(H.to_matrix())
        _, evecs = np.linalg.eigh(mat)
        gs = evecs[:, 0]

        Z0 = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=6)
        mag_z = float(np.real(gs.conj() @ _to_dense(Z0.to_matrix()) @ gs))
        np.testing.assert_allclose(mag_z, 0.0, atol=1e-10)

    def test_z2_symmetry_broken_at_g_positive(self, builder):
        """At g>0, <Z_i> != 0 (longitudinal field breaks Z2 symmetry)."""
        lattice = make_lattice("chain_1d", 6, J=1.0, h=1.5)
        H = builder.build_tfim_longitudinal(lattice, g=0.5)
        mat = _to_dense(H.to_matrix())
        _, evecs = np.linalg.eigh(mat)
        gs = evecs[:, 0]

        Z0 = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=6)
        mag_z = float(np.real(gs.conj() @ _to_dense(Z0.to_matrix()) @ gs))
        assert abs(mag_z) > 0.01, f"Expected Z2 broken: <Z>={mag_z}"

    @pytest.mark.parametrize("topology", ["chain_1d", "ladder", "triangular"])
    def test_works_with_multiple_topologies(self, builder, topology):
        """Hamiltonian builds successfully for different topologies."""
        n = 6
        lattice = make_lattice(topology, n, J=1.0, h=1.5)
        H = builder.build_tfim_longitudinal(lattice, g=0.3)
        assert H.num_qubits == n
        mat = _to_dense(H.to_matrix())
        np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)

    def test_per_bond_J_array(self, builder):
        """Works with non-uniform coupling J as array."""
        # Override J with per-bond array
        J_arr = np.array([1.0, 0.5, 1.5])
        lattice_custom = make_lattice("chain_1d", 4, J=J_arr, h=1.5)
        H = builder.build_tfim_longitudinal(lattice_custom, g=0.2)
        assert H.num_qubits == 4
        mat = _to_dense(H.to_matrix())
        np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Observables Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTFIMLongitudinalObservables:
    """Tests for build_tfim_longitudinal_observables()."""

    def test_returns_three_observable_lists(self, builder, chain4):
        ops_x, ops_z, ops_zz = builder.build_tfim_longitudinal_observables(chain4)
        assert len(ops_x) == 4  # per-site X
        assert len(ops_z) == 4  # per-site Z
        assert len(ops_zz) == 3  # per-bond ZZ (N-1 bonds)

    def test_observables_are_sparse_pauli_ops(self, builder, chain4):
        ops_x, ops_z, ops_zz = builder.build_tfim_longitudinal_observables(chain4)
        for op in ops_x + ops_z + ops_zz:
            assert isinstance(op, SparsePauliOp)
            assert op.num_qubits == 4

    def test_observables_are_hermitian(self, builder, chain4):
        ops_x, ops_z, ops_zz = builder.build_tfim_longitudinal_observables(chain4)
        for op in ops_x + ops_z + ops_zz:
            mat = _to_dense(op.to_matrix())
            np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Circuit Builder Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTFIMLongitudinalCircuit:
    """Tests for HVACircuitBuilder.create_tfim_longitudinal()."""

    def test_parameter_count_p1(self, hva, chain4):
        qc, theta = hva.create_tfim_longitudinal(4, 1, chain4)
        assert len(theta) == 3  # 3 params/layer
        assert qc.num_parameters == 3

    def test_parameter_count_p2(self, hva, chain4):
        qc, theta = hva.create_tfim_longitudinal(4, 2, chain4)
        assert len(theta) == 6  # 3 params/layer × 2
        assert qc.num_parameters == 6

    def test_correct_qubit_count(self, hva, chain6):
        qc, _ = hva.create_tfim_longitudinal(6, 2, chain6)
        assert qc.num_qubits == 6

    def test_p_greater_than_2_raises(self, hva, chain4):
        with pytest.raises(ValueError, match="exceeds"):
            hva.create_tfim_longitudinal(4, 3, chain4)

    def test_qubit_mismatch_raises(self, hva, chain4):
        with pytest.raises(ValueError, match="does not match"):
            hva.create_tfim_longitudinal(6, 1, chain4)

    def test_initial_state_is_plus(self, hva, chain4):
        """Circuit starts with H gates on all qubits (|+⟩^N)."""
        qc, theta = hva.create_tfim_longitudinal(4, 1, chain4)
        # Assign zeros → only H gates active → |+⟩^N state
        bound = qc.assign_parameters(np.zeros(3))
        sv = Statevector(bound)
        # |+⟩^N = (1/sqrt(2^N)) * sum_i |i⟩
        expected_amp = 1.0 / np.sqrt(2**4)
        np.testing.assert_allclose(
            np.abs(sv.data),
            expected_amp,
            atol=1e-10,
        )

    def test_reduces_to_standard_hva_when_theta_z_zero(self, hva, chain4):
        """When θ_z=0 for all layers, the state matches standard TFIM HVA."""
        qc_ext, _ = hva.create_tfim_longitudinal(4, 1, chain4)
        qc_std, _ = hva.create(4, 1, chain4)

        # Extended: params = [θ_zz, θ_x, θ_z=0]
        # Standard: params = [θ_zz, θ_x]
        theta_zz, theta_x = 0.5, -0.3
        sv_ext = Statevector(qc_ext.assign_parameters([theta_zz, theta_x, 0.0]))
        sv_std = Statevector(qc_std.assign_parameters([theta_zz, theta_x]))

        fid = float(state_fidelity(sv_ext, sv_std))
        np.testing.assert_allclose(fid, 1.0, atol=1e-10)

    def test_theta_z_nonzero_produces_different_state(self, hva, chain4):
        """Non-zero θ_z produces a different state from standard HVA."""
        qc_ext, _ = hva.create_tfim_longitudinal(4, 1, chain4)
        qc_std, _ = hva.create(4, 1, chain4)

        theta_zz, theta_x, theta_z = 0.5, -0.3, 0.4
        sv_ext = Statevector(qc_ext.assign_parameters([theta_zz, theta_x, theta_z]))
        sv_std = Statevector(qc_std.assign_parameters([theta_zz, theta_x]))

        fid = float(state_fidelity(sv_ext, sv_std))
        assert fid < 0.99, f"θ_z≠0 should give different state, got fid={fid}"

    @pytest.mark.parametrize("topology", ["chain_1d", "ladder"])
    def test_works_with_different_topologies(self, hva, topology):
        lattice = make_lattice(topology, 6, J=1.0, h=1.0)
        qc, theta = hva.create_tfim_longitudinal(6, 2, lattice)
        assert qc.num_qubits == 6
        assert qc.num_parameters == 6


# ─────────────────────────────────────────────────────────────────────────────
# Model Registry Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTFIMLongitudinalRegistry:
    """Tests for the tfim_longitudinal model in the registry."""

    def test_registered_in_list(self):
        assert "tfim_longitudinal" in list_models()

    def test_spec_params_per_layer(self):
        spec = get_model_spec("tfim_longitudinal")
        assert spec.params_per_layer == 3

    def test_spec_initial_state(self):
        spec = get_model_spec("tfim_longitudinal")
        assert spec.initial_state == "plus"

    def test_spec_hamiltonian_kwargs_default_g(self):
        spec = get_model_spec("tfim_longitudinal")
        assert spec.hamiltonian_kwargs["g"] == 0.0

    def test_spec_fidelity_threshold(self):
        spec = get_model_spec("tfim_longitudinal")
        assert spec.fidelity_threshold == 0.90

    def test_with_g_creates_new_spec(self):
        spec = get_model_spec("tfim_longitudinal")
        new_spec = spec.with_g(0.5)
        assert new_spec.hamiltonian_kwargs["g"] == 0.5
        # Original unchanged
        assert spec.hamiltonian_kwargs["g"] == 0.0

    def test_with_g_preserves_other_fields(self):
        spec = get_model_spec("tfim_longitudinal")
        new_spec = spec.with_g(0.3)
        assert new_spec.name == spec.name
        assert new_spec.params_per_layer == spec.params_per_layer
        assert new_spec.initial_state == spec.initial_state
        assert new_spec.fidelity_threshold == spec.fidelity_threshold

    def test_build_hamiltonian_callable(self):
        """Registry's build_hamiltonian produces valid Hamiltonian."""
        spec = get_model_spec("tfim_longitudinal")
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.5)
        H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
        assert isinstance(H, SparsePauliOp)
        assert H.num_qubits == 4

    def test_create_circuit_callable(self):
        """Registry's create_circuit produces valid circuit."""
        spec = get_model_spec("tfim_longitudinal")
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.5)
        qc, theta = spec.create_circuit(4, 2, lattice)
        assert qc.num_parameters == 6
        assert qc.num_qubits == 4


# ─────────────────────────────────────────────────────────────────────────────
# VQE Expressibility Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTFIMLongitudinalExpressibility:
    """Tests that the extended HVA can reach ground states of H(g>0)."""

    def _get_ground_state(self, builder, lattice, g):
        """Helper: exact diag for ground state."""
        H = builder.build_tfim_longitudinal(lattice, g=g)
        mat = _to_dense(H.to_matrix())
        evals, evecs = np.linalg.eigh(mat)
        return H, evals[0], evecs[:, 0]

    def _vqe_fidelity(self, hva, builder, lattice, g, n_restarts=10):
        """Helper: run VQE and return best fidelity."""
        from qiskit.primitives import StatevectorEstimator
        from scipy.optimize import minimize

        N = lattice.n_qubits
        H, e_exact, gs = self._get_ground_state(builder, lattice, g)
        qc, _ = hva.create_tfim_longitudinal(N, 2, lattice)
        estimator = StatevectorEstimator()

        def cost(params):
            bound = qc.assign_parameters(params)
            return float(estimator.run([(bound, H)]).result()[0].data.evs)

        best_fid = 0.0
        rng = np.random.default_rng(42)
        for _ in range(n_restarts):
            x0 = rng.uniform(-np.pi, np.pi, qc.num_parameters)
            result = minimize(
                cost,
                x0,
                method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * qc.num_parameters,
                options={"maxiter": 500, "ftol": 1e-14},
            )
            sv = Statevector(qc.assign_parameters(result.x))
            fid = float(state_fidelity(sv, Statevector(gs)))
            best_fid = max(best_fid, fid)

        return best_fid

    def test_expressibility_g03_h20(self, hva, builder):
        """Extended HVA reaches fid≥0.99 at h=2.0, g=0.3 (easy regime)."""
        lattice = make_lattice("chain_1d", 4, J=1.0, h=2.0)
        fid = self._vqe_fidelity(hva, builder, lattice, g=0.3)
        assert fid >= 0.99, f"Expected fid≥0.99, got {fid:.4f}"

    def test_expressibility_g05_h15(self, hva, builder):
        """Extended HVA reaches fid≥0.98 at h=1.5, g=0.5 (harder regime)."""
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.5)
        fid = self._vqe_fidelity(hva, builder, lattice, g=0.5)
        assert fid >= 0.98, f"Expected fid≥0.98, got {fid:.4f}"

    def test_expressibility_g05_h10(self, hva, builder):
        """Extended HVA reaches fid≥0.95 at h=1.0, g=0.5 (near critical)."""
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)
        fid = self._vqe_fidelity(hva, builder, lattice, g=0.5, n_restarts=20)
        assert fid >= 0.95, f"Expected fid≥0.95, got {fid:.4f}"

    def test_standard_hva_fails_at_g05(self, builder):
        """Standard HVA (no RZ) has fid < 0.90 at g=0.5 (confirms E4 finding)."""
        from qiskit.primitives import StatevectorEstimator
        from scipy.optimize import minimize

        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.5)
        H = builder.build_tfim_longitudinal(lattice, g=0.5)
        mat = _to_dense(H.to_matrix())
        _, evecs = np.linalg.eigh(mat)
        gs = evecs[:, 0]

        hva_std = HVACircuitBuilder()
        qc_std, _ = hva_std.create(4, 2, lattice)
        estimator = StatevectorEstimator()

        def cost(params):
            bound = qc_std.assign_parameters(params)
            return float(estimator.run([(bound, H)]).result()[0].data.evs)

        best_fid = 0.0
        rng = np.random.default_rng(42)
        for _ in range(20):
            x0 = rng.uniform(-np.pi, np.pi, qc_std.num_parameters)
            result = minimize(
                cost,
                x0,
                method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * qc_std.num_parameters,
                options={"maxiter": 500, "ftol": 1e-14},
            )
            sv = Statevector(qc_std.assign_parameters(result.x))
            fid = float(state_fidelity(sv, Statevector(gs)))
            best_fid = max(best_fid, fid)

        # Standard HVA should NOT be expressive enough
        assert best_fid < 0.90, f"Standard HVA should fail at g=0.5 but got fid={best_fid:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Hardware Viability Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTFIMLongitudinalHardwareViability:
    """Tests that the extended HVA maintains hardware viability."""

    def test_no_additional_2q_gates_vs_standard(self, hva):
        """RZ layer adds NO 2-qubit gates (hardware overhead unchanged)."""
        lattice = make_lattice("chain_1d", 6, J=1.0, h=1.0)

        qc_ext, _ = hva.create_tfim_longitudinal(6, 1, lattice)
        qc_std, _ = hva.create(6, 1, lattice)

        # Count gate types in undecomposed circuits
        # RZZ is the only 2-qubit gate in both circuits
        ext_ops = qc_ext.count_ops()
        std_ops = qc_std.count_ops()

        # Both should have the same number of RZZ gates (= num_edges)
        assert ext_ops.get("rzz", 0) == std_ops.get("rzz", 0)

        # Extended has additional RZ gates (single-qubit, free on hardware)
        assert ext_ops.get("rz", 0) == 6  # One RZ per qubit

    def test_p1_circuit_depth_reasonable(self, hva):
        """p=1 extended HVA depth is shallow enough for NISQ."""
        lattice = make_lattice("chain_1d", 10, J=1.0, h=1.0)
        qc, _ = hva.create_tfim_longitudinal(10, 1, lattice)
        # Depth should be moderate (H + RZZ layer + RX layer + RZ layer)
        assert qc.depth() < 50
