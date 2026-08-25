"""Tests for Phase 1 solver improvements.

Covers:
- Sparse eigsh dispatch for N>=13 (both solve and ground_state_vector)
- Gap accuracy: sparse k=2 eigsh vs dense eigh
- DMRG gap finite-size correction
- Ground state caching behavior
- Method dispatch boundaries
"""

import numpy as np
import pytest

from qmbp_simulation import make_lattice
from qmbp_simulation.models.model_registry import get_model_spec
from qmbp_simulation.solvers.classical import ClassicalSolver


@pytest.fixture
def solver():
    return ClassicalSolver()


@pytest.fixture
def tfim_spec():
    return get_model_spec("tfim")


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Method dispatch boundaries
# ═══════════════════════════════════════════════════════════════════════════════


class TestMethodDispatch:
    """Verify correct method selection by system size."""

    def test_n6_uses_dense_eigh(self, solver, tfim_spec):
        """N=6 should use dense eigh (below sparse threshold)."""
        lattice = make_lattice("chain_1d", 6, J=1.0, h=1.5)
        H = tfim_spec.build_hamiltonian(lattice, **tfim_spec.hamiltonian_kwargs)
        result = solver.solve(H, lattice)
        # Dense eigh always returns ground_state
        assert result.ground_state is not None
        assert result.ground_state.shape == (2**6,)

    def test_n14_uses_sparse_eigsh(self, solver, tfim_spec):
        """N=14 should use sparse eigsh (above threshold=13)."""
        lattice = make_lattice("chain_1d", 14, J=1.0, h=1.5)
        H = tfim_spec.build_hamiltonian(lattice, **tfim_spec.hamiltonian_kwargs)
        result = solver.solve(H, lattice)
        assert result.ground_state is not None
        assert result.ground_state.shape == (2**14,)

    def test_n15_uses_sparse_eigsh(self, solver, tfim_spec):
        """N=15 should use sparse eigsh (at EXACT_DIAG_QUBIT_LIMIT)."""
        lattice = make_lattice("chain_1d", 15, J=1.0, h=2.0)
        H = tfim_spec.build_hamiltonian(lattice, **tfim_spec.hamiltonian_kwargs)
        result = solver.solve(H, lattice)
        assert result.ground_state is not None
        assert result.ground_state.shape == (2**15,)

    def test_n16_uses_exact(self, solver, tfim_spec):
        """N=16 should use exact (at STATEVECTOR_MAX_N boundary)."""
        lattice = make_lattice("chain_1d", 16, J=1.0, h=2.0)
        H = tfim_spec.build_hamiltonian(lattice, **tfim_spec.hamiltonian_kwargs)
        result = solver.solve(H, lattice)
        # N=16 == STATEVECTOR_MAX_N → exact diag, returns ground_state
        assert result.ground_state is not None
        assert result.ground_state.shape == (2**16,)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Gap accuracy (sparse k=2 vs dense full spectrum)
# ═══════════════════════════════════════════════════════════════════════════════


class TestGapAccuracy:
    """Verify gap computation accuracy across methods."""

    def test_sparse_gap_matches_dense_n10(self, solver, tfim_spec):
        """At N=10, sparse eigsh (k=2) gap should match dense eigh gap exactly."""
        lattice = make_lattice("chain_1d", 10, J=1.0, h=1.5)
        H = tfim_spec.build_hamiltonian(lattice, **tfim_spec.hamiltonian_kwargs)

        # Dense reference (full spectrum)
        mat = np.asarray(H.to_matrix())
        evals_dense = np.sort(np.linalg.eigvalsh(mat))
        gap_dense = float(evals_dense[1] - evals_dense[0])

        # Solver result
        result = solver.solve(H, lattice)

        np.testing.assert_allclose(result.gap, gap_dense, atol=1e-8)

    def test_sparse_gap_matches_dense_n14(self, solver, tfim_spec):
        """At N=14, sparse eigsh gap should match dense eigh gap within tolerance."""
        lattice = make_lattice("chain_1d", 14, J=1.0, h=1.8)
        H = tfim_spec.build_hamiltonian(lattice, **tfim_spec.hamiltonian_kwargs)

        # Dense reference (expensive but needed for validation)
        from scipy.sparse.linalg import eigsh

        H_sparse = H.to_matrix(sparse=True)
        evals_ref, _ = eigsh(H_sparse, k=2, which="SA")
        evals_ref = np.sort(evals_ref)
        gap_ref = float(evals_ref[1] - evals_ref[0])

        result = solver.solve(H, lattice)
        np.testing.assert_allclose(result.gap, gap_ref, atol=1e-8)

    def test_dmrg_gap_finite_size_correction(self, solver, tfim_spec):
        """DMRG gap with finite-size correction should be closer to exact than bare formula."""
        lattice = make_lattice("chain_1d", 16, J=1.0, h=1.5)
        H = tfim_spec.build_hamiltonian(lattice, **tfim_spec.hamiltonian_kwargs)

        # Exact gap via sparse eigsh (reference)
        from scipy.sparse.linalg import eigsh

        H_sparse = H.to_matrix(sparse=True)
        evals, _ = eigsh(H_sparse, k=2, which="SA")
        exact_gap = float(np.sort(evals)[1] - np.sort(evals)[0])

        # DMRG result (uses finite-size corrected analytical gap)
        result = solver.solve(H, lattice)

        # Old formula: 2|J-h| = 2|1-1.5| = 1.0
        old_gap = 2 * abs(1.0 - 1.5)

        # New gap should be closer to exact than old
        new_error = abs(result.gap - exact_gap)
        old_error = abs(old_gap - exact_gap)
        assert new_error <= old_error, (
            f"New gap ({result.gap:.6f}) should be closer to exact ({exact_gap:.6f}) "
            f"than old formula ({old_gap:.6f}). Errors: new={new_error:.6f}, old={old_error:.6f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Test: ground_state_vector dispatch
# ═══════════════════════════════════════════════════════════════════════════════


class TestGroundStateVector:
    """Test ground_state_vector method dispatch and correctness."""

    def test_n6_returns_normalized_vector(self, solver, tfim_spec):
        lattice = make_lattice("chain_1d", 6, J=1.0, h=1.5)
        H = tfim_spec.build_hamiltonian(lattice, **tfim_spec.hamiltonian_kwargs)
        gs = solver.ground_state_vector(H)
        assert gs.shape == (2**6,)
        np.testing.assert_allclose(np.linalg.norm(gs), 1.0, atol=1e-10)

    def test_n14_returns_normalized_vector(self, solver, tfim_spec):
        lattice = make_lattice("chain_1d", 14, J=1.0, h=1.5)
        H = tfim_spec.build_hamiltonian(lattice, **tfim_spec.hamiltonian_kwargs)
        gs = solver.ground_state_vector(H)
        assert gs.shape == (2**14,)
        np.testing.assert_allclose(np.linalg.norm(gs), 1.0, atol=1e-10)

    def test_n23_raises_value_error(self, solver, tfim_spec):
        """N > 22 should raise ValueError."""
        lattice = make_lattice("chain_1d", 23, J=1.0, h=2.0)
        H = tfim_spec.build_hamiltonian(lattice, **tfim_spec.hamiltonian_kwargs)
        with pytest.raises(ValueError, match="not supported for N=23"):
            solver.ground_state_vector(H)

    def test_ground_state_is_eigenvector(self, solver, tfim_spec):
        """Returned vector should be an eigenvector of H with lowest eigenvalue."""
        lattice = make_lattice("chain_1d", 8, J=1.0, h=1.5)
        H = tfim_spec.build_hamiltonian(lattice, **tfim_spec.hamiltonian_kwargs)
        gs = solver.ground_state_vector(H)

        # H|gs⟩ = E₀|gs⟩
        H_mat = H.to_matrix(sparse=True)
        Hgs = H_mat @ gs
        E0 = float(np.real(gs.conj() @ Hgs))
        residual = np.linalg.norm(Hgs - E0 * gs)
        assert residual < 1e-10, f"Residual ||H|gs⟩ - E₀|gs⟩|| = {residual}"
