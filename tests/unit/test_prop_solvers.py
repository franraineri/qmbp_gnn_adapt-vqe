"""Property-based tests for qmbp_simulation.solvers submodule.

Uses Hypothesis to verify universal properties of the ClassicalSolver
across many random inputs.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from qmbp_simulation.models import HamiltonianBuilder, make_lattice
from qmbp_simulation.solvers import ClassicalSolver

# ─────────────────────────────────────────────────────────────────────────────
# Strategies
# ─────────────────────────────────────────────────────────────────────────────


def chain_lattice_strategy() -> st.SearchStrategy:
    """Generate valid chain_1d lattices with n_qubits in [2, 6] and h in [0.1, 3.0]."""
    return st.tuples(
        st.integers(min_value=2, max_value=6),
        st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Property 3: Classical solver returns structurally valid results
# **Validates: Requirements 3.1, 3.4**
# ─────────────────────────────────────────────────────────────────────────────


class TestProperty3ClassicalSolverReturnsStructurallyValidResults:
    """Property 3: For any valid chain_1d lattice (n_qubits 2-6) and h in [0.1, 3.0],
    the ClassicalSolver returns a GroundTruthResult with:
    - gap >= 0 (spectral gap is non-negative)
    - per_site_mag_x has shape (n_qubits,)
    - per_bond_corr_zz has shape (n_edges,) where n_edges = len(lattice.edges)
    - ground_energy is finite (not NaN or inf)
    - ground_state is either None or has shape (2**n_qubits,)
    - If ground_state is not None, it should be normalized (|ψ|² ≈ 1)
    """

    @given(lattice_params=chain_lattice_strategy())
    @settings(max_examples=50, deadline=None)
    def test_gap_is_nonnegative(self, lattice_params):
        """Spectral gap E₁ - E₀ must be >= 0 for any valid input."""
        n_qubits, h_val = lattice_params
        lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=h_val)
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        H = builder.build(lattice)
        result = solver.solve(H, lattice)

        assert result.gap >= 0.0, f"Negative gap={result.gap} for n_qubits={n_qubits}, h={h_val}"

    @given(lattice_params=chain_lattice_strategy())
    @settings(max_examples=50, deadline=None)
    def test_per_site_mag_x_shape(self, lattice_params):
        """per_site_mag_x must have shape (n_qubits,)."""
        n_qubits, h_val = lattice_params
        lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=h_val)
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        H = builder.build(lattice)
        result = solver.solve(H, lattice)

        assert result.per_site_mag_x.shape == (n_qubits,), (
            f"Expected shape ({n_qubits},), got {result.per_site_mag_x.shape}"
        )

    @given(lattice_params=chain_lattice_strategy())
    @settings(max_examples=50, deadline=None)
    def test_per_bond_corr_zz_shape(self, lattice_params):
        """per_bond_corr_zz must have shape (n_edges,)."""
        n_qubits, h_val = lattice_params
        lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=h_val)
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        H = builder.build(lattice)
        result = solver.solve(H, lattice)

        n_edges = len(lattice.edges)
        assert result.per_bond_corr_zz.shape == (n_edges,), (
            f"Expected shape ({n_edges},), got {result.per_bond_corr_zz.shape}"
        )

    @given(lattice_params=chain_lattice_strategy())
    @settings(max_examples=50, deadline=None)
    def test_ground_energy_is_finite(self, lattice_params):
        """ground_energy must be finite (not NaN or inf)."""
        n_qubits, h_val = lattice_params
        lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=h_val)
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        H = builder.build(lattice)
        result = solver.solve(H, lattice)

        assert np.isfinite(result.ground_energy), (
            f"Non-finite ground_energy={result.ground_energy} for n={n_qubits}, h={h_val}"
        )

    @given(lattice_params=chain_lattice_strategy())
    @settings(max_examples=50, deadline=None)
    def test_ground_state_shape_or_none(self, lattice_params):
        """ground_state is either None or has shape (2**n_qubits,)."""
        n_qubits, h_val = lattice_params
        lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=h_val)
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        H = builder.build(lattice)
        result = solver.solve(H, lattice)

        if result.ground_state is not None:
            expected_dim = 2**n_qubits
            assert result.ground_state.shape == (expected_dim,), (
                f"Expected shape ({expected_dim},), got {result.ground_state.shape}"
            )

    @given(lattice_params=chain_lattice_strategy())
    @settings(max_examples=50, deadline=None)
    def test_ground_state_is_normalized(self, lattice_params):
        """If ground_state is not None, |ψ|² ≈ 1."""
        n_qubits, h_val = lattice_params
        lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=h_val)
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        H = builder.build(lattice)
        result = solver.solve(H, lattice)

        if result.ground_state is not None:
            norm_sq = float(np.sum(np.abs(result.ground_state) ** 2))
            np.testing.assert_allclose(
                norm_sq,
                1.0,
                atol=1e-10,
                err_msg=f"|ψ|²={norm_sq} != 1 for n={n_qubits}, h={h_val}",
            )
