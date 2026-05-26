"""Unit tests for qmbp_simulation.solvers module."""

from __future__ import annotations

import numpy as np

from qmbp_simulation.models import HamiltonianBuilder, make_lattice
from qmbp_simulation.solvers import ClassicalSolver


class TestClassicalSolver:
    """Test ClassicalSolver.solve() returns valid GroundTruthResult."""

    def test_solve_returns_valid_result(self, small_lattice):
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        H = builder.build(small_lattice)
        result = solver.solve(H, small_lattice)

        assert result.h_value == small_lattice.h
        assert isinstance(result.ground_energy, float)
        assert np.isfinite(result.ground_energy)
        assert result.gap >= 0.0

    def test_ground_state_shape(self, small_lattice):
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        H = builder.build(small_lattice)
        result = solver.solve(H, small_lattice)

        if result.ground_state is not None:
            expected_dim = 2**small_lattice.n_qubits
            assert result.ground_state.shape == (expected_dim,)

    def test_gap_is_nonnegative(self):
        """Gap must be ≥ 0 for all h values."""
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        for h in [0.5, 1.0, 1.5, 2.0]:
            lattice = make_lattice("chain_1d", 4, J=1.0, h=h)
            H = builder.build(lattice)
            result = solver.solve(H, lattice)
            assert result.gap >= 0.0, f"Negative gap at h={h}"

    def test_gap_fallback_behavior(self):
        """Solver should still return a valid gap even for small systems."""
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        # N=2 is the smallest valid system
        lattice = make_lattice("chain_1d", 2, J=1.0, h=1.0)
        H = builder.build(lattice)
        result = solver.solve(H, lattice)
        assert result.gap >= 0.0
        assert np.isfinite(result.gap)

    def test_observables_are_finite(self, small_lattice):
        """mag_x and corr_zz must be finite."""
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        H = builder.build(small_lattice)
        result = solver.solve(H, small_lattice)
        assert np.isfinite(result.mag_x)
        assert np.isfinite(result.corr_zz)
