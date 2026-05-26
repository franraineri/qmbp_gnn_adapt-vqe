"""Property-based tests for qmbp_simulation.optimizers submodule.

Uses Hypothesis to verify universal properties of VQE optimizer behavior
across many random inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.execution import NoiselessBackend
from qmbp_simulation.models import (
    HamiltonianBuilder,
    VQEConfig,
    make_lattice,
)
from qmbp_simulation.optimizers import VQEOptimizer

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures for property tests (module-level for speed)
# ─────────────────────────────────────────────────────────────────────────────

_LATTICE = make_lattice("chain_1d", 4, J=1.0, h=1.5)
_BUILDER = HVACircuitBuilder()
_QC, _THETA = _BUILDER.create(4, 1, _LATTICE)
_H = HamiltonianBuilder().build(_LATTICE)
_N_PARAMS = _QC.num_parameters  # 2 for p=1


# ─────────────────────────────────────────────────────────────────────────────
# Property 7: Descending sweep order enforcement
# **Validates: Requirements 6.3, 6.5, 6.6, 19.3**
# ─────────────────────────────────────────────────────────────────────────────


class TestProperty7DescendingSweepOrderEnforcement:
    """Property 7: For any h_values array where h_values[0] < h_values[-1]
    (ascending overall), descending_sweep() SHALL raise ValueError.

    The implementation checks if the first element is less than the last element
    to detect non-descending order.
    """

    @given(
        h_start=st.floats(min_value=0.01, max_value=1.5),
        h_end=st.floats(min_value=1.6, max_value=3.0),
        n_points=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=30, deadline=None)
    def test_ascending_h_values_raises_valueerror(self, h_start, h_end, n_points):
        """Any h_values where first < last raises ValueError."""
        # Generate ascending h_values (h_start < h_end guaranteed by ranges)
        h_values = np.linspace(h_start, h_end, n_points)

        config = VQEConfig(p_layers=1, n_restarts=1, maxiter=5)
        optimizer = VQEOptimizer(config)

        with pytest.raises(ValueError, match="descending order"):
            optimizer.descending_sweep(h_values, _QC, _LATTICE)

    @given(
        h_values=st.lists(
            st.floats(min_value=0.1, max_value=3.0),
            min_size=2,
            max_size=5,
        ).filter(lambda xs: xs[0] < xs[-1]),
    )
    @settings(max_examples=30, deadline=None)
    def test_random_non_descending_raises_valueerror(self, h_values):
        """Any random h_values list where first < last raises ValueError."""
        h_arr = np.array(h_values)

        config = VQEConfig(p_layers=1, n_restarts=1, maxiter=5)
        optimizer = VQEOptimizer(config)

        with pytest.raises(ValueError, match="descending order"):
            optimizer.descending_sweep(h_arr, _QC, _LATTICE)


# ─────────────────────────────────────────────────────────────────────────────
# Property 8: Warm-start propagation preserves parameters
# **Validates: Requirements 6.3, 6.5, 6.6, 19.3**
# ─────────────────────────────────────────────────────────────────────────────


class TestProperty8WarmStartPropagationPreservesParameters:
    """Property 8: For any non-zero h_value and any previous_theta array,
    get_initial_guess() with previous_theta SHALL return a copy (not the same
    object) with identical values. Modifying the copy must not affect the original.
    """

    @given(
        previous_theta=st.lists(
            st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=8,
        ).map(np.array),
        h_value=st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_returns_copy_not_same_object(self, previous_theta, h_value):
        """Returned array is a distinct object from previous_theta."""
        config = VQEConfig(p_layers=1)
        n_params = len(previous_theta)

        result = VQEOptimizer.get_initial_guess(
            n_params=n_params,
            h_value=h_value,
            config=config,
            previous_theta=previous_theta,
        )

        # Must be a different object
        assert result is not previous_theta
        # Must have same values
        np.testing.assert_array_equal(result, previous_theta)

    @given(
        previous_theta=st.lists(
            st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=8,
        ).map(np.array),
        h_value=st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_modifying_copy_does_not_affect_original(self, previous_theta, h_value):
        """Modifying the returned array does not mutate previous_theta."""
        config = VQEConfig(p_layers=1)
        n_params = len(previous_theta)
        original_values = previous_theta.copy()

        result = VQEOptimizer.get_initial_guess(
            n_params=n_params,
            h_value=h_value,
            config=config,
            previous_theta=previous_theta,
        )

        # Mutate the result
        result[:] = 999.0

        # Original must be unchanged
        np.testing.assert_array_equal(previous_theta, original_values)


# ─────────────────────────────────────────────────────────────────────────────
# Property 9: Optimization trajectory recording
# **Validates: Requirements 6.3, 6.5, 6.6, 19.3**
# ─────────────────────────────────────────────────────────────────────────────


class TestProperty9OptimizationTrajectoryRecording:
    """Property 9: For any VQE optimization with enable_callbacks=True,
    the returned VQEResult.trajectory SHALL be non-None and contain at least
    one energy entry.

    Uses a small circuit (N=4, p=1) with NoiselessBackend and minimal
    maxiter/n_restarts for speed.
    """

    @given(
        initial_params=st.lists(
            st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False, allow_infinity=False),
            min_size=_N_PARAMS,
            max_size=_N_PARAMS,
        ).map(np.array),
    )
    @settings(max_examples=30, deadline=None)
    def test_trajectory_non_none_when_callbacks_enabled(self, initial_params):
        """trajectory is not None when enable_callbacks=True."""
        config = VQEConfig(
            p_layers=1,
            n_restarts=1,
            maxiter=5,
            enable_callbacks=True,
        )
        backend = NoiselessBackend()
        optimizer = VQEOptimizer(config, backend=backend)

        result = optimizer.optimize(
            hamiltonian=_H,
            circuit=_QC,
            initial_guess=initial_params,
        )

        assert result.trajectory is not None, (
            "trajectory should be non-None when enable_callbacks=True"
        )
        assert len(result.trajectory.energies) >= 1, (
            "trajectory should contain at least one energy entry"
        )
