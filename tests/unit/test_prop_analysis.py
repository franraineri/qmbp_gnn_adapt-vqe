"""Property-based tests for qmbp_simulation.analysis module.

# Feature: framework-restructure, Property 20: Analysis metrics satisfy invariants
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from qmbp_simulation.analysis import (
    compute_energy_decomposition,
    compute_snr,
    compute_theta_smoothness,
)

# ---------------------------------------------------------------------------
# Property 20: Analysis metrics satisfy invariants
# For any valid inputs:
# - compute_snr() SHALL return a value >= 0
# - compute_theta_smoothness() SHALL return a value >= 0 (or None for < 2 points)
# - compute_energy_decomposition() SHALL return components where
#   error_from_circuit + error_from_mpnn == |e_predicted - e_exact| within 1e-12
# **Validates: Requirements 9.2, 9.3, 9.5**
# ---------------------------------------------------------------------------


# --- Strategies ---

# Finite observable values (no NaN, no Inf)
finite_floats = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)

# Positive shots (1 to 10000)
positive_shots = st.integers(min_value=1, max_value=10000)


# 2D theta arrays with shape (n_points, n_params), n_points >= 2
theta_arrays_valid = st.builds(
    lambda rows, cols: np.random.default_rng(42).standard_normal((rows, cols)),
    rows=st.integers(min_value=3, max_value=10),
    cols=st.integers(min_value=2, max_value=4),
)

# Single-point theta arrays (should return None)
theta_arrays_single = st.builds(
    lambda cols: np.random.default_rng(42).standard_normal((1, cols)),
    cols=st.integers(min_value=2, max_value=4),
)

# Energy values where e_exact <= e_vqe_ceiling <= e_predicted (all negative)
energy_triples = st.tuples(
    st.floats(min_value=-100.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-100.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-100.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
).map(lambda t: tuple(sorted(t)))  # sorted ascending: (e_exact, e_vqe_ceiling, e_predicted)


# --- Property Tests: SNR ---


@settings(max_examples=50, deadline=None)
@given(observable_value=finite_floats, shots=positive_shots)
def test_snr_non_negative(observable_value: float, shots: int) -> None:
    """compute_snr returns a non-negative value for any finite observable and positive shots.

    **Validates: Requirements 9.2, 9.3, 9.5**
    """
    result = compute_snr(observable_value, shots)
    assert result >= 0.0, f"SNR should be >= 0, got {result}"


@settings(max_examples=50, deadline=None)
@given(observable_value=finite_floats, shots=positive_shots)
def test_snr_formula_correct(observable_value: float, shots: int) -> None:
    """compute_snr equals |observable_value| * sqrt(shots).

    **Validates: Requirements 9.2, 9.3, 9.5**
    """
    result = compute_snr(observable_value, shots)
    expected = abs(observable_value) * np.sqrt(shots)
    np.testing.assert_allclose(result, expected, atol=1e-10)


# --- Property Tests: Theta Smoothness ---


@settings(max_examples=50, deadline=None)
@given(theta=theta_arrays_valid)
def test_theta_smoothness_non_negative(theta: np.ndarray) -> None:
    """compute_theta_smoothness returns a non-negative value for arrays with >= 2 points.

    **Validates: Requirements 9.2, 9.3, 9.5**
    """
    result = compute_theta_smoothness(theta)
    assert result is not None, "Should not be None for arrays with >= 2 rows"
    assert result >= 0.0, f"Smoothness should be >= 0, got {result}"


@settings(max_examples=50, deadline=None)
@given(theta=theta_arrays_single)
def test_theta_smoothness_none_for_single_point(theta: np.ndarray) -> None:
    """compute_theta_smoothness returns None for arrays with fewer than 2 points.

    **Validates: Requirements 9.2, 9.3, 9.5**
    """
    result = compute_theta_smoothness(theta)
    assert result is None, f"Should be None for single-point array, got {result}"


# --- Property Tests: Energy Decomposition ---


@settings(max_examples=50, deadline=None)
@given(energies=energy_triples)
def test_energy_decomposition_sums_correctly(
    energies: tuple[float, float, float],
) -> None:
    """error_from_circuit + error_from_mpnn == |e_predicted - e_exact| within 1e-12.

    **Validates: Requirements 9.2, 9.3, 9.5**
    """
    e_exact, e_vqe_ceiling, e_predicted = energies

    result = compute_energy_decomposition(e_exact, e_vqe_ceiling, e_predicted)

    total_error = abs(e_predicted - e_exact)
    component_sum = result["error_from_circuit"] + result["error_from_mpnn"]

    np.testing.assert_allclose(
        component_sum,
        total_error,
        atol=1e-12,
        err_msg=(
            f"Decomposition sum mismatch: "
            f"error_from_circuit={result['error_from_circuit']}, "
            f"error_from_mpnn={result['error_from_mpnn']}, "
            f"sum={component_sum}, total_error={total_error}"
        ),
    )


@settings(max_examples=50, deadline=None)
@given(energies=energy_triples)
def test_energy_decomposition_components_non_negative(
    energies: tuple[float, float, float],
) -> None:
    """Both error_from_circuit and error_from_mpnn are non-negative.

    **Validates: Requirements 9.2, 9.3, 9.5**
    """
    e_exact, e_vqe_ceiling, e_predicted = energies

    result = compute_energy_decomposition(e_exact, e_vqe_ceiling, e_predicted)

    assert result["error_from_circuit"] >= 0.0, (
        f"error_from_circuit should be >= 0, got {result['error_from_circuit']}"
    )
    assert result["error_from_mpnn"] >= 0.0, (
        f"error_from_mpnn should be >= 0, got {result['error_from_mpnn']}"
    )
