"""Property-based tests for qmbp_simulation.pipeline module.

# Feature: framework-restructure, Property 15: Dataset save/load round-trip
# Feature: framework-restructure, Property 16: Dataset load rejects invalid cost function
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qmbp_simulation.pipeline import load_phase12_dataset, save_phase12_dataset

# ---------------------------------------------------------------------------
# Strategies for generating valid dataset components
# ---------------------------------------------------------------------------

# Number of h-points in the dataset (small for speed, ≥2 for meaningful tests)
n_points_strategy = st.integers(min_value=2, max_value=10)

# Number of HVA layers (1 or 2)
p_layers_strategy = st.integers(min_value=1, max_value=2)


@st.composite
def valid_dataset(draw):
    """Generate a valid Phase 1+2 dataset with consistent array shapes."""
    n_points = draw(n_points_strategy)
    p_layers = draw(p_layers_strategy)
    n_qubits = draw(st.integers(min_value=4, max_value=10))

    # h_values: descending sweep from ~2.0 to ~0.5
    h_values = np.sort(
        draw(
            st.lists(
                st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False),
                min_size=n_points,
                max_size=n_points,
            )
        )
    )[::-1]  # Descending order

    J = draw(st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False))

    # 1D arrays of shape (n_points,)
    ground_energies = np.array(
        draw(
            st.lists(
                st.floats(min_value=-20.0, max_value=0.0, allow_nan=False, allow_infinity=False),
                min_size=n_points,
                max_size=n_points,
            )
        )
    )
    gaps = np.array(
        draw(
            st.lists(
                st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
                min_size=n_points,
                max_size=n_points,
            )
        )
    )
    mag_x = np.array(
        draw(
            st.lists(
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=n_points,
                max_size=n_points,
            )
        )
    )
    corr_zz = np.array(
        draw(
            st.lists(
                st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=n_points,
                max_size=n_points,
            )
        )
    )
    vqe_energies = np.array(
        draw(
            st.lists(
                st.floats(min_value=-20.0, max_value=0.0, allow_nan=False, allow_infinity=False),
                min_size=n_points,
                max_size=n_points,
            )
        )
    )
    fidelities = np.array(
        draw(
            st.lists(
                st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
                min_size=n_points,
                max_size=n_points,
            )
        )
    )

    # 2D array: theta_opt has shape (n_points, 2*p_layers)
    theta_opt = np.array(
        draw(
            st.lists(
                st.lists(
                    st.floats(
                        min_value=-np.pi,
                        max_value=np.pi,
                        allow_nan=False,
                        allow_infinity=False,
                    ),
                    min_size=2 * p_layers,
                    max_size=2 * p_layers,
                ),
                min_size=n_points,
                max_size=n_points,
            )
        )
    )

    return {
        "h_values": h_values,
        "J": J,
        "n_qubits": n_qubits,
        "p_layers": p_layers,
        "ground_energies": ground_energies,
        "gaps": gaps,
        "mag_x": mag_x,
        "corr_zz": corr_zz,
        "theta_opt": theta_opt,
        "vqe_energies": vqe_energies,
        "fidelities": fidelities,
    }


# ---------------------------------------------------------------------------
# Property 15: Dataset save/load round-trip
# For any valid Phase 1+2 dataset, save_phase12_dataset() followed by
# load_phase12_dataset() SHALL preserve all array values within tolerance.
# **Validates: Requirements 10.1, 10.2, 10.3, 20.1**
# ---------------------------------------------------------------------------


@settings(max_examples=20, deadline=None)
@given(data=valid_dataset())
def test_dataset_roundtrip_arrays_preserved(data: dict) -> None:
    """All numpy arrays are preserved within tolerance after save/load.

    **Validates: Requirements 10.1, 10.2, 10.3, 20.1**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "test_dataset.npz"
        save_phase12_dataset(filepath, **data)
        loaded = load_phase12_dataset(filepath)

        # 1D arrays
        np.testing.assert_allclose(loaded["h_values"], data["h_values"], atol=1e-10)
        np.testing.assert_allclose(loaded["ground_energies"], data["ground_energies"], atol=1e-10)
        np.testing.assert_allclose(loaded["gaps"], data["gaps"], atol=1e-10)
        np.testing.assert_allclose(loaded["mag_x"], data["mag_x"], atol=1e-10)
        np.testing.assert_allclose(loaded["corr_zz"], data["corr_zz"], atol=1e-10)
        np.testing.assert_allclose(loaded["vqe_energies"], data["vqe_energies"], atol=1e-10)
        np.testing.assert_allclose(loaded["fidelities"], data["fidelities"], atol=1e-10)

        # 2D array
        np.testing.assert_allclose(loaded["theta_opt"], data["theta_opt"], atol=1e-10)


@settings(max_examples=20, deadline=None)
@given(data=valid_dataset())
def test_dataset_roundtrip_scalars_preserved(data: dict) -> None:
    """Scalar values (J, n_qubits, p_layers) are exactly preserved.

    **Validates: Requirements 10.1, 10.2, 10.3, 20.1**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "test_dataset.npz"
        save_phase12_dataset(filepath, **data)
        loaded = load_phase12_dataset(filepath)

        assert float(loaded["J"]) == pytest.approx(data["J"], abs=1e-10)
        assert int(loaded["n_qubits"]) == data["n_qubits"]
        assert int(loaded["p_layers"]) == data["p_layers"]


@settings(max_examples=20, deadline=None)
@given(data=valid_dataset())
def test_dataset_roundtrip_metadata_valid(data: dict) -> None:
    """Loaded dataset contains valid cost_function and version metadata.

    **Validates: Requirements 10.1, 10.2, 10.3, 20.1**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "test_dataset.npz"
        save_phase12_dataset(filepath, **data)
        loaded = load_phase12_dataset(filepath)

        assert str(loaded["cost_function"]) == "energy"
        assert "version" in loaded


# ---------------------------------------------------------------------------
# Property 16: Dataset load rejects invalid cost function
# For any dataset file where cost_function metadata is not "energy",
# load_phase12_dataset() SHALL raise ValueError.
# **Validates: Requirements 10.1, 10.2, 10.3, 20.1**
# ---------------------------------------------------------------------------

# Strategy for non-"energy" cost function strings
non_energy_cost_functions = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=1,
    max_size=30,
).filter(lambda s: s != "energy")


@settings(max_examples=20, deadline=None)
@given(
    cost_fn=non_energy_cost_functions,
    n_points=st.integers(min_value=2, max_value=5),
)
def test_load_rejects_non_energy_cost_function(cost_fn: str, n_points: int) -> None:
    """Loading a dataset with cost_function != 'energy' raises ValueError.

    **Validates: Requirements 10.1, 10.2, 10.3, 20.1**
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        filepath = Path(tmp_dir) / "bad_dataset.npz"

        # Manually create a dataset with invalid cost_function
        np.savez(
            filepath,
            cost_function=cost_fn,
            version="v7.0",
            h_values=np.linspace(2.0, 0.5, n_points),
            J=1.0,
            n_qubits=4,
            p_layers=1,
            ground_energies=np.random.rand(n_points) * -5,
            gaps=np.random.rand(n_points) * 0.5 + 0.1,
            mag_x=np.random.rand(n_points),
            corr_zz=np.random.rand(n_points),
            theta_opt=np.random.rand(n_points, 2),
            vqe_energies=np.random.rand(n_points) * -4.5,
            fidelities=np.random.rand(n_points) * 0.1 + 0.9,
        )

        with pytest.raises(ValueError, match="cost_function"):
            load_phase12_dataset(filepath)
