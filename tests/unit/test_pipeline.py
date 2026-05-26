"""Unit tests for qmbp_simulation.pipeline module."""

from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.pipeline import load_phase12_dataset, save_phase12_dataset


class TestDatasetRoundTrip:
    """Test save_phase12_dataset / load_phase12_dataset round-trip."""

    def _make_sample_data(self, n_points=5):
        return dict(
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

    def test_save_load_preserves_arrays(self, tmp_path):
        data = self._make_sample_data()
        filepath = tmp_path / "test_dataset.npz"
        save_phase12_dataset(filepath, **data)
        loaded = load_phase12_dataset(filepath)

        np.testing.assert_allclose(loaded["h_values"], data["h_values"], atol=1e-10)
        np.testing.assert_allclose(loaded["theta_opt"], data["theta_opt"], atol=1e-10)
        np.testing.assert_allclose(loaded["ground_energies"], data["ground_energies"], atol=1e-10)

    def test_metadata_preserved(self, tmp_path):
        data = self._make_sample_data()
        filepath = tmp_path / "test_dataset.npz"
        save_phase12_dataset(filepath, **data)
        loaded = load_phase12_dataset(filepath)

        assert str(loaded["cost_function"]) == "energy"
        assert int(loaded["n_qubits"]) == 4
        assert int(loaded["p_layers"]) == 1


class TestCostFunctionValidation:
    """Test rejection of non-energy cost function datasets."""

    def test_non_energy_cost_function_raises(self, tmp_path):
        """Manually create a dataset with wrong cost_function."""
        filepath = tmp_path / "bad_dataset.npz"
        np.savez(
            filepath,
            cost_function="hybrid_observable",
            version="v7.0",
            h_values=np.array([1.0, 1.5]),
            J=1.0,
            n_qubits=4,
            p_layers=1,
            ground_energies=np.array([-3.0, -4.0]),
            gaps=np.array([0.5, 0.3]),
            mag_x=np.array([0.5, 0.8]),
            corr_zz=np.array([0.3, 0.1]),
            theta_opt=np.array([[0.1, 0.2], [0.3, 0.4]]),
            vqe_energies=np.array([-2.9, -3.8]),
            fidelities=np.array([0.95, 0.97]),
        )
        with pytest.raises(ValueError, match="cost_function"):
            load_phase12_dataset(filepath)
