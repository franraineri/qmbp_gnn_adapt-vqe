"""Integration tests for backward compatibility with legacy formats."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pytest
import torch

from qmbp_simulation.pipeline import load_phase12_dataset
from qmbp_simulation.predictors import MPNNPredictor, load_mpnn_checkpoint

pytestmark = pytest.mark.integration


@pytest.fixture
def legacy_dataset_path(tmp_path):
    """Create a minimal legacy dataset in the old format (v6.0 schema)."""
    filepath = tmp_path / "legacy_dataset.npz"
    np.savez(
        filepath,
        # Old schema: has cost_function="energy" and version="v6.0"
        cost_function="energy",
        version="v6.0",
        h_values=np.array([2.0, 1.5, 1.0, 0.5]),
        J=np.float64(1.0),
        n_qubits=np.int64(4),
        p_layers=np.int64(2),
        ground_energies=np.array([-5.2, -4.8, -4.3, -3.9]),
        gaps=np.array([0.4, 0.35, 0.2, 0.15]),
        mag_x=np.array([0.9, 0.7, 0.5, 0.3]),
        corr_zz=np.array([0.1, 0.3, 0.5, 0.7]),
        theta_opt=np.random.rand(4, 4),
        vqe_energies=np.array([-5.1, -4.7, -4.2, -3.8]),
        fidelities=np.array([0.99, 0.97, 0.95, 0.93]),
    )
    return filepath


@pytest.fixture
def legacy_checkpoint_path(tmp_path):
    """Create a legacy MPNN checkpoint (raw state_dict, no metadata)."""
    model = MPNNPredictor(node_features=2, hidden_dim=64, n_layers=3, output_dim=4)
    path = tmp_path / "legacy_model.pt"
    # V6.0 format: just the raw state_dict, no architecture metadata
    torch.save(model.state_dict(), path)
    return path


class TestLegacyDatasetLoading:
    """Test loading legacy datasets saved by previous save_phase12_dataset."""

    def test_loads_v6_dataset_with_deprecation_warning(self, legacy_dataset_path):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_phase12_dataset(legacy_dataset_path)
            # Should emit deprecation warning for v6.0 schema
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "legacy" in str(deprecation_warnings[0].message).lower()

    def test_legacy_data_arrays_accessible(self, legacy_dataset_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            data = load_phase12_dataset(legacy_dataset_path)

        assert "h_values" in data
        assert "theta_opt" in data
        assert "ground_energies" in data
        np.testing.assert_array_equal(data["h_values"], np.array([2.0, 1.5, 1.0, 0.5]))


class TestLegacyCheckpointLoading:
    """Test loading legacy MPNN checkpoints."""

    def test_loads_raw_state_dict_checkpoint(self, legacy_checkpoint_path, caplog):
        """V6.0 checkpoints are raw state_dicts without metadata."""
        with caplog.at_level(logging.WARNING, logger="qmbp_simulation"):
            model = load_mpnn_checkpoint(str(legacy_checkpoint_path))

        assert isinstance(model, MPNNPredictor)
        # Should log warning about legacy format
        assert any("legacy" in r.message.lower() for r in caplog.records)

    def test_loaded_model_can_forward(self, legacy_checkpoint_path):
        """Loaded legacy model should be functional."""
        from torch_geometric.data import Data

        model = load_mpnn_checkpoint(str(legacy_checkpoint_path))
        model.eval()
        # Create minimal input — single graph with 4 nodes
        x = torch.randn(4, 2)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        with torch.no_grad():
            output = model(data)
        # MPNN output: (1, output_dim) for single graph or (output_dim,)
        assert output.numel() == 4  # output_dim=4
