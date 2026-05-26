"""Unit tests for qmbp_simulation.predictors module."""

from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.models import make_lattice
from qmbp_simulation.predictors import (
    MPNNPredictor,
    build_graph_dataset,
    load_mpnn_checkpoint,
    save_mpnn_checkpoint,
)


class TestBuildGraphDataset:
    """Test build_graph_dataset output shapes."""

    def test_output_length_matches_input(self):
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)
        h_values = np.array([2.0, 1.5, 1.0, 0.5])
        theta_opt = np.random.rand(4, 2)
        e_exact = np.array([-5.0, -4.5, -4.0, -3.5])
        fidelities = np.ones(4)  # All pass filter

        dataset = build_graph_dataset(lattice, h_values, theta_opt, e_exact, fidelities)
        assert len(dataset) == 4

    def test_node_features_shape(self):
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)
        h_values = np.array([2.0, 1.5, 1.0])
        theta_opt = np.random.rand(3, 2)
        e_exact = np.array([-5.0, -4.5, -4.0])
        fidelities = np.ones(3)

        dataset = build_graph_dataset(lattice, h_values, theta_opt, e_exact, fidelities)
        # Each graph has x shape [n_qubits, 2] (h_feat, coord)
        assert dataset[0].x.shape == (4, 2)
        # Target shape matches theta_opt columns
        assert dataset[0].y.shape == (2,)


class TestFidelityFilter:
    """Test fidelity filter enforcement."""

    def test_low_fidelity_points_filtered(self):
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)
        h_values = np.array([2.0, 1.5, 1.0, 0.5, 0.3])
        theta_opt = np.random.rand(5, 2)
        e_exact = np.array([-5.0, -4.5, -4.0, -3.5, -3.0])
        # Only first 3 pass the 0.93 threshold
        fidelities = np.array([0.99, 0.95, 0.94, 0.80, 0.70])

        dataset = build_graph_dataset(lattice, h_values, theta_opt, e_exact, fidelities)
        assert len(dataset) == 3

    def test_fewer_than_3_passing_raises(self):
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.0)
        h_values = np.array([2.0, 1.5, 1.0])
        theta_opt = np.random.rand(3, 2)
        e_exact = np.array([-5.0, -4.5, -4.0])
        # Only 2 pass
        fidelities = np.array([0.99, 0.95, 0.50])

        with pytest.raises(ValueError, match="Fewer than 3"):
            build_graph_dataset(lattice, h_values, theta_opt, e_exact, fidelities)


class TestCheckpointRoundTrip:
    """Test checkpoint save/load round-trip."""

    def test_save_load_preserves_state(self, tmp_path):
        model = MPNNPredictor(node_features=2, hidden_dim=32, n_layers=2, output_dim=2)
        path = str(tmp_path / "test_model.pt")
        save_mpnn_checkpoint(model, path, training_metadata={"epoch": 100})

        loaded = load_mpnn_checkpoint(path)
        # Compare state dicts
        for key in model.state_dict():
            import torch

            assert torch.equal(model.state_dict()[key], loaded.state_dict()[key]), (
                f"Mismatch in {key}"
            )
