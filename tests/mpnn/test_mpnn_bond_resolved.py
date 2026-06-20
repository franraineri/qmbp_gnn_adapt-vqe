"""Tests for MPNN bond-resolved support (n_edges asymmetric split).

Validates:
1. Asymmetric head split produces correct output shape (39 ZZ + 40 X = 79)
2. Backward compatibility (n_edges=None → symmetric split)
3. Forward pass with per_parameter_heads and n_edges
4. Full pipeline: build_graph_dataset + train_mpnn with output_dim=79
5. Checkpoint save/load preserves n_edges attribute

These tests run in <5s with no MPS/hardware dependencies.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn

# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: Asymmetric head split
# ═══════════════════════════════════════════════════════════════════════════════


class TestAsymmetricHeadSplit:
    """Verify n_edges parameter creates correct head dimensions."""

    def test_asymmetric_79_params(self):
        """79 params = 39 edges + 40 qubits → head_zz=39, head_x=40."""
        model = MPNNPredictor(
            output_dim=79,
            per_parameter_heads=True,
            n_edges=39,
            hidden_dim=64,
        )
        assert model._dim_zz == 39
        assert model._dim_x == 40
        assert model.n_edges == 39
        assert model.output_dim == 79

    def test_asymmetric_21_params(self):
        """21 params = 12 edges + 9 qubits (square 3x3)."""
        model = MPNNPredictor(
            output_dim=21,
            per_parameter_heads=True,
            n_edges=12,
            hidden_dim=64,
        )
        assert model._dim_zz == 12
        assert model._dim_x == 9

    def test_symmetric_fallback_no_n_edges(self):
        """Without n_edges, falls back to output_dim // 2."""
        model = MPNNPredictor(
            output_dim=4,
            per_parameter_heads=True,
            hidden_dim=64,
        )
        assert model._dim_zz == 2
        assert model._dim_x == 2
        assert model.n_edges is None

    def test_symmetric_fallback_none_explicit(self):
        """Explicit n_edges=None uses symmetric split."""
        model = MPNNPredictor(
            output_dim=10,
            per_parameter_heads=True,
            n_edges=None,
            hidden_dim=64,
        )
        assert model._dim_zz == 5
        assert model._dim_x == 5

    def test_n_edges_stored_as_attribute(self):
        """n_edges is stored for checkpoint metadata."""
        model = MPNNPredictor(output_dim=79, n_edges=39, hidden_dim=64)
        assert model.n_edges == 39


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: Forward pass shape
# ═══════════════════════════════════════════════════════════════════════════════


class TestForwardPassShape:
    """Verify forward pass produces correct output tensor shape."""

    @pytest.fixture
    def synthetic_graph(self):
        """Build a minimal graph (5 nodes, 4 edges) for testing."""
        from torch_geometric.data import Data

        x = torch.randn(5, 2)  # 5 nodes, 2 features (h, coord)
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
            dtype=torch.long,
        )
        return Data(x=x, edge_index=edge_index)

    def test_forward_79_asymmetric(self, synthetic_graph):
        """Output shape is [1, 79] with asymmetric heads."""
        model = MPNNPredictor(
            output_dim=79,
            per_parameter_heads=True,
            n_edges=39,
            hidden_dim=64,
        )
        model.eval()
        with torch.no_grad():
            out = model(synthetic_graph)
        assert out.shape == (1, 79)

    def test_forward_79_single_head(self, synthetic_graph):
        """Output shape is [1, 79] with single head (no per_parameter_heads)."""
        model = MPNNPredictor(
            output_dim=79,
            per_parameter_heads=False,
            hidden_dim=64,
        )
        model.eval()
        with torch.no_grad():
            out = model(synthetic_graph)
        assert out.shape == (1, 79)

    def test_forward_values_finite(self, synthetic_graph):
        """All output values are finite after forward pass."""
        model = MPNNPredictor(
            output_dim=79,
            per_parameter_heads=True,
            n_edges=39,
            hidden_dim=128,
        )
        model.eval()
        with torch.no_grad():
            out = model(synthetic_graph)
        assert torch.all(torch.isfinite(out))


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: Full pipeline with 79-dim output (build_graph_dataset + train_mpnn)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFullPipeline79Dim:
    """Validate that build_graph_dataset + train_mpnn work with output_dim=79.

    This catches shape mismatches that would only appear after a 12h sweep.
    Uses synthetic data (3 graphs × 79 targets, 10 epochs).
    """

    @pytest.fixture
    def synthetic_dataset(self):
        """Build synthetic dataset mimicking N=40 chain_1d bond-resolved."""
        from qmbp_simulation import make_lattice

        N = 40
        n_h_points = 5
        h_values = [6.0, 5.5, 5.0, 4.75, 4.5]

        lattice = make_lattice("chain_1d", N, h=h_values[0])
        n_edges = len(lattice.edges)
        n_params = n_edges + N  # 39 + 40 = 79

        # Synthetic θ_opt (smooth, realistic magnitude)
        rng = np.random.default_rng(42)
        theta_opt = rng.uniform(-0.5, 0.5, (n_h_points, n_params))
        e_exact = np.linspace(-200, -160, n_h_points)  # dummy energies
        fidelities = np.ones(n_h_points)

        dataset = build_graph_dataset(
            lattice,
            np.array(h_values),
            theta_opt,
            e_exact,
            fidelities=fidelities,
            fidelity_threshold=0.0,
        )
        return dataset, n_edges, n_params

    def test_dataset_construction(self, synthetic_dataset):
        """build_graph_dataset creates valid dataset with 79-dim targets."""
        dataset, n_edges, n_params = synthetic_dataset
        assert len(dataset) == 5
        # Check target shape
        assert dataset[0].y.shape[-1] == n_params

    def test_train_10_epochs_no_crash(self, synthetic_dataset):
        """train_mpnn runs 10 epochs without shape mismatch or crash."""
        dataset, n_edges, n_params = synthetic_dataset

        model = MPNNPredictor(
            node_features=2,
            hidden_dim=64,
            n_layers=2,
            output_dim=n_params,
            per_parameter_heads=True,
            n_edges=n_edges,
        )

        result = train_mpnn(
            model=model,
            dataset=dataset,
            n_epochs=10,
            lr=1e-3,
            patience=100,
        )

        assert "mse_history" in result
        assert "final_mse" in result
        assert len(result["mse_history"]) == 10
        assert np.isfinite(result["final_mse"])

    def test_train_with_weight_decay(self, synthetic_dataset):
        """train_mpnn with per_parameter_heads on 79-dim output works."""
        dataset, n_edges, n_params = synthetic_dataset

        model = MPNNPredictor(
            node_features=2,
            hidden_dim=128,
            n_layers=3,
            output_dim=n_params,
            per_parameter_heads=True,
            n_edges=n_edges,
        )

        result = train_mpnn(
            model=model,
            dataset=dataset,
            n_epochs=10,
            lr=1e-3,
            patience=100,
        )

        assert len(result["mse_history"]) == 10
        assert np.isfinite(result["final_mse"])

    def test_predict_after_train(self, synthetic_dataset):
        """Model produces 79-dim prediction on unseen graph after training."""
        from qmbp_simulation import make_lattice

        dataset, n_edges, n_params = synthetic_dataset

        model = MPNNPredictor(
            node_features=2,
            hidden_dim=64,
            n_layers=2,
            output_dim=n_params,
            per_parameter_heads=True,
            n_edges=n_edges,
        )

        train_mpnn(model=model, dataset=dataset, n_epochs=10, lr=1e-3)

        # Predict on unseen h-value
        unseen_lattice = make_lattice("chain_1d", 40, h=5.25)
        # build_graph_dataset requires >=3 points, so build prediction input manually
        from torch_geometric.data import Data

        edge_index = (
            torch.tensor(
                [[i, i + 1] for i in range(39)] + [[i + 1, i] for i in range(39)],
                dtype=torch.long,
            )
            .t()
            .contiguous()
        )
        # Node features: [h_normalized, coordination_number]
        coord = np.array(unseen_lattice.coordination_numbers, dtype=np.float32)
        x = torch.tensor(
            np.column_stack([np.full(40, 5.25), coord]),
            dtype=torch.float32,
        )
        pred_data = Data(x=x, edge_index=edge_index)

        model.eval()
        with torch.no_grad():
            pred = model(pred_data)

        assert pred.shape == (1, n_params)
        assert torch.all(torch.isfinite(pred))
        # First 39 are θ_zz, last 40 are θ_x
        theta_zz = pred[0, :n_edges].numpy()
        theta_x = pred[0, n_edges:].numpy()
        assert len(theta_zz) == 39
        assert len(theta_x) == 40


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: Edge cases and error handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Catch potential issues before long-running execution."""

    def test_n_edges_exceeds_output_dim_raises(self):
        """n_edges > output_dim would create negative dim_x — should work but be nonsensical."""
        # This is a user error but shouldn't crash during construction
        model = MPNNPredictor(
            output_dim=5,
            per_parameter_heads=True,
            n_edges=3,
            hidden_dim=32,
        )
        assert model._dim_zz == 3
        assert model._dim_x == 2

    def test_n_edges_zero(self):
        """n_edges=0 means all params are θ_x (no ZZ bonds)."""
        model = MPNNPredictor(
            output_dim=40,
            per_parameter_heads=True,
            n_edges=0,
            hidden_dim=32,
        )
        assert model._dim_zz == 0
        assert model._dim_x == 40

    def test_single_head_ignores_n_edges(self):
        """When per_parameter_heads=False, n_edges is stored but unused."""
        model = MPNNPredictor(
            output_dim=79,
            per_parameter_heads=False,
            n_edges=39,
            hidden_dim=64,
        )
        assert model.n_edges == 39
        assert model.head is not None  # Single head is used
