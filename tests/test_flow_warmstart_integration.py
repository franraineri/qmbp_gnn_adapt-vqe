"""Integration smoke tests for FlowWarmstartManager.

Fast, deterministic checks (no Hypothesis) covering:
- Constructor defaults and custom params
- _extract_embedding frozen-encoder invariant
- train() correctness and empty-dataset error
- sample() before train raises RuntimeError
- trainable_param_count delegation

These complement the PBT tests in test_flow_warmstart.py by providing
fast deterministic verification (~0.3s total) suitable for CI pre-checks.

Run: pytest tests/test_flow_warmstart_integration.py -v
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from qmbp_simulation.analysis.flow_warmstart import (
    FlowWarmstartManager,
    _extract_embedding,
)
from qmbp_simulation.analysis.normalizing_flow import EmbeddingMAF
from qmbp_simulation.predictors import MPNNPredictor

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_graph(n_nodes: int = 4, theta_dim: int = 4) -> Data:
    """Create a minimal chain graph with random features."""
    x = torch.randn(n_nodes, 2)
    src = list(range(n_nodes - 1)) + list(range(1, n_nodes))
    dst = list(range(1, n_nodes)) + list(range(n_nodes - 1))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.randn(theta_dim)
    return Data(x=x, edge_index=edge_index, y=y)


def _make_dataset(n_data: int = 5, theta_dim: int = 4) -> list[Data]:
    return [_make_graph(theta_dim=theta_dim) for _ in range(n_data)]


# ── Constructor Tests ────────────────────────────────────────────────────


class TestFlowWarmstartConstructor:
    """Verify constructor defaults and custom parameter acceptance."""

    def test_defaults(self):
        mgr = FlowWarmstartManager()
        assert mgr.embedding_dim == 64
        assert mgr.theta_dim == 4
        assert mgr.n_flow_layers == 2
        assert mgr.hidden_dim == 32
        assert mgr.n_epochs == 500
        assert mgr.lr == 1e-3
        assert mgr.patience == 0
        assert mgr.flow_model is None
        assert mgr.is_trained is False

    def test_custom_params(self):
        mgr = FlowWarmstartManager(
            embedding_dim=128,
            theta_dim=8,
            n_flow_layers=3,
            hidden_dim=64,
            n_epochs=200,
            lr=5e-4,
            patience=50,
        )
        assert mgr.embedding_dim == 128
        assert mgr.theta_dim == 8
        assert mgr.n_flow_layers == 3
        assert mgr.hidden_dim == 64
        assert mgr.n_epochs == 200
        assert mgr.lr == 5e-4
        assert mgr.patience == 50


# ── _extract_embedding Tests ─────────────────────────────────────────────


class TestExtractEmbedding:
    """Verify frozen-encoder embedding extraction."""

    def test_output_shape(self):
        model = MPNNPredictor(node_features=2, hidden_dim=16, output_dim=4)
        model.eval()
        data = _make_graph(n_nodes=4)
        z = _extract_embedding(model, data)
        assert z.shape == (1, 16)

    def test_frozen_encoder_invariant(self):
        model = MPNNPredictor(node_features=2, hidden_dim=32, output_dim=4)
        model.eval()
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        data = _make_graph()
        _extract_embedding(model, data)
        for name, p in model.named_parameters():
            assert torch.equal(params_before[name], p), f"Param '{name}' changed!"

    def test_device_consistency(self):
        model = MPNNPredictor(node_features=2, hidden_dim=16, output_dim=4)
        model.eval()
        data = _make_graph()
        z = _extract_embedding(model, data)
        assert z.device == next(model.parameters()).device


# ── train() Tests ────────────────────────────────────────────────────────


class TestFlowWarmstartTrain:
    """Verify train() behavior and invariants."""

    def test_basic_train(self):
        model = MPNNPredictor(node_features=2, hidden_dim=32, output_dim=4)
        model.eval()
        dataset = _make_dataset(n_data=5, theta_dim=4)
        mgr = FlowWarmstartManager(embedding_dim=32, theta_dim=4, n_epochs=10, lr=1e-3)
        result = mgr.train(model, dataset)

        assert mgr.is_trained is True
        assert mgr.flow_model is not None
        assert "nll_history" in result
        assert "final_nll" in result
        assert len(result["nll_history"]) == 10
        assert isinstance(result["final_nll"], float)
        assert abs(result["final_nll"] - result["nll_history"][-1]) < 1e-9

    def test_frozen_encoder_after_train(self):
        model = MPNNPredictor(node_features=2, hidden_dim=32, output_dim=4)
        model.eval()
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        dataset = _make_dataset()
        mgr = FlowWarmstartManager(embedding_dim=32, theta_dim=4, n_epochs=5)
        mgr.train(model, dataset)
        for name, p in model.named_parameters():
            assert torch.allclose(params_before[name], p, atol=0.0, rtol=0.0)

    def test_empty_dataset_raises(self):
        model = MPNNPredictor(node_features=2, hidden_dim=32, output_dim=4)
        mgr = FlowWarmstartManager(embedding_dim=32, theta_dim=4)
        with pytest.raises(ValueError, match="dataset is empty"):
            mgr.train(model, [])

    def test_trainable_param_count_after_train(self):
        model = MPNNPredictor(node_features=2, hidden_dim=32, output_dim=4)
        model.eval()
        dataset = _make_dataset()
        mgr = FlowWarmstartManager(embedding_dim=32, theta_dim=4, n_epochs=5)
        mgr.train(model, dataset)
        count = mgr.trainable_param_count()
        assert isinstance(count, int)
        assert count > 0
        assert count == mgr.flow_model.trainable_param_count()


# ── Error Handling Tests ─────────────────────────────────────────────────


class TestFlowWarmstartErrors:
    """Verify error handling before training."""

    def test_trainable_param_count_before_train_raises(self):
        mgr = FlowWarmstartManager()
        with pytest.raises(RuntimeError, match="not been trained"):
            mgr.trainable_param_count()

    def test_sample_before_train_raises(self):
        mgr = FlowWarmstartManager()
        with pytest.raises(RuntimeError, match="not been trained"):
            mgr.sample(_make_graph())

    def test_trainable_param_count_delegation(self):
        """Manual flow_model assignment delegates correctly."""
        mgr = FlowWarmstartManager()
        flow = EmbeddingMAF(embedding_dim=64, theta_dim=4)
        mgr.flow_model = flow
        count = mgr.trainable_param_count()
        assert count == flow.trainable_param_count()
        assert count > 0
