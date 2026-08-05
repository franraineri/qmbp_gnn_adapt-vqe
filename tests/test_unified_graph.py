"""Tests for unified Hamiltonian+Circuit graph builder and BondResolvedMPNN integration.

Covers:
- Backward compatibility with Hamiltonian-only graphs
- Unified graph structure correctness (node counts, edge connectivity)
- BondResolvedMPNN forward pass with node_type masking
- Input validation and error prevention
- Graph metrics computation
"""

import numpy as np
import pytest
import torch

from qmbp_simulation import make_lattice
from qmbp_simulation.predictors import (
    BondResolvedMPNN,
    build_bond_resolved_graph,
    build_unified_bond_resolved_graph,
    build_unified_dataset,
    compute_graph_metrics,
    validate_unified_graph,
    NODE_TYPE_QUBIT,
    NODE_TYPE_ZZ_GATE,
    NODE_TYPE_RX_GATE,
)


class TestUnifiedGraphBuilder:
    """Tests for build_unified_bond_resolved_graph()."""

    def test_backward_compat_features(self):
        """include_circuit_nodes=False produces same graph as original builder."""
        lattice = make_lattice("chain_1d", 10, h=1.5)
        n_edges = len(lattice.edges)
        N = 10

        theta_opt = np.random.uniform(-0.5, 0.5, n_edges + N)
        orig = build_bond_resolved_graph(lattice, h_value=1.5, theta_opt=theta_opt)
        unified = build_unified_bond_resolved_graph(
            lattice, h_value=1.5, p_layers=1, theta_opt=theta_opt,
            include_circuit_nodes=False,
        )

        assert unified.x.shape[0] == orig.x.shape[0] == N
        assert torch.equal(unified.edge_list, orig.edge_list)
        assert torch.allclose(unified.y, orig.y)
        assert (unified.node_type == 0).all()
        # Unified has 4 features (with type column), original has 3
        assert unified.x.shape[1] == 4
        assert orig.x.shape[1] == 3
        # First 3 features must match
        assert torch.allclose(unified.x[:, :3], orig.x, atol=1e-6)

    @pytest.mark.parametrize("topology,N,expected_edges", [
        ("chain_1d", 10, 9),
        ("chain_1d", 6, 5),
        ("ladder", 10, 13),
        ("square", 16, 24),
    ])
    def test_node_counts(self, topology, N, expected_edges):
        """Unified graph has correct node counts for various topologies."""
        lattice = make_lattice(topology, N, h=1.0)
        n_edges = len(lattice.edges)
        p = 1

        graph = build_unified_bond_resolved_graph(
            lattice, h_value=1.0, p_layers=p, include_circuit_nodes=True,
        )

        expected_nodes = N + n_edges * p + N * p
        assert graph.x.shape[0] == expected_nodes
        assert graph.x.shape[1] == 4
        assert (graph.node_type == NODE_TYPE_QUBIT).sum() == N
        assert (graph.node_type == NODE_TYPE_ZZ_GATE).sum() == n_edges * p
        assert (graph.node_type == NODE_TYPE_RX_GATE).sum() == N * p
        assert graph.n_qubit_nodes == N
        assert graph.n_edges_unique == n_edges

    def test_p2_graph_structure(self):
        """p=2 doubles gate nodes and adds inter-layer edges."""
        lattice = make_lattice("chain_1d", 6, h=1.0)
        N = 6
        n_edges = 5
        p = 2

        graph = build_unified_bond_resolved_graph(
            lattice, h_value=1.0, p_layers=p, include_circuit_nodes=True,
        )

        # p=2: N + 2*n_edges + 2*N = 6 + 10 + 12 = 28
        expected_nodes = N + n_edges * p + N * p
        assert graph.x.shape[0] == expected_nodes
        assert (graph.node_type == NODE_TYPE_ZZ_GATE).sum() == n_edges * p
        assert (graph.node_type == NODE_TYPE_RX_GATE).sum() == N * p

        # p=2 graph should have MORE edges than p=1 (inter-layer connections)
        graph_p1 = build_unified_bond_resolved_graph(
            lattice, h_value=1.0, p_layers=1, include_circuit_nodes=True,
        )
        assert graph.edge_index.shape[1] > graph_p1.edge_index.shape[1]

    def test_qubit_nodes_are_first(self):
        """Qubit nodes must be indices 0..N-1 (layout invariant)."""
        lattice = make_lattice("chain_1d", 10, h=1.5)
        graph = build_unified_bond_resolved_graph(
            lattice, h_value=1.5, p_layers=2, include_circuit_nodes=True,
        )
        # First N nodes must all be qubit type
        assert (graph.node_type[:10] == NODE_TYPE_QUBIT).all()
        # Remaining nodes must NOT be qubit type
        assert (graph.node_type[10:] != NODE_TYPE_QUBIT).all()

    def test_edge_index_bounds(self):
        """No edge index exceeds total node count."""
        lattice = make_lattice("triangular", 12, h=1.0)
        graph = build_unified_bond_resolved_graph(
            lattice, h_value=1.0, p_layers=2, include_circuit_nodes=True,
        )
        total_nodes = graph.x.shape[0]
        assert graph.edge_index.max().item() < total_nodes
        assert graph.edge_index.min().item() >= 0

    def test_feature_normalization_ranges(self):
        """Gate node features are normalized to [0, 1] range."""
        lattice = make_lattice("chain_1d", 10, h=2.0)
        graph = build_unified_bond_resolved_graph(
            lattice, h_value=2.0, p_layers=2, include_circuit_nodes=True,
        )
        # Gate nodes (type 1 and 2): first two features are normalized indices
        gate_mask = graph.node_type != NODE_TYPE_QUBIT
        gate_features = graph.x[gate_mask]
        # feat1 (layer_norm) and feat2 (bond/qubit norm) should be in (0, 1)
        assert gate_features[:, 0].min() > 0
        assert gate_features[:, 0].max() < 1
        assert gate_features[:, 1].min() > 0
        assert gate_features[:, 1].max() < 1


class TestBondResolvedMPNNUnified:
    """Tests for BondResolvedMPNN with unified graph inputs."""

    def test_forward_unified_graph(self):
        """Forward pass with unified graph produces correct output shape."""
        lattice = make_lattice("chain_1d", 10, h=1.5)
        N = 10
        n_edges = 9

        theta_opt = np.random.uniform(-0.5, 0.5, n_edges + N)
        graph = build_unified_bond_resolved_graph(
            lattice, h_value=1.5, p_layers=1, theta_opt=theta_opt,
            include_circuit_nodes=True,
        )

        model = BondResolvedMPNN(node_features=4, hidden_dim=64, n_layers=2)
        model.eval()
        with torch.no_grad():
            pred = model(graph)

        # Output: [1, n_edges + N] regardless of gate node count
        assert pred.shape == (1, n_edges + N)

    def test_forward_backward_compat(self):
        """Forward pass without node_type still works (original graphs)."""
        lattice = make_lattice("chain_1d", 10, h=1.5)
        N = 10
        n_edges = 9

        theta_opt = np.random.uniform(-0.5, 0.5, n_edges + N)
        graph = build_bond_resolved_graph(lattice, h_value=1.5, theta_opt=theta_opt)

        model = BondResolvedMPNN(node_features=3, hidden_dim=64, n_layers=2)
        model.eval()
        with torch.no_grad():
            pred = model(graph)

        assert pred.shape == (1, n_edges + N)

    def test_gate_nodes_improve_embeddings(self):
        """Gate nodes influence qubit embeddings via message passing."""
        lattice = make_lattice("chain_1d", 6, h=1.5)
        N = 6
        n_edges = 5

        # Same model, same weights — compare predictions with/without circuit nodes
        model = BondResolvedMPNN(node_features=4, hidden_dim=64, n_layers=2)
        model.eval()

        graph_ham = build_unified_bond_resolved_graph(
            lattice, h_value=1.5, p_layers=1, include_circuit_nodes=False,
        )
        graph_unified = build_unified_bond_resolved_graph(
            lattice, h_value=1.5, p_layers=1, include_circuit_nodes=True,
        )

        with torch.no_grad():
            pred_ham = model(graph_ham)
            pred_unified = model(graph_unified)

        # Predictions should DIFFER (gate nodes contribute to message passing)
        assert not torch.allclose(pred_ham, pred_unified, atol=1e-6), \
            "Gate nodes should influence predictions (different embeddings)"

    @pytest.mark.parametrize("topology", ["chain_1d", "square", "ladder"])
    def test_forward_multiple_topologies(self, topology):
        """Forward pass works across different topologies."""
        N_map = {"chain_1d": 10, "square": 16, "ladder": 10}
        N = N_map[topology]
        lattice = make_lattice(topology, N, h=1.0)
        n_edges = len(lattice.edges)

        graph = build_unified_bond_resolved_graph(
            lattice, h_value=1.0, p_layers=1, include_circuit_nodes=True,
        )

        model = BondResolvedMPNN(node_features=4, hidden_dim=64, n_layers=2)
        model.eval()
        with torch.no_grad():
            pred = model(graph)

        assert pred.shape == (1, n_edges + N)
        assert torch.all(torch.isfinite(pred)), "Predictions must be finite"


class TestValidation:
    """Tests for graph validation and error prevention."""

    def test_valid_graph_no_issues(self):
        """Well-formed graph passes validation."""
        lattice = make_lattice("chain_1d", 6, h=1.0)
        theta = np.random.uniform(-0.5, 0.5, 5 + 6)
        graph = build_unified_bond_resolved_graph(
            lattice, h_value=1.0, p_layers=1, theta_opt=theta,
            include_circuit_nodes=True,
        )
        issues = validate_unified_graph(graph)
        assert issues == []

    def test_corrupt_edge_list_detected(self):
        """Validation catches edge_list referencing non-qubit indices."""
        lattice = make_lattice("chain_1d", 6, h=1.0)
        graph = build_unified_bond_resolved_graph(
            lattice, h_value=1.0, p_layers=1, include_circuit_nodes=True,
        )
        graph.edge_list = torch.tensor([[0, 20], [1, 25]], dtype=torch.long)
        issues = validate_unified_graph(graph)
        assert any("edge_list references" in i for i in issues)

    def test_wrong_target_size_detected(self):
        """Validation catches target y with wrong length."""
        lattice = make_lattice("chain_1d", 6, h=1.0)
        theta = np.random.uniform(-0.5, 0.5, 5 + 6)
        graph = build_unified_bond_resolved_graph(
            lattice, h_value=1.0, p_layers=1, theta_opt=theta,
            include_circuit_nodes=True,
        )
        # Corrupt target length
        graph.y = torch.zeros(3)
        issues = validate_unified_graph(graph)
        assert any("Target y" in i for i in issues)

    def test_input_validation_p_layers(self):
        """ValueError raised for invalid p_layers."""
        lattice = make_lattice("chain_1d", 10, h=1.5)
        with pytest.raises(ValueError, match="p_layers"):
            build_unified_bond_resolved_graph(lattice, h_value=1.5, p_layers=0)

    def test_input_validation_h_nan(self):
        """ValueError raised for NaN h_value."""
        lattice = make_lattice("chain_1d", 10, h=1.5)
        with pytest.raises(ValueError, match="h_value"):
            build_unified_bond_resolved_graph(lattice, h_value=float("nan"))

    def test_input_validation_theta_shape(self):
        """ValueError raised for wrong theta_opt shape."""
        lattice = make_lattice("chain_1d", 10, h=1.5)
        bad_theta = np.zeros(5)
        with pytest.raises(ValueError, match="theta_opt shape"):
            build_unified_bond_resolved_graph(
                lattice, h_value=1.5, p_layers=1, theta_opt=bad_theta,
            )

    def test_dataset_builder_shape_mismatch(self):
        """ValueError raised when theta_opts columns don't match expected params."""
        lattice = make_lattice("chain_1d", 6, h=1.0)
        h_values = np.array([2.0, 1.5, 1.0])
        bad_thetas = np.random.uniform(-0.5, 0.5, (3, 5))  # wrong cols
        with pytest.raises(ValueError, match="column count"):
            build_unified_dataset(lattice, h_values, bad_thetas, p_layers=1)

    def test_runtime_error_on_corrupt_forward(self):
        """RuntimeError raised if edge_list indices exceed qubit embedding size."""
        lattice = make_lattice("chain_1d", 6, h=1.0)
        graph = build_unified_bond_resolved_graph(
            lattice, h_value=1.0, p_layers=1, include_circuit_nodes=True,
        )
        # Corrupt edge_list to point beyond qubit nodes
        graph.edge_list = torch.tensor([[0, 99]], dtype=torch.long)

        model = BondResolvedMPNN(node_features=4, hidden_dim=64, n_layers=2)
        model.eval()
        with pytest.raises(RuntimeError, match="edge_list contains index"):
            with torch.no_grad():
                model(graph)


class TestGraphMetrics:
    """Tests for compute_graph_metrics()."""

    def test_metrics_unified(self):
        """Metrics are correct for unified graph."""
        lattice = make_lattice("chain_1d", 10, h=1.5)
        graph = build_unified_bond_resolved_graph(
            lattice, h_value=1.5, p_layers=1, include_circuit_nodes=True,
        )
        metrics = compute_graph_metrics(graph)

        assert metrics["n_qubit_nodes"] == 10
        assert metrics["n_zz_gates"] == 9
        assert metrics["n_rx_gates"] == 10
        assert metrics["n_gate_nodes"] == 19
        assert metrics["total_nodes"] == 29
        assert metrics["include_circuit_nodes"] is True
        np.testing.assert_allclose(metrics["node_expansion_ratio"], 29 / 10)

    def test_metrics_hamiltonian_only(self):
        """Metrics for Hamiltonian-only graph show no expansion."""
        lattice = make_lattice("chain_1d", 10, h=1.5)
        graph = build_unified_bond_resolved_graph(
            lattice, h_value=1.5, p_layers=1, include_circuit_nodes=False,
        )
        metrics = compute_graph_metrics(graph)

        assert metrics["n_gate_nodes"] == 0
        assert metrics["include_circuit_nodes"] is False
        np.testing.assert_allclose(metrics["node_expansion_ratio"], 1.0)

    def test_metrics_serializable(self):
        """Metrics dict is JSON-serializable (for result envelopes)."""
        import json
        from qmbp_simulation.utils.helpers import json_serialize

        lattice = make_lattice("chain_1d", 10, h=1.5)
        graph = build_unified_bond_resolved_graph(
            lattice, h_value=1.5, p_layers=1, include_circuit_nodes=True,
        )
        metrics = compute_graph_metrics(graph)
        # Should not raise
        json.dumps(metrics, default=json_serialize)
