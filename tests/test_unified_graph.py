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
    NODE_TYPE_QUBIT,
    NODE_TYPE_RX_GATE,
    NODE_TYPE_ZZ_GATE,
    BondResolvedMPNN,
    build_bond_resolved_graph,
    build_graph_for_model,
    build_unified_bond_resolved_graph,
    build_unified_dataset,
    compute_graph_metrics,
    validate_unified_graph,
)
from qmbp_simulation.predictors.unified_graph import UNIFIED_NODE_FEATURES
from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN


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
            lattice,
            h_value=1.5,
            p_layers=1,
            theta_opt=theta_opt,
            include_circuit_nodes=False,
        )

        assert unified.x.shape[0] == orig.x.shape[0] == N
        assert torch.equal(unified.edge_list, orig.edge_list)
        assert torch.allclose(unified.y, orig.y)
        assert (unified.node_type == 0).all()
        # Unified has 5 features (h, coord, N/100, type, coloring), original has 3
        assert unified.x.shape[1] == 5
        assert orig.x.shape[1] == 3
        # First 3 features must match
        assert torch.allclose(unified.x[:, :3], orig.x, atol=1e-6)

    @pytest.mark.parametrize(
        "topology,N,expected_edges",
        [
            ("chain_1d", 10, 9),
            ("chain_1d", 6, 5),
            ("ladder", 10, 13),
            ("square", 16, 24),
        ],
    )
    def test_node_counts(self, topology, N, expected_edges):
        """Unified graph has correct node counts for various topologies."""
        lattice = make_lattice(topology, N, h=1.0)
        n_edges = len(lattice.edges)
        p = 1

        graph = build_unified_bond_resolved_graph(
            lattice,
            h_value=1.0,
            p_layers=p,
            include_circuit_nodes=True,
        )

        # +1 for the virtual global node
        n_global = 1 if getattr(graph, "has_global_node", False) else 0
        expected_nodes = N + n_edges * p + N * p + n_global
        assert graph.x.shape[0] == expected_nodes
        assert graph.x.shape[1] == 5
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
            lattice,
            h_value=1.0,
            p_layers=p,
            include_circuit_nodes=True,
        )

        # p=2: N + 2*n_edges + 2*N (+1 virtual global node)
        n_global = 1 if getattr(graph, "has_global_node", False) else 0
        expected_nodes = N + n_edges * p + N * p + n_global
        assert graph.x.shape[0] == expected_nodes
        assert (graph.node_type == NODE_TYPE_ZZ_GATE).sum() == n_edges * p
        assert (graph.node_type == NODE_TYPE_RX_GATE).sum() == N * p

        # p=2 graph should have MORE edges than p=1 (inter-layer connections)
        graph_p1 = build_unified_bond_resolved_graph(
            lattice,
            h_value=1.0,
            p_layers=1,
            include_circuit_nodes=True,
        )
        assert graph.edge_index.shape[1] > graph_p1.edge_index.shape[1]

    def test_qubit_nodes_are_first(self):
        """Qubit nodes must be indices 0..N-1 (layout invariant)."""
        lattice = make_lattice("chain_1d", 10, h=1.5)
        graph = build_unified_bond_resolved_graph(
            lattice,
            h_value=1.5,
            p_layers=2,
            include_circuit_nodes=True,
        )
        # First N nodes must all be qubit type
        assert (graph.node_type[:10] == NODE_TYPE_QUBIT).all()
        # Remaining nodes must NOT be qubit type
        assert (graph.node_type[10:] != NODE_TYPE_QUBIT).all()

    def test_edge_index_bounds(self):
        """No edge index exceeds total node count."""
        lattice = make_lattice("triangular", 12, h=1.0)
        graph = build_unified_bond_resolved_graph(
            lattice,
            h_value=1.0,
            p_layers=2,
            include_circuit_nodes=True,
        )
        total_nodes = graph.x.shape[0]
        assert graph.edge_index.max().item() < total_nodes
        assert graph.edge_index.min().item() >= 0

    def test_feature_normalization_ranges(self):
        """Gate node features are normalized to [0, 1] range."""
        lattice = make_lattice("chain_1d", 10, h=2.0)
        graph = build_unified_bond_resolved_graph(
            lattice,
            h_value=2.0,
            p_layers=2,
            include_circuit_nodes=True,
        )
        # Gate nodes (type 1 and 2): first two features are normalized indices.
        # Exclude the virtual global node (type 3), whose features are raw h/coord.
        gate_mask = (graph.node_type == NODE_TYPE_ZZ_GATE) | (graph.node_type == NODE_TYPE_RX_GATE)
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
            lattice,
            h_value=1.5,
            p_layers=1,
            theta_opt=theta_opt,
            include_circuit_nodes=True,
        )

        model = BondResolvedMPNN(node_features=5, hidden_dim=64, n_layers=2)
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
        model = BondResolvedMPNN(node_features=5, hidden_dim=64, n_layers=2)
        model.eval()

        graph_ham = build_unified_bond_resolved_graph(
            lattice,
            h_value=1.5,
            p_layers=1,
            include_circuit_nodes=False,
        )
        graph_unified = build_unified_bond_resolved_graph(
            lattice,
            h_value=1.5,
            p_layers=1,
            include_circuit_nodes=True,
        )

        with torch.no_grad():
            pred_ham = model(graph_ham)
            pred_unified = model(graph_unified)

        # Predictions should DIFFER (gate nodes contribute to message passing)
        assert not torch.allclose(pred_ham, pred_unified, atol=1e-6), (
            "Gate nodes should influence predictions (different embeddings)"
        )

    @pytest.mark.parametrize("topology", ["chain_1d", "square", "ladder"])
    def test_forward_multiple_topologies(self, topology):
        """Forward pass works across different topologies."""
        N_map = {"chain_1d": 10, "square": 16, "ladder": 10}
        N = N_map[topology]
        lattice = make_lattice(topology, N, h=1.0)
        n_edges = len(lattice.edges)

        graph = build_unified_bond_resolved_graph(
            lattice,
            h_value=1.0,
            p_layers=1,
            include_circuit_nodes=True,
        )

        model = BondResolvedMPNN(node_features=5, hidden_dim=64, n_layers=2)
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
            lattice,
            h_value=1.0,
            p_layers=1,
            theta_opt=theta,
            include_circuit_nodes=True,
        )
        issues = validate_unified_graph(graph)
        assert issues == []

    def test_corrupt_edge_list_detected(self):
        """Validation catches edge_list referencing non-qubit indices."""
        lattice = make_lattice("chain_1d", 6, h=1.0)
        graph = build_unified_bond_resolved_graph(
            lattice,
            h_value=1.0,
            p_layers=1,
            include_circuit_nodes=True,
        )
        graph.edge_list = torch.tensor([[0, 20], [1, 25]], dtype=torch.long)
        issues = validate_unified_graph(graph)
        assert any("edge_list references" in i for i in issues)

    def test_wrong_target_size_detected(self):
        """Validation catches target y with wrong length."""
        lattice = make_lattice("chain_1d", 6, h=1.0)
        theta = np.random.uniform(-0.5, 0.5, 5 + 6)
        graph = build_unified_bond_resolved_graph(
            lattice,
            h_value=1.0,
            p_layers=1,
            theta_opt=theta,
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
                lattice,
                h_value=1.5,
                p_layers=1,
                theta_opt=bad_theta,
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
            lattice,
            h_value=1.0,
            p_layers=1,
            include_circuit_nodes=True,
        )
        # Corrupt edge_list to point beyond qubit nodes
        graph.edge_list = torch.tensor([[0, 99]], dtype=torch.long)

        model = BondResolvedMPNN(node_features=5, hidden_dim=64, n_layers=2)
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
            lattice,
            h_value=1.5,
            p_layers=1,
            include_circuit_nodes=True,
        )
        metrics = compute_graph_metrics(graph)

        assert metrics["n_qubit_nodes"] == 10
        assert metrics["n_zz_gates"] == 9
        assert metrics["n_rx_gates"] == 10
        assert metrics["n_gate_nodes"] == 19
        # 10 qubit + 19 gate + 1 virtual global node = 30
        n_global = 1 if getattr(graph, "has_global_node", False) else 0
        expected_total = 29 + n_global
        assert metrics["total_nodes"] == expected_total
        assert metrics["include_circuit_nodes"] is True
        np.testing.assert_allclose(metrics["node_expansion_ratio"], expected_total / 10)

    def test_metrics_hamiltonian_only(self):
        """Metrics for Hamiltonian-only graph show no expansion."""
        lattice = make_lattice("chain_1d", 10, h=1.5)
        graph = build_unified_bond_resolved_graph(
            lattice,
            h_value=1.5,
            p_layers=1,
            include_circuit_nodes=False,
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
            lattice,
            h_value=1.5,
            p_layers=1,
            include_circuit_nodes=True,
        )
        metrics = compute_graph_metrics(graph)
        # Should not raise
        json.dumps(metrics, default=json_serialize)


class TestFeatureDimAdaptivity:
    """Regression tests for graph/model feature-dim adaptivity (orbit feature).

    Prevents the class of bug where a model trained with the orbit feature
    (node_features = UNIFIED_NODE_FEATURES + 1) receives a graph built without
    it (or vice-versa), causing a size mismatch in message passing. The system
    must self-adapt via build_graph_for_model() and the forward() safety net.
    """

    def test_build_graph_for_model_matches_5feat_model(self):
        """A 5-feature model gets a 5-feature graph (orbit off)."""
        lattice = make_lattice("chain_1d", 6, h=2.5)
        model = UnifiedMPNN(node_features=UNIFIED_NODE_FEATURES, hidden_dim=32, n_layers=2)
        g = build_graph_for_model(model, lattice, h_value=2.5, p_layers=1)
        assert g.x.shape[1] == UNIFIED_NODE_FEATURES
        model.eval()
        with torch.no_grad():
            pred = model(g)
        assert torch.all(torch.isfinite(pred))

    def test_build_graph_for_model_matches_6feat_model(self):
        """A 6-feature model (orbit) gets a 6-feature graph automatically."""
        lattice = make_lattice("chain_1d", 6, h=2.5)
        model = UnifiedMPNN(node_features=UNIFIED_NODE_FEATURES + 1, hidden_dim=32, n_layers=2)
        g = build_graph_for_model(model, lattice, h_value=2.5, p_layers=1)
        assert g.x.shape[1] == UNIFIED_NODE_FEATURES + 1
        model.eval()
        with torch.no_grad():
            pred = model(g)
        assert torch.all(torch.isfinite(pred))

    def test_forward_safety_net_pads_short_graph(self):
        """A 6-feature model fed a 5-feature graph reconciles (pads), no crash."""
        lattice = make_lattice("chain_1d", 6, h=2.5)
        model = UnifiedMPNN(node_features=UNIFIED_NODE_FEATURES + 1, hidden_dim=32, n_layers=2)
        g5 = build_unified_bond_resolved_graph(lattice, h_value=2.5, p_layers=1, include_circuit_nodes=True)
        assert g5.x.shape[1] == UNIFIED_NODE_FEATURES
        model.eval()
        with torch.no_grad():
            pred = model(g5)  # must not raise despite dim mismatch
        assert torch.all(torch.isfinite(pred))

    def test_forward_safety_net_trims_long_graph(self):
        """A 5-feature model fed a 6-feature graph reconciles (trims), no crash."""
        lattice = make_lattice("chain_1d", 6, h=2.5)
        model = UnifiedMPNN(node_features=UNIFIED_NODE_FEATURES, hidden_dim=32, n_layers=2)
        g6 = build_unified_bond_resolved_graph(
            lattice,
            h_value=2.5,
            p_layers=1,
            include_circuit_nodes=True,
            include_orbit_feature=True,
        )
        assert g6.x.shape[1] == UNIFIED_NODE_FEATURES + 1
        model.eval()
        with torch.no_grad():
            pred = model(g6)  # must not raise despite dim mismatch
        assert torch.all(torch.isfinite(pred))

    def test_orbit_graph_has_extra_column(self):
        """include_orbit_feature adds exactly one node-feature column."""
        lattice = make_lattice("chain_1d", 6, h=2.5)
        g_off = build_unified_bond_resolved_graph(lattice, h_value=2.5, p_layers=1, include_circuit_nodes=True)
        g_on = build_unified_bond_resolved_graph(
            lattice,
            h_value=2.5,
            p_layers=1,
            include_circuit_nodes=True,
            include_orbit_feature=True,
        )
        assert g_on.x.shape[1] == g_off.x.shape[1] + 1


class TestLongitudinalRZExtension:
    """Bond-resolved longitudinal ansatz: RZ nodes + θ_z head (include_rz_nodes)."""

    def test_tfim_graph_unchanged_by_default(self):
        """Default include_rz_nodes=False builds an identical graph to before."""
        lattice = make_lattice("heavy_hex", 6, h=2.0)
        g_default = build_unified_bond_resolved_graph(lattice, 2.0, p_layers=1)
        g_explicit = build_unified_bond_resolved_graph(lattice, 2.0, p_layers=1, include_rz_nodes=False)
        assert g_default.x.shape[0] == g_explicit.x.shape[0]
        assert not getattr(g_default, "has_rz_nodes", False)
        assert g_default.n_rz_gates == 0
        from qmbp_simulation.predictors import NODE_TYPE_RZ_GATE

        assert (g_default.node_type == NODE_TYPE_RZ_GATE).sum().item() == 0

    def test_longitudinal_graph_node_count(self):
        """include_rz_nodes adds exactly N·p RZ gate nodes."""
        from qmbp_simulation.predictors import NODE_TYPE_RZ_GATE

        lattice = make_lattice("heavy_hex", 6, h=2.0)
        N = 6
        g_tfim = build_unified_bond_resolved_graph(lattice, 2.0, p_layers=1)
        for p in (1, 2):
            g = build_unified_bond_resolved_graph(lattice, 2.0, p_layers=p, include_rz_nodes=True)
            assert g.has_rz_nodes
            assert g.n_rz_gates == N * p
            assert (g.node_type == NODE_TYPE_RZ_GATE).sum().item() == N * p
            if p == 1:
                assert g.x.shape[0] == g_tfim.x.shape[0] + N

    def test_longitudinal_theta_shape_validation(self):
        """theta_opt must be (n_edges + 2N)·p; wrong shape raises."""
        lattice = make_lattice("heavy_hex", 6, h=2.0)
        N = 6
        n_edges = len(lattice.edges)
        good = np.random.uniform(-1, 1, (n_edges + 2 * N))
        g = build_unified_bond_resolved_graph(lattice, 2.0, p_layers=1, theta_opt=good, include_rz_nodes=True)
        assert len(g.y) == n_edges + 2 * N
        with pytest.raises(ValueError, match="shape mismatch"):
            build_unified_bond_resolved_graph(
                lattice, 2.0, p_layers=1, theta_opt=np.zeros(n_edges + N), include_rz_nodes=True
            )

    def test_longitudinal_edge_attr_consistency(self):
        """edge_attr length matches edge_index (RZ edges included)."""
        lattice = make_lattice("chain_1d", 6, h=2.0)
        g = build_unified_bond_resolved_graph(lattice, 2.0, p_layers=2, include_rz_nodes=True)
        assert g.edge_attr.shape[0] == g.edge_index.shape[1]

    def test_model_tfim_output_shape_unchanged(self):
        """predict_z=False model returns (n_edges + N)·p — no regression."""
        lattice = make_lattice("heavy_hex", 6, h=2.0)
        N = 6
        n_edges = len(lattice.edges)
        m = UnifiedMPNN(node_features=5, hidden_dim=32, n_layers=2, edge_dim=2)
        m.eval()
        g = build_unified_bond_resolved_graph(lattice, 2.0, p_layers=1)
        out = m(g)
        assert out.shape[1] == n_edges + N
        assert m.z_head is None

    def test_model_longitudinal_output_shape(self):
        """predict_z=True model on longitudinal graph returns (n_edges + 2N)·p."""
        lattice = make_lattice("heavy_hex", 6, h=2.0)
        N = 6
        n_edges = len(lattice.edges)
        m = UnifiedMPNN(node_features=5, hidden_dim=32, n_layers=2, edge_dim=2, predict_z=True)
        m.eval()
        for p in (1, 2):
            g = build_unified_bond_resolved_graph(lattice, 2.0, p_layers=p, include_rz_nodes=True)
            out = m(g)
            assert out.shape[1] == (n_edges + 2 * N) * p

    def test_model_graceful_degrade_on_tfim_graph(self):
        """A longitudinal model fed a plain TFIM graph degrades to (n_edges + N)."""
        lattice = make_lattice("heavy_hex", 6, h=2.0)
        N = 6
        n_edges = len(lattice.edges)
        m = UnifiedMPNN(node_features=5, hidden_dim=32, n_layers=2, edge_dim=2, predict_z=True)
        m.eval()
        g_tfim = build_unified_bond_resolved_graph(lattice, 2.0, p_layers=1)
        out = m(g_tfim)
        assert out.shape[1] == n_edges + N

    def test_checkpoint_roundtrip_preserves_predict_z(self, tmp_path):
        """save/load preserves predict_z and reproduces the prediction."""
        from qmbp_simulation.predictors.unified_mpnn import (
            load_unified_checkpoint,
            save_unified_checkpoint,
        )

        lattice = make_lattice("chain_1d", 6, h=2.0)
        m = UnifiedMPNN(node_features=5, hidden_dim=32, n_layers=2, edge_dim=2, predict_z=True)
        m.eval()
        g = build_unified_bond_resolved_graph(lattice, 2.0, p_layers=1, include_rz_nodes=True)
        out = m(g)
        path = str(tmp_path / "long.pt")
        save_unified_checkpoint(m, path)
        m2 = load_unified_checkpoint(path)
        assert m2.predict_z
        assert torch.allclose(out, m2(g), atol=1e-5)

    def test_tfim_checkpoint_has_no_z_head(self, tmp_path):
        """A plain TFIM checkpoint loads back without a z_head (backward compat)."""
        from qmbp_simulation.predictors.unified_mpnn import (
            load_unified_checkpoint,
            save_unified_checkpoint,
        )

        m = UnifiedMPNN(node_features=5, hidden_dim=32, n_layers=2, edge_dim=2)
        path = str(tmp_path / "tfim.pt")
        save_unified_checkpoint(m, path)
        m2 = load_unified_checkpoint(path)
        assert not m2.predict_z
        assert m2.z_head is None

    def test_build_graph_for_model_adds_rz_for_longitudinal(self):
        """build_graph_for_model auto-enables RZ nodes for a predict_z model.

        Regression: without this, a longitudinal model fed through the adaptive
        graph builder silently gets a TFIM graph and outputs θ short by the θ_z
        block. The graph feature dim must match AND the prediction must include
        the per-site Z block.
        """
        lattice = make_lattice("chain_1d", 6, h=2.5)
        N = 6
        n_edges = len(lattice.edges)
        model = UnifiedMPNN(node_features=5, hidden_dim=32, n_layers=2, edge_dim=2, predict_z=True)
        model.eval()
        g = build_graph_for_model(model, lattice, h_value=2.5, p_layers=1)
        assert getattr(g, "has_rz_nodes", False), "RZ nodes not auto-enabled"
        with torch.no_grad():
            pred = model(g)
        assert pred.shape[1] == n_edges + 2 * N

    def test_build_graph_for_model_no_rz_for_tfim(self):
        """A plain TFIM model still gets a TFIM graph (no RZ nodes)."""
        lattice = make_lattice("chain_1d", 6, h=2.5)
        model = UnifiedMPNN(node_features=5, hidden_dim=32, n_layers=2, edge_dim=2)
        g = build_graph_for_model(model, lattice, h_value=2.5, p_layers=1)
        assert not getattr(g, "has_rz_nodes", False)

    def test_build_unified_dataset_longitudinal(self):
        """build_unified_dataset validates the (n_edges + 2N) column count."""
        lattice = make_lattice("chain_1d", 4, h=2.0)
        N = 4
        n_edges = len(lattice.edges)
        h_values = np.array([1.5, 2.0, 2.5])
        theta_opts = np.random.uniform(-1, 1, (3, n_edges + 2 * N))
        ds = build_unified_dataset(lattice, h_values, theta_opts, p_layers=1, include_rz_nodes=True)
        assert len(ds) == 3
        assert all(g.has_rz_nodes for g in ds)
        with pytest.raises(ValueError, match="column count"):
            build_unified_dataset(lattice, h_values, np.zeros((3, n_edges + N)), p_layers=1, include_rz_nodes=True)

    def test_longitudinal_training_decreases_mse(self):
        """End-to-end: 3-block training reduces MSE on a small longitudinal set."""
        from qmbp_simulation.predictors.unified_mpnn import train_unified_mpnn

        lattice = make_lattice("chain_1d", 4, h=2.0)
        N = 4
        n_edges = len(lattice.edges)
        rng = np.random.default_rng(0)
        ds = [
            build_unified_bond_resolved_graph(
                lattice,
                h,
                p_layers=1,
                theta_opt=rng.uniform(-1, 1, (n_edges + 2 * N)),
                include_rz_nodes=True,
            )
            for h in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        ]
        m = UnifiedMPNN(node_features=5, hidden_dim=32, n_layers=2, edge_dim=2, predict_z=True)
        res = train_unified_mpnn(m, ds, n_epochs=150, lr=1e-2, val_fraction=0.0, fidelity_loss_weight=0.0)
        assert res["final_mse"] < res["mse_history"][0]
