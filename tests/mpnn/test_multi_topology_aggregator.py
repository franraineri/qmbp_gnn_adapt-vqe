"""Tests for MultiTopologyAggregator and multi-topology training pipeline.

Validates:
1. MultiTopologyAggregator scans all topologies correctly
2. Quality filtering works across topologies
3. Feature dimension consistency is enforced
4. Dataset tagging (topology attribute on graphs)
5. Integration with existing MultiNAggregator (reuse, not duplication)
6. No data loss: all high-quality points are included
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def fake_npz_dir(tmp_path, monkeypatch):
    """Create fake multi_n_training NPZ files for testing."""
    from qmbp_simulation.predictors import multi_n_aggregator

    npz_dir = tmp_path / "data" / "multi_n_training"
    npz_dir.mkdir(parents=True)
    monkeypatch.setattr(multi_n_aggregator, "_PROJECT_ROOT", tmp_path)

    # Create fake NPZ for chain_1d N=4
    n_pts = 8
    np.savez(
        npz_dir / "chain_1d_N4_p1.npz",
        h_values=np.linspace(2.0, 4.0, n_pts),
        theta_opt=np.random.randn(n_pts, 7).astype(np.float64),
        e_vqe=np.linspace(-4.0, -3.0, n_pts),
        e_exact=np.linspace(-4.01, -3.01, n_pts),
        gaps=np.ones(n_pts) * 2.0,
        quality_tier=np.array(["verified"] * 6 + ["approximate"] * 2),
        de_gaps=np.full(n_pts, 0.005),
    )

    # Create fake NPZ for ladder N=4 (needs 8 params: 4 edges + 4 qubits)
    np.savez(
        npz_dir / "ladder_N4_p1.npz",
        h_values=np.linspace(2.0, 4.0, n_pts),
        theta_opt=np.random.randn(n_pts, 8).astype(np.float64),
        e_vqe=np.linspace(-5.0, -4.0, n_pts),
        e_exact=np.linspace(-5.01, -4.01, n_pts),
        gaps=np.ones(n_pts) * 1.8,
        quality_tier=np.array(["verified"] * 5 + ["unverified"] * 3),
        de_gaps=np.full(n_pts, 0.006),
    )

    # Create fake NPZ for square N=4 (low quality — should be filtered)
    # N=4 square: 4 edges + 4 qubits = 8 params
    np.savez(
        npz_dir / "square_N4_p1.npz",
        h_values=np.linspace(2.0, 4.0, n_pts),
        theta_opt=np.random.randn(n_pts, 8).astype(np.float64),
        e_vqe=np.linspace(-6.0, -5.0, n_pts),
        e_exact=np.linspace(-7.0, -6.0, n_pts),
        gaps=np.ones(n_pts) * 0.5,
        quality_tier=np.array(["unverified"] * n_pts),
        de_gaps=np.full(n_pts, 2.0),  # All failing
    )

    return tmp_path


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: MultiTopologyAggregator
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiTopologyAggregator:
    """Tests for the multi-topology data aggregation."""

    def test_auto_detects_topologies(self, fake_npz_dir, monkeypatch):
        """Should find all topologies that have NPZ files."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiTopologyAggregator

        # Monkeypatch exclusion registry to return empty (no exclusions in test)
        from qmbp_simulation.analysis import metrics
        monkeypatch.setattr(metrics, "_EXCLUSION_REGISTRY_PATH", fake_npz_dir / "excl.json")

        agg = MultiTopologyAggregator(min_verified_points=1)
        assert set(agg.topologies) == {"chain_1d", "ladder", "square"}

    def test_scan_returns_per_topology_summary(self, fake_npz_dir, monkeypatch):
        """scan() returns {topology: {N: n_points}}."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiTopologyAggregator
        from qmbp_simulation.analysis import metrics
        monkeypatch.setattr(metrics, "_EXCLUSION_REGISTRY_PATH", fake_npz_dir / "excl.json")

        agg = MultiTopologyAggregator(min_verified_points=1)
        summary = agg.scan()

        assert "chain_1d" in summary
        assert "ladder" in summary
        assert 4 in summary["chain_1d"]  # N=4 was created

    def test_build_dataset_filters_by_quality(self, fake_npz_dir, monkeypatch):
        """Low-quality topologies (all unverified failing) should be excluded."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiTopologyAggregator
        from qmbp_simulation.analysis import metrics
        monkeypatch.setattr(metrics, "_EXCLUSION_REGISTRY_PATH", fake_npz_dir / "excl.json")

        agg = MultiTopologyAggregator(min_verified_points=5)
        agg.scan()
        dataset = agg.build_combined_dataset(max_de_gap=0.10)

        # chain_1d has 6 verified → included
        # ladder has 5 verified → included (at boundary)
        # square has 0 verified, all de_gap=2.0 → excluded
        topos_in_dataset = set(g.topology for g in dataset)
        assert "chain_1d" in topos_in_dataset
        assert "ladder" in topos_in_dataset
        assert "square" not in topos_in_dataset

    def test_feature_dimensions_consistent(self, fake_npz_dir, monkeypatch):
        """All graphs must have same feature dimension regardless of topology."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiTopologyAggregator
        from qmbp_simulation.analysis import metrics
        monkeypatch.setattr(metrics, "_EXCLUSION_REGISTRY_PATH", fake_npz_dir / "excl.json")

        agg = MultiTopologyAggregator(min_verified_points=1)
        agg.scan()
        dataset = agg.build_combined_dataset(max_de_gap=0.10)

        if len(dataset) > 1:
            dims = set(g.x.shape[1] for g in dataset)
            assert len(dims) == 1, f"Inconsistent dims: {dims}"

    def test_graphs_tagged_with_topology(self, fake_npz_dir, monkeypatch):
        """Each graph should have a .topology attribute for traceability."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiTopologyAggregator
        from qmbp_simulation.analysis import metrics
        monkeypatch.setattr(metrics, "_EXCLUSION_REGISTRY_PATH", fake_npz_dir / "excl.json")

        agg = MultiTopologyAggregator(min_verified_points=1)
        agg.scan()
        dataset = agg.build_combined_dataset(max_de_gap=0.10)

        for g in dataset:
            assert hasattr(g, "topology"), "Graph missing topology tag"
            assert g.topology in ("chain_1d", "ladder", "square")

    def test_explicit_topology_filter(self, fake_npz_dir, monkeypatch):
        """When topologies parameter is set, only those are included."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiTopologyAggregator
        from qmbp_simulation.analysis import metrics
        monkeypatch.setattr(metrics, "_EXCLUSION_REGISTRY_PATH", fake_npz_dir / "excl.json")

        agg = MultiTopologyAggregator(
            topologies=["chain_1d"],
            min_verified_points=1,
        )
        agg.scan()
        dataset = agg.build_combined_dataset(max_de_gap=0.10)

        topos = set(g.topology for g in dataset)
        assert topos == {"chain_1d"}

    def test_no_data_loss_verified_points(self, fake_npz_dir, monkeypatch):
        """All verified points that pass quality filter must be in dataset."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiTopologyAggregator
        from qmbp_simulation.analysis import metrics
        monkeypatch.setattr(metrics, "_EXCLUSION_REGISTRY_PATH", fake_npz_dir / "excl.json")

        agg = MultiTopologyAggregator(min_verified_points=1)
        agg.scan()
        dataset = agg.build_combined_dataset(max_de_gap=0.10)

        # chain_1d had 6 verified + 2 approximate = 8 pts, all with de_gap=0.005
        # All should pass the 0.10 threshold
        chain_graphs = [g for g in dataset if g.topology == "chain_1d"]
        # At minimum: 6 verified + 2 approximate (may have augmentation too)
        assert len(chain_graphs) >= 8, f"Expected ≥8, got {len(chain_graphs)}"

    def test_summary_method(self, fake_npz_dir, monkeypatch):
        """summary() returns structured info."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiTopologyAggregator
        from qmbp_simulation.analysis import metrics
        monkeypatch.setattr(metrics, "_EXCLUSION_REGISTRY_PATH", fake_npz_dir / "excl.json")

        agg = MultiTopologyAggregator(min_verified_points=1)
        agg.scan()
        s = agg.summary()

        assert "topologies" in s
        assert "per_topology" in s
        assert "total_points" in s
        assert s["total_points"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Integration with train_unified_mpnn
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultiTopologyTrainingIntegration:
    """Integration tests: aggregation → model creation → training loop."""

    def test_model_trains_on_multi_topology_data(self, fake_npz_dir, monkeypatch):
        """UnifiedMPNN can train on combined multi-topology dataset."""
        import torch
        from qmbp_simulation.predictors.multi_n_aggregator import MultiTopologyAggregator
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, train_unified_mpnn
        from qmbp_simulation.analysis import metrics
        monkeypatch.setattr(metrics, "_EXCLUSION_REGISTRY_PATH", fake_npz_dir / "excl.json")

        agg = MultiTopologyAggregator(min_verified_points=1)
        agg.scan()
        dataset = agg.build_combined_dataset(max_de_gap=0.10)

        assert len(dataset) >= 3, "Need ≥3 graphs for training"

        feat_dim = dataset[0].x.shape[1]
        model = UnifiedMPNN(
            node_features=feat_dim,
            hidden_dim=32,  # Small for test speed
            n_layers=2,
            dropout=0.0,
            type_embedding_dim=8,
        )

        # Train for just a few epochs (verify no crash)
        result = train_unified_mpnn(
            model, dataset,
            n_epochs=10,
            lr=1e-3,
            patience=5,
            seed=42,
            val_fraction=0.0,  # No val for tiny dataset
        )

        assert result["final_mse"] < float("inf")
        assert result["n_epochs_run"] == 10
        assert result["stop_reason"] == "completed"

    def test_model_produces_valid_output_per_topology(self, fake_npz_dir, monkeypatch):
        """Trained model produces correct output shape for each topology's graphs."""
        import torch
        from qmbp_simulation.predictors.multi_n_aggregator import MultiTopologyAggregator
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, train_unified_mpnn
        from qmbp_simulation.analysis import metrics
        monkeypatch.setattr(metrics, "_EXCLUSION_REGISTRY_PATH", fake_npz_dir / "excl.json")

        agg = MultiTopologyAggregator(min_verified_points=1)
        agg.scan()
        dataset = agg.build_combined_dataset(max_de_gap=0.10)

        feat_dim = dataset[0].x.shape[1]
        model = UnifiedMPNN(
            node_features=feat_dim,
            hidden_dim=32,
            n_layers=2,
            type_embedding_dim=8,
        )

        train_unified_mpnn(model, dataset, n_epochs=5, val_fraction=0.0)

        model.eval()
        with torch.no_grad():
            for g in dataset[:5]:
                out = model(g)
                # Output should match y shape
                assert out.shape[-1] == g.y.shape[0], (
                    f"Output {out.shape} != target {g.y.shape} for {g.topology}"
                )
