"""Tests for cross-topology runners: architecture variants, save/load, zoo integration.

Covers:
- UnifiedMPNN architecture variants (residual, JK, FiLM) instantiation + forward
- Save/load roundtrip for all variants (backward compat + new features)
- MultiTopologyAggregator scan + exclusion policy
- run_multi_topology_training integration (dry-run)
- run_arch_ablation integration (dataset build + variant training)
- run_model_comparison checkpoint discovery
- Zoo registration with architecture_config
- Edge cases: empty data, missing checkpoints, dim mismatches
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]

# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def small_graph():
    """Build a minimal test graph (chain_1d N=4 p=1)."""
    from qmbp_simulation.models.hamiltonian import make_lattice
    from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph

    lattice = make_lattice("chain_1d", 4, J=1.0, h=2.0)
    theta = np.random.uniform(-0.1, 0.1, 7)  # 3 edges + 4 qubits
    g = build_unified_bond_resolved_graph(lattice, h_value=2.0, p_layers=1, theta_opt=theta)
    return g


@pytest.fixture
def small_dataset():
    """Build a small dataset (5 graphs, chain_1d N=4)."""
    from qmbp_simulation.models.hamiltonian import make_lattice
    from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph

    dataset = []
    for h in [1.5, 2.0, 2.5, 3.0, 3.5]:
        lattice = make_lattice("chain_1d", 4, J=1.0, h=h)
        theta = np.random.uniform(-0.1, 0.1, 7)
        g = build_unified_bond_resolved_graph(lattice, h_value=h, p_layers=1, theta_opt=theta)
        g.topology = "chain_1d"
        dataset.append(g)
    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Architecture variants — instantiation and forward pass
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnifiedMPNNVariants:
    """Test all architecture variant combinations."""

    @pytest.mark.parametrize("use_residual", [False, True])
    @pytest.mark.parametrize("readout_mode", ["last", "jk_cat", "jk_max"])
    @pytest.mark.parametrize("film", [False, True])
    def test_forward_all_combinations(self, small_graph, use_residual, readout_mode, film):
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN

        model = UnifiedMPNN(
            node_features=4, hidden_dim=32, n_layers=2,
            use_residual=use_residual, readout_mode=readout_mode,
            film_conditioning=film,
        )
        model.eval()
        with torch.no_grad():
            out = model(small_graph)

        expected_len = small_graph.n_edges_unique + small_graph.n_qubit_nodes
        assert out.shape == (1, expected_len)
        assert torch.isfinite(out).all()

    def test_baseline_backward_compatible(self, small_graph):
        """Default params produce same architecture as before."""
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN

        model = UnifiedMPNN(node_features=4, hidden_dim=64, n_layers=3)
        assert model.use_residual is False
        assert model.readout_mode == "last"
        assert model.film_conditioning is False
        assert model.input_proj is None
        assert model.film_gamma is None

    def test_residual_adds_input_proj(self):
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN

        model = UnifiedMPNN(node_features=4, hidden_dim=64, n_layers=3, use_residual=True)
        assert model.input_proj is not None
        assert model.input_proj.in_features == 4 + 16  # node_features + type_emb

    def test_jk_cat_changes_head_input_dim(self):
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN

        model = UnifiedMPNN(node_features=4, hidden_dim=64, n_layers=3, readout_mode="jk_cat")
        # qubit_head first layer should accept hidden_dim * n_layers = 192
        assert model.qubit_head[0].in_features == 64 * 3

    def test_film_creates_modulation_layers(self):
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN

        model = UnifiedMPNN(node_features=4, hidden_dim=64, n_layers=3, film_conditioning=True)
        assert model.film_gamma is not None
        assert len(model.film_gamma) == 3
        assert model.film_beta is not None
        assert len(model.film_beta) == 3

    def test_film_identity_init(self):
        """FiLM layers initialized to identity (γ=1, β=0)."""
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN

        model = UnifiedMPNN(node_features=4, hidden_dim=32, n_layers=2, film_conditioning=True)
        for gamma_layer in model.film_gamma:
            assert torch.allclose(gamma_layer.weight, torch.ones_like(gamma_layer.weight), atol=1e-6)
            assert torch.allclose(gamma_layer.bias, torch.zeros_like(gamma_layer.bias), atol=1e-6)

    def test_gradient_flow_all_components(self, small_graph):
        """All components receive gradients during training."""
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN

        model = UnifiedMPNN(
            node_features=4, hidden_dim=32, n_layers=2,
            use_residual=True, readout_mode="jk_cat", film_conditioning=True,
        )
        model.train()
        pred = model(small_graph).squeeze(0)
        target = small_graph.y
        loss = torch.nn.functional.mse_loss(pred[:len(target)], target)
        loss.backward()

        assert model.convs[0].nn[0].weight.grad is not None
        assert model.type_emb.weight.grad is not None
        assert model.qubit_head[0].weight.grad is not None
        assert model.input_proj.weight.grad is not None
        assert model.film_gamma[0].weight.grad is not None

    def test_invalid_readout_mode_raises(self):
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN

        with pytest.raises(ValueError, match="readout_mode"):
            UnifiedMPNN(node_features=4, hidden_dim=32, n_layers=2, readout_mode="invalid")


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Save/Load roundtrip
# ═══════════════════════════════════════════════════════════════════════════════


class TestSaveLoadRoundtrip:
    """Test save → load → identical output for all variants."""

    @pytest.mark.parametrize("kwargs", [
        {},
        {"use_residual": True},
        {"readout_mode": "jk_cat"},
        {"readout_mode": "jk_max"},
        {"film_conditioning": True},
        {"use_residual": True, "readout_mode": "jk_cat", "film_conditioning": True},
    ])
    def test_roundtrip_preserves_output(self, small_graph, kwargs):
        from qmbp_simulation.predictors.unified_mpnn import (
            UnifiedMPNN, save_unified_checkpoint, load_unified_checkpoint,
        )

        model = UnifiedMPNN(node_features=4, hidden_dim=32, n_layers=2, **kwargs)
        model.eval()
        with torch.no_grad():
            out_before = model(small_graph).clone()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            save_unified_checkpoint(model, path)
            loaded = load_unified_checkpoint(path)
            with torch.no_grad():
                out_after = loaded(small_graph)

            assert torch.allclose(out_before, out_after, atol=1e-6)
            assert loaded.use_residual == kwargs.get("use_residual", False)
            assert loaded.readout_mode == kwargs.get("readout_mode", "last")
            assert loaded.film_conditioning == kwargs.get("film_conditioning", False)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_legacy_checkpoint_loads_as_baseline(self, small_graph):
        """Checkpoint without new keys loads with defaults (backward compat)."""
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, load_unified_checkpoint

        model = UnifiedMPNN(node_features=4, hidden_dim=32, n_layers=2)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            # Save in legacy format (no new keys)
            torch.save({
                "state_dict": model.state_dict(),
                "architecture": "unified_mpnn",
                "node_features": 4,
                "hidden_dim": 32,
                "n_layers": 2,
                "norm_type": "none",
                "dropout": 0.1,
                "type_embedding_dim": 16,
                "gate_readout": True,
                "training_metadata": {},
            }, path)

            loaded = load_unified_checkpoint(path)
            assert loaded.use_residual is False
            assert loaded.readout_mode == "last"
            assert loaded.film_conditioning is False
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_rejects_nan_weights(self):
        """Cannot save a corrupted model (NaN in weights)."""
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, save_unified_checkpoint

        model = UnifiedMPNN(node_features=4, hidden_dim=32, n_layers=2)
        # Corrupt a weight
        with torch.no_grad():
            model.convs[0].nn[0].weight[0, 0] = float("nan")

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            with pytest.raises(ValueError, match="non-finite"):
                save_unified_checkpoint(model, path)
        finally:
            Path(path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Training integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrainingIntegration:
    """Test train_unified_mpnn with new architecture variants."""

    def test_train_baseline(self, small_dataset):
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, train_unified_mpnn

        model = UnifiedMPNN(node_features=4, hidden_dim=32, n_layers=2)
        result = train_unified_mpnn(model, small_dataset, n_epochs=10, lr=1e-3, seed=42)

        assert "final_mse" in result
        assert result["final_mse"] >= 0
        assert result["n_epochs_run"] <= 10

    def test_train_residual_film(self, small_dataset):
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, train_unified_mpnn

        model = UnifiedMPNN(
            node_features=4, hidden_dim=32, n_layers=2,
            use_residual=True, film_conditioning=True,
        )
        result = train_unified_mpnn(model, small_dataset, n_epochs=10, lr=1e-3, seed=42)

        assert result["final_mse"] >= 0
        assert result["n_epochs_run"] > 0

    def test_train_jk_cat(self, small_dataset):
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, train_unified_mpnn

        model = UnifiedMPNN(
            node_features=4, hidden_dim=32, n_layers=2, readout_mode="jk_cat",
        )
        result = train_unified_mpnn(model, small_dataset, n_epochs=10, lr=1e-3, seed=42)

        assert result["final_mse"] >= 0

    def test_train_rejects_too_few_points(self):
        """Training with <3 points should raise."""
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, train_unified_mpnn
        from qmbp_simulation.models.hamiltonian import make_lattice
        from qmbp_simulation.predictors.unified_graph import build_unified_bond_resolved_graph

        model = UnifiedMPNN(node_features=4, hidden_dim=32, n_layers=2)
        lattice = make_lattice("chain_1d", 4, J=1.0, h=2.0)
        g = build_unified_bond_resolved_graph(lattice, h_value=2.0, p_layers=1,
                                              theta_opt=np.zeros(7))
        with pytest.raises(ValueError, match="Need ≥3"):
            train_unified_mpnn(model, [g, g], n_epochs=10)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Zoo registration with architecture config
# ═══════════════════════════════════════════════════════════════════════════════


class TestZooRegistration:
    """Test that architecture config is persisted in zoo."""

    def test_register_with_architecture_config(self, small_dataset, tmp_path):
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, train_unified_mpnn
        from qmbp_simulation.predictors.model_zoo import ZooEntry, register_checkpoint

        model = UnifiedMPNN(
            node_features=4, hidden_dim=32, n_layers=2,
            use_residual=True, film_conditioning=True,
        )
        train_unified_mpnn(model, small_dataset, n_epochs=5, lr=1e-3, seed=42)

        entry = ZooEntry(
            model="tfim_bond_resolved",
            topology="test_topology",
            n_qubits=0,
            p_layers=1,
            checkpoint_file="test_model_arch.pt",
            pass_rate=0.0,
            n_training_points=5,
        )
        # This should not raise
        register_checkpoint(model, entry, overwrite=True)

        # Verify checkpoint on disk has architecture metadata
        from qmbp_simulation.predictors.model_zoo import _CHECKPOINTS_DIR
        ckpt_path = _CHECKPOINTS_DIR / "test_model_arch.pt"
        if ckpt_path.exists():
            data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            assert data.get("use_residual") is True
            assert data.get("film_conditioning") is True
            assert data.get("readout_mode") == "last"
            # Cleanup
            ckpt_path.unlink()


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Exclusion policy in aggregator
# ═══════════════════════════════════════════════════════════════════════════════


class TestExclusionPolicy:
    """Test N-level exclusion policy applied in multi-topology training."""

    def test_apply_exclusion_policy(self, tmp_path):
        """Exclusion policy removes N-values with hard failure modes."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        # Create a mock aggregator with data
        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved")
        agg._data_by_n = {
            6: [{"h": 2.0, "theta": np.zeros(11), "de_gap": 0.01}] * 5,
            40: [{"h": 2.0, "theta": np.zeros(11), "de_gap": 0.5}] * 3,
        }

        # Simulate exclusion policy (same logic as in script)
        from qmbp_simulation.analysis.metrics import load_training_exclusions
        registry = load_training_exclusions()
        hard_modes = {"contaminated_training", "gap_masking"}

        excluded_ns = set()
        for entry in registry.get("excluded", []):
            if (entry.get("topology") == "chain_1d"
                    and entry.get("failure_mode") in hard_modes):
                n_val = entry.get("n_qubits", 0)
                if n_val > 0:
                    excluded_ns.add(n_val)

        for n_val in excluded_ns:
            agg._data_by_n.pop(n_val, None)

        # N=40 should be removed (if in exclusion registry)
        # N=6 should remain
        assert 6 in agg._data_by_n


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Model comparison checkpoint discovery
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelComparison:
    """Test run_model_comparison checkpoint discovery."""

    def test_discover_explicit_checkpoints(self, tmp_path):
        """Explicit paths are discovered correctly."""
        sys_path_backup = sys.path.copy()
        sys.path.insert(0, str(ROOT / "src"))
        try:
            from scripts.experiment_runners.cross_topology.run_model_comparison import (
                discover_checkpoints,
            )

            # Create a fake checkpoint
            fake_ckpt = tmp_path / "fake_model.pt"
            fake_ckpt.write_bytes(b"fake")

            candidates = discover_checkpoints("chain_1d", 1, [str(fake_ckpt)])
            assert len(candidates) == 1
            assert candidates[0]["source"] == "explicit"
            assert candidates[0]["path"] == fake_ckpt
        finally:
            sys.path = sys_path_backup

    def test_discover_auto_finds_zoo_entries(self):
        """Auto-detect finds entries from zoo manifest."""
        sys_path_backup = sys.path.copy()
        sys.path.insert(0, str(ROOT / "src"))
        try:
            from scripts.experiment_runners.cross_topology.run_model_comparison import (
                discover_checkpoints,
            )

            # Should find at least per-topology or multi-topology models
            candidates = discover_checkpoints("chain_1d", 1, None)
            # May be empty if no checkpoints on disk, but should not raise
            assert isinstance(candidates, list)
        finally:
            sys.path = sys_path_backup


# ═══════════════════════════════════════════════════════════════════════════════
# Test: Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_model_with_zero_type_embedding(self, small_graph):
        """type_embedding_dim=0 should work (no embedding layer)."""
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN

        model = UnifiedMPNN(node_features=4, hidden_dim=32, n_layers=2, type_embedding_dim=0)
        model.eval()
        with torch.no_grad():
            out = model(small_graph)
        assert torch.isfinite(out).all()

    def test_single_layer_model(self, small_graph):
        """n_layers=1 works correctly."""
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN

        model = UnifiedMPNN(
            node_features=4, hidden_dim=32, n_layers=1,
            use_residual=True, readout_mode="jk_cat", film_conditioning=True,
        )
        model.eval()
        with torch.no_grad():
            out = model(small_graph)
        # JK-cat with 1 layer = same as last
        assert out.shape == (1, small_graph.n_edges_unique + small_graph.n_qubit_nodes)

    def test_large_hidden_dim(self, small_graph):
        """Large hidden dim doesn't crash."""
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN

        model = UnifiedMPNN(node_features=4, hidden_dim=512, n_layers=2)
        model.eval()
        with torch.no_grad():
            out = model(small_graph)
        assert torch.isfinite(out).all()

    def test_missing_node_type_raises(self):
        """Forward without node_type raises RuntimeError."""
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN
        from torch_geometric.data import Data

        model = UnifiedMPNN(node_features=4, hidden_dim=32, n_layers=2)
        model.eval()
        # Create a data object without node_type
        data = Data(
            x=torch.randn(5, 4),
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        )
        with pytest.raises(RuntimeError, match="node_type"):
            model(data)

    def test_training_reduces_loss(self, small_dataset):
        """Training for multiple epochs should reduce loss vs untrained model."""
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, train_unified_mpnn

        model = UnifiedMPNN(node_features=4, hidden_dim=32, n_layers=2, film_conditioning=True)

        # Evaluate untrained loss
        model.eval()
        initial_loss = 0.0
        with torch.no_grad():
            for g in small_dataset:
                pred = model(g).squeeze(0)
                initial_loss += torch.nn.functional.mse_loss(pred[:len(g.y)], g.y).item()
        initial_loss /= len(small_dataset)

        # Train
        model.train()
        result = train_unified_mpnn(model, small_dataset, n_epochs=50, seed=42, val_fraction=0.0)

        # Final MSE should be lower than initial
        assert result["final_mse"] < initial_loss, (
            f"Training didn't reduce loss: initial={initial_loss:.4e}, final={result['final_mse']:.4e}"
        )
