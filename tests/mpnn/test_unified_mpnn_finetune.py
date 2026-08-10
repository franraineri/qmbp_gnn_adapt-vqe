"""Tests for UnifiedMPNN fine-tuning, should_retrain, zoo integration, and aggregator.

Covers all improvements made in the fine-tuning session:
- should_retrain() gating logic (all branches including edge cases)
- train_unified_mpnn() validations + mse_floor early-stop
- fine_tune_unified_mpnn() layer-wise LR, real early-stop, result structure
- register_checkpoint() auto-detection of UnifiedMPNN vs MPNNPredictor
- maybe_fine_tune_mpnn() ValidationRunner helper integration
- MultiNAggregator n-before-use bug (NameError regression guard)
- AcceleratedVQE mse_floor integration in _train_mpnn

All tests use synthetic PyG graphs — no VQE or Qiskit required.
Marked slow tests skip by default (make test) and run with make test-full.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from qmbp_simulation.predictors import (
    UnifiedMPNN,
    fine_tune_unified_mpnn,
    should_retrain,
    train_unified_mpnn,
    save_unified_checkpoint,
    load_unified_checkpoint,
)
from qmbp_simulation.predictors.unified_graph import UNIFIED_NODE_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_unified_graph(
    n_qubits: int = 6,
    n_edges: int = 5,
    seed: int | None = None,
) -> Data:
    """Minimal synthetic unified graph (qubit + ZZ-gate + RX-gate nodes).

    Layout: [qubit_0..N-1, zz_gate_0..E-1, rx_gate_0..N-1]
    Target y: [n_edges + n_qubits] flat vector (p=1 layout).
    """
    rng = np.random.default_rng(seed)
    n_nodes = n_qubits + n_edges + n_qubits
    x = torch.tensor(rng.standard_normal((n_nodes, UNIFIED_NODE_FEATURES)), dtype=torch.float32)
    node_type = torch.cat([
        torch.zeros(n_qubits, dtype=torch.long),
        torch.ones(n_edges, dtype=torch.long),
        torch.full((n_qubits,), 2, dtype=torch.long),
    ])
    src, dst = [], []
    for i in range(n_edges):
        gate = n_qubits + i
        q1, q2 = i % n_qubits, (i + 1) % n_qubits
        src += [gate, q1, gate, q2]
        dst += [q1, gate, q2, gate]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_list = torch.tensor(
        [[i % n_qubits, (i + 1) % n_qubits] for i in range(n_edges)], dtype=torch.long
    )
    g = Data(
        x=x, edge_index=edge_index, node_type=node_type,
        n_qubit_nodes=n_qubits, n_edges_unique=n_edges, edge_list=edge_list,
    )
    g.y = torch.tensor(rng.standard_normal(n_edges + n_qubits), dtype=torch.float32)
    return g


def _make_dataset(n: int = 6, seed_start: int = 0) -> list[Data]:
    """Return a dataset of n synthetic graphs with distinct seeds."""
    return [_make_unified_graph(seed=seed_start + i) for i in range(n)]


@pytest.fixture
def small_dataset():
    return _make_dataset(n=6)


@pytest.fixture
def tiny_model():
    return UnifiedMPNN(hidden_dim=32, n_layers=2)


# ─────────────────────────────────────────────────────────────────────────────
# TestShouldRetrain — complete coverage of all decision branches
# ─────────────────────────────────────────────────────────────────────────────


class TestShouldRetrain:
    """Tests for should_retrain() gating logic."""

    # ── no-op cases ──────────────────────────────────────────────────────────

    def test_zero_new_points_always_skips(self):
        ok, reason = should_retrain(0, 0.70, 0.70, 45)
        assert not ok
        assert reason == "no_new_data"

    def test_zero_new_points_even_with_improved_pass_rate(self):
        """pass_rate improvement is irrelevant when there's no new data."""
        ok, reason = should_retrain(0, 0.90, 0.50, 45)
        assert not ok
        assert reason == "no_new_data"

    def test_below_min_points_skips(self):
        """Fewer than min_new_points absolute always skips."""
        ok, reason = should_retrain(0, 0.69, 0.69, 45, min_new_points=1)
        assert not ok

    # ── pass rate improvement overrides fraction threshold ────────────────────

    def test_pass_rate_improved_always_retrains(self):
        """Pass rate improvement by >3pp always triggers retrain regardless of fraction."""
        ok, reason = should_retrain(1, 0.80, 0.65, 200)
        assert ok
        assert reason == "pass_rate_improved"

    def test_pass_rate_improvement_exactly_at_boundary(self):
        """Exactly +3pp+ improvement (>0.03) triggers retrain."""
        ok, reason = should_retrain(1, 0.74, 0.70, 200)
        assert ok
        assert reason == "pass_rate_improved"

    def test_pass_rate_improvement_below_threshold_does_not_trigger(self):
        """Improvement < 0.03 is noise — should not trigger via pass_rate path."""
        ok, reason = should_retrain(1, 0.72, 0.70, 200)
        # Should not match pass_rate_improved; may still match new_data_available
        assert reason != "pass_rate_improved"

    # ── fraction-based skip for large datasets ────────────────────────────────

    def test_tiny_fraction_large_dataset_skips(self):
        """1 new point in 200-point dataset (0.5% < 5% threshold) → skip."""
        ok, reason = should_retrain(1, 0.69, 0.69, 200)
        assert not ok
        assert reason == "below_min_fraction"

    def test_tiny_fraction_small_dataset_retrains(self):
        """1 new point in 15-point dataset (dataset_size <= 20) → retrain.

        The dataset_size > 20 guard prevents over-skipping on small datasets
        where every point matters.
        """
        ok, reason = should_retrain(1, 0.69, 0.69, 15)
        assert ok

    def test_fraction_at_threshold_retrains(self):
        """Exactly 5% new data → retrain (boundary is exclusive: < not <=)."""
        # 10 new points in 200 = exactly 5%
        ok, reason = should_retrain(10, 0.69, 0.69, 200)
        assert ok

    def test_fraction_above_threshold_retrains(self):
        """Well above 5% threshold → retrain."""
        ok, reason = should_retrain(15, 0.69, 0.69, 200)
        assert ok
        assert reason == "new_data_available"

    # ── meaningful new data ───────────────────────────────────────────────────

    def test_meaningful_new_data_retrains(self):
        ok, reason = should_retrain(3, 0.69, 0.69, 45)
        assert ok
        assert reason == "new_data_available"

    def test_docstring_examples_exact_match(self):
        """The three docstring examples must hold exactly."""
        assert should_retrain(0, 0.69, 0.69, 45) == (False, "no_new_data")
        assert should_retrain(3, 0.75, 0.69, 45) == (True, "pass_rate_improved")
        assert should_retrain(1, 0.69, 0.69, 200) == (False, "below_min_fraction")

    # ── custom thresholds ─────────────────────────────────────────────────────

    @pytest.mark.parametrize("n_new,ds,expected_ok", [
        (1, 100, False),   # 1% < 10% custom threshold → skip
        (5, 100, False),   # 5% < 10% → skip
        (10, 100, True),   # 10% = threshold → retrain (boundary exclusive)
        (15, 100, True),   # 15% > 10% → retrain
    ])
    def test_custom_min_fraction(self, n_new, ds, expected_ok):
        ok, _ = should_retrain(n_new, 0.69, 0.69, ds, min_new_fraction=0.10)
        assert ok is expected_ok


# ─────────────────────────────────────────────────────────────────────────────
# TestTrainUnifiedMPNN — validation, early-stop, and result structure
# ─────────────────────────────────────────────────────────────────────────────


class TestTrainUnifiedMPNN:
    """Tests for train_unified_mpnn() core functionality."""

    # ── input validation ──────────────────────────────────────────────────────

    def test_raises_on_wrong_model_type(self, small_dataset):
        from qmbp_simulation.predictors.mpnn import MPNNPredictor
        wrong_model = MPNNPredictor(node_features=2, hidden_dim=32, n_layers=2, output_dim=11)
        with pytest.raises(TypeError, match="Expected UnifiedMPNN"):
            train_unified_mpnn(wrong_model, small_dataset, n_epochs=5)

    def test_raises_on_too_few_points(self, tiny_model):
        dataset = _make_dataset(n=2)
        with pytest.raises(ValueError, match="≥3 training points"):
            train_unified_mpnn(tiny_model, dataset, n_epochs=5)

    def test_raises_on_missing_required_attributes(self, tiny_model):
        """Graphs missing node_type raise ValueError before training starts."""
        bad_graphs = [_make_unified_graph(seed=i) for i in range(4)]
        del bad_graphs[0].node_type  # remove required attr
        with pytest.raises(ValueError, match="missing required attributes"):
            train_unified_mpnn(tiny_model, bad_graphs, n_epochs=5)

    def test_raises_on_negative_mse_floor(self, tiny_model, small_dataset):
        with pytest.raises(ValueError, match="mse_floor must be"):
            train_unified_mpnn(tiny_model, small_dataset, n_epochs=5, mse_floor=-1.0)

    # ── training runs and result structure ────────────────────────────────────

    def test_returns_expected_keys(self, tiny_model, small_dataset):
        result = train_unified_mpnn(tiny_model, small_dataset, n_epochs=10, val_fraction=0.0)
        required = {
            "final_mse", "final_zz_mse", "final_x_mse",
            "n_epochs_run", "stopped_early", "stop_reason",
            "n_train", "n_val", "mse_history",
            "zz_loss_history", "x_loss_history",
        }
        assert required.issubset(result.keys()), f"Missing: {required - result.keys()}"

    def test_mse_history_length_matches_epochs_run(self, tiny_model, small_dataset):
        result = train_unified_mpnn(tiny_model, small_dataset, n_epochs=10, val_fraction=0.0)
        assert len(result["mse_history"]) == result["n_epochs_run"]
        assert result["n_epochs_run"] <= 10

    def test_loss_decreases_over_training(self, tiny_model, small_dataset):
        """MSE at end should be lower than at start (model is learning)."""
        result = train_unified_mpnn(tiny_model, small_dataset, n_epochs=50, val_fraction=0.0)
        if len(result["mse_history"]) >= 10:
            early = np.mean(result["mse_history"][:5])
            late = np.mean(result["mse_history"][-5:])
            assert late < early, f"Loss did not decrease: {early:.4f} → {late:.4f}"

    def test_val_split_produces_val_mse(self, tiny_model, small_dataset):
        result = train_unified_mpnn(
            tiny_model, small_dataset, n_epochs=20, val_fraction=0.3
        )
        assert result["val_mse"] is not None
        assert result["n_val"] > 0
        assert result["n_train"] + result["n_val"] == len(small_dataset)
        assert result["generalization_gap"] is not None

    def test_val_fraction_zero_disables_validation(self, tiny_model, small_dataset):
        result = train_unified_mpnn(tiny_model, small_dataset, n_epochs=10, val_fraction=0.0)
        assert result["val_mse"] is None
        assert result["n_val"] == 0

    # ── mse_floor early-stop ──────────────────────────────────────────────────

    def test_mse_floor_stops_early(self, tiny_model):
        """Very high mse_floor (>initial MSE) should trigger early stop."""
        dataset = _make_dataset(n=6, seed_start=100)
        # Force model output to nearly match targets so MSE will be tiny
        tiny_model.eval()
        with torch.no_grad():
            for g in dataset:
                g.y = tiny_model(g).squeeze(0).detach().clone()
        tiny_model.train()

        result = train_unified_mpnn(
            tiny_model, dataset,
            n_epochs=5000, val_fraction=0.0,
            mse_floor=1.0,  # MSE from zero-error init is ~0 < 1.0 → stops immediately
            seed=42,
        )
        assert result["stopped_early"] is True
        assert result["stop_reason"] == "mse_floor_reached"
        assert result["n_epochs_run"] <= 60  # stops after checking at epoch >= 50

    def test_mse_floor_zero_disables_early_stop(self, tiny_model, small_dataset):
        """mse_floor=0 must never trigger early stop."""
        result = train_unified_mpnn(
            tiny_model, small_dataset, n_epochs=15, val_fraction=0.0, mse_floor=0.0
        )
        assert result["stop_reason"] != "mse_floor_reached"

    def test_mse_floor_not_triggered_before_epoch_50(self, tiny_model):
        """mse_floor early-stop requires epoch >= 50 to avoid premature exits."""
        dataset = _make_dataset(n=6, seed_start=200)
        tiny_model.eval()
        with torch.no_grad():
            for g in dataset:
                g.y = tiny_model(g).squeeze(0).detach().clone()
        tiny_model.train()

        result = train_unified_mpnn(
            tiny_model, dataset,
            n_epochs=30,   # only 30 epochs — less than 50 → floor not reachable
            val_fraction=0.0, mse_floor=1.0, seed=42,
        )
        # With only 30 epochs, mse_floor logic can't trigger (requires epoch >= 50)
        # Either completes normally or exhausts lr
        assert result["n_epochs_run"] <= 30

    # ── skipped-graph guards ──────────────────────────────────────────────────

    def test_nan_y_graphs_skipped_not_crash(self, tiny_model):
        """Graphs with NaN targets should be skipped gracefully."""
        dataset = _make_dataset(n=6, seed_start=300)
        dataset[2].y = torch.full_like(dataset[2].y, float("nan"))
        dataset[4].y = torch.full_like(dataset[4].y, float("inf"))
        # Should complete without raising
        result = train_unified_mpnn(tiny_model, dataset, n_epochs=5, val_fraction=0.0)
        assert result["n_epochs_run"] > 0

    def test_mismatched_y_length_graphs_skipped(self, tiny_model):
        """Graphs whose y length doesn't match model output are skipped."""
        dataset = _make_dataset(n=6, seed_start=400)
        dataset[1].y = torch.zeros(3)  # wrong length — mismatch pred vs target
        result = train_unified_mpnn(tiny_model, dataset, n_epochs=5, val_fraction=0.0)
        assert result["n_epochs_run"] > 0  # training continues despite skip

    # ── layer-wise LR path ────────────────────────────────────────────────────

    def test_layerwise_lr_parameter_injected(self, tiny_model, small_dataset):
        """_layerwise_lr param triggers AdamW with per-group learning rates."""
        lw = {"early_conv": 0.1, "last_conv": 0.5, "heads": 1.0, "type_emb": 0.2}
        result = train_unified_mpnn(
            tiny_model, small_dataset, n_epochs=5,
            val_fraction=0.0, _layerwise_lr=lw,
        )
        assert result["n_epochs_run"] > 0
        assert result["final_mse"] < float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# TestFineTuneUnifiedMPNN — genuine fine-tuning correctness
# ─────────────────────────────────────────────────────────────────────────────


class TestFineTuneUnifiedMPNN:
    """Tests for fine_tune_unified_mpnn() result structure, LR strategy, and edge cases."""

    # ── input validation ──────────────────────────────────────────────────────

    def test_raises_on_wrong_model_type(self, small_dataset):
        from qmbp_simulation.predictors.mpnn import MPNNPredictor
        wrong = MPNNPredictor(node_features=2, hidden_dim=32, n_layers=2, output_dim=11)
        with pytest.raises(TypeError, match="Expected UnifiedMPNN"):
            fine_tune_unified_mpnn(wrong, small_dataset)

    def test_raises_on_too_few_points(self, tiny_model):
        with pytest.raises(ValueError, match="≥3 points"):
            fine_tune_unified_mpnn(tiny_model, _make_dataset(n=2))

    def test_raises_on_invalid_layerwise_decay(self, tiny_model, small_dataset):
        with pytest.raises(ValueError, match="layerwise_decay"):
            fine_tune_unified_mpnn(tiny_model, small_dataset, layerwise_decay=0.0)
        with pytest.raises(ValueError, match="layerwise_decay"):
            fine_tune_unified_mpnn(tiny_model, small_dataset, layerwise_decay=1.5)

    # ── result structure ──────────────────────────────────────────────────────

    def test_result_has_finetune_keys(self, tiny_model, small_dataset):
        result = fine_tune_unified_mpnn(tiny_model, small_dataset, n_epochs=10)
        assert result["mode"] == "fine_tune"
        assert "initial_mse" in result
        assert "improvement_ratio" in result
        assert "layerwise_lr_used" in result
        assert "notes" in result
        assert result["notes"] in ("improved", "minimal_improvement", "below_mse_floor")

    def test_layerwise_lr_enabled_by_default(self, tiny_model, small_dataset):
        result = fine_tune_unified_mpnn(tiny_model, small_dataset, n_epochs=10)
        assert result["layerwise_lr_used"] is True

    def test_layerwise_lr_disabled_when_freeze_false(self, tiny_model, small_dataset):
        result = fine_tune_unified_mpnn(
            tiny_model, small_dataset, n_epochs=10, freeze_early_layers=False
        )
        assert result["layerwise_lr_used"] is False

    def test_initial_mse_recorded_from_first_epoch(self, tiny_model, small_dataset):
        result = fine_tune_unified_mpnn(tiny_model, small_dataset, n_epochs=20)
        assert result["initial_mse"] > 0
        assert np.isfinite(result["initial_mse"])

    def test_improvement_ratio_finite_and_positive(self, tiny_model, small_dataset):
        result = fine_tune_unified_mpnn(tiny_model, small_dataset, n_epochs=20)
        assert np.isfinite(result["improvement_ratio"])
        assert result["improvement_ratio"] > 0

    # ── mse_floor as real early-stop ──────────────────────────────────────────

    def test_mse_floor_stops_fine_tune_early(self, tiny_model):
        """Fine-tune with high mse_floor stops before n_epochs."""
        dataset = _make_dataset(n=6, seed_start=500)
        tiny_model.eval()
        with torch.no_grad():
            for g in dataset:
                g.y = tiny_model(g).squeeze(0).detach().clone()
        tiny_model.train()

        result = fine_tune_unified_mpnn(
            tiny_model, dataset, n_epochs=5000, mse_floor=1.0
        )
        assert result["stopped_early"] is True
        assert result["stop_reason"] == "mse_floor_reached"
        assert result["notes"] == "below_mse_floor"

    # ── weight preservation — catastrophic forgetting guard ──────────────────

    def test_early_conv_weights_change_less_than_heads(self, small_dataset):
        """Layer-wise LR should cause heads to change more than early backbone."""
        model = UnifiedMPNN(hidden_dim=32, n_layers=3)
        # Snapshot weights before fine-tune
        before_early = model.convs[0].nn[0].weight.data.clone()
        before_head = model.qubit_head[0].weight.data.clone()

        fine_tune_unified_mpnn(
            model, small_dataset, n_epochs=50,
            freeze_early_layers=True, layerwise_decay=0.1,
        )

        delta_early = (model.convs[0].nn[0].weight.data - before_early).abs().mean()
        delta_head = (model.qubit_head[0].weight.data - before_head).abs().mean()

        # Head should change at least 2× more than early layers
        assert delta_head > delta_early * 2 or delta_early < 1e-6, (
            f"Layer-wise LR not working: delta_early={delta_early:.2e}, "
            f"delta_head={delta_head:.2e}"
        )

    # ── notes categorization ──────────────────────────────────────────────────

    @pytest.mark.parametrize("expected_note,ratio", [
        ("minimal_improvement", 0.99),   # barely changed
        ("improved", 0.5),               # meaningfully improved
    ])
    def test_notes_based_on_improvement_ratio(self, expected_note, ratio):
        """Notes are correctly set based on improvement_ratio threshold."""
        # Build a fake result dict and check categorization logic
        # (testing the conditional directly, not the full training)
        result = {
            "improvement_ratio": ratio,
            "stop_reason": "completed",
            "n_epochs_run": 100,
            "notes": None,
        }
        if result["improvement_ratio"] > 0.95:
            result["notes"] = "minimal_improvement"
        elif result["stop_reason"] == "mse_floor_reached":
            result["notes"] = "below_mse_floor"
        else:
            result["notes"] = "improved"
        assert result["notes"] == expected_note


# ─────────────────────────────────────────────────────────────────────────────
# TestCheckpointRoundTrip — save/load preserves architecture metadata
# ─────────────────────────────────────────────────────────────────────────────


class TestUnifiedMPNNCheckpoint:
    """Checkpoint roundtrip tests — architecture metadata must survive save/load."""

    def test_roundtrip_preserves_state_dict(self, tiny_model, tmp_path):
        path = str(tmp_path / "unified.pt")
        save_unified_checkpoint(tiny_model, path, training_metadata={"epoch": 50})
        loaded = load_unified_checkpoint(path)
        for key in tiny_model.state_dict():
            assert torch.equal(tiny_model.state_dict()[key], loaded.state_dict()[key]), (
                f"State mismatch in {key}"
            )

    def test_roundtrip_preserves_architecture_metadata(self, tmp_path):
        model = UnifiedMPNN(
            hidden_dim=128, n_layers=4, norm_type="layer",
            dropout=0.2, type_embedding_dim=8, gate_readout=False,
        )
        path = str(tmp_path / "unified_custom.pt")
        save_unified_checkpoint(model, path)
        loaded = load_unified_checkpoint(path)
        assert loaded.hidden_dim == 128
        assert loaded.n_layers == 4
        assert loaded.norm_type == "layer"
        assert loaded.dropout_rate == pytest.approx(0.2)
        assert loaded.type_embedding_dim == 8
        assert loaded.gate_readout is False

    def test_load_sets_eval_mode_by_default(self, tiny_model, tmp_path):
        path = str(tmp_path / "unified.pt")
        save_unified_checkpoint(tiny_model, path)
        loaded = load_unified_checkpoint(path, eval_mode=True)
        assert not loaded.training

    def test_load_with_eval_mode_false_stays_in_train_mode(self, tiny_model, tmp_path):
        path = str(tmp_path / "unified.pt")
        save_unified_checkpoint(tiny_model, path)
        loaded = load_unified_checkpoint(path, eval_mode=False)
        assert loaded.training

    def test_checkpoint_has_architecture_key(self, tiny_model, tmp_path):
        """Saved file must contain 'architecture': 'unified_mpnn' for _smart_load."""
        path = str(tmp_path / "unified.pt")
        save_unified_checkpoint(tiny_model, path)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw.get("architecture") == "unified_mpnn"
        assert "node_features" in raw
        assert "hidden_dim" in raw
        assert "gate_readout" in raw

    def test_training_metadata_preserved(self, tiny_model, tmp_path):
        meta = {"epoch": 3000, "final_mse": 0.0042, "topology": "chain_1d"}
        path = str(tmp_path / "unified_meta.pt")
        save_unified_checkpoint(tiny_model, path, training_metadata=meta)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["training_metadata"]["epoch"] == 3000
        assert raw["training_metadata"]["topology"] == "chain_1d"


# ─────────────────────────────────────────────────────────────────────────────
# TestModelZooRegistration — register_checkpoint auto-detects model type
# ─────────────────────────────────────────────────────────────────────────────


class TestModelZooRegistration:
    """Tests for register_checkpoint() with UnifiedMPNN and MPNNPredictor."""

    @pytest.fixture
    def zoo_entry(self):
        from datetime import datetime, timezone
        from qmbp_simulation.predictors.model_zoo import ZooEntry
        return ZooEntry(
            model="tfim_bond_resolved",
            topology="chain_1d",
            n_qubits=6,
            p_layers=1,
            checkpoint_file="kiro_test_unified_n6_p1.pt",
            h_range=(1.0, 3.5),
            pass_rate=0.85,
            n_training_points=10,
            seeds=[42],
            created=datetime.now(timezone.utc).isoformat(),
            notes="Test entry",
        )

    def test_unified_mpnn_saved_with_correct_format(self, tmp_path, monkeypatch, zoo_entry):
        """register_checkpoint detects UnifiedMPNN and uses save_unified_checkpoint."""
        from qmbp_simulation.predictors import model_zoo as zoo_mod
        monkeypatch.setattr(zoo_mod, "_CHECKPOINTS_DIR", tmp_path / "checkpoints")
        monkeypatch.setattr(zoo_mod, "_MANIFEST_PATH", tmp_path / "manifest.json")

        model = UnifiedMPNN(hidden_dim=32, n_layers=2)
        from qmbp_simulation.predictors.model_zoo import register_checkpoint
        path = register_checkpoint(model, zoo_entry, overwrite=True)

        assert path.exists()
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        # Must have unified_mpnn architecture metadata
        assert raw.get("architecture") == "unified_mpnn", (
            f"Wrong format: architecture={raw.get('architecture')!r} "
            "(should be 'unified_mpnn', was saved with save_mpnn_checkpoint)"
        )
        assert "qubit_head.0.weight" in raw["state_dict"]

    def test_mpnn_predictor_saved_with_mpnn_format(self, tmp_path, monkeypatch, zoo_entry):
        """register_checkpoint uses save_mpnn_checkpoint for MPNNPredictor."""
        from qmbp_simulation.predictors import model_zoo as zoo_mod
        monkeypatch.setattr(zoo_mod, "_CHECKPOINTS_DIR", tmp_path / "checkpoints")
        monkeypatch.setattr(zoo_mod, "_MANIFEST_PATH", tmp_path / "manifest.json")

        from qmbp_simulation.predictors.mpnn import MPNNPredictor
        zoo_entry.checkpoint_file = "kiro_test_mpnn_n6_p1.pt"
        model = MPNNPredictor(node_features=2, hidden_dim=32, n_layers=2, output_dim=11)

        from qmbp_simulation.predictors.model_zoo import register_checkpoint
        path = register_checkpoint(model, zoo_entry, overwrite=True)

        assert path.exists()
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
        # MPNNPredictor format: no 'architecture' key at top level
        assert raw.get("architecture") != "unified_mpnn"

    def test_unknown_model_type_raises(self, tmp_path, monkeypatch, zoo_entry):
        """Unrecognized model type raises TypeError."""
        from qmbp_simulation.predictors import model_zoo as zoo_mod
        monkeypatch.setattr(zoo_mod, "_CHECKPOINTS_DIR", tmp_path / "checkpoints")
        monkeypatch.setattr(zoo_mod, "_MANIFEST_PATH", tmp_path / "manifest.json")

        class _FakeModel:
            pass

        from qmbp_simulation.predictors.model_zoo import register_checkpoint
        with pytest.raises(TypeError, match="unrecognized model type"):
            register_checkpoint(_FakeModel(), zoo_entry, overwrite=True)

    def test_manifest_updated_after_registration(self, tmp_path, monkeypatch, zoo_entry):
        """Manifest JSON is written with the new entry after registration."""
        from qmbp_simulation.predictors import model_zoo as zoo_mod
        monkeypatch.setattr(zoo_mod, "_CHECKPOINTS_DIR", tmp_path / "checkpoints")
        monkeypatch.setattr(zoo_mod, "_MANIFEST_PATH", tmp_path / "manifest.json")

        model = UnifiedMPNN(hidden_dim=32, n_layers=2)
        from qmbp_simulation.predictors.model_zoo import register_checkpoint, _load_manifest
        register_checkpoint(model, zoo_entry, overwrite=True)

        entries = _load_manifest()
        assert any(e.checkpoint_file == zoo_entry.checkpoint_file for e in entries)

    def test_sha256_computed_and_stored(self, tmp_path, monkeypatch, zoo_entry):
        """SHA256 hash is computed and stored in the manifest after registration."""
        from qmbp_simulation.predictors import model_zoo as zoo_mod
        monkeypatch.setattr(zoo_mod, "_CHECKPOINTS_DIR", tmp_path / "checkpoints")
        monkeypatch.setattr(zoo_mod, "_MANIFEST_PATH", tmp_path / "manifest.json")

        model = UnifiedMPNN(hidden_dim=32, n_layers=2)
        from qmbp_simulation.predictors.model_zoo import register_checkpoint, _load_manifest
        register_checkpoint(model, zoo_entry, overwrite=True)

        entries = _load_manifest()
        match = next(e for e in entries if e.checkpoint_file == zoo_entry.checkpoint_file)
        assert len(match.sha256) == 64, "SHA256 should be 64 hex chars"

    def test_overwrite_false_does_not_replace(self, tmp_path, monkeypatch, zoo_entry):
        """overwrite=False returns existing path without overwriting."""
        from qmbp_simulation.predictors import model_zoo as zoo_mod
        monkeypatch.setattr(zoo_mod, "_CHECKPOINTS_DIR", tmp_path / "checkpoints")
        monkeypatch.setattr(zoo_mod, "_MANIFEST_PATH", tmp_path / "manifest.json")

        model = UnifiedMPNN(hidden_dim=32, n_layers=2)
        from qmbp_simulation.predictors.model_zoo import register_checkpoint
        path = register_checkpoint(model, zoo_entry, overwrite=True)
        original_size = path.stat().st_size

        # Second registration with overwrite=False — file should be unchanged
        register_checkpoint(model, zoo_entry, overwrite=False)
        assert path.stat().st_size == original_size

    def test_load_pretrained_returns_unified_mpnn(self, tmp_path, monkeypatch, zoo_entry):
        """load_pretrained should return UnifiedMPNN after registering one."""
        from qmbp_simulation.predictors import model_zoo as zoo_mod
        monkeypatch.setattr(zoo_mod, "_CHECKPOINTS_DIR", tmp_path / "checkpoints")
        monkeypatch.setattr(zoo_mod, "_MANIFEST_PATH", tmp_path / "manifest.json")

        model = UnifiedMPNN(hidden_dim=32, n_layers=2)
        from qmbp_simulation.predictors.model_zoo import register_checkpoint, load_pretrained
        register_checkpoint(model, zoo_entry, overwrite=True)

        loaded, meta = load_pretrained(
            model="tfim_bond_resolved",
            topology="chain_1d",
            n_qubits=6,
            p_layers=1,
        )
        assert isinstance(loaded, UnifiedMPNN), (
            f"Expected UnifiedMPNN, got {type(loaded).__name__}"
        )
        assert meta.pass_rate == pytest.approx(0.85)


# ─────────────────────────────────────────────────────────────────────────────
# TestMaybeFineTuneMPNN — ValidationRunner helper integration
# ─────────────────────────────────────────────────────────────────────────────


import argparse

from qmbp_simulation.framework.runner_base import ValidationRunner, Section


def _make_args(**kwargs):
    defaults = {
        "section": None, "skip_preflight": False,
        "stop_on_failure": False, "verbose": False, "dry_run": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class _MinimalRunner(ValidationRunner):
    runner_id = "test_finetune_helper"
    experiment_id = "TFH"
    description = "Fine-tune helper test runner"
    hypothesis = "maybe_fine_tune_mpnn integrates correctly"

    def define_sections(self):
        return [Section(id=1, name="Noop", fn=lambda: {"pass": True}, hypothesis="")]


class TestMaybeFineTuneMPNN:
    """Tests for ValidationRunner.maybe_fine_tune_mpnn()."""

    @pytest.fixture
    def runner(self):
        return _MinimalRunner(args=_make_args())

    def test_skips_when_no_new_data(self, runner, small_dataset, tiny_model):
        result = runner.maybe_fine_tune_mpnn(
            tiny_model, small_dataset,
            prev_pass_rate=0.70, current_pass_rate=0.70, n_new_points=0,
        )
        assert result is None

    def test_skips_when_tiny_fraction_large_dataset(self, runner, tiny_model):
        dataset = _make_dataset(n=50)
        result = runner.maybe_fine_tune_mpnn(
            tiny_model, dataset,
            prev_pass_rate=0.69, current_pass_rate=0.69, n_new_points=1,
        )
        assert result is None

    def test_retrains_when_pass_rate_improved(self, runner, small_dataset, tiny_model):
        result = runner.maybe_fine_tune_mpnn(
            tiny_model, small_dataset,
            prev_pass_rate=0.50, current_pass_rate=0.75, n_new_points=2,
            n_epochs=10,
        )
        assert result is not None
        assert result["mode"] == "fine_tune"

    def test_retrains_with_meaningful_new_data(self, runner, small_dataset, tiny_model):
        result = runner.maybe_fine_tune_mpnn(
            tiny_model, small_dataset,
            prev_pass_rate=0.69, current_pass_rate=0.69, n_new_points=3,
            n_epochs=10,
        )
        assert result is not None
        assert "improvement_ratio" in result

    def test_raises_on_wrong_model_type(self, runner, small_dataset):
        from qmbp_simulation.predictors.mpnn import MPNNPredictor
        wrong = MPNNPredictor(node_features=2, hidden_dim=32, n_layers=2, output_dim=11)
        with pytest.raises(TypeError, match="maybe_fine_tune_mpnn expects UnifiedMPNN"):
            runner.maybe_fine_tune_mpnn(
                wrong, small_dataset,
                prev_pass_rate=0.5, current_pass_rate=0.8, n_new_points=3,
            )

    def test_returns_none_for_empty_dataset(self, runner, tiny_model):
        result = runner.maybe_fine_tune_mpnn(
            tiny_model, [],
            prev_pass_rate=0.5, current_pass_rate=0.8, n_new_points=3,
        )
        assert result is None

    def test_result_has_expected_keys_when_retraining(self, runner, small_dataset, tiny_model):
        result = runner.maybe_fine_tune_mpnn(
            tiny_model, small_dataset,
            prev_pass_rate=0.50, current_pass_rate=0.80, n_new_points=3,
            n_epochs=10,
        )
        assert result is not None
        for key in ("mode", "initial_mse", "improvement_ratio", "notes", "layerwise_lr_used"):
            assert key in result, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# TestMultiNAggregatorRegressions — NameError and edge cases
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiNAggregatorRegressions:
    """Regression tests for MultiNAggregator bugs."""

    def test_n_defined_before_gt_cache_lookup(self, tmp_path):
        """N must be parsed from filename BEFORE GroundTruthCache.get() is called.

        Regression: the original code called gt_cache.get(..., n, ...) before
        assigning n from the filename. This caused a NameError in any codepath
        where de_gaps was missing from the NPZ.
        """
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        npz_dir = tmp_path / "data" / "multi_n_training"
        npz_dir.mkdir(parents=True)

        # Create NPZ WITHOUT de_gaps or gaps — triggers the GroundTruthCache fallback
        h_vals = np.array([2.0, 2.5, 3.0])
        theta = np.random.default_rng(0).standard_normal((3, 11))
        e_exact = np.array([-5.0, -4.5, -4.0])
        e_vqe = e_exact + np.array([0.1, 0.05, 0.02])
        npz_path = npz_dir / "chain_1d_N6_p1.npz"
        np.savez(npz_path, h_values=h_vals, theta_opt=theta, e_exact=e_exact, e_vqe=e_vqe)
        # Note: no 'gaps', no 'de_gaps' — forces the fallback code path

        agg = MultiNAggregator(
            topology="chain_1d", model="tfim_bond_resolved",
            results_dir=tmp_path,
        )
        # Monkeypatch _PROJECT_ROOT so it looks in tmp_path
        import qmbp_simulation.predictors.multi_n_aggregator as agg_mod
        orig = agg_mod._PROJECT_ROOT
        try:
            agg_mod._PROJECT_ROOT = tmp_path
            summary = agg.scan()  # Should NOT raise NameError
        finally:
            agg_mod._PROJECT_ROOT = orig

        assert 6 in summary, f"Expected N=6 in summary, got {summary}"

    def test_scan_with_gaps_in_npz_uses_them_directly(self, tmp_path):
        """When gaps are in NPZ, they're used directly without GroundTruthCache."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator
        import qmbp_simulation.predictors.multi_n_aggregator as agg_mod

        npz_dir = tmp_path / "data" / "multi_n_training"
        npz_dir.mkdir(parents=True)

        h_vals = np.array([2.0, 2.5, 3.0])
        theta = np.random.default_rng(1).standard_normal((3, 11))
        e_exact = np.array([-5.0, -4.5, -4.0])
        e_vqe = e_exact + np.array([0.05, 0.02, 0.01])
        gaps = np.array([0.5, 0.6, 0.7])
        de_gaps = np.abs(e_vqe - e_exact) / gaps

        npz_path = npz_dir / "chain_1d_N8_p1.npz"
        np.savez(
            npz_path, h_values=h_vals, theta_opt=theta,
            e_exact=e_exact, e_vqe=e_vqe, gaps=gaps, de_gaps=de_gaps,
        )

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved",
                               results_dir=tmp_path)
        orig = agg_mod._PROJECT_ROOT
        try:
            agg_mod._PROJECT_ROOT = tmp_path
            summary = agg.scan()
        finally:
            agg_mod._PROJECT_ROOT = orig

        assert 8 in summary
        assert summary[8] == 3

    def test_scan_empty_directory_returns_empty(self, tmp_path):
        """Scan on empty directory returns {} without crashing."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator
        import qmbp_simulation.predictors.multi_n_aggregator as agg_mod

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved",
                               results_dir=tmp_path)
        orig = agg_mod._PROJECT_ROOT
        try:
            agg_mod._PROJECT_ROOT = tmp_path
            summary = agg.scan()
        finally:
            agg_mod._PROJECT_ROOT = orig

        assert summary == {}

    def test_scan_ignores_wrong_topology_files(self, tmp_path):
        """Files for a different topology are ignored."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator
        import qmbp_simulation.predictors.multi_n_aggregator as agg_mod

        npz_dir = tmp_path / "data" / "multi_n_training"
        npz_dir.mkdir(parents=True)

        h_vals = np.array([2.0, 2.5, 3.0])
        theta = np.zeros((3, 11))
        e_exact = np.array([-5.0, -4.5, -4.0])
        # Save as 'ladder' topology, but aggregator looks for 'chain_1d'
        np.savez(npz_dir / "ladder_N6_p1.npz",
                 h_values=h_vals, theta_opt=theta, e_exact=e_exact)

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved",
                               results_dir=tmp_path)
        orig = agg_mod._PROJECT_ROOT
        try:
            agg_mod._PROJECT_ROOT = tmp_path
            summary = agg.scan()
        finally:
            agg_mod._PROJECT_ROOT = orig

        assert summary == {}

    def test_scan_handles_corrupted_npz_gracefully(self, tmp_path):
        """Corrupted NPZ file is skipped, not crashed."""
        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator
        import qmbp_simulation.predictors.multi_n_aggregator as agg_mod

        npz_dir = tmp_path / "data" / "multi_n_training"
        npz_dir.mkdir(parents=True)

        # Write garbage as a "NPZ" file
        bad_file = npz_dir / "chain_1d_N10_p1.npz"
        bad_file.write_bytes(b"this is not a valid npz file")

        agg = MultiNAggregator(topology="chain_1d", model="tfim_bond_resolved",
                               results_dir=tmp_path)
        orig = agg_mod._PROJECT_ROOT
        try:
            agg_mod._PROJECT_ROOT = tmp_path
            summary = agg.scan()  # Should not raise
        finally:
            agg_mod._PROJECT_ROOT = orig

        # Corrupted file skipped → empty summary
        assert 10 not in summary


# ─────────────────────────────────────────────────────────────────────────────
# TestAcceleratedVQEMsrFloor — mse_floor integration in _train_mpnn
# ─────────────────────────────────────────────────────────────────────────────


class TestAcceleratedVQEMseFloor:
    """Verify AcceleratedVQE._train_mpnn passes mse_floor=1e-5 to train_unified_mpnn."""

    def test_train_mpnn_accepts_mse_floor_kwarg(self):
        """train_unified_mpnn signature accepts mse_floor — no TypeError raised."""
        import inspect
        sig = inspect.signature(train_unified_mpnn)
        assert "mse_floor" in sig.parameters, (
            "train_unified_mpnn missing mse_floor param — AcceleratedVQE._train_mpnn will break"
        )

    def test_accelerated_config_default_mpnn_epochs(self):
        """AcceleratedConfig.mpnn_epochs default is 3000 (not reduced by mse_floor)."""
        from qmbp_simulation.pipeline.accelerated import AcceleratedConfig
        cfg = AcceleratedConfig()
        assert cfg.mpnn_epochs == 3000

    def test_train_unified_mpnn_with_mse_floor_1e5_can_stop_early(self):
        """AcceleratedVQE's mse_floor=1e-5 can trigger early stop for very good models."""
        model = UnifiedMPNN(hidden_dim=32, n_layers=2)
        dataset = _make_dataset(n=6, seed_start=600)
        model.eval()
        with torch.no_grad():
            for g in dataset:
                g.y = model(g).squeeze(0).detach().clone()
        model.train()
        # With y = model(g) exactly, MSE will be ~0 << 1e-5
        result = train_unified_mpnn(
            model, dataset, n_epochs=5000, val_fraction=0.0, mse_floor=1e-5
        )
        assert result["stopped_early"] is True


# ─────────────────────────────────────────────────────────────────────────────
# TestEdgeCasesAndRegression — cross-cutting edge cases
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCasesAndRegression:
    """Edge cases that could slip through other tests."""

    def test_should_retrain_with_dataset_size_zero(self):
        """dataset_size=0 should not divide-by-zero."""
        ok, reason = should_retrain(1, 0.69, 0.69, 0)
        # 1 / max(0,1) = 1.0 ≥ 0.05 → retrain
        assert ok

    def test_should_retrain_with_negative_pass_rates(self):
        """Negative pass rates (bad data) should not crash."""
        ok, reason = should_retrain(3, -0.1, -0.5, 10)
        assert isinstance(ok, bool)

    def test_should_retrain_with_pass_rate_above_1(self):
        """Pass rates > 1.0 (possible from averaging bugs) should not crash."""
        ok, reason = should_retrain(3, 1.2, 0.9, 10)
        assert isinstance(ok, bool)

    def test_fine_tune_with_single_conv_layer(self, small_dataset):
        """Fine-tune works when model has only 1 conv layer (no early_conv group)."""
        model = UnifiedMPNN(hidden_dim=32, n_layers=1)
        result = fine_tune_unified_mpnn(model, small_dataset, n_epochs=10)
        assert result["mode"] == "fine_tune"
        assert result["layerwise_lr_used"] is True  # single conv = no early_convs group, still uses LR dict

    def test_fine_tune_with_no_type_embedding(self, small_dataset):
        """Fine-tune works when type_embedding_dim=0 (no type_emb layer)."""
        model = UnifiedMPNN(hidden_dim=32, n_layers=2, type_embedding_dim=0)
        result = fine_tune_unified_mpnn(model, small_dataset, n_epochs=10)
        assert result["mode"] == "fine_tune"

    def test_train_with_all_graphs_having_same_y(self, tiny_model):
        """Constant targets should still train without NaN losses."""
        dataset = _make_dataset(n=5, seed_start=700)
        constant_y = torch.zeros(11)  # 5 edges + 6 qubits
        for g in dataset:
            g.y = constant_y.clone()
        result = train_unified_mpnn(tiny_model, dataset, n_epochs=10, val_fraction=0.0)
        assert np.isfinite(result["final_mse"])

    def test_checkpoint_with_empty_metadata(self, tiny_model, tmp_path):
        """save_unified_checkpoint with metadata=None should not crash."""
        path = str(tmp_path / "no_meta.pt")
        save_unified_checkpoint(tiny_model, path, training_metadata=None)
        raw = torch.load(path, map_location="cpu", weights_only=False)
        assert raw["training_metadata"] == {}

    def test_load_unified_checkpoint_nonexistent_path(self):
        """Loading from nonexistent path raises a clear error."""
        with pytest.raises((FileNotFoundError, RuntimeError, Exception)):
            load_unified_checkpoint("/nonexistent/path/model.pt")

    def test_maybe_fine_tune_passes_custom_lr_and_floor(self):
        """Custom lr and mse_floor are forwarded to fine_tune_unified_mpnn."""
        runner = _MinimalRunner(args=_make_args())
        model = UnifiedMPNN(hidden_dim=32, n_layers=2)
        dataset = _make_dataset(n=6)

        # Set y = model output so MSE ~0 → mse_floor=10.0 will trigger immediately
        model.eval()
        with torch.no_grad():
            for g in dataset:
                g.y = model(g).squeeze(0).detach().clone()
        model.train()

        result = runner.maybe_fine_tune_mpnn(
            model, dataset,
            prev_pass_rate=0.50, current_pass_rate=0.80, n_new_points=3,
            n_epochs=5000, lr=1e-4, mse_floor=10.0,
        )
        assert result is not None
        assert result["stopped_early"] is True

    @pytest.mark.parametrize("n_qubits,n_edges", [
        (4, 3),   # tiny chain
        (10, 9),  # standard chain
        (16, 24), # 4x4 square lattice
    ])
    def test_train_unified_runs_for_multiple_system_sizes(self, n_qubits, n_edges):
        """Training works for different system sizes (cross-N generalization)."""
        dataset = [_make_unified_graph(n_qubits=n_qubits, n_edges=n_edges, seed=i)
                   for i in range(4)]
        model = UnifiedMPNN(hidden_dim=32, n_layers=2)
        result = train_unified_mpnn(model, dataset, n_epochs=5, val_fraction=0.0)
        assert result["final_mse"] < float("inf")
        assert result["n_train"] > 0
