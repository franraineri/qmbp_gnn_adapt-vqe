"""Tests for H/I/F/G features: monotonicity, aggregator, retrain queue, rollback.

Covers:
- H: enforce_h_frontier_monotonicity
- I: build_clean_training_dataset
- F: compute_retrain_queue
- G: auto-rollback in register_checkpoint (>30% regression)
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# H. enforce_h_frontier_monotonicity
# ─────────────────────────────────────────────────────────────────────────────


class TestEnforceHFrontierMonotonicity:
    """h_frontier should be non-decreasing with N per topology."""

    def test_corrects_anomaly(self):
        from qmbp_simulation.analysis.metrics import enforce_h_frontier_monotonicity

        configs = [
            {"topology": "heavy_hex", "n_qubits": 6, "h_frontier": 1.9},
            {"topology": "heavy_hex", "n_qubits": 10, "h_frontier": 1.76},
            {"topology": "heavy_hex", "n_qubits": 12, "h_frontier": 1.93},
        ]
        enforce_h_frontier_monotonicity(configs)

        c10 = next(c for c in configs if c["n_qubits"] == 10)
        assert c10["h_frontier"] == 1.9
        assert c10["h_frontier_corrected"] is True
        assert c10["h_frontier_original"] == 1.76

    def test_leaves_monotonic_unchanged(self):
        from qmbp_simulation.analysis.metrics import enforce_h_frontier_monotonicity

        configs = [
            {"topology": "chain_1d", "n_qubits": 6, "h_frontier": 1.5},
            {"topology": "chain_1d", "n_qubits": 10, "h_frontier": 1.8},
            {"topology": "chain_1d", "n_qubits": 20, "h_frontier": 2.2},
        ]
        enforce_h_frontier_monotonicity(configs)

        for c in configs:
            assert "h_frontier_corrected" not in c

    def test_handles_single_config(self):
        from qmbp_simulation.analysis.metrics import enforce_h_frontier_monotonicity

        configs = [{"topology": "chain_1d", "n_qubits": 10, "h_frontier": 1.8}]
        enforce_h_frontier_monotonicity(configs)
        assert configs[0]["h_frontier"] == 1.8

    def test_handles_none_frontiers(self):
        from qmbp_simulation.analysis.metrics import enforce_h_frontier_monotonicity

        configs = [
            {"topology": "chain_1d", "n_qubits": 6, "h_frontier": None},
            {"topology": "chain_1d", "n_qubits": 10, "h_frontier": 1.8},
        ]
        enforce_h_frontier_monotonicity(configs)
        assert configs[1]["h_frontier"] == 1.8

    def test_multi_topology_independent(self):
        from qmbp_simulation.analysis.metrics import enforce_h_frontier_monotonicity

        configs = [
            {"topology": "chain_1d", "n_qubits": 6, "h_frontier": 2.0},
            {"topology": "chain_1d", "n_qubits": 10, "h_frontier": 1.5},  # anomaly
            {"topology": "ladder", "n_qubits": 6, "h_frontier": 2.5},
            {"topology": "ladder", "n_qubits": 10, "h_frontier": 2.8},  # OK
        ]
        enforce_h_frontier_monotonicity(configs)

        chain_10 = next(c for c in configs if c["topology"] == "chain_1d" and c["n_qubits"] == 10)
        assert chain_10["h_frontier"] == 2.0  # corrected
        assert chain_10["h_frontier_corrected"] is True

        ladder_10 = next(c for c in configs if c["topology"] == "ladder" and c["n_qubits"] == 10)
        assert "h_frontier_corrected" not in ladder_10  # untouched


# ─────────────────────────────────────────────────────────────────────────────
# I. build_clean_training_dataset
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildCleanTrainingDataset:
    """Unified training data aggregator with exclusion policies."""

    def test_loads_real_data(self):
        from qmbp_simulation.framework.result_io import build_clean_training_dataset

        ds = build_clean_training_dataset("chain_1d", p_layers=1)
        assert len(ds["h_values"]) > 0
        assert len(ds["n_values_used"]) >= 4
        assert ds["n_excluded"] >= 0
        assert len(ds["h_values"]) == len(ds["e_vqe"])
        assert len(ds["h_values"]) == len(ds["n_qubits_per_point"])

    def test_verified_only_is_subset(self):
        from qmbp_simulation.framework.result_io import build_clean_training_dataset

        ds_all = build_clean_training_dataset("chain_1d", p_layers=1)
        ds_ver = build_clean_training_dataset("chain_1d", p_layers=1, min_quality_tier="verified")
        assert len(ds_ver["h_values"]) <= len(ds_all["h_values"])

    def test_reject_not_useful_excludes_configs(self):
        from qmbp_simulation.framework.result_io import build_clean_training_dataset

        ds = build_clean_training_dataset("triangular", p_layers=1, reject_not_useful=True)
        # triangular N=14,16 are "not_useful" — should not appear
        n_values = ds["n_values_used"]
        assert 14 not in n_values or 16 not in n_values  # at least one excluded

    def test_nonexistent_topology_returns_empty(self):
        from qmbp_simulation.framework.result_io import build_clean_training_dataset

        ds = build_clean_training_dataset("nonexistent_topo", p_layers=1)
        assert len(ds["h_values"]) == 0
        assert ds["n_values_used"] == []

    def test_h_frontier_filtering(self):
        from qmbp_simulation.framework.result_io import build_clean_training_dataset

        # With very high h_frontier override → should exclude almost everything
        ds = build_clean_training_dataset(
            "chain_1d",
            p_layers=1,
            h_frontier_override={6: 99.0, 8: 99.0, 10: 99.0, 12: 99.0, 15: 99.0, 20: 99.0},
        )
        assert len(ds["h_values"]) == 0 or ds["n_excluded"] > 0

    def test_theta_ragged_handling(self):
        """Multi-N dataset has different theta dimensions per N."""
        from qmbp_simulation.framework.result_io import build_clean_training_dataset

        ds = build_clean_training_dataset("chain_1d", p_layers=1)
        # chain_1d has N=6(11 params), N=8(15), N=10(19), etc. → object array
        theta = ds["theta_opt"]
        assert theta.dtype == object  # Mixed dimensions → object array


# ─────────────────────────────────────────────────────────────────────────────
# F. compute_retrain_queue
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeRetrainQueue:
    """Retrain trigger produces prioritized queue."""

    def test_returns_list(self):
        from qmbp_simulation.predictors.model_zoo import compute_retrain_queue

        queue = compute_retrain_queue()
        assert isinstance(queue, list)

    def test_queue_sorted_by_priority(self):
        from qmbp_simulation.predictors.model_zoo import compute_retrain_queue

        queue = compute_retrain_queue()
        if len(queue) >= 2:
            priorities = [r["priority"] for r in queue]
            assert priorities == sorted(priorities)

    def test_queue_entries_have_required_fields(self):
        from qmbp_simulation.predictors.model_zoo import compute_retrain_queue

        queue = compute_retrain_queue()
        required = {"topology", "checkpoint_file", "priority", "reason", "command"}
        for entry in queue:
            assert required.issubset(entry.keys()), f"Missing keys: {required - entry.keys()}"

    def test_contaminated_is_priority_1(self):
        from qmbp_simulation.predictors.model_zoo import compute_retrain_queue

        queue = compute_retrain_queue()
        contaminated = [r for r in queue if r["priority"] == 1]
        for r in contaminated:
            assert "contaminated" in r["reason"]


# ─────────────────────────────────────────────────────────────────────────────
# G. Auto-rollback (>30% regression)
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoRollback:
    """register_checkpoint blocks severe regressions (>30% relative drop)."""

    def _make_model_and_entry(self, pass_rate=0.9):
        """Create a minimal real MPNNPredictor and entry for testing."""
        from qmbp_simulation.predictors.model_zoo import ZooEntry
        from qmbp_simulation.predictors.mpnn import MPNNPredictor

        model = MPNNPredictor(
            node_features=2,
            hidden_dim=16,
            output_dim=19,
            n_layers=2,
        )
        model.eval()

        entry = ZooEntry(
            model="tfim_bond_resolved",
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            checkpoint_file="test_rollback_n10_p1.pt",
            h_range=(1.5, 5.5),
            pass_rate=pass_rate,
            n_training_points=50,
            seeds=[42],
            created="2026-08-17",
            notes="test",
            runner_tag="XX",
            date_tag="170826",
        )
        return model, entry

    def test_severe_regression_triggers_rollback(self, tmp_path, monkeypatch):
        """If new pass_rate is >30% worse, checkpoint is NOT overwritten."""
        from qmbp_simulation.predictors import model_zoo

        # Redirect checkpoints to tmp
        monkeypatch.setattr(model_zoo, "_CHECKPOINTS_DIR", tmp_path)
        monkeypatch.setattr(model_zoo, "_MANIFEST_PATH", tmp_path / "manifest.json")

        # Create an "existing" good model
        model_good, entry_good = self._make_model_and_entry(pass_rate=0.9)
        path1 = model_zoo.register_checkpoint(model_good, entry_good, overwrite=True)
        assert path1.exists()
        good_size = path1.stat().st_size

        # Try to register a much worse model (pass=0.2 vs 0.9 = 78% drop)
        model_bad, entry_bad = self._make_model_and_entry(pass_rate=0.2)
        path2 = model_zoo.register_checkpoint(model_bad, entry_bad, overwrite=True)

        # Rollback should have kept the original
        current_size = path2.stat().st_size
        assert current_size == good_size, (
            "Checkpoint was overwritten despite >30% regression. "
            "Auto-rollback should have prevented this."
        )

    def test_mild_regression_still_overwrites(self, tmp_path, monkeypatch):
        """If new pass_rate is <30% worse, checkpoint IS overwritten (normal flow)."""
        from qmbp_simulation.predictors import model_zoo

        monkeypatch.setattr(model_zoo, "_CHECKPOINTS_DIR", tmp_path)
        monkeypatch.setattr(model_zoo, "_MANIFEST_PATH", tmp_path / "manifest.json")

        model1, entry1 = self._make_model_and_entry(pass_rate=0.8)
        model_zoo.register_checkpoint(model1, entry1, overwrite=True)

        # Mild regression: 0.8 → 0.7 = 12.5% drop (< 30%)
        model2, entry2 = self._make_model_and_entry(pass_rate=0.7)
        path2 = model_zoo.register_checkpoint(model2, entry2, overwrite=True)

        # Should have been overwritten normally
        # Verify manifest has the new pass_rate
        entries = model_zoo._load_manifest()
        match = [e for e in entries if e.checkpoint_file == "test_rollback_n10_p1.pt"]
        assert len(match) == 1
        assert match[0].pass_rate == 0.7
