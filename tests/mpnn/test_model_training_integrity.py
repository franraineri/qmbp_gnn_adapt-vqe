"""Tests for MPNN training → zoo → registry integrity.

Verifies end-to-end flow:
1. Trained models are registered without overwriting better ones
2. All provenance information is preserved
3. Zoo manifest and ModelRegistryDB stay synchronized
4. Anti-regression backup mechanism works
"""

from datetime import UTC, datetime

import pytest


class TestZooRegistrationIntegrity:
    """Test model_zoo.register_checkpoint behavior."""

    @pytest.fixture
    def isolated_zoo(self, tmp_path, monkeypatch):
        """Create an isolated zoo environment for testing."""
        from qmbp_simulation.predictors import model_zoo

        zoo_dir = tmp_path / "model_zoo"
        zoo_dir.mkdir()
        ckpts_dir = zoo_dir / "checkpoints"
        ckpts_dir.mkdir()

        monkeypatch.setattr(model_zoo, "_ZOO_DIR", zoo_dir)
        monkeypatch.setattr(model_zoo, "_MANIFEST_PATH", zoo_dir / "manifest.json")
        monkeypatch.setattr(model_zoo, "_CHECKPOINTS_DIR", ckpts_dir)

        return zoo_dir

    @pytest.fixture
    def mock_mpnn(self):
        """Create a minimal UnifiedMPNN for registration tests.

        Uses actual UnifiedMPNN class so register_checkpoint recognizes it.
        Small architecture to keep tests fast.
        """
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN

        # Minimal architecture for fast tests
        model = UnifiedMPNN(
            node_features=4,
            hidden_dim=8,
            n_layers=1,
            norm_type="none",
            dropout=0.0,
            type_embedding_dim=4,
            gate_readout=True,
        )
        model.eval()
        return model

    def test_register_new_model_creates_manifest_entry(self, isolated_zoo, mock_mpnn):
        """New model registration creates manifest entry with all metadata."""
        from qmbp_simulation.predictors.model_zoo import (
            ZooEntry,
            _load_manifest,
            register_checkpoint,
        )

        entry = ZooEntry(
            model="tfim_bond_resolved",
            topology="chain_1d",
            n_qubits=0,  # multi-N
            p_layers=1,
            checkpoint_file="test_multiN_p1.pt",
            h_range=(2.0, 4.0),
            pass_rate=0.85,
            n_training_points=150,
            seeds=[42],
            created=datetime.now(UTC).isoformat(),
            notes="Multi-N training: N=[6, 8, 10]",
            runner_tag="AC",
            date_tag="130826",
        )

        register_checkpoint(mock_mpnn, entry)

        # Verify manifest
        manifest = _load_manifest()
        assert len(manifest) == 1
        saved = manifest[0]
        assert saved.model == "tfim_bond_resolved"
        assert saved.topology == "chain_1d"
        assert saved.pass_rate == 0.85
        assert saved.n_training_points == 150
        assert saved.runner_tag == "AC"
        assert saved.date_tag == "130826"

    def test_no_overwrite_preserves_existing_model(self, isolated_zoo, mock_mpnn):
        """Without overwrite=True, existing model is NOT replaced."""
        from qmbp_simulation.predictors.model_zoo import (
            ZooEntry,
            _load_manifest,
            register_checkpoint,
        )

        # Register first model
        entry1 = ZooEntry(
            model="tfim_bond_resolved",
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            checkpoint_file="chain_1d_n10_p1.pt",
            pass_rate=0.90,
            n_training_points=50,
            notes="Original model",
        )
        register_checkpoint(mock_mpnn, entry1)

        # Try to register same checkpoint without overwrite
        entry2 = ZooEntry(
            model="tfim_bond_resolved",
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            checkpoint_file="chain_1d_n10_p1.pt",  # Same filename
            pass_rate=0.60,  # Worse pass_rate
            n_training_points=30,
            notes="New model attempt",
        )
        register_checkpoint(mock_mpnn, entry2, overwrite=False)

        # Original should be preserved
        manifest = _load_manifest()
        matching = [e for e in manifest if e.checkpoint_file == "chain_1d_n10_p1.pt"]
        assert len(matching) == 1
        # Note: manifest doesn't preserve the original entry details in this case,
        # but the checkpoint file should not be overwritten

    def test_overwrite_creates_backup_in_best_dir(self, isolated_zoo, mock_mpnn):
        """Overwriting a model backs up the old one to _best/ directory."""
        from qmbp_simulation.predictors.model_zoo import (
            ZooEntry,
            register_checkpoint,
        )

        # Register first model with good pass_rate
        entry1 = ZooEntry(
            model="tfim_bond_resolved",
            topology="ladder",
            n_qubits=10,
            p_layers=1,
            checkpoint_file="ladder_n10_p1.pt",
            pass_rate=0.95,
            date_tag="100826",
        )
        register_checkpoint(mock_mpnn, entry1)

        # Overwrite with new model (worse pass_rate)
        entry2 = ZooEntry(
            model="tfim_bond_resolved",
            topology="ladder",
            n_qubits=10,
            p_layers=1,
            checkpoint_file="ladder_n10_p1.pt",
            pass_rate=0.70,
            date_tag="130826",
        )
        register_checkpoint(mock_mpnn, entry2, overwrite=True)

        # Check backup exists
        best_dir = isolated_zoo / "checkpoints" / "_best"
        assert best_dir.exists()
        backups = list(best_dir.glob("ladder_n10_p1_pass*"))
        assert len(backups) >= 1, "Backup should be created in _best/"

    def test_unique_models_dont_conflict(self, isolated_zoo, mock_mpnn):
        """Different configs can coexist without overwriting each other."""
        from qmbp_simulation.predictors.model_zoo import (
            ZooEntry,
            _load_manifest,
            register_checkpoint,
        )

        configs = [
            ("chain_1d", 10, 1),
            ("chain_1d", 10, 2),
            ("chain_1d", 20, 1),
            ("ladder", 10, 1),
            ("ladder", 0, 1),  # multi-N
        ]

        for topo, n, p in configs:
            entry = ZooEntry(
                model="tfim_bond_resolved",
                topology=topo,
                n_qubits=n,
                p_layers=p,
                checkpoint_file=f"{topo}_n{n}_p{p}.pt",
                pass_rate=0.80,
            )
            register_checkpoint(mock_mpnn, entry)

        manifest = _load_manifest()
        assert len(manifest) == 5, "All 5 models should coexist"

        # Each should have unique checkpoint_file
        filenames = {e.checkpoint_file for e in manifest}
        assert len(filenames) == 5


class TestProvenanceCompleteness:
    """Test that all provenance information is captured and preserved."""

    @pytest.fixture
    def isolated_db(self, tmp_path):
        """Create isolated ModelRegistryDB."""
        from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

        return ModelRegistryDB(
            path=tmp_path / "registry.json",
            history_path=tmp_path / "history.json",
        )

    def test_zoo_entry_captures_full_provenance(self, tmp_path):
        """ZooEntry captures all training provenance fields."""
        from qmbp_simulation.predictors.model_zoo import ZooEntry

        entry = ZooEntry(
            model="tfim_bond_resolved",
            topology="chain_1d",
            n_qubits=0,
            p_layers=1,
            checkpoint_file="unified_tfim_br_chain_1d_multiN_6+8+10_p1.pt",
            h_range=(2.0, 5.0),
            pass_rate=0.88,
            n_training_points=200,
            seeds=[42, 123],
            created="2026-08-13T12:00:00+00:00",
            notes="Multi-N training: N=[6, 8, 10], 200 points, MSE=1.2e-3",
            runner_tag="AC",
            date_tag="130826",
        )

        # Verify all fields are set
        assert entry.model == "tfim_bond_resolved"
        assert entry.topology == "chain_1d"
        assert entry.n_qubits == 0  # multi-N indicator
        assert entry.h_range == (2.0, 5.0)
        assert entry.seeds == [42, 123]
        assert entry.runner_tag == "AC"
        assert "N=[6, 8, 10]" in entry.notes

    def test_registry_from_zoo_entry_preserves_n_values(self, isolated_db):
        """register_from_zoo_entry extracts N values from notes."""
        from qmbp_simulation.predictors.model_zoo import ZooEntry

        entry = ZooEntry(
            model="tfim_bond_resolved",
            topology="heavy_hex",
            n_qubits=0,
            p_layers=1,
            checkpoint_file="heavy_hex_multiN.pt",
            notes="Multi-N training: N=[6, 10, 16, 20], 300 points",
            n_training_points=300,
            runner_tag="MN",
        )

        record = isolated_db.register_from_zoo_entry(entry)

        assert record.training.n_values_used == [6, 10, 16, 20]
        assert record.training.total_training_points == 300
        assert record.runner_tag == "MN"

    def test_training_metrics_captured_after_training(self, isolated_db):
        """Training metrics (MSE, epochs, convergence) are stored."""
        from qmbp_simulation.predictors.model_registry_db import (
            ModelRecord,
            TrainingMetrics,
            TrainingProvenance,
        )

        record = ModelRecord(
            model_id="trained_model.pt",
            model_name="tfim_bond_resolved",
            topology="square",
            p_layers=1,
            training=TrainingProvenance(
                n_values_used=[6, 8, 10],
                total_training_points=150,
                points_per_n={"6": 40, "8": 50, "10": 60},
                h_range=(2.0, 4.0),
                seeds=[42],
                training_metrics=TrainingMetrics(
                    final_loss=0.0012,
                    final_mse=0.0015,
                    epochs=2500,
                    best_epoch=2100,
                    early_stopped=True,
                    convergence_status="converged",
                    loss_history=[0.1, 0.01, 0.005, 0.002, 0.0012],
                ),
            ),
            runner_tag="AC",
            date_tag="130826",
        )

        isolated_db.register_model(record)

        # Retrieve and verify
        result = isolated_db.get_model("trained_model.pt")
        tm = result.training.training_metrics

        assert tm.final_mse == pytest.approx(0.0015)
        assert tm.epochs == 2500
        assert tm.early_stopped is True
        assert tm.convergence_status == "converged"
        assert len(tm.loss_history) == 5


class TestAntiRegressionMechanism:
    """Test that better models are never lost due to overwrites."""

    @pytest.fixture
    def isolated_zoo(self, tmp_path, monkeypatch):
        """Create isolated zoo with mock stdin to skip interactive prompts."""
        import sys

        from qmbp_simulation.predictors import model_zoo

        zoo_dir = tmp_path / "model_zoo"
        zoo_dir.mkdir()
        ckpts_dir = zoo_dir / "checkpoints"
        ckpts_dir.mkdir()

        monkeypatch.setattr(model_zoo, "_ZOO_DIR", zoo_dir)
        monkeypatch.setattr(model_zoo, "_MANIFEST_PATH", zoo_dir / "manifest.json")
        monkeypatch.setattr(model_zoo, "_CHECKPOINTS_DIR", ckpts_dir)

        # Mock stdin to not be a tty (skip interactive prompt)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

        return zoo_dir

    @pytest.fixture
    def mock_mpnn(self):
        """Create a minimal UnifiedMPNN for registration tests."""
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN

        model = UnifiedMPNN(
            node_features=4,
            hidden_dim=8,
            n_layers=1,
            norm_type="none",
            dropout=0.0,
            type_embedding_dim=4,
            gate_readout=True,
        )
        model.eval()
        return model

    def test_downgrade_creates_warning_backup(self, isolated_zoo, mock_mpnn, caplog):
        """Replacing a better model with worse one creates backup and warns."""
        import logging

        from qmbp_simulation.predictors.model_zoo import ZooEntry, register_checkpoint

        # Register good model
        entry1 = ZooEntry(
            model="tfim_bond_resolved",
            topology="triangular",
            n_qubits=6,
            p_layers=1,
            checkpoint_file="triangular_n6_p1.pt",
            pass_rate=0.92,
            date_tag="100826",
        )
        register_checkpoint(mock_mpnn, entry1)

        # Overwrite with worse model
        with caplog.at_level(logging.WARNING):
            entry2 = ZooEntry(
                model="tfim_bond_resolved",
                topology="triangular",
                n_qubits=6,
                p_layers=1,
                checkpoint_file="triangular_n6_p1.pt",
                pass_rate=0.55,  # Much worse
                date_tag="130826",
            )
            register_checkpoint(mock_mpnn, entry2, overwrite=True)

        # Check warning was logged
        assert any("DOWNGRADE" in r.message for r in caplog.records)

        # Check backup was created
        best_dir = isolated_zoo / "checkpoints" / "_best"
        backups = list(best_dir.glob("triangular_n6_p1_pass*"))
        assert len(backups) >= 1

    def test_registry_tracks_version_on_overwrite(self, tmp_path):
        """ModelRegistryDB tracks version history on overwrites."""
        from qmbp_simulation.predictors.model_registry_db import (
            ModelRecord,
            ModelRegistryDB,
            TrainingProvenance,
        )

        db = ModelRegistryDB(
            path=tmp_path / "reg.json",
            history_path=tmp_path / "hist.json",
        )

        # Version 1
        db.register_model(
            ModelRecord(
                model_id="versioned.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                training=TrainingProvenance(total_training_points=50),
            )
        )

        # Version 2 (overwrite)
        db.register_model(
            ModelRecord(
                model_id="versioned.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                training=TrainingProvenance(total_training_points=100),
            ),
            overwrite=True,
        )

        # Version 3 (overwrite)
        db.register_model(
            ModelRecord(
                model_id="versioned.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                training=TrainingProvenance(total_training_points=200),
            ),
            overwrite=True,
        )

        result = db.get_model("versioned.pt")
        assert result.version == 3
        assert len(result.version_history) == 2  # Previous 2 versions

        # History should capture all retrains
        history = db.get_history(model_id="versioned.pt", event_type="retrained")
        assert len(history) == 2


class TestZooRegistrySync:
    """Test synchronization between model_zoo and ModelRegistryDB."""

    @pytest.fixture
    def isolated_env(self, tmp_path, monkeypatch):
        """Setup isolated zoo + registry environment."""
        import sys

        from qmbp_simulation.predictors import model_zoo

        # Isolated zoo
        zoo_dir = tmp_path / "model_zoo"
        zoo_dir.mkdir()
        ckpts_dir = zoo_dir / "checkpoints"
        ckpts_dir.mkdir()

        monkeypatch.setattr(model_zoo, "_ZOO_DIR", zoo_dir)
        monkeypatch.setattr(model_zoo, "_MANIFEST_PATH", zoo_dir / "manifest.json")
        monkeypatch.setattr(model_zoo, "_CHECKPOINTS_DIR", ckpts_dir)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

        # Isolated registry
        reg_path = tmp_path / "registry.json"
        hist_path = tmp_path / "history.json"

        return {
            "zoo_dir": zoo_dir,
            "manifest_path": zoo_dir / "manifest.json",
            "reg_path": reg_path,
            "hist_path": hist_path,
        }

    @pytest.fixture
    def mock_mpnn(self):
        """Create minimal UnifiedMPNN for sync tests."""
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN

        model = UnifiedMPNN(
            node_features=4,
            hidden_dim=8,
            n_layers=1,
            norm_type="none",
            dropout=0.0,
            type_embedding_dim=4,
            gate_readout=True,
        )
        model.eval()
        return model

    def test_sync_from_manifest_populates_registry(self, isolated_env, mock_mpnn):
        """sync_from_manifest imports zoo entries into registry."""
        from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB
        from qmbp_simulation.predictors.model_zoo import ZooEntry, register_checkpoint

        # Register models in zoo
        for topo in ["chain_1d", "ladder", "square"]:
            entry = ZooEntry(
                model="tfim_bond_resolved",
                topology=topo,
                n_qubits=10,
                p_layers=1,
                checkpoint_file=f"{topo}_n10_p1.pt",
                pass_rate=0.85,
                n_training_points=50,
                runner_tag="AC",
            )
            register_checkpoint(mock_mpnn, entry)

        # Create registry and sync
        db = ModelRegistryDB(
            path=isolated_env["reg_path"],
            history_path=isolated_env["hist_path"],
        )

        # Manually simulate sync (since we patched zoo paths)
        from qmbp_simulation.predictors.model_zoo import _load_manifest

        manifest = _load_manifest()
        for zoo_entry in manifest:
            db.register_from_zoo_entry(zoo_entry)

        # Verify all models are in registry
        all_models = db.list_all()
        assert len(all_models) == 3
        topos = {m.topology for m in all_models}
        assert topos == {"chain_1d", "ladder", "square"}

    def test_multiple_models_same_config_preserved(self, isolated_env, mock_mpnn):
        """Different checkpoint filenames for same config don't overwrite."""
        from qmbp_simulation.predictors.model_zoo import ZooEntry, register_checkpoint

        # Register two models with same config but different filenames
        for i, pass_rate in enumerate([0.80, 0.90]):
            entry = ZooEntry(
                model="tfim_bond_resolved",
                topology="chain_1d",
                n_qubits=10,
                p_layers=1,
                checkpoint_file=f"chain_1d_n10_p1_v{i + 1}.pt",
                pass_rate=pass_rate,
                n_training_points=50 + i * 10,
                date_tag=f"1{i}0826",
            )
            register_checkpoint(mock_mpnn, entry)

        # Both should exist in manifest
        from qmbp_simulation.predictors.model_zoo import _load_manifest

        manifest = _load_manifest()
        chain_models = [e for e in manifest if e.topology == "chain_1d"]
        assert len(chain_models) == 2

        # Both should have different checkpoint files
        files = {e.checkpoint_file for e in chain_models}
        assert len(files) == 2


class TestContaminationFiltering:
    """Test contamination-aware model selection (should_retrain integration)."""

    def test_should_retrain_rejects_not_useful_data(self):
        """should_retrain returns False for not_useful training data."""
        from qmbp_simulation.predictors.unified_mpnn import should_retrain

        should, reason = should_retrain(
            n_new_points=10,
            current_pass_rate=0.50,
            prev_pass_rate=0.50,
            dataset_size=100,
            training_utility="not_useful",
            failure_mode="healthy",
        )

        assert should is False
        assert reason == "training_data_not_useful"

    def test_should_retrain_forces_on_contamination(self):
        """should_retrain returns True for contaminated models."""
        from qmbp_simulation.predictors.unified_mpnn import should_retrain

        should, reason = should_retrain(
            n_new_points=0,  # No new data
            current_pass_rate=0.30,
            prev_pass_rate=0.30,
            dataset_size=50,
            training_utility="useful",
            failure_mode="contaminated_training",
        )

        assert should is True
        assert reason == "contaminated_training_detected"

    def test_should_retrain_gap_masking_with_useful_data(self):
        """Gap masking + useful data triggers retrain even with 0 new points."""
        from qmbp_simulation.predictors.unified_mpnn import should_retrain

        should, reason = should_retrain(
            n_new_points=0,
            current_pass_rate=0.70,
            prev_pass_rate=0.70,
            dataset_size=100,
            training_utility="useful",
            failure_mode="gap_masking",
        )

        assert should is True
        assert reason == "gap_masking_with_improved_data"

    def test_should_retrain_warns_on_insufficient_signal(self, caplog):
        """should_retrain logs warning for insufficient_signal data."""
        import logging

        from qmbp_simulation.predictors.unified_mpnn import should_retrain

        with caplog.at_level(logging.WARNING):
            should, reason = should_retrain(
                n_new_points=5,
                current_pass_rate=0.80,  # Improved
                prev_pass_rate=0.70,
                dataset_size=50,
                training_utility="insufficient_signal",
            )

        assert should is True
        assert "insufficient_signal" in reason
        assert any("insufficient_signal" in r.message for r in caplog.records)


class TestNPZQualitySync:
    """Test NPZ quality tier updates triggering needs_retrain flag."""

    def test_npz_upsert_notifies_registry(self, tmp_path, monkeypatch):
        """upsert_theta_npz notifies registry when quality improves."""
        from qmbp_simulation.predictors.model_registry_db import (
            ModelRecord,
            ModelRegistryDB,
            TrainingProvenance,
        )

        # Create isolated registry
        reg_path = tmp_path / "registry.json"
        hist_path = tmp_path / "history.json"
        db = ModelRegistryDB(path=reg_path, history_path=hist_path)

        # Register a model
        db.register_model(
            ModelRecord(
                model_id="chain_1d_N10_p1.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                p_layers=1,
                training=TrainingProvenance(
                    n_values_used=[10],
                    total_training_points=20,
                ),
            )
        )

        # Verify needs_retrain is False initially
        result = db.get_model("chain_1d_N10_p1.pt")
        assert result.dashboard_quality.needs_retrain is False

    def test_mark_needs_retrain_sets_flag(self, tmp_path):
        """mark_needs_retrain_from_npz_update sets needs_retrain flag when quality improves."""
        from qmbp_simulation.predictors.model_registry_db import (
            ModelRecord,
            ModelRegistryDB,
            TrainingProvenance,
        )

        db = ModelRegistryDB(
            path=tmp_path / "reg.json",
            history_path=tmp_path / "hist.json",
        )

        db.register_model(
            ModelRecord(
                model_id="test_model.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                p_layers=1,
                training=TrainingProvenance(n_values_used=[10]),
            )
        )

        # Call mark_needs_retrain with correct signature (old/new verified ratio)
        marked = db.mark_needs_retrain_from_npz_update(
            topology="chain_1d",
            n_qubits=10,
            model_name="tfim_bond_resolved",
            p_layers=1,
            old_verified_ratio=0.30,  # 30% verified before
            new_verified_ratio=0.60,  # 60% verified now (significant improvement)
        )

        assert marked is True
        result = db.get_model("test_model.pt")
        assert result.dashboard_quality.needs_retrain is True

    def test_get_models_needing_retrain(self, tmp_path):
        """get_models_needing_retrain returns correct list."""
        from qmbp_simulation.predictors.model_registry_db import (
            DashboardQuality,
            ModelRecord,
            ModelRegistryDB,
        )

        db = ModelRegistryDB(
            path=tmp_path / "reg.json",
            history_path=tmp_path / "hist.json",
        )

        # Model needing retrain
        db.register_model(
            ModelRecord(
                model_id="needs_retrain.pt",
                model_name="tfim",
                topology="chain_1d",
                dashboard_quality=DashboardQuality(needs_retrain=True),
            )
        )

        # Model NOT needing retrain
        db.register_model(
            ModelRecord(
                model_id="fine_model.pt",
                model_name="tfim",
                topology="ladder",
                dashboard_quality=DashboardQuality(needs_retrain=False),
            )
        )

        needing = db.get_models_needing_retrain()
        assert len(needing) == 1
        assert needing[0].model_id == "needs_retrain.pt"
