"""Tests for qmbp_simulation.predictors.model_registry_db.

Verifies:
1. CRUD operations (register, get, query, update)
2. Persistence (save/load roundtrip)
3. Query filters (topology, min_points, min_n_values, status)
4. Lifecycle transitions (active → superseded → archived)
5. Evaluation tracking (append evaluations)
6. sync_from_manifest (idempotent population from zoo manifest)
7. enrich_points_per_n (NPZ scan enrichment)
8. Serialization edge cases (tuples, empty lists)
"""

import json

import numpy as np
import pytest

from qmbp_simulation.predictors.model_registry_db import (
    DashboardQuality,
    EvaluationRecord,
    ModelRecord,
    ModelRegistryDB,
    TrainingMetrics,
    TrainingProvenance,
)


@pytest.fixture
def tmp_registry(tmp_path):
    """Create a ModelRegistryDB backed by a temp file."""
    path = tmp_path / "test_registry.json"
    hist_path = tmp_path / "test_history.json"
    return ModelRegistryDB(path=path, history_path=hist_path)


@pytest.fixture
def sample_record():
    """A sample ModelRecord for testing."""
    return ModelRecord(
        model_id="test_model_chain_1d_p1.pt",
        model_name="tfim_bond_resolved",
        architecture="UnifiedMPNN",
        topology="chain_1d",
        p_layers=1,
        checkpoint_path="data/model_zoo/checkpoints/test_model_chain_1d_p1.pt",
        created="2026-08-12T00:00:00+00:00",
        runner_tag="AC",
        date_tag="120826",
        training=TrainingProvenance(
            n_values_used=[6, 8, 10],
            total_training_points=120,
            points_per_n={"6": 30, "8": 40, "10": 50},
            h_range=(2.0, 4.0),
            seeds=[42],
            data_source="multi_n_training",
        ),
        status="active",
        notes="Test model",
    )


@pytest.fixture
def populated_registry(tmp_registry, sample_record):
    """Registry with a few models pre-loaded."""
    tmp_registry.register_model(sample_record)

    # Second model - different topology
    record2 = ModelRecord(
        model_id="test_model_ladder_p1.pt",
        model_name="tfim_bond_resolved",
        architecture="UnifiedMPNN",
        topology="ladder",
        p_layers=1,
        checkpoint_path="data/model_zoo/checkpoints/test_model_ladder_p1.pt",
        created="2026-08-11T00:00:00+00:00",
        runner_tag="AC",
        date_tag="110826",
        training=TrainingProvenance(
            n_values_used=[4, 6, 8, 10, 12],
            total_training_points=250,
            points_per_n={"4": 20, "6": 50, "8": 60, "10": 70, "12": 50},
            h_range=(1.5, 5.0),
            seeds=[42],
            data_source="multi_n_training",
        ),
        status="active",
        notes="Ladder multi-N model",
    )
    tmp_registry.register_model(record2)

    # Third model - archived
    record3 = ModelRecord(
        model_id="old_model_chain_1d.pt",
        model_name="tfim_bond_resolved",
        topology="chain_1d",
        p_layers=1,
        training=TrainingProvenance(
            n_values_used=[6, 8],
            total_training_points=50,
            points_per_n={"6": 25, "8": 25},
        ),
        status="archived",
        notes="Old model, replaced",
    )
    tmp_registry.register_model(record3)

    return tmp_registry


class TestModelRegistryDBCRUD:
    """Test basic Create/Read/Update/Delete operations."""

    def test_register_and_get(self, tmp_registry, sample_record):
        tmp_registry.register_model(sample_record)
        result = tmp_registry.get_model("test_model_chain_1d_p1.pt")

        assert result is not None
        assert result.model_id == "test_model_chain_1d_p1.pt"
        assert result.topology == "chain_1d"
        assert result.training.total_training_points == 120
        assert result.training.n_values_used == [6, 8, 10]
        assert result.training.points_per_n == {"6": 30, "8": 40, "10": 50}

    def test_get_nonexistent(self, tmp_registry):
        result = tmp_registry.get_model("nonexistent.pt")
        assert result is None

    def test_register_no_overwrite(self, tmp_registry, sample_record):
        tmp_registry.register_model(sample_record)
        # Try to register again without overwrite
        modified = ModelRecord(
            model_id=sample_record.model_id,
            model_name="different",
            topology="square",
            training=TrainingProvenance(total_training_points=999),
        )
        tmp_registry.register_model(modified, overwrite=False)

        # Original should be preserved
        result = tmp_registry.get_model(sample_record.model_id)
        assert result.topology == "chain_1d"
        assert result.training.total_training_points == 120

    def test_register_with_overwrite(self, tmp_registry, sample_record):
        tmp_registry.register_model(sample_record)
        modified = ModelRecord(
            model_id=sample_record.model_id,
            model_name="tfim_bond_resolved",
            topology="chain_1d",
            p_layers=1,
            training=TrainingProvenance(
                n_values_used=[6, 8, 10, 12],
                total_training_points=200,
                points_per_n={"6": 30, "8": 40, "10": 50, "12": 80},
            ),
            status="active",
        )
        tmp_registry.register_model(modified, overwrite=True)

        result = tmp_registry.get_model(sample_record.model_id)
        assert result.training.total_training_points == 200
        assert result.training.n_values_used == [6, 8, 10, 12]

    def test_list_all_excludes_archived(self, populated_registry):
        active = populated_registry.list_all(include_archived=False)
        assert len(active) == 2
        assert all(r.status == "active" for r in active)

    def test_list_all_includes_archived(self, populated_registry):
        all_models = populated_registry.list_all(include_archived=True)
        assert len(all_models) == 3


class TestModelRegistryDBQueries:
    """Test query filtering capabilities."""

    def test_query_by_topology(self, populated_registry):
        results = populated_registry.query(topology="chain_1d")
        assert len(results) == 1
        assert results[0].topology == "chain_1d"

    def test_query_by_min_training_points(self, populated_registry):
        results = populated_registry.query(min_training_points=200)
        assert len(results) == 1
        assert results[0].model_id == "test_model_ladder_p1.pt"

    def test_query_by_min_n_values(self, populated_registry):
        results = populated_registry.query(min_n_values=4)
        assert len(results) == 1
        assert results[0].topology == "ladder"

    def test_query_combined_filters(self, populated_registry):
        results = populated_registry.query(topology="ladder", min_training_points=100)
        assert len(results) == 1

        results = populated_registry.query(topology="chain_1d", min_training_points=200)
        assert len(results) == 0

    def test_query_excludes_archived_by_default(self, populated_registry):
        results = populated_registry.query(topology="chain_1d")
        assert len(results) == 1
        assert results[0].status == "active"

    def test_query_include_archived(self, populated_registry):
        results = populated_registry.query(topology="chain_1d", status=None)
        assert len(results) == 2

    def test_query_no_filters_returns_all_active(self, populated_registry):
        results = populated_registry.query()
        assert len(results) == 2


class TestModelRegistryDBPersistence:
    """Test save/load roundtrip and data integrity."""

    def test_persistence_roundtrip(self, tmp_path, sample_record):
        path = tmp_path / "roundtrip.json"
        hist_path = tmp_path / "roundtrip_hist.json"
        db1 = ModelRegistryDB(path=path, history_path=hist_path)
        db1.register_model(sample_record)

        # Create new instance from same file
        db2 = ModelRegistryDB(path=path, history_path=hist_path)
        result = db2.get_model(sample_record.model_id)

        assert result is not None
        assert result.model_id == sample_record.model_id
        assert result.training.n_values_used == [6, 8, 10]
        assert result.training.points_per_n == {"6": 30, "8": 40, "10": 50}
        assert result.training.h_range == (2.0, 4.0)

    def test_h_range_tuple_survives_json(self, tmp_path, sample_record):
        """h_range stored as list in JSON but reconstructed as tuple."""
        path = tmp_path / "tuple_test.json"
        hist_path = tmp_path / "tuple_test_hist.json"
        db = ModelRegistryDB(path=path, history_path=hist_path)
        db.register_model(sample_record)

        # Read raw JSON to verify format
        with open(path) as f:
            raw = json.load(f)
        assert raw[0]["training"]["h_range"] == [2.0, 4.0]

        # Reload and verify tuple reconstruction
        db2 = ModelRegistryDB(path=path, history_path=hist_path)
        result = db2.get_model(sample_record.model_id)
        assert isinstance(result.training.h_range, tuple)
        assert result.training.h_range == (2.0, 4.0)

    def test_empty_registry_file_created_on_first_save(self, tmp_path):
        path = tmp_path / "new_registry.json"
        assert not path.exists()

        db = ModelRegistryDB(path=path, history_path=tmp_path / "h.json")
        record = ModelRecord(
            model_id="first.pt",
            model_name="tfim",
            training=TrainingProvenance(total_training_points=10),
        )
        db.register_model(record)
        assert path.exists()

    def test_load_nonexistent_file_gives_empty(self, tmp_path):
        path = tmp_path / "does_not_exist.json"
        db = ModelRegistryDB(path=path, history_path=tmp_path / "h.json")
        assert db.list_all(include_archived=True) == []


class TestModelRegistryDBLifecycle:
    """Test model lifecycle transitions."""

    def test_mark_superseded(self, populated_registry):
        populated_registry.mark_superseded(
            "test_model_chain_1d_p1.pt",
            superseded_by="new_model_chain_1d_p1.pt",
        )
        result = populated_registry.get_model("test_model_chain_1d_p1.pt")
        assert result.status == "superseded"
        assert result.superseded_by == "new_model_chain_1d_p1.pt"

    def test_archive(self, populated_registry):
        populated_registry.archive("test_model_chain_1d_p1.pt")
        result = populated_registry.get_model("test_model_chain_1d_p1.pt")
        assert result.status == "archived"

    def test_superseded_excluded_from_default_query(self, populated_registry):
        populated_registry.mark_superseded(
            "test_model_chain_1d_p1.pt",
            superseded_by="new.pt",
        )
        results = populated_registry.query(topology="chain_1d")
        assert len(results) == 0


class TestModelRegistryDBEvaluations:
    """Test evaluation tracking."""

    def test_add_evaluation(self, populated_registry):
        evaluation = EvaluationRecord(
            evaluated_at="2026-08-12T12:00:00+00:00",
            target_n_values=[20, 30],
            pass_rate_5pct=0.85,
            pass_rate_dual=0.70,
            mean_de_gap=0.032,
            mean_abs_error_per_site=0.009,
            notes="Evaluated on large-N extrapolation",
        )
        populated_registry.add_evaluation("test_model_chain_1d_p1.pt", evaluation)

        result = populated_registry.get_model("test_model_chain_1d_p1.pt")
        assert len(result.evaluations) == 1
        assert result.evaluations[0].pass_rate_5pct == 0.85
        assert result.evaluations[0].target_n_values == [20, 30]

    def test_multiple_evaluations_appended(self, populated_registry):
        eval1 = EvaluationRecord(
            evaluated_at="2026-08-12T12:00:00+00:00",
            target_n_values=[20],
            pass_rate_dual=0.50,
        )
        eval2 = EvaluationRecord(
            evaluated_at="2026-08-12T18:00:00+00:00",
            target_n_values=[20, 30],
            pass_rate_dual=0.75,
            notes="After iterative improve",
        )
        populated_registry.add_evaluation("test_model_chain_1d_p1.pt", eval1)
        populated_registry.add_evaluation("test_model_chain_1d_p1.pt", eval2)

        result = populated_registry.get_model("test_model_chain_1d_p1.pt")
        assert len(result.evaluations) == 2
        assert result.evaluations[-1].pass_rate_dual == 0.75

    def test_add_evaluation_nonexistent_model(self, populated_registry):
        evaluation = EvaluationRecord(pass_rate_dual=0.5)
        # Should not raise, just warn
        populated_registry.add_evaluation("nonexistent.pt", evaluation)


class TestModelRegistryDBSummary:
    """Test summary/statistics generation."""

    def test_summary_counts(self, populated_registry):
        s = populated_registry.summary()
        assert s["total_models"] == 3
        assert s["active_models"] == 2
        assert "chain_1d" in s["topologies"]
        assert "ladder" in s["topologies"]
        assert s["total_training_points"] == 370  # 120 + 250
        assert s["max_n_trained"] == 12

    def test_summary_empty_registry(self, tmp_registry):
        s = tmp_registry.summary()
        assert s["total_models"] == 0
        assert s["active_models"] == 0
        assert s["max_n_trained"] == 0


class TestModelRegistryDBFromZooEntry:
    """Test register_from_zoo_entry convenience method."""

    def test_from_zoo_entry_multi_n(self, tmp_registry):
        from qmbp_simulation.predictors.model_zoo import ZooEntry

        zoo_entry = ZooEntry(
            model="tfim_bond_resolved",
            topology="square",
            n_qubits=0,
            p_layers=1,
            checkpoint_file="unified_tfim_br_square_multiN_4+6+8_p1.pt",
            h_range=(2.0, 5.0),
            pass_rate=0.85,
            n_training_points=150,
            seeds=[42],
            created="2026-08-12T00:00:00+00:00",
            notes="Multi-N training: N=[4, 6, 8], 150 points",
            runner_tag="AC",
            date_tag="120826",
        )
        record = tmp_registry.register_from_zoo_entry(zoo_entry)

        assert record.model_id == "unified_tfim_br_square_multiN_4+6+8_p1.pt"
        assert record.topology == "square"
        assert record.training.n_values_used == [4, 6, 8]
        assert record.training.total_training_points == 150
        assert record.training.data_source == "multi_n_training"

    def test_from_zoo_entry_single_n(self, tmp_registry):
        from qmbp_simulation.predictors.model_zoo import ZooEntry

        zoo_entry = ZooEntry(
            model="tfim_bond_resolved",
            topology="chain_1d",
            n_qubits=10,
            p_layers=1,
            checkpoint_file="model_chain_1d_n10_p1.pt",
            h_range=(2.0, 3.5),
            pass_rate=1.0,
            n_training_points=14,
            seeds=[42],
            created="2026-08-12T00:00:00+00:00",
            notes="Single-N model",
            runner_tag="XX",
            date_tag="120826",
        )
        record = tmp_registry.register_from_zoo_entry(zoo_entry)

        assert record.training.n_values_used == [10]
        assert record.training.data_source == "single_n"

    def test_from_zoo_entry_infers_n_from_notes(self, tmp_registry):
        from qmbp_simulation.predictors.model_zoo import ZooEntry

        zoo_entry = ZooEntry(
            model="tfim_bond_resolved",
            topology="ladder",
            n_qubits=0,
            p_layers=1,
            checkpoint_file="some_model_ladder.pt",
            notes="Multi-N training: N=[4, 6, 8, 10, 12], 200 points",
            n_training_points=200,
        )
        record = tmp_registry.register_from_zoo_entry(zoo_entry)
        assert record.training.n_values_used == [4, 6, 8, 10, 12]


class TestModelRegistryDBEnrichment:
    """Test NPZ enrichment functionality."""

    def test_enrich_points_per_n_with_mock_npz(self, tmp_path):
        """Create mock NPZ files and verify enrichment picks them up."""
        registry_path = tmp_path / "registry.json"
        npz_dir = tmp_path / "data" / "multi_n_training"
        npz_dir.mkdir(parents=True)

        # Create mock NPZ files
        for n, pts in [(6, 30), (8, 45), (10, 60)]:
            h_values = np.linspace(2.0, 4.0, pts)
            np.savez(npz_dir / f"chain_1d_N{n}_p1.npz", h_values=h_values)

        # Create registry with a record that has n_values but no points_per_n
        db = ModelRegistryDB(path=registry_path, history_path=tmp_path / "hist.json")
        record = ModelRecord(
            model_id="test_enrich.pt",
            model_name="tfim_bond_resolved",
            topology="chain_1d",
            p_layers=1,
            training=TrainingProvenance(
                n_values_used=[6, 8, 10],
                total_training_points=0,
                points_per_n={},
            ),
        )
        db.register_model(record)

        # Monkey-patch the project root for this test
        import qmbp_simulation.predictors.model_registry_db as registry_module

        original_root = registry_module._PROJECT_ROOT
        registry_module._PROJECT_ROOT = tmp_path

        try:
            enriched = db.enrich_points_per_n()
            assert enriched == 1

            result = db.get_model("test_enrich.pt")
            assert result.training.points_per_n == {"6": 30, "8": 45, "10": 60}
            assert result.training.total_training_points == 135
        finally:
            registry_module._PROJECT_ROOT = original_root


class TestModelRegistryDBHistory:
    """Test history (audit trail) and regression detection."""

    def test_register_creates_history_event(self, tmp_path):
        path = tmp_path / "reg.json"
        hist_path = tmp_path / "hist.json"
        db = ModelRegistryDB(path=path, history_path=hist_path)

        record = ModelRecord(
            model_id="new_model.pt",
            model_name="tfim_bond_resolved",
            topology="chain_1d",
            training=TrainingProvenance(
                n_values_used=[6, 8],
                total_training_points=80,
            ),
            runner_tag="AC",
        )
        db.register_model(record)

        history = db.get_history(model_id="new_model.pt")
        assert len(history) == 1
        assert history[0].event_type == "registered"
        assert history[0].details["training_points"] == 80
        assert history[0].details["runner_tag"] == "AC"

    def test_overwrite_creates_retrained_event(self, tmp_path):
        path = tmp_path / "reg.json"
        hist_path = tmp_path / "hist.json"
        db = ModelRegistryDB(path=path, history_path=hist_path)

        record = ModelRecord(
            model_id="model_v1.pt",
            model_name="tfim_bond_resolved",
            topology="chain_1d",
            training=TrainingProvenance(
                n_values_used=[6, 8],
                total_training_points=80,
            ),
        )
        db.register_model(record)

        # Overwrite with new version
        record_v2 = ModelRecord(
            model_id="model_v1.pt",
            model_name="tfim_bond_resolved",
            topology="chain_1d",
            training=TrainingProvenance(
                n_values_used=[6, 8, 10],
                total_training_points=150,
            ),
        )
        db.register_model(record_v2, overwrite=True)

        history = db.get_history(model_id="model_v1.pt")
        assert len(history) == 2
        retrain_event = next(e for e in history if e.event_type == "retrained")
        assert retrain_event.details["old_training_points"] == 80
        assert retrain_event.details["new_training_points"] == 150

    def test_evaluation_creates_history_event(self, tmp_path):
        path = tmp_path / "reg.json"
        hist_path = tmp_path / "hist.json"
        db = ModelRegistryDB(path=path, history_path=hist_path)

        record = ModelRecord(
            model_id="eval_model.pt",
            model_name="tfim_bond_resolved",
            topology="ladder",
            training=TrainingProvenance(total_training_points=100),
        )
        db.register_model(record)

        evaluation = EvaluationRecord(
            evaluated_at="2026-08-12T12:00:00+00:00",
            target_n_values=[20],
            pass_rate_dual=0.80,
            mean_de_gap=0.03,
            notes="First eval",
        )
        db.add_evaluation("eval_model.pt", evaluation)

        history = db.get_history(event_type="evaluated")
        assert len(history) == 1
        assert history[0].details["pass_rate_dual"] == 0.80

    def test_regression_detection_in_evaluations(self, tmp_path):
        path = tmp_path / "reg.json"
        hist_path = tmp_path / "hist.json"
        db = ModelRegistryDB(path=path, history_path=hist_path)

        record = ModelRecord(
            model_id="regressing_model.pt",
            model_name="tfim_bond_resolved",
            topology="chain_1d",
            training=TrainingProvenance(total_training_points=100),
        )
        db.register_model(record)

        # First eval — good
        db.add_evaluation(
            "regressing_model.pt",
            EvaluationRecord(
                evaluated_at="2026-08-10T00:00:00+00:00",
                pass_rate_dual=0.85,
            ),
        )

        # Second eval — regression!
        db.add_evaluation(
            "regressing_model.pt",
            EvaluationRecord(
                evaluated_at="2026-08-12T00:00:00+00:00",
                pass_rate_dual=0.50,
            ),
        )

        # Should have regression event in history
        regression_events = db.get_history(event_type="regression_detected")
        assert len(regression_events) == 1
        assert regression_events[0].details["prev_best_pass_rate"] == 0.85
        assert regression_events[0].details["new_pass_rate"] == 0.50

    def test_detect_regressions_method(self, tmp_path):
        path = tmp_path / "reg.json"
        hist_path = tmp_path / "hist.json"
        db = ModelRegistryDB(path=path, history_path=hist_path)

        record = ModelRecord(
            model_id="model_with_regression.pt",
            model_name="tfim_bond_resolved",
            topology="square",
            training=TrainingProvenance(total_training_points=100),
            evaluations=[
                EvaluationRecord(pass_rate_dual=0.90),
                EvaluationRecord(pass_rate_dual=0.70),  # Regression
            ],
        )
        db.register_model(record)

        regressions = db.detect_regressions(threshold=0.05)
        assert len(regressions) == 1
        assert regressions[0]["prev_pass_rate"] == 0.90
        assert regressions[0]["curr_pass_rate"] == 0.70

    def test_no_regression_within_threshold(self, tmp_path):
        path = tmp_path / "reg.json"
        hist_path = tmp_path / "hist.json"
        db = ModelRegistryDB(path=path, history_path=hist_path)

        record = ModelRecord(
            model_id="stable_model.pt",
            model_name="tfim_bond_resolved",
            topology="chain_1d",
            training=TrainingProvenance(total_training_points=100),
            evaluations=[
                EvaluationRecord(pass_rate_dual=0.85),
                EvaluationRecord(pass_rate_dual=0.82),  # Within 5% threshold
            ],
        )
        db.register_model(record)

        regressions = db.detect_regressions(threshold=0.05)
        assert len(regressions) == 0

    def test_superseded_creates_history_event(self, tmp_path):
        path = tmp_path / "reg.json"
        hist_path = tmp_path / "hist.json"
        db = ModelRegistryDB(path=path, history_path=hist_path)

        record = ModelRecord(
            model_id="old_model.pt",
            model_name="tfim_bond_resolved",
            topology="chain_1d",
        )
        db.register_model(record)
        db.mark_superseded("old_model.pt", superseded_by="new_model.pt")

        history = db.get_history(event_type="superseded")
        assert len(history) == 1
        assert history[0].details["superseded_by"] == "new_model.pt"

    def test_archive_creates_history_event(self, tmp_path):
        path = tmp_path / "reg.json"
        hist_path = tmp_path / "hist.json"
        db = ModelRegistryDB(path=path, history_path=hist_path)

        record = ModelRecord(
            model_id="to_archive.pt",
            model_name="tfim_bond_resolved",
            topology="ladder",
        )
        db.register_model(record)
        db.archive("to_archive.pt")

        history = db.get_history(event_type="archived")
        assert len(history) == 1
        assert history[0].model_id == "to_archive.pt"

    def test_get_model_timeline(self, tmp_path):
        path = tmp_path / "reg.json"
        hist_path = tmp_path / "hist.json"
        db = ModelRegistryDB(path=path, history_path=hist_path)

        record = ModelRecord(
            model_id="timeline_model.pt",
            model_name="tfim_bond_resolved",
            topology="chain_1d",
            training=TrainingProvenance(total_training_points=100),
        )
        db.register_model(record)
        db.add_evaluation("timeline_model.pt", EvaluationRecord(pass_rate_dual=0.80))
        db.add_evaluation("timeline_model.pt", EvaluationRecord(pass_rate_dual=0.90))

        timeline = db.get_model_timeline("timeline_model.pt")
        assert len(timeline) == 3  # registered + 2 evaluations
        assert timeline[0]["event"] == "registered"
        assert timeline[1]["event"] == "evaluated"
        assert timeline[2]["event"] == "evaluated"

    def test_history_summary(self, tmp_path):
        path = tmp_path / "reg.json"
        hist_path = tmp_path / "hist.json"
        db = ModelRegistryDB(path=path, history_path=hist_path)

        record = ModelRecord(
            model_id="summary_model.pt",
            model_name="tfim_bond_resolved",
            topology="chain_1d",
            training=TrainingProvenance(total_training_points=50),
        )
        db.register_model(record)
        db.add_evaluation("summary_model.pt", EvaluationRecord(pass_rate_dual=0.70))

        summary = db.history_summary()
        assert summary["total_events"] == 2
        assert summary["event_types"]["registered"] == 1
        assert summary["event_types"]["evaluated"] == 1

    def test_history_persists_across_instances(self, tmp_path):
        path = tmp_path / "reg.json"
        hist_path = tmp_path / "hist.json"

        db1 = ModelRegistryDB(path=path, history_path=hist_path)
        record = ModelRecord(
            model_id="persist_test.pt",
            model_name="tfim_bond_resolved",
            topology="chain_1d",
        )
        db1.register_model(record)

        # New instance should see the history
        db2 = ModelRegistryDB(path=path, history_path=hist_path)
        history = db2.get_history(model_id="persist_test.pt")
        assert len(history) == 1
        assert history[0].event_type == "registered"

    def test_history_filter_by_topology(self, tmp_path):
        path = tmp_path / "reg.json"
        hist_path = tmp_path / "hist.json"
        db = ModelRegistryDB(path=path, history_path=hist_path)

        db.register_model(
            ModelRecord(
                model_id="chain_model.pt",
                model_name="tfim",
                topology="chain_1d",
            )
        )
        db.register_model(
            ModelRecord(
                model_id="ladder_model.pt",
                model_name="tfim",
                topology="ladder",
            )
        )

        chain_events = db.get_history(topology="chain_1d")
        assert len(chain_events) == 1
        assert chain_events[0].model_id == "chain_model.pt"


class TestModelRegistryDBEdgeCases:
    """Edge cases and boundary conditions."""

    def test_register_model_with_empty_training(self, tmp_path):
        """Model with no training provenance still registers fine."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        record = ModelRecord(
            model_id="empty_training.pt",
            model_name="tfim",
            topology="chain_1d",
            training=TrainingProvenance(),
        )
        db.register_model(record)
        result = db.get_model("empty_training.pt")
        assert result.training.n_values_used == []
        assert result.training.total_training_points == 0
        assert result.training.points_per_n == {}

    def test_register_model_with_special_chars_in_notes(self, tmp_path):
        """Notes with unicode and special characters survive roundtrip."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        notes = "ΔE/gap < 5% ✓ — trained with h∈[2.0, 5.0] × N=[4,6,8]"
        record = ModelRecord(
            model_id="unicode_model.pt",
            model_name="tfim_bond_resolved",
            topology="chain_1d",
            notes=notes,
        )
        db.register_model(record)

        # Reload from disk
        db2 = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        result = db2.get_model("unicode_model.pt")
        assert result.notes == notes

    def test_query_p_layers_zero(self, tmp_path):
        """p_layers=0 is a valid query (but unlikely to match)."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="p1.pt",
                model_name="tfim",
                topology="chain_1d",
                p_layers=1,
            )
        )
        results = db.query(p_layers=0)
        assert len(results) == 0

    def test_query_multiple_topologies_separate_calls(self, tmp_path):
        """Querying different topologies returns disjoint sets."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="chain.pt",
                model_name="tfim",
                topology="chain_1d",
            )
        )
        db.register_model(
            ModelRecord(
                model_id="ladder.pt",
                model_name="tfim",
                topology="ladder",
            )
        )
        db.register_model(
            ModelRecord(
                model_id="square.pt",
                model_name="tfim",
                topology="square",
            )
        )

        chain = db.query(topology="chain_1d")
        ladder = db.query(topology="ladder")
        square = db.query(topology="square")
        assert len(chain) == 1 and chain[0].model_id == "chain.pt"
        assert len(ladder) == 1 and ladder[0].model_id == "ladder.pt"
        assert len(square) == 1 and square[0].model_id == "square.pt"

    def test_register_many_models_performance(self, tmp_path):
        """Registry handles 100+ models without issue."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        for i in range(100):
            db.register_model(
                ModelRecord(
                    model_id=f"model_{i:03d}.pt",
                    model_name="tfim_bond_resolved",
                    topology=["chain_1d", "ladder", "square", "heavy_hex"][i % 4],
                    p_layers=1,
                    training=TrainingProvenance(total_training_points=i * 10),
                )
            )

        assert len(db.list_all()) == 100
        heavy_hex = db.query(topology="heavy_hex")
        assert len(heavy_hex) == 25
        large = db.query(min_training_points=500)
        assert len(large) == 50  # i >= 50 → pts >= 500

    def test_summary_with_no_n_values(self, tmp_path):
        """Summary handles models with empty n_values_used gracefully."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="no_n.pt",
                model_name="tfim",
                topology="chain_1d",
                training=TrainingProvenance(n_values_used=[], total_training_points=50),
            )
        )
        s = db.summary()
        assert s["active_models"] == 1
        assert s["max_n_trained"] == 0

    def test_overwrite_preserves_evaluations_in_history(self, tmp_path):
        """When overwriting, old evaluation pass_rate is captured in retrain event."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        record_v1 = ModelRecord(
            model_id="evolving.pt",
            model_name="tfim_bond_resolved",
            topology="chain_1d",
            training=TrainingProvenance(total_training_points=50),
            evaluations=[EvaluationRecord(pass_rate_dual=0.80)],
        )
        db.register_model(record_v1)

        record_v2 = ModelRecord(
            model_id="evolving.pt",
            model_name="tfim_bond_resolved",
            topology="chain_1d",
            training=TrainingProvenance(total_training_points=100),
        )
        db.register_model(record_v2, overwrite=True)

        retrain_events = db.get_history(model_id="evolving.pt", event_type="retrained")
        assert len(retrain_events) == 1
        assert retrain_events[0].details["old_pass_rate"] == 0.80


class TestModelRegistryDBMultiModelRegressions:
    """Test regression detection across multiple models."""

    def test_regressions_only_for_active_models(self, tmp_path):
        """Archived models don't appear in regression scan."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")

        # Active model with regression
        db.register_model(
            ModelRecord(
                model_id="active_reg.pt",
                model_name="tfim",
                topology="chain_1d",
                evaluations=[
                    EvaluationRecord(pass_rate_dual=0.90),
                    EvaluationRecord(pass_rate_dual=0.60),
                ],
            )
        )
        # Archived model with regression (should be ignored)
        db.register_model(
            ModelRecord(
                model_id="archived_reg.pt",
                model_name="tfim",
                topology="ladder",
                status="archived",
                evaluations=[
                    EvaluationRecord(pass_rate_dual=0.95),
                    EvaluationRecord(pass_rate_dual=0.40),
                ],
            )
        )

        regressions = db.detect_regressions()
        assert len(regressions) == 1
        assert regressions[0]["model_id"] == "active_reg.pt"

    def test_multiple_regressions_in_sequence(self, tmp_path):
        """Multiple consecutive drops each count as a regression."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="declining.pt",
                model_name="tfim",
                topology="chain_1d",
                evaluations=[
                    EvaluationRecord(pass_rate_dual=0.90),
                    EvaluationRecord(pass_rate_dual=0.80),  # 10% drop
                    EvaluationRecord(pass_rate_dual=0.60),  # 20% drop
                ],
            )
        )

        regressions = db.detect_regressions(threshold=0.05)
        assert len(regressions) == 2

    def test_recovery_after_regression_not_flagged(self, tmp_path):
        """A recovery (improvement) after regression is not flagged."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="recovery.pt",
                model_name="tfim",
                topology="chain_1d",
                evaluations=[
                    EvaluationRecord(pass_rate_dual=0.90),
                    EvaluationRecord(pass_rate_dual=0.70),  # Regression
                    EvaluationRecord(pass_rate_dual=0.85),  # Recovery (not a regression)
                ],
            )
        )

        regressions = db.detect_regressions(threshold=0.05)
        # Only the drop from 0.90 → 0.70 is a regression
        assert len(regressions) == 1
        assert regressions[0]["prev_pass_rate"] == 0.90
        assert regressions[0]["curr_pass_rate"] == 0.70

    def test_custom_threshold(self, tmp_path):
        """Threshold parameter controls sensitivity."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="borderline.pt",
                model_name="tfim",
                topology="chain_1d",
                evaluations=[
                    EvaluationRecord(pass_rate_dual=0.80),
                    EvaluationRecord(pass_rate_dual=0.72),  # 8% drop
                ],
            )
        )

        # With 10% threshold → not a regression
        assert len(db.detect_regressions(threshold=0.10)) == 0
        # With 5% threshold → is a regression
        assert len(db.detect_regressions(threshold=0.05)) == 1


class TestModelRegistryDBHistoryQueries:
    """Advanced history query scenarios."""

    def test_history_limit_returns_most_recent(self, tmp_path):
        """Limit returns the N most recent events."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        for i in range(10):
            db.register_model(
                ModelRecord(
                    model_id=f"model_{i}.pt",
                    model_name="tfim",
                    topology="chain_1d",
                )
            )

        recent = db.get_history(limit=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0].model_id == "model_9.pt"
        assert recent[1].model_id == "model_8.pt"
        assert recent[2].model_id == "model_7.pt"

    def test_history_combined_filters(self, tmp_path):
        """Multiple filters combine with AND logic."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="chain_a.pt",
                model_name="tfim",
                topology="chain_1d",
            )
        )
        db.register_model(
            ModelRecord(
                model_id="ladder_a.pt",
                model_name="tfim",
                topology="ladder",
            )
        )
        db.add_evaluation("chain_a.pt", EvaluationRecord(pass_rate_dual=0.80))

        # Only chain_1d + evaluated
        chain_evals = db.get_history(topology="chain_1d", event_type="evaluated")
        assert len(chain_evals) == 1

        # Ladder + evaluated → nothing
        ladder_evals = db.get_history(topology="ladder", event_type="evaluated")
        assert len(ladder_evals) == 0

    def test_timeline_empty_for_unknown_model(self, tmp_path):
        """Timeline for non-existent model returns empty list."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        timeline = db.get_model_timeline("nonexistent.pt")
        assert timeline == []

    def test_history_summary_with_no_events(self, tmp_path):
        """Empty history gives summary with zero counts."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        s = db.history_summary()
        assert s["total_events"] == 0

    def test_full_lifecycle_history(self, tmp_path):
        """Complete lifecycle: register → evaluate → supersede produces correct timeline."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")

        db.register_model(
            ModelRecord(
                model_id="lifecycle.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                training=TrainingProvenance(total_training_points=100),
            )
        )
        db.add_evaluation(
            "lifecycle.pt",
            EvaluationRecord(
                pass_rate_dual=0.85,
                target_n_values=[20],
            ),
        )
        db.mark_superseded("lifecycle.pt", superseded_by="lifecycle_v2.pt")

        timeline = db.get_model_timeline("lifecycle.pt")
        assert len(timeline) == 3
        event_types = [e["event"] for e in timeline]
        assert event_types == ["registered", "evaluated", "superseded"]

    def test_regression_detected_event_includes_target_n(self, tmp_path):
        """Regression event details include target_n when available."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="regr_n.pt",
                model_name="tfim",
                topology="chain_1d",
                training=TrainingProvenance(total_training_points=50),
            )
        )
        db.add_evaluation(
            "regr_n.pt",
            EvaluationRecord(
                pass_rate_dual=0.90,
                target_n_values=[20, 30],
            ),
        )
        db.add_evaluation(
            "regr_n.pt",
            EvaluationRecord(
                pass_rate_dual=0.50,
                target_n_values=[40],
            ),
        )

        reg_events = db.get_history(event_type="regression_detected")
        assert len(reg_events) == 1
        assert reg_events[0].details["target_n"] == [40]


class TestModelRegistryDBConcurrency:
    """Test behavior with concurrent-like patterns (sequential but simulating multi-access)."""

    def test_two_instances_share_state(self, tmp_path):
        """Two DB instances pointing to same file see each other's writes."""
        path = tmp_path / "shared.json"
        hist = tmp_path / "shared_hist.json"

        db1 = ModelRegistryDB(path=path, history_path=hist)
        db1.register_model(
            ModelRecord(
                model_id="from_db1.pt",
                model_name="tfim",
                topology="chain_1d",
            )
        )

        # db2 loads fresh from disk
        db2 = ModelRegistryDB(path=path, history_path=hist)
        assert db2.get_model("from_db1.pt") is not None

        db2.register_model(
            ModelRecord(
                model_id="from_db2.pt",
                model_name="tfim",
                topology="ladder",
            )
        )

        # db1 needs to reload to see db2's changes (no live sync)
        db1_fresh = ModelRegistryDB(path=path, history_path=hist)
        assert db1_fresh.get_model("from_db2.pt") is not None

    def test_rapid_evaluations_all_persisted(self, tmp_path):
        """Multiple rapid evaluations are all correctly persisted."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="rapid.pt",
                model_name="tfim",
                topology="chain_1d",
            )
        )

        for i in range(20):
            db.add_evaluation(
                "rapid.pt",
                EvaluationRecord(
                    pass_rate_dual=0.5 + i * 0.02,
                    notes=f"eval_{i}",
                ),
            )

        # Reload and verify
        db2 = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        result = db2.get_model("rapid.pt")
        assert len(result.evaluations) == 20
        assert result.evaluations[-1].pass_rate_dual == pytest.approx(0.88)


class TestModelRegistryDBSerialization:
    """Test serialization correctness for various data types."""

    def test_empty_evaluations_list_survives_roundtrip(self, tmp_path):
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="no_evals.pt",
                model_name="tfim",
                topology="chain_1d",
                evaluations=[],
            )
        )
        db2 = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        result = db2.get_model("no_evals.pt")
        assert result.evaluations == []

    def test_large_points_per_n_dict(self, tmp_path):
        """points_per_n with many keys serializes correctly."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        ppn = {str(n): n * 10 for n in range(4, 52, 2)}  # 24 entries
        db.register_model(
            ModelRecord(
                model_id="large_ppn.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                training=TrainingProvenance(
                    n_values_used=list(range(4, 52, 2)),
                    total_training_points=sum(ppn.values()),
                    points_per_n=ppn,
                ),
            )
        )

        db2 = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        result = db2.get_model("large_ppn.pt")
        assert len(result.training.points_per_n) == 24
        assert result.training.points_per_n["10"] == 100

    def test_float_precision_in_h_range(self, tmp_path):
        """h_range floats maintain precision through serialization."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="precise.pt",
                model_name="tfim",
                topology="chain_1d",
                training=TrainingProvenance(h_range=(1.23456789, 4.98765432)),
            )
        )

        db2 = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        result = db2.get_model("precise.pt")
        assert result.training.h_range[0] == pytest.approx(1.23456789)
        assert result.training.h_range[1] == pytest.approx(4.98765432)

    def test_evaluation_with_zero_values(self, tmp_path):
        """EvaluationRecord with all zeros is valid and serializes."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="zero_eval.pt",
                model_name="tfim",
                topology="chain_1d",
            )
        )
        db.add_evaluation(
            "zero_eval.pt",
            EvaluationRecord(
                pass_rate_5pct=0.0,
                pass_rate_dual=0.0,
                mean_de_gap=0.0,
                mean_abs_error_per_site=0.0,
            ),
        )

        db2 = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        result = db2.get_model("zero_eval.pt")
        assert len(result.evaluations) == 1
        assert result.evaluations[0].pass_rate_dual == 0.0

    def test_model_id_with_dots_and_underscores(self, tmp_path):
        """Model IDs with complex naming patterns work."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        complex_id = "unified_tfim_br_heavy_hex_multiN_4+6+10+12+16_p1_v2.3_final.pt"
        db.register_model(
            ModelRecord(
                model_id=complex_id,
                model_name="tfim_bond_resolved",
                topology="heavy_hex",
            )
        )
        result = db.get_model(complex_id)
        assert result is not None
        assert result.model_id == complex_id


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for new improvements (1, 3, 4, 8, 9)
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrainingMetrics:
    """Test TrainingMetrics tracking (Improvement #1)."""

    def test_set_training_metrics(self, tmp_path):
        """set_training_metrics updates metrics correctly."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="metrics_model.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
            )
        )

        success = db.set_training_metrics(
            "metrics_model.pt",
            final_loss=0.0015,
            final_mse=0.002,
            epochs=150,
            best_epoch=120,
            training_time_seconds=45.3,
            learning_rate=0.001,
            batch_size=32,
            early_stopped=True,
            convergence_status="converged",
        )
        assert success

        result = db.get_model("metrics_model.pt")
        tm = result.training.training_metrics
        assert tm.final_loss == pytest.approx(0.0015)
        assert tm.final_mse == pytest.approx(0.002)
        assert tm.epochs == 150
        assert tm.best_epoch == 120
        assert tm.early_stopped is True
        assert tm.convergence_status == "converged"

    def test_training_metrics_persist(self, tmp_path):
        """Training metrics survive persistence roundtrip."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="persist_metrics.pt",
                model_name="tfim",
                topology="ladder",
                training=TrainingProvenance(
                    training_metrics=TrainingMetrics(
                        final_loss=0.001,
                        epochs=100,
                        convergence_status="plateau",
                        loss_history=[0.1, 0.05, 0.02, 0.01, 0.001],
                    )
                ),
            )
        )

        db2 = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        result = db2.get_model("persist_metrics.pt")
        tm = result.training.training_metrics
        assert tm.final_loss == pytest.approx(0.001)
        assert tm.epochs == 100
        assert tm.convergence_status == "plateau"
        assert len(tm.loss_history) == 5

    def test_set_training_metrics_nonexistent_model(self, tmp_path):
        """set_training_metrics returns False for nonexistent model."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        success = db.set_training_metrics("nonexistent.pt", epochs=10)
        assert success is False


class TestTaggingSystem:
    """Test tagging functionality (Improvement #3)."""

    def test_add_tag(self, tmp_path):
        """add_tag adds a tag successfully."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="tag_model.pt",
                model_name="tfim",
                topology="chain_1d",
            )
        )

        assert db.add_tag("tag_model.pt", "production") is True
        result = db.get_model("tag_model.pt")
        assert "production" in result.tags

    def test_add_duplicate_tag(self, tmp_path):
        """add_tag returns False for duplicate tag."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="dup_tag.pt",
                model_name="tfim",
                topology="chain_1d",
                tags=["existing"],
            )
        )

        assert db.add_tag("dup_tag.pt", "existing") is False

    def test_remove_tag(self, tmp_path):
        """remove_tag removes a tag."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="remove_tag.pt",
                model_name="tfim",
                topology="chain_1d",
                tags=["to_remove", "keep"],
            )
        )

        assert db.remove_tag("remove_tag.pt", "to_remove") is True
        result = db.get_model("remove_tag.pt")
        assert "to_remove" not in result.tags
        assert "keep" in result.tags

    def test_query_by_tag(self, tmp_path):
        """query_by_tag finds models with specific tag."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="tagged1.pt",
                model_name="tfim",
                topology="chain_1d",
                tags=["production", "validated"],
            )
        )
        db.register_model(
            ModelRecord(
                model_id="tagged2.pt",
                model_name="tfim",
                topology="ladder",
                tags=["production"],
            )
        )
        db.register_model(
            ModelRecord(
                model_id="untagged.pt",
                model_name="tfim",
                topology="square",
            )
        )

        production = db.query_by_tag("production")
        assert len(production) == 2
        assert {r.model_id for r in production} == {"tagged1.pt", "tagged2.pt"}

        validated = db.query_by_tag("validated")
        assert len(validated) == 1

    def test_list_all_tags(self, tmp_path):
        """list_all_tags returns tag counts."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="m1.pt",
                model_name="tfim",
                topology="chain_1d",
                tags=["a", "b"],
            )
        )
        db.register_model(
            ModelRecord(
                model_id="m2.pt",
                model_name="tfim",
                topology="ladder",
                tags=["a", "c"],
            )
        )
        db.register_model(
            ModelRecord(
                model_id="m3.pt",
                model_name="tfim",
                topology="square",
                tags=["a"],
            )
        )

        all_tags = db.list_all_tags()
        assert all_tags["a"] == 3
        assert all_tags["b"] == 1
        assert all_tags["c"] == 1

    def test_tag_events_in_history(self, tmp_path):
        """Tag operations create history events."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="tag_hist.pt",
                model_name="tfim",
                topology="chain_1d",
            )
        )
        db.add_tag("tag_hist.pt", "test_tag")
        db.remove_tag("tag_hist.pt", "test_tag")

        history = db.get_history(model_id="tag_hist.pt")
        event_types = [e.event_type for e in history]
        assert "tag_added" in event_types
        assert "tag_removed" in event_types


class TestBestModelSelection:
    """Test get_best_for_deployment (Improvement #4)."""

    def test_prefers_multi_n_with_good_pass_rate(self, tmp_path):
        """Prefers multi-N model with good pass_rate."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")

        # Single-N model
        db.register_model(
            ModelRecord(
                model_id="single_n.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                p_layers=1,
                training=TrainingProvenance(
                    n_values_used=[10],
                    total_training_points=50,
                ),
                evaluations=[EvaluationRecord(pass_rate_dual=0.90)],
            )
        )

        # Multi-N model with good pass_rate
        db.register_model(
            ModelRecord(
                model_id="multi_n.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                p_layers=1,
                training=TrainingProvenance(
                    n_values_used=[6, 8, 10],
                    total_training_points=150,
                ),
                evaluations=[EvaluationRecord(pass_rate_dual=0.85)],
            )
        )

        best = db.get_best_for_deployment(
            topology="chain_1d",
            model_name="tfim_bond_resolved",
            p_layers=1,
            n_target=20,
        )
        assert best.model_id == "multi_n.pt"

    def test_falls_back_to_single_n_when_multi_n_poor(self, tmp_path):
        """Falls back to single-N when multi-N has poor pass_rate."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")

        # Single-N model with good pass_rate
        db.register_model(
            ModelRecord(
                model_id="good_single.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                p_layers=1,
                training=TrainingProvenance(
                    n_values_used=[10],
                    total_training_points=50,
                ),
                evaluations=[EvaluationRecord(pass_rate_dual=0.90)],
            )
        )

        # Multi-N model with poor pass_rate (below 0.40 threshold)
        db.register_model(
            ModelRecord(
                model_id="bad_multi.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                p_layers=1,
                training=TrainingProvenance(
                    n_values_used=[6, 8, 10],
                    total_training_points=150,
                ),
                evaluations=[EvaluationRecord(pass_rate_dual=0.30)],
            )
        )

        best = db.get_best_for_deployment(
            topology="chain_1d",
            model_name="tfim_bond_resolved",
            p_layers=1,
            n_target=20,
            min_pass_rate=0.40,
        )
        # With strict min_pass_rate, should prefer single-N or skip multi-N
        # Since multi-N has 0.30 < 0.40, and single-N has 0.90, should get single-N
        assert best.model_id == "good_single.pt"

    def test_require_tag_filter(self, tmp_path):
        """require_tag filters candidates."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")

        db.register_model(
            ModelRecord(
                model_id="tagged.pt",
                model_name="tfim",
                topology="chain_1d",
                p_layers=1,
                tags=["production"],
                training=TrainingProvenance(n_values_used=[10], total_training_points=50),
            )
        )
        db.register_model(
            ModelRecord(
                model_id="untagged.pt",
                model_name="tfim",
                topology="chain_1d",
                p_layers=1,
                training=TrainingProvenance(n_values_used=[10], total_training_points=100),
            )
        )

        # With tag filter, only tagged model considered
        best = db.get_best_for_deployment(
            topology="chain_1d",
            model_name="tfim",
            p_layers=1,
            n_target=20,
            require_tag="production",
        )
        assert best.model_id == "tagged.pt"

    def test_returns_none_when_no_candidates(self, tmp_path):
        """Returns None when no models match."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="wrong_topo.pt",
                model_name="tfim",
                topology="ladder",
                p_layers=1,
            )
        )

        best = db.get_best_for_deployment(
            topology="chain_1d",
            model_name="tfim",
            p_layers=1,
            n_target=20,
        )
        assert best is None


class TestIntegrityValidation:
    """Test integrity validation (Improvement #8)."""

    def test_validate_missing_checkpoint(self, tmp_path):
        """Detects missing checkpoint file."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="missing_ckpt.pt",
                model_name="tfim",
                topology="chain_1d",
            )
        )

        report = db.validate_integrity("missing_ckpt.pt")
        assert report.checkpoint_exists is False
        assert report.all_ok is False
        assert any("missing" in i.lower() for i in report.issues)

    def test_validate_integrity_creates_history_event(self, tmp_path):
        """Validation creates history event."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="validate_hist.pt",
                model_name="tfim",
                topology="chain_1d",
            )
        )

        db.validate_integrity("validate_hist.pt")

        history = db.get_history(event_type="integrity_checked")
        assert len(history) == 1
        assert history[0].model_id == "validate_hist.pt"

    def test_validate_all_integrity(self, tmp_path):
        """validate_all_integrity checks all active models."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="model1.pt",
                model_name="tfim",
                topology="chain_1d",
            )
        )
        db.register_model(
            ModelRecord(
                model_id="model2.pt",
                model_name="tfim",
                topology="ladder",
            )
        )
        db.register_model(
            ModelRecord(
                model_id="archived.pt",
                model_name="tfim",
                topology="square",
                status="archived",
            )
        )

        results = db.validate_all_integrity()
        assert results["n_checked"] == 2  # Excludes archived
        assert results["n_issues"] == 2  # Both missing checkpoints


class TestDashboardIntegration:
    """Test dashboard integration (Improvement #9)."""

    def test_dashboard_quality_default(self, tmp_path):
        """DashboardQuality initializes with defaults."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="default_dq.pt",
                model_name="tfim",
                topology="chain_1d",
            )
        )

        result = db.get_model("default_dq.pt")
        assert result.dashboard_quality.training_utility == ""
        assert result.dashboard_quality.needs_retrain is False

    def test_dashboard_quality_persists(self, tmp_path):
        """DashboardQuality survives roundtrip."""
        from qmbp_simulation.predictors.model_registry_db import DashboardQuality

        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="dq_persist.pt",
                model_name="tfim",
                topology="chain_1d",
                dashboard_quality=DashboardQuality(
                    training_utility="useful",
                    training_utility_reason="Test reason",
                    pass_rate_dual_criterion=0.85,
                    needs_retrain=True,
                    h_frontier=2.5,
                ),
            )
        )

        db2 = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        result = db2.get_model("dq_persist.pt")
        assert result.dashboard_quality.training_utility == "useful"
        assert result.dashboard_quality.needs_retrain is True
        assert result.dashboard_quality.h_frontier == pytest.approx(2.5)

    def test_get_training_health(self, tmp_path):
        """get_training_health returns correct summary."""
        from qmbp_simulation.predictors.model_registry_db import DashboardQuality

        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="health_model.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                dashboard_quality=DashboardQuality(
                    training_utility="useful",
                    needs_retrain=False,
                    model_stale=False,
                ),
                evaluations=[EvaluationRecord(pass_rate_dual=0.85)],
            )
        )

        health = db.get_training_health("health_model.pt")
        assert health["recommendation"] == "use"
        assert health["quality_issues"] == []

    def test_get_training_health_flags_issues(self, tmp_path):
        """get_training_health detects quality issues."""
        from qmbp_simulation.predictors.model_registry_db import DashboardQuality

        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="unhealthy.pt",
                model_name="tfim",
                topology="chain_1d",
                dashboard_quality=DashboardQuality(
                    training_utility="not_useful",
                    training_utility_reason="Contaminated data",
                    needs_retrain=True,
                ),
                evaluations=[EvaluationRecord(pass_rate_dual=0.25)],
            )
        )

        health = db.get_training_health("unhealthy.pt")
        assert health["recommendation"] == "retrain"
        assert len(health["quality_issues"]) >= 2  # Multiple issues


# ═══════════════════════════════════════════════════════════════════════════════
# Tests for Model Versioning (Improvement #5)
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelVersioning:
    """Test model versioning functionality (Improvement #5)."""

    def test_new_model_starts_at_version_1(self, tmp_path):
        """New models start at version 1."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="new_model.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
            )
        )

        result = db.get_model("new_model.pt")
        assert result.version == 1
        assert result.version_history == []

    def test_overwrite_increments_version(self, tmp_path):
        """Overwriting a model increments its version."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")

        # Register v1
        db.register_model(
            ModelRecord(
                model_id="versioned.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                training=TrainingProvenance(total_training_points=50),
            )
        )

        # Overwrite → v2
        db.register_model(
            ModelRecord(
                model_id="versioned.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                training=TrainingProvenance(total_training_points=100),
            ),
            overwrite=True,
        )

        result = db.get_model("versioned.pt")
        assert result.version == 2
        assert len(result.version_history) == 1
        assert result.version_history[0] == "versioned.pt"

    def test_multiple_overwrites_chain_versions(self, tmp_path):
        """Multiple overwrites create a version chain."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")

        # v1
        db.register_model(
            ModelRecord(
                model_id="multi_ver.pt",
                model_name="tfim",
                topology="chain_1d",
                training=TrainingProvenance(total_training_points=30),
            )
        )

        # v2
        db.register_model(
            ModelRecord(
                model_id="multi_ver.pt",
                model_name="tfim",
                topology="chain_1d",
                training=TrainingProvenance(total_training_points=60),
            ),
            overwrite=True,
        )

        # v3
        db.register_model(
            ModelRecord(
                model_id="multi_ver.pt",
                model_name="tfim",
                topology="chain_1d",
                training=TrainingProvenance(total_training_points=90),
            ),
            overwrite=True,
        )

        result = db.get_model("multi_ver.pt")
        assert result.version == 3
        assert len(result.version_history) == 2
        assert result.version_history == ["multi_ver.pt", "multi_ver.pt"]

    def test_version_info_includes_correct_fields(self, tmp_path):
        """get_version_info returns all expected fields."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="info_model.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                created="2026-08-12T10:00:00+00:00",
            )
        )

        info = db.get_version_info("info_model.pt")
        assert info is not None
        assert info["model_id"] == "info_model.pt"
        assert info["version"] == 1
        assert info["version_history"] == []
        assert info["is_latest"] is True
        assert info["superseding_model"] is None
        assert info["topology"] == "chain_1d"

    def test_get_version_info_nonexistent(self, tmp_path):
        """get_version_info returns None for nonexistent model."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        info = db.get_version_info("nonexistent.pt")
        assert info is None

    def test_version_chain_builds_correctly(self, tmp_path):
        """get_version_chain returns correct chain."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")

        # Create and overwrite
        db.register_model(
            ModelRecord(
                model_id="chain_model.pt",
                model_name="tfim",
                topology="chain_1d",
                created="2026-08-01T00:00:00+00:00",
                training=TrainingProvenance(total_training_points=40),
            )
        )
        db.register_model(
            ModelRecord(
                model_id="chain_model.pt",
                model_name="tfim",
                topology="chain_1d",
                created="2026-08-10T00:00:00+00:00",
                training=TrainingProvenance(total_training_points=80),
            ),
            overwrite=True,
        )

        chain = db.get_version_chain("chain_model.pt")
        assert len(chain) == 2
        assert chain[0]["version"] == 1
        assert chain[1]["version"] == 2
        assert chain[1]["training_points"] == 80

    def test_version_persists_across_instances(self, tmp_path):
        """Version info survives persistence roundtrip."""
        db1 = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db1.register_model(
            ModelRecord(
                model_id="persist_ver.pt",
                model_name="tfim",
                topology="chain_1d",
            )
        )
        db1.register_model(
            ModelRecord(
                model_id="persist_ver.pt",
                model_name="tfim",
                topology="chain_1d",
                training=TrainingProvenance(total_training_points=100),
            ),
            overwrite=True,
        )

        # Reload
        db2 = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        result = db2.get_model("persist_ver.pt")
        assert result.version == 2
        assert len(result.version_history) == 1

    def test_list_versions_filters_correctly(self, tmp_path):
        """list_versions applies filters correctly."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")

        db.register_model(
            ModelRecord(
                model_id="chain_v1.pt",
                model_name="tfim",
                topology="chain_1d",
                p_layers=1,
            )
        )
        db.register_model(
            ModelRecord(
                model_id="ladder_v1.pt",
                model_name="tfim",
                topology="ladder",
                p_layers=1,
            )
        )
        db.register_model(
            ModelRecord(
                model_id="ladder_v1.pt",
                model_name="tfim",
                topology="ladder",
                p_layers=1,
                training=TrainingProvenance(total_training_points=100),
            ),
            overwrite=True,
        )

        # All
        all_versions = db.list_versions()
        assert len(all_versions) == 2

        # Filter by topology
        chain_only = db.list_versions(topology="chain_1d")
        assert len(chain_only) == 1
        assert chain_only[0]["topology"] == "chain_1d"

        # Ladder should show v2
        ladder_only = db.list_versions(topology="ladder")
        assert len(ladder_only) == 1
        assert ladder_only[0]["version"] == 2

    def test_get_latest_version(self, tmp_path):
        """get_latest_version returns model with highest version."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")

        # Create two different models for same config
        db.register_model(
            ModelRecord(
                model_id="model_a.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                p_layers=1,
                version=1,
            )
        )
        db.register_model(
            ModelRecord(
                model_id="model_b.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                p_layers=1,
                version=3,  # Higher version
            )
        )

        latest = db.get_latest_version("chain_1d", "tfim_bond_resolved", 1)
        assert latest is not None
        assert latest.model_id == "model_b.pt"
        assert latest.version == 3

    def test_get_latest_version_no_match(self, tmp_path):
        """get_latest_version returns None when no match."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="other.pt",
                model_name="tfim",
                topology="ladder",
                p_layers=1,
            )
        )

        latest = db.get_latest_version("chain_1d", "tfim", 1)
        assert latest is None

    def test_version_recorded_in_history(self, tmp_path):
        """Version info is recorded in history events."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")

        db.register_model(
            ModelRecord(
                model_id="hist_ver.pt",
                model_name="tfim",
                topology="chain_1d",
            )
        )
        db.register_model(
            ModelRecord(
                model_id="hist_ver.pt",
                model_name="tfim",
                topology="chain_1d",
                training=TrainingProvenance(total_training_points=100),
            ),
            overwrite=True,
        )

        # Check registered event has version
        registered_events = db.get_history(model_id="hist_ver.pt", event_type="registered")
        assert len(registered_events) == 1
        assert registered_events[0].details.get("version") == 1

        # Check retrained event has both versions
        retrained_events = db.get_history(model_id="hist_ver.pt", event_type="retrained")
        assert len(retrained_events) == 1
        assert retrained_events[0].details.get("old_version") == 1
        assert retrained_events[0].details.get("new_version") == 2

    def test_version_chain_empty_for_v1(self, tmp_path):
        """Version chain for v1 model has only one entry."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="single_ver.pt",
                model_name="tfim",
                topology="chain_1d",
                created="2026-08-12T00:00:00+00:00",
                training=TrainingProvenance(total_training_points=50),
            )
        )

        chain = db.get_version_chain("single_ver.pt")
        assert len(chain) == 1
        assert chain[0]["version"] == 1
        assert chain[0]["model_id"] == "single_ver.pt"

    def test_version_chain_empty_for_nonexistent(self, tmp_path):
        """Version chain for nonexistent model is empty."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        chain = db.get_version_chain("nonexistent.pt")
        assert chain == []


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPipelineIntegration:
    """Test pipeline integration features (TASK 6).

    Tests the integration between:
    - register_checkpoint_with_training_metrics()
    - set_training_metrics() automatic capture
    - Auto-tagging based on pass_rate
    """

    def test_training_metrics_capture_from_dict(self, tmp_path):
        """Training result dict is correctly captured in registry."""
        path = tmp_path / "r.json"
        hist_path = tmp_path / "h.json"
        db = ModelRegistryDB(path=path, history_path=hist_path)

        # Register a model
        db.register_model(
            ModelRecord(
                model_id="metrics_test.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                p_layers=1,
            )
        )

        # Simulate train_unified_mpnn return dict
        training_result = {
            "final_mse": 0.00123,
            "n_epochs_run": 2500,
            "stopped_early": True,
            "stop_reason": "lr_exhausted",
            "mse_history": [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.00123],
        }

        # Set training metrics
        success = db.set_training_metrics(
            "metrics_test.pt",
            final_loss=training_result["final_mse"],
            final_mse=training_result["final_mse"],
            epochs=training_result["n_epochs_run"],
            early_stopped=training_result["stopped_early"],
            convergence_status=training_result["stop_reason"],
            loss_history=training_result["mse_history"],
        )

        assert success is True

        # Verify metrics were stored
        record = db.get_model("metrics_test.pt")
        metrics = record.training.training_metrics
        assert metrics.final_mse == 0.00123
        assert metrics.epochs == 2500
        assert metrics.early_stopped is True
        assert metrics.convergence_status == "lr_exhausted"
        assert len(metrics.loss_history) == 7

    def test_training_metrics_persist_across_reload(self, tmp_path):
        """Training metrics survive JSON roundtrip."""
        path = tmp_path / "r.json"
        hist_path = tmp_path / "h.json"

        # Create and populate
        db1 = ModelRegistryDB(path=path, history_path=hist_path)
        db1.register_model(
            ModelRecord(
                model_id="persist_metrics.pt",
                model_name="tfim_bond_resolved",
                topology="ladder",
            )
        )
        db1.set_training_metrics(
            "persist_metrics.pt",
            final_mse=0.0025,
            epochs=3000,
            convergence_status="converged",
        )

        # Reload and verify
        db2 = ModelRegistryDB(path=path, history_path=hist_path)
        record = db2.get_model("persist_metrics.pt")
        assert record.training.training_metrics.final_mse == 0.0025
        assert record.training.training_metrics.epochs == 3000
        assert record.training.training_metrics.convergence_status == "converged"

    def test_set_training_metrics_nonexistent_model(self, tmp_path):
        """set_training_metrics returns False for nonexistent model."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        success = db.set_training_metrics(
            "nonexistent.pt",
            final_mse=0.001,
        )
        assert success is False

    def test_set_training_metrics_partial_update(self, tmp_path):
        """set_training_metrics only updates provided fields."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="partial.pt",
                model_name="tfim",
                topology="chain_1d",
                training=TrainingProvenance(
                    training_metrics=TrainingMetrics(
                        final_mse=0.01,
                        epochs=1000,
                        learning_rate=0.001,
                    ),
                ),
            )
        )

        # Update only final_mse
        db.set_training_metrics("partial.pt", final_mse=0.005)

        record = db.get_model("partial.pt")
        metrics = record.training.training_metrics
        assert metrics.final_mse == 0.005  # Updated
        assert metrics.epochs == 1000  # Unchanged
        assert metrics.learning_rate == 0.001  # Unchanged


class TestRegisterCheckpointWithTrainingMetrics:
    """Test the new register_checkpoint_with_training_metrics function."""

    def test_truncate_loss_history(self):
        """Loss history truncation preserves first/last 10 + samples."""
        from qmbp_simulation.predictors.model_zoo import _truncate_loss_history

        # Small history — no truncation
        small = [0.1, 0.05, 0.02, 0.01]
        assert _truncate_loss_history(small) == small

        # Exactly 50 — no truncation
        exactly_50 = list(range(50))
        assert _truncate_loss_history(exactly_50) == exactly_50

        # Large history — truncated
        large = list(range(200))
        truncated = _truncate_loss_history(large, max_points=50)

        # Should have <= 50 points
        assert len(truncated) <= 50

        # First 10 preserved
        assert truncated[:10] == list(range(10))

        # Last 10 preserved
        assert truncated[-10:] == list(range(190, 200))

    def test_truncate_loss_history_empty(self):
        """Empty history returns empty."""
        from qmbp_simulation.predictors.model_zoo import _truncate_loss_history

        assert _truncate_loss_history([]) == []
        assert _truncate_loss_history(None) is None


class TestAutoTagging:
    """Test auto-tagging based on pass_rate thresholds."""

    def test_add_tag(self, tmp_path):
        """add_tag correctly adds tags to model."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="tag_test.pt",
                model_name="tfim",
                topology="chain_1d",
            )
        )

        db.add_tag("tag_test.pt", "production")
        db.add_tag("tag_test.pt", "thesis-figure-5")

        record = db.get_model("tag_test.pt")
        assert "production" in record.tags
        assert "thesis-figure-5" in record.tags

    def test_add_tag_duplicate(self, tmp_path):
        """Adding duplicate tag is idempotent."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="dup_tag.pt",
                model_name="tfim",
                topology="chain_1d",
            )
        )

        db.add_tag("dup_tag.pt", "validated")
        db.add_tag("dup_tag.pt", "validated")

        record = db.get_model("dup_tag.pt")
        assert record.tags.count("validated") == 1

    def test_remove_tag(self, tmp_path):
        """remove_tag correctly removes tags."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="remove_tag.pt",
                model_name="tfim",
                topology="chain_1d",
                tags=["production", "validated", "experimental"],
            )
        )

        db.remove_tag("remove_tag.pt", "validated")

        record = db.get_model("remove_tag.pt")
        assert "validated" not in record.tags
        assert "production" in record.tags
        assert "experimental" in record.tags

    def test_query_by_tag(self, tmp_path):
        """query_by_tag finds models with specific tag."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")

        db.register_model(
            ModelRecord(
                model_id="prod1.pt",
                model_name="tfim",
                topology="chain_1d",
                tags=["production", "cross-n-validated"],
            )
        )
        db.register_model(
            ModelRecord(
                model_id="exp1.pt",
                model_name="tfim",
                topology="chain_1d",
                tags=["experimental"],
            )
        )
        db.register_model(
            ModelRecord(
                model_id="prod2.pt",
                model_name="tfim",
                topology="ladder",
                tags=["production"],
            )
        )

        production_models = db.query_by_tag("production")
        assert len(production_models) == 2
        model_ids = {m.model_id for m in production_models}
        assert model_ids == {"prod1.pt", "prod2.pt"}

    def test_list_all_tags(self, tmp_path):
        """list_all_tags returns all unique tags with counts across models."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")

        db.register_model(
            ModelRecord(
                model_id="m1.pt",
                model_name="tfim",
                topology="chain_1d",
                tags=["production", "cross-n-validated"],
            )
        )
        db.register_model(
            ModelRecord(
                model_id="m2.pt",
                model_name="tfim",
                topology="ladder",
                tags=["experimental", "thesis-figure-7"],
            )
        )
        db.register_model(
            ModelRecord(
                model_id="m3.pt",
                model_name="tfim",
                topology="square",
                tags=["production"],
            )
        )

        all_tags = db.list_all_tags()
        # Returns dict[str, int] - tag -> count
        assert "production" in all_tags
        assert all_tags["production"] == 2  # appears in m1 and m3
        assert "cross-n-validated" in all_tags
        assert all_tags["cross-n-validated"] == 1
        assert "experimental" in all_tags
        assert "thesis-figure-7" in all_tags

    def test_tag_events_in_history(self, tmp_path):
        """Tag add/remove creates history events."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="tag_hist.pt",
                model_name="tfim",
                topology="chain_1d",
            )
        )

        db.add_tag("tag_hist.pt", "production")
        db.remove_tag("tag_hist.pt", "production")

        tag_added = db.get_history(event_type="tag_added")
        assert len(tag_added) == 1
        assert tag_added[0].details.get("tag") == "production"

        tag_removed = db.get_history(event_type="tag_removed")
        assert len(tag_removed) == 1
        assert tag_removed[0].details.get("tag") == "production"


class TestBestForDeployment:
    """Test get_best_for_deployment selection logic."""

    def test_get_best_for_deployment_priority(self, tmp_path):
        """get_best_for_deployment follows priority hierarchy."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")

        # Multi-N model with good training points (should be preferred)
        db.register_model(
            ModelRecord(
                model_id="multi_n_model.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                p_layers=1,
                training=TrainingProvenance(
                    n_values_used=[6, 8, 10, 12],
                    total_training_points=400,
                ),
                evaluations=[EvaluationRecord(pass_rate_dual=0.60)],
            )
        )

        # Single-N model with high pass rate
        db.register_model(
            ModelRecord(
                model_id="single_n_model.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                p_layers=1,
                training=TrainingProvenance(
                    n_values_used=[10],
                    total_training_points=100,
                ),
                evaluations=[EvaluationRecord(pass_rate_dual=0.95)],
            )
        )

        # With prefer_multi_n=True (default), multi-N should be selected
        best = db.get_best_for_deployment(
            topology="chain_1d",
            model_name="tfim_bond_resolved",
            p_layers=1,
            n_target=20,
        )

        assert best is not None
        # Multi-N should be preferred when it has acceptable pass_rate
        assert best.model_id == "multi_n_model.pt"

    def test_get_best_for_deployment_no_match(self, tmp_path):
        """get_best_for_deployment returns None when no match."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="ladder.pt",
                model_name="tfim",
                topology="ladder",
                p_layers=1,
            )
        )

        best = db.get_best_for_deployment(
            topology="chain_1d",
            model_name="tfim",
            p_layers=1,
            n_target=20,
        )
        assert best is None

    def test_get_best_for_deployment_excludes_archived(self, tmp_path):
        """get_best_for_deployment skips archived models."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")

        # Only archived model available
        db.register_model(
            ModelRecord(
                model_id="archived_best.pt",
                model_name="tfim",
                topology="chain_1d",
                p_layers=1,
                status="archived",
                training=TrainingProvenance(n_values_used=[10], total_training_points=100),
                evaluations=[EvaluationRecord(pass_rate_dual=1.0)],
            )
        )

        # query() with status="active" should filter this out
        best = db.get_best_for_deployment(
            topology="chain_1d",
            model_name="tfim",
            p_layers=1,
            n_target=20,
        )

        # Should be None since only archived model exists
        assert best is None


class TestIntegrityValidation:
    """Test integrity validation features."""

    def test_validate_integrity_missing_checkpoint(self, tmp_path):
        """validate_integrity detects missing checkpoint file."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="missing.pt",
                model_name="tfim",
                topology="chain_1d",
                checkpoint_path="data/model_zoo/checkpoints/missing.pt",
            )
        )

        report = db.validate_integrity("missing.pt")

        assert report.checkpoint_exists is False
        assert report.all_ok is False
        assert any("checkpoint" in issue.lower() for issue in report.issues)

    def test_validate_all_integrity(self, tmp_path):
        """validate_all_integrity checks all active models."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")

        # Create checkpoints dir with one file
        ckpt_dir = tmp_path / "data" / "model_zoo" / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "exists.pt").write_bytes(b"fake checkpoint")

        db.register_model(
            ModelRecord(
                model_id="exists.pt",
                model_name="tfim",
                topology="chain_1d",
                checkpoint_path=str(ckpt_dir / "exists.pt"),
            )
        )
        db.register_model(
            ModelRecord(
                model_id="missing.pt",
                model_name="tfim",
                topology="ladder",
                checkpoint_path=str(ckpt_dir / "missing.pt"),
            )
        )

        # Patch PROJECT_ROOT for test
        import qmbp_simulation.predictors.model_registry_db as registry_module

        original_root = registry_module._PROJECT_ROOT
        registry_module._PROJECT_ROOT = tmp_path

        try:
            report = db.validate_all_integrity()

            # Should have checked both models
            assert report["n_checked"] == 2

            # At least one should have issues (missing.pt)
            assert report["n_issues"] >= 1

            # models_with_issues should contain the missing one
            issue_ids = {m["model_id"] for m in report["models_with_issues"]}
            assert "missing.pt" in issue_ids
        finally:
            registry_module._PROJECT_ROOT = original_root

    def test_integrity_check_creates_history_event(self, tmp_path):
        """validate_integrity records event in history."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="integrity_hist.pt",
                model_name="tfim",
                topology="chain_1d",
            )
        )

        db.validate_integrity("integrity_hist.pt")

        events = db.get_history(event_type="integrity_checked")
        assert len(events) == 1
        assert events[0].model_id == "integrity_hist.pt"


class TestDashboardIntegration:
    """Test dashboard quality integration features."""

    def test_enrich_from_dashboard(self, tmp_path):
        """enrich_from_dashboard populates dashboard_quality field."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")
        db.register_model(
            ModelRecord(
                model_id="dashboard_test.pt",
                model_name="tfim_bond_resolved",
                topology="chain_1d",
                p_layers=1,
            )
        )

        # Create mock dashboard file
        import qmbp_simulation.predictors.model_registry_db as registry_module

        original_dashboard_path = registry_module._DASHBOARD_PATH
        mock_dashboard_path = tmp_path / "dashboard.json"
        registry_module._DASHBOARD_PATH = mock_dashboard_path

        dashboard = {
            "configs": [
                {
                    "topology": "chain_1d",
                    "p_layers": 1,
                    "model": "tfim_bond_resolved",
                    "training_utility": "useful",
                    "training_utility_reason": "pass_rate_dual > 50%",
                    "pass_rate_dual_criterion": 0.85,
                    "mean_de_gap": 0.023,
                    "zoo_model_available": True,
                    "n_points": 150,
                    "h_frontier": 3.2,
                },
            ],
        }

        with open(mock_dashboard_path, "w") as f:
            json.dump(dashboard, f)

        try:
            enriched = db.enrich_from_dashboard()
            assert enriched == 1

            record = db.get_model("dashboard_test.pt")
            dq = record.dashboard_quality
            assert dq.training_utility == "useful"
            assert dq.pass_rate_dual_criterion == 0.85
            assert dq.n_points == 150
        finally:
            registry_module._DASHBOARD_PATH = original_dashboard_path

    def test_get_training_health(self, tmp_path):
        """get_training_health returns health summary for a specific model."""
        db = ModelRegistryDB(path=tmp_path / "r.json", history_path=tmp_path / "h.json")

        db.register_model(
            ModelRecord(
                model_id="healthy.pt",
                model_name="tfim",
                topology="chain_1d",
                dashboard_quality=DashboardQuality(
                    training_utility="useful",
                    pass_rate_dual_criterion=0.90,
                    n_points=200,
                ),
                evaluations=[EvaluationRecord(pass_rate_dual=0.85)],
            )
        )
        db.register_model(
            ModelRecord(
                model_id="unhealthy.pt",
                model_name="tfim",
                topology="ladder",
                dashboard_quality=DashboardQuality(
                    training_utility="not_useful",
                    training_utility_reason="Low pass rate",
                    pass_rate_dual_criterion=0.20,
                    n_points=30,
                    needs_retrain=True,
                ),
            )
        )

        # Test healthy model
        health = db.get_training_health("healthy.pt")
        assert health["training_utility"] == "useful"
        assert health["recommendation"] == "use"
        assert len(health["quality_issues"]) == 0

        # Test unhealthy model
        health = db.get_training_health("unhealthy.pt")
        assert health["training_utility"] == "not_useful"
        assert health["recommendation"] == "retrain"
        assert len(health["quality_issues"]) >= 1

        # Test nonexistent model
        health = db.get_training_health("nonexistent.pt")
        assert "error" in health
