"""Model Registry Database — queryable training provenance for MPNN models.

Complementary to model_zoo.py (which handles checkpoint load/save),
this module provides a rich queryable database for:
- Which models are available
- What data was used to train each model
- Per-N point breakdowns
- Model history (regression detection, lifecycle audit trail)
- Evaluation scores, deployment results
- Training metrics (loss, MSE, epochs)
- Model tagging/labeling for semantic search
- Integrity validation (checkpoint + NPZ existence)
- Dashboard integration for quality tier tracking

Storage:
- data/model_zoo/model_registry.json (current state, list of records)
- data/model_zoo/model_history.json (append-only event log for audit/regression)

Usage
-----
>>> from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB
>>> db = ModelRegistryDB()
>>> db.query(topology="chain_1d", min_training_points=50)
>>> db.register_model(entry)
>>> db.get_model("unified_tfim_br_chain_1d_multiN_6+8+10+12+20_p1.pt")
>>> db.get_history("unified_tfim_br_chain_1d_multiN_6+8+10+12+20_p1.pt")
>>> db.detect_regressions()
>>> db.validate_integrity("model.pt")
>>> db.get_best_for_deployment("chain_1d", "tfim_bond_resolved", 1, n_target=20)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_REGISTRY_PATH = _PROJECT_ROOT / "data" / "model_zoo" / "model_registry.json"
_HISTORY_PATH = _PROJECT_ROOT / "data" / "model_zoo" / "model_history.json"
_CHECKPOINTS_DIR = _PROJECT_ROOT / "data" / "model_zoo" / "checkpoints"
_DASHBOARD_PATH = _PROJECT_ROOT / "data" / "model_quality_dashboard.json"
_MANIFEST_PATH = _PROJECT_ROOT / "data" / "model_zoo" / "manifest.json"

# Minimum pass_rate for multi-N model selection (below this, prefer single-N)
MULTI_N_MIN_PASS_RATE: float = 0.40


# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT #1: Training Metrics Tracking
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrainingMetrics:
    """Metrics from the MPNN training process.

    Captures training dynamics for reproducibility and diagnostics.

    Attributes
    ----------
    final_loss : float
        Final training loss value.
    final_mse : float
        Final mean squared error on training set.
    final_val_mse : float
        Final validation MSE (if val_fraction > 0).
    generalization_gap : float
        val_mse - train_mse (positive = overfitting).
    epochs : int
        Total epochs trained.
    best_epoch : int
        Epoch with best validation loss (if early stopping used).
    training_time_seconds : float
        Wall-clock training time.
    learning_rate : float
        Learning rate used.
    batch_size : int
        Batch size used.
    early_stopped : bool
        Whether training stopped early due to plateau.
    stop_reason : str
        Why training stopped: "completed" | "lr_exhausted" | "overfitting_detected" |
        "mse_floor_reached" | "all_graphs_skipped"
    convergence_status : str
        "converged" | "plateau" | "max_epochs" | "diverged"
    loss_history : list[float]
        Optional: loss curve (can be empty to save space).
    weight_distribution : dict[str, int]
        Quality tier distribution in training data: {"verified (1.0)": N, ...}
    """

    final_loss: float = 0.0
    final_mse: float = 0.0
    final_val_mse: float | None = None
    generalization_gap: float | None = None
    epochs: int = 0
    best_epoch: int = 0
    training_time_seconds: float = 0.0
    learning_rate: float = 0.001
    batch_size: int = 32
    early_stopped: bool = False
    stop_reason: str = "unknown"
    convergence_status: str = "unknown"  # converged | plateau | max_epochs | diverged
    loss_history: list[float] = field(default_factory=list)
    weight_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class ModelArchitectureConfig:
    """MPNN model architecture configuration for reproducibility.

    Attributes
    ----------
    hidden_dim : int
        Hidden dimension of the MPNN.
    n_conv_layers : int
        Number of message-passing layers.
    n_heads : int
        Number of attention heads (if applicable).
    dropout : float
        Dropout rate during training.
    activation : str
        Activation function name (e.g., "relu", "gelu").
    include_circuit_nodes : bool
        Whether the graph includes circuit layer nodes.
    """

    hidden_dim: int = 64
    n_conv_layers: int = 3
    n_heads: int = 1
    dropout: float = 0.0
    activation: str = "relu"
    include_circuit_nodes: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer configuration for reproducibility.

    Attributes
    ----------
    optimizer_name : str
        Optimizer class name (e.g., "AdamW", "Adam").
    learning_rate : float
        Initial learning rate.
    weight_decay : float
        L2 regularization strength.
    scheduler : str
        LR scheduler name (e.g., "ReduceLROnPlateau").
    scheduler_patience : int
        Patience for scheduler (if applicable).
    scheduler_factor : float
        LR reduction factor (if applicable).
    layerwise_lr : dict[str, float]
        Layer-wise LR multipliers (for fine-tuning).
    """

    optimizer_name: str = "AdamW"
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    scheduler: str = "ReduceLROnPlateau"
    scheduler_patience: int = 300
    scheduler_factor: float = 0.5
    layerwise_lr: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingProvenance:
    """Per-model training data provenance.

    Captures complete reproducibility information for the training data.

    Attributes
    ----------
    n_values_used : list[int]
        System sizes (N) used in training.
    total_training_points : int
        Total number of (h, θ) pairs used.
    points_per_n : dict[str, int]
        Per-N breakdown of training points.
    h_range : tuple[float, float]
        (h_min, h_max) covered by training data.
    max_de_gap_filter : float
        Maximum ΔE/gap allowed during data filtering.
    seeds : list[int]
        Random seeds used in VQE data generation.
    data_source : str
        "multi_n_training" | "single_n" | "manual"
    training_data_hash : str
        MD5 hash of training NPZ files (detects if data changed).
    validation_split_seed : int
        Seed used for train/val split (for reproducibility).
    training_metrics : TrainingMetrics
        Metrics from the training process.
    architecture_config : ModelArchitectureConfig
        Model architecture configuration.
    optimizer_config : OptimizerConfig
        Optimizer and scheduler configuration.
    """

    n_values_used: list[int] = field(default_factory=list)
    total_training_points: int = 0
    points_per_n: dict[str, int] = field(default_factory=dict)
    h_range: tuple[float, float] = (1.0, 3.5)
    max_de_gap_filter: float = 0.10
    seeds: list[int] = field(default_factory=lambda: [42])
    data_source: str = "multi_n_training"  # "multi_n_training" | "single_n" | "manual"
    # IMPROVEMENT: Training data hash for staleness detection
    training_data_hash: str = ""
    # IMPROVEMENT: Validation split seed for reproducibility
    validation_split_seed: int = 42
    # Training metrics
    training_metrics: TrainingMetrics = field(default_factory=TrainingMetrics)
    # IMPROVEMENT: Architecture config for reproducibility
    architecture_config: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    # IMPROVEMENT: Optimizer config for reproducibility
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)


@dataclass
class EvaluationRecord:
    """Evaluation/deployment result for a model (filled post-analysis).

    Placeholder for future integration — scores, pass rates at different N, etc.
    """

    evaluated_at: str = ""  # ISO timestamp
    target_n_values: list[int] = field(default_factory=list)
    pass_rate_5pct: float = 0.0
    pass_rate_dual: float = 0.0
    mean_de_gap: float = 0.0
    mean_abs_error_per_site: float = 0.0
    notes: str = ""
    # ── Statistical confidence (95% CI) — populated from compute_deploy_summary ──
    pass_rate_dual_ci: list[float] = field(default_factory=list)  # [lower, upper]
    mean_de_gap_ci: list[float] = field(default_factory=list)  # [lower, upper]
    # ── Per-h raw arrays (per N) — enables paired significance tests between ──
    # models without re-parsing markdown. Format: {str(n): [values...]}.
    per_h_de_gaps: dict[str, list[float]] = field(default_factory=dict)
    per_h_abs_errors: dict[str, list[float]] = field(default_factory=dict)
    per_h_h_values: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class HistoryEvent:
    """A single event in the model history log (append-only).

    Types:
    - "registered": Model first added to registry
    - "retrained": Model replaced by retraining (old metrics preserved)
    - "evaluated": Evaluation result recorded
    - "superseded": Model marked as superseded by another
    - "archived": Model archived (soft-deleted)
    - "pass_rate_updated": Pass rate updated via update_zoo_pass_rate
    - "regression_detected": Automatic regression detection event
    - "tag_added": Tag added to model
    - "tag_removed": Tag removed from model
    - "integrity_checked": Integrity validation performed
    - "failure_diagnosed": Failure diagnostics run
    - "training_data_changed": NPZ files updated (model now stale)
    - "auto_retrain_triggered": Automatic retrain triggered and why
    - "quality_degraded": Metrics worsened without retrain (data problem)
    - "needs_retrain_flagged": Model marked for retrain due to data improvement
    - "needs_retrain_cleared": Flag cleared after successful retrain
    - "dashboard_synced": Dashboard quality data synced to record
    - "auto_versioned": Model checkpoint auto-versioned to _versions/ before overwrite
    """

    timestamp: str  # ISO format
    event_type: str
    model_id: str
    topology: str = ""
    details: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT #9: Dashboard Quality Integration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FailureDiagnosticSummary:
    """Summarized failure diagnostic for embedding in ModelRecord.

    Lightweight version of FailureDiagnostic from analysis/failures_tests.py
    for storage in the registry without heavy imports.
    """

    primary_mode: str = ""  # "gap_masking", "generalization_failure", "intrinsic_vqe_error", "contaminated_training", "healthy", "mixed", "unknown"
    confidence: float = 0.0  # 0-1 confidence in classification
    secondary_modes: list[str] = field(default_factory=list)
    explanation: str = ""

    # Key metrics (subset of full FailureDiagnostic)
    per_site_verified: float | None = None
    gap_masked_fraction: float | None = None
    contamination_severity: str | None = None  # "severe", "moderate", "mild", "none"
    h_range_overlap: float | None = None

    diagnosed_at: str = ""  # ISO timestamp


@dataclass
class DashboardQuality:
    """Quality metrics from model_quality_dashboard.json.

    Provides data health indicators for training decisions.
    """

    training_utility: str = ""  # "useful" | "insufficient_signal" | "not_useful"
    training_utility_reason: str = ""
    pass_rate_dual_criterion: float = 0.0
    mean_de_gap: float = 0.0
    zoo_model_available: bool = False
    zoo_integrity_ok: bool = True
    needs_retrain: bool = False
    model_stale: bool = False
    n_points: int = 0
    h_frontier: float = 0.0
    last_synced: str = ""  # ISO timestamp of last sync

    # IMPROVEMENT #10: Failure Diagnostic Integration
    failure_diagnostic: FailureDiagnosticSummary = field(default_factory=FailureDiagnosticSummary)


# ═══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT #8: Integrity Validation Result
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class IntegrityReport:
    """Result of integrity validation for a model.

    All fields should be True for a healthy model.
    """

    checkpoint_exists: bool = False
    hash_matches: bool = False  # True if no hash stored (legacy) or hash matches
    manifest_consistent: bool = False  # Zoo manifest and registry agree
    all_ok: bool = False
    issues: list[str] = field(default_factory=list)

    # IMPROVEMENT #10: Training data quality checks
    training_data_exists: bool = False  # NPZ files exist for training N values
    training_data_verified_ratio: float = 0.0  # Fraction of verified points in training data
    training_data_issues: list[str] = field(default_factory=list)
    failure_mode_detected: str = ""  # Primary failure mode from diagnostics


@dataclass
class ModelRecord:
    """Complete record for a trained MPNN model.

    Attributes
    ----------
    model_id : str
        Unique identifier = checkpoint filename (without path).
    model_name : str
        Hamiltonian model (tfim_bond_resolved, tfim, etc.)
    architecture : str
        Model architecture (UnifiedMPNN, MPNNPredictor, BondResolvedMPNN).
    topology : str
        Lattice topology.
    p_layers : int
        HVA depth used.
    checkpoint_path : str
        Relative path to checkpoint file (from project root).
    created : str
        ISO timestamp of model creation.
    runner_tag : str
        2-letter tag identifying the runner that trained it.
    date_tag : str
        DDMMYY format date tag.
    training : TrainingProvenance
        Detailed training data provenance.
    evaluations : list[EvaluationRecord]
        Evaluation results (appended over time).
    status : str
        "active" | "archived" | "superseded"
    superseded_by : str
        model_id of replacement model (if superseded).
    notes : str
        Free-form notes.
    tags : list[str]
        Structured tags for semantic search (IMPROVEMENT #3).
        Examples: ["production", "cross-n-validated", "hardware-ready", "thesis-figure-7"]
    dashboard_quality : DashboardQuality
        Quality metrics from dashboard (IMPROVEMENT #9).
    version : int
        Model version number (IMPROVEMENT #5). Starts at 1, increments on retrain.
    version_history : list[str]
        List of previous model_ids that this model supersedes (version chain).
    """

    model_id: str
    model_name: str
    architecture: str = "UnifiedMPNN"
    topology: str = ""
    p_layers: int = 1
    checkpoint_path: str = ""
    created: str = ""
    runner_tag: str = "XX"
    date_tag: str = ""
    training: TrainingProvenance = field(default_factory=TrainingProvenance)
    evaluations: list[EvaluationRecord] = field(default_factory=list)
    status: str = "active"
    superseded_by: str = ""
    notes: str = ""
    # IMPROVEMENT #3: Tagging System
    tags: list[str] = field(default_factory=list)
    # IMPROVEMENT #9: Dashboard Quality Integration
    dashboard_quality: DashboardQuality = field(default_factory=DashboardQuality)
    # IMPROVEMENT #5: Model Versioning
    version: int = 1
    version_history: list[str] = field(default_factory=list)


class ModelRegistryDB:
    """Queryable database of trained MPNN models.

    Thread-safe for reads. Writes are append-only and flush immediately.
    Maintains an append-only history log for regression detection and audit.
    """

    def __init__(self, path: Path | None = None, history_path: Path | None = None):
        self._path = path or _REGISTRY_PATH
        self._history_path = history_path or _HISTORY_PATH
        self._records: list[ModelRecord] = []
        self._history: list[HistoryEvent] | None = None  # Lazy-loaded
        self._batch_mode: bool = False  # Defer saves when True
        self._dirty: bool = False  # Tracks unsaved record changes
        self._history_dirty: bool = False  # Tracks unsaved history changes
        self._load()

    # ─── Batch Context Manager ──────────────────────────────────────────────

    class _BatchContext:
        """Context manager to batch multiple writes into a single flush.

        Supports nesting: only the outermost batch triggers the final flush.
        On exception, dirty state is cleared WITHOUT flushing (rollback semantics).
        """

        def __init__(self, db: ModelRegistryDB):
            self._db = db
            self._was_already_batching = db._batch_mode

        def __enter__(self):
            self._db._batch_mode = True
            return self._db

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Only the outermost batch controls flush
            if self._was_already_batching:
                return  # Inner batch — leave batch_mode on, let outer handle flush

            self._db._batch_mode = False

            if exc_type is not None:
                # Exception occurred — discard dirty state (rollback)
                # Records in memory may be inconsistent, reload from disk
                self._db._dirty = False
                self._db._history_dirty = False
                self._db._load()
                if self._db._history is not None:
                    self._db._load_history()
                return  # Don't suppress the exception

            # Normal exit — flush accumulated writes
            if self._db._dirty:
                self._db._flush_records()
            if self._db._history_dirty:
                self._db._flush_history()

    def batch(self) -> _BatchContext:
        """Context manager to defer disk writes until block exits.

        Use for operations that would otherwise call _save() multiple times.

        Example
        -------
        >>> with db.batch():
        ...     db.register_model(record, overwrite=True)
        ...     db.add_tag(record.model_id, "production")
        ...     db.set_training_metrics(record.model_id, final_mse=0.01)
        # Single disk write here instead of 3
        """
        return self._BatchContext(self)

    # ─── Persistence ────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load registry from disk."""
        if not self._path.exists():
            self._records = []
            return
        with open(self._path) as f:
            raw = json.load(f)
        # Handle both formats: list (canonical) and dict with numeric keys (legacy)
        if isinstance(raw, dict):
            items = list(raw.values())
        else:
            items = raw
        self._records = [self._deserialize(item) for item in items if isinstance(item, dict)]
        logger.debug("ModelRegistryDB: loaded %d records", len(self._records))

    def _save(self) -> None:
        """Mark records as dirty; flush immediately unless in batch mode."""
        if self._batch_mode:
            self._dirty = True
        else:
            self._flush_records()

    def _flush_records(self) -> None:
        """Actually write registry to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        raw = [self._serialize(r) for r in self._records]
        with open(self._path, "w") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)
        self._dirty = False
        logger.debug("ModelRegistryDB: saved %d records", len(self._records))

    # ─── History Persistence (lazy load + append optimization) ──────────────

    def _ensure_history_loaded(self) -> None:
        """Lazy-load history on first access."""
        if self._history is None:
            self._load_history()

    def _load_history(self) -> None:
        """Load history log from disk."""
        if not self._history_path.exists():
            self._history = []
            return
        with open(self._history_path) as f:
            raw = json.load(f)
        self._history = [HistoryEvent(**item) for item in raw]
        logger.debug("ModelRegistryDB: loaded %d history events", len(self._history))

    def _save_history(self) -> None:
        """Mark history as dirty; flush immediately unless in batch mode."""
        if self._batch_mode:
            self._history_dirty = True
        else:
            self._flush_history()

    def _flush_history(self) -> None:
        """Actually write history to disk."""
        if self._history is None:
            return
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        raw = [asdict(e) for e in self._history]
        with open(self._history_path, "w") as f:
            json.dump(raw, f, indent=2, ensure_ascii=False)
        self._history_dirty = False

    def _record_event(
        self,
        event_type: str,
        model_id: str,
        topology: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Append an event to the history log."""
        self._ensure_history_loaded()
        event = HistoryEvent(
            timestamp=datetime.now(UTC).isoformat(),
            event_type=event_type,
            model_id=model_id,
            topology=topology,
            details=details or {},
        )
        self._history.append(event)
        self._save_history()

    @staticmethod
    def _serialize(record: ModelRecord) -> dict[str, Any]:
        """Convert ModelRecord to JSON-serializable dict."""
        d = asdict(record)
        # Ensure h_range is list (JSON doesn't have tuples)
        if "training" in d and "h_range" in d["training"]:
            d["training"]["h_range"] = list(d["training"]["h_range"])
        return d

    @staticmethod
    def _deserialize(raw: dict[str, Any]) -> ModelRecord:
        """Reconstruct ModelRecord from raw dict."""
        # Parse training provenance
        training_raw = raw.pop("training", {})
        if "h_range" in training_raw and isinstance(training_raw["h_range"], list):
            training_raw["h_range"] = tuple(training_raw["h_range"])

        # Parse nested TrainingMetrics
        metrics_raw = training_raw.pop("training_metrics", {})
        training_metrics = TrainingMetrics(**metrics_raw) if metrics_raw else TrainingMetrics()

        # Parse nested ModelArchitectureConfig
        arch_raw = training_raw.pop("architecture_config", {})
        architecture_config = (
            ModelArchitectureConfig(**arch_raw) if arch_raw else ModelArchitectureConfig()
        )

        # Parse nested OptimizerConfig
        opt_raw = training_raw.pop("optimizer_config", {})
        optimizer_config = OptimizerConfig(**opt_raw) if opt_raw else OptimizerConfig()

        training = TrainingProvenance(
            training_metrics=training_metrics,
            architecture_config=architecture_config,
            optimizer_config=optimizer_config,
            **training_raw,
        )

        # Parse evaluations
        evals_raw = raw.pop("evaluations", [])
        evaluations = [EvaluationRecord(**e) for e in evals_raw]

        # Parse dashboard_quality (IMPROVEMENT #9)
        dashboard_raw = raw.pop("dashboard_quality", {})
        # Parse nested FailureDiagnosticSummary (IMPROVEMENT #10)
        failure_diag_raw = dashboard_raw.pop("failure_diagnostic", {})
        failure_diagnostic = (
            FailureDiagnosticSummary(**failure_diag_raw)
            if failure_diag_raw
            else FailureDiagnosticSummary()
        )
        dashboard_quality = (
            DashboardQuality(failure_diagnostic=failure_diagnostic, **dashboard_raw)
            if dashboard_raw
            else DashboardQuality()
        )

        return ModelRecord(
            training=training, evaluations=evaluations, dashboard_quality=dashboard_quality, **raw
        )

    # ─── Registration ───────────────────────────────────────────────────────

    def register_model(self, record: ModelRecord, *, overwrite: bool = False) -> None:
        """Register a model in the database.

        Parameters
        ----------
        record : ModelRecord
            Complete model record.
        overwrite : bool
            If True, replaces existing record with same model_id.
            The new record inherits version+1 and adds old model_id to version_history.
        """
        existing_idx = next(
            (i for i, r in enumerate(self._records) if r.model_id == record.model_id),
            None,
        )
        if existing_idx is not None:
            if overwrite:
                old_record = self._records[existing_idx]

                # IMPROVEMENT #5: Version management on overwrite
                # Increment version and preserve history chain
                record.version = old_record.version + 1
                record.version_history = old_record.version_history + [old_record.model_id]

                # Record retrain event with old metrics for regression detection
                self._record_event(
                    "retrained",
                    record.model_id,
                    topology=record.topology,
                    details={
                        "old_training_points": old_record.training.total_training_points,
                        "new_training_points": record.training.total_training_points,
                        "old_n_values": old_record.training.n_values_used,
                        "new_n_values": record.training.n_values_used,
                        "old_pass_rate": (
                            old_record.evaluations[-1].pass_rate_dual
                            if old_record.evaluations
                            else 0.0
                        ),
                        "old_version": old_record.version,
                        "new_version": record.version,
                    },
                )
                self._records[existing_idx] = record
                logger.info(
                    "ModelRegistryDB: updated %s (v%d → v%d)",
                    record.model_id,
                    old_record.version,
                    record.version,
                )
            else:
                logger.warning(
                    "ModelRegistryDB: %s already exists. Use overwrite=True.", record.model_id
                )
                return
        else:
            # New model — ensure version starts at 1
            if record.version < 1:
                record.version = 1
            self._records.append(record)
            self._record_event(
                "registered",
                record.model_id,
                topology=record.topology,
                details={
                    "training_points": record.training.total_training_points,
                    "n_values": record.training.n_values_used,
                    "runner_tag": record.runner_tag,
                    "version": record.version,
                },
            )
            logger.info("ModelRegistryDB: registered %s (v%d)", record.model_id, record.version)
        self._save()

    def register_from_zoo_entry(
        self,
        zoo_entry: ZooEntry,  # noqa: F821
        *,
        architecture: str = "UnifiedMPNN",
        points_per_n: dict[str, int] | None = None,
        n_values_used: list[int] | None = None,
        overwrite: bool = False,
    ) -> ModelRecord:
        """Create a ModelRecord from a ZooEntry and register it.

        Convenience method for integration with register_checkpoint().
        """
        # Infer n_values from checkpoint filename or notes if not provided
        inferred_n_values = n_values_used or []
        if not inferred_n_values and "N=" in (zoo_entry.notes or ""):
            import re

            match = re.search(r"N=\[([^\]]+)\]", zoo_entry.notes)
            if match:
                inferred_n_values = [int(x.strip()) for x in match.group(1).split(",")]

        if not inferred_n_values and "multiN_" in zoo_entry.checkpoint_file:
            # Parse from filename: unified_tfim_br_chain_1d_multiN_6+8+10+12+20_p1.pt
            import re

            match = re.search(r"multiN_([\d+]+)_", zoo_entry.checkpoint_file)
            if match:
                inferred_n_values = [int(x) for x in match.group(1).split("+")]

        # Single-N model
        if not inferred_n_values and zoo_entry.n_qubits > 0:
            inferred_n_values = [zoo_entry.n_qubits]

        training = TrainingProvenance(
            n_values_used=inferred_n_values,
            total_training_points=zoo_entry.n_training_points,
            points_per_n=points_per_n or {},
            h_range=zoo_entry.h_range,
            seeds=zoo_entry.seeds,
            data_source="multi_n_training" if zoo_entry.n_qubits == 0 else "single_n",
        )

        record = ModelRecord(
            model_id=zoo_entry.checkpoint_file,
            model_name=zoo_entry.model,
            architecture=architecture,
            topology=zoo_entry.topology,
            p_layers=zoo_entry.p_layers,
            checkpoint_path=f"data/model_zoo/checkpoints/{zoo_entry.checkpoint_file}",
            created=zoo_entry.created,
            runner_tag=zoo_entry.runner_tag,
            date_tag=zoo_entry.date_tag,
            training=training,
            status="active",
            notes=zoo_entry.notes,
        )
        self.register_model(record, overwrite=overwrite)
        return record

    # ─── Evaluation Tracking ────────────────────────────────────────────────

    def add_evaluation(self, model_id: str, evaluation: EvaluationRecord) -> None:
        """Append an evaluation record to a model.

        Parameters
        ----------
        model_id : str
            The model to update.
        evaluation : EvaluationRecord
            Evaluation result to append.
        """
        idx = next(
            (i for i, r in enumerate(self._records) if r.model_id == model_id),
            None,
        )
        if idx is None:
            logger.warning("ModelRegistryDB: model %s not found", model_id)
            return

        # Check for regression against previous evaluations
        record = self._records[idx]
        prev_evals = record.evaluations
        if prev_evals and evaluation.pass_rate_dual > 0:
            prev_best = (
                max(e.pass_rate_dual for e in prev_evals if e.pass_rate_dual > 0)
                if any(e.pass_rate_dual > 0 for e in prev_evals)
                else 0.0
            )
            if prev_best > 0 and evaluation.pass_rate_dual < prev_best - 0.05:
                self._record_event(
                    "regression_detected",
                    model_id,
                    topology=record.topology,
                    details={
                        "prev_best_pass_rate": prev_best,
                        "new_pass_rate": evaluation.pass_rate_dual,
                        "delta": evaluation.pass_rate_dual - prev_best,
                        "target_n": evaluation.target_n_values,
                    },
                )
                logger.warning(
                    "ModelRegistryDB: REGRESSION on %s: %.0f%% → %.0f%% (Δ=%.0f%%)",
                    model_id,
                    prev_best * 100,
                    evaluation.pass_rate_dual * 100,
                    (evaluation.pass_rate_dual - prev_best) * 100,
                )

        record.evaluations.append(evaluation)
        self._save()

        # Record evaluation event in history
        self._record_event(
            "evaluated",
            model_id,
            topology=record.topology,
            details={
                "pass_rate_dual": evaluation.pass_rate_dual,
                "pass_rate_5pct": evaluation.pass_rate_5pct,
                "mean_de_gap": evaluation.mean_de_gap,
                "target_n": evaluation.target_n_values,
                "notes": evaluation.notes,
            },
        )
        logger.info("ModelRegistryDB: added evaluation to %s", model_id)

    # ─── Queries ────────────────────────────────────────────────────────────

    def query(
        self,
        *,
        model_name: str | None = None,
        topology: str | None = None,
        p_layers: int | None = None,
        min_training_points: int | None = None,
        min_n_values: int | None = None,
        status: str | None = "active",
        architecture: str | None = None,
    ) -> list[ModelRecord]:
        """Query models matching the given criteria.

        All filters are AND-combined. None = no filter on that field.
        """
        results = []
        for r in self._records:
            if model_name and r.model_name != model_name:
                continue
            if topology and r.topology != topology:
                continue
            if p_layers is not None and r.p_layers != p_layers:
                continue
            if status and r.status != status:
                continue
            if architecture and r.architecture != architecture:
                continue
            if min_training_points and r.training.total_training_points < min_training_points:
                continue
            if min_n_values and len(r.training.n_values_used) < min_n_values:
                continue
            results.append(r)
        return results

    def get_model(self, model_id: str) -> ModelRecord | None:
        """Get a single model by its ID (checkpoint filename)."""
        return next((r for r in self._records if r.model_id == model_id), None)

    def list_all(self, *, include_archived: bool = False) -> list[ModelRecord]:
        """List all models, optionally including archived ones."""
        if include_archived:
            return list(self._records)
        return [r for r in self._records if r.status != "archived"]

    def summary(self) -> dict[str, Any]:
        """Quick summary of the registry contents."""
        active = [r for r in self._records if r.status == "active"]
        return {
            "total_models": len(self._records),
            "active_models": len(active),
            "topologies": sorted(set(r.topology for r in active)),
            "model_names": sorted(set(r.model_name for r in active)),
            "total_training_points": sum(r.training.total_training_points for r in active),
            "max_n_trained": max(
                (max(r.training.n_values_used) for r in active if r.training.n_values_used),
                default=0,
            ),
        }

    # ─── Topology-Based Queries ─────────────────────────────────────────────

    def list_by_topology(
        self, topology: str, *, include_archived: bool = False
    ) -> list[ModelRecord]:
        """List all models for a specific topology.

        Parameters
        ----------
        topology : str
            Target topology (e.g., "chain_1d", "multi_topology").
        include_archived : bool
            Whether to include archived models.

        Returns
        -------
        list[ModelRecord]
            Matching models sorted by training points (descending).
        """
        results = [
            r
            for r in self._records
            if r.topology == topology and (include_archived or r.status != "archived")
        ]
        return sorted(results, key=lambda r: r.training.total_training_points, reverse=True)

    def get_multi_topology_models(self, *, p_layers: int = 1) -> list[ModelRecord]:
        """Get all multi-topology models.


        Convenience method that filters for topology="multi_topology".
        Used by the MT selection policy and comparison scripts.
        """
        return [r for r in self.list_by_topology("multi_topology") if r.p_layers == p_layers]

    def get_best_for_topology(
        self, topology: str, *, p_layers: int = 1, n_target: int = 0
    ) -> ModelRecord | None:
        """Get the best active model for a given topology.

        Scoring: pass_rate (if evaluated) > training_points > proximity.

        Parameters
        ----------
        topology : str
            Target topology.
        p_layers : int
            HVA depth filter.
        n_target : int
            If > 0, prefer models trained on N values close to n_target.

        Returns
        -------
        ModelRecord | None
            Best model or None if no models exist.
        """
        candidates = [
            r
            for r in self.list_by_topology(topology)
            if r.p_layers == p_layers and r.status == "active"
        ]
        if not candidates:
            return None

        def _score(r: ModelRecord) -> float:
            tm = r.training.training_metrics
            pr = tm.pass_rate if tm and hasattr(tm, "pass_rate") else 0.0
            pts = r.training.total_training_points
            proximity = (
                1.0 / (1.0 + abs(max(r.training.n_values_used or [0]) - n_target))
                if n_target > 0 and r.training.n_values_used
                else 1.0
            )
            return pr * 1000 + pts * proximity

        return max(candidates, key=_score)

    def prune_test_entries(self, *, dry_run: bool = True) -> list[str]:
        """Remove test/orphan entries from the registry.

        Identifies entries with model_id matching test patterns
        (test_*, kiro_test_*) and archives or removes them.

        Parameters
        ----------
        dry_run : bool
            If True, just report what would be pruned without modifying.

        Returns
        -------
        list[str]
            List of model_ids that were (or would be) pruned.
        """
        test_patterns = ("test_", "kiro_test_")
        to_prune = [
            r.model_id
            for r in self._records
            if any(r.model_id.lower().startswith(p) for p in test_patterns)
        ]
        if not dry_run and to_prune:
            self._records = [r for r in self._records if r.model_id not in to_prune]
            self._save()
            for mid in to_prune:
                self._record_event("pruned", mid, details={"reason": "test_entry"})
        return to_prune

    def get_lineage(self, model_id: str, *, max_depth: int = 10) -> list[dict]:
        """Trace the fine-tuning lineage of a model.

        Follows the ``fine_tuned_from`` field in architecture_config to
        reconstruct the training ancestry chain. Useful for thesis
        documentation (demonstrates iterative improvement pipeline).

        Parameters
        ----------
        model_id : str
            Starting model to trace back from.
        max_depth : int
            Maximum ancestry depth to prevent infinite loops (default: 10).

        Returns
        -------
        list[dict]
            Lineage chain from newest to oldest, each entry:
            {
                "model_id": str,
                "topology": str,
                "n_training_points": int,
                "created": str,
                "fine_tuned_from": str | None,
                "depth": int,
            }
        """
        lineage = []
        current_id = model_id
        visited = set()

        for depth in range(max_depth):
            if current_id in visited:
                break  # Cycle detected
            visited.add(current_id)

            record = self.get_model(current_id)
            if record is None:
                # Try fuzzy match (partial model_id)
                matches = [r for r in self._records if current_id in r.model_id]
                record = matches[0] if matches else None

            if record is None:
                break

            arch_config = {}
            if record.training and hasattr(record.training, "architecture"):
                arch_config = (
                    record.training.architecture.__dict__ if record.training.architecture else {}
                )

            parent = arch_config.get("fine_tuned_from", None)

            lineage.append(
                {
                    "model_id": record.model_id,
                    "topology": record.topology,
                    "n_training_points": record.training.total_training_points,
                    "created": record.created,
                    "fine_tuned_from": parent,
                    "depth": depth,
                }
            )

            if not parent:
                break  # Root of the lineage

            # Find parent model_id (parent is a filename, not model_id)
            parent_matches = [
                r.model_id
                for r in self._records
                if parent in r.model_id or r.model_id.endswith(parent)
            ]
            current_id = parent_matches[0] if parent_matches else parent

        return lineage

    # ─── Lifecycle ──────────────────────────────────────────────────────────

    def mark_superseded(self, model_id: str, superseded_by: str) -> None:
        """Mark a model as superseded by a newer version."""
        idx = next(
            (i for i, r in enumerate(self._records) if r.model_id == model_id),
            None,
        )
        if idx is None:
            logger.warning("ModelRegistryDB: model %s not found", model_id)
            return
        self._records[idx].status = "superseded"
        self._records[idx].superseded_by = superseded_by
        self._save()
        self._record_event(
            "superseded",
            model_id,
            topology=self._records[idx].topology,
            details={"superseded_by": superseded_by},
        )

    def archive(self, model_id: str) -> None:
        """Archive a model (soft-delete)."""
        idx = next(
            (i for i, r in enumerate(self._records) if r.model_id == model_id),
            None,
        )
        if idx is None:
            return
        self._records[idx].status = "archived"
        self._save()
        self._record_event(
            "archived",
            model_id,
            topology=self._records[idx].topology,
        )

    # ─── Bulk Operations ────────────────────────────────────────────────────

    def sync_from_manifest(self) -> dict[str, int]:
        """Populate/sync registry from existing zoo manifest.

        Scans data/model_zoo/manifest.json and registers any models not yet
        in the registry. Does NOT overwrite existing entries.

        Returns
        -------
        dict with "added" and "skipped" counts.
        """
        from qmbp_simulation.predictors.model_zoo import _load_manifest

        manifest_entries = _load_manifest()
        added, skipped = 0, 0

        for entry in manifest_entries:
            if self.get_model(entry.checkpoint_file) is not None:
                skipped += 1
                continue
            self.register_from_zoo_entry(entry)
            added += 1

        logger.info("ModelRegistryDB sync_from_manifest: added=%d, skipped=%d", added, skipped)
        return {"added": added, "skipped": skipped}

    def enrich_points_per_n(self) -> int:
        """Scan NPZ files to fill in points_per_n where missing.

        Returns number of records enriched.
        """
        import numpy as np

        npz_dir = _PROJECT_ROOT / "data" / "multi_n_training"
        if not npz_dir.exists():
            return 0

        enriched = 0
        for record in self._records:
            if record.training.points_per_n:
                continue  # Already has data
            if not record.training.n_values_used:
                continue

            points_per_n: dict[str, int] = {}
            for n in record.training.n_values_used:
                npz_file = npz_dir / f"{record.topology}_N{n}_p{record.p_layers}.npz"
                if npz_file.exists():
                    data = np.load(npz_file, allow_pickle=True)
                    # Count points (h_values array length)
                    if "h_values" in data:
                        points_per_n[str(n)] = len(data["h_values"])
                    elif "h" in data:
                        points_per_n[str(n)] = len(data["h"])

            if points_per_n:
                record.training.points_per_n = points_per_n
                record.training.total_training_points = sum(points_per_n.values())
                enriched += 1

        if enriched:
            self._save()
        return enriched

    # ─── History Queries ────────────────────────────────────────────────────

    def get_history(
        self,
        model_id: str | None = None,
        event_type: str | None = None,
        topology: str | None = None,
        limit: int | None = None,
    ) -> list[HistoryEvent]:
        """Query history events with optional filters.

        Parameters
        ----------
        model_id : str | None
            Filter by model ID (exact match).
        event_type : str | None
            Filter by event type.
        topology : str | None
            Filter by topology.
        limit : int | None
            Max number of events to return (most recent first).
        """
        self._ensure_history_loaded()
        results = self._history
        if model_id:
            results = [e for e in results if e.model_id == model_id]
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if topology:
            results = [e for e in results if e.topology == topology]
        # Most recent first
        results = sorted(results, key=lambda e: e.timestamp, reverse=True)
        if limit:
            results = results[:limit]
        return results

    def detect_regressions(self, threshold: float = 0.05) -> list[dict[str, Any]]:
        """Detect regressions by comparing consecutive evaluations per model.

        A regression is when a model's pass_rate_dual drops by more than
        `threshold` between consecutive evaluations.

        Returns list of regression records with model_id, topology, old/new rates.
        """
        regressions = []
        for record in self._records:
            if record.status != "active":
                continue
            evals_with_rate = [e for e in record.evaluations if e.pass_rate_dual > 0]
            if len(evals_with_rate) < 2:
                continue

            for i in range(1, len(evals_with_rate)):
                prev = evals_with_rate[i - 1].pass_rate_dual
                curr = evals_with_rate[i].pass_rate_dual
                if curr < prev - threshold:
                    regressions.append(
                        {
                            "model_id": record.model_id,
                            "topology": record.topology,
                            "prev_pass_rate": prev,
                            "curr_pass_rate": curr,
                            "delta": curr - prev,
                            "eval_index": i,
                            "evaluated_at": evals_with_rate[i].evaluated_at,
                        }
                    )
        return regressions

    def get_model_timeline(self, model_id: str) -> list[dict[str, Any]]:
        """Get a chronological timeline of all events for a model.

        Combines registration, retrains, evaluations, and lifecycle events
        into a single ordered list for easy visualization.
        """
        events = self.get_history(model_id=model_id)
        # Reverse to chronological order
        events.reverse()
        timeline = []
        for e in events:
            entry = {
                "timestamp": e.timestamp,
                "event": e.event_type,
                **e.details,
            }
            timeline.append(entry)
        return timeline

    def history_summary(self) -> dict[str, Any]:
        """Summary statistics of the history log."""
        self._ensure_history_loaded()
        if not self._history:
            return {"total_events": 0}

        type_counts: dict[str, int] = {}
        for e in self._history:
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1

        return {
            "total_events": len(self._history),
            "event_types": type_counts,
            "first_event": self._history[0].timestamp if self._history else None,
            "last_event": self._history[-1].timestamp if self._history else None,
            "regressions_detected": type_counts.get("regression_detected", 0),
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # IMPROVEMENT #3: Tagging System
    # ═══════════════════════════════════════════════════════════════════════════

    def add_tag(self, model_id: str, tag: str) -> bool:
        """Add a tag to a model.

        Parameters
        ----------
        model_id : str
            Model to tag.
        tag : str
            Tag to add (e.g., "production", "cross-n-validated", "thesis-figure-7").

        Returns
        -------
        bool
            True if tag was added, False if already present or model not found.
        """
        record = self.get_model(model_id)
        if record is None:
            logger.warning("ModelRegistryDB: model %s not found", model_id)
            return False

        if tag in record.tags:
            return False

        record.tags.append(tag)
        self._save()
        self._record_event(
            "tag_added",
            model_id,
            topology=record.topology,
            details={"tag": tag},
        )
        logger.info("ModelRegistryDB: added tag '%s' to %s", tag, model_id)
        return True

    def remove_tag(self, model_id: str, tag: str) -> bool:
        """Remove a tag from a model.

        Returns
        -------
        bool
            True if tag was removed, False if not present or model not found.
        """
        record = self.get_model(model_id)
        if record is None:
            logger.warning("ModelRegistryDB: model %s not found", model_id)
            return False

        if tag not in record.tags:
            return False

        record.tags.remove(tag)
        self._save()
        self._record_event(
            "tag_removed",
            model_id,
            topology=record.topology,
            details={"tag": tag},
        )
        return True

    def query_by_tag(self, tag: str, *, include_archived: bool = False) -> list[ModelRecord]:
        """Find all models with a specific tag.

        Parameters
        ----------
        tag : str
            Tag to search for.
        include_archived : bool
            Whether to include archived models.

        Returns
        -------
        list[ModelRecord]
            Models with the specified tag.
        """
        results = []
        for r in self._records:
            if not include_archived and r.status == "archived":
                continue
            if tag in r.tags:
                results.append(r)
        return results

    def list_all_tags(self) -> dict[str, int]:
        """Get all tags used across models with their counts.

        Returns
        -------
        dict[str, int]
            Tag → count mapping, sorted by count descending.
        """
        tag_counts: dict[str, int] = {}
        for r in self._records:
            if r.status == "archived":
                continue
            for tag in r.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return dict(sorted(tag_counts.items(), key=lambda x: -x[1]))

    # ═══════════════════════════════════════════════════════════════════════════
    # IMPROVEMENT #4: Best Model Selection (integrates with model_zoo.py)
    # ═══════════════════════════════════════════════════════════════════════════

    def get_best_for_deployment(
        self,
        topology: str,
        model_name: str,
        p_layers: int,
        n_target: int,
        *,
        min_pass_rate: float = MULTI_N_MIN_PASS_RATE,
        prefer_multi_n: bool = True,
        require_tag: str | None = None,
    ) -> ModelRecord | None:
        """Select the best model for deployment to n_target.

        Centralizes selection logic that was scattered across model_zoo.py.
        Uses the same priority hierarchy:

        1. Multi-N model with pass_rate >= min_pass_rate (if prefer_multi_n)
        2. Single-N model with best composite score
        3. Any multi-N model (with warning)
        4. None (no suitable model)

        Parameters
        ----------
        topology : str
            Target topology.
        model_name : str
            Hamiltonian model name.
        p_layers : int
            HVA depth.
        n_target : int
            Target system size.
        min_pass_rate : float
            Minimum pass_rate for multi-N preference.
        prefer_multi_n : bool
            Whether to prefer multi-N over single-N models.
        require_tag : str | None
            If set, only consider models with this tag.

        Returns
        -------
        ModelRecord | None
            Best model, or None if none suitable.
        """
        candidates = self.query(
            topology=topology,
            model_name=model_name,
            p_layers=p_layers,
            status="active",
        )

        if require_tag:
            candidates = [c for c in candidates if require_tag in c.tags]

        if not candidates:
            return None

        # Separate multi-N and single-N
        multi_n = [c for c in candidates if len(c.training.n_values_used) > 1]
        single_n = [c for c in candidates if len(c.training.n_values_used) == 1]

        # Priority 1: Multi-N with good pass_rate
        if prefer_multi_n and multi_n:
            # Get pass_rate from latest evaluation
            def _get_pass_rate(r: ModelRecord) -> float:
                if r.evaluations:
                    return (
                        max(e.pass_rate_dual for e in r.evaluations if e.pass_rate_dual > 0) or 0.0
                    )
                return 0.0

            good_multi_n = [
                c for c in multi_n if _get_pass_rate(c) >= min_pass_rate or _get_pass_rate(c) == 0.0
            ]
            if good_multi_n:
                # Prefer most training points
                return max(good_multi_n, key=lambda c: c.training.total_training_points)

        # Priority 2: Best-scored single-N
        if single_n:

            def _score(r: ModelRecord) -> float:
                """Composite score: points × proximity × pass_rate."""
                n_trained = r.training.n_values_used[0] if r.training.n_values_used else 0
                proximity = 1.0 / (1.0 + abs(n_trained - n_target))
                pass_rate = 0.0
                if r.evaluations:
                    pass_rate = max(
                        (e.pass_rate_dual for e in r.evaluations if e.pass_rate_dual > 0),
                        default=0.0,
                    )
                pr_weight = max(pass_rate, 0.3)  # Floor for unevaluated models
                return r.training.total_training_points * proximity * pr_weight

            return max(single_n, key=_score)

        # Priority 3: Any multi-N (with warning)
        if multi_n:
            logger.warning(
                "get_best_for_deployment: Using multi-N model with low/unknown pass_rate "
                "for %s/%s p=%d. Consider retraining.",
                topology,
                model_name,
                p_layers,
            )
            return max(multi_n, key=lambda c: c.training.total_training_points)

        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # IMPROVEMENT #8: Integrity Validation (reuses model_zoo.py functions)
    # ═══════════════════════════════════════════════════════════════════════════

    def validate_integrity(self, model_id: str) -> IntegrityReport:
        """Validate integrity of a registered model.

        Checks:
        - checkpoint_exists: .pt file exists on disk
        - hash_matches: SHA256 matches manifest (if stored)
        - manifest_consistent: Zoo manifest and registry agree

        Parameters
        ----------
        model_id : str
            Model to validate.

        Returns
        -------
        IntegrityReport
            Validation results.
        """
        from qmbp_simulation.predictors.model_zoo import (
            _compute_file_hash,
            _load_manifest,
        )

        report = IntegrityReport()
        issues: list[str] = []

        record = self.get_model(model_id)
        if record is None:
            issues.append(f"Model {model_id} not found in registry")
            report.issues = issues
            return report

        # Check 1: Checkpoint exists
        ckpt_path = _CHECKPOINTS_DIR / model_id
        report.checkpoint_exists = ckpt_path.exists()
        if not report.checkpoint_exists:
            issues.append(f"Checkpoint file missing: {ckpt_path}")

        # Check 2: Hash matches (compare with zoo manifest)
        manifest_entries = _load_manifest()
        zoo_entry = next(
            (e for e in manifest_entries if e.checkpoint_file == model_id),
            None,
        )

        if zoo_entry is None:
            issues.append("Model not found in zoo manifest")
            report.manifest_consistent = False
        else:
            report.manifest_consistent = True

            # Verify hash if checkpoint exists and hash is stored
            if report.checkpoint_exists and zoo_entry.sha256:
                actual_hash = _compute_file_hash(ckpt_path)
                report.hash_matches = actual_hash == zoo_entry.sha256
                if not report.hash_matches:
                    issues.append(
                        f"Hash mismatch: expected {zoo_entry.sha256[:16]}..., "
                        f"got {actual_hash[:16]}..."
                    )
            elif not zoo_entry.sha256:
                # Legacy entry without hash — consider OK
                report.hash_matches = True
            else:
                report.hash_matches = False

        report.issues = issues
        report.all_ok = (
            report.checkpoint_exists
            and report.hash_matches
            and report.manifest_consistent
            and len(issues) == 0
        )

        # Record in history
        self._record_event(
            "integrity_checked",
            model_id,
            topology=record.topology,
            details={
                "all_ok": report.all_ok,
                "checkpoint_exists": report.checkpoint_exists,
                "hash_matches": report.hash_matches,
                "issues": issues,
            },
        )

        return report

    def validate_all_integrity(self) -> dict[str, Any]:
        """Validate integrity of all active models.

        Returns
        -------
        dict
            Summary with n_checked, n_ok, n_issues, and per-model details.
        """
        results = {
            "n_checked": 0,
            "n_ok": 0,
            "n_issues": 0,
            "models_with_issues": [],
        }

        for record in self._records:
            if record.status != "active":
                continue

            results["n_checked"] += 1
            report = self.validate_integrity(record.model_id)

            if report.all_ok:
                results["n_ok"] += 1
            else:
                results["n_issues"] += 1
                results["models_with_issues"].append(
                    {
                        "model_id": record.model_id,
                        "issues": report.issues,
                    }
                )

        return results

    # ═══════════════════════════════════════════════════════════════════════════
    # IMPROVEMENT #9: Dashboard Integration
    # ═══════════════════════════════════════════════════════════════════════════

    def enrich_from_dashboard(self) -> int:
        """Enrich model records with data from model_quality_dashboard.json.

        Imports quality metrics like training_utility, needs_retrain, etc.

        Returns
        -------
        int
            Number of records enriched.
        """
        if not _DASHBOARD_PATH.exists():
            logger.warning("Dashboard not found: %s", _DASHBOARD_PATH)
            return 0

        with open(_DASHBOARD_PATH) as f:
            dashboard = json.load(f)

        configs = dashboard.get("configs", [])
        enriched = 0
        now = datetime.now(UTC).isoformat()

        for record in self._records:
            # Find matching config in dashboard
            matching = [
                c
                for c in configs
                if (
                    c.get("topology") == record.topology
                    and c.get("p_layers") == record.p_layers
                    and c.get("model") == record.model_name
                )
            ]

            if not matching:
                continue

            # Aggregate across N values if multi-N model
            if len(matching) == 1:
                cfg = matching[0]
            else:
                # Multi-N: aggregate stats
                cfg = {
                    "training_utility": "useful"
                    if any(c.get("training_utility") == "useful" for c in matching)
                    else "not_useful",
                    "training_utility_reason": f"Aggregated from {len(matching)} configs",
                    "pass_rate_dual_criterion": sum(
                        c.get("pass_rate_dual_criterion", 0) for c in matching
                    )
                    / len(matching),
                    "mean_de_gap": sum(c.get("mean_de_gap", 0) for c in matching) / len(matching),
                    "zoo_model_available": any(c.get("zoo_model_available") for c in matching),
                    "zoo_integrity_ok": all(c.get("zoo_integrity_ok", True) for c in matching),
                    "needs_retrain": any(c.get("needs_retrain") for c in matching),
                    "model_stale": any(c.get("model_stale") for c in matching),
                    "n_points": sum(c.get("n_points", 0) for c in matching),
                    "h_frontier": min(c.get("h_frontier", 0) for c in matching)
                    if matching
                    else 0.0,
                }

            # Update dashboard_quality
            record.dashboard_quality = DashboardQuality(
                training_utility=cfg.get("training_utility", ""),
                training_utility_reason=cfg.get("training_utility_reason", ""),
                pass_rate_dual_criterion=cfg.get("pass_rate_dual_criterion", 0.0),
                mean_de_gap=cfg.get("mean_de_gap", 0.0),
                zoo_model_available=cfg.get("zoo_model_available", False),
                zoo_integrity_ok=cfg.get("zoo_integrity_ok", True),
                needs_retrain=cfg.get("needs_retrain", False),
                model_stale=cfg.get("model_stale", False),
                n_points=cfg.get("n_points", 0),
                h_frontier=cfg.get("h_frontier", 0.0),
                last_synced=now,
            )
            enriched += 1

        if enriched:
            self._save()
            logger.info("ModelRegistryDB: enriched %d records from dashboard", enriched)

        return enriched

    def get_training_health(self, model_id: str) -> dict[str, Any]:
        """Get training data health summary for a model.

        Combines registry data with dashboard quality metrics.

        Returns
        -------
        dict
            Health summary including:
            - training_utility: from dashboard
            - needs_retrain: bool
            - quality_issues: list of detected problems
            - recommendation: "use" | "retrain" | "investigate"
        """
        record = self.get_model(model_id)
        if record is None:
            return {"error": f"Model {model_id} not found"}

        dq = record.dashboard_quality
        issues: list[str] = []

        # Check training utility
        if dq.training_utility == "not_useful":
            issues.append(f"Training data marked 'not_useful': {dq.training_utility_reason}")
        elif dq.training_utility == "insufficient_signal":
            issues.append("Training data has insufficient signal")

        # Check if needs retrain
        if dq.needs_retrain:
            issues.append("Dashboard indicates model needs retraining")

        # Check staleness
        if dq.model_stale:
            issues.append("Model is stale (newer training data available)")

        # Check integrity
        if not dq.zoo_integrity_ok:
            issues.append("Zoo integrity check failed")

        # Check pass_rate from evaluations
        latest_pass_rate = 0.0
        if record.evaluations:
            latest_pass_rate = record.evaluations[-1].pass_rate_dual

        if latest_pass_rate > 0 and latest_pass_rate < MULTI_N_MIN_PASS_RATE:
            issues.append(f"Low pass_rate: {latest_pass_rate:.0%} < {MULTI_N_MIN_PASS_RATE:.0%}")

        # Determine recommendation
        if not issues:
            recommendation = "use"
        elif len(issues) >= 2 or dq.training_utility == "not_useful":
            recommendation = "retrain"
        else:
            recommendation = "investigate"

        return {
            "model_id": model_id,
            "topology": record.topology,
            "training_utility": dq.training_utility,
            "training_utility_reason": dq.training_utility_reason,
            "needs_retrain": dq.needs_retrain,
            "model_stale": dq.model_stale,
            "latest_pass_rate": latest_pass_rate,
            "quality_issues": issues,
            "recommendation": recommendation,
            "dashboard_synced": dq.last_synced or "never",
        }

    def set_training_metrics(
        self,
        model_id: str,
        *,
        final_loss: float | None = None,
        final_mse: float | None = None,
        epochs: int | None = None,
        best_epoch: int | None = None,
        training_time_seconds: float | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        early_stopped: bool | None = None,
        convergence_status: str | None = None,
        loss_history: list[float] | None = None,
    ) -> bool:
        """Update training metrics for a model (IMPROVEMENT #1).

        Called after training completes to record training dynamics.

        Returns
        -------
        bool
            True if updated, False if model not found.
        """
        record = self.get_model(model_id)
        if record is None:
            logger.warning("ModelRegistryDB: model %s not found", model_id)
            return False

        metrics = record.training.training_metrics

        if final_loss is not None:
            metrics.final_loss = final_loss
        if final_mse is not None:
            metrics.final_mse = final_mse
        if epochs is not None:
            metrics.epochs = epochs
        if best_epoch is not None:
            metrics.best_epoch = best_epoch
        if training_time_seconds is not None:
            metrics.training_time_seconds = training_time_seconds
        if learning_rate is not None:
            metrics.learning_rate = learning_rate
        if batch_size is not None:
            metrics.batch_size = batch_size
        if early_stopped is not None:
            metrics.early_stopped = early_stopped
        if convergence_status is not None:
            metrics.convergence_status = convergence_status
        if loss_history is not None:
            metrics.loss_history = loss_history

        self._save()
        logger.info("ModelRegistryDB: updated training metrics for %s", model_id)
        return True

    # ═══════════════════════════════════════════════════════════════════════════
    # IMPROVEMENT #5: Model Versioning
    # ═══════════════════════════════════════════════════════════════════════════

    def get_version_info(self, model_id: str) -> dict[str, Any] | None:
        """Get version information for a model.

        Parameters
        ----------
        model_id : str
            Model to query.

        Returns
        -------
        dict | None
            Version info including current version, history, and latest flag.
        """
        record = self.get_model(model_id)
        if record is None:
            return None

        # Check if there's a newer version (this model appears in another's history)
        is_latest = True
        superseding_model = None
        for r in self._records:
            if r.status == "active" and model_id in r.version_history:
                is_latest = False
                superseding_model = r.model_id
                break

        return {
            "model_id": model_id,
            "version": record.version,
            "version_history": record.version_history,
            "is_latest": is_latest,
            "superseding_model": superseding_model,
            "created": record.created,
            "topology": record.topology,
        }

    def get_version_chain(self, model_id: str) -> list[dict[str, Any]]:
        """Get the full version chain for a model.

        Returns all versions in the chain (past and current), ordered chronologically.

        Parameters
        ----------
        model_id : str
            Any model in the version chain (current or historical).

        Returns
        -------
        list[dict]
            Version chain with version number, model_id, created timestamp, and status.
        """
        record = self.get_model(model_id)
        if record is None:
            return []

        chain = []

        # Add historical versions from version_history
        for i, hist_id in enumerate(record.version_history):
            # Try to find the historical record (may be archived)
            hist_record = self.get_model(hist_id)
            if hist_record:
                chain.append(
                    {
                        "version": i + 1,
                        "model_id": hist_id,
                        "created": hist_record.created,
                        "status": hist_record.status,
                        "training_points": hist_record.training.total_training_points,
                    }
                )
            else:
                # Historical record not in DB (only ID preserved)
                chain.append(
                    {
                        "version": i + 1,
                        "model_id": hist_id,
                        "created": "unknown",
                        "status": "unknown",
                        "training_points": 0,
                    }
                )

        # Add current version
        chain.append(
            {
                "version": record.version,
                "model_id": model_id,
                "created": record.created,
                "status": record.status,
                "training_points": record.training.total_training_points,
            }
        )

        return chain

    def list_versions(
        self,
        topology: str | None = None,
        model_name: str | None = None,
        p_layers: int | None = None,
    ) -> list[dict[str, Any]]:
        """List all models with their version info.

        Parameters
        ----------
        topology : str | None
            Filter by topology.
        model_name : str | None
            Filter by model name.
        p_layers : int | None
            Filter by p_layers.

        Returns
        -------
        list[dict]
            List of models with version info, sorted by (topology, model_name, version desc).
        """
        results = []
        for r in self._records:
            if r.status == "archived":
                continue
            if topology and r.topology != topology:
                continue
            if model_name and r.model_name != model_name:
                continue
            if p_layers is not None and r.p_layers != p_layers:
                continue

            results.append(
                {
                    "model_id": r.model_id,
                    "topology": r.topology,
                    "model_name": r.model_name,
                    "p_layers": r.p_layers,
                    "version": r.version,
                    "n_versions_in_history": len(r.version_history),
                    "training_points": r.training.total_training_points,
                    "created": r.created,
                }
            )

        # Sort by topology, model_name, then version descending
        results.sort(key=lambda x: (x["topology"], x["model_name"], -x["version"]))
        return results

    def get_latest_version(
        self,
        topology: str,
        model_name: str,
        p_layers: int,
    ) -> ModelRecord | None:
        """Get the latest version of a model for a given config.

        Parameters
        ----------
        topology : str
            Target topology.
        model_name : str
            Hamiltonian model name.
        p_layers : int
            HVA depth.

        Returns
        -------
        ModelRecord | None
            Latest version, or None if no model exists.
        """
        candidates = self.query(
            topology=topology,
            model_name=model_name,
            p_layers=p_layers,
            status="active",
        )

        if not candidates:
            return None

        # Return the one with highest version
        return max(candidates, key=lambda r: r.version)

    # ═══════════════════════════════════════════════════════════════════════════
    # IMPROVEMENT #10: Failure Diagnostic Integration
    # ═══════════════════════════════════════════════════════════════════════════

    def run_failure_diagnostics(
        self,
        model_id: str,
        *,
        extrapolation_data: dict[int, dict] | None = None,
        force: bool = False,
    ) -> FailureDiagnosticSummary | None:
        """Run failure diagnostics for a model and update its record.

        Uses classify_topology_failure_mode_from_dashboard() from failures_tests.py
        to diagnose the primary failure mode and populate DashboardQuality.failure_diagnostic.

        Parameters
        ----------
        model_id : str
            Model to diagnose.
        extrapolation_data : dict[int, dict] | None
            Optional per-N data for cross-N analysis. If None, only dashboard-based
            diagnostics are run.
        force : bool
            If True, re-run diagnostics even if already present.

        Returns
        -------
        FailureDiagnosticSummary | None
            Diagnostic summary, or None if model not found or diagnostics failed.
        """
        from qmbp_simulation.analysis.failures_tests import (
            classify_topology_failure_mode_from_dashboard,
        )

        record = self.get_model(model_id)
        if record is None:
            logger.warning("ModelRegistryDB: model %s not found", model_id)
            return None

        # Check if already diagnosed and not forced
        existing = record.dashboard_quality.failure_diagnostic
        if existing.primary_mode and not force:
            logger.debug(
                "ModelRegistryDB: %s already diagnosed (primary_mode=%s)",
                model_id,
                existing.primary_mode,
            )
            return existing

        # Load dashboard configs for this topology
        if not _DASHBOARD_PATH.exists():
            logger.warning("Dashboard not found: %s", _DASHBOARD_PATH)
            return None

        with open(_DASHBOARD_PATH) as f:
            dashboard = json.load(f)

        configs = dashboard.get("configs", [])
        topology_configs = [
            c
            for c in configs
            if (
                c.get("topology") == record.topology
                and c.get("p_layers") == record.p_layers
                and c.get("model") == record.model_name
            )
        ]

        if not topology_configs:
            logger.warning(
                "No dashboard configs for %s/%s/p%d",
                record.topology,
                record.model_name,
                record.p_layers,
            )
            return None

        # Run diagnostics
        try:
            full_diag = classify_topology_failure_mode_from_dashboard(
                topology=record.topology,
                dashboard_configs=topology_configs,
                extrapolation_data=extrapolation_data,
            )
        except Exception as exc:
            logger.error("Failed to run diagnostics for %s: %s", model_id, exc)
            return None

        # Convert to summary
        now = datetime.now(UTC).isoformat()
        summary = FailureDiagnosticSummary(
            primary_mode=full_diag.primary_mode,
            confidence=full_diag.confidence,
            secondary_modes=full_diag.secondary_modes,
            explanation=full_diag.explanation,
            per_site_verified=full_diag.per_site_verified,
            gap_masked_fraction=full_diag.training_gap_masked_fraction,
            contamination_severity=None,  # Populated from _diagnose_contaminated_training_dashboard if available
            h_range_overlap=full_diag.h_range_overlap_fraction,
            diagnosed_at=now,
        )

        # Update record
        record.dashboard_quality.failure_diagnostic = summary
        self._save()

        # Auto-tag based on failure mode
        self._auto_tag_from_diagnosis(record, summary)

        # Record event
        self._record_event(
            "failure_diagnosed",
            model_id,
            topology=record.topology,
            details={
                "primary_mode": summary.primary_mode,
                "confidence": summary.confidence,
                "secondary_modes": summary.secondary_modes,
            },
        )

        logger.info(
            "ModelRegistryDB: diagnosed %s → %s (conf=%.0f%%)",
            model_id,
            summary.primary_mode,
            summary.confidence * 100,
        )
        return summary

    def _auto_tag_from_diagnosis(
        self,
        record: ModelRecord,
        diag: FailureDiagnosticSummary,
    ) -> None:
        """Auto-tag model based on failure diagnostic results.

        Tags added:
        - "gap-masked": primary mode is gap_masking
        - "contaminated": primary mode is contaminated_training
        - "ansatz-limited": primary mode is intrinsic_vqe_error
        - "cross-n-degraded": primary mode is generalization_failure
        - "clean": primary mode is healthy with high confidence
        """
        # Remove old diagnostic tags (to allow re-tagging on re-diagnosis)
        diag_tags = {"gap-masked", "contaminated", "ansatz-limited", "cross-n-degraded", "clean"}
        record.tags = [t for t in record.tags if t not in diag_tags]

        # Add new tag based on primary mode
        mode_to_tag = {
            "gap_masking": "gap-masked",
            "contaminated_training": "contaminated",
            "intrinsic_vqe_error": "ansatz-limited",
            "generalization_failure": "cross-n-degraded",
            "healthy": "clean" if diag.confidence >= 0.8 else None,
        }

        new_tag = mode_to_tag.get(diag.primary_mode)
        if new_tag and new_tag not in record.tags:
            record.tags.append(new_tag)
            self._record_event(
                "tag_added",
                record.model_id,
                topology=record.topology,
                details={"tag": new_tag, "source": "auto_diagnosis"},
            )

    def run_all_failure_diagnostics(
        self,
        *,
        topology: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Run failure diagnostics for all active models.

        Parameters
        ----------
        topology : str | None
            If set, only diagnose models for this topology.
        force : bool
            If True, re-run diagnostics even if already present.

        Returns
        -------
        dict
            Summary with n_diagnosed, n_skipped, n_failed, mode_distribution.
        """
        results = {
            "n_diagnosed": 0,
            "n_skipped": 0,
            "n_failed": 0,
            "mode_distribution": {},
        }

        for record in self._records:
            if record.status != "active":
                continue
            if topology and record.topology != topology:
                continue

            summary = self.run_failure_diagnostics(record.model_id, force=force)

            if summary is None:
                results["n_failed"] += 1
            elif (
                summary.primary_mode
                and not force
                and record.dashboard_quality.failure_diagnostic.diagnosed_at
            ):
                results["n_skipped"] += 1
            else:
                results["n_diagnosed"] += 1
                mode = summary.primary_mode
                results["mode_distribution"][mode] = results["mode_distribution"].get(mode, 0) + 1

        logger.info(
            "ModelRegistryDB: failure diagnostics complete — diagnosed=%d, skipped=%d, failed=%d",
            results["n_diagnosed"],
            results["n_skipped"],
            results["n_failed"],
        )
        return results

    def validate_integrity_with_diagnostics(self, model_id: str) -> IntegrityReport:
        """Extended integrity validation including training data quality checks.

        Runs standard integrity checks plus:
        - Training data existence (NPZ files for each N in n_values_used)
        - Training data quality (verified ratio from quality_tier)
        - Failure mode detection from diagnostics

        Parameters
        ----------
        model_id : str
            Model to validate.

        Returns
        -------
        IntegrityReport
            Extended validation results including training data issues.
        """
        import numpy as np

        # Run standard integrity checks first
        report = self.validate_integrity(model_id)

        record = self.get_model(model_id)
        if record is None:
            return report

        # Check training data existence
        npz_dir = _PROJECT_ROOT / "data" / "multi_n_training"
        training_issues: list[str] = []
        n_values = record.training.n_values_used

        total_verified = 0
        total_points = 0
        n_npz_found = 0

        for n in n_values:
            npz_file = npz_dir / f"{record.topology}_N{n}_p{record.p_layers}.npz"
            if npz_file.exists():
                n_npz_found += 1
                try:
                    data = np.load(str(npz_file), allow_pickle=True)
                    tiers = data.get("quality_tier", np.array([]))
                    n_pts = len(data.get("h_values", []))
                    total_points += n_pts

                    if len(tiers) > 0:
                        n_verified = sum(1 for t in tiers if str(t) == "verified")
                        total_verified += n_verified
                except Exception as exc:
                    training_issues.append(f"Failed to read NPZ for N={n}: {exc}")
            else:
                training_issues.append(f"Missing NPZ for N={n}")

        report.training_data_exists = n_npz_found > 0
        report.training_data_verified_ratio = total_verified / max(total_points, 1)

        # Check failure mode from diagnostics
        diag = record.dashboard_quality.failure_diagnostic
        if diag.primary_mode and diag.primary_mode not in ("healthy", "unknown"):
            training_issues.append(f"Failure mode detected: {diag.primary_mode}")
            report.failure_mode_detected = diag.primary_mode

        # Flag concerning patterns
        if report.training_data_verified_ratio < 0.30 and total_points > 0:
            training_issues.append(
                f"Low verified ratio: {report.training_data_verified_ratio:.0%} "
                f"({total_verified}/{total_points})"
            )

        report.training_data_issues = training_issues

        # Update all_ok to include training data status
        report.all_ok = (
            report.checkpoint_exists
            and report.hash_matches
            and report.manifest_consistent
            and len(report.issues) == 0
            and len(training_issues) == 0
        )

        return report

    def get_comprehensive_health(self, model_id: str) -> dict[str, Any]:
        """Get comprehensive health report combining integrity, quality, and diagnostics.

        This is the unified health check that combines:
        - Checkpoint integrity (validate_integrity_with_diagnostics)
        - Training data quality (get_training_health)
        - Failure diagnostics (run_failure_diagnostics)

        Returns
        -------
        dict
            Comprehensive health report with:
            - status: "healthy" | "warning" | "critical"
            - integrity: IntegrityReport summary
            - quality: DashboardQuality summary
            - diagnostics: FailureDiagnosticSummary
            - issues: list of all detected issues
            - recommendation: "deploy" | "investigate" | "retrain" | "do_not_use"
        """
        record = self.get_model(model_id)
        if record is None:
            return {"error": f"Model {model_id} not found", "status": "critical"}

        # Run diagnostics if not already done
        if not record.dashboard_quality.failure_diagnostic.primary_mode:
            self.run_failure_diagnostics(model_id)

        # Get integrity with training data checks
        integrity = self.validate_integrity_with_diagnostics(model_id)

        # Get training health
        health = self.get_training_health(model_id)

        # Collect all issues
        all_issues: list[str] = []
        all_issues.extend(integrity.issues)
        all_issues.extend(integrity.training_data_issues)
        all_issues.extend(health.get("quality_issues", []))

        diag = record.dashboard_quality.failure_diagnostic
        if diag.primary_mode and diag.primary_mode not in ("healthy", "unknown"):
            all_issues.append(
                f"Failure mode: {diag.primary_mode} ({diag.confidence:.0%} confidence)"
            )

        # Determine status and recommendation
        n_critical = sum(1 for i in all_issues if "missing" in i.lower() or "corrupt" in i.lower())
        n_severe = sum(
            1 for i in all_issues if "contaminated" in i.lower() or "severe" in i.lower()
        )

        if n_critical > 0 or not integrity.checkpoint_exists:
            status = "critical"
            recommendation = "do_not_use"
        elif n_severe > 0 or diag.primary_mode in ("contaminated_training", "intrinsic_vqe_error"):
            status = "warning"
            recommendation = "retrain"
        elif len(all_issues) > 2:
            status = "warning"
            recommendation = "investigate"
        elif diag.primary_mode == "healthy" and integrity.all_ok:
            status = "healthy"
            recommendation = "deploy"
        else:
            status = "warning"
            recommendation = "investigate"

        return {
            "model_id": model_id,
            "topology": record.topology,
            "status": status,
            "recommendation": recommendation,
            "integrity": {
                "checkpoint_exists": integrity.checkpoint_exists,
                "hash_matches": integrity.hash_matches,
                "manifest_consistent": integrity.manifest_consistent,
                "training_data_exists": integrity.training_data_exists,
                "training_data_verified_ratio": integrity.training_data_verified_ratio,
                "all_ok": integrity.all_ok,
            },
            "quality": {
                "training_utility": record.dashboard_quality.training_utility,
                "pass_rate_dual": record.dashboard_quality.pass_rate_dual_criterion,
                "needs_retrain": record.dashboard_quality.needs_retrain,
                "model_stale": record.dashboard_quality.model_stale,
            },
            "diagnostics": {
                "primary_mode": diag.primary_mode,
                "confidence": diag.confidence,
                "secondary_modes": diag.secondary_modes,
                "explanation": diag.explanation,
            },
            "issues": all_issues,
            "tags": record.tags,
        }

    def get_models_by_failure_mode(
        self,
        failure_mode: str,
        *,
        min_confidence: float = 0.5,
    ) -> list[ModelRecord]:
        """Query models by their diagnosed failure mode.

        Parameters
        ----------
        failure_mode : str
            Failure mode to filter by (e.g., "gap_masking", "contaminated_training").
        min_confidence : float
            Minimum confidence threshold.

        Returns
        -------
        list[ModelRecord]
            Models matching the failure mode criteria.
        """
        results = []
        for r in self._records:
            if r.status != "active":
                continue
            diag = r.dashboard_quality.failure_diagnostic
            if diag.primary_mode == failure_mode and diag.confidence >= min_confidence:
                results.append(r)
        return results

    def generate_health_dashboard(self) -> dict[str, Any]:
        """Generate a comprehensive health dashboard for all active models.

        Returns
        -------
        dict
            Dashboard with:
            - summary: overall statistics
            - by_status: models grouped by health status
            - by_failure_mode: models grouped by failure mode
            - action_items: prioritized list of issues to address
        """
        summary = {
            "total_active": 0,
            "healthy": 0,
            "warning": 0,
            "critical": 0,
        }

        by_status: dict[str, list[str]] = {"healthy": [], "warning": [], "critical": []}
        by_failure_mode: dict[str, list[str]] = {}
        action_items: list[dict[str, Any]] = []

        for record in self._records:
            if record.status != "active":
                continue

            summary["total_active"] += 1
            health = self.get_comprehensive_health(record.model_id)

            status = health.get("status", "unknown")
            if status in summary:
                summary[status] += 1

            by_status.setdefault(status, []).append(record.model_id)

            diag = record.dashboard_quality.failure_diagnostic
            if diag.primary_mode:
                by_failure_mode.setdefault(diag.primary_mode, []).append(record.model_id)

            # Generate action items for non-healthy models
            if status != "healthy":
                action_items.append(
                    {
                        "model_id": record.model_id,
                        "topology": record.topology,
                        "priority": "high" if status == "critical" else "medium",
                        "recommendation": health.get("recommendation"),
                        "issues": health.get("issues", [])[:3],  # Top 3 issues
                    }
                )

        # Sort action items by priority
        action_items.sort(key=lambda x: 0 if x["priority"] == "high" else 1)

        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "summary": summary,
            "by_status": by_status,
            "by_failure_mode": by_failure_mode,
            "action_items": action_items,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # ResultIndex → ModelRegistryDB Sync (IMPROVEMENT: auto-update on save)
    # ═══════════════════════════════════════════════════════════════════════════

    def sync_evaluation_from_result(
        self,
        result_envelope: dict[str, Any],
        *,
        auto_create: bool = False,
    ) -> bool:
        """Sync evaluation metrics from a save_experiment_result envelope.

        Called automatically by save_experiment_result() when a new result
        is saved. Updates the corresponding ModelRecord with evaluation metrics.

        Parameters
        ----------
        result_envelope : dict
            The result envelope from save_experiment_result containing:
            - config: {topology, model, n_qubits, p_layers, ...}
            - summary: {pass_rate, mean_de_gap, ...}
            - results: per-section results
        auto_create : bool
            If True and no matching model exists, attempt to find the
            corresponding zoo entry and register it. Default False.

        Returns
        -------
        bool
            True if a model was updated, False otherwise.
        """
        config = result_envelope.get("config", {})
        summary = result_envelope.get("summary", {})

        # Extract config fields
        topology = config.get("topology")
        model_name = config.get("model") or config.get("model_name")
        p_layers = config.get("p_layers") or config.get("p", 1)
        n_qubits = config.get("n_qubits") or config.get("n", 0)

        if not topology or not model_name:
            logger.debug("sync_evaluation_from_result: missing topology or model in config")
            return False

        # Find matching model record(s)
        records = self.query(
            topology=topology,
            model_name=model_name,
            p_layers=p_layers,
            status="active",
        )

        if not records and auto_create:
            # Try to find and register from zoo manifest
            try:
                from qmbp_simulation.predictors.model_zoo import list_pretrained

                zoo_entries = list_pretrained(
                    model=model_name,
                    topology=topology,
                    p_layers=p_layers,
                )
                if zoo_entries:
                    self.register_from_zoo_entry(zoo_entries[0])
                    records = self.query(
                        topology=topology,
                        model_name=model_name,
                        p_layers=p_layers,
                    )
            except Exception as e:
                logger.debug("sync_evaluation_from_result: auto_create failed: %s", e)

        if not records:
            logger.debug(
                "sync_evaluation_from_result: no matching model for %s/%s p=%d",
                topology,
                model_name,
                p_layers,
            )
            return False

        # Build evaluation record from summary
        pass_rate_dual = summary.get("pass_rate_dual") or summary.get("pass_rate", 0.0)
        pass_rate_5pct = summary.get("pass_rate_5pct") or summary.get("pass_rate_single", 0.0)
        mean_de_gap = summary.get("mean_de_gap", 0.0)

        # Skip if no meaningful metrics
        if pass_rate_dual == 0 and pass_rate_5pct == 0 and mean_de_gap == 0:
            logger.debug("sync_evaluation_from_result: no metrics in summary")
            return False

        evaluation = EvaluationRecord(
            evaluated_at=result_envelope.get("timestamp", datetime.now(UTC).isoformat()),
            target_n_values=[n_qubits] if n_qubits > 0 else [],
            pass_rate_5pct=pass_rate_5pct,
            pass_rate_dual=pass_rate_dual,
            mean_de_gap=mean_de_gap,
            notes="auto-synced from result_io",
        )

        # Update all matching records (typically just one)
        updated = False
        for record in records:
            self.add_evaluation(record.model_id, evaluation)
            updated = True
            logger.info(
                "sync_evaluation_from_result: updated %s with pass_rate=%.0f%%",
                record.model_id,
                pass_rate_dual * 100,
            )

        return updated

    # ═══════════════════════════════════════════════════════════════════════════
    # NPZ Quality Tier → Zoo Sync (IMPROVEMENT: mark needs_retrain on tier change)
    # ═══════════════════════════════════════════════════════════════════════════

    def mark_needs_retrain_from_npz_update(
        self,
        topology: str,
        n_qubits: int,
        model_name: str,
        p_layers: int,
        old_verified_ratio: float,
        new_verified_ratio: float,
        *,
        improvement_threshold: float = 0.10,
    ) -> bool:
        """Mark model as needs_retrain if NPZ quality tier improved significantly.

        Called when upsert_theta_npz updates quality_tier values. If the
        verified_ratio improved by more than threshold, the model trained
        on this data is now stale and should be retrained.

        Parameters
        ----------
        topology : str
            Lattice topology.
        n_qubits : int
            System size (0 for multi-N models).
        model_name : str
            Hamiltonian model name.
        p_layers : int
            HVA depth.
        old_verified_ratio : float
            Previous fraction of verified points (0.0-1.0).
        new_verified_ratio : float
            New fraction of verified points (0.0-1.0).
        improvement_threshold : float
            Minimum improvement to trigger needs_retrain. Default 10%.

        Returns
        -------
        bool
            True if a model was marked for retrain, False otherwise.
        """
        improvement = new_verified_ratio - old_verified_ratio

        if improvement < improvement_threshold:
            logger.debug(
                "mark_needs_retrain_from_npz_update: improvement %.1f%% < threshold %.1f%%",
                improvement * 100,
                improvement_threshold * 100,
            )
            return False

        # Find matching model records
        records = self.query(
            topology=topology,
            model_name=model_name,
            p_layers=p_layers,
            status="active",
        )

        if not records:
            logger.debug(
                "mark_needs_retrain_from_npz_update: no matching model for %s/%s p=%d",
                topology,
                model_name,
                p_layers,
            )
            return False

        marked = False
        for record in records:
            # Only mark if the record's training data includes this N
            if n_qubits > 0 and n_qubits not in record.training.n_values_used:
                if record.training.n_values_used:  # Skip if multi-N doesn't include this N
                    continue

            record.dashboard_quality.needs_retrain = True
            record.dashboard_quality.model_stale = True

            self._record_event(
                "needs_retrain_flagged",
                record.model_id,
                topology=topology,
                details={
                    "reason": "npz_quality_tier_improved",
                    "n_qubits": n_qubits,
                    "old_verified_ratio": old_verified_ratio,
                    "new_verified_ratio": new_verified_ratio,
                    "improvement": improvement,
                },
            )

            marked = True
            logger.info(
                "mark_needs_retrain_from_npz_update: Marked %s for retrain "
                "(verified_ratio improved from %.0f%% to %.0f%%)",
                record.model_id,
                old_verified_ratio * 100,
                new_verified_ratio * 100,
            )

        if marked:
            self._save()

        return marked

    def update_failure_diagnostic(
        self,
        model_id: str,
        primary_mode: str,
        confidence: float,
        secondary_modes: list[str] | None = None,
        explanation: str = "",
        **extra_metrics,
    ) -> bool:
        """Update failure diagnostic for a model from failures_tests.py analysis.

        Called after running diagnose_* functions to persist the diagnosis
        in the registry for future model selection decisions.

        Parameters
        ----------
        model_id : str
            Model to update.
        primary_mode : str
            Primary failure mode from diagnose_* functions.
            Valid: "gap_masking", "generalization_failure", "intrinsic_vqe_error",
            "contaminated_training", "healthy", "mixed", "unknown".
        confidence : float
            Confidence in the classification (0.0-1.0).
        secondary_modes : list[str] | None
            Secondary failure modes detected.
        explanation : str
            Human-readable explanation of the diagnosis.
        **extra_metrics :
            Additional metrics like per_site_verified, gap_masked_fraction, etc.

        Returns
        -------
        bool
            True if model was found and updated, False otherwise.
        """
        record = self.get_model(model_id)
        if record is None:
            logger.warning("update_failure_diagnostic: model %s not found", model_id)
            return False

        diag = record.dashboard_quality.failure_diagnostic
        diag.primary_mode = primary_mode
        diag.confidence = confidence
        diag.secondary_modes = secondary_modes or []
        diag.explanation = explanation
        diag.diagnosed_at = datetime.now(UTC).isoformat()

        # Update extra metrics if provided
        if "per_site_verified" in extra_metrics:
            diag.per_site_verified = extra_metrics["per_site_verified"]
        if "gap_masked_fraction" in extra_metrics:
            diag.gap_masked_fraction = extra_metrics["gap_masked_fraction"]
        if "contamination_severity" in extra_metrics:
            diag.contamination_severity = extra_metrics["contamination_severity"]
        if "h_range_overlap" in extra_metrics:
            diag.h_range_overlap = extra_metrics["h_range_overlap"]

        # Mark needs_retrain if contaminated
        if primary_mode == "contaminated_training" and confidence > 0.7:
            record.dashboard_quality.needs_retrain = True
            logger.warning(
                "update_failure_diagnostic: %s flagged as contaminated (confidence=%.0f%%). "
                "Model marked for retrain with clean data.",
                model_id,
                confidence * 100,
            )

        self._save()

        self._record_event(
            "failure_diagnostic_updated",
            model_id,
            topology=record.topology,
            details={
                "primary_mode": primary_mode,
                "confidence": confidence,
                "secondary_modes": secondary_modes or [],
            },
        )

        logger.info(
            "update_failure_diagnostic: %s → %s (confidence=%.0f%%)",
            model_id,
            primary_mode,
            confidence * 100,
        )
        return True

    def clear_needs_retrain(self, model_id: str, reason: str = "retrained") -> bool:
        """Clear the needs_retrain flag after retraining.

        Call this after successfully retraining a model to clear the flag.

        Parameters
        ----------
        model_id : str
            Model that was retrained.
        reason : str
            Reason for clearing (for audit trail).

        Returns
        -------
        bool
            True if flag was cleared, False if model not found or flag wasn't set.
        """
        record = self.get_model(model_id)
        if record is None:
            logger.warning("clear_needs_retrain: model %s not found", model_id)
            return False

        if not record.dashboard_quality.needs_retrain:
            logger.debug("clear_needs_retrain: %s didn't have needs_retrain set", model_id)
            return False

        record.dashboard_quality.needs_retrain = False
        record.dashboard_quality.model_stale = False

        self._save()

        self._record_event(
            "needs_retrain_cleared",
            model_id,
            topology=record.topology,
            details={"reason": reason},
        )

        logger.info("clear_needs_retrain: cleared flag for %s (reason: %s)", model_id, reason)
        return True

    def get_models_needing_retrain(self) -> list[ModelRecord]:
        """Get all active models that are flagged for retraining.

        Returns
        -------
        list[ModelRecord]
            Models with needs_retrain=True, sorted by priority (contaminated first).
        """
        needing = [
            r for r in self._records if r.status == "active" and r.dashboard_quality.needs_retrain
        ]

        # Sort by priority: contaminated > stale > other
        def _priority(r: ModelRecord) -> int:
            if r.dashboard_quality.failure_diagnostic.primary_mode == "contaminated_training":
                return 0
            if r.dashboard_quality.model_stale:
                return 1
            return 2

        return sorted(needing, key=_priority)

    # ═══════════════════════════════════════════════════════════════════════════
    # Training Data Hash & Staleness Detection
    # ═══════════════════════════════════════════════════════════════════════════

    def compute_training_data_hash(
        self,
        topology: str,
        n_values: list[int],
        p_layers: int,
        model_name: str = "tfim_bond_resolved",
    ) -> str:
        """Compute MD5 hash of training NPZ files.

        Used to detect if training data has changed since model was trained.

        Parameters
        ----------
        topology : str
            Lattice topology.
        n_values : list[int]
            System sizes used in training.
        p_layers : int
            HVA depth.
        model_name : str
            Hamiltonian model name (affects NPZ path).

        Returns
        -------
        str
            MD5 hash of concatenated NPZ file contents, or "" if files missing.
        """
        npz_dir = _PROJECT_ROOT / "data" / "multi_n_training"

        hasher = hashlib.md5()
        found_any = False

        for n in sorted(n_values):
            npz_file = npz_dir / f"{topology}_N{n}_p{p_layers}.npz"
            if npz_file.exists():
                found_any = True
                # Hash file content
                with open(npz_file, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)

        return hasher.hexdigest() if found_any else ""

    def check_training_data_changed(self, model_id: str) -> tuple[bool, str]:
        """Check if training data has changed since model was trained.

        Compares stored training_data_hash with current NPZ hash.

        Parameters
        ----------
        model_id : str
            Model to check.

        Returns
        -------
        tuple[bool, str]
            (changed, reason) - True if data changed, with explanation.
        """
        record = self.get_model(model_id)
        if record is None:
            return False, "Model not found"

        stored_hash = record.training.training_data_hash
        if not stored_hash:
            return False, "No hash stored (legacy model)"

        current_hash = self.compute_training_data_hash(
            topology=record.topology,
            n_values=record.training.n_values_used,
            p_layers=record.p_layers,
            model_name=record.model_name,
        )

        if not current_hash:
            return False, "Training NPZ files not found"

        if current_hash != stored_hash:
            return True, f"Hash changed: {stored_hash[:8]}... → {current_hash[:8]}..."

        return False, "No change detected"

    def detect_stale_models(self) -> list[dict[str, Any]]:
        """Detect models whose training data has changed.

        Scans all active models and flags those with changed training data.

        Returns
        -------
        list[dict]
            List of stale model reports with model_id, reason, old_hash, new_hash.
        """
        stale_models = []

        for record in self._records:
            if record.status != "active":
                continue

            changed, reason = self.check_training_data_changed(record.model_id)

            if changed:
                current_hash = self.compute_training_data_hash(
                    topology=record.topology,
                    n_values=record.training.n_values_used,
                    p_layers=record.p_layers,
                    model_name=record.model_name,
                )

                stale_models.append(
                    {
                        "model_id": record.model_id,
                        "topology": record.topology,
                        "reason": reason,
                        "old_hash": record.training.training_data_hash[:16],
                        "new_hash": current_hash[:16] if current_hash else "",
                    }
                )

                # Record event and mark as stale
                if not record.dashboard_quality.model_stale:
                    record.dashboard_quality.model_stale = True
                    record.dashboard_quality.needs_retrain = True

                    self._record_event(
                        "training_data_changed",
                        record.model_id,
                        topology=record.topology,
                        details={
                            "old_hash": record.training.training_data_hash[:16],
                            "new_hash": current_hash[:16],
                            "reason": "NPZ files modified since training",
                        },
                    )

        if stale_models:
            self._save()
            logger.info("detect_stale_models: found %d stale models", len(stale_models))

        return stale_models

    # ═══════════════════════════════════════════════════════════════════════════
    # Auto-Sync After Training (Integration with train_unified_mpnn)
    # ═══════════════════════════════════════════════════════════════════════════

    def register_with_training_metrics(
        self,
        zoo_entry: ZooEntry,  # noqa: F821
        training_result: dict[str, Any],
        *,
        architecture_config: dict[str, Any] | None = None,
        optimizer_config: dict[str, Any] | None = None,
        n_values_used: list[int] | None = None,
        auto_diagnose: bool = True,
        auto_sync_dashboard: bool = True,
    ) -> ModelRecord:
        """Register model with full training metrics from train_unified_mpnn().

        This is the primary integration point for automatic post-training
        registration with metrics, diagnostics, and dashboard sync.

        Parameters
        ----------
        zoo_entry : ZooEntry
            Zoo entry from register_checkpoint().
        training_result : dict
            Result dict from train_unified_mpnn() containing:
            - final_mse, val_mse, generalization_gap
            - n_epochs_run, stopped_early, stop_reason
            - mse_history, weight_distribution
        architecture_config : dict | None
            Model architecture: {hidden_dim, n_conv_layers, n_heads, ...}
        optimizer_config : dict | None
            Optimizer config: {learning_rate, weight_decay, scheduler_patience, ...}
        n_values_used : list[int] | None
            System sizes used in training (if not inferrable from zoo_entry).
        auto_diagnose : bool
            If True, run failure diagnostics after registration.
        auto_sync_dashboard : bool
            If True, sync dashboard quality data after registration.

        Returns
        -------
        ModelRecord
            The registered model record with all metrics populated.
        """
        # Infer n_values if not provided
        inferred_n_values = n_values_used or []
        if not inferred_n_values and "multiN_" in zoo_entry.checkpoint_file:
            import re

            match = re.search(r"multiN_([\d+]+)_", zoo_entry.checkpoint_file)
            if match:
                inferred_n_values = [int(x) for x in match.group(1).split("+")]
        if not inferred_n_values and zoo_entry.n_qubits > 0:
            inferred_n_values = [zoo_entry.n_qubits]

        # Build TrainingMetrics from training_result
        training_metrics = TrainingMetrics(
            final_loss=training_result.get("final_mse", 0.0),
            final_mse=training_result.get("final_mse", 0.0),
            final_val_mse=training_result.get("val_mse"),
            generalization_gap=training_result.get("generalization_gap"),
            epochs=training_result.get("n_epochs_run", 0),
            early_stopped=training_result.get("stopped_early", False),
            stop_reason=training_result.get("stop_reason", "unknown"),
            convergence_status=self._infer_convergence_status(training_result),
            loss_history=training_result.get("mse_history", [])[-100:],  # Keep last 100
            weight_distribution=training_result.get("weight_distribution", {}),
        )

        # Build architecture config
        arch_config = ModelArchitectureConfig()
        if architecture_config:
            arch_config = ModelArchitectureConfig(
                hidden_dim=architecture_config.get("hidden_dim", 64),
                n_conv_layers=architecture_config.get("n_conv_layers", 3),
                n_heads=architecture_config.get("n_heads", 1),
                dropout=architecture_config.get("dropout", 0.0),
                activation=architecture_config.get("activation", "relu"),
                include_circuit_nodes=architecture_config.get("include_circuit_nodes", True),
            )

        # Build optimizer config
        opt_config = OptimizerConfig()
        if optimizer_config:
            opt_config = OptimizerConfig(
                optimizer_name=optimizer_config.get("optimizer_name", "AdamW"),
                learning_rate=optimizer_config.get("learning_rate", 0.001),
                weight_decay=optimizer_config.get("weight_decay", 1e-4),
                scheduler=optimizer_config.get("scheduler", "ReduceLROnPlateau"),
                scheduler_patience=optimizer_config.get("scheduler_patience", 300),
                scheduler_factor=optimizer_config.get("scheduler_factor", 0.5),
                layerwise_lr=optimizer_config.get("layerwise_lr", {}),
            )

        # Compute training data hash
        training_data_hash = self.compute_training_data_hash(
            topology=zoo_entry.topology,
            n_values=inferred_n_values,
            p_layers=zoo_entry.p_layers,
            model_name=zoo_entry.model,
        )

        # Build TrainingProvenance
        training = TrainingProvenance(
            n_values_used=inferred_n_values,
            total_training_points=zoo_entry.n_training_points,
            h_range=zoo_entry.h_range,
            seeds=zoo_entry.seeds,
            data_source="multi_n_training" if zoo_entry.n_qubits == 0 else "single_n",
            training_data_hash=training_data_hash,
            validation_split_seed=optimizer_config.get("seed", 42) if optimizer_config else 42,
            training_metrics=training_metrics,
            architecture_config=arch_config,
            optimizer_config=opt_config,
        )

        # Build ModelRecord
        record = ModelRecord(
            model_id=zoo_entry.checkpoint_file,
            model_name=zoo_entry.model,
            architecture="UnifiedMPNN",
            topology=zoo_entry.topology,
            p_layers=zoo_entry.p_layers,
            checkpoint_path=f"data/model_zoo/checkpoints/{zoo_entry.checkpoint_file}",
            created=zoo_entry.created,
            runner_tag=zoo_entry.runner_tag,
            date_tag=zoo_entry.date_tag,
            training=training,
            status="active",
            notes=zoo_entry.notes,
        )

        # Register (may overwrite existing)
        self.register_model(record, overwrite=True)

        # Auto-sync dashboard quality
        if auto_sync_dashboard:
            self._sync_dashboard_for_model(record.model_id)

        # Auto-run failure diagnostics
        if auto_diagnose:
            self.run_failure_diagnostics(record.model_id, force=True)

        logger.info(
            "register_with_training_metrics: registered %s with MSE=%.2e, epochs=%d, stop=%s",
            record.model_id,
            training_metrics.final_mse,
            training_metrics.epochs,
            training_metrics.stop_reason,
        )

        return record

    def _infer_convergence_status(self, training_result: dict[str, Any]) -> str:
        """Infer convergence status from training result."""
        stop_reason = training_result.get("stop_reason", "unknown")

        if stop_reason == "mse_floor_reached":
            return "converged"
        elif stop_reason == "lr_exhausted":
            return "plateau"
        elif stop_reason == "overfitting_detected":
            return "diverged"
        elif stop_reason == "completed":
            # Check if MSE is still decreasing
            history = training_result.get("mse_history", [])
            if len(history) >= 100:
                recent_avg = sum(history[-10:]) / 10
                older_avg = sum(history[-100:-90]) / 10
                if recent_avg < older_avg * 0.95:
                    return "max_epochs"  # Still improving, ran out of epochs
            return "converged"

        return "unknown"

    def _sync_dashboard_for_model(self, model_id: str) -> bool:
        """Sync dashboard quality data for a single model.

        Called automatically after registration to populate DashboardQuality.

        Returns True if sync succeeded.
        """
        record = self.get_model(model_id)
        if record is None:
            return False

        if not _DASHBOARD_PATH.exists():
            logger.debug("_sync_dashboard_for_model: dashboard not found")
            return False

        try:
            with open(_DASHBOARD_PATH) as f:
                dashboard = json.load(f)
        except Exception as exc:
            logger.warning("_sync_dashboard_for_model: failed to load dashboard: %s", exc)
            return False

        configs = dashboard.get("configs", [])
        matching = [
            c
            for c in configs
            if (
                c.get("topology") == record.topology
                and c.get("p_layers") == record.p_layers
                and c.get("model") == record.model_name
            )
        ]

        if not matching:
            logger.debug("_sync_dashboard_for_model: no matching config for %s", model_id)
            return False

        # Aggregate if multiple N values
        if len(matching) == 1:
            cfg = matching[0]
        else:
            cfg = {
                "training_utility": "useful"
                if any(c.get("training_utility") == "useful" for c in matching)
                else "not_useful",
                "training_utility_reason": f"Aggregated from {len(matching)} configs",
                "pass_rate_dual_criterion": sum(
                    c.get("pass_rate_dual_criterion", 0) for c in matching
                )
                / len(matching),
                "mean_de_gap": sum(c.get("mean_de_gap", 0) for c in matching) / len(matching),
                "zoo_model_available": True,
                "zoo_integrity_ok": True,
                "needs_retrain": False,
                "model_stale": False,
                "n_points": sum(c.get("n_points", 0) for c in matching),
                "h_frontier": min(c.get("h_frontier", 0) for c in matching) if matching else 0.0,
            }

        now = datetime.now(UTC).isoformat()

        # Preserve existing failure diagnostic
        existing_diag = record.dashboard_quality.failure_diagnostic

        record.dashboard_quality = DashboardQuality(
            training_utility=cfg.get("training_utility", ""),
            training_utility_reason=cfg.get("training_utility_reason", ""),
            pass_rate_dual_criterion=cfg.get("pass_rate_dual_criterion", 0.0),
            mean_de_gap=cfg.get("mean_de_gap", 0.0),
            zoo_model_available=True,
            zoo_integrity_ok=True,
            needs_retrain=False,  # Just trained, so not stale
            model_stale=False,
            n_points=cfg.get("n_points", 0),
            h_frontier=cfg.get("h_frontier", 0.0),
            last_synced=now,
            failure_diagnostic=existing_diag,
        )

        self._save()

        self._record_event(
            "dashboard_synced",
            model_id,
            topology=record.topology,
            details={
                "training_utility": cfg.get("training_utility"),
                "n_points": cfg.get("n_points"),
            },
        )

        logger.debug("_sync_dashboard_for_model: synced %s", model_id)
        return True

    def record_quality_degradation(
        self,
        model_id: str,
        old_pass_rate: float,
        new_pass_rate: float,
        reason: str = "",
    ) -> None:
        """Record quality degradation event (metrics worsened without retrain).

        This indicates a problem with the underlying data, not the model itself.

        Parameters
        ----------
        model_id : str
            Affected model.
        old_pass_rate : float
            Previous pass_rate.
        new_pass_rate : float
            New (worse) pass_rate.
        reason : str
            Explanation of the degradation.
        """
        record = self.get_model(model_id)
        if record is None:
            logger.warning("record_quality_degradation: model %s not found", model_id)
            return

        delta = new_pass_rate - old_pass_rate

        self._record_event(
            "quality_degraded",
            model_id,
            topology=record.topology,
            details={
                "old_pass_rate": old_pass_rate,
                "new_pass_rate": new_pass_rate,
                "delta": delta,
                "reason": reason,
            },
        )

        logger.warning(
            "record_quality_degradation: %s degraded %.0f%% → %.0f%% (Δ=%.0f%%). %s",
            model_id,
            old_pass_rate * 100,
            new_pass_rate * 100,
            delta * 100,
            reason,
        )

    def record_auto_retrain_triggered(
        self,
        model_id: str,
        trigger_reason: str,
        new_model_id: str | None = None,
    ) -> None:
        """Record that automatic retrain was triggered.

        Parameters
        ----------
        model_id : str
            Model being retrained.
        trigger_reason : str
            Why retrain was triggered (e.g., "training_data_improved",
            "contamination_detected", "staleness_threshold").
        new_model_id : str | None
            ID of the new model (if retrain completed).
        """
        record = self.get_model(model_id)
        if record is None:
            logger.warning("record_auto_retrain_triggered: model %s not found", model_id)
            return

        self._record_event(
            "auto_retrain_triggered",
            model_id,
            topology=record.topology,
            details={
                "trigger_reason": trigger_reason,
                "new_model_id": new_model_id,
                "completed": new_model_id is not None,
            },
        )

        logger.info(
            "record_auto_retrain_triggered: %s triggered by %s%s",
            model_id,
            trigger_reason,
            f" → {new_model_id}" if new_model_id else " (pending)",
        )
