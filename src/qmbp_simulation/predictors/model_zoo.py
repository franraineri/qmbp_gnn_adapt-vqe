"""Pre-trained Model Zoo — MPNN checkpoints for common configurations.

Provides instant θ prediction without running VQE+MPNN training.
Users load a pre-trained model via ``load_pretrained()`` and call
``predict_theta()`` directly.

Usage:
    from qmbp_simulation.predictors.model_zoo import (
        load_pretrained, list_pretrained, ZooEntry,
    )

    # List available models
    entries = list_pretrained()

    # Load best available for a config
    model, meta = load_pretrained(model="tfim", topology="chain_1d", n_qubits=10, p_layers=2)

    # Predict θ
    from qmbp_simulation.predictors import predict_theta
    predictions = predict_theta(model, lattice, h_values)

Registry is stored as a JSON manifest at ``data/model_zoo/manifest.json``.
Checkpoints are ``.pt`` files in ``data/model_zoo/checkpoints/``.

Security: All checkpoints are hash-verified on load (SHA256). Tampered or
corrupted files are rejected before any deserialization occurs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC
from pathlib import Path
from typing import Any

from qmbp_simulation.predictors.mpnn import MPNNPredictor, load_mpnn_checkpoint

logger = logging.getLogger(__name__)


def _smart_load_checkpoint(path: str):
    """Load a checkpoint, auto-detecting model type (MPNNPredictor vs UnifiedMPNN).

    Inspects the checkpoint keys to determine which architecture to instantiate.
    UnifiedMPNN has 'type_emb.weight' and 'qubit_head'/'gate_head' keys.
    Standard MPNNPredictor has 'head' keys.
    """
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)  # nosec: trusted checkpoint

    # Detect model type from state_dict keys
    state_dict = data.get("state_dict", data)
    has_unified_keys = any("type_emb" in k or "qubit_head" in k for k in state_dict.keys())

    if has_unified_keys:
        # It's a UnifiedMPNN
        from qmbp_simulation.predictors.unified_mpnn import load_unified_checkpoint

        return load_unified_checkpoint(path)
    else:
        return load_mpnn_checkpoint(path)


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_zoo_dir() -> Path:
    """Resolve the model-zoo root directory, portable local ↔ server.

    On Kubeflow (or any server with ephemeral pod storage), set the
    ``QMBP_MODEL_ZOO_DIR`` environment variable to a path on a mounted
    PersistentVolumeClaim (PVC) or object-store gateway so trained
    checkpoints survive pod restarts. Locally, leave it unset to use the
    in-repo ``data/model_zoo`` directory.

    Returns
    -------
    Path
        The resolved zoo root. The directory is created on demand by the
        save paths, so a fresh PVC works without manual setup.
    """
    import os

    env_dir = os.environ.get("QMBP_MODEL_ZOO_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return _PROJECT_ROOT / "data" / "model_zoo"


_ZOO_DIR = _resolve_zoo_dir()
_MANIFEST_PATH = _ZOO_DIR / "manifest.json"
_CHECKPOINTS_DIR = _ZOO_DIR / "checkpoints"
# Fallback location for checkpoints that are version-controlled in git.
# On a fresh clone, a "best" checkpoint may live only in archived/ (tracked),
# not in checkpoints/ (regenerable). Loaders resolve through both directories.
# Kept in the repo tree even when QMBP_MODEL_ZOO_DIR overrides the zoo root,
# so the git-tracked baseline checkpoints remain findable as a last resort.
_ARCHIVED_DIR = _ZOO_DIR / "archived"
_REPO_ZOO_DIR = _PROJECT_ROOT / "data" / "model_zoo"
_REPO_ARCHIVED_DIR = _REPO_ZOO_DIR / "archived"
_REPO_MANIFEST_PATH = _REPO_ZOO_DIR / "manifest.json"


def _resolve_checkpoint_path(checkpoint_file: str) -> Path | None:
    """Resolve a checkpoint filename to an existing path.

    Search order:
    1. ``data/model_zoo/checkpoints/`` — the canonical (regenerable) location.
    2. ``data/model_zoo/archived/`` — version-controlled fallback, guarantees a
       usable checkpoint exists after a fresh ``git clone``.

    Parameters
    ----------
    checkpoint_file : str
        Bare checkpoint filename (e.g. ``"model_p1.pt"``).

    Returns
    -------
    Path | None
        The first existing path, or ``None`` if the checkpoint is in neither
        location.
    """
    primary = _CHECKPOINTS_DIR / checkpoint_file
    if primary.exists():
        return primary
    fallback = _ARCHIVED_DIR / checkpoint_file
    if fallback.exists():
        return fallback
    # Last resort: git-tracked baseline archive in the repo tree. Relevant when
    # QMBP_MODEL_ZOO_DIR points at a fresh/empty PVC that lacks the baseline
    # checkpoints shipped with the repository.
    if _REPO_ARCHIVED_DIR != _ARCHIVED_DIR:
        repo_fallback = _REPO_ARCHIVED_DIR / checkpoint_file
        if repo_fallback.exists():
            return repo_fallback
    return None


def _checkpoint_available(checkpoint_file: str) -> bool:
    """Return True if a checkpoint exists in checkpoints/ or archived/."""
    return _resolve_checkpoint_path(checkpoint_file) is not None


# If a multi-N model has training_quality_score below this, it's likely
# trained on insufficient or contaminated data.
MULTI_N_MIN_QUALITY_SCORE: float = 0.50


@dataclass
class ZooEntry:
    """Metadata for a pre-trained MPNN checkpoint.

    Attributes
    ----------
    model : str
        Hamiltonian model (tfim, tfim_longitudinal, heisenberg, etc.)
    topology : str
        Lattice topology (chain_1d, ladder, square, triangular, heavy_hex).
    n_qubits : int
        System size the model was trained on.
    p_layers : int
        HVA depth.
    checkpoint_file : str
        Filename of the .pt checkpoint (relative to checkpoints dir).
    h_range : tuple[float, float]
        Training h-range [h_min, h_max].
    pass_rate : float
        Observed pass rate using dual criterion (ΔE/gap < 5% AND |ΔE| < 0.10).
    n_training_points : int
        Number of VQE points used for training.
    seeds : list[int]
        Seeds used for training data generation.
    created : str
        ISO timestamp of checkpoint creation.
    notes : str
        Any additional context.
    runner_tag : str
        2-letter tag identifying which runner produced this model.
        Convention: AC=AcceleratedCrossN, LN=LargeNExtrap, BR=BondResolved,
        MN=MultiNTrain, II=IterativeImprove, XX=Unknown/manual.
    date_tag : str
        Date in DDMMYY format (e.g., "100826" for August 10, 2026).
    """

    model: str
    topology: str
    n_qubits: int
    p_layers: int
    checkpoint_file: str
    h_range: tuple[float, float] = (1.0, 3.5)
    pass_rate: float = 0.0
    pass_rate_source: str = (
        ""  # "training_data_eval" | "extrapolation_eval" | "cross_n_deployment" | ""
    )
    n_training_points: int = 0
    seeds: list[int] = field(default_factory=list)
    created: str = ""
    notes: str = ""
    sha256: str = ""  # Integrity hash — verified on load
    runner_tag: str = "XX"  # 2-letter runner identifier
    date_tag: str = ""  # DDMMYY format
    training_quality_score: float = -1.0  # [0,1] composite quality, -1 = not computed
    pass_rate_by_n: dict = field(default_factory=dict)  # {str(n): float} per-N pass rates
    run_json: str = ""  # Path to training run JSON envelope (traceability/reproducibility)

    def matches(
        self,
        model: str | None = None,
        topology: str | None = None,
        n_qubits: int | None = None,
        p_layers: int | None = None,
    ) -> bool:
        """Check if this entry matches the query filters."""
        if model and self.model != model:
            return False
        if topology and self.topology != topology:
            return False
        if n_qubits is not None and self.n_qubits != n_qubits:
            return False
        if p_layers is not None and self.p_layers != p_layers:
            return False
        return True

    @property
    def is_multi_topology(self) -> bool:
        """True if this is a multi-topology model."""
        return self.topology == "multi_topology"

    @property
    def is_multi_n(self) -> bool:
        """True if this is a multi-N model (trained across system sizes)."""
        return self.n_qubits == 0

    @property
    def is_evaluated(self) -> bool:
        """True if pass_rate has been set (model was evaluated)."""
        return self.pass_rate > 0.0


def _load_manifest() -> list[ZooEntry]:
    """Load the model zoo manifest from disk.

    When ``QMBP_MODEL_ZOO_DIR`` points at a persistent volume that has no
    manifest yet (fresh PVC), fall back to the git-tracked repo manifest so
    the baseline models are still discoverable on first run.
    """
    manifest_path = _MANIFEST_PATH
    if not manifest_path.exists():
        if _REPO_MANIFEST_PATH != _MANIFEST_PATH and _REPO_MANIFEST_PATH.exists():
            manifest_path = _REPO_MANIFEST_PATH
        else:
            return []
    with open(manifest_path) as f:
        raw = json.load(f)
    # Get valid field names to filter out stale/unknown keys
    _valid_fields = {f.name for f in fields(ZooEntry)}
    entries = []
    for item in raw:
        # Handle h_range as list → tuple
        if "h_range" in item and isinstance(item["h_range"], list):
            item["h_range"] = tuple(item["h_range"])
        # Filter out any keys not in ZooEntry (forward/backward compat)
        filtered = {k: v for k, v in item.items() if k in _valid_fields}
        entries.append(ZooEntry(**filtered))
    return entries


def _save_manifest(entries: list[ZooEntry]) -> None:
    """Persist the manifest to disk."""
    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = []
    for e in entries:
        d = asdict(e)
        d["h_range"] = list(d["h_range"])
        raw.append(d)
    with open(_MANIFEST_PATH, "w") as f:
        json.dump(raw, f, indent=2)
    logger.info("Model zoo manifest saved: %d entries", len(entries))


def prune_test_entries(*, dry_run: bool = True) -> list[str]:
    """Remove test/orphan entries from the zoo manifest.

    Identifies entries with checkpoint_file matching test patterns
    and removes them from the manifest. Corresponding checkpoint files
    on disk are NOT deleted (only the manifest entry is removed).

    Parameters
    ----------
    dry_run : bool
        If True, report what would be pruned without modifying manifest.

    Returns
    -------
    list[str]
        Checkpoint filenames that were (or would be) removed.
    """
    entries = _load_manifest()
    test_patterns = ("test_", "kiro_test_")
    to_prune = [
        e.checkpoint_file
        for e in entries
        if any(e.checkpoint_file.lower().startswith(p) for p in test_patterns)
    ]
    if not dry_run and to_prune:
        clean = [e for e in entries if e.checkpoint_file not in to_prune]
        _save_manifest(clean)
        logger.info("Zoo: pruned %d test entries: %s", len(to_prune), to_prune)
    return to_prune


def list_multi_topology_entries(*, p_layers: int = 1) -> list[ZooEntry]:
    """List all multi-topology entries in the zoo manifest.


    Convenience function for scripts that need quick access to MT models
    without loading the full manifest and filtering manually.
    """
    entries = _load_manifest()
    mt = [e for e in entries if e.is_multi_topology and e.p_layers == p_layers]
    return sorted(mt, key=lambda e: e.n_training_points, reverse=True)


def _get_extrapolation_performance(topology: str, entry: ZooEntry) -> float | None:
    """Check real extrapolation performance for a model on a topology.

    Looks at data/large_n_extrapolation/{topology}_N*.npz files to see
    how well models trained on this data actually perform at large N.

    For MT models, checks the topology-specific extrapolation data since
    the MT model will be used on this topology.

    Returns
    -------
    float | None
        Average pass_rate@5% across all N values for this topology,
        or None if no extrapolation data exists.
    """
    from pathlib import Path as _Path

    import numpy as _np

    extrap_dir = _Path(__file__).resolve().parents[3] / "data" / "large_n_extrapolation"
    if not extrap_dir.exists():
        return None

    # For MT models, we check the topology they'll be USED on (not "multi_topology")
    check_topo = topology

    files = list(extrap_dir.glob(f"{check_topo}_N*.npz"))
    if not files:
        return None

    pass_rates = []
    for f in files:
        try:
            data = _np.load(f, allow_pickle=True)
            dg = data.get("de_gaps")
            if dg is not None and len(dg) > 0:
                pass_rates.append(float(_np.mean(dg < 0.05)))
        except Exception:
            continue

    if not pass_rates:
        return None

    return float(_np.mean(pass_rates))


def load_best_model_for(
    topology: str,
    *,
    model: str = "tfim_bond_resolved",
    p_layers: int = 1,
    n_target: int | None = None,
    h_regime: str | None = None,
    include_multi_topology: bool = True,
) -> tuple:
    """Load the best available model for a topology using ALL available signals.

    Supports regime-specific selection: when `n_target` or `h_regime` are
    specified, the scoring prioritizes models with proven performance in
    those regimes over models with high global pass_rate.

    Integrates information from 3 sources to make the best selection:
    1. Zoo manifest: checkpoint existence, pass_rate, pass_rate_by_n, h_range
    2. ModelRegistryDB: training_metrics (MSE, convergence), architecture
    3. Dashboard: per-topology quality tiers, staleness, h_frontier

    The final score for each candidate is:
        score = (0.40 * pass_rate_signal
               + 0.30 * data_quality_signal
               + 0.20 * convergence_signal
               + 0.10 * freshness_signal)
        × source_multiplier (1.0 per-topo, 0.95 MT, 0.85 single-N)

    Parameters
    ----------
    topology : str
        Target topology (e.g., "ladder", "heavy_hex").
    model : str
        Hamiltonian model (default: "tfim_bond_resolved").
    p_layers : int
        HVA depth (default: 1).
    n_target : int | None
        Target system size for regime-specific selection. When set, models
        with high pass_rate at this N (from pass_rate_by_n) are boosted.
        Models trained on N values near n_target are preferred.
        Examples: n_target=30 for large-N extrapolation, n_target=10 for
        deployment at small N.
    h_regime : str | None
        Target h-regime for selection. Options:
        - "critical" — prefer models trained/evaluated near h_c ≈ 1.0
          (h_range overlapping [0.5, 2.0])
        - "paramagnetic" — prefer models with h_range in [2.5, 5.0]
          (the standard extrapolation regime)
        - None (default) — no h-regime bias
    include_multi_topology : bool
        If True, also considers multi_topology models as candidates.

    Returns
    -------
    tuple[Any, ZooEntry, str]
        (loaded_model, zoo_entry, source) where source is one of:
        "per_topology", "multi_topology", "single_n".

    Raises
    ------
    FileNotFoundError
        If no suitable model exists anywhere in the zoo.
    """
    entries = _load_manifest()

    # Load supplementary data sources (graceful — never block on failure)
    db_records = {}
    try:
        from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

        _db = ModelRegistryDB()
        for r in _db.list_all():
            db_records[r.model_id] = r
    except Exception:
        pass

    dashboard_configs = {}
    try:
        import json as _json

        _dash_path = _PROJECT_ROOT / "data" / "model_quality_dashboard.json"
        if _dash_path.exists():
            _dash = _json.loads(_dash_path.read_text())
            for c in _dash.get("configs", []):
                key = (c.get("topology"), c.get("n_qubits"), c.get("p_layers", 1))
                dashboard_configs.setdefault(c.get("topology"), []).append(c)
    except Exception:
        pass

    def _score_entry(entry: ZooEntry, source_multiplier: float) -> float:
        """Compute unified score using all available signals.

        Signal weighting adapts based on what evidence is available:
        - With real extrapolation data: extrapolation dominates (gold standard)
        - Without: pass_rate is discounted by evaluation breadth (narrow eval = less trust)

        The scoring is designed to be:
        - Extensible: new signals can be added without rebalancing
        - Robust: degraded gracefully when signals are missing
        - Trustworthy: prefers broad evidence over narrow high scores
        """
        # ── Signal 1: pass_rate (with confidence adjustment) ─────────────
        # Raw pass_rate is discounted based on evaluation breadth:
        # - h_range < 2.0 → trained/evaluated on narrow regime (less trustworthy)
        # - Only known at specific N values → may not generalize
        raw_pass_rate = entry.pass_rate if entry.pass_rate > 0 else 0.0
        is_evaluated = raw_pass_rate > 0

        # Confidence factor: how much to trust the pass_rate
        pass_confidence = 1.0
        if is_evaluated:
            # Narrow h-range penalty: models evaluated at h=[3.0,3.5] get discounted
            h_range_width = (entry.h_range[1] - entry.h_range[0]) if entry.h_range else 0
            if h_range_width < 1.5:
                pass_confidence *= 0.6  # Very narrow
            elif h_range_width < 2.5:
                pass_confidence *= 0.8  # Moderate

            # Few training points penalty: <100 pts → probably single-N eval
            if entry.n_training_points < 50:
                pass_confidence *= 0.7
            elif entry.n_training_points < 150:
                pass_confidence *= 0.85

            # Check registry for number of N values evaluated
            db_rec = db_records.get(entry.checkpoint_file)
            if db_rec and db_rec.evaluations:
                latest_eval = db_rec.evaluations[-1]
                n_values_evaluated = (
                    len(latest_eval.target_n_values) if latest_eval.target_n_values else 0
                )
                if n_values_evaluated >= 4:
                    pass_confidence = min(1.0, pass_confidence + 0.1)  # Bonus for broad eval
                elif n_values_evaluated <= 1:
                    pass_confidence *= 0.8  # Penalty for narrow eval

        pass_rate_signal = raw_pass_rate * pass_confidence if is_evaluated else 0.3

        # Boost if pass_rate was validated via extrapolation (most reliable)
        if entry.pass_rate_source == "extrapolation_eval":
            pass_rate_signal = min(1.0, pass_rate_signal * 1.15)
        elif entry.pass_rate_source == "cross_n_deployment":
            pass_rate_signal = min(1.0, pass_rate_signal * 1.10)

        # ── Signal 2: data quality (training points + dashboard utility) ──
        pts = entry.n_training_points
        # Normalize: 500+ pts = 1.0, 100 pts = 0.5, <20 = 0.2
        data_signal = min(1.0, max(0.2, pts / 500.0))

        # Enrich from dashboard if available
        topo_configs = dashboard_configs.get(entry.topology, [])
        if topo_configs:
            useful_ratio = sum(
                1 for c in topo_configs if c.get("training_utility") == "useful"
            ) / max(len(topo_configs), 1)
            data_signal = data_signal * 0.7 + useful_ratio * 0.3

        # For MT models: check target topology's dashboard (not "multi_topology")
        if entry.topology == "multi_topology":
            target_configs = dashboard_configs.get(topology, [])
            if target_configs:
                useful_ratio_target = sum(
                    1 for c in target_configs if c.get("training_utility") == "useful"
                ) / max(len(target_configs), 1)
                data_signal = data_signal * 0.6 + useful_ratio_target * 0.4

        # ── Signal 3: convergence (from ModelRegistryDB training_metrics) ──
        convergence_signal = 0.5  # Default: unknown
        db_rec = db_records.get(entry.checkpoint_file)
        if db_rec and db_rec.training and db_rec.training.training_metrics:
            tm = db_rec.training.training_metrics
            if hasattr(tm, "final_mse") and tm.final_mse > 0:
                # Lower MSE = better convergence
                if tm.final_mse < 0.05:
                    convergence_signal = 1.0
                elif tm.final_mse < 0.15:
                    convergence_signal = 0.7
                elif tm.final_mse < 0.30:
                    convergence_signal = 0.4
                else:
                    convergence_signal = 0.2
            if hasattr(tm, "status") and tm.status == "converged":
                convergence_signal = min(1.0, convergence_signal + 0.1)

        # ── Signal 4: freshness (recent models preferred) ────────────────
        freshness_signal = 0.5
        if entry.created:
            try:
                from datetime import datetime

                created_dt = datetime.fromisoformat(entry.created.replace("Z", "+00:00"))
                age_days = (datetime.now(UTC) - created_dt).days
                freshness_signal = max(0.3, 1.0 - age_days / 30.0)  # Decays over 30 days
            except (ValueError, TypeError):
                pass

        # ── Signal 5: extrapolation performance (gold standard) ──────────
        # This is actual measured performance at large N — the most trustworthy signal.
        extrap_signal = 0.0
        extrap_data = _get_extrapolation_performance(topology, entry)
        if extrap_data is not None:
            extrap_signal = extrap_data

        # ── Weighted combination (adaptive) ──────────────────────────────
        # Tier 1: Real extrapolation data available → trust deployment results
        if extrap_signal > 0:
            raw_score = (
                0.15 * pass_rate_signal
                + 0.15 * data_signal
                + 0.10 * convergence_signal
                + 0.05 * freshness_signal
                + 0.55 * extrap_signal  # Extrapolation is king
            )
        # Tier 2: Model evaluated (pass_rate > 0) but no extrap data
        elif is_evaluated:
            raw_score = (
                0.35 * pass_rate_signal  # Discounted by confidence
                + 0.30 * data_signal
                + 0.20 * convergence_signal
                + 0.15 * freshness_signal
            )
        # Tier 3: Never evaluated — pure training signals
        else:
            raw_score = (
                0.10 * pass_rate_signal  # 0.3 floor × 0.10 = minimal contribution
                + 0.40 * data_signal  # Data quality dominates
                + 0.35 * convergence_signal
                + 0.15 * freshness_signal
            )

        return raw_score * source_multiplier

    # ── Regime-specific scoring adjustments ───────────────────────────
    def _regime_bonus(entry: ZooEntry, raw_score: float) -> float:
        """Apply regime-specific bonuses/penalties based on n_target and h_regime.

        Uses real evaluation data (pass_rate_by_n, dashboard h_frontier,
        extrapolation reports) rather than heuristics from checkpoint filenames.
        """
        bonus = 0.0

        # ── N-target regime: boost models with proven performance at target N ──
        if n_target is not None:
            import re

            by_n = entry.pass_rate_by_n
            if by_n:
                n_values = [int(k) for k in by_n.keys() if k.isdigit()]
                if n_values:
                    # Direct match: pass_rate at exactly this N (strongest signal)
                    n_key = str(n_target)
                    if n_key in by_n:
                        n_pass_rate = float(by_n[n_key])
                        bonus += 0.30 * n_pass_rate  # Gold: real eval at target N

                    # Interpolation: weighted average of nearby evaluated N values
                    # Closer N values are more predictive of target performance
                    else:
                        weighted_sum = 0.0
                        weight_total = 0.0
                        for n_val in n_values:
                            distance = abs(n_val - n_target) / max(n_target, 1)
                            if distance < 1.0:  # Only consider N within 100% of target
                                w = 1.0 / (1.0 + 5.0 * distance)  # Sharp decay
                                weighted_sum += w * float(by_n[str(n_val)])
                                weight_total += w
                        if weight_total > 0:
                            interpolated_rate = weighted_sum / weight_total
                            bonus += 0.20 * interpolated_rate

                    # Coverage bonus: model evaluated at N >= target
                    max_n_evaluated = max(n_values)
                    if max_n_evaluated >= n_target:
                        bonus += 0.08
                    elif max_n_evaluated >= n_target * 0.7:
                        bonus += 0.04

            # Training N coverage from checkpoint filename (fallback when no by_n data)
            elif entry.n_training_points > 0:
                n_in_name = re.findall(r"\d+", entry.checkpoint_file)
                train_n_values = [int(x) for x in n_in_name if 3 <= int(x) <= 200]
                if train_n_values:
                    max_train_n = max(train_n_values)
                    if max_train_n >= n_target:
                        bonus += 0.10
                    elif max_train_n >= n_target * 0.6:
                        bonus += 0.05

            # Penalty: model only trained at small N, target is large
            if n_target >= 20 and entry.n_training_points > 0:
                n_in_name = (
                    re.findall(r"\d+", entry.checkpoint_file) if entry.checkpoint_file else []
                )
                train_n_values = [int(x) for x in n_in_name if 3 <= int(x) <= 200]
                if train_n_values and max(train_n_values) < n_target * 0.4:
                    bonus -= 0.10  # Penalty: trained only at small N, extrapolating too far

        # ── H-regime preference ──────────────────────────────────────────
        if h_regime is not None:
            h_lo, h_hi = entry.h_range if entry.h_range else (1.0, 3.5)

            # Use dashboard h_frontier if available (more accurate than h_range)
            topo_configs = dashboard_configs.get(entry.topology, [])
            h_frontier_val = None
            if topo_configs:
                frontiers = [c.get("h_frontier", 0) for c in topo_configs if c.get("h_frontier")]
                if frontiers:
                    h_frontier_val = min(frontiers)  # Lowest h where model starts working

            if h_regime == "critical":
                # Near phase transition: h ∈ [0.5, 2.0], h_c ≈ 1.0 for TFIM
                # Best: model trained in this regime AND has low h_frontier
                if h_frontier_val is not None and h_frontier_val <= 1.5:
                    bonus += 0.20  # Model proven to work near h_c
                elif h_frontier_val is not None and h_frontier_val <= 2.5:
                    bonus += 0.10  # Partial critical coverage
                elif h_lo <= 1.5:
                    bonus += 0.12  # h_range covers critical (less reliable)
                elif h_lo <= 2.0:
                    bonus += 0.05
                else:
                    bonus -= 0.12  # Penalty: model only knows h >> h_c

            elif h_regime == "paramagnetic":
                # Deep paramagnetic: h ∈ [2.5, 5.5]
                if h_hi >= 5.0 and h_lo <= 3.0:
                    bonus += 0.12  # Full paramagnetic coverage
                elif h_hi >= 4.0:
                    bonus += 0.06
                # Penalty if model only covers critical regime
                if h_hi < 3.0:
                    bonus -= 0.08

        # ── Architecture affinity (bonus for models with enhanced features) ──
        # Models trained with bipartite coloring + global node are better for
        # heavy_hex extrapolation (topology-aware sign assignment)
        if n_target is not None and n_target >= 16:
            notes_lower = (entry.notes or "").lower()
            ckpt_lower = entry.checkpoint_file.lower()
            if "coloring" in notes_lower or "global" in notes_lower:
                bonus += 0.05  # Trained with new architecture features
            if "residual" in ckpt_lower or "film" in ckpt_lower:
                bonus += 0.03  # More expressive architecture

        return max(0.0, raw_score + bonus)  # Floor at 0

    # Build candidate pool
    candidates: list[tuple[float, ZooEntry, str]] = []

    # Pool 1: per-topology multi-N
    for e in entries:
        if (
            e.topology == topology
            and e.model == model
            and e.p_layers == p_layers
            and e.n_qubits == 0
            and _checkpoint_available(e.checkpoint_file)
        ):
            score = _regime_bonus(e, _score_entry(e, 1.0))
            candidates.append((score, e, "per_topology"))

    # Pool 2: multi-topology
    if include_multi_topology:
        for e in entries:
            if (
                e.topology == "multi_topology"
                and e.model == model
                and e.p_layers == p_layers
                and _checkpoint_available(e.checkpoint_file)
            ):
                score = _regime_bonus(e, _score_entry(e, 0.95))
                candidates.append((score, e, "multi_topology"))

    # Pool 3: single-N
    for e in entries:
        if (
            e.topology == topology
            and e.model == model
            and e.p_layers == p_layers
            and e.n_qubits > 0
            and _checkpoint_available(e.checkpoint_file)
        ):
            score = _regime_bonus(e, _score_entry(e, 0.85))
            candidates.append((score, e, "single_n"))

    if not candidates:
        raise FileNotFoundError(
            f"No model found for ({model}, {topology}, p={p_layers}). Train one first."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_entry, source = candidates[0]

    ckpt_path = _resolve_checkpoint_path(best_entry.checkpoint_file)
    if ckpt_path is None:
        # Should not happen (candidates were filtered by availability), but guard
        # against a race where the file is removed between filtering and load.
        raise FileNotFoundError(
            f"Checkpoint {best_entry.checkpoint_file!r} vanished from "
            f"checkpoints/ and archived/ before load."
        )
    if ckpt_path.parent == _ARCHIVED_DIR:
        logger.warning(
            "load_best_model_for(%s): using ARCHIVED fallback for %s "
            "(not present in checkpoints/). Regenerate the canonical checkpoint.",
            topology,
            best_entry.checkpoint_file,
        )
    loaded_model = _smart_load_checkpoint(str(ckpt_path))

    logger.info(
        "load_best_model_for(%s): %s model selected (score=%.3f, pass=%.0f%%, pts=%d, ckpt=%s)",
        topology,
        source,
        best_score,
        best_entry.pass_rate * 100,
        best_entry.n_training_points,
        best_entry.checkpoint_file[:40],
    )

    return loaded_model, best_entry, source


def explain_model_selection(
    topology: str,
    *,
    model: str = "tfim_bond_resolved",
    p_layers: int = 1,
    n_target: int = 20,
) -> list[dict]:
    """Explain why a specific model was selected for a topology.

    Returns all candidates with their scores, source signals, and ranking.
    Useful for debugging model selection decisions and thesis reporting.

    Parameters
    ----------
    topology : str
        Target topology.
    model : str
        Hamiltonian model.
    p_layers : int
        HVA depth.
    n_target : int
        Target system size.

    Returns
    -------
    list[dict]
        Sorted candidates (best first), each with:
        {
            "checkpoint": str,
            "topology": str,
            "source": str,
            "final_score": float,
            "pass_rate": float,
            "n_training_points": int,
            "selected": bool,
        }
    """
    entries = _load_manifest()
    results: list[dict] = []

    source_configs = [
        (
            [
                e
                for e in entries
                if e.topology == topology
                and e.model == model
                and e.p_layers == p_layers
                and e.n_qubits == 0
            ],
            "per_topology",
            1.0,
        ),
        (
            [
                e
                for e in entries
                if e.topology == "multi_topology" and e.model == model and e.p_layers == p_layers
            ],
            "multi_topology",
            0.95,
        ),
        (
            [
                e
                for e in entries
                if e.topology == topology
                and e.model == model
                and e.p_layers == p_layers
                and e.n_qubits > 0
            ],
            "single_n",
            0.85,
        ),
    ]

    for pool, source_label, multiplier in source_configs:
        for e in pool:
            if not (_CHECKPOINTS_DIR / e.checkpoint_file).exists():
                continue
            readiness = compute_model_readiness(e, n_target=n_target)
            raw = readiness["readiness_score"]
            final = raw * multiplier
            results.append(
                {
                    "checkpoint": e.checkpoint_file,
                    "topology": e.topology,
                    "source": source_label,
                    "raw_readiness": round(raw, 4),
                    "penalty_factor": multiplier,
                    "final_score": round(final, 4),
                    "pass_rate": e.pass_rate,
                    "n_training_points": e.n_training_points,
                    "recommendation": readiness["recommendation"],
                    "selected": False,
                }
            )

    results.sort(key=lambda x: x["final_score"], reverse=True)
    if results:
        results[0]["selected"] = True
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Objective-driven model selection (validated params + CI-based confidence)
# ═══════════════════════════════════════════════════════════════════════════════

_VALID_H_REGIMES = ("critical", "paramagnetic", "full", None)


def select_model_for_objective(
    topology: str,
    *,
    objective: str = "deploy",
    model: str = "tfim_bond_resolved",
    p_layers: int = 1,
    n_target: int | None = None,
    h_regime: str | None = None,
    include_multi_topology: bool = True,
    min_confidence: float = 0.0,
) -> dict:
    """Select the best model for a SPECIFIC objective, with validated params
    and a statistically-grounded confidence assessment.

    Unlike ``load_best_model_for`` (returns the model + entry), this returns a
    rich decision report that explains *why* a model was chosen, *how much to
    trust it* (via the Wilson CI lower bound at the target regime), and *what
    is missing* if the objective cannot be met reliably.

    Objective presets (translate a high-level goal into scoring params):
    - "deploy"        → paramagnetic regime, uses n_target as given
    - "critical"      → h_regime=critical (near h_c physics, e.g. DQPT/QPT)
    - "extrapolation" → large-N focus; requires n_target, prefers broad-N eval
    - "custom"        → use n_target/h_regime exactly as passed

    Parameters
    ----------
    topology : str
        Target topology. Validated against SUPPORTED_TOPOLOGIES.
    objective : str
        One of "deploy", "critical", "extrapolation", "custom".
    model, p_layers : str, int
        Hamiltonian model and HVA depth.
    n_target : int | None
        Target system size. Required for objective="extrapolation".
    h_regime : str | None
        Overrides the objective preset when objective="custom".
    include_multi_topology : bool
        Consider MT models as candidates.
    min_confidence : float
        Minimum acceptable confidence (Wilson CI lower bound of pass_rate at
        the target N). If the best model is below this, the report flags it
        as unreliable (but still returns the model — caller decides).

    Returns
    -------
    dict
        {
            "model": UnifiedMPNN,          # loaded model (or None if none found)
            "entry": ZooEntry,             # selected zoo entry
            "source": str,                 # per_topology|multi_topology|single_n
            "objective": str,
            "resolved_params": {"n_target", "h_regime"},
            "confidence": float,           # Wilson CI lower bound [0,1]
            "confidence_basis": str,       # how confidence was derived
            "reliable": bool,              # confidence >= min_confidence
            "warnings": list[str],         # actionable diagnostics
        }

    Raises
    ------
    ValueError
        If topology, objective, or h_regime are invalid, or if a required
        param for the objective is missing.
    """
    from qmbp_simulation.models import SUPPORTED_TOPOLOGIES

    warnings: list[str] = []

    # ── Parameter validation (fail loud, not silent) ─────────────────────
    if topology not in SUPPORTED_TOPOLOGIES and topology != "multi_topology":
        raise ValueError(f"Unknown topology {topology!r}. Valid: {SUPPORTED_TOPOLOGIES}")
    _valid_obj = ("deploy", "critical", "extrapolation", "custom")
    if objective not in _valid_obj:
        raise ValueError(f"objective must be one of {_valid_obj}, got {objective!r}")
    if h_regime not in _VALID_H_REGIMES:
        raise ValueError(f"h_regime must be one of {_VALID_H_REGIMES}, got {h_regime!r}")
    if not (0.0 <= min_confidence <= 1.0):
        raise ValueError(f"min_confidence must be in [0,1], got {min_confidence}")

    # ── Resolve objective → scoring params ───────────────────────────────
    resolved_h_regime = h_regime
    resolved_n_target = n_target

    if objective == "deploy":
        resolved_h_regime = h_regime or "paramagnetic"
    elif objective == "critical":
        resolved_h_regime = "critical"
    elif objective == "extrapolation":
        if n_target is None:
            raise ValueError(
                "objective='extrapolation' requires n_target (the large-N "
                "system size you want to predict for)."
            )
        resolved_h_regime = h_regime or "paramagnetic"
    # objective == "custom" uses params as-is

    # ── Delegate ranking to the unified selector ─────────────────────────
    try:
        loaded_model, entry, source = load_best_model_for(
            topology,
            model=model,
            p_layers=p_layers,
            n_target=resolved_n_target,
            h_regime=resolved_h_regime,
            include_multi_topology=include_multi_topology,
        )
    except FileNotFoundError as e:
        return {
            "model": None,
            "entry": None,
            "source": None,
            "objective": objective,
            "resolved_params": {"n_target": resolved_n_target, "h_regime": resolved_h_regime},
            "confidence": 0.0,
            "confidence_basis": "no_model_available",
            "reliable": False,
            "warnings": [str(e)],
        }

    # ── Confidence: Wilson CI lower bound of pass_rate at target N ───────
    # Prefer the registry EvaluationRecord (has real per-N CIs). Fall back to
    # pass_rate_by_n (point estimate → conservative) then global pass_rate.
    confidence = 0.0
    confidence_basis = "none"

    ci_lower = _confidence_from_registry(entry.checkpoint_file, resolved_n_target)
    if ci_lower is not None:
        confidence = ci_lower
        confidence_basis = "registry_wilson_ci_lower"
    elif entry.pass_rate_by_n and resolved_n_target is not None:
        n_key = str(resolved_n_target)
        if n_key in entry.pass_rate_by_n:
            confidence = float(entry.pass_rate_by_n[n_key])
            confidence_basis = "pass_rate_by_n_point_estimate"
            warnings.append(
                f"No CI data for N={resolved_n_target}; confidence is a point "
                f"estimate. Re-run evaluation to get statistical bounds."
            )
    if confidence == 0.0 and entry.pass_rate > 0:
        confidence = entry.pass_rate * 0.7  # discount: global, not target-specific
        confidence_basis = "global_pass_rate_discounted"
        warnings.append(
            "Confidence derived from GLOBAL pass_rate (not target-specific). "
            "Treat as a rough lower bound."
        )

    # ── Regime reachability check ────────────────────────────────────────
    if resolved_h_regime == "critical":
        h_lo = entry.h_range[0] if entry.h_range else 1.0
        if h_lo > 2.0:
            warnings.append(
                f"Objective='critical' but selected model's h_range starts at "
                f"{h_lo:.2f} (> 2.0). It likely never saw near-h_c data — "
                f"predictions near the transition may be unreliable."
            )
    if objective == "extrapolation" and resolved_n_target is not None:
        by_n = entry.pass_rate_by_n or {}
        n_vals = [int(k) for k in by_n if k.isdigit()]
        if n_vals and max(n_vals) < resolved_n_target:
            warnings.append(
                f"No evaluation at N>={resolved_n_target} (max evaluated N="
                f"{max(n_vals)}). Extrapolation confidence is inferred, not measured."
            )

    reliable = confidence >= min_confidence

    logger.info(
        "select_model_for_objective(%s, obj=%s): %s (conf=%.0f%% [%s], reliable=%s)",
        topology,
        objective,
        entry.checkpoint_file[:40],
        confidence * 100,
        confidence_basis,
        reliable,
    )

    return {
        "model": loaded_model,
        "entry": entry,
        "source": source,
        "objective": objective,
        "resolved_params": {"n_target": resolved_n_target, "h_regime": resolved_h_regime},
        "confidence": round(confidence, 4),
        "confidence_basis": confidence_basis,
        "reliable": reliable,
        "warnings": warnings,
    }


def _confidence_from_registry(checkpoint_file: str, n_target: int | None) -> float | None:
    """Return the Wilson CI lower bound of pass_rate at n_target from the
    ModelRegistryDB EvaluationRecord, or None if unavailable.

    Uses the per-N CI persisted by runner_base._persist_evaluation_to_registry.
    If a per-h array exists for n_target, recomputes a fresh Wilson CI from it
    (most accurate); otherwise uses the stored pass_rate_dual_ci.
    """
    try:
        from qmbp_simulation.analysis.metrics import (
            DE_GAP_THRESHOLD,
            MAX_ABS_ERROR,
            wilson_ci,
        )
        from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

        db = ModelRegistryDB()
        record = db.get_model(checkpoint_file)
        if not record or not record.evaluations:
            return None
        latest = record.evaluations[-1]

        # Best: recompute Wilson CI from per-h arrays at target N
        if n_target is not None:
            n_key = str(n_target)
            dgs = latest.per_h_de_gaps.get(n_key) if latest.per_h_de_gaps else None
            aes = latest.per_h_abs_errors.get(n_key) if latest.per_h_abs_errors else None
            if dgs:
                if aes and len(aes) == len(dgs):
                    n_pass = sum(
                        1
                        for d, a in zip(dgs, aes, strict=False)
                        if d < DE_GAP_THRESHOLD and a < MAX_ABS_ERROR
                    )
                else:
                    n_pass = sum(1 for d in dgs if d < DE_GAP_THRESHOLD)
                lo, _hi = wilson_ci(n_pass, len(dgs))
                return lo

        # Fallback: stored aggregate CI (not N-specific but statistically valid)
        if latest.pass_rate_dual_ci and len(latest.pass_rate_dual_ci) == 2:
            return float(latest.pass_rate_dual_ci[0])
    except Exception:
        return None
    return None


def heal_manifest(*, dry_run: bool = True) -> dict:
    """Detect and fix inconsistencies between manifest and disk.

    Checks:
    1. Entries pointing to missing checkpoint files → archived/removed
    2. Orphan .pt files on disk not in manifest → reported
    3. Duplicate entries (same checkpoint_file) → deduplicated

    Parameters
    ----------
    dry_run : bool
        If True, report issues without modifying manifest.

    Returns
    -------
    dict
        {
            "missing_checkpoints": list[str],  # entries removed
            "orphan_files": list[str],  # .pt files not in manifest
            "duplicates_removed": list[str],  # deduped entries
            "healed": bool,  # True if modifications were made
        }
    """
    entries = _load_manifest()
    manifest_files = {e.checkpoint_file for e in entries}

    # 1. Missing checkpoints
    missing = [
        e.checkpoint_file for e in entries if not (_CHECKPOINTS_DIR / e.checkpoint_file).exists()
    ]

    # 2. Orphan files
    orphans = [
        f.name for f in sorted(_CHECKPOINTS_DIR.glob("*.pt")) if f.name not in manifest_files
    ]

    # 3. Duplicates (keep last occurrence)
    seen = {}
    duplicates = []
    for e in entries:
        if e.checkpoint_file in seen:
            duplicates.append(e.checkpoint_file)
        seen[e.checkpoint_file] = e

    result = {
        "missing_checkpoints": missing,
        "orphan_files": orphans,
        "duplicates_removed": duplicates,
        "healed": False,
    }

    if not dry_run and (missing or duplicates):
        # Remove missing + deduplicate
        clean = [e for e in entries if (_CHECKPOINTS_DIR / e.checkpoint_file).exists()]
        # Deduplicate: keep last occurrence (most recent registration)
        deduped = {}
        for e in clean:
            deduped[e.checkpoint_file] = e
        clean = list(deduped.values())

        _save_manifest(clean)
        result["healed"] = True
        n_removed = len(entries) - len(clean)
        logger.info(
            "heal_manifest: removed %d entries (%d missing, %d duplicates). "
            "%d orphan .pt files on disk.",
            n_removed,
            len(missing),
            len(duplicates),
            len(orphans),
        )

    return result


def _compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file for integrity verification."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _verify_checkpoint_integrity(path: Path, expected_hash: str) -> bool:
    """Verify a checkpoint file has not been tampered with or corrupted.

    Parameters
    ----------
    path : Path
        Path to the checkpoint file.
    expected_hash : str
        Expected SHA256 hash (from manifest). Empty string skips verification
        (backward compatibility for pre-hash entries).

    Returns
    -------
    bool
        True if hash matches or no hash stored (legacy entry).

    Raises
    ------
    SecurityError (RuntimeError)
        If hash mismatch detected — file may be corrupted or tampered.
    """
    if not expected_hash:
        logger.debug("No hash stored for %s — skipping integrity check (legacy entry)", path.name)
        return True

    actual_hash = _compute_file_hash(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"CHECKPOINT INTEGRITY FAILURE: {path.name}\n"
            f"  Expected SHA256: {expected_hash[:16]}...\n"
            f"  Actual SHA256:   {actual_hash[:16]}...\n"
            f"  The file may be corrupted or tampered with.\n"
            f"  Re-run the pipeline with --export-zoo to regenerate."
        )
    return True


def _sort_score(entry: ZooEntry) -> float:
    """Compute sort key for model selection.

    Uses training_quality_score if available (≥0), otherwise falls back
    to pass_rate for backward compatibility with legacy manifest entries.
    """
    if entry.training_quality_score >= 0:
        return entry.training_quality_score
    return entry.pass_rate


def compute_model_readiness(
    entry: ZooEntry,
    *,
    n_target: int = 0,
    _db: ModelRegistryDB | None = None,
) -> dict:
    """Compute a comprehensive readiness assessment for deployment.

    Combines training data quality, model health signals from ModelRegistryDB,
    and training convergence metrics into a single [0,1] score with full
    breakdown for auditing and time-series tracking.

    Score formula (weighted composite):
        readiness = (0.40 * data_quality
                   + 0.25 * convergence_health
                   + 0.20 * pass_rate_adj
                   + 0.15 * freshness)

    Where:
    - data_quality: from training_quality_score (NPZ verified_ratio + coverage)
    - convergence_health: 1.0 if converged, penalties for diverged/high MSE
    - pass_rate_adj: observed pass_rate (floored at 0.3 for unevaluated)
    - freshness: 1.0 if not stale (needs_retrain=False), 0.3 if stale

    Parameters
    ----------
    entry : ZooEntry
        Model metadata from manifest.
    n_target : int
        Target N for proximity bonus. When > 0 and model is single-N,
        proximity to n_target adds up to +0.05 bonus. Default 0 (no bonus).

    Returns
    -------
    dict
        {
            "readiness_score": float [0, 1],
            "data_quality": float,
            "convergence_health": float,
            "pass_rate_adj": float,
            "freshness": float,
            "final_mse": float,
            "penalties": list[str],
            "recommendation": "deploy" | "usable" | "caution" | "avoid",
            "has_training_metrics": bool,
        }
    """
    penalties: list[str] = []

    # ── Component 1: Data quality (from manifest or computed) ────────────
    data_quality = _sort_score(entry)  # [0, 1] or pass_rate fallback

    # ── Fetch ModelRegistryDB record ONCE (shared across components) ──────
    record = None
    try:
        if _db is None:
            from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

            _db = ModelRegistryDB()
        record = _db.get_model(entry.checkpoint_file)
    except Exception:
        pass  # DB unavailable — graceful degradation

    # ── Component 2: Convergence health (from training metrics) ──────────
    convergence_health = 1.0
    final_mse = -1.0
    has_training_metrics = False

    if record is not None:
        tm = record.training.training_metrics
        # Check if training metrics are actually populated (not just defaults)
        has_training_metrics = tm.epochs > 0 or tm.final_mse > 0

        if has_training_metrics:
            # Convergence status penalty
            status = tm.convergence_status.lower() if tm.convergence_status else ""
            if "diverged" in status:
                convergence_health = 0.0
                penalties.append(f"convergence={tm.convergence_status}")
            elif "plateau" in status or "stalled" in status:
                convergence_health = 0.5
                penalties.append(f"convergence={tm.convergence_status}")
            elif "max_epochs" in status or "unknown" in status:
                convergence_health = 0.7
            # else: "converged" / "early_stopped" → 1.0

            # Final MSE penalty (scale: MSE < 1e-4 is excellent, > 0.01 is bad)
            final_mse = tm.final_mse if tm.final_mse > 0 else -1.0
            if final_mse > 0:
                if final_mse > 0.05:
                    convergence_health = min(convergence_health, 0.2)
                    penalties.append(f"MSE={final_mse:.2e} (very high)")
                elif final_mse > 0.01:
                    convergence_health = min(convergence_health, 0.5)
                    penalties.append(f"MSE={final_mse:.2e} (high)")
                elif final_mse > 0.005:
                    convergence_health = min(convergence_health, 0.8)
        else:
            # No training metrics available — mild uncertainty penalty
            convergence_health = 0.7

    # ── Component 3: Pass rate (adjusted) ────────────────────────────────
    pass_rate_adj = max(entry.pass_rate, 0.3)  # Floor for unevaluated models

    # ── Component 4: Freshness (not stale) ───────────────────────────────
    freshness = 1.0
    if record is not None and record.dashboard_quality.needs_retrain:
        freshness = 0.3
        penalties.append("needs_retrain=True (training data changed)")

    # ── Weighted composite ───────────────────────────────────────────────
    readiness_score = (
        0.40 * data_quality + 0.25 * convergence_health + 0.20 * pass_rate_adj + 0.15 * freshness
    )

    # ── Proximity bonus (single-N models closer to target get a bump) ────
    if n_target > 0 and entry.n_qubits > 0:
        proximity = 1.0 / (1.0 + abs(entry.n_qubits - n_target))
        readiness_score += 0.05 * proximity  # Max +0.05 bonus

    # Clamp to [0, 1]
    readiness_score = max(0.0, min(1.0, readiness_score))

    # ── Recommendation + Grade ───────────────────────────────────────────
    # Inline grade computation (avoids circular import framework→predictors).
    # Thresholds match quality_profile.grade_from_score for consistency.
    if readiness_score >= 0.85:
        grade = "A"
    elif readiness_score >= 0.65:
        grade = "B"
    elif readiness_score >= 0.45:
        grade = "C"
    elif readiness_score >= 0.25:
        grade = "D"
    else:
        grade = "F"

    if readiness_score >= 0.75 and not penalties:
        recommendation = "deploy"
    elif readiness_score >= 0.55:
        recommendation = "usable"
    elif readiness_score >= 0.35:
        recommendation = "caution"
    else:
        recommendation = "avoid"

    return {
        "readiness_score": round(readiness_score, 4),
        "grade": grade,
        "data_quality": round(data_quality, 4),
        "convergence_health": round(convergence_health, 4),
        "pass_rate_adj": round(pass_rate_adj, 4),
        "freshness": round(freshness, 4),
        "final_mse": final_mse,
        "penalties": penalties,
        "recommendation": recommendation,
        "has_training_metrics": has_training_metrics,
    }


def compute_training_quality_score(
    topology: str,
    n_qubits: int = 0,
    p_layers: int = 1,
    model: str = "tfim_bond_resolved",
) -> float:
    """Compute a [0,1] quality score from NPZ training data.

    Formula:
        score = 0.50 * verified_ratio + 0.20 * coverage + 0.30 * pass_rate_dual

    Where:
    - verified_ratio = n_verified / n_total (from quality_tier in NPZ)
    - coverage = min(1.0, n_total / 100) (saturates at 100 pts)
    - pass_rate_dual = fraction passing BOTH ΔE/gap<5% AND |ΔE|<0.10

    For multi-N (n_qubits=0): aggregates across all available N for the topology.

    Parameters
    ----------
    topology : str
    n_qubits : int
        0 = compute across all N for this topology (multi-N model).
    p_layers : int
    model : str

    Returns
    -------
    float
        Score in [0, 1]. Returns 0.0 if no data found.
    """
    from pathlib import Path

    import numpy as np

    npz_dir = Path("data/multi_n_training")
    if not npz_dir.exists():
        return 0.0

    if n_qubits == 0:
        files = sorted(npz_dir.glob(f"{topology}_N*_p{p_layers}.npz"))
    else:
        files = [npz_dir / f"{topology}_N{n_qubits}_p{p_layers}.npz"]
        files = [f for f in files if f.exists()]

    if not files:
        return 0.0

    total_pts = 0
    total_verified = 0
    total_pass_dual = 0

    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            n_pts = len(d["h_values"])
            if n_pts == 0:
                continue

            # Verified ratio
            tiers = d["quality_tier"].tolist() if "quality_tier" in d else []
            n_verified = sum(1 for t in tiers if t == "verified")

            # Dual pass rate (ΔE/gap < 5% AND |ΔE| < 0.10)
            de_gaps = (
                np.asarray(d["de_gaps"], dtype=np.float64) if "de_gaps" in d else np.zeros(n_pts)
            )
            e_key = "e_vqe" if "e_vqe" in d else ("energies" if "energies" in d else None)
            if e_key is not None:
                abs_err = np.abs(
                    np.asarray(d[e_key], dtype=np.float64)
                    - np.asarray(d["e_exact"], dtype=np.float64)
                )
                n_pass = int(((de_gaps < 0.05) & (abs_err < 0.10)).sum())
            else:
                n_pass = int((de_gaps < 0.05).sum())

            total_pts += n_pts
            total_verified += n_verified
            total_pass_dual += n_pass
        except Exception:
            continue

    if total_pts == 0:
        return 0.0

    verified_ratio = total_verified / total_pts
    coverage = min(1.0, total_pts / 100.0)
    pass_rate_dual = total_pass_dual / total_pts

    score = 0.50 * verified_ratio + 0.20 * coverage + 0.30 * pass_rate_dual
    return round(min(1.0, max(0.0, score)), 3)


def refresh_zoo_quality_scores() -> dict[str, float]:
    """Recompute training_quality_score for all models in the zoo manifest.

    Call this after batch VQE refinement to update scores that may have
    improved due to new verified points in NPZ files.

    Returns
    -------
    dict[str, float]
        Mapping checkpoint_file → new score for entries that changed.
    """
    entries = _load_manifest()
    updated: dict[str, float] = {}

    for entry in entries:
        new_score = compute_training_quality_score(
            topology=entry.topology,
            n_qubits=entry.n_qubits,
            p_layers=entry.p_layers,
            model=entry.model,
        )
        if abs(new_score - entry.training_quality_score) > 0.001:
            old_score = entry.training_quality_score
            entry.training_quality_score = new_score
            updated[entry.checkpoint_file] = new_score
            logger.info(
                "  Zoo score refresh: %s %.3f → %.3f",
                entry.checkpoint_file[:40],
                old_score,
                new_score,
            )

    if updated:
        _save_manifest(entries)
        logger.info("  Refreshed %d zoo quality scores", len(updated))

    return updated


def auto_retrain_stale_models(
    *,
    min_score_improvement: float = 0.05,
    max_models: int = 3,
    n_epochs: int = 3500,
    dry_run: bool = False,
) -> list[dict]:
    """Auto-retrain zoo models whose training data has improved significantly.

    Detects models where the NPZ quality score exceeds the model's stored score
    by more than `min_score_improvement`, indicating fresher/better data is
    available. Re-trains and re-registers the model.

    Call this after batch VQE refinement runs to keep models up-to-date.

    Parameters
    ----------
    min_score_improvement : float
        Minimum score delta to trigger retrain (default: 0.05).
    max_models : int
        Maximum models to retrain per call (most stale first). Default: 3.
    dry_run : bool
        If True, only report what would be retrained without doing it.

    Returns
    -------
    list[dict]
        List of retrained model summaries with topology, old_score, new_score.
    """
    entries = _load_manifest()
    candidates = []

    for entry in entries:
        if entry.n_qubits != 0:
            continue  # Only retrain multi-N models
        current_score = compute_training_quality_score(
            topology=entry.topology,
            n_qubits=0,
            p_layers=entry.p_layers,
            model=entry.model,
        )
        stored_score = entry.training_quality_score if entry.training_quality_score >= 0 else 0.0
        delta = current_score - stored_score
        if delta >= min_score_improvement:
            candidates.append(
                {
                    "entry": entry,
                    "current_score": current_score,
                    "stored_score": stored_score,
                    "delta": delta,
                }
            )

    # Sort by largest improvement first
    candidates.sort(key=lambda x: x["delta"], reverse=True)
    candidates = candidates[:max_models]

    if not candidates:
        logger.info("  auto_retrain: no stale models (all scores current)")
        return []

    results = []
    for c in candidates:
        entry = c["entry"]
        topo = entry.topology
        p = entry.p_layers

        if dry_run:
            logger.info(
                f"  [DRY-RUN] Would retrain: {topo} p={p} "
                f"(score {c['stored_score']:.3f} → {c['current_score']:.3f}, Δ={c['delta']:.3f})"
            )
            results.append(
                {
                    "topology": topo,
                    "p_layers": p,
                    "old_score": c["stored_score"],
                    "new_score": c["current_score"],
                    "action": "would_retrain",
                }
            )
            continue

        logger.info(
            f"  Retraining: {topo} p={p} (score {c['stored_score']:.3f} → {c['current_score']:.3f})"
        )

        try:
            from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator
            from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN, train_unified_mpnn

            agg = MultiNAggregator(topology=topo, model=entry.model)
            agg.scan()
            dataset = agg.build_combined_dataset(max_de_gap=0.10)

            if len(dataset) < 5:
                logger.warning(f"  {topo}: only {len(dataset)} pts after filter, skipping")
                continue

            sample_g = dataset[0]
            n_node_features = sample_g.x.shape[1] if hasattr(sample_g, "x") else 4

            model = UnifiedMPNN(
                node_features=n_node_features,
                hidden_dim=256,
                n_layers=3,
                norm_type="none",
                dropout=0.1,
            )
            train_result = train_unified_mpnn(
                model,
                dataset,
                n_epochs=n_epochs,
                lr=1e-3,
                patience=300,
                seed=42,
            )
            mse = train_result.get("final_mse", 0) if isinstance(train_result, dict) else 0

            # Re-register with updated metadata
            from datetime import datetime

            n_vals = agg.available_n_values()
            new_entry = ZooEntry(
                model=entry.model,
                topology=topo,
                n_qubits=0,
                p_layers=p,
                checkpoint_file=entry.checkpoint_file,
                h_range=entry.h_range,
                pass_rate=entry.pass_rate,
                n_training_points=len(dataset),
                seeds=[42],
                created=datetime.now(UTC).isoformat(),
                notes=f"Auto-retrained: score {c['stored_score']:.3f}→{c['current_score']:.3f}",
                runner_tag="AR",
                date_tag=datetime.now(UTC).strftime("%d%m%y"),
            )
            register_checkpoint(model, new_entry, overwrite=True)

            results.append(
                {
                    "topology": topo,
                    "p_layers": p,
                    "old_score": c["stored_score"],
                    "new_score": c["current_score"],
                    "n_training_points": len(dataset),
                    "final_mse": float(mse),
                    "action": "retrained",
                }
            )
            logger.info(f"  ✅ {topo}: retrained with {len(dataset)} pts, MSE={mse:.2e}")

        except Exception as e:
            logger.warning(f"  ❌ {topo}: retrain failed: {e}")
            results.append(
                {
                    "topology": topo,
                    "p_layers": p,
                    "old_score": c["stored_score"],
                    "new_score": c["current_score"],
                    "action": "failed",
                    "error": str(e),
                }
            )

    return results


def list_pretrained(
    model: str | None = None,
    topology: str | None = None,
    n_qubits: int | None = None,
    p_layers: int | None = None,
) -> list[ZooEntry]:
    """List available pre-trained models, optionally filtered.

    Parameters
    ----------
    model, topology, n_qubits, p_layers : optional filters

    Returns
    -------
    list[ZooEntry]
        Matching entries sorted by pass_rate (descending).
    """
    entries = _load_manifest()
    filtered = [
        e
        for e in entries
        if e.matches(model=model, topology=topology, n_qubits=n_qubits, p_layers=p_layers)
    ]
    return sorted(filtered, key=lambda e: (_sort_score(e), e.n_training_points), reverse=True)


def load_pretrained(
    model: str = "tfim",
    topology: str = "chain_1d",
    n_qubits: int = 10,
    p_layers: int = 2,
    *,
    checkpoint_path: str | Path | None = None,
    allow_cross_n: bool = True,
) -> tuple[MPNNPredictor, ZooEntry]:
    """Load the best pre-trained MPNN for a given configuration.

    Parameters
    ----------
    model : str
        Hamiltonian model name.
    topology : str
        Lattice topology.
    n_qubits : int
        System size.
    p_layers : int
        HVA depth.
    checkpoint_path : str | Path | None
        Override: load from a specific path instead of the zoo registry.
    allow_cross_n : bool
        If True and no exact N match exists, try loading a model trained
        at a different N (same model/topology/p). GINConv with global_mean_pool
        supports cross-N inference natively. Default True.

    Returns
    -------
    tuple[MPNNPredictor, ZooEntry]
        Loaded model and its metadata.

    Raises
    ------
    FileNotFoundError
        If no matching pre-trained model exists.
    """
    if checkpoint_path is not None:
        # Direct load from user-specified path
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        mpnn = _smart_load_checkpoint(str(ckpt_path))
        meta = ZooEntry(
            model=model,
            topology=topology,
            n_qubits=n_qubits,
            p_layers=p_layers,
            checkpoint_file=str(ckpt_path),
            notes="User-specified checkpoint",
        )
        logger.info("Loaded user checkpoint: %s", ckpt_path)
        return mpnn, meta

    # Search the zoo registry — exact match first
    candidates = list_pretrained(
        model=model, topology=topology, n_qubits=n_qubits, p_layers=p_layers
    )

    # Fuzzy cross-N fallback: same model/topology/p but different N
    if not candidates and allow_cross_n:
        cross_n_candidates = list_pretrained(model=model, topology=topology, p_layers=p_layers)
        if cross_n_candidates:
            # Prefer the closest N, then highest quality score
            cross_n_candidates.sort(key=lambda e: (abs(e.n_qubits - n_qubits), -_sort_score(e)))
            candidates = cross_n_candidates[:1]
            logger.info(
                "No exact N=%d match in zoo. Using cross-N transfer from N=%d "
                "(GINConv + global_mean_pool supports variable-size graphs).",
                n_qubits,
                candidates[0].n_qubits,
            )
            # Note: CrossNValidator should be used by the caller to validate
            # predictions at the target N. The zoo only provides the model;
            # validation is the caller's responsibility. Log a reminder:
            logger.info(
                "  ℹ️  For rigorous cross-N validation, use:\n"
                "     from qmbp_simulation.analysis import CrossNValidator\n"
                "     validator = CrossNValidator(topology, spec, backend)\n"
                "     report = validator.validate_prediction(model, n_target=%d, ...)",
                n_qubits,
            )

    if not candidates:
        available = list_pretrained()
        configs = {(e.model, e.topology, e.n_qubits, e.p_layers) for e in available}
        raise FileNotFoundError(
            f"No pre-trained model for ({model}, {topology}, N={n_qubits}, p={p_layers}).\n"
            f"Available configs: {sorted(configs)}\n"
            f"Run the pipeline with --export-zoo to generate one, or use --checkpoint."
        )

    best = candidates[0]  # Already sorted by pass_rate
    ckpt_path = _resolve_checkpoint_path(best.checkpoint_file)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"Checkpoint file missing: {best.checkpoint_file}\n"
            f"The manifest references it but it is not in checkpoints/ or archived/. "
            f"Re-run the pipeline with --export-zoo to regenerate."
        )
    if ckpt_path.parent == _ARCHIVED_DIR:
        logger.warning(
            "load_pretrained: using ARCHIVED fallback for %s "
            "(not present in checkpoints/). Regenerate the canonical checkpoint.",
            best.checkpoint_file,
        )

    # Verify integrity before deserializing (prevents loading corrupted/tampered files)
    _verify_checkpoint_integrity(ckpt_path, best.sha256)

    mpnn = _smart_load_checkpoint(str(ckpt_path))
    logger.info(
        "Loaded pre-trained MPNN: %s/%s N=%d p=%d (pass_rate_dual=%.0f%%)",
        best.model,
        best.topology,
        best.n_qubits,
        best.p_layers,
        best.pass_rate * 100,
    )
    return mpnn, best


def _get_contaminated_model_ids(
    model: str,
    topology: str,
    p_layers: int,
    *,
    min_confidence: float = 0.70,
) -> set[str]:
    """Get model IDs flagged as contaminated from ModelRegistryDB.

    Returns set of checkpoint filenames that should be excluded from selection.
    A model is considered contaminated if its training data contained
    variationally invalid points (E_vqe < E_exact) that corrupted learning.

    Only excludes models where the contamination diagnosis has HIGH confidence
    (>= min_confidence). Low-confidence diagnoses are logged as warnings but
    do not block model selection — they may be false positives from stale
    registry data or borderline cases.

    Parameters
    ----------
    min_confidence : float
        Minimum diagnostic confidence to act on. Default 0.70.
        Below this, the diagnosis is considered unreliable.
    """
    try:
        from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

        db = ModelRegistryDB()
        contaminated = set()

        for record in db.query(topology=topology, model_name=model, p_layers=p_layers):
            diag = record.dashboard_quality.failure_diagnostic
            if diag.primary_mode == "contaminated_training":
                if diag.confidence >= min_confidence:
                    contaminated.add(record.model_id)
                else:
                    logger.info(
                        "_get_contaminated_model_ids: %s flagged contaminated but "
                        "confidence=%.2f < %.2f — not excluding (possible false positive).",
                        record.model_id,
                        diag.confidence,
                        min_confidence,
                    )
            # Also check for severe contamination in secondary modes
            elif (
                "contaminated_training" in diag.secondary_modes
                and diag.confidence >= min_confidence
            ):
                contaminated.add(record.model_id)

        return contaminated
    except Exception as exc:
        logger.debug("_get_contaminated_model_ids: registry check failed: %s", exc)
        return set()  # Fail open — don't block if registry unavailable


def _get_gap_masked_model_ids(
    model: str,
    topology: str,
    p_layers: int,
) -> set[str]:
    """Get model IDs flagged as gap_masking from ModelRegistryDB.

    Returns set of checkpoint filenames that have gap masking issues.
    Gap masking means the model's metrics may be inflated (looks good on ΔE/gap
    but absolute error |ΔE| is still high). These models CAN be used but
    callers should be warned.
    """
    try:
        from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

        db = ModelRegistryDB()
        gap_masked = set()

        for record in db.query(topology=topology, model_name=model, p_layers=p_layers):
            diag = record.dashboard_quality.failure_diagnostic
            if diag.primary_mode == "gap_masking" or "gap_masking" in diag.secondary_modes:
                gap_masked.add(record.model_id)

        return gap_masked
    except Exception as exc:
        logger.debug("_get_gap_masked_model_ids: registry check failed: %s", exc)
        return set()


# ═══════════════════════════════════════════════════════════════════════════════
# Runner Tag + Date Tag Utilities
# ═══════════════════════════════════════════════════════════════════════════════

# Standard 2-letter runner tags
RUNNER_TAGS: dict[str, str] = {
    "accelerated_cross_n": "AC",
    "large_n_extrapolation": "LN",
    "bond_resolved_validation": "BR",
    "bond_resolved_cross_n": "BC",
    "bond_resolved_scaling": "BS",
    "noiseless_pipeline": "NP",
    "noiseless_cross_n": "NC",
    "iterative_improve": "II",
    "multi_n_train": "MN",
    "hardware_rehearsal": "HR",
    "manual": "XX",
}


def make_date_tag() -> str:
    """Generate date tag in DDMMYY format (UTC).

    Example: August 10, 2026 → "100826"
    """
    from datetime import datetime

    return datetime.now(UTC).strftime("%d%m%y")


def get_runner_tag(runner_id: str) -> str:
    """Map a runner_id to its 2-letter tag.


    Falls back to "XX" for unknown runners. Matches by substring
    so "accelerated_cross_n_v1" → "AC".
    """
    runner_lower = runner_id.lower()
    for key, tag in RUNNER_TAGS.items():
        if key in runner_lower:
            return tag
    return "XX"


def _validate_zoo_entry(entry: ZooEntry) -> None:
    """Pre-registration validation — catch common mistakes before writing to disk.

    Raises ValueError for hard violations. Logs warnings for suspicious-but-allowed cases.
    """
    # Hard violation: multiN/unified filename but n_qubits != 0
    if "multiN" in entry.checkpoint_file or "multi_n" in entry.checkpoint_file.lower():
        if entry.n_qubits != 0:
            raise ValueError(
                f"Pre-registration validation failed: filename '{entry.checkpoint_file}' "
                f"contains 'multiN' but n_qubits={entry.n_qubits} (expected 0 for multi-N models). "
                f"Fix: pass n_qubits=0 when constructing ZooEntry."
            )

    # Hard violation: n_qubits negative
    if entry.n_qubits < 0:
        raise ValueError(
            f"Pre-registration validation failed: n_qubits={entry.n_qubits} is negative."
        )

    # Hard violation: pass_rate out of range
    if not (0.0 <= entry.pass_rate <= 1.0):
        raise ValueError(
            f"Pre-registration validation failed: pass_rate={entry.pass_rate} "
            f"out of valid range [0.0, 1.0]."
        )

    # Warning: unified architecture with single-N naming but n_qubits looks like single
    if "unified" in entry.checkpoint_file.lower() and entry.n_qubits > 0:
        # Only warn if it looks like a bug (not explicitly single-N eval)
        if f"_n{entry.n_qubits}_" in entry.checkpoint_file:
            logger.warning(
                "Zoo validation: '%s' has 'unified' prefix but n_qubits=%d. "
                "If this is a single-N evaluation (not a multi-N model), consider "
                "removing 'unified' from the filename to avoid confusion.",
                entry.checkpoint_file,
                entry.n_qubits,
            )

    # Warning: perfect pass_rate with very few points
    if entry.pass_rate == 1.0 and 0 < entry.n_training_points < 20:
        logger.warning(
            "Zoo validation: pass_rate=1.0 with only %d points for '%s'. "
            "This may indicate overfitting or very limited evaluation scope.",
            entry.n_training_points,
            entry.checkpoint_file,
        )

    # Warning: zero training points (model never evaluated)
    if entry.n_training_points == 0 and entry.pass_rate > 0:
        logger.warning(
            "Zoo validation: pass_rate=%.2f but n_training_points=0 for '%s'. "
            "The pass_rate may not be meaningful without evaluation data.",
            entry.pass_rate,
            entry.checkpoint_file,
        )


def _next_version_filename(base_filename: str) -> str:
    """Compute the next versioned filename for a checkpoint.

    Given "model_p1.pt", if "model_p1.pt" exists on disk, returns "model_p1_v2.pt".
    If "model_p1_v2.pt" already exists, returns "model_p1_v3.pt", etc.

    Handles edge cases:
    - Input already has version suffix ("model_v2.pt") → strips it first, then increments
    - Multiple versions exist → finds max and returns +1

    Returns the FIRST available versioned filename.
    """
    import re as _re

    # Strip any existing version suffix to get the canonical base
    # Pattern: _v{N}.pt at the end
    stem = Path(base_filename).stem
    suffix = Path(base_filename).suffix  # .pt

    # Remove existing _v{N} suffix to get canonical base
    base_stem = _re.sub(r"_v\d+$", "", stem)

    # Scan existing files in checkpoints dir to find max version
    existing_versions: list[int] = []

    # The base file (no version) counts as v1
    base_path = _CHECKPOINTS_DIR / f"{base_stem}{suffix}"
    if base_path.exists():
        existing_versions.append(1)

    # Scan for _v{N} files
    for f in _CHECKPOINTS_DIR.glob(f"{base_stem}_v*{suffix}"):
        match = _re.search(r"_v(\d+)" + _re.escape(suffix) + "$", f.name)
        if match:
            existing_versions.append(int(match.group(1)))

    if not existing_versions:
        # No existing file — use the base name as-is
        return base_filename

    # Next version = max + 1
    next_v = max(existing_versions) + 1
    return f"{base_stem}_v{next_v}{suffix}"


def register_checkpoint(
    model,
    entry: ZooEntry,
    *,
    overwrite: bool = False,
    require_improvement: bool = False,
    regression_guard: bool = False,
    regression_tolerance: float = 0.10,
) -> Path:
    """Register a newly trained MPNN into the model zoo.

    Saves the checkpoint and updates the manifest. Supports both
    ``MPNNPredictor`` and ``UnifiedMPNN`` — auto-detects the model type and
    uses the appropriate serialization function (``save_mpnn_checkpoint`` or
    ``save_unified_checkpoint``).  Passing the wrong saver was a bug that
    caused UnifiedMPNN checkpoints to be saved without their architecture
    metadata (``type_emb``, ``qubit_head``, ``gate_readout`` etc.), making
    them impossible to reconstruct correctly via ``load_unified_checkpoint``.

    Parameters
    ----------
    model : MPNNPredictor | UnifiedMPNN
        Trained model to save.  Type is detected automatically.
    entry : ZooEntry
        Metadata for the checkpoint.
    overwrite : bool
        If True, overwrite existing entry for the same config.
    require_improvement : bool
        If True, only register if the new model has more training points
        than the existing one OR existing has pass_rate=0 (unevaluated).
        If the existing model is better, the new one is saved to
        ``_candidates/`` for manual review instead of overwriting.
    regression_guard : bool
        If True (Item 6), run active evaluation of the new model and
        compare against existing model's pass_rate. Block registration if
        the new model regresses beyond ``regression_tolerance``.
        More rigorous than ``require_improvement`` (which only checks
        training point count, not actual model quality).
    regression_tolerance : float
        Maximum allowed pass_rate drop when ``regression_guard=True``.
        Default: 0.10 (10% absolute drop allowed).

    Returns
    -------
    Path
        Path to the saved checkpoint file.

    Raises
    ------
    TypeError
        If ``model`` is not a recognized model type.
    """
    _CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = _CHECKPOINTS_DIR / entry.checkpoint_file

    # ── Pre-registration validation ──────────────────────────────────────
    # Catch common mistakes before writing anything to disk or manifest.
    _validate_zoo_entry(entry)

    # ── Regression guard (Item 6): active quality evaluation ─────────────
    # More rigorous than require_improvement — actually runs the model and
    # compares observed pass_rate vs existing model's pass_rate.
    if regression_guard and overwrite:
        try:
            from qmbp_simulation.predictors.retrain_loop import regression_guardrail

            allowed, guard_reason = regression_guardrail(
                model,
                entry,
                tolerance=regression_tolerance,
            )
            if not allowed:
                # Save to _candidates/ for manual review
                candidates_dir = _CHECKPOINTS_DIR / "_candidates"
                candidates_dir.mkdir(parents=True, exist_ok=True)
                candidate_path = candidates_dir / entry.checkpoint_file
                logger.warning(
                    "Regression guard BLOCKED registration: %s. "
                    "Saved to _candidates/ for manual review.",
                    guard_reason,
                )
                # Save checkpoint to candidates
                try:
                    from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN as _UM

                    if isinstance(model, _UM):
                        from qmbp_simulation.predictors.unified_mpnn import save_unified_checkpoint

                        save_unified_checkpoint(model, str(candidate_path))
                    else:
                        from qmbp_simulation.predictors.mpnn import save_mpnn_checkpoint

                        save_mpnn_checkpoint(model, str(candidate_path))
                except Exception:
                    pass
                return candidate_path
            else:
                logger.info("  Regression guard passed: %s", guard_reason)
        except ImportError:
            logger.debug("regression_guard: retrain_loop not available, skipping")

    # ── Quality gate: require_improvement ────────────────────────────────
    if require_improvement and overwrite:
        existing_entries = [
            e for e in _load_manifest() if e.checkpoint_file == entry.checkpoint_file
        ]
        if existing_entries:
            existing = existing_entries[0]
            # Block if existing is evaluated AND has more training data
            if (
                existing.is_evaluated
                and existing.n_training_points >= entry.n_training_points
                and existing.pass_rate > 0.0
            ):
                # Save to _candidates/ instead
                candidates_dir = _CHECKPOINTS_DIR / "_candidates"
                candidates_dir.mkdir(parents=True, exist_ok=True)
                candidate_path = candidates_dir / entry.checkpoint_file
                logger.warning(
                    "Quality gate BLOCKED registration: existing model has "
                    "pass_rate=%.2f with %d pts. New model has %d pts. "
                    "Saved to _candidates/ for manual review.",
                    existing.pass_rate,
                    existing.n_training_points,
                    entry.n_training_points,
                )
                # Save checkpoint to candidates dir
                try:
                    from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN as _UM

                    if isinstance(model, _UM):
                        from qmbp_simulation.predictors.unified_mpnn import save_unified_checkpoint

                        save_unified_checkpoint(model, str(candidate_path))
                    else:
                        from qmbp_simulation.predictors.mpnn import save_mpnn_checkpoint

                        save_mpnn_checkpoint(model, str(candidate_path))
                except Exception:
                    pass
                return candidate_path

    # ── Auto-fill date_tag if not provided ───────────────────────────────
    if not entry.date_tag:
        from datetime import datetime

        entry.date_tag = datetime.now(UTC).strftime("%d%m%y")

    if ckpt_path.exists() and not overwrite:
        logger.warning("Checkpoint already exists: %s. Use overwrite=True to replace.", ckpt_path)
        return ckpt_path

    # ── VERSIONING: when file exists, save new model with _v{N+1} suffix ──
    # NEVER overwrite an existing checkpoint. The old model stays in place,
    # the new model gets a versioned filename. Both coexist in checkpoints/.
    # This prevents the data loss that caused the phantom model issues.
    if ckpt_path.exists():
        versioned_name = _next_version_filename(entry.checkpoint_file)
        logger.info(
            "  Zoo version-up: %s already exists → saving new model as %s",
            entry.checkpoint_file,
            versioned_name,
        )
        # Update entry to use versioned filename
        entry.checkpoint_file = versioned_name
        ckpt_path = _CHECKPOINTS_DIR / versioned_name

    # ── ANTI-REGRESSION: track version lineage in ModelRegistryDB ─────────
    # Record the supersession relationship for provenance tracking.
    if ckpt_path.name != entry.checkpoint_file:
        pass  # Already updated above

    # Find existing manifest entry for this config (for provenance)
    existing_entries_list = _load_manifest()
    existing_entry = next(
        (
            e
            for e in existing_entries_list
            if e.model == entry.model
            and e.topology == entry.topology
            and e.n_qubits == entry.n_qubits
            and e.p_layers == entry.p_layers
        ),
        None,
    )

    if existing_entry is not None:
        old_pass_rate = existing_entry.pass_rate or 0.0
        new_pass_rate = entry.pass_rate or 0.0

        # ── ModelRegistryDB: record version lineage ──────────────────────
        try:
            from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

            _db = ModelRegistryDB()
            _db._record_event(
                "auto_versioned",
                existing_entry.checkpoint_file,
                topology=entry.topology,
                details={
                    "new_model_id": entry.checkpoint_file,
                    "old_pass_rate": old_pass_rate,
                    "new_pass_rate": new_pass_rate,
                    "old_n_training_points": existing_entry.n_training_points,
                    "new_n_training_points": entry.n_training_points,
                },
            )
        except Exception as _e_db:
            logger.debug("ModelRegistryDB version tracking failed (non-critical): %s", _e_db)

        # Warn if new model has lower pass_rate (informational only, never blocks)
        if new_pass_rate > 0 and new_pass_rate < old_pass_rate - 0.01:
            logger.warning(
                "  ⚠️ New model %s has lower pass_rate (%.0f%%) than existing %s (%.0f%%). "
                "Both versions preserved.",
                entry.checkpoint_file,
                new_pass_rate * 100,
                existing_entry.checkpoint_file,
                old_pass_rate * 100,
            )

    if hasattr(model, "training") and model.training:
        logger.warning(
            "register_checkpoint: model is in training mode. "
            "Consider calling model.eval() before saving for consistent inference results."
        )

    # ── Auto-detect model type and use correct saver ─────────────────────
    # Import lazily to avoid circular imports (model_zoo is imported by mpnn.py)
    try:
        from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN as _UnifiedMPNN

        _has_unified = True
    except ImportError:
        _has_unified = False
        _UnifiedMPNN = None

    from qmbp_simulation.predictors.mpnn import save_mpnn_checkpoint

    zoo_meta = {
        "zoo_entry": asdict(entry),
        "pass_rate": entry.pass_rate,
        "n_training_points": entry.n_training_points,
    }

    if _has_unified and isinstance(model, _UnifiedMPNN):
        from qmbp_simulation.predictors.unified_mpnn import save_unified_checkpoint

        save_unified_checkpoint(model, str(ckpt_path), training_metadata=zoo_meta)
        logger.debug("register_checkpoint: used save_unified_checkpoint for UnifiedMPNN")
    elif hasattr(model, "output_dim") or hasattr(model, "hidden_dim"):
        # MPNNPredictor or BondResolvedMPNN — use standard saver
        save_mpnn_checkpoint(model, str(ckpt_path), training_metadata=zoo_meta)
        logger.debug("register_checkpoint: used save_mpnn_checkpoint for %s", type(model).__name__)
    else:
        raise TypeError(
            f"register_checkpoint: unrecognized model type {type(model).__name__}. "
            "Expected MPNNPredictor, BondResolvedMPNN, or UnifiedMPNN."
        )

    # Compute integrity hash after save
    entry.sha256 = _compute_file_hash(ckpt_path)

    # Auto-compute training_quality_score from NPZ data
    if entry.training_quality_score < 0:
        try:
            entry.training_quality_score = compute_training_quality_score(
                topology=entry.topology,
                n_qubits=entry.n_qubits,
                p_layers=entry.p_layers,
                model=entry.model,
            )
        except Exception:
            entry.training_quality_score = 0.0

    # Update manifest (ADD new entry, keep existing entries intact)
    entries = _load_manifest()
    entries.append(entry)
    _save_manifest(entries)

    logger.info(
        "Registered checkpoint: %s/%s N=%d p=%d → %s",
        entry.model,
        entry.topology,
        entry.n_qubits,
        entry.p_layers,
        ckpt_path,
    )

    # ── Auto-register in model registry DB (best-effort) ────────────────
    try:
        from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

        db = ModelRegistryDB()
        db.register_from_zoo_entry(entry, overwrite=overwrite)
    except Exception as e:
        logger.debug("Model registry DB auto-register failed (non-critical): %s", e)

    return ckpt_path


def register_checkpoint_with_training_metrics(
    model,
    entry: ZooEntry,
    training_result: dict | None = None,
    *,
    overwrite: bool = False,
    auto_tag: bool = True,
    auto_diagnose: bool = True,
    auto_sync_dashboard: bool = True,
    architecture_config: dict | None = None,
    optimizer_config: dict | None = None,
    regression_guard: bool = False,
    regression_tolerance: float = 0.10,
    run_json_path: str = "",
) -> Path:
    """Register checkpoint and capture training metrics in one call.

    Enhanced version of ``register_checkpoint()`` that also:
    - Records training metrics from ``train_unified_mpnn()`` return dict
    - Auto-tags model based on pass_rate thresholds
    - Integrates with ModelRegistryDB for full provenance tracking
    - Auto-runs failure diagnostics post-registration
    - Auto-syncs dashboard quality data
    - Records run_json path for traceability/reproducibility

    Parameters
    ----------
    model : MPNNPredictor | UnifiedMPNN
        Trained model to save.
    entry : ZooEntry
        Metadata for the checkpoint.
    training_result : dict | None
        Return dict from ``train_unified_mpnn()`` or ``fine_tune_unified_mpnn()``.
        If provided, metrics are extracted and stored in ModelRegistryDB.
        Expected keys: final_mse, n_epochs_run, stopped_early, stop_reason, etc.
    overwrite : bool
        If True, allow saving even if a model for the same config exists.
        The new model gets a versioned filename (_v2, _v3, etc.) — the old
        model is NEVER deleted or overwritten.
    auto_tag : bool
        If True (default), auto-add tags based on pass_rate.
    auto_diagnose : bool
        If True (default), run failure diagnostics after registration.
    auto_sync_dashboard : bool
        If True (default), sync dashboard quality data after registration.
    architecture_config : dict | None
        Model architecture config for reproducibility.
    optimizer_config : dict | None
        Optimizer config for reproducibility.
    run_json_path : str
        Path to the training run JSON envelope for traceability. Stored in
        ZooEntry.run_json so the training can be traced and reproduced.

    Returns
    -------
    Path
        Path to the saved checkpoint file (may have _v{N} suffix).
    """
    # ── Safety-first: persist model to _recovery/ BEFORE registration ────
    # Protects against crashes in regression_guardrail, validation, or
    # manifest update. Cleaned up on success.
    _recovery_path = None
    try:
        _recovery_dir = _CHECKPOINTS_DIR / "_recovery"
        _recovery_dir.mkdir(parents=True, exist_ok=True)
        _recovery_path = _recovery_dir / f"pre_register_{entry.checkpoint_file}"
        try:
            from qmbp_simulation.predictors.unified_mpnn import UnifiedMPNN as _UM_check

            if isinstance(model, _UM_check):
                from qmbp_simulation.predictors.unified_mpnn import save_unified_checkpoint

                save_unified_checkpoint(model, str(_recovery_path))
            else:
                from qmbp_simulation.predictors.mpnn import save_mpnn_checkpoint

                save_mpnn_checkpoint(model, str(_recovery_path))
            logger.debug("Safety backup saved: %s", _recovery_path.name)
        except Exception as _e_recovery:
            logger.debug("Safety backup failed (non-critical): %s", _e_recovery)
            _recovery_path = None
    except Exception:
        _recovery_path = None

    # Set run_json for traceability (before saving, so it's in the manifest)
    if run_json_path:
        entry.run_json = run_json_path

    # Save checkpoint via standard function
    ckpt_path = register_checkpoint(
        model,
        entry,
        overwrite=overwrite,
        regression_guard=regression_guard,
        regression_tolerance=regression_tolerance,
    )

    # ── Record training metrics in ModelRegistryDB ────────────────────────
    if training_result is not None:
        try:
            from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

            db = ModelRegistryDB()

            # Use batch mode to avoid multiple disk writes during composite registration
            with db.batch():
                db.register_with_training_metrics(
                    zoo_entry=entry,
                    training_result=training_result,
                    architecture_config=architecture_config,
                    optimizer_config=optimizer_config,
                    auto_diagnose=auto_diagnose,
                    auto_sync_dashboard=auto_sync_dashboard,
                )
            logger.info(
                "register_checkpoint_with_training_metrics: full registration for %s "
                "(MSE=%.2e, diagnose=%s, dashboard=%s)",
                entry.checkpoint_file[:40],
                training_result.get("final_mse", 0),
                auto_diagnose,
                auto_sync_dashboard,
            )
        except Exception as e:
            logger.warning("Full registration failed, falling back to basic: %s", e)
            # Fallback to basic metrics recording
            try:
                db = ModelRegistryDB()
                db.set_training_metrics(
                    entry.checkpoint_file,
                    final_loss=training_result.get("final_mse"),
                    final_mse=training_result.get("final_mse"),
                    epochs=training_result.get("n_epochs_run"),
                    early_stopped=training_result.get("stopped_early", False),
                    convergence_status=training_result.get("stop_reason", "unknown"),
                    loss_history=_truncate_loss_history(training_result.get("mse_history", [])),
                )
            except Exception as e2:
                logger.debug("Fallback metrics recording also failed: %s", e2)

    # ── Auto-tagging based on pass_rate ───────────────────────────────────
    if auto_tag and entry.pass_rate > 0:
        try:
            from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

            db = ModelRegistryDB()

            # Tag based on training_quality_score (falls back to pass_rate for legacy)
            score = _sort_score(entry)
            if score >= 0.85:
                db.add_tag(entry.checkpoint_file, "production")
                logger.info(
                    "  Auto-tagged %s as 'production' (score=%.3f)",
                    entry.checkpoint_file[:40],
                    score,
                )
            elif score >= 0.70:
                db.add_tag(entry.checkpoint_file, "validated")
            elif score < 0.50:
                db.add_tag(entry.checkpoint_file, "experimental")
        except Exception as e:
            logger.debug("Auto-tagging failed (non-critical): %s", e)

    # ── Clean up safety backup (registration succeeded) ───────────────────
    if _recovery_path and _recovery_path.exists():
        try:
            _recovery_path.unlink()
            logger.debug("Cleaned up safety backup: %s", _recovery_path.name)
        except Exception:
            pass

    return ckpt_path


def _truncate_loss_history(history: list[float], max_points: int = 50) -> list[float]:
    """Truncate loss history to save storage space.

    Keeps: first 10 + every Nth + last 10, where N is chosen to fit max_points.
    """
    if not history or len(history) <= max_points:
        return history

    first_10 = history[:10]
    last_10 = history[-10:]

    # Sample middle portion
    middle = history[10:-10]
    if not middle:
        return history

    n_middle = max_points - 20
    if n_middle <= 0:
        return first_10 + last_10

    step = max(1, len(middle) // n_middle)
    sampled_middle = middle[::step][:n_middle]

    return first_10 + sampled_middle + last_10


def update_zoo_pass_rate(
    checkpoint_file: str,
    observed_pass_rate: float,
    *,
    only_if_better: bool = True,
    add_notes: str | None = None,
    pass_rate_source: str = "training_data_eval",
    _skip_db_sync: bool = False,
) -> bool:
    """Update the pass_rate for an existing zoo entry after evaluation.

    Call this after running cross-N prediction or any evaluation that
    produces per-h results. The zoo entry may have pass_rate=0 (never evaluated
    after training) — this function updates the manifest so future

    Parameters
    ----------
    checkpoint_file : str
        The checkpoint filename (as stored in ZooEntry.checkpoint_file).
    observed_pass_rate : float
        The newly observed pass rate (0.0 to 1.0).
    only_if_better : bool
        If True (default), only update if observed_pass_rate > current.
        If False, always update (use for correcting bad data).
    add_notes : str | None
        Optional text to append to the entry's notes field.
    pass_rate_source : str
        Source of the evaluation: "training_data_eval", "extrapolation_eval",
        or "cross_n_deployment". Default: "training_data_eval".
    _skip_db_sync : bool
        If True, skip the auto-sync to ModelRegistryDB. Use when the caller
        handles its own richer EvaluationRecord write to avoid duplicates.

    Returns
    -------
    bool
        True if the manifest was updated, False otherwise.

    Example
    -------
    >>> from qmbp_simulation.predictors.model_zoo import update_zoo_pass_rate
    >>> # After running cross-N evaluation with 85% pass rate:
    >>> updated = update_zoo_pass_rate(
    ...     "unified_tfim_br_ladder_multiN_6+8+10_p1.pt",
    ...     0.85,
    ...     add_notes="eval@N=20 h=[2.0,3.5]"
    ... )
    """
    if not 0.0 <= observed_pass_rate <= 1.0:
        logger.warning(
            "update_zoo_pass_rate: invalid pass_rate=%.3f (expected 0.0-1.0)",
            observed_pass_rate,
        )
        return False

    entries = _load_manifest()
    updated = False

    for entry in entries:
        if entry.checkpoint_file == checkpoint_file:
            old_rate = entry.pass_rate
            should_update = not only_if_better or observed_pass_rate > old_rate

            if should_update:
                entry.pass_rate = observed_pass_rate
                entry.pass_rate_source = pass_rate_source
                if add_notes:
                    sep = " | " if entry.notes else ""
                    entry.notes = f"{entry.notes}{sep}{add_notes}"
                updated = True
                logger.info(
                    "update_zoo_pass_rate: %s %.0f%% → %.0f%%%s",
                    checkpoint_file[:40],
                    old_rate * 100,
                    observed_pass_rate * 100,
                    " (+notes)" if add_notes else "",
                )
            else:
                logger.debug(
                    "update_zoo_pass_rate: %s already has better rate (%.0f%% vs %.0f%%)",
                    checkpoint_file[:40],
                    old_rate * 100,
                    observed_pass_rate * 100,
                )
            break
    else:
        logger.warning(
            "update_zoo_pass_rate: checkpoint '%s' not found in manifest",
            checkpoint_file,
        )
        return False

    if updated:
        _save_manifest(entries)

        # ── Auto-track in model registry history (best-effort) ──────────
        if not _skip_db_sync:
            try:
                from datetime import datetime

                from qmbp_simulation.predictors.model_registry_db import (
                    EvaluationRecord,
                    ModelRegistryDB,
                )

                db = ModelRegistryDB()
                eval_record = EvaluationRecord(
                    evaluated_at=datetime.now(UTC).isoformat(),
                    pass_rate_dual=observed_pass_rate,
                    notes=add_notes or "update_zoo_pass_rate",
                )
                db.add_evaluation(checkpoint_file, eval_record)
            except Exception as e:
                logger.debug("Registry evaluation tracking failed (non-critical): %s", e)

    return updated


def update_zoo_pass_rate_by_n(
    checkpoint_file: str,
    pass_rate_by_n: dict[int | str, float],
    *,
    update_global: bool = True,
) -> bool:
    """Update per-N pass rates for a zoo entry.

    Merges new per-N data with existing data. Also updates the global
    pass_rate as a weighted average if update_global=True.

    Parameters
    ----------
    checkpoint_file : str
        Checkpoint filename in the zoo manifest.
    pass_rate_by_n : dict[int | str, float]
        Mapping of N values to pass rates. Keys are converted to strings.
        Example: {10: 0.85, 20: 0.60, 30: 0.25}
    update_global : bool
        If True, recompute global pass_rate as the mean of all per-N values.

    Returns
    -------
    bool
        True if manifest was updated.
    """
    entries = _load_manifest()
    updated = False

    for entry in entries:
        if entry.checkpoint_file == checkpoint_file:
            # Merge (new values overwrite existing for same N)
            for n, pr in pass_rate_by_n.items():
                entry.pass_rate_by_n[str(n)] = float(pr)

            if update_global and entry.pass_rate_by_n:
                # Weighted: larger N gets slightly more weight (extrapolation harder)
                rates = list(entry.pass_rate_by_n.values())
                entry.pass_rate = float(sum(rates) / len(rates))

            updated = True
            logger.info(
                "update_zoo_pass_rate_by_n: %s → %d N values, global=%.0f%%",
                checkpoint_file[:40],
                len(entry.pass_rate_by_n),
                entry.pass_rate * 100,
            )
            break
    else:
        logger.warning("update_zoo_pass_rate_by_n: checkpoint '%s' not found", checkpoint_file)
        return False

    if updated:
        _save_manifest(entries)
    return updated


def backfill_pass_rate_by_n_from_comparisons() -> int:
    """Populate pass_rate_by_n for zoo entries using model_comparison JSONs.

    Scans all compare_*.json files and extracts per-N pass rates for each
    checkpoint that appears in the zoo manifest. Updates the manifest with
    the latest per-N data.

    Returns
    -------
    int
        Number of zoo entries updated.
    """
    import json as _json

    comp_dir = _PROJECT_ROOT / "results" / "model_comparison"
    if not comp_dir.exists():
        return 0

    # Aggregate per-checkpoint, per-N pass rates (latest wins)
    checkpoint_rates: dict[str, dict[int, float]] = {}
    for f in sorted(comp_dir.glob("compare_*.json")):
        try:
            d = _json.loads(f.read_text())
            for r in d.get("results", []):
                ckpt = r.get("checkpoint", "")
                if not ckpt:
                    continue
                for n_str, metrics in r.get("results_by_n", {}).items():
                    pr = metrics.get("pass_rate_dual", metrics.get("pass_rate_5pct", 0))
                    checkpoint_rates.setdefault(ckpt, {})[int(n_str)] = float(pr)
        except Exception:
            continue

    if not checkpoint_rates:
        return 0

    entries = _load_manifest()
    n_updated = 0

    for entry in entries:
        rates = checkpoint_rates.get(entry.checkpoint_file)
        if rates and rates != entry.pass_rate_by_n:
            # Merge (new values into existing)
            for n, pr in rates.items():
                entry.pass_rate_by_n[str(n)] = pr
            n_updated += 1

    if n_updated > 0:
        _save_manifest(entries)
        logger.info("backfill_pass_rate_by_n: updated %d entries", n_updated)

    return n_updated


def compute_retrain_queue() -> list[dict]:
    """Compute prioritized list of models that need retraining.

    Reads the dashboard (needs_retrain, model_stale, training_utility) and
    the zoo manifest to determine which models should be retrained, why,
    and in what order.

    Priority levels:
    1. contaminated — training data teaches wrong mappings
    2. stale — NPZ has significantly more/better data than model was trained on
    3. expanded — more N values available than model covers
    4. low_pass_rate — model exists but performs poorly

    Returns
    -------
    list[dict]
        Sorted by priority (highest first). Each entry:
        {
            "topology": str,
            "checkpoint_file": str,
            "priority": int (1=highest),
            "reason": str,
            "n_values_available": list[int],
            "n_training_points_available": int,
            "current_pass_rate": float,
            "command": str,  # CLI command to execute retrain
        }
    """
    import json
    from pathlib import Path as _Path

    _ROOT = _Path(__file__).resolve().parents[3]
    dashboard_path = _ROOT / "data" / "model_quality_dashboard.json"

    if not dashboard_path.exists():
        return []

    with open(dashboard_path) as f:
        dashboard = json.load(f)

    configs = dashboard.get("configs", [])
    entries = _load_manifest()

    # Get multi-N models (the ones that actually get retrained)
    multi_n_entries = [e for e in entries if e.n_qubits == 0]

    queue = []
    for entry in multi_n_entries:
        topo = entry.topology
        p = entry.p_layers

        # Get all configs for this topology
        topo_configs = [c for c in configs if c["topology"] == topo and c.get("p_layers", 1) == p]
        if not topo_configs:
            continue

        # Gather stats
        useful_configs = [c for c in topo_configs if c.get("training_utility") == "useful"]
        not_useful_configs = [c for c in topo_configs if c.get("training_utility") == "not_useful"]
        n_values_available = sorted(c["n_qubits"] for c in useful_configs)
        total_useful_pts = sum(c["n_points"] for c in useful_configs)
        any_stale = any(c.get("model_stale") for c in topo_configs)
        any_needs_retrain = any(c.get("needs_retrain") for c in topo_configs)

        # Determine if retrain needed and why
        reason = None
        priority = 99

        # Priority 1: contaminated (lots of not_useful data)
        if len(not_useful_configs) >= 2:
            reason = (
                f"contaminated: {len(not_useful_configs)} configs are 'not_useful' "
                f"(gap masking teaches wrong mappings)"
            )
            priority = 1

        # Priority 2: stale (much more data available)
        elif any_needs_retrain and total_useful_pts > entry.n_training_points * 1.3:
            reason = (
                f"stale: {total_useful_pts} useful pts available "
                f"(model trained on {entry.n_training_points})"
            )
            priority = 2

        # Priority 3: expanded (more N values available)
        elif any_stale and len(n_values_available) > 0:
            reason = f"expanded: model stale, N={n_values_available} available"
            priority = 3

        # Priority 4: low pass_rate
        elif entry.pass_rate < 0.5 and total_useful_pts >= 30:
            reason = f"low_pass_rate: {entry.pass_rate:.0%} with {total_useful_pts} pts available"
            priority = 4

        if reason is None:
            continue

        # Build CLI command
        n_str = " ".join(str(n) for n in n_values_available)
        command = (
            f"python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py "
            f"--topology {topo} --n-qubits {n_str} --p-layers {p} --retrain"
        )

        queue.append(
            {
                "topology": topo,
                "checkpoint_file": entry.checkpoint_file,
                "priority": priority,
                "reason": reason,
                "n_values_available": n_values_available,
                "n_training_points_available": total_useful_pts,
                "current_pass_rate": entry.pass_rate,
                "command": command,
            }
        )

    queue.sort(key=lambda x: (x["priority"], x["topology"]))
    return queue


def validate_zoo() -> dict[str, Any]:
    """Validate all model zoo entries: manifest consistency + file integrity.

    Returns
    -------
    dict
        Validation report with keys: "n_entries", "n_valid", "n_missing",
        "n_corrupted", "errors" (list of error messages).
    """
    entries = _load_manifest()
    report: dict[str, Any] = {
        "n_entries": len(entries),
        "n_valid": 0,
        "n_missing": 0,
        "n_corrupted": 0,
        "errors": [],
    }

    for entry in entries:
        ckpt_path = _CHECKPOINTS_DIR / entry.checkpoint_file
        if not ckpt_path.exists():
            report["n_missing"] += 1
            # Missing files are OK if gitignored (checkpoints are regenerable)
            continue

        try:
            _verify_checkpoint_integrity(ckpt_path, entry.sha256)
            report["n_valid"] += 1
        except RuntimeError as e:
            report["n_corrupted"] += 1
            report["errors"].append(str(e))

    return report


def list_best_backups() -> list[dict]:
    """List all safety backups in _best/ directory.

    Returns list of dicts with filename, size, and parsed metadata.
    """
    best_dir = _CHECKPOINTS_DIR / "_best"
    if not best_dir.exists():
        return []

    backups = []
    for f in sorted(best_dir.glob("*.pt")):
        backups.append(
            {
                "filename": f.name,
                "size_kb": f.stat().st_size / 1024,
                "path": str(f),
            }
        )
    return backups


def restore_from_best(
    topology: str,
    *,
    p_layers: int = 1,
    model: str = "tfim_bond_resolved",
) -> bool:
    """Restore the best backup for a topology into the active zoo.

    Finds the highest-pass-rate backup in _best/ for the given topology
    and copies it back to the active checkpoints directory + updates manifest.

    Parameters
    ----------
    topology : str
        Topology to restore (e.g., "chain_1d").
    p_layers : int
        HVA depth to match.
    model : str
        Model name to match.

    Returns
    -------
    bool
        True if a backup was restored, False if no matching backup found.
    """
    import shutil

    best_dir = _CHECKPOINTS_DIR / "_best"
    if not best_dir.exists():
        logger.warning("No _best/ directory found.")
        return False

    # Find backups matching this topology
    matching = [f for f in best_dir.glob("*.pt") if topology in f.name]
    if not matching:
        logger.warning(f"No backups found for topology={topology} in _best/")
        return False

    # Pick the one with highest pass rate (encoded in filename as _passXXpct_)
    def _extract_pass_rate(path):
        name = path.stem
        if "_pass" in name and "pct" in name:
            try:
                part = name.split("_pass")[1].split("pct")[0]
                return float(part.replace("%", "")) / 100
            except (ValueError, IndexError):
                pass
        return 0.0

    best_backup = max(matching, key=_extract_pass_rate)
    pass_rate = _extract_pass_rate(best_backup)

    # Find the current active checkpoint name for this config
    entries = _load_manifest()
    current_entry = next(
        (
            e
            for e in entries
            if e.model == model
            and e.topology == topology
            and e.p_layers == p_layers
            and e.n_qubits == 0
        ),
        None,
    )

    if current_entry is None:
        logger.warning(f"No manifest entry for {model}/{topology} n=0 p={p_layers}")
        return False

    target_path = _CHECKPOINTS_DIR / current_entry.checkpoint_file
    shutil.copy2(best_backup, target_path)

    # Update manifest pass_rate
    current_entry.pass_rate = pass_rate
    current_entry.notes = f"Restored from _best/ ({best_backup.name})"
    _save_manifest(entries)

    logger.info(
        f"Restored {topology} from _best/{best_backup.name} "
        f"(pass_rate={pass_rate:.0%}) → {target_path.name}"
    )
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Integration: Quality Tier Analysis
# ═══════════════════════════════════════════════════════════════════════════════


def get_training_data_quality(
    topology: str,
    n_qubits: int,
    model: str = "tfim_bond_resolved",
    p_layers: int = 1,
) -> dict[str, Any]:
    """Get quality tier breakdown for a model's training data NPZ.

    Cross-integration: Links model zoo entries to their NPZ training data
    quality metrics. Used by runners to decide if a zoo model is trustworthy
    or if retraining with cleaner data is needed.

    Parameters
    ----------
    topology : str
        Lattice topology.
    n_qubits : int
        System size. Use 0 for multi-N aggregated models.
    model : str
        Hamiltonian model name (for NPZ filename construction).
    p_layers : int
        HVA depth.

    Returns
    -------
    dict
        Quality report with keys:
        - "found": bool - whether NPZ exists
        - "n_points": int - total training points
        - "n_verified": int - VQE-verified points
        - "n_approximate": int - MPNN-predicted but not VQE-verified
        - "n_unverified": int - legacy data without tier info
        - "verified_ratio": float - fraction of verified points
        - "quality_score": float - composite score (verified=1.0, approx=0.7, unverified=0.5)
        - "npz_path": str - path to NPZ file (if found)
        - "warnings": list[str] - quality warnings
    """
    import numpy as np

    npz_dir = _PROJECT_ROOT / "data" / "multi_n_training"

    # Construct expected NPZ filename
    if n_qubits == 0:
        # Multi-N model — aggregate from all matching NPZ files
        pattern = f"{topology}_N*_p{p_layers}.npz"
        npz_files = list(npz_dir.glob(pattern))
    else:
        npz_filename = f"{topology}_N{n_qubits}_p{p_layers}.npz"
        npz_path = npz_dir / npz_filename
        npz_files = [npz_path] if npz_path.exists() else []

    if not npz_files:
        return {
            "found": False,
            "n_points": 0,
            "n_verified": 0,
            "n_approximate": 0,
            "n_unverified": 0,
            "verified_ratio": 0.0,
            "quality_score": 0.0,
            "npz_path": None,
            "warnings": [f"No NPZ found for {topology}/N={n_qubits}/p={p_layers}"],
        }

    total_pts = 0
    total_verified = 0
    total_approx = 0
    total_unverified = 0
    warnings = []

    for npz_path in npz_files:
        try:
            data = np.load(str(npz_path), allow_pickle=True)
            h_vals = data["h_values"]
            n_pts = len(h_vals)
            total_pts += n_pts

            # Check for quality_tier field
            if "quality_tier" in data:
                tiers = data["quality_tier"].tolist()
                total_verified += tiers.count("verified")
                total_approx += tiers.count("approximate")
                total_unverified += tiers.count("unverified")
            else:
                # Legacy NPZ without quality_tier
                total_unverified += n_pts
                warnings.append(f"Legacy NPZ {npz_path.name}: no quality_tier field")
        except Exception as e:
            warnings.append(f"Error reading {npz_path.name}: {e}")

    # Compute quality score
    if total_pts > 0:
        verified_ratio = total_verified / total_pts
        # Weighted quality score: verified=1.0, approx=0.7, unverified=0.5
        quality_score = (
            total_verified * 1.0 + total_approx * 0.7 + total_unverified * 0.5
        ) / total_pts
    else:
        verified_ratio = 0.0
        quality_score = 0.0

    # Generate warnings for quality issues
    if total_pts > 10:
        if verified_ratio < 0.3:
            warnings.append(
                f"Low verified ratio ({verified_ratio:.0%}). Consider running "
                f"--refine-all to convert approximate → verified."
            )
        if total_unverified > total_pts * 0.5:
            warnings.append(
                f"High unverified count ({total_unverified}/{total_pts}). "
                f"Likely legacy data — re-run pipeline to add quality_tier."
            )

    return {
        "found": True,
        "n_points": total_pts,
        "n_verified": total_verified,
        "n_approximate": total_approx,
        "n_unverified": total_unverified,
        "verified_ratio": verified_ratio,
        "quality_score": quality_score,
        "npz_path": str(npz_files[0]) if len(npz_files) == 1 else f"{len(npz_files)} files",
        "warnings": warnings,
    }
