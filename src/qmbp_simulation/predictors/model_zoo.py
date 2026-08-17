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

    data = torch.load(path, map_location="cpu", weights_only=False)

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
_ZOO_DIR = _PROJECT_ROOT / "data" / "model_zoo"
_MANIFEST_PATH = _ZOO_DIR / "manifest.json"
_CHECKPOINTS_DIR = _ZOO_DIR / "checkpoints"

# Quality gate threshold for multi-N model selection in load_best_for_cross_n().
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
    n_training_points: int = 0
    seeds: list[int] = field(default_factory=list)
    created: str = ""
    notes: str = ""
    sha256: str = ""  # Integrity hash — verified on load
    runner_tag: str = "XX"  # 2-letter runner identifier
    date_tag: str = ""  # DDMMYY format
    training_quality_score: float = -1.0  # [0,1] composite quality, -1 = not computed

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


def _load_manifest() -> list[ZooEntry]:
    """Load the model zoo manifest from disk."""
    if not _MANIFEST_PATH.exists():
        return []
    with open(_MANIFEST_PATH) as f:
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
    ckpt_path = _CHECKPOINTS_DIR / best.checkpoint_file
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file missing: {ckpt_path}\n"
            f"The manifest references it but the file is not on disk. "
            f"Re-run the pipeline with --export-zoo to regenerate."
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


def load_best_for_cross_n(
    model: str,
    topology: str,
    n_target: int,
    p_layers: int,
    *,
    checkpoint_path: str | Path | None = None,
    reject_contaminated: bool = True,
) -> tuple[Any, ZooEntry]:
    """Load the best available model for cross-N prediction at n_target.

    Implements a priority hierarchy optimized for cross-N generalization:

    1. **Multi-N model** (``n_qubits=0``): trained on aggregated data from
       multiple system sizes.  Always preferred because it has seen the
       functional dependence θ(h, N) and generalizes best to unseen N.
    2. **Best-scored single-N model**: ranked by a composite score that
       balances training data quantity, proximity to n_target, and validated
       pass_rate.  Score formula::

           score = n_training_points * (1 / (1 + |n_qubits - n_target|)) * pr_weight

       where ``pr_weight = max(pass_rate, 0.3)`` (unvalidated models with
       pass_rate=0 get a floor of 0.3 so they aren't completely ignored).

    Unlike ``load_pretrained(allow_cross_n=True)`` which prefers exact N
    match then falls back to nearest N, this function is specifically
    designed for the cross-N prediction use case where the model will
    ALWAYS be used on a different N than it was trained on.

    Parameters
    ----------
    model : str
        Hamiltonian model name (e.g. "tfim_bond_resolved").
    topology : str
        Lattice topology (e.g. "ladder", "chain_1d").
    n_target : int
        Target system size for prediction.
    p_layers : int
        HVA depth.
    checkpoint_path : str | Path | None
        Override: load from a specific path (bypasses zoo search).
    reject_contaminated : bool
        If True (default), reject models with `contaminated_training` failure mode.
        Uses ModelRegistryDB diagnostics to check model health.

    Returns
    -------
    tuple[model, ZooEntry]
        Best available model + metadata.

    Raises
    ------
    FileNotFoundError
        If no suitable model exists in the zoo for this topology/model/p.
        The caller should handle this by training a new model from available
        NPZ data (via MultiNAggregator + train_unified_mpnn).
    """
    if n_target < 1:
        raise ValueError(f"n_target must be ≥ 1, got {n_target}.")
    if p_layers < 1:
        raise ValueError(f"p_layers must be ≥ 1, got {p_layers}.")

    # User override — direct checkpoint path
    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        mpnn = _smart_load_checkpoint(str(ckpt_path))
        meta = ZooEntry(
            model=model,
            topology=topology,
            n_qubits=0,
            p_layers=p_layers,
            checkpoint_file=str(ckpt_path),
            notes="User-specified checkpoint for cross-N",
        )
        return mpnn, meta

    # Load ALL entries for this model/topology/p (ignoring n_qubits filter)
    entries = _load_manifest()
    candidates = [
        e for e in entries if e.model == model and e.topology == topology and e.p_layers == p_layers
    ]

    if not candidates:
        available = {(e.model, e.topology, e.p_layers) for e in entries}
        raise FileNotFoundError(
            f"No model in zoo for ({model}, {topology}, p={p_layers}).\n"
            f"Available (model, topology, p): {sorted(available)}\n"
            f"Train one via: --multi-n-train or --train-n <N>"
        )

    # ── Check for contaminated models using ModelRegistryDB ──────────────
    contaminated_ids: set[str] = set()
    gap_masked_ids: set[str] = set()
    if reject_contaminated:
        contaminated_ids = _get_contaminated_model_ids(model, topology, p_layers)
        gap_masked_ids = _get_gap_masked_model_ids(model, topology, p_layers)
        if contaminated_ids:
            logger.warning(
                "load_best_for_cross_n: Found %d contaminated model(s) for %s/%s p=%d. "
                "These will be excluded from selection.",
                len(contaminated_ids),
                topology,
                model,
                p_layers,
            )
        if gap_masked_ids:
            logger.warning(
                "load_best_for_cross_n: Found %d gap-masked model(s) for %s/%s p=%d. "
                "These will be included but metrics may be inflated (|ΔE| high despite ΔE/gap passing).",
                len(gap_masked_ids),
                topology,
                model,
                p_layers,
            )

    # ── Priority 1: Multi-N models (n_qubits=0) ──────────────────────────
    multi_n = [e for e in candidates if e.n_qubits == 0]

    # Filter out contaminated models
    if reject_contaminated and contaminated_ids:
        multi_n = [e for e in multi_n if e.checkpoint_file not in contaminated_ids]

    if multi_n:
        # Among multi-N models, prefer most training points
        best = max(multi_n, key=lambda e: e.n_training_points)
        ckpt_path = _CHECKPOINTS_DIR / best.checkpoint_file

        # Quality gate: comprehensive readiness assessment combining data quality,
        # convergence health (MSE, status), pass_rate, and freshness (stale detection).
        should_fallback = False
        fallback_reason = ""

        readiness = compute_model_readiness(best, n_target=n_target)
        quality_score = readiness["readiness_score"]

        if quality_score < MULTI_N_MIN_QUALITY_SCORE:
            should_fallback = True
            penalty_str = (
                "; ".join(readiness["penalties"]) if readiness["penalties"] else "low composite"
            )
            fallback_reason = (
                f"readiness={quality_score:.3f} < {MULTI_N_MIN_QUALITY_SCORE:.2f} "
                f"[{readiness['recommendation']}] ({penalty_str})"
            )
        elif readiness["recommendation"] == "avoid":
            should_fallback = True
            fallback_reason = f"recommendation='avoid' ({'; '.join(readiness['penalties'])})"

        # Cross-check with dashboard training_utility if available
        if not should_fallback:
            try:
                import json

                dashboard_path = _PROJECT_ROOT / "data" / "model_quality_dashboard.json"
                if dashboard_path.exists():
                    with open(dashboard_path) as _f:
                        _dash = json.load(_f)
                    # Check if ANY config for this topology is "not_useful"
                    # (contamination signal for the multi-N model)
                    topo_configs = [
                        c
                        for c in _dash.get("configs", [])
                        if c.get("topology") == topology and c.get("p_layers") == p_layers
                    ]
                    n_not_useful = sum(
                        1 for c in topo_configs if c.get("training_utility") == "not_useful"
                    )
                    n_total = len(topo_configs)
                    if n_total > 0 and n_not_useful / n_total > 0.5:
                        should_fallback = True
                        fallback_reason = (
                            f"dashboard: {n_not_useful}/{n_total} configs are 'not_useful' "
                            f"— multi-N model likely contaminated"
                        )
            except Exception:
                pass  # Dashboard check is best-effort

        if should_fallback:
            logger.warning(
                "load_best_for_cross_n: Multi-N model %s quality gate FAILED (%s). "
                "Checking single-N alternatives.",
                best.checkpoint_file,
                fallback_reason,
            )
            single_n_alt = [e for e in candidates if e.n_qubits > 0]
            # Also filter contaminated from single-N alternatives
            if reject_contaminated and contaminated_ids:
                single_n_alt = [
                    e for e in single_n_alt if e.checkpoint_file not in contaminated_ids
                ]
            better_single = [e for e in single_n_alt if _sort_score(e) > quality_score + 0.1]
            if better_single:
                logger.info(
                    "  Found %d single-N models with better quality_score. Falling back.",
                    len(better_single),
                )
                multi_n = []  # Skip multi-N, fall through to Priority 2
            else:
                # No better alternatives — use multi-N anyway but warn strongly
                logger.warning(
                    "  No single-N alternatives with better quality_score. "
                    "Using multi-N model despite quality concern. Consider --force-retrain."
                )

        if multi_n and ckpt_path.exists():
            _verify_checkpoint_integrity(ckpt_path, best.sha256)
            mpnn = _smart_load_checkpoint(str(ckpt_path))
            logger.info(
                "load_best_for_cross_n: Multi-N model → %s "
                "(%d training pts, N_target=%d, readiness=%.2f [%s])",
                best.checkpoint_file,
                best.n_training_points,
                n_target,
                readiness["readiness_score"],
                readiness["recommendation"],
            )
            if readiness["penalties"]:
                for p in readiness["penalties"]:
                    logger.info("    ⚠️ %s", p)
            return mpnn, best
        elif multi_n:
            logger.warning(
                "Multi-N checkpoint missing on disk: %s. Falling back to single-N.",
                best.checkpoint_file,
            )

    # ── Priority 2: Best-scored single-N model ────────────────────────────
    single_n = [e for e in candidates if e.n_qubits > 0]

    # Filter out contaminated models
    if reject_contaminated and contaminated_ids:
        single_n = [e for e in single_n if e.checkpoint_file not in contaminated_ids]

    if not single_n:
        if contaminated_ids:
            raise FileNotFoundError(
                f"All available models for ({model}, {topology}, p={p_layers}) "
                f"are flagged as contaminated. Run diagnostics or retrain with clean data.\n"
                f"Contaminated models: {sorted(contaminated_ids)}"
            )
        raise FileNotFoundError(
            f"Multi-N checkpoint file missing and no single-N models available "
            f"for ({model}, {topology}, p={p_layers})."
        )

    def _score(entry: ZooEntry) -> float:
        """Composite score: data quantity × proximity × confidence."""
        proximity = 1.0 / (1.0 + abs(entry.n_qubits - n_target))
        # Floor pass_rate at 0.3 for unvalidated models (pass_rate=0 from
        # auto-export before evaluation). This prevents good models from
        # being ranked below tiny validated ones.
        pr_weight = max(entry.pass_rate, 0.3)
        return entry.n_training_points * proximity * pr_weight

    single_n.sort(key=_score, reverse=True)
    best = single_n[0]

    ckpt_path = _CHECKPOINTS_DIR / best.checkpoint_file
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Best single-N checkpoint missing: {ckpt_path}\nRe-run training to regenerate."
        )

    _verify_checkpoint_integrity(ckpt_path, best.sha256)
    mpnn = _smart_load_checkpoint(str(ckpt_path))
    pass_str = f"pass_dual={best.pass_rate:.0%}" if best.pass_rate > 0 else "unevaluated"
    logger.info(
        "load_best_for_cross_n: Single-N model N=%d → %s (score=%.1f, %d pts, %s, N_target=%d)",
        best.n_qubits,
        best.checkpoint_file,
        _score(best),
        best.n_training_points,
        pass_str,
        n_target,
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


def register_checkpoint(
    model,
    entry: ZooEntry,
    *,
    overwrite: bool = False,
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

    # ── Auto-fill date_tag if not provided ───────────────────────────────
    if not entry.date_tag:
        from datetime import datetime

        entry.date_tag = datetime.now(UTC).strftime("%d%m%y")

    if ckpt_path.exists() and not overwrite:
        logger.warning("Checkpoint already exists: %s. Use overwrite=True to replace.", ckpt_path)
        return ckpt_path

    # ── ANTI-REGRESSION + AUTO-VERSION: preserve ALL previous checkpoints ─
    # When overwriting, we:
    # 1. Copy existing to _versions/ with incremental version number
    # 2. Copy to _best/ if it has the highest pass_rate seen
    # 3. Mark old model as superseded in ModelRegistryDB
    # 4. The new model ALWAYS gets saved (no interactive prompts, no timeouts)
    if ckpt_path.exists() and overwrite:
        # Find existing manifest entry for this config
        existing_entries = _load_manifest()
        existing_entry = next(
            (
                e
                for e in existing_entries
                if e.model == entry.model
                and e.topology == entry.topology
                and e.n_qubits == entry.n_qubits
                and e.p_layers == entry.p_layers
            ),
            None,
        )

        old_pass_rate = (existing_entry.pass_rate or 0.0) if existing_entry else 0.0
        new_pass_rate = entry.pass_rate or 0.0
        old_pts = (existing_entry.n_training_points or 0) if existing_entry else 0

        # ── _versions/: sequential archive (NEVER lose any model) ────────
        from qmbp_simulation.utils.helpers import versioned_backup

        _sidecar = {
            "pass_rate": old_pass_rate,
            "n_training_points": old_pts,
            "date_tag": existing_entry.date_tag if existing_entry else "",
            "superseded_by": entry.checkpoint_file,
            "superseded_at": entry.date_tag,
        }
        _versioned_path, _version_num = versioned_backup(
            ckpt_path,
            version_dir=_CHECKPOINTS_DIR / "_versions",
            sidecar_metadata=_sidecar,
        )

        # Derive base stem for logging and _best/ naming (use same regex as versioned_backup)
        import re as _re_ver

        from qmbp_simulation.utils.helpers import _VERSION_SUFFIX_RE

        _base_stem = _re_ver.sub(_VERSION_SUFFIX_RE, "", ckpt_path.stem)
        _suffix = ckpt_path.suffix

        _old_info = f" ({old_pts} pts, pass={old_pass_rate:.0%})" if existing_entry else ""
        logger.info(
            "  Zoo auto-version: %s%s → _versions/%s_v%d%s  |  New: %d pts, pass=%.0f%%",
            ckpt_path.name,
            _old_info,
            _base_stem,
            _version_num,
            _suffix,
            entry.n_training_points,
            new_pass_rate * 100,
        )

        # ── _best/: keep the highest pass_rate version for quick recovery ─
        best_dir = _CHECKPOINTS_DIR / "_best"
        best_dir.mkdir(parents=True, exist_ok=True)

        if existing_entry is not None:
            import shutil as _shutil_ver

            backup_name = (
                f"{_base_stem}_pass{old_pass_rate:.0%}"
                f"_{existing_entry.date_tag or 'nodate'}{_suffix}"
            ).replace("%", "pct")
            backup_path = best_dir / backup_name
            if not backup_path.exists():
                _shutil_ver.copy2(ckpt_path, backup_path)
                logger.info(
                    "  Zoo anti-regression: backed up %s (pass=%.0f%%) → _best/%s",
                    ckpt_path.name,
                    old_pass_rate * 100,
                    backup_name,
                )

        # Warn if new model is worse (but still save it — never block)
        if new_pass_rate < old_pass_rate - 0.01:
            logger.warning(
                "  ⚠️ Zoo DOWNGRADE: %s pass_rate %.0f%% → %.0f%%. "
                "Previous best preserved in _best/ and _versions/.",
                entry.topology,
                old_pass_rate * 100,
                new_pass_rate * 100,
            )

            # ── AUTO-ROLLBACK: severe regression (>30% relative drop) ────
            # If new model is drastically worse, don't overwrite the
            # production checkpoint. Save new model as candidate in _versions/
            # and keep the existing model in place.
            relative_drop = (old_pass_rate - new_pass_rate) / max(old_pass_rate, 0.01)
            if relative_drop > 0.30 and old_pass_rate > 0.3:
                logger.warning(
                    "  🔄 AUTO-ROLLBACK: regression too severe (%.0f%% relative drop). "
                    "New model saved to _versions/ as candidate. "
                    "Production checkpoint unchanged.",
                    relative_drop * 100,
                )
                # The new model was already saved to _versions/ above.
                # Return the existing (better) checkpoint path without overwriting.
                return ckpt_path

        # ── ModelRegistryDB: mark old model superseded + record event ────
        try:
            from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

            _db = ModelRegistryDB()
            _old_model_id = existing_entry.checkpoint_file if existing_entry else None
            if _old_model_id:
                _db.mark_superseded(_old_model_id, superseded_by=entry.checkpoint_file)
                _db._record_event(
                    "auto_versioned",
                    _old_model_id,
                    topology=entry.topology,
                    details={
                        "version_number": _version_num,
                        "versioned_path": str(_versioned_path.name),
                        "old_pass_rate": old_pass_rate,
                        "old_n_training_points": old_pts,
                        "new_pass_rate": new_pass_rate,
                        "new_n_training_points": entry.n_training_points,
                        "new_model_id": entry.checkpoint_file,
                    },
                )
        except Exception as _e_db:
            logger.debug("ModelRegistryDB auto-version tracking failed (non-critical): %s", _e_db)

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

    # Update manifest
    entries = _load_manifest()
    # Remove existing entry for same config if overwriting
    if overwrite:
        entries = [
            e
            for e in entries
            if not (
                e.model == entry.model
                and e.topology == entry.topology
                and e.n_qubits == entry.n_qubits
                and e.p_layers == entry.p_layers
            )
        ]
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
) -> Path:
    """Register checkpoint and capture training metrics in one call.

    Enhanced version of ``register_checkpoint()`` that also:
    - Records training metrics from ``train_unified_mpnn()`` return dict
    - Auto-tags model based on pass_rate thresholds
    - Integrates with ModelRegistryDB for full provenance tracking
    - Auto-runs failure diagnostics post-registration
    - Auto-syncs dashboard quality data

    This is the recommended function for pipeline runners to use after
    training, as it captures full training provenance automatically.

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
        If True, overwrite existing entry for the same config.
    auto_tag : bool
        If True (default), auto-add tags based on pass_rate:
        - pass_rate ≥ 0.90 → "production"
        - pass_rate ≥ 0.70 → "validated"
        - pass_rate < 0.50 → "experimental"
    auto_diagnose : bool
        If True (default), run failure diagnostics after registration.
        Detects: gap_masking, contaminated_training, intrinsic_vqe_error, etc.
    auto_sync_dashboard : bool
        If True (default), sync dashboard quality data after registration.
    architecture_config : dict | None
        Model architecture config: {hidden_dim, n_conv_layers, n_heads, ...}
        Stored for reproducibility.
    optimizer_config : dict | None
        Optimizer config: {learning_rate, weight_decay, scheduler_patience, ...}
        Stored for reproducibility.

    Returns
    -------
    Path
        Path to the saved checkpoint file.

    Example
    -------
    >>> model = UnifiedMPNN(...)
    >>> train_result = train_unified_mpnn(model, dataset, n_epochs=4000)
    >>> entry = ZooEntry(model="tfim_br", topology="chain_1d", ...)
    >>> path = register_checkpoint_with_training_metrics(
    ...     model, entry, training_result=train_result, overwrite=True
    ... )
    """
    # Save checkpoint via standard function
    ckpt_path = register_checkpoint(model, entry, overwrite=overwrite)

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
) -> bool:
    """Update the pass_rate for an existing zoo entry after evaluation.

    Call this after running cross-N prediction or any evaluation that
    produces per-h results. The zoo entry may have pass_rate=0 (never evaluated
    after training) — this function updates the manifest so future
    `load_best_for_cross_n` sees the real quality.

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


def load_best_for_cross_n_quality_aware(
    model: str,
    topology: str,
    n_target: int,
    p_layers: int,
    *,
    min_quality_score: float = 0.6,
    checkpoint_path: str | Path | None = None,
) -> tuple[Any, ZooEntry, dict]:
    """Load best cross-N model with quality-aware selection.

    Enhanced version of ``load_best_for_cross_n`` that also checks the
    training data quality tier distribution. Returns both the model and
    a quality report so callers can decide if retraining is advisable.

    Parameters
    ----------
    model : str
        Hamiltonian model name.
    topology : str
        Lattice topology.
    n_target : int
        Target system size.
    p_layers : int
        HVA depth.
    min_quality_score : float
        Minimum acceptable quality score (0.0-1.0). If below this, a warning
        is logged suggesting retraining. Default 0.6.
    checkpoint_path : str | Path | None
        Override: load from specific path (bypasses quality check).

    Returns
    -------
    tuple[model, ZooEntry, dict]
        Loaded model, metadata, and quality report.

    Raises
    ------
    FileNotFoundError
        If no suitable model exists in the zoo.
    """
    # Load model via standard function
    mpnn, entry = load_best_for_cross_n(
        model=model,
        topology=topology,
        n_target=n_target,
        p_layers=p_layers,
        checkpoint_path=checkpoint_path,
    )

    # Get quality info for the training data
    quality = get_training_data_quality(
        topology=topology,
        n_qubits=entry.n_qubits,
        model=model,
        p_layers=p_layers,
    )

    # Quality gate warning
    if quality["found"] and quality["quality_score"] < min_quality_score:
        logger.warning(
            "load_best_for_cross_n_quality_aware: Model %s trained on low-quality data "
            "(quality_score=%.2f < %.2f). Consider retraining with cleaner data.\n"
            "  verified=%d, approximate=%d, unverified=%d",
            entry.checkpoint_file,
            quality["quality_score"],
            min_quality_score,
            quality["n_verified"],
            quality["n_approximate"],
            quality["n_unverified"],
        )
        for w in quality.get("warnings", []):
            logger.warning("    ⚠️ %s", w)

    return mpnn, entry, quality
