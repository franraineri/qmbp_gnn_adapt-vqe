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
        if n_qubits and self.n_qubits != n_qubits:
            return False
        if p_layers and self.p_layers != p_layers:
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
        e for e in entries
        if e.matches(model=model, topology=topology, n_qubits=n_qubits, p_layers=p_layers)
    ]
    return sorted(filtered, key=lambda e: e.pass_rate, reverse=True)


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
            model=model, topology=topology, n_qubits=n_qubits,
            p_layers=p_layers, checkpoint_file=str(ckpt_path),
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
        cross_n_candidates = list_pretrained(
            model=model, topology=topology, p_layers=p_layers
        )
        if cross_n_candidates:
            # Prefer the closest N, then highest pass_rate
            cross_n_candidates.sort(
                key=lambda e: (abs(e.n_qubits - n_qubits), -e.pass_rate)
            )
            candidates = cross_n_candidates[:1]
            logger.info(
                "No exact N=%d match in zoo. Using cross-N transfer from N=%d "
                "(GINConv + global_mean_pool supports variable-size graphs).",
                n_qubits, candidates[0].n_qubits,
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
        "Loaded pre-trained MPNN: %s/%s N=%d p=%d (pass_rate=%.0f%%)",
        best.model, best.topology, best.n_qubits, best.p_layers, best.pass_rate * 100,
    )
    return mpnn, best


def load_best_for_cross_n(
    model: str,
    topology: str,
    n_target: int,
    p_layers: int,
    *,
    checkpoint_path: str | Path | None = None,
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
            model=model, topology=topology, n_qubits=0,
            p_layers=p_layers, checkpoint_file=str(ckpt_path),
            notes="User-specified checkpoint for cross-N",
        )
        return mpnn, meta

    # Load ALL entries for this model/topology/p (ignoring n_qubits filter)
    entries = _load_manifest()
    candidates = [
        e for e in entries
        if e.model == model and e.topology == topology and e.p_layers == p_layers
    ]

    if not candidates:
        available = {(e.model, e.topology, e.p_layers) for e in entries}
        raise FileNotFoundError(
            f"No model in zoo for ({model}, {topology}, p={p_layers}).\n"
            f"Available (model, topology, p): {sorted(available)}\n"
            f"Train one via: --multi-n-train or --train-n <N>"
        )

    # ── Priority 1: Multi-N models (n_qubits=0) ──────────────────────────
    multi_n = [e for e in candidates if e.n_qubits == 0]
    if multi_n:
        # Among multi-N models, prefer most training points
        best = max(multi_n, key=lambda e: e.n_training_points)
        ckpt_path = _CHECKPOINTS_DIR / best.checkpoint_file

        # Quality gate: multi-N model with pass_rate < 40% is likely trained
        # on contaminated data (gap-masked or variationally invalid points).
        # Fall through to single-N models which may be better.
        if best.pass_rate > 0 and best.pass_rate < 0.40:
            logger.warning(
                "load_best_for_cross_n: Multi-N model %s has low pass_rate=%.0f%%. "
                "Likely trained on contaminated data. Checking single-N alternatives.",
                best.checkpoint_file, best.pass_rate * 100,
            )
            # Check if a single-N model has better pass_rate
            single_n = [e for e in candidates if e.n_qubits > 0]
            better_single = [e for e in single_n if e.pass_rate > best.pass_rate + 0.1]
            if better_single:
                # Fall through to single-N selection below
                logger.info(
                    "  Found %d single-N models with better pass_rate. Using those.",
                    len(better_single),
                )
                multi_n = []  # Skip multi-N, fall through to Priority 2

        if multi_n and ckpt_path.exists():
            _verify_checkpoint_integrity(ckpt_path, best.sha256)
            mpnn = _smart_load_checkpoint(str(ckpt_path))
            logger.info(
                "load_best_for_cross_n: Multi-N model → %s "
                "(%d training pts, N_target=%d)",
                best.checkpoint_file, best.n_training_points, n_target,
            )
            return mpnn, best
        elif multi_n:
            logger.warning(
                "Multi-N checkpoint missing on disk: %s. Falling back to single-N.",
                best.checkpoint_file,
            )

    # ── Priority 2: Best-scored single-N model ────────────────────────────
    single_n = [e for e in candidates if e.n_qubits > 0]
    if not single_n:
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
            f"Best single-N checkpoint missing: {ckpt_path}\n"
            f"Re-run training to regenerate."
        )

    _verify_checkpoint_integrity(ckpt_path, best.sha256)
    mpnn = _smart_load_checkpoint(str(ckpt_path))
    pass_str = f"pass={best.pass_rate:.0%}" if best.pass_rate > 0 else "unevaluated"
    logger.info(
        "load_best_for_cross_n: Single-N model N=%d → %s "
        "(score=%.1f, %d pts, %s, N_target=%d)",
        best.n_qubits, best.checkpoint_file,
        _score(best), best.n_training_points,
        pass_str, n_target,
    )
    return mpnn, best


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
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%d%m%y")


def get_runner_tag(runner_id: str) -> str:
    """Map a runner_id to its 2-letter tag.

    Falls back to "XX" for unknown runners. Matches by substring
    so "accelerated_cross_n_v1" → "AC".

    Parameters
    ----------
    runner_id : str
        The runner_id attribute (e.g., "accelerated_cross_n_v1").

    Returns
    -------
    str
        2-letter tag (e.g., "AC").
    """
    runner_lower = runner_id.lower()
    for key, tag in RUNNER_TAGS.items():
        if key in runner_lower:
            return tag
    return "XX"


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

    # ── Auto-fill date_tag if not provided ───────────────────────────────
    if not entry.date_tag:
        from datetime import datetime, timezone
        entry.date_tag = datetime.now(timezone.utc).strftime("%d%m%y")

    if ckpt_path.exists() and not overwrite:
        logger.warning("Checkpoint already exists: %s. Use overwrite=True to replace.", ckpt_path)
        return ckpt_path

    # ── Validate model state before saving ───────────────────────────────
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

    # Update manifest
    entries = _load_manifest()
    # Remove existing entry for same config if overwriting
    if overwrite:
        entries = [
            e for e in entries
            if not (e.model == entry.model and e.topology == entry.topology
                    and e.n_qubits == entry.n_qubits and e.p_layers == entry.p_layers)
        ]
    entries.append(entry)
    _save_manifest(entries)

    logger.info(
        "Registered checkpoint: %s/%s N=%d p=%d → %s",
        entry.model, entry.topology, entry.n_qubits, entry.p_layers, ckpt_path,
    )
    return ckpt_path


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
            should_update = (
                not only_if_better or
                observed_pass_rate > old_rate
            )

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
                    f" (+notes)" if add_notes else "",
                )
            else:
                logger.debug(
                    "update_zoo_pass_rate: %s already has better rate (%.0f%% vs %.0f%%)",
                    checkpoint_file[:40], old_rate * 100, observed_pass_rate * 100,
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
    return updated


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
        model=model, topology=topology, n_target=n_target,
        p_layers=p_layers, checkpoint_path=checkpoint_path,
    )

    # Get quality info for the training data
    quality = get_training_data_quality(
        topology=topology, n_qubits=entry.n_qubits,
        model=model, p_layers=p_layers,
    )

    # Quality gate warning
    if quality["found"] and quality["quality_score"] < min_quality_score:
        logger.warning(
            "load_best_for_cross_n_quality_aware: Model %s trained on low-quality data "
            "(quality_score=%.2f < %.2f). Consider retraining with cleaner data.\n"
            "  verified=%d, approximate=%d, unverified=%d",
            entry.checkpoint_file,
            quality["quality_score"], min_quality_score,
            quality["n_verified"], quality["n_approximate"], quality["n_unverified"],
        )
        for w in quality.get("warnings", []):
            logger.warning("    ⚠️ %s", w)

    return mpnn, entry, quality
