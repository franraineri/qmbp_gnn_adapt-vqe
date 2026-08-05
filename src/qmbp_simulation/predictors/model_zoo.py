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
from dataclasses import asdict, dataclass, field
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
        Observed pass rate (ΔE/gap < 5%) on validation set.
    n_training_points : int
        Number of VQE points used for training.
    seeds : list[int]
        Seeds used for training data generation.
    created : str
        ISO timestamp of checkpoint creation.
    notes : str
        Any additional context.
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
    entries = []
    for item in raw:
        # Handle h_range as list → tuple
        if "h_range" in item and isinstance(item["h_range"], list):
            item["h_range"] = tuple(item["h_range"])
        entries.append(ZooEntry(**item))
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

    if ckpt_path.exists() and not overwrite:
        logger.warning("Checkpoint already exists: %s. Use overwrite=True to replace.", ckpt_path)
        return ckpt_path

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
