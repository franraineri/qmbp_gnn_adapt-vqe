"""
Shared utility functions — seeding, JSON serialization, and timing.

This module is the leaf node of the dependency graph: it has NO imports
from other qmbp_simulation submodules.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────


def set_global_seed(seed: int) -> None:
    """Seed NumPy, PyTorch, and Python random for reproducibility.

    Parameters
    ----------
    seed : int
        Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# JSON Serialization
# ─────────────────────────────────────────────────────────────────────────────


def json_serialize(obj: Any) -> Any:
    """Recursively convert Python/numpy objects to JSON-serializable types.

    Handles:
    - numpy bool → bool
    - numpy arrays → list
    - numpy integer/floating scalars → int/float
    - dataclasses → dict (via asdict)
    - datetime → ISO format string
    - Path objects → str
    - NaN/Inf floats → None

    Parameters
    ----------
    obj : Any
        Object to serialize.

    Returns
    -------
    Any
        JSON-serializable equivalent.
    """
    if obj is None:
        return None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if is_dataclass(obj) and not isinstance(obj, type):
        return json_serialize(asdict(obj))
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_serialize(item) for item in obj]
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, int | str | bool):
        return obj
    # Fallback: try numeric conversion, else str
    try:
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    except (TypeError, ValueError):
        return str(obj)


def json_dump(obj: Any, path: Path, indent: int = 2) -> None:
    """Serialize obj to JSON and write to path.


    Uses `json_serialize` as the default handler for non-standard types.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=json_serialize)


# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TimerResult:
    """Result from the timer context manager.

    Attributes
    ----------
    elapsed_s : float
        Wall-clock elapsed time in seconds.
    label : str
        Descriptive label for the timed block.
    """

    elapsed_s: float = 0.0
    label: str = ""


@contextmanager
def timer(label: str = "") -> Generator[TimerResult, None, None]:
    """Context manager that measures wall-clock time."""
    result = TimerResult(label=label)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_s = time.perf_counter() - start


# ─────────────────────────────────────────────────────────────────────────────
# Theta Canonicalization (HVA parameter gauge symmetry)
# ─────────────────────────────────────────────────────────────────────────────


def canonicalize_theta(theta: np.ndarray, *, period: float = np.pi) -> np.ndarray:
    """Canonicalize VQE parameters to the fundamental domain.

    The HVA circuit has gauge symmetries that make multiple θ values produce
    the same quantum state:

    1. **Periodicity**: RZZ(2θ) and RX(2θ) both have period π in θ.
       Verified: |ψ(θ)⟩ = |ψ(θ+π)⟩ (fidelity=1.0, same state).
    2. **Z₂ symmetry**: (-θ_zz, -θ_x) gives the same *energy* but a
       different state (fidelity≈0.9998). We canonicalize by sign to ensure
       consistent MPNN targets.

    This function maps θ to a canonical representative:
    - Wrap each parameter to [-period/2, period/2] using modular arithmetic.
    - Apply Z₂ convention: ensure the last parameter is non-negative.

    Parameters
    ----------
    theta : np.ndarray or array-like
        Parameter vector from VQE optimization. Shape (n_params,).
    period : float
        Periodicity of the gate parameters. Default π for standard HVA
        (RZZ(2θ) and RX(2θ) both have period π in θ). Use 2π for circuits
        with single-angle rotations (RZ(θ), RX(θ)).

    Returns
    -------
    np.ndarray
        Canonicalized θ in the fundamental domain.

    Notes
    -----
    This function handles the most common gauge equivalences in HVA circuits.
    It does NOT detect genuine local minima with different energy — those must
    be filtered by energy comparison or `filter_consistent_theta`.

    For bond-resolved HVA (many parameters per layer), translational invariance
    can create additional equivalences not handled here.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.size == 0:
        return theta

    result = theta.copy()
    half_period = period / 2.0

    # Step 1: Wrap each parameter to [-period/2, period/2]
    result = ((result + half_period) % period) - half_period

    # Step 2: Z₂ convention — ensure last parameter is non-negative
    if result[-1] < 0:
        result = -result

    return result



def canonicalize_sweep_z2(
    theta_array: np.ndarray,
    h_values: np.ndarray | None = None,
    *,
    period: float = np.pi,
) -> np.ndarray:
    """Canonicalize a sweep of θ vectors to enforce Z₂ continuity.

    Greedily chooses between θ_i and -θ_i at each point to minimize
    ||θ_i - θ_{i-1}||₂, enforcing branch continuity across the sweep.
    """
    theta_array = np.asarray(theta_array, dtype=np.float64)
    if theta_array.ndim != 2 or theta_array.shape[0] < 2:
        if theta_array.ndim == 1:
            return canonicalize_theta(theta_array, period=period)
        if theta_array.shape[0] == 1:
            return canonicalize_theta(theta_array[0], period=period).reshape(1, -1)
        return theta_array.copy()

    K, n_params = theta_array.shape

    if h_values is not None:
        h_values = np.asarray(h_values, dtype=np.float64)
        sort_idx = np.argsort(h_values)
        unsort_idx = np.argsort(sort_idx)
        theta_sorted = theta_array[sort_idx].copy()
    else:
        theta_sorted = theta_array.copy()
        unsort_idx = None

    half = period / 2.0
    theta_sorted = ((theta_sorted + half) % period) - half

    if theta_sorted[0, -1] < 0:
        theta_sorted[0] = -theta_sorted[0]

    for i in range(1, K):
        d_same = np.sum((theta_sorted[i] - theta_sorted[i - 1]) ** 2)
        d_flip = np.sum((-theta_sorted[i] - theta_sorted[i - 1]) ** 2)
        if d_flip < d_same:
            theta_sorted[i] = -theta_sorted[i]

    if unsort_idx is not None:
        return theta_sorted[unsort_idx]
    return theta_sorted


def filter_consistent_theta(
    theta_array: np.ndarray,
    *,
    outlier_sigma: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter θ_opt array to remove points in different local minima.

    After canonicalization (mod-π + Z₂), most gauge-equivalent θ converge
    to the same canonical value. However, the VQE can still find genuine
    local minima (different state, different energy) that canonicalization
    cannot fix. These appear as outliers far from the cluster of normal points.

    Uses robust MAD-based outlier detection: removes points whose distance
    from the median θ exceeds `outlier_sigma × MAD`.

    Parameters
    ----------
    theta_array : np.ndarray
        Array of canonicalized θ vectors, shape (n_points, n_params).
        MUST be canonicalized first (call canonicalize_theta on each row).
    outlier_sigma : float
        Number of MAD-scaled deviations to consider as outlier. Default 5.0
        (very conservative — only catches gross outliers like basin jumps
        where Δθ ≈ π/2 or larger).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (filtered_theta, mask) where mask[i] = True if point i was kept.

    Notes
    -----
    For typical TFIM HVA p=1 data:
    - Normal cluster: θ_zz ∈ [0.04, 0.12], θ_x ≈ π/8. MAD ≈ 0.01.
    - Periodic basin outliers: θ_x ≈ 3π/8 (distance ~0.8 from cluster).
    - Threshold at 5σ: 0.01 × 5 × 1.48 ≈ 0.07 → catches outliers at 0.8.

    For random/synthetic data (property tests):
    - MAD is large (~0.3+) → threshold is large → nothing gets filtered.
    """
    theta_array = np.asarray(theta_array, dtype=float)

    if len(theta_array) < 3:
        return theta_array, np.ones(len(theta_array), dtype=bool)

    # Compute distance of each point from the median (robust center)
    median_theta = np.median(theta_array, axis=0)
    distances = np.linalg.norm(theta_array - median_theta, axis=1)

    # MAD (median absolute deviation) — robust scale estimator
    med_dist = np.median(distances)
    mad = np.median(np.abs(distances - med_dist))

    if mad < 1e-10:
        # All points are nearly identical — no outliers detectable
        return theta_array, np.ones(len(theta_array), dtype=bool)

    # Adaptive threshold: median_distance + sigma × scaled_MAD
    # The 1.4826 factor converts MAD to Gaussian-equivalent σ
    threshold = med_dist + outlier_sigma * 1.4826 * mad
    mask = distances <= threshold

    return theta_array[mask], mask


def augment_theta_symmetries(
    theta: np.ndarray,
    *,
    period: float = np.pi,
    include_z2: bool = True,
    include_shift: bool = False,
    noise_std: float = 0.0,
    n_noise_variants: int = 1,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Generate symmetry-equivalent θ variants for data augmentation.

    HVA circuits have gauge symmetries that produce identical or
    near-identical quantum states. This function exploits these symmetries
    to multiply training data without additional VQE cost.

    Symmetries used:
    1. **Z₂ reflection**: (-θ) produces the same energy as (+θ) for TFIM HVA.
       For canonicalized θ ∈ [-π/2, π/2], -θ is also in [-π/2, π/2] —
       no wrapping needed (exact symmetry preserved).
    2. **Period shift**: (θ + π) produces the same state as θ (periodicity).
    3. **Optional noise**: small Gaussian perturbation for regularization.

    Parameters
    ----------
    theta : np.ndarray
        Single parameter vector, shape (n_params,). Should be canonicalized.
    period : float
        Periodicity of gate parameters. Default π (standard HVA).
    include_z2 : bool
        Include Z₂ reflection (-θ). Default True.
    include_shift : bool
        Include period-shifted variants. Default False.
    noise_std : float
        Standard deviation of Gaussian noise to add. Default 0.0 (no noise).
        Values like 0.01-0.05 provide regularization without degrading quality.
    n_noise_variants : int
        Number of noisy variants to generate per base (original + Z₂).
        Default 1. Use 2-3 for very small datasets (<20 points).
    seed : int | None
        Random seed for noise generation.

    Returns
    -------
    list[np.ndarray]
        List of augmented θ variants (does NOT include the original).
        Z₂ variants preserve exact symmetry. Noisy variants are clamped
        to [-π/2, π/2].
    """
    theta = np.asarray(theta, dtype=float)
    if theta.size == 0:
        return []

    variants: list[np.ndarray] = []
    half_period = period / 2.0

    # Z₂ symmetry: -θ gives same energy for TFIM HVA.
    # If θ ∈ [-π/2, π/2] (canonicalized), then -θ ∈ [-π/2, π/2] too.
    # NO wrapping needed — this preserves the exact symmetry.
    if include_z2:
        z2 = -theta.copy()
        # Only clamp (not wrap) — handles edge case where θ_i = ±π/2 exactly
        z2 = np.clip(z2, -half_period, half_period)
        variants.append(z2)

    # Period shift: θ + π (same state due to 2θ in gates)
    if include_shift:
        shifted = theta + period
        shifted = ((shifted + half_period) % period) - half_period
        if not np.allclose(shifted, theta, atol=1e-6):
            variants.append(shifted)

    # Gaussian noise augmentation
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        bases = [theta] + ([variants[0]] if include_z2 and variants else [])
        for base in bases:
            for _ in range(n_noise_variants):
                noisy = base + rng.normal(0, noise_std, size=base.shape)
                # Clamp to canonical domain (not wrap — avoids discontinuities)
                noisy = np.clip(noisy, -half_period, half_period)
                variants.append(noisy)

    return variants


# ─────────────────────────────────────────────────────────────────────────────
# File Versioning (anti-data-loss utility)
# ─────────────────────────────────────────────────────────────────────────────

_VERSION_SUFFIX_RE: str = r"_v\d+$"


def versioned_backup(
    file_path: Path,
    version_dir: Path | None = None,
    *,
    sidecar_metadata: dict[str, Any] | None = None,
) -> tuple[Path, int]:
    """Create a versioned copy of a file, NEVER overwriting existing versions.

    Implements sequential versioning: file.pt → _versions/file_v1.pt, file_v2.pt, ...
    Strips existing _vN suffixes from the filename to prevent stacking
    (e.g., file_v2.pt → file_v3.pt, NOT file_v2_v1.pt).

    Parameters
    ----------
    file_path : Path
        Path to the existing file to version. Must exist.
    version_dir : Path | None
        Directory to store versioned copies. Defaults to ``file_path.parent / "_versions"``.
    sidecar_metadata : dict | None
        If provided, writes a JSON sidecar file alongside the versioned copy
        with this metadata (useful for identifying versions without loading binary files).

    Returns
    -------
    tuple[Path, int]
        (path_to_versioned_copy, version_number)

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist.

    Examples
    --------
    >>> from pathlib import Path
    >>> versioned_path, v_num = versioned_backup(Path("model.pt"))
    >>> # Creates _versions/model_v1.pt (or _v2, _v3, ... if previous exist)
    >>> versioned_path, v_num = versioned_backup(
    ...     Path("data_v2.npz"),
    ...     sidecar_metadata={"reason": "retrained", "pass_rate": 0.83},
    ... )
    >>> # Creates _versions/data_v3.npz + _versions/data_v3.json
    """
    import re
    import shutil

    if not file_path.exists():
        raise FileNotFoundError(f"versioned_backup: {file_path} does not exist")

    if version_dir is None:
        version_dir = file_path.parent / "_versions"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Strip existing _vN suffix to prevent stacking (file_v2 → file, not file_v2_v1)
    raw_stem = file_path.stem
    base_stem = re.sub(_VERSION_SUFFIX_RE, "", raw_stem)
    suffix = file_path.suffix

    # Find next available version number
    version_num = 1
    while (version_dir / f"{base_stem}_v{version_num}{suffix}").exists():
        version_num += 1

    versioned_path = version_dir / f"{base_stem}_v{version_num}{suffix}"
    shutil.copy2(file_path, versioned_path)

    # Write optional JSON sidecar
    if sidecar_metadata is not None:
        sidecar_path = versioned_path.with_suffix(".json")
        if not sidecar_path.exists():
            try:
                sidecar_data = {
                    "version_number": version_num,
                    "original_filename": file_path.name,
                    **sidecar_metadata,
                }
                with open(sidecar_path, "w") as f:
                    json.dump(sidecar_data, f, indent=2, default=json_serialize)
            except Exception:
                pass  # Non-critical — sidecar is informational only

    return versioned_path, version_num


# ─────────────────────────────────────────────────────────────────────────────
# Batch Write Mixin (deferred persistence pattern)
# ─────────────────────────────────────────────────────────────────────────────


class BatchWriteMixin:
    """Mixin providing batch write semantics for JSON-persisted stores.

    Subclasses must define:
    - ``_flush()``: actually write data to disk
    - ``_reload()``: reload data from disk (for rollback on exception)

    Attributes set by mixin:
    - ``_batch_mode``: bool, True inside a batch context
    - ``_dirty``: bool, True if there are unsaved changes

    Usage
    -----
    Subclass this and call ``self._mark_dirty()`` wherever you currently
    call ``self._save()``. Then wrap operations with ``with obj.batch(): ...``.

    Example
    -------
    >>> class MyStore(BatchWriteMixin):
    ...     def _flush(self): json.dump(...)
    ...     def _reload(self): self.data = json.load(...)
    ...     def add(self, item):
    ...         self.data.append(item)
    ...         self._mark_dirty()
    ...
    >>> store = MyStore()
    >>> with store.batch():
    ...     store.add("a")
    ...     store.add("b")  # Single flush at exit
    """

    _batch_mode: bool = False
    _dirty: bool = False

    class _BatchCtx:
        """Nestable batch context with rollback on exception."""

        def __init__(self, owner: BatchWriteMixin):
            self._owner = owner
            self._was_batching = owner._batch_mode

        def __enter__(self):
            self._owner._batch_mode = True
            return self._owner

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._was_batching:
                return  # Nested — let outer handle

            self._owner._batch_mode = False
            if exc_type is not None:
                # Rollback: discard dirty state, reload from source
                self._owner._dirty = False
                self._owner._reload()
                return  # Don't suppress exception

            if self._owner._dirty:
                self._owner._flush()

    def batch(self) -> BatchWriteMixin._BatchCtx:
        """Context manager to defer disk writes until block exits.

        Supports nesting: only the outermost batch triggers flush.
        On exception: rollback (reload from disk, no partial writes).
        """
        return self._BatchCtx(self)

    def _mark_dirty(self) -> None:
        """Mark store as having unsaved changes. Flushes immediately if not batching."""
        if self._batch_mode:
            self._dirty = True
        else:
            self._flush()

    def _flush(self) -> None:
        """Override: actually persist data to disk."""
        raise NotImplementedError("Subclass must implement _flush()")

    def _reload(self) -> None:
        """Override: reload data from disk (for rollback)."""
        raise NotImplementedError("Subclass must implement _reload()")


# ─────────────────────────────────────────────────────────────────────────────
# Atomic NPZ write (crash-safe persistence)
# ─────────────────────────────────────────────────────────────────────────────


def atomic_savez(path: Path, **arrays) -> None:
    """Write NPZ file atomically using temp-file + rename.

    If the process is killed during write, the original file remains intact
    (or doesn't exist yet). Prevents corrupted NPZ files that cause
    load failures on subsequent runs.

    Parameters
    ----------
    path : Path
        Destination NPZ file path.
    **arrays
        Keyword arguments passed directly to np.savez.

    Raises
    ------
    Any exception from np.savez is re-raised after cleanup of the temp file.

    Example
    -------
    >>> from qmbp_simulation.utils.helpers import atomic_savez
    >>> atomic_savez(Path("data/results.npz"), h_values=h_arr, theta=theta_arr)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.npz")
    try:
        np.savez(tmp_path, **arrays)
        tmp_path.rename(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def compute_dataset_fingerprint(dataset: list) -> str:
    """Compute a stable fingerprint for a PyG dataset.

    Creates an 8-char hex hash from dataset composition (size, node/edge counts,
    target sum). Changes when the dataset composition changes — enables tracking
    whether two ablation/training runs used the same underlying data.

    Parameters
    ----------
    dataset : list[Data]
        PyG graph dataset (each element must have .x, .edge_index, .y).

    Returns
    -------
    str
        8-character hex hash uniquely identifying this dataset content.

    Example
    -------
    >>> from qmbp_simulation.utils.helpers import compute_dataset_fingerprint
    >>> fp = compute_dataset_fingerprint(my_dataset)
    >>> print(fp)  # e.g., "a3f8b21c"
    """
    import hashlib

    n_graphs = len(dataset)
    total_nodes = sum(g.x.shape[0] for g in dataset)
    total_edges = sum(g.edge_index.shape[1] for g in dataset)
    y_sum = sum(float(g.y.sum()) for g in dataset)

    key = f"{n_graphs}:{total_nodes}:{total_edges}:{y_sum:.6f}"
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def persist_training_curve(
    result: dict,
    output_dir: Path,
    prefix: str = "training",
) -> Path | None:
    """Persist training loss curves from a train_unified_mpnn result dict.

    Saves mse_history, val_mse_history, zz_loss_history, and x_loss_history
    as compressed NPZ for post-hoc analysis (overfitting detection, LR schedule
    evaluation, convergence speed comparison).

    Parameters
    ----------
    result : dict
        Return dict from train_unified_mpnn() or fine_tune_unified_mpnn().
        Must contain "mse_history" key at minimum.
    output_dir : Path
        Directory to save the curve file.
    prefix : str
        Filename prefix (default: "training"). Timestamp is appended.

    Returns
    -------
    Path | None
        Path to saved NPZ file, or None if no data to persist.

    Example
    -------
    >>> from qmbp_simulation.utils.helpers import persist_training_curve
    >>> curve_path = persist_training_curve(train_result, Path("results/curves"))
    """
    from datetime import datetime

    mse_history = result.get("mse_history", [])
    if not mse_history:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    curve_path = output_dir / f"{prefix}_{ts}.npz"

    curve_data = {
        "mse_history": np.array(mse_history, dtype=np.float64),
        "val_mse_history": np.array(result.get("val_mse_history", []), dtype=np.float64),
        "zz_loss_history": np.array(result.get("zz_loss_history", []), dtype=np.float64),
        "x_loss_history": np.array(result.get("x_loss_history", []), dtype=np.float64),
    }

    np.savez_compressed(curve_path, **curve_data)
    return curve_path


def tile_theta_for_higher_p(
    theta_p_low: np.ndarray,
    p_target: int,
    p_source: int = 1,
    *,
    noise_std: float = 0.05,
    noise_layers: str = "extra",
    seed: int | None = None,
    expected_n_params: int | None = None,
) -> np.ndarray:
    """Construct warm-start θ for higher p_layers by tiling lower-p solution.

    For HVA circuits, repeating the same layer parameters is a natural
    stationary point — the VQE converges much faster from this initialization
    than from random. A small perturbation on additional layers breaks
    symmetry and allows the optimizer to explore the expanded parameter space.

    Parameters
    ----------
    theta_p_low : np.ndarray
        Optimized θ from a lower p_layers circuit. Shape (n_params_per_layer * p_source,).
    p_target : int
        Target number of HVA layers (must be > p_source).
    p_source : int
        Number of layers in theta_p_low. Default 1.
    noise_std : float
        Standard deviation of Gaussian noise added to break symmetry.
        Default 0.05 rad (~3°). Set to 0.0 for exact tiling.
    noise_layers : str
        Which layers get noise: "extra" (only new layers, default),
        "all" (all layers including original).
    seed : int | None
        Random seed for reproducibility.
    expected_n_params : int | None
        If provided, validates that the output has exactly this many parameters.
        Raises ValueError if mismatch (helps catch topology/p inconsistencies).

    Returns
    -------
    np.ndarray
        θ initialization for p_target layers. Shape (n_params_per_layer * p_target,).

    Raises
    ------
    ValueError
        If p_target <= p_source, theta is empty, dimensions inconsistent,
        or expected_n_params doesn't match output.

    Examples
    --------
    >>> theta_p1 = np.array([0.1, -0.2, 0.3])  # 3 params per layer, p=1
    >>> theta_p2 = tile_theta_for_higher_p(theta_p1, p_target=2)
    >>> theta_p2.shape
    (6,)
    >>> np.allclose(theta_p2[:3], theta_p1)  # first layer is exact copy
    True
    """
    theta_p_low = np.asarray(theta_p_low, dtype=np.float64)

    if p_target <= p_source:
        raise ValueError(
            f"p_target ({p_target}) must be > p_source ({p_source}). "
            f"Use theta directly if p_target == p_source."
        )

    if theta_p_low.size == 0:
        raise ValueError("theta_p_low is empty.")

    if theta_p_low.size % p_source != 0:
        raise ValueError(
            f"theta_p_low size ({theta_p_low.size}) is not divisible by "
            f"p_source ({p_source}). Cannot determine params_per_layer."
        )

    params_per_layer = theta_p_low.size // p_source
    n_params_target = params_per_layer * p_target

    # Tile: repeat the per-layer block to fill p_target layers
    # For p_source=1 → p_target=2: [θ_layer1] → [θ_layer1, θ_layer1]
    # For p_source=2 → p_target=4: [θ_l1, θ_l2] → [θ_l1, θ_l2, θ_l1, θ_l2]
    n_repeats = p_target // p_source
    remainder = p_target % p_source

    theta_tiled = np.tile(theta_p_low, n_repeats)
    if remainder > 0:
        # Append first `remainder` layers for non-integer multiples
        extra_params = params_per_layer * remainder
        theta_tiled = np.concatenate([theta_tiled, theta_p_low[:extra_params]])

    assert theta_tiled.size == n_params_target, (
        f"Tiling bug: got {theta_tiled.size}, expected {n_params_target}"
    )

    # Add noise to break layer symmetry
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, noise_std, n_params_target)

        if noise_layers == "extra":
            # Only perturb layers beyond the original p_source
            noise[: theta_p_low.size] = 0.0
        elif noise_layers == "all":
            pass  # noise everywhere
        else:
            raise ValueError(f"noise_layers must be 'extra' or 'all', got '{noise_layers}'")

        theta_tiled += noise

    if expected_n_params is not None and theta_tiled.size != expected_n_params:
        raise ValueError(
            f"Tiled theta has {theta_tiled.size} params but circuit expects "
            f"{expected_n_params}. This likely means the topology/N doesn't "
            f"produce (n_edges + N) * p_target = {expected_n_params} parameters. "
            f"(theta_p_low.size={theta_p_low.size}, p_source={p_source}, "
            f"p_target={p_target}, params_per_layer={params_per_layer})"
        )

    return theta_tiled


def load_theta_from_npz(
    topology: str,
    n_qubits: int,
    p_layers: int = 1,
    h_values: np.ndarray | None = None,
    source: str = "multi_n_training",
) -> dict[float, np.ndarray] | None:
    """Load θ_opt from an existing NPZ file for warm-start or analysis.

    General-purpose NPZ loader for any (topology, N, p) combination.
    Reads from `data/{source}/{topology}_N{n}_p{p}.npz` and returns
    a dict mapping h → θ_opt for each available h-point.

    Use cases:
    - Warm-starting p=2 VQE from p=1 solutions (tile after loading)
    - Loading checkpoint data for analysis/comparison
    - Seeding iterative improvement from previous round

    Parameters
    ----------
    topology : str
        Lattice topology name.
    n_qubits : int
        System size.
    p_layers : int
        HVA depth of the data to load. Default 1.
    h_values : np.ndarray | None
        If provided, only return entries for these h-values (nearest match
        with tolerance 1e-4). If None, return all available.
    source : str
        Subdirectory under `data/`. Default "multi_n_training".
        Other option: "large_n_extrapolation".

    Returns
    -------
    dict[float, np.ndarray] | None
        Mapping h → θ_opt array. None if no data exists or all entries
        have NaN/Inf values.
    """
    project_root = Path(__file__).resolve().parents[3]
    npz_path = project_root / "data" / source / f"{topology}_N{n_qubits}_p{p_layers}.npz"

    if not npz_path.exists():
        return None

    try:
        data = np.load(npz_path, allow_pickle=True)
        h_stored = np.asarray(data["h_values"], dtype=np.float64)
        theta_stored = data["theta_opt"]

        if h_values is None:
            result = {}
            for i in range(len(h_stored)):
                theta_i = np.asarray(theta_stored[i], dtype=np.float64)
                if np.all(np.isfinite(theta_i)):
                    result[float(h_stored[i])] = theta_i
            return result if result else None

        # Match requested h-values to nearest stored (tolerance 1e-4)
        h_values = np.asarray(h_values, dtype=np.float64)
        result = {}
        for h_req in h_values:
            diffs = np.abs(h_stored - h_req)
            idx = int(np.argmin(diffs))
            if diffs[idx] < 1e-4:
                theta_i = np.asarray(theta_stored[idx], dtype=np.float64)
                if np.all(np.isfinite(theta_i)):
                    result[float(h_req)] = theta_i

        return result if result else None
    except Exception:
        return None


def load_p1_theta_for_warmstart(
    topology: str,
    n_qubits: int,
    h_values: np.ndarray | None = None,
) -> dict[float, np.ndarray] | None:
    """Load θ_opt from p=1 NPZ for use as warm-start seed when running p>1.

    Convenience wrapper around ``load_theta_from_npz`` for the common
    pattern of loading p=1 data to tile into p=2 initialization.

    Parameters
    ----------
    topology : str
        Lattice topology name.
    n_qubits : int
        System size.
    h_values : np.ndarray | None
        If provided, only return entries for these h-values (nearest match).
        If None, return all available.

    Returns
    -------
    dict[float, np.ndarray] | None
        Mapping h → θ_p1 array. None if no p=1 data exists.
    """
    return load_theta_from_npz(topology, n_qubits, p_layers=1, h_values=h_values)


def compute_npz_fingerprint(npz_path: Path | str) -> str:
    """Compute a stable fingerprint for an NPZ training data file.

    Hashes the content of h_values + theta_opt + e_vqe arrays to create
    a unique identifier. Changes when any training point is added, modified,
    or removed. Used for:
    - Tracking whether a model was trained on this exact data version
    - Detecting post-training data corruption/modification
    - Enabling reproducibility verification

    Parameters
    ----------
    npz_path : Path | str
        Path to the NPZ file.

    Returns
    -------
    str
        12-character hex hash uniquely identifying the data content.

    Example
    -------
    >>> fp = compute_npz_fingerprint("data/multi_n_training/chain_1d_N10_p1.npz")
    >>> print(fp)  # e.g., "a3f8b21c4e9d"
    """
    import hashlib

    npz_path = Path(npz_path)
    if not npz_path.exists():
        return "missing"

    try:
        data = np.load(str(npz_path), allow_pickle=True)
        h = np.asarray(data.get("h_values", []), dtype=np.float64)
        theta = data.get("theta_opt", np.array([]))
        e = np.asarray(data.get("e_vqe", []), dtype=np.float64)

        # Hash the raw bytes of the key arrays
        hasher = hashlib.sha256()
        hasher.update(h.tobytes())
        if hasattr(theta, 'tobytes'):
            hasher.update(np.asarray(theta, dtype=np.float64).tobytes())
        hasher.update(e.tobytes())
        return hasher.hexdigest()[:12]
    except Exception:
        return "error"
