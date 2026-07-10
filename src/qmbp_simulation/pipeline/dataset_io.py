"""
Dataset I/O — Phase 1+2 dataset persistence with metadata validation.

Migrated from src/poc/v6/pipeline_utils.py. Implements:
  - Dataset save with cost_function, version, and library version metadata
  - Dataset load with cost_function="energy" enforcement (V5.x failure guard)
  - Deprecation warning for old schema versions

Requirements: 10.1, 10.2, 10.3, 20.1, 20.4
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "v7.0"
EXPECTED_COST_FUNCTION = "energy"

# Schema versions considered legacy (emit deprecation warning)
_LEGACY_VERSIONS = {"v5.0", "v5.1", "v6.0"}


def get_library_versions() -> dict[str, str]:
    """Collect current library versions for reproducibility metadata."""
    versions = {}
    try:
        import qiskit

        versions["qiskit"] = qiskit.__version__
    except Exception:
        versions["qiskit"] = "unknown"
    try:
        import numpy as _np

        versions["numpy"] = _np.__version__
    except Exception:
        versions["numpy"] = "unknown"
    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except Exception:
        versions["scipy"] = "unknown"
    try:
        import torch

        versions["torch"] = torch.__version__
    except Exception:
        versions["torch"] = "unknown"
    try:
        import torch_geometric

        versions["torch_geometric"] = torch_geometric.__version__
    except Exception:
        versions["torch_geometric"] = "unknown"
    return versions


def save_phase12_dataset(
    filepath: str | Path,
    *,
    h_values: np.ndarray,
    J: float,
    n_qubits: int,
    p_layers: int,
    ground_energies: np.ndarray,
    gaps: np.ndarray,
    mag_x: np.ndarray,
    corr_zz: np.ndarray,
    theta_opt: np.ndarray,
    vqe_energies: np.ndarray,
    fidelities: np.ndarray,
    **kwargs,
) -> None:
    """Save Phase 1+2 dataset with metadata for Phase 3 validation.

    Includes cost_function="energy", version, and library versions.
    Additional keyword arguments are stored as extra arrays in the .npz file.

    Parameters
    ----------
    filepath : str | Path
        Output path for the .npz file.
    h_values : np.ndarray
        Transverse field values used in the sweep.
    J : float
        Coupling constant.
    n_qubits : int
        Number of lattice sites.
    p_layers : int
        HVA circuit depth.
    ground_energies : np.ndarray
        Exact ground state energies per h-point.
    gaps : np.ndarray
        Spectral gaps per h-point.
    mag_x : np.ndarray
        Bulk-averaged magnetization per h-point.
    corr_zz : np.ndarray
        Bulk-averaged ZZ correlation per h-point.
    theta_opt : np.ndarray
        Optimized VQE parameters per h-point.
    vqe_energies : np.ndarray
        VQE energies per h-point.
    fidelities : np.ndarray
        State fidelities per h-point.
    **kwargs
        Additional arrays to store (e.g., per_site_mag_x, per_bond_corr_zz).
    """
    versions = get_library_versions()

    save_dict = dict(
        # Metadata
        cost_function=EXPECTED_COST_FUNCTION,
        version=PIPELINE_VERSION,
        lib_qiskit=versions.get("qiskit", "unknown"),
        lib_numpy=versions.get("numpy", "unknown"),
        lib_scipy=versions.get("scipy", "unknown"),
        lib_torch=versions.get("torch", "unknown"),
        lib_torch_geometric=versions.get("torch_geometric", "unknown"),
        # Data
        h_values=h_values,
        J=J,
        n_qubits=n_qubits,
        p_layers=p_layers,
        ground_energies=ground_energies,
        gaps=gaps,
        mag_x=mag_x,
        corr_zz=corr_zz,
        theta_opt=theta_opt,
        vqe_energies=vqe_energies,
        fidelities=fidelities,
    )

    # Store any additional keyword arguments
    for key, value in kwargs.items():
        if value is not None:
            save_dict[key] = value

    np.savez(filepath, **save_dict)  # type: ignore[arg-type]
    logger.info(
        f"Dataset saved: {filepath} (version={PIPELINE_VERSION}, cost={EXPECTED_COST_FUNCTION})"
    )


def load_phase12_dataset(filepath: str | Path) -> dict:
    """Load Phase 1+2 dataset with cost_function validation.

    Validates that the dataset was produced with pure energy cost function.
    Emits a deprecation warning if the schema version is older than current.

    Parameters
    ----------
    filepath : str | Path
        Path to the .npz dataset file.

    Returns
    -------
    dict
        Dataset contents as a dictionary of arrays and metadata.

    Raises
    ------
    ValueError
        If cost_function metadata does not equal "energy".
    """
    data = dict(np.load(filepath, allow_pickle=True))

    # Validate cost_function metadata
    cost_fn = str(data.get("cost_function", ""))
    if cost_fn != EXPECTED_COST_FUNCTION:
        raise ValueError(
            f"Phase coupling mismatch: dataset has cost_function='{cost_fn}', "
            f"but the pipeline expects cost_function='{EXPECTED_COST_FUNCTION}'. "
            f"This is the V5.x failure mode — Phase 2 cost function was changed "
            f"without updating Phase 3. Re-run Phase 2 with pure energy cost."
        )

    # ── Dataset integrity validation ─────────────────────────────────
    _validate_dataset_integrity(data, filepath)

    # Check schema version and emit deprecation warning for old versions
    version = str(data.get("version", "unknown"))
    if version in _LEGACY_VERSIONS:
        warnings.warn(
            f"Dataset at '{filepath}' uses legacy schema version '{version}'. "
            f"Current version is '{PIPELINE_VERSION}'. The dataset is still "
            f"loadable but may lack newer metadata fields. Consider re-generating "
            f"with the current pipeline.",
            DeprecationWarning,
            stacklevel=2,
        )

    logger.info(f"Dataset loaded: {filepath} (version={version}, cost={cost_fn})")
    return data


# ── Dataset integrity validation ─────────────────────────────────────────────


def _validate_dataset_integrity(data: dict, filepath: str | Path) -> None:
    """Validate dataset array shapes, NaN/Inf, and value ranges.

    Raises ValueError on critical issues, emits warnings on non-critical ones.

    Parameters
    ----------
    data : dict
        Loaded dataset dictionary.
    filepath : str | Path
        Path for error messages.
    """
    h_values = data.get("h_values")
    if h_values is None:
        raise ValueError(f"Dataset at '{filepath}' is missing 'h_values' array.")

    h_values = np.asarray(h_values)
    n_points = len(h_values)

    if n_points == 0:
        raise ValueError(f"Dataset at '{filepath}' has empty h_values array.")

    # Shape consistency checks
    _required_arrays = {
        "ground_energies": n_points,
        "gaps": n_points,
        "vqe_energies": n_points,
        "fidelities": n_points,
    }
    for name, expected_len in _required_arrays.items():
        arr = data.get(name)
        if arr is None:
            continue  # Optional arrays are OK to skip
        arr = np.asarray(arr)
        if len(arr) != expected_len:
            raise ValueError(
                f"Dataset at '{filepath}': array '{name}' has length {len(arr)}, "
                f"expected {expected_len} (matching h_values)."
            )

    # theta_opt shape: [n_points, n_params]
    theta_opt = data.get("theta_opt")
    if theta_opt is not None:
        theta_opt = np.asarray(theta_opt)
        if theta_opt.ndim == 2 and theta_opt.shape[0] != n_points:
            raise ValueError(
                f"Dataset at '{filepath}': theta_opt has {theta_opt.shape[0]} rows, "
                f"expected {n_points}."
            )

    # NaN/Inf checks on critical arrays
    for name in ("ground_energies", "vqe_energies", "gaps"):
        arr = data.get(name)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        n_bad = np.sum(~np.isfinite(arr))
        if n_bad > 0:
            raise ValueError(
                f"Dataset at '{filepath}': array '{name}' contains "
                f"{n_bad} NaN/Inf values. Dataset is corrupted."
            )

    # Fidelity range check [0, 1]
    fidelities = data.get("fidelities")
    if fidelities is not None:
        fidelities = np.asarray(fidelities, dtype=float)
        if np.any(fidelities < -0.01) or np.any(fidelities > 1.01):
            warnings.warn(
                f"Dataset at '{filepath}': fidelities contain values outside "
                f"[0, 1] range (min={fidelities.min():.4f}, max={fidelities.max():.4f}). "
                f"This may indicate corrupted VQE results.",
                RuntimeWarning,
                stacklevel=3,
            )

    # Gap non-negativity check
    gaps = data.get("gaps")
    if gaps is not None:
        gaps = np.asarray(gaps, dtype=float)
        if np.any(gaps < 0):
            warnings.warn(
                f"Dataset at '{filepath}': gaps contain negative values "
                f"(min={gaps.min():.6f}). This violates spectral gap definition.",
                RuntimeWarning,
                stacklevel=3,
            )

    # Descending order check (warning only — some datasets may be ascending)
    if n_points >= 2 and h_values[0] < h_values[-1]:
        warnings.warn(
            f"Dataset at '{filepath}': h_values are in ascending order "
            f"(h[0]={h_values[0]:.2f}, h[-1]={h_values[-1]:.2f}). "
            f"Pipeline expects descending order (h=2→0).",
            RuntimeWarning,
            stacklevel=3,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Non-uniform h-grid generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_nonuniform_h_grid(
    h_min: float,
    h_max: float,
    n_points: int,
    h_critical: float | None = None,
    dense_fraction: float = 0.4,
    dense_radius: float = 0.5,
) -> np.ndarray:
    """Generate a non-uniform h-grid with denser sampling near h_critical.

    Produces a descending grid (h_max → h_min) with more points concentrated
    around the critical region where the phase transition occurs and the VQE
    landscape is most complex.

    Parameters
    ----------
    h_min : float
        Minimum field value.
    h_max : float
        Maximum field value.
    n_points : int
        Total number of grid points.
    h_critical : float | None
        Critical field value for dense sampling. If None, uses midpoint.
    dense_fraction : float
        Fraction of points allocated to the dense region (default 0.4 = 40%).
    dense_radius : float
        Half-width of the dense region around h_critical (default 0.5).

    Returns
    -------
    np.ndarray
        Sorted descending h-values with denser sampling near h_critical.

    Examples
    --------
    >>> grid = generate_nonuniform_h_grid(1.0, 5.0, 30, h_critical=1.5)
    >>> len(grid)
    30
    >>> grid[0] > grid[-1]  # descending
    True
    """
    if h_min >= h_max:
        raise ValueError(f"h_min ({h_min}) must be < h_max ({h_max}).")
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}.")

    if h_critical is None:
        h_critical = (h_min + h_max) / 2.0

    logger.debug(
        "generate_nonuniform_h_grid: [%.2f, %.2f], n=%d, h_c=%.2f, dense_frac=%.1f, radius=%.2f",
        h_min,
        h_max,
        n_points,
        h_critical,
        dense_fraction,
        dense_radius,
    )

    # Clamp h_critical to valid range
    h_critical = max(h_min + dense_radius * 0.5, min(h_max - dense_radius * 0.5, h_critical))

    # Split points between dense and sparse regions
    n_dense = max(3, int(n_points * dense_fraction))
    n_sparse = n_points - n_dense

    # Dense region: ±dense_radius around h_critical
    dense_lo = max(h_min, h_critical - dense_radius)
    dense_hi = min(h_max, h_critical + dense_radius)
    dense_points = np.linspace(dense_lo, dense_hi, n_dense)

    # Sparse region: remainder of [h_min, h_max] excluding dense zone
    sparse_points = []
    n_below = n_sparse // 2
    n_above = n_sparse - n_below

    if dense_lo > h_min and n_below > 0:
        sparse_points.extend(np.linspace(h_min, dense_lo, n_below + 1)[:-1].tolist())
    if dense_hi < h_max and n_above > 0:
        sparse_points.extend(np.linspace(dense_hi, h_max, n_above + 1)[1:].tolist())

    # Combine, deduplicate, sort descending
    all_points = np.concatenate([dense_points, np.array(sparse_points)])
    all_points = np.unique(all_points)

    # Ensure exactly n_points by resampling from the combined range
    if len(all_points) != n_points:
        # Interpolate to get exactly n_points from the non-uniform distribution
        # This preserves the density pattern while guaranteeing exact count
        target_indices = np.linspace(0, len(all_points) - 1, n_points)
        all_points = np.interp(target_indices, np.arange(len(all_points)), all_points)

    # Sort descending (for warm-start sweep)
    return np.sort(all_points)[::-1]
