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

    np.savez(filepath, **save_dict)
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
