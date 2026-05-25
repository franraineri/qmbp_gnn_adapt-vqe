"""
Pipeline Utilities — Phase coupling safeguards, dataset metadata, and
observable locality checks.

Implements the V6 pipeline integrity requirements:
  - Dataset metadata with cost_function, version, library versions (Req 8.1)
  - Phase 3 data loading validation (Req 8.2)
  - Observable locality assertion (Req 8.4)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "v6.0"
EXPECTED_COST_FUNCTION = "energy"


# ── Task 9.1: Dataset metadata ──────────────────────────────────────────


def get_library_versions() -> dict[str, str]:
    """Collect current library versions for reproducibility metadata."""
    versions = {}
    try:
        import qiskit

        versions["qiskit"] = qiskit.__version__
    except Exception:
        versions["qiskit"] = "unknown"
    try:
        import numpy

        versions["numpy"] = numpy.__version__
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
    per_site_mag_x: np.ndarray | None = None,
    per_bond_corr_zz: np.ndarray | None = None,
) -> None:
    """Save Phase 1+2 dataset with metadata for Phase 3 validation.

    Includes cost_function="energy", version="v6.0", and library versions.
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

    if per_site_mag_x is not None:
        save_dict["per_site_mag_x"] = per_site_mag_x
    if per_bond_corr_zz is not None:
        save_dict["per_bond_corr_zz"] = per_bond_corr_zz

    np.savez(filepath, **save_dict)
    logger.info(
        f"Dataset saved: {filepath} (version={PIPELINE_VERSION}, cost={EXPECTED_COST_FUNCTION})"
    )


# ── Task 9.2: Phase 3 data loading validation ───────────────────────────


def load_phase12_dataset(filepath: str | Path) -> dict:
    """Load Phase 1+2 dataset with cost_function validation.

    Raises
    ------
    ValueError
        If cost_function metadata does not match expected value.
    """
    data = dict(np.load(filepath, allow_pickle=True))

    # Validate cost_function metadata
    cost_fn = str(data.get("cost_function", ""))
    if cost_fn != EXPECTED_COST_FUNCTION:
        raise ValueError(
            f"Phase coupling mismatch: dataset has cost_function='{cost_fn}', "
            f"but Phase 3 expects cost_function='{EXPECTED_COST_FUNCTION}'. "
            f"This is the V5.x failure mode — Phase 2 cost function was changed "
            f"without updating Phase 3. Re-run Phase 2 with pure energy cost."
        )

    version = str(data.get("version", "unknown"))
    logger.info(f"Dataset loaded: {filepath} (version={version}, cost={cost_fn})")
    return data


# ── Task 9.3: Observable locality assertion ──────────────────────────────


def assert_observable_locality(
    observables: list[SparsePauliOp],
    lattice_edges: list[tuple[int, int]],
    max_weight: int = 2,
) -> None:
    """Verify all hardware-path observables act on ≤ max_weight adjacent qubits.

    Parameters
    ----------
    observables : list[SparsePauliOp]
        All observables to be measured on hardware.
    lattice_edges : list[tuple[int, int]]
        Valid adjacent qubit pairs from the lattice.
    max_weight : int
        Maximum number of non-identity qubits per observable (default 2).

    Raises
    ------
    ValueError
        If any observable violates the locality constraint.
    """
    edge_set = set()
    for i, j in lattice_edges:
        edge_set.add((min(i, j), max(i, j)))

    for idx, obs in enumerate(observables):
        for label_obj in obs.paulis:
            label = str(label_obj)
            # Count non-identity positions
            non_id_positions = [i for i, c in enumerate(reversed(label)) if c != "I"]
            weight = len(non_id_positions)

            if weight > max_weight:
                raise ValueError(
                    f"Observable {idx} has weight {weight} (label='{label}'), "
                    f"exceeding max_weight={max_weight}. "
                    f"Only local observables (≤ {max_weight} adjacent qubits) "
                    f"are allowed on hardware."
                )

            if weight == 2:
                pair = (min(non_id_positions), max(non_id_positions))
                if pair not in edge_set:
                    raise ValueError(
                        f"Observable {idx} (label='{label}') acts on qubits "
                        f"{pair}, which are not adjacent in the lattice. "
                        f"Only nearest-neighbor observables are allowed."
                    )
