"""Analysis Metrics — Pure computation helpers for pipeline diagnostics.

Provides signal-to-noise ratio, parameter smoothness, classification
confidence, and energy decomposition computations. These are stateless
functions with no side effects.

This module has NO heavy imports (no Qiskit, no PyTorch).
"""

from __future__ import annotations

import numpy as np


def compute_snr(observable_value: float, shots: int) -> float:
    """Signal-to-noise ratio: |⟨O⟩| * √shots.

    Quantifies measurement reliability. Higher SNR indicates the observable
    signal dominates over shot noise (σ = 1/√shots).

    Parameters
    ----------
    observable_value : float
        Measured expectation value ⟨O⟩.
    shots : int
        Number of measurement shots (must be positive integer).

    Returns
    -------
    float
        Non-negative SNR value: |observable_value| * sqrt(shots).

    Raises
    ------
    ValueError
        If shots is not a positive integer.
    """
    if not isinstance(shots, int | np.integer) or shots <= 0:
        raise ValueError(f"shots must be a positive integer, got {shots}")
    return float(abs(observable_value) * np.sqrt(shots))


def compute_theta_smoothness(theta_array: np.ndarray) -> float | None:
    """Maximum parameter discontinuity across the h-sweep.

    Computes max_i ||θ(h_i) - θ(h_{i-1})||_∞ — the largest infinity-norm
    difference between consecutive θ vectors. Small values indicate a smooth
    parameter landscape (good for MPNN learnability); large values indicate
    discontinuities where the MPNN will struggle.

    Parameters
    ----------
    theta_array : np.ndarray
        Shape (n_h_points, n_params). Rows ordered by h (descending,
        matching the VQE sweep direction).

    Returns
    -------
    float | None
        Non-negative smoothness metric, or None if fewer than 2 h-points.
    """
    if theta_array.shape[0] < 2:
        return None

    max_diff = 0.0
    for i in range(1, theta_array.shape[0]):
        diff = float(np.max(np.abs(theta_array[i] - theta_array[i - 1])))
        max_diff = max(max_diff, diff)
    return max_diff


def compute_classification_confidence(
    mag_x: float,
    corr_zz: float,
    shots: int,
) -> float:
    """Phase classification confidence: |⟨X⟩ - ⟨ZZ⟩| * √shots.

    Measures how confidently the pipeline can distinguish between the
    paramagnetic (⟨X⟩ dominant) and antiferromagnetic (⟨ZZ⟩ dominant) phases.
    Higher values indicate clearer phase separation relative to shot noise.

    Parameters
    ----------
    mag_x : float
        Measured transverse magnetization ⟨X⟩.
    corr_zz : float
        Measured nearest-neighbor ZZ correlation ⟨ZZ⟩.
    shots : int
        Number of measurement shots (must be positive integer).

    Returns
    -------
    float
        Non-negative classification confidence value.

    Raises
    ------
    ValueError
        If shots is not a positive integer.
    """
    if not isinstance(shots, int | np.integer) or shots <= 0:
        raise ValueError(f"shots must be a positive integer, got {shots}")
    return float(abs(mag_x - corr_zz) * np.sqrt(shots))


def compute_energy_decomposition(
    e_exact: float,
    e_vqe_ceiling: float,
    e_predicted: float,
) -> dict[str, float]:
    """Decompose total energy error into circuit vs MPNN contributions.

    Separates the total prediction error |e_predicted - e_exact| into:
    - error_from_circuit: |e_vqe_ceiling - e_exact| — physics limit of HVA p=2
    - error_from_mpnn: |e_predicted - e_vqe_ceiling| — ML prediction error

    Invariant: error_from_circuit + error_from_mpnn == |e_predicted - e_exact|
    within floating-point tolerance (1e-12).

    Parameters
    ----------
    e_exact : float
        Exact ground state energy (from Phase 1 exact diagonalization).
    e_vqe_ceiling : float
        Best achievable energy with HVA p=2: E_VQE(θ_opt).
    e_predicted : float
        Energy using MPNN-predicted parameters: E_VQE(θ_MPNN).

    Returns
    -------
    dict[str, float]
        Keys: e_exact, e_vqe_ceiling, e_mpnn_predicted,
              error_from_circuit, error_from_mpnn.
    """
    error_from_circuit = abs(e_vqe_ceiling - e_exact)
    error_from_mpnn = abs(e_predicted - e_vqe_ceiling)

    return {
        "e_exact": float(e_exact),
        "e_vqe_ceiling": float(e_vqe_ceiling),
        "e_mpnn_predicted": float(e_predicted),
        "error_from_circuit": float(error_from_circuit),
        "error_from_mpnn": float(error_from_mpnn),
    }
