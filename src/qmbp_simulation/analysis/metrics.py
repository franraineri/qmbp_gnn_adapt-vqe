"""Analysis Metrics — Pure computation helpers for pipeline diagnostics.

Provides signal-to-noise ratio, parameter smoothness, classification
confidence, energy decomposition, and fraction-near-ground-state
computations. These are stateless functions with no side effects.

This module has NO heavy imports (no Qiskit, no PyTorch).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


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

    # Vectorized: compute all consecutive differences at once
    diffs = np.abs(np.diff(theta_array, axis=0))
    return float(np.max(diffs))


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


def compute_fraction_near_gs(
    cost_fn,
    n_params: int,
    n_samples: int = 200,
    threshold: float = 0.05,
    gap: float = 1.0,
    e_exact: float = 0.0,
    bounds: tuple[float, float] = (-np.pi, np.pi),
    seed: int | None = None,
) -> dict[str, float]:
    """Fraction of random parameter initializations near the ground state.

    A training-free metric that estimates how accessible the ground state
    is from random starting points. Higher values indicate an easier
    optimization landscape at a given h-value.

    Parameters
    ----------
    cost_fn : callable
        Energy function E(theta) -> float.
    n_params : int
        Number of variational parameters.
    n_samples : int
        Number of random samples to evaluate.
    threshold : float
        ΔE/gap threshold for "near ground state" (default 5%).
    gap : float
        Spectral gap for normalization.
    e_exact : float
        Exact ground state energy.
    bounds : tuple[float, float]
        Parameter bounds (default [-pi, pi]).
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    dict[str, float]
        Keys: fraction_near_gs, n_near, n_samples, threshold, mean_de_gap.
    """
    rng = np.random.default_rng(seed)
    gap_safe = max(abs(gap), 1e-10)

    n_near = 0
    de_gaps = np.zeros(n_samples)

    for i in range(n_samples):
        theta = rng.uniform(bounds[0], bounds[1], n_params)
        energy = cost_fn(theta)
        de_gap = abs(energy - e_exact) / gap_safe
        de_gaps[i] = de_gap
        if de_gap < threshold:
            n_near += 1

    fraction = n_near / n_samples

    return {
        "fraction_near_gs": float(fraction),
        "n_near": n_near,
        "n_samples": n_samples,
        "threshold": float(threshold),
        "mean_de_gap": float(np.mean(de_gaps)),
    }
