"""Post-VQE theta alignment — eliminates parameter discontinuities.

When VQE multi-restart finds equivalent minima on different branches of the
energy landscape at adjacent h-points, the resulting θ(h) curve has
discontinuities (θ_smoothness ≈ π). This module detects such jumps and
re-optimizes the affected points using the neighbor's θ as seed, producing
a smooth θ(h) suitable for MPNN interpolation.

The alignment is model-agnostic — it operates purely on the θ arrays and
an energy evaluation function, independent of the Hamiltonian structure.

Usage
-----
    from qmbp_simulation.analysis.theta_alignment import align_theta_sweep

    aligned_results = align_theta_sweep(
        vqe_results=vqe_results,
        circuit=circuit,
        hamiltonians=hamiltonians,
        backend=backend,
        jump_threshold=2.0,
        max_reopt_iters=200,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

    from qmbp_simulation.execution.backends import ExecutionBackend
    from qmbp_simulation.models.data_models import VQEResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AlignmentReport:
    """Summary of the theta alignment pass.

    Attributes
    ----------
    n_points : int
        Total number of h-points in the sweep.
    n_jumps_detected : int
        Number of consecutive-pair discontinuities detected.
    n_realigned : int
        Number of points successfully re-optimized with neighbor seed.
    n_failed : int
        Number of re-optimization attempts that did not improve.
    jump_indices : list[int]
        Indices where jumps were detected.
    original_smoothness : float
        max ||θ_i - θ_{i-1}||_∞ before alignment.
    final_smoothness : float
        max ||θ_i - θ_{i-1}||_∞ after alignment.
    energy_degradation_max : float
        Maximum energy increase from alignment (should be ≤ 0 or tiny positive).
    """

    n_points: int = 0
    n_jumps_detected: int = 0
    n_realigned: int = 0
    n_failed: int = 0
    jump_indices: list[int] = field(default_factory=list)
    original_smoothness: float = 0.0
    final_smoothness: float = 0.0
    energy_degradation_max: float = 0.0

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        return {
            "n_points": self.n_points,
            "n_jumps_detected": self.n_jumps_detected,
            "n_realigned": self.n_realigned,
            "n_failed": self.n_failed,
            "jump_indices": self.jump_indices,
            "original_smoothness": self.original_smoothness,
            "final_smoothness": self.final_smoothness,
            "energy_degradation_max": self.energy_degradation_max,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Core alignment functions
# ═══════════════════════════════════════════════════════════════════════════════


def detect_jumps(
    theta_array: np.ndarray,
    threshold: float = 2.0,
) -> list[int]:
    """Detect indices where θ has a discontinuity.

    A jump is detected at index i if ||θ[i] - θ[i-1]||_∞ > threshold.
    The threshold default of 2.0 (~2π/3) catches branch switches while
    ignoring normal parameter evolution.

    Parameters
    ----------
    theta_array : np.ndarray
        Shape (n_points, n_params). Ordered by h (descending sweep).
    threshold : float
        L-infinity norm threshold for declaring a jump.

    Returns
    -------
    list[int]
        Indices of points where a jump FROM the previous point was detected.
        E.g., [5, 12] means θ[5] jumped relative to θ[4], and θ[12] jumped
        relative to θ[11].
    """
    if theta_array.shape[0] < 2:
        return []

    diffs = np.max(np.abs(np.diff(theta_array, axis=0)), axis=1)
    jump_mask = diffs > threshold
    return [int(i + 1) for i in np.where(jump_mask)[0]]


def _reoptimize_point(
    seed_theta: np.ndarray,
    hamiltonian: SparsePauliOp,
    circuit: QuantumCircuit,
    backend: ExecutionBackend,
    maxiter: int = 200,
) -> tuple[np.ndarray, float]:
    """Re-optimize a single point using neighbor's θ as seed.

    Uses L-BFGS-B with tight bounds — the goal is NOT global search
    but rather to descend from the neighbor seed into the nearest local
    minimum (which should be the same branch).

    Returns
    -------
    tuple[np.ndarray, float]
        (theta_opt, energy)
    """
    from scipy.optimize import minimize

    def cost(params: np.ndarray) -> float:
        return backend.evaluate(circuit, hamiltonian, params)

    bounds = [(-np.pi, np.pi)] * len(seed_theta)
    result = minimize(
        cost,
        seed_theta.copy(),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
    )
    return result.x.copy(), float(result.fun)


def align_theta_sweep(
    vqe_results: list[VQEResult],
    circuit: QuantumCircuit,
    hamiltonians: list[SparsePauliOp],
    backend: ExecutionBackend,
    *,
    jump_threshold: float = 2.0,
    max_reopt_iters: int = 200,
    energy_tolerance: float = 1e-4,
) -> tuple[list[VQEResult], AlignmentReport]:
    """Align θ across the h-sweep to eliminate branch discontinuities.

    For each detected jump, re-optimizes the jumped point using the
    previous point's θ as seed. Accepts the new θ only if it achieves
    energy within `energy_tolerance` of the original (i.e., it found
    the same-energy minimum on the smooth branch).

    The alignment propagates forward: if θ[5] is realigned, subsequent
    points θ[6], θ[7], ... are checked against the new θ[5].

    Parameters
    ----------
    vqe_results : list[VQEResult]
        VQE results from descending sweep (ordered h_max → h_min).
    circuit : QuantumCircuit
        Parameterized HVA circuit.
    hamiltonians : list[SparsePauliOp]
        Hamiltonian for each h-point (same order as vqe_results).
    backend : ExecutionBackend
        Backend for energy evaluation during re-optimization.
    jump_threshold : float
        L-infinity norm threshold for detecting jumps (default 2.0 ≈ 2π/3).
    max_reopt_iters : int
        Max L-BFGS-B iterations for re-optimization (default 200).
    energy_tolerance : float
        Accept realignment if |E_new - E_original| < tolerance (default 1e-4).

    Returns
    -------
    tuple[list[VQEResult], AlignmentReport]
        Aligned VQE results (new list, originals unchanged) and report.
    """
    import copy

    n_points = len(vqe_results)
    if n_points < 2:
        report = AlignmentReport(n_points=n_points)
        return list(vqe_results), report

    if len(hamiltonians) != n_points:
        raise ValueError(
            f"hamiltonians length ({len(hamiltonians)}) must match vqe_results length ({n_points})."
        )

    # Extract theta array for jump detection
    theta_array = np.array([r.theta_opt for r in vqe_results])
    original_smoothness = float(np.max(np.abs(np.diff(theta_array, axis=0))))

    # Detect jumps on the original data
    jumps = detect_jumps(theta_array, threshold=jump_threshold)

    if not jumps:
        logger.info("θ alignment: no jumps detected (sweep is smooth).")
        report = AlignmentReport(
            n_points=n_points,
            original_smoothness=original_smoothness,
            final_smoothness=original_smoothness,
        )
        return list(vqe_results), report

    logger.info(
        f"θ alignment: {len(jumps)} jump(s) detected at indices {jumps} "
        f"(smoothness={original_smoothness:.3f}, threshold={jump_threshold:.1f})"
    )

    # Deep copy results to avoid mutating originals
    aligned = [copy.deepcopy(r) for r in vqe_results]
    n_realigned = 0
    n_failed = 0
    energy_degradation_max = 0.0

    # Iterative alignment: after fixing a jump, re-check downstream
    # because the new θ at idx may resolve (or create) subsequent jumps.
    processed = set()
    max_passes = 3  # Prevent infinite loops

    for pass_num in range(max_passes):
        current_theta = np.array([r.theta_opt for r in aligned])
        current_jumps = detect_jumps(current_theta, threshold=jump_threshold)
        # Only process new jumps not already handled
        new_jumps = [j for j in current_jumps if j not in processed]
        if not new_jumps:
            break

        for idx in new_jumps:
            processed.add(idx)
            prev_theta = aligned[idx - 1].theta_opt
            original_energy = aligned[idx].energy
            H_idx = hamiltonians[idx]

            # Re-optimize using neighbor's θ as seed
            new_theta, new_energy = _reoptimize_point(
                seed_theta=prev_theta,
                hamiltonian=H_idx,
                circuit=circuit,
                backend=backend,
                maxiter=max_reopt_iters,
            )

            # Accept if energy is comparable (same minimum, different branch)
            energy_diff = new_energy - original_energy
            if energy_diff < energy_tolerance:
                # Success: realigned to smooth branch
                aligned[idx].theta_opt = new_theta
                aligned[idx].energy = new_energy
                aligned[idx].energy_error = abs(new_energy - original_energy)
                energy_degradation_max = max(energy_degradation_max, energy_diff)
                n_realigned += 1

                new_smoothness_at_idx = float(np.max(np.abs(new_theta - prev_theta)))
                logger.info(
                    f"  idx={idx} (h={aligned[idx].h_value:.3f}): REALIGNED "
                    f"||Δθ||_∞: {float(np.max(np.abs(vqe_results[idx].theta_opt - prev_theta))):.3f}"
                    f" → {new_smoothness_at_idx:.3f}, "
                    f"ΔE={energy_diff:.2e}"
                )
            else:
                # Re-optimization found a worse minimum — keep original
                n_failed += 1
                logger.info(
                    f"  idx={idx} (h={aligned[idx].h_value:.3f}): KEPT ORIGINAL "
                    f"(reopt energy worse by {energy_diff:.2e})"
                )

    # Compute final smoothness
    final_theta = np.array([r.theta_opt for r in aligned])
    final_smoothness = float(np.max(np.abs(np.diff(final_theta, axis=0))))

    report = AlignmentReport(
        n_points=n_points,
        n_jumps_detected=len(jumps),
        n_realigned=n_realigned,
        n_failed=n_failed,
        jump_indices=jumps,
        original_smoothness=original_smoothness,
        final_smoothness=final_smoothness,
        energy_degradation_max=energy_degradation_max,
    )

    logger.info(
        f"θ alignment complete: {n_realigned}/{len(jumps)} realigned, "
        f"smoothness {original_smoothness:.3f} → {final_smoothness:.3f}"
    )
    return aligned, report


def align_theta_array(
    theta_array: np.ndarray,
    energies: np.ndarray,
    circuit: QuantumCircuit,
    hamiltonians: list[SparsePauliOp],
    backend: ExecutionBackend,
    *,
    jump_threshold: float = 2.0,
    max_reopt_iters: int = 200,
    energy_tolerance: float = 1e-4,
) -> tuple[np.ndarray, AlignmentReport]:
    """Align a raw theta array (for use in noiseless pipeline dict-based flow).

    Same logic as align_theta_sweep but operates on numpy arrays directly
    instead of VQEResult objects. Used by the noiseless experiment runner.

    Parameters
    ----------
    theta_array : np.ndarray
        Shape (n_points, n_params). Ordered by h (descending).
    energies : np.ndarray
        Shape (n_points,). VQE energies corresponding to each theta row.
    circuit : QuantumCircuit
        Parameterized HVA circuit.
    hamiltonians : list[SparsePauliOp]
        Hamiltonian per h-point.
    backend : ExecutionBackend
        For energy evaluation.
    jump_threshold : float
        L-inf threshold for jump detection.
    max_reopt_iters : int
        Max iterations for re-optimization.
    energy_tolerance : float
        Energy tolerance for accepting realignment.

    Returns
    -------
    tuple[np.ndarray, AlignmentReport]
        Aligned theta array (copy) and report.
    """
    n_points = theta_array.shape[0]
    if n_points < 2:
        report = AlignmentReport(n_points=n_points)
        return theta_array.copy(), report

    if len(hamiltonians) != n_points:
        raise ValueError(
            f"hamiltonians length ({len(hamiltonians)}) must match theta_array rows ({n_points})."
        )

    original_smoothness = float(np.max(np.abs(np.diff(theta_array, axis=0))))
    jumps = detect_jumps(theta_array, threshold=jump_threshold)

    if not jumps:
        logger.info("θ alignment: no jumps detected (sweep is smooth).")
        report = AlignmentReport(
            n_points=n_points,
            original_smoothness=original_smoothness,
            final_smoothness=original_smoothness,
        )
        return theta_array.copy(), report

    logger.info(
        f"θ alignment: {len(jumps)} jump(s) at indices {jumps} "
        f"(smoothness={original_smoothness:.3f})"
    )

    aligned = theta_array.copy()
    aligned_energies = energies.copy()
    n_realigned = 0
    n_failed = 0
    energy_degradation_max = 0.0

    # Iterative alignment: re-check downstream after each fix
    processed = set()
    max_passes = 3

    for pass_num in range(max_passes):
        current_jumps = detect_jumps(aligned, threshold=jump_threshold)
        new_jumps = [j for j in current_jumps if j not in processed]
        if not new_jumps:
            break

        for idx in new_jumps:
            processed.add(idx)
            prev_theta = aligned[idx - 1]
            original_energy = aligned_energies[idx]

            new_theta, new_energy = _reoptimize_point(
                seed_theta=prev_theta,
                hamiltonian=hamiltonians[idx],
                circuit=circuit,
                backend=backend,
                maxiter=max_reopt_iters,
            )

            energy_diff = new_energy - original_energy
            if energy_diff < energy_tolerance:
                aligned[idx] = new_theta
                aligned_energies[idx] = new_energy
                energy_degradation_max = max(energy_degradation_max, energy_diff)
                n_realigned += 1
                logger.info(f"  idx={idx}: REALIGNED, ΔE={energy_diff:.2e}")
            else:
                n_failed += 1
                logger.info(f"  idx={idx}: KEPT ORIGINAL (reopt worse by {energy_diff:.2e})")

    final_smoothness = float(np.max(np.abs(np.diff(aligned, axis=0))))

    report = AlignmentReport(
        n_points=n_points,
        n_jumps_detected=len(jumps),
        n_realigned=n_realigned,
        n_failed=n_failed,
        jump_indices=jumps,
        original_smoothness=original_smoothness,
        final_smoothness=final_smoothness,
        energy_degradation_max=energy_degradation_max,
    )

    logger.info(
        f"θ alignment complete: {n_realigned}/{len(jumps)} realigned, "
        f"smoothness {original_smoothness:.3f} → {final_smoothness:.3f}"
    )
    return aligned, report


# ═══════════════════════════════════════════════════════════════════════════════
# Outlier Detection & Filtering
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class OutlierReport:
    """Summary of the outlier filtering pass.

    Attributes
    ----------
    n_points : int
        Total number of points in the input.
    n_outliers : int
        Number of points flagged as outliers.
    outlier_indices : list[int]
        Indices of outlier points.
    outlier_h_values : list[float]
        h-values of outlier points (for logging).
    method : str
        Detection method used.
    threshold : float
        Threshold parameter used for detection.
    """

    n_points: int
    n_outliers: int
    outlier_indices: list[int] = field(default_factory=list)
    outlier_h_values: list[float] = field(default_factory=list)
    method: str = "neighbor_deviation"
    threshold: float = 2.0


def detect_theta_outliers(
    theta_array: np.ndarray,
    h_values: np.ndarray,
    fidelities: np.ndarray | None = None,
    threshold: float = 2.0,
    fidelity_floor: float = 0.5,
) -> OutlierReport:
    """Detect outlier θ-points that would corrupt MPNN training.

    An outlier is a point where the VQE converged to a fundamentally wrong
    local minimum, producing θ values inconsistent with neighbors. These
    points appear as isolated spikes in the θ(h) curve.

    Detection uses two complementary criteria:
    1. **Neighbor deviation**: θ(h_i) deviates from the linear interpolation
       of its neighbors by more than `threshold` standard deviations.
    2. **Fidelity floor** (optional): F(h_i) < `fidelity_floor` when the
       neighbors have F >> fidelity_floor.

    Parameters
    ----------
    theta_array : np.ndarray [n_points, n_params]
        Optimized parameters for each h-point (already aligned).
    h_values : np.ndarray [n_points]
        Field values corresponding to each row of theta_array.
    fidelities : np.ndarray | None [n_points]
        VQE fidelities. If provided, used as secondary outlier signal.
    threshold : float
        Number of standard deviations for neighbor-deviation detection.
        Lower = more aggressive filtering. Default 2.0.
    fidelity_floor : float
        Points with fidelity below this AND neighbors above it are outliers.
        Only used if `fidelities` is provided. Default 0.5.

    Returns
    -------
    OutlierReport
        Report with indices and h-values of detected outliers.
    """
    n_points, n_params = theta_array.shape
    outlier_mask = np.zeros(n_points, dtype=bool)

    logger.info(
        "  🔍 detect_theta_outliers: %d points, %d params, threshold=%.1f, fidelity_floor=%.2f",
        n_points,
        n_params,
        threshold,
        fidelity_floor,
    )

    # Method 1: First-difference spike detection
    # An outlier θ_i creates large |θ_i - θ_{i-1}| AND |θ_{i+1} - θ_i|.
    # Normal smooth variation has one small diff per step. An outlier has
    # TWO large diffs flanking it.
    if n_points >= 3:
        # Compute L2 norm of first differences
        first_diffs = np.linalg.norm(np.diff(theta_array, axis=0), axis=1)

        # Robust baseline: median of first differences
        median_diff = np.median(first_diffs)
        mad_diff = np.median(np.abs(first_diffs - median_diff))
        sigma_diff = max(mad_diff * 1.4826, median_diff * 0.1)
        diff_cutoff = median_diff + threshold * sigma_diff

        # A point is an outlier if BOTH adjacent first-diffs exceed the cutoff
        for i in range(1, n_points - 1):
            left_diff = first_diffs[i - 1]  # |θ_i - θ_{i-1}|
            right_diff = first_diffs[i]  # |θ_{i+1} - θ_i|
            if left_diff > diff_cutoff and right_diff > diff_cutoff:
                outlier_mask[i] = True
    # Method 2: Fidelity-based detection (if available)
    if fidelities is not None and n_points >= 3:
        for i in range(1, n_points - 1):
            f_i = fidelities[i]
            f_neighbors = (fidelities[i - 1] + fidelities[i + 1]) / 2
            # Flag if this point has very low fidelity while neighbors are fine
            if f_i < fidelity_floor and f_neighbors > fidelity_floor + 0.1:
                outlier_mask[i] = True

    outlier_indices = list(np.where(outlier_mask)[0])
    outlier_h = [float(h_values[i]) for i in outlier_indices]

    report = OutlierReport(
        n_points=n_points,
        n_outliers=len(outlier_indices),
        outlier_indices=outlier_indices,
        outlier_h_values=outlier_h,
        method="neighbor_deviation+fidelity",
        threshold=threshold,
    )

    if outlier_indices:
        logger.info(
            f"Outlier detection: {len(outlier_indices)}/{n_points} outliers at h={outlier_h}"
        )
    else:
        logger.info("Outlier detection: no outliers found.")

    return report


def filter_theta_outliers(
    theta_array: np.ndarray,
    h_values: np.ndarray,
    e_exact: np.ndarray,
    fidelities: np.ndarray | None = None,
    threshold: float = 2.0,
    fidelity_floor: float = 0.5,
    replace_strategy: str = "interpolate",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, OutlierReport]:
    """Filter or replace outlier θ-points before MPNN training.

    Detects outliers using `detect_theta_outliers`, then either removes them
    from the arrays or replaces them with interpolated values.

    Parameters
    ----------
    theta_array : np.ndarray [n_points, n_params]
        Optimized parameters for each h-point.
    h_values : np.ndarray [n_points]
        Field values.
    e_exact : np.ndarray [n_points]
        Exact energies.
    fidelities : np.ndarray | None [n_points]
        VQE fidelities (optional).
    threshold : float
        Detection threshold (default 2.0).
    fidelity_floor : float
        Fidelity floor for detection (default 0.5).
    replace_strategy : str
        "remove" — drop outlier points entirely.
        "interpolate" — replace θ with linear interpolation from neighbors.
        Default: "interpolate" (preserves dataset size for MPNN).

    Returns
    -------
    (theta_clean, h_clean, e_exact_clean, fidelities_clean, report)
        Filtered/cleaned arrays and the outlier detection report.
    """
    report = detect_theta_outliers(theta_array, h_values, fidelities, threshold, fidelity_floor)

    logger.info(
        "  🧹 filter_theta_outliers: strategy=%s, %d outliers detected",
        replace_strategy,
        report.n_outliers,
    )

    if report.n_outliers == 0:
        return theta_array, h_values, e_exact, fidelities, report

    if replace_strategy == "remove":
        mask = np.ones(len(h_values), dtype=bool)
        mask[report.outlier_indices] = False
        theta_clean = theta_array[mask]
        h_clean = h_values[mask]
        e_clean = e_exact[mask]
        f_clean = fidelities[mask] if fidelities is not None else None
        logger.info(
            f"Outlier filter (remove): {report.n_outliers} points removed, "
            f"{len(h_clean)} remaining."
        )
        return theta_clean, h_clean, e_clean, f_clean, report

    elif replace_strategy == "interpolate":
        theta_clean = theta_array.copy()
        for idx in report.outlier_indices:
            if idx == 0:
                # First point: use next valid point
                theta_clean[idx] = theta_array[idx + 1]
            elif idx == len(h_values) - 1:
                # Last point: use previous valid point
                theta_clean[idx] = theta_array[idx - 1]
            else:
                # Interior: simple average of neighbors (robust to h-ordering)
                theta_clean[idx] = (theta_array[idx - 1] + theta_array[idx + 1]) / 2.0
        logger.info(
            f"Outlier filter (interpolate): {report.n_outliers} points "
            f"replaced with neighbor interpolation."
        )
        return theta_clean, h_values, e_exact, fidelities, report

    else:
        raise ValueError(
            f"Unknown replace_strategy='{replace_strategy}'. Use 'remove' or 'interpolate'."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-h Energy Validation Guard
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EnergyGuardReport:
    """Summary of the cross-h energy validation pass.

    Attributes
    ----------
    n_points : int
        Total number of h-points.
    n_suspicious : int
        Points flagged as likely local minima.
    n_repaired : int
        Points successfully repaired (lower energy found).
    suspicious_indices : list[int]
        Indices of suspicious points.
    energy_improvements : list[float]
        Energy improvement at each repaired point (negative = better).
    """

    n_points: int
    n_suspicious: int
    n_repaired: int
    suspicious_indices: list[int] = field(default_factory=list)
    energy_improvements: list[float] = field(default_factory=list)


def cross_h_energy_guard(
    vqe_energies: np.ndarray,
    exact_energies: np.ndarray,
    gaps: np.ndarray,
    theta_array: np.ndarray,
    h_values: np.ndarray,
    reoptimize_fn: Callable[[int, np.ndarray], tuple[float, np.ndarray]] | None = None,
    de_gap_threshold: float = 0.05,
    neighbor_ratio: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, EnergyGuardReport]:
    """Detect and repair VQE points that are likely stuck in local minima.

    A point is suspicious if its ΔE/gap is significantly worse than both
    neighbors. This indicates the warm-start propagation failed at that
    point, causing VQE to settle in a bad local minimum.

    For detected suspicious points, attempts repair by re-optimizing with
    the better neighbor's θ as warm-start seed (via `reoptimize_fn`).

    Parameters
    ----------
    vqe_energies : np.ndarray [n_points]
        VQE energies from the sweep.
    exact_energies : np.ndarray [n_points]
        Exact ground state energies.
    gaps : np.ndarray [n_points]
        Spectral gaps at each h-point.
    theta_array : np.ndarray [n_points, n_params]
        VQE parameters from the sweep.
    h_values : np.ndarray [n_points]
        Field values.
    reoptimize_fn : callable | None
        Function `(index, theta_seed) -> (energy, theta_opt)` that re-runs
        VQE at h_values[index] using theta_seed as initial guess.
        If None, only detection is performed (no repair).
    de_gap_threshold : float
        ΔE/gap threshold below which a point is considered good (default 0.05).
    neighbor_ratio : float
        A point is suspicious if its ΔE/gap exceeds `neighbor_ratio` times
        the average ΔE/gap of its neighbors (default 3.0).

    Returns
    -------
    (energies_out, theta_out, report)
        Possibly repaired arrays and the guard report.
    """
    n_points = len(vqe_energies)
    de_gaps = np.abs(vqe_energies - exact_energies) / np.maximum(gaps, 1e-10)

    logger.info(
        "  🛡️ cross_h_energy_guard: %d points, de_gap_threshold=%.2f, neighbor_ratio=%.1f",
        n_points,
        de_gap_threshold,
        neighbor_ratio,
    )

    suspicious: list[int] = []

    for i in range(1, n_points - 1):
        de_i = de_gaps[i]
        de_left = de_gaps[i - 1]
        de_right = de_gaps[i + 1]
        neighbor_avg = (de_left + de_right) / 2.0

        # Suspicious: this point is much worse than both neighbors
        if (
            de_i > de_gap_threshold and de_left < de_gap_threshold and de_right < de_gap_threshold
        ) or (
            de_i > de_gap_threshold and neighbor_avg > 0 and de_i > neighbor_ratio * neighbor_avg
        ):
            suspicious.append(i)

    if suspicious:
        logger.info(
            f"Energy guard: {len(suspicious)} suspicious points at indices "
            f"{suspicious} (h={[float(h_values[i]) for i in suspicious]})"
        )

    # Attempt repair
    energies_out = vqe_energies.copy()
    theta_out = theta_array.copy()
    n_repaired = 0
    improvements: list[float] = []

    if reoptimize_fn is not None and suspicious:
        for idx in suspicious:
            # Use the better neighbor's θ as seed
            if de_gaps[idx - 1] <= de_gaps[idx + 1]:
                seed_theta = theta_array[idx - 1]
            else:
                seed_theta = theta_array[idx + 1]

            try:
                new_energy, new_theta = reoptimize_fn(idx, seed_theta)
                improvement = new_energy - vqe_energies[idx]
                if new_energy < vqe_energies[idx]:
                    energies_out[idx] = new_energy
                    theta_out[idx] = new_theta
                    n_repaired += 1
                    improvements.append(improvement)
                    logger.info(
                        f"  idx={idx} h={h_values[idx]:.3f}: REPAIRED (ΔE={improvement:.4e})"
                    )
                else:
                    logger.info(
                        f"  idx={idx} h={h_values[idx]:.3f}: kept original "
                        f"(reopt worse by {improvement:.4e})"
                    )
            except Exception as e:
                logger.warning(f"  idx={idx} h={h_values[idx]:.3f}: reopt failed ({e})")
    elif suspicious and reoptimize_fn is None:
        logger.info("  Energy guard: detection only (no reoptimize_fn provided).")

    report = EnergyGuardReport(
        n_points=n_points,
        n_suspicious=len(suspicious),
        n_repaired=n_repaired,
        suspicious_indices=suspicious,
        energy_improvements=improvements,
    )

    return energies_out, theta_out, report
