#!/usr/bin/env python
"""QPT Detection via Energy Derivatives at Large N.

Loads E(h) data from multi_n_training and large_n_extrapolation NPZ files,
computes d^2E/dh^2 numerically, identifies h_c(N), and performs
finite-size scaling analysis.

This validates that the GNN+HVA pipeline captures the physics of the
quantum phase transition (h_c ~ 1.0 for TFIM chain_1d).

NOTE: This is NOT a claim of quantum advantage. DMRG calculates E(h) for
chain_1d N=1000 in minutes. The value is in demonstrating that the MPNN
correctly predicts h_c — a consistency check of the learned physics.

Usage:
    # Using exact ground truth energies
    python scripts/analysis/qpt_detection.py --topology chain_1d --save

    # Using MPNN-predicted energies (tests if GNN captures QPT)
    python scripts/analysis/qpt_detection.py --topology chain_1d --use-predicted --save

    # Compare both (key thesis result)
    python scripts/analysis/qpt_detection.py --topology chain_1d --compare --save

    # Restrict h range
    python scripts/analysis/qpt_detection.py --topology chain_1d --h-range 0.5 3.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Ensure project root on path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_energy_curves(
    topology: str,
    p_layers: int = 1,
    use_predicted: bool = False,
    *,
    min_points: int = 5,
) -> dict[int, dict]:
    """Load E(h) curves for all available N.

    Scans both multi_n_training and large_n_extrapolation directories for NPZ
    files matching the topology and p_layers. Prefers multi_n_training (denser
    h-grid) when both exist for the same N.

    Parameters
    ----------
    topology : str
        Lattice topology (e.g., "chain_1d", "heavy_hex").
    p_layers : int
        HVA circuit depth. Default: 1.
    use_predicted : bool
        If True, use e_vqe (MPNN-predicted / VQE-optimized energy).
        If False, use e_exact (ground truth from ED/DMRG).
    min_points : int
        Minimum number of h-points required to include a curve.

    Returns
    -------
    dict[int, dict]
        {N: {"h": ndarray, "E": ndarray, "gap": ndarray, "source": str}}
    """
    data_root = _project_root / "data"
    results: dict[int, dict] = {}

    # Scan multi_n_training first (denser grids, priority)
    training_dir = data_root / "multi_n_training"
    extrap_dir = data_root / "large_n_extrapolation"

    pattern = f"{topology}_N*_p{p_layers}.npz"

    for search_dir, source_label in [
        (training_dir, "multi_n_training"),
        (extrap_dir, "large_n_extrapolation"),
    ]:
        if not search_dir.exists():
            continue
        for npz_file in sorted(search_dir.glob(pattern)):
            if "_baselines" in str(npz_file):
                continue
            # Extract N from filename
            stem = npz_file.stem
            try:
                n_str = stem.split("_N")[1].split("_p")[0]
                n = int(n_str)
            except (IndexError, ValueError):
                continue

            # Skip if already loaded from higher-priority source
            if n in results:
                continue

            try:
                data = np.load(npz_file, allow_pickle=True)
                h_values = np.asarray(data["h_values"], dtype=float)

                # Select energy source
                if use_predicted:
                    if "e_vqe" in data:
                        energies = np.asarray(data["e_vqe"], dtype=float)
                    else:
                        continue  # No predicted energies
                else:
                    if "e_exact" in data:
                        energies = np.asarray(data["e_exact"], dtype=float)
                    else:
                        continue  # No exact energies

                gaps = np.asarray(data.get("gaps", np.zeros_like(h_values)), dtype=float)

                # Filter out NaN/inf
                valid_mask = np.isfinite(energies) & np.isfinite(h_values)
                h_values = h_values[valid_mask]
                energies = energies[valid_mask]
                gaps = gaps[valid_mask] if len(gaps) == len(valid_mask) else np.zeros_like(h_values)

                if len(h_values) >= min_points:
                    # Sort by h for derivative computation
                    sort_idx = np.argsort(h_values)
                    results[n] = {
                        "h": h_values[sort_idx],
                        "E": energies[sort_idx],
                        "gap": gaps[sort_idx],
                        "source": source_label,
                        "file": str(npz_file.name),
                    }
            except Exception as e:
                logger.debug(f"  Failed to load {npz_file.name}: {e}")
                continue

    # ── Fallback: supplement with GroundTruthCache for N not in NPZ ──────
    # The GT cache may have entries (especially for large N computed by DMRG)
    # that haven't been aggregated into NPZ files yet.
    if not use_predicted:
        try:
            from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

            gt_cache = GroundTruthCache()
            # Group GT entries by N for this topology
            from collections import defaultdict

            gt_by_n: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
            prefix = f"{topology}|"
            for key, entry in gt_cache._data.items():
                if not key.startswith(prefix):
                    continue
                parts = key.split("|")
                if len(parts) < 4:
                    continue
                try:
                    n_gt = int(parts[1])
                    model_gt = parts[2]
                except (ValueError, IndexError):
                    continue
                # Only include matching models (tfim or tfim_bond_resolved)
                if model_gt not in ("tfim", "tfim_bond_resolved"):
                    continue
                # Enrich: add GT data even if NPZ exists (NPZ may lack low-h points)
                h_gt = float(parts[3])
                e_gt = float(entry["energy"])
                gap_gt = float(entry.get("gap", 0.0))
                if np.isfinite(e_gt) and np.isfinite(h_gt):
                    gt_by_n[n_gt].append((h_gt, e_gt, gap_gt))

            for n_gt, data_list in gt_by_n.items():
                if len(data_list) < min_points:
                    continue
                if n_gt in results:
                    # Merge GT data into existing NPZ curve (adds low-h points)
                    existing_h = set(round(hv, 2) for hv in results[n_gt]["h"])
                    new_points = [
                        (h, e, g) for h, e, g in data_list if round(h, 2) not in existing_h
                    ]
                    if new_points:
                        h_new = np.array([p[0] for p in new_points])
                        e_new = np.array([p[1] for p in new_points])
                        gap_new = np.array([p[2] for p in new_points])
                        results[n_gt]["h"] = np.concatenate([results[n_gt]["h"], h_new])
                        results[n_gt]["E"] = np.concatenate([results[n_gt]["E"], e_new])
                        results[n_gt]["gap"] = np.concatenate([results[n_gt]["gap"], gap_new])
                        sort_idx = np.argsort(results[n_gt]["h"])
                        results[n_gt]["h"] = results[n_gt]["h"][sort_idx]
                        results[n_gt]["E"] = results[n_gt]["E"][sort_idx]
                        results[n_gt]["gap"] = results[n_gt]["gap"][sort_idx]
                else:
                    # New N from GT cache only
                    data_list.sort(key=lambda x: x[0])
                    h_arr = np.array([d[0] for d in data_list])
                    e_arr = np.array([d[1] for d in data_list])
                    gap_arr = np.array([d[2] for d in data_list])
                    results[n_gt] = {
                        "h": h_arr,
                        "E": e_arr,
                        "gap": gap_arr,
                        "source": "ground_truth_cache",
                        "file": "ground_truth_cache.json",
                    }
        except (ImportError, OSError) as e:
            logger.debug(f"  GT cache fallback unavailable: {e}")

    # ── Harvest E(h) data points from DQPT trajectories ─────────────────────
    # Each DQPT run stores e_exact_h_pre (ground state energy at the pre-quench field).
    # By scanning multiple DQPT runs with different h_pre, we get free E(h) data points
    # that can supplement sparse QPT coverage (especially for N=12-18 heavy_hex).
    dqpt_dir = _project_root / "data" / "dqpt_trajectories"
    if dqpt_dir.exists():
        from collections import defaultdict

        dqpt_by_n: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
        for npz_file in dqpt_dir.glob(f"{topology}_N*.npz"):
            try:
                data = np.load(npz_file, allow_pickle=True)
                if "e_exact_h_pre" not in data:
                    continue
                n_dqpt = int(data["n_qubits"])
                h_val = float(data["h_pre"])
                e_val = float(data["e_exact_h_pre"])
                gap_val = float(data["gap_h_pre"]) if "gap_h_pre" in data else 0.0
                if np.isfinite(e_val) and np.isfinite(h_val):
                    dqpt_by_n[n_dqpt].append((h_val, e_val, gap_val))
            except Exception:
                continue

        for n_dqpt, data_list in dqpt_by_n.items():
            if n_dqpt in results:
                # Merge: add DQPT points that aren't already in the existing curve
                existing_h = set(round(hv, 2) for hv in results[n_dqpt]["h"])
                new_points = [(h, e, g) for h, e, g in data_list if round(h, 2) not in existing_h]
                if new_points:
                    h_new = np.array([p[0] for p in new_points])
                    e_new = np.array([p[1] for p in new_points])
                    gap_new = np.array([p[2] for p in new_points])
                    results[n_dqpt]["h"] = np.concatenate([results[n_dqpt]["h"], h_new])
                    results[n_dqpt]["E"] = np.concatenate([results[n_dqpt]["E"], e_new])
                    results[n_dqpt]["gap"] = np.concatenate([results[n_dqpt]["gap"], gap_new])
                    # Re-sort by h
                    sort_idx = np.argsort(results[n_dqpt]["h"])
                    results[n_dqpt]["h"] = results[n_dqpt]["h"][sort_idx]
                    results[n_dqpt]["E"] = results[n_dqpt]["E"][sort_idx]
                    results[n_dqpt]["gap"] = results[n_dqpt]["gap"][sort_idx]
            elif len(data_list) >= min_points:
                # New N from DQPT data alone
                data_list.sort(key=lambda x: x[0])
                results[n_dqpt] = {
                    "h": np.array([d[0] for d in data_list]),
                    "E": np.array([d[1] for d in data_list]),
                    "gap": np.array([d[2] for d in data_list]),
                    "source": "dqpt_trajectories",
                    "file": "data/dqpt_trajectories/",
                }

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Public API: h_c Lookup (for integration with quality_profile, retrain triggers)
# ═══════════════════════════════════════════════════════════════════════════════


def get_h_critical(
    topology: str,
    n_qubits: int | None = None,
    p_layers: int = 1,
) -> float | None:
    """Get the detected critical field h_c for a topology.

    This is the main integration point for other modules that need h_c
    (e.g., compute_quality_profile for regional breakdown, retrain triggers).

    Uses cached results from `results/analysis/qpt_detection_{topology}_exact.json`
    if available, otherwise runs a quick detection on available data.

    Parameters
    ----------
    topology : str
        Lattice topology.
    n_qubits : int | None
        If provided, returns h_c for that specific N.
        If None, returns h_c(inf) from FSS (or best available N).
    p_layers : int
        HVA depth.

    Returns
    -------
    float | None
        Critical field value, or None if insufficient data.
    """
    # Try cached result first (fast path)
    cache_path = _project_root / "results" / "analysis" / f"qpt_detection_{topology}_exact.json"
    if cache_path.exists():
        try:
            import json

            with open(cache_path) as f:
                cached = json.load(f)

            if n_qubits is not None:
                # Return h_c for specific N
                h_c_by_n = cached.get("h_c_by_n", cached.get("h_c_reliable", {}))
                val = h_c_by_n.get(str(n_qubits))
                return float(val) if val is not None else None
            else:
                hc = _h_c_inf_from_fss(cached.get("finite_size_scaling"))
                if hc is not None:
                    return hc
                # Fallback: largest reliable N
                h_c_rel = cached.get("h_c_reliable", {})
                if h_c_rel:
                    largest_n = max(h_c_rel.keys(), key=lambda x: int(x))
                    return float(h_c_rel[largest_n])
        except Exception:
            pass

    # Fallback: quick computation (only if data exists)
    try:
        result = run_qpt_analysis(topology, p_layers, use_predicted=False)
        if "error" in result:
            return None

        if n_qubits is not None:
            h_c_by_n = result.get("h_c_reliable", result.get("h_c_by_n", {}))
            val = h_c_by_n.get(str(n_qubits))
            return float(val) if val is not None else None

        hc = _h_c_inf_from_fss(result.get("finite_size_scaling"))
        if hc is not None:
            return hc
        h_c_rel = result.get("h_c_reliable", {})
        if h_c_rel:
            largest_n = max(h_c_rel.keys(), key=lambda x: int(x))
            return float(h_c_rel[largest_n])
    except Exception:
        pass

    return None


def _h_c_inf_from_fss(fss: dict | None) -> float | None:
    """Extract the best h_c(inf) estimate from a finite_size_scaling result.

    Prefers the robust median (h_c_inf_robust) when the curve_fit was flagged
    unreliable (flat h_c across N or undefined covariance), otherwise uses the
    fitted h_c_inf. Returns None if the FSS dict has no usable estimate.
    """
    if not fss:
        return None
    # If the fit was explicitly marked unreliable, prefer the robust median.
    if fss.get("fit_reliable") is False and fss.get("h_c_inf_robust") is not None:
        return float(fss["h_c_inf_robust"])
    if "h_c_inf" in fss and fss["h_c_inf"] is not None:
        return float(fss["h_c_inf"])
    if fss.get("h_c_inf_robust") is not None:
        return float(fss["h_c_inf_robust"])
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Derivative Computation
# ═══════════════════════════════════════════════════════════════════════════════

# Default QPT-relevant h-range per model.
# For TFIM variants, the QPT occurs at h_c ~ 1.0 (1D) to ~3.0 (2D square).
# Restrict to h < 2.0 for TFIM since:
#   1. h_c is always < 2.0 for all finite-coordination lattices
#   2. Non-uniform VQE training data creates d2E artifacts at h > 2.0
#   3. The energy curve E(h) is provably smooth and monotone for h in [0, 2.0]
_QPT_SEARCH_RANGE: dict[str, tuple[float, float]] = {
    "tfim": (0.3, 2.0),
    "tfim_bond_resolved": (0.3, 2.0),
    "default": (0.3, 3.5),
}


def compute_second_derivative(
    h_values: np.ndarray,
    energies: np.ndarray,
    *,
    n_interp: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute d^2E/dh^2 via finite differences on a uniform grid.

    When h-values are non-uniformly spaced (common in VQE training data),
    np.gradient introduces artifacts proportional to 1/(Δh)^2. To avoid this,
    we interpolate onto a uniform grid before differentiating.

    Parameters
    ----------
    h_values : np.ndarray
        Field values (may be non-uniformly spaced).
    energies : np.ndarray
        Corresponding energies.
    n_interp : int | None
        Number of uniformly-spaced points for interpolation.
        Default: min(200, 2*len(h_values)) to avoid under-sampling.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (h_uniform, d2E) on the uniform grid.
    """
    if n_interp is None:
        n_interp = min(200, max(len(h_values), 2 * len(h_values)))

    # Guard: need at least 3 points for second derivative
    if len(h_values) < 3:
        return h_values, np.zeros_like(h_values)

    # Guard: degenerate case (all h identical or range too small)
    h_span = h_values.max() - h_values.min()
    if h_span < 1e-10:
        return h_values, np.zeros_like(h_values)

    # Check if already approximately uniform
    dh = np.diff(h_values)
    if len(dh) > 0 and (dh.max() / max(dh.min(), 1e-10)) < 1.5:
        # Approximately uniform — use directly
        dE = np.gradient(energies, h_values)
        d2E = np.gradient(dE, h_values)
        return h_values, d2E

    # Non-uniform: interpolate to uniform grid
    h_uniform = np.linspace(h_values.min(), h_values.max(), n_interp)
    E_uniform = np.interp(h_uniform, h_values, energies)

    dE = np.gradient(E_uniform, h_uniform)
    d2E = np.gradient(dE, h_uniform)
    return h_uniform, d2E


def find_critical_field(
    h_values: np.ndarray,
    d2E: np.ndarray,
    *,
    h_min_search: float | None = None,
    h_max_search: float | None = None,
) -> tuple[float, float]:
    """Identify h_c as the position of the peak (maximum |d2E/dh2|).

    For the TFIM, d2E/dh2 has a pronounced minimum (most negative) at h_c,
    corresponding to maximum susceptibility / sharpest energy curvature.

    Parameters
    ----------
    h_values : np.ndarray
        Field values.
    d2E : np.ndarray
        Second derivative of energy.
    h_min_search : float | None
        Restrict search to h > h_min_search (avoids edge artifacts).
    h_max_search : float | None
        Restrict search to h < h_max_search.

    Returns
    -------
    tuple[float, float]
        (h_c, peak_magnitude) — critical field and the absolute value of d2E at that point.
    """
    mask = np.ones(len(h_values), dtype=bool)
    if h_min_search is not None:
        mask &= h_values >= h_min_search
    if h_max_search is not None:
        mask &= h_values <= h_max_search

    # Exclude edge points (2 on each side) to avoid gradient boundary effects
    if len(h_values) > 6:
        mask[:2] = False
        mask[-2:] = False

    h_masked = h_values[mask]
    d2E_masked = d2E[mask]

    if len(h_masked) == 0:
        return float("nan"), 0.0

    # The QPT signature is the MINIMUM of d2E (most negative second derivative)
    # This corresponds to the inflection point of E(h) — maximum curvature
    idx_min = np.argmin(d2E_masked)
    h_c = float(h_masked[idx_min])
    peak_magnitude = float(abs(d2E_masked[idx_min]))

    return h_c, peak_magnitude


# ═══════════════════════════════════════════════════════════════════════════════
# Finite-Size Scaling
# ═══════════════════════════════════════════════════════════════════════════════


def finite_size_scaling(h_c_by_n: dict[int, float]) -> dict:
    """Fit h_c(N) = h_c(inf) + a/N^nu to extract thermodynamic h_c.

    Includes IQR-based outlier rejection: points where h_c deviates more
    than 1.5*IQR from the median are excluded before fitting. This handles
    cases where a single N has corrupted data (e.g., N=10 giving h_c=1.6
    while all others give ~0.9).

    Parameters
    ----------
    h_c_by_n : dict[int, float]
        {N: h_c(N)} measured at each system size.

    Returns
    -------
    dict
        Scaling analysis results with h_c_inf, a, nu, r_squared, etc.
    """
    from scipy.optimize import curve_fit

    N_values = np.array(sorted(h_c_by_n.keys()), dtype=float)
    h_c_values = np.array([h_c_by_n[int(n)] for n in N_values])

    if len(N_values) < 3:
        return {
            "error": f"Need at least 3 N-values for fit, got {len(N_values)}",
            "N_values": N_values.tolist(),
            "h_c_values": h_c_values.tolist(),
        }

    # IQR outlier rejection: remove h_c values > 1.5*IQR from Q1/Q3
    q1, q3 = np.percentile(h_c_values, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        inlier_mask = (h_c_values >= lower) & (h_c_values <= upper)
        n_outliers = int((~inlier_mask).sum())
        if n_outliers > 0 and inlier_mask.sum() >= 3:
            N_values = N_values[inlier_mask]
            h_c_values = h_c_values[inlier_mask]
    else:
        n_outliers = 0

    # After outlier removal, re-check minimum
    if len(N_values) < 3:
        return {
            "error": f"Need at least 3 N-values for fit, got {len(N_values)}",
            "N_values": N_values.tolist(),
            "h_c_values": h_c_values.tolist(),
        }

    def scaling_law(N, h_inf, a, nu):
        return h_inf + a / N**nu

    # Guard: the 3-parameter law h_inf + a/N^nu needs at least 3 distinct
    # N-values with variation in h_c to be identifiable. If h_c is flat
    # (e.g., h_c converges fast — good physics), curve_fit cannot estimate
    # the covariance and emits OptimizeWarning. We detect this and downgrade
    # the fit to a robust median estimate rather than reporting garbage errors.
    import warnings

    from scipy.optimize import OptimizeWarning

    h_c_spread = float(h_c_values.max() - h_c_values.min())

    try:
        with warnings.catch_warnings():
            # OptimizeWarning ("Covariance could not be estimated") is expected
            # for flat/degenerate data — handle it explicitly below instead of
            # letting it print scary noise during valid runs.
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(
                scaling_law,
                N_values,
                h_c_values,
                p0=[1.0, 1.0, 1.0],  # Initial guess (TFIM: h_c=1, nu=1)
                bounds=([0.0, -10, 0.1], [5.0, 10, 3.0]),
                maxfev=10000,
            )
        h_inf, a, nu = popt
        perr = np.sqrt(np.diag(pcov))

        # Detect degenerate fit: non-finite covariance means the parameters
        # were not identifiable (flat data or too few effective points).
        cov_undefined = not np.all(np.isfinite(perr))

        # R^2
        residuals = h_c_values - scaling_law(N_values, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((h_c_values - np.mean(h_c_values)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # A fit is reliable only if covariance is defined AND h_c actually
        # varies with N. Otherwise the median is a more honest estimate.
        fit_reliable = bool(cov_undefined is False and h_c_spread > 1e-3)

        result = {
            "h_c_inf": float(h_inf),
            "h_c_inf_err": float(perr[0]) if np.isfinite(perr[0]) else None,
            "a": float(a),
            "nu": float(nu),
            "nu_err": float(perr[2]) if np.isfinite(perr[2]) else None,
            "r_squared": float(r_squared),
            "fit_reliable": fit_reliable,
            "cov_undefined": bool(cov_undefined),
            "h_c_spread": h_c_spread,
            "N_values": N_values.tolist(),
            "h_c_values": h_c_values.tolist(),
            "fit_residuals": residuals.tolist(),
            "n_outliers_removed": n_outliers,
        }

        # When the fit is unreliable, provide a robust fallback so downstream
        # code (get_h_critical) has a sensible h_c(inf) instead of a spurious
        # extrapolation to the parameter bound.
        if not fit_reliable:
            robust_hc = float(np.median(h_c_values))
            result["h_c_inf_robust"] = robust_hc
            result["fit_note"] = (
                "Fit unreliable (covariance undefined or h_c ~ flat across N). "
                "h_c converges fast with N — use h_c_inf_robust (median) instead."
            )
            logger.debug(
                "FSS fit unreliable (spread=%.4f, cov_undefined=%s). Robust h_c=%.4f",
                h_c_spread,
                cov_undefined,
                robust_hc,
            )

        return result
    except Exception as e:
        # Fall back to a robust median estimate rather than a hard error, so
        # downstream consumers still get a usable h_c(inf).
        return {
            "error": str(e),
            "h_c_inf_robust": float(np.median(h_c_values)),
            "fit_reliable": False,
            "N_values": N_values.tolist(),
            "h_c_values": h_c_values.tolist(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def run_qpt_analysis(
    topology: str,
    p_layers: int = 1,
    use_predicted: bool = False,
    h_range: tuple[float, float] | None = None,
    *,
    min_points_for_detection: int = 8,
) -> dict:
    """Run full QPT detection analysis for a given topology.

    Parameters
    ----------
    topology : str
        Lattice topology.
    p_layers : int
        HVA depth.
    use_predicted : bool
        Use MPNN-predicted energies vs exact.
    h_range : tuple | None
        Restrict h range for analysis. If None, uses the model-appropriate
        QPT search range (h < 2.5 for TFIM variants) to avoid VQE artifacts.
    min_points_for_detection : int
        Minimum number of h-points required in the search range.

    Returns
    -------
    dict
        Full analysis results.
    """
    source_label = "MPNN-predicted" if use_predicted else "exact (GT)"

    # Load data
    curves = load_energy_curves(topology, p_layers, use_predicted)
    if not curves:
        return {"error": f"No data found for {topology} p={p_layers}", "source": source_label}

    # Determine effective h-range for QPT search
    effective_range = h_range or _QPT_SEARCH_RANGE.get("tfim_bond_resolved", (0.3, 3.0))

    # Compute derivatives and detect h_c for each N
    h_c_by_n: dict[int, float] = {}
    per_n_results: dict[int, dict] = {}

    for n in sorted(curves.keys()):
        data = curves[n]
        h, E = data["h"].copy(), data["E"].copy()

        # Filter to QPT-relevant range
        mask = (h >= effective_range[0]) & (h <= effective_range[1])
        h, E = h[mask], E[mask]

        if len(h) < min_points_for_detection:
            continue

        # Compute second derivative (with uniform interpolation)
        h_d2, d2E = compute_second_derivative(h, E)

        # Find critical field (exclude edge artifacts)
        h_c, peak_mag = find_critical_field(
            h_d2,
            d2E,
            h_min_search=effective_range[0] + 0.1,
            h_max_search=effective_range[1] - 0.1,
        )

        # Quality filter: reject if h_c is within 10% of data boundary
        # (indicates insufficient coverage rather than real QPT detection)
        # Also reject if data range doesn't cover at least [h_c - 0.3, h_c + 0.3]
        # (need context on BOTH sides of the QPT to confirm it's real)
        if not np.isnan(h_c):
            h_data_min, h_data_max = float(h.min()), float(h.max())
            data_span = h_data_max - h_data_min

            # Check 1: h_c near data boundary
            near_lower = (h_c - h_data_min) < 0.15 * data_span
            near_upper = (h_data_max - h_c) < 0.15 * data_span

            # Check 2: insufficient data range around h_c
            # We need at least 0.3 of h-range on each side of h_c
            insufficient_below = (h_c - h_data_min) < 0.3
            insufficient_above = (h_data_max - h_c) < 0.3

            is_edge_artifact = near_lower or near_upper or insufficient_below or insufficient_above

            h_c_by_n[n] = h_c
            per_n_results[n] = {
                "h_c": h_c,
                "peak_magnitude": peak_mag,
                "n_points": len(h),
                "h_range": [h_data_min, h_data_max],
                "source": data["source"],
                "edge_artifact": is_edge_artifact,
            }

    # Finite-size scaling (exclude edge artifacts for reliable fit)
    h_c_clean = {
        n: h_c_by_n[n] for n in h_c_by_n if not per_n_results.get(n, {}).get("edge_artifact", False)
    }
    fss = finite_size_scaling(h_c_clean) if len(h_c_clean) >= 3 else None

    return {
        "topology": topology,
        "p_layers": p_layers,
        "source": source_label,
        "n_values_analyzed": sorted(h_c_by_n.keys()),
        "n_values_reliable": sorted(h_c_clean.keys()),
        "h_c_by_n": {str(k): v for k, v in h_c_by_n.items()},
        "h_c_reliable": {str(k): v for k, v in h_c_clean.items()},
        "per_n_results": {str(k): v for k, v in per_n_results.items()},
        "finite_size_scaling": fss,
        "h_range_used": list(effective_range),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="QPT Detection via Energy Derivatives — "
        "validates that the pipeline captures h_c correctly"
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="chain_1d",
        help="Lattice topology (default: chain_1d)",
    )
    parser.add_argument(
        "--p-layers",
        type=int,
        default=1,
        help="HVA circuit depth (default: 1)",
    )
    parser.add_argument(
        "--use-predicted",
        action="store_true",
        help="Use MPNN-predicted energies instead of exact GT",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both exact and predicted, show comparison",
    )
    parser.add_argument(
        "--h-range",
        type=float,
        nargs=2,
        default=None,
        help="Restrict h range for analysis (e.g., 0.5 3.0)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    h_range = tuple(args.h_range) if args.h_range else None

    if args.compare:
        # Run both exact and predicted
        print(f"\n{'=' * 60}")
        print(f"QPT Detection: {args.topology} — COMPARISON (GT vs MPNN)")
        print(f"{'=' * 60}")

        result_exact = run_qpt_analysis(args.topology, args.p_layers, False, h_range)
        result_pred = run_qpt_analysis(args.topology, args.p_layers, True, h_range)

        _print_analysis(result_exact, "EXACT (Ground Truth)")
        _print_analysis(result_pred, "MPNN-PREDICTED")

        # Comparison
        hc_exact = result_exact.get("h_c_by_n", {})
        hc_pred = result_pred.get("h_c_by_n", {})
        common_n = sorted(set(hc_exact.keys()) & set(hc_pred.keys()))

        if common_n:
            print(f"\n{'─' * 60}")
            print("COMPARISON: h_c(exact) vs h_c(predicted)")
            print(f"{'─' * 60}")
            print(
                f"{'N':>4} | {'h_c(exact)':>10} | {'h_c(pred)':>10} | {'Δh_c':>8} | {'|Δ|/h_c':>8}"
            )
            print("-" * 55)
            for n_str in common_n:
                hc_e = hc_exact[n_str]
                hc_p = hc_pred[n_str]
                delta = hc_p - hc_e
                rel = abs(delta) / hc_e if hc_e > 0 else float("inf")
                print(f"{n_str:>4} | {hc_e:>10.4f} | {hc_p:>10.4f} | {delta:>+8.4f} | {rel:>8.2%}")

            mean_rel_error = np.mean(
                [abs(hc_pred[n] - hc_exact[n]) / hc_exact[n] for n in common_n if hc_exact[n] > 0]
            )
            print(f"\n  Mean |Δh_c|/h_c = {mean_rel_error:.2%}")
            captures_qpt = mean_rel_error < 0.05
            print(
                f"  MPNN captures QPT: {'YES' if captures_qpt else 'NO'} "
                f"(threshold: <5% relative error)"
            )

        if args.save:
            output = {
                "topology": args.topology,
                "comparison": {
                    "exact": result_exact,
                    "predicted": result_pred,
                    "mean_relative_error": float(mean_rel_error) if common_n else None,
                    "captures_qpt": captures_qpt if common_n else None,
                },
            }
            _save_results(output, args.topology, "comparison")

    else:
        # Single run
        result = run_qpt_analysis(args.topology, args.p_layers, args.use_predicted, h_range)
        source = "MPNN-predicted" if args.use_predicted else "exact (GT)"

        print(f"\n{'=' * 60}")
        print(f"QPT Detection: {args.topology} (source: {source})")
        print(f"{'=' * 60}")

        _print_analysis(result, source)

        if args.save:
            _save_results(result, args.topology, "predicted" if args.use_predicted else "exact")


def _print_analysis(result: dict, label: str) -> None:
    """Pretty-print QPT analysis results."""
    if "error" in result:
        print(f"\n  ERROR: {result['error']}")
        return

    print(f"\n── {label} ──")
    print(f"  N values analyzed: {result['n_values_analyzed']}")

    hc = result.get("h_c_by_n", {})
    per_n = result.get("per_n_results", {})

    if hc:
        print(f"\n  {'N':>4} | {'h_c':>6} | {'|d²E/dh²|_max':>14} | {'n_pts':>6} | {'source':>20}")
        print("  " + "-" * 65)
        for n_str in sorted(hc.keys(), key=lambda x: int(x)):
            info = per_n.get(n_str, {})
            print(
                f"  {n_str:>4} | {hc[n_str]:>6.3f} | "
                f"{info.get('peak_magnitude', 0):>14.4f} | "
                f"{info.get('n_points', 0):>6} | "
                f"{info.get('source', ''):>20}"
            )

    fss = result.get("finite_size_scaling")
    if fss:
        print("\n  ── Finite-Size Scaling ──")
        if "error" in fss:
            print(f"  Fit failed: {fss['error']}")
        else:
            print(f"  h_c(∞) = {fss['h_c_inf']:.4f} ± {fss['h_c_inf_err']:.4f}")
            print(f"  Exponent ν = {fss['nu']:.3f} ± {fss['nu_err']:.3f}")
            print(f"  R² = {fss['r_squared']:.4f}")
            # For TFIM chain_1d, exact h_c = 1.0
            if result.get("topology") == "chain_1d":
                error_from_exact = abs(fss["h_c_inf"] - 1.0)
                print(
                    f"  Error from exact (h_c=1.0): {error_from_exact:.4f} "
                    f"({error_from_exact * 100:.1f}%)"
                )


def _save_results(output: dict, topology: str, suffix: str) -> None:
    """Save results to JSON file."""
    from qmbp_simulation.utils.helpers import json_serialize

    out_dir = _project_root / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"qpt_detection_{topology}_{suffix}.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=json_serialize)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
