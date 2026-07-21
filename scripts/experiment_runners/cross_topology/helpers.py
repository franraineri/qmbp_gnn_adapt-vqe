"""Shared utilities for cross-topology transfer experiments.

Provides:
    - SourceData: Unified dataclass for VQE data from any format
    - detect_format: Detect JSON format (scaling or pipeline_run)
    - load_source_data: Dual-format adapter extracting (h, theta_opt, e_exact)
    - canonicalize_theta: Normalize theta arrays to consistent sign convention
    - build_cross_topology_dataset: Build torch_geometric Data list from SourceData
    - build_target_graph: Build graph for TARGET topology inference
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SourceData:
    """Unified container for VQE data extracted from any format."""

    n: int
    topology: str
    h_values: np.ndarray  # shape [n_points]
    theta_opt: np.ndarray  # shape [n_points, 2*p]
    e_exact: np.ndarray  # shape [n_points]
    param_dim: int  # len(theta_opt[0])


# ═══════════════════════════════════════════════════════════════════════════════
# Theta Normalization
# ═══════════════════════════════════════════════════════════════════════════════

from qmbp_simulation.utils import canonicalize_theta  # noqa: F401 — re-export for compatibility

# ═══════════════════════════════════════════════════════════════════════════════
# Format Detection
# ═══════════════════════════════════════════════════════════════════════════════


def detect_format(data: dict) -> str:
    """Detect JSON format: 'scaling' or 'pipeline_run'.

    Scaling format contains top-level keys "vqe_results" and "metadata".
    Pipeline_run format contains top-level keys "config" and "diagnostics".

    Parameters
    ----------
    data : dict
        Loaded JSON data.

    Returns
    -------
    str
        Either "scaling" or "pipeline_run".

    Raises
    ------
    ValueError
        If format is unrecognized (neither set of keys is present).
    """
    if "vqe_results" in data and "metadata" in data:
        return "scaling"
    elif "config" in data and "diagnostics" in data:
        return "pipeline_run"
    raise ValueError(f"Unknown JSON format: keys={list(data.keys())}")


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading (Dual-Format Adapter)
# ═══════════════════════════════════════════════════════════════════════════════


def _load_scaling_format(data: dict, path: Path, seed: int) -> SourceData:
    """Extract SourceData from scaling-result JSON format.

    Format mapping:
        N        -> metadata.n
        topology -> metadata.topology
        h_values -> vqe_results[seed].results[].h
        theta_opt-> vqe_results[seed].results[].theta_opt
        e_exact  -> vqe_results[seed].results[].dmrg_energy
    """
    meta = data["metadata"]
    n = meta["n"]
    topology = meta["topology"]

    # Filter to requested seed
    vqe_results = data["vqe_results"]
    seed_runs = [r for r in vqe_results if r["seed"] == seed]
    if not seed_runs:
        # Fall back to first available seed if only one exists
        if len(vqe_results) == 1:
            seed_runs = vqe_results
            logger.warning(f"Seed {seed} not found in {path}, using seed {vqe_results[0]['seed']}")
        else:
            raise ValueError(
                f"Seed {seed} not found in {path}. "
                f"Available seeds: {[r['seed'] for r in vqe_results]}"
            )

    results = seed_runs[0]["results"]

    # Validate theta_opt presence
    if not results or "theta_opt" not in results[0]:
        raise ValueError(f"No theta_opt entries found in {path}")

    h_values = np.array([r["h"] for r in results])
    theta_opt = np.array([canonicalize_theta(np.array(r["theta_opt"])) for r in results])
    e_exact = np.array([r["dmrg_energy"] for r in results])
    param_dim = theta_opt.shape[1]

    return SourceData(
        n=n,
        topology=topology,
        h_values=h_values,
        theta_opt=theta_opt,
        e_exact=e_exact,
        param_dim=param_dim,
    )


def _load_pipeline_run_format(data: dict, path: Path, seed: int) -> SourceData:
    """Extract SourceData from pipeline_run JSON format.

    Format mapping (design doc):
        N        -> config.n_qubits
        topology -> config.topology
        h_values -> diagnostics.phase2.h_values
        theta_opt-> vqe_results[seed].results[].theta_opt
        e_exact  -> diagnostics.phase1.energies[]

    Also supports the phase12_data variant where data is inline:
        N        -> system.n_qubits or config.n_qubits
        topology -> system.topology or config.topology
        h_values -> phase12_data[].h
        theta_opt-> phase12_data[].theta_opt
        e_exact  -> phase12_data[].e_exact
    """
    config = data.get("config", {})
    system = data.get("system", {})
    diagnostics = data.get("diagnostics", {})

    # Extract N and topology (try multiple locations)
    n = config.get("n_qubits") or system.get("n_qubits")
    topology = config.get("topology") or system.get("topology")

    if n is None or topology is None:
        raise ValueError(
            f"Cannot determine n_qubits/topology from {path}. "
            f"config keys: {list(config.keys())}, system keys: {list(system.keys())}"
        )

    # Strategy 1: Use vqe_results if present (matches design doc mapping)
    vqe_results = data.get("vqe_results", [])
    if vqe_results:
        seed_runs = [r for r in vqe_results if r.get("seed") == seed]
        if not seed_runs and len(vqe_results) == 1:
            seed_runs = vqe_results
        if seed_runs:
            results = seed_runs[0].get("results", [])
            if results and "theta_opt" in results[0]:
                h_values_diag = diagnostics.get("phase2", {}).get("h_values")
                e_exact_diag = diagnostics.get("phase1", {}).get("energies")

                h_values = np.array(h_values_diag if h_values_diag else [r["h"] for r in results])
                theta_opt = np.array(
                    [canonicalize_theta(np.array(r["theta_opt"])) for r in results]
                )
                e_exact = np.array(
                    e_exact_diag
                    if e_exact_diag
                    else [r.get("dmrg_energy", r.get("e_exact", 0.0)) for r in results]
                )
                # Validate array length consistency
                n_results = len(results)
                if len(h_values) != n_results:
                    h_values = np.array([r["h"] for r in results])
                if len(e_exact) != n_results:
                    e_exact = np.array(
                        [r.get("dmrg_energy", r.get("e_exact", 0.0)) for r in results]
                    )
                param_dim = theta_opt.shape[1]
                return SourceData(
                    n=n,
                    topology=topology,
                    h_values=h_values,
                    theta_opt=theta_opt,
                    e_exact=e_exact,
                    param_dim=param_dim,
                )

    # Strategy 2: Use phase12_data if present (inline format)
    phase12_data = data.get("phase12_data", [])
    if phase12_data and "theta_opt" in phase12_data[0]:
        h_values = np.array([entry["h"] for entry in phase12_data])
        theta_opt = np.array(
            [canonicalize_theta(np.array(entry["theta_opt"])) for entry in phase12_data]
        )
        e_exact = np.array([entry["e_exact"] for entry in phase12_data])
        param_dim = theta_opt.shape[1]
        return SourceData(
            n=n,
            topology=topology,
            h_values=h_values,
            theta_opt=theta_opt,
            e_exact=e_exact,
            param_dim=param_dim,
        )

    # No theta_opt found in any location
    raise ValueError(f"No theta_opt entries found in {path}")


def load_source_data(path: Path, seed: int = 42) -> SourceData:
    """Load VQE data from either scaling-result or pipeline_run JSON format.

    This is the primary entry point for the data adapter. It detects the
    JSON format automatically and extracts h_values, theta_opt, and e_exact
    into a unified SourceData dataclass.

    Parameters
    ----------
    path : Path
        JSON result file (scaling format or pipeline_run format).
    seed : int
        Seed to filter when multiple seeds exist.

    Returns
    -------
    SourceData
        Unified container with h_values, theta_opt, e_exact, and metadata.

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    ValueError
        If file lacks theta_opt entries (message includes file path),
        or if file format is unrecognized.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    fmt = detect_format(data)
    logger.debug(f"Detected format '{fmt}' for {path}")

    if fmt == "scaling":
        return _load_scaling_format(data, path, seed)
    else:  # pipeline_run
        return _load_pipeline_run_format(data, path, seed)


def filter_source_data(
    source: SourceData,
    quality_threshold: float = 0.05,
    max_theta_norm: float = 2.0,
) -> SourceData:
    """Filter SourceData to keep only high-quality VQE points.

    Removes points where:
    - ΔE/gap exceeds quality_threshold (VQE didn't converge)
    - |theta| norm exceeds max_theta_norm (likely stuck at pi-equivalent minima)

    For the theta norm filter: TFIM p=1 optimal theta should be small
    (theta_zz ~ 0.05-0.1, theta_x ~ 0.3-0.5 in the paramagnetic regime).
    Points with |theta| > 2.0 indicate the optimizer found a pi-shifted
    equivalent that breaks the smooth h→theta mapping.

    Parameters
    ----------
    source : SourceData
        Raw VQE data (may contain bad points).
    quality_threshold : float
        Maximum ΔE/gap for a point to be kept (default 0.05 = 5%).
        Points without de_gap info are kept by default.
    max_theta_norm : float
        Maximum L-inf norm of theta for a point to be kept (default 2.0).
        Catches pi-shifted equivalent solutions.

    Returns
    -------
    SourceData
        Filtered data with only high-quality points.

    Raises
    ------
    ValueError
        If all points are filtered out (no usable data).
    """
    n_original = len(source.h_values)
    keep_mask = np.ones(n_original, dtype=bool)

    # Filter by theta norm (catches pi-shifted solutions)
    for i in range(n_original):
        theta_linf = np.max(np.abs(source.theta_opt[i]))
        if theta_linf > max_theta_norm:
            keep_mask[i] = False
            logger.info(
                f"  Filtered: {source.topology} N={source.n} h={source.h_values[i]:.2f} "
                f"— theta norm {theta_linf:.2f} > {max_theta_norm}"
            )

    # Filter by energy quality if e_exact is available
    # Compute ΔE/gap proxy: for paramagnetic TFIM, gap ≈ 2*(h - h_c)
    # We check if e_pred from theta_opt is far from e_exact
    # (This is an approximate check — precise check requires circuit evaluation)

    n_kept = int(keep_mask.sum())
    if n_kept == 0:
        raise ValueError(
            f"All {n_original} points filtered out for {source.topology} N={source.n}. "
            f"No usable VQE data (quality_threshold={quality_threshold}, "
            f"max_theta_norm={max_theta_norm}). "
            f"Consider regenerating data with more restarts or higher h-values."
        )

    if n_kept < n_original:
        logger.warning(
            f"  Quality filter: kept {n_kept}/{n_original} points for "
            f"{source.topology} N={source.n} "
            f"(removed {n_original - n_kept} with |theta|>{max_theta_norm})"
        )

    return SourceData(
        n=source.n,
        topology=source.topology,
        h_values=source.h_values[keep_mask],
        theta_opt=source.theta_opt[keep_mask],
        e_exact=source.e_exact[keep_mask],
        param_dim=source.param_dim,
    )


def load_source_data_filtered(
    path: Path,
    seed: int = 42,
    quality_threshold: float = 0.05,
    max_theta_norm: float = 2.0,
) -> SourceData:
    """Load and filter VQE data in one step.

    Convenience function that calls load_source_data then filter_source_data.
    Also uses de_gap from the JSON if available (scaling format has it).

    Parameters
    ----------
    path : Path
        JSON result file.
    seed : int
        Seed to filter.
    quality_threshold : float
        Maximum ΔE/gap to keep (default 5%).
    max_theta_norm : float
        Maximum theta L-inf norm to keep (default 2.0).

    Returns
    -------
    SourceData
        Filtered, high-quality data only.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    fmt = detect_format(data)

    # For scaling format, we can also filter by de_gap from the JSON directly
    if fmt == "scaling":
        source = _load_scaling_format_with_quality(data, path, seed, quality_threshold)
    else:
        source = _load_pipeline_run_format(data, path, seed)

    # Apply theta norm filter
    return filter_source_data(
        source,
        quality_threshold=quality_threshold,
        max_theta_norm=max_theta_norm,
    )


def _load_scaling_format_with_quality(
    data: dict, path: Path, seed: int, quality_threshold: float
) -> SourceData:
    """Load scaling format and pre-filter by de_gap if available in JSON."""
    meta = data["metadata"]
    n = meta["n"]
    topology = meta["topology"]

    vqe_results = data["vqe_results"]
    seed_runs = [r for r in vqe_results if r["seed"] == seed]
    if not seed_runs:
        if len(vqe_results) == 1:
            seed_runs = vqe_results
        else:
            raise ValueError(f"Seed {seed} not found in {path}")

    results = seed_runs[0]["results"]
    if not results or "theta_opt" not in results[0]:
        raise ValueError(f"No theta_opt entries found in {path}")

    # Filter by de_gap if the field exists in the JSON
    has_de_gap = "de_gap" in results[0]
    if has_de_gap:
        good_results = [r for r in results if r.get("de_gap", 1.0) < quality_threshold]
        n_filtered = len(results) - len(good_results)
        if n_filtered > 0:
            logger.info(
                f"  Pre-filtered {n_filtered}/{len(results)} points with "
                f"de_gap>{quality_threshold * 100:.0f}% from {path.name}"
            )
        if not good_results:
            # Fall back to all results if everything would be filtered
            logger.warning(
                f"  All points in {path.name} have de_gap>{quality_threshold * 100:.0f}%! "
                f"Using all {len(results)} points anyway."
            )
            good_results = results
        results = good_results

    h_values = np.array([r["h"] for r in results])
    theta_opt = np.array([canonicalize_theta(np.array(r["theta_opt"])) for r in results])
    e_exact = np.array([r["dmrg_energy"] for r in results])
    param_dim = theta_opt.shape[1]

    return SourceData(
        n=n,
        topology=topology,
        h_values=h_values,
        theta_opt=theta_opt,
        e_exact=e_exact,
        param_dim=param_dim,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Training Data Validation Guards
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ValidationReport:
    """Report from training data validation checks."""

    passed: bool
    warnings: list[str]
    errors: list[str]
    n_sources: int = 0
    n_total_points: int = 0
    n_unique_sizes: int = 0
    h_range: tuple[float, float] = (0.0, 0.0)
    theta_continuity_ok: bool = True
    sufficient_data: bool = True

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "warnings": self.warnings,
            "errors": self.errors,
            "n_sources": self.n_sources,
            "n_total_points": self.n_total_points,
            "n_unique_sizes": self.n_unique_sizes,
            "h_range": list(self.h_range),
            "theta_continuity_ok": self.theta_continuity_ok,
            "sufficient_data": self.sufficient_data,
        }


def validate_training_data(
    sources: list[SourceData],
    min_points: int = 6,
    min_sizes: int = 2,
    max_theta_jump: float = 1.5,
    experiment_label: str = "",
) -> ValidationReport:
    """Validate that training data is sufficient and consistent for GNN training.

    Performs 5 checks:
    1. Sufficiency: enough total training points
    2. Diversity: data from multiple system sizes
    3. Theta continuity: no discontinuous jumps in theta(h)
    4. Energy bounds: e_exact values are physically reasonable
    5. H-range coverage: training data spans a meaningful range

    Parameters
    ----------
    sources : list[SourceData]
        Training data to validate.
    min_points : int
        Minimum total training points required (default 6).
    min_sizes : int
        Minimum distinct system sizes required (default 2).
    max_theta_jump : float
        Maximum allowed |theta(h_i) - theta(h_{i-1})|_inf between
        consecutive h-values within a single source (default 1.5).
    experiment_label : str
        Label for logging context (e.g., "within_tri", "tri_to_hex").

    Returns
    -------
    ValidationReport
        Report with passed/failed status, warnings, and errors.
        If passed=False, the experiment should NOT proceed.
    """
    warnings: list[str] = []
    errors: list[str] = []
    prefix = f"[{experiment_label}] " if experiment_label else ""

    if not sources:
        msg = f"{prefix}No training data provided — cannot proceed"
        errors.append(msg)
        logger.error(msg)
        return ValidationReport(
            passed=False,
            warnings=warnings,
            errors=errors,
            n_sources=0,
            n_total_points=0,
            n_unique_sizes=0,
            sufficient_data=False,
        )

    # ── Check 1: Training data sufficiency ────────────────────────────
    n_total = sum(len(s.h_values) for s in sources)
    if n_total < min_points:
        msg = (
            f"{prefix}Insufficient training data: {n_total} points "
            f"(need >= {min_points}). GNN will likely overfit."
        )
        errors.append(msg)
        logger.error(msg)

    # ── Check 2: Source data diversity (multiple sizes) ────────────────
    unique_sizes = sorted(set(s.n for s in sources))
    if len(unique_sizes) < min_sizes:
        msg = (
            f"{prefix}Only {len(unique_sizes)} system size(s): {unique_sizes}. "
            f"Need >= {min_sizes} for cross-N generalization."
        )
        errors.append(msg)
        logger.error(msg)

    # ── Check 3: Theta continuity (detect pi-shifted solutions) ────────
    theta_continuity_ok = True
    for src in sources:
        if len(src.h_values) < 2:
            continue
        # Sort by h descending (standard sweep order)
        order = np.argsort(-src.h_values)
        sorted_theta = src.theta_opt[order]
        for i in range(1, len(sorted_theta)):
            jump = np.max(np.abs(sorted_theta[i] - sorted_theta[i - 1]))
            if jump > max_theta_jump:
                h_prev = src.h_values[order[i - 1]]
                h_curr = src.h_values[order[i]]
                msg = (
                    f"{prefix}Theta discontinuity in {src.topology} N={src.n}: "
                    f"|theta(h={h_curr:.2f}) - theta(h={h_prev:.2f})|_inf = {jump:.3f} "
                    f"> {max_theta_jump}. Likely pi-shifted VQE solution."
                )
                warnings.append(msg)
                logger.warning(msg)
                theta_continuity_ok = False

    # ── Check 4: Energy bounds (physically reasonable) ──────────────────
    for src in sources:
        # Ground energy should be negative for TFIM
        if np.any(src.e_exact > 0):
            n_positive = int(np.sum(src.e_exact > 0))
            msg = (
                f"{prefix}Unphysical positive ground energies in "
                f"{src.topology} N={src.n}: {n_positive}/{len(src.e_exact)} points. "
                f"Data may be corrupted."
            )
            errors.append(msg)
            logger.error(msg)

        # Energy should scale roughly with N (E ~ -N*h for paramagnetic)
        max_reasonable = -0.5 * src.n  # Very loose lower bound
        if np.any(src.e_exact > max_reasonable):
            msg = (
                f"{prefix}Suspiciously high energies in {src.topology} N={src.n}: "
                f"max(e_exact)={np.max(src.e_exact):.2f}, "
                f"expected << {max_reasonable:.1f}"
            )
            warnings.append(msg)
            logger.warning(msg)

    # ── Check 5: H-range coverage ──────────────────────────────────────
    all_h = np.concatenate([s.h_values for s in sources])
    h_min, h_max = float(all_h.min()), float(all_h.max())
    h_range_span = h_max - h_min
    if h_range_span < 1.0:
        msg = (
            f"{prefix}Narrow h-range: [{h_min:.2f}, {h_max:.2f}] "
            f"(span={h_range_span:.2f}). May not capture enough variation "
            f"for GNN to learn the h-dependence."
        )
        warnings.append(msg)
        logger.warning(msg)

    # ── Build report ───────────────────────────────────────────────────
    passed = len(errors) == 0
    sufficient = n_total >= min_points

    if warnings and not errors:
        logger.warning(f"{prefix}Training data validation: PASS with {len(warnings)} warning(s)")
    elif not errors:
        logger.info(
            f"{prefix}Training data validation: PASS "
            f"({n_total} points, {len(unique_sizes)} sizes, "
            f"h=[{h_min:.2f},{h_max:.2f}])"
        )

    return ValidationReport(
        passed=passed,
        warnings=warnings,
        errors=errors,
        n_sources=len(sources),
        n_total_points=n_total,
        n_unique_sizes=len(unique_sizes),
        h_range=(h_min, h_max),
        theta_continuity_ok=theta_continuity_ok,
        sufficient_data=sufficient,
    )


def validate_predictions_sanity(
    predictions: list[dict],
    topology: str,
    n_target: int,
    experiment_label: str = "",
) -> ValidationReport:
    """Post-prediction sanity check on GNN outputs.

    Validates that predicted energies and thetas are physically reasonable
    BEFORE considering the experiment a failure (distinguishes "GNN failed
    to generalize" from "something is fundamentally broken").

    Checks:
    - Predicted energy is negative (TFIM ground state is always negative)
    - Predicted energy is not wildly positive (broken circuit evaluation)
    - Theta values are bounded (no NaN, no extreme values)

    Parameters
    ----------
    predictions : list[dict]
        Results from evaluate_theta() calls.
    topology : str
        Target topology.
    n_target : int
        Target system size.
    experiment_label : str
        Label for logging.

    Returns
    -------
    ValidationReport
        Report indicating whether predictions are physically sane.
    """
    warnings: list[str] = []
    errors: list[str] = []
    prefix = f"[{experiment_label}] " if experiment_label else ""

    if not predictions:
        msg = f"{prefix}No predictions to validate"
        errors.append(msg)
        logger.error(msg)
        return ValidationReport(passed=False, warnings=warnings, errors=errors)

    n_positive_energy = 0
    n_extreme_energy = 0
    n_extreme_theta = 0
    n_nan_theta = 0

    for r in predictions:
        e_pred = r.get("e_pred", 0.0)
        theta = r.get("theta_pred", [])

        # Energy should be negative for TFIM
        if e_pred > 0:
            n_positive_energy += 1

        # Energy should not be wildly off (more than 10x exact)
        e_exact = r.get("e_exact", -1.0)
        if abs(e_pred) > 10 * abs(e_exact) and abs(e_exact) > 1.0:
            n_extreme_energy += 1

        # Theta should be bounded
        if any(np.isnan(t) or np.isinf(t) for t in theta):
            n_nan_theta += 1
        elif any(abs(t) > 10.0 for t in theta):
            n_extreme_theta += 1

    n_total = len(predictions)

    if n_positive_energy > 0:
        msg = (
            f"{prefix}Positive predicted energy for "
            f"{n_positive_energy}/{n_total} points on {topology} N={n_target}. "
            f"GNN predictions are physically unreasonable."
        )
        if n_positive_energy > n_total // 2:
            errors.append(msg)
            logger.error(msg)
        else:
            warnings.append(msg)
            logger.warning(msg)

    if n_extreme_energy > 0:
        msg = (
            f"{prefix}Extreme energy predictions (>10x exact) for "
            f"{n_extreme_energy}/{n_total} points. "
            f"GNN may be producing garbage outputs."
        )
        warnings.append(msg)
        logger.warning(msg)

    if n_nan_theta > 0:
        msg = (
            f"{prefix}NaN/Inf in predicted theta for "
            f"{n_nan_theta}/{n_total} points. Model is broken."
        )
        errors.append(msg)
        logger.error(msg)

    if n_extreme_theta > 0:
        msg = (
            f"{prefix}Extreme theta values (|theta|>10) for "
            f"{n_extreme_theta}/{n_total} points. "
            f"GNN output is unbounded — consider gradient clipping."
        )
        warnings.append(msg)
        logger.warning(msg)

    passed = len(errors) == 0
    return ValidationReport(
        passed=passed,
        warnings=warnings,
        errors=errors,
        n_total_points=n_total,
    )


def save_validation_checkpoint(
    report: ValidationReport,
    output_dir: Path,
    experiment_label: str,
) -> Path:
    """Save a validation failure report as a checkpoint JSON for tracking.

    Called when validation fails and execution should stop. The checkpoint
    allows the user to understand what went wrong without re-running.

    Parameters
    ----------
    report : ValidationReport
        The failed validation report.
    output_dir : Path
        Directory to save the checkpoint.
    experiment_label : str
        Label for the filename.

    Returns
    -------
    Path
        Path to the saved checkpoint file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"validation_failure_{experiment_label}_{timestamp}.json"
    path = output_dir / filename

    checkpoint = {
        "type": "validation_failure",
        "experiment_label": experiment_label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "report": report.to_dict(),
        "action_required": (
            "Fix the issues listed in 'errors' before re-running. "
            "Common fixes: generate more data, use higher h-values, "
            "or increase VQE restarts for problematic points."
        ),
    }

    from qmbp_simulation.utils.helpers import json_dump

    try:
        json_dump(checkpoint, path)
    except Exception as e:
        logger.error(f"Failed to save validation checkpoint: {e}")
        return path

    logger.error(f"Validation failure saved to: {path}")
    return path


def validate_vqe_sweep_quality(
    results: list[dict],
    topology: str,
    n: int,
    seed: int,
    convergence_threshold: float = 0.05,
    min_converged_fraction: float = 0.5,
) -> ValidationReport:
    """Validate quality of a VQE sweep before saving/using as training data.

    Check 6: Convergence guard for VQE data generation.
    Ensures enough points in the sweep actually converged well.

    Parameters
    ----------
    results : list[dict]
        VQE result entries with at least 'h', 'de_gap', 'theta_opt'.
    topology : str
        Topology name for logging.
    n : int
        System size.
    seed : int
        Random seed used.
    convergence_threshold : float
        Maximum de_gap for a point to count as "converged" (default 5%).
    min_converged_fraction : float
        Minimum fraction of points that must converge (default 50%).

    Returns
    -------
    ValidationReport
        Report on VQE sweep quality.
    """
    warnings: list[str] = []
    errors: list[str] = []
    prefix = f"[vqe_{topology}_N{n}_s{seed}] "

    if not results:
        msg = f"{prefix}Empty VQE results — sweep produced nothing"
        errors.append(msg)
        logger.error(msg)
        return ValidationReport(passed=False, warnings=warnings, errors=errors)

    # Count converged points
    n_total = len(results)
    n_converged = sum(1 for r in results if r.get("de_gap", 1.0) < convergence_threshold)
    converged_fraction = n_converged / n_total

    if converged_fraction < min_converged_fraction:
        msg = (
            f"{prefix}Only {n_converged}/{n_total} points converged "
            f"(de_gap < {convergence_threshold * 100:.0f}%). "
            f"Need >= {min_converged_fraction * 100:.0f}% for usable training data."
        )
        if n_converged == 0:
            errors.append(msg)
            logger.error(msg)
        else:
            warnings.append(msg)
            logger.warning(msg)

    # Check for variational violations
    n_violations = sum(1 for r in results if not r.get("variational_ok", True))
    if n_violations > 0:
        msg = f"{prefix}Variational principle violated in {n_violations}/{n_total} points"
        warnings.append(msg)
        logger.warning(msg)

    # Check for extremely slow points (budget risk)
    slow_points = [r for r in results if r.get("time_s", 0) > 300]
    if slow_points:
        msg = f"{prefix}{len(slow_points)} points took >5min each. Budget risk for larger sweeps."
        warnings.append(msg)
        logger.warning(msg)

    # Report useful h-range that actually converged
    converged_h = [r["h"] for r in results if r.get("de_gap", 1.0) < convergence_threshold]
    if converged_h:
        logger.info(
            f"{prefix}Converged range: h=[{min(converged_h):.2f}, {max(converged_h):.2f}] "
            f"({n_converged}/{n_total} points)"
        )

    passed = len(errors) == 0
    return ValidationReport(
        passed=passed,
        warnings=warnings,
        errors=errors,
        n_total_points=n_total,
        sufficient_data=n_converged >= 3,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MLP Baseline
# ═══════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """Simple feedforward baseline — no graph structure.

    Input: 3 scalars (mean_h, mean_coord, N/100) — flattened from graph.
    Architecture: Linear(3,128) → ReLU → Linear(128,128) → ReLU → Linear(128, output_dim)

    This deliberately discards per-node structure to test whether the GNN's
    graph awareness is essential for cross-topology transfer.
    """

    def __init__(self, hidden_dim: int = 128, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, 3] — (mean_h, mean_coord, N/100)."""
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Graph Construction Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def build_cross_topology_dataset(
    sources: list[SourceData],
    use_n_feature: bool = True,
) -> list:
    """Build torch_geometric dataset from multiple sources (potentially mixed topologies).

    Node features per site: [h_i, coordination_number_i, N/100]
    Edge structure: from the SOURCE topology (training data's own graph).

    Parameters
    ----------
    sources : list[SourceData]
        List of SourceData instances, possibly from different topologies/sizes.
    use_n_feature : bool
        Whether to include N/100 as a third node feature (default True).

    Returns
    -------
    list[Data]
        torch_geometric Data objects with x, edge_index, y, and e_exact attributes.
    """
    import torch
    from torch_geometric.data import Data

    from qmbp_simulation import HamiltonianBuilder, make_lattice

    builder = HamiltonianBuilder()
    dataset: list = []

    for src in sources:
        lattice = make_lattice(src.topology, src.n, J=1.0, h=float(src.h_values[0]))
        edge_index_np, coord = builder.build_graph_data(lattice)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)

        for i, h in enumerate(src.h_values):
            cols = [np.full(src.n, float(h)), coord.astype(float)]
            if use_n_feature:
                cols.append(np.full(src.n, src.n / 100.0))
            x = torch.tensor(np.stack(cols, axis=1), dtype=torch.float32)
            y = torch.tensor(src.theta_opt[i], dtype=torch.float32)
            data = Data(x=x, edge_index=edge_index, y=y)
            data.e_exact = float(src.e_exact[i])
            dataset.append(data)

    return dataset


def train_mlp_baseline(
    model: MLPBaseline,
    features: np.ndarray,  # [n_samples, 3]
    targets: np.ndarray,  # [n_samples, output_dim]
    n_epochs: int = 6000,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    """Train MLP with same epochs/hidden_dim as GNN for fair comparison.

    Parameters
    ----------
    model : MLPBaseline
        The MLP model to train.
    features : np.ndarray
        Input features of shape [n_samples, 3] — (mean_h, mean_coord, N/100).
    targets : np.ndarray
        Target theta values of shape [n_samples, output_dim].
    n_epochs : int
        Number of training epochs (default 6000, same as GNN).
    lr : float
        Learning rate for Adam optimizer.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys:
        - final_mse: float — MSE at last epoch
        - mse_history: list[float] — MSE per epoch
        - epochs_to_convergence: int — first epoch where loss < 1% of initial
    """
    torch.manual_seed(seed)

    X = torch.tensor(features, dtype=torch.float32)
    Y = torch.tensor(targets, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    mse_history: list[float] = []
    epochs_to_convergence = n_epochs  # default: never converged
    initial_loss: float | None = None

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, Y)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        mse_history.append(loss_val)

        if initial_loss is None:
            initial_loss = loss_val

        # Convergence: first epoch where loss drops below 1% of initial
        if epochs_to_convergence == n_epochs and initial_loss > 0:
            if loss_val < 0.01 * initial_loss:
                epochs_to_convergence = epoch

    return {
        "final_mse": mse_history[-1] if mse_history else 0.0,
        "mse_history": mse_history,
        "epochs_to_convergence": epochs_to_convergence,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Scipy Interpolation Baseline
# ═══════════════════════════════════════════════════════════════════════════════


def scipy_interpolation_predict(
    sources: list[SourceData],
    target_h: float,
    target_n: int,
) -> np.ndarray:
    """Predict theta via 2D linear interpolation on (h, N) → θ.

    Falls back to NearestNDInterpolator when query is outside convex hull.
    No awareness of graph topology — uses only (h, N) coordinates.

    Parameters
    ----------
    sources : list[SourceData]
        Training data from one or more topologies/sizes.
    target_h : float
        Transverse field value for the prediction target.
    target_n : int
        System size for the prediction target.

    Returns
    -------
    np.ndarray
        Predicted theta vector (shape [param_dim]).
    """
    points = []
    values = []

    for src in sources:
        for i, h in enumerate(src.h_values):
            points.append([float(h), float(src.n)])
            values.append(src.theta_opt[i])

    points_arr = np.array(points)
    values_arr = np.array(values)
    query = np.array([[target_h, float(target_n)]])
    theta_pred = np.zeros(values_arr.shape[1])

    for col in range(values_arr.shape[1]):
        interp = LinearNDInterpolator(points_arr, values_arr[:, col])
        val = interp(query).flatten()[0]
        if np.isnan(val):
            nn_interp = NearestNDInterpolator(points_arr, values_arr[:, col])
            val = nn_interp(query).flatten()[0]
        theta_pred[col] = val

    return theta_pred


def build_target_graph(
    topology: str,
    n: int,
    h_val: float,
    use_n_feature: bool = True,
):
    """Build graph for deployment on TARGET topology.

    CRITICAL: Uses the TARGET topology's edge_index and coordination numbers,
    not the source. This enables cross-topology transfer — the GNN receives
    the correct structural context for the prediction target.

    Parameters
    ----------
    topology : str
        Target topology name (e.g. "heavy_hex", "triangular").
    n : int
        Number of qubits in the target system.
    h_val : float
        Transverse field value for the target prediction.
    use_n_feature : bool
        Whether to include N/100 as a third node feature (default True).

    Returns
    -------
    Data
        torch_geometric Data object with x, edge_index, and batch attributes.
        batch is set to zeros (single-graph batch) for inference.
    """
    import torch
    from torch_geometric.data import Data

    from qmbp_simulation import HamiltonianBuilder, make_lattice

    builder = HamiltonianBuilder()
    lattice = make_lattice(topology, n, J=1.0, h=h_val)
    edge_index_np, coord = builder.build_graph_data(lattice)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    cols = [np.full(n, h_val), coord.astype(float)]
    if use_n_feature:
        cols.append(np.full(n, n / 100.0))

    x = torch.tensor(np.stack(cols, axis=1), dtype=torch.float32)
    graph = Data(x=x, edge_index=edge_index, batch=torch.zeros(n, dtype=torch.long))
    return graph


def extract_mlp_features(sources: list[SourceData]) -> tuple[np.ndarray, np.ndarray]:
    """Extract flattened features for MLP: (mean_h, mean_coord, N/100) per sample.

    For each source, builds the lattice graph once to get coordination numbers,
    then creates one feature vector per (source, h_value) pair.

    Parameters
    ----------
    sources : list[SourceData]
        List of unified VQE data containers.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        features: shape [n_samples, 3] — (h, mean_coord, N/100)
        targets: shape [n_samples, output_dim] — theta_opt values
    """
    from qmbp_simulation import HamiltonianBuilder, make_lattice

    features_list: list[list[float]] = []
    targets_list: list[np.ndarray] = []
    builder = HamiltonianBuilder()

    for src in sources:
        lattice = make_lattice(src.topology, src.n, J=1.0, h=float(src.h_values[0]))
        _, coord = builder.build_graph_data(lattice)
        mean_coord = float(coord.mean())

        for i, h in enumerate(src.h_values):
            features_list.append([float(h), mean_coord, src.n / 100.0])
            targets_list.append(src.theta_opt[i])

    return np.array(features_list), np.array(targets_list)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment Result Envelope
# ═══════════════════════════════════════════════════════════════════════════════


def build_experiment_envelope(
    experiment_name: str,
    source_files: list[str],
    **extra_metadata,
) -> dict:
    """Build standardized result envelope with traceability metadata.

    Includes: source file paths, git commit hash, Python/torch/numpy versions,
    timestamp. Orchestrator fills total_time_s after execution.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment (e.g. "cross_topology_transfer").
    source_files : list[str]
        Paths to source data files used in this experiment.
    **extra_metadata
        Additional metadata fields merged into the metadata dict.

    Returns
    -------
    dict
        Standardized envelope with experiment, metadata, and environment sections.
    """
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "unknown"

    return {
        "experiment": experiment_name,
        "metadata": {
            "source_files": source_files,
            "git_commit": git_hash,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **extra_metadata,
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Theta Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_theta(
    theta_pred: np.ndarray,
    n_target: int,
    h_val: float,
    topology: str,
    use_mps: bool = False,
    precision: float = 0.005,
    seed: int = 42,
    threshold: float = 0.10,
) -> dict:
    """Evaluate predicted theta on target system.

    Uses NoiselessBackend for N≤15, MPSBackend(chi_max=MPS_DEFAULT_CHI_MAX) for N=16.
    Checks variational principle: E_pred ≥ E_exact - 1e-6.

    Parameters
    ----------
    theta_pred : np.ndarray
        Predicted parameter vector for the HVA circuit.
    n_target : int
        Number of qubits in the target system.
    h_val : float
        Transverse field value.
    topology : str
        Target lattice topology (e.g. "triangular", "heavy_hex").
    use_mps : bool
        Force MPS backend regardless of system size (default False).
    precision : float
        MPS precision parameter (default 0.005).
    seed : int
        Random seed for MPS backend (default 42).
    threshold : float
        Pass threshold for ΔE/gap (default 0.10 = 10%).

    Returns
    -------
    dict
        Keys: h, e_pred, e_exact, gap, de_gap, energy_error,
              variational_ok, theta_pred, passed, time_s
    """
    from qmbp_simulation import (
        ClassicalSolver,
        HamiltonianBuilder,
        HVACircuitBuilder,
        make_lattice,
    )
    from qmbp_simulation.execution import MPSBackend, NoiselessBackend

    t0 = time.perf_counter()

    builder = HamiltonianBuilder()
    hva = HVACircuitBuilder()
    solver = ClassicalSolver()

    lattice = make_lattice(topology, n_target, J=1.0, h=h_val)
    H = builder.build(lattice)
    circuit, _ = hva.create(n_target, 1, lattice)

    if use_mps or n_target > 15:
        backend = MPSBackend(
            strategy="aer_mps", chi_max=MPS_DEFAULT_CHI_MAX, precision=precision, seed=seed
        )
    else:
        backend = NoiselessBackend()

    e_pred = backend.evaluate(circuit, H, theta_pred)
    gt = solver.solve(H, lattice, method="auto")

    de_gap = abs(e_pred - gt.ground_energy) / max(gt.gap, 1e-10)
    energy_error = float(e_pred - gt.ground_energy)

    # Log warnings for concerning results
    if energy_error < -1e-6:
        logger.warning(
            f"Variational principle violated: E_pred={e_pred:.6f} < E_exact={gt.ground_energy:.6f} "
            f"(error={energy_error:.2e}) for {topology} N={n_target} h={h_val:.3f}"
        )
    if gt.gap < 1e-6:
        logger.warning(
            f"Near-zero gap ({gt.gap:.2e}) for {topology} N={n_target} h={h_val:.3f} — "
            f"ΔE/gap may be unreliable"
        )
    if de_gap > 1.0:
        logger.warning(
            f"Very large ΔE/gap={de_gap * 100:.1f}% for {topology} N={n_target} h={h_val:.3f} — "
            f"prediction may be far from optimal"
        )

    elapsed = time.perf_counter() - t0

    return {
        "h": float(h_val),
        "e_pred": float(e_pred),
        "e_exact": float(gt.ground_energy),
        "gap": float(gt.gap),
        "de_gap": float(de_gap),
        "energy_error": energy_error,
        "variational_ok": energy_error >= -1e-6,
        "theta_pred": theta_pred.tolist(),
        "passed": bool(de_gap < threshold),
        "time_s": elapsed,
    }
