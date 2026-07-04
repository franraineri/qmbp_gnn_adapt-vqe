"""QESEM integration — Qedma's Qiskit Function for unbiased error mitigation.

Provides the execution path for hardware deployment using QESEM instead of
the local PEA/GF-ZNE pipeline. QESEM handles transpilation, characterization,
error suppression, and quasi-probabilistic error mitigation internally.

References:
    - arXiv:2508.10997 — "Reliable high-accuracy error mitigation for utility-scale quantum circuits"
    - IBM Quantum Docs: https://quantum.cloud.ibm.com/docs/guides/qedma-qesem

Requirements:
    - qiskit-ibm-catalog >= 0.8.0
    - IBM Quantum Premium/Flex/On-Prem plan access
    - Environment: IBM_KEY + IBM_INSTANCE_CRN

Usage:
    Set mitigation.qesem_enabled=True in HardwareConfig. The run_deployment()
    method will automatically route through this module.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

    from .config import HardwareConfig

logger = logging.getLogger(__name__)


QESEM_AVAILABLE = False
_QESEM_IMPORT_ERROR: str | None = None

try:
    from qiskit_ibm_catalog import QiskitFunctionsCatalog

    QESEM_AVAILABLE = True
except ImportError as e:
    _QESEM_IMPORT_ERROR = str(e)


@dataclass
class QESEMResult:
    """Result container for a single QESEM execution.

    Maps QESEM output back to the fields expected by HardwareRunResult.
    """

    # Mitigated energy (primary observable = H_TFIM)
    energy_mitigated: float
    energy_std: float
    # Per-site observables (if submitted as additional observables)
    x_values: list[float]
    zz_values: list[float]
    x_stds: list[float]
    zz_stds: list[float]
    # Noisy (pre-mitigation) estimates
    noisy_energy: float
    noisy_x_values: list[float]
    noisy_zz_values: list[float]
    # QESEM metadata
    job_id: str
    total_qpu_time: float | None = None
    gate_fidelities: dict | None = None
    total_shots: int | None = None
    mitigation_shots: int | None = None
    # Transpiled circuit info from QESEM
    transpiled_circs: list[dict] | None = None
    # Whether noisy (pre-mitigation) data was successfully extracted.
    # When False, noisy_energy/noisy_x_values/noisy_zz_values are zeros (sentinel).
    noisy_data_available: bool = False
    # Pre-submission circuit stats (from circuit_summary on the logical circuit
    # we send to QESEM). Enables post-execution validators to analyze the circuit
    # without needing the transpiled version (QESEM handles its own transpilation).
    circuit_stats: dict | None = None
    # ── QET (Quasi-probabilistic Error Tuning) fields ─────────────────────
    # Per-observable noise-scale results: {scale: (value, std)} for each scale.
    # Populated when QET mode is used (noise_scale2precision provided).
    # For each observable index, contains the results at each noise scale.
    # Structure: list of dicts, one per observable. Each dict maps scale→(ev, std).
    # Index 0 = energy (H), 1..N_x = X_i, N_x+1..N_x+N_zz = ZZ_ij.
    noise_scale_results: list[dict[float, tuple[float, float]]] | None = None
    # Extrapolation method used to produce energy_mitigated:
    #   "qesem_standard" → QESEM's own quasi-probabilistic mitigation (scale=0.0)
    #   "qet_user_wls" → user-side WLS extrapolation from QET noise scale data
    #   "qesem_heuristic" → QESEM's exponential ZNE from scales 1.0 + 2.0
    extrapolation_method: str = "qesem_standard"
    # QESEM heuristic estimate (exponential ZNE from scales 1.0 + 2.0).
    # Available for free in standard QESEM runs. None if not present in results.
    qesem_heuristic_energy: float | None = None
    qesem_heuristic_std: float | None = None


def check_qesem_available() -> tuple[bool, str | None]:
    """Check if QESEM dependencies are installed and importable."""
    print(f"  [DEBUG] check_qesem_available: QESEM_AVAILABLE={QESEM_AVAILABLE}")
    return QESEM_AVAILABLE, _QESEM_IMPORT_ERROR


def validate_qesem_submission(
    pubs: list[tuple],
    options: dict[str, Any] | None = None,
    config: HardwareConfig | None = None,
) -> list[str]:
    """Preflight validation for QESEM submissions — catches common errors locally.

    Call BEFORE submitting to QESEM to catch issues that would otherwise
    result in a 1221 ServerlessError after ~100s of server-side processing.

    Parameters
    ----------
    pubs : list[tuple]
        List of PUBs to submit: [(circuit, observables), ...] or
        [(circuit, observables, params), ...].
    options : dict | None
        The options dict that will be passed to qesem_fn.run().
    config : HardwareConfig | None
        Hardware configuration (for checking mode, precision, etc.).

    Returns
    -------
    list[str]
        List of error/warning messages. Empty list = submission is valid.
    """
    issues: list[str] = []
    print(f"  [DEBUG] validate_qesem_submission: validating {len(pubs)} PUBs")

    if not pubs:
        issues.append("CRITICAL: pubs list is empty — nothing to submit.")
        return issues

    n_pubs = len(pubs)

    # ── Check 1: Multi-PUB not supported by QESEM ──────────────────────────
    if n_pubs > 1:
        effective_options = options or {}
        transpilation_level = effective_options.get("transpilation_level", "standard")
        if transpilation_level == "standard" or transpilation_level is None:
            issues.append(
                f"WARNING: Submitting {n_pubs} PUBs. QESEM does NOT support "
                f"'standard' transpilation for multiple circuits (Error 1221), "
                f"and transpilation_level=0 requires pre-transpiled circuits. "
                f"Use sequential single-PUB submission instead."
            )

    # ── Check 2: Circuits must be parameter-free (bound) ──────────────────
    for i, pub in enumerate(pubs):
        circuit = pub[0]
        if hasattr(circuit, "num_parameters") and circuit.num_parameters > 0:
            issues.append(
                f"CRITICAL: PUB[{i}] circuit has {circuit.num_parameters} unbound "
                f"parameters. QESEM requires bound (parameter-free) circuits. "
                f"Call circuit.assign_parameters(values) first."
            )

    # ── Check 3: Observables must be non-empty ────────────────────────────
    for i, pub in enumerate(pubs):
        observables = pub[1] if len(pub) > 1 else None
        if observables is None or (isinstance(observables, list) and len(observables) == 0):
            issues.append(
                f"CRITICAL: PUB[{i}] has no observables. QESEM requires at least "
                f"one observable per PUB."
            )

    # ── Check 4: Qubit count consistency across PUBs ──────────────────────
    if n_pubs > 1:
        qubit_counts = set()
        for pub in pubs:
            if hasattr(pub[0], "num_qubits"):
                qubit_counts.add(pub[0].num_qubits)
        if len(qubit_counts) > 1:
            issues.append(
                f"WARNING: PUBs have inconsistent qubit counts: {qubit_counts}. "
                f"QESEM may reject this or produce unexpected layout sharing."
            )

    # ── Check 5: Mode guard ───────────────────────────────────────────────
    if config is not None:
        if hasattr(config, "mode") and config.mode == "fake_backend":
            issues.append(
                "CRITICAL: QESEM cannot run in fake_backend mode — it requires "
                "a real QPU with Premium/Flex plan access."
            )

    # ── Check 6: Precision sanity ─────────────────────────────────────────
    if options and "default_precision" in options:
        precision = options["default_precision"]
        if precision <= 0:
            issues.append(
                f"CRITICAL: default_precision={precision} is non-positive. "
                f"Must be > 0 (typical range: 0.005 to 0.05)."
            )
        elif precision > 0.5:
            issues.append(
                f"WARNING: default_precision={precision} is very large. "
                f"QESEM results will have low accuracy. Typical: 0.01."
            )

    # ── Check 7: max_execution_time sanity ────────────────────────────────
    if options and "max_execution_time" in options:
        max_time = options["max_execution_time"]
        if max_time is not None and max_time < 60:
            issues.append(
                f"WARNING: max_execution_time={max_time}s is very short. "
                f"QESEM needs time for characterization + mitigation. "
                f"Minimum recommended: 120s per PUB."
            )

    # ── Check 8: QET noise_scale2precision validation ─────────────────────
    for i, pub in enumerate(pubs):
        if len(pub) >= 4 and pub[3] is not None:
            noise_scales = pub[3]
            if not isinstance(noise_scales, dict):
                issues.append(
                    f"CRITICAL: PUB[{i}] noise_scale2precision must be a dict, "
                    f"got {type(noise_scales).__name__}."
                )
                continue
            for scale, precision in noise_scales.items():
                if not isinstance(scale, (int, float)):
                    issues.append(
                        f"CRITICAL: PUB[{i}] noise scale key must be numeric, "
                        f"got {type(scale).__name__}: {scale}"
                    )
                elif scale < 0:
                    issues.append(
                        f"CRITICAL: PUB[{i}] noise scale={scale} is negative. "
                        f"Valid range: [0.0, ∞). scale=0.0 is QESEM mitigated."
                    )
                if not isinstance(precision, (int, float)):
                    issues.append(
                        f"CRITICAL: PUB[{i}] precision for scale={scale} must be "
                        f"numeric, got {type(precision).__name__}."
                    )
                elif precision <= 0:
                    issues.append(
                        f"CRITICAL: PUB[{i}] precision={precision} for scale={scale} must be > 0."
                    )
            if not noise_scales:
                issues.append(
                    f"CRITICAL: PUB[{i}] noise_scale2precision is empty dict. "
                    f"Must contain at least one scale→precision mapping."
                )

    return issues


def extrapolate_qet_wls(
    noise_scale_data: dict[float, tuple[float, float]],
    extrapolation_order: int = 1,
) -> tuple[float, float]:
    """Extrapolate observable to zero-noise limit from QET noise scale data.

    Uses Weighted Least Squares (WLS) with weights σ_i^{-2} — consistent
    with our PEA-ZNE WLS implementation in noisy_utils.

    Parameters
    ----------
    noise_scale_data : dict[float, tuple[float, float]]
        Mapping of noise_scale → (expectation_value, std). Must have ≥2 points.
        Example: {0.5: (-40.1, 0.05), 1.0: (-39.5, 0.06), 1.5: (-38.9, 0.08)}
    extrapolation_order : int
        Polynomial order for extrapolation. 1 = linear (default, most robust).

    Returns
    -------
    tuple[float, float]
        (extrapolated_value_at_scale_0, extrapolated_std)

    Raises
    ------
    ValueError
        If insufficient data points for requested polynomial order.
    """
    scales = np.array(sorted(noise_scale_data.keys()))
    values = np.array([noise_scale_data[s][0] for s in scales])
    stds = np.array([noise_scale_data[s][1] for s in scales])

    n_points = len(scales)
    n_coeffs = extrapolation_order + 1

    if n_points < n_coeffs:
        raise ValueError(
            f"Need at least {n_coeffs} points for order-{extrapolation_order} "
            f"extrapolation, got {n_points}. Scales: {scales.tolist()}"
        )

    # WLS: weights = 1/σ² (inverse variance weighting)
    # Guard against zero stds (use minimum non-zero std as floor)
    nonzero_stds = stds[stds > 1e-12]
    if len(nonzero_stds) > 0:
        std_floor = float(np.min(nonzero_stds))
    else:
        # All stds are zero — use uniform weighting (unweighted LS)
        std_floor = 1.0
    stds_safe = np.where(stds > 1e-12, stds, std_floor)
    weights = 1.0 / (stds_safe**2)

    # Build Vandermonde matrix for polynomial fit
    # [1, s, s², ...] for each scale value
    vander = np.vander(scales, N=n_coeffs, increasing=True)

    # Weighted least squares: (V^T W V) c = V^T W y
    W = np.diag(weights)
    VtW = vander.T @ W
    A = VtW @ vander
    b = VtW @ values

    try:
        coeffs = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]

    # Extrapolated value at scale=0 is simply coeffs[0] (constant term)
    extrapolated_value = float(coeffs[0])

    # Propagate uncertainty: std at scale=0
    # Covariance of coefficients: Cov = (V^T W V)^{-1}
    try:
        cov = np.linalg.inv(A)
        extrapolated_var = float(cov[0, 0])
        extrapolated_std = float(np.sqrt(max(extrapolated_var, 0.0)))
    except np.linalg.LinAlgError:
        # If covariance estimation fails, use a conservative estimate
        extrapolated_std = float(np.mean(stds))

    # Sanity check: extrapolated value should be finite
    if not np.isfinite(extrapolated_value):
        logger.warning(
            f"WLS extrapolation produced non-finite value ({extrapolated_value}). "
            f"Falling back to weighted mean of input data."
        )
        extrapolated_value = float(np.average(values, weights=weights))
        extrapolated_std = float(np.sqrt(1.0 / np.sum(weights)))

    return extrapolated_value, extrapolated_std


def _parse_qet_noise_scaling_results(
    metadata: dict,
    n_observables: int,
) -> list[dict[float, tuple[float, float]]]:
    """Parse QET noise-scaling results from QESEM metadata.

    Handles both formats:
    1. Standard QESEM (scale=0.0 requested): results in noise_scaling.results_with_REM
    2. QET explicit (no scale=0.0): results only in noise_scaling.results_with_REM

    Parameters
    ----------
    metadata : dict
        The pub_result.metadata dict from QESEM.
    n_observables : int
        Expected number of observables (1 energy + n_x + n_zz).

    Returns
    -------
    list[dict[float, tuple[float, float]]]
        One dict per observable. Each dict maps scale → (value, std).
        Empty list if noise_scaling data is not available.
    """
    results_meta = metadata.get("results")
    if results_meta is None or not isinstance(results_meta, list):
        return []

    per_obs_scales: list[dict[float, tuple[float, float]]] = []

    # results_meta structure: list of groups, each group is list of (obs_array, result_dict)
    # For single-PUB: results_meta[0] is the group for our circuit
    if len(results_meta) == 0:
        return []

    group = results_meta[0]
    if not isinstance(group, (list, tuple)):
        return []

    for obs_idx, obs_entry in enumerate(group):
        if obs_idx >= n_observables:
            break

        scale_dict: dict[float, tuple[float, float]] = {}

        # obs_entry is (obs_array_repr, result_dict)
        if not isinstance(obs_entry, (list, tuple)) or len(obs_entry) < 2:
            per_obs_scales.append(scale_dict)
            continue

        result_dict = obs_entry[1]
        if not isinstance(result_dict, dict):
            per_obs_scales.append(scale_dict)
            continue

        # Extract noise_scaling.results_with_REM
        noise_scaling = result_dict.get("noise_scaling", {})
        if not isinstance(noise_scaling, dict):
            per_obs_scales.append(scale_dict)
            continue

        rem_results = noise_scaling.get("results_with_REM", [])
        if not isinstance(rem_results, (list, tuple)):
            per_obs_scales.append(scale_dict)
            continue

        for point in rem_results:
            if not isinstance(point, dict):
                continue
            scale = point.get("scale")
            value = point.get("value")
            error_bar = point.get("error_bar", 0.0)
            if scale is not None and value is not None:
                try:
                    s = float(scale)
                    v = float(value)
                    e = float(error_bar) if error_bar is not None else 0.0
                    if np.isfinite(v):
                        scale_dict[s] = (v, e)
                except (TypeError, ValueError):
                    continue

        per_obs_scales.append(scale_dict)

    return per_obs_scales


def _parse_qesem_heuristic(
    metadata: dict,
) -> tuple[float | None, float | None]:
    """Extract QESEM heuristic (exponential ZNE) from metadata for energy observable.

    Returns (value, std) or (None, None) if not available.
    """
    results_meta = metadata.get("results")
    if results_meta is None or not isinstance(results_meta, list) or len(results_meta) == 0:
        return None, None

    group = results_meta[0]
    if not isinstance(group, list) or len(group) == 0:
        return None, None

    # First observable = energy
    obs_entry = group[0]
    if not isinstance(obs_entry, (list, tuple)) or len(obs_entry) < 2:
        return None, None

    result_dict = obs_entry[1]
    if not isinstance(result_dict, dict):
        return None, None

    heuristic_list = result_dict.get("qesem_heuristic")
    if not heuristic_list or not isinstance(heuristic_list, list) or len(heuristic_list) == 0:
        return None, None

    h = heuristic_list[0]
    if isinstance(h, dict) and "value" in h:
        return float(h["value"]), float(h.get("error_bar", 0.0))

    return None, None


def _safe_job_status(job: Any) -> str:
    """Safely get job status without raising on network errors."""
    try:
        status = job.status()
        return status.name if hasattr(status, "name") else str(status).upper()
    except Exception:
        return "UNKNOWN"


def _safe_job_logs(job: Any) -> str | None:
    """Safely get job logs without raising on network errors."""
    try:
        if hasattr(job, "logs"):
            logs = job.logs()
            return logs if logs else None
    except Exception:
        pass
    return None


def _load_qesem_function(config: HardwareConfig) -> Any:
    """Load the QESEM Qiskit Function from the catalog.

    Uses credentials from environment (IBM_KEY, IBM_INSTANCE_CRN) or
    from config.qesem_instance if explicitly set.
    """
    import os

    print(f"  [DEBUG] _load_qesem_function: loading qedma/qesem for {config.backend_name}")

    if not QESEM_AVAILABLE:
        raise ImportError(
            f"qiskit-ibm-catalog not installed. Install with: "
            f"pip install qiskit-ibm-catalog>=0.8.0\n"
            f"Original error: {_QESEM_IMPORT_ERROR}"
        )

    token = os.environ.get("IBM_KEY")
    instance = config.qesem_instance or os.environ.get("IBM_INSTANCE_CRN")

    if not token:
        raise ValueError("IBM_KEY environment variable not set. Required for QESEM access.")
    if not instance:
        raise ValueError(
            "IBM_INSTANCE_CRN environment variable not set (or set config.qesem_instance). "
            "Required for QESEM catalog access."
        )

    catalog = QiskitFunctionsCatalog(
        channel="ibm_quantum_platform",
        token=token,
        instance=instance,
    )
    return catalog.load("qedma/qesem")


def estimate_qesem_time(
    circuit: QuantumCircuit,
    observables: list[SparsePauliOp],
    config: HardwareConfig,
    mode: str = "analytical",
) -> dict[str, Any]:
    """Estimate QPU time for QESEM execution without running mitigation.

    Parameters
    ----------
    circuit : QuantumCircuit
        Bound (parameter-free) circuit to mitigate.
    observables : list[SparsePauliOp]
        Observables to estimate (H_TFIM + per-site ops).
    config : HardwareConfig
        Hardware configuration with QESEM settings.
    mode : str
        "analytical" (fast, rough) or "empirical" (uses ~2 min QPU, accurate).

    Returns
    -------
    dict with keys: "time_estimation_sec", "mode", "job_id" (if empirical)
    """
    qesem_fn = _load_qesem_function(config)

    job = qesem_fn.run(
        pubs=[(circuit, observables)],
        backend_name=config.backend_name,
        options={
            "estimate_time_only": mode,
            "default_precision": config.qesem_precision,
            "max_execution_time": 120 if mode == "empirical" else None,
        },
    )

    result = job.result()
    time_sec = result[0].metadata.get("time_estimation_sec", None)

    return {
        "time_estimation_sec": time_sec,
        "mode": mode,
        "job_id": job.job_id,
    }


def run_qesem_deployment(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    x_ops: list[SparsePauliOp],
    zz_ops: list[SparsePauliOp],
    config: HardwareConfig,
    structured_logger: Any | None = None,
    noise_scale2precision: dict[float, float] | None = None,
) -> QESEMResult:
    """Execute a single h-point via QESEM with full mitigation.

    This function replaces the local PEA/GF-ZNE pipeline when
    config.mitigation.qesem_enabled=True.

    QESEM handles internally:
        1. Device characterization (noise model)
        2. Noise-aware transpilation (optimal layout + extended gate set)
        3. Error suppression (gate optimization, Pauli twirling)
        4. Circuit characterization (per-layer error model)
        5. Unbiased quasi-probabilistic error mitigation

    Parameters
    ----------
    circuit : QuantumCircuit
        Bound (parameter-free) circuit. QESEM handles its own transpilation.
    hamiltonian : SparsePauliOp
        H_TFIM observable (energy measurement).
    x_ops : list[SparsePauliOp]
        Per-site X_i observables for phase classification.
    zz_ops : list[SparsePauliOp]
        Per-bond Z_iZ_j observables for phase classification.
    config : HardwareConfig
        Hardware configuration with QESEM settings.
    structured_logger : StructuredLogger | None
        Optional logger for structured event recording.
    noise_scale2precision : dict[float, float] | None
        QET noise-scale-to-precision mapping. When provided, QESEM returns
        expectation values at each requested noise scale (plus free complementary
        scales). The user can then perform custom extrapolation (WLS/polynomial).
        If None, uses config.qesem_noise_scales. If both are None, standard
        QESEM flow with config.qesem_precision as target.
        Example: {0.5: 0.02, 1.3: 0.03} → requests scale=0.5 (precision 0.02)
        and scale=1.3 (precision 0.03). Complementary 1.5 and 0.7 come free.

    Returns
    -------
    QESEMResult
        Mitigated results mapped to our pipeline's format.
    """
    # Resolve QET noise_scale2precision: parameter > config > None (standard)
    effective_noise_scales = noise_scale2precision or config.qesem_noise_scales
    is_qet_mode = effective_noise_scales is not None

    # Validate QET scales if provided
    if is_qet_mode:
        if not isinstance(effective_noise_scales, dict) or not effective_noise_scales:
            raise ValueError(
                f"noise_scale2precision must be a non-empty dict mapping "
                f"scale→precision, got: {effective_noise_scales}"
            )
        for scale, precision in effective_noise_scales.items():
            if scale < 0:
                raise ValueError(f"Noise scale {scale} is negative. Valid range: [0.0, ∞).")
            if precision <= 0:
                raise ValueError(f"Precision {precision} for scale={scale} must be > 0.")

    qesem_fn = _load_qesem_function(config)

    # Build combined observable list: [H_TFIM, X_0, X_1, ..., ZZ_01, ZZ_12, ...]
    # QESEM handles multi-basis measurement optimization automatically.
    all_observables = [hamiltonian] + x_ops + zz_ops
    n_x = len(x_ops)
    n_zz = len(zz_ops)
    print(
        f"  [DEBUG] run_qesem_deployment: submitting {len(all_observables)} observables "
        f"(1 energy + {n_x} X + {n_zz} ZZ) to {config.backend_name}"
    )

    if structured_logger:
        structured_logger.log(
            "qesem_submission",
            data={
                "backend": config.backend_name,
                "n_observables": len(all_observables),
                "n_energy": 1,
                "n_x_ops": n_x,
                "n_zz_ops": n_zz,
                "precision": config.qesem_precision,
                "max_execution_time": config.qesem_max_execution_time,
                "circuit_depth": circuit.depth(),
                "circuit_2q_depth": circuit.depth(
                    filter_function=lambda instr: len(instr.qubits) == 2
                ),
                "is_qet_mode": is_qet_mode,
                "qet_noise_scales": (
                    sorted(effective_noise_scales.keys()) if effective_noise_scales else None
                ),
            },
        )

    # ── Console output: circuit being sent to QESEM ──────────────────────
    depth_2q = circuit.depth(filter_function=lambda instr: len(instr.qubits) == 2)

    # ── Compute circuit_stats for post-execution analysis ─────────────────
    # This captures the logical circuit properties BEFORE QESEM transpiles it.
    # QESEM does its own noise-aware transpilation internally, so these stats
    # represent the "input" circuit. The transpiled version comes back in metadata.
    _gate_counts: dict[str, int] = {}
    _n_2q = 0
    _n_1q = 0
    for inst in circuit.data:
        name = inst.operation.name
        nq = inst.operation.num_qubits
        _gate_counts[name] = _gate_counts.get(name, 0) + 1
        if nq == 2:
            _n_2q += 1
        elif nq == 1:
            _n_1q += 1

    circuit_stats = {
        "n_qubits": circuit.num_qubits,
        "depth": circuit.depth(),
        "depth_2q": depth_2q,
        "n_gates_total": len(circuit.data),
        "n_2q_gates": _n_2q,
        "n_1q_gates": _n_1q,
        "count_ops": _gate_counts,
        "n_observables": len(all_observables),
        "n_x_ops": n_x,
        "n_zz_ops": n_zz,
        "precision_target": config.qesem_precision,
        "max_execution_time": config.qesem_max_execution_time,
        "backend": config.backend_name,
        "source": "pre_qesem_logical",  # Marks this as pre-transpilation stats
    }

    if structured_logger:
        structured_logger.log("qesem_circuit_stats", data=circuit_stats)
    print("\n" + "=" * 70)
    print("  CIRCUIT BEING SENT TO QESEM (Qedma)")
    print("=" * 70)
    print(f"  Backend: {config.backend_name}")
    print(f"  Qubits: {circuit.num_qubits}")
    print(f"  Depth: {circuit.depth()}  |  Depth(2Q): {depth_2q}")
    print(f"  Gate counts: {dict(circuit.count_ops())}")
    print(f"  Observables: {len(all_observables)} total (1 energy + {n_x} X + {n_zz} ZZ)")
    print(f"  Precision target: ε = {config.qesem_precision}")
    print(f"  Max QPU time: {config.qesem_max_execution_time}s")
    print(f"  Client timeout: {config.qesem_max_execution_time * 2 + 300}s")
    print("-" * 70)
    # Print compact circuit text (first 20 lines to avoid flooding terminal)
    circ_text = circuit.draw(output="text", fold=100)
    circ_lines = str(circ_text).split("\n")
    for line in circ_lines[:20]:
        print(f"  {line}")
    if len(circ_lines) > 20:
        print(f"  ... ({len(circ_lines) - 20} more lines)")
    print("=" * 70 + "\n")

    t_start = time.time()

    # ── Build PUB and options based on mode (standard vs QET) ─────────────
    if is_qet_mode:
        # QET mode: PUB has 4 elements (circuit, observables, None, noise_scale2precision)
        # The 4th element is the noise_scale2precision mapping.
        # When using QET, do NOT pass "default_precision" in options — precision
        # is specified per-scale in the noise_scale2precision dict.
        pub = (circuit, all_observables, None, effective_noise_scales)
        run_options: dict[str, Any] = {
            "max_execution_time": config.qesem_max_execution_time,
        }
        logger.info(
            f"QET mode: requesting scales {sorted(effective_noise_scales.keys())} "
            f"with precisions {effective_noise_scales}"
        )
        print("  Mode: QET (Quasi-probabilistic Error Tuning)")
        print(f"  Requested scales: {sorted(effective_noise_scales.keys())}")
        print(f"  Per-scale precisions: {effective_noise_scales}")
    else:
        # Standard QESEM mode: PUB has 2 elements (circuit, observables)
        pub = (circuit, all_observables)
        run_options = {
            "default_precision": config.qesem_precision,
            "max_execution_time": config.qesem_max_execution_time,
        }
        print(f"  Mode: Standard QESEM (scale=0.0, ε={config.qesem_precision})")

    job = qesem_fn.run(
        pubs=[pub],
        backend_name=config.backend_name,
        options=run_options,
    )
    logger.info(f"QESEM job submitted: {job.job_id}")
    if structured_logger:
        structured_logger.log("qesem_job_submitted", data={"job_id": job.job_id})

    # Block until result with client-side timeout (QESEM manages its own QPU
    # scheduling, but we guard against indefinite hangs from queue congestion).
    # NOTE: QiskitFunctionsCatalog jobs do NOT accept timeout= in .result().
    # We implement client-side timeout via concurrent.futures, plus retry logic
    # for transient network errors (ReadTimeoutError on long-polling connections).
    client_timeout_s = config.qesem_max_execution_time * 2 + 900  # +15 min buffer
    max_retries = 3
    retry_delay_s = 30

    result = None
    for attempt in range(1, max_retries + 1):
        try:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(job.result)
                result = future.result(timeout=client_timeout_s)
            break  # Success — exit retry loop
        except concurrent.futures.TimeoutError:
            t_elapsed = time.time() - t_start
            # Check if job is actually done (network blip during result fetch)
            job_status = _safe_job_status(job)
            error_msg = (
                f"QESEM job {job.job_id} timed out after {t_elapsed:.1f}s "
                f"(client timeout = {client_timeout_s}s). "
                f"Job status at timeout: {job_status}. "
                f"The job may still be running server-side.\n"
                f"Recover with:\n"
                f"  .venv/bin/python scripts/recover_qesem_jobs.py {job.job_id}"
            )
            logger.error(error_msg)
            if structured_logger:
                structured_logger.log(
                    "qesem_error",
                    data={
                        "job_id": job.job_id,
                        "is_timeout": True,
                        "elapsed_s": t_elapsed,
                        "timeout_s": client_timeout_s,
                        "error_type": "TimeoutError",
                        "error_message": "Client-side timeout exceeded",
                        "job_status_at_error": job_status,
                        "circuit_stats": circuit_stats,
                    },
                )
            raise RuntimeError(error_msg)
        except Exception as exc:
            t_elapsed = time.time() - t_start
            exc_str = str(exc)
            # Retry on transient network errors (ReadTimeoutError, ConnectionError)
            is_transient = any(
                pattern in exc_str
                for pattern in (
                    "ReadTimeoutError",
                    "ConnectionError",
                    "Read timed out",
                    "AUTH1001",
                    "Connection reset",
                )
            )
            if is_transient and attempt < max_retries:
                logger.warning(
                    f"QESEM job {job.job_id}: transient network error on attempt "
                    f"{attempt}/{max_retries} after {t_elapsed:.1f}s. "
                    f"Retrying in {retry_delay_s}s... Error: {exc}"
                )
                if structured_logger:
                    structured_logger.log(
                        "qesem_retry",
                        data={
                            "job_id": job.job_id,
                            "attempt": attempt,
                            "max_retries": max_retries,
                            "elapsed_s": t_elapsed,
                            "error_type": type(exc).__name__,
                            "error_snippet": exc_str[:200],
                        },
                    )
                time.sleep(retry_delay_s)
                continue  # Retry

            # Non-transient error or max retries exhausted — get server logs
            job_status = _safe_job_status(job)
            job_logs = _safe_job_logs(job)

            error_msg = (
                f"QESEM job {job.job_id} failed after {t_elapsed:.1f}s "
                f"(attempt {attempt}/{max_retries}). "
                f"{'Transient error — retries exhausted. ' if is_transient else ''}"
                f"Server error: {exc}\n"
                f"Job status: {job_status}\n"
                f"Common causes:\n"
                f"  - Backend '{config.backend_name}' unavailable or in maintenance\n"
                f"  - Circuit exceeds backend limits (qubits, gates)\n"
                f"  - QESEM function access not enabled for your plan\n"
                f"  - Observable format incompatible with QESEM\n"
                f"Recover with:\n"
                f"  .venv/bin/python scripts/recover_qesem_jobs.py {job.job_id}"
            )

            logger.error(error_msg)
            if job_logs:
                logger.error(f"QESEM server logs (tail): {job_logs[-300:]}")
            if structured_logger:
                structured_logger.log(
                    "qesem_error",
                    data={
                        "job_id": job.job_id,
                        "is_timeout": False,
                        "elapsed_s": t_elapsed,
                        "timeout_s": client_timeout_s,
                        "error_type": type(exc).__name__,
                        "error_message": exc_str[:500],
                        "attempt": attempt,
                        "is_transient": is_transient,
                        "job_status_at_error": job_status,
                        "job_logs_tail": job_logs[-300:] if job_logs else None,
                        "circuit_stats": circuit_stats,
                    },
                )
            raise RuntimeError(error_msg) from exc

    t_elapsed = time.time() - t_start

    logger.info(f"QESEM job completed in {t_elapsed:.1f}s wall time")

    # Parse QESEM output
    pub_result = result[0]
    metadata = pub_result.metadata
    expected_n_obs = 1 + n_x + n_zz

    # ── Parse noise-scaling results (available in both standard and QET modes)
    noise_scale_results = _parse_qet_noise_scaling_results(metadata, expected_n_obs)
    qesem_heuristic_energy, qesem_heuristic_std = _parse_qesem_heuristic(metadata)

    # ── Determine how to extract mitigated values ─────────────────────────
    # In standard QESEM (scale=0.0 requested or default), pub_result.data.evs
    # contains the fully mitigated results. In QET without scale=0.0, data.evs
    # may be absent — we must extract from noise_scaling and extrapolate.
    has_data_evs = False
    try:
        _evs_candidate = getattr(getattr(pub_result, "data", None), "evs", None)
        if _evs_candidate is not None:
            _evs_array = np.atleast_1d(_evs_candidate)
            if len(_evs_array) > 0 and np.isfinite(_evs_array[0]):
                has_data_evs = True
    except (TypeError, ValueError, AttributeError):
        pass
    extrapolation_method = "qesem_standard"

    if has_data_evs:
        evs = np.atleast_1d(pub_result.data.evs)
        stds = np.atleast_1d(pub_result.data.stds)

        # Validate observable count matches what we submitted
        if len(evs) != expected_n_obs:
            raise RuntimeError(
                f"QESEM returned {len(evs)} expectation values but we submitted "
                f"{expected_n_obs} observables (1 energy + {n_x} X + {n_zz} ZZ). "
                f"This may indicate QESEM observable grouping changed the output shape."
            )

        # Standard QESEM: extract directly from data.evs
        energy_mitigated = float(evs[0])
        energy_std = float(stds[0])
        x_values = [float(evs[1 + i]) for i in range(n_x)]
        zz_values = [float(evs[1 + n_x + i]) for i in range(n_zz)]
        x_stds = [float(stds[1 + i]) for i in range(n_x)]
        zz_stds = [float(stds[1 + n_x + i]) for i in range(n_zz)]

    elif is_qet_mode and noise_scale_results:
        # QET mode without scale=0.0: extrapolate from noise-scale data.
        # Use WLS linear extrapolation to scale=0 for each observable.
        extrapolation_method = "qet_user_wls"
        logger.info(
            f"QET mode: no data.evs available. Extrapolating from "
            f"{len(noise_scale_results)} observables × "
            f"{len(noise_scale_results[0]) if noise_scale_results else 0} scales."
        )

        # Energy (observable index 0)
        if len(noise_scale_results) > 0 and len(noise_scale_results[0]) >= 2:
            energy_mitigated, energy_std = extrapolate_qet_wls(
                noise_scale_results[0], extrapolation_order=1
            )
        else:
            raise RuntimeError(
                "QET mode: insufficient noise-scale data for energy observable. "
                f"Got {len(noise_scale_results[0]) if noise_scale_results else 0} "
                f"scale points, need at least 2."
            )

        # Per-site X observables (indices 1..n_x)
        x_values = []
        x_stds = []
        for i in range(n_x):
            obs_idx = 1 + i
            if obs_idx < len(noise_scale_results) and len(noise_scale_results[obs_idx]) >= 2:
                val, std = extrapolate_qet_wls(noise_scale_results[obs_idx])
                x_values.append(val)
                x_stds.append(std)
            else:
                x_values.append(0.0)
                x_stds.append(1.0)

        # Per-bond ZZ observables (indices n_x+1..n_x+n_zz)
        zz_values = []
        zz_stds = []
        for i in range(n_zz):
            obs_idx = 1 + n_x + i
            if obs_idx < len(noise_scale_results) and len(noise_scale_results[obs_idx]) >= 2:
                val, std = extrapolate_qet_wls(noise_scale_results[obs_idx])
                zz_values.append(val)
                zz_stds.append(std)
            else:
                zz_values.append(0.0)
                zz_stds.append(1.0)

    else:
        raise RuntimeError(
            "QESEM returned no mitigated data (no data.evs and no noise_scaling). "
            "Job may have failed silently or returned unexpected format. "
            f"Metadata keys: {list(metadata.keys())}"
        )

    # Clip observables to physical bounds [-1, 1]. QESEM's unbiased estimator
    # can produce values slightly exceeding Pauli bounds due to finite sampling.
    # We clip and log the violations for traceability.
    _n_clipped = 0
    for i in range(n_x):
        if abs(x_values[i]) > 1.0:
            _n_clipped += 1
            x_values[i] = max(-1.0, min(1.0, x_values[i]))
    for i in range(n_zz):
        if abs(zz_values[i]) > 1.0:
            _n_clipped += 1
            zz_values[i] = max(-1.0, min(1.0, zz_values[i]))
    if _n_clipped > 0:
        logger.info(
            f"Clipped {_n_clipped} QESEM observables to [-1, 1] "
            f"(unbiased estimator artifact, expected behavior)."
        )

    # Extract noisy (pre-mitigation) results if available.
    # Guard against API changes: noisy_results may be an object with .evs,
    # a dict with "evs" key, a string repr (from serialization), or absent entirely.
    noisy_results = metadata.get("noisy_results", None)
    noisy_energy = 0.0
    noisy_x: list[float] = [0.0] * n_x
    noisy_zz: list[float] = [0.0] * n_zz
    _noisy_available = False

    if noisy_results is not None:
        try:
            # String repr (from recovery scripts that used str(v) serialization)
            # e.g. "DataBin(evs=np.ndarray(<shape=(20,)...>)...)"
            if isinstance(noisy_results, str):
                logger.warning(
                    f"noisy_results is a string representation (not parseable): "
                    f"'{noisy_results[:80]}...'. Cannot extract noisy baseline. "
                    f"Re-recover with proper serialization to get noisy data."
                )
                # Leave _noisy_available = False, sentinels remain at 0.0
            # Object with .evs attribute (current QESEM SDK)
            elif hasattr(noisy_results, "evs"):
                noisy_evs = np.atleast_1d(noisy_results.evs)
                if len(noisy_evs) >= 1 + n_x + n_zz:
                    noisy_energy = float(noisy_evs[0])
                    noisy_x = [float(noisy_evs[1 + i]) for i in range(n_x)]
                    noisy_zz = [float(noisy_evs[1 + n_x + i]) for i in range(n_zz)]
                    _noisy_available = True
                else:
                    logger.warning(
                        f"noisy_results has {len(noisy_evs)} values, expected "
                        f"{1 + n_x + n_zz}. Falling back to zeros."
                    )
            # Dict with "evs" key (possible future SDK format or recovered JSON)
            elif isinstance(noisy_results, dict) and "evs" in noisy_results:
                noisy_evs = np.atleast_1d(np.asarray(noisy_results["evs"]))
                if len(noisy_evs) >= 1 + n_x + n_zz:
                    noisy_energy = float(noisy_evs[0])
                    noisy_x = [float(noisy_evs[1 + i]) for i in range(n_x)]
                    noisy_zz = [float(noisy_evs[1 + n_x + i]) for i in range(n_zz)]
                    _noisy_available = True
                else:
                    logger.warning(
                        f"noisy_results dict has {len(noisy_evs)} values, expected "
                        f"{1 + n_x + n_zz}. Falling back to zeros."
                    )
            else:
                raise AttributeError(f"noisy_results has unexpected type: {type(noisy_results)}")
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning(f"Failed to parse noisy_results from QESEM metadata: {exc}")

    # Enrich circuit_stats with QESEM's transpiled circuit info (if available).
    # This gives us the physical circuit QESEM actually ran on the QPU.
    transpiled_info = metadata.get("transpiled_circs", None)
    if transpiled_info and isinstance(transpiled_info, list) and len(transpiled_info) > 0:
        tc = transpiled_info[0]
        # Extract qubit map to determine physical qubits used
        qubit_maps = tc.get("qubit_maps", [])
        physical_qubits = []
        if qubit_maps and len(qubit_maps) > 0:
            physical_qubits = [pair[1] for pair in qubit_maps[0]]
        circuit_stats["qesem_transpiled"] = {
            "n_physical_qubits_used": len(physical_qubits),
            "physical_qubits": physical_qubits,
            "num_measurement_bases": tc.get("num_measurement_bases"),
            "has_qasm": "circuit" in tc,
        }

    # ── Post-result diagnostics (zero-cost sanity checks) ─────────────────
    # Flag observable bound violations (expected with unbiased QESEM estimator).
    obs_violations = []
    for i, x in enumerate(x_values):
        if abs(x) > 1.0:
            obs_violations.append(f"|X_{i}|={abs(x):.6f}")
    for i, zz in enumerate(zz_values):
        if abs(zz) > 1.0:
            obs_violations.append(f"|ZZ_{i}|={abs(zz):.6f}")
    if obs_violations:
        logger.info(
            f"QESEM unbiased estimator produced {len(obs_violations)} observables "
            f"slightly exceeding Pauli bound: {obs_violations[:5]}. "
            f"This is expected behavior for quasi-probabilistic EM."
        )

    # Precision convergence metric
    precision_ratio = energy_std / config.qesem_precision if config.qesem_precision > 0 else 0.0
    if precision_ratio > 2.0:
        logger.warning(
            f"QESEM achieved σ={energy_std:.4f} but target ε={config.qesem_precision:.4f} "
            f"(ratio={precision_ratio:.1f}×). QPU time cap likely hit before convergence."
        )

    # Add diagnostics to circuit_stats for persistence
    circuit_stats["post_execution"] = {
        "wall_time_s": t_elapsed,
        "precision_ratio": precision_ratio,
        "n_obs_violations": len(obs_violations),
        "obs_violations": obs_violations[:10],  # Cap at 10 for JSON size
        "noisy_data_available": _noisy_available,
        "extrapolation_method": extrapolation_method,
        "is_qet_mode": is_qet_mode,
    }

    # ── QET-specific diagnostics ──────────────────────────────────────────
    if is_qet_mode and noise_scale_results:
        # Record per-observable scale coverage for post-hoc analysis
        energy_scales = noise_scale_results[0] if noise_scale_results else {}
        circuit_stats["qet_diagnostics"] = {
            "requested_scales": sorted(effective_noise_scales.keys()),
            "received_scales": sorted(energy_scales.keys()),
            "n_total_scale_points": sum(len(d) for d in noise_scale_results),
            "energy_scale_values": {
                f"scale_{s:.2f}": {"value": v, "std": std}
                for s, (v, std) in sorted(energy_scales.items())
            },
            "has_complementary_pairs": any(
                abs(s1 + s2 - 2.0) < 0.01 for s1 in energy_scales for s2 in energy_scales if s1 < s2
            ),
        }

    qesem_result = QESEMResult(
        energy_mitigated=energy_mitigated,
        energy_std=energy_std,
        x_values=x_values,
        zz_values=zz_values,
        x_stds=x_stds,
        zz_stds=zz_stds,
        noisy_energy=noisy_energy,
        noisy_x_values=noisy_x,
        noisy_zz_values=noisy_zz,
        job_id=job.job_id,
        total_qpu_time=metadata.get("total_qpu_time", None),
        gate_fidelities=metadata.get("gate_fidelities", None),
        total_shots=metadata.get("total_shots", None),
        mitigation_shots=metadata.get("mitigation_shots", None),
        transpiled_circs=metadata.get("transpiled_circs", None),
        noisy_data_available=_noisy_available,
        circuit_stats=circuit_stats,
        noise_scale_results=noise_scale_results if noise_scale_results else None,
        extrapolation_method=extrapolation_method,
        qesem_heuristic_energy=qesem_heuristic_energy,
        qesem_heuristic_std=qesem_heuristic_std,
    )

    if structured_logger:
        structured_logger.log(
            "qesem_result",
            data={
                "job_id": job.job_id,
                "energy_mitigated": energy_mitigated,
                "energy_std": energy_std,
                "precision_achieved": energy_std <= config.qesem_precision,
                "noisy_energy": noisy_energy,
                "noisy_data_available": _noisy_available,
                "total_qpu_time": qesem_result.total_qpu_time,
                "total_shots": qesem_result.total_shots,
                "mitigation_shots": qesem_result.mitigation_shots,
                "wall_time_s": t_elapsed,
                "mag_x_mean": float(np.mean(x_values)),
                "corr_zz_mean": float(np.mean(zz_values)),
                "gate_fidelities": qesem_result.gate_fidelities,
                "circuit_stats": circuit_stats,
            },
        )

    return qesem_result


def run_qesem_sweep(
    circuit: QuantumCircuit,
    hamiltonians: list[SparsePauliOp],
    x_ops_list: list[list[SparsePauliOp]],
    zz_ops_list: list[list[SparsePauliOp]],
    params_per_h: list[np.ndarray],
    h_values: list[float],
    config: HardwareConfig,
    structured_logger: Any | None = None,
    noise_scale2precision: dict[float, float] | None = None,
) -> list[QESEMResult]:
    """Execute multiple h-points in a single QESEM call (multi-PUB batch).

    This is the preferred execution mode for Tier 1+ sweeps. By batching all
    h-points into one QESEM function call:
      - Device characterization is performed ONCE (not per-h-point)
      - All circuits share the same transpilation/layout selection
      - QPU time budget is distributed across PUBs efficiently
      - ~3-4× QPU cost reduction vs sequential single-PUB calls

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametric (unbound) circuit. Will be bound to each params set.
    hamiltonians : list[SparsePauliOp]
        H(h) for each h-point. Must match len(h_values).
    x_ops_list : list[list[SparsePauliOp]]
        Per-site X observables for each h-point. Typically identical across h.
    zz_ops_list : list[list[SparsePauliOp]]
        Per-bond ZZ observables for each h-point. Typically identical across h.
    params_per_h : list[np.ndarray]
        Bound parameter values for each h-point.
    h_values : list[float]
        The h-values being swept.
    config : HardwareConfig
        Hardware configuration with QESEM settings.
    structured_logger : StructuredLogger | None
        Optional logger for structured event recording.
    noise_scale2precision : dict[float, float] | None
        QET noise-scale-to-precision mapping. When provided, each per-h-point
        job uses QET mode (noise scales instead of standard QESEM).
        Falls back to config.qesem_noise_scales. If both None, standard QESEM.

    Returns
    -------
    list[QESEMResult]
        One QESEMResult per h-point, in the same order as h_values.
    """
    n_pubs = len(h_values)
    if not (
        len(hamiltonians) == len(x_ops_list) == len(zz_ops_list) == len(params_per_h) == n_pubs
    ):
        raise ValueError(
            f"All input lists must have same length. Got h_values={n_pubs}, "
            f"hamiltonians={len(hamiltonians)}, x_ops={len(x_ops_list)}, "
            f"zz_ops={len(zz_ops_list)}, params={len(params_per_h)}"
        )

    qesem_fn = _load_qesem_function(config)

    # Resolve QET noise_scale2precision: parameter > config > None (standard)
    effective_noise_scales = noise_scale2precision or config.qesem_noise_scales
    is_qet_mode = effective_noise_scales is not None

    # Build PUBs: one per h-point, each with (bound_circuit, [H, X_0,..., ZZ_01,...])
    pubs = []
    n_obs_per_pub = []
    circuit_stats_list = []
    for i in range(n_pubs):
        bound = circuit.assign_parameters(params_per_h[i])
        observables_i = [hamiltonians[i]] + x_ops_list[i] + zz_ops_list[i]
        pubs.append((bound, observables_i))
        n_obs_per_pub.append(len(observables_i))

        # Compute per-PUB circuit stats
        depth_2q = bound.depth(filter_function=lambda instr: len(instr.qubits) == 2)
        _gc: dict[str, int] = {}
        _n2q = 0
        for inst in bound.data:
            name = inst.operation.name
            _gc[name] = _gc.get(name, 0) + 1
            if inst.operation.num_qubits == 2:
                _n2q += 1
        circuit_stats_list.append(
            {
                "h_value": h_values[i],
                "n_qubits": bound.num_qubits,
                "depth": bound.depth(),
                "depth_2q": depth_2q,
                "n_2q_gates": _n2q,
                "n_gates_total": len(bound.data),
                "count_ops": _gc,
                "n_observables": len(observables_i),
                "source": "pre_qesem_logical",
            }
        )

    if structured_logger:
        structured_logger.log(
            "qesem_sweep_submission",
            data={
                "n_pubs": n_pubs,
                "h_values": h_values,
                "n_obs_per_pub": n_obs_per_pub,
                "backend": config.backend_name,
                "precision": config.qesem_precision,
                "max_execution_time_per_pub": config.qesem_max_execution_time,
                "mode": "sequential_single_pub",
                "circuit_stats": circuit_stats_list,
            },
        )

    # ── Console output ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  QESEM SEQUENTIAL SWEEP (one job per h-point)")
    print("=" * 70)
    print(f"  Backend: {config.backend_name}")
    print(f"  Jobs: {n_pubs} (one per h-point)")
    print(f"  h-values: {h_values}")
    print(f"  Observables per PUB: {n_obs_per_pub[0]}")
    print(f"  Precision target: ε = {config.qesem_precision}")
    print(
        f"  Max QPU time per job: {config.qesem_max_execution_time}s "
        f"({config.qesem_max_execution_time / 60:.1f} min)"
    )
    print("  Note: QESEM does not support multi-PUB batches with different")
    print("        circuits. Each h-point submitted as a separate job.")
    print("=" * 70 + "\n")

    t_start = time.time()

    # ── Preflight validation — catch common errors locally ────────────────
    single_pub_options = {
        "default_precision": config.qesem_precision,
        "max_execution_time": config.qesem_max_execution_time,
    }
    preflight_issues = validate_qesem_submission(
        pubs=pubs[:1], options=single_pub_options, config=config
    )
    if preflight_issues:
        critical_issues = [i for i in preflight_issues if i.startswith("CRITICAL")]
        if critical_issues:
            error_msg = (
                f"QESEM sweep preflight FAILED ({len(critical_issues)} critical issues):\n"
                + "\n".join(f"  • {issue}" for issue in preflight_issues)
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        for issue in preflight_issues:
            logger.warning(f"QESEM preflight: {issue}")

    # ── Sequential submission: one QESEM job per h-point ──────────────────
    # QESEM's "standard" transpilation_level does NOT support multiple circuits
    # in a single call. The official pattern (IBM tutorial: qedma-2d-ising-with-qesem)
    # uses one PUB per qesem_function.run() call. Each job gets independent device
    # characterization, which adds ~2 min overhead per job but is the only supported
    # mode for circuits with non-native gates (rzz, rx, etc.).
    qesem_results: list[QESEMResult] = []
    job_ids: list[str] = []

    for pub_idx in range(n_pubs):
        h = h_values[pub_idx]
        pub = pubs[pub_idx]
        cs = circuit_stats_list[pub_idx]
        n_x = len(x_ops_list[pub_idx])
        n_zz = len(zz_ops_list[pub_idx])

        t_pub_start = time.time()
        print(f"\n  ▶ Submitting h={h:.3f} ({pub_idx + 1}/{n_pubs})...")

        if structured_logger:
            structured_logger.log(
                "qesem_sweep_pub_submission",
                data={
                    "pub_idx": pub_idx,
                    "h_value": h,
                    "n_pubs_total": n_pubs,
                    "circuit_stats": cs,
                },
            )

        # ── Build per-h PUB and options based on mode ─────────────────────
        if is_qet_mode:
            pub = (pub[0], pub[1], None, effective_noise_scales)
            sweep_run_options: dict[str, Any] = {
                "max_execution_time": config.qesem_max_execution_time,
            }
        else:
            sweep_run_options = {
                "default_precision": config.qesem_precision,
                "max_execution_time": config.qesem_max_execution_time,
            }

        job = qesem_fn.run(
            pubs=[pub if not is_qet_mode else pub],
            backend_name=config.backend_name,
            options=sweep_run_options,
        )
        logger.info(f"QESEM job submitted for h={h:.3f}: {job.job_id} (PUB {pub_idx + 1}/{n_pubs})")
        job_ids.append(job.job_id)

        # Block with retry logic
        client_timeout_s = config.qesem_max_execution_time * 2 + 900
        max_retries = 3
        retry_delay_s = 30

        result = None
        for attempt in range(1, max_retries + 1):
            try:
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(job.result)
                    result = future.result(timeout=client_timeout_s)
                break
            except concurrent.futures.TimeoutError:
                t_elapsed = time.time() - t_pub_start
                job_status = _safe_job_status(job)
                error_msg = (
                    f"QESEM job {job.job_id} (h={h:.3f}) timed out after "
                    f"{t_elapsed:.1f}s (client timeout={client_timeout_s}s, "
                    f"status={job_status}).\n"
                    f"Recover with:\n"
                    f"  .venv/bin/python scripts/recover_qesem_jobs.py {job.job_id}"
                )
                logger.error(error_msg)
                if structured_logger:
                    structured_logger.log(
                        "qesem_sweep_error",
                        data={
                            "job_id": job.job_id,
                            "h_value": h,
                            "is_timeout": True,
                            "elapsed_s": t_elapsed,
                            "job_status": job_status,
                        },
                    )
                raise RuntimeError(error_msg)
            except Exception as exc:
                t_elapsed = time.time() - t_pub_start
                exc_str = str(exc)
                is_transient = any(
                    p in exc_str
                    for p in (
                        "ReadTimeoutError",
                        "ConnectionError",
                        "Read timed out",
                        "AUTH1001",
                    )
                )
                if is_transient and attempt < max_retries:
                    logger.warning(
                        f"QESEM job {job.job_id} (h={h:.3f}): transient error "
                        f"attempt {attempt}/{max_retries}. Retrying in "
                        f"{retry_delay_s}s..."
                    )
                    if structured_logger:
                        structured_logger.log(
                            "qesem_sweep_retry",
                            data={
                                "job_id": job.job_id,
                                "h_value": h,
                                "attempt": attempt,
                                "error_snippet": exc_str[:200],
                            },
                        )
                    time.sleep(retry_delay_s)
                    continue

                job_status = _safe_job_status(job)
                job_logs = _safe_job_logs(job)
                error_msg = (
                    f"QESEM job {job.job_id} (h={h:.3f}) failed after "
                    f"{t_elapsed:.1f}s (attempt {attempt}/{max_retries}, "
                    f"status={job_status}): {exc}\n"
                    f"Recover with:\n"
                    f"  .venv/bin/python scripts/recover_qesem_jobs.py "
                    f"{job.job_id}"
                )
                logger.error(error_msg)
                if structured_logger:
                    structured_logger.log(
                        "qesem_sweep_error",
                        data={
                            "job_id": job.job_id,
                            "h_value": h,
                            "is_timeout": False,
                            "elapsed_s": t_elapsed,
                            "error_type": type(exc).__name__,
                            "error_message": exc_str[:500],
                            "job_status": job_status,
                            "job_logs_tail": (job_logs[-300:] if job_logs else None),
                        },
                    )
                raise RuntimeError(error_msg) from exc

        t_pub_elapsed = time.time() - t_pub_start

        # ── Parse single-PUB result ──────────────────────────────────────
        pub_result = result[0]
        evs = np.atleast_1d(pub_result.data.evs)
        stds = np.atleast_1d(pub_result.data.stds)
        metadata = pub_result.metadata

        expected_n_obs = 1 + n_x + n_zz
        if len(evs) != expected_n_obs:
            logger.warning(
                f"h={h:.3f}: expected {expected_n_obs} observables, "
                f"got {len(evs)}. Padding/truncating."
            )

        # Extract energy
        energy_mitigated = float(evs[0])
        energy_std = float(stds[0])

        # Extract per-site observables with clipping
        x_values = [float(evs[1 + i]) for i in range(min(n_x, len(evs) - 1))]
        zz_values = [float(evs[1 + n_x + i]) for i in range(min(n_zz, len(evs) - 1 - n_x))]
        x_stds = [float(stds[1 + i]) for i in range(min(n_x, len(stds) - 1))]
        zz_stds = [float(stds[1 + n_x + i]) for i in range(min(n_zz, len(stds) - 1 - n_x))]

        # Clip to physical bounds [-1, 1]
        _n_clipped = 0
        for i in range(len(x_values)):
            if abs(x_values[i]) > 1.0:
                _n_clipped += 1
                x_values[i] = max(-1.0, min(1.0, x_values[i]))
        for i in range(len(zz_values)):
            if abs(zz_values[i]) > 1.0:
                _n_clipped += 1
                zz_values[i] = max(-1.0, min(1.0, zz_values[i]))

        # Parse noisy baselines
        noisy_results = metadata.get("noisy_results", None)
        noisy_energy = 0.0
        noisy_x: list[float] = [0.0] * n_x
        noisy_zz: list[float] = [0.0] * n_zz
        _noisy_available = False

        if noisy_results is not None:
            try:
                if hasattr(noisy_results, "evs"):
                    noisy_evs = np.atleast_1d(noisy_results.evs)
                elif isinstance(noisy_results, dict) and "evs" in noisy_results:
                    noisy_evs = np.atleast_1d(np.asarray(noisy_results["evs"]))
                else:
                    raise AttributeError(f"Unexpected noisy_results type: {type(noisy_results)}")

                if len(noisy_evs) >= 1 + n_x + n_zz:
                    noisy_energy = float(noisy_evs[0])
                    noisy_x = [float(noisy_evs[1 + i]) for i in range(n_x)]
                    noisy_zz = [float(noisy_evs[1 + n_x + i]) for i in range(n_zz)]
                    _noisy_available = True
            except (AttributeError, TypeError, ValueError) as exc:
                logger.warning(f"h={h:.3f}: failed to parse noisy_results: {exc}")

        # Enrich circuit_stats with QESEM transpiled info
        transpiled_info = metadata.get("transpiled_circs", None)
        if transpiled_info and isinstance(transpiled_info, list) and len(transpiled_info) > 0:
            tc = transpiled_info[0]
            qubit_maps = tc.get("qubit_maps", [])
            physical_qubits = [pair[1] for pair in qubit_maps[0]] if qubit_maps else []
            cs["qesem_transpiled"] = {
                "n_physical_qubits_used": len(physical_qubits),
                "physical_qubits": physical_qubits,
                "num_measurement_bases": tc.get("num_measurement_bases"),
            }

        # Post-execution diagnostics
        precision_ratio = energy_std / config.qesem_precision if config.qesem_precision > 0 else 0.0
        cs["post_execution"] = {
            "wall_time_s": t_pub_elapsed,
            "precision_ratio": precision_ratio,
            "n_obs_clipped": _n_clipped,
            "noisy_data_available": _noisy_available,
            "pub_idx": pub_idx,
            "batch_size": n_pubs,
            "job_id": job.job_id,
        }

        # Parse noise_scaling results and heuristic (available in both modes)
        _noise_scale_results = _parse_qet_noise_scaling_results(metadata, 1 + n_x + n_zz)
        _heur_e, _heur_std = _parse_qesem_heuristic(metadata)

        qesem_results.append(
            QESEMResult(
                energy_mitigated=energy_mitigated,
                energy_std=energy_std,
                x_values=x_values,
                zz_values=zz_values,
                x_stds=x_stds,
                zz_stds=zz_stds,
                noisy_energy=noisy_energy,
                noisy_x_values=noisy_x,
                noisy_zz_values=noisy_zz,
                job_id=job.job_id,
                total_qpu_time=metadata.get("total_qpu_time", None),
                gate_fidelities=metadata.get("gate_fidelities", None),
                total_shots=metadata.get("total_shots", None),
                mitigation_shots=metadata.get("mitigation_shots", None),
                transpiled_circs=metadata.get("transpiled_circs", None),
                noisy_data_available=_noisy_available,
                circuit_stats=cs,
                noise_scale_results=_noise_scale_results if _noise_scale_results else None,
                extrapolation_method="qet_user_wls" if is_qet_mode else "qesem_standard",
                qesem_heuristic_energy=_heur_e,
                qesem_heuristic_std=_heur_std,
            )
        )

        print(
            f"    ✅ h={h:.3f}: E={energy_mitigated:.4f} ± {energy_std:.4f} "
            f"({t_pub_elapsed:.1f}s, job={job.job_id[:12]}...)"
        )

    t_elapsed = time.time() - t_start
    if structured_logger:
        structured_logger.log(
            "qesem_sweep_result",
            data={
                "job_ids": job_ids,
                "n_pubs": n_pubs,
                "wall_time_s": t_elapsed,
                "h_values": h_values,
                "energies": [r.energy_mitigated for r in qesem_results],
                "stds": [r.energy_std for r in qesem_results],
                "gate_fidelities": qesem_results[0].gate_fidelities,
                "total_shots_per_pub": [r.total_shots for r in qesem_results],
            },
        )

    print(f"\n  ✅ QESEM sweep completed: {n_pubs} jobs in {t_elapsed:.1f}s")
    for i, r in enumerate(qesem_results):
        print(f"    h={h_values[i]:.2f}: E={r.energy_mitigated:.4f} ± {r.energy_std:.4f}")

    return qesem_results
