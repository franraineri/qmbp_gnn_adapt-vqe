"""Preflight checks for hardware execution.

Verifies backend status, calibration quality, topology connectivity,
and cost ceiling feasibility before submitting any jobs.
Designed to fail fast and save credits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qmbp_simulation.execution.noisy_utils import build_adjacency

if TYPE_CHECKING:
    from qmbp_simulation.execution.hardware.config import HardwareConfig
    from qmbp_simulation.framework.logging import StructuredLogger

# ZNE perturbative thresholds — validated empirically (project-status.md)
# Gate-folding: each 2Q gate gets folded → noise scales linearly with gate count.
#   At >18 CX, the folded circuit is too deep for linear extrapolation.
# PEA: noise amplification is via the noise MODEL, not circuit depth.
#   PEA works up to ~50 CX (validated on heavy_hex N=10: 41-44 CX, R²>0.99).
_ZNE_CX_THRESHOLD_GF = 18
_ZNE_CX_THRESHOLD_PEA = 50
# Legacy alias (used in validate_circuit_for_zne when amplifier unknown)
_ZNE_CX_THRESHOLD = 18


def compute_mean_2q_error(backend) -> float | None:
    """Compute mean error rate across all 2-qubit gates on the backend.

    Scans backend.target for standard 2Q gate types (ecr, cz, cx) and
    averages their error rates, skipping entries where error is None.

    Parameters
    ----------
    backend : BackendV2
        Any Qiskit BackendV2 with calibration data.

    Returns
    -------
    float | None
        Mean 2Q error rate, or None if no error data is available.
    """
    target = backend.target
    errors: list[float] = []
    gate_names = ["ecr", "cz", "cx"]

    for gate_name in gate_names:
        if gate_name not in target.operation_names:
            continue
        try:
            qargs_list = target.qargs_for_operation_name(gate_name)
        except Exception:
            continue
        if qargs_list is None:
            continue
        for qargs in qargs_list:
            if len(qargs) != 2:
                continue
            try:
                props = target[gate_name].get(qargs)
                if props is not None and props.error is not None:
                    errors.append(props.error)
            except Exception:
                continue

    return sum(errors) / len(errors) if errors else None


def run_preflight_checks(
    backend,
    config: HardwareConfig,
    logger: StructuredLogger,
) -> dict[str, Any]:
    """Run pre-execution checks without submitting any jobs.

    In fake_backend mode, only topology connectivity is checked.
    In hardware mode, checks backend status, queue depth, calibration
    quality, topology connectivity, and cost ceiling feasibility.

    Parameters
    ----------
    backend : BackendV2
        The resolved backend (real or fake).
    config : HardwareConfig
        Execution configuration with mode and n_qubits.
    logger : StructuredLogger
        Logger for recording preflight events.

    Returns
    -------
    dict[str, Any]
        Results of each check plus an "abort" boolean field.
    """
    checks: dict[str, Any] = {"abort": False}

    # --- Topology connectivity (both modes) ---
    try:
        adj = build_adjacency(backend)
        connected_nodes = len(adj)
        checks["topology_connected_nodes"] = connected_nodes
        checks["topology_sufficient"] = connected_nodes >= config.n_qubits
        if connected_nodes < config.n_qubits:
            checks["abort"] = True
            checks["abort_reason"] = (
                f"Topology has {connected_nodes} connected nodes, need {config.n_qubits}"
            )
    except Exception as exc:
        checks["topology_error"] = str(exc)
        checks["abort"] = True
        checks["abort_reason"] = f"Topology check failed: {exc}"

    # --- Cost ceiling feasibility (both modes) ---
    # Ensure requested shots don't exceed max_total_shots for a single evaluation
    shots_per_eval = config.shots * config.n_layouts
    checks["shots_per_eval"] = shots_per_eval
    if shots_per_eval > config.max_total_shots:
        checks["abort"] = True
        checks["abort_reason"] = (
            f"shots×layouts ({shots_per_eval:,}) exceeds max_total_shots "
            f"({config.max_total_shots:,}). Reduce shots or n_layouts."
        )

    # --- In fake_backend mode, only topology + cost ceiling matter ---
    if config.mode == "fake_backend":
        logger.log("preflight", data=checks)
        return checks

    # --- Backend operational status ---
    try:
        status = backend.status()
        operational = getattr(status, "operational", True)
        checks["backend_operational"] = operational
        if not operational:
            checks["abort"] = True
            checks["abort_reason"] = (
                f"Backend not operational: {getattr(status, 'status_msg', 'unknown')}"
            )
    except Exception as exc:
        checks["backend_status_error"] = str(exc)
        # Don't abort on status check failure — backend may still work

    # --- Queue depth ---
    try:
        status = backend.status()
        pending = getattr(status, "pending_jobs", None)
        if pending is not None:
            checks["queue_pending_jobs"] = pending
            if pending > 50:
                checks["queue_warning"] = (
                    f"High queue depth ({pending} jobs). "
                    "Consider executing during off-peak hours (UTC 2-6 AM)."
                )
    except Exception:
        pass  # Queue info is best-effort

    # --- Calibration quality (mean 2Q error) ---
    try:
        mean_error = compute_mean_2q_error(backend)
        checks["mean_2q_error"] = mean_error
        if mean_error is not None and mean_error > 0.01:
            checks["abort"] = True
            checks["abort_reason"] = (
                f"Mean 2Q error {mean_error:.4f} exceeds 1% threshold. "
                "Defer execution until calibration improves."
            )
    except Exception as exc:
        checks["calibration_error"] = str(exc)

    logger.log("preflight", data=checks)
    return checks


def validate_circuit_for_zne(
    circuit,
    config: HardwareConfig,
    logger: StructuredLogger,
) -> dict[str, Any]:
    """Validate circuit gate count is within ZNE perturbative regime.

    Must be called AFTER circuit construction but BEFORE job submission.
    Counts 2-qubit gates (CX, CZ, ECR) and warns/aborts if above threshold.

    The threshold is amplifier-aware:
    - Gate-folding: 18 CX (each gate folded → depth triples at factor=3)
    - PEA: 50 CX (noise amplified via model, not circuit depth)
    - Adaptive: uses GF threshold (conservative — will fall back to PEA if needed)

    Parameters
    ----------
    circuit : QuantumCircuit
        The parametrized circuit (before or after binding).
    config : HardwareConfig
        Execution configuration (includes mitigation.zne_amplifier).
    logger : StructuredLogger
        Logger for recording check events.

    Returns
    -------
    dict[str, Any]
        Check results with "abort" boolean if CX count is too high.
    """
    checks: dict[str, Any] = {"abort": False}

    # Determine threshold based on amplifier
    amplifier = config.mitigation.zne_amplifier if config.mitigation else "gate_folding"
    if amplifier == "pea":
        threshold = _ZNE_CX_THRESHOLD_PEA
    elif amplifier == "adaptive":
        # Adaptive starts with GF, so use GF threshold for abort
        # (PEA fallback will handle higher gate counts gracefully)
        threshold = _ZNE_CX_THRESHOLD_PEA
    else:
        threshold = _ZNE_CX_THRESHOLD_GF

    # Count 2-qubit gates
    two_q_gates = {"cx", "cz", "ecr", "rzz", "rxx", "ryy", "cp"}
    gate_count = 0
    gate_breakdown: dict[str, int] = {}
    for instruction in circuit.data:
        name = instruction.operation.name.lower()
        if name in two_q_gates:
            gate_count += 1
            gate_breakdown[name] = gate_breakdown.get(name, 0) + 1

    checks["two_qubit_gate_count"] = gate_count
    checks["gate_breakdown"] = gate_breakdown
    checks["zne_threshold"] = threshold
    checks["amplifier"] = amplifier

    if gate_count > threshold:
        checks["abort"] = True
        checks["abort_reason"] = (
            f"Circuit has {gate_count} 2Q gates (threshold: {threshold} for {amplifier}). "
            f"ZNE extrapolation will fail in non-perturbative regime. "
            f"Use p=1 for N≥10 or reduce circuit depth."
        )
        logger.log("circuit_zne_abort", data=checks)
    elif gate_count > threshold * 0.8:
        checks["zne_warning"] = (
            f"Circuit has {gate_count} 2Q gates "
            f"({gate_count / threshold:.0%} of {amplifier} threshold). "
            f"ZNE may have reduced R². Monitor extrapolation quality."
        )
        logger.log("circuit_zne_warning", data=checks)
    else:
        logger.log("circuit_zne_ok", data=checks)

    return checks


@dataclass
class QPUCostEstimate:
    """Estimated QPU cost for a hardware deployment run.

    All times in seconds. Use this to verify the run fits within
    IBM's max_execution_time limits before submitting jobs.
    """

    n_h_points: int
    circuits_per_h: int
    shots_per_h: int
    total_circuits: int
    total_shots: int
    est_time_per_h_s: float
    est_total_s: float
    max_execution_time_s: int
    fits_per_job: bool
    amplifier: str
    estimated_clops: int


def estimate_qpu_cost(
    config: HardwareConfig,
    n_h_points: int = 4,
    include_spsa: bool = True,
) -> QPUCostEstimate:
    """Estimate QPU time and shot budget for a hardware deployment.

    Uses IBM Eagle r3 throughput estimates (~2500 CLOPS for 10-qubit circuits)
    to predict wall-clock time. Includes ZNE overhead (noise factors) and
    optional SPSA refinement cost.

    Parameters
    ----------
    config : HardwareConfig
        Hardware configuration (shots, n_layouts, amplifier, etc.)
    n_h_points : int
        Number of h-values to evaluate (default: 4).
    include_spsa : bool
        Include worst-case SPSA overhead (30% probability × 200 iters).

    Returns
    -------
    QPUCostEstimate
        Detailed cost breakdown with fits_per_job boolean.
    """
    shots = config.shots
    n_layouts = config.n_layouts
    amplifier = config.mitigation.zne_amplifier

    # IBM Eagle r3 approximate throughput
    ESTIMATED_CLOPS = 2500

    # Noise factors per amplifier
    if amplifier == "pea":
        n_noise_factors = 3
        overhead_factor = 1.5  # noise learning phase
    elif amplifier == "adaptive":
        n_noise_factors = 6  # worst case: GF(3) fails + PEA(3) runs
        overhead_factor = 1.5
    else:
        n_noise_factors = 3
        overhead_factor = 1.0

    circuits_per_h = n_layouts * n_noise_factors
    shots_per_h = circuits_per_h * shots
    time_per_circuit_s = shots / ESTIMATED_CLOPS
    time_per_h_s = circuits_per_h * time_per_circuit_s * overhead_factor

    # Observable measurement overhead (2 groups: X, ZZ)
    obs_time_per_h = 2 * time_per_circuit_s

    # SPSA worst case (30% probability × 200 iter × 2 evals)
    spsa_per_h = 0.0
    if include_spsa:
        spsa_per_h = 0.3 * 200 * 2 * time_per_circuit_s

    total_per_h = time_per_h_s + obs_time_per_h + spsa_per_h
    total_s = n_h_points * total_per_h

    max_exec = config.job_timeout_s
    fits = time_per_h_s < max_exec

    return QPUCostEstimate(
        n_h_points=n_h_points,
        circuits_per_h=circuits_per_h,
        shots_per_h=shots_per_h,
        total_circuits=n_h_points * circuits_per_h,
        total_shots=n_h_points * shots_per_h,
        est_time_per_h_s=total_per_h,
        est_total_s=total_s,
        max_execution_time_s=max_exec,
        fits_per_job=fits,
        amplifier=amplifier,
        estimated_clops=ESTIMATED_CLOPS,
    )
