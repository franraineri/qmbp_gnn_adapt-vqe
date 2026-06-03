"""Preflight checks for hardware execution.

Verifies backend status, calibration quality, topology connectivity,
and cost ceiling feasibility before submitting any jobs.
Designed to fail fast and save credits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qmbp_simulation.execution.noisy_utils import build_adjacency

if TYPE_CHECKING:
    from qmbp_simulation.execution.hardware.config import HardwareConfig
    from qmbp_simulation.framework.logging import StructuredLogger

# ZNE perturbative threshold — validated empirically (project-status.md)
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

    Parameters
    ----------
    circuit : QuantumCircuit
        The parametrized circuit (before or after binding).
    config : HardwareConfig
        Execution configuration.
    logger : StructuredLogger
        Logger for recording check events.

    Returns
    -------
    dict[str, Any]
        Check results with "abort" boolean if CX count is too high.
    """
    checks: dict[str, Any] = {"abort": False}

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
    checks["zne_threshold"] = _ZNE_CX_THRESHOLD

    if gate_count > _ZNE_CX_THRESHOLD:
        checks["abort"] = True
        checks["abort_reason"] = (
            f"Circuit has {gate_count} 2Q gates (threshold: {_ZNE_CX_THRESHOLD}). "
            f"ZNE extrapolation will fail in non-perturbative regime. "
            f"Use p=1 for N≥10 or reduce circuit depth."
        )
        logger.log("circuit_zne_abort", data=checks)
    elif gate_count > _ZNE_CX_THRESHOLD * 0.8:
        checks["zne_warning"] = (
            f"Circuit has {gate_count} 2Q gates ({gate_count / _ZNE_CX_THRESHOLD:.0%} of threshold). "
            f"ZNE may have reduced R². Monitor extrapolation quality."
        )
        logger.log("circuit_zne_warning", data=checks)
    else:
        logger.log("circuit_zne_ok", data=checks)

    return checks
