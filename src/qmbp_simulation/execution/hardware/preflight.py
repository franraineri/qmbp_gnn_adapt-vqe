"""Preflight checks for hardware execution.

Verifies backend status, calibration quality, and topology connectivity
before submitting any jobs. Designed to fail fast and save credits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qmbp_simulation.execution.noisy_utils import build_adjacency

if TYPE_CHECKING:
    from qmbp_simulation.execution.hardware.config import HardwareConfig
    from qmbp_simulation.framework.logging import StructuredLogger


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
    quality, and topology connectivity.

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

    # --- In fake_backend mode, only topology matters ---
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
