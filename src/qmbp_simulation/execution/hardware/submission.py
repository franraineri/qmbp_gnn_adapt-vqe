"""Job submission with submit-all-then-collect pattern and retry logic.

Key patterns: submit ALL jobs first, THEN collect (parallel execution);
retry on submit failure (max 3 attempts, 30s delay); abort if >1 fails;
discard NaN/Inf from results.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qmbp_simulation.execution.noisy_utils import (
    LayoutSelection,
    build_adjacency,
    find_layouts_bfs,
    select_layouts_low_ces,
)

if TYPE_CHECKING:
    from qmbp_simulation.execution.hardware.config import HardwareConfig
    from qmbp_simulation.framework.logging import StructuredLogger


def select_layouts_for_hardware(
    bound_circuit: QuantumCircuit,
    backend: Any,
    config: HardwareConfig,
    logger: StructuredLogger,
) -> LayoutSelection:
    """Select lowest-CES layouts via noisy_utils pipeline."""
    adj = build_adjacency(backend)
    candidates = find_layouts_bfs(
        adj,
        config.n_qubits,
        n_candidates=config.n_candidates,
        seed=config.layout_seed,
    )
    if len(candidates) < config.n_layouts:
        logger.log(
            "layout_warning",
            data={
                "message": f"Only {len(candidates)} candidates found",
                "required": config.n_layouts,
            },
        )
    layout_selection = select_layouts_low_ces(
        bound_circuit,
        backend,
        candidates,
        n_select=config.n_layouts,
        optimization_level=config.optimization_level,
        max_ces=config.max_ces,
    )
    logger.log(
        "layout_selection",
        data={
            "n_selected": len(layout_selection.layouts),
            "ces_values": layout_selection.ces_values,
        },
    )
    return layout_selection


def submit_all_then_collect(
    isa_circuits: list[QuantumCircuit],
    hamiltonian: SparsePauliOp,
    backend: Any,
    config: HardwareConfig,
    logger: StructuredLogger,
    estimator: Any | None = None,
) -> list[dict[str, Any]]:
    """Submit all jobs then collect. Raises RuntimeError if >1 job fails."""
    # Phase 1: Submit all
    submitted: list[tuple[int, Any]] = []
    for idx, isa_circ in enumerate(isa_circuits):
        h_mapped = hamiltonian.apply_layout(isa_circ.layout)
        job = _submit_with_retry(
            (isa_circ, h_mapped),
            backend,
            config,
            logger,
            idx,
            estimator,
        )
        if job is not None:
            submitted.append((idx, job))
            logger.log(
                "job_submitted",
                data={
                    "layout_idx": idx,
                    "job_id": job.job_id() if hasattr(job, "job_id") else str(idx),
                },
            )
    # Phase 2: Collect all
    results: list[dict[str, Any]] = []
    failed_count = 0
    for idx, job in submitted:
        try:
            # Local PrimitiveJob (fake_backend) has no wait_for_final_state;
            # IBM Runtime jobs do. Handle both gracefully.
            if hasattr(job, "wait_for_final_state"):
                job.wait_for_final_state(timeout=config.job_timeout_s)
                status = job.status().name if hasattr(job.status(), "name") else str(job.status())
                if status != "DONE":
                    failed_count += 1
                    logger.log("job_failed", data={"layout_idx": idx, "status": status})
                    continue

            res = job.result()
            ev = float(res[0].data.evs)
            std = float(res[0].data.stds) if hasattr(res[0].data, "stds") else 0.0
            if np.isfinite(ev):
                jid = job.job_id() if hasattr(job, "job_id") else str(idx)
                results.append({"layout_idx": idx, "job_id": jid, "energy": ev, "std": std})
                logger.log(
                    "job_completion",
                    data={
                        "layout_idx": idx,
                        "job_id": jid,
                        "energy": ev,
                        "std": std,
                    },
                )
            else:
                failed_count += 1
                logger.log("job_nan_discarded", data={"layout_idx": idx, "value": ev})
        except TimeoutError:
            failed_count += 1
            logger.log(
                "job_timeout",
                data={"layout_idx": idx, "timeout_s": config.job_timeout_s},
            )
        except Exception as e:
            failed_count += 1
            logger.log("job_error", data={"layout_idx": idx, "error": str(e)})

    if failed_count > 1:
        raise RuntimeError(
            f"More than 1 job failed ({failed_count}/{len(submitted)}). "
            f"Session aborted. Check execution_log for details."
        )
    return results


def build_estimator_options(config: HardwareConfig) -> dict[str, Any]:
    """Build EstimatorOptions dict for hardware or fake_backend mode."""
    if config.mode == "fake_backend":
        return {
            "default_precision": 1.0 / math.sqrt(config.shots),
            "seed_simulator": config.layout_seed,
        }
    options: dict[str, Any] = {"default_shots": config.shots}
    if config.mitigation.dd_enabled:
        options["dynamical_decoupling"] = {"enable": True, "sequence_type": "XpXm"}
    if config.mitigation.twirling_enabled:
        options["twirling"] = {
            "enable_gates": True,
            "enable_measure": True,
            "num_randomizations": 32,
        }
    if config.mitigation.trex_enabled:
        options.setdefault("resilience", {})["measure_mitigation"] = True
    options.setdefault("resilience", {})["zne_mitigation"] = False  # always off
    return options


def _submit_with_retry(
    pub: tuple[QuantumCircuit, SparsePauliOp],
    backend: Any,
    config: HardwareConfig,
    logger: StructuredLogger,
    layout_idx: int,
    estimator: Any | None = None,
) -> Any | None:
    """Submit a PUB with retry logic. Returns job or None on failure."""
    for attempt in range(config.max_retries):
        try:
            est = estimator if estimator is not None else _get_estimator(backend, config)
            return est.run([pub])
        except Exception as e:
            logger.log(
                "submit_retry",
                data={
                    "layout_idx": layout_idx,
                    "attempt": attempt,
                    "error": str(e),
                },
            )
            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay_s)
    return None


def _get_estimator(backend: Any, config: HardwareConfig) -> Any:
    """Create a configured estimator for the given backend and config."""
    options = build_estimator_options(config)
    if config.mode == "fake_backend":
        from qiskit.primitives import BackendEstimatorV2

        return BackendEstimatorV2(backend=backend, options=options)
    from qiskit_ibm_runtime import EstimatorV2

    return EstimatorV2(backend=backend, options=options)
