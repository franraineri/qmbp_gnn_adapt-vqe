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
    """Submit all jobs then collect. Raises RuntimeError if >1 job fails.

    For hardware mode, wraps submissions in a Batch context to batch all
    layout jobs into a single QPU session (reduces queue wait from N× to 1×).
    For fake_backend mode, submits locally without Batch (not supported).
    """
    if config.mode == "hardware" and estimator is None:
        return _submit_all_batched(isa_circuits, hamiltonian, backend, config, logger)
    return _submit_all_sequential(isa_circuits, hamiltonian, backend, config, logger, estimator)


def _submit_all_batched(
    isa_circuits: list[QuantumCircuit],
    hamiltonian: SparsePauliOp,
    backend: Any,
    config: HardwareConfig,
    logger: StructuredLogger,
) -> list[dict[str, Any]]:
    """Submit all layout jobs within a single Batch context (hardware only).

    Creates one Batch → one EstimatorV2 → submits all PUBs → collects results.
    This avoids N separate queue waits for N layouts.
    """
    from qiskit_ibm_runtime import Batch, EstimatorV2

    submitted: list[tuple[int, Any]] = []
    results: list[dict[str, Any]] = []
    failed_count = 0

    try:
        with Batch(backend=backend) as batch:
            # Create one estimator for the entire batch
            est = EstimatorV2(mode=batch)
            # Apply options via structured API
            _apply_estimator_options(est, config)

            # Phase 1: Submit all PUBs
            for idx, isa_circ in enumerate(isa_circuits):
                h_mapped = hamiltonian.apply_layout(isa_circ.layout)
                try:
                    job = est.run([(isa_circ, h_mapped)])
                    submitted.append((idx, job))
                    logger.log(
                        "job_submitted",
                        data={
                            "layout_idx": idx,
                            "job_id": job.job_id() if hasattr(job, "job_id") else str(idx),
                            "batch_mode": True,
                        },
                    )
                except Exception as e:
                    logger.log(
                        "submit_error",
                        data={"layout_idx": idx, "error": str(e), "batch_mode": True},
                    )
    except Exception as e:
        # Batch creation failed — fall back to sequential submission
        logger.log("batch_creation_failed", data={"error": str(e), "fallback": "sequential"})
        return _submit_all_sequential(isa_circuits, hamiltonian, backend, config, logger, None)

    # Phase 2: Collect all results
    results, failed_count = _collect_results(submitted, config, logger)

    if failed_count > 1:
        raise RuntimeError(
            f"More than 1 job failed ({failed_count}/{len(submitted)}). "
            f"Batch aborted. Check execution_log for details."
        )
    return results


def _submit_all_sequential(
    isa_circuits: list[QuantumCircuit],
    hamiltonian: SparsePauliOp,
    backend: Any,
    config: HardwareConfig,
    logger: StructuredLogger,
    estimator: Any | None = None,
) -> list[dict[str, Any]]:
    """Submit jobs sequentially (for fake_backend or when a pre-built estimator is given)."""
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
    results, failed_count = _collect_results(submitted, config, logger)

    if failed_count > 1:
        raise RuntimeError(
            f"More than 1 job failed ({failed_count}/{len(submitted)}). "
            f"Session aborted. Check execution_log for details."
        )
    return results


def _collect_results(
    submitted: list[tuple[int, Any]],
    config: HardwareConfig,
    logger: StructuredLogger,
) -> tuple[list[dict[str, Any]], int]:
    """Collect results from submitted jobs. Returns (results, failed_count)."""
    results: list[dict[str, Any]] = []
    failed_count = 0

    for idx, job in submitted:
        try:
            # Local PrimitiveJob (fake_backend) has no wait_for_final_state;
            # IBM Runtime jobs do. Handle both gracefully.
            if hasattr(job, "wait_for_final_state"):
                job.wait_for_final_state(timeout=config.job_timeout_s)
                # Handle both enum-style and string-style status returns
                raw_status = job.status()
                if hasattr(raw_status, "name"):
                    status = raw_status.name
                else:
                    status = str(raw_status).upper()
                if status != "DONE":
                    failed_count += 1
                    logger.log("job_failed", data={"layout_idx": idx, "status": status})
                    continue

            res = job.result()
            ev = float(res[0].data.evs)
            std = float(res[0].data.stds) if hasattr(res[0].data, "stds") else 0.0
            if np.isfinite(ev):
                jid = job.job_id() if hasattr(job, "job_id") else str(idx)
                # Capture QPU usage metrics (IBM Runtime provides these for real jobs)
                usage_info: dict[str, Any] = {}
                if hasattr(job, "metrics"):
                    try:
                        metrics = job.metrics()
                        # quantum_seconds is numeric (int/float)
                        qs = metrics.get("usage", {}).get("quantum_seconds", None)
                        usage_info["qpu_seconds"] = qs if isinstance(qs, (int, float)) else None
                        # "running" timestamp is an ISO string, NOT numeric seconds
                        # Store as-is for provenance, never sum it
                        running_ts = metrics.get("timestamps", {}).get("running", None)
                        usage_info["running_timestamp"] = running_ts
                    except Exception:
                        pass  # metrics() not available for local jobs
                results.append(
                    {
                        "layout_idx": idx,
                        "job_id": jid,
                        "energy": ev,
                        "std": std,
                        **usage_info,
                    }
                )
                logger.log(
                    "job_completion",
                    data={
                        "layout_idx": idx,
                        "job_id": jid,
                        "energy": ev,
                        "std": std,
                        **usage_info,
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

    return results, failed_count


def _apply_estimator_options(estimator: Any, config: HardwareConfig) -> None:
    """Apply mitigation options to an EstimatorV2 via the structured API."""
    options_dict = build_estimator_options(config)
    if "default_shots" in options_dict:
        estimator.options.default_shots = options_dict["default_shots"]
    if "dynamical_decoupling" in options_dict:
        dd = options_dict["dynamical_decoupling"]
        estimator.options.dynamical_decoupling.enable = dd.get("enable", True)
        estimator.options.dynamical_decoupling.sequence_type = dd.get("sequence_type", "XpXm")
    if "twirling" in options_dict:
        tw = options_dict["twirling"]
        estimator.options.twirling.enable_gates = tw.get("enable_gates", True)
        estimator.options.twirling.enable_measure = tw.get("enable_measure", True)
        estimator.options.twirling.num_randomizations = tw.get("num_randomizations", 32)
    if "resilience" in options_dict:
        res = options_dict["resilience"]
        if "measure_mitigation" in res:
            estimator.options.resilience.measure_mitigation = res["measure_mitigation"]
        if "zne_mitigation" in res:
            estimator.options.resilience.zne_mitigation = res["zne_mitigation"]
        if "zne" in res:
            zne = res["zne"]
            if "amplifier" in zne:
                estimator.options.resilience.zne.amplifier = zne["amplifier"]
            if "noise_factors" in zne:
                estimator.options.resilience.zne.noise_factors = zne["noise_factors"]
        if "layer_noise_learning" in res:
            lnl = res["layer_noise_learning"]
            estimator.options.resilience.layer_noise_learning.num_randomizations = lnl.get(
                "num_randomizations", 32
            )
            estimator.options.resilience.layer_noise_learning.shots_per_randomization = lnl.get(
                "shots_per_randomization", 128
            )


def build_estimator_options(config: HardwareConfig) -> dict[str, Any]:
    """Build EstimatorOptions dict for hardware or fake_backend mode.

    Supports ZNE with two amplifier strategies:
    - "gate_folding" (default): digital gate folding U→U·U†·U
    - "pea": Probabilistic Error Amplification (learns noise model first)
    """
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
            "num_randomizations": config.mitigation.num_randomizations,
        }
    if config.mitigation.trex_enabled:
        options.setdefault("resilience", {})["measure_mitigation"] = True

    # ZNE configuration
    if config.mitigation.zne_enabled:
        options.setdefault("resilience", {})["zne_mitigation"] = True
        zne_opts: dict[str, Any] = {}
        # Amplifier selection
        amplifier = config.mitigation.zne_amplifier
        if amplifier and amplifier != "gate_folding":
            zne_opts["amplifier"] = amplifier
        # Custom noise factors
        if config.mitigation.zne_noise_factors:
            zne_opts["noise_factors"] = config.mitigation.zne_noise_factors
        if zne_opts:
            options.setdefault("resilience", {})["zne"] = zne_opts
        # PEA-specific: layer noise learning options
        if amplifier == "pea":
            options.setdefault("resilience", {})["layer_noise_learning"] = {
                "num_randomizations": config.mitigation.num_randomizations,
                "shots_per_randomization": config.mitigation.shots_per_randomization,
            }
    else:
        options.setdefault("resilience", {})["zne_mitigation"] = False
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
    """Create a configured estimator for the given backend and config.

    For fake_backend: uses Qiskit's local BackendEstimatorV2 with dict options.
    For hardware: uses IBM Runtime EstimatorV2 with structured options API.
    """
    options_dict = build_estimator_options(config)
    if config.mode == "fake_backend":
        from qiskit.primitives import BackendEstimatorV2

        return BackendEstimatorV2(backend=backend, options=options_dict)

    from qiskit_ibm_runtime import EstimatorV2

    estimator = EstimatorV2(mode=backend)
    _apply_estimator_options(estimator, config)
    return estimator
