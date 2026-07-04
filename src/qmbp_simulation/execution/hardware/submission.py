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

from qmbp_simulation.execution.hardware.layout_optimizer import (
    MAPOMATIC_AVAILABLE,
    select_optimal_layouts,
)
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
    """Select optimal layouts for hardware execution.

    Uses mapomatic VF2 subgraph isomorphism when available and enabled
    (config.use_mapomatic=True). Falls back to BFS + CES selection otherwise.

    P2-A: Dynamic n_layouts escalation — if CES spread across selected layouts
    is insufficient for reliable ZNE extrapolation (spread < min_ces_spread),
    escalates from config.n_layouts to config.n_layouts_max (default 5) to
    improve diversity. This only triggers when additional layouts are available.
    """
    # P2-A: Dynamic layout count parameters
    min_ces_spread = getattr(config, "min_ces_spread", 0.02)
    n_layouts_max = getattr(config, "n_layouts_max", 5)

    # ── PRIORITY 1: Try known SWAP-free layout FIRST ──
    # This is the highest-priority path because it guarantees zero routing
    # overhead (verified: all 9 logical edges map to physical CZ).
    # Only skipped if the layout produces high n_2q (calibration changed topology).
    fallback = getattr(config, "fallback_layout_kingston", None)
    if fallback and len(fallback) >= config.n_qubits:
        try:
            layout_selection = select_layouts_low_ces(
                bound_circuit,
                backend,
                [fallback],  # Single known-good layout as only candidate
                n_select=1,
                optimization_level=config.optimization_level,
                max_ces=config.max_ces,
            )
            if layout_selection.layouts:
                # Verify the transpiled circuit has low 2Q count (SWAP-free)
                tc = layout_selection.transpiled_circuits[0]
                n_2q = sum(dict(tc.count_ops()).get(g, 0) for g in ["cz", "cx", "ecr"])
                if n_2q <= config.n_qubits * 2:  # Heuristic: ≤2× logical edges = minimal routing
                    logger.log(
                        "layout_method",
                        data={
                            "method": "known_swap_free",
                            "layout": fallback,
                            "n_2q": n_2q,
                            "swap_free": n_2q <= len(fallback),
                        },
                    )
                    logger.log(
                        "layout_selection",
                        data={
                            "n_selected": 1,
                            "ces_values": layout_selection.ces_values,
                            "method": "known_swap_free",
                        },
                    )
                    return layout_selection
                else:
                    logger.log(
                        "layout_fallback_rejected",
                        data={
                            "reason": f"n_2q={n_2q} > threshold={config.n_qubits * 2}",
                            "layout": fallback,
                        },
                    )
        except Exception as exc:
            logger.log(
                "layout_fallback_error",
                data={"error": str(exc), "layout": fallback},
            )

    # ── PRIORITY 2: VF2 (mapomatic) ──
    if config.use_mapomatic and MAPOMATIC_AVAILABLE:
        logger.log(
            "layout_method",
            data={"method": "mapomatic_vf2", "strategy": config.layout_strategy},
        )
        layout_selection = select_optimal_layouts(
            bound_circuit,
            backend,
            n_select=config.n_layouts,
            max_ces=config.max_ces,
            max_2q_error=config.layout_max_2q_error,
            min_t1_us=config.layout_min_t1_us,
            optimization_level=config.optimization_level,
            call_limit=config.layout_call_limit,
            exclude_qubits=set(config.layout_exclude_qubits),
            defective_edge_threshold=0.10,
            strategy=config.layout_strategy,
        )
        if layout_selection.layouts:
            # P2-A: Check CES spread and escalate if insufficient
            layout_selection = _maybe_escalate_layouts(
                layout_selection,
                bound_circuit,
                backend,
                config,
                logger,
                min_ces_spread=min_ces_spread,
                n_layouts_max=n_layouts_max,
                method="mapomatic_vf2",
            )
            logger.log(
                "layout_selection",
                data={
                    "n_selected": len(layout_selection.layouts),
                    "ces_values": layout_selection.ces_values,
                    "method": "mapomatic_vf2",
                },
            )
            return layout_selection
        # VF2 found nothing usable — fall through to BFS
        logger.log(
            "layout_fallback",
            data={"reason": "mapomatic returned empty, using BFS"},
        )

    # ── PRIORITY 3: BFS fallback ──
    if config.use_mapomatic and not MAPOMATIC_AVAILABLE:
        logger.log(
            "layout_method",
            data={
                "method": "bfs_fallback",
                "reason": "mapomatic not installed",
            },
        )

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

    # P2-A: Check CES spread and escalate if insufficient (BFS path)
    layout_selection = _maybe_escalate_layouts(
        layout_selection,
        bound_circuit,
        backend,
        config,
        logger,
        min_ces_spread=min_ces_spread,
        n_layouts_max=n_layouts_max,
        method="bfs",
    )

    logger.log(
        "layout_selection",
        data={
            "n_selected": len(layout_selection.layouts),
            "ces_values": layout_selection.ces_values,
            "method": "bfs",
        },
    )
    return layout_selection


def _maybe_escalate_layouts(
    layout_selection: LayoutSelection,
    bound_circuit: QuantumCircuit,
    backend: Any,
    config: HardwareConfig,
    logger: StructuredLogger,
    *,
    min_ces_spread: float = 0.02,
    n_layouts_max: int = 5,
    method: str = "unknown",
) -> LayoutSelection:
    """P2-A: Escalate n_layouts from 3→5 if CES spread is insufficient.

    CES spread is defined as max(CES) - min(CES) across selected layouts.
    Insufficient spread means ZNE extrapolation has poor leverage (all points
    cluster at similar noise levels). Escalating to more layouts increases the
    chance of finding diverse CES values.

    Only escalates when:
    1. Current spread < min_ces_spread
    2. Current n_layouts < n_layouts_max
    3. Additional layouts are actually available

    Parameters
    ----------
    layout_selection : LayoutSelection
        Initial layout selection with n_layouts layouts.
    min_ces_spread : float
        Minimum acceptable CES spread (default 0.02).
    n_layouts_max : int
        Maximum layouts to select on escalation (default 5).
    method : str
        Selection method for logging ("mapomatic_vf2" or "bfs").

    Returns
    -------
    LayoutSelection
        Original selection if spread is OK, or expanded selection if escalated.
    """
    ces_values = layout_selection.ces_values
    n_current = len(ces_values)

    # Guard: need at least 2 layouts to compute spread
    if n_current < 2:
        return layout_selection

    # Guard: already at or above maximum
    if n_current >= n_layouts_max:
        return layout_selection

    ces_spread = max(ces_values) - min(ces_values)
    if ces_spread >= min_ces_spread:
        return layout_selection

    # CES spread insufficient → escalate
    logger.log(
        "layout_escalation_triggered",
        data={
            "reason": "ces_spread_insufficient",
            "ces_spread": ces_spread,
            "min_ces_spread": min_ces_spread,
            "n_current": n_current,
            "n_target": n_layouts_max,
            "method": method,
        },
    )

    # Re-select with more layouts
    if method == "mapomatic_vf2" and MAPOMATIC_AVAILABLE:
        expanded = select_optimal_layouts(
            bound_circuit,
            backend,
            n_select=n_layouts_max,
            max_ces=config.max_ces,
            max_2q_error=getattr(config, "layout_max_2q_error", 0.01),
            min_t1_us=getattr(config, "layout_min_t1_us", 50.0),
            optimization_level=config.optimization_level,
            call_limit=getattr(config, "layout_call_limit", 100_000),
            exclude_qubits=set(getattr(config, "layout_exclude_qubits", [])),
            defective_edge_threshold=0.10,
            strategy=getattr(config, "layout_strategy", "lowest_cost"),
        )
        if expanded.layouts and len(expanded.layouts) > n_current:
            new_spread = max(expanded.ces_values) - min(expanded.ces_values)
            logger.log(
                "layout_escalation_result",
                data={
                    "n_layouts_new": len(expanded.layouts),
                    "ces_spread_new": new_spread,
                    "ces_spread_old": ces_spread,
                    "improvement": new_spread > ces_spread,
                },
            )
            return expanded
    else:
        # BFS path: re-select with expanded count
        adj = build_adjacency(backend)
        candidates = find_layouts_bfs(
            adj,
            config.n_qubits,
            n_candidates=config.n_candidates,
            seed=config.layout_seed,
        )
        expanded = select_layouts_low_ces(
            bound_circuit,
            backend,
            candidates,
            n_select=n_layouts_max,
            optimization_level=config.optimization_level,
            max_ces=config.max_ces,
        )
        if expanded.layouts and len(expanded.layouts) > n_current:
            new_spread = max(expanded.ces_values) - min(expanded.ces_values)
            logger.log(
                "layout_escalation_result",
                data={
                    "n_layouts_new": len(expanded.layouts),
                    "ces_spread_new": new_spread,
                    "ces_spread_old": ces_spread,
                    "improvement": new_spread > ces_spread,
                },
            )
            return expanded

    # Escalation didn't help — keep original
    logger.log(
        "layout_escalation_no_improvement",
        data={"n_current": n_current, "method": method},
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

            # ── Warm-up shot (QPU session primer) ──────────────────────
            # Submit a small (100-shot) execution of the first circuit to
            # stabilize the QPU session. Result is discarded. This helps
            # avoid transient errors at session start on some IBM devices.
            try:
                warmup_circ = isa_circuits[0]
                h_warmup = hamiltonian.apply_layout(warmup_circ.layout)
                warmup_est = EstimatorV2(mode=batch)
                warmup_est.options.default_shots = 100
                warmup_job = warmup_est.run([(warmup_circ, h_warmup)])
                # Wait for warmup but don't use the result
                if hasattr(warmup_job, "wait_for_final_state"):
                    warmup_job.wait_for_final_state(timeout=60)
                logger.log("warmup_completed", data={"shots": 100})
            except Exception as warmup_exc:
                # Warmup failure is non-fatal — continue with real jobs
                logger.log("warmup_skipped", data={"error": str(warmup_exc)})

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


def wait_for_qpu_execution(
    job: Any,
    qpu_timeout_s: int | None = 900,
    poll_interval_s: float = 5.0,
) -> None:
    """Wait for a job to finish, timing out only on QPU execution time.

    Unlike ``job.wait_for_final_state(timeout=T)`` which counts queue wait
    time against the timeout, this function:
      1. Waits indefinitely for the job to leave QUEUED state.
      2. Once the job enters RUNNING, starts a QPU-only timer.
      3. Raises TimeoutError only if QPU execution exceeds ``qpu_timeout_s``.

    For local/fake-backend jobs that lack ``status()`` or ``wait_for_final_state``,
    falls back to ``job.wait_for_final_state()`` without timeout.

    Parameters
    ----------
    job : RuntimeJobV2 or PrimitiveJob
        The submitted job object.
    qpu_timeout_s : int | None
        Maximum seconds to wait after the job starts running on QPU.
        None means wait indefinitely (no timeout even on QPU execution).
    poll_interval_s : float
        Seconds between status polls while waiting (default: 5.0).

    Raises
    ------
    TimeoutError
        If the QPU execution phase exceeds ``qpu_timeout_s``.
    """
    # Local PrimitiveJob (fake_backend) — no polling needed
    if not hasattr(job, "status"):
        if hasattr(job, "wait_for_final_state"):
            job.wait_for_final_state()
        return

    # Phase 1: Wait for job to leave QUEUED (no timeout on queue)
    _QUEUED_STATES = {"QUEUED", "VALIDATING", "INITIALIZING"}
    _TERMINAL_STATES = {"DONE", "ERROR", "CANCELLED"}

    while True:
        raw_status = job.status()
        status = raw_status.name if hasattr(raw_status, "name") else str(raw_status).upper()
        if status not in _QUEUED_STATES:
            break
        time.sleep(poll_interval_s)

    # If already terminal (e.g., cancelled while in queue), return
    if status in _TERMINAL_STATES:
        return

    # Phase 2: Job is RUNNING — apply QPU execution timeout
    if qpu_timeout_s is None:
        # No timeout: wait indefinitely for completion
        if hasattr(job, "wait_for_final_state"):
            job.wait_for_final_state()
        return

    t_running_start = time.time()
    while True:
        raw_status = job.status()
        status = raw_status.name if hasattr(raw_status, "name") else str(raw_status).upper()
        if status in _TERMINAL_STATES:
            return
        elapsed = time.time() - t_running_start
        if elapsed > qpu_timeout_s:
            raise TimeoutError(
                f"QPU execution exceeded {qpu_timeout_s}s "
                f"(running for {elapsed:.0f}s). "
                f"Queue wait was excluded from this timeout."
            )
        time.sleep(poll_interval_s)


def _save_raw_job_to_disk(job: Any, save_dir: Any, label: str = "") -> None:
    """Persist full raw QPU output from a completed job to disk.

    Saves: evs, stds, ZNE noise-factor data, ensemble_stds, metrics,
    submitted options, session_id — everything IBM provides.
    Non-blocking: never raises, silently returns on failure.
    """
    from datetime import UTC, datetime
    from pathlib import Path

    import numpy as np

    save_dir = Path(save_dir)
    job_id = job.job_id() if hasattr(job, "job_id") else "local"

    # Metrics
    metrics_data: dict[str, Any] = {}
    if hasattr(job, "metrics"):
        try:
            metrics = job.metrics()
            if isinstance(metrics, dict):
                usage = metrics.get("usage", {})
                timestamps = metrics.get("timestamps", {})
                metrics_data = {
                    "qpu_seconds": usage.get("quantum_seconds"),
                    "billed_seconds": usage.get("seconds"),
                    "created": timestamps.get("created"),
                    "running": timestamps.get("running"),
                    "finished": timestamps.get("finished"),
                }
        except Exception:
            pass

    # Job metadata
    job_metadata: dict[str, Any] = {}
    for attr in ("program_id", "session_id", "tags"):
        val = getattr(job, attr, None)
        if val is not None:
            job_metadata[attr] = val
    try:
        inputs = getattr(job, "inputs", None)
        if isinstance(inputs, dict):
            options_input = inputs.get("options", {})
            if options_input:
                job_metadata["submitted_options"] = options_input
    except Exception:
        pass

    # PUB results with all available data
    pub_results: list[dict[str, Any]] = []
    try:
        job_result = job.result()
        for pub_idx, pub_result in enumerate(job_result):
            evs = pub_result.data.evs
            stds = getattr(pub_result.data, "stds", None)
            evs_val = float(evs) if np.isscalar(evs) else evs.tolist()
            stds_val = (
                float(stds)
                if stds is not None and np.isscalar(stds)
                else (stds.tolist() if stds is not None else None)
            )
            pub_data: dict[str, Any] = {"pub_idx": pub_idx, "evs": evs_val, "stds": stds_val}
            for extra in ("evs_noise_factors", "stds_noise_factors", "ensemble_stds", "metadata"):
                val = getattr(pub_result.data, extra, None)
                if val is not None:
                    pub_data[extra] = val.tolist() if hasattr(val, "tolist") else val
            pub_results.append(pub_data)
    except Exception:
        pass

    output = {
        "job_id": job_id,
        "saved_at": datetime.now(UTC).isoformat(),
        "metrics": metrics_data,
        "job_metadata": job_metadata,
        "pub_results": pub_results,
        "n_pubs": len(pub_results),
    }

    prefix = f"{label}_" if label else ""
    out_path = save_dir / f"{prefix}raw_qpu_{job_id}.json"
    try:
        from qmbp_simulation.utils.helpers import json_dump

        json_dump(output, out_path)
    except Exception:
        pass


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
            # Use QPU-only timeout: waits indefinitely in queue, then
            # applies job_timeout_s only to actual QPU execution time.
            wait_for_qpu_execution(job, qpu_timeout_s=config.job_timeout_s)
            # Handle both enum-style and string-style status returns
            if hasattr(job, "status"):
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
                        # Total billed seconds (QPU + classical)
                        billed = metrics.get("usage", {}).get("seconds", None)
                        usage_info["billed_seconds"] = (
                            billed if isinstance(billed, (int, float)) else None
                        )
                        # "running" timestamp is an ISO string, NOT numeric seconds
                        # Store as-is for provenance, never sum it
                        running_ts = metrics.get("timestamps", {}).get("running", None)
                        usage_info["running_timestamp"] = running_ts
                        # Created timestamp for queue wait derivation
                        created_ts = metrics.get("timestamps", {}).get("created", None)
                        usage_info["created_timestamp"] = created_ts
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
                # Save full raw QPU output for post-hoc analysis
                if config.mode == "hardware" and config.output_dir:
                    try:
                        from pathlib import Path as _Path

                        _raw_dir = _Path(config.output_dir) / "raw_qpu_output"
                        _raw_dir.mkdir(parents=True, exist_ok=True)
                        _save_raw_job_to_disk(job, _raw_dir, f"layout{idx}")
                    except Exception:
                        pass  # Never block execution for raw save
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
        if "skip_reset_qubits" in dd:
            estimator.options.dynamical_decoupling.skip_reset_qubits = dd["skip_reset_qubits"]
    if "twirling" in options_dict:
        tw = options_dict["twirling"]
        estimator.options.twirling.enable_gates = tw.get("enable_gates", True)
        estimator.options.twirling.enable_measure = tw.get("enable_measure", False)
        if "num_randomizations" in tw:
            estimator.options.twirling.num_randomizations = tw["num_randomizations"]
        if "shots_per_randomization" in tw:
            estimator.options.twirling.shots_per_randomization = tw["shots_per_randomization"]
        if "strategy" in tw:
            estimator.options.twirling.strategy = tw["strategy"]
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
            if "extrapolator" in zne:
                estimator.options.resilience.zne.extrapolator = zne["extrapolator"]
        if "layer_noise_learning" in res:
            lnl = res["layer_noise_learning"]
            estimator.options.resilience.layer_noise_learning.num_randomizations = lnl.get(
                "num_randomizations", 32
            )
            estimator.options.resilience.layer_noise_learning.shots_per_randomization = lnl.get(
                "shots_per_randomization", 128
            )
            if "layer_pair_depths" in lnl:
                estimator.options.resilience.layer_noise_learning.layer_pair_depths = lnl[
                    "layer_pair_depths"
                ]


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
        options["dynamical_decoupling"] = {
            "enable": True,
            "sequence_type": "XpXm",
            "skip_reset_qubits": True,  # |0⟩ qubits post-reset need no DD
        }
    if config.mitigation.twirling_enabled:
        twirl_opts: dict[str, Any] = {
            "enable_gates": True,
            "enable_measure": config.mitigation.trex_enabled,  # measure twirling for TREX
        }
        # Only set num_randomizations when PEA is active (it controls the noise
        # learning budget). For gate_folding, let Runtime auto-distribute shots.
        # "adaptive" maps to "pea" on Runtime, so it also needs the budget.
        if config.mitigation.zne_amplifier in ("pea", "adaptive"):
            twirl_opts["num_randomizations"] = config.mitigation.num_randomizations
            twirl_opts["shots_per_randomization"] = config.mitigation.shots_per_randomization
        # Twirling strategy: "active-circuit" avoids twirling idle qubits in
        # dense circuits (IBM tutorial recommendation for utility-scale).
        if config.mitigation.twirling_strategy:
            twirl_opts["strategy"] = config.mitigation.twirling_strategy
        options["twirling"] = twirl_opts
    if config.mitigation.trex_enabled:
        options.setdefault("resilience", {})["measure_mitigation"] = True

    # ZNE configuration
    if config.mitigation.zne_enabled:
        options.setdefault("resilience", {})["zne_mitigation"] = True
        zne_opts: dict[str, Any] = {}
        # Amplifier selection: map framework values to IBM Runtime values.
        # "adaptive" is a local-only strategy (GF→PEA fallback) — on real hardware,
        # use "pea" for server-side ZNE since PEA is the validated primary.
        amplifier = config.mitigation.zne_amplifier
        runtime_amplifier = "pea" if amplifier == "adaptive" else amplifier
        if runtime_amplifier and runtime_amplifier != "gate_folding":
            zne_opts["amplifier"] = runtime_amplifier
        # Custom noise factors
        if config.mitigation.zne_noise_factors:
            zne_opts["noise_factors"] = config.mitigation.zne_noise_factors
        # Extrapolator: always send both exponential and linear for PEA.
        # IBM Runtime heuristically picks the best ("automatic"), but having
        # both available in the result data enables post-hoc comparison.
        # Ref: IBM PEA tutorial (2026) uses ("exponential", "linear").
        if runtime_amplifier == "pea":
            zne_opts["extrapolator"] = ("exponential", "linear")
        if zne_opts:
            options.setdefault("resilience", {})["zne"] = zne_opts
        # PEA-specific: layer noise learning options
        if runtime_amplifier == "pea":
            lnl_opts: dict[str, Any] = {
                "num_randomizations": config.mitigation.num_randomizations,
                "shots_per_randomization": config.mitigation.shots_per_randomization,
            }
            # layer_pair_depths controls the identity-pair insertion depths used
            # to learn the exponential noise decay per layer. IBM tutorial uses
            # [0,1,2,4,6,12,24] for deep circuits. For HVA p=1 (1 layer of 2Q
            # gates), [0,1,2,4,8] suffices. None = Runtime default.
            if config.mitigation.layer_pair_depths is not None:
                lnl_opts["layer_pair_depths"] = config.mitigation.layer_pair_depths
            options.setdefault("resilience", {})["layer_noise_learning"] = lnl_opts
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
