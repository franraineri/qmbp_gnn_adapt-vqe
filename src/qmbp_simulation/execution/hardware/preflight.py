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


def compute_layout_2q_error(backend, layout: list[int]) -> float | None:
    """Compute mean 2Q error rate only for qubits in the selected layout.

    Unlike compute_mean_2q_error (which averages over the ENTIRE chip including
    degraded qubits), this function measures error only on the edges that will
    actually be used by the transpiled circuit. This gives a realistic estimate
    of noise impact for our specific 10-qubit subgraph.

    Parameters
    ----------
    backend : BackendV2
        Any Qiskit BackendV2 with calibration data.
    layout : list[int]
        Physical qubit indices selected by the layout.

    Returns
    -------
    float | None
        Mean 2Q error rate for edges within the layout, or None if unavailable.
    """
    target = backend.target
    layout_set = set(layout)
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
            # Only include edges where BOTH qubits are in our layout
            if qargs[0] in layout_set and qargs[1] in layout_set:
                try:
                    props = target[gate_name].get(qargs)
                    if props is not None and props.error is not None:
                        errors.append(props.error)
                except Exception:
                    continue

    return sum(errors) / len(errors) if errors else None


def compute_mean_readout_error(backend) -> float | None:
    """Compute mean readout error rate across all qubits on the backend.

    Scans backend.target for the "measure" operation and averages its
    error rates across all qubits, skipping entries where error is None.

    Parameters
    ----------
    backend : BackendV2
        Any Qiskit BackendV2 with calibration data.

    Returns
    -------
    float | None
        Mean readout error rate, or None if no error data is available.
    """
    target = backend.target
    errors: list[float] = []

    if "measure" not in target.operation_names:
        return None

    try:
        qargs_list = target.qargs_for_operation_name("measure")
    except Exception:
        return None

    if qargs_list is None:
        return None

    for qargs in qargs_list:
        try:
            props = target["measure"].get(qargs)
            if props is not None and props.error is not None:
                errors.append(props.error)
        except Exception:
            continue

    return sum(errors) / len(errors) if errors else None


def compute_min_t1_t2(backend) -> tuple[float | None, float | None]:
    """Compute minimum T1 and T2 coherence times across all qubits.

    Uses backend.target.qubit_properties to extract T1/T2 values,
    filtering out None entries. Returns the minimum observed values
    (worst-case qubit), which determines the coherence floor.

    Parameters
    ----------
    backend : BackendV2
        Any Qiskit BackendV2 with qubit property data.

    Returns
    -------
    tuple[float | None, float | None]
        (min_t1, min_t2) in seconds, or None if data is unavailable.
    """
    try:
        qubit_props = backend.target.qubit_properties
    except Exception:
        return None, None

    if qubit_props is None:
        return None, None

    t1_vals: list[float] = []
    t2_vals: list[float] = []

    for qp in qubit_props:
        if qp is None:
            continue
        t1 = getattr(qp, "t1", None)
        t2 = getattr(qp, "t2", None)
        if t1 is not None:
            t1_vals.append(t1)
        if t2 is not None:
            t2_vals.append(t2)

    min_t1 = min(t1_vals) if t1_vals else None
    min_t2 = min(t2_vals) if t2_vals else None
    return min_t1, min_t2


def _compute_t1_percentile(backend, percentile: int = 5) -> float | None:
    """Compute a given percentile of T1 across all qubits.

    More robust than min_T1 for large processors where 1-5 qubits may
    have TLS-induced low T1 while the rest are healthy. The 5th percentile
    gives a better picture of "widespread" decoherence vs isolated defects.

    Parameters
    ----------
    backend : BackendV2
        Any Qiskit BackendV2 with qubit property data.
    percentile : int
        Percentile to compute (default: 5 = bottom 5%).

    Returns
    -------
    float | None
        T1 at the given percentile in seconds, or None if unavailable.
    """
    import numpy as np

    try:
        qubit_props = backend.target.qubit_properties
    except Exception:
        return None

    if qubit_props is None:
        return None

    t1_vals: list[float] = []
    for qp in qubit_props:
        if qp is None:
            continue
        t1 = getattr(qp, "t1", None)
        if t1 is not None:
            t1_vals.append(t1)

    if not t1_vals:
        return None

    return float(np.percentile(t1_vals, percentile))


def check_native_gate_support(backend, required_gates: list[str]) -> dict[str, bool]:
    """Check whether required gates appear in the backend's native gate set.

    Parameters
    ----------
    backend : BackendV2
        Any Qiskit BackendV2.
    required_gates : list[str]
        Gate names to check (e.g. ["ecr", "rz", "sx", "x", "measure"]).

    Returns
    -------
    dict[str, bool]
        Mapping gate_name -> True if present in backend.target.operation_names.
    """
    op_names = set(backend.target.operation_names)
    return {gate: gate in op_names for gate in required_gates}


def run_preflight_checks(
    backend,
    config: HardwareConfig,
    logger: StructuredLogger,
) -> dict[str, Any]:
    """Run pre-execution checks without submitting any jobs.

    In fake_backend mode, only topology connectivity is checked.
    In hardware mode, checks backend status, queue depth, calibration
    quality, topology connectivity, cost ceiling feasibility, readout
    error rates, T1/T2 coherence, and native gate support.

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
    # On 150+ qubit processors (Kingston/Torino/Sherbrooke), the chip-wide
    # mean 2Q error includes degraded qubits that BFS + CES layout selection
    # actively avoids. Typical global means are 2-4% even when the selected
    # 10-qubit subgraph achieves ~1%. We abort only above 5% (catastrophic)
    # and warn above 3% (degraded but usable with good layout selection).
    try:
        mean_error = compute_mean_2q_error(backend)
        checks["mean_2q_error"] = mean_error
        if mean_error is not None and mean_error > 0.05:
            checks["abort"] = True
            checks["abort_reason"] = (
                f"Mean 2Q error {mean_error:.4f} exceeds 5% threshold. "
                "Defer execution until calibration improves."
            )
        elif mean_error is not None and mean_error > 0.03:
            checks["calibration_warning"] = (
                f"Mean 2Q error {mean_error:.4f} is elevated (>3%). "
                "Layout selection should avoid worst qubits. Monitor ZNE R²."
            )
            logger.log("preflight_2q_warning", data={"mean_2q_error": mean_error})
    except Exception as exc:
        checks["calibration_error"] = str(exc)

    # --- Readout error check (hardware only) ---
    # TREX mitigates readout errors, so this is a warning, not abort.
    try:
        mean_readout_error = compute_mean_readout_error(backend)
        checks["mean_readout_error"] = mean_readout_error
        if mean_readout_error is not None and mean_readout_error > 0.03:
            checks["readout_warning"] = (
                f"Mean readout error {mean_readout_error:.4f} exceeds 3%. "
                "TREX mitigation is enabled — this is mitigated but may increase variance."
            )
            logger.log("preflight_readout_warning", data={"mean_readout_error": mean_readout_error})
    except Exception as exc:
        checks["readout_check_error"] = str(exc)

    # --- T1/T2 coherence check (hardware only) ---
    # Note: min_T1 is the WORST qubit on the entire chip. On large processors
    # (133-156 qubits) there are typically 1-5 qubits with very low T1 due to
    # TLS defects. Our layout selection (BFS + CES) avoids these qubits.
    # We report min_T1 as informational but only abort if the 5th-percentile
    # T1 is critically low (indicates widespread decoherence, not isolated defects).
    try:
        min_t1, min_t2 = compute_min_t1_t2(backend)
        checks["min_t1_us"] = min_t1 * 1e6 if min_t1 is not None else None
        checks["min_t2_us"] = min_t2 * 1e6 if min_t2 is not None else None

        # Compute 5th-percentile T1 for a more robust decoherence check
        p5_t1 = _compute_t1_percentile(backend, percentile=5)
        checks["p5_t1_us"] = p5_t1 * 1e6 if p5_t1 is not None else None

        if p5_t1 is not None and p5_t1 < 30e-6:
            # 5% of qubits below 30μs → widespread decoherence, abort
            checks["abort"] = True
            checks["abort_reason"] = (
                f"5th-percentile T1 = {p5_t1 * 1e6:.1f}μs < 30μs — "
                f"widespread decoherence (min T1 = {min_t1 * 1e6:.1f}μs)."
            )
            logger.log(
                "preflight_t1_abort",
                data={
                    "min_t1_us": min_t1 * 1e6 if min_t1 else None,
                    "p5_t1_us": p5_t1 * 1e6,
                },
            )
        elif min_t1 is not None and min_t1 < 50e-6:
            # Some bad qubits but most are fine — warn but don't abort
            checks["t1_warning"] = (
                f"Min T1 = {min_t1 * 1e6:.1f}μs (isolated defect). "
                f"P5 T1 = {p5_t1 * 1e6:.1f}μs — layout selection will avoid bad qubits."
            )
            logger.log(
                "preflight_t1_warning",
                data={
                    "min_t1_us": min_t1 * 1e6,
                    "p5_t1_us": p5_t1 * 1e6 if p5_t1 else None,
                },
            )
    except Exception as exc:
        checks["t1_t2_check_error"] = str(exc)

    # --- Native gate support check (hardware only) ---
    try:
        required_gates = ["cz", "rz", "sx", "x", "measure"]
        gate_support = check_native_gate_support(backend, required_gates)
        checks["native_gate_support"] = gate_support
        # Check for either CZ (Heron) or ECR (Eagle) as native 2Q gate
        has_native_2q = gate_support.get("cz", False) or gate_support.get("ecr", False)
        if not has_native_2q:
            checks["native_gate_warning"] = (
                "Neither CZ nor ECR found in native gate set. "
                "Transpiler may add overhead converting to native 2Q gate."
            )
            logger.log(
                "preflight_native_gate_warning",
                data={"gate_support": gate_support},
            )
    except Exception as exc:
        checks["native_gate_check_error"] = str(exc)

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
    - Adaptive: uses PEA threshold for abort (since PEA handles higher gate
      counts); warns at GF threshold.

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
        # Adaptive uses PEA threshold for abort (since PEA handles higher gate
        # counts); warns at GF threshold.
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


def validate_transpiled_circuit_quality(
    transpiled_circuit,
    backend,
    layout: list[int] | None,
    logger: StructuredLogger,
    *,
    error_budget_abort_threshold: float = 0.50,
    error_budget_warn_threshold: float = 0.30,
    depth_2q_warn_threshold: int = 30,
    defective_edge_threshold: float = 0.10,
) -> dict[str, Any]:
    """Validate transpiled circuit quality before QPU submission.

    Runs AFTER transpilation and layout selection but BEFORE job submission.
    Uses ResourceEstimation-derived metrics (depth_2q, count_ops) combined
    with calibration data to predict whether the circuit will produce usable
    results.

    Checks:
      1. Error budget (Σ n_gate_i × ε_i from calibration) — abort if > 0.50
      2. depth_2q — warn if critical path through 2Q gates is too long
      3. Defective edges — abort if layout uses edges with error > threshold
      4. Active qubits — sanity check that routing didn't expand circuit

    Parameters
    ----------
    transpiled_circuit : QuantumCircuit
        The transpiled (ISA) circuit for one layout.
    backend : BackendV2
        Backend with calibration data.
    layout : list[int] | None
        Physical qubit indices for this layout.
    logger : StructuredLogger
        Logger for recording check events.
    error_budget_abort_threshold : float
        Abort if error budget exceeds this (default: 0.50, F < 60%).
    error_budget_warn_threshold : float
        Warn if error budget exceeds this (default: 0.30, F < 74%).
    depth_2q_warn_threshold : int
        Warn if depth_2q exceeds this (default: 30 layers).
    defective_edge_threshold : float
        An edge is "defective" if its error rate exceeds this (default: 10%).

    Returns
    -------
    dict[str, Any]
        Check results with "abort" boolean if quality is insufficient.
    """
    import numpy as np

    checks: dict[str, Any] = {"abort": False}

    # ── depth_2q check ──
    try:
        depth_2q = transpiled_circuit.depth(filter_function=lambda x: x.operation.num_qubits == 2)
        depth_total = transpiled_circuit.depth()
        n_2q = sum(1 for inst in transpiled_circuit.data if inst.operation.num_qubits == 2)
        checks["depth_2q"] = depth_2q
        checks["depth_total"] = depth_total
        checks["n_2q_gates"] = n_2q

        if depth_2q > depth_2q_warn_threshold:
            checks["depth_2q_warning"] = (
                f"depth_2q={depth_2q} exceeds {depth_2q_warn_threshold}. "
                f"Circuit may suffer significant decoherence during 2Q layers. "
                f"Consider p=1 or smaller N."
            )
            logger.log("preflight_depth_2q_warning", data=checks)
    except Exception as exc:
        checks["depth_2q_error"] = str(exc)

    # ── Error budget from calibration ──
    try:
        target = backend.target
        layout_set = set(layout) if layout else None

        # Collect per-gate-type error rate (layout-filtered)
        gate_errors: dict[str, float] = {}
        for gate_name in target.operation_names:
            try:
                qargs_list = target.qargs_for_operation_name(gate_name)
            except Exception:
                continue
            if qargs_list is None:
                continue
            errs = []
            for qa in qargs_list:
                if layout_set and not all(q in layout_set for q in qa):
                    continue
                try:
                    props = target[gate_name].get(qa)
                    if props and props.error is not None and props.error < 0.5:
                        errs.append(props.error)
                except Exception:
                    continue
            if errs:
                gate_errors[gate_name] = float(np.mean(errs))

        # Count ops in circuit
        count_ops: dict[str, int] = {}
        for inst in transpiled_circuit.data:
            name = inst.operation.name
            count_ops[name] = count_ops.get(name, 0) + 1

        # Compute budget
        error_budget = 0.0
        for gate, count in count_ops.items():
            rate = gate_errors.get(gate, 0.0)
            error_budget += count * rate

        fidelity_estimate = float(np.exp(-error_budget))
        checks["error_budget"] = error_budget
        checks["fidelity_estimate"] = fidelity_estimate
        checks["error_budget_source"] = "calibration"

        if error_budget > error_budget_abort_threshold:
            checks["abort"] = True
            checks["abort_reason"] = (
                f"Error budget {error_budget:.3f} exceeds {error_budget_abort_threshold} "
                f"(predicted fidelity {fidelity_estimate:.1%}). "
                f"Circuit is too deep for this calibration state."
            )
            logger.log("preflight_error_budget_abort", data=checks)
        elif error_budget > error_budget_warn_threshold:
            checks["error_budget_warning"] = (
                f"Error budget {error_budget:.3f} is elevated "
                f"(predicted fidelity {fidelity_estimate:.1%}). "
                f"PEA-ZNE recommended for recovery."
            )
            logger.log("preflight_error_budget_warning", data=checks)
    except Exception as exc:
        checks["error_budget_error"] = str(exc)

    # ── Defective edge detection ──
    if layout is not None:
        try:
            layout_set = set(layout)
            defective_edges: list[tuple] = []
            for gate_name in ["cz", "ecr", "cx"]:
                if gate_name not in target.operation_names:
                    continue
                try:
                    qargs_list = target.qargs_for_operation_name(gate_name)
                except Exception:
                    continue
                if qargs_list is None:
                    continue
                for qa in qargs_list:
                    if len(qa) != 2:
                        continue
                    if qa[0] in layout_set and qa[1] in layout_set:
                        props = target[gate_name].get(qa)
                        if props and props.error is not None:
                            if props.error > defective_edge_threshold:
                                defective_edges.append((qa[0], qa[1], props.error))

            checks["defective_edges_in_layout"] = len(defective_edges)
            if defective_edges:
                checks["abort"] = True
                worst = max(defective_edges, key=lambda x: x[2])
                checks["abort_reason"] = (
                    f"{len(defective_edges)} edge(s) in layout have error > "
                    f"{defective_edge_threshold:.0%}. "
                    f"Worst: qubits ({worst[0]},{worst[1]}) error={worst[2]:.3f}. "
                    f"Select a different layout or wait for recalibration."
                )
                checks["defective_edge_details"] = [
                    {"q0": e[0], "q1": e[1], "error": e[2]} for e in defective_edges
                ]
                logger.log("preflight_defective_edges", data=checks)
        except Exception as exc:
            checks["defective_edge_check_error"] = str(exc)

    # ── Active qubits sanity ──
    try:
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import ResourceEstimation

        re_pm = PassManager([ResourceEstimation()])
        re_pm.run(transpiled_circuit)
        prop = re_pm.property_set
        width = prop.get("width")
        num_tf = prop.get("num_tensor_factors")
        if width is not None and num_tf is not None:
            active_qubits = width - num_tf + 1
            checks["active_qubits"] = active_qubits
            expected_n = len(layout) if layout else None
            if expected_n and active_qubits > expected_n:
                checks["routing_expansion_warning"] = (
                    f"Transpiled circuit uses {active_qubits} active qubits "
                    f"(expected {expected_n}). "
                    f"Routing may have introduced SWAP chains to non-layout qubits."
                )
                logger.log("preflight_routing_expansion", data=checks)
    except Exception:
        pass  # ResourceEstimation is best-effort

    logger.log("preflight_transpiled_quality", data=checks)
    return checks


@dataclass
class QPUCostEstimate:
    """Estimated QPU cost for a hardware deployment run.

    All times in seconds. Use this to verify the run fits within
    IBM's max_execution_time limits before submitting jobs.

    The estimate includes depth-aware CLOPS scaling, amortized PEA noise
    learning, classical latency overhead, and optimistic/pessimistic SPSA
    scenarios for realistic budget planning.

    Fields
    ------
    est_total_s : float
        Expected total time (SPSA weighted by trigger probability).
    est_total_optimistic_s : float
        Total time if SPSA never triggers (best case, ~70% likely).
    est_total_pessimistic_s : float
        Total time if SPSA triggers on every h-point (worst case).
    fits_per_job : bool
        Whether a single h-point (optimistic, no SPSA) fits per-job timeout.
    fits_full_sweep_10min : bool
        Whether the full optimistic sweep fits within 600s.
    effective_clops : int
        Depth/N-aware CLOPS used for this estimate (not the fixed reference).
    pea_noise_learning_s : float
        One-time PEA noise learning cost (amortized, not per-h-point).
    classical_latency_s : float
        Total classical overhead (compilation, deserialization, network).
    """

    n_h_points: int
    circuits_per_h: int
    shots_per_h: int
    total_circuits: int
    total_shots: int
    est_time_per_h_s: float
    est_total_s: float
    est_total_optimistic_s: float
    est_total_pessimistic_s: float
    max_execution_time_s: int
    fits_per_job: bool
    fits_full_sweep_10min: bool
    amplifier: str
    estimated_clops: int
    effective_clops: int
    pea_noise_learning_s: float
    classical_latency_s: float
    time_per_circuit_s: float
    spsa_per_h_if_triggered_s: float


# ═══════════════════════════════════════════════════════════════════════════════
# QPU Throughput Model (reusable, backend-agnostic)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class QPUThroughputProfile:
    """Throughput profile for a specific IBM QPU generation.

    Encapsulates all hardware-specific CLOPS characteristics so the cost
    estimator can be reused across different backends without modification.

    Parameters
    ----------
    name : str
        Human-readable label (e.g., "ibm_torino", "heron_r2", "nighthawk").
    base_clops : int
        Reference CLOPS at (ref_n_qubits, ref_depth).
    ref_n_qubits : int
        Qubit count for the reference CLOPS measurement.
    ref_depth : int
        Circuit depth for the reference CLOPS measurement.
    width_exponent : float
        Scaling exponent for qubit count. Higher = more penalty for wider circuits.
    depth_exponent : float
        Scaling exponent for circuit depth. Higher = more penalty for deeper circuits.
    total_qubits : int
        Total number of physical qubits on the device (for utilization ratio).
    clops_floor : int
        Minimum CLOPS (heavily-utilized circuits can't go below this).
    clops_ceiling : int
        Maximum CLOPS (hardware clock limits even for trivial circuits).
    classical_latency_per_job_s : float
        Compilation + network + deserialization overhead per job submission.

    Examples
    --------
    >>> torino = QPUThroughputProfile.ibm_torino()
    >>> torino.estimate_clops(n_qubits=10, circuit_depth=25)
    2500

    >>> nighthawk = QPUThroughputProfile.ibm_nighthawk()
    >>> nighthawk.estimate_clops(n_qubits=10, circuit_depth=25)
    5000
    """

    name: str
    base_clops: int
    ref_n_qubits: int
    ref_depth: int
    width_exponent: float
    depth_exponent: float
    total_qubits: int
    clops_floor: int
    clops_ceiling: int
    classical_latency_per_job_s: float

    def estimate_clops(
        self,
        n_qubits: int,
        circuit_depth: int | None = None,
        cx_count: int | None = None,
    ) -> int:
        """Estimate effective CLOPS for given circuit parameters.

        Uses a power-law scaling model from the reference point:
            CLOPS(N, D) = base_clops × (ref_N / N)^α × (ref_D / D)^β

        Parameters
        ----------
        n_qubits : int
            Number of qubits in the circuit.
        circuit_depth : int | None
            Circuit depth. If None, estimated from cx_count or n_qubits.
        cx_count : int | None
            Known 2Q gate count. Used to estimate depth if depth not given.

        Returns
        -------
        int
            Estimated effective CLOPS, clamped to [floor, ceiling].
        """
        if circuit_depth is None:
            if cx_count is not None:
                # depth ≈ 1.4 × n_cx (empirical average parallelism for HVA)
                circuit_depth = max(10, int(cx_count * 1.4))
            else:
                cx_est = _interpolate_cx_count(n_qubits)
                circuit_depth = max(10, int(cx_est * 1.4))

        width_factor = (self.ref_n_qubits / max(n_qubits, 1)) ** self.width_exponent
        depth_factor = (self.ref_depth / max(circuit_depth, 1)) ** self.depth_exponent

        effective = int(self.base_clops * width_factor * depth_factor)
        return max(self.clops_floor, min(self.clops_ceiling, effective))

    def time_per_circuit(self, shots: int, n_qubits: int, **kwargs) -> float:
        """Compute seconds per circuit execution for given shot count.

        Parameters
        ----------
        shots : int
            Number of shots per circuit.
        n_qubits : int
            Number of qubits (for CLOPS scaling).
        **kwargs
            Passed to estimate_clops (circuit_depth, cx_count).

        Returns
        -------
        float
            Seconds for one circuit execution at the given shot budget.
        """
        clops = self.estimate_clops(n_qubits, **kwargs)
        return shots / clops

    @classmethod
    def ibm_torino(cls) -> QPUThroughputProfile:
        """IBM Torino (Heron r1, 133 qubits, 2024-2026).

        Conservative profile validated against FakeTorino benchmarks.
        Width scaling mild (sparse utilization at N≤20, penalty above N=60).
        Depth scaling moderate (shallow HVA benefits from less T1 decay).
        """
        return cls(
            name="ibm_torino",
            base_clops=2500,
            ref_n_qubits=10,
            ref_depth=25,
            width_exponent=0.3,
            depth_exponent=0.4,
            total_qubits=133,
            clops_floor=1000,
            clops_ceiling=15000,
            classical_latency_per_job_s=15.0,
        )

    @classmethod
    def ibm_heron_r2(cls) -> QPUThroughputProfile:
        """IBM Heron r2 (2025+, 156 qubits, CZ native, TLS mitigation).

        Based on ibm_kingston specs (April 2025):
        - CLOPS: 340K (published), effective ~3750 at our circuit depths
        - Native 2Q: CZ (1.95E-3 median error, 8.28E-4 best)
        - T1: 258.88 μs, T2: 131.6 μs (median)
        - Heavy-hex topology, 156 qubits

        CZ-native means our HVA ZZ terms transpile directly without
        ECR decomposition overhead → ~30% fewer 2Q gates post-transpilation.
        """
        return cls(
            name="ibm_heron_r2",
            base_clops=3750,
            ref_n_qubits=10,
            ref_depth=25,
            width_exponent=0.3,
            depth_exponent=0.4,
            total_qubits=156,
            clops_floor=1500,
            clops_ceiling=20000,
            classical_latency_per_job_s=12.0,
        )

    @classmethod
    def ibm_kingston(cls) -> QPUThroughputProfile:
        """IBM Kingston (Heron r2, 156 qubits, US-East, available April 2025).

        Alias for ibm_heron_r2 with Kingston-specific naming.
        """
        profile = cls.ibm_heron_r2()
        profile.name = "ibm_kingston"
        return profile

    @classmethod
    def ibm_nighthawk(cls) -> QPUThroughputProfile:
        """IBM Nighthawk (2027+, projected specs: 3× EPLG, T1=350μs).

        Projected profile for next-gen hardware. Square lattice topology
        eliminates most SWAP routing → lower effective depth.
        """
        return cls(
            name="ibm_nighthawk",
            base_clops=5000,
            ref_n_qubits=10,
            ref_depth=20,
            width_exponent=0.25,
            depth_exponent=0.35,
            total_qubits=200,
            clops_floor=2000,
            clops_ceiling=30000,
            classical_latency_per_job_s=10.0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SPSA Cost Model (reusable)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SPSACostModel:
    """SPSA conditional refinement cost model.

    Encapsulates SPSA trigger probability and iteration budget so the
    cost estimator can adapt to different VQE quality assumptions.

    Parameters
    ----------
    trigger_probability : float
        Probability that SPSA refinement is needed (ΔE/gap > threshold).
        Default 0.30 for well-trained MPNN. Use 0.0 to disable, 1.0 for
        worst-case planning, or higher (0.5-0.8) for untested configs.
    n_iterations : int
        SPSA iterations if triggered (validated: 200 from V7-4A).
    evals_per_iteration : int
        Function evaluations per SPSA step (always 2 for gradient estimate).
    """

    trigger_probability: float = 0.30
    n_iterations: int = 200
    evals_per_iteration: int = 2

    def cost_if_triggered(self, time_per_circuit_s: float) -> float:
        """QPU seconds consumed if SPSA triggers on one h-point."""
        return self.n_iterations * self.evals_per_iteration * time_per_circuit_s

    def expected_cost(self, time_per_circuit_s: float) -> float:
        """Expected QPU seconds (probability × cost_if_triggered)."""
        return self.trigger_probability * self.cost_if_triggered(time_per_circuit_s)

    @classmethod
    def disabled(cls) -> SPSACostModel:
        """No SPSA refinement (confident MPNN predictions)."""
        return cls(trigger_probability=0.0)

    @classmethod
    def conservative(cls) -> SPSACostModel:
        """Conservative: 50% trigger rate (first-time deployment)."""
        return cls(trigger_probability=0.50, n_iterations=200)

    @classmethod
    def aggressive(cls) -> SPSACostModel:
        """Worst-case budget planning: always triggers."""
        return cls(trigger_probability=1.0, n_iterations=200)


# ═══════════════════════════════════════════════════════════════════════════════
# CX Count Interpolation (reusable across cost models)
# ═══════════════════════════════════════════════════════════════════════════════

# Empirical CX counts for HVA p=1 on heavy_hex (from transpiler audit).
# Used to estimate circuit depth when actual circuit is not available.
# Extend this dict as new N values are validated.
_HVA_P1_CX_COUNTS: dict[int, int] = {
    4: 6,
    6: 10,
    8: 14,
    10: 18,
    12: 22,
    16: 30,
    20: 38,
    40: 78,
    50: 98,
    80: 158,
}


def _interpolate_cx_count(n_qubits: int) -> int:
    """Interpolate CX gate count for HVA p=1 from known data points.

    Uses linear interpolation between bracketing known values, or
    linear extrapolation beyond the range. The data points in
    _HVA_P1_CX_COUNTS can be extended without modifying this function.
    """
    known = sorted(_HVA_P1_CX_COUNTS.items())

    if n_qubits <= known[0][0]:
        return known[0][1]
    if n_qubits >= known[-1][0]:
        n1, cx1 = known[-2]
        n2, cx2 = known[-1]
        slope = (cx2 - cx1) / (n2 - n1)
        return int(cx2 + slope * (n_qubits - n2))

    for i in range(len(known) - 1):
        n_lo, cx_lo = known[i]
        n_hi, cx_hi = known[i + 1]
        if n_lo <= n_qubits <= n_hi:
            frac = (n_qubits - n_lo) / (n_hi - n_lo)
            return int(cx_lo + frac * (cx_hi - cx_lo))

    return known[-1][1]


def estimate_effective_clops(
    n_qubits: int,
    circuit_depth: int | None = None,
    profile: QPUThroughputProfile | None = None,
) -> int:
    """Estimate effective CLOPS for a given system size and circuit depth.

    Convenience wrapper around QPUThroughputProfile.estimate_clops().
    Defaults to ibm_torino profile if none provided.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    circuit_depth : int | None
        Circuit depth (layers). If None, estimated from HVA p=1 CX count.
    profile : QPUThroughputProfile | None
        Hardware profile. Defaults to ibm_torino.

    Returns
    -------
    int
        Estimated effective CLOPS.
    """
    if profile is None:
        profile = QPUThroughputProfile.ibm_kingston()
    return profile.estimate_clops(n_qubits, circuit_depth=circuit_depth)


# ═══════════════════════════════════════════════════════════════════════════════
# QPU Cost Estimator (composable, scalable)
# ═══════════════════════════════════════════════════════════════════════════════


def estimate_qpu_cost(
    config: HardwareConfig,
    n_h_points: int = 4,
    include_spsa: bool = True,
    circuit_depth: int | None = None,
    cx_count: int | None = None,
    profile: QPUThroughputProfile | None = None,
    spsa_model: SPSACostModel | None = None,
) -> QPUCostEstimate:
    """Estimate QPU time and shot budget for a hardware deployment.

    Composable cost estimator that accepts pluggable hardware profiles
    and SPSA models. Use this for budget planning before committing QPU time.

    Features:
    1. Depth-aware CLOPS — smaller circuits run faster per shot.
    2. Amortized PEA noise learning — one-time cost, not per-h-point.
    3. Optimistic/pessimistic SPSA — show both scenarios for budget planning.
    4. Classical latency — compilation + network overhead per job.
    5. Pluggable hardware profiles — same code for Torino, Heron, Nighthawk.
    6. Configurable SPSA model — adjust trigger probability per scenario.

    Parameters
    ----------
    config : HardwareConfig
        Hardware configuration (shots, n_layouts, amplifier, etc.)
    n_h_points : int
        Number of h-values to evaluate (default: 4).
    include_spsa : bool
        Include SPSA overhead in estimates (default: True).
    circuit_depth : int | None
        Known circuit depth. If None, estimated from cx_count or n_qubits.
    cx_count : int | None
        Known 2Q gate count. Used to estimate depth if depth not given.
    profile : QPUThroughputProfile | None
        Hardware throughput profile. Defaults to ibm_torino.
    spsa_model : SPSACostModel | None
        SPSA cost model. Defaults to P(trigger)=0.30, 200 iters.

    Returns
    -------
    QPUCostEstimate
        Detailed cost breakdown with optimistic/pessimistic/expected times.

    Examples
    --------
    >>> from qmbp_simulation.execution.hardware.config import HardwareConfig
    >>> config = HardwareConfig(n_qubits=10, shots=16384, n_layouts=3)
    >>> est = estimate_qpu_cost(config, n_h_points=3)
    >>> print(f"Optimistic: {est.est_total_optimistic_s:.0f}s")
    >>> print(f"Expected:   {est.est_total_s:.0f}s")
    >>> print(f"Pessimistic: {est.est_total_pessimistic_s:.0f}s")

    >>> # Planning for Nighthawk with conservative SPSA
    >>> est_nh = estimate_qpu_cost(
    ...     config, n_h_points=3,
    ...     profile=QPUThroughputProfile.ibm_nighthawk(),
    ...     spsa_model=SPSACostModel.conservative(),
    ... )
    """
    if profile is None:
        profile = QPUThroughputProfile.ibm_kingston()
    if spsa_model is None:
        spsa_model = SPSACostModel() if include_spsa else SPSACostModel.disabled()

    shots = config.shots
    n_layouts = config.n_layouts
    n_qubits = config.n_qubits
    amplifier = config.mitigation.zne_amplifier

    # ── 1. Depth-aware CLOPS ─────────────────────────────────────────────────
    effective_clops = profile.estimate_clops(
        n_qubits, circuit_depth=circuit_depth, cx_count=cx_count
    )
    time_per_circuit_s = shots / effective_clops

    # ── 2. Amplifier-specific circuit count ──────────────────────────────────
    if amplifier == "pea":
        n_noise_factors = 3
    elif amplifier == "adaptive":
        # Worst case: GF(3) fails + PEA(3) runs
        n_noise_factors = 6
    else:  # gate_folding
        n_noise_factors = 3

    circuits_per_h = n_layouts * n_noise_factors
    shots_per_h = circuits_per_h * shots

    # ZNE energy evaluation time (pure QPU, no overhead yet)
    zne_qpu_per_h = circuits_per_h * time_per_circuit_s

    # ── 3. Amortized PEA noise learning ──────────────────────────────────────
    # PEA runs LayerNoiseLearning ONCE per unique circuit structure:
    #   num_randomizations circuits, each with shots_per_randomization shots.
    # This cost is amortized across all h-points (same HVA structure).
    pea_noise_learning_s = 0.0
    if amplifier in ("pea", "adaptive"):
        num_rand = config.mitigation.num_randomizations  # default: 32
        shots_rand = config.mitigation.shots_per_randomization  # default: 128
        pea_noise_learning_s = num_rand * (shots_rand / effective_clops)

    # ── 4. Observable measurement ────────────────────────────────────────────
    # Two measurement groups (⟨X⟩ basis, ⟨ZZ⟩ basis) per h-point.
    # EstimatorV2 batches PUBs in same job → overhead ≈ 1 job submission
    # but 2× the shot budget for measurements.
    obs_qpu_per_h = 2 * time_per_circuit_s

    # ── 5. SPSA conditional refinement ───────────────────────────────────────
    spsa_per_h_if_triggered = spsa_model.cost_if_triggered(time_per_circuit_s)
    spsa_expected_per_h = spsa_model.expected_cost(time_per_circuit_s)

    # ── 6. Classical latency budget ──────────────────────────────────────────
    latency = profile.classical_latency_per_job_s
    jobs_per_h_optimistic = 2  # ZNE + observables
    jobs_per_h_pessimistic = 3  # ZNE + observables + SPSA
    classical_per_h_optimistic = jobs_per_h_optimistic * latency
    classical_per_h_pessimistic = jobs_per_h_pessimistic * latency
    # PEA noise learning is an additional job (one-time)
    classical_pea_startup = latency if amplifier in ("pea", "adaptive") else 0.0

    # ── 7. Assemble per-h-point times ────────────────────────────────────────
    per_h_optimistic = zne_qpu_per_h + obs_qpu_per_h + classical_per_h_optimistic

    per_h_pessimistic = (
        zne_qpu_per_h + obs_qpu_per_h + spsa_per_h_if_triggered + classical_per_h_pessimistic
    )

    p = spsa_model.trigger_probability
    per_h_expected = (
        zne_qpu_per_h
        + obs_qpu_per_h
        + spsa_expected_per_h
        + classical_per_h_optimistic * (1 - p)
        + classical_per_h_pessimistic * p
    )

    # ── 8. Total sweep times ─────────────────────────────────────────────────
    startup = pea_noise_learning_s + classical_pea_startup

    total_optimistic = startup + n_h_points * per_h_optimistic
    total_pessimistic = startup + n_h_points * per_h_pessimistic
    total_expected = startup + n_h_points * per_h_expected

    # ── 9. Budget checks ─────────────────────────────────────────────────────
    max_exec = config.job_timeout_s
    fits_per_job = max_exec is None or per_h_optimistic < max_exec
    fits_full_sweep_10min = total_optimistic < 600

    return QPUCostEstimate(
        n_h_points=n_h_points,
        circuits_per_h=circuits_per_h,
        shots_per_h=shots_per_h,
        total_circuits=n_h_points * circuits_per_h,
        total_shots=n_h_points * shots_per_h,
        est_time_per_h_s=per_h_expected,
        est_total_s=total_expected,
        est_total_optimistic_s=total_optimistic,
        est_total_pessimistic_s=total_pessimistic,
        max_execution_time_s=max_exec,
        fits_per_job=fits_per_job,
        fits_full_sweep_10min=fits_full_sweep_10min,
        amplifier=amplifier,
        estimated_clops=profile.base_clops,
        effective_clops=effective_clops,
        pea_noise_learning_s=pea_noise_learning_s,
        classical_latency_s=startup + n_h_points * classical_per_h_optimistic,
        time_per_circuit_s=time_per_circuit_s,
        spsa_per_h_if_triggered_s=spsa_per_h_if_triggered,
    )
