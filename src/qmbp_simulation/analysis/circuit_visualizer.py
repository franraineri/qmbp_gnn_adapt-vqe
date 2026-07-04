"""Circuit visualization utilities for hardware deployment.

Provides reusable functions to print, save, and export quantum circuits
at different stages (logical, transpiled, bound) for debugging and thesis
documentation.

Usage:
    from qmbp_simulation.analysis.circuit_visualizer import (
        print_circuit,
        save_circuit_diagram,
        circuit_summary,
    )

    # Quick terminal print
    print_circuit(circuit, params=theta, title="HVA p=1 N=10")

    # Save as PNG/PDF for thesis
    save_circuit_diagram(transpiled_circuit, "figures/hva_transpiled.png")

    # Get text summary (gate counts, depth, etc.)
    info = circuit_summary(circuit)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from qiskit.circuit import QuantumCircuit


def print_circuit(
    circuit: QuantumCircuit,
    params: np.ndarray | None = None,
    title: str | None = None,
    style: str = "text",
    fold: int = 120,
) -> str:
    """Print a quantum circuit to the terminal and return the string.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to visualize. Can be parameterized or bound.
    params : np.ndarray | None
        If provided and circuit is parameterized, bind these values first.
    title : str | None
        Optional title printed above the circuit.
    style : str
        Drawing style: "text" (ASCII, terminal-friendly) or "mpl" (matplotlib).
    fold : int
        Line width for text mode (default 120 chars).

    Returns
    -------
    str
        The text representation of the circuit.
    """
    qc = circuit
    if params is not None and circuit.num_parameters > 0:
        qc = circuit.assign_parameters(params)

    if title:
        print(f"\n{'─' * 60}")
        print(f"  {title}")
        print(f"{'─' * 60}")

    if style == "text":
        text = qc.draw(output="text", fold=fold).__str__()
        print(text)
        return text
    elif style == "mpl":
        # For matplotlib, display in notebook or save
        qc.draw(output="mpl", fold=fold)
        return f"[matplotlib figure displayed for {qc.num_qubits}q circuit]"
    else:
        raise ValueError(f"Unknown style: {style}. Use 'text' or 'mpl'.")


def save_circuit_diagram(
    circuit: QuantumCircuit,
    output_path: str | Path,
    params: np.ndarray | None = None,
    title: str | None = None,
    fold: int = 30,
    dpi: int = 300,
    style: str | None = None,
) -> Path:
    """Save a circuit diagram as PNG or PDF.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to visualize.
    output_path : str | Path
        Output file path (.png, .pdf, .svg supported).
    params : np.ndarray | None
        If provided and circuit is parameterized, bind these values first.
    title : str | None
        Title for the figure (matplotlib only).
    fold : int
        Gates per line before folding (default 30 for readability).
    dpi : int
        Resolution for raster formats (default 300 for thesis quality).
    style : str | None
        Matplotlib style. None uses Qiskit default. Options: "iqp", "bw", "clifford".

    Returns
    -------
    Path
        The path where the diagram was saved.
    """
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for saving

    qc = circuit
    if params is not None and circuit.num_parameters > 0:
        qc = circuit.assign_parameters(params)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = qc.draw(
        output="mpl",
        fold=fold,
        style=style or {"backgroundcolor": "#FFFFFF"},
    )
    if title:
        fig.suptitle(title, fontsize=12, y=0.98)

    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)

    return output_path


def circuit_summary(
    circuit: QuantumCircuit,
    params: np.ndarray | None = None,
) -> dict[str, Any]:
    """Get a comprehensive summary of circuit properties.

    Returns gate counts, depth, parameter info, and qubit usage — useful for
    logging, preflight checks, and thesis tables.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to analyze.
    params : np.ndarray | None
        If provided, includes bound parameter values in summary.

    Returns
    -------
    dict
        Summary with keys: n_qubits, depth, n_params, gate_counts, n_2q, n_1q, etc.
    """
    qc = circuit
    if params is not None and circuit.num_parameters > 0:
        qc = circuit.assign_parameters(params)

    # Gate counts by type
    gate_counts: dict[str, int] = {}
    n_2q = 0
    n_1q = 0
    for inst in qc.data:
        name = inst.operation.name
        nq = inst.operation.num_qubits
        gate_counts[name] = gate_counts.get(name, 0) + 1
        if nq == 2:
            n_2q += 1
        elif nq == 1:
            n_1q += 1

    summary = {
        "n_qubits": qc.num_qubits,
        "n_clbits": qc.num_clbits,
        "depth": qc.depth(),
        "n_parameters": circuit.num_parameters,
        "n_gates_total": len(qc.data),
        "n_2q_gates": n_2q,
        "n_1q_gates": n_1q,
        "gate_counts": gate_counts,
    }

    if params is not None:
        summary["params"] = params.tolist() if hasattr(params, "tolist") else list(params)
        summary["params_norm"] = float(np.linalg.norm(params))

    return summary


def print_circuit_comparison(
    logical: QuantumCircuit,
    transpiled: QuantumCircuit,
    params: np.ndarray | None = None,
    title: str = "Circuit Comparison (Logical vs Transpiled)",
) -> dict[str, Any]:
    """Print side-by-side comparison of logical and transpiled circuits.

    Shows how transpilation changes gate counts, depth, and qubit usage.
    Useful for understanding hardware overhead before QPU submission.

    Parameters
    ----------
    logical : QuantumCircuit
        Original parameterized circuit (before transpilation).
    transpiled : QuantumCircuit
        Transpiled circuit (ISA circuit, after layout + routing).
    params : np.ndarray | None
        Parameters to bind for the logical circuit display.
    title : str
        Section title.

    Returns
    -------
    dict
        Comparison metrics (depth_overhead, gate_overhead, etc.)
    """
    log_summary = circuit_summary(logical, params)
    trans_summary = circuit_summary(transpiled)

    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")
    print(f"  {'Property':<25} {'Logical':>10} {'Transpiled':>12} {'Overhead':>10}")
    print(f"  {'─' * 25} {'─' * 10} {'─' * 12} {'─' * 10}")
    print(
        f"  {'Qubits':<25} {log_summary['n_qubits']:>10} {trans_summary['n_qubits']:>12} "
        f"{trans_summary['n_qubits'] - log_summary['n_qubits']:>+10}"
    )
    print(
        f"  {'Depth':<25} {log_summary['depth']:>10} {trans_summary['depth']:>12} "
        f"{trans_summary['depth'] - log_summary['depth']:>+10}"
    )
    print(
        f"  {'2Q gates':<25} {log_summary['n_2q_gates']:>10} {trans_summary['n_2q_gates']:>12} "
        f"{trans_summary['n_2q_gates'] - log_summary['n_2q_gates']:>+10}"
    )
    print(
        f"  {'1Q gates':<25} {log_summary['n_1q_gates']:>10} {trans_summary['n_1q_gates']:>12} "
        f"{trans_summary['n_1q_gates'] - log_summary['n_1q_gates']:>+10}"
    )
    print(
        f"  {'Total gates':<25} {log_summary['n_gates_total']:>10} "
        f"{trans_summary['n_gates_total']:>12} "
        f"{trans_summary['n_gates_total'] - log_summary['n_gates_total']:>+10}"
    )
    print()

    # Gate breakdown for transpiled
    print("  Transpiled gate breakdown:")
    for gate, count in sorted(trans_summary["gate_counts"].items(), key=lambda x: -x[1]):
        print(f"    {gate:<12} {count:>5}")
    print()

    return {
        "logical": log_summary,
        "transpiled": trans_summary,
        "depth_overhead": trans_summary["depth"] - log_summary["depth"],
        "gate_2q_overhead": trans_summary["n_2q_gates"] - log_summary["n_2q_gates"],
        "qubit_overhead": trans_summary["n_qubits"] - log_summary["n_qubits"],
    }


def _layer_gap(dag, node_a, node_b) -> int:
    """Compute idle cycles between two consecutive ops on the same wire.

    Returns the number of DAG layers between node_a and node_b minus 1
    (i.e., the number of idle cycles between two ops on a qubit).
    Uses topological sorting to assign layer indices to nodes.
    """
    # Build a map from node._node_id to its layer index
    # dag.layers() returns layers in topological order
    node_to_layer: dict[int, int] = {}
    for layer_idx, layer in enumerate(dag.layers()):
        for node in layer["graph"].op_nodes():
            node_to_layer[node._node_id] = layer_idx

    layer_a = node_to_layer.get(node_a._node_id, 0)
    layer_b = node_to_layer.get(node_b._node_id, 0)
    gap = layer_b - layer_a - 1
    return max(0, gap)


def _compute_idle_metrics(circuit: QuantumCircuit) -> dict[str, Any]:
    """Compute idle-time metrics via DAG analysis.

    Analyzes the circuit DAG to determine how many cycles each qubit
    spends idle (not participating in any gate), and the longest
    consecutive idle stretch on any qubit.

    Parameters
    ----------
    circuit : QuantumCircuit
        A transpiled circuit to analyze.

    Returns
    -------
    dict[str, Any]
        idle_cycles_per_qubit: float average idle cycles per active qubit.
        max_idle_stretch: int longest consecutive idle on any qubit.
    """
    from qiskit.converters import circuit_to_dag

    dag = circuit_to_dag(circuit)
    idle_wires = set(dag.idle_wires())
    active_qubits = [q for q in dag.qubits if q not in idle_wires]

    if not active_qubits:
        return {"idle_cycles_per_qubit": 0.0, "max_idle_stretch": 0}

    # Pre-compute node-to-layer mapping once for efficiency
    node_to_layer: dict[int, int] = {}
    for layer_idx, layer in enumerate(dag.layers()):
        for node in layer["graph"].op_nodes():
            node_to_layer[node._node_id] = layer_idx

    total_idle = 0
    max_stretch = 0
    depth = circuit.depth()

    for qubit in active_qubits:
        nodes = list(dag.nodes_on_wire(qubit, only_ops=True))
        if len(nodes) <= 1:
            idle_for_qubit = max(0, depth - 1)
            total_idle += idle_for_qubit
            max_stretch = max(max_stretch, idle_for_qubit)
            continue
        # Compute gaps between consecutive ops using pre-built layer map
        stretches = []
        for i in range(len(nodes) - 1):
            layer_a = node_to_layer.get(nodes[i]._node_id, 0)
            layer_b = node_to_layer.get(nodes[i + 1]._node_id, 0)
            gap = max(0, layer_b - layer_a - 1)
            stretches.append(gap)
        total_idle += sum(stretches)
        if stretches:
            max_stretch = max(max_stretch, max(stretches))

    n_active = len(active_qubits)
    return {
        "idle_cycles_per_qubit": total_idle / n_active if n_active > 0 else 0.0,
        "max_idle_stretch": max_stretch,
    }


def transpiled_circuit_stats(circuit: QuantumCircuit) -> dict[str, Any]:
    """Unified resource statistics for a transpiled (ISA) circuit.

    Combines our ad-hoc gate counting with Qiskit's ResourceEstimation pass
    to produce a complete picture of circuit resources. This is the canonical
    function for extracting hardware-relevant metrics from a transpiled circuit.

    Metrics returned:
      - depth: total circuit depth (critical path through all gates)
      - depth_2q: critical path through 2-qubit gates only (strongest
        predictor of hardware error — 2Q gates dominate noise budget)
      - n_2q_gates, n_1q_gates, total_gates: gate counts
      - count_ops: per-gate-type breakdown (e.g. {cz: 18, rz: 64, sx: 56})
      - num_tensor_factors: disconnected sub-circuits (idle qubits + 1 active)
      - width: total physical qubits in transpiled circuit
      - active_qubits: width − num_tensor_factors + 1

    Parameters
    ----------
    circuit : QuantumCircuit
        A transpiled (ISA) circuit. Must already be bound (no free parameters).

    Returns
    -------
    dict[str, Any]
        Complete resource statistics.

    Examples
    --------
    >>> from qmbp_simulation.analysis.circuit_visualizer import transpiled_circuit_stats
    >>> stats = transpiled_circuit_stats(transpiled)
    >>> print(f"2Q-depth: {stats['depth_2q']}, CZ gates: {stats['count_ops'].get('cz', 0)}")
    """
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import ResourceEstimation

    # ── Fast ad-hoc counting (no DAG overhead) ──
    n_2q = sum(1 for inst in circuit.data if inst.operation.num_qubits == 2)
    n_1q = sum(1 for inst in circuit.data if inst.operation.num_qubits == 1)
    depth = circuit.depth()
    depth_2q = circuit.depth(filter_function=lambda x: x.operation.num_qubits == 2)

    stats: dict[str, Any] = {
        "depth": depth,
        "depth_2q": depth_2q,
        "n_2q_gates": n_2q,
        "n_1q_gates": n_1q,
        "total_gates": len(circuit.data),
    }

    # ── ResourceEstimation pass (count_ops + tensor factors) ──
    try:
        re_pm = PassManager([ResourceEstimation()])
        re_pm.run(circuit)
        prop = re_pm.property_set
        count_ops = prop.get("count_ops")
        stats["count_ops"] = dict(count_ops) if count_ops else {}
        num_tf = prop.get("num_tensor_factors")
        width = prop.get("width")
        stats["num_tensor_factors"] = num_tf
        stats["width"] = width
        if width is not None and num_tf is not None:
            stats["active_qubits"] = width - num_tf + 1
    except Exception:
        # Fallback: build count_ops manually
        gate_counts: dict[str, int] = {}
        for inst in circuit.data:
            name = inst.operation.name
            gate_counts[name] = gate_counts.get(name, 0) + 1
        stats["count_ops"] = gate_counts
        stats["num_tensor_factors"] = None
        stats["width"] = circuit.num_qubits
        stats["active_qubits"] = None

    # ── Idle/parallelism metrics (§6.1-6.5) ──
    idle_metrics = _compute_idle_metrics(circuit)
    stats["idle_cycles_per_qubit"] = idle_metrics["idle_cycles_per_qubit"]
    stats["max_idle_stretch"] = idle_metrics["max_idle_stretch"]
    n_2q_for_ratio = stats["n_2q_gates"]
    active = stats.get("active_qubits") or circuit.num_qubits
    stats["parallelism_ratio"] = n_2q_for_ratio / depth_2q if depth_2q > 0 else 0.0
    stats["gate_density_2q"] = (
        n_2q_for_ratio / (active * depth_2q) if (depth_2q > 0 and active > 0) else 0.0
    )

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Improvement 1: Error Budget from Calibration Data
# ══════════════════════════════════════════════════════════════════════════════


def compute_error_budget(
    circuit: QuantumCircuit,
    backend=None,
    layout: list[int] | None = None,
) -> dict[str, Any]:
    """Estimate hardware error budget from gate counts + calibration data.

    Computes the predicted total error and fidelity using:
      F_predicted ≈ exp(-Σ n_gate_i × ε_gate_i)

    When a real backend with calibration data is provided, uses actual error
    rates from the Target. Otherwise falls back to typical IBM Heron rates.

    Parameters
    ----------
    circuit : QuantumCircuit
        A transpiled (ISA) circuit. Must already be bound.
    backend : BackendV2 | None
        If provided, reads real calibration error rates from backend.target.
        If None, uses typical IBM Heron/Kingston error rates as fallback.
    layout : list[int] | None
        Physical qubit indices used. If provided, only averages error rates
        for edges within this layout (more accurate than chip-wide average).

    Returns
    -------
    dict[str, Any]
        Keys:
          - error_budget: total accumulated error probability
          - fidelity_estimate: exp(-error_budget)
          - per_gate_contribution: {gate_name: n × ε}
          - error_rates_used: {gate_name: ε} (source: "calibration" or "typical")
          - source: "calibration" | "typical_fallback"
          - depth_2q: critical-path 2Q depth (for decoherence estimation)
    """
    stats = transpiled_circuit_stats(circuit)
    count_ops = stats["count_ops"]
    depth_2q = stats["depth_2q"]

    # ── Get error rates ──
    error_rates: dict[str, float] = {}
    source = "typical_fallback"

    if backend is not None:
        source = "calibration"
        target = backend.target
        error_rates = _extract_error_rates_from_target(target, layout)

    # Fallback for any gate type not found in calibration
    TYPICAL_RATES = {
        "cz": 8e-3,
        "ecr": 8e-3,
        "cx": 8e-3,
        "sx": 2.5e-4,
        "x": 2.5e-4,
        "rz": 0.0,
        "id": 0.0,
        "delay": 0.0,
        "barrier": 0.0,
        "measure": 0.0,
    }
    for gate in count_ops:
        if gate not in error_rates:
            error_rates[gate] = TYPICAL_RATES.get(gate, 1e-4)

    # ── Compute budget ──
    per_gate_contribution: dict[str, float] = {}
    total_error = 0.0
    for gate, count in count_ops.items():
        rate = error_rates.get(gate, 0.0)
        contribution = count * rate
        per_gate_contribution[gate] = contribution
        total_error += contribution

    fidelity = float(np.exp(-total_error))

    return {
        "error_budget": total_error,
        "fidelity_estimate": fidelity,
        "per_gate_contribution": per_gate_contribution,
        "error_rates_used": error_rates,
        "source": source,
        "depth_2q": depth_2q,
        "n_2q_gates": stats["n_2q_gates"],
        "count_ops": count_ops,
    }


def _extract_error_rates_from_target(target, layout: list[int] | None = None) -> dict[str, float]:
    """Extract per-gate-type average error rates from a BackendV2 Target.

    Parameters
    ----------
    target : Target
        Qiskit Target object with calibration properties.
    layout : list[int] | None
        If provided, only average over edges/qubits in this layout.

    Returns
    -------
    dict[str, float]
        Average error rate per gate type.
    """
    layout_set = set(layout) if layout else None
    gate_errors: dict[str, list[float]] = {}

    for gate_name in target.operation_names:
        try:
            qargs_list = target.qargs_for_operation_name(gate_name)
        except Exception:
            continue
        if qargs_list is None:
            continue

        for qargs in qargs_list:
            # Filter by layout if provided
            if layout_set is not None:
                if not all(q in layout_set for q in qargs):
                    continue
            try:
                props = target[gate_name].get(qargs)
                if props is not None and props.error is not None:
                    gate_errors.setdefault(gate_name, []).append(props.error)
            except Exception:
                continue

    return {gate: sum(errs) / len(errs) for gate, errs in gate_errors.items() if errs}


# ══════════════════════════════════════════════════════════════════════════════
# Improvement 2: Depth-2Q vs ΔE/gap Correlation Tracker
# ══════════════════════════════════════════════════════════════════════════════


def build_error_prediction(
    circuit: QuantumCircuit,
    h_value: float,
    backend=None,
    layout: list[int] | None = None,
    kappa: float | None = None,
) -> dict[str, Any]:
    """Build a pre-execution error prediction for a given h-point.

    Call this BEFORE QPU execution. After hardware results come back,
    compare `prediction["fidelity_estimate"]` vs actual ΔE/gap to validate
    whether depth_2q and error_budget are good predictors.

    Parameters
    ----------
    circuit : QuantumCircuit
        Transpiled (ISA) circuit for this h-point.
    h_value : float
        The field strength being evaluated.
    backend : BackendV2 | None
        Backend with calibration data (for error rates).
    layout : list[int] | None
        Physical qubits used in this layout.
    kappa : float | None
        Landscape curvature from compute_kappa_per_h (if available).

    Returns
    -------
    dict[str, Any]
        Pre-execution prediction with keys:
          - h: field value
          - depth_2q: 2Q critical path
          - error_budget: predicted total error
          - fidelity_estimate: predicted circuit fidelity
          - kappa: landscape curvature (if provided)
          - predicted_risk: "low" | "medium" | "high" based on combined metrics
          - explanation: human-readable risk assessment
    """
    budget = compute_error_budget(circuit, backend=backend, layout=layout)

    # Combined risk from error_budget + kappa
    eb = budget["error_budget"]
    d2q = budget["depth_2q"]

    if eb > 0.40:
        risk = "high"
        explanation = f"Error budget {eb:.3f} exceeds 0.40 — ZNE unlikely to recover."
    elif eb > 0.25:
        risk = "medium"
        explanation = f"Error budget {eb:.3f} in marginal zone — PEA-ZNE recommended."
    else:
        risk = "low"
        explanation = f"Error budget {eb:.3f} well within perturbative regime."

    # Kappa refinement: low kappa + moderate error = higher risk
    if kappa is not None and kappa < 45 and risk == "medium":
        risk = "high"
        explanation += f" κ={kappa:.0f} (flat landscape near h_c) amplifies noise impact."

    return {
        "h": h_value,
        "depth_2q": d2q,
        "n_2q_gates": budget["n_2q_gates"],
        "error_budget": eb,
        "fidelity_estimate": budget["fidelity_estimate"],
        "kappa": kappa,
        "predicted_risk": risk,
        "explanation": explanation,
        "source": budget["source"],
    }


def validate_prediction_vs_result(
    prediction: dict[str, Any],
    actual_de_gap: float,
    actual_zne_r2: float | None = None,
) -> dict[str, Any]:
    """Compare pre-execution prediction with actual hardware result.

    Call this AFTER QPU execution with the measured ΔE/gap to build a
    correlation dataset for validating depth_2q as an error predictor.

    Parameters
    ----------
    prediction : dict
        Output from ``build_error_prediction``.
    actual_de_gap : float
        Measured ΔE/gap from hardware execution.
    actual_zne_r2 : float | None
        ZNE extrapolation R² (quality indicator).

    Returns
    -------
    dict[str, Any]
        Validation record with prediction, actual result, and correlation
        metrics. Accumulate these across h-points and seeds to build
        the depth_2q ↔ ΔE/gap correlation plot for the thesis.
    """
    predicted_risk = prediction["predicted_risk"]
    # Was our prediction correct?
    if actual_de_gap < 0.05:
        actual_outcome = "pass"
    elif actual_de_gap < 0.10:
        actual_outcome = "marginal"
    else:
        actual_outcome = "fail"

    risk_to_expected = {"low": "pass", "medium": "marginal", "high": "fail"}
    prediction_correct = risk_to_expected.get(predicted_risk) == actual_outcome

    return {
        "h": prediction["h"],
        "depth_2q": prediction["depth_2q"],
        "error_budget": prediction["error_budget"],
        "fidelity_estimate": prediction["fidelity_estimate"],
        "kappa": prediction.get("kappa"),
        "predicted_risk": predicted_risk,
        "actual_de_gap": actual_de_gap,
        "actual_zne_r2": actual_zne_r2,
        "actual_outcome": actual_outcome,
        "prediction_correct": prediction_correct,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Improvement 3: Layout Selection by depth_2q
# ══════════════════════════════════════════════════════════════════════════════


def rank_layouts_by_depth_2q(
    transpiled_circuits: list[QuantumCircuit],
    layouts: list[list[int]] | None = None,
) -> list[dict[str, Any]]:
    """Rank transpiled circuits by depth_2q (lowest = best for hardware).

    Use this to select the primary layout for ZNE — the layout with
    the lowest 2Q critical path will accumulate less decoherence error.

    Parameters
    ----------
    transpiled_circuits : list[QuantumCircuit]
        Pre-transpiled circuits (one per layout).
    layouts : list[list[int]] | None
        Corresponding physical qubit layouts (for logging).

    Returns
    -------
    list[dict[str, Any]]
        Per-layout stats sorted by depth_2q ascending (best first).
        Each entry has: layout_idx, depth_2q, depth, n_2q_gates, layout.
    """
    ranked = []
    for i, circ in enumerate(transpiled_circuits):
        depth_2q = circ.depth(filter_function=lambda x: x.operation.num_qubits == 2)
        depth = circ.depth()
        n_2q = sum(1 for inst in circ.data if inst.operation.num_qubits == 2)
        entry = {
            "layout_idx": i,
            "depth_2q": depth_2q,
            "depth": depth,
            "n_2q_gates": n_2q,
        }
        if layouts is not None and i < len(layouts):
            entry["layout"] = layouts[i]
        ranked.append(entry)

    ranked.sort(key=lambda x: (x["depth_2q"], x["n_2q_gates"]))
    return ranked


def select_best_layout_for_zne(
    transpiled_circuits: list[QuantumCircuit],
    layouts: list[list[int]] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Select the layout with lowest depth_2q as ZNE primary.

    Returns the index into transpiled_circuits and the ranking info.
    If all layouts have equal depth_2q, returns index 0 (CES-selected).

    Parameters
    ----------
    transpiled_circuits : list[QuantumCircuit]
        Pre-transpiled circuits from layout selection.
    layouts : list[list[int]] | None
        Physical qubit layouts.

    Returns
    -------
    tuple[int, dict]
        (best_index, ranking_info) where best_index is the position in
        the input list and ranking_info has all metrics.
    """
    ranked = rank_layouts_by_depth_2q(transpiled_circuits, layouts)
    best = ranked[0]
    return best["layout_idx"], best


# ══════════════════════════════════════════════════════════════════════════════
# Improvement 4: Decoherence Penalty & Advanced Transpilation Metrics
# ══════════════════════════════════════════════════════════════════════════════


def compute_decoherence_penalty(
    circuit: QuantumCircuit,
    backend=None,
    layout: list[int] | None = None,
    *,
    t_2q_gate_ns: float = 84.0,
    t_1q_gate_ns: float = 28.0,
) -> dict[str, Any]:
    """Compute decoherence penalty from idle qubits and circuit duration.

    Models T1-decay during circuit execution. Idle qubits lose coherence
    proportional to their idle time / T1. This estimates the total
    decoherence fraction that the error budget model (gate-error only) misses.

    Based on the insight from Ma et al. (arXiv:2411.15631) that T1/T2
    are the single most important predictor for simulator execution time,
    and the physical model: P(decay) = 1 - exp(-t_idle / T1).

    Parameters
    ----------
    circuit : QuantumCircuit
        Transpiled (ISA) circuit.
    backend : BackendV2 | None
        Backend with qubit_properties for T1/T2 data. If None, uses
        typical Heron r2 values (T1=258μs, T2=131μs).
    layout : list[int] | None
        Physical qubit indices. Used to select correct T1/T2 values.
    t_2q_gate_ns : float
        Duration of a 2Q gate in nanoseconds (Heron CZ: ~84ns).
    t_1q_gate_ns : float
        Duration of a 1Q gate in nanoseconds (Heron SX: ~28ns).

    Returns
    -------
    dict[str, Any]
        Keys:
          - circuit_duration_ns: estimated total circuit execution time
          - decoherence_fraction: mean P(decay) across active qubits
          - worst_qubit_decay: maximum single-qubit decoherence probability
          - t1_budget_ratio: circuit_duration / min_T1 (should be << 1)
          - idle_decoherence_budget: contribution of idle time to total error
          - depth_2q_time_ns: time spent in 2Q critical path
    """
    stats = transpiled_circuit_stats(circuit)
    idle_metrics = _compute_idle_metrics(circuit)
    depth = stats["depth"]
    depth_2q = stats["depth_2q"]
    n_2q = stats["n_2q_gates"]

    # Estimate circuit duration: depth_2q layers of 2Q gates + remaining 1Q layers
    depth_1q_only = max(0, depth - depth_2q)
    circuit_duration_ns = depth_2q * t_2q_gate_ns + depth_1q_only * t_1q_gate_ns
    depth_2q_time_ns = depth_2q * t_2q_gate_ns

    # Get T1/T2 values for layout qubits
    t1_values_s: list[float] = []
    t2_values_s: list[float] = []
    TYPICAL_T1_S = 258e-6  # Heron r2 median
    TYPICAL_T2_S = 131e-6  # Heron r2 median

    if backend is not None and layout is not None:
        try:
            qubit_props = backend.target.qubit_properties
            if qubit_props is not None:
                for phys_q in layout:
                    if phys_q < len(qubit_props) and qubit_props[phys_q] is not None:
                        t1 = getattr(qubit_props[phys_q], "t1", None)
                        t2 = getattr(qubit_props[phys_q], "t2", None)
                        t1_values_s.append(t1 if t1 is not None else TYPICAL_T1_S)
                        t2_values_s.append(t2 if t2 is not None else TYPICAL_T2_S)
                    else:
                        t1_values_s.append(TYPICAL_T1_S)
                        t2_values_s.append(TYPICAL_T2_S)
        except Exception:
            pass

    # Fallback: fill with typical values if we couldn't read from backend
    n_active = stats.get("active_qubits") or circuit.num_qubits
    while len(t1_values_s) < n_active:
        t1_values_s.append(TYPICAL_T1_S)
        t2_values_s.append(TYPICAL_T2_S)

    # Compute per-qubit decoherence probability
    # P(decay_i) = 1 - exp(-t_idle_i / T1_i)
    # Approximation: average idle fraction × circuit duration / T1
    avg_idle_cycles = idle_metrics["idle_cycles_per_qubit"]
    # Each idle cycle ≈ one gate-layer duration (mix of 1Q and 2Q)
    avg_gate_ns = (t_2q_gate_ns + t_1q_gate_ns) / 2.0
    avg_idle_time_ns = avg_idle_cycles * avg_gate_ns

    per_qubit_decay: list[float] = []
    for i in range(min(n_active, len(t1_values_s))):
        t1_ns = t1_values_s[i] * 1e9
        # Total time qubit exists: circuit_duration_ns
        # Time qubit is idle: proportional to idle_cycles / depth
        idle_frac = avg_idle_cycles / max(depth, 1)
        qubit_idle_ns = circuit_duration_ns * idle_frac
        decay_prob = 1.0 - np.exp(-qubit_idle_ns / t1_ns) if t1_ns > 0 else 1.0
        per_qubit_decay.append(decay_prob)

    mean_decay = float(np.mean(per_qubit_decay)) if per_qubit_decay else 0.0
    worst_decay = float(np.max(per_qubit_decay)) if per_qubit_decay else 0.0

    # T1 budget ratio: circuit_duration / min_T1 — must be << 1
    min_t1_ns = min(t * 1e9 for t in t1_values_s) if t1_values_s else TYPICAL_T1_S * 1e9
    t1_budget_ratio = circuit_duration_ns / min_t1_ns if min_t1_ns > 0 else float("inf")

    # Idle decoherence budget: total error from idle decay (complement to gate errors)
    idle_decoherence_budget = mean_decay * n_active

    return {
        "circuit_duration_ns": circuit_duration_ns,
        "decoherence_fraction": mean_decay,
        "worst_qubit_decay": worst_decay,
        "t1_budget_ratio": t1_budget_ratio,
        "idle_decoherence_budget": idle_decoherence_budget,
        "depth_2q_time_ns": depth_2q_time_ns,
        "avg_idle_time_ns": avg_idle_time_ns,
        "n_active_qubits": n_active,
    }


def compute_parallelism_efficiency(circuit: QuantumCircuit) -> dict[str, Any]:
    """Compute parallelism efficiency vs theoretical maximum.

    Measures how well the transpiler parallelized 2Q gates. On heavy-hex
    with N=10, at most floor(N/2)=5 CZ gates can execute simultaneously.
    The actual utilization is n_2q / (depth_2q × max_parallel_2q).

    Also computes gate cancellation rate: (logical_2q - actual_2q) / logical_2q
    when logical gate count is inferable from the circuit structure.

    Reference: The parallelism metric from Tomesh et al. (SupermarQ, 2022)
    and gate-aware depth from arXiv:2505.16908.

    Parameters
    ----------
    circuit : QuantumCircuit
        Transpiled (ISA) circuit.

    Returns
    -------
    dict[str, Any]
        Keys:
          - parallelism_efficiency: actual / theoretical max parallelism [0, 1]
          - theoretical_max_parallel_2q: max simultaneous 2Q ops on layout
          - actual_avg_parallel_2q: average 2Q ops per 2Q layer
          - serialization_overhead: extra depth from imperfect scheduling
          - liveness: fraction of qubit-time slots that are active
          - gate_density: total_gates / (n_qubits × depth)
    """
    stats = transpiled_circuit_stats(circuit)
    n_2q = stats["n_2q_gates"]
    depth_2q = stats["depth_2q"]
    depth = stats["depth"]
    n_qubits = circuit.num_qubits
    active = stats.get("active_qubits") or n_qubits

    # Maximum parallel 2Q gates on a subgraph: floor(N/2) for linear chain
    # (each 2Q gate consumes 2 qubits, so at most N/2 can be simultaneous)
    theoretical_max_parallel_2q = max(1, active // 2)

    # Actual average parallelism: n_2q / depth_2q
    actual_avg_parallel = n_2q / depth_2q if depth_2q > 0 else 0.0

    # Efficiency: how much of theoretical max we achieve
    parallelism_efficiency = (
        actual_avg_parallel / theoretical_max_parallel_2q
        if theoretical_max_parallel_2q > 0
        else 0.0
    )

    # Serialization overhead: minimum possible depth_2q vs actual
    # Minimum depth_2q if perfectly parallelized = ceil(n_2q / max_parallel)
    min_depth_2q = (
        int(np.ceil(n_2q / theoretical_max_parallel_2q))
        if theoretical_max_parallel_2q > 0
        else n_2q
    )
    serialization_overhead = (depth_2q - min_depth_2q) / min_depth_2q if min_depth_2q > 0 else 0.0

    # Liveness: fraction of (qubit × time) slots used by gates
    total_slots = active * depth if (active > 0 and depth > 0) else 1
    total_gates = stats["total_gates"]
    # Each 2Q gate occupies 2 slots, each 1Q gate occupies 1 slot
    n_1q = stats["n_1q_gates"]
    occupied_slots = n_2q * 2 + n_1q
    liveness = min(1.0, occupied_slots / total_slots)

    # Gate density: total gates / (qubits × depth)
    gate_density = total_gates / total_slots if total_slots > 0 else 0.0

    return {
        "parallelism_efficiency": min(1.0, parallelism_efficiency),
        "theoretical_max_parallel_2q": theoretical_max_parallel_2q,
        "actual_avg_parallel_2q": actual_avg_parallel,
        "serialization_overhead": serialization_overhead,
        "min_depth_2q": min_depth_2q,
        "liveness": liveness,
        "gate_density": gate_density,
    }


def compute_shot_noise_floor(
    shots: int,
    expected_observable: float | None = None,
    n_qubits: int | None = None,
) -> dict[str, Any]:
    """Compute shot noise floor and signal-to-noise ratio.

    Determines whether the shot budget is sufficient to resolve the
    expected observable magnitude. Critical for near-critical h-values
    where ⟨X⟩ ≈ 0.008 at N=10.

    The estimator flags when σ_shot > |⟨O⟩|, meaning the measurement
    is noise-dominated and more shots are needed.

    Parameters
    ----------
    shots : int
        Total shot budget per circuit.
    expected_observable : float | None
        Expected magnitude |⟨O⟩|. If None, uses conservative estimate
        for TFIM near criticality (0.01 for N=10).
    n_qubits : int | None
        System size (used for default observable estimate).

    Returns
    -------
    dict[str, Any]
        Keys:
          - sigma_shot: standard deviation from shot noise = 1/√shots
          - expected_signal: |⟨O⟩| used for comparison
          - snr: signal-to-noise ratio = |⟨O⟩| × √shots
          - shots_sufficient: True if SNR > 2 (signal resolvable)
          - min_shots_for_snr2: minimum shots for SNR=2
          - min_shots_for_snr5: minimum shots for SNR=5
          - noise_dominated: True if σ > |signal|
    """
    sigma = 1.0 / np.sqrt(shots) if shots > 0 else float("inf")

    # Default signal estimate for TFIM N=10 near h_c ≈ 1.0
    if expected_observable is None:
        # Conservative: ⟨X⟩ per site near criticality scales as ~0.5/N
        n = n_qubits if n_qubits is not None else 10
        expected_observable = 0.5 / n  # ~0.05 for N=10, ~0.025 for N=20

    signal = abs(expected_observable)
    snr = signal * np.sqrt(shots) if shots > 0 else 0.0

    # Minimum shots for target SNR
    min_shots_snr2 = int(np.ceil((2.0 / signal) ** 2)) if signal > 0 else float("inf")
    min_shots_snr5 = int(np.ceil((5.0 / signal) ** 2)) if signal > 0 else float("inf")

    return {
        "sigma_shot": float(sigma),
        "expected_signal": signal,
        "snr": float(snr),
        "shots_sufficient": snr > 2.0,
        "min_shots_for_snr2": min_shots_snr2,
        "min_shots_for_snr5": min_shots_snr5,
        "noise_dominated": sigma > signal,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Improvement 5: Unified Circuit Feasibility Score
# ══════════════════════════════════════════════════════════════════════════════


def compute_circuit_feasibility(
    circuit: QuantumCircuit,
    backend=None,
    layout: list[int] | None = None,
    shots: int = 16384,
    expected_observable: float | None = None,
    kappa: float | None = None,
) -> dict[str, Any]:
    """Unified go/no-go feasibility score combining fidelity + cost + decoherence.

    Produces a single composite score in [0, 1] that integrates:
      - Gate error budget (from compute_error_budget)
      - Decoherence penalty (from compute_decoherence_penalty)
      - Shot noise sufficiency (from compute_shot_noise_floor)
      - Readout error contribution (from backend calibration)
      - Parallelism efficiency (scheduling quality)

    The score is weighted to reflect hardware deployment priorities:
      - Gate fidelity: 40% (dominant noise source)
      - Decoherence: 25% (idle time T1 decay)
      - Shot noise: 20% (measurement resolution)
      - Readout: 10% (TREX-mitigated, lower weight)
      - Scheduling: 5% (overhead, not a hard blocker)

    Parameters
    ----------
    circuit : QuantumCircuit
        Transpiled (ISA) circuit.
    backend : BackendV2 | None
        Backend with calibration data.
    layout : list[int] | None
        Physical qubit indices.
    shots : int
        Shot budget per circuit.
    expected_observable : float | None
        Expected |⟨O⟩| for SNR calculation.
    kappa : float | None
        Landscape curvature (optional refinement).

    Returns
    -------
    dict[str, Any]
        Keys:
          - feasibility_score: composite [0, 1] (higher = more feasible)
          - verdict: "go" | "caution" | "no-go"
          - component_scores: individual sub-scores
          - bottleneck: name of lowest-scoring component
          - recommendations: list of actionable suggestions
    """
    # ── Gather component metrics ──
    error_budget = compute_error_budget(circuit, backend=backend, layout=layout)
    decoherence = compute_decoherence_penalty(circuit, backend=backend, layout=layout)
    shot_noise = compute_shot_noise_floor(
        shots, expected_observable=expected_observable, n_qubits=circuit.num_qubits
    )
    parallelism = compute_parallelism_efficiency(circuit)

    # ── Readout error (from backend or typical) ──
    readout_error = 0.01  # Typical Heron r2
    if backend is not None:
        from qmbp_simulation.execution.hardware.preflight import compute_mean_readout_error

        re = compute_mean_readout_error(backend)
        if re is not None:
            readout_error = re

    # ── Component scores (each in [0, 1], higher = better) ──
    # Gate fidelity score: exp(-error_budget) mapped to [0,1]
    gate_score = error_budget["fidelity_estimate"]  # Already in [0, 1]

    # Decoherence score: 1 - decoherence_fraction (capped at 1)
    deco_score = max(0.0, 1.0 - decoherence["decoherence_fraction"] * 10)
    # Scale ×10 because raw fraction is typically ~0.001-0.05

    # Shot noise score: SNR-based, sigmoid mapping
    snr = shot_noise["snr"]
    shot_score = min(1.0, snr / 5.0)  # SNR=5 → perfect, SNR<1 → poor

    # Readout score: 1 - readout_error × N_qubits (each qubit measured)
    n_active = decoherence["n_active_qubits"]
    readout_score = max(0.0, 1.0 - readout_error * n_active)

    # Scheduling score: parallelism efficiency
    sched_score = parallelism["parallelism_efficiency"]

    # ── Weighted composite ──
    weights = {
        "gate_fidelity": 0.40,
        "decoherence": 0.25,
        "shot_noise": 0.20,
        "readout": 0.10,
        "scheduling": 0.05,
    }
    scores = {
        "gate_fidelity": gate_score,
        "decoherence": deco_score,
        "shot_noise": shot_score,
        "readout": readout_score,
        "scheduling": sched_score,
    }

    feasibility = sum(weights[k] * scores[k] for k in weights)

    # ── Verdict ──
    if feasibility >= 0.70:
        verdict = "go"
    elif feasibility >= 0.45:
        verdict = "caution"
    else:
        verdict = "no-go"

    # Kappa refinement: near-critical landscape is harder
    if kappa is not None and kappa < 45 and verdict == "caution":
        verdict = "no-go"
        feasibility *= 0.85  # Penalize flat landscape

    # ── Bottleneck identification ──
    bottleneck = min(scores, key=scores.get)  # type: ignore[arg-type]

    # ── Recommendations ──
    recommendations: list[str] = []
    if scores["gate_fidelity"] < 0.60:
        recommendations.append("Reduce circuit depth (use p=1) or wait for recalibration.")
    if scores["decoherence"] < 0.50:
        recommendations.append(
            "High idle-time decoherence. Add dynamical decoupling or improve scheduling."
        )
    if scores["shot_noise"] < 0.50:
        recommendations.append(
            f"Insufficient shots for signal resolution. "
            f"Minimum {shot_noise['min_shots_for_snr2']} shots for SNR=2."
        )
    if scores["readout"] < 0.70:
        recommendations.append("Elevated readout errors. Ensure TREX is enabled.")
    if scores["scheduling"] < 0.30:
        recommendations.append(
            "Poor gate parallelism. Consider alternative layout with less serialization."
        )

    return {
        "feasibility_score": float(feasibility),
        "verdict": verdict,
        "component_scores": scores,
        "bottleneck": bottleneck,
        "recommendations": recommendations,
        "details": {
            "error_budget": error_budget["error_budget"],
            "fidelity_estimate": error_budget["fidelity_estimate"],
            "decoherence_fraction": decoherence["decoherence_fraction"],
            "t1_budget_ratio": decoherence["t1_budget_ratio"],
            "snr": shot_noise["snr"],
            "readout_error": readout_error,
            "parallelism_efficiency": parallelism["parallelism_efficiency"],
        },
    }
