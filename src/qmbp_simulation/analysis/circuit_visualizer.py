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
