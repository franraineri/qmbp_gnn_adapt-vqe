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
