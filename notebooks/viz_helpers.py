"""Visualization helpers for demo notebooks.

DESIGN PRINCIPLE: This module is a thin presentation layer on top of existing
qmbp_simulation analysis infrastructure. It reuses:

- `qmbp_simulation.analysis.circuit_visualizer` — circuit_summary, save_circuit_diagram
- `qmbp_simulation.models.HamiltonianBuilder.build_graph_data` — edge_index, coordination
- `qmbp_simulation.models.make_lattice` — lattice construction

This module adds ONLY the matplotlib rendering logic for notebooks.
No physics logic, no circuit construction, no data processing is duplicated here.

Functions:
    Circuit visualization:
        draw_circuit        — Render circuit as matplotlib figure (wraps Qiskit mpl draw)
        draw_hva_structure  — Schematic HVA layer diagram
        draw_circuit_stats  — Bar chart of gate counts from circuit_summary()

    Lattice/topology visualization:
        draw_lattice            — Single lattice with spin sites + bonds
        draw_topology_comparison — Multiple topologies side by side

    Pipeline diagrams:
        draw_pipeline_diagram   — 3-phase architecture overview
        draw_gnn_input_graph    — What the MPNN/GNN-QEM sees as input

    Training diagnostics:
        draw_training_curves    — MSE + energy validation during MPNN training
        draw_theta_landscape    — θ*(h) parameter trajectories
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

    from qmbp_simulation.models import LatticeConfig


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Circuit Visualization (wraps qmbp_simulation.analysis.circuit_visualizer)
# ═══════════════════════════════════════════════════════════════════════════════


def draw_circuit(
    circuit: "QuantumCircuit",
    params: np.ndarray | None = None,
    title: str = "",
    figsize: tuple | None = None,
    fold: int | None = None,
    style: dict | None = None,
) -> plt.Figure:
    """Render a quantum circuit as a matplotlib figure.

    Wraps Qiskit's `circuit.draw(output='mpl')` with sensible defaults
    and auto-sizing. For gate-level analysis use `draw_circuit_stats()`.

    Parameters
    ----------
    circuit : QuantumCircuit
        The Qiskit circuit to visualize.
    params : np.ndarray | None
        If provided and circuit is parameterized, bind these values first.
    title : str
        Optional title above the circuit.
    figsize : tuple | None
        Figure size. Auto-calculated if None.
    fold : int | None
        Gates per line before wrapping. Auto if None.
    style : dict | None
        Qiskit drawing style dict. Default: white background.
    """
    qc = circuit
    if params is not None and circuit.num_parameters > 0:
        qc = circuit.assign_parameters(params)

    if figsize is None:
        figsize = (max(8, qc.depth() * 0.7), max(3, qc.num_qubits * 0.55))

    if fold is None:
        fold = max(15, 80 // max(1, qc.num_qubits))

    draw_style = style or {"backgroundcolor": "#FFFFFF"}

    fig = qc.draw(output="mpl", style=draw_style, fold=fold)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.set_size_inches(figsize)
    return fig


def draw_circuit_stats(circuit: "QuantumCircuit", title: str = "") -> plt.Figure:
    """Bar chart of circuit gate counts using circuit_summary().

    Reuses `qmbp_simulation.analysis.circuit_visualizer.circuit_summary` for
    accurate gate counting, then renders as a horizontal bar chart.
    """
    from qmbp_simulation.analysis import circuit_summary

    info = circuit_summary(circuit)
    gate_counts = info["gate_counts"]

    fig, ax = plt.subplots(figsize=(7, max(3, len(gate_counts) * 0.5)))
    gates = sorted(gate_counts.keys())
    counts = [gate_counts[g] for g in gates]
    colors = ["#e74c3c" if g in ("rzz", "cx", "cz", "rxx", "ryy") else "#3498db" for g in gates]

    ax.barh(gates, counts, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Count")
    ax.set_title(title or f"Gate Counts (depth={info['depth']}, 2Q={info['n_2q_gates']})")
    ax.grid(True, alpha=0.3, axis="x")

    # Annotate totals
    ax.text(
        0.98, 0.02,
        f"Total: {info['n_gates_total']} gates\n"
        f"1Q: {info['n_1q_gates']} | 2Q: {info['n_2q_gates']}\n"
        f"Depth: {info['depth']} | Qubits: {info['n_qubits']}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    fig.tight_layout()
    return fig


def draw_hva_structure(
    n_qubits: int,
    p_layers: int,
    edges: list,
    model_name: str = "TFIM",
) -> plt.Figure:
    """Draw a schematic of HVA layer structure (ZZ→X blocks repeated p times).

    This is a DIDACTIC diagram — not the actual circuit. Shows the conceptual
    layer decomposition of the HVA ansatz.
    """
    fig, ax = plt.subplots(figsize=(max(8, p_layers * 3.5), max(3, n_qubits * 0.35)))
    ax.set_xlim(-0.5, p_layers * 3 + 1.5)
    ax.set_ylim(-0.8, n_qubits - 0.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Qubit lines
    for q in range(n_qubits):
        ax.plot([-0.3, p_layers * 3 + 1.2], [q, q], "k-", lw=0.5, alpha=0.3)
        ax.text(-0.5, q, "|+⟩", ha="right", va="center", fontsize=9, family="monospace")
        ax.text(p_layers * 3 + 1.4, q, f"q{q}", ha="left", va="center", fontsize=8)

    c_zz, c_x = "#3498db", "#e74c3c"

    for layer in range(p_layers):
        x0 = layer * 3 + 0.5

        # ZZ block
        rect = mpatches.FancyBboxPatch(
            (x0, -0.3), 1.0, n_qubits - 0.4,
            boxstyle="round,pad=0.1", fc=c_zz, alpha=0.15, ec=c_zz, lw=1.5,
        )
        ax.add_patch(rect)
        ax.text(x0 + 0.5, n_qubits / 2 - 0.5, r"$e^{-i\theta_{zz}H_{ZZ}}$",
                ha="center", va="center", fontsize=8, color=c_zz, fontweight="bold")

        # X block
        rect = mpatches.FancyBboxPatch(
            (x0 + 1.3, -0.3), 1.0, n_qubits - 0.4,
            boxstyle="round,pad=0.1", fc=c_x, alpha=0.15, ec=c_x, lw=1.5,
        )
        ax.add_patch(rect)
        ax.text(x0 + 1.8, n_qubits / 2 - 0.5, r"$e^{-i\theta_{x}H_{X}}$",
                ha="center", va="center", fontsize=8, color=c_x, fontweight="bold")

        ax.text(x0 + 1.15, -0.7, f"Layer {layer+1}", ha="center", fontsize=9, style="italic")

    ax.set_title(
        f"HVA Structure — {model_name} (N={n_qubits}, p={p_layers}, {len(edges)} bonds)",
        fontsize=12, fontweight="bold", pad=10,
    )
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Lattice / Topology Visualization
# ═══════════════════════════════════════════════════════════════════════════════


def _get_positions(topology: str, n_qubits: int, edges: list) -> dict:
    """Compute 2D node positions from topology name.

    Uses known geometric layouts for common topologies, falls back to
    networkx spring_layout for unknown ones.
    """
    if topology == "chain_1d":
        return {i: (i, 0) for i in range(n_qubits)}
    elif topology == "ladder":
        n_rungs = n_qubits // 2
        pos = {}
        for i in range(n_rungs):
            pos[i] = (i, 0)
            pos[i + n_rungs] = (i, 1)
        return pos
    elif topology == "square":
        side = int(np.ceil(np.sqrt(n_qubits)))
        return {i: (i % side, i // side) for i in range(n_qubits)}
    elif topology == "triangular":
        side = int(np.ceil(np.sqrt(n_qubits)))
        pos = {}
        for i in range(n_qubits):
            row, col = i // side, i % side
            pos[i] = (col + 0.5 * (row % 2), row * 0.866)
        return pos
    elif topology == "heavy_hex":
        n_top = (n_qubits + 1) // 2
        pos = {}
        for i in range(n_top):
            pos[i] = (i * 1.5, 1)
        for i in range(n_qubits - n_top):
            pos[n_top + i] = (i * 1.5 + 0.75, 0)
        return pos
    else:
        G = nx.Graph()
        G.add_nodes_from(range(n_qubits))
        G.add_edges_from(edges)
        return nx.spring_layout(G, seed=42)


def draw_lattice(
    topology: str,
    n_qubits: int,
    edges: list,
    h_value: float | None = None,
    highlight_qubits: list | None = None,
    title: str | None = None,
    show_labels: bool = True,
) -> plt.Figure:
    """Draw a lattice with spin sites (nodes) and interaction bonds (edges).

    Parameters
    ----------
    topology : str
        Topology name.
    n_qubits : int
        Number of sites.
    edges : list
        List of (i, j) interaction pairs.
    h_value : float | None
        Transverse field annotation.
    highlight_qubits : list | None
        Qubits to highlight in red.
    title : str | None
        Plot title.
    show_labels : bool
        Whether to show qubit index labels.
    """
    fig, ax = plt.subplots(figsize=(max(6, n_qubits * 0.7), 5))

    G = nx.Graph()
    G.add_nodes_from(range(n_qubits))
    G.add_edges_from(edges)
    pos = _get_positions(topology, n_qubits, edges)

    # Edges
    nx.draw_networkx_edges(G, pos, ax=ax, width=2.0, alpha=0.6, edge_color="#2c3e50")

    # Nodes
    node_colors = ["#3498db"] * n_qubits
    if highlight_qubits:
        for q in highlight_qubits:
            if q < n_qubits:
                node_colors[q] = "#e74c3c"

    node_size = max(250, 2000 // n_qubits)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_size, edgecolors="black", linewidths=1.5)

    if show_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=max(7, 12 - n_qubits // 4),
                                font_color="white", font_weight="bold")

    # Edge labels (only for small systems)
    if len(edges) <= 15:
        edge_labels = {(i, j): "J" for i, j in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax,
                                     font_size=7, font_color="#7f8c8d")

    # Annotations
    if h_value is not None:
        ax.text(0.02, 0.98, f"h = {h_value:.2f}", transform=ax.transAxes,
                fontsize=11, va="top", bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

    z_max = max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0
    ax.text(0.02, 0.02, f"N={n_qubits} sites, {len(edges)} bonds, z_max={z_max}",
            transform=ax.transAxes, fontsize=9, va="bottom", color="#7f8c8d")

    ax.set_title(title or f"Spin Lattice — {topology} (N={n_qubits})",
                 fontsize=13, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def draw_topology_comparison(
    topologies: list[str],
    n_qubits: int = 10,
) -> plt.Figure:
    """Draw multiple topologies side by side for comparison.

    Reuses `make_lattice` from the package to get edges per topology.
    """
    from qmbp_simulation import make_lattice

    n = len(topologies)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, topo in zip(axes, topologies):
        lattice = make_lattice(topo, n_qubits, J=1.0, h=2.0)
        G = nx.Graph()
        G.add_nodes_from(range(n_qubits))
        G.add_edges_from(lattice.edges)
        pos = _get_positions(topo, n_qubits, lattice.edges)

        nx.draw_networkx_edges(G, pos, ax=ax, width=1.5, alpha=0.5, edge_color="#2c3e50")
        node_size = max(150, 1200 // n_qubits)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#3498db",
                               node_size=node_size, edgecolors="black", linewidths=1.0)
        if n_qubits <= 12:
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=max(6, 9 - n_qubits // 5),
                                    font_color="white", font_weight="bold")

        z_max = max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 0
        ax.set_title(f"{topo}\n({len(lattice.edges)} edges, z={z_max})", fontsize=10)
        ax.axis("off")

    fig.suptitle(f"Lattice Topologies (N={n_qubits})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Pipeline / Architecture Diagrams
# ═══════════════════════════════════════════════════════════════════════════════


def draw_pipeline_diagram() -> plt.Figure:
    """Draw the 3-phase GNN-HVA pipeline architecture."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    phases = [
        (0.5, "Phase 1\nGround Truth", "#2ecc71", "ExactDiag / DMRG\n→ E₀(h), gap(h)"),
        (4.0, "Phase 2\nVQE Warm-Start", "#3498db", "Descending sweep\n→ θ*(h) dataset"),
        (7.5, "Phase 3\nMPNN Predictor", "#9b59b6", "GINConv training\n→ h → θ (one-shot)"),
    ]

    for x, title, color, desc in phases:
        rect = mpatches.FancyBboxPatch(
            (x, 0.8), 2.8, 2.0,
            boxstyle="round,pad=0.15", fc=color, alpha=0.12, ec=color, lw=2,
        )
        ax.add_patch(rect)
        ax.text(x + 1.4, 2.3, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)
        ax.text(x + 1.4, 1.4, desc, ha="center", va="center", fontsize=9, color="#2c3e50")

    # Arrows
    for x1, x2 in [(3.4, 3.9), (6.9, 7.4)]:
        ax.annotate("", xy=(x2, 1.8), xytext=(x1, 1.8),
                    arrowprops=dict(arrowstyle="-|>", lw=2, color="#2c3e50"))

    # Deploy
    ax.annotate("", xy=(11.2, 1.8), xytext=(10.4, 1.8),
                arrowprops=dict(arrowstyle="-|>", lw=2, color="#e74c3c"))
    ax.text(11.5, 1.8, "Deploy\n(zero-shot)", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#e74c3c")

    ax.text(6.0, 0.2, "29–500× speedup vs random-init VQE  |  ΔE/gap < 5%",
            ha="center", fontsize=10, style="italic", color="#7f8c8d")
    ax.set_title("GNN-HVA Pipeline for Quantum Phase Classification",
                 fontsize=13, fontweight="bold", pad=5)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GNN Graph Structure Visualization
# ═══════════════════════════════════════════════════════════════════════════════


def draw_gnn_input_graph(
    n_qubits: int,
    edges: list,
    h_value: float,
    node_features_desc: str = "[h, coord]",
) -> plt.Figure:
    """Visualize the MPNN/GNN-QEM input: Hamiltonian graph with node features.

    Reuses `HamiltonianBuilder.build_graph_data` conceptually — shows the same
    graph structure that `build_graph_dataset()` constructs for PyG Data objects.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    G = nx.Graph()
    G.add_nodes_from(range(n_qubits))
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42, k=2.0 / max(1, n_qubits**0.3))

    coords = [G.degree(i) for i in range(n_qubits)]
    max_coord = max(coords) if coords else 1

    # Edges
    nx.draw_networkx_edges(G, pos, ax=ax, width=2.5, alpha=0.4, edge_color="#e67e22")

    # Nodes colored by coordination
    node_colors = plt.cm.Blues(np.array(coords) / max_coord * 0.7 + 0.3)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=600, edgecolors="black", linewidths=2)

    # Feature labels
    labels = {i: f"q{i}\n[{h_value},{coords[i]}]" for i in range(n_qubits)}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7, font_weight="bold")

    # Edge labels
    if len(edges) <= 15:
        edge_labels = {(i, j): "J=1" for i, j in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=7, font_color="#d35400")

    ax.set_title(
        f"MPNN Input Graph — Node features: {node_features_desc}\n"
        f"(N={n_qubits}, h={h_value}, {len(edges)} edges)",
        fontsize=11, fontweight="bold",
    )
    ax.axis("off")
    ax.text(0.02, 0.02, "Node color ∝ coordination number",
            transform=ax.transAxes, fontsize=8, color="#7f8c8d")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Training Diagnostics (for Notebook 01 post-Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════


def draw_training_curves(train_result: dict, title: str = "") -> plt.Figure:
    """Plot MPNN training curves (MSE loss + optional energy validation).

    Parameters
    ----------
    train_result : dict
        Output from `train_mpnn()`: keys 'mse_history', 'energy_val_history'.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    mse_history = train_result.get("mse_history", [])
    if mse_history:
        ax.semilogy(mse_history, "b-", lw=1.0, alpha=0.7, label="MSE Loss")

    energy_hist = train_result.get("energy_val_history", [])
    if energy_hist:
        epochs_e = [e for e, _ in energy_hist]
        vals_e = [v for _, v in energy_hist]
        ax2 = ax.twinx()
        ax2.plot(epochs_e, vals_e, "r.-", ms=4, lw=0.8, label="ΔE/gap (val)")
        ax2.set_ylabel("ΔE/gap", color="red")
        ax2.axhline(0.05, color="red", ls="--", alpha=0.5, label="5% threshold")
        ax2.legend(loc="upper right")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss", color="blue")
    ax.set_title(title or "MPNN Training Progress")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def draw_theta_landscape(
    h_values: np.ndarray,
    theta_opt: np.ndarray,
    param_names: list[str] | None = None,
    title: str = "",
) -> plt.Figure:
    """Plot θ*(h) parameter trajectories — the smooth landscape MPNN learns.

    Parameters
    ----------
    h_values : np.ndarray [n_points]
    theta_opt : np.ndarray [n_points, n_params]
    param_names : list[str] | None
        Names for each parameter dimension.
    """
    n_params = theta_opt.shape[1]
    if param_names is None:
        param_names = [f"θ_{i}" for i in range(n_params)]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.Set1(np.linspace(0, 1, n_params))

    for j in range(n_params):
        ax.plot(h_values, theta_opt[:, j], ".-", ms=4, lw=1.5,
                color=colors[j], label=param_names[j])

    ax.axvline(1.0, color="gray", ls=":", alpha=0.5, label="h_c = 1.0")
    ax.set_xlabel("h (transverse field)")
    ax.set_ylabel("θ (radians)")
    ax.set_title(title or "Optimal Parameters θ*(h) — What the MPNN Learns")
    ax.legend(ncol=min(4, n_params))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
