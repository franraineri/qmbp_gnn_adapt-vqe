"""Reusable, swappable helpers for the GNN-HVA pipeline anatomy notebook.

This module is intentionally thin: it ONLY adds what the project modules do
not already provide (lattice-graph drawing, MPNN graph drawing, small
matplotlib conveniences). Everything else — Hamiltonian, HVA circuit, VQE,
MPNN, circuit rendering — is imported from ``qmbp_simulation`` so the notebook
stays a faithful view of the production pipeline, not a re-implementation.

Design: each phase in the notebook calls one function here or one project
function, so parts can be swapped independently (change the config, re-run one
cell, without touching the rest).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Config — the single place to change to explore other regimes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """All knobs for the anatomy walkthrough. Edit one field, re-run the cells.

    The defaults reproduce the didactic target: 1D TFIM, N=10, p=1, a small
    h-sweep straddling the critical point h_c = J = 1.
    """

    topology: str = "chain_1d"
    n_qubits: int = 10
    p_layers: int = 1
    j_coupling: float = 1.0
    h_values: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)
    h_focus: float = 1.0  # the h-point used for single-circuit close-ups
    vqe_maxiter: int = 300
    vqe_restarts: int = 1
    seed: int = 42
    # ── Zoo auto-selection (phase A) ──
    # The pretrained zoo models are all `tfim_bond_resolved` → we use the
    # bond-resolved HVA ansatz (per-bond θ_zz + per-site θ_x). The model is
    # picked automatically for the chosen objective.
    zoo_model: str = "tfim_bond_resolved"
    zoo_objective: str = "critical"  # deploy | warmstart | critical | extrapolation
    # ── Transpilation (phase B) ──
    basis_gates: tuple[str, ...] = ("cz", "rz", "sx", "x")  # IBM Heron-like native set
    opt_level: int = 3
    figures_dir: Path = field(default=Path(__file__).parent / "figures")

    def __post_init__(self) -> None:
        self.figures_dir = Path(self.figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def n_params_bond_resolved(self, n_edges: int) -> int:
        """Bond-resolved HVA parameter count: (n_edges + n_qubits) * p_layers."""
        return (n_edges + self.n_qubits) * self.p_layers


# ─────────────────────────────────────────────────────────────────────────────
# Lattice graph drawing (not provided by project modules)
# ─────────────────────────────────────────────────────────────────────────────


def draw_lattice_graph(
    lattice: Any,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Path | None:
    """Draw the lattice as a node-edge graph (the physical connectivity).

    Uses the same ``edge_index`` the MPNN consumes (via
    ``HamiltonianBuilder.build_graph_data``) so the picture matches what the
    network actually sees. Node labels show the coordination number.

    Returns the saved path, or None if drawing is unavailable.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from qmbp_simulation.models import HamiltonianBuilder

    builder = HamiltonianBuilder()
    edge_index, coord = builder.build_graph_data(lattice)
    n = lattice.n_qubits

    # 1D chain layout: place qubits left-to-right. Generic fallback: circle.
    if lattice.n_qubits > 1 and _is_chain_like(lattice):
        xs = np.arange(n, dtype=float)
        ys = np.zeros(n, dtype=float)
    else:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs, ys = np.cos(angles), np.sin(angles)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), 2.6 if _is_chain_like(lattice) else 6))

    # edges (undirected: edge_index is symmetric, draw each pair once)
    drawn = set()
    for a, b in zip(edge_index[0], edge_index[1], strict=False):
        key = (min(int(a), int(b)), max(int(a), int(b)))
        if key in drawn:
            continue
        drawn.add(key)
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color="#888", lw=1.5, zorder=1)

    ax.scatter(xs, ys, s=650, c="#4C78A8", edgecolors="white", linewidths=2, zorder=2)
    for i in range(n):
        ax.text(xs[i], ys[i], f"q{i}", ha="center", va="center", color="white",
                fontsize=9, fontweight="bold", zorder=3)
        ax.text(xs[i], ys[i] - 0.28, f"z={int(coord[i])}", ha="center", va="top",
                fontsize=7, color="#555", zorder=3)

    ax.set_title(title or f"{lattice.topology} lattice · N={n} · {len(drawn)} bonds")
    ax.axis("off")
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path
    plt.close(fig)
    return None


def _is_chain_like(lattice: Any) -> bool:
    return str(getattr(lattice, "topology", "")).startswith("chain")


# ─────────────────────────────────────────────────────────────────────────────
# MPNN input-graph drawing (feature annotations)
# ─────────────────────────────────────────────────────────────────────────────


# Node-type codes used by the unified graph (qubit / ZZ gate / RX gate / global / RZ gate)
_NODE_TYPE_COLORS = {
    0: ("#4C78A8", "q"),    # qubit
    1: ("#E45756", "zz"),   # ZZ gate
    2: ("#F58518", "rx"),   # RX gate
    3: ("#9D755D", "g"),    # global
    4: ("#72B7B2", "rz"),   # RZ gate
}


def draw_mpnn_graph(
    data: Any,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Path | None:
    """Draw a torch_geometric ``Data`` object the MPNN ingests.

    Handles BOTH the simple homogeneous graph (features [h, coord] per node) and
    the unified heterogeneous graph (qubit + ZZ/RX/RZ gate + global nodes,
    distinguished by ``data.node_type``). Nodes are colored by type; a legend
    explains the mapping. For the unified graph we use a spring-like layered
    layout so gate nodes sit above their qubits.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    x = data.x.detach().cpu().numpy()
    edge_index = data.edge_index.detach().cpu().numpy()
    n = x.shape[0]
    node_type = (
        data.node_type.detach().cpu().numpy()
        if hasattr(data, "node_type") and data.node_type is not None
        else np.zeros(n, dtype=int)
    )
    is_unified = bool(np.any(node_type != 0))

    # ── Layout ──
    xs = np.zeros(n, dtype=float)
    ys = np.zeros(n, dtype=float)
    if is_unified:
        # Row per node type: qubits at y=0, ZZ gates y=1, RX gates y=2, others y=3.
        row_of = {0: 0.0, 1: 1.2, 2: 2.4, 4: 3.0, 3: 3.6}
        for t in _NODE_TYPE_COLORS:
            idx = np.where(node_type == t)[0]
            if len(idx) == 0:
                continue
            # Spread this type's nodes evenly across the full width [0, n-1]
            xs[idx] = np.linspace(0, max(n - 1, 1), len(idx))
            ys[idx] = row_of.get(t, 4.0)
    else:
        xs = np.arange(n, dtype=float)

    fig, ax = plt.subplots(figsize=(max(7, n * 0.5), 4.2 if is_unified else 3.0))

    # edges (draw each undirected pair once; skip in very dense unified graphs)
    drawn = set()
    max_edges_to_draw = 400
    for a, b in zip(edge_index[0], edge_index[1], strict=False):
        key = (min(int(a), int(b)), max(int(a), int(b)))
        if key in drawn:
            continue
        drawn.add(key)
        if len(drawn) > max_edges_to_draw:
            continue
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color="#ccc", lw=0.7, zorder=1)

    # nodes colored by type
    for t, (color, prefix) in _NODE_TYPE_COLORS.items():
        idx = np.where(node_type == t)[0]
        if len(idx) == 0:
            continue
        ax.scatter(xs[idx], ys[idx], s=280 if is_unified else 650, c=color,
                   edgecolors="white", linewidths=1.5, zorder=2)

    if not is_unified:
        for i in range(n):
            ax.text(xs[i], ys[i], f"q{i}", ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold", zorder=3)
            feat = ", ".join(f"{v:.2f}" for v in x[i])
            ax.text(xs[i], ys[i] + 0.30, f"[{feat}]", ha="center", va="bottom",
                    fontsize=7, color="#333", zorder=3)
        subtitle = f"{n} nodes · features/node = {x.shape[1]} ([h, coord])"
    else:
        # legend instead of per-node labels (too many nodes)
        present = [t for t in _NODE_TYPE_COLORS if np.any(node_type == t)]
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_NODE_TYPE_COLORS[t][0],
                   markersize=10, label={0: "qubit", 1: "ZZ gate", 2: "RX gate",
                                         3: "global", 4: "RZ gate"}[t])
            for t in present
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
        counts = {name: int(np.sum(node_type == t)) for t, (_, name) in _NODE_TYPE_COLORS.items()
                  if np.any(node_type == t)}
        subtitle = f"{n} nodes · features/node = {x.shape[1]} · types: {counts}"

    ax.set_title(title or f"MPNN input graph · {subtitle}")
    ax.axis("off")
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path
    plt.close(fig)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Small plotting conveniences
# ─────────────────────────────────────────────────────────────────────────────


def save_line_plot(
    x: np.ndarray,
    ys: dict[str, np.ndarray],
    output_path: str | Path,
    xlabel: str,
    ylabel: str,
    title: str,
    markers: bool = True,
) -> Path:
    """Save a simple multi-series line plot (energy curves, θ vs h, etc.)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for label, y in ys.items():
        ax.plot(x, y, marker="o" if markers else None, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(ys) > 1:
        ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_matrix_heatmap(
    matrix: np.ndarray,
    output_path: str | Path,
    title: str,
) -> Path:
    """Save a heatmap of a (small) dense matrix — e.g. the Hamiltonian."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(np.real(matrix), cmap="RdBu_r", aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("column (basis index)")
    ax.set_ylabel("row (basis index)")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
