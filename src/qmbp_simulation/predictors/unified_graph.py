"""Unified Hamiltonian+Circuit graph builder for bond-resolved MPNN.

Implements Qracle-style unified graph representation (Zhang et al., 2025)
adapted for our bond-resolved HVA framework. Encodes both the Hamiltonian
structure AND the circuit structure (gate nodes) into a single graph.

The model (BondResolvedMPNN) uses node_type masking to predict only on
qubit nodes (θ_x) and lattice edges (θ_zz), while gate nodes provide
structural context during message passing.

Node types:
  - type=0: Qubit nodes (N)  — features: [h_i, coord_i, N/100, 0]
  - type=1: ZZ gate nodes (n_edges × p) — features: [layer/p, bond/n_edges, N/100, 1]
  - type=2: RX gate nodes (N × p) — features: [layer/p, qubit/N, N/100, 2]

Edge types (encoded implicitly via connectivity, not edge_attr):
  - Hamiltonian: qubit ↔ qubit (lattice.edges, bidirectional)
  - Gate-qubit: gate_node ↔ qubit(s) it acts on (bidirectional)
  - Intra-layer: sequential ordering within layer (gate_zz → gate_x)

References
----------
- Zhang et al. (2025) "Qracle" arXiv:2505.01236
- Integration plan: internal/documentation/next-steps/04_qracle_unified_graph.md
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from qmbp_simulation.models import LatticeConfig, HamiltonianBuilder

logger = logging.getLogger(__name__)



# Node type constants
NODE_TYPE_QUBIT = 0
NODE_TYPE_ZZ_GATE = 1
NODE_TYPE_RX_GATE = 2

# Feature dimension (fixed for all node types)
UNIFIED_NODE_FEATURES = 4  # [feat1, feat2, N/100, node_type]


def build_unified_bond_resolved_graph(
    lattice: LatticeConfig,
    h_value: float,
    p_layers: int = 1,
    theta_opt: np.ndarray | None = None,
    include_circuit_nodes: bool = True,
    n_feature: bool = True,
) -> Data:
    """Build a unified Hamiltonian+Circuit graph for bond-resolved prediction.

    When ``include_circuit_nodes=False``, produces the same graph as
    ``build_bond_resolved_graph`` (backward compatible, 3 features per node).
    When ``include_circuit_nodes=True``, adds gate nodes and inter-layer
    edges following the Qracle pattern (4 features per node).

    Parameters
    ----------
    lattice : LatticeConfig
        Lattice with edges defined.
    h_value : float
        Transverse field value.
    p_layers : int
        Number of HVA layers (determines gate node count).
    theta_opt : np.ndarray | None
        Target θ_opt vector [θ_zz_edges, θ_x_nodes] × p_layers. None for inference.
    include_circuit_nodes : bool
        If True, add ZZ gate nodes and RX gate nodes to the graph.
        If False, produce a Hamiltonian-only graph (backward compatible).
    n_feature : bool
        Include N/100 as a node feature (recommended for cross-N).

    Returns
    -------
    Data
        Graph with x, edge_index, edge_list, node_type, and optionally y.
        - node_type: tensor [total_nodes] with values 0/1/2
        - n_qubit_nodes: number of qubit nodes (for masking in forward)
        - n_edges_unique: number of unique lattice edges (for θ_zz)
    """
    builder = HamiltonianBuilder()
    edge_index_np, coord = builder.build_graph_data(lattice)

    N = lattice.n_qubits
    n_edges = len(lattice.edges)
    edges_unique = np.array(sorted(lattice.edges))

    # ── Input validation ─────────────────────────────────────────────
    if N < 2:
        raise ValueError(f"n_qubits must be >= 2, got {N}")
    if n_edges == 0:
        raise ValueError(f"Lattice has no edges (topology={lattice.topology})")
    if p_layers < 1:
        raise ValueError(f"p_layers must be >= 1, got {p_layers}")
    if not np.isfinite(h_value):
        raise ValueError(f"h_value must be finite, got {h_value}")
    if theta_opt is not None:
        # Ensure float64 (handles legacy dtype=object arrays from NPZ)
        theta_opt = np.asarray(theta_opt, dtype=np.float64)
        expected_params = (n_edges + N) * p_layers
        if theta_opt.shape != (expected_params,):
            raise ValueError(
                f"theta_opt shape mismatch: expected ({expected_params},) "
                f"= ({n_edges} edges + {N} qubits) × {p_layers} layers, "
                f"got {theta_opt.shape}"
            )
        if not np.all(np.isfinite(theta_opt)):
            n_bad = np.sum(~np.isfinite(theta_opt))
            logger.warning(
                "theta_opt contains %d non-finite values (NaN/Inf). "
                "Graph will be built but training may diverge.", n_bad
            )


    if not include_circuit_nodes:
        # ── Backward-compatible path: Hamiltonian-only graph ─────────
        # Same as build_bond_resolved_graph but with 4 features (padded type=0)
        h_feat = np.full(N, float(h_value))
        n_scale = N / 100.0 if n_feature else 0.0
        # Features: [h_i, coord_i, N/100, node_type=0]
        node_features = np.column_stack([
            h_feat,
            coord.astype(float),
            np.full(N, n_scale),
            np.zeros(N),  # node_type = 0 (qubit)
        ])

        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        edge_list = torch.tensor(edges_unique, dtype=torch.long)
        node_type = torch.zeros(N, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_list=edge_list)
        data.node_type = node_type
        data.n_qubit_nodes = N
        data.n_nodes = N
        data.n_edges_unique = n_edges

        if theta_opt is not None:
            data.y = torch.tensor(theta_opt, dtype=torch.float32)

        return data

    # ── Unified graph: Hamiltonian + Circuit nodes ───────────────────
    # Node layout: [qubit_0..qubit_{N-1}, zz_gates..., rx_gates...]
    n_zz_gates = n_edges * p_layers
    n_rx_gates = N * p_layers
    total_nodes = N + n_zz_gates + n_rx_gates

    n_scale = N / 100.0 if n_feature else 0.0


    # ── Build node features ──────────────────────────────────────────
    # Qubit nodes: [h_i, coord_i, N/100, type=0]
    qubit_features = np.column_stack([
        np.full(N, float(h_value)),
        coord.astype(float),
        np.full(N, n_scale),
        np.zeros(N),  # type=0
    ])

    # ZZ gate nodes: [layer/p_layers, bond_idx/n_edges, N/100, type=1]
    # Normalized indices for scale-invariance across system sizes
    zz_feat_list = []
    for layer in range(p_layers):
        layer_norm = (layer + 0.5) / p_layers  # center in [0, 1]
        for bond_idx in range(n_edges):
            bond_norm = (bond_idx + 0.5) / n_edges
            zz_feat_list.append([layer_norm, bond_norm, n_scale, 1.0])
    zz_features = np.array(zz_feat_list) if zz_feat_list else np.empty((0, 4))

    # RX gate nodes: [layer/p_layers, qubit_idx/N, N/100, type=2]
    rx_feat_list = []
    for layer in range(p_layers):
        layer_norm = (layer + 0.5) / p_layers
        for qubit_idx in range(N):
            qubit_norm = (qubit_idx + 0.5) / N
            rx_feat_list.append([layer_norm, qubit_norm, n_scale, 2.0])
    rx_features = np.array(rx_feat_list) if rx_feat_list else np.empty((0, 4))

    # Stack all node features
    all_features = np.vstack([qubit_features, zz_features, rx_features])
    x = torch.tensor(all_features, dtype=torch.float32)

    # Node type tensor
    node_type = torch.cat([
        torch.full((N,), NODE_TYPE_QUBIT, dtype=torch.long),
        torch.full((n_zz_gates,), NODE_TYPE_ZZ_GATE, dtype=torch.long),
        torch.full((n_rx_gates,), NODE_TYPE_RX_GATE, dtype=torch.long),
    ])


    # ── Build edge index ─────────────────────────────────────────────
    # We collect all edges as (src, dst) pairs, then make symmetric.
    edge_src = []
    edge_dst = []

    # 1. Hamiltonian edges: qubit ↔ qubit (from lattice, already in edge_index_np)
    edge_src.extend(edge_index_np[0].tolist())
    edge_dst.extend(edge_index_np[1].tolist())

    # 2. Gate-qubit edges: ZZ gate ↔ both qubits it acts on
    zz_base_idx = N  # ZZ gate nodes start after qubit nodes
    for layer in range(p_layers):
        for bond_idx, (qi, qj) in enumerate(lattice.edges):
            gate_node = zz_base_idx + layer * n_edges + bond_idx
            # Bidirectional: gate ↔ qubit_i, gate ↔ qubit_j
            edge_src.extend([gate_node, qi, gate_node, qj])
            edge_dst.extend([qi, gate_node, qj, gate_node])

    # 3. Gate-qubit edges: RX gate ↔ the qubit it acts on
    rx_base_idx = N + n_zz_gates  # RX gate nodes start after ZZ gates
    for layer in range(p_layers):
        for qubit_idx in range(N):
            gate_node = rx_base_idx + layer * N + qubit_idx
            # Bidirectional: gate ↔ qubit
            edge_src.extend([gate_node, qubit_idx])
            edge_dst.extend([qubit_idx, gate_node])

    # 4. Intra-layer sequential edges: ZZ gates → RX gates (circuit ordering)
    #    Within each layer: after all ZZ gates execute, RX gates follow.
    #    Connect each ZZ gate to the RX gates on its qubits (data flow).
    for layer in range(p_layers):
        for bond_idx, (qi, qj) in enumerate(lattice.edges):
            zz_node = zz_base_idx + layer * n_edges + bond_idx
            rx_node_i = rx_base_idx + layer * N + qi
            rx_node_j = rx_base_idx + layer * N + qj
            # ZZ → RX (unidirectional: circuit causal order)
            edge_src.extend([zz_node, zz_node])
            edge_dst.extend([rx_node_i, rx_node_j])


    # 5. Inter-layer edges: RX gate (layer l) → ZZ gate (layer l+1)
    #    Connects sequential layers for deep HVA (p>1).
    for layer in range(p_layers - 1):
        for qubit_idx in range(N):
            rx_node = rx_base_idx + layer * N + qubit_idx
            # Connect to ZZ gates in next layer that touch this qubit
            for bond_idx, (qi, qj) in enumerate(lattice.edges):
                if qubit_idx == qi or qubit_idx == qj:
                    zz_next = zz_base_idx + (layer + 1) * n_edges + bond_idx
                    edge_src.append(rx_node)
                    edge_dst.append(zz_next)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)

    # ── Edge list for θ_zz prediction (same as Hamiltonian-only) ─────
    edge_list = torch.tensor(edges_unique, dtype=torch.long)

    # ── Assemble Data object ─────────────────────────────────────────
    data = Data(x=x, edge_index=edge_index, edge_list=edge_list)
    data.node_type = node_type
    data.n_qubit_nodes = N
    data.n_nodes = total_nodes
    data.n_edges_unique = n_edges

    if theta_opt is not None:
        data.y = torch.tensor(theta_opt, dtype=torch.float32)

    return data


def build_unified_dataset(
    lattice: LatticeConfig,
    h_values: np.ndarray,
    theta_opts: np.ndarray,
    p_layers: int = 1,
    include_circuit_nodes: bool = True,
) -> list[Data]:
    """Build a list of unified graphs for training the BondResolvedMPNN.

    Parameters
    ----------
    lattice : LatticeConfig
        Lattice specification (topology, edges, N).
    h_values : np.ndarray of shape [n_points]
        Transverse field values.
    theta_opts : np.ndarray of shape [n_points, n_params]
        Optimized parameters per h-point.
    p_layers : int
        Number of HVA layers.
    include_circuit_nodes : bool
        Whether to include circuit structure in the graph.

    Returns
    -------
    list[Data]
        One graph per h-point, ready for train_bond_resolved_mpnn.
    """
    if len(h_values) != len(theta_opts):
        raise ValueError(
            f"h_values ({len(h_values)}) and theta_opts ({len(theta_opts)}) "
            f"must have the same length."
        )

    # Ensure float64 (handles legacy dtype=object arrays from NPZ)
    theta_opts = np.asarray(theta_opts, dtype=np.float64)
    n_edges = len(lattice.edges)
    N = lattice.n_qubits
    expected_params = (n_edges + N) * p_layers
    if theta_opts.shape[1] != expected_params:
        raise ValueError(
            f"theta_opts column count ({theta_opts.shape[1]}) != expected "
            f"({expected_params}) = ({n_edges} edges + {N} qubits) × {p_layers} layers"
        )

    dataset = []
    for h, theta in zip(h_values, theta_opts):
        graph = build_unified_bond_resolved_graph(
            lattice=lattice,
            h_value=float(h),
            p_layers=p_layers,
            theta_opt=theta,
            include_circuit_nodes=include_circuit_nodes,
        )
        dataset.append(graph)

    return dataset


def validate_unified_graph(data: Data) -> list[str]:
    """Validate a unified graph for structural correctness.

    Run this after building a graph to catch construction errors before
    they propagate to training (where they manifest as NaN loss or
    mysterious shape mismatches).

    Parameters
    ----------
    data : Data
        Graph from build_unified_bond_resolved_graph().

    Returns
    -------
    list[str]
        List of issues found. Empty list = graph is valid.
    """
    issues: list[str] = []

    # 1. Required attributes
    for attr in ("x", "edge_index", "edge_list", "node_type", "n_qubit_nodes", "n_edges_unique"):
        if not hasattr(data, attr) or getattr(data, attr) is None:
            issues.append(f"Missing required attribute: {attr}")

    if issues:
        return issues  # Can't continue without basic attributes

    N = data.n_qubit_nodes
    n_edges = data.n_edges_unique
    total_nodes = data.x.shape[0]

    # 2. Node type consistency
    n_qubits_actual = (data.node_type == NODE_TYPE_QUBIT).sum().item()
    if n_qubits_actual != N:
        issues.append(
            f"node_type qubit count ({n_qubits_actual}) != n_qubit_nodes ({N})"
        )

    # 3. Edge index bounds
    max_idx = data.edge_index.max().item()
    if max_idx >= total_nodes:
        issues.append(
            f"edge_index contains index {max_idx} >= total_nodes ({total_nodes})"
        )
    min_idx = data.edge_index.min().item()
    if min_idx < 0:
        issues.append(f"edge_index contains negative index {min_idx}")

    # 4. Edge list bounds (should reference qubit indices 0..N-1)
    if data.edge_list.max().item() >= N:
        issues.append(
            f"edge_list references index {data.edge_list.max().item()} >= N ({N}). "
            f"edge_list must reference qubit nodes only."
        )
    if data.edge_list.shape[0] != n_edges:
        issues.append(
            f"edge_list has {data.edge_list.shape[0]} edges, expected {n_edges}"
        )

    # 5. Feature dimensions
    if data.x.shape[1] != UNIFIED_NODE_FEATURES:
        issues.append(
            f"Feature dim is {data.x.shape[1]}, expected {UNIFIED_NODE_FEATURES}"
        )

    # 6. NaN/Inf in features
    if not torch.all(torch.isfinite(data.x)):
        n_bad = (~torch.isfinite(data.x)).sum().item()
        issues.append(f"Node features contain {n_bad} non-finite values")

    # 7. Target shape (if present)
    if hasattr(data, "y") and data.y is not None:
        # Infer p_layers from gate node count
        n_zz_gates = (data.node_type == NODE_TYPE_ZZ_GATE).sum().item()
        p_layers = n_zz_gates // n_edges if n_edges > 0 else 1
        expected_y_len = (n_edges + N) * p_layers
        if data.y.shape[0] != expected_y_len:
            issues.append(
                f"Target y has {data.y.shape[0]} elements, expected "
                f"{expected_y_len} = ({n_edges} edges + {N} qubits) × {p_layers} layers"
            )

    # 8. Qubit nodes must be first N nodes (layout invariant)
    first_n_types = data.node_type[:N]
    if not (first_n_types == NODE_TYPE_QUBIT).all():
        issues.append(
            "Qubit nodes must be the first N nodes in the graph "
            "(layout invariant for edge_list indexing)"
        )

    return issues


def compute_graph_metrics(data: Data) -> dict[str, Any]:
    """Compute structural metrics for a unified graph.

    These metrics are saved alongside experiment results to track
    how graph structure correlates with prediction quality.

    Parameters
    ----------
    data : Data
        Graph from build_unified_bond_resolved_graph().

    Returns
    -------
    dict with keys:
        - n_qubit_nodes: number of qubit nodes
        - n_gate_nodes: number of gate nodes (ZZ + RX)
        - n_zz_gates: number of ZZ gate nodes
        - n_rx_gates: number of RX gate nodes
        - total_nodes: total graph nodes
        - total_edges: total directed edges
        - node_expansion_ratio: total_nodes / n_qubit_nodes
        - edge_expansion_ratio: total_edges / hamiltonian_edges
        - avg_degree: average node degree
        - include_circuit_nodes: whether circuit nodes are present
        - graph_density: edge_count / (nodes * (nodes - 1))
    """
    from typing import Any  # noqa: F811

    N = data.n_qubit_nodes
    n_edges_unique = data.n_edges_unique
    total_nodes = data.x.shape[0]
    total_edges = data.edge_index.shape[1]
    hamiltonian_edges = 2 * n_edges_unique  # bidirectional

    n_zz = (data.node_type == NODE_TYPE_ZZ_GATE).sum().item()
    n_rx = (data.node_type == NODE_TYPE_RX_GATE).sum().item()
    include_circuit = (n_zz + n_rx) > 0

    avg_degree = total_edges / total_nodes if total_nodes > 0 else 0.0
    max_possible_edges = total_nodes * (total_nodes - 1)
    density = total_edges / max_possible_edges if max_possible_edges > 0 else 0.0

    # ── Gate heterogeneity metrics ──────────────────────────────────
    # Measures how varied the ZZ gate node connectivity is.
    # High heterogeneity = non-symmetric topology (square, triangular)
    # Low heterogeneity = symmetric topology (chain_1d — all gates equivalent)
    # This is the core predictor for whether UnifiedMPNN will outperform
    # BondResolvedMPNN: type-aware readout helps only when gates differ.
    gate_degree_std = 0.0
    gate_degree_cv = 0.0  # coefficient of variation
    qubit_degree_std = 0.0
    qubit_degree_cv = 0.0
    gate_neighborhood_cv = 0.0  # 2nd-order: variance of qubit-degrees around each gate
    if total_nodes > 0 and total_edges > 0:
        import torch as _torch
        edge_index = data.edge_index
        # Compute per-node degree
        degrees = _torch.zeros(total_nodes, dtype=_torch.long)
        degrees.scatter_add_(0, edge_index[0], _torch.ones(edge_index.shape[1], dtype=_torch.long))

        if n_zz > 0:
            zz_mask = data.node_type == NODE_TYPE_ZZ_GATE
            zz_degrees = degrees[zz_mask].float()
            gate_degree_std = float(zz_degrees.std().item()) if n_zz > 1 else 0.0
            gate_mean = float(zz_degrees.mean().item())
            gate_degree_cv = gate_degree_std / gate_mean if gate_mean > 0 else 0.0

        qubit_mask = data.node_type == NODE_TYPE_QUBIT
        qubit_degrees = degrees[qubit_mask].float()
        if N > 1:
            qubit_degree_std = float(qubit_degrees.std().item())
            qubit_mean = float(qubit_degrees.mean().item())
            qubit_degree_cv = qubit_degree_std / qubit_mean if qubit_mean > 0 else 0.0

        # 2nd-order heterogeneity: for each ZZ gate, sum the degrees of its
        # connected qubits. This captures positional information — a ZZ gate
        # between two high-degree qubits (center of lattice) vs two low-degree
        # qubits (edge of lattice) will have different neighborhood sums.
        if n_zz > 0 and N > 0:
            zz_indices = _torch.where(data.node_type == NODE_TYPE_ZZ_GATE)[0]
            neighborhood_scores = _torch.zeros(n_zz, dtype=_torch.float)
            for idx_local, gate_idx in enumerate(zz_indices):
                # Find qubit neighbors of this gate node
                gate_neighbors = edge_index[1, edge_index[0] == gate_idx.item()]
                qubit_neighbors = gate_neighbors[gate_neighbors < N]
                if len(qubit_neighbors) > 0:
                    neighborhood_scores[idx_local] = degrees[qubit_neighbors].float().sum()
            if n_zz > 1:
                ns_mean = float(neighborhood_scores.mean().item())
                ns_std = float(neighborhood_scores.std().item())
                gate_neighborhood_cv = ns_std / ns_mean if ns_mean > 0 else 0.0

    return {
        "n_qubit_nodes": N,
        "n_gate_nodes": n_zz + n_rx,
        "n_zz_gates": n_zz,
        "n_rx_gates": n_rx,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "hamiltonian_edges": hamiltonian_edges,
        "node_expansion_ratio": total_nodes / N if N > 0 else 0.0,
        "edge_expansion_ratio": total_edges / hamiltonian_edges if hamiltonian_edges > 0 else 0.0,
        "avg_degree": avg_degree,
        "graph_density": density,
        "include_circuit_nodes": include_circuit,
        "gate_degree_std": gate_degree_std,
        "gate_degree_cv": gate_degree_cv,
        "qubit_degree_std": qubit_degree_std,
        "qubit_degree_cv": qubit_degree_cv,
        "gate_neighborhood_cv": gate_neighborhood_cv,
    }
