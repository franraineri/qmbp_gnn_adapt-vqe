"""Noise-aware layout optimization using VF2 subgraph isomorphism.

Wraps mapomatic for exhaustive layout discovery and provides custom
cost functions that integrate domain-specific metrics via BackendV2 Target API.

Architecture: Multi-layer filtering pipeline
  - Layer 0: Pre-VF2 CouplingMap pruning (exclude bad qubits/edges)
  - Layer 1: VF2 search with call_limit (mapomatic)
  - Layer 2: BackendV2 Target-based scoring (fidelity product + penalties)
  - Layer 3: Top-N selection + transpilation → LayoutSelection

References:
    - Mapomatic: https://github.com/qiskit-community/mapomatic
    - Nation et al., Quantum Sci. Technol. 8 035006 (2023)
    - VF2PostLayout: qiskit.transpiler.passes.VF2PostLayout (Qiskit native)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qmbp_simulation.execution.noisy_utils import (
    LayoutSelection,
    compute_circuit_ces,
    find_layouts_bfs,
)

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Mapomatic availability check (graceful degradation)
# ═══════════════════════════════════════════════════════════════════════════

try:
    import mapomatic as mm

    MAPOMATIC_AVAILABLE = True
except ImportError:
    mm = None  # type: ignore[assignment]
    MAPOMATIC_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class LayoutOptimizationResult:
    """Complete result of noise-aware layout optimization.

    Attributes
    ----------
    selected_layouts : list[list[int]]
        Final selected physical qubit layouts.
    fidelity_costs : list[float]
        Per-layout fidelity cost (1 - product_fidelity).
    ces_values : list[float]
        Per-layout CES post-transpilation.
    transpiled_circuits : list[QuantumCircuit]
        Pre-transpiled circuits for selected layouts.
    total_vf2_layouts_found : int
        How many layouts VF2 discovered (before filtering).
    filtered_cmap_edges : int
        Number of edges in the filtered CouplingMap.
    original_cmap_edges : int
        Number of edges in the original CouplingMap.
    filtering_stats : dict
        Detailed stats: excluded qubits, excluded edges, etc.
    backend_name : str
        Name of the backend used.
    strategy_used : str
        Layout selection strategy applied.
    elapsed_s : float
        Total wall-clock time for optimization.
    method : str
        "mapomatic_vf2" or "bfs_fallback".
    """

    selected_layouts: list[list[int]] = field(default_factory=list)
    fidelity_costs: list[float] = field(default_factory=list)
    ces_values: list[float] = field(default_factory=list)
    transpiled_circuits: list[QuantumCircuit] = field(default_factory=list)
    total_vf2_layouts_found: int = 0
    filtered_cmap_edges: int = 0
    original_cmap_edges: int = 0
    filtering_stats: dict[str, Any] = field(default_factory=dict)
    backend_name: str = ""
    strategy_used: str = ""
    elapsed_s: float = 0.0
    method: str = "mapomatic_vf2"


# ═══════════════════════════════════════════════════════════════════════════
# Layer 0: Pre-VF2 CouplingMap pruning
# ═══════════════════════════════════════════════════════════════════════════


def build_filtered_coupling_map(
    backend,
    *,
    max_2q_error: float = 0.01,
    min_t1_us: float = 50.0,
    exclude_qubits: set[int] | None = None,
) -> tuple[CouplingMap, dict[str, Any]]:
    """Build a CouplingMap containing only high-quality edges.

    Pre-filters the VF2 search space by removing edges and qubits
    that do not meet calibration quality thresholds. This drastically
    reduces the number of isomorphisms VF2 needs to explore.

    Parameters
    ----------
    backend : BackendV2
        Backend with live calibration data in ``backend.target``.
    max_2q_error : float
        Maximum allowed 2Q gate error rate per edge. Edges above this
        are excluded from the coupling map. Default: 0.01 (1%).
        For ibm_kingston (Heron R2, CZ median 0.18%), 0.01 keeps ~90% of edges.
    min_t1_us : float
        Minimum T1 time in microseconds. Qubits below this threshold
        are excluded (along with all their edges). Default: 50 µs.
    exclude_qubits : set[int] | None
        Manual blacklist of physical qubit indices to exclude.

    Returns
    -------
    tuple[CouplingMap, dict[str, Any]]
        - Filtered CouplingMap (only high-quality edges)
        - Stats dict with filtering details
    """
    target = backend.target
    exclude = exclude_qubits or set()
    stats: dict[str, Any] = {
        "total_qubits": backend.num_qubits,
        "excluded_qubits_manual": sorted(exclude),
        "excluded_qubits_t1": [],
        "excluded_edges_error": [],
        "original_edges": 0,
        "filtered_edges": 0,
    }

    # ── Filter qubits by T1 ──
    bad_qubits_t1: set[int] = set()
    for qubit_idx in range(backend.num_qubits):
        if qubit_idx in exclude:
            continue
        try:
            # T1 is accessible via target qubit properties
            qubit_props = target.qubit_properties
            if qubit_props and qubit_idx < len(qubit_props):
                qp = qubit_props[qubit_idx]
                if qp and hasattr(qp, "t1") and qp.t1 is not None:
                    t1_us = qp.t1 * 1e6  # Convert seconds to microseconds
                    if t1_us < min_t1_us:
                        bad_qubits_t1.add(qubit_idx)
        except (AttributeError, IndexError, TypeError):
            pass  # T1 not available — do not exclude

    stats["excluded_qubits_t1"] = sorted(bad_qubits_t1)
    all_excluded = exclude | bad_qubits_t1

    # ── Filter edges by 2Q gate error ──
    good_edges: list[list[int]] = []
    all_edges: list[list[int]] = []
    excluded_edges: list[dict[str, Any]] = []

    gate_names_2q = ["cz", "ecr", "cx"]
    for gate_name in gate_names_2q:
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
            q0, q1 = qa
            all_edges.append([q0, q1])

            # Exclude if either qubit is in blacklist
            if q0 in all_excluded or q1 in all_excluded:
                continue

            # Check gate error
            try:
                props = target[gate_name].get(qa)
                if props and props.error is not None:
                    if props.error <= max_2q_error:
                        good_edges.append([q0, q1])
                    else:
                        excluded_edges.append(
                            {"edge": [q0, q1], "gate": gate_name, "error": props.error}
                        )
                else:
                    # No error data — include by default (conservative)
                    good_edges.append([q0, q1])
            except Exception:
                good_edges.append([q0, q1])

    stats["original_edges"] = len(all_edges)
    stats["filtered_edges"] = len(good_edges)
    stats["excluded_edges_error"] = excluded_edges[:20]  # Cap for JSON size
    stats["excluded_edges_count"] = len(excluded_edges)
    stats["retention_rate"] = len(good_edges) / max(len(all_edges), 1)

    if not good_edges:
        _logger.warning(
            "build_filtered_coupling_map: no edges pass filter "
            "(max_2q_error=%.4f, min_t1_us=%.1f). Returning full coupling map.",
            max_2q_error,
            min_t1_us,
        )
        return backend.coupling_map, stats

    return CouplingMap(good_edges), stats


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1: VF2 layout discovery (mapomatic wrapper)
# ═══════════════════════════════════════════════════════════════════════════


def find_vf2_layouts(
    circuit: QuantumCircuit,
    backend_or_cmap,
    *,
    call_limit: int = 100_000,
    strict_direction: bool = True,
    max_layouts: int = 200,
) -> list[list[int]]:
    """Find all matching subgraph layouts via VF2 isomorphism.

    Uses mapomatic's VF2 mapper (via rustworkx) to find all subgraphs
    of the backend topology that are isomorphic to the circuit's
    interaction graph. This guarantees SWAP-free layouts.

    Parameters
    ----------
    circuit : QuantumCircuit
        Deflated circuit (active qubits only, no idle wires).
    backend_or_cmap : BackendV2 | CouplingMap
        Target backend or pre-filtered CouplingMap.
    call_limit : int
        Maximum VF2 mapper calls. Default: 100,000 (sufficient for
        N=10 on 156-qubit backends). Reduce for faster search at the
        cost of completeness.
    strict_direction : bool
        If True, respect native gate direction (CZ/ECR directionality).
        Reduces layouts to those not needing direction flipping.
    max_layouts : int
        Maximum layouts to return. If VF2 finds more, truncate.
        Default: 200 (more than enough for scoring phase).

    Returns
    -------
    list[list[int]]
        Found layouts (each a list of physical qubit indices).
        Empty list if mapomatic is unavailable or no layouts found.

    Notes
    -----
    Falls back to ``find_layouts_bfs()`` if mapomatic is not installed.
    The fallback does NOT guarantee SWAP-free layouts but provides
    connected subgraphs as candidates.
    """
    if not MAPOMATIC_AVAILABLE:
        _logger.info(
            "mapomatic not available — falling back to find_layouts_bfs(). "
            "Install with: pip install mapomatic>=0.14"
        )
        return []

    try:
        layouts = mm.matching_layouts(
            circuit,
            backend_or_cmap,
            strict_direction=strict_direction,
            call_limit=call_limit,
        )
    except Exception as exc:
        _logger.warning("mapomatic matching_layouts failed: %s. Returning empty.", exc)
        return []

    if len(layouts) > max_layouts:
        _logger.info(
            "VF2 found %d layouts, truncating to max_layouts=%d", len(layouts), max_layouts
        )
        layouts = layouts[:max_layouts]

    return layouts


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: BackendV2 Target-based scoring
# ═══════════════════════════════════════════════════════════════════════════


def compute_layout_fidelity_cost(
    circuit: QuantumCircuit,
    layouts: list[list[int]],
    backend,
    *,
    defective_edge_threshold: float = 0.10,
    include_readout: bool = True,
) -> list[tuple[list[int], float]]:
    """Score layouts by fidelity cost using BackendV2 Target API.

    Computes the product fidelity for each layout based on the circuit's
    gate structure and live calibration data. Lower cost = better layout.

    Cost = 1 - Π(1 - ε_gate) × Π(1 - ε_readout)

    Layouts with any edge exceeding ``defective_edge_threshold`` receive
    a +1.0 penalty, effectively relegating them to the bottom of the ranking.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to evaluate (deflated, with active qubits only).
    layouts : list[list[int]]
        Candidate layouts from VF2 or BFS discovery.
    backend : BackendV2
        Backend with calibration data in ``backend.target``.
    defective_edge_threshold : float
        Edge error above this triggers a penalty. Default: 0.10 (10%).
    include_readout : bool
        If True, include measurement readout error in fidelity product.

    Returns
    -------
    list[tuple[list[int], float]]
        Sorted list of (layout, cost) tuples, ascending by cost (best first).
    """
    target = backend.target
    out: list[tuple[list[int], float]] = []

    for layout in layouts:
        fid = 1.0
        has_defect = False
        n_virtual = circuit.num_qubits

        # Defensive: skip layouts that don't match circuit size
        if len(layout) < n_virtual:
            _logger.debug(
                "Layout %s has %d qubits but circuit needs %d — skipping",
                layout[:3],
                len(layout),
                n_virtual,
            )
            continue

        for inst in circuit.data:
            op_name = inst.operation.name

            if inst.operation.num_qubits == 2 and op_name not in ("barrier", "delay"):
                q0 = circuit.find_bit(inst.qubits[0]).index
                q1 = circuit.find_bit(inst.qubits[1]).index

                # Map virtual → physical
                phys_q0 = layout[q0]
                phys_q1 = layout[q1]

                # Try both directions (backend may only have one)
                props = None
                if op_name in target.operation_names:
                    props = target[op_name].get((phys_q0, phys_q1))
                    if props is None:
                        props = target[op_name].get((phys_q1, phys_q0))

                if props and props.error is not None:
                    fid *= 1.0 - props.error
                    if props.error > defective_edge_threshold:
                        has_defect = True
                else:
                    # No calibration data — assume typical error
                    fid *= 1.0 - 0.005

            elif op_name in ("sx", "x", "rz", "ry", "rx"):
                q0 = circuit.find_bit(inst.qubits[0]).index
                phys_q0 = layout[q0]
                if op_name in target.operation_names:
                    props = target[op_name].get((phys_q0,))
                    if props and props.error is not None:
                        fid *= 1.0 - props.error

            elif op_name == "measure" and include_readout:
                q0 = circuit.find_bit(inst.qubits[0]).index
                phys_q0 = layout[q0]
                if "measure" in target.operation_names:
                    props = target["measure"].get((phys_q0,))
                    if props and props.error is not None:
                        fid *= 1.0 - props.error

        error = 1.0 - fid
        if has_defect:
            error += 1.0  # Massive penalty — pushed to bottom

        out.append((layout, error))

    out.sort(key=lambda x: x[1])
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3: Top-N selection + transpilation → LayoutSelection
# ═══════════════════════════════════════════════════════════════════════════


def select_optimal_layouts(
    circuit: QuantumCircuit,
    backend,
    *,
    n_select: int = 3,
    max_ces: float = 0.5,
    max_2q_error: float = 0.01,
    min_t1_us: float = 50.0,
    optimization_level: int = 2,
    call_limit: int = 100_000,
    max_layouts: int = 200,
    exclude_qubits: set[int] | None = None,
    defective_edge_threshold: float = 0.10,
    strategy: Literal["lowest_cost", "ces_spread", "hybrid"] = "lowest_cost",
) -> LayoutSelection:
    """Full multi-layer layout optimization pipeline.

    Orchestrates: Layer 0 (prune) → Layer 1 (VF2) → Layer 2 (score) →
    Layer 3 (select + transpile). Returns a LayoutSelection compatible
    with the existing noisy_utils pipeline.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized or bound circuit. If parameterized, must be bound
        before calling (needed for accurate transpilation).
    backend : BackendV2
        Target backend with calibration data.
    n_select : int
        Number of layouts to select. Default: 3.
    max_ces : float
        Maximum CES allowed per layout. Layouts above this are discarded
        post-transpilation. Default: 0.5.
    max_2q_error : float
        Layer 0 threshold: exclude edges with error above this.
    min_t1_us : float
        Layer 0 threshold: exclude qubits with T1 below this.
    optimization_level : int
        Transpiler optimization level for final transpilation.
    call_limit : int
        Layer 1: VF2 call limit.
    max_layouts : int
        Layer 1: maximum VF2 layouts to evaluate.
    exclude_qubits : set[int] | None
        Layer 0: manual qubit blacklist.
    defective_edge_threshold : float
        Layer 2: edge error above this triggers fidelity penalty.
    strategy : {"lowest_cost", "ces_spread", "hybrid"}
        Layout selection strategy:
        - "lowest_cost": Pick top-N with lowest fidelity cost (best for PEA).
        - "ces_spread": Pick layouts maximizing CES diversity (for GF-ZNE).
        - "hybrid": Top-(N-1) lowest cost + 1 higher-CES (for adaptive ZNE).

    Returns
    -------
    LayoutSelection
        Selected layouts with CES values and pre-transpiled circuits.
        Compatible with all downstream pipeline functions.
    """
    t_start = time.time()

    # ── Layer 0: Filter CouplingMap ──
    filtered_cmap, filter_stats = build_filtered_coupling_map(
        backend,
        max_2q_error=max_2q_error,
        min_t1_us=min_t1_us,
        exclude_qubits=exclude_qubits,
    )

    original_edges = filter_stats["original_edges"]
    filtered_edges = filter_stats["filtered_edges"]
    _logger.info(
        "Layer 0: CouplingMap filtered %d → %d edges (%.0f%% retained)",
        original_edges,
        filtered_edges,
        filter_stats["retention_rate"] * 100,
    )

    # ── Layer 1: VF2 discovery ──
    vf2_layouts: list[list[int]] = []
    method = "mapomatic_vf2"

    if MAPOMATIC_AVAILABLE:
        # Deflate circuit for VF2 (remove idle qubits)
        try:
            deflated = mm.deflate_circuit(circuit)
        except Exception:
            deflated = circuit

        vf2_layouts = find_vf2_layouts(
            deflated,
            filtered_cmap,
            call_limit=call_limit,
            strict_direction=True,
            max_layouts=max_layouts,
        )
        _logger.info("Layer 1: VF2 found %d candidate layouts", len(vf2_layouts))

    # Fallback to BFS if VF2 yields nothing
    if not vf2_layouts:
        _logger.info("Layer 1: VF2 returned 0 layouts — falling back to BFS")
        method = "bfs_fallback"
        from qmbp_simulation.execution.noisy_utils import build_adjacency

        adj = build_adjacency(backend)
        vf2_layouts = find_layouts_bfs(
            adj,
            circuit.num_qubits,
            n_candidates=40,
            seed=42,
        )
        _logger.info("Layer 1 (BFS fallback): found %d candidates", len(vf2_layouts))

    total_found = len(vf2_layouts)

    if not vf2_layouts:
        _logger.warning("No layouts found by either VF2 or BFS. Returning empty.")
        return LayoutSelection()

    # ── Layer 2: Score by fidelity cost ──
    scored = compute_layout_fidelity_cost(
        circuit if method == "bfs_fallback" else deflated,
        vf2_layouts,
        backend,
        defective_edge_threshold=defective_edge_threshold,
        include_readout=True,
    )

    # Filter out penalized layouts (cost > 1.0 means defective)
    scored_clean = [(lay, cost) for lay, cost in scored if cost < 1.0]
    if not scored_clean:
        # All have defects — take the least bad
        scored_clean = scored[:n_select]
        _logger.warning(
            "All %d layouts have defective edges. Using least-bad %d.",
            len(scored),
            len(scored_clean),
        )

    _logger.info(
        "Layer 2: %d layouts pass quality filter (of %d scored)",
        len(scored_clean),
        len(scored),
    )

    # ── Layer 3: Strategy-based selection + transpilation ──
    selected_layouts, selected_costs = _apply_strategy(
        scored_clean, n_select=n_select, strategy=strategy
    )

    # Transpile only the final selected layouts
    transpiled_circuits: list[QuantumCircuit] = []
    ces_values: list[float] = []
    final_layouts: list[list[int]] = []
    final_costs: list[float] = []

    for layout, cost in zip(selected_layouts, selected_costs, strict=True):
        pm = generate_preset_pass_manager(
            optimization_level=optimization_level,
            backend=backend,
            initial_layout=layout,
        )
        transpiled = pm.run(circuit)
        ces, _ = compute_circuit_ces(transpiled, backend)

        # Apply max_ces filter
        if ces <= max_ces:
            transpiled_circuits.append(transpiled)
            ces_values.append(ces)
            final_layouts.append(layout)
            final_costs.append(cost)
        else:
            _logger.debug(
                "Layout %s excluded: CES=%.3f > max_ces=%.3f",
                layout[:3],
                ces,
                max_ces,
            )

    # If all filtered out by CES, take the single lowest-CES from original
    if not final_layouts and selected_layouts:
        _logger.warning("All selected layouts exceed max_ces=%.3f. Using lowest.", max_ces)
        layout = selected_layouts[0]
        pm = generate_preset_pass_manager(
            optimization_level=optimization_level,
            backend=backend,
            initial_layout=layout,
        )
        transpiled = pm.run(circuit)
        ces, _ = compute_circuit_ces(transpiled, backend)
        transpiled_circuits.append(transpiled)
        ces_values.append(ces)
        final_layouts.append(layout)
        final_costs.append(selected_costs[0])

    elapsed = time.time() - t_start
    _logger.info(
        "Layout optimization complete: %d layouts selected in %.2fs "
        "(method=%s, strategy=%s, VF2_found=%d)",
        len(final_layouts),
        elapsed,
        method,
        strategy,
        total_found,
    )

    return LayoutSelection(
        layouts=final_layouts,
        ces_values=ces_values,
        transpiled_circuits=transpiled_circuits,
    )


def _apply_strategy(
    scored: list[tuple[list[int], float]],
    *,
    n_select: int,
    strategy: str,
) -> tuple[list[list[int]], list[float]]:
    """Apply selection strategy to scored layouts.

    Parameters
    ----------
    scored : list[tuple[list[int], float]]
        Sorted (layout, cost) pairs (best first).
    n_select : int
        Number of layouts to pick.
    strategy : str
        One of "lowest_cost", "ces_spread", "hybrid".

    Returns
    -------
    tuple[list[list[int]], list[float]]
        Selected layouts and their costs.
    """
    if not scored:
        return [], []

    n_avail = len(scored)
    n_pick = min(n_select, n_avail)

    if strategy == "lowest_cost":
        # Simply take the top-N best (lowest cost)
        picked = scored[:n_pick]

    elif strategy == "ces_spread":
        # Pick layouts spread across the cost range for CES diversity
        if n_avail <= n_pick:
            picked = scored
        else:
            indices = [0]  # Always include best
            if n_pick > 2:
                step = (n_avail - 1) / (n_pick - 1)
                for i in range(1, n_pick - 1):
                    indices.append(int(round(i * step)))
            indices.append(n_avail - 1)  # Include worst of the clean set
            indices = sorted(set(indices))[:n_pick]
            picked = [scored[i] for i in indices]

    elif strategy == "hybrid":
        # Top-(N-1) lowest + 1 from higher cost (for adaptive ZNE)
        if n_avail <= n_pick:
            picked = scored
        else:
            top = scored[: n_pick - 1]
            # Pick one from the upper quartile (but not defective)
            upper_idx = max(n_pick - 1, int(n_avail * 0.75))
            upper_pick = scored[min(upper_idx, n_avail - 1)]
            picked = top + [upper_pick]

    else:
        _logger.warning("Unknown strategy '%s', falling back to lowest_cost", strategy)
        picked = scored[:n_pick]

    layouts = [p[0] for p in picked]
    costs = [p[1] for p in picked]
    return layouts, costs


# ═══════════════════════════════════════════════════════════════════════════
# Multi-backend ranking
# ═══════════════════════════════════════════════════════════════════════════


def rank_backends(
    circuit: QuantumCircuit,
    backends: list,
    *,
    n_top: int = 3,
    max_2q_error: float = 0.01,
    min_t1_us: float = 50.0,
    call_limit: int = 100_000,
    cost_function: Callable | None = None,
) -> list[tuple[list[int], str, float]]:
    """Rank multiple backends for a given circuit.

    For each backend, finds VF2 layouts on the filtered CouplingMap,
    scores them, and returns the best layout per backend sorted globally
    by cost.

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to deploy (deflated or bound).
    backends : list[BackendV2]
        List of candidate backends.
    n_top : int
        Return top-N results across all backends.
    max_2q_error : float
        Layer 0 filter threshold.
    min_t1_us : float
        Layer 0 filter threshold.
    call_limit : int
        VF2 call limit per backend.
    cost_function : Callable | None
        Custom cost function. If None, uses compute_layout_fidelity_cost.

    Returns
    -------
    list[tuple[list[int], str, float]]
        Sorted list of (best_layout, backend_name, cost).
        Best overall first.
    """
    results: list[tuple[list[int], str, float]] = []

    for backend in backends:
        try:
            backend_name = getattr(backend, "name", str(backend))

            # Layer 0
            filtered_cmap, _ = build_filtered_coupling_map(
                backend,
                max_2q_error=max_2q_error,
                min_t1_us=min_t1_us,
            )

            # Layer 1
            if MAPOMATIC_AVAILABLE:
                try:
                    deflated = mm.deflate_circuit(circuit)
                except Exception:
                    deflated = circuit
                layouts = find_vf2_layouts(deflated, filtered_cmap, call_limit=call_limit)
            else:
                layouts = []

            if not layouts:
                continue

            # Layer 2
            if cost_function is not None:
                scored = cost_function(circuit, layouts, backend)
            else:
                scored = compute_layout_fidelity_cost(
                    deflated if MAPOMATIC_AVAILABLE else circuit,
                    layouts,
                    backend,
                )

            if scored:
                best_layout, best_cost = scored[0]
                results.append((best_layout, backend_name, best_cost))

        except Exception as exc:
            _logger.warning("Failed to evaluate backend %s: %s", backend, exc)
            continue

    results.sort(key=lambda x: x[2])
    return results[:n_top]
