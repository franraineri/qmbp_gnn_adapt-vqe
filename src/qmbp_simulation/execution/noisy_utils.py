"""Noisy simulation utilities — layout selection, CES computation, ZNE.

Provides reusable building blocks for inhomogeneous ZNE experiments on
FakeTorino (or any BackendV2). Centralizes the patterns that were previously
duplicated across scripts/run_noisy_v2_batch.py, run_v1_p1_noisy.py,
run_v3_per_obs_zne.py, and run_zne_robustness.py.

Key design decisions (from auditoría 2026-05-25):
  - ALWAYS pass seed_simulator for reproducibility.
  - ALWAYS pass default_precision = 1/sqrt(shots) for correct shot count.
  - ALWAYS use circuit CES (post-transpilation) for layout selection,
    never topology CES (pre-transpilation).

Usage:
    from qmbp_simulation.execution.noisy_utils import (
        build_adjacency,
        find_layouts_bfs,
        compute_circuit_ces,
        select_layouts_by_circuit_ces,
        noisy_estimate,
        linear_zne,
        NoisyEstimatorConfig,
    )
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class NoisyEstimatorConfig:
    """Configuration for reproducible noisy estimation.

    Encapsulates the three critical parameters that were previously
    missing or inconsistent across scripts.

    Parameters
    ----------
    shots : int
        Number of shots for the simulation. Determines precision via
        default_precision = 1/sqrt(shots).
    seed_simulator : int
        Base seed for the simulator RNG. Each layout/call gets
        seed_simulator + offset for independence while maintaining
        reproducibility.
    optimization_level : int
        Transpiler optimization level (default 2 for production).
    """

    shots: int = 16384
    seed_simulator: int = 42
    optimization_level: int = 2

    @property
    def precision(self) -> float:
        """Correct precision for BackendEstimatorV2: 1/sqrt(shots)."""
        return 1.0 / np.sqrt(self.shots)


# ═══════════════════════════════════════════════════════════════════════════
# Topology utilities
# ═══════════════════════════════════════════════════════════════════════════


def build_adjacency(backend) -> dict[int, list[int]]:
    """Build adjacency graph from backend target.

    Parameters
    ----------
    backend : BackendV2
        Any Qiskit BackendV2 (FakeTorino, real hardware, etc.)

    Returns
    -------
    dict[int, list[int]]
        Adjacency list: qubit_id -> list of neighbor qubit_ids.
    """
    adj: dict[int, list[int]] = {}
    target = backend.target
    for op_name in target.operation_names:
        qargs = target.qargs_for_operation_name(op_name)
        if qargs is None:
            continue
        for qa in qargs:
            if len(qa) == 2:
                q0, q1 = qa
                adj.setdefault(q0, []).append(q1)
                adj.setdefault(q1, []).append(q0)
    # Deduplicate
    for k in adj:
        adj[k] = list(set(adj[k]))
    return adj


def find_layouts_bfs(
    adj: dict[int, list[int]],
    n_qubits: int,
    n_candidates: int = 40,
    seed: int = 42,
) -> list[list[int]]:
    """Find connected subsets of size n_qubits via BFS on the topology.

    Parameters
    ----------
    adj : dict[int, list[int]]
        Adjacency list from build_adjacency().
    n_qubits : int
        Required subset size (e.g. 6, 10).
    n_candidates : int
        Maximum number of candidate layouts to find.
    seed : int
        RNG seed for reproducible starting-node selection.

    Returns
    -------
    list[list[int]]
        List of layouts (each a list of physical qubit indices).
        Deduplicated by sorted tuple.
    """
    rng = random.Random(seed)
    all_nodes = list(adj.keys())
    found: list[list[int]] = []
    seen_keys: set[tuple[int, ...]] = set()
    starts = rng.sample(all_nodes, min(100, len(all_nodes)))

    for start in starts:
        visited = [start]
        queue = deque(adj.get(start, []))
        visited_set = {start}
        while queue and len(visited) < n_qubits:
            node = queue.popleft()
            if node in visited_set:
                continue
            visited_set.add(node)
            visited.append(node)
            for nb in adj.get(node, []):
                if nb not in visited_set:
                    queue.append(nb)
        if len(visited) >= n_qubits:
            key = tuple(sorted(visited[:n_qubits]))
            if key not in seen_keys:
                seen_keys.add(key)
                found.append(visited[:n_qubits])
        if len(found) >= n_candidates:
            break

    return found


# ═══════════════════════════════════════════════════════════════════════════
# Circuit Error Score (CES)
# ═══════════════════════════════════════════════════════════════════════════


def compute_circuit_ces(transpiled: QuantumCircuit, backend) -> tuple[float, int]:
    """Compute ACTUAL circuit CES from a transpiled circuit.

    CES = sum of 2-qubit gate errors for all 2Q gates in the transpiled
    circuit. This is the correct metric for ZNE layout diversity — NOT
    the topology CES (which ignores SWAP routing).

    Parameters
    ----------
    transpiled : QuantumCircuit
        Already-transpiled circuit (bound parameters, ISA-compliant).
    backend : BackendV2
        Backend whose target provides gate error rates.

    Returns
    -------
    tuple[float, int]
        (ces_value, n_2q_gates) — total CES and count of 2Q gates.
    """
    ces = 0.0
    n_2q = 0
    target = backend.target
    for inst in transpiled.data:
        if inst.operation.num_qubits == 2:
            n_2q += 1
            qubits = [transpiled.find_bit(q).index for q in inst.qubits]
            q0, q1 = min(qubits), max(qubits)
            gate_props = target[inst.operation.name].get((q0, q1))
            if gate_props and gate_props.error is not None:
                ces += gate_props.error
            else:
                ces += 0.01  # Default error for unknown gates
    return ces, n_2q


@dataclass
class LayoutSelection:
    """Result of circuit-CES-based layout selection.

    Attributes
    ----------
    layouts : list[list[int]]
        Selected physical qubit layouts.
    ces_values : list[float]
        Circuit CES for each selected layout.
    transpiled_circuits : list[QuantumCircuit]
        Pre-transpiled circuits (one per layout). Reuse these to avoid
        redundant transpilation.
    """

    layouts: list[list[int]] = field(default_factory=list)
    ces_values: list[float] = field(default_factory=list)
    transpiled_circuits: list[QuantumCircuit] = field(default_factory=list)


def select_layouts_by_circuit_ces(
    bound_circuit: QuantumCircuit,
    backend,
    candidate_layouts: list[list[int]],
    n_select: int = 3,
    optimization_level: int = 2,
) -> LayoutSelection:
    """Select layouts by ACTUAL circuit CES with maximum spread.

    Transpiles the bound circuit to each candidate layout, computes the
    real circuit CES, then picks n_select layouts with maximum CES
    diversity (first, last, and evenly spaced in between).

    This is more expensive than topology CES but gives TRUE diversity.
    The transpiled circuits are cached in the result to avoid re-transpilation.

    Parameters
    ----------
    bound_circuit : QuantumCircuit
        Parameter-bound circuit to transpile.
    backend : BackendV2
        Target backend (e.g. FakeTorino).
    candidate_layouts : list[list[int]]
        Candidate layouts from find_layouts_bfs().
    n_select : int
        Number of layouts to select.
    optimization_level : int
        Transpiler optimization level.

    Returns
    -------
    LayoutSelection
        Selected layouts with CES values and pre-transpiled circuits.
    """
    circuit_ces_list: list[float] = []
    transpiled_list: list[QuantumCircuit] = []

    for layout in candidate_layouts:
        pm = generate_preset_pass_manager(
            optimization_level=optimization_level,
            backend=backend,
            initial_layout=layout,
        )
        transpiled = pm.run(bound_circuit)
        ces, _ = compute_circuit_ces(transpiled, backend)
        circuit_ces_list.append(ces)
        transpiled_list.append(transpiled)

    # Select n_select with max spread (first + last + evenly spaced)
    sorted_idx = np.argsort(circuit_ces_list)
    if len(sorted_idx) >= n_select:
        indices = [sorted_idx[0]]
        if n_select > 2:
            step = (len(sorted_idx) - 1) / (n_select - 1)
            for i in range(1, n_select - 1):
                indices.append(sorted_idx[int(round(i * step))])
        indices.append(sorted_idx[-1])
        indices = sorted(set(indices))[:n_select]
    else:
        indices = list(sorted_idx)

    return LayoutSelection(
        layouts=[candidate_layouts[i] for i in indices],
        ces_values=[circuit_ces_list[i] for i in indices],
        transpiled_circuits=[transpiled_list[i] for i in indices],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Noisy estimation with correct options
# ═══════════════════════════════════════════════════════════════════════════


def noisy_estimate(
    transpiled: QuantumCircuit,
    observable: SparsePauliOp,
    backend,
    config: NoisyEstimatorConfig,
    seed_offset: int = 0,
) -> float:
    """Execute a single noisy estimation with correct seed and precision.

    This is the ONLY correct way to call BackendEstimatorV2 in this project.
    It enforces:
      - seed_simulator for reproducibility
      - default_precision = 1/sqrt(shots) for correct shot count

    Parameters
    ----------
    transpiled : QuantumCircuit
        Already-transpiled, parameter-bound circuit.
    observable : SparsePauliOp
        Observable to measure (already layout-mapped).
    backend : BackendV2
        Noisy backend (e.g. FakeTorino).
    config : NoisyEstimatorConfig
        Estimation configuration (shots, seed, etc.)
    seed_offset : int
        Added to config.seed_simulator for per-call independence.

    Returns
    -------
    float
        Expectation value ⟨O⟩.
    """
    from qiskit.primitives import BackendEstimatorV2

    estimator = BackendEstimatorV2(
        backend=backend,
        options={
            "seed_simulator": config.seed_simulator + seed_offset,
            "default_precision": config.precision,
        },
    )
    job = estimator.run([(transpiled, observable)])
    return float(job.result()[0].data.evs)


def noisy_estimate_batch(
    transpiled: QuantumCircuit,
    observables: list[SparsePauliOp],
    backend,
    config: NoisyEstimatorConfig,
    seed_offset: int = 0,
) -> list[float]:
    """Execute multiple observable measurements on the same transpiled circuit.

    Uses a single estimator instance for all observables (same seed/precision),
    which is correct since BackendEstimatorV2 is stateless between .run() calls.

    Parameters
    ----------
    transpiled : QuantumCircuit
        Already-transpiled, parameter-bound circuit.
    observables : list[SparsePauliOp]
        Observables to measure (already layout-mapped).
    backend : BackendV2
        Noisy backend.
    config : NoisyEstimatorConfig
        Estimation configuration.
    seed_offset : int
        Added to config.seed_simulator.

    Returns
    -------
    list[float]
        Expectation values for each observable.
    """
    from qiskit.primitives import BackendEstimatorV2

    estimator = BackendEstimatorV2(
        backend=backend,
        options={
            "seed_simulator": config.seed_simulator + seed_offset,
            "default_precision": config.precision,
        },
    )
    results = []
    for obs in observables:
        job = estimator.run([(transpiled, obs)])
        results.append(float(job.result()[0].data.evs))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# ZNE extrapolation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ZNEResult:
    """Result of a linear ZNE extrapolation.

    Attributes
    ----------
    extrapolated_value : float
        Value at CES=0 (the mitigated estimate).
    r_squared : float
        R² of the linear fit. R²>0.8 indicates good linearity.
    slope : float
        Slope of E(CES) — positive means noise increases energy.
    ces_values : np.ndarray
        CES values used in the fit.
    measured_values : np.ndarray
        Measured values at each CES point.
    """

    extrapolated_value: float
    r_squared: float
    slope: float
    ces_values: np.ndarray
    measured_values: np.ndarray


def linear_zne(ces_values: np.ndarray, measured_values: np.ndarray) -> ZNEResult:
    """Linear ZNE extrapolation to CES=0.

    Fits a line E(CES) = a*CES + b and extrapolates to CES=0.

    Parameters
    ----------
    ces_values : np.ndarray
        Circuit Error Scores for each layout.
    measured_values : np.ndarray
        Measured expectation values at each CES point.

    Returns
    -------
    ZNEResult
        Extrapolation result with R², slope, and raw data.
    """
    ces_arr = np.asarray(ces_values, dtype=float)
    vals_arr = np.asarray(measured_values, dtype=float)

    if len(ces_arr) < 2 or np.std(ces_arr) < 1e-10:
        return ZNEResult(
            extrapolated_value=float(np.mean(vals_arr)),
            r_squared=0.0,
            slope=0.0,
            ces_values=ces_arr,
            measured_values=vals_arr,
        )

    coeffs = np.polyfit(ces_arr, vals_arr, 1)
    extrap = float(np.polyval(coeffs, 0.0))
    y_pred = np.polyval(coeffs, ces_arr)
    ss_res = np.sum((vals_arr - y_pred) ** 2)
    ss_tot = np.sum((vals_arr - np.mean(vals_arr)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0

    return ZNEResult(
        extrapolated_value=extrap,
        r_squared=r2,
        slope=float(coeffs[0]),
        ces_values=ces_arr,
        measured_values=vals_arr,
    )


# ═══════════════════════════════════════════════════════════════════════════
# High-level ZNE deployment
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ZNEDeploymentResult:
    """Full ZNE deployment result for one h-value.

    Attributes
    ----------
    energy_zne : ZNEResult
        ZNE extrapolation of total energy.
    per_site_zne : list[ZNEResult] | None
        Per-site ⟨X_i⟩ ZNE results (if per_site=True).
    per_layout_data : list[dict]
        Raw data per layout: {ces, n_2q, energy, per_site_x}.
    """

    energy_zne: ZNEResult
    per_site_zne: list[ZNEResult] | None = None
    per_layout_data: list[dict] = field(default_factory=list)


def run_zne_deployment(
    bound_circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    backend,
    layout_selection: LayoutSelection,
    config: NoisyEstimatorConfig,
    n_qubits: int,
    per_site: bool = False,
) -> ZNEDeploymentResult:
    """Run full ZNE deployment: measure energy (and optionally per-site X)
    across multiple layouts, then extrapolate to CES=0.

    Parameters
    ----------
    bound_circuit : QuantumCircuit
        Parameter-bound circuit (not yet transpiled — use layout_selection
        which already has pre-transpiled circuits).
    hamiltonian : SparsePauliOp
        Total Hamiltonian to measure.
    backend : BackendV2
        Noisy backend.
    layout_selection : LayoutSelection
        Pre-computed layout selection from select_layouts_by_circuit_ces().
    config : NoisyEstimatorConfig
        Estimation configuration.
    n_qubits : int
        Number of logical qubits.
    per_site : bool
        If True, also measure per-site ⟨X_i⟩ for each layout.

    Returns
    -------
    ZNEDeploymentResult
        Complete ZNE results including extrapolation and raw data.
    """
    per_layout_data = []

    for li, transpiled in enumerate(layout_selection.transpiled_circuits):
        ces = layout_selection.ces_values[li]
        _, n_2q = compute_circuit_ces(transpiled, backend)

        # Energy measurement
        h_mapped = hamiltonian.apply_layout(transpiled.layout)
        energy = noisy_estimate(transpiled, h_mapped, backend, config, seed_offset=li)

        # Per-site X_i (optional)
        per_site_x: list[float] | None = None
        if per_site:
            x_obs_mapped = []
            for i in range(n_qubits):
                op = SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=n_qubits)
                x_obs_mapped.append(op.apply_layout(transpiled.layout))
            per_site_x = noisy_estimate_batch(
                transpiled, x_obs_mapped, backend, config, seed_offset=li
            )

        per_layout_data.append(
            {
                "ces": ces,
                "n_2q": n_2q,
                "energy": energy,
                "per_site_x": per_site_x,
            }
        )

    # ZNE extrapolation — energy
    ces_arr = np.array([d["ces"] for d in per_layout_data])
    e_arr = np.array([d["energy"] for d in per_layout_data])
    energy_zne = linear_zne(ces_arr, e_arr)

    # ZNE extrapolation — per-site (if requested)
    per_site_zne_results: list[ZNEResult] | None = None
    if per_site:
        per_site_zne_results = []
        for site_i in range(n_qubits):
            site_vals = np.array([d["per_site_x"][site_i] for d in per_layout_data])
            per_site_zne_results.append(linear_zne(ces_arr, site_vals))

    return ZNEDeploymentResult(
        energy_zne=energy_zne,
        per_site_zne=per_site_zne_results,
        per_layout_data=per_layout_data,
    )
