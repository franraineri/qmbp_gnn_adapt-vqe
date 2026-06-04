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

Three ZNE strategies:
  1. **Inhomogeneous CES-ZNE** (original): different layouts → different CES
     → linear extrapolation to CES=0. Fails on heavy_hex (uniform CES).
  2. **Gate-folding ZNE** (2026-06-04): single layout, amplify noise
     via gate folding (U→UU†U) at factors [1,3,5] → extrapolation to
     noise_factor=0. Works regardless of topology/layout uniformity.
  3. **PEA-ZNE** (2026-06-04): Probabilistic Error Amplification. Learns
     the noise model from backend calibration data, then amplifies noise
     by scaling depolarizing rates at factors [1,3,5]. Preserves circuit
     structure (no depth increase). More physically accurate than gate-folding
     for correlated noise. ~50% overhead from noise learning on real hardware.

Usage:
    from qmbp_simulation.execution.noisy_utils import (
        build_adjacency,
        find_layouts_bfs,
        compute_circuit_ces,
        select_layouts_by_circuit_ces,
        noisy_estimate,
        linear_zne,
        NoisyEstimatorConfig,
        # Gate-folding ZNE
        fold_gates,
        run_gate_folding_zne,
        GateFoldingZNEResult,
        # PEA (Probabilistic Error Amplification)
        run_pea_zne,
        run_pea_zne_deployment,
        PEAResult,
        PEADeploymentResult,
    )
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

_logger = logging.getLogger(__name__)

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


def select_layouts_low_ces(
    bound_circuit: QuantumCircuit,
    backend,
    candidate_layouts: list[list[int]],
    n_select: int = 3,
    optimization_level: int = 2,
    max_ces: float | None = None,
) -> LayoutSelection:
    """Select layouts with LOWEST CES values (perturbative regime).

    Unlike select_layouts_by_circuit_ces which maximizes CES spread for
    ZNE extrapolation, this function picks the n_select layouts with the
    lowest total CES. This is optimal for p=1 circuits where:
      - The circuit is already shallow (few CX gates)
      - We want ALL layouts in the perturbative regime (low CES)
      - ZNE works best when all points are in the linear E(CES) region

    Validated by multi-seed p=1 ZNE experiment (2026-05-28):
      - seed 42 (low CES layouts): +73% gain
      - seed 44 (high CES layout): -39% gain
    The difference is entirely due to layout CES values.

    Parameters
    ----------
    bound_circuit : QuantumCircuit
        Parameter-bound circuit to transpile.
    backend : BackendV2
        Target backend (e.g. FakeTorino).
    candidate_layouts : list[list[int]]
        Candidate layouts from find_layouts_bfs().
    n_select : int
        Number of layouts to select (default: 3).
    optimization_level : int
        Transpiler optimization level (default: 2).
    max_ces : float | None
        Maximum allowed CES per layout. Layouts above this are excluded.
        If None, no filtering is applied (just picks lowest n_select).

    Returns
    -------
    LayoutSelection
        Selected layouts sorted by CES (lowest first).
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

    # Sort by CES ascending (lowest first)
    sorted_idx = np.argsort(circuit_ces_list)

    # Apply max_ces filter if specified
    if max_ces is not None:
        sorted_idx = [i for i in sorted_idx if circuit_ces_list[i] <= max_ces]

    # Take the n_select lowest
    indices = list(sorted_idx[:n_select])

    if len(indices) == 0:
        # Fallback: if all layouts exceed max_ces, take the single lowest
        indices = [int(np.argmin(circuit_ces_list))]

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


# ═══════════════════════════════════════════════════════════════════════════
# Gate-folding ZNE — noise amplification via U → U·U†·U
# ═══════════════════════════════════════════════════════════════════════════


TWO_QUBIT_GATES = frozenset({"cx", "cz", "ecr", "rzz", "rxx", "ryy", "cp"})


def fold_gates(circuit: QuantumCircuit, noise_factor: int = 1) -> QuantumCircuit:
    """Apply digital gate folding to amplify noise on 2-qubit gates.

    Replaces each 2-qubit gate U with U·(U†·U)^k where noise_factor = 2k+1.
    For noise_factor=1 (no amplification), returns the circuit unchanged.
    For noise_factor=3, each 2Q gate becomes U·U†·U (3× noise).
    For noise_factor=5, each 2Q gate becomes U·U†·U·U†·U (5× noise).

    This is the "digital gate folding" method described in IBM's ZNE
    documentation. It works regardless of layout/topology uniformity.

    Parameters
    ----------
    circuit : QuantumCircuit
        Transpiled (ISA) circuit to fold. Must be parameter-free.
    noise_factor : int
        Odd integer ≥ 1. Number of times each 2Q gate is effectively
        applied (1 = original, 3 = one fold, 5 = two folds).

    Returns
    -------
    QuantumCircuit
        Folded circuit with amplified noise on 2-qubit gates.

    Raises
    ------
    ValueError
        If noise_factor is not an odd positive integer.
    """
    if noise_factor < 1 or noise_factor % 2 == 0:
        raise ValueError(
            f"noise_factor must be an odd positive integer, got {noise_factor}. "
            f"Valid values: 1 (no fold), 3 (1 fold), 5 (2 folds), etc."
        )
    if noise_factor == 1:
        _logger.debug("[gate_folding] noise_factor=1, returning circuit unchanged")
        return circuit.copy()

    n_folds = (noise_factor - 1) // 2
    _logger.debug(
        f"[gate_folding] noise_factor={noise_factor}, n_folds={n_folds}, "
        f"circuit depth={circuit.depth()}"
    )

    folded = QuantumCircuit(circuit.qubits, circuit.clbits, name=f"folded_{noise_factor}x")
    n_2q_folded = 0
    n_1q_kept = 0

    for instruction in circuit.data:
        gate = instruction.operation
        qubits = instruction.qubits
        gate_name = gate.name.lower()

        # Always add the original gate
        folded.append(instruction)

        # Only fold 2-qubit gates (where noise dominates)
        if gate_name in TWO_QUBIT_GATES:
            for _ in range(n_folds):
                # U† (inverse)
                folded.append(gate.inverse(), qubits, [])
                # U (original again)
                folded.append(gate, qubits, [])
            n_2q_folded += 1
        else:
            n_1q_kept += 1

    _logger.debug(
        f"[gate_folding] Folded {n_2q_folded} 2Q gates × {n_folds} folds, "
        f"{n_1q_kept} 1Q gates unchanged. "
        f"New depth: {folded.depth()} (was {circuit.depth()})"
    )
    return folded


@dataclass
class GateFoldingZNEResult:
    """Result of gate-folding ZNE extrapolation.

    Attributes
    ----------
    extrapolated_value : float
        Energy extrapolated to noise_factor=0 (mitigated estimate).
    r_squared : float
        R² of the linear fit E(noise_factor).
    slope : float
        Slope of E vs noise_factor — indicates noise sensitivity.
    noise_factors : list[int]
        Noise amplification factors used (e.g. [1, 3, 5]).
    measured_values : list[float]
        Measured energies at each noise factor.
    extrapolator : str
        Extrapolation method used ("linear" or "exponential").
    """

    extrapolated_value: float
    r_squared: float
    slope: float
    noise_factors: list[int]
    measured_values: list[float]
    extrapolator: str = "linear"


def _extrapolate_linear(
    noise_factors: np.ndarray, measured: np.ndarray
) -> tuple[float, float, float]:
    """Linear extrapolation E(nf) = a*nf + b → E(0) = b."""
    coeffs = np.polyfit(noise_factors, measured, 1)
    extrap = float(np.polyval(coeffs, 0.0))
    y_pred = np.polyval(coeffs, noise_factors)
    ss_res = np.sum((measured - y_pred) ** 2)
    ss_tot = np.sum((measured - np.mean(measured)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0
    return extrap, r2, float(coeffs[0])


def _extrapolate_exponential(
    noise_factors: np.ndarray, measured: np.ndarray
) -> tuple[float, float, float]:
    """Exponential extrapolation E(nf) = a * exp(b*nf) + c → E(0) = a + c.

    Falls back to linear if exponential fit fails (e.g. non-monotonic data).
    """
    try:
        from scipy.optimize import curve_fit

        def exp_model(x, a, b, c):
            return a * np.exp(b * x) + c

        # Initial guess: a=range, b=positive(noise increases E), c=min
        e_range = float(np.max(measured) - np.min(measured))
        p0 = [e_range, 0.1, float(np.min(measured))]
        popt, _ = curve_fit(exp_model, noise_factors, measured, p0=p0, maxfev=5000)
        extrap = float(exp_model(0, *popt))
        y_pred = exp_model(noise_factors, *popt)
        ss_res = np.sum((measured - y_pred) ** 2)
        ss_tot = np.sum((measured - np.mean(measured)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0
        slope = float(popt[0] * popt[1])  # derivative at 0
        _logger.debug(
            f"[gate_folding_zne] Exponential fit: a={popt[0]:.4f}, "
            f"b={popt[1]:.4f}, c={popt[2]:.4f}, R²={r2:.4f}"
        )
        return extrap, r2, slope
    except Exception as e:
        _logger.warning(
            f"[gate_folding_zne] Exponential fit failed ({e}), falling back to linear extrapolation"
        )
        return _extrapolate_linear(noise_factors, measured)


def run_gate_folding_zne(
    transpiled_circuit: QuantumCircuit,
    observable: SparsePauliOp,
    backend,
    config: NoisyEstimatorConfig,
    noise_factors: tuple[int, ...] = (1, 3, 5),
    extrapolator: str = "linear",
    seed_offset: int = 0,
) -> GateFoldingZNEResult:
    """Run gate-folding ZNE on a single transpiled circuit.

    Amplifies noise by folding 2-qubit gates at each noise factor,
    then extrapolates to zero noise.

    Parameters
    ----------
    transpiled_circuit : QuantumCircuit
        Already-transpiled, parameter-bound ISA circuit.
    observable : SparsePauliOp
        Observable to measure (already layout-mapped via apply_layout).
    backend : BackendV2
        Noisy backend (e.g. FakeTorino).
    config : NoisyEstimatorConfig
        Shots and seed configuration.
    noise_factors : tuple[int, ...]
        Odd integers for noise amplification. Default (1, 3, 5).
    extrapolator : str
        "linear" or "exponential". Default "linear".
    seed_offset : int
        Added to config.seed_simulator for independence across calls.

    Returns
    -------
    GateFoldingZNEResult
        Extrapolation result with R², slope, and per-factor measurements.
    """
    _logger.info(
        f"[gate_folding_zne] Starting ZNE with noise_factors={noise_factors}, "
        f"extrapolator={extrapolator}, shots={config.shots}"
    )

    # Validate noise factors
    for nf in noise_factors:
        if nf < 1 or nf % 2 == 0:
            raise ValueError(f"All noise_factors must be odd positive integers, got {nf}")

    if len(noise_factors) < 2:
        raise ValueError(
            f"At least 2 noise factors required for extrapolation, got {len(noise_factors)}"
        )

    # Validate circuit has no unbound parameters
    if transpiled_circuit.num_parameters > 0:
        raise ValueError(
            f"Circuit has {transpiled_circuit.num_parameters} unbound parameters. "
            f"Bind parameters before calling run_gate_folding_zne()."
        )

    measured_values: list[float] = []

    for i, nf in enumerate(noise_factors):
        folded = fold_gates(transpiled_circuit, noise_factor=nf)
        energy = noisy_estimate(
            folded,
            observable,
            backend,
            config,
            seed_offset=seed_offset + i * 100,
        )
        measured_values.append(energy)
        _logger.info(f"[gate_folding_zne]   factor={nf}: E={energy:.6f}, depth={folded.depth()}")

    # Extrapolation
    nf_arr = np.array(noise_factors, dtype=float)
    meas_arr = np.array(measured_values, dtype=float)

    if extrapolator == "exponential":
        extrap, r2, slope = _extrapolate_exponential(nf_arr, meas_arr)
        method_used = "exponential"
    else:
        extrap, r2, slope = _extrapolate_linear(nf_arr, meas_arr)
        method_used = "linear"

    _logger.info(
        f"[gate_folding_zne] Result: E_extrapolated={extrap:.6f}, "
        f"R²={r2:.4f}, slope={slope:.6f}, method={method_used}"
    )

    return GateFoldingZNEResult(
        extrapolated_value=extrap,
        r_squared=r2,
        slope=slope,
        noise_factors=list(noise_factors),
        measured_values=measured_values,
        extrapolator=method_used,
    )


@dataclass
class GateFoldingDeploymentResult:
    """Full gate-folding ZNE deployment result for one h-value.

    Combines layout averaging with gate-folding ZNE for best results.

    Attributes
    ----------
    energy_gf_zne : GateFoldingZNEResult
        Primary: gate-folding ZNE on the best (lowest-CES) layout.
    energy_layout_avg : float | None
        Secondary: simple average across multiple low-CES layouts.
    per_layout_gf_zne : list[GateFoldingZNEResult] | None
        Gate-folding ZNE per layout (if multi_layout=True).
    best_layout_idx : int
        Index of the layout used for primary ZNE.
    ces_values : list[float]
        CES of each layout used.
    """

    energy_gf_zne: GateFoldingZNEResult
    energy_layout_avg: float | None = None
    per_layout_gf_zne: list[GateFoldingZNEResult] | None = None
    best_layout_idx: int = 0
    ces_values: list[float] = field(default_factory=list)


def run_gate_folding_zne_deployment(
    bound_circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    backend,
    layout_selection: LayoutSelection,
    config: NoisyEstimatorConfig,
    noise_factors: tuple[int, ...] = (1, 3, 5),
    extrapolator: str = "linear",
    multi_layout: bool = False,
) -> GateFoldingDeploymentResult:
    """Run gate-folding ZNE deployment with optional multi-layout averaging.

    Strategy:
      - Primary: Gate-folding ZNE on the lowest-CES layout (most reliable).
      - Secondary (optional): Run GF-ZNE on all layouts, average extrapolated
        values for additional variance reduction.

    Parameters
    ----------
    bound_circuit : QuantumCircuit
        Parameter-bound circuit (layout_selection has pre-transpiled versions).
    hamiltonian : SparsePauliOp
        Total Hamiltonian to measure.
    backend : BackendV2
        Noisy backend.
    layout_selection : LayoutSelection
        Pre-computed layout selection (use select_layouts_low_ces).
    config : NoisyEstimatorConfig
        Estimation configuration.
    noise_factors : tuple[int, ...]
        Noise amplification factors for gate folding (default [1,3,5]).
    extrapolator : str
        "linear" or "exponential" (default "linear").
    multi_layout : bool
        If True, run GF-ZNE on ALL layouts and average results.

    Returns
    -------
    GateFoldingDeploymentResult
        Full deployment result with primary ZNE + optional layout averaging.
    """
    _logger.info(
        f"[gf_zne_deployment] Starting: {len(layout_selection.layouts)} layouts, "
        f"noise_factors={noise_factors}, multi_layout={multi_layout}"
    )

    ces_values = [float(c) for c in layout_selection.ces_values]

    # Use lowest-CES layout as primary (index 0, since select_layouts_low_ces sorts)
    best_idx = 0
    transpiled_best = layout_selection.transpiled_circuits[best_idx]
    h_mapped_best = hamiltonian.apply_layout(transpiled_best.layout)

    _logger.info(
        f"[gf_zne_deployment] Primary layout idx={best_idx}, "
        f"CES={ces_values[best_idx]:.4f}, depth={transpiled_best.depth()}"
    )

    primary_result = run_gate_folding_zne(
        transpiled_best,
        h_mapped_best,
        backend,
        config,
        noise_factors=noise_factors,
        extrapolator=extrapolator,
        seed_offset=0,
    )

    # Optional: multi-layout GF-ZNE for variance reduction
    per_layout_results: list[GateFoldingZNEResult] | None = None
    layout_avg: float | None = None

    if multi_layout and len(layout_selection.transpiled_circuits) > 1:
        per_layout_results = [primary_result]
        for li in range(1, len(layout_selection.transpiled_circuits)):
            transpiled_li = layout_selection.transpiled_circuits[li]
            h_mapped_li = hamiltonian.apply_layout(transpiled_li.layout)
            _logger.info(f"[gf_zne_deployment] Layout {li}: CES={ces_values[li]:.4f}")
            result_li = run_gate_folding_zne(
                transpiled_li,
                h_mapped_li,
                backend,
                config,
                noise_factors=noise_factors,
                extrapolator=extrapolator,
                seed_offset=(li + 1) * 1000,
            )
            per_layout_results.append(result_li)

        # Average extrapolated values across layouts
        extrap_values = [r.extrapolated_value for r in per_layout_results]
        layout_avg = float(np.mean(extrap_values))
        _logger.info(
            f"[gf_zne_deployment] Multi-layout average: {layout_avg:.6f} "
            f"(std={np.std(extrap_values):.6f}, n={len(extrap_values)})"
        )

    return GateFoldingDeploymentResult(
        energy_gf_zne=primary_result,
        energy_layout_avg=layout_avg,
        per_layout_gf_zne=per_layout_results,
        best_layout_idx=best_idx,
        ces_values=ces_values,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PEA (Probabilistic Error Amplification) — local simulation
# ═══════════════════════════════════════════════════════════════════════════
#
# PEA amplifies noise by injecting single-qubit Pauli errors proportional
# to the learned noise model, rather than digitally folding gates.
# This preserves the circuit structure while achieving more physically
# accurate noise amplification.
#
# For LOCAL simulation (FakeTorino), we approximate PEA by:
# 1. Learning the Pauli error rates from the backend's noise model
#    (extracting per-gate error probabilities from BackendV2 properties).
# 2. Constructing Pauli noise channels with amplified rates.
# 3. Running the circuit through AerSimulator with the amplified noise model.
#
# On REAL hardware (IBM Runtime), PEA is handled server-side via:
#   options.resilience.zne.amplifier = "pea"
# and the LayerNoiseLearning runs automatically.
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PEAResult:
    """Result of PEA-based ZNE extrapolation.

    Attributes
    ----------
    extrapolated_value : float
        Energy extrapolated to noise_factor=0 (mitigated estimate).
    r_squared : float
        R² of the extrapolation fit.
    slope : float
        Slope of E vs noise_factor.
    noise_factors : list[float]
        Noise amplification factors used (e.g. [1, 3, 5]).
    measured_values : list[float]
        Measured energies at each noise factor.
    extrapolator : str
        Extrapolation method used ("linear" or "exponential").
    learned_error_rates : dict[str, float] | None
        Summary of learned per-layer error rates (for diagnostics).
    """

    extrapolated_value: float
    r_squared: float
    slope: float
    noise_factors: list[float]
    measured_values: list[float]
    extrapolator: str = "linear"
    learned_error_rates: dict[str, float] | None = None


def _learn_noise_rates(backend) -> dict[tuple[int, ...], float]:
    """Learn per-gate error rates from backend properties.

    Extracts two-qubit gate error rates from the BackendV2 target,
    which represents the Pauli-Lindblad error model for each entangling
    gate in the device.

    Parameters
    ----------
    backend : BackendV2
        Backend with noise information (FakeTorino or real).

    Returns
    -------
    dict[tuple[int, ...], float]
        Mapping from qubit pair → gate error probability.
    """
    target = backend.target
    error_rates: dict[tuple[int, ...], float] = {}

    # Try CZ first (Torino native), then ECR, then CX
    for gate_name in ("cz", "ecr", "cx"):
        if gate_name not in target.operation_names:
            continue
        qargs = target.qargs_for_operation_name(gate_name)
        if qargs is None:
            continue
        for qa in qargs:
            if len(qa) == 2:
                props = target[gate_name].get(qa)
                if props is not None and props.error is not None:
                    error_rates[qa] = float(props.error)
        if error_rates:
            break

    if not error_rates:
        _logger.warning(
            "[pea] Could not extract gate error rates from backend. "
            "Falling back to uniform error rate of 0.01."
        )
        # Fallback: use topology to infer pairs with uniform error
        adj = build_adjacency(backend)
        for q, neighbors in adj.items():
            for n in neighbors:
                if (q, n) not in error_rates and (n, q) not in error_rates:
                    error_rates[(q, n)] = 0.01

    _logger.info(
        f"[pea] Learned noise rates for {len(error_rates)} gate pairs. "
        f"Mean error: {np.mean(list(error_rates.values())):.4f}, "
        f"Max error: {np.max(list(error_rates.values())):.4f}"
    )
    return error_rates


def _build_amplified_noise_model(
    backend,
    transpiled_circuit: QuantumCircuit,
    noise_factor: float,
    learned_rates: dict[tuple[int, ...], float],
):
    """Build a noise model with amplified depolarizing error on 2Q gates.

    Simulates PEA by scaling the depolarizing error rates on 2-qubit gates
    by the given noise_factor. For noise_factor=1, returns the original
    noise model. For noise_factor=3, the depolarizing probability is 3×.

    The amplification is capped so that the total error probability
    never exceeds 0.75 (the depolarizing channel limit for 2 qubits).

    Parameters
    ----------
    backend : BackendV2
        Original noisy backend.
    transpiled_circuit : QuantumCircuit
        The transpiled circuit (to identify which qubits are used).
    noise_factor : float
        Amplification factor (1 = original, 3 = 3× noise, etc.).
    learned_rates : dict[tuple[int, ...], float]
        Per-gate error rates from _learn_noise_rates().

    Returns
    -------
    NoiseModel
        Qiskit Aer noise model with amplified 2-qubit gate errors.
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    # Determine native 2Q gate name
    native_2q_gate = None
    for gate_name in ("cz", "ecr", "cx"):
        if gate_name in backend.target.operation_names:
            native_2q_gate = gate_name
            break

    if native_2q_gate is None:
        _logger.warning("[pea] No native 2Q gate found, returning original noise model")
        return NoiseModel.from_backend(backend)

    # For ALL factors (including 1), build a consistent depolarizing-only model.
    # This ensures the extrapolation axis is coherent: at factor=1 we get the
    # learned depolarizing noise, at factor=3 we get 3× that noise.
    # Using from_backend for factor=1 but clean model for factor>1 would
    # create an inconsistency that destroys R² (different noise types).
    amplified_model = NoiseModel()

    # Add amplified depolarizing noise on all learned 2Q gate pairs
    for pair, base_rate in learned_rates.items():
        amplified_rate = min(base_rate * noise_factor, 0.75)
        amp_error = depolarizing_error(amplified_rate, 2)
        amplified_model.add_quantum_error(amp_error, native_2q_gate, list(pair))

    # Add basis gates so AerSimulator accepts the circuits
    basis = list(backend.target.operation_names)
    amplified_model.add_basis_gates(basis)

    _logger.debug(
        f"[pea] Built amplified noise model: factor={noise_factor}, "
        f"n_2q_pairs={len(learned_rates)}, native_gate={native_2q_gate}"
    )
    return amplified_model


def _pea_estimate(
    transpiled_circuit: QuantumCircuit,
    observable: SparsePauliOp,
    backend,
    config: NoisyEstimatorConfig,
    noise_factor: float,
    learned_rates: dict[tuple[int, ...], float],
    seed_offset: int = 0,
) -> float:
    """Execute a single PEA-amplified noisy estimation.

    Instead of folding gates, builds a noise model with amplified
    error rates and runs the ORIGINAL circuit through it.

    Parameters
    ----------
    transpiled_circuit : QuantumCircuit
        Already-transpiled, parameter-bound ISA circuit.
    observable : SparsePauliOp
        Observable to measure (already layout-mapped).
    backend : BackendV2
        Original noisy backend (for noise model extraction).
    config : NoisyEstimatorConfig
        Estimation configuration.
    noise_factor : float
        Noise amplification factor (1.0 = no amplification).
    learned_rates : dict[tuple[int, ...], float]
        Per-gate error rates from _learn_noise_rates().
    seed_offset : int
        Added to config.seed_simulator for independence.

    Returns
    -------
    float
        Expectation value measured under amplified noise.
    """
    from qiskit.primitives import BackendEstimatorV2
    from qiskit_aer import AerSimulator

    amplified_model = _build_amplified_noise_model(
        backend, transpiled_circuit, noise_factor, learned_rates
    )

    # Create AerSimulator with amplified noise — from_backend preserves
    # coupling map and basis gates from the original backend.
    sim_backend = AerSimulator.from_backend(backend, noise_model=amplified_model)

    estimator = BackendEstimatorV2(
        backend=sim_backend,
        options={
            "seed_simulator": config.seed_simulator + seed_offset,
            "default_precision": config.precision,
        },
    )
    job = estimator.run([(transpiled_circuit, observable)])
    energy = float(job.result()[0].data.evs)

    if not np.isfinite(energy):
        _logger.warning(
            f"[pea] Non-finite energy at noise_factor={noise_factor}: {energy}. "
            f"Returning NaN (will degrade extrapolation R²)."
        )

    return energy


def run_pea_zne(
    transpiled_circuit: QuantumCircuit,
    observable: SparsePauliOp,
    backend,
    config: NoisyEstimatorConfig,
    noise_factors: tuple[float, ...] = (1, 3, 5),
    extrapolator: str = "linear",
    seed_offset: int = 0,
) -> PEAResult:
    """Run PEA-based ZNE on a single transpiled circuit.

    Amplifies noise probabilistically by scaling the learned noise model
    at each noise factor, then extrapolates to zero noise. This is the
    local simulation equivalent of IBM Runtime's amplifier="pea".

    Compared to gate-folding ZNE:
    - Same circuit structure at all noise factors (no depth increase).
    - Amplification targets ONLY the learned noise (not coherent errors).
    - More physically accurate for correlated noise environments.
    - Requires qiskit-aer for local noise model manipulation.

    Parameters
    ----------
    transpiled_circuit : QuantumCircuit
        Already-transpiled, parameter-bound ISA circuit.
    observable : SparsePauliOp
        Observable to measure (already layout-mapped via apply_layout).
    backend : BackendV2
        Noisy backend (e.g. FakeTorino) — provides the noise model.
    config : NoisyEstimatorConfig
        Shots and seed configuration.
    noise_factors : tuple[float, ...]
        Noise amplification factors (default [1, 3, 5]).
        Can be non-integer for PEA (unlike gate-folding which requires odd ints).
    extrapolator : str
        "linear" or "exponential". Default "linear".
    seed_offset : int
        Added to config.seed_simulator for independence across calls.

    Returns
    -------
    PEAResult
        Extrapolation result with R², slope, per-factor measurements,
        and learned error rate summary.
    """
    _logger.info(
        f"[pea_zne] Starting PEA-ZNE: noise_factors={noise_factors}, "
        f"extrapolator={extrapolator}, shots={config.shots}"
    )

    # Input validation
    if len(noise_factors) < 2:
        raise ValueError(
            f"At least 2 noise factors required for extrapolation, got {len(noise_factors)}"
        )
    for nf in noise_factors:
        if nf < 1:
            raise ValueError(f"All noise_factors must be ≥ 1, got {nf}")

    if transpiled_circuit.num_parameters > 0:
        raise ValueError(
            f"Circuit has {transpiled_circuit.num_parameters} unbound parameters. "
            f"Bind parameters before calling run_pea_zne()."
        )

    # Phase 1: Learn noise rates from backend
    learned_rates = _learn_noise_rates(backend)
    rate_summary = {
        "n_pairs": len(learned_rates),
        "mean_error": float(np.mean(list(learned_rates.values()))),
        "max_error": float(np.max(list(learned_rates.values()))),
        "min_error": float(np.min(list(learned_rates.values()))),
    }

    # Phase 2: Measure at each noise factor
    measured_values: list[float] = []
    for i, nf in enumerate(noise_factors):
        energy = _pea_estimate(
            transpiled_circuit,
            observable,
            backend,
            config,
            noise_factor=nf,
            learned_rates=learned_rates,
            seed_offset=seed_offset + i * 100,
        )
        measured_values.append(energy)
        _logger.info(
            f"[pea_zne]   factor={nf}: E={energy:.6f} "
            f"(circuit depth unchanged: {transpiled_circuit.depth()})"
        )

    # Phase 3: Extrapolation
    nf_arr = np.array(noise_factors, dtype=float)
    meas_arr = np.array(measured_values, dtype=float)

    if extrapolator == "exponential":
        extrap, r2, slope = _extrapolate_exponential(nf_arr, meas_arr)
        method_used = "exponential"
    else:
        extrap, r2, slope = _extrapolate_linear(nf_arr, meas_arr)
        method_used = "linear"

    _logger.info(
        f"[pea_zne] Result: E_extrapolated={extrap:.6f}, "
        f"R²={r2:.4f}, slope={slope:.6f}, method={method_used}"
    )

    return PEAResult(
        extrapolated_value=extrap,
        r_squared=r2,
        slope=slope,
        noise_factors=list(noise_factors),
        measured_values=measured_values,
        extrapolator=method_used,
        learned_error_rates=rate_summary,
    )


@dataclass
class PEADeploymentResult:
    """Full PEA-ZNE deployment result for one h-value.

    Combines layout selection with PEA noise amplification.

    Attributes
    ----------
    energy_pea_zne : PEAResult
        Primary: PEA-ZNE on the best (lowest-CES) layout.
    energy_layout_avg : float | None
        Secondary: simple average across multiple low-CES layouts.
    per_layout_pea_zne : list[PEAResult] | None
        PEA-ZNE per layout (if multi_layout=True).
    best_layout_idx : int
        Index of the layout used for primary ZNE.
    ces_values : list[float]
        CES of each layout used.
    """

    energy_pea_zne: PEAResult
    energy_layout_avg: float | None = None
    per_layout_pea_zne: list[PEAResult] | None = None
    best_layout_idx: int = 0
    ces_values: list[float] = field(default_factory=list)


def run_pea_zne_deployment(
    bound_circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    backend,
    layout_selection: LayoutSelection,
    config: NoisyEstimatorConfig,
    noise_factors: tuple[float, ...] = (1, 3, 5),
    extrapolator: str = "linear",
    multi_layout: bool = False,
) -> PEADeploymentResult:
    """Run PEA-ZNE deployment with optional multi-layout averaging.

    Strategy:
      - Primary: PEA-ZNE on the lowest-CES layout (most reliable).
      - Secondary (optional): Run PEA-ZNE on all layouts, average extrapolated
        values for additional variance reduction.

    Parameters
    ----------
    bound_circuit : QuantumCircuit
        Parameter-bound circuit (layout_selection has pre-transpiled versions).
    hamiltonian : SparsePauliOp
        Total Hamiltonian to measure.
    backend : BackendV2
        Noisy backend (provides noise model for PEA learning).
    layout_selection : LayoutSelection
        Pre-computed layout selection (use select_layouts_low_ces).
    config : NoisyEstimatorConfig
        Estimation configuration.
    noise_factors : tuple[float, ...]
        Noise amplification factors for PEA (default [1, 3, 5]).
    extrapolator : str
        "linear" or "exponential" (default "linear").
    multi_layout : bool
        If True, run PEA-ZNE on ALL layouts and average results.

    Returns
    -------
    PEADeploymentResult
        Full deployment result with primary PEA-ZNE + optional layout averaging.
    """
    _logger.info(
        f"[pea_deployment] Starting: {len(layout_selection.layouts)} layouts, "
        f"noise_factors={noise_factors}, multi_layout={multi_layout}"
    )

    ces_values = [float(c) for c in layout_selection.ces_values]

    # Use lowest-CES layout as primary
    best_idx = 0
    transpiled_best = layout_selection.transpiled_circuits[best_idx]
    h_mapped_best = hamiltonian.apply_layout(transpiled_best.layout)

    _logger.info(
        f"[pea_deployment] Primary layout idx={best_idx}, "
        f"CES={ces_values[best_idx]:.4f}, depth={transpiled_best.depth()}"
    )

    primary_result = run_pea_zne(
        transpiled_best,
        h_mapped_best,
        backend,
        config,
        noise_factors=noise_factors,
        extrapolator=extrapolator,
        seed_offset=0,
    )

    # Optional: multi-layout PEA-ZNE for variance reduction
    per_layout_results: list[PEAResult] | None = None
    layout_avg: float | None = None

    if multi_layout and len(layout_selection.transpiled_circuits) > 1:
        per_layout_results = [primary_result]
        for li in range(1, len(layout_selection.transpiled_circuits)):
            transpiled_li = layout_selection.transpiled_circuits[li]
            h_mapped_li = hamiltonian.apply_layout(transpiled_li.layout)
            _logger.info(f"[pea_deployment] Layout {li}: CES={ces_values[li]:.4f}")
            result_li = run_pea_zne(
                transpiled_li,
                h_mapped_li,
                backend,
                config,
                noise_factors=noise_factors,
                extrapolator=extrapolator,
                seed_offset=(li + 1) * 1000,
            )
            per_layout_results.append(result_li)

        # Average extrapolated values across layouts
        extrap_values = [r.extrapolated_value for r in per_layout_results]
        layout_avg = float(np.mean(extrap_values))
        _logger.info(
            f"[pea_deployment] Multi-layout average: {layout_avg:.6f} "
            f"(std={np.std(extrap_values):.6f}, n={len(extrap_values)})"
        )

    return PEADeploymentResult(
        energy_pea_zne=primary_result,
        energy_layout_avg=layout_avg,
        per_layout_pea_zne=per_layout_results,
        best_layout_idx=best_idx,
        ces_values=ces_values,
    )
