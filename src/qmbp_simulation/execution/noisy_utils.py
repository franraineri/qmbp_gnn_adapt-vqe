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
from datetime import UTC

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
        return float(1.0 / np.sqrt(self.shots))


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
    ces_spread_threshold: float = 0.3,
) -> LayoutSelection:
    """Select layouts by ACTUAL circuit CES with maximum spread.

    Transpiles the bound circuit to each candidate layout, computes the
    real circuit CES, then picks n_select layouts with maximum CES
    diversity (first, last, and evenly spaced in between).

    This is more expensive than topology CES but gives TRUE diversity.
    The transpiled circuits are cached in the result to avoid re-transpilation.

    CES Spread Guard (P0-A):
        After selection, checks whether the CES spread is sufficient for
        reliable ZNE extrapolation. The spread ratio is defined as:
            spread_ratio = (max_ces - min_ces) / mean_ces
        If spread_ratio < ces_spread_threshold (default 0.3), the returned
        LayoutSelection has `ces_spread_sufficient = False`, signaling
        the caller to fall back to gate-folding ZNE instead of CES-ZNE.
        This prevents the known failure mode on heavy_hex where all layouts
        have nearly uniform CES (~0.15), giving R²≈0.04 for CES-ZNE.

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
    ces_spread_threshold : float
        Minimum relative spread (max-min)/mean for CES-ZNE to be
        considered reliable. Default 0.3 (validated: heavy_hex uniform
        CES has spread_ratio ≈ 0.05, chain_1d has spread_ratio ≈ 1.2).

    Returns
    -------
    LayoutSelection
        Selected layouts with CES values and pre-transpiled circuits.
        The `ces_spread_sufficient` attribute indicates whether CES
        spread is adequate for inhomogeneous ZNE extrapolation.
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

    selected_ces = [circuit_ces_list[i] for i in indices]

    # CES spread guard: check if spread is sufficient for ZNE extrapolation
    ces_mean = float(np.mean(selected_ces)) if selected_ces else 0.0
    ces_range = (max(selected_ces) - min(selected_ces)) if len(selected_ces) >= 2 else 0.0
    spread_ratio = ces_range / ces_mean if ces_mean > 1e-10 else 0.0
    ces_spread_sufficient = spread_ratio >= ces_spread_threshold

    if not ces_spread_sufficient:
        _logger.warning(
            f"[select_layouts_by_circuit_ces] CES spread insufficient for ZNE: "
            f"spread_ratio={spread_ratio:.3f} < threshold={ces_spread_threshold}. "
            f"CES values: {[f'{c:.4f}' for c in selected_ces]}. "
            f"Recommend falling back to gate-folding ZNE."
        )

    result = LayoutSelection(
        layouts=[candidate_layouts[i] for i in indices],
        ces_values=selected_ces,
        transpiled_circuits=[transpiled_list[i] for i in indices],
    )
    # Attach spread metadata (used by run_zne_deployment for fallback logic)
    result.ces_spread_sufficient = ces_spread_sufficient  # type: ignore[attr-defined]
    result.ces_spread_ratio = spread_ratio  # type: ignore[attr-defined]

    return result


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
        sorted_idx = [i for i in sorted_idx if circuit_ces_list[i] <= max_ces]  # type: ignore[assignment]

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
    *,
    return_job: bool = False,
) -> float | tuple[float, Any]:
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
    return_job : bool
        If True, return (energy, job) tuple for QPU metrics extraction.
        Default False for backward compatibility.

    Returns
    -------
    float
        Expectation value ⟨O⟩ (when return_job=False).
    tuple[float, Any]
        (energy, job) when return_job=True. The job object exposes
        .metrics() for QPU time extraction on real hardware.
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
    energy = float(job.result()[0].data.evs)
    if return_job:
        return energy, job
    return energy


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


def linear_zne(
    ces_values: np.ndarray,
    measured_values: np.ndarray,
    sigmas: np.ndarray | None = None,
) -> ZNEResult:
    """Linear ZNE extrapolation to CES=0.

    Fits a line E(CES) = a*CES + b and extrapolates to CES=0.
    Supports Weighted Least Squares (WLS) when per-point uncertainties
    (sigmas) are provided — the statistically optimal estimator for
    heteroscedastic shot-noise data.

    Parameters
    ----------
    ces_values : np.ndarray
        Circuit Error Scores for each layout.
    measured_values : np.ndarray
        Measured expectation values at each CES point.
    sigmas : np.ndarray or None
        Per-point standard deviations for WLS weighting. Computed as
        1/√shots per layout when shots differ, or from bootstrap.
        If None or all-equal, OLS is used (equivalent result).

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

    # Use WLS when valid per-point sigmas are provided
    use_wls = (
        sigmas is not None
        and len(sigmas) == len(vals_arr)
        and np.all(np.isfinite(sigmas))
        and np.all(sigmas > 0)
        and np.std(sigmas) > 1e-15
    )

    if use_wls:
        weights = 1.0 / (sigmas**2)
        coeffs = np.polyfit(ces_arr, vals_arr, 1, w=np.sqrt(weights))
    else:
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
    fallback_to_gf: bool = True,
    gf_noise_factors: tuple[int, ...] = (1, 3, 5),
) -> ZNEDeploymentResult:
    """Run full ZNE deployment: measure energy (and optionally per-site X)
    across multiple layouts, then extrapolate to CES=0.

    CES Spread Guard (P0-A):
        When `fallback_to_gf=True` (default), checks whether the layout
        selection has sufficient CES spread for reliable inhomogeneous ZNE.
        If spread is insufficient (spread_ratio < 0.3, typical on heavy_hex),
        automatically falls back to gate-folding ZNE on the lowest-CES
        layout. This prevents the known failure mode where uniform CES
        produces R²≈0.04.

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
    fallback_to_gf : bool
        If True (default), fall back to gate-folding ZNE when CES spread
        is insufficient for reliable inhomogeneous ZNE extrapolation.
    gf_noise_factors : tuple[int, ...]
        Noise factors for the gate-folding fallback (default [1,3,5]).

    Returns
    -------
    ZNEDeploymentResult
        Complete ZNE results including extrapolation and raw data.
        When fallback is triggered, the result uses GF-ZNE on the best
        layout. The `fallback_triggered` attribute indicates this case.
    """
    # ── P0-A: CES spread soft guard — fallback to GF-ZNE if spread insufficient ──
    ces_spread_ok = getattr(layout_selection, "ces_spread_sufficient", True)
    if fallback_to_gf and not ces_spread_ok:
        spread_ratio = getattr(layout_selection, "ces_spread_ratio", 0.0)
        _logger.warning(
            f"[run_zne_deployment] CES spread insufficient "
            f"(ratio={spread_ratio:.3f}). Falling back to gate-folding ZNE "
            f"with noise_factors={gf_noise_factors} on lowest-CES layout."
        )
        # Use the lowest-CES layout (index 0, sorted by select_layouts_*)
        best_transpiled = layout_selection.transpiled_circuits[0]
        h_mapped_best = hamiltonian.apply_layout(best_transpiled.layout)

        gf_result = run_gate_folding_zne(
            best_transpiled,
            h_mapped_best,
            backend,
            config,
            noise_factors=gf_noise_factors,
            seed_offset=0,
        )

        # Convert GateFoldingZNEResult to ZNEResult for interface compatibility
        energy_zne = ZNEResult(
            extrapolated_value=gf_result.extrapolated_value,
            r_squared=gf_result.r_squared,
            slope=gf_result.slope,
            ces_values=np.array(layout_selection.ces_values),
            measured_values=np.array([gf_result.measured_values[0]]),
        )

        result = ZNEDeploymentResult(
            energy_zne=energy_zne,
            per_site_zne=None,
            per_layout_data=[
                {
                    "ces": layout_selection.ces_values[0],
                    "n_2q": sum(
                        1 for inst in best_transpiled.data if inst.operation.num_qubits == 2
                    ),
                    "energy": gf_result.measured_values[0],
                    "per_site_x": None,
                }
            ],
        )
        result.fallback_triggered = True  # type: ignore[attr-defined]
        result.fallback_method = "gate_folding"  # type: ignore[attr-defined]
        result.gf_result = gf_result  # type: ignore[attr-defined]
        return result

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

    # ZNE extrapolation — energy (with WLS using shot-noise sigma per layout)
    ces_arr = np.array([d["ces"] for d in per_layout_data])
    e_arr = np.array([d["energy"] for d in per_layout_data])
    # σ per layout: precision is uniform (same shots), so OLS is appropriate
    # for CES-ZNE unless layouts use different shot counts.
    energy_zne = linear_zne(ces_arr, e_arr)

    # ZNE extrapolation — per-site (if requested)
    per_site_zne_results: list[ZNEResult] | None = None
    if per_site:
        per_site_zne_results = []
        for site_i in range(n_qubits):
            site_vals = np.array([d["per_site_x"][site_i] for d in per_layout_data])  # type: ignore[index]
            per_site_zne_results.append(linear_zne(ces_arr, site_vals))

    result = ZNEDeploymentResult(
        energy_zne=energy_zne,
        per_site_zne=per_site_zne_results,
        per_layout_data=per_layout_data,
    )
    result.fallback_triggered = False  # type: ignore[attr-defined]
    return result


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
    noise_factors: np.ndarray,
    measured: np.ndarray,
    sigmas: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Linear extrapolation E(nf) = a*nf + b → E(0) = b.

    Supports Weighted Least Squares (WLS) when per-point uncertainties
    are provided. WLS is the statistically optimal estimator for
    heteroscedastic data (different noise at each amplification factor).

    Parameters
    ----------
    noise_factors : np.ndarray
        Noise amplification factors (x-axis).
    measured : np.ndarray
        Measured energies at each noise factor (y-axis).
    sigmas : np.ndarray or None
        Per-point standard deviations (1/σ² weighting). If None or
        all-equal, falls back to OLS (equivalent result).

    Returns
    -------
    tuple[float, float, float]
        (extrapolated_value, r_squared, slope).
    """
    # Use WLS when valid per-point sigmas are provided
    use_wls = (
        sigmas is not None
        and len(sigmas) == len(measured)
        and np.all(np.isfinite(sigmas))
        and np.all(sigmas > 0)
        and np.std(sigmas) > 1e-15  # Skip WLS if all sigmas are identical
    )

    if use_wls:
        # WLS: weight_i = 1/σ_i² (inverse-variance weighting)
        weights = 1.0 / (sigmas**2)
        coeffs = np.polyfit(noise_factors, measured, 1, w=np.sqrt(weights))
    else:
        coeffs = np.polyfit(noise_factors, measured, 1)

    extrap = float(np.polyval(coeffs, 0.0))
    y_pred = np.polyval(coeffs, noise_factors)
    ss_res = np.sum((measured - y_pred) ** 2)
    ss_tot = np.sum((measured - np.mean(measured)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0
    return extrap, r2, float(coeffs[0])


def _extrapolate_exponential(
    noise_factors: np.ndarray,
    measured: np.ndarray,
    sigmas: np.ndarray | None = None,
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
        return _extrapolate_linear(noise_factors, measured, sigmas=sigmas)


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

    # ── Depth guard: warn when shallow circuits exit the ZNE linear regime ──
    # For gate-folding ZNE, the effective 2Q gate count at the deepest fold is
    # n_2q * max_factor. The linear regime limit is ~18 CX (project constraint).
    # Beyond this, the linear E(noise_factor) assumption breaks down and ZNE
    # can overcorrect, yielding negative improvement. See project-status.md §ZNE.
    n_2q_orig = sum(1 for inst in transpiled_circuit.data if inst.operation.num_qubits == 2)
    max_factor = max(noise_factors)
    folded_2q_at_max = n_2q_orig * max_factor
    ZNE_LINEAR_REGIME_LIMIT = 18  # validated CX budget for linear ZNE (project-status.md)

    if n_2q_orig < 6:
        # Very shallow circuit: folding makes the circuit deeper than the original
        # by a large multiple. Gate-folding ZNE is unreliable in this regime.
        # Cap noise factors to keep folded depth within linear regime.
        safe_max_factor = max(1, ZNE_LINEAR_REGIME_LIMIT // max(n_2q_orig, 1))
        # Ensure safe_max_factor is odd
        if safe_max_factor % 2 == 0:
            safe_max_factor -= 1
        safe_max_factor = max(1, safe_max_factor)
        if safe_max_factor < max_factor:
            _logger.warning(
                f"[gate_folding_zne] Shallow circuit: {n_2q_orig} 2Q gates × "
                f"max_factor={max_factor} = {folded_2q_at_max} effective 2Q gates. "
                f"Exceeds linear regime ({ZNE_LINEAR_REGIME_LIMIT}). "
                f"Auto-capping noise_factors to max={safe_max_factor} to avoid "
                f"negative ZNE improvement. "
                f"Consider using PEA-ZNE instead (not depth-limited)."
            )
            # Filter noise_factors to only include odd factors ≤ safe_max_factor
            noise_factors = tuple(nf for nf in noise_factors if nf <= safe_max_factor)
            if len(noise_factors) < 2:
                # Fallback: use (1, safe_max_factor) — minimum viable ZNE
                noise_factors = (1, safe_max_factor) if safe_max_factor >= 3 else (1, 3)
            _logger.info(f"[gate_folding_zne] Using capped noise_factors={noise_factors}")
    elif folded_2q_at_max > ZNE_LINEAR_REGIME_LIMIT:
        _logger.warning(
            f"[gate_folding_zne] Circuit has {n_2q_orig} 2Q gates × "
            f"max_factor={max_factor} = {folded_2q_at_max} effective 2Q gates. "
            f"Exceeds linear regime ({ZNE_LINEAR_REGIME_LIMIT}). ZNE linearity "
            f"may be degraded. Consider reducing max noise_factor to "
            f"{max(1, ZNE_LINEAR_REGIME_LIMIT // n_2q_orig)!s}."
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

    # Compute per-noise-factor sigmas for WLS (shot noise scales with √nf
    # because folded circuits have nf× more 2Q gates → more depolarization →
    # larger variance). σ_i = precision × √(nf_i) is a first-order estimate.
    # This gives higher weight to the less-noisy (low nf) data points.
    sigmas = np.array([config.precision * np.sqrt(float(nf)) for nf in noise_factors])

    if extrapolator == "exponential":
        extrap, r2, slope = _extrapolate_exponential(nf_arr, meas_arr, sigmas=sigmas)
        method_used = "exponential"
    else:
        extrap, r2, slope = _extrapolate_linear(nf_arr, meas_arr, sigmas=sigmas)
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


def _get_circuit_qubits(transpiled_circuit: QuantumCircuit) -> set[int]:
    """Extract the set of physical qubit indices used by a transpiled circuit.

    Parameters
    ----------
    transpiled_circuit : QuantumCircuit
        An already-transpiled ISA circuit.

    Returns
    -------
    set[int]
        Physical qubit indices that participate in at least one instruction.
    """
    used = set()
    for inst in transpiled_circuit.data:
        for qubit in inst.qubits:
            used.add(transpiled_circuit.qubits.index(qubit))
    return used


def _filter_rates_to_circuit(
    learned_rates: dict[tuple[int, ...], float],
    circuit_qubits: set[int],
) -> dict[tuple[int, ...], float]:
    """Filter noise rates to only pairs relevant to the circuit.

    A pair is relevant if at least one qubit in the pair is used by the
    circuit. Noise on pairs where NEITHER qubit participates in any gate
    cannot affect the circuit's measurement outcomes.

    Parameters
    ----------
    learned_rates : dict[tuple[int, ...], float]
        All gate error pairs from _learn_noise_rates().
    circuit_qubits : set[int]
        Physical qubits used by the transpiled circuit.

    Returns
    -------
    dict[tuple[int, ...], float]
        Filtered subset of learned_rates.
    """
    return {
        pair: rate
        for pair, rate in learned_rates.items()
        if pair[0] in circuit_qubits or pair[1] in circuit_qubits
    }


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

    Performance: When called with pre-filtered rates (via
    _filter_rates_to_circuit), this function is 5-15× faster because
    it only constructs depolarizing_error objects for circuit-relevant
    qubit pairs. The results are bit-exact: noise on unused qubits cannot
    affect circuit outcomes.

    Note
    ----
    **Approximation**: This is a simplified isotropic depolarizing model.
    Real PEA (IBM Runtime) learns a full Pauli-Lindblad noise model with
    up to 15 generators per 2-qubit pair (XX, XY, XZ, YX, ...) and
    amplifies each channel independently. Our local simulation uses
    isotropic depolarizing as a practical simplification. This is accurate
    for FakeTorino (whose calibration data is already mostly depolarizing)
    but may differ from real hardware results by ~5-10% in extrapolated
    energy values due to the anisotropic structure of physical noise.

    Parameters
    ----------
    backend : BackendV2
        Original noisy backend.
    transpiled_circuit : QuantumCircuit
        The transpiled circuit (to identify which qubits are used).
    noise_factor : float
        Amplification factor (1 = original, 3 = 3× noise, etc.).
    learned_rates : dict[tuple[int, ...], float]
        Per-gate error rates from _learn_noise_rates(). May be pre-filtered
        to circuit-relevant pairs for performance.

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
    prebuilt_noise_model=None,
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
        May be pre-filtered to circuit-relevant pairs for performance.
    seed_offset : int
        Added to config.seed_simulator for independence.
    prebuilt_noise_model : NoiseModel | None
        If provided, skip _build_amplified_noise_model() and use this model
        directly. Used by run_pea_zne() to avoid redundant model construction
        when models are pre-built for all noise factors.

    Returns
    -------
    float
        Expectation value measured under amplified noise.
    """
    from qiskit.primitives import BackendEstimatorV2
    from qiskit_aer import AerSimulator

    if prebuilt_noise_model is not None:
        amplified_model = prebuilt_noise_model
    else:
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


# ─── Parallel Noise Factor Execution ─────────────────────────────────────
# For N≥20, each noise factor simulation takes 0.5-2s (dominated by Aer).
# Since each factor uses an independent AerSimulator + independent seed,
# they can run in parallel via ThreadPoolExecutor (Aer releases the GIL
# during its C++ simulation kernel).
#
# ThreadPoolExecutor (not ProcessPoolExecutor) because:
# - No pickle/serialization of QuantumCircuit or NoiseModel needed
# - Lower spawn overhead (~1ms vs ~100ms)
# - Shared memory (no data copying)
# - Aer's C++ backend genuinely releases the GIL → true parallelism
# ──────────────────────────────────────────────────────────────────────────

# Auto-enable parallel execution when circuit uses ≥ this many qubits.
# Below this threshold, the sequential overhead is negligible (~60ms total).
_PEA_PARALLEL_QUBIT_THRESHOLD = 14


def _measure_noise_factors(
    noise_factors: tuple[float, ...],
    noise_models: dict,
    transpiled_circuit: QuantumCircuit,
    observable: SparsePauliOp,
    backend,
    config: NoisyEstimatorConfig,
    relevant_rates: dict[tuple[int, ...], float],
    seed_offset: int,
    parallel: bool = False,
) -> list[float]:
    """Measure energy at each noise factor, optionally in parallel.

    Each noise factor simulation is fully independent (own AerSimulator,
    own seed, own noise model). When `parallel=True`, uses ThreadPoolExecutor
    to overlap the Aer C++ simulation across factors.

    Parameters
    ----------
    noise_factors : tuple[float, ...]
        Noise amplification factors.
    noise_models : dict
        Pre-built noise models keyed by factor.
    transpiled_circuit : QuantumCircuit
        ISA circuit (parameter-bound).
    observable : SparsePauliOp
        Layout-mapped observable.
    backend : BackendV2
        Original noisy backend (for from_backend).
    config : NoisyEstimatorConfig
        Shots and seed configuration.
    relevant_rates : dict
        Filtered noise rates (for fallback if model missing).
    seed_offset : int
        Base seed offset.
    parallel : bool
        If True, execute noise factors concurrently via threads.
        Auto-enabled for circuits with ≥ _PEA_PARALLEL_QUBIT_THRESHOLD qubits.

    Returns
    -------
    list[float]
        Measured energies in the same order as noise_factors.
    """

    def _run_single(args: tuple) -> float:
        i, nf = args
        return _pea_estimate(
            transpiled_circuit,
            observable,
            backend,
            config,
            noise_factor=nf,
            learned_rates=relevant_rates,
            seed_offset=seed_offset + i * 100,
            prebuilt_noise_model=noise_models[nf],
        )

    if parallel and len(noise_factors) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n_workers = min(len(noise_factors), 4)
        _logger.info(
            f"[pea_zne] Parallel execution: {len(noise_factors)} factors, {n_workers} threads"
        )
        # Use submit + index tracking to preserve order (map already preserves
        # order, but explicit indexing is clearer for debugging)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(_run_single, (i, nf)): i for i, nf in enumerate(noise_factors)
            }
            results = [0.0] * len(noise_factors)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    _logger.error(f"[pea_zne] Thread for factor={noise_factors[idx]} failed: {exc}")
                    # Propagate NaN so extrapolation degrades gracefully
                    results[idx] = float("nan")
        return results

    # Sequential fallback (small circuits or single factor)
    return [_run_single((i, nf)) for i, nf in enumerate(noise_factors)]


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

    # Optimization: filter rates to circuit-relevant qubits only.
    # Noise on qubit pairs outside the circuit's light cone cannot affect
    # measurement outcomes. This reduces noise model construction time by
    # total_pairs/relevant_pairs (typically 10-15× for N=6 on 133-qubit Torino).
    circuit_qubits = _get_circuit_qubits(transpiled_circuit)
    relevant_rates = _filter_rates_to_circuit(learned_rates, circuit_qubits)

    _logger.info(
        f"[pea_zne] Noise filtering: {len(relevant_rates)}/{len(learned_rates)} "
        f"pairs relevant to {len(circuit_qubits)} circuit qubits"
    )

    rate_summary = {
        "n_pairs": len(learned_rates),
        "n_pairs_relevant": len(relevant_rates),
        "mean_error": float(np.mean(list(learned_rates.values()))),
        "max_error": float(np.max(list(learned_rates.values()))),
        "min_error": float(np.min(list(learned_rates.values()))),
    }

    # Optimization: pre-build all noise models before the measurement loop.
    # This avoids repeated Python loop overhead and allows the interpreter to
    # optimize object allocation in a single batch.
    noise_models = {}
    for nf in noise_factors:
        noise_models[nf] = _build_amplified_noise_model(
            backend, transpiled_circuit, nf, relevant_rates
        )

    # Phase 2: Measure at each noise factor (parallel if beneficial)
    measured_values = _measure_noise_factors(
        noise_factors=noise_factors,
        noise_models=noise_models,
        transpiled_circuit=transpiled_circuit,
        observable=observable,
        backend=backend,
        config=config,
        relevant_rates=relevant_rates,
        seed_offset=seed_offset,
        parallel=len(circuit_qubits) >= _PEA_PARALLEL_QUBIT_THRESHOLD,
    )

    for i, nf in enumerate(noise_factors):
        _logger.info(
            f"[pea_zne]   factor={nf}: E={measured_values[i]:.6f} "
            f"(circuit depth unchanged: {transpiled_circuit.depth()})"
        )

    # Phase 3: Extrapolation
    nf_arr = np.array(noise_factors, dtype=float)
    meas_arr = np.array(measured_values, dtype=float)

    # WLS sigmas for PEA: noise amplification increases variance proportionally
    # to the noise factor (σ_i ∝ √nf_i for depolarizing noise).
    sigmas = np.array([config.precision * np.sqrt(float(nf)) for nf in noise_factors])

    if extrapolator == "exponential":
        extrap, r2, slope = _extrapolate_exponential(nf_arr, meas_arr, sigmas=sigmas)
        method_used = "exponential"
    else:
        extrap, r2, slope = _extrapolate_linear(nf_arr, meas_arr, sigmas=sigmas)
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


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive ZNE — automatic GF → PEA fallback by R² threshold
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AdaptiveZNEResult:
    """Result from adaptive tiered ZNE strategy (QESEM-inspired).

    Default strategy ("pea_primary"): PEA first, GF only if PEA unavailable.
    Legacy strategy ("gf_primary"): GF first, PEA if R² < threshold.

    Rationale for pea_primary (Kim et al. Nature 618, 2023; QESEM arXiv:2508.10997):
    Characterization-based methods (PEA) model the actual noise channel via
    Sparse Pauli-Lindblad fitting, yielding +94.4% mean gain. Gate-folding
    assumes uniform noise scaling and achieves only +20.6% gain. Critically,
    GF can produce R²>0.99 with ΔE/gap=89.8% (HW_REHEARSAL_V2 section 5),
    proving that R² alone is insufficient for accuracy assessment.

    Attributes
    ----------
    extrapolated_value : float
        Best ZNE-extrapolated energy (from whichever amplifier was used).
    r_squared : float
        R² of the selected extrapolation.
    amplifier_used : str
        "gate_folding" or "pea" — which amplifier produced the final result.
    gf_result : GateFoldingZNEResult | None
        Gate-folding result (present if GF was attempted).
    pea_result : PEAResult | None
        PEA result (present if PEA was attempted).
    fallback_triggered : bool
        True if the primary method failed and fallback was used.
    """

    extrapolated_value: float
    r_squared: float
    amplifier_used: str
    gf_result: GateFoldingZNEResult | None
    pea_result: PEAResult | None
    fallback_triggered: bool


def run_adaptive_zne(
    transpiled_circuit: QuantumCircuit,
    observable: SparsePauliOp,
    backend,
    config: NoisyEstimatorConfig,
    noise_factors: tuple[float, ...] = (1, 3, 5),
    r2_threshold: float = 0.90,
    extrapolator: str = "linear",
    seed_offset: int = 0,
    strategy: str = "pea_primary",
) -> AdaptiveZNEResult:
    """Run tiered ZNE with configurable primary/fallback strategy.

    Strategies (inspired by QESEM — arXiv:2508.10997):
      - "pea_primary" (default, RECOMMENDED): PEA first, GF only if PEA
        unavailable (missing qiskit-aer). PEA uses learned noise model
        (Sparse Pauli-Lindblad) for physically accurate amplification.
        Validated: +94.4% mean gain vs +20.6% for GF (ZNE_CROSS_TOPO).
      - "gf_primary" (legacy): GF first, PEA fallback if R² < threshold.
        WARNING: GF can produce high R² (>0.99) with terrible accuracy
        (ΔE/gap=89.8% observed in HW_REHEARSAL_V2 section 5). High R²
        only means the extrapolation is *consistent*, not *accurate*.

    The key insight from Kim et al. (Nature 618, 2023) and follow-up work
    (QESEM 2025, Stabilized Noise 2025): characterization-based methods
    (PEA, PEC, QESEM) always outperform heuristic methods (gate-folding)
    because they model the *actual* noise channel rather than assuming
    noise scales uniformly with gate repetition.

    Parameters
    ----------
    transpiled_circuit : QuantumCircuit
        Already-transpiled, parameter-bound ISA circuit.
    observable : SparsePauliOp
        Observable to measure (already layout-mapped).
    backend : BackendV2
        Noisy backend (e.g. FakeTorino).
    config : NoisyEstimatorConfig
        Shots and seed configuration.
    noise_factors : tuple[float, ...]
        Noise amplification factors (default [1, 3, 5]).
    r2_threshold : float
        Minimum R² for the primary method to be accepted (default 0.90).
    extrapolator : str
        "linear" or "exponential" (default "linear").
    seed_offset : int
        Added to config.seed_simulator for independence.
    strategy : str
        "pea_primary" (recommended) or "gf_primary" (legacy).

    Returns
    -------
    AdaptiveZNEResult
        Result with the best available extrapolation and provenance.
    """
    if strategy == "pea_primary":
        return _adaptive_pea_primary(
            transpiled_circuit,
            observable,
            backend,
            config,
            noise_factors,
            r2_threshold,
            extrapolator,
            seed_offset,
        )
    elif strategy == "gf_primary":
        return _adaptive_gf_primary(
            transpiled_circuit,
            observable,
            backend,
            config,
            noise_factors,
            r2_threshold,
            extrapolator,
            seed_offset,
        )
    else:
        raise ValueError(
            f"Unknown adaptive ZNE strategy: {strategy!r}. "
            f"Use 'pea_primary' (recommended) or 'gf_primary'."
        )


def _adaptive_pea_primary(
    transpiled_circuit: QuantumCircuit,
    observable: SparsePauliOp,
    backend,
    config: NoisyEstimatorConfig,
    noise_factors: tuple[float, ...],
    r2_threshold: float,
    extrapolator: str,
    seed_offset: int,
) -> AdaptiveZNEResult:
    """PEA-primary strategy: try PEA first, GF only if PEA unavailable."""
    # Step 1: Attempt PEA (characterization-based, highest accuracy)
    pea_result: PEAResult | None = None
    try:
        pea_result = run_pea_zne(
            transpiled_circuit,
            observable,
            backend,
            config,
            noise_factors=noise_factors,
            extrapolator=extrapolator,
            seed_offset=seed_offset,
        )
    except Exception as e:
        _logger.warning(
            f"[adaptive_zne] PEA failed ({type(e).__name__}: {e}), falling back to gate-folding"
        )

    if pea_result is not None and pea_result.r_squared >= r2_threshold:
        _logger.info(
            f"[adaptive_zne] PEA R²={pea_result.r_squared:.3f} >= {r2_threshold}, "
            f"accepting PEA result (strategy=pea_primary)"
        )
        return AdaptiveZNEResult(
            extrapolated_value=pea_result.extrapolated_value,
            r_squared=pea_result.r_squared,
            amplifier_used="pea",
            gf_result=None,
            pea_result=pea_result,
            fallback_triggered=False,
        )

    # Step 2: Fallback to gate-folding (if PEA failed or R² insufficient)
    _logger.warning(
        f"[adaptive_zne] PEA {'unavailable' if pea_result is None else f'R²={pea_result.r_squared:.3f} < {r2_threshold}'}, "
        f"falling back to gate-folding"
    )
    gf_noise_factors = tuple(int(nf) for nf in noise_factors)
    gf_result = run_gate_folding_zne(
        transpiled_circuit,
        observable,
        backend,
        config,
        noise_factors=gf_noise_factors,
        extrapolator=extrapolator,
        seed_offset=seed_offset + 5000,
    )

    return AdaptiveZNEResult(
        extrapolated_value=gf_result.extrapolated_value,
        r_squared=gf_result.r_squared,
        amplifier_used="gate_folding",
        gf_result=gf_result,
        pea_result=pea_result,
        fallback_triggered=True,
    )


def _adaptive_gf_primary(
    transpiled_circuit: QuantumCircuit,
    observable: SparsePauliOp,
    backend,
    config: NoisyEstimatorConfig,
    noise_factors: tuple[float, ...],
    r2_threshold: float,
    extrapolator: str,
    seed_offset: int,
) -> AdaptiveZNEResult:
    """Legacy GF-primary strategy: GF first, PEA if R² < threshold.

    WARNING: High GF R² does NOT guarantee accuracy. Gate-folding assumes
    noise scales uniformly with gate repetition, which is incorrect for
    correlated noise channels. Use pea_primary instead.
    """
    # Step 1: Try gate-folding (cheap)
    gf_noise_factors = tuple(int(nf) for nf in noise_factors)
    gf_result = run_gate_folding_zne(
        transpiled_circuit,
        observable,
        backend,
        config,
        noise_factors=gf_noise_factors,
        extrapolator=extrapolator,
        seed_offset=seed_offset,
    )

    if gf_result.r_squared >= r2_threshold:
        _logger.info(
            f"[adaptive_zne] GF R²={gf_result.r_squared:.3f} >= {r2_threshold}, "
            f"accepting gate-folding result (strategy=gf_primary)"
        )
        return AdaptiveZNEResult(
            extrapolated_value=gf_result.extrapolated_value,
            r_squared=gf_result.r_squared,
            amplifier_used="gate_folding",
            gf_result=gf_result,
            pea_result=None,
            fallback_triggered=False,
        )

    # Step 2: Fallback to PEA (more expensive but higher quality)
    _logger.warning(
        f"[adaptive_zne] GF R²={gf_result.r_squared:.3f} < {r2_threshold}, "
        f"falling back to PEA (strategy=gf_primary)"
    )
    pea_result = run_pea_zne(
        transpiled_circuit,
        observable,
        backend,
        config,
        noise_factors=noise_factors,
        extrapolator=extrapolator,
        seed_offset=seed_offset + 5000,
    )

    return AdaptiveZNEResult(
        extrapolated_value=pea_result.extrapolated_value,
        r_squared=pea_result.r_squared,
        amplifier_used="pea",
        gf_result=gf_result,
        pea_result=pea_result,
        fallback_triggered=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Dual-Branch Affine Correction — physics-constrained post-processing
# ═══════════════════════════════════════════════════════════════════════════
# Reference: Wang et al. (2026), "Scalable Quantum Error Mitigation with
# Physically Informed Graph Neural Networks", arXiv:2604.16815.
# The GEM framework applies a dual-branch affine correction to maintain
# consistency with physical constraints. We adapt this idea: the mitigated
# energy must lie in [E_ground, E_max] where E_max is the maximum eigenvalue.
# For TFIM, E_max = 0 (trivial upper bound) or the max-energy state.
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AffineCorrectedResult:
    """Result of dual-branch affine correction on a mitigated energy.

    Attributes
    ----------
    corrected_energy : float
        Energy after affine correction (clipped to physical bounds).
    original_energy : float
        Input energy before correction.
    correction_applied : bool
        True if the energy was actually modified.
    correction_magnitude : float
        |corrected - original| (0 if no correction).
    lower_bound : float
        Physical lower bound used (E_ground or estimate).
    upper_bound : float
        Physical upper bound used (E_max or 0).
    """

    corrected_energy: float
    original_energy: float
    correction_applied: bool
    correction_magnitude: float
    lower_bound: float
    upper_bound: float


def affine_correct_energy(
    mitigated_energy: float,
    e_ground: float,
    e_upper: float | None = None,
    n_qubits: int | None = None,
    h_value: float | None = None,
    n_bonds: int | None = None,
) -> AffineCorrectedResult:
    """Apply dual-branch affine correction to a ZNE-mitigated energy.

    Physics constraint: for a finite spin system, the ground state energy
    has a rigorous lower bound (E_ground from exact diag or variational
    principle) and an upper bound (E_max eigenvalue or trivial bound).

    For TFIM H = -J·ΣZZ - h·ΣX with N spins:
      - E_lower = E_ground (from Phase 1 exact diag)
      - E_upper = +|J|·N_bonds + |h|·N (all spins anti-aligned)
      - Trivial: E_upper ≈ 0 works for h >> J (paramagnetic phase)

    The correction clips the mitigated energy to [E_lower, E_upper] and
    optionally applies soft affine rescaling when the result is close to
    but outside bounds (reduces discontinuity at boundary).

    Inspired by: Wang et al. (arXiv:2604.16815) GEM dual-branch affine
    correction which maintains consistency with physical constraints.

    Parameters
    ----------
    mitigated_energy : float
        ZNE-mitigated energy estimate.
    e_ground : float
        Ground state energy (rigorous lower bound). From exact diag.
    e_upper : float | None
        Upper bound on energy. If None, estimated from system params.
    n_qubits : int | None
        Number of qubits (used to estimate e_upper if not provided).
    h_value : float | None
        Transverse field value (used to estimate e_upper if not provided).
    n_bonds : int | None
        Number of ZZ bonds in the lattice. If None, defaults to n_qubits-1
        (1D chain). For heavy_hex N=10: n_bonds=11. Pass explicitly for
        non-1D topologies to get correct upper bounds.

    Returns
    -------
    AffineCorrectedResult
        Corrected energy with metadata about the correction.
    """
    # Estimate upper bound if not provided
    if e_upper is None:
        if n_qubits is not None and h_value is not None:
            # TFIM trivial upper bound: all eigenvalues ≤ |J|*N_bonds + |h|*N
            # For 1D chain: n_bonds = N-1. For heavy_hex: pass n_bonds explicitly.
            effective_n_bonds = n_bonds if n_bonds is not None else (n_qubits - 1)
            e_upper = float(abs(1.0) * effective_n_bonds + abs(h_value) * n_qubits)
        else:
            # Fallback: use 0 as upper bound (valid for h > J paramagnetic)
            e_upper = 0.0

    # Ensure e_ground < e_upper
    if e_ground >= e_upper:
        _logger.warning(
            f"[affine_correct] e_ground={e_ground:.4f} >= e_upper={e_upper:.4f}, "
            f"swapping or skipping correction"
        )
        return AffineCorrectedResult(
            corrected_energy=mitigated_energy,
            original_energy=mitigated_energy,
            correction_applied=False,
            correction_magnitude=0.0,
            lower_bound=e_ground,
            upper_bound=e_upper,
        )

    # Apply clipping with soft margin (5% of range)
    energy_range = e_upper - e_ground
    margin = 0.05 * energy_range

    corrected = mitigated_energy
    if mitigated_energy < e_ground - margin:
        # Far below ground state: hard clip to lower bound
        corrected = e_ground
    elif mitigated_energy > e_upper + margin:
        # Far above upper bound: hard clip to upper bound
        corrected = e_upper
    elif mitigated_energy < e_ground:
        # Slightly below ground state (within margin): clip to lower bound.
        # Physics: variational principle forbids E < E_ground. Any sub-ground
        # estimate is ZNE overshoot — clip to the rigorous bound.
        # BUG FIX (2026-06-22): Previous "soft interpolation" formula
        # (corrected = e_ground - margin*(1-alpha)*0.5) moved energy FURTHER
        # below e_ground instead of toward it, amplifying errors up to 614×.
        # Affected 3/18 hardware runs (verdict flipped PASS→FAIL).
        corrected = e_ground
    elif mitigated_energy > e_upper:
        # Slightly above upper bound (within margin): clip to upper bound.
        # Same fix as lower bound — simple clip is correct and safe.
        corrected = e_upper

    correction_applied = abs(corrected - mitigated_energy) > 1e-10
    correction_mag = abs(corrected - mitigated_energy)

    if correction_applied:
        _logger.info(
            f"[affine_correct] Corrected {mitigated_energy:.6f} → {corrected:.6f} "
            f"(bounds=[{e_ground:.4f}, {e_upper:.4f}], Δ={correction_mag:.6f})"
        )

    return AffineCorrectedResult(
        corrected_energy=corrected,
        original_energy=mitigated_energy,
        correction_applied=correction_applied,
        correction_magnitude=correction_mag,
        lower_bound=e_ground,
        upper_bound=e_upper,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Block-Level ZNE — fold only one HVA layer instead of the full circuit
# ═══════════════════════════════════════════════════════════════════════════
# Reference: "Enhanced Extrapolation-Based Quantum Error Mitigation Using
# Repetitive Structure in Quantum Algorithms", arXiv:2507.23314 (Jul 2025).
# Key insight: for algorithms with repeating blocks (like HVA layers),
# characterize the error of ONE block with shallow circuits, then extrapolate.
# This yields better ZNE for p≥2 because:
#   1. The folded block stays shallow → less decoherence during measurement
#   2. Noise amplification is more uniform (same block structure)
#   3. Extrapolation is more linear (single-block error model)
# ═══════════════════════════════════════════════════════════════════════════


def fold_single_layer(
    circuit: QuantumCircuit,
    layer_index: int,
    noise_factor: int = 3,
    n_layers: int = 1,
) -> QuantumCircuit:
    """Apply gate folding only to 2-qubit gates in a specific HVA layer.

    For an HVA circuit with p layers, this folds gates only in layer
    `layer_index`, leaving other layers untouched. This is more precise
    than full-circuit folding because it amplifies noise in a controlled,
    structured way that matches the algorithm's repetitive structure.

    The layer boundaries are identified by counting 2-qubit gate groups:
    each HVA layer contains a fixed number of 2Q gates determined by
    the lattice connectivity.

    Parameters
    ----------
    circuit : QuantumCircuit
        Transpiled (ISA) circuit. Must be parameter-free.
    layer_index : int
        Which HVA layer to fold (0-indexed). Must be < n_layers.
    noise_factor : int
        Odd integer ≥ 1. Folding factor for gates in the target layer.
    n_layers : int
        Total number of HVA layers in the circuit (p).

    Returns
    -------
    QuantumCircuit
        Circuit with the specified layer's 2Q gates folded.

    Raises
    ------
    ValueError
        If noise_factor is not odd, or layer_index >= n_layers.
    """
    if noise_factor < 1 or noise_factor % 2 == 0:
        raise ValueError(f"noise_factor must be odd positive integer, got {noise_factor}")
    if layer_index >= n_layers:
        raise ValueError(f"layer_index={layer_index} >= n_layers={n_layers}")
    if noise_factor == 1:
        return circuit.copy()

    n_folds = (noise_factor - 1) // 2

    # Count total 2Q gates to determine gates-per-layer
    total_2q = sum(1 for inst in circuit.data if inst.operation.name.lower() in TWO_QUBIT_GATES)
    if total_2q == 0 or n_layers == 0:
        return circuit.copy()

    gates_per_layer = total_2q // n_layers
    if gates_per_layer == 0:
        return circuit.copy()

    # Determine which 2Q gates belong to the target layer
    start_gate = layer_index * gates_per_layer
    end_gate = start_gate + gates_per_layer

    folded = QuantumCircuit(
        circuit.qubits, circuit.clbits, name=f"block_fold_L{layer_index}_{noise_factor}x"
    )
    gate_2q_counter = 0
    n_folded = 0

    for instruction in circuit.data:
        gate = instruction.operation
        qubits = instruction.qubits
        gate_name = gate.name.lower()

        folded.append(instruction)

        if gate_name in TWO_QUBIT_GATES:
            if start_gate <= gate_2q_counter < end_gate:
                # This gate is in the target layer — fold it
                for _ in range(n_folds):
                    folded.append(gate.inverse(), qubits, [])
                    folded.append(gate, qubits, [])
                n_folded += 1
            gate_2q_counter += 1

    _logger.debug(
        f"[block_fold] Layer {layer_index}/{n_layers}: folded {n_folded}/{gates_per_layer} "
        f"gates × {n_folds} folds (factor={noise_factor}). "
        f"Depth: {circuit.depth()} → {folded.depth()}"
    )
    return folded


@dataclass
class BlockZNEResult:
    """Result of block-level (single-layer) ZNE extrapolation.

    Attributes
    ----------
    extrapolated_value : float
        Energy extrapolated to noise_factor=0.
    r_squared : float
        R² of the extrapolation fit.
    slope : float
        Slope of E vs noise_factor for the target layer.
    layer_index : int
        Which HVA layer was used for noise amplification.
    noise_factors : list[int]
        Noise factors applied to the target layer.
    measured_values : list[float]
        Measured energies at each noise factor.
    """

    extrapolated_value: float
    r_squared: float
    slope: float
    layer_index: int
    noise_factors: list[int]
    measured_values: list[float]


def run_block_zne(
    transpiled_circuit: QuantumCircuit,
    observable: SparsePauliOp,
    backend,
    config: NoisyEstimatorConfig,
    n_layers: int,
    layer_index: int = 0,
    noise_factors: tuple[int, ...] = (1, 3, 5),
    extrapolator: str = "linear",
    seed_offset: int = 0,
) -> BlockZNEResult:
    """Run block-level ZNE: fold only one HVA layer and extrapolate.

    For p≥2 circuits, this is more precise than full-circuit gate-folding
    because the noise amplification is structurally uniform (same repeating
    block) and the folded circuit is shallower.

    Reference: arXiv:2507.23314 — "Enhanced Extrapolation-Based Quantum
    Error Mitigation Using Repetitive Structure in Quantum Algorithms"

    Parameters
    ----------
    transpiled_circuit : QuantumCircuit
        Already-transpiled, parameter-bound ISA circuit.
    observable : SparsePauliOp
        Observable to measure (already layout-mapped).
    backend : BackendV2
        Noisy backend.
    config : NoisyEstimatorConfig
        Shots and seed configuration.
    n_layers : int
        Number of HVA layers (p) in the circuit.
    layer_index : int
        Which layer to fold for noise amplification (default: 0 = first).
    noise_factors : tuple[int, ...]
        Odd integers for noise amplification (default: [1, 3, 5]).
    extrapolator : str
        "linear" or "exponential" (default: "linear").
    seed_offset : int
        Added to config.seed_simulator.

    Returns
    -------
    BlockZNEResult
        Extrapolation result with per-layer noise characterization.
    """
    _logger.info(
        f"[block_zne] Starting: layer={layer_index}/{n_layers}, noise_factors={noise_factors}"
    )

    if transpiled_circuit.num_parameters > 0:
        raise ValueError(f"Circuit has {transpiled_circuit.num_parameters} unbound parameters.")

    measured_values: list[float] = []
    for i, nf in enumerate(noise_factors):
        folded = fold_single_layer(
            transpiled_circuit, layer_index, noise_factor=nf, n_layers=n_layers
        )
        energy = noisy_estimate(
            folded,
            observable,
            backend,
            config,
            seed_offset=seed_offset + i * 100,
        )
        measured_values.append(energy)
        _logger.info(f"[block_zne]   factor={nf}: E={energy:.6f}, depth={folded.depth()}")

    nf_arr = np.array(noise_factors, dtype=float)
    meas_arr = np.array(measured_values, dtype=float)

    # WLS sigmas: same principle as gate-folding ZNE — higher noise factors
    # produce noisier measurements (σ_i ∝ √nf_i).
    sigmas = np.array([config.precision * np.sqrt(float(nf)) for nf in noise_factors])

    if extrapolator == "exponential":
        extrap, r2, slope = _extrapolate_exponential(nf_arr, meas_arr, sigmas=sigmas)
    else:
        extrap, r2, slope = _extrapolate_linear(nf_arr, meas_arr, sigmas=sigmas)

    _logger.info(f"[block_zne] Result: E={extrap:.6f}, R²={r2:.4f}, slope={slope:.6f}")

    return BlockZNEResult(
        extrapolated_value=extrap,
        r_squared=r2,
        slope=slope,
        layer_index=layer_index,
        noise_factors=list(noise_factors),
        measured_values=measured_values,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TLS-Aware Scheduling — monitor calibration stability for hardware runs
# ═══════════════════════════════════════════════════════════════════════════
# Reference: "Error mitigation with stabilized noise in superconducting
# quantum processors", Nature Communications (2025), arXiv:2407.02467.
# Key insight: qubit-TLS (Two-Level System) interactions cause quasi-static
# noise fluctuations that degrade error mitigation. Monitoring T1/T2 drift
# between runs allows us to:
#   1. Detect when calibration has drifted (abort/re-calibrate)
#   2. Schedule experiments during stable windows
#   3. Filter out runs affected by TLS events
#
# Also: IBM "Detection of time-varying noise in superconducting qubits for
# quantum error mitigation", APS Global Physics Summit 2025 — demonstrates
# post-selection based on anomalous output variance reduces errors from
# 5.4% to 1.6%.
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CalibrationSnapshot:
    """Snapshot of backend calibration data for drift monitoring.

    Attributes
    ----------
    timestamp : str
        ISO timestamp when snapshot was taken.
    qubit_t1 : dict[int, float]
        T1 (µs) per qubit.
    qubit_t2 : dict[int, float]
        T2 (µs) per qubit.
    gate_errors_2q : dict[str, float]
        2-qubit gate errors keyed by "q0-q1".
    readout_errors : dict[int, float]
        Readout error per qubit.
    """

    timestamp: str
    qubit_t1: dict[int, float]
    qubit_t2: dict[int, float]
    gate_errors_2q: dict[str, float]
    readout_errors: dict[int, float]

    @property
    def mean_t1_us(self) -> float:
        """Mean T1 across all captured qubits (µs)."""
        if not self.qubit_t1:
            return 0.0
        return float(np.mean(list(self.qubit_t1.values())))

    @property
    def mean_t2_us(self) -> float:
        """Mean T2 across all captured qubits (µs)."""
        if not self.qubit_t2:
            return 0.0
        return float(np.mean(list(self.qubit_t2.values())))

    @property
    def mean_2q_error(self) -> float:
        """Mean 2-qubit gate error across all captured gates."""
        if not self.gate_errors_2q:
            return 0.0
        return float(np.mean(list(self.gate_errors_2q.values())))


@dataclass
class DriftReport:
    """Report of calibration drift between two snapshots.

    Attributes
    ----------
    t1_drift_pct : float
        Mean |ΔT1/T1| across relevant qubits (%).
    t2_drift_pct : float
        Mean |ΔT2/T2| across relevant qubits (%).
    gate_error_drift_pct : float
        Mean |Δerr/err| across 2Q gates (%).
    max_single_drift_pct : float
        Worst-case single-qubit drift (%).
    is_stable : bool
        True if all drifts below thresholds.
    recommendation : str
        "proceed", "re-calibrate", or "abort".
    t1_threshold_pct : float
        T1 drift threshold used for this report (default 20%).
    """

    t1_drift_pct: float
    t2_drift_pct: float
    gate_error_drift_pct: float
    max_single_drift_pct: float
    is_stable: bool
    recommendation: str
    t1_threshold_pct: float = 20.0

    @property
    def should_abort(self) -> bool:
        """True if drift is severe enough to warrant aborting the run."""
        return self.recommendation == "abort"

    @property
    def abort_recommended(self) -> bool:
        """True if drift exceeds threshold (abort or re-calibrate)."""
        return self.recommendation in ("abort", "re-calibrate")

    @property
    def threshold_pct(self) -> float:
        """T1 drift threshold that was used for this assessment."""
        return self.t1_threshold_pct


def take_calibration_snapshot(
    backend,
    qubits: list[int] | None = None,
) -> CalibrationSnapshot:
    """Capture current calibration data from a backend.

    Works with both FakeTorino (local) and real IBM backends.

    Parameters
    ----------
    backend : BackendV2
        IBM backend (real or fake).
    qubits : list[int] | None
        Specific qubits to monitor. If None, captures all.

    Returns
    -------
    CalibrationSnapshot
        Current calibration state.
    """
    from datetime import datetime

    target = backend.target
    all_qubits = qubits or list(range(backend.num_qubits))

    t1_data: dict[int, float] = {}
    t2_data: dict[int, float] = {}
    readout_data: dict[int, float] = {}
    gate_2q_data: dict[str, float] = {}

    for q in all_qubits:
        # T1, T2 from qubit properties
        props = target.qubit_properties
        if props and q < len(props) and props[q] is not None:
            t1 = getattr(props[q], "t1", None)
            t2 = getattr(props[q], "t2", None)
            if t1 is not None:
                t1_data[q] = t1 * 1e6  # Convert to µs
            if t2 is not None:
                t2_data[q] = t2 * 1e6

        # Readout error
        if "measure" in target.operation_names:
            meas_props = target["measure"].get((q,))
            if meas_props and meas_props.error is not None:
                readout_data[q] = meas_props.error

    # 2Q gate errors
    for op_name in target.operation_names:
        qargs = target.qargs_for_operation_name(op_name)
        if qargs is None:
            continue
        for qa in qargs:
            if len(qa) == 2:
                q0, q1 = qa
                if q0 in all_qubits or q1 in all_qubits:
                    gate_props = target[op_name].get((q0, q1))
                    if gate_props and gate_props.error is not None:
                        key = f"{q0}-{q1}"
                        gate_2q_data[key] = gate_props.error

    return CalibrationSnapshot(
        timestamp=datetime.now(UTC).isoformat(),
        qubit_t1=t1_data,
        qubit_t2=t2_data,
        gate_errors_2q=gate_2q_data,
        readout_errors=readout_data,
    )


def check_calibration_drift(
    before: CalibrationSnapshot,
    after: CalibrationSnapshot,
    t1_threshold_pct: float = 20.0,
    t2_threshold_pct: float = 30.0,
    gate_error_threshold_pct: float = 50.0,
) -> DriftReport:
    """Compare two calibration snapshots to detect drift.

    Thresholds based on IBM's findings (Nature Comms 2025): TLS events
    can cause T1 drops of 30-50% on individual qubits. A 20% mean drift
    in T1 is the recommended abort threshold.

    Parameters
    ----------
    before : CalibrationSnapshot
        Calibration at the start of the experiment.
    after : CalibrationSnapshot
        Calibration at the end (or mid-point).
    t1_threshold_pct : float
        Maximum allowed mean T1 drift (default: 20%).
    t2_threshold_pct : float
        Maximum allowed mean T2 drift (default: 30%).
    gate_error_threshold_pct : float
        Maximum allowed mean gate error drift (default: 50%).

    Returns
    -------
    DriftReport
        Drift analysis with stability assessment and recommendation.
    """
    # T1 drift
    t1_drifts = []
    for q in before.qubit_t1:
        if q in after.qubit_t1 and before.qubit_t1[q] > 0:
            drift = abs(after.qubit_t1[q] - before.qubit_t1[q]) / before.qubit_t1[q]
            t1_drifts.append(drift * 100)

    # T2 drift
    t2_drifts = []
    for q in before.qubit_t2:
        if q in after.qubit_t2 and before.qubit_t2[q] > 0:
            drift = abs(after.qubit_t2[q] - before.qubit_t2[q]) / before.qubit_t2[q]
            t2_drifts.append(drift * 100)

    # Gate error drift
    gate_drifts = []
    for key in before.gate_errors_2q:
        if key in after.gate_errors_2q and before.gate_errors_2q[key] > 0:
            drift = (
                abs(after.gate_errors_2q[key] - before.gate_errors_2q[key])
                / before.gate_errors_2q[key]
            )
            gate_drifts.append(drift * 100)

    mean_t1 = float(np.mean(t1_drifts)) if t1_drifts else 0.0
    mean_t2 = float(np.mean(t2_drifts)) if t2_drifts else 0.0
    mean_gate = float(np.mean(gate_drifts)) if gate_drifts else 0.0
    max_single = float(max(t1_drifts + t2_drifts + gate_drifts, default=0.0))

    # Stability assessment
    is_stable = (
        mean_t1 <= t1_threshold_pct
        and mean_t2 <= t2_threshold_pct
        and mean_gate <= gate_error_threshold_pct
    )

    if not is_stable:
        if mean_t1 > t1_threshold_pct * 2 or max_single > 100:
            recommendation = "abort"
        else:
            recommendation = "re-calibrate"
    else:
        recommendation = "proceed"

    _logger.info(
        f"[tls_monitor] Drift: T1={mean_t1:.1f}%, T2={mean_t2:.1f}%, "
        f"gates={mean_gate:.1f}%, max_single={max_single:.1f}% → {recommendation}"
    )

    return DriftReport(
        t1_drift_pct=mean_t1,
        t2_drift_pct=mean_t2,
        gate_error_drift_pct=mean_gate,
        max_single_drift_pct=max_single,
        is_stable=is_stable,
        recommendation=recommendation,
        t1_threshold_pct=t1_threshold_pct,
    )
