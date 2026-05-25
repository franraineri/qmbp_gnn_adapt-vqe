"""
Shared Experiment Utilities — Hamed V7 Full Experiments.

Provides common metrics computation, result serialization, phase classification,
and aggregation logic shared across all 5 technique experiment scripts.

Usage:
    from experiment_utils import (
        ExperimentMetrics, SubExperimentResult,
        compute_metrics, classify_phase, save_experiment_result,
        aggregate_results, compute_fidelity,
    )
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.poc.v6 import (
    ClassicalSolver,
    HamiltonianBuilder,
    make_lattice,
)

# ── Data Models ──────────────────────────────────────────────────────────────


@dataclass
class ExperimentMetrics:
    """Standard metrics reported for every sub-experiment run."""

    energy: float
    energy_error: float  # ΔE = |E_method - E_exact|
    relative_error: float  # ΔE / gap
    fidelity: float | None  # State fidelity (None if unavailable)
    phase_label: str  # "ferromagnetic" or "paramagnetic"
    phase_correct: bool  # Whether classification matches exact
    wall_time_s: float
    n_evaluations: int
    seed: int


@dataclass
class SubExperimentResult:
    """Complete result for one sub-experiment (e.g., 1A, 3C)."""

    experiment_id: str  # e.g., "1A", "3C", "4E"
    technique: int  # 1-5
    description: str
    config: dict = field(default_factory=dict)
    metrics: list[ExperimentMetrics] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    success: bool = True
    error: str | None = None


# ── Core Functions ───────────────────────────────────────────────────────────


def compute_metrics(
    energy: float,
    exact_energy: float,
    gap: float,
    circuit=None,
    params=None,
    exact_state=None,
    wall_time_s: float = 0.0,
    n_evaluations: int = 0,
    seed: int = 0,
    h_value: float = 1.0,
    mag_x_exact: float = 0.0,
    corr_zz_exact: float = 0.0,
) -> ExperimentMetrics:
    """Compute all standard metrics for a single experiment run.

    Parameters
    ----------
    energy : float
        Energy obtained by the method under test.
    exact_energy : float
        Exact ground state energy from ClassicalSolver.
    gap : float
        Spectral gap (E1 - E0). Must be > 0.
    circuit : QuantumCircuit, optional
        Parameterized circuit (for fidelity computation).
    params : np.ndarray, optional
        Optimized parameters for the circuit.
    exact_state : np.ndarray, optional
        Exact ground state vector (for fidelity computation).
    wall_time_s : float
        Wall clock time in seconds.
    n_evaluations : int
        Number of function evaluations used.
    seed : int
        Random seed used for this run.
    h_value : float
        Transverse field value (for phase classification fallback).
    mag_x_exact : float
        Exact bulk-averaged ⟨X⟩ (for phase correctness check).
    corr_zz_exact : float
        Exact bulk-averaged ⟨ZZ⟩ (for phase correctness check).

    Returns
    -------
    ExperimentMetrics
        Computed metrics with all fields populated.
    """
    energy_error = abs(energy - exact_energy)
    relative_error = energy_error / gap if gap > 0 else float("inf")

    # Fidelity
    fidelity = None
    if circuit is not None and params is not None and exact_state is not None:
        fidelity = compute_fidelity(circuit, params, exact_state)

    # Phase classification from circuit output (if available)
    if circuit is not None and params is not None:
        phase_label, _, _ = _classify_from_statevector(circuit, params)
    else:
        # Fallback: use h_value heuristic (h > 1 → paramagnetic)
        phase_label = "paramagnetic" if h_value > 1.0 else "ferromagnetic"

    # Determine exact phase from exact observables
    exact_phase = "paramagnetic" if abs(mag_x_exact) >= abs(corr_zz_exact) else "ferromagnetic"
    # If no exact observables provided, use h_value heuristic
    if mag_x_exact == 0.0 and corr_zz_exact == 0.0:
        exact_phase = "paramagnetic" if h_value > 1.0 else "ferromagnetic"

    phase_correct = phase_label == exact_phase

    return ExperimentMetrics(
        energy=energy,
        energy_error=energy_error,
        relative_error=relative_error,
        fidelity=fidelity,
        phase_label=phase_label,
        phase_correct=phase_correct,
        wall_time_s=wall_time_s,
        n_evaluations=n_evaluations,
        seed=seed,
    )


def classify_phase(circuit, params, lattice, hamiltonian_builder) -> tuple[str, float, float]:
    """Classify phase from circuit output using ⟨X⟩ vs ⟨ZZ⟩ comparison.

    Evaluates the parameterized circuit at the given parameters and computes
    bulk-averaged magnetization ⟨X⟩ and nearest-neighbor correlation ⟨ZZ⟩.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized HVA circuit.
    params : np.ndarray
        Parameter values to bind.
    lattice : LatticeConfig
        Lattice configuration (for edge list).
    hamiltonian_builder : HamiltonianBuilder
        Builder instance (used for observable construction).

    Returns
    -------
    tuple[str, float, float]
        (phase_label, mag_x, corr_zz) where:
        - "ferromagnetic" if |corr_zz| > |mag_x|
        - "paramagnetic" if |mag_x| >= |corr_zz|
    """
    from qiskit.quantum_info import SparsePauliOp, Statevector

    n_qubits = circuit.num_qubits

    # Bind parameters and get statevector
    bound_circuit = circuit.assign_parameters(dict(zip(circuit.parameters, params, strict=False)))
    sv = Statevector(bound_circuit)

    # Compute ⟨X⟩ = (1/N) Σ ⟨Xi⟩
    mag_x_total = 0.0
    for i in range(n_qubits):
        label = "I" * (n_qubits - 1 - i) + "X" + "I" * i
        op = SparsePauliOp.from_list([(label, 1.0)])
        mag_x_total += sv.expectation_value(op).real
    mag_x = mag_x_total / n_qubits

    # Compute ⟨ZZ⟩ = (1/|edges|) Σ ⟨ZiZj⟩
    edges = lattice.edges
    corr_zz_total = 0.0
    for i, j in edges:
        label_list = ["I"] * n_qubits
        label_list[n_qubits - 1 - i] = "Z"
        label_list[n_qubits - 1 - j] = "Z"
        label = "".join(label_list)
        op = SparsePauliOp.from_list([(label, 1.0)])
        corr_zz_total += sv.expectation_value(op).real
    corr_zz = corr_zz_total / len(edges) if edges else 0.0

    # Classification rule: |corr_zz| > |mag_x| → ferromagnetic
    phase_label = "ferromagnetic" if abs(corr_zz) > abs(mag_x) else "paramagnetic"

    return phase_label, mag_x, corr_zz


def save_experiment_result(
    result: SubExperimentResult,
    output_dir: Path,
    prefix: str = "",
) -> Path:
    """Save result as JSON with timestamp. Returns path to saved file.

    Parameters
    ----------
    result : SubExperimentResult
        The experiment result to serialize.
    output_dir : Path
        Directory to save the JSON file.
    prefix : str
        Optional filename prefix (e.g., "nevergrad").

    Returns
    -------
    Path
        Path to the saved JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename
    parts = [prefix] if prefix else []
    parts.append(result.experiment_id)
    parts.append(result.timestamp)
    filename = "_".join(parts) + ".json"

    filepath = output_dir / filename

    # Serialize
    data = _result_to_dict(result)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=_json_serializer)

    return filepath


def aggregate_results(
    metrics_list: list[ExperimentMetrics],
) -> dict:
    """Compute summary statistics (mean, std, min, max) across seeds.

    Parameters
    ----------
    metrics_list : list[ExperimentMetrics]
        List of metrics from multiple runs/seeds.

    Returns
    -------
    dict
        Summary with keys: mean_energy_error, std_energy_error,
        mean_relative_error, std_relative_error, mean_fidelity,
        std_fidelity, mean_wall_time, phase_accuracy, n_runs.
    """
    if not metrics_list:
        return {"n_runs": 0}

    errors = [m.energy_error for m in metrics_list]
    rel_errors = [m.relative_error for m in metrics_list]
    wall_times = [m.wall_time_s for m in metrics_list]
    n_evals = [m.n_evaluations for m in metrics_list]
    phase_correct = [m.phase_correct for m in metrics_list]

    # Fidelity (may be None for some runs)
    fidelities = [m.fidelity for m in metrics_list if m.fidelity is not None]

    summary = {
        "n_runs": len(metrics_list),
        # Energy error
        "mean_energy_error": float(np.mean(errors)),
        "std_energy_error": float(np.std(errors)),
        "min_energy_error": float(np.min(errors)),
        "max_energy_error": float(np.max(errors)),
        # Relative error
        "mean_relative_error": float(np.mean(rel_errors)),
        "std_relative_error": float(np.std(rel_errors)),
        "min_relative_error": float(np.min(rel_errors)),
        "max_relative_error": float(np.max(rel_errors)),
        # Wall time
        "mean_wall_time_s": float(np.mean(wall_times)),
        "std_wall_time_s": float(np.std(wall_times)),
        "min_wall_time_s": float(np.min(wall_times)),
        "max_wall_time_s": float(np.max(wall_times)),
        # Evaluations
        "mean_n_evaluations": float(np.mean(n_evals)),
        "std_n_evaluations": float(np.std(n_evals)),
        "min_n_evaluations": float(np.min(n_evals)),
        "max_n_evaluations": float(np.max(n_evals)),
        # Phase accuracy
        "phase_accuracy": float(np.mean(phase_correct)),
    }

    if fidelities:
        summary["mean_fidelity"] = float(np.mean(fidelities))
        summary["std_fidelity"] = float(np.std(fidelities))
        summary["min_fidelity"] = float(np.min(fidelities))
        summary["max_fidelity"] = float(np.max(fidelities))
    else:
        summary["mean_fidelity"] = None
        summary["std_fidelity"] = None
        summary["min_fidelity"] = None
        summary["max_fidelity"] = None

    return summary


def compute_fidelity(circuit, params, exact_state) -> float:
    """Compute state fidelity |⟨ψ_exact|ψ_vqe⟩|².

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized circuit.
    params : np.ndarray
        Parameter values to bind.
    exact_state : np.ndarray
        Exact ground state vector.

    Returns
    -------
    float
        Fidelity in [0, 1].
    """
    from qiskit.quantum_info import Statevector, state_fidelity

    bound_circuit = circuit.assign_parameters(dict(zip(circuit.parameters, params, strict=False)))
    sv_vqe = Statevector(bound_circuit)
    sv_exact = Statevector(exact_state)

    fid = state_fidelity(sv_vqe, sv_exact)
    return float(fid)


# ── Private Helpers ──────────────────────────────────────────────────────────


def _classify_from_statevector(circuit, params) -> tuple[str, float, float]:
    """Classify phase directly from circuit statevector (no lattice needed).

    Uses a simple heuristic: compute ⟨X⟩ and ⟨ZZ⟩ for nearest-neighbor chain.
    """
    from qiskit.quantum_info import SparsePauliOp, Statevector

    n_qubits = circuit.num_qubits
    bound_circuit = circuit.assign_parameters(dict(zip(circuit.parameters, params, strict=False)))
    sv = Statevector(bound_circuit)

    # ⟨X⟩ average
    mag_x_total = 0.0
    for i in range(n_qubits):
        label = "I" * (n_qubits - 1 - i) + "X" + "I" * i
        op = SparsePauliOp.from_list([(label, 1.0)])
        mag_x_total += sv.expectation_value(op).real
    mag_x = mag_x_total / n_qubits

    # ⟨ZZ⟩ nearest-neighbor (chain assumption)
    corr_zz_total = 0.0
    n_bonds = n_qubits - 1
    for k in range(n_bonds):
        i, j = k, k + 1
        label_list = ["I"] * n_qubits
        label_list[n_qubits - 1 - i] = "Z"
        label_list[n_qubits - 1 - j] = "Z"
        label = "".join(label_list)
        op = SparsePauliOp.from_list([(label, 1.0)])
        corr_zz_total += sv.expectation_value(op).real
    corr_zz = corr_zz_total / n_bonds if n_bonds > 0 else 0.0

    phase_label = "ferromagnetic" if abs(corr_zz) > abs(mag_x) else "paramagnetic"
    return phase_label, mag_x, corr_zz


TECHNIQUE_NAMES = {
    1: "Nevergrad (gradient-free VQE)",
    2: "QRC warm-start",
    3: "MPS simulation",
    4: "SPSA hardware optimizer",
    5: "Noise-aware training",
}


def _result_to_dict(result: SubExperimentResult) -> dict[str, Any]:
    """Convert SubExperimentResult to a JSON-serializable dict."""
    data = {
        "experiment_id": result.experiment_id,
        "technique": result.technique,
        "technique_name": TECHNIQUE_NAMES.get(result.technique, "unknown"),
        "description": result.description,
        "timestamp": result.timestamp,
        "config": result.config,
        "metrics": [asdict(m) for m in result.metrics],
        "summary": result.summary,
        "success": result.success,
        "error": result.error,
    }
    return data


def _json_serializer(obj):
    """Custom JSON serializer for numpy types and other non-serializable objects."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ── Shared Experiment Helpers ────────────────────────────────────────────────


def evaluate_energy_statevector(circuit, hamiltonian, params) -> float:
    """Evaluate ⟨H⟩ using exact StatevectorEstimator.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized circuit.
    hamiltonian : SparsePauliOp
        Hamiltonian operator.
    params : np.ndarray
        Parameter values to bind.

    Returns
    -------
    float
        Expectation value ⟨ψ(θ)|H|ψ(θ)⟩.
    """
    from qiskit.primitives import StatevectorEstimator

    estimator = StatevectorEstimator()
    bound = circuit.assign_parameters(dict(zip(circuit.parameters, params, strict=False)))
    job = estimator.run([(bound, hamiltonian)])
    return float(job.result()[0].data.evs)


def create_fake_torino_estimator(n_shots: int = 8192):
    """Create FakeTorino BackendEstimatorV2 for noisy simulation.

    Parameters
    ----------
    n_shots : int
        Number of shots per circuit execution.

    Returns
    -------
    tuple[BackendEstimatorV2, FakeTorino]
        (estimator, backend) pair. Returns (None, None) if unavailable.

    Raises
    ------
    ImportError
        If qiskit_ibm_runtime is not installed.
    """
    try:
        from qiskit.primitives import BackendEstimatorV2
        from qiskit_ibm_runtime.fake_provider import FakeTorino
    except ImportError:
        return None, None

    backend = FakeTorino()
    estimator = BackendEstimatorV2(backend=backend)
    return estimator, backend


def evaluate_energy_fake_torino(circuit, hamiltonian, params, estimator) -> float:
    """Evaluate ⟨H⟩ using FakeTorino BackendEstimatorV2.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized circuit.
    hamiltonian : SparsePauliOp
        Hamiltonian operator.
    params : np.ndarray
        Parameter values to bind.
    estimator : BackendEstimatorV2
        Pre-configured FakeTorino estimator.

    Returns
    -------
    float
        Noisy expectation value.
    """
    bound = circuit.assign_parameters(dict(zip(circuit.parameters, params, strict=False)))
    job = estimator.run([(bound, hamiltonian)], precision=0.01)
    return float(job.result()[0].data.evs)


def warm_start_sweep(
    circuit,
    h_values_desc: list[float],
    builder: HamiltonianBuilder,
    solver: ClassicalSolver,
    optimizer_fn,
    initial_theta: np.ndarray | None = None,
    N: int = 6,
) -> list[dict]:
    """Run optimizer with warm-start from adjacent h-point (descending sweep).

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized HVA circuit.
    h_values_desc : list[float]
        h-values sorted DESCENDING (h=2→0).
    builder : HamiltonianBuilder
        Hamiltonian builder instance.
    solver : ClassicalSolver
        Classical solver for exact ground truth.
    optimizer_fn : callable
        Function signature: (circuit, H, initial_guess) → (theta_opt, energy, n_evals, wall_time_s)
    initial_theta : np.ndarray | None
        Initial parameters for first h-value. If None, uses small random.
    N : int
        Number of qubits.

    Returns
    -------
    list[dict]
        List of result dicts per h-value with keys:
        h, theta_opt, energy, exact_energy, gap, energy_error, relative_error,
        n_evaluations, wall_time_s.
    """
    n_params = circuit.num_parameters
    if initial_theta is None:
        initial_theta = np.random.uniform(-0.01, 0.01, n_params)

    results = []
    prev_theta = initial_theta.copy()

    for h in h_values_desc:
        lattice = make_lattice("chain_1d", N, J=1.0, h=h)
        H = builder.build(lattice)
        exact = solver.solve(H, lattice)

        theta_opt, energy, n_evals, wall_time = optimizer_fn(circuit, H, prev_theta)

        energy_error = abs(energy - exact.ground_energy)
        gap = exact.gap if exact.gap > 0 else 1e-10
        relative_error = energy_error / gap

        results.append(
            {
                "h": h,
                "theta_opt": theta_opt,
                "energy": energy,
                "exact_energy": exact.ground_energy,
                "gap": gap,
                "energy_error": energy_error,
                "relative_error": relative_error,
                "n_evaluations": n_evals,
                "wall_time_s": wall_time,
                "exact_state": exact.ground_state,
            }
        )

        prev_theta = theta_opt  # Warm-start for next point

    return results


def build_experiment_circuit(N: int, p: int = 2):
    """Build HVA circuit and return (circuit, n_params, base_lattice).

    Convenience wrapper for experiment scripts.

    Parameters
    ----------
    N : int
        Number of qubits.
    p : int
        Number of HVA layers (default 2).

    Returns
    -------
    tuple[QuantumCircuit, int, LatticeConfig]
        (circuit, n_params, base_lattice)
    """
    from src.poc.v6 import HVACircuitBuilder

    base_lattice = make_lattice("chain_1d", N, J=1.0, h=1.0)
    hva = HVACircuitBuilder()
    qc, _ = hva.create(N, p, base_lattice)
    return qc, qc.num_parameters, base_lattice
