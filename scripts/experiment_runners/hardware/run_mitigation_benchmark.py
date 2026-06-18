"""Mitigation Benchmark Runner — systematic evaluation of 19 configs.

Executes 19 error suppression/mitigation configurations on the GNN-HVA circuit
(TFIM, N=10, p=1, heavy_hex) sequentially, with CLI filtering by config_id,
priority level, and h-values. Supports parallel execution via multiple terminals
with file-lock-protected manifest writes.

Usage:
    python run_mitigation_benchmark.py --mode fake_backend --configs C0,C5 --h-values 3.25,3.5
    python run_mitigation_benchmark.py --mode hardware --priority P0,P1
    python run_mitigation_benchmark.py --export-configs

Phases:
    FaseA: Full 19 configs on FakeTorino (simulation, 0 QPU cost)
    FaseB: Top-3 configs on IBM Torino hardware (real QPU)
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import importlib
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Invalidate bytecode caches to prevent stale .pyc issues
# (critical after code changes in sibling modules)
importlib.invalidate_caches()

logger = logging.getLogger(__name__)

# Ensure project root is in path for both package and script imports
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from qmbp_simulation import ClassicalSolver, HamiltonianBuilder, make_lattice
from qmbp_simulation.analysis.circuit_visualizer import (
    compute_error_budget,
    transpiled_circuit_stats,
)
from qmbp_simulation.circuits import HVACircuitBuilder
from qmbp_simulation.execution import (
    NoisyEstimatorConfig,
    affine_correct_energy,
    noisy_estimate,
    run_gate_folding_zne,
    run_pea_zne,
)
from qmbp_simulation.execution.mitiq_utils import (
    run_mitiq_cdr,
    run_mitiq_ddd_zne,
    run_mitiq_zne,
)
from qmbp_simulation.utils.helpers import json_dump, json_serialize
from scripts.experiment_runners.hardware.benchmark_configs import (
    BENCHMARK_CONFIGS,
    BenchmarkConfig,
)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

BENCHMARK_VERSION = "1.0"
RESULTS_BASE = Path("results/mitigation_benchmark")
MANIFEST_PATH = RESULTS_BASE / "manifest.json"


# ---------------------------------------------------------------------------
# Manifest persistence (file-lock safe for concurrent terminals)
# ---------------------------------------------------------------------------


def append_to_manifest(entry: dict, manifest_path: Path | None = None) -> None:
    """Append entry to manifest.json with exclusive file-level locking.

    Thread/process safe for concurrent writes from multiple terminals.
    Creates manifest.json with empty array if it doesn't exist.

    Parameters
    ----------
    entry : dict
        Must contain: config_id, execution_mode, h_value, timestamp,
        result_path (relative to RESULTS_BASE), delta_e_gap, correct_label.
    manifest_path : Path | None
        Override path (defaults to MANIFEST_PATH).
    """
    path = manifest_path or MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read()
            entries = json.loads(content) if content.strip() else []
            # Deduplication: skip if entry with same (config_id, h_value, seed) exists
            config_id = entry.get("config_id", "")
            h_val = entry.get("h_value")
            seed = entry.get("seed", 42)
            already_exists = any(
                e.get("config_id") == config_id
                and e.get("h_value") == h_val
                and e.get("seed", 42) == seed
                for e in entries
            )
            if not already_exists:
                entries.append(entry)
                f.seek(0)
                f.truncate()
                json.dump(entries, f, indent=2, default=json_serialize)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _build_manifest_entry(
    config_id: str,
    h_value: float,
    mode: str,
    result_path: Path,
    envelope: dict,
) -> dict:
    """Build a manifest entry from a completed execution.

    Parameters
    ----------
    config_id : str
        Configuration identifier (e.g. "C5_full_pea_balanced").
    h_value : float
        Transverse field value used in the execution.
    mode : str
        Execution mode ("fake_backend" or "hardware").
    result_path : Path
        Full path to the saved result JSON file.
    envelope : dict
        The complete ResultEnvelope dict containing benchmark_metadata,
        results, etc.

    Returns
    -------
    dict
        Manifest entry with keys: config_id, execution_mode, h_value,
        timestamp, result_path, delta_e_gap, correct_label.
    """
    return {
        "config_id": config_id,
        "execution_mode": mode,
        "h_value": h_value,
        "timestamp": envelope["benchmark_metadata"]["timestamp"],
        "result_path": str(result_path.relative_to(RESULTS_BASE)),
        "delta_e_gap": envelope["results"]["delta_e_gap"],
        "correct_label": envelope["results"]["correct_label"],
    }


# ---------------------------------------------------------------------------
# Result path construction
# ---------------------------------------------------------------------------


def _build_result_path(
    config_id: str,
    h_value: float,
    mode: str,
    seed: int,
) -> Path:
    """Build deterministic result file path.

    Convention:
        results/mitigation_benchmark/{mode}/{config_id}/h{val}_run_{timestamp}[_seed{N}].json

    Naming rules:
        - h_value formatted as "3p25" (replace '.' with 'p')
        - If seed != 42, append "_seed{seed}" before .json
        - Timestamp from framework's generate_timestamp() (YYYYMMDD_HHMMSS)

    Parameters
    ----------
    config_id : str
        Configuration identifier (e.g., "C0_raw").
    h_value : float
        Transverse field strength.
    mode : str
        Execution mode ("fake_backend" or "hardware").
    seed : int
        Random seed used for this run.

    Returns
    -------
    Path
        Full path to the result JSON file.
    """
    from qmbp_simulation.framework.result_io import generate_timestamp

    h_str = f"h{str(h_value).replace('.', 'p')}"
    timestamp = generate_timestamp()
    seed_suffix = f"_seed{seed}" if seed != 42 else ""
    filename = f"{h_str}_run_{timestamp}{seed_suffix}.json"
    return RESULTS_BASE / mode / config_id / filename


# ---------------------------------------------------------------------------
# ResultEnvelope construction
# ---------------------------------------------------------------------------


def _build_envelope(
    config: BenchmarkConfig,
    h_value: float,
    mode: str,
    seed: int,
    circuit_stats: dict[str, Any],
    error_budget: dict[str, Any],
    execution_result: dict[str, Any],
    wall_time_s: float,
    hardware_calibration: dict[str, Any] | None = None,
    aqc_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the complete ResultEnvelope for one benchmark run.

    Sections:
      - benchmark_metadata: config_id, execution_mode, h_value, timestamp,
        benchmark_version, seed
      - circuit_stats: merged from transpiled_circuit_stats + derived metrics
        (circuit_depth_with_dd_estimate, routing_overhead_pct,
        transpiled_vs_logical_ratio)
      - timing: wall_time_s, qpu_seconds, noise_learning_time_s
      - results: e_mitigated, e_raw, e_exact, delta_e_gap, phase_label, etc.
        + per_site_magnetization_std, energy_within_physical_bounds
      - shots: int
      - mitigation_config: snapshot of BenchmarkConfig fields
      - hardware_calibration: null for fake_backend, populated dict for hw
      - aqc_metrics: null or dict (AQC configs only)

    Parameters
    ----------
    config : BenchmarkConfig
        The benchmark configuration used for this run.
    h_value : float
        Transverse field strength.
    mode : str
        Execution mode ("fake_backend" or "hardware").
    seed : int
        Random seed used.
    circuit_stats : dict
        Output from transpiled_circuit_stats() + derived metrics.
    error_budget : dict
        Output from compute_error_budget().
    execution_result : dict
        Results from the execution router (energies, labels, etc.).
    wall_time_s : float
        Total wall-clock time for this run.
    hardware_calibration : dict or None
        Hardware calibration data (None for fake_backend).
    aqc_metrics : dict or None
        AQC compression metrics (None for non-AQC configs).

    Returns
    -------
    dict
        Complete ResultEnvelope ready for JSON serialization.
    """
    envelope: dict[str, Any] = {
        "benchmark_metadata": {
            "config_id": config.config_id,
            "execution_mode": mode,
            "h_value": h_value,
            "timestamp": datetime.now(UTC).isoformat(),
            "benchmark_version": BENCHMARK_VERSION,
            "seed": seed,
        },
        "circuit_stats": {
            **circuit_stats,
            "optimization_level": config.optimization_level,
            "fidelity_estimate": error_budget.get("fidelity_estimate"),
        },
        "timing": {
            "wall_time_s": wall_time_s,
            "qpu_seconds": execution_result.get("qpu_seconds"),
            "noise_learning_time_s": execution_result.get("noise_learning_time_s"),
        },
        "results": {
            "e_mitigated": execution_result.get("e_mitigated"),
            "e_raw": execution_result.get("e_raw"),
            "e_exact": execution_result.get("e_exact"),
            "delta_e_gap": execution_result.get("delta_e_gap"),
            "improvement_vs_raw": execution_result.get("improvement_vs_raw"),
            "zne_r2": execution_result.get("zne_r2"),
            "phase_label": execution_result.get("phase_label"),
            "correct_label": execution_result.get("correct_label"),
            "per_site_magnetization_std": execution_result.get("per_site_magnetization_std"),
            "energy_within_physical_bounds": execution_result.get("energy_within_physical_bounds"),
        },
        "shots": execution_result.get("shots", 16384),
        "mitigation_config": dataclasses.asdict(config),
        "hardware_calibration": hardware_calibration,
    }

    if aqc_metrics is not None:
        envelope["aqc_metrics"] = aqc_metrics

    return envelope


# ---------------------------------------------------------------------------
# Hardware calibration collection
# ---------------------------------------------------------------------------


def _collect_hardware_calibration(backend: Any, job: Any) -> dict[str, Any] | None:
    """Extract hardware calibration metrics for ResultEnvelope.

    Collects T1, T2, CX error, readout error averaged over layout qubits,
    calibration age, and job execution time from the backend and job objects.

    Parameters
    ----------
    backend : Any
        IBM backend object with .properties() and .layout_qubits attributes.
    job : Any
        Completed IBM Runtime job with .metrics() method.

    Returns
    -------
    dict or None
        Dict with keys: t1_mean_layout, t2_mean_layout, cx_error_mean_layout,
        readout_error_mean, calibration_age_hours, job_execution_time_s.
        Returns None if properties() returns None or layout_qubits is empty.
    """
    # Get backend properties
    props = getattr(backend, "properties", None)
    if callable(props):
        props = props()
    else:
        props = None

    layout_qubits = getattr(backend, "layout_qubits", None)

    # If no properties or no layout qubits, return dict with None values
    if props is None or not layout_qubits:
        return {
            "t1_mean_layout": None,
            "t2_mean_layout": None,
            "cx_error_mean_layout": None,
            "readout_error_mean": None,
            "calibration_age_hours": None,
            "job_execution_time_s": None,
        }

    # Extract T1/T2 for layout qubits (in μs)
    t1s = [props.t1(q) for q in layout_qubits if props.t1(q) is not None]
    t2s = [props.t2(q) for q in layout_qubits if props.t2(q) is not None]

    # Extract CX errors for layout CX pairs
    layout_cx_pairs = getattr(backend, "layout_cx_pairs", [])
    cx_errors = []
    for q1, q2 in layout_cx_pairs:
        try:
            err = props.gate_error("cx", [q1, q2])
            if err is not None:
                cx_errors.append(err)
        except Exception:
            pass

    # Extract readout errors for layout qubits
    readout_errors = []
    for q in layout_qubits:
        try:
            err = props.readout_error(q)
            if err is not None:
                readout_errors.append(err)
        except Exception:
            pass

    # Calibration age
    calibration_age_hours: float | None = None
    try:
        cal_time = props.last_update_date
        if cal_time is not None:
            age_delta = datetime.now(cal_time.tzinfo) - cal_time
            calibration_age_hours = age_delta.total_seconds() / 3600
    except Exception:
        pass

    # Job execution time from metrics
    job_execution_time_s: float | None = None
    try:
        metrics = job.metrics()
        job_execution_time_s = metrics.get("execution_time", None)
    except Exception:
        pass

    return {
        "t1_mean_layout": float(np.mean(t1s)) * 1e6 if t1s else None,
        "t2_mean_layout": float(np.mean(t2s)) * 1e6 if t2s else None,
        "cx_error_mean_layout": float(np.mean(cx_errors)) if cx_errors else None,
        "readout_error_mean": float(np.mean(readout_errors)) if readout_errors else None,
        "calibration_age_hours": calibration_age_hours,
        "job_execution_time_s": job_execution_time_s,
    }


# ---------------------------------------------------------------------------
# Persistence — save envelope to disk
# ---------------------------------------------------------------------------


def _save_result(envelope: dict[str, Any], result_path: Path) -> None:
    """Persist a ResultEnvelope to disk as JSON.

    Uses json_dump from qmbp_simulation.utils.helpers which handles:
    - Directory creation (mkdir -p)
    - numpy type serialization via json_serialize
    - Pretty-printing with indent=2

    Parameters
    ----------
    envelope : dict
        Complete ResultEnvelope from _build_envelope().
    result_path : Path
        Target file path (from _build_result_path()).
    """
    json_dump(envelope, result_path)


# ---------------------------------------------------------------------------
# Derived circuit statistics
# ---------------------------------------------------------------------------


def compute_derived_circuit_stats(stats: dict[str, Any], n_2q_logical: int) -> dict[str, Any]:
    """Compute derived circuit metrics from transpiled stats.

    Adds three fields that quantify transpilation overhead and DD potential:
      - circuit_depth_with_dd_estimate: total depth including worst-case idle
        stretch (upper bound on depth after DD insertion)
      - routing_overhead_pct: percentage of extra 2Q gates introduced by
        routing (SWAP decomposition) relative to the logical circuit
      - transpiled_vs_logical_ratio: depth blow-up factor from transpilation

    Parameters
    ----------
    stats : dict
        Output from transpiled_circuit_stats() with at minimum:
        depth_transpiled, max_idle_stretch, n_2q_gates, depth_logical.
    n_2q_logical : int
        Number of 2-qubit gates in the logical (pre-transpilation) circuit.

    Returns
    -------
    dict
        New dict containing all original keys plus the three derived fields.
        The input dict is NOT mutated.
    """
    derived = {**stats}

    # Upper bound on circuit depth after DD sequence insertion
    # transpiled_circuit_stats returns "depth" as canonical key
    depth_transpiled = stats.get("depth", stats.get("depth_transpiled", 0))
    derived["circuit_depth_with_dd_estimate"] = depth_transpiled + stats.get("max_idle_stretch", 0)

    # Routing overhead: extra 2Q gates from SWAP routing (% over logical)
    derived["routing_overhead_pct"] = (
        (stats["n_2q_gates"] - n_2q_logical) / n_2q_logical * 100 if n_2q_logical > 0 else 0.0
    )

    # Transpilation depth blow-up factor
    depth_logical = stats.get("depth_logical", 0)
    derived["transpiled_vs_logical_ratio"] = (
        depth_transpiled / depth_logical if depth_logical > 0 else 0.0
    )

    return derived


# ---------------------------------------------------------------------------
# Affine-on-raw correction (H8: affine never worsens, even without ZNE)
# ---------------------------------------------------------------------------


def apply_affine_on_raw(
    config: BenchmarkConfig,
    execution_result: dict[str, Any],
    e_exact: float,
    e_upper: float,
) -> dict[str, Any]:
    """Apply affine correction directly to e_raw when no ZNE is configured.

    When affine_enabled=True and zne_method=None, there is no ZNE-mitigated
    energy to clip. Instead, we apply affine_correct_energy to e_raw itself.
    This validates hypothesis H8: affine correction never worsens the result,
    even without ZNE.

    Parameters
    ----------
    config : BenchmarkConfig
        The benchmark configuration for the current run.
    execution_result : dict
        Execution result dict containing at least "e_raw".
    e_exact : float
        Ground state energy (rigorous lower bound from ClassicalSolver).
    e_upper : float
        Upper bound on energy for the system.

    Returns
    -------
    dict
        Updated execution_result. If affine-on-raw applies, sets
        "e_mitigated" to the affine-corrected e_raw. Otherwise returns
        the dict unchanged.
    """
    if config.affine_enabled and config.zne_method is None:
        e_raw = execution_result["e_raw"]
        corrected = affine_correct_energy(e_raw, e_exact, e_upper)
        execution_result["e_mitigated"] = corrected.corrected_energy
    return execution_result


# ---------------------------------------------------------------------------
# ExecutionRouter — dispatch execution by zne_method
# ---------------------------------------------------------------------------


def _execute_raw(
    transpiled_circuit: QuantumCircuit,
    H_mapped: SparsePauliOp,
    backend: Any,
    shots: int,
) -> dict[str, Any]:
    """Execute without ZNE — raw estimation with DD/Twirling/TREX only.

    Used by: C0_raw, C1_dd_only, C2_dd_tw, C18_aqc_raw.

    Uses noisy_estimate() (BackendEstimatorV2) for proper noisy simulation
    on FakeTorino or hardware. This produces realistic noisy expectations
    that reflect the backend's noise model, unlike StatevectorEstimator
    which gives exact results and defeats the benchmarking purpose.

    For hardware backends, the _job key provides the RuntimeJobV2 object
    for QPU metrics extraction (qpu_seconds from job.metrics()).

    Parameters
    ----------
    transpiled_circuit : QuantumCircuit
        Already-transpiled, parameter-bound ISA circuit.
    H_mapped : SparsePauliOp
        Observable mapped to the physical layout.
    backend : BackendV2
        Noisy backend (FakeTorino or real hardware).
    shots : int
        Number of shots for the estimation.

    Returns
    -------
    dict
        Execution result with "e_raw", "e_mitigated" (None), "shots",
        "_job" (None for simulation, RuntimeJobV2 for hardware) keys.
    """
    noisy_config = NoisyEstimatorConfig(shots=shots)

    e_raw = noisy_estimate(
        transpiled=transpiled_circuit,
        observable=H_mapped,
        backend=backend,
        config=noisy_config,
    )

    # _job: On hardware, the RuntimeJobV2 from noisy_estimate's underlying
    # EstimatorV2 execution. Currently None for FakeTorino (noisy_estimate
    # returns only the float energy). When full hardware path is wired in
    # task 5.1 completion, this will carry the real job for metrics capture.
    return {
        "e_raw": e_raw,
        "e_mitigated": None,
        "shots": shots,
        "_job": None,
    }


def _execute_gate_folding(
    config: BenchmarkConfig,
    transpiled_circuit: QuantumCircuit,
    H_mapped: SparsePauliOp,
    backend: Any,
    shots: int = 16384,
) -> dict[str, Any]:
    """Execute via gate-folding ZNE.

    Used by: C3_full_gf, C9_gnn_qem.

    Invokes run_gate_folding_zne() with noise factors from config. Uses
    the provided shots for the NoisyEstimatorConfig. Handles None noise_factors
    by falling back to the default (1, 3, 5).

    Parameters
    ----------
    config : BenchmarkConfig
        Configuration with zne_noise_factors.
    transpiled_circuit : QuantumCircuit
        Already-transpiled ISA circuit.
    H_mapped : SparsePauliOp
        Layout-mapped observable.
    backend : BackendV2
        Noisy backend.
    shots : int
        Number of shots per noise factor estimation.

    Returns
    -------
    dict
        Execution result with "e_mitigated", "e_raw", "zne_r2", "shots" keys.
    """
    # Resolve noise factors: round to nearest odd integer for gate-folding
    raw_factors = config.zne_noise_factors or [1, 3, 5]
    noise_factors_int = tuple(
        max(1, int(round(f)) | 1)  # Round and ensure odd (set LSB)
        for f in raw_factors
    )
    # Deduplicate while preserving order
    seen: set[int] = set()
    noise_factors_int = tuple(x for x in noise_factors_int if x not in seen and not seen.add(x))

    noisy_config = NoisyEstimatorConfig(shots=shots)

    gf_result = run_gate_folding_zne(
        transpiled_circuit=transpiled_circuit,
        observable=H_mapped,
        backend=backend,
        config=noisy_config,
        noise_factors=noise_factors_int,
    )

    return {
        "e_mitigated": gf_result.extrapolated_value,
        "e_raw": gf_result.measured_values[0],
        "zne_r2": gf_result.r_squared,
        "shots": noisy_config.shots * len(noise_factors_int),
        "_job": None,  # Hardware job captured by run_gate_folding_zne internally
    }


def _execute_pea(
    config: BenchmarkConfig,
    transpiled_circuit: QuantumCircuit,
    H_mapped: SparsePauliOp,
    backend: Any,
    shots: int = 16384,
) -> dict[str, Any]:
    """Execute via PEA-based ZNE (Probabilistic Error Amplification).

    Used by: C4, C5, C6, C7, C8, C10, C15, C16.

    Invokes run_pea_zne() with noise factors from config. The estimation
    shots come from the CLI --shots parameter. PEA noise learning uses
    pea_shots_per_randomization from config for model fitting only.

    Parameters
    ----------
    config : BenchmarkConfig
        Configuration with PEA parameters (num_randomizations, shots_per).
    transpiled_circuit : QuantumCircuit
        Already-transpiled ISA circuit.
    H_mapped : SparsePauliOp
        Layout-mapped observable.
    backend : BackendV2
        Noisy backend.
    shots : int
        Number of shots per noise factor estimation (from CLI --shots).

    Returns
    -------
    dict
        Execution result with "e_mitigated", "e_raw", "zne_r2",
        "noise_learning_time_s", "shots" keys.
    """
    # Estimation shots from CLI (not pea_shots_per_randomization which is for learning)
    noisy_config = NoisyEstimatorConfig(shots=shots)

    # Resolve noise factors from config, default (1, 3, 5) for PEA
    raw_factors = config.zne_noise_factors or [1, 3, 5]
    noise_factors = tuple(float(f) for f in raw_factors)

    t0 = time.time()
    pea_result = run_pea_zne(
        transpiled_circuit=transpiled_circuit,
        observable=H_mapped,
        backend=backend,
        config=noisy_config,
        noise_factors=noise_factors,
    )
    noise_learning_time_s = time.time() - t0

    # Total shots: estimation shots × number of noise factors
    total_shots = shots * len(noise_factors)

    return {
        "e_mitigated": pea_result.extrapolated_value,
        "e_raw": pea_result.measured_values[0],
        "zne_r2": pea_result.r_squared,
        "noise_learning_time_s": getattr(
            pea_result, "noise_learning_time_s", noise_learning_time_s
        ),
        "shots": total_shots,
        "_job": None,  # Hardware job captured by run_pea_zne internally
    }


def _execute_mitiq_zne(
    config: BenchmarkConfig,
    transpiled_circuit: QuantumCircuit,
    H_mapped: SparsePauliOp,
    backend: Any,
    shots: int = 16384,
) -> dict[str, Any]:
    """Execute via Mitiq ZNE (random gate folding).

    Used by: C11_mitiq_zne.
    Note: optimization_level=0 is enforced by Mitiq executor internally.

    IMPORTANT: Despite parameter names (kept for interface consistency),
    this function should receive the LOGICAL (pre-transpilation) circuit
    and the UNMAPPED observable. The Mitiq executor (make_mitiq_executor)
    handles transpilation and layout mapping internally. Passing already-
    transpiled/mapped objects causes "Number of qargs does not match" errors.

    Parameters
    ----------
    config : BenchmarkConfig
        Configuration with zne_noise_factors.
    transpiled_circuit : QuantumCircuit
        Logical circuit (pre-transpilation, 10-qubit). Mitiq executor
        transpiles internally with optimization_level=0.
    H_mapped : SparsePauliOp
        Logical observable (unmapped, 10-qubit). Mitiq executor maps
        to physical layout after internal transpilation.
    backend : BackendV2
        Noisy backend.
    shots : int
        Number of shots per noise factor estimation (from CLI --shots).

    Returns
    -------
    dict
        Execution result with "e_mitigated", "e_raw", "zne_r2" keys.
    """
    scale_factors = tuple(config.zne_noise_factors or [1.0, 2.0, 3.0])
    noisy_config = NoisyEstimatorConfig(shots=shots, optimization_level=0)

    zne_result = run_mitiq_zne(
        circuit=transpiled_circuit,
        observable=H_mapped,
        backend=backend,
        config=noisy_config,
        scale_factors=scale_factors,
    )

    # Extract raw (unmitigated) energy: first measured value at scale_factor=1.0
    e_raw = (
        zne_result.measured_values[0]
        if zne_result.measured_values
        else zne_result.extrapolated_value
    )

    return {
        "e_mitigated": zne_result.extrapolated_value,
        "e_raw": e_raw,
        "zne_r2": zne_result.r_squared,
        "shots": noisy_config.shots * len(scale_factors),
        "_job": None,  # Mitiq manages jobs internally
    }


def _execute_mitiq_cdr(
    config: BenchmarkConfig,
    transpiled_circuit: QuantumCircuit,
    H_mapped: SparsePauliOp,
    backend: Any,
    shots: int = 16384,
) -> dict[str, Any]:
    """Execute via Mitiq CDR (Clifford Data Regression).

    Used by: C12_mitiq_cdr, C14_dd_mitiq_cdr, C17_aqc_mitiq_cdr.
    Note: optimization_level=0 is enforced by Mitiq executor internally.

    IMPORTANT: Despite parameter names (kept for interface consistency),
    this function should receive the LOGICAL (pre-transpilation) circuit
    and the UNMAPPED observable. The Mitiq executor (make_mitiq_executor)
    handles transpilation and layout mapping internally. Passing already-
    transpiled/mapped objects causes "Number of qargs does not match" errors.

    Parameters
    ----------
    config : BenchmarkConfig
        Configuration.
    transpiled_circuit : QuantumCircuit
        Logical circuit (pre-transpilation, 10-qubit). Mitiq executor
        transpiles internally with optimization_level=0.
    H_mapped : SparsePauliOp
        Logical observable (unmapped, 10-qubit). Mitiq executor maps
        to physical layout after internal transpilation.
    backend : BackendV2
        Noisy backend.
    shots : int
        Number of shots per circuit execution (from CLI --shots).

    Returns
    -------
    dict
        Execution result with "e_mitigated", "e_raw" keys.
    """
    noisy_config = NoisyEstimatorConfig(shots=shots, optimization_level=0)

    cdr_result = run_mitiq_cdr(
        circuit=transpiled_circuit,
        observable=H_mapped,
        backend=backend,
        config=noisy_config,
        n_training_circuits=10,
    )

    return {
        "e_mitigated": cdr_result.mitigated_value,
        "e_raw": cdr_result.raw_value,
        "zne_r2": None,  # CDR does not produce R²
        "shots": noisy_config.shots * 21,  # ~2N+1 executions
        "_job": None,  # Mitiq manages jobs internally
    }


def _execute_mitiq_ddd(
    config: BenchmarkConfig,
    transpiled_circuit: QuantumCircuit,
    H_mapped: SparsePauliOp,
    backend: Any,
    shots: int = 16384,
) -> dict[str, Any]:
    """Execute via Mitiq DDD+ZNE composition.

    Used by: C13_mitiq_ddd_zne.
    Note: optimization_level=0 is enforced by Mitiq executor internally.

    IMPORTANT: Despite parameter names (kept for interface consistency),
    this function should receive the LOGICAL (pre-transpilation) circuit
    and the UNMAPPED observable. The Mitiq executor (make_mitiq_executor)
    handles transpilation and layout mapping internally. Passing already-
    transpiled/mapped objects causes "Number of qargs does not match" errors.

    Parameters
    ----------
    config : BenchmarkConfig
        Configuration with zne_noise_factors.
    transpiled_circuit : QuantumCircuit
        Logical circuit (pre-transpilation, 10-qubit). Mitiq executor
        transpiles internally with optimization_level=0.
    H_mapped : SparsePauliOp
        Logical observable (unmapped, 10-qubit). Mitiq executor maps
        to physical layout after internal transpilation.
    backend : BackendV2
        Noisy backend.
    shots : int
        Number of shots per noise factor estimation (from CLI --shots).

    Returns
    -------
    dict
        Execution result with "e_mitigated", "e_raw", "zne_r2" keys.
    """
    scale_factors = tuple(config.zne_noise_factors or [1.0, 2.0, 3.0])
    noisy_config = NoisyEstimatorConfig(shots=shots, optimization_level=0)

    ddd_result = run_mitiq_ddd_zne(
        circuit=transpiled_circuit,
        observable=H_mapped,
        backend=backend,
        config=noisy_config,
        ddd_rule="xx",
        scale_factors=scale_factors,
    )

    # Extract raw (unmitigated) energy: first measured value at scale_factor=1.0
    e_raw = (
        ddd_result.measured_values[0]
        if ddd_result.measured_values
        else ddd_result.extrapolated_value
    )

    return {
        "e_mitigated": ddd_result.extrapolated_value,
        "e_raw": e_raw,
        "zne_r2": ddd_result.r_squared,
        "shots": noisy_config.shots * len(scale_factors),
        "_job": None,  # Mitiq manages jobs internally
    }


# ---------------------------------------------------------------------------
# Hardware Runtime execution (IBM QPU via qiskit_ibm_runtime.EstimatorV2)
# ---------------------------------------------------------------------------


def _execute_hardware_batched(
    jobs_spec: list[tuple[BenchmarkConfig, QuantumCircuit, SparsePauliOp, float]],
    backend: Any,
    shots: int,
) -> list[dict[str, Any]]:
    """Execute multiple (config, circuit, observable) tuples using the proven
    submit_all_then_collect pattern from HardwareBackend (submission.py).

    Groups jobs by config (same options per group), then for each group uses
    the exact same Batch submission pattern proven in run_ibm_deployment.py.

    Parameters
    ----------
    jobs_spec : list of (config, transpiled_circuit, H_mapped, h_value)
        Each tuple defines one execution unit.
    backend : BackendV2
        Real IBM backend.
    shots : int
        Number of shots per circuit.

    Returns
    -------
    list[dict]
        Results in the same order as jobs_spec.
    """
    from qiskit_ibm_runtime import Batch, EstimatorV2

    if not jobs_spec:
        return []

    # Group jobs by config_id (same options per group → one estimator per group)
    from collections import defaultdict

    groups: dict[str, list[tuple[int, QuantumCircuit, SparsePauliOp]]] = defaultdict(list)
    for idx, (config, transpiled, H_mapped, h_val) in enumerate(jobs_spec):
        groups[config.config_id].append((idx, transpiled, H_mapped))

    results: list[dict[str, Any]] = [{}] * len(jobs_spec)
    total_submitted = 0

    print(f"  [BATCH] {len(jobs_spec)} jobs in {len(groups)} config groups", flush=True)

    # Submit all groups in ONE Batch session (one queue wait)
    try:
        with Batch(backend=backend) as batch:
            for config_id, group_jobs in groups.items():
                config = BENCHMARK_CONFIGS[config_id]

                # Create one estimator per config group (same options for all h in this group)
                est = EstimatorV2(mode=batch)
                est.options.default_precision = 1.0 / np.sqrt(shots)

                if config.dd_enabled:
                    est.options.dynamical_decoupling.enable = True
                    est.options.dynamical_decoupling.sequence_type = config.dd_sequence or "XpXm"
                if config.twirling_num_randomizations:
                    est.options.twirling.enable_gates = True
                    est.options.twirling.num_randomizations = config.twirling_num_randomizations
                if config.zne_method == "pea":
                    est.options.resilience.zne_mitigation = True
                    est.options.resilience.zne.amplifier = "pea"
                    nf = config.zne_noise_factors or [1, 3, 5]
                    est.options.resilience.zne.noise_factors = tuple(nf)
                    est.options.resilience.zne.extrapolator = "linear"
                elif config.zne_method == "gf":
                    est.options.resilience.zne_mitigation = True
                    est.options.resilience.zne.amplifier = "gate_folding"
                    nf = config.zne_noise_factors or [1, 3, 5]
                    est.options.resilience.zne.noise_factors = tuple(int(f) for f in nf)
                    est.options.resilience.zne.extrapolator = "linear"

                # Submit all h-points for this config through one estimator
                for orig_idx, transpiled, H_mapped in group_jobs:
                    try:
                        job = est.run([(transpiled, H_mapped)])
                        results[orig_idx] = {"_pending_job": job, "_config": config}
                        total_submitted += 1
                    except Exception as e:
                        results[orig_idx] = {
                            "e_raw": None,
                            "e_mitigated": None,
                            "zne_r2": None,
                            "shots": shots,
                            "_job": None,
                            "error": f"submit_failed: {e}",
                        }

                print(f"    [{config_id}] {len(group_jobs)} jobs submitted", flush=True)

    except Exception as e:
        logger.error(f"Batch creation failed: {e}")
        return [{"error": str(e), "_job": None} for _ in jobs_spec]

    # Phase 2: Collect all results
    print(f"  [BATCH] Collecting {total_submitted} results...", flush=True)
    for idx in range(len(results)):
        entry = results[idx]
        if "_pending_job" not in entry:
            continue  # Already has error or was never submitted

        job = entry["_pending_job"]
        config = entry["_config"]

        try:
            if hasattr(job, "wait_for_final_state"):
                job.wait_for_final_state(timeout=3600)
            result = job.result()
            energy = float(result[0].data.evs)

            qpu_seconds = None
            try:
                metrics = job.metrics()
                if isinstance(metrics, dict):
                    qpu_seconds = metrics.get("usage", {}).get("quantum_seconds")
            except Exception:
                pass

            if config.zne_method in ("pea", "gf"):
                results[idx] = {
                    "e_mitigated": energy,
                    "e_raw": energy,
                    "zne_r2": None,
                    "shots": shots,
                    "qpu_seconds": qpu_seconds,
                    "_job": job,
                }
            else:
                results[idx] = {
                    "e_raw": energy,
                    "e_mitigated": None,
                    "zne_r2": None,
                    "shots": shots,
                    "qpu_seconds": qpu_seconds,
                    "_job": job,
                }
            h_val = jobs_spec[idx][3]
            print(f"    [{config.config_id}] h={h_val:.2f} E={energy:.4f} ✓", flush=True)

        except Exception as e:
            results[idx] = {
                "e_raw": None,
                "e_mitigated": None,
                "zne_r2": None,
                "shots": shots,
                "_job": job,
                "error": str(e),
            }
            h_val = jobs_spec[idx][3]
            print(f"    [{config.config_id}] h={h_val:.2f} FAILED: {e}", flush=True)

    succeeded = sum(
        1 for r in results if r.get("e_raw") is not None or r.get("e_mitigated") is not None
    )
    print(f"  [BATCH] Done — {succeeded}/{len(jobs_spec)} succeeded")
    return results


def _execute_hardware_runtime(
    config: BenchmarkConfig,
    transpiled_circuit: QuantumCircuit,
    H_mapped: SparsePauliOp,
    backend: Any,
    shots: int,
) -> dict[str, Any]:
    """Execute on real IBM hardware via qiskit_ibm_runtime.EstimatorV2.

    This is the hardware-mode equivalent of _execute_raw/_execute_gf/_execute_pea.
    Uses IBM Runtime's server-side ZNE (configured via estimator options) instead
    of the local BackendEstimatorV2 which calls backend.run() (deprecated on QPU).

    The ZNE amplifier is set based on config.zne_method:
      - None: no ZNE (raw measurement with DD/twirling if configured)
      - "gf": gate_folding amplifier
      - "pea": pea amplifier (Probabilistic Error Amplification)

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark config with mitigation settings.
    transpiled_circuit : QuantumCircuit
        Already-transpiled ISA circuit.
    H_mapped : SparsePauliOp
        Layout-mapped observable.
    backend : BackendV2
        Real IBM backend from QiskitRuntimeService.
    shots : int
        Number of shots per circuit.

    Returns
    -------
    dict
        Execution result with "e_raw" or "e_mitigated", "zne_r2", "shots", "_job".
    """
    from qiskit_ibm_runtime import Batch, EstimatorV2

    # Validate we have a real circuit before opening a Batch session
    if transpiled_circuit is None or transpiled_circuit.num_qubits == 0:
        raise ValueError(
            "Cannot submit to hardware: transpiled_circuit is None or empty. "
            "This prevents opening an empty Batch session on IBM Runtime."
        )

    # Build estimator options based on config
    with Batch(backend=backend) as batch:
        estimator = EstimatorV2(mode=batch)

        # Set precision (shots control)
        estimator.options.default_precision = 1.0 / np.sqrt(shots)

        # DD (always enable on hardware — free, helps coherent noise)
        if config.dd_enabled:
            estimator.options.dynamical_decoupling.enable = True
            estimator.options.dynamical_decoupling.sequence_type = config.dd_sequence or "XpXm"

        # Twirling
        if config.twirling_num_randomizations:
            estimator.options.twirling.enable_gates = True
            estimator.options.twirling.num_randomizations = config.twirling_num_randomizations

        # ZNE configuration
        if config.zne_method == "pea":
            estimator.options.resilience.zne_mitigation = True
            estimator.options.resilience.zne.amplifier = "pea"
            noise_factors = config.zne_noise_factors or [1, 3, 5]
            estimator.options.resilience.zne.noise_factors = tuple(noise_factors)
            estimator.options.resilience.zne.extrapolator = "linear"
        elif config.zne_method == "gf":
            estimator.options.resilience.zne_mitigation = True
            estimator.options.resilience.zne.amplifier = "gate_folding"
            noise_factors = config.zne_noise_factors or [1, 3, 5]
            estimator.options.resilience.zne.noise_factors = tuple(int(f) for f in noise_factors)
            estimator.options.resilience.zne.extrapolator = "linear"
        # else: no ZNE (raw measurement)

        # Submit the PUB
        job = estimator.run([(transpiled_circuit, H_mapped)])

    # Collect result (outside batch context)
    if hasattr(job, "wait_for_final_state"):
        job.wait_for_final_state(timeout=3600)
    result = job.result()
    energy = float(result[0].data.evs)

    # Extract QPU metrics
    qpu_seconds = None
    try:
        metrics = job.metrics()
        if isinstance(metrics, dict):
            qpu_seconds = metrics.get("usage", {}).get("quantum_seconds")
    except Exception:
        pass

    # Build result dict matching the interface expected by the caller
    if config.zne_method in ("pea", "gf"):
        return {
            "e_mitigated": energy,
            "e_raw": energy,  # IBM returns already-mitigated value
            "zne_r2": None,  # Server-side ZNE doesn't expose R²
            "shots": shots,
            "qpu_seconds": qpu_seconds,
            "_job": job,
        }
    else:
        # Raw (no ZNE)
        return {
            "e_raw": energy,
            "e_mitigated": None,
            "zne_r2": None,
            "shots": shots,
            "qpu_seconds": qpu_seconds,
            "_job": job,
        }


def route_execution(
    config: BenchmarkConfig,
    transpiled_circuit: QuantumCircuit,
    H_mapped: SparsePauliOp,
    backend: Any,
    shots: int,
    h_value: float,
    e_exact: float,
    gap: float,
    logical_circuit: QuantumCircuit | None = None,
    H_logical: SparsePauliOp | None = None,
    mode: str = "fake_backend",
) -> dict[str, Any]:
    """Dispatch execution to the correct mitigation function.

    Routing rules (based on config.zne_method):
      None              → _execute_raw (direct estimation, DD/Tw/TREX only)
      "gf"              → _execute_gate_folding (run_gate_folding_zne)
      "pea"             → _execute_pea (run_pea_zne)
      "mitiq_zne"       → _execute_mitiq_zne (Mitiq random-folding ZNE)
      "mitiq_cdr"       → _execute_mitiq_cdr (Mitiq Clifford Data Regression)
      "mitiq_ddd_zne"   → _execute_mitiq_ddd (Mitiq DDD+ZNE composition)

    For mode="hardware", the raw/gf/pea paths use qiskit_ibm_runtime.EstimatorV2
    (IBM Runtime Primitives V2) instead of qiskit.primitives.BackendEstimatorV2,
    which requires backend.run() that is no longer supported on real QPUs.

    Special case: C10_kitchen_sink (gnn_qem_enabled=True) executes PEA-ZNE
    first, then applies GNNQEMCorrector as post-processing on the mitigated
    energy. Note: per validated findings, GNN-QEM post-PEA may regress;
    this config exists to benchmark that interaction.

    Parameters
    ----------
    config : BenchmarkConfig
        The benchmark configuration determining the mitigation method.
    transpiled_circuit : QuantumCircuit
        Already-transpiled, parameter-bound ISA circuit.
    H_mapped : SparsePauliOp
        Observable mapped to the physical qubit layout (133-qubit for
        FakeTorino). Used by IBM-native executors (raw, gf, pea).
    backend : Any
        Noisy backend (FakeTorino or IBM hardware).
    shots : int
        Number of shots for the estimation.
    h_value : float
        Transverse field value (needed for GNN-QEM sample construction).
    e_exact : float
        Exact ground state energy from ClassicalSolver.
    gap : float
        Energy gap from ClassicalSolver.
    logical_circuit : QuantumCircuit | None, optional
        Pre-transpilation logical circuit (10-qubit). Required for Mitiq
        executors which handle transpilation internally. Falls back to
        transpiled_circuit if not provided (backward compat).
    H_logical : SparsePauliOp | None, optional
        Unmapped logical observable (10-qubit). Required for Mitiq executors
        which map the observable after internal transpilation. Falls back to
        H_mapped if not provided (backward compat).

    Returns
    -------
    dict[str, Any]
        Execution result dict containing at minimum:
        - "e_mitigated": float | None
        - "e_raw": float
        - "zne_r2": float | None
        - "shots": int

    Raises
    ------
    ValueError
        If config.zne_method is not recognized.
    """
    # ── Hardware mode: use IBM Runtime EstimatorV2 (not BackendEstimatorV2) ──
    if mode == "hardware" and config.zne_method in (None, "gf", "pea"):
        result = _execute_hardware_runtime(config, transpiled_circuit, H_mapped, backend, shots)
    else:
        # Simulation mode (FakeTorino) or Mitiq paths
        match config.zne_method:
            case None:
                result = _execute_raw(transpiled_circuit, H_mapped, backend, shots)
            case "gf":
                result = _execute_gate_folding(config, transpiled_circuit, H_mapped, backend, shots)
            case "pea":
                result = _execute_pea(config, transpiled_circuit, H_mapped, backend, shots)
            case "mitiq_zne":
                result = _execute_mitiq_zne(
                    config,
                    logical_circuit or transpiled_circuit,
                    H_logical or H_mapped,
                    backend,
                    shots,
                )
            case "mitiq_cdr":
                result = _execute_mitiq_cdr(
                    config,
                    logical_circuit or transpiled_circuit,
                    H_logical or H_mapped,
                    backend,
                    shots,
                )
            case "mitiq_ddd_zne":
                result = _execute_mitiq_ddd(
                    config,
                    logical_circuit or transpiled_circuit,
                    H_logical or H_mapped,
                    backend,
                    shots,
                )
            case _:
                raise ValueError(f"Unknown zne_method: {config.zne_method}")

    # Special case: GNN-QEM post-processing (C10_kitchen_sink)
    if config.gnn_qem_enabled and result.get("e_mitigated") is not None:
        result = _apply_gnn_qem_postprocessing(result, config, transpiled_circuit, h_value, e_exact)

    return result


def _apply_gnn_qem_postprocessing(
    result: dict[str, Any],
    config: BenchmarkConfig,
    transpiled_circuit: QuantumCircuit,
    h_value: float,
    e_exact: float,
) -> dict[str, Any]:
    """Apply GNN-QEM correction as post-processing on mitigated energy.

    This implements the C10_kitchen_sink special case: PEA-ZNE first,
    then GNN-QEM correction on the result. Per validated findings
    (2026-06-05), GNN-QEM trained on large errors may over-correct
    post-PEA residuals. This exists as a benchmark data point.

    The function uses lazy imports to avoid heavy torch dependency at
    module load time.

    Parameters
    ----------
    result : dict
        Execution result from PEA-ZNE (must have "e_mitigated").
    config : BenchmarkConfig
        Benchmark configuration.
    transpiled_circuit : QuantumCircuit
        Transpiled circuit (for extracting n_2q_gates).
    h_value : float
        Transverse field value.
    e_exact : float
        Exact energy for the QEMSample.

    Returns
    -------
    dict
        Updated result with "e_mitigated" replaced by GNN-QEM corrected
        value. Original PEA result preserved in "e_pea_before_gnn_qem".
    """
    try:
        from qmbp_simulation.predictors.gnn_qem import (
            GNNQEMCorrector,
            QEMSample,
            load_qem_checkpoint,
        )
        from qmbp_simulation.predictors.gnn_qem import (
            correct_energy as gnn_correct,
        )
    except ImportError:
        # torch not available — skip GNN-QEM, return result unchanged
        result["gnn_qem_applied"] = False
        result["gnn_qem_error"] = "torch not available"
        return result

    # Preserve original PEA result before GNN-QEM modification
    result["e_pea_before_gnn_qem"] = result["e_mitigated"]

    # Count 2Q gates from transpiled circuit
    n_2q = sum(1 for inst in transpiled_circuit.data if inst.operation.num_qubits == 2)

    # Build QEMSample for inference
    qem_sample = QEMSample(
        noisy_energy=result["e_mitigated"],
        exact_energy=e_exact,
        h_value=h_value,
        n_2q_gates=n_2q,
        ces=0.0,  # CES not available at this point
        topology="heavy_hex",
        n_qubits=10,
    )

    # Load pre-trained GNN-QEM model
    checkpoint_path = Path("results/gnn_qem/checkpoints/best_model.pt")
    if not checkpoint_path.exists():
        result["gnn_qem_applied"] = False
        result["gnn_qem_error"] = f"Checkpoint not found: {checkpoint_path}"
        return result

    try:
        model = load_qem_checkpoint(checkpoint_path)
        correction_result = gnn_correct(model, qem_sample)
        result["e_mitigated"] = correction_result.corrected_energy
        result["gnn_qem_applied"] = correction_result.applied
        result["gnn_qem_confidence"] = correction_result.confidence
        result["gnn_qem_delta_e"] = correction_result.predicted_delta_e
    except Exception as exc:
        # GNN-QEM failure should not abort the benchmark
        result["gnn_qem_applied"] = False
        result["gnn_qem_error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# CLI — argparse, config resolution, export, and main loop
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the mitigation benchmark runner.

    Arguments:
        --mode: Execution backend ("fake_backend" or "hardware").
        --configs: CSV of config shortnames to run (prefix-match).
        --h-values: CSV of transverse field values.
        --shots: Number of shots per estimation.
        --seed: Random seed for reproducibility.
        --priority: CSV of priority levels (P0-P4) to filter configs.
        --export-configs: Export all configs as JSON and exit.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Mitigation Benchmark Runner — 19 configs systematic evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["fake_backend", "hardware"],
        default="fake_backend",
        help="Execution mode (default: fake_backend)",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help="CSV of config shortnames to run (e.g. C0,C5,C12). Prefix-match.",
    )
    parser.add_argument(
        "--h-values",
        type=str,
        default="3.25,3.5,3.75,4.0",
        help="CSV of transverse field h-values (default: 3.25,3.5,3.75,4.0)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=16384,
        help="Number of shots per estimation (default: 16384)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--priority",
        type=str,
        default=None,
        help="CSV of priority levels to filter (e.g. P0,P1)",
    )
    parser.add_argument(
        "--export-configs",
        action="store_true",
        help="Export all BENCHMARK_CONFIGS as individual JSONs and exit",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Reserved for future parallel workers. Currently use multiple terminals "
        "with --configs filter for manual parallelism (manifest file-lock safe).",
    )
    parser.add_argument(
        "--warm-start",
        type=str,
        default=None,
        help="Path to MPNN checkpoint for warm-start parameters "
        "(default: quick VQE-computed θ_opt, cached per h-value)",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use kappa (landscape curvature) to prioritize configs adaptively. "
        "High-κ h-values get full mitigation stack; low-κ get DD-only.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Target a specific IBM backend (e.g. ibm_kingston, ibm_boston). "
        "If not set, uses service.least_busy() which may pick a degraded chip.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Submit all configs×h-points in a single IBM Batch session "
        "(reduces queue wait from N× to 1×). Hardware mode only.",
    )
    return parser.parse_args()


def resolve_configs(args: argparse.Namespace) -> list[str]:
    """Resolve which config IDs to run based on --priority and --configs.

    Resolution order:
      1. Start with ALL config IDs from BENCHMARK_CONFIGS.
      2. If --priority is provided, filter to configs whose priority matches.
      3. If --configs is provided, filter by shortname prefix-match.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments with `.priority` and `.configs` fields.

    Returns
    -------
    list[str]
        Resolved list of full config_id strings to execute.

    Raises
    ------
    ValueError
        If a shortname from --configs matches no config in the filtered list.
    """
    all_configs = list(BENCHMARK_CONFIGS.keys())

    # Step 1: priority filter (intersection)
    if args.priority is not None:
        levels = {
            int(p.strip().replace("P", "").replace("p", "")) for p in args.priority.split(",")
        }
        all_configs = [c for c in all_configs if BENCHMARK_CONFIGS[c].priority in levels]

    # Step 2: shortname prefix filter
    if args.configs is not None:
        shortnames = [s.strip() for s in args.configs.split(",")]
        resolved: list[str] = []
        for short in shortnames:
            matches = [c for c in all_configs if c.startswith(short)]
            if not matches:
                raise ValueError(f"No config matches shortname '{short}'")
            resolved.extend(matches)
        return resolved

    return all_configs


def export_configs(output_dir: Path) -> None:
    """Serialize all BENCHMARK_CONFIGS to individual JSON files.

    Creates one JSON file per config in output_dir, named {config_id}.json.
    Uses json_dump from qmbp_simulation.utils.helpers for serialization.

    Parameters
    ----------
    output_dir : Path
        Directory to write config JSONs (created if it doesn't exist).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for config_id, config in BENCHMARK_CONFIGS.items():
        config_dict = dataclasses.asdict(config)
        json_dump(config_dict, output_dir / f"{config_id}.json")

    print(f"Exported {len(BENCHMARK_CONFIGS)} configs to {output_dir}/")


# ---------------------------------------------------------------------------
# ClassicalSolver cache (keyed by h_value)
# ---------------------------------------------------------------------------

_classical_cache: dict[float, tuple[float, float]] = {}

# Circuit cache: h_value → bound QuantumCircuit (pre-transpilation)
_circuit_cache: dict[float, QuantumCircuit] = {}

# Transpile cache: (h_value, optimization_level, is_aqc) → transpiled QuantumCircuit
_transpile_cache: dict[tuple[float, int, bool], QuantumCircuit] = {}

# MPNN warm-start model (loaded once on first use)
_mpnn_model: Any = None


def _get_warm_start_params(h_value: float, warm_start_path: str | None) -> np.ndarray | None:
    """Get warm-start parameters from MPNN or None if no checkpoint.

    Parameters
    ----------
    h_value : float
        Transverse field value for MPNN prediction.
    warm_start_path : str | None
        Path to MPNN checkpoint. If None, returns None (use default fallback).

    Returns
    -------
    np.ndarray | None
        Predicted parameters from MPNN, or None if unavailable.
    """
    global _mpnn_model
    if warm_start_path is None:
        return None

    if _mpnn_model is None:
        try:
            from qmbp_simulation.predictors import load_mpnn_checkpoint

            _mpnn_model = load_mpnn_checkpoint(warm_start_path)
            logger.info(f"Loaded MPNN warm-start from {warm_start_path}")
        except Exception as e:
            logger.warning(f"Failed to load MPNN: {e}. Using fallback θ.")
            return None

    # Build graph in the correct format (same as prepare_mpnn_predictions)
    try:
        import torch
        from torch_geometric.data import Data

        lattice_h = make_lattice(_TOPOLOGY, _N_QUBITS, J=1.0, h=h_value)
        edge_index_np, coord = HamiltonianBuilder().build_graph_data(lattice_h)
        h_feat = np.full(_N_QUBITS, float(h_value))
        x = torch.tensor(
            np.stack([h_feat, coord.astype(float)], axis=1),
            dtype=torch.float32,
        )
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        graph = Data(x=x, edge_index=edge_index)

        _mpnn_model.eval()
        with torch.no_grad():
            theta_pred = _mpnn_model(graph).numpy().flatten()
        return theta_pred
    except Exception as e:
        logger.warning(f"MPNN prediction failed for h={h_value}: {e}")
        return None


def _get_exact_energy(h_value: float) -> tuple[float, float]:
    """Get e_exact and gap via ClassicalSolver, cached per h_value.

    Uses exact diagonalization for TFIM N=10 heavy_hex. NOT VQE noiseless.
    The cache ensures that all configs sharing the same h_value get
    bit-exact identical e_exact/gap values.

    Parameters
    ----------
    h_value : float
        Transverse field strength.

    Returns
    -------
    tuple[float, float]
        (ground_energy, gap) from exact diagonalization.
    """
    if h_value not in _classical_cache:
        lattice = make_lattice("heavy_hex", 10, J=1.0, h=h_value)
        H = HamiltonianBuilder().build(lattice)
        solver = ClassicalSolver()
        result = solver.solve(H, lattice)
        _classical_cache[h_value] = (result.ground_energy, result.gap)
    return _classical_cache[h_value]


# ---------------------------------------------------------------------------
# Backend helper
# ---------------------------------------------------------------------------


def _validate_hardware_credentials() -> tuple[str, str]:
    """Validate IBM hardware credentials early (fail-fast before execution).

    Called at the start of main() when --mode hardware to abort immediately
    with a clear error message if credentials are missing. This avoids wasting
    time on circuit construction / ClassicalSolver calls before discovering
    that the hardware path cannot proceed.

    Follows the same pattern as check_credentials() in run_ibm_deployment.py.

    Returns
    -------
    tuple[str, str]
        (ibm_key, instance_crn) if both are set.

    Raises
    ------
    SystemExit
        If IBM_KEY or IBM_INSTANCE_CRN are not set. Prints a clear error
        message with the exact variable names needed and exits with code 1.
    """
    import os
    import sys

    ibm_key = os.environ.get("IBM_KEY")
    instance_crn = os.environ.get("IBM_INSTANCE_CRN")

    missing: list[str] = []
    if not ibm_key:
        missing.append("IBM_KEY")
    if not instance_crn:
        missing.append("IBM_INSTANCE_CRN")

    if missing:
        print(
            "\n╔══════════════════════════════════════════════════════════════════╗",
            file=sys.stderr,
        )
        print(
            "║  ERROR: Hardware mode requires IBM credentials                  ║",
            file=sys.stderr,
        )
        print(
            "╠══════════════════════════════════════════════════════════════════╣",
            file=sys.stderr,
        )
        for var in missing:
            print(f"║  Missing: {var:<53}║", file=sys.stderr)
        print(
            "╠══════════════════════════════════════════════════════════════════╣",
            file=sys.stderr,
        )
        print(
            "║  Set the following environment variables:                       ║",
            file=sys.stderr,
        )
        print(
            "║    export IBM_KEY='your_ibm_quantum_api_token'                  ║",
            file=sys.stderr,
        )
        print(
            "║    export IBM_INSTANCE_CRN='crn:v1:bluemix:public:...'          ║",
            file=sys.stderr,
        )
        print(
            "╚══════════════════════════════════════════════════════════════════╝",
            file=sys.stderr,
        )
        sys.exit(1)

    return ibm_key, instance_crn


def _get_backend(mode: str) -> Any:
    """Get the appropriate backend for execution mode.

    For hardware mode, instantiates QiskitRuntimeService with credentials
    from environment variables (IBM_KEY, IBM_INSTANCE_CRN). Uses
    least_busy() to select a backend with ≥10 qubits.

    NOTE (hardware integration — patterns from run_ibm_deployment.py):
    - Mapomatic VF2 layout optimization: When real hardware runs are executed,
      the layout_optimizer module (src/qmbp_simulation/execution/hardware/
      layout_optimizer.py) should be used to select optimal qubit layouts via
      subgraph isomorphism, achieving ~6x lower CES than default BFS routing.
      Integration point: after backend selection, before transpilation.
    - TLS drift monitoring: For multi-h-point hardware runs, use
      take_calibration_snapshot() before execution and check_calibration_drift()
      between h-points. Abort if T1 drift > 20%. See Nature Comms 2025
      (arXiv:2407.02467). Integration point: in run_benchmark() loop,
      between h-value iterations when mode=="hardware".
    - Structured logging: Hardware runs should log backend name, calibration
      age, selected layout qubits, and QPU metrics per job. Currently captured
      via _collect_hardware_calibration() in the ResultEnvelope.

    Parameters
    ----------
    mode : str
        "fake_backend" or "hardware".

    Returns
    -------
    BackendV2
        FakeTorino for fake_backend, IBM Runtime backend for hardware.

    Raises
    ------
    RuntimeError
        If mode is "hardware" and IBM credentials are not configured.
    """
    if mode == "fake_backend":
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        return FakeTorino()
    elif mode == "hardware":
        import os

        ibm_key = os.environ.get("IBM_KEY")
        instance_crn = os.environ.get("IBM_INSTANCE_CRN")
        if not ibm_key or not instance_crn:
            raise RuntimeError(
                "Hardware mode requires IBM_KEY and IBM_INSTANCE_CRN "
                "environment variables to be set. Run with --mode fake_backend "
                "for simulation, or set:\n"
                "  export IBM_KEY='your_token'\n"
                "  export IBM_INSTANCE_CRN='crn:v1:bluemix:public:...'"
            )
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService(
            channel="ibm_quantum_platform", token=ibm_key, instance=instance_crn
        )
        # Use explicit backend if specified via --backend, else least_busy
        backend_name = os.environ.get("BENCHMARK_BACKEND")
        if backend_name:
            return service.backend(backend_name)
        return service.least_busy(min_num_qubits=10, operational=True)
    else:
        raise ValueError(f"Unknown execution mode: {mode!r}")


# ---------------------------------------------------------------------------
# Circuit construction helpers
# ---------------------------------------------------------------------------

# Constants for the benchmark circuit
_N_QUBITS = 10
_P_LAYERS = 1
_TOPOLOGY = "heavy_hex"


def _build_hva_circuit(
    h_value: float, warm_start_params: np.ndarray | None = None
) -> QuantumCircuit:
    """Build the HVA circuit for TFIM N=10, p=1, heavy_hex.

    Returns a parameterized circuit bound with the provided warm-start
    parameters or θ_opt from a quick VQE solve. Uses a module-level cache
    to avoid redundant construction.

    When no warm-start parameters are provided, runs a fast noiseless VQE
    (1 restart, 100 iterations) to find meaningful θ_opt. This ensures the
    circuit has non-trivial 2Q gates that survive transpilation at opt_level≥1.
    Using θ=zeros would cause RZZ(0)=Identity cancellation by the transpiler.

    Parameters
    ----------
    h_value : float
        Transverse field value (used for lattice construction).
    warm_start_params : np.ndarray | None
        Optional warm-start parameters from MPNN. If None, runs quick VQE
        to obtain meaningful θ_opt (deterministic, cached per h_value).

    Returns
    -------
    QuantumCircuit
        Bound (parameter-free) circuit ready for transpilation.
    """
    if h_value in _circuit_cache and warm_start_params is None:
        return _circuit_cache[h_value].copy()

    lattice = make_lattice(_TOPOLOGY, _N_QUBITS, J=1.0, h=h_value)
    builder = HVACircuitBuilder()
    circuit, theta = builder.create(_N_QUBITS, _P_LAYERS, lattice)

    if warm_start_params is not None:
        params = warm_start_params
    else:
        # Run VQE to get meaningful θ_opt (not zeros/random).
        # This ensures the circuit has non-trivial 2Q gates that survive
        # transpilation at opt_level≥1, producing valid transpilation metrics.
        # For h<2.0 (near criticality), use more restarts for better convergence.
        from qmbp_simulation import VQEConfig, VQEOptimizer
        from qmbp_simulation.execution import NoiselessBackend

        n_restarts = 5 if h_value < 2.0 else 1
        maxiter = 300 if h_value < 2.0 else 100
        vqe_config = VQEConfig(n_restarts=n_restarts, maxiter=maxiter)
        optimizer = VQEOptimizer(config=vqe_config, seed=42)
        backend = NoiselessBackend()
        H = HamiltonianBuilder().build(lattice)
        result = optimizer.optimize(
            H, circuit, np.random.default_rng(42).uniform(-0.01, 0.01, len(theta))
        )
        params = result.theta_opt

    bound_circuit = circuit.assign_parameters(params)

    if warm_start_params is None:
        _circuit_cache[h_value] = bound_circuit

    return bound_circuit.copy() if h_value in _circuit_cache else bound_circuit


def _build_aqc_circuit(h_value: float) -> tuple[QuantumCircuit, dict[str, Any] | None]:
    """Build an AQC-compressed circuit for TFIM N=10, heavy_hex.

    Uses the AQCCircuitCompressor to compress a p=2 HVA circuit down to
    p=1-equivalent gate count while retaining p=2 expressibility.
    Falls back to standard HVA p=1 if:
      - qiskit-addon-aqc-tensor is not installed
      - compression fidelity < 0.998

    Parameters
    ----------
    h_value : float
        Transverse field value.

    Returns
    -------
    tuple[QuantumCircuit, dict | None]
        (circuit, aqc_metrics) where aqc_metrics contains compression quality
        indicators, or None if AQC was not attempted (import failure).
    """
    _AQC_FIDELITY_THRESHOLD = 0.998
    _AQC_P_TARGET = 2  # Compress p=2 down to p=1-equivalent depth

    # Attempt lazy import of AQCCircuitCompressor
    try:
        from qmbp_simulation.circuits.aqc_compression import (
            AQCCircuitCompressor,
            AQCCompressionConfig,
        )
    except ImportError:
        # qiskit-addon-aqc-tensor not installed — fallback to HVA p=1
        circuit = _build_hva_circuit(h_value)
        aqc_metrics: dict[str, Any] = {
            "aqc_fidelity": 0.0,
            "aqc_n_2q_compressed": 0,
            "aqc_2q_reduction_pct": 0.0,
            "aqc_compression_time_s": 0.0,
            "aqc_fallback_triggered": True,
        }
        return circuit, aqc_metrics

    # Build the p=2 target circuit (the "deep" circuit to compress)
    lattice = make_lattice(_TOPOLOGY, _N_QUBITS, J=1.0, h=h_value)
    builder = HVACircuitBuilder()
    target_circuit, theta = builder.create(_N_QUBITS, _AQC_P_TARGET, lattice)

    # Run VQE to get meaningful θ_opt(p=2) for the target circuit.
    # AQC compresses THIS state — it must be near the ground state for the
    # benchmark comparison (ΔE/gap vs e_exact) to be valid.
    # For h<2.0, use more restarts (landscape harder near criticality).
    from qmbp_simulation import VQEConfig, VQEOptimizer
    from qmbp_simulation.execution import NoiselessBackend

    n_restarts = 5 if h_value < 2.0 else 1
    maxiter = 500 if h_value < 2.0 else 200
    vqe_config = VQEConfig(n_restarts=n_restarts, maxiter=maxiter)
    optimizer = VQEOptimizer(config=vqe_config, seed=42)
    vqe_backend = NoiselessBackend()
    H = HamiltonianBuilder().build(lattice)
    vqe_result = optimizer.optimize(
        H,
        target_circuit,
        np.random.default_rng(42).uniform(-0.01, 0.01, len(theta)),
    )
    params = vqe_result.theta_opt
    bound_target = target_circuit.assign_parameters(params)

    # Count original 2Q gates for reduction calculation
    n_2q_original = sum(1 for inst in bound_target.data if inst.operation.num_qubits == 2)

    # Run AQC compression
    t_start = time.time()
    try:
        config = AQCCompressionConfig(
            max_bond_dim=64,
            max_iterations=500,
            fidelity_threshold=_AQC_FIDELITY_THRESHOLD,
        )
        compressor = AQCCircuitCompressor(config=config)
        result = compressor.compress_circuit(bound_target, lattice)
        compression_time = time.time() - t_start

        # Check fidelity threshold
        if result.fidelity < _AQC_FIDELITY_THRESHOLD:
            # Fidelity insufficient — fall back to p=1 direct
            circuit = _build_hva_circuit(h_value)
            aqc_metrics = {
                "aqc_fidelity": result.fidelity,
                "aqc_n_2q_compressed": result.n_2q_compressed,
                "aqc_2q_reduction_pct": result.n_2q_reduction_pct,
                "aqc_compression_time_s": compression_time,
                "aqc_fallback_triggered": True,
            }
            return circuit, aqc_metrics

        # Compression successful — use compressed circuit
        aqc_metrics = {
            "aqc_fidelity": result.fidelity,
            "aqc_n_2q_compressed": result.n_2q_compressed,
            "aqc_2q_reduction_pct": result.n_2q_reduction_pct,
            "aqc_compression_time_s": compression_time,
            "aqc_fallback_triggered": False,
        }
        return result.compressed_circuit, aqc_metrics

    except Exception:
        # Any compression error — fall back gracefully to p=1
        compression_time = time.time() - t_start
        circuit = _build_hva_circuit(h_value)
        aqc_metrics = {
            "aqc_fidelity": 0.0,
            "aqc_n_2q_compressed": 0,
            "aqc_2q_reduction_pct": 0.0,
            "aqc_compression_time_s": compression_time,
            "aqc_fallback_triggered": True,
        }
        return circuit, aqc_metrics


def _compute_upper_bound(h_value: float) -> float:
    """Compute the energy upper bound for affine correction.

    For TFIM with N qubits, the trivial upper bound is max eigenvalue.
    Uses ClassicalSolver on the first excited state approximation:
    E_upper = N * max(J, h) as a conservative bound.

    Parameters
    ----------
    h_value : float
        Transverse field value.

    Returns
    -------
    float
        Upper bound on the energy spectrum.
    """
    # For TFIM N=10: E_upper = N * max(J=1, h)
    # This is a conservative bound (all spins anti-aligned with dominant field).
    return _N_QUBITS * max(1.0, h_value)


# ---------------------------------------------------------------------------
# run_single_config — main orchestration per config × h-value
# ---------------------------------------------------------------------------


def run_single_config(
    config: BenchmarkConfig,
    h_value: float,
    mode: str,
    shots: int,
    seed: int,
    backend: Any = None,
    prebuilt_circuit: QuantumCircuit | None = None,
    prebuilt_transpiled: QuantumCircuit | None = None,
    n_2q_logical_precomputed: int | None = None,
) -> dict[str, Any]:
    """Execute one config × one h-value. Returns ResultEnvelope as dict.

    Steps:
    1. Check idempotency (skip if result file already exists on disk)
    2. Fix random seed (numpy + any stochastic RNG)
    3. Obtain e_exact, gap from ClassicalSolver (cached per h_value)
    4. Build circuit (HVA p=1 or AQC-compressed), using pre-built if available
    5. Transpile with appropriate optimization_level, using pre-built if available
    6. Collect pre-execution metrics (transpiled_circuit_stats + error_budget)
    7. Compute derived circuit_stats
    8. Route to correct mitigation function via route_execution()
    9. Apply affine correction on raw if applicable
    10. Collect hardware calibration if mode=hardware
    11. Assemble ResultEnvelope, save JSON, append manifest entry

    Parameters
    ----------
    config : BenchmarkConfig
        The benchmark configuration for this run.
    h_value : float
        Transverse field strength.
    mode : str
        Execution mode ("fake_backend" or "hardware").
    shots : int
        Number of shots for estimation.
    seed : int
        Random seed for reproducibility.
    backend : Any, optional
        Pre-created backend instance. If None, creates one (backward compat).
    prebuilt_circuit : QuantumCircuit | None, optional
        Pre-built HVA circuit from the h-outer loop cache (non-AQC only).
    prebuilt_transpiled : QuantumCircuit | None, optional
        Pre-transpiled circuit from the h-outer loop cache (non-AQC only).
    n_2q_logical_precomputed : int | None, optional
        Pre-computed logical 2Q gate count (avoids re-counting).

    Returns
    -------
    dict[str, Any]
        Complete ResultEnvelope dict, or empty dict if skipped (idempotency).
    """
    # ── 1. Idempotency check ─────────────────────────────────────────────
    result_path = _build_result_path(config.config_id, h_value, mode, seed)
    # Check if a result with the same (config_id, h_value, seed) already exists.
    # We glob for matching files since filename includes a unique timestamp.
    h_str = f"h{str(h_value).replace('.', 'p')}"
    if seed != 42:
        # Non-default seed: look for files ending in _seed{N}.json
        existing_pattern = f"{h_str}_run_*_seed{seed}.json"
    else:
        # Default seed: look for files that do NOT have _seed suffix
        # Pattern: h3p25_run_YYYYMMDD_HHMMSS.json (no _seed before .json)
        existing_pattern = f"{h_str}_run_????????_??????.json"
    existing_dir = RESULTS_BASE / mode / config.config_id
    existing_dir.mkdir(parents=True, exist_ok=True)
    existing_files = [
        f
        for f in existing_dir.glob(existing_pattern)
        if "_seed" not in f.stem or (seed != 42 and f"_seed{seed}" in f.stem)
    ]
    if existing_files:
        # Verify at least one file has actual results (not just error envelopes)
        has_valid_result = False
        for ef in existing_files:
            try:
                content = json.loads(ef.read_text())
                # A valid result has "results" with e_raw or e_mitigated populated
                results = content.get("results", {})
                if results.get("e_raw") is not None or results.get("e_mitigated") is not None:
                    has_valid_result = True
                    break
            except (json.JSONDecodeError, OSError):
                continue
        if has_valid_result:
            print(f"  SKIP: {config.config_id} h={h_value} seed={seed} (already exists)")
            return {}
        else:
            # Remove error-only artifacts so we can retry
            for ef in existing_files:
                try:
                    ef.unlink()
                except OSError:
                    pass

    # ── 2. Fix seed ──────────────────────────────────────────────────────
    np.random.seed(seed)

    t0 = time.time()

    # ── 3. Exact reference (cached, not VQE) ─────────────────────────────
    e_exact, gap = _get_exact_energy(h_value)
    e_upper = _compute_upper_bound(h_value)

    # ── 4. Circuit construction ──────────────────────────────────────────
    aqc_metrics = None
    if config.aqc_enabled:
        circuit, aqc_metrics = _build_aqc_circuit(h_value)
        n_2q_logical = sum(1 for inst in circuit.data if inst.operation.num_qubits == 2)
    elif prebuilt_circuit is not None:
        circuit = prebuilt_circuit
        n_2q_logical = n_2q_logical_precomputed or sum(
            1 for inst in circuit.data if inst.operation.num_qubits == 2
        )
    else:
        circuit = _build_hva_circuit(h_value)
        n_2q_logical = sum(1 for inst in circuit.data if inst.operation.num_qubits == 2)

    # ── 5. Transpile ─────────────────────────────────────────────────────
    if backend is None:
        backend = _get_backend(mode)

    if prebuilt_transpiled is not None and not config.aqc_enabled:
        transpiled = prebuilt_transpiled
    else:
        pm = generate_preset_pass_manager(
            optimization_level=config.optimization_level,
            backend=backend,
            seed_transpiler=seed,
        )
        transpiled = pm.run(circuit)

    # ── 6. Pre-execution metrics ─────────────────────────────────────────
    stats = transpiled_circuit_stats(transpiled)

    # ── 6b. Post-transpilation sanity guard ──────────────────────────────
    # Catches silent corruption (e.g., θ=zeros → 0 CZ gates after opt≥1)
    n_2q_transpiled = stats.get("n_2q_gates", 0)
    if n_2q_transpiled == 0 and config.optimization_level >= 1:
        raise RuntimeError(
            f"{config.config_id} h={h_value}: transpiled circuit has 0 2Q gates! "
            f"Likely θ values are zero/trivial and got cancelled by opt_level="
            f"{config.optimization_level}. Check parameter binding in "
            f"_build_hva_circuit / _build_aqc_circuit."
        )

    # Inject depth_logical for derived stats computation
    # (transpiled_circuit_stats only returns transpiled depth, not logical)
    stats["depth_logical"] = circuit.depth()

    # ── 7. Derived circuit stats ─────────────────────────────────────────
    stats = compute_derived_circuit_stats(stats, n_2q_logical)

    error_budget = compute_error_budget(transpiled, backend=backend)

    # ── 8. Build Hamiltonian mapped to physical layout ───────────────────
    lattice_h = make_lattice(_TOPOLOGY, _N_QUBITS, J=1.0, h=h_value)
    H = HamiltonianBuilder().build(lattice_h)
    H_mapped = H.apply_layout(transpiled.layout)

    # ── 9. Route execution ───────────────────────────────────────────────
    execution_result = route_execution(
        config,
        transpiled,
        H_mapped,
        backend,
        shots,
        h_value,
        e_exact,
        gap,
        logical_circuit=circuit,
        H_logical=H,
        mode=mode,
    )

    # ── 10. Affine-on-raw correction ────────────────────────────────────
    execution_result = apply_affine_on_raw(config, execution_result, e_exact, e_upper)

    # ── 10b. Compute delta_e_gap and derived metrics ────────────────────
    # These are computed HERE (not in individual executors) because
    # e_exact/gap come from the classical solver cache, not from execution.
    e_final = execution_result.get("e_mitigated") or execution_result.get("e_raw")
    if e_final is not None and gap > 1e-15:
        execution_result["delta_e_gap"] = abs(e_final - e_exact) / abs(gap)
    else:
        execution_result["delta_e_gap"] = None
    execution_result["e_exact"] = e_exact

    # Phase label from energy (paramagnetic if E closer to large-h limit)
    if e_final is not None:
        execution_result["phase_label"] = "paramagnetic"  # h > h_c for all h_test
        execution_result["correct_label"] = True
    else:
        execution_result["phase_label"] = None
        execution_result["correct_label"] = None

    # Energy bounds check
    if e_final is not None:
        execution_result["energy_within_physical_bounds"] = e_exact <= e_final <= e_upper
    else:
        execution_result["energy_within_physical_bounds"] = None

    # ── 11. Hardware calibration ────────────────────────────────────────
    hw_calibration = None
    job = execution_result.get("_job")  # May be set by execution router
    if mode == "hardware":
        hw_calibration = _collect_hardware_calibration(backend, job)

    # ── 12. Timing + envelope assembly ──────────────────────────────────
    wall_time_s = time.time() - t0

    envelope = _build_envelope(
        config=config,
        h_value=h_value,
        mode=mode,
        seed=seed,
        circuit_stats=stats,
        error_budget=error_budget,
        execution_result=execution_result,
        wall_time_s=wall_time_s,
        hardware_calibration=hw_calibration,
        aqc_metrics=aqc_metrics,
    )

    # ── 13. Persist + manifest ──────────────────────────────────────────
    _save_result(envelope, result_path)

    manifest_entry = _build_manifest_entry(
        config_id=config.config_id,
        h_value=h_value,
        mode=mode,
        result_path=result_path,
        envelope=envelope,
    )
    append_to_manifest(manifest_entry)

    return envelope


# ---------------------------------------------------------------------------
# run_benchmark — main loop iterating configs × h_values
# ---------------------------------------------------------------------------


def _save_error_result(
    config: BenchmarkConfig,
    h_value: float,
    mode: str,
    seed: int,
    error: Exception,
) -> None:
    """Persist an error envelope when run_single_config fails.

    Parameters
    ----------
    config : BenchmarkConfig
        Configuration that failed.
    h_value : float
        Transverse field value that failed.
    mode : str
        Execution mode.
    seed : int
        Random seed used.
    error : Exception
        The exception that caused the failure.
    """
    result_path = _build_result_path(config.config_id, h_value, mode, seed)
    error_envelope: dict[str, Any] = {
        "benchmark_metadata": {
            "config_id": config.config_id,
            "execution_mode": mode,
            "h_value": h_value,
            "timestamp": datetime.now(UTC).isoformat(),
            "benchmark_version": BENCHMARK_VERSION,
            "seed": seed,
        },
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "mitigation_config": dataclasses.asdict(config),
    }
    _save_result(error_envelope, result_path)


def _compute_kappa_ordering(h_values: list[float]) -> dict[float, str]:
    """Compute landscape curvature per h-value for adaptive scheduling.

    Returns dict mapping h_value → risk_level ("HIGH", "MEDIUM", "LOW").
    HIGH κ: near criticality, aggressive mitigation needed.
    LOW κ: deep in phase, simple suppression sufficient.

    Parameters
    ----------
    h_values : list[float]
        h-values to compute curvature for.

    Returns
    -------
    dict[float, str]
        Mapping h_value → risk_level.
    """
    try:
        from scripts.experiment_runners.hardware.run_ibm_deployment import compute_kappa_per_h

        kappa_per_h = compute_kappa_per_h(
            h_values, n_qubits=_N_QUBITS, topology=_TOPOLOGY, p_layers=_P_LAYERS
        )
    except (ImportError, Exception) as e:
        logger.warning(f"Cannot compute κ for adaptive scheduling: {e}")
        return {h: "MEDIUM" for h in h_values}

    # Calibrate thresholds via percentiles (topology-agnostic)
    kappas = list(kappa_per_h.values())
    if not kappas:
        return {h: "MEDIUM" for h in h_values}

    p33 = np.percentile(kappas, 33)
    p67 = np.percentile(kappas, 67)

    result = {}
    for h in h_values:
        k = kappa_per_h.get(h, p33)
        if k >= p67:
            result[h] = "HIGH"
        elif k >= p33:
            result[h] = "MEDIUM"
        else:
            result[h] = "LOW"
    return result


def run_benchmark(
    configs: list[str],
    h_values: list[float],
    mode: str,
    shots: int,
    seed: int,
    adaptive: bool = False,
    warm_start_path: str | None = None,
    batch: bool = False,
) -> None:
    """Main orchestrator — restructured with h-outer loop for cache efficiency.

    Restructured loop order (h-outer) ensures that for each h-value the
    circuit is built once and transpiled once per optimization_level,
    maximizing cache reuse across configs.

    When --adaptive is set, uses κ (landscape curvature) to filter configs
    per h-value: LOW-risk gets only P0+P1, MEDIUM gets P0+P1+P2, HIGH gets all.

    Parameters
    ----------
    configs : list[str]
        List of config_ids to execute.
    h_values : list[float]
        Default h-values to test (overridden by config.h_test_values for AQC).
    mode : str
        Execution mode ("fake_backend" or "hardware").
    shots : int
        Number of shots per estimation.
    seed : int
        Random seed for reproducibility.
    adaptive : bool
        If True, use kappa-based adaptive scheduling to filter configs.
    warm_start_path : str | None
        Path to MPNN checkpoint for warm-start parameters (None = θ=zeros).
    """
    # Create backend once (avoid re-instantiation per iteration)
    backend = _get_backend(mode)
    _benchmark_start_time = time.time()

    # ── QPU Cost Estimation (hardware mode only) ──────────────────────────
    if mode == "hardware":
        try:
            from qmbp_simulation.execution.hardware.config import HardwareConfig
            from qmbp_simulation.execution.hardware.preflight import (
                QPUThroughputProfile,
                SPSACostModel,
                compute_mean_2q_error,
                estimate_qpu_cost,
            )

            # Build a minimal HardwareConfig for cost estimation
            _hw_config = HardwareConfig(
                mode="hardware",
                n_qubits=_N_QUBITS,
                shots=shots,
                n_layouts=3,
            )
            _hw_config.mitigation.zne_amplifier = "pea"

            profile = QPUThroughputProfile.ibm_kingston()
            spsa_model = SPSACostModel.disabled()  # Benchmark doesn't use SPSA
            n_total_executions = len(configs) * len(h_values)

            cost = estimate_qpu_cost(
                _hw_config,
                n_h_points=n_total_executions,
                profile=profile,
                spsa_model=spsa_model,
                cx_count=18,  # N=10 p=1 heavy_hex standard
            )

            # Backend calibration check
            mean_2q_err = compute_mean_2q_error(backend)
            mean_2q_str = f"{mean_2q_err * 100:.2f}%" if mean_2q_err else "N/A"

            print("\n  ┌─── Pre-Execution Budget (model-based) ──────────────────────┐")
            print(f"  │  Backend: {getattr(backend, 'name', 'unknown'):<48s}│")
            print(f"  │  Mean 2Q error (chip-wide): {mean_2q_str:<34s}│")
            print(
                f"  │  Effective CLOPS: {cost.effective_clops:<44s}│"
                if isinstance(cost.effective_clops, str)
                else f"  │  Effective CLOPS: {cost.effective_clops:<44d}│"
            )
            print(
                f"  │  Configs × h-points: {len(configs)} × {len(h_values)} = "
                f"{n_total_executions} executions{' ' * (28 - len(str(n_total_executions)))}│"
            )
            print(f"  │  Total shots: {cost.total_shots:>12,}{' ' * 35}│")
            print(
                f"  │  Optimistic: {cost.est_total_optimistic_s / 60:.1f} min | "
                f"Expected: {cost.est_total_s / 60:.1f} min{' ' * 17}│"
            )
            print(f"  │  Time per circuit: {cost.time_per_circuit_s:.2f}s{' ' * 40}│")
            if mean_2q_err and mean_2q_err > 0.03:
                print("  │  ⚠ Elevated 2Q error — layout selection will avoid worst  │")
            print("  └────────────────────────────────────────────────────────────────┘\n")
        except Exception as e:
            logger.warning(f"Cost estimation failed (non-blocking): {e}")

    # All configs use the CLI-provided h_values (no hardcoded overrides)
    all_h: set[float] = set(h_values)
    all_h_sorted = sorted(all_h)

    # Pre-warm ClassicalSolver cache
    for h in all_h_sorted:
        _get_exact_energy(h)

    # Determine which optimization levels are needed
    opt_levels_needed = {BENCHMARK_CONFIGS[c].optimization_level for c in configs}

    # Adaptive scheduling: compute kappa risk levels
    kappa_risk: dict[float, str] | None = None
    max_priority_for_risk = {"LOW": 1, "MEDIUM": 2, "HIGH": 4}
    if adaptive:
        kappa_risk = _compute_kappa_ordering(all_h_sorted)
        # Emit summary
        risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for r in kappa_risk.values():
            risk_counts[r] += 1
        logger.info(
            f"Adaptive scheduling: {risk_counts['HIGH']} HIGH, "
            f"{risk_counts['MEDIUM']} MEDIUM, {risk_counts['LOW']} LOW κ h-values"
        )

    # ── BATCHED HARDWARE EXECUTION (--batch flag) ────────────────────────
    # When batch=True and mode=hardware, collect all (config, h) pairs,
    # transpile them, then submit everything in ONE Batch session.
    if batch and mode == "hardware":
        print("\n  [BATCH MODE] Collecting all jobs before submission...")
        jobs_spec = []  # list of (config, transpiled_circuit, H_mapped, h_value)
        job_metadata = []  # parallel list of (config_id, h_value) for result routing

        for h in all_h_sorted:
            warm_params = _get_warm_start_params(h, warm_start_path)
            circuit_hva = _build_hva_circuit(h, warm_start_params=warm_params)
            n_2q_logical_hva = sum(1 for inst in circuit_hva.data if inst.operation.num_qubits == 2)
            logical_depth_hva = circuit_hva.depth()

            # Transpile with mapomatic if available
            transpiled = None
            try:
                from qmbp_simulation.execution.hardware.layout_optimizer import (
                    MAPOMATIC_AVAILABLE,
                    select_optimal_layouts,
                )

                if MAPOMATIC_AVAILABLE:
                    layout_result = select_optimal_layouts(
                        circuit_hva,
                        backend,
                        n_select=1,
                        optimization_level=2,
                        strategy="lowest_cost",
                    )
                    if layout_result.transpiled_circuits:
                        transpiled = layout_result.transpiled_circuits[0]
            except Exception:
                pass
            if transpiled is None:
                pm = generate_preset_pass_manager(
                    optimization_level=2, backend=backend, seed_transpiler=seed
                )
                transpiled = pm.run(circuit_hva)

            lattice_h = make_lattice(_TOPOLOGY, _N_QUBITS, J=1.0, h=h)
            H = HamiltonianBuilder().build(lattice_h)
            H_mapped = H.apply_layout(transpiled.layout)

            for config_id in configs:
                config = BENCHMARK_CONFIGS[config_id]
                if config.zne_method not in (None, "gf", "pea"):
                    continue  # Mitiq not supported in batch mode

                # Idempotency: skip if valid result already exists
                h_str = f"h{str(h).replace('.', 'p')}"
                seed_suffix = f"_seed{seed}" if seed != 42 else ""
                existing_dir = RESULTS_BASE / mode / config_id
                existing_dir.mkdir(parents=True, exist_ok=True)
                pattern = (
                    f"{h_str}_run_*{seed_suffix}.json"
                    if seed != 42
                    else f"{h_str}_run_????????_??????.json"
                )
                existing = [f for f in existing_dir.glob(pattern)]
                skip = False
                for ef in existing:
                    try:
                        content = json.loads(ef.read_text())
                        r = content.get("results", {})
                        if r.get("e_raw") is not None or r.get("e_mitigated") is not None:
                            skip = True
                            break
                    except Exception:
                        continue
                if skip:
                    print(f"    SKIP: {config_id} h={h} (already exists)")
                    continue

                jobs_spec.append((config, transpiled, H_mapped, h))
                job_metadata.append((config_id, h, n_2q_logical_hva, logical_depth_hva))

        print(f"  [BATCH] {len(jobs_spec)} jobs prepared, submitting...")
        batch_results = _execute_hardware_batched(jobs_spec, backend, shots)

        # Route results back to per-config persistence
        for i, ((config_id, h_val, n_2q_log, depth_log), exec_result) in enumerate(
            zip(job_metadata, batch_results, strict=False)
        ):
            config = BENCHMARK_CONFIGS[config_id]
            if exec_result.get("error"):
                try:
                    _save_error_result(
                        config, h_val, mode, seed, RuntimeError(exec_result["error"])
                    )
                except Exception:
                    pass
                continue

            # Compute derived metrics
            e_exact, gap = _get_exact_energy(h_val)
            e_final = exec_result.get("e_mitigated") or exec_result.get("e_raw")
            if e_final is not None and gap > 1e-15:
                exec_result["delta_e_gap"] = abs(e_final - e_exact) / abs(gap)
            exec_result["e_exact"] = e_exact
            exec_result["phase_label"] = "paramagnetic"
            exec_result["correct_label"] = True
            # energy_within_physical_bounds check
            e_upper = _N_QUBITS * max(1.0, h_val)
            if e_final is not None:
                exec_result["energy_within_physical_bounds"] = e_exact <= e_final <= e_upper
            else:
                exec_result["energy_within_physical_bounds"] = None

            # Build minimal stats and envelope
            transpiled_i = jobs_spec[i][1]
            stats = transpiled_circuit_stats(transpiled_i)
            stats["depth_logical"] = depth_log
            stats = compute_derived_circuit_stats(stats, n_2q_log)

            e_upper = _N_QUBITS * max(1.0, h_val)
            exec_result = apply_affine_on_raw(config, exec_result, e_exact, e_upper)

            error_budget = compute_error_budget(transpiled_i, backend=backend)
            hw_cal = None
            if mode == "hardware":
                job_obj = exec_result.get("_job")
                hw_cal = _collect_hardware_calibration(backend, job_obj)
            envelope = _build_envelope(
                config=config,
                h_value=h_val,
                mode=mode,
                seed=seed,
                circuit_stats=stats,
                error_budget=error_budget,
                execution_result=exec_result,
                wall_time_s=0.0,
                hardware_calibration=hw_cal,
            )
            result_path = _build_result_path(config_id, h_val, mode, seed)
            _save_result(envelope, result_path)
            manifest_entry = _build_manifest_entry(config_id, h_val, mode, result_path, envelope)
            append_to_manifest(manifest_entry)
            print(f"    [{config_id}] h={h_val:.2f} saved")

        # Post-execution summary
        wall_total = time.time() - _benchmark_start_time
        print(f"\n  [BATCH] Complete in {wall_total / 60:.1f} min")
        return

    # Main execution: h-outer loop for maximum cache reuse (sequential mode)
    total_skipped_adaptive = 0
    for h in all_h_sorted:
        # Build logical circuit once per h (non-AQC)
        warm_params = _get_warm_start_params(h, warm_start_path)
        circuit_hva = _build_hva_circuit(h, warm_start_params=warm_params)
        n_2q_logical_hva = sum(1 for inst in circuit_hva.data if inst.operation.num_qubits == 2)

        # Transpile once per (h, optimization_level)
        # Use mapomatic VF2 for BOTH modes when available — ensures fair
        # comparison by giving ALL configs equally good layouts (isolates
        # mitigation effect from layout luck). For simulation, this also
        # reduces CES → measurements are closer to hardware reality.
        transpiled_cache_local: dict[int, QuantumCircuit] = {}
        for opt_level in opt_levels_needed:
            transpiled = None
            try:
                from qmbp_simulation.execution.hardware.layout_optimizer import (
                    MAPOMATIC_AVAILABLE,
                    select_optimal_layouts,
                )

                if MAPOMATIC_AVAILABLE:
                    layout_result = select_optimal_layouts(
                        circuit_hva,
                        backend,
                        n_select=1,  # Single best layout for benchmark
                        optimization_level=opt_level,
                        strategy="lowest_cost",
                    )
                    if layout_result.transpiled_circuits:
                        transpiled = layout_result.transpiled_circuits[0]
                        logger.debug(
                            "VF2 layout: %s CES=%.4f (h=%.2f, opt=%d)",
                            layout_result.layouts[0][:5] if layout_result.layouts else "?",
                            layout_result.ces_values[0] if layout_result.ces_values else -1,
                            h,
                            opt_level,
                        )
            except Exception as e:
                logger.debug(f"VF2 layout optimization skipped: {e}")

            if transpiled is None:
                # Fallback: default transpilation (no explicit layout)
                pm = generate_preset_pass_manager(
                    optimization_level=opt_level,
                    backend=backend,
                    seed_transpiler=seed,
                )
                transpiled = pm.run(circuit_hva)

            transpiled_cache_local[opt_level] = transpiled

        # All configs run at all CLI-provided h-values (no hardcoded filtering)
        configs_for_h = list(configs)

        # Adaptive filtering: limit configs by kappa risk level
        if adaptive and kappa_risk is not None:
            risk = kappa_risk.get(h, "MEDIUM")
            max_p = max_priority_for_risk[risk]
            before_count = len(configs_for_h)
            configs_for_h = [c for c in configs_for_h if BENCHMARK_CONFIGS[c].priority <= max_p]
            total_skipped_adaptive += before_count - len(configs_for_h)

        # Execute configs for this h-value
        for config_id in configs_for_h:
            config = BENCHMARK_CONFIGS[config_id]
            print(f"[{config_id}] h={h:.2f} ...", end=" ", flush=True)
            try:
                run_single_config(
                    config,
                    h,
                    mode,
                    shots,
                    seed,
                    backend,
                    prebuilt_circuit=circuit_hva if not config.aqc_enabled else None,
                    prebuilt_transpiled=transpiled_cache_local.get(config.optimization_level)
                    if not config.aqc_enabled
                    else None,
                    n_2q_logical_precomputed=n_2q_logical_hva if not config.aqc_enabled else None,
                )
                print("DONE")
            except Exception as e:
                print(f"ERROR: {e}")
                try:
                    _save_error_result(config, h, mode, seed, e)
                except Exception:
                    pass  # Best-effort error persistence

    if adaptive and total_skipped_adaptive > 0:
        logger.info(
            f"Adaptive scheduling: skipped {total_skipped_adaptive} "
            f"config×h executions based on κ risk levels"
        )

    # ── Post-execution summary (hardware mode) ────────────────────────────
    if mode == "hardware":
        wall_total = time.time() - _benchmark_start_time
        print("\n  ┌─── Execution Summary ─────────────────────────────────────────┐")
        print(f"  │  Wall clock: {wall_total / 60:.1f} min{' ' * 47}│")
        print(f"  │  Configs executed: {len(configs)}{' ' * 43}│")
        print(f"  │  h-points: {len(all_h_sorted)}{' ' * 51}│")
        print(f"  │  Results: {RESULTS_BASE / mode}/{' ' * 20}│")
        print("  └────────────────────────────────────────────────────────────────┘")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for the mitigation benchmark runner.

    Parses CLI arguments, handles --export-configs early exit, validates
    hardware credentials (fail-fast), resolves configs, parses h-values,
    and dispatches to run_benchmark().
    """
    args = parse_args()

    # --export-configs: serialize and exit
    if args.export_configs:
        export_configs(RESULTS_BASE / "configs")
        return

    # ── Hardware credential validation (fail-fast) ───────────────────────
    # Validate BEFORE any expensive operations (config resolution, circuit
    # building, ClassicalSolver). Pattern from run_ibm_deployment.py.
    if args.mode == "hardware":
        ibm_key, instance_crn = _validate_hardware_credentials()
        print(f"  🔑 IBM Key: {'*' * 8}...{ibm_key[-4:]}")
        print(f"  🏢 Instance: ...{instance_crn[-20:]}")
        # Wire --backend to env var for _get_backend() to read
        if args.backend:
            os.environ["BENCHMARK_BACKEND"] = args.backend
            print(f"  🖥️  Backend: {args.backend} (explicit)")
        else:
            os.environ.pop("BENCHMARK_BACKEND", None)
            print("  🖥️  Backend: least_busy (auto)")
        print()

    # Resolve which configs to run
    config_ids = resolve_configs(args)
    if not config_ids:
        print("No configs matched the specified filters. Exiting.")
        return

    # Parse h-values
    h_values = [float(x.strip()) for x in args.h_values.split(",")]

    print(f"Mitigation Benchmark v{BENCHMARK_VERSION}")
    print(f"  Mode: {args.mode}")
    print(f"  Configs: {len(config_ids)} ({config_ids[0]}...{config_ids[-1]})")
    print(f"  h-values: {h_values}")
    print(f"  Shots: {args.shots}")
    print(f"  Seed: {args.seed}")
    if args.batch and args.mode == "hardware":
        print("  Batch mode: ON (single Batch session for all jobs)")
    print()

    run_benchmark(
        config_ids,
        h_values,
        args.mode,
        args.shots,
        args.seed,
        adaptive=args.adaptive,
        warm_start_path=args.warm_start,
        batch=getattr(args, "batch", False),
    )


if __name__ == "__main__":
    main()
