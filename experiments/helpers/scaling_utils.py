"""Shared scaling utilities for experiment runners.

Reusable functions for:
- Power-law fitting (log-log regression with R²)
- Transpilation metrics (build → transpile → measure)
- MPS chi-convergence evaluation (same theta at multiple chi)

These utilities are designed to be imported by scaling experiment runners
and any future script that needs circuit scaling or MPS precision analysis.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

logger = logging.getLogger(__name__)


# Power-law fitting now lives in the package (pure numpy util) so src/ can reuse
# it without importing from experiments/. Re-exported here for runner compat.
from qmbp_simulation.utils.helpers import fit_power_law  # noqa: F401

# ═══════════════════════════════════════════════════════════════════════════════
# Transpilation Metrics
# ═══════════════════════════════════════════════════════════════════════════════


def compute_transpilation_metrics(
    circuit: QuantumCircuit,
    backend,
    optimization_level: int = 2,
) -> dict[str, Any]:
    """Transpile a circuit and return comprehensive resource metrics.

    Wraps the transpilation process with timing, then uses the canonical
    `transpiled_circuit_stats()` for post-transpile metrics. Also computes
    pre-transpile 2Q gate count for routing overhead calculation.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrized or bound circuit to transpile.
    backend : BackendV2
        Target backend (real or fake).
    optimization_level : int
        Qiskit transpiler optimization level (default: 2).

    Returns
    -------
    dict with keys:
        - cx_count_pre_transpile: int
        - cx_count_post_transpile: int
        - n_swap_gates: int
        - routing_overhead_ratio: float
        - depth_total: int
        - depth_2q: int
        - n_1q_gates: int
        - n_2q_gates: int
        - transpile_time_s: float
        - gate_counts: dict[str, int]
        - transpiled_circuit: QuantumCircuit (the result)
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    from qmbp_simulation.analysis.circuit_visualizer import transpiled_circuit_stats

    # Pre-transpile 2Q gate count
    two_q_gates = {"cx", "cz", "ecr", "rzz", "rxx", "ryy", "cp"}
    cx_pre = sum(1 for inst in circuit.data if inst.operation.name.lower() in two_q_gates)

    # Transpile with timing
    t0 = time.perf_counter()
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend)
    transpiled = pm.run(circuit)
    transpile_time = time.perf_counter() - t0

    # Post-transpile metrics via canonical utility
    stats = transpiled_circuit_stats(transpiled)

    # Extract SWAP count from gate breakdown
    gate_counts = stats.get("count_ops", {})
    n_swap = gate_counts.get("swap", 0)
    cx_post = stats["n_2q_gates"]

    return {
        "cx_count_pre_transpile": cx_pre,
        "cx_count_post_transpile": cx_post,
        "n_swap_gates": n_swap,
        "routing_overhead_ratio": cx_post / max(cx_pre, 1),
        "depth_total": stats["depth"],
        "depth_2q": stats["depth_2q"],
        "n_1q_gates": stats["n_1q_gates"],
        "n_2q_gates": cx_post,
        "transpile_time_s": round(transpile_time, 3),
        "gate_counts": gate_counts,
        "transpiled_circuit": transpiled,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MPS Chi-Convergence Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_at_multiple_chi(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    theta: np.ndarray,
    chi_values: list[int],
    seed: int = 42,
) -> dict[int, dict[str, float]]:
    """Evaluate the same θ at multiple MPS bond dimensions.

    This isolates MPS truncation error from VQE optimization noise.
    The energy at chi_max is the reference; deviations at lower chi
    represent pure truncation error.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parametrized HVA circuit.
    hamiltonian : SparsePauliOp
        System Hamiltonian.
    theta : np.ndarray
        Optimal parameters to evaluate.
    chi_values : list[int]
        Bond dimensions to test (should be sorted ascending).
    seed : int
        Random seed for MPS backend.

    Returns
    -------
    dict[int, dict] mapping chi → {energy, time_s}
    """
    from qmbp_simulation.execution.mps_backend import MPSBackend

    results: dict[int, dict[str, float]] = {}

    for chi in sorted(chi_values):
        backend = MPSBackend(strategy="aer_mps", chi_max=chi, seed=seed)
        t0 = time.perf_counter()
        energy = backend.evaluate(circuit, hamiltonian, theta)
        elapsed = time.perf_counter() - t0

        results[chi] = {
            "energy": energy,
            "time_s": round(elapsed, 3),
        }

    return results


def analyze_chi_convergence(
    chi_results: dict[int, dict[str, float]],
    e_exact: float | None = None,
    gap: float = 1.0,
    convergence_threshold: float = 1e-4,
) -> dict[str, Any]:
    """Analyze chi-convergence from evaluate_at_multiple_chi results.

    Parameters
    ----------
    chi_results : dict[int, dict]
        Output from evaluate_at_multiple_chi.
    e_exact : float | None
        Exact energy reference. If None, uses highest chi as reference.
    gap : float
        Spectral gap for ΔE/gap computation.
    convergence_threshold : float
        |ΔE| below this means "converged".

    Returns
    -------
    dict with:
        - ref_energy: float (reference used)
        - ref_source: str ("exact_diag" or "chi=<max>")
        - per_chi: dict[int, {abs_error, de_gap, converged}]
        - min_converged_chi: int | None
        - chi64_abs_error: float | None
        - chi64_de_gap: float | None
        - chi64_is_sufficient: bool | None
    """
    chi_values = sorted(chi_results.keys())
    chi_max = max(chi_values)

    e_ref = e_exact if e_exact is not None else chi_results[chi_max]["energy"]
    ref_source = "exact_diag" if e_exact is not None else f"chi={chi_max}"

    per_chi: dict[int, dict] = {}
    min_converged_chi = None

    for chi in chi_values:
        e_chi = chi_results[chi]["energy"]
        abs_error = abs(e_chi - e_ref)
        de_gap = abs_error / max(gap, 1e-10)
        converged = abs_error < convergence_threshold

        per_chi[chi] = {
            "abs_error": abs_error,
            "de_gap": de_gap,
            "converged": converged,
        }

        if converged and min_converged_chi is None:
            min_converged_chi = chi

    chi64_data = per_chi.get(64)
    chi64_abs_error = chi64_data["abs_error"] if chi64_data else None
    chi64_de_gap = chi64_data["de_gap"] if chi64_data else None
    chi64_sufficient = chi64_abs_error < convergence_threshold if chi64_abs_error is not None else None

    return {
        "ref_energy": e_ref,
        "ref_source": ref_source,
        "per_chi": per_chi,
        "min_converged_chi": min_converged_chi,
        "chi64_abs_error": chi64_abs_error,
        "chi64_de_gap": chi64_de_gap,
        "chi64_is_sufficient": chi64_sufficient,
    }
