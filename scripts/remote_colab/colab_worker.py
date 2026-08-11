#!/usr/bin/env python3
"""
Colab Worker — Executes compute-heavy tasks on the remote VM.

This script runs INSIDE the Colab VM. It reads a job JSON from stdin or a file,
executes the requested computations (DMRG, VQE, etc.), and writes results to stdout/file.

Supported tasks:
  - "dmrg": Compute ground truth (energy, gap) via DMRG/exact diag
  - "vqe": Run VQE optimization with given initial parameters

Usage (on VM):
    python colab_worker.py --job /content/job.json --output /content/results.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Ensure qmbp_simulation is importable (sys.path fallback for Colab)
import importlib

if importlib.util.find_spec("qmbp_simulation") is None:
    sys.path.insert(0, "/content/qmbp/src")


def run_dmrg_task(task: dict) -> dict:
    """Compute ground truth for a single (topology, N, model, h) point.

    Parameters
    ----------
    task : dict
        Required keys: topology, n_qubits, model, h
        Optional: chi_max, J

    Returns
    -------
    dict with: h, energy, gap, gap_method, mag_x, corr_zz, elapsed_s
    """
    # Import only what we need — avoid qiskit dependency for pure DMRG
    from qmbp_simulation.models.hamiltonian import HamiltonianBuilder, make_lattice
    from qmbp_simulation.models.model_registry import get_model_spec
    from qmbp_simulation.solvers.classical import ClassicalSolver

    topology = task["topology"]
    n_qubits = task["n_qubits"]
    model = task.get("model", "tfim")
    h = task["h"]
    chi_max = task.get("chi_max")
    J = task.get("J", 1.0)

    t0 = time.time()

    lattice = make_lattice(topology=topology, n_qubits=n_qubits, h=h, J=J)
    spec = get_model_spec(model)
    builder = HamiltonianBuilder()
    H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)

    solver = ClassicalSolver()
    gt = solver.solve(H, lattice, chi_max=chi_max)

    elapsed = time.time() - t0

    return {
        "h": h,
        "energy": float(gt.ground_energy),
        "gap": float(gt.gap),
        "gap_method": gt.gap_method,
        "mag_x": float(gt.mag_x) if gt.mag_x is not None else None,
        "corr_zz": float(gt.corr_zz) if gt.corr_zz is not None else None,
        "elapsed_s": round(elapsed, 2),
    }


def run_vqe_task(task: dict) -> dict:
    """Run VQE optimization for a single point.

    Parameters
    ----------
    task : dict
        Required: topology, n_qubits, model, h, p_layers
        Optional: theta_init (list), maxiter, n_restarts, method, J

    Returns
    -------
    dict with: h, energy, theta_opt (list), n_iters, elapsed_s
    """
    from qmbp_simulation import HamiltonianBuilder, VQEOptimizer, make_lattice
    from qmbp_simulation.models.model_registry import get_model_spec

    topology = task["topology"]
    n_qubits = task["n_qubits"]
    model = task.get("model", "tfim")
    h = task["h"]
    p_layers = task["p_layers"]
    theta_init = task.get("theta_init")
    maxiter = task.get("maxiter", 500)
    n_restarts = task.get("n_restarts", 3)
    method = task.get("method", "L-BFGS-B")
    J = task.get("J", 1.0)

    t0 = time.time()

    lattice = make_lattice(topology=topology, n_qubits=n_qubits, h=h, J=J)
    spec = get_model_spec(model)
    builder = HamiltonianBuilder()
    H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
    qc, _ = spec.create_circuit(n_qubits, p_layers, lattice, **spec.circuit_kwargs)

    if theta_init is not None:
        theta_init = np.array(theta_init)

    optimizer = VQEOptimizer()
    result = optimizer.optimize(
        circuit=qc,
        hamiltonian=H,
        initial_theta=theta_init,
        method=method,
        maxiter=maxiter,
        n_restarts=n_restarts,
    )

    elapsed = time.time() - t0

    return {
        "h": h,
        "energy": float(result.optimal_energy),
        "theta_opt": [float(x) for x in result.optimal_params],
        "n_iters": result.n_iterations,
        "elapsed_s": round(elapsed, 2),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Colab Worker — compute heavy tasks")
    parser.add_argument("--job", required=True, help="Path to job JSON file")
    parser.add_argument("--output", required=True, help="Path to write results JSON")
    args = parser.parse_args()

    # Load job
    with open(args.job) as f:
        job = json.load(f)

    task_type = job.get("task_type", "dmrg")
    tasks = job.get("tasks", [])
    metadata = job.get("metadata", {})

    logger.info(f"Worker started: {len(tasks)} {task_type} tasks")
    logger.info(f"  Metadata: {metadata}")

    results = []
    n_success = 0
    n_fail = 0

    for i, task in enumerate(tasks):
        logger.info(f"  [{i + 1}/{len(tasks)}] {task_type} h={task.get('h', '?')}...")
        try:
            if task_type == "dmrg":
                result = run_dmrg_task(task)
            elif task_type == "vqe":
                result = run_vqe_task(task)
            else:
                raise ValueError(f"Unknown task_type: {task_type}")
            result["status"] = "ok"
            results.append(result)
            n_success += 1
            logger.info(
                f"    ✓ E={result['energy']:.8f}, gap={result.get('gap', 'N/A')}, "
                f"{result['elapsed_s']}s"
            )
        except Exception as e:
            logger.error(f"    ✗ {type(e).__name__}: {e}")
            results.append({"h": task.get("h"), "status": "error", "error": str(e)})
            n_fail += 1

    # Write output
    output = {
        "task_type": task_type,
        "metadata": metadata,
        "n_tasks": len(tasks),
        "n_success": n_success,
        "n_fail": n_fail,
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nDone: {n_success} OK, {n_fail} failed → {args.output}")


if __name__ == "__main__":
    main()
