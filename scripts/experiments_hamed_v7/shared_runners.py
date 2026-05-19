"""
Shared Experiment Runners — Common optimization and evaluation functions.

Extracted from the 5 technique scripts to avoid duplication.
All experiment scripts should import from here instead of re-implementing.

Functions:
    - run_lbfgsb: L-BFGS-B optimizer (noiseless)
    - run_lbfgsb_with_restarts: L-BFGS-B with multiple restarts
    - run_cobyla_noiseless: COBYLA on noiseless cost
    - run_cobyla_noisy: COBYLA on noisy cost
    - run_spsa: SPSA optimizer (for shot-noise scenarios)
    - run_nevergrad: Nevergrad optimizer wrapper
    - noisy_cost_function: Shot-noise cost function factory
    - vqe_descending_sweep: Generate training data via descending VQE sweep
    - setup_experiment: Common circuit/lattice/solver setup
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment_utils import evaluate_energy_statevector

from src.poc.v6 import (
    ClassicalSolver,
    HamiltonianBuilder,
    HVACircuitBuilder,
    make_lattice,
)

# ── Experiment Setup ─────────────────────────────────────────────────────────


def setup_experiment(N: int, p: int = 2):
    """Common experiment setup: build circuit, lattice, solver instances.

    Parameters
    ----------
    N : int
        Number of qubits.
    p : int
        HVA layers (default 2).

    Returns
    -------
    dict with keys: circuit, n_params, base_lattice, builder, solver, hva
    """
    builder = HamiltonianBuilder()
    solver = ClassicalSolver()
    hva = HVACircuitBuilder()
    base_lattice = make_lattice("chain_1d", N, J=1.0, h=1.0)
    qc, _ = hva.create(N, p, base_lattice)

    return {
        "circuit": qc,
        "n_params": qc.num_parameters,
        "base_lattice": base_lattice,
        "builder": builder,
        "solver": solver,
        "hva": hva,
    }


def get_exact_solution(builder, solver, N: int, h: float):
    """Get exact ground truth for a given h-value.

    Uses module-level cache to avoid redundant exact diag/DMRG calls
    across seeds and optimizer variants.

    Returns
    -------
    dict with keys: lattice, hamiltonian, exact (GroundTruthResult)
    """
    cache_key = (N, "chain_1d", 1.0, h)
    if cache_key not in _EXACT_CACHE:
        lattice = make_lattice("chain_1d", N, J=1.0, h=h)
        H = builder.build(lattice)
        exact = solver.solve(H, lattice)
        _EXACT_CACHE[cache_key] = {"lattice": lattice, "hamiltonian": H, "exact": exact}
    return _EXACT_CACHE[cache_key]


# Module-level cache for exact solutions (avoids redundant diag across seeds)
_EXACT_CACHE: dict[tuple, dict] = {}


# ── Noiseless Optimizers ─────────────────────────────────────────────────────


def run_lbfgsb(circuit, hamiltonian, initial_guess, maxiter=1000):
    """Run L-BFGS-B optimizer for VQE (noiseless).

    Returns
    -------
    tuple: (params_opt, energy, n_evaluations, wall_time_s, converged)
    """
    from scipy.optimize import minimize

    eval_count = [0]

    def cost_fn(params):
        eval_count[0] += 1
        return evaluate_energy_statevector(circuit, hamiltonian, params)

    t0 = time.time()
    result = minimize(
        cost_fn,
        initial_guess,
        method="L-BFGS-B",
        bounds=[(-np.pi, np.pi)] * len(initial_guess),
        options={"maxiter": maxiter, "ftol": 1e-14},
    )
    elapsed = time.time() - t0

    return result.x, result.fun, eval_count[0], elapsed, result.success


def run_lbfgsb_with_restarts(circuit, hamiltonian, initial_guess, n_restarts=5, maxiter=1000):
    """Run L-BFGS-B with multiple restarts (standard protocol).

    Returns
    -------
    tuple: (params_opt, energy, total_evals, wall_time_s)
    """
    best_params = initial_guess.copy()
    best_energy = float("inf")
    total_evals = 0
    n_params = len(initial_guess)

    t0 = time.time()
    for restart in range(n_restarts + 1):
        x0 = initial_guess if restart == 0 else best_params + np.random.normal(0, 0.1, n_params)
        x0 = np.clip(x0, -np.pi, np.pi)
        params, energy, evals, _, _ = run_lbfgsb(circuit, hamiltonian, x0, maxiter=maxiter)
        total_evals += evals
        if energy < best_energy:
            best_energy = energy
            best_params = params
    elapsed = time.time() - t0

    return best_params, best_energy, total_evals, elapsed


def run_cobyla_noiseless(circuit, hamiltonian, initial_guess, maxiter=500):
    """Run COBYLA on noiseless cost function.

    Returns
    -------
    tuple: (params_opt, energy, n_evaluations, wall_time_s)
    """
    from scipy.optimize import minimize

    eval_count = [0]

    def cost_fn(params):
        eval_count[0] += 1
        return evaluate_energy_statevector(circuit, hamiltonian, params)

    t0 = time.time()
    result = minimize(
        cost_fn,
        initial_guess,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.5},
    )
    elapsed = time.time() - t0

    return result.x, result.fun, eval_count[0], elapsed


# ── Noisy Optimizers ─────────────────────────────────────────────────────────


def noisy_cost_function(circuit, hamiltonian, n_shots=1024):
    """Create a cost function that simulates shot noise.

    Adds Gaussian noise with std = 1/sqrt(n_shots) to the exact energy,
    mimicking the statistical uncertainty from finite measurements.

    Returns
    -------
    tuple: (cost_fn, eval_count_ref)
        cost_fn: callable(params) -> float
        eval_count_ref: list[int] — mutable counter
    """
    from qiskit.primitives import StatevectorEstimator

    estimator = StatevectorEstimator()
    eval_count = [0]
    shot_noise_std = 1.0 / np.sqrt(n_shots)

    def cost_fn(params):
        eval_count[0] += 1
        bound = circuit.assign_parameters(params)
        job = estimator.run([(bound, hamiltonian)])
        exact_energy = float(job.result()[0].data.evs)
        return exact_energy + np.random.normal(0, shot_noise_std)

    return cost_fn, eval_count


def run_spsa(
    cost_fn,
    initial_guess,
    n_iterations=200,
    a=0.1,
    c=0.1,
    alpha=0.602,
    gamma=0.101,
    A_frac=0.1,
):
    """Run SPSA optimizer.

    Standard SPSA with gain sequences:
        a_k = a / (k + 1 + A)^alpha
        c_k = c / (k + 1)^gamma

    Parameters
    ----------
    cost_fn : callable
        Noisy cost function.
    initial_guess : np.ndarray
        Starting parameters.
    n_iterations : int
        Number of SPSA iterations.
    a, c, alpha, gamma : float
        SPSA hyperparameters.
    A_frac : float
        Stability constant as fraction of n_iterations.

    Returns
    -------
    tuple: (best_theta, best_energy, n_evals, wall_time_s)
    """
    n_params = len(initial_guess)
    theta = initial_guess.copy()
    A = n_iterations * A_frac

    t0 = time.time()
    best_energy = cost_fn(theta)
    best_theta = theta.copy()
    n_evals = 1

    for k in range(n_iterations):
        a_k = a / (k + 1 + A) ** alpha
        c_k = c / (k + 1) ** gamma

        # Bernoulli perturbation
        delta = 2 * np.random.randint(0, 2, n_params) - 1

        # Two-sided gradient estimate
        theta_plus = theta + c_k * delta
        theta_minus = theta - c_k * delta

        y_plus = cost_fn(theta_plus)
        y_minus = cost_fn(theta_minus)
        n_evals += 2

        # Gradient estimate
        g_hat = (y_plus - y_minus) / (2 * c_k * delta)

        # Update
        theta = theta - a_k * g_hat
        theta = np.clip(theta, -np.pi, np.pi)

        # Track best
        min_y = min(y_plus, y_minus)
        if min_y < best_energy:
            best_energy = min_y
            best_theta = theta_plus.copy() if y_plus < y_minus else theta_minus.copy()

    # Final evaluation
    final_energy = cost_fn(theta)
    n_evals += 1
    if final_energy < best_energy:
        best_energy = final_energy
        best_theta = theta.copy()

    elapsed = time.time() - t0
    return best_theta, best_energy, n_evals, elapsed


def run_cobyla_noisy(circuit, hamiltonian, initial_guess, n_shots=4096, maxiter=300):
    """Run COBYLA on noisy cost function.

    Returns
    -------
    tuple: (params_opt, energy, n_evaluations, wall_time_s)
    """
    cost_fn, eval_count = noisy_cost_function(circuit, hamiltonian, n_shots)

    from scipy.optimize import minimize

    t0 = time.time()
    result = minimize(
        cost_fn,
        initial_guess,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.3},
    )
    elapsed = time.time() - t0

    return result.x, result.fun, eval_count[0], elapsed


def run_nevergrad(circuit, hamiltonian, initial_guess, optimizer_name, budget=500):
    """Run a Nevergrad optimizer for VQE.

    Returns
    -------
    tuple: (params_opt, energy, n_evaluations, wall_time_s)
    """
    import nevergrad as ng

    def cost_fn(params):
        return evaluate_energy_statevector(circuit, hamiltonian, np.array(params))

    parametrization = ng.p.Array(init=initial_guess).set_bounds(-np.pi, np.pi)
    optimizer = ng.optimizers.registry[optimizer_name](
        parametrization=parametrization, budget=budget
    )

    t0 = time.time()
    recommendation = optimizer.minimize(cost_fn)
    elapsed = time.time() - t0

    params_opt = recommendation.value
    energy = cost_fn(params_opt)

    return params_opt, energy, budget, elapsed


# ── Training Data Generation ─────────────────────────────────────────────────


def vqe_descending_sweep(
    circuit,
    h_values,
    builder,
    solver,
    N: int,
    initial_guess=None,
    noiseless: bool = True,
    n_shots: int = 4096,
    maxiter: int = 1000,
):
    """Generate VQE training data via descending h-sweep.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized HVA circuit.
    h_values : array-like
        h-values (will be sorted descending internally).
    builder : HamiltonianBuilder
    solver : ClassicalSolver
    N : int
        Number of qubits.
    initial_guess : np.ndarray | None
        Starting parameters. If None, uses small random.
    noiseless : bool
        If True, uses L-BFGS-B. If False, uses COBYLA with shot noise.
    n_shots : int
        Shot count for noisy mode.
    maxiter : int
        Max optimizer iterations.

    Returns
    -------
    dict with keys:
        h_values: sorted ascending
        theta_opt: array (n_points, n_params)
        energies: array (n_points,)
        exact_energies: array (n_points,)
        fidelities: array (n_points,) — only for noiseless
        wall_time_s: float
    """
    from experiment_utils import compute_fidelity

    h_sorted = np.sort(h_values)[::-1]  # descending
    n_params = circuit.num_parameters

    if initial_guess is None:
        initial_guess = np.random.uniform(-0.01, 0.01, n_params)

    theta_results = []
    energies = []
    exact_energies = []
    fidelities = []
    prev_theta = initial_guess.copy()

    t0 = time.time()
    for h in h_sorted:
        lattice = make_lattice("chain_1d", N, J=1.0, h=h)
        H = builder.build(lattice)
        exact = solver.solve(H, lattice)

        if noiseless:
            theta_opt, energy, _, _ = run_cobyla_noiseless(circuit, H, prev_theta, maxiter=maxiter)
            # Also try L-BFGS-B and keep best
            theta_lb, energy_lb, _, _ = run_lbfgsb(circuit, H, prev_theta, maxiter=maxiter)[:4]
            if energy_lb < energy:
                theta_opt, energy = theta_lb, energy_lb
        else:
            cost_fn, _ = noisy_cost_function(circuit, H, n_shots)
            from scipy.optimize import minimize

            result = minimize(
                cost_fn,
                prev_theta,
                method="COBYLA",
                options={"maxiter": maxiter, "rhobeg": 0.3},
            )
            theta_opt, energy = result.x, result.fun

        theta_results.append(theta_opt)
        energies.append(energy)
        exact_energies.append(exact.ground_energy)

        if noiseless and exact.ground_state is not None:
            fid = compute_fidelity(circuit, theta_opt, exact.ground_state)
            fidelities.append(fid)
        else:
            fidelities.append(None)

        prev_theta = theta_opt.copy()

    elapsed = time.time() - t0

    # Reverse back to ascending order
    theta_results = list(reversed(theta_results))
    energies = list(reversed(energies))
    exact_energies = list(reversed(exact_energies))
    fidelities = list(reversed(fidelities))
    h_ascending = np.sort(h_values)

    return {
        "h_values": h_ascending,
        "theta_opt": np.array(theta_results),
        "energies": np.array(energies),
        "exact_energies": np.array(exact_energies),
        "fidelities": np.array([f if f is not None else 0.0 for f in fidelities]),
        "wall_time_s": elapsed,
    }


# ── MPNN Training with Best-State Tracking ───────────────────────────────


def train_mpnn_with_best_state(
    model,
    dataset,
    n_epochs: int = 3000,
    lr: float = 1e-3,
    patience: int = 200,
    restore_best: bool = True,
):
    """Train MPNN and optionally restore best model state.

    Wraps src.poc.v6.mpnn_predictor.train_mpnn with best-state tracking.
    This prevents training overshoot from degrading the final model,
    which is especially important with noisy training data (experiments 5B/5C/5D).

    Parameters
    ----------
    model : MPNNPredictor
    dataset : list[Data]
    n_epochs : int
    lr : float
    patience : int
    restore_best : bool
        If True, restore model weights from the epoch with lowest MSE.

    Returns
    -------
    dict — same as train_mpnn() result, plus 'best_epoch' key.
    """
    import torch

    from src.poc.v6.mpnn_predictor import train_mpnn

    result = train_mpnn(model, dataset, n_epochs=n_epochs, lr=lr, patience=patience)

    if restore_best and result["mse_history"]:
        best_epoch = int(np.argmin(result["mse_history"]))
        best_mse = result["mse_history"][best_epoch]

        # If final MSE is worse than best, retrain to best epoch
        # (Only worth it if difference is significant)
        if result["final_mse"] > best_mse * 1.05:
            # Retrain to best epoch (cheaper than storing all states)
            model_class = model.__class__
            fresh_model = model_class(
                node_features=model.node_features,
                hidden_dim=model.hidden_dim,
                n_layers=model.n_layers,
                output_dim=model.output_dim,
                per_parameter_heads=model.per_parameter_heads,
                use_edge_features=model.use_edge_features,
            )
            # Copy initial weights and retrain to best_epoch
            torch.manual_seed(42)
            result2 = train_mpnn(
                fresh_model,
                dataset,
                n_epochs=best_epoch + 1,
                lr=lr,
                patience=patience,
            )
            # Copy state back to original model
            model.load_state_dict(fresh_model.state_dict())
            result["final_mse"] = result2["final_mse"]
            result["restored_to_best"] = True

        result["best_epoch"] = best_epoch
    else:
        result["best_epoch"] = len(result.get("mse_history", [])) - 1

    return result


def clear_exact_cache():
    """Clear the ground truth cache (useful between different N values)."""
    _EXACT_CACHE.clear()
