"""
SPSA Optimizer — Simultaneous Perturbation Stochastic Approximation.

Designed for noisy/hardware settings where gradient-free methods are needed.
Hyperparameters validated from V7 experiment 4A grid search:
optimal config is a=0.1, c=0.05, A_frac=0.05.

Algorithm
---------
For each iteration k:
    1. Generate random perturbation Δ (Bernoulli ±1)
    2. Evaluate f(θ + c_k*Δ) and f(θ - c_k*Δ)
    3. Estimate gradient: g_k = (f+ - f-) / (2*c_k*Δ)
    4. Update: θ = θ - a_k * g_k

Gain sequences:
    a_k = a / (k + 1 + A)^alpha
    c_k = c / (k + 1)^gamma
    A = A_frac * n_iterations

References
----------
- Spall (1998): SPSA algorithm.
- V7 Experiment 4A: Grid search over 36 configs × 10 seeds.
- V7 Experiment 4C: SPSA 3× better than COBYLA under shot noise.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qmbp_simulation.execution import ExecutionBackend
from qmbp_simulation.models import VQEResult

logger = logging.getLogger(__name__)


class SPSAOptimizer:
    """SPSA optimizer for noisy/hardware settings.

    Hyperparameters validated from V7 experiment 4A grid search:
    optimal config is a=0.1, c=0.05, A_frac=0.05.

    Parameters
    ----------
    backend : ExecutionBackend
        Execution backend for energy evaluation (noisy or hardware).
    a : float
        Initial step size gain (default 0.1).
    c : float
        Initial perturbation magnitude (default 0.1).
    alpha : float
        Step size decay exponent (default 0.602).
    gamma : float
        Perturbation decay exponent (default 0.101).
    A_frac : float
        Stability constant as fraction of n_iterations (default 0.1).
    bounds : tuple[float, float]
        Parameter bounds for clipping (default (-π, π)).
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        backend: ExecutionBackend,
        a: float = 0.1,
        c: float = 0.1,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A_frac: float = 0.1,
        bounds: tuple[float, float] = (-np.pi, np.pi),
        seed: int | None = None,
    ) -> None:
        self._backend = backend
        self.a = a
        self.c = c
        self.alpha = alpha
        self.gamma = gamma
        self.A_frac = A_frac
        self._bounds = bounds
        self._rng = np.random.default_rng(seed)

    def optimize(
        self,
        circuit: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        initial_guess: np.ndarray,
        n_iterations: int = 200,
    ) -> VQEResult:
        """Run SPSA optimization.

        Parameters
        ----------
        circuit : QuantumCircuit
            Parameterized HVA circuit.
        hamiltonian : SparsePauliOp
            Observable to minimize.
        initial_guess : np.ndarray
            Starting parameters.
        n_iterations : int
            Number of SPSA iterations (default 200).

        Returns
        -------
        VQEResult
            Optimization result with best parameters and energy.
        """
        n_params = len(initial_guess)
        theta = initial_guess.copy()
        A = n_iterations * self.A_frac

        t0 = time.time()

        # Initial evaluation
        best_energy = self._backend.evaluate(circuit, hamiltonian, theta)
        best_theta = theta.copy()
        n_evals = 1

        for k in range(n_iterations):
            # Gain sequences
            a_k = self.a / (k + 1 + A) ** self.alpha
            c_k = self.c / (k + 1) ** self.gamma

            # Bernoulli ±1 perturbation (seeded RNG)
            delta = 2 * self._rng.integers(0, 2, n_params) - 1

            # Two-sided function evaluations
            theta_plus = theta + c_k * delta
            theta_minus = theta - c_k * delta

            y_plus = self._backend.evaluate(circuit, hamiltonian, theta_plus)
            y_minus = self._backend.evaluate(circuit, hamiltonian, theta_minus)
            n_evals += 2

            # Gradient estimate
            g_hat = (y_plus - y_minus) / (2 * c_k * delta)

            # Parameter update with configurable bounds
            theta = theta - a_k * g_hat
            theta = np.clip(theta, self._bounds[0], self._bounds[1])

            # Track best energy found
            min_y = min(y_plus, y_minus)
            if min_y < best_energy:
                best_energy = min_y
                best_theta = theta_plus.copy() if y_plus < y_minus else theta_minus.copy()

        # Final evaluation at converged point
        final_energy = self._backend.evaluate(circuit, hamiltonian, theta)
        n_evals += 1
        if final_energy < best_energy:
            best_energy = final_energy
            best_theta = theta.copy()

        elapsed = time.time() - t0

        logger.info(
            f"SPSA completed: {n_iterations} iterations, "
            f"{n_evals} evaluations, "
            f"energy={best_energy:.6f}, "
            f"wall_time={elapsed:.2f}s"
        )

        return VQEResult(
            h_value=0.0,  # Set by caller in sweep context
            theta_opt=best_theta.copy(),
            energy=float(best_energy),
            energy_error=0.0,  # Caller sets if exact reference available
            fidelity=0.0,  # Caller sets if exact state available
            n_iterations=n_evals,
            trajectory=None,
        )
