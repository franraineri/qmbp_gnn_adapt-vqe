"""
VQE Optimizer — Multi-start L-BFGS-B with diagnostic callbacks and trajectory
logging.

Implements the V4 descending sweep pattern (h=2→0) with warm-start propagation,
expanded bounds [-π, π], and full optimization trajectory recording for
diagnostic plots (Energy vs. Iterations, Parameter Trajectory).

References
----------
- V4 PoC: multi_start_vqe() pattern with warm-start propagation.
- Mele et al. (2026): shallow-circuit constraint (p ≤ 2).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
from qiskit.primitives import StatevectorEstimator

from .config import (
    VQEConfig,
    VQEResult,
    OptimizationTrajectory,
    GroundTruthResult,
)

logger = logging.getLogger(__name__)


# ── Task 5.1: OptimizationCallback ───────────────────────────────────────

class OptimizationCallback:
    """Logs energy, gradient norm, and parameter vector at every L-BFGS-B
    iteration.

    Used by injecting into ``scipy.optimize.minimize`` via the ``callback``
    parameter.  L-BFGS-B's callback receives only the current parameter
    vector, so we re-evaluate energy via the cost function reference.
    """

    def __init__(self, cost_fn: callable) -> None:
        self.cost_fn = cost_fn
        self.energies: list[float] = []
        self.grad_norms: list[float] = []
        self.param_vectors: list[np.ndarray] = []
        self._iteration = 0

    def __call__(self, xk: np.ndarray) -> None:
        """Called at every optimizer iteration with current parameters."""
        self._iteration += 1
        energy = float(self.cost_fn(xk))
        self.energies.append(energy)
        self.param_vectors.append(xk.copy())

        # Approximate gradient norm from energy change between iterations.
        # We avoid finite-difference gradient here because it costs n_params
        # extra circuit evaluations per iteration — too expensive for the
        # 27-point sweep.  Instead, use the energy drop as a proxy.
        if len(self.energies) >= 2:
            delta_e = abs(self.energies[-1] - self.energies[-2])
            self.grad_norms.append(delta_e)
        else:
            self.grad_norms.append(float("nan"))

    def to_trajectory(self, converged: bool, n_restarts_used: int) -> OptimizationTrajectory:
        """Convert logged data to an OptimizationTrajectory dataclass."""
        return OptimizationTrajectory(
            energies=self.energies,
            grad_norms=self.grad_norms,
            param_vectors=self.param_vectors,
            converged=converged,
            n_restarts_used=n_restarts_used,
        )


# ── Task 5.2–5.5: VQEOptimizer ───────────────────────────────────────────

class VQEOptimizer:
    """Optimize HVA parameters via multi-start L-BFGS-B with diagnostic
    callbacks and descending sweep orchestration.
    """

    def __init__(self, config: Optional[VQEConfig] = None) -> None:
        self.config = config or VQEConfig()
        self._estimator = StatevectorEstimator()

    # ── Task 5.2: optimize() ─────────────────────────────────────────

    def optimize(
        self,
        hamiltonian: SparsePauliOp,
        circuit: QuantumCircuit,
        initial_guess: np.ndarray,
        exact_energy: Optional[float] = None,
        exact_state: Optional[np.ndarray] = None,
    ) -> VQEResult:
        """Run multi-start L-BFGS-B VQE for a single h-point.

        Parameters
        ----------
        hamiltonian : SparsePauliOp
        circuit : QuantumCircuit (parameterized HVA)
        initial_guess : np.ndarray — warm-start parameters
        exact_energy : float | None — for computing energy_error
        exact_state : np.ndarray | None — for computing fidelity

        Returns
        -------
        VQEResult
        """
        cfg = self.config

        def cost_fn(params: np.ndarray) -> float:
            bound = circuit.assign_parameters(params)
            job = self._estimator.run([(bound, hamiltonian)])
            return float(job.result()[0].data.evs)

        # Set up callback if enabled
        callback = OptimizationCallback(cost_fn) if cfg.enable_callbacks else None

        bounds = [cfg.bounds] * len(initial_guess)

        # Warm-start run
        best = minimize(
            cost_fn, initial_guess,
            method="L-BFGS-B",
            bounds=bounds,
            callback=callback,
            options={"maxiter": cfg.maxiter, "ftol": cfg.ftol},
        )

        n_restarts_used = 0

        # Random restarts
        for _ in range(cfg.n_restarts):
            x0 = best.x + np.random.normal(0, cfg.restart_sigma, len(best.x))
            # Clip to bounds
            x0 = np.clip(x0, cfg.bounds[0], cfg.bounds[1])
            trial = minimize(
                cost_fn, x0,
                method="L-BFGS-B",
                bounds=bounds,
                callback=callback,
                options={"maxiter": cfg.maxiter, "ftol": cfg.ftol},
            )
            if trial.fun < best.fun:
                best = trial
                n_restarts_used += 1

        # Compute validation metrics
        energy_error = 0.0
        fidelity = 0.0
        if exact_energy is not None:
            energy_error = abs(best.fun - exact_energy)
        if exact_state is not None:
            sv_ansatz = Statevector(circuit.assign_parameters(best.x))
            fidelity = float(state_fidelity(sv_ansatz, Statevector(exact_state)))

        # Build trajectory
        trajectory = None
        if callback is not None:
            converged = best.success
            trajectory = callback.to_trajectory(converged, n_restarts_used)

        return VQEResult(
            h_value=0.0,  # Set by caller in sweep
            theta_opt=best.x.copy(),
            energy=float(best.fun),
            energy_error=energy_error,
            fidelity=fidelity,
            n_iterations=int(best.nit),
            trajectory=trajectory,
        )

    # ── Task 5.3: warm-start seeding ─────────────────────────────────

    @staticmethod
    def get_initial_guess(
        n_params: int,
        h_value: float,
        config: VQEConfig,
        previous_theta: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Determine the initial guess for a given h-point.

        Parameters
        ----------
        n_params : int — number of circuit parameters
        h_value : float — current transverse field value
        config : VQEConfig
        previous_theta : np.ndarray | None — θ_opt from previous h-point

        Returns
        -------
        np.ndarray — initial parameter guess
        """
        # Task 5.3: enforce θ=0 for h=0 when warm_start_seed_zeros=True
        if config.warm_start_seed_zeros and abs(h_value) < 1e-12:
            return np.zeros(n_params)

        # Warm-start from previous h-point if available
        if previous_theta is not None:
            return previous_theta.copy()

        # First point: small random perturbation
        return np.random.uniform(-0.01, 0.01, n_params)

    # ── Task 5.4: descending sweep ───────────────────────────────────

    def descending_sweep(
        self,
        h_values: np.ndarray,
        circuit: QuantumCircuit,
        lattice: "LatticeConfig",
        exact_data: Optional[list[GroundTruthResult]] = None,
    ) -> list[VQEResult]:
        """Run VQE across h-values in descending order (h=2→0), propagating
        θ_opt as warm-start.

        Parameters
        ----------
        h_values : np.ndarray — h-values in ascending order
        circuit : QuantumCircuit — parameterized HVA
        lattice : LatticeConfig — lattice specification
        exact_data : list[GroundTruthResult] | None — exact results per h-point

        Returns
        -------
        list[VQEResult] — results in ascending h order (matching h_values)
        """
        from .hamiltonian_builder import HamiltonianBuilder
        import copy

        builder = HamiltonianBuilder()
        n_params = circuit.num_parameters
        results_dict: dict[int, VQEResult] = {}

        current_guess = self.get_initial_guess(
            n_params, h_values[-1], self.config
        )

        # Sweep descending: from highest h to lowest
        for idx in reversed(range(len(h_values))):
            h = float(h_values[idx])

            # Build Hamiltonian for this h — use deep copy to avoid
            # sharing mutable fields (edges, coordination_numbers)
            lat_h = copy.deepcopy(lattice)
            lat_h.h = h
            H = builder.build(lat_h)

            # Get exact reference if available
            exact_e = None
            exact_psi = None
            if exact_data is not None and idx < len(exact_data):
                exact_e = exact_data[idx].ground_energy
                exact_psi = exact_data[idx].ground_state

            # Task 5.3: warm-start seeding
            current_guess = self.get_initial_guess(
                n_params, h, self.config, current_guess
            )

            result = self.optimize(
                H, circuit, current_guess,
                exact_energy=exact_e, exact_state=exact_psi,
            )
            result.h_value = h

            status = "✅" if result.fidelity >= 0.995 else "⚠️"
            logger.info(
                f"  {status} h={h:.2f}: fid={result.fidelity:.6f}, "
                f"ΔE={result.energy_error:.2e}, nit={result.n_iterations}"
            )

            results_dict[idx] = result
            # Propagate θ_opt as warm-start (no wrapping — V4 lesson)
            current_guess = result.theta_opt

        # Return in ascending h order
        return [results_dict[i] for i in range(len(h_values))]
