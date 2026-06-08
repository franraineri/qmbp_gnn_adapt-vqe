"""VQE Optimizer — Multi-start L-BFGS-B with diagnostic callbacks and trajectory
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
from collections.abc import Callable

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
from scipy.optimize import minimize

from qmbp_simulation.execution import ExecutionBackend, NoiselessBackend
from qmbp_simulation.models import (
    GroundTruthResult,
    HamiltonianBuilder,
    LatticeConfig,
    OptimizationTrajectory,
    VQEConfig,
    VQEResult,
)

logger = logging.getLogger(__name__)


# ── OptimizationCallback ─────────────────────────────────────────────────


class OptimizationCallback:
    """Logs energy, gradient norm, and parameter vector at every L-BFGS-B
    iteration.

    Used by injecting into ``scipy.optimize.minimize`` via the ``callback``
    parameter.  L-BFGS-B's callback receives only the current parameter
    vector, so we re-evaluate energy via the cost function reference.
    """

    def __init__(self, cost_fn: Callable[..., float]) -> None:
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

        # Approximate convergence rate from energy change between iterations.
        # NOTE: This is |ΔE| (energy drop), NOT the true gradient norm.
        # Used as a convergence proxy — true gradient would cost n_params
        # extra circuit evaluations per iteration.
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


# ── VQEOptimizer ─────────────────────────────────────────────────────────


class VQEOptimizer:
    """Optimize HVA parameters via multi-start L-BFGS-B with diagnostic
    callbacks and descending sweep orchestration.

    Parameters
    ----------
    config : VQEConfig | None
        VQE configuration. Defaults to ``VQEConfig()`` if not provided.
    backend : ExecutionBackend | None
        Execution backend for energy evaluation. Defaults to ``NoiselessBackend()``.
    """

    def __init__(
        self,
        config: VQEConfig | None = None,
        backend: ExecutionBackend | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or VQEConfig()
        self._backend = backend or NoiselessBackend()
        # Seeded RNG for restart perturbations — ensures reproducibility
        # independent of global numpy state.
        self._rng = np.random.default_rng(seed)

    # ── optimize() ───────────────────────────────────────────────────

    def optimize(
        self,
        hamiltonian: SparsePauliOp,
        circuit: QuantumCircuit,
        initial_guess: np.ndarray,
        exact_energy: float | None = None,
        exact_state: np.ndarray | None = None,
    ) -> VQEResult:
        """Run multi-start VQE for a single h-point.

        Uses a pure energy cost function (⟨H⟩ only — never hybrid/observable).
        Optimizer method is determined by ``config.method``:
        - ``"L-BFGS-B"``: gradient-based with bounds (default, for exact backends)
        - ``"COBYLA"``: gradient-free (for shot-based/noisy backends)
        - ``"Nelder-Mead"``: simplex method (alternative gradient-free)

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
        backend = self._backend

        def cost_fn(params: np.ndarray) -> float:
            """Pure energy cost function — evaluates ⟨H⟩ only."""
            return backend.evaluate(circuit, hamiltonian, params)

        # Set up callback if enabled
        callback = OptimizationCallback(cost_fn) if cfg.enable_callbacks else None

        # Record initial energy in callback so trajectory is never empty
        if callback is not None:
            initial_energy = cost_fn(initial_guess)
            callback.energies.append(float(initial_energy))
            callback.param_vectors.append(initial_guess.copy())
            callback.grad_norms.append(float("nan"))

        bounds = [cfg.bounds] * len(initial_guess)

        # Warm-start run
        best = self._run_minimize(cost_fn, initial_guess, cfg, bounds, callback)

        n_restarts_used = 0

        # Random restarts
        for restart_idx in range(cfg.n_restarts):
            x0 = best.x + self._rng.normal(0, cfg.restart_sigma, len(best.x))
            # Clip to bounds
            x0 = np.clip(x0, cfg.bounds[0], cfg.bounds[1])
            trial = self._run_minimize(cost_fn, x0, cfg, bounds, callback)
            if not trial.success:
                logger.debug(
                    f"VQE restart {restart_idx + 1}/{cfg.n_restarts} did not converge: "
                    f"{trial.message}"
                )
            if trial.fun < best.fun:
                best = trial
                n_restarts_used += 1

        # Log convergence status
        if not best.success:
            n_iters = getattr(best, "nit", getattr(best, "nfev", "?"))
            logger.warning(
                f"VQE best result did not converge (nit={n_iters}, "
                f"message='{best.message}'). Energy={best.fun:.6f}"
            )

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
            n_iterations=int(getattr(best, "nit", getattr(best, "nfev", 0))),
            trajectory=trajectory,
        )

    # ── optimizer dispatch ──────────────────────────────────────────

    @staticmethod
    def _run_minimize(cost_fn, x0, cfg, bounds, callback):
        """Dispatch to the correct scipy.optimize.minimize method.

        - L-BFGS-B: gradient-based, uses bounds and ftol
        - COBYLA: gradient-free, no bounds/ftol (TypeError if passed)
        - Nelder-Mead: simplex, no bounds, uses fatol
        """
        method = cfg.method
        if method == "L-BFGS-B":
            return minimize(
                cost_fn,
                x0,
                method="L-BFGS-B",
                bounds=bounds,
                callback=callback,
                options={"maxiter": cfg.maxiter, "ftol": cfg.ftol},
            )
        elif method == "COBYLA":
            return minimize(
                cost_fn,
                x0,
                method="COBYLA",
                callback=callback,
                options={"maxiter": cfg.maxiter, "rhobeg": cfg.restart_sigma},
            )
        elif method == "Nelder-Mead":
            return minimize(
                cost_fn,
                x0,
                method="Nelder-Mead",
                callback=callback,
                options={"maxiter": cfg.maxiter, "fatol": cfg.ftol},
            )
        else:
            # Should not reach here — VQEConfig validates method
            raise ValueError(f"Unsupported method: {method}")

    # ── warm-start seeding ───────────────────────────────────────────

    @staticmethod
    def get_initial_guess(
        n_params: int,
        h_value: float,
        config: VQEConfig,
        previous_theta: np.ndarray | None = None,
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
        # Warm-start from previous h-point if available (always preferred)
        if previous_theta is not None:
            return previous_theta.copy()

        # Enforce θ=0 for h=0 when warm_start_seed_zeros=True and no
        # previous theta is available (first point in sweep at h=0)
        if config.warm_start_seed_zeros and abs(h_value) < 1e-12:
            return np.zeros(n_params)

        # First point: small random perturbation
        return np.random.uniform(-0.01, 0.01, n_params)

    # ── descending sweep ─────────────────────────────────────────────

    def descending_sweep(
        self,
        h_values: np.ndarray,
        circuit: QuantumCircuit,
        lattice: LatticeConfig,
        exact_data: list[GroundTruthResult] | None = None,
    ) -> list[VQEResult]:
        """Run VQE across h-values in descending order (h=2→0), propagating
        θ_opt as warm-start.

        Parameters
        ----------
        h_values : np.ndarray — h-values (must be in descending order)
        circuit : QuantumCircuit — parameterized HVA
        lattice : LatticeConfig — lattice specification
        exact_data : list[GroundTruthResult] | None — exact results per h-point

        Returns
        -------
        list[VQEResult] — results in same order as h_values

        Raises
        ------
        ValueError
            If h_values are in ascending order (violates V4 lesson:
            ascending breaks θ smoothness).
        """
        # Validate descending order
        if len(h_values) == 0:
            raise ValueError("h_values cannot be empty.")
        if len(h_values) >= 2 and h_values[0] < h_values[-1]:
            raise ValueError(
                "h_values must be in descending order (h=2→0). "
                "Ascending sweep breaks θ smoothness (V4 lesson). "
                f"Got h_values[0]={h_values[0]}, h_values[-1]={h_values[-1]}."
            )

        builder = HamiltonianBuilder()
        n_params = circuit.num_parameters
        results: list[VQEResult] = []

        current_guess = self.get_initial_guess(n_params, h_values[0], self.config)

        # Sweep in provided order (must be descending)
        for idx, h in enumerate(h_values):
            h = float(h)

            # Build Hamiltonian for this h — reuse lattice structure,
            # only update the scalar h field
            lat_h = LatticeConfig(
                topology=lattice.topology,
                n_qubits=lattice.n_qubits,
                J=lattice.J,
                h=h,
                edges=lattice.edges,
                coordination_numbers=lattice.coordination_numbers,
                periodic=lattice.periodic,
            )
            H = builder.build(lat_h)

            # Get exact reference if available
            exact_e = None
            exact_psi = None
            if exact_data is not None and idx < len(exact_data):
                exact_e = exact_data[idx].ground_energy
                exact_psi = exact_data[idx].ground_state

            # Warm-start seeding
            current_guess = self.get_initial_guess(n_params, h, self.config, current_guess)

            result = self.optimize(
                H,
                circuit,
                current_guess,
                exact_energy=exact_e,
                exact_state=exact_psi,
            )
            result.h_value = h

            # NaN detection: catch corrupted optimization early
            if np.any(~np.isfinite(result.theta_opt)):
                logger.error(
                    f"NaN/Inf detected in θ_opt at h={h:.4f}. "
                    f"Warm-start chain corrupted. Attempting recovery from previous good θ."
                )
                # Recovery: use the last known good theta (current_guess before this iteration)
                # rather than random init, to preserve warm-start continuity
                result.theta_opt = current_guess.copy()
                result.energy = float("inf")  # Mark as unreliable
                result.fidelity = 0.0

            # Variational principle check (lightweight, per-point)
            if (
                exact_e is not None
                and np.isfinite(result.energy)
                and result.energy < exact_e - 1e-8
            ):
                logger.warning(
                    f"⚠️  Variational principle violated at h={h:.4f}: "
                    f"E_VQE={result.energy:.8f} < E_exact={exact_e:.8f} "
                    f"(Δ={exact_e - result.energy:.2e})"
                )

            status = "✅" if result.fidelity >= 0.995 else "⚠️"
            logger.info(
                f"  {status} h={h:.2f}: fid={result.fidelity:.6f}, "
                f"ΔE={result.energy_error:.2e}, nit={result.n_iterations}"
            )

            results.append(result)
            # Propagate θ_opt as warm-start (no wrapping — V4 lesson)
            current_guess = result.theta_opt

        return results
