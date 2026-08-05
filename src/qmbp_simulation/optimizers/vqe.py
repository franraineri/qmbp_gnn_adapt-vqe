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
import time
from collections.abc import Callable

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
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
from qmbp_simulation.models.constants import (
    COBYLA_AUTO_SWITCH_THRESHOLD,
    VQE_WALL_CLOCK_LIMIT_PER_POINT,
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

        # ── Workaround: Qiskit mimalloc GC deadlock on macOS ARM64 ────────
        # Qiskit's Rust-accelerated CircuitData uses mimalloc which can deadlock
        # during Python's garbage collection when freeing circuit objects. This
        # manifests as the process hanging in sleep() inside
        # _mi_arenas_page_unabandon. Disabling GC during VQE optimization
        # prevents this. GC is re-enabled on exit (always, via finally).
        import gc as _gc
        _gc_was_enabled = _gc.isenabled()
        _gc.disable()

        try:
            return self._optimize_inner(
                cfg, backend, hamiltonian, circuit, initial_guess,
                exact_energy, exact_state,
            )
        finally:
            if _gc_was_enabled:
                _gc.enable()

    def _optimize_inner(
        self,
        cfg: VQEConfig,
        backend,
        hamiltonian: SparsePauliOp,
        circuit: QuantumCircuit,
        initial_guess: np.ndarray,
        exact_energy: float | None,
        exact_state: np.ndarray | None,
    ) -> VQEResult:
        """Core optimization logic (called with GC disabled)."""
        logger.debug(
            "VQEOptimizer.optimize: n_params=%d, method=%s, maxiter=%d, "
            "n_restarts=%d, exact_energy=%s",
            len(initial_guess),
            cfg.method,
            cfg.maxiter,
            cfg.n_restarts,
            f"{exact_energy:.6f}" if exact_energy is not None else "None",
        )

        def cost_fn(params: np.ndarray) -> float:
            """Pure energy cost function — evaluates ⟨H⟩ only."""
            return backend.evaluate(circuit, hamiltonian, params)

        # Auto-select optimizer method based on backend type.
        # Noiseless/MPS backends have smooth, deterministic landscapes where
        # gradient-based methods (L-BFGS-B) converge 10-100x faster than COBYLA.
        # Noisy/hardware backends have stochastic evaluations where FD gradients
        # are unreliable — COBYLA is safer there.
        effective_method = cfg.method
        n_params = len(initial_guess)
        backend_name = self._backend.name

        _is_noiseless = ("noiseless" in backend_name or "mps" in backend_name)

        if _is_noiseless and cfg.method == "COBYLA" and n_params > 4:
            effective_method = "L-BFGS-B"
            logger.info(
                f"    ⚙️ Auto-upgraded COBYLA → L-BFGS-B "
                f"(noiseless backend '{backend_name}', n_params={n_params})"
            )
        elif not _is_noiseless and cfg.method == "L-BFGS-B" and n_params > COBYLA_AUTO_SWITCH_THRESHOLD:
            effective_method = "COBYLA"
            logger.info(
                f"    ⚙️ Auto-switched L-BFGS-B → COBYLA "
                f"(noisy backend '{backend_name}', n_params={n_params})"
            )

        # Create effective config with possibly overridden method.
        # For L-BFGS-B: each iteration costs (2n+1) FD evals. With n=48 and
        # multi-hour runs when users pass large values (designed for COBYLA).
        from dataclasses import replace

        effective_maxiter = cfg.maxiter
        if effective_method == "L-BFGS-B" and n_params > 10:
            # L-BFGS-B converges in 5-20 iters with warm-start.
            # Cap at 50 to prevent accidental multi-hour runs.
            max_lbfgsb_iters = 50
            if cfg.maxiter > max_lbfgsb_iters:
                effective_maxiter = max_lbfgsb_iters
                logger.info(
                    f"    ⚙️ Capped maxiter {cfg.maxiter} → {max_lbfgsb_iters} "
                    f"(L-BFGS-B iters cost {2*n_params+1} evals each)"
                )

        effective_cfg = replace(cfg, method=effective_method, maxiter=effective_maxiter)

        # Emit immediately visible progress info
        print(
            f"    [VQE] method={effective_cfg.method}, maxiter={effective_cfg.maxiter}, "
            f"n_params={n_params}, restarts={effective_cfg.n_restarts}",
            flush=True,
        )

        # Lightweight progress callback: logs every N iterations so the
        # user sees activity during long-running VQE calls (even when
        # enable_callbacks=False). Does NOT re-evaluate cost_fn (zero overhead).
        _iter_count = [0]
        # For high-dimensional params (expensive FD gradient), log every iteration.
        # For low-dimensional, log every 25 to avoid spam.
        _progress_interval = 25

        def _progress_callback(xk: np.ndarray) -> None:
            _iter_count[0] += 1
            if _iter_count[0] % _progress_interval == 0:
                print(
                    f"    [VQE] iter={_iter_count[0]}/{effective_cfg.maxiter}",
                    flush=True,
                )

        # Set up callback if enabled
        callback = OptimizationCallback(cost_fn) if cfg.enable_callbacks else None

        # Use progress callback when full callbacks are disabled and
        # parameter space is high-dimensional (potentially long-running).
        progress_cb = _progress_callback if (not cfg.enable_callbacks and n_params > 4) else None

        # Record initial energy in callback so trajectory is never empty
        if callback is not None:
            initial_energy = cost_fn(initial_guess)
            callback.energies.append(float(initial_energy))
            callback.param_vectors.append(initial_guess.copy())
            callback.grad_norms.append(float("nan"))

        from qmbp_simulation.models.constants import (
            VQE_RESTART_IMPROVEMENT_TOL,
            VQE_RESTART_STAGNATION_THRESHOLD,
        )

        # Effective callback for scipy: full trajectory OR lightweight progress
        effective_cb = callback or progress_cb

        bounds = [cfg.bounds] * len(initial_guess)

        # Warm-start run
        best = self._run_minimize(cost_fn, initial_guess, effective_cfg, bounds, effective_cb)

        n_restarts_used = 0
        _consecutive_no_improvement = 0

        # Random restarts with stagnation detection
        for restart_idx in range(effective_cfg.n_restarts):
            # Early-stop: if N consecutive restarts failed to improve,
            # the landscape is well-explored — skip remaining restarts.
            if _consecutive_no_improvement >= VQE_RESTART_STAGNATION_THRESHOLD:
                logger.debug(
                    f"VQE early-stop: {_consecutive_no_improvement} consecutive restarts "
                    f"without improvement (threshold={VQE_RESTART_STAGNATION_THRESHOLD}). "
                    f"Skipping remaining {effective_cfg.n_restarts - restart_idx} restarts."
                )
                break

            logger.info(
                f"    VQE restart {restart_idx + 1}/{effective_cfg.n_restarts}, "
                f"best_E={best.fun:.6f}"
            )
            for handler in logging.getLogger().handlers:
                handler.flush()

            x0 = best.x + self._rng.normal(0, cfg.restart_sigma, len(best.x))
            # Clip to bounds
            x0 = np.clip(x0, cfg.bounds[0], cfg.bounds[1])
            # Reset iter counter for each restart
            if progress_cb is not None:
                _iter_count[0] = 0
            trial = self._run_minimize(cost_fn, x0, effective_cfg, bounds, effective_cb)
            if not trial.success:
                logger.debug(
                    f"VQE restart {restart_idx + 1}/{effective_cfg.n_restarts} did not converge: "
                    f"{trial.message}"
                )
            # Stagnation detection: meaningful improvement resets counter
            improvement = best.fun - trial.fun  # positive = trial is better
            if improvement > VQE_RESTART_IMPROVEMENT_TOL:
                _consecutive_no_improvement = 0
            else:
                _consecutive_no_improvement += 1
            # Always track the absolute best energy (exact comparison)
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
            # Variational principle check at optimize level (covers non-sweep usage)
            if np.isfinite(best.fun) and best.fun < exact_energy - 1e-8:
                violation = exact_energy - best.fun
                if violation >= 0.1:
                    logger.error(
                        "❌ CRITICAL variational principle violation in optimize(): "
                        "E_VQE=%.8f < E_exact=%.8f (Δ=%.4e). "
                        "Check Hamiltonian/backend consistency.",
                        best.fun,
                        exact_energy,
                        violation,
                    )
                elif violation >= 1e-3:
                    logger.warning(
                        "⚠️  Variational principle violation in optimize(): "
                        "E_VQE=%.8f < E_exact=%.8f (Δ=%.2e).",
                        best.fun,
                        exact_energy,
                        violation,
                    )
        if exact_state is not None:
            try:
                fidelity = float(backend.compute_fidelity(circuit, best.x, exact_state))
            except Exception as exc:
                logger.warning(
                    "compute_fidelity failed (N=%d): %s. Setting fidelity=0.0.",
                    circuit.num_qubits,
                    exc,
                )
                fidelity = 0.0
            # Fidelity sanity guard (compute_fidelity should clip, but double-check)
            if fidelity > 1.0 + 1e-6:
                logger.error(
                    "Fidelity %.8f > 1.0 — this should not happen. "
                    "Possible bug in compute_fidelity or exact_state normalization.",
                    fidelity,
                )
                fidelity = 1.0
            elif fidelity > 1.0:
                fidelity = 1.0

        # Build trajectory
        trajectory = None
        if callback is not None:
            converged = best.success
            trajectory = callback.to_trajectory(converged, n_restarts_used)

        # Compute energy variance: Var(H) = ⟨H²⟩ - ⟨H⟩²
        # This quantifies how close |ψ(θ_opt)⟩ is to an eigenstate of H.
        energy_variance = None
        try:
            energy_variance = self._backend.compute_energy_variance(circuit, hamiltonian, best.x)
        except Exception as e:
            logger.debug("Energy variance computation failed: %s", e)

        # Error prevention: unexpected NaN for small N on exact backends
        if energy_variance is not None and np.isnan(energy_variance):
            from qmbp_simulation.models.constants import STATEVECTOR_MAX_N

            n = circuit.num_qubits
            if n <= STATEVECTOR_MAX_N and "noiseless" in self._backend.name:
                logger.error(
                    "UNEXPECTED: energy_variance is NaN for N=%d on %s. "
                    "This indicates a bug in compute_energy_variance or the backend.",
                    n,
                    self._backend.name,
                )

        # Error prevention: inconsistency between variance and de_gap
        if (
            energy_variance is not None
            and np.isfinite(energy_variance)
            and energy_variance < 0.001
            and energy_error > 0
            and exact_energy is not None
        ):
            gap_estimate = max(abs(exact_energy) * 0.01, 0.1)  # conservative gap estimate
            de_gap_estimate = energy_error / gap_estimate
            if de_gap_estimate > 0.10:
                logger.warning(
                    "Inconsistency: Var(H)≈%.2e (near-eigenstate) but energy_error=%.4f. "
                    "Possible: excited eigenstate, incorrect E_exact, or gap mismatch.",
                    energy_variance,
                    energy_error,
                )

        return VQEResult(
            h_value=0.0,  # Set by caller in sweep
            theta_opt=best.x.copy(),
            energy=float(best.fun),
            energy_error=energy_error,
            fidelity=fidelity,
            n_iterations=int(getattr(best, "nit", getattr(best, "nfev", 0))),
            trajectory=trajectory,
            energy_variance=energy_variance,
        )

    # ── optimizer dispatch ──────────────────────────────────────────

    @staticmethod
    def _run_minimize(cost_fn, x0, cfg, bounds, callback):
        """Dispatch to the correct scipy.optimize.minimize method.

        - L-BFGS-B: gradient-based, uses bounds and ftol
        - COBYLA: gradient-free, no bounds/ftol (TypeError if passed)
        - Nelder-Mead: simplex, no bounds, uses fatol

        All methods have a function-evaluation cap to guarantee termination
        even when convergence tolerance is unreachable (e.g., flat landscapes
        near the critical point with high-dimensional parameter spaces).

        The cap formula is: maxfun = maxiter * min(n_params + 5, 50).
        For a typical case (n_params=4, maxiter=500): maxfun = 4500.
        For high-dim (n_params=79, maxiter=500): maxfun = 25000.
        """
        method = cfg.method
        n_params = len(x0)
        # Unified maxfun cap formula: generous but finite.
        maxfun = cfg.maxiter * min(n_params + 5, 50)

        if method == "L-BFGS-B":
            # Signal-aware wrapper: L-BFGS-B also runs in compiled Fortran
            # and won't respond to Ctrl+C until control returns to Python.
            import signal as _signal

            _interrupted = [False]
            _orig_handler = _signal.getsignal(_signal.SIGINT)

            def _interrupt_handler(signum, frame):
                _interrupted[0] = True

            _signal.signal(_signal.SIGINT, _interrupt_handler)

            _eval_count = [0]

            def _interruptible_cost(params):
                _eval_count[0] += 1
                if _interrupted[0]:
                    _signal.signal(_signal.SIGINT, _orig_handler)
                    raise KeyboardInterrupt("L-BFGS-B interrupted by user (Ctrl+C)")
                return cost_fn(params)

            try:
                result = minimize(
                    _interruptible_cost,
                    x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    callback=callback,
                    options={
                        "maxiter": cfg.maxiter,
                        "ftol": cfg.ftol,
                        "maxfun": maxfun,
                    },
                )
            finally:
                _signal.signal(_signal.SIGINT, _orig_handler)
            return result
        elif method == "COBYLA":
            # COBYLA does not support maxfev in options — wrap cost_fn with
            # an evaluation counter AND wall-clock timeout that force termination.
            _eval_count = [0]
            _last_value = [0.0]
            _start_time = time.monotonic()
            # Wall-clock cap: use VQE_WALL_CLOCK_LIMIT_PER_POINT for a single
            # optimize() call (restart included separately by the caller).
            # Divide by (n_restarts + 1) to keep total per-point time bounded.
            _timeout_s = VQE_WALL_CLOCK_LIMIT_PER_POINT / max(cfg.n_restarts + 1, 1)

            # Signal-aware interruption: COBYLA runs in Fortran and doesn't
            # yield to Python signal handlers. We periodically check for pending
            # interrupts by calling PyErr_CheckSignals (via signal.pthread_sigmask
            # is not portable; instead we use a lightweight try/except pattern).
            import signal as _signal

            _interrupted = [False]
            _orig_handler = _signal.getsignal(_signal.SIGINT)

            def _interrupt_handler(signum, frame):
                """Set flag so COBYLA cost function can raise on next eval."""
                _interrupted[0] = True

            _signal.signal(_signal.SIGINT, _interrupt_handler)

            def _capped_cost(params):
                _eval_count[0] += 1
                # Interrupt check: raise KeyboardInterrupt from within the cost
                # function — this is the only way to escape COBYLA's Fortran loop.
                if _interrupted[0]:
                    _signal.signal(_signal.SIGINT, _orig_handler)
                    raise KeyboardInterrupt(
                        "COBYLA interrupted by user (Ctrl+C)"
                    )
                # Eval cap
                if _eval_count[0] > maxfun:
                    return _last_value[0]
                # Wall-clock cap
                if time.monotonic() - _start_time > _timeout_s:
                    return _last_value[0]
                val = cost_fn(params)
                _last_value[0] = val
                return val

            try:
                result = minimize(
                    _capped_cost,
                    x0,
                    method="COBYLA",
                    callback=callback,
                    options={
                        "maxiter": cfg.maxiter,
                        "rhobeg": cfg.restart_sigma,
                        "catol": 1e-10,
                    },
                )
            finally:
                # Always restore original handler
                _signal.signal(_signal.SIGINT, _orig_handler)
            # Annotate result with actual eval count for diagnostics
            result.nfev = _eval_count[0]
            elapsed = time.monotonic() - _start_time
            if elapsed > _timeout_s * 0.9:
                logger.debug(
                    "COBYLA hit wall-clock cap (%.1fs / %.1fs limit, %d evals)",
                    elapsed,
                    _timeout_s,
                    _eval_count[0],
                )
            return result
        elif method == "Nelder-Mead":
            return minimize(
                cost_fn,
                x0,
                method="Nelder-Mead",
                callback=callback,
                options={
                    "maxiter": cfg.maxiter,
                    "maxfev": maxfun,
                    "fatol": cfg.ftol,
                    "xatol": 1e-10,
                },
            )
        else:
            # Should not reach here — VQEConfig validates method
            raise ValueError(f"Unsupported method: {method}")

    # ── warm-start seeding ───────────────────────────────────────────

    @staticmethod
    def compute_adaptive_restarts(
        h_value: float,
        gap: float | None = None,
        n_restarts_base: int = 5,
        *,
        h_critical: float = 1.0,
        gap_easy_threshold: float = 0.5,
        gap_hard_threshold: float = 0.1,
    ) -> int:
        """Compute adaptive number of restarts based on landscape difficulty.

        Near the critical point (small gap), the energy landscape is flat
        and multi-restart search is essential. Far from criticality (large gap),
        the warm-start converges immediately and extra restarts are wasted.

        This implements the observation from B4 experiment: condition number
        varies 14→1400 across the h-range, but NO saddle points exist.
        More restarts help at high κ (small gap), not at low κ (easy regime).

        Parameters
        ----------
        h_value : float
            Current transverse field value.
        gap : float | None
            Spectral gap at this h-value (from exact diag). If None,
            uses h-value heuristic only.
        n_restarts_base : int
            Configured n_restarts (the maximum; adaptive reduces from here).
        h_critical : float
            Estimated critical h (TFIM: ~1.0). Used for h-distance heuristic.
        gap_easy_threshold : float
            Gap above this = easy landscape → reduce restarts to 1.
        gap_hard_threshold : float
            Gap below this = hard landscape → use full n_restarts_base.

        Returns
        -------
        int
            Adaptive number of restarts (1 ≤ result ≤ n_restarts_base).
        """
        if n_restarts_base <= 1:
            return 1

        # If gap is available, use it directly (most reliable signal)
        if gap is not None and gap > 0:
            if gap >= gap_easy_threshold:
                # Easy: paramagnetic limit, warm-start always converges
                return 1
            elif gap >= gap_hard_threshold:
                # Medium: interpolate between 1 and base
                frac = (gap - gap_hard_threshold) / (gap_easy_threshold - gap_hard_threshold)
                return max(1, int(round(n_restarts_base * (1 - frac * 0.8))))
            else:
                # Hard: near/at critical point, use full budget
                return n_restarts_base

        # Fallback: h-distance heuristic (when gap is not available)
        h_distance = abs(h_value - h_critical)
        if h_distance > 1.0:
            # Far from critical: very easy landscape
            return max(1, n_restarts_base // 3)
        elif h_distance > 0.3:
            # Moderate distance
            return max(1, n_restarts_base // 2)
        else:
            # Near critical: full budget
            return n_restarts_base

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
        adaptive: bool = False,
    ) -> list[VQEResult]:
        """Run VQE across h-values in descending order (h=2→0), propagating
        θ_opt as warm-start.

        Parameters
        ----------
        h_values : np.ndarray — h-values (must be in descending order)
        circuit : QuantumCircuit — parameterized HVA
        lattice : LatticeConfig — lattice specification
        exact_data : list[GroundTruthResult] | None — exact results per h-point
        adaptive : bool
            If True, use adaptive restart allocation from sweep_strategies:
            more restarts near h_critical or when previous point had high ΔE/gap,
            fewer restarts for easy (paramagnetic) points. Default False
            preserves existing fixed-restart behavior.

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

            # Notify cached backends of current h-value for key generation
            if hasattr(self._backend, "set_h"):
                self._backend.set_h(h)

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

            # ── Adaptive restart allocation ──
            # When adaptive=True, vary n_restarts based on neighbor's convergence
            # quality and proximity to h_critical. Easy points (deep paramagnetic)
            # get fewer restarts; hard points (near QPT) get more.
            if adaptive:
                from dataclasses import replace as _replace

                from qmbp_simulation.optimizers.sweep_strategies import (
                    AdaptiveRestartConfig,
                    compute_adaptive_restarts,
                )

                # Determine previous point's ΔE/gap (neighbor signal)
                prev_de_gap = None
                if idx > 0 and exact_data is not None and idx - 1 < len(exact_data):
                    prev_e_exact = exact_data[idx - 1].ground_energy
                    prev_gap = exact_data[idx - 1].gap
                    prev_vqe_e = results[idx - 1].energy
                    if prev_gap > 1e-10:
                        prev_de_gap = abs(prev_vqe_e - prev_e_exact) / prev_gap

                adaptive_cfg = AdaptiveRestartConfig(
                    base_restarts=max(1, self.config.n_restarts // 3),
                    max_restarts=self.config.n_restarts,
                    critical_restarts=self.config.n_restarts,
                    h_critical=1.0,  # TFIM QPT
                )
                n_restarts_adaptive = compute_adaptive_restarts(
                    h, prev_de_gap=prev_de_gap, config=adaptive_cfg
                )

                # Temporarily override config for this point
                original_config = self.config
                self.config = _replace(self.config, n_restarts=n_restarts_adaptive)

            t_point_start = time.perf_counter()
            result = self.optimize(
                H,
                circuit,
                current_guess,
                exact_energy=exact_e,
                exact_state=exact_psi,
            )

            # Restore original config after adaptive override
            if adaptive:
                self.config = original_config

            t_point_elapsed = time.perf_counter() - t_point_start
            result.h_value = h

            # Wall-clock timeout warning: detect when a single h-point
            # takes unreasonably long (likely COBYLA spinning on flat landscape)
            if t_point_elapsed > VQE_WALL_CLOCK_LIMIT_PER_POINT:
                logger.warning(
                    f"⏱️  VQE at h={h:.4f} took {t_point_elapsed:.1f}s "
                    f"(limit={VQE_WALL_CLOCK_LIMIT_PER_POINT:.0f}s). "
                    f"Possible flat landscape or excessive restarts. "
                    f"n_params={n_params}, method={self.config.method}"
                )

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
            # Severity escalation: small violations (< 0.01) are numerical noise
            # from eigsh vs statevector mismatch. Large violations (≥ 0.1) indicate
            # a real bug in the Hamiltonian, circuit, or solver.
            if (
                exact_e is not None
                and np.isfinite(result.energy)
                and result.energy < exact_e - 1e-8
            ):
                violation = exact_e - result.energy
                if violation >= 0.1:
                    logger.error(
                        f"❌ CRITICAL variational principle violation at h={h:.4f}: "
                        f"E_VQE={result.energy:.8f} < E_exact={exact_e:.8f} "
                        f"(Δ={violation:.4e}). This is NOT numerical noise — "
                        f"check Hamiltonian consistency between solver and VQE backend."
                    )
                elif violation >= 0.01:
                    logger.warning(
                        f"⚠️  Variational principle violated at h={h:.4f}: "
                        f"E_VQE={result.energy:.8f} < E_exact={exact_e:.8f} "
                        f"(Δ={violation:.2e}). Likely eigsh tolerance vs statevector "
                        f"mismatch for this system size."
                    )
                else:
                    logger.warning(
                        f"⚠️  Variational principle violated at h={h:.4f}: "
                        f"E_VQE={result.energy:.8f} < E_exact={exact_e:.8f} "
                        f"(Δ={violation:.2e}) — numerical noise (benign)."
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

    # ── Bidirectional sweep ──────────────────────────────────────────

    def bidirectional_sweep(
        self,
        h_values: np.ndarray,
        circuit: QuantumCircuit,
        lattice: LatticeConfig,
        exact_data: list[GroundTruthResult] | None = None,
        adaptive: bool = False,
    ) -> list[VQEResult]:
        """Run VQE with both descending and ascending sweeps, keeping the best.

        For each h-point, takes the result with lower energy from the two
        sweep directions. This eliminates warm-start propagation errors where
        a bad local minimum in one direction infects all downstream points.

        When ``adaptive=True``, uses selective ascending: only re-optimizes
        points where the descending pass had ΔE/gap > 2% (plus their
        neighbors). This typically saves 50-80% of ascending pass cost
        while recovering most of the improvements.

        Parameters
        ----------
        h_values : np.ndarray
            h-values in descending order (h_max → h_min).
        circuit : QuantumCircuit
            Parameterized HVA circuit.
        lattice : LatticeConfig
            Lattice specification.
        exact_data : list[GroundTruthResult] | None
            Exact results per h-point (same order as h_values).
        adaptive : bool
            If True, use selective ascending (only re-optimize suspicious
            points) and adaptive restart allocation. Default False preserves
            full bidirectional behavior.

        Returns
        -------
        list[VQEResult]
            Best results per h-point (same order as h_values).
        """
        logger.info(
            "  🔄 Bidirectional VQE sweep: %d h-points, %d params, topology=%s%s",
            len(h_values),
            circuit.num_parameters,
            lattice.topology,
            " (adaptive)" if adaptive else "",
        )
        logger.info("  Bidirectional sweep: running descending pass...")
        results_desc = self.descending_sweep(
            h_values, circuit, lattice, exact_data, adaptive=adaptive
        )

        # ── Selective vs Full ascending pass ──
        if adaptive and exact_data is not None:
            return self._selective_ascending_merge(
                h_values, circuit, lattice, exact_data, results_desc
            )

        # Full ascending pass (original behavior)
        h_ascending = h_values[::-1]
        exact_ascending = list(reversed(exact_data)) if exact_data else None

        logger.info("  Bidirectional sweep: running ascending pass...")
        results_asc_reversed = self._ascending_sweep(h_ascending, circuit, lattice, exact_ascending)
        # Reverse back to match original h-order
        results_asc = list(reversed(results_asc_reversed))

        # Merge: take lower energy at each point
        merged: list[VQEResult] = []
        n_improved = 0
        for i in range(len(h_values)):
            if results_asc[i].energy < results_desc[i].energy:
                merged.append(results_asc[i])
                n_improved += 1
            else:
                merged.append(results_desc[i])

        logger.info(
            f"  Bidirectional merge: {n_improved}/{len(h_values)} points "
            f"improved by ascending pass."
        )
        return merged

    def _selective_ascending_merge(
        self,
        h_values: np.ndarray,
        circuit: QuantumCircuit,
        lattice: LatticeConfig,
        exact_data: list[GroundTruthResult],
        results_desc: list[VQEResult],
    ) -> list[VQEResult]:
        """Selective ascending: only re-optimize suspicious points.

        Uses select_suspicious_points() from sweep_strategies to identify
        which h-points need re-optimization. Saves 50-80% compute vs full
        ascending pass while recovering most improvements.
        """
        from qmbp_simulation.optimizers.sweep_strategies import (
            SelectiveAscendingConfig,
            select_suspicious_points,
        )

        # Build results dicts for select_suspicious_points
        desc_for_selection = []
        for i, (r, gt) in enumerate(zip(results_desc, exact_data, strict=False)):
            gap = gt.gap if gt.gap and gt.gap > 1e-10 else 0.1
            de_gap = abs(r.energy - gt.ground_energy) / gap
            desc_for_selection.append({"h": float(h_values[i]), "de_gap": de_gap})

        config = SelectiveAscendingConfig(
            de_gap_threshold=0.02,
            include_neighbors=True,
            max_fraction=0.5,
        )
        targeted_indices, report = select_suspicious_points(desc_for_selection, config=config)

        if report.fell_back_to_full:
            logger.info(
                f"  Selective ascending: >50% suspicious → full ascending "
                f"({report.n_suspicious}/{report.n_total_points} suspicious)"
            )
            # Fall back to full ascending
            h_ascending = h_values[::-1]
            exact_ascending = list(reversed(exact_data))
            results_asc_reversed = self._ascending_sweep(
                h_ascending, circuit, lattice, exact_ascending
            )
            results_asc = list(reversed(results_asc_reversed))
            merged = []
            n_improved = 0
            for i in range(len(h_values)):
                if results_asc[i].energy < results_desc[i].energy:
                    merged.append(results_asc[i])
                    n_improved += 1
                else:
                    merged.append(results_desc[i])
            logger.info(f"  Full ascending merge: {n_improved}/{len(h_values)} improved.")
            return merged

        if not targeted_indices:
            logger.info("  Selective ascending: no suspicious points — skipping ascending pass.")
            return results_desc

        logger.info(
            f"  Selective ascending: {report.n_targeted}/{report.n_total_points} points "
            f"targeted ({report.n_suspicious} suspicious + neighbors)"
        )

        # Re-optimize only targeted points in ascending direction
        # Start from the lowest-h targeted point and sweep upward
        merged = list(results_desc)  # Start with descending results
        builder = HamiltonianBuilder()
        n_improved = 0

        # Sort targeted indices by h ascending (sweep direction)
        targeted_sorted = sorted(targeted_indices)  # Already in ascending order from selector

        # Use the best available θ as warm-start for ascending re-optimization
        for idx in targeted_sorted:
            h = float(h_values[idx])

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

            exact_e = exact_data[idx].ground_energy
            exact_psi = exact_data[idx].ground_state

            # Warm-start from the neighbor below (ascending direction)
            # Find the nearest completed point below this one
            warm_theta = results_desc[idx].theta_opt  # fallback to descending result
            for prev_idx in range(idx + 1, len(h_values)):
                if prev_idx in targeted_sorted or prev_idx >= len(merged):
                    continue
                warm_theta = merged[prev_idx].theta_opt
                break

            result_asc = self.optimize(
                H, circuit, warm_theta,
                exact_energy=exact_e,
                exact_state=exact_psi,
            )
            result_asc.h_value = h

            # Keep better energy
            if result_asc.energy < merged[idx].energy:
                merged[idx] = result_asc
                n_improved += 1

        logger.info(
            f"  Selective ascending: {n_improved}/{len(targeted_indices)} targeted points improved."
        )
        return merged

    def _ascending_sweep(
        self,
        h_values: np.ndarray,
        circuit: QuantumCircuit,
        lattice: LatticeConfig,
        exact_data: list[GroundTruthResult] | None = None,
    ) -> list[VQEResult]:
        """Internal ascending sweep (h_min → h_max) with warm-start.

        Same logic as descending_sweep but without the descending-order check.
        """
        builder = HamiltonianBuilder()
        n_params = circuit.num_parameters
        results: list[VQEResult] = []

        current_guess = self.get_initial_guess(n_params, h_values[0], self.config)

        for idx, h in enumerate(h_values):
            h = float(h)

            # Notify cached backends of current h-value for key generation
            if hasattr(self._backend, "set_h"):
                self._backend.set_h(h)

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

            exact_e = None
            exact_psi = None
            if exact_data is not None and idx < len(exact_data):
                exact_e = exact_data[idx].ground_energy
                exact_psi = exact_data[idx].ground_state

            current_guess = self.get_initial_guess(n_params, h, self.config, current_guess)

            t_point_start = time.perf_counter()
            result = self.optimize(
                H,
                circuit,
                current_guess,
                exact_energy=exact_e,
                exact_state=exact_psi,
            )
            t_point_elapsed = time.perf_counter() - t_point_start
            result.h_value = h

            if t_point_elapsed > VQE_WALL_CLOCK_LIMIT_PER_POINT:
                logger.warning(
                    f"⏱️  VQE ascending at h={h:.4f} took {t_point_elapsed:.1f}s "
                    f"(limit={VQE_WALL_CLOCK_LIMIT_PER_POINT:.0f}s). "
                    f"n_params={n_params}, method={self.config.method}"
                )

            if np.any(~np.isfinite(result.theta_opt)):
                result.theta_opt = current_guess.copy()
                result.energy = float("inf")
                result.fidelity = 0.0

            results.append(result)
            current_guess = result.theta_opt

        return results
