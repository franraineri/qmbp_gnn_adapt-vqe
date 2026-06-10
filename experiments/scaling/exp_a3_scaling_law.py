"""A3: Finite-Size Scaling of the Valid Regime Boundary.

Hypothesis: The valid regime boundary h_min(N) follows a power law
    h_min = h_c + alpha * N^beta
that can be extracted from N=4,6,8,10 data and connected to
known TFIM critical exponents (nu=1 for 1D).

Method:
    For each N, find h_min where DE/gap first drops below 5% using
    binary search on a fine h-grid. Then fit the scaling law.

Expected outcome:
    beta ~ 0.8-1.2 (consistent with TFIM universality class).

Thesis value: HIGH — connects pipeline performance to known physics.

Performance Optimizations (2026-05-27):
    The original implementation was prohibitively slow for N>=12 (~25+ min)
    due to: (1) statevector backend scaling as 2^N, (2) fine linear h-scan
    over ~25 points, (3) fixed 3 restarts everywhere.

    Applied optimizations:
    1. MPS backend for N>=12 (not just N>=14). MPS is exact for 1D HVA
       (proven in V7 exp 3A/3B: |MPS-SV|=1e-14, chi=64 sufficient).
       Avoids 2^N scaling, uses O(N·chi²) instead.
    2. Bisection search: Use the known scaling law prediction to start near
       the expected boundary, then bisect with ~8-10 evaluations instead of
       25-30. Falls back to linear scan if bisection fails.
    3. Adaptive restarts: 1 restart far from boundary (trivial landscape),
       3 near boundary, 5 for N>=14. B4 proved no saddle points exist.
    4. Adaptive maxiter: 100 far from boundary (converges in <20 iters),
       300 near boundary where precision matters.
    5. Aggressive warm-start: track best theta per-h and reuse nearest
       evaluated point during bisection.

    Expected speedup: ~25-50x combined → N=12 in ~1-2 min, N=14 in ~3-5 min.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import AnalysisConfig, ExperimentConfig, SystemConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics


class ExperimentA3(BaseExperiment):
    """Finite-size scaling of the valid regime boundary."""

    logger = logging.getLogger(__name__)

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="A3",
            category="A",
            description="Finite-size scaling law for valid regime boundary h_min(N)",
            hypothesis=(
                "h_min(N) = h_c + alpha * N^(beta) with beta ~ 0.5-1.0 "
                "(TFIM universality class, nu=1 in 1D)"
            ),
            system=SystemConfig(
                n_qubits=6,
                p_layers=2,
                h_values=[],  # A3 generates its own h-grid internally per N
                h_test=[],  # No MPNN deployment — boundary search only
            ),
            analysis=AnalysisConfig(scaling_n_values=[4, 6, 8, 10]),
            seeds=[42, 43, 44],
            verbose=True,
        )

    def setup(self) -> None:
        """Override: we don't build a single circuit — we build per-N."""
        from qmbp_simulation import (
            ClassicalSolver,
            HamiltonianBuilder,
            HVACircuitBuilder,
        )

        self._setup_logging()
        self.builder = HamiltonianBuilder()
        self.solver = ClassicalSolver()
        self.hva = HVACircuitBuilder()
        self.logger = logging.getLogger(__name__)
        self.logger.info("A3 setup: will test N=%s", self.config.analysis.scaling_n_values)

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Find h_min for each N via descending h-scan."""
        N_values = self.config.analysis.scaling_n_values
        p = self.config.system.p_layers
        metrics = []

        for N in N_values:
            t0 = time.time()
            h_min = self._find_boundary(N, p, seed)
            elapsed = time.time() - t0

            m = ExperimentMetrics(
                h_value=h_min if h_min else 0.0,
                energy=0.0,
                exact_energy=0.0,
                energy_error=0.0,
                gap=1.0,
                relative_error=0.0,
                seed=seed,
                wall_time_s=elapsed,
                n_evaluations=0,
                converged=h_min is not None,
                technique_metadata={
                    "N": N,
                    "p": p,
                    "h_min": h_min,
                    "boundary_found": h_min is not None,
                },
            )
            metrics.append(m)
            if h_min:
                self.logger.info(f"  N={N}: h_min={h_min:.3f} ({elapsed:.1f}s)")

        return metrics

    def _make_cost_fn(self, N: int, p: int, qc):
        """Create the cost function with the appropriate backend.

        Backend selection (optimized 2026-05-27):
        - N <= 10: StatevectorEstimator (exact, fast for small Hilbert spaces)
        - N >= 12: AerSimulator MPS (exact for 1D HVA, chi=64 sufficient)
          Proven in V7 exp 3A/3B: |MPS - Statevector| = 1e-14.
          Avoids 2^N memory/time scaling, uses O(N·chi²) instead.
          N=12: MPS ~4x faster than statevector (4096 dim avoided).
          N=14: MPS ~16x faster (16384 dim avoided).
        """
        use_mps = N >= 12
        if use_mps:
            from qiskit.quantum_info import Statevector
            from qiskit_aer import AerSimulator

            mps_sim = AerSimulator(
                method="matrix_product_state",
                matrix_product_state_max_bond_dimension=64,
                matrix_product_state_truncation_threshold=1e-12,
            )
            self.logger.info(f"    N={N}: using MPS backend (chi=64)")

            def cost_fn(params, H):
                bound = qc.assign_parameters(params)
                bound_save = bound.copy()
                bound_save.save_statevector()
                result = mps_sim.run(bound_save).result()
                sv = Statevector(result.get_statevector())
                return float(sv.expectation_value(H).real)
        else:
            from qiskit.primitives import StatevectorEstimator

            estimator = StatevectorEstimator()

            def cost_fn(params, H):
                bound = qc.assign_parameters(params)
                job = estimator.run([(bound, H)])
                return float(job.result()[0].data.evs)

        return cost_fn

    def _predict_boundary(self, N: int) -> float:
        """Predict h_min using the established scaling law.

        Uses topology-dependent calibrated formulas. For chain_1d, the
        prediction is very accurate (R²=1.0000 from V8 results).
        """
        topology = self.config.system.topology
        if topology == "chain_1d":
            # Original fit at N=4-20 (exact diag): 1.0 + 0.020*N^1.31
            # For N>30 (MPS regime), add +0.50 offset: use run_scaling_validation.py
            return 1.0 + 0.020 * N**1.31
        elif topology == "ladder":
            return 1.0 + 0.04 * N**1.33
        else:
            return 1.0 + 0.06 * N**1.33

    def _adaptive_restarts(self, N: int, h: float, h_predicted: float) -> int:
        """Determine number of VQE restarts based on proximity to boundary.

        Optimization rationale (B4 result): ALL VQE minima are genuine
        (0 saddle points in HVA landscape). Far from boundary, the landscape
        is trivial and 1 restart suffices. Near boundary, use more restarts
        for precision.
        """
        distance = abs(h - h_predicted)
        if N >= 14:
            # Larger systems need more restarts near boundary
            return 5 if distance < 0.2 else 3 if distance < 0.5 else 2
        elif N >= 12:
            return 4 if distance < 0.2 else 2 if distance < 0.5 else 1
        else:
            # N <= 10: original behavior (fast enough)
            return 3

    def _adaptive_maxiter(self, h: float, h_predicted: float) -> int:
        """Determine L-BFGS-B maxiter based on proximity to boundary.

        Far from boundary (h >> h_min), VQE converges in <20 iterations.
        Near boundary, full 300 iterations may be needed for precision.
        """
        distance = abs(h - h_predicted)
        if distance > 0.5:
            return 100
        elif distance > 0.2:
            return 200
        else:
            return 300

    def _evaluate_at_h(
        self,
        h: float,
        N: int,
        p: int,
        cost_fn,
        prev_theta: np.ndarray,
        rng: np.random.Generator,
        h_predicted: float,
    ) -> tuple[float, np.ndarray]:
        """Run VQE at a single h-point and return (de_gap, best_theta).

        Uses adaptive restarts and maxiter based on distance to predicted
        boundary. Warm-starts from prev_theta.
        """
        from scipy.optimize import minimize

        from qmbp_simulation import make_lattice

        topology = self.config.system.topology
        n_params = len(prev_theta)

        lattice = make_lattice(topology, N, J=1.0, h=h)
        H = self.builder.build(lattice)
        exact = self.solver.solve(H, lattice)

        n_restarts = self._adaptive_restarts(N, h, h_predicted)
        maxiter = self._adaptive_maxiter(h, h_predicted)

        # Multi-start optimization
        best_energy = float("inf")
        best_theta = prev_theta.copy()

        for restart in range(n_restarts):
            if restart == 0:
                x0 = prev_theta.copy()
            else:
                x0 = best_theta + rng.normal(0, 0.1, n_params)
                x0 = np.clip(x0, -np.pi, np.pi)
            result = minimize(
                cost_fn,
                x0,
                method="L-BFGS-B",
                args=(H,),
                bounds=[(-np.pi, np.pi)] * n_params,
                options={"maxiter": maxiter, "ftol": 1e-12},
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_theta = result.x.copy()

        # Compute DE/gap
        gap = exact.gap if exact.gap > 1e-10 else max(2 * abs(1.0 - h), 2 * np.pi / N)
        de_gap = abs(best_energy - exact.ground_energy) / gap

        return de_gap, best_theta

    def _find_boundary_bisection(
        self,
        N: int,
        p: int,
        seed: int,
        cost_fn,
        n_params: int,
    ) -> float | None:
        """Find h_min using bisection around the predicted boundary.

        Strategy:
        1. Use scaling law to predict h_min.
        2. Verify one point above (should pass) and one below (should fail).
        3. Bisect to resolution of 0.05 (matching original grid).

        Falls back to linear scan if bisection assumptions fail
        (e.g., for new topologies where prediction is inaccurate).
        """
        h_predicted = self._predict_boundary(N)
        rng = np.random.default_rng(seed)
        prev_theta = rng.uniform(-0.01, 0.01, n_params)

        # Define search bracket: [h_low, h_high]
        # h_high should be in valid regime (de_gap < 5%)
        # h_low should be in invalid regime (de_gap >= 5%)
        h_high = min(5.0, h_predicted + 0.8)
        h_low = max(0.5, h_predicted - 0.6)

        # First, warm-start by descending from h_high to near the boundary
        # This gives good theta initialization (critical for accuracy)
        warmup_points = np.arange(h_high, h_predicted + 0.1, -0.15)
        for h in warmup_points:
            _, prev_theta = self._evaluate_at_h(
                h,
                N,
                p,
                cost_fn,
                prev_theta,
                rng,
                h_predicted,
            )

        # Verify bracket: h_high should pass, h_low should fail
        de_high, theta_high = self._evaluate_at_h(
            h_predicted + 0.3,
            N,
            p,
            cost_fn,
            prev_theta,
            rng,
            h_predicted,
        )
        de_low, theta_low = self._evaluate_at_h(
            h_predicted - 0.3,
            N,
            p,
            cost_fn,
            prev_theta,
            rng,
            h_predicted,
        )

        if de_high >= 0.05:
            # Prediction too low — boundary is higher than expected
            # Fall back to linear scan from higher h
            self.logger.info(
                f"    N={N}: bisection bracket failed (de_high={de_high:.3f}), "
                f"falling back to linear scan"
            )
            return self._find_boundary_linear(N, p, seed, cost_fn, n_params)

        if de_low < 0.05:
            # Prediction too high — boundary is lower than expected
            # Shift bracket down
            h_high = h_predicted - 0.3
            h_low = max(0.5, h_predicted - 1.0)
            theta_high = theta_low
            # Re-evaluate at new h_low
            de_low, theta_low = self._evaluate_at_h(
                h_low,
                N,
                p,
                cost_fn,
                theta_high,
                rng,
                h_predicted,
            )
            if de_low < 0.05:
                # Still passing — boundary is very low
                return self._find_boundary_linear(N, p, seed, cost_fn, n_params)

        # Bisection: find h where de_gap crosses 5%
        # Use theta from the high side (valid regime) for warm-start
        h_a = h_predicted - 0.3 if de_low >= 0.05 else h_low
        h_b = h_predicted + 0.3 if de_high < 0.05 else h_high
        theta_best = theta_high.copy()

        resolution = 0.05  # Match original grid resolution
        n_bisect_steps = 0
        max_bisect = 10  # At most 10 bisection steps

        while (h_b - h_a) > resolution and n_bisect_steps < max_bisect:
            h_mid = (h_a + h_b) / 2.0
            de_mid, theta_mid = self._evaluate_at_h(
                h_mid,
                N,
                p,
                cost_fn,
                theta_best,
                rng,
                h_predicted,
            )
            n_bisect_steps += 1

            if de_mid < 0.05:
                # h_mid is in valid regime — boundary is lower
                h_b = h_mid
                theta_best = theta_mid.copy()
            else:
                # h_mid is in invalid regime — boundary is higher
                h_a = h_mid

        # Return the lowest h that passes (h_b after bisection)
        # Round to nearest 0.05 for consistency with original results
        boundary = round(h_b / 0.05) * 0.05
        self.logger.info(
            f"    N={N}: bisection found boundary at h={boundary:.2f} ({n_bisect_steps} steps)"
        )
        return boundary

    def _find_boundary_linear(
        self,
        N: int,
        p: int,
        seed: int,
        cost_fn,
        n_params: int,
    ) -> float | None:
        """Fallback: linear descending scan (original algorithm).

        Used when bisection bracket fails (e.g., new topologies where
        the scaling law prediction is inaccurate). Still uses adaptive
        restarts and maxiter for speedup.
        """
        h_predicted = self._predict_boundary(N)

        h_start = min(5.0, h_predicted + 1.0)
        h_stop = max(0.45, h_predicted - 0.8)
        h_test_points = np.arange(h_start, h_stop, -0.05)
        rng = np.random.default_rng(seed)
        prev_theta = rng.uniform(-0.01, 0.01, n_params)
        boundary_h = None

        for h in h_test_points:
            de_gap, prev_theta = self._evaluate_at_h(
                h,
                N,
                p,
                cost_fn,
                prev_theta,
                rng,
                h_predicted,
            )

            if de_gap < 0.05:
                boundary_h = h
            else:
                if boundary_h is not None:
                    break

        return boundary_h

    def _find_boundary(self, N: int, p: int, seed: int) -> float | None:
        """Find h_min where DE/gap < 5% using optimized bisection.

        For N <= 10 (fast), uses the original linear scan for backward
        compatibility with established results. For N >= 12, uses the
        optimized bisection approach.

        Backend selection:
        - N <= 10: StatevectorEstimator (exact, fast)
        - N >= 12: AerSimulator MPS (exact for 1D, O(N·chi²) scaling)
        """
        from qmbp_simulation import HVACircuitBuilder, make_lattice

        topology = self.config.system.topology
        hva = HVACircuitBuilder()
        base_lattice = make_lattice(topology, N, J=1.0, h=1.0)
        qc, _ = hva.create(N, p, base_lattice)
        n_params = qc.num_parameters

        cost_fn = self._make_cost_fn(N, p, qc)

        # For N <= 10, use original linear scan (fast, backward-compatible)
        # For N >= 12, use optimized bisection (25-50x speedup)
        if N <= 10:
            return self._find_boundary_linear(N, p, seed, cost_fn, n_params)
        else:
            return self._find_boundary_bisection(N, p, seed, cost_fn, n_params)
