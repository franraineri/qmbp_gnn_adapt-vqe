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
            system=SystemConfig(n_qubits=6, p_layers=2),
            analysis=AnalysisConfig(scaling_n_values=[4, 6, 8, 10, 14, 20]),
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
        """Find h_min for each N via binary search."""
        np.random.seed(seed)
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

    def _find_boundary(self, N: int, p: int, seed: int) -> float | None:
        """Binary search for h_min where DE/gap < 5%.

        Uses StatevectorEstimator for N<=10, AerSimulator MPS for N>=14.
        """
        from scipy.optimize import minimize

        from qmbp_simulation import HVACircuitBuilder, make_lattice

        hva = HVACircuitBuilder()
        base_lattice = make_lattice("chain_1d", N, J=1.0, h=1.0)
        qc, _ = hva.create(N, p, base_lattice)
        n_params = qc.num_parameters

        # Choose backend: MPS for N>=14, statevector for smaller
        use_mps = N >= 14
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

        # Optimized h-grid: start above expected h_min, stop well below
        h_predicted = 1.0 + 0.019 * N**1.33
        h_start = min(3.5, h_predicted + 0.8)
        h_stop = max(0.45, h_predicted - 0.6)  # Don't scan far below expected boundary
        h_test_points = np.arange(h_start, h_stop, -0.05)
        prev_theta = np.random.uniform(-0.01, 0.01, n_params)
        n_restarts = 5 if N >= 14 else 3  # More restarts for larger N
        boundary_h = None

        for h in h_test_points:
            lattice = make_lattice("chain_1d", N, J=1.0, h=h)
            H = self.builder.build(lattice)
            exact = self.solver.solve(H, lattice)

            # Multi-start optimization
            best_energy = float("inf")
            best_theta = prev_theta.copy()

            for restart in range(n_restarts):
                if restart == 0:
                    x0 = prev_theta.copy()
                else:
                    x0 = best_theta + np.random.normal(0, 0.1, n_params)
                    x0 = np.clip(x0, -np.pi, np.pi)
                result = minimize(
                    cost_fn,
                    x0,
                    method="L-BFGS-B",
                    args=(H,),
                    bounds=[(-np.pi, np.pi)] * n_params,
                    options={"maxiter": 300, "ftol": 1e-12},
                )
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_theta = result.x.copy()

            # Compute DE/gap
            gap = exact.gap if exact.gap > 1e-10 else max(2 * abs(1.0 - h), 2 * np.pi / N)
            de_gap = abs(best_energy - exact.ground_energy) / gap
            prev_theta = best_theta.copy()

            if de_gap < 0.05:
                boundary_h = h
            else:
                if boundary_h is not None:
                    break

        return boundary_h
