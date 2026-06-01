"""S3: Landscape Analysis at N=20 (F3 + B4 Extension).

Hypothesis: The energy landscape at N=20 is qualitatively different from
N=6/N=10 — higher condition numbers and/or more local minima explain why
G3 (1 restart + freeze) failed at N=20.

Method:
    1. F3-style fluctuation analysis at N=20 p=2 using MPS backend
    2. B4-style Hessian analysis at VQE minima (from V7 3C data)
    3. Compare κ(N=20) vs κ(N=6) and κ(N=10)
    4. Count distinct minima found across restarts

Expected outcome:
    κ(N=20) >> κ(N=6) at h=2.0 (flatter landscape), explaining why
    1 restart is insufficient. Alternatively, multiple distinct local
    minima may exist at N=20 that don't exist at N=6.

Thesis value: MEDIUM-HIGH — explains G3 negative result quantitatively.

References:
    - B4: 0 saddle points at N=6 and N=10, κ N-independent
    - G3: 1 restart + freeze FAILS at N=20 (ΔE/gap=1.26)
    - F3: fluctuation >1.0 everywhere at N=6 (no barren plateaus)
"""

from __future__ import annotations

import logging
import time

import numpy as np

from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import (
    AnalysisConfig,
    ExperimentConfig,
    SystemConfig,
    VQEConfig,
)
from qmbp_simulation.framework.metrics import ExperimentMetrics

logger = logging.getLogger(__name__)


class ExperimentS3(BaseExperiment):
    """Landscape analysis at N=20: fluctuation + Hessian."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="S3",
            category="S",
            description="Landscape analysis at N=20 — explains G3 failure",
            hypothesis=(
                "κ(N=20) >> κ(N=6) at h=2.0, or multiple distinct local minima "
                "exist at N=20 that don't exist at smaller N."
            ),
            system=SystemConfig(
                n_qubits=20,
                p_layers=2,
                h_values=[2.0, 1.75, 1.5],
            ),
            vqe=VQEConfig(n_restarts=7, maxiter=300, sigma=0.3),
            analysis=AnalysisConfig(
                fluctuation_n_samples=50,
                compute_hessian=True,
            ),
            seeds=[42, 43, 44],
            verbose=True,
            auto_warm_cold_comparison=False,
        )

    def setup(self) -> None:
        """Setup MPS backend for N=20."""
        from qiskit_aer import AerSimulator

        from qmbp_simulation import (
            ClassicalSolver,
            HamiltonianBuilder,
            HVACircuitBuilder,
            make_lattice,
        )

        self._setup_logging()
        self.builder = HamiltonianBuilder()
        self.solver = ClassicalSolver()
        self.hva = HVACircuitBuilder()

        N = self.config.system.n_qubits
        p = self.config.system.p_layers
        lattice = make_lattice("chain_1d", N, J=1.0, h=1.0)
        self.circuit, _ = self.hva.create(N, p, lattice)

        self._mps_sim = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=64,
            matrix_product_state_truncation_threshold=1e-12,
        )
        logger.info("S3 setup: N=%d, p=%d, backend=MPS(chi=64)", N, p)

    def _evaluate_mps(self, params, H) -> float:
        """Evaluate energy via MPS."""
        from qiskit.quantum_info import Statevector

        bound = self.circuit.assign_parameters(params)
        bound_save = bound.copy()
        bound_save.save_statevector()
        result = self._mps_sim.run(bound_save).result()
        sv = Statevector(result.get_statevector())
        return float(sv.expectation_value(H).real)

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Run landscape analysis: fluctuation + Hessian + multi-restart."""
        from scipy.optimize import minimize

        from qmbp_simulation import make_lattice

        np.random.seed(seed)
        N = self.config.system.n_qubits
        n_params = self.circuit.num_parameters
        h_values = self.config.system.h_values
        n_samples = self.config.analysis.fluctuation_n_samples
        metrics = []

        for h in sorted(h_values, reverse=True):
            t0 = time.time()
            lattice = make_lattice("chain_1d", N, J=1.0, h=h)
            H = self.builder.build(lattice)
            result = self.solver.solve(H, lattice)
            exact_energy = result.ground_energy
            gap = result.gap if result.gap > 1e-10 else max(2 * abs(1 - h), 2 * np.pi / N)

            # --- Fluctuation analysis (F3-style) ---
            energies = []
            for _ in range(n_samples):
                theta_rand = np.random.uniform(-np.pi, np.pi, n_params)
                e = self._evaluate_mps(theta_rand, H)
                energies.append(e)

            energies = np.array(energies)
            e_mean = np.mean(energies)
            e_var = np.var(energies)
            fluctuation = e_var / (e_mean**2) if abs(e_mean) > 1e-10 else 0.0
            fraction_near_gs = np.mean(energies < exact_energy + gap)

            # --- Multi-restart VQE (find distinct minima) ---
            minima_energies = []
            minima_thetas = []
            prev_theta = np.random.uniform(-0.01, 0.01, n_params)

            for r in range(self.config.vqe.n_restarts):
                x0 = (
                    prev_theta.copy()
                    if r == 0
                    else (
                        minima_thetas[0] + np.random.normal(0, self.config.vqe.sigma, n_params)
                        if minima_thetas
                        else np.random.uniform(-0.5, 0.5, n_params)
                    )
                )
                x0 = np.clip(x0, -np.pi, np.pi)

                res = minimize(
                    lambda p, _H=H: self._evaluate_mps(p, _H),
                    x0,
                    method="L-BFGS-B",
                    bounds=[(-np.pi, np.pi)] * n_params,
                    options={"maxiter": self.config.vqe.maxiter, "ftol": 1e-14},
                )
                minima_energies.append(res.fun)
                minima_thetas.append(res.x.copy())

            # Count distinct minima (energy difference > gap/10)
            sorted_energies = sorted(minima_energies)
            distinct_minima = 1
            for i in range(1, len(sorted_energies)):
                if abs(sorted_energies[i] - sorted_energies[i - 1]) > gap / 10:
                    distinct_minima += 1

            best_energy = min(minima_energies)
            best_idx = minima_energies.index(best_energy)
            best_theta = minima_thetas[best_idx]
            de_gap = abs(best_energy - exact_energy) / gap

            # --- Hessian at best minimum ---
            hessian_eigenvalues = None
            condition_number = None
            min_type = "unknown"
            if self.config.analysis.compute_hessian:
                hess = self._compute_hessian(best_theta, H, epsilon=5e-3)
                eigenvalues = np.linalg.eigvalsh(hess)
                hessian_eigenvalues = eigenvalues.tolist()
                condition_number = float(max(abs(eigenvalues)) / max(min(abs(eigenvalues)), 1e-10))
                min_type = "minimum" if all(eigenvalues > -1e-6) else "saddle"

            elapsed = time.time() - t0

            m = ExperimentMetrics(
                h_value=h,
                energy=best_energy,
                exact_energy=exact_energy,
                energy_error=abs(best_energy - exact_energy),
                gap=gap,
                relative_error=de_gap,
                seed=seed,
                wall_time_s=elapsed,
                converged=de_gap < 0.05,
                technique_metadata={
                    "N": N,
                    "fluctuation": float(fluctuation),
                    "fraction_near_gs": float(fraction_near_gs),
                    "n_distinct_minima": distinct_minima,
                    "n_restarts": self.config.vqe.n_restarts,
                    "min_type": min_type,
                    "condition_number": condition_number,
                    "hessian_eigenvalues": hessian_eigenvalues,
                    "energy_spread": float(max(minima_energies) - min(minima_energies)),
                },
            )
            metrics.append(m)
            logger.info(
                "  h=%.2f: fluct=%.3f, frac_gs=%.3f, κ=%s, distinct=%d, ΔE/gap=%.4f (%.0fs)",
                h,
                fluctuation,
                fraction_near_gs,
                f"{condition_number:.0f}" if condition_number else "N/A",
                distinct_minima,
                de_gap,
                elapsed,
            )

        return metrics

    def _compute_hessian(self, theta: np.ndarray, H, epsilon: float = 5e-3) -> np.ndarray:
        """Compute Hessian via central finite differences."""
        n = len(theta)
        hess = np.zeros((n, n))
        e0 = self._evaluate_mps(theta, H)

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    theta_p = theta.copy()
                    theta_m = theta.copy()
                    theta_p[i] += epsilon
                    theta_m[i] -= epsilon
                    hess[i, i] = (
                        self._evaluate_mps(theta_p, H) - 2 * e0 + self._evaluate_mps(theta_m, H)
                    ) / epsilon**2
                else:
                    theta_pp = theta.copy()
                    theta_pm = theta.copy()
                    theta_mp = theta.copy()
                    theta_mm = theta.copy()
                    theta_pp[i] += epsilon
                    theta_pp[j] += epsilon
                    theta_pm[i] += epsilon
                    theta_pm[j] -= epsilon
                    theta_mp[i] -= epsilon
                    theta_mp[j] += epsilon
                    theta_mm[i] -= epsilon
                    theta_mm[j] -= epsilon
                    hess[i, j] = (
                        self._evaluate_mps(theta_pp, H)
                        - self._evaluate_mps(theta_pm, H)
                        - self._evaluate_mps(theta_mp, H)
                        + self._evaluate_mps(theta_mm, H)
                    ) / (4 * epsilon**2)
                    hess[j, i] = hess[i, j]

        return hess
