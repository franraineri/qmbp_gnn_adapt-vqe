"""C3: Sign Canonicalization for p=1 N=20 MPNN Deployment.

Hypothesis: Enforcing θ_x > 0 (Strategy A) resolves the Z₂ sign ambiguity
at N=20 p=1, enabling the MPNN to learn a consistent mapping and deploy
successfully at all h >= 2.25 (not just h=3.0).

Method:
    1. Run VQE at N=20, p=1 with 3 seeds
    2. Detect sign inconsistencies
    3. Canonicalize dataset (enforce θ_x > 0)
    4. Train MPNN with/without canonicalization
    5. Deploy and compare ΔE/gap

References:
    - Wiersema et al. (2020) PRX Quantum 1, 020319 — HVA Z₂ symmetry
    - Mele et al. (2022) PRA 106, L060401 — parameter transferability
"""

from __future__ import annotations

import logging
import time

import numpy as np

from experiments.helpers.sign_equivariant import (
    canonicalize_dataset,
    detect_sign_inconsistency,
)
from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import ExperimentConfig, SystemConfig, VQEConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics


class ExperimentC3(BaseExperiment):
    """Sign canonicalization for p=1 N=20 deployment."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="C3",
            category="C",
            description="Sign canonicalization enables MPNN deployment at N=20 p=1",
            hypothesis=(
                "Enforcing θ_x > 0 resolves Z₂ ambiguity at N=20 p=1, "
                "enabling MPNN to deploy at all h >= 2.25 (not just h=3.0)."
            ),
            system=SystemConfig(
                n_qubits=20,
                p_layers=1,
                h_values=[2.25, 2.5, 2.75, 3.0, 3.5, 4.0],
                h_test=[2.5, 3.0, 3.5],
            ),
            vqe=VQEConfig(n_restarts=3, sigma=0.3, maxiter=100),
            seeds=[42, 43, 44],
            verbose=True,
        )

    def setup(self) -> None:
        """Override: build p=1 circuit at N=20."""
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
        self.logger = logging.getLogger(__name__)

        N = self.config.system.n_qubits
        p = self.config.system.p_layers
        base_lattice = make_lattice("chain_1d", N, J=1.0, h=1.0)
        self.circuit, _ = self.hva.create(N, p, base_lattice)
        self.logger.info(f"C3 setup: N={N}, p={p}, n_params={self.circuit.num_parameters}")

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Run VQE sweep with MPS, then deploy with/without canonicalization."""
        from qiskit.quantum_info import Statevector
        from qiskit_aer import AerSimulator
        from scipy.optimize import minimize

        np.random.seed(seed)
        N = self.config.system.n_qubits
        n_params = self.circuit.num_parameters
        h_train = self.config.system.h_values
        h_test = self.config.system.h_test

        # MPS backend for N=20
        mps_sim = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=64,
            matrix_product_state_truncation_threshold=1e-12,
        )

        def cost_fn_mps(params, H):
            bound = self.circuit.assign_parameters(params)
            bound_save = bound.copy()
            bound_save.save_statevector()
            result = mps_sim.run(bound_save).result()
            sv = Statevector(result.get_statevector())
            return float(sv.expectation_value(H).real)

        # Phase 2: VQE descending sweep
        theta_data = {}
        prev_theta = np.random.uniform(-0.01, 0.01, n_params)

        for h in sorted(h_train, reverse=True):
            sol = self.get_exact_solution(h, N)
            H = sol["hamiltonian"]

            best_energy = float("inf")
            best_theta = prev_theta.copy()

            for restart in range(self.config.vqe.n_restarts):
                if restart == 0:
                    x0 = prev_theta.copy()
                else:
                    x0 = best_theta + np.random.normal(0, self.config.vqe.sigma, n_params)
                    x0 = np.clip(x0, -np.pi, np.pi)

                result = minimize(
                    cost_fn_mps,
                    x0,
                    method="L-BFGS-B",
                    args=(H,),
                    bounds=[(-np.pi, np.pi)] * n_params,
                    options={"maxiter": self.config.vqe.maxiter, "ftol": 1e-14},
                )
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_theta = result.x.copy()

            theta_data[h] = best_theta.copy()
            prev_theta = best_theta.copy()

        # Detect sign inconsistency and canonicalize
        h_sorted = sorted(theta_data.keys())
        theta_array = np.array([theta_data[h] for h in h_sorted])
        h_array = np.array(h_sorted)
        inconsistency = detect_sign_inconsistency(theta_array, h_array)
        theta_canonical = canonicalize_dataset(theta_array, reference_index=-1)

        # Phase 4: Deploy at test points
        metrics = []
        for h_t in h_test:
            t0 = time.time()
            sol = self.get_exact_solution(h_t, N)
            H = sol["hamiltonian"]
            exact_energy = sol["exact"].ground_energy
            gap = sol["exact"].gap
            if gap < 1e-10:
                gap = max(2 * abs(1 - h_t), 2 * np.pi / N)

            # Predict via interpolation (raw)
            theta_pred_raw = self._interpolate_theta(h_array, theta_array, h_t)
            e_raw = cost_fn_mps(theta_pred_raw, H)
            de_gap_raw = abs(e_raw - exact_energy) / gap

            # Predict via interpolation (canonicalized)
            theta_pred_canon = self._interpolate_theta(h_array, theta_canonical, h_t)
            e_canon = cost_fn_mps(theta_pred_canon, H)
            de_gap_canon = abs(e_canon - exact_energy) / gap

            elapsed = time.time() - t0

            m = ExperimentMetrics(
                h_value=h_t,
                energy=e_canon,
                exact_energy=exact_energy,
                energy_error=abs(e_canon - exact_energy),
                gap=gap,
                relative_error=de_gap_canon,
                seed=seed,
                wall_time_s=elapsed,
                theta_opt=theta_pred_canon.tolist(),
                technique_metadata={
                    "de_gap_raw": float(de_gap_raw),
                    "de_gap_canonicalized": float(de_gap_canon),
                    "canonicalization_improvement_pct": float(
                        (de_gap_raw - de_gap_canon) / max(de_gap_raw, 1e-10) * 100
                    ),
                    "sign_inconsistency": inconsistency,
                },
            )
            metrics.append(m)

        return metrics

    def _interpolate_theta(
        self, h_array: np.ndarray, theta_array: np.ndarray, h_target: float
    ) -> np.ndarray:
        """Simple linear interpolation of θ at h_target."""
        n_params = theta_array.shape[1]
        theta_pred = np.zeros(n_params)
        for p in range(n_params):
            theta_pred[p] = np.interp(h_target, h_array, theta_array[:, p])
        return theta_pred
