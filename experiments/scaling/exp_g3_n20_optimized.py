"""G3: N=20 p=2 Optimized Pipeline (Capstone Experiment).

Hypothesis: Combining all V8 findings (1 restart, 2 frozen params at h>=1.5,
restricted regime h∈[1.5,2.5]), the N=20 p=2 pipeline achieves ΔE/gap < 3%
in ≤15 min total.

Method:
    1. MPS VQE with optimized config (1 restart, freeze at h>=1.5)
    2. MPNN training on restricted regime
    3. Deploy at h_test = [1.75, 2.0, 2.25]
    4. Compare time and accuracy vs V7 3C (5 restarts, full params)

Thesis value: HIGH — capstone demonstrating V8 improvements compose.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from qmbp_simulation.framework import BaseExperiment, ExperimentConfig
from qmbp_simulation.framework.config import MPNNConfig, SystemConfig, VQEConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics

logger = logging.getLogger(__name__)


class ExperimentG3(BaseExperiment):
    """N=20 p=2 optimized pipeline capstone."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="G3",
            category="G",
            description="N=20 p=2 optimized pipeline (capstone)",
            hypothesis=(
                "V8 optimizations (1 restart + freeze) achieve ΔE/gap < 3% "
                "at N=20 p=2 in ≤15 min, vs 50 min with V7 config."
            ),
            system=SystemConfig(
                n_qubits=20,
                p_layers=2,
                h_values=[2.5, 2.25, 2.0, 1.75, 1.5],
                h_test=[2.125],  # Interpolation point not in training set
            ),
            vqe=VQEConfig(
                n_restarts=1,  # B4: no saddle points
                maxiter=300,
                freeze_params=[2, 3],  # B2: freeze θ_zz2, θ_x2
                freeze_after_h=1.5,
            ),
            mpnn=MPNNConfig(hidden_dim=128, n_layers=3, n_epochs=6000, lr=1e-3),
            seeds=DEFAULT_SEEDS,
            verbose=True,
        )

    def setup(self) -> None:
        """Override: use MPS via AerSimulator for N=20."""
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.models import HamiltonianBuilder, make_lattice
        from qmbp_simulation.solvers import ClassicalSolver

        warnings = self.config.validate()
        for w in warnings:
            logger.warning(f"Config warning: {w}")

        self.builder = HamiltonianBuilder()
        self.solver = ClassicalSolver()
        self.hva = HVACircuitBuilder()

        N = self.config.system.n_qubits
        p = self.config.system.p_layers
        topology = self.config.system.topology
        base_lattice = make_lattice(topology, N, J=1.0, h=1.0)
        self.circuit, _ = self.hva.create(N, p, base_lattice)
        self._base_lattice = base_lattice  # Cache for graph construction

        # MPS backend for N=20 (validated exact for 1D HVA, V7 3A/3B)
        from qiskit_aer import AerSimulator

        self._mps_sim = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=64,
            matrix_product_state_truncation_threshold=1e-12,
        )
        logger.info(f"G3 setup: N={N}, p={p}, topology={topology}, backend=MPS(chi=64)")

    def _evaluate_mps(self, params, H):
        """Evaluate energy via MPS (exact, no shot noise)."""
        from qiskit.quantum_info import Statevector

        bound = self.circuit.assign_parameters(params)
        bound_save = bound.copy()
        bound_save.save_statevector()
        result = self._mps_sim.run(bound_save).result()
        sv = Statevector(result.get_statevector())
        return float(sv.expectation_value(H).real)

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Run optimized pipeline at N=20."""
        from scipy.optimize import minimize

        from qmbp_simulation.predictors import MPNNPredictor, train_mpnn

        np.random.seed(seed)
        torch.manual_seed(seed)
        N = self.config.system.n_qubits
        n_params = self.circuit.num_parameters
        h_train = self.config.system.h_values
        h_test = self.config.system.h_test

        # Phase 2: VQE with optimized config
        logger.info(f"    Phase 2: VQE sweep ({len(h_train)} points, 1 restart)...")
        t_vqe_start = time.time()
        vqe_data = {}
        prev_theta = np.random.uniform(-0.01, 0.01, n_params)

        for h in sorted(h_train, reverse=True):
            sol = self.get_exact_solution(float(h), N)
            H = sol["hamiltonian"]

            # Determine active params (freeze at h >= freeze_after_h)
            bounds = [(-np.pi, np.pi)] * n_params
            if self.config.vqe.freeze_params and h >= (self.config.vqe.freeze_after_h or 99):
                frozen = self.config.vqe.freeze_params
                # Fix frozen params at current values
                for fi in frozen:
                    bounds[fi] = (prev_theta[fi], prev_theta[fi])

            best_e, best_theta = float("inf"), prev_theta.copy()
            for r in range(self.config.vqe.n_restarts):
                x0 = (
                    prev_theta.copy() if r == 0 else best_theta + np.random.normal(0, 0.1, n_params)
                )
                x0 = np.clip(x0, -np.pi, np.pi)
                res = minimize(
                    lambda p, _H=H: self._evaluate_mps(p, _H),
                    x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": self.config.vqe.maxiter, "ftol": 1e-14},
                )
                if res.fun < best_e:
                    best_e = res.fun
                    best_theta = res.x.copy()

            vqe_data[float(h)] = best_theta.copy()
            prev_theta = best_theta.copy()

        t_vqe = time.time() - t_vqe_start
        logger.info(f"    Phase 2 complete: {t_vqe:.1f}s")

        # Phase 3: MPNN training
        logger.info("    Phase 3: MPNN training...")
        t_mpnn_start = time.time()
        dataset = self._build_dataset(h_train, vqe_data, N)
        model = MPNNPredictor(
            node_features=2,
            hidden_dim=self.config.mpnn.hidden_dim,
            n_layers=self.config.mpnn.n_layers,
            output_dim=n_params,
        )
        train_mpnn(
            model,
            dataset,
            n_epochs=self.config.mpnn.n_epochs,
            lr=self.config.mpnn.lr,
            patience=self.config.mpnn.patience,
        )
        t_mpnn = time.time() - t_mpnn_start
        logger.info(f"    Phase 3 complete: {t_mpnn:.1f}s")

        # Phase 4: Deploy
        metrics = []
        for h_t in h_test:
            sol = self.get_exact_solution(float(h_t), N)
            H = sol["hamiltonian"]
            e_exact = sol["exact"].ground_energy
            gap = (
                sol["exact"].gap
                if sol["exact"].gap > 1e-10
                else max(2 * abs(1 - h_t), 2 * np.pi / N)
            )

            theta_pred = self._predict(model, float(h_t), N)
            e_pred = self._evaluate_mps(theta_pred, H)
            de_gap = abs(e_pred - e_exact) / gap

            m = ExperimentMetrics(
                h_value=float(h_t),
                energy=e_pred,
                exact_energy=e_exact,
                energy_error=abs(e_pred - e_exact),
                gap=gap,
                relative_error=de_gap,
                seed=seed,
                wall_time_s=t_vqe + t_mpnn,
                technique_metadata={
                    "vqe_time_s": t_vqe,
                    "mpnn_time_s": t_mpnn,
                    "total_time_s": t_vqe + t_mpnn,
                    "n_restarts": self.config.vqe.n_restarts,
                    "n_frozen_params": len(self.config.vqe.freeze_params or []),
                },
            )
            metrics.append(m)

        return metrics

    def _build_dataset(self, h_values, vqe_data, N):
        """Build PyG dataset using the lattice's actual edges and coordination."""
        from experiments.helpers.graph_utils import build_experiment_dataset

        h_sorted = sorted(h_values)
        theta_array = np.array([vqe_data[float(h)] for h in h_sorted])
        return build_experiment_dataset(self, np.array(h_sorted), theta_array)

    def _predict(self, model, h, N):
        """Predict θ at unseen h using the MPNN and lattice graph structure."""
        from experiments.helpers.graph_utils import predict_theta

        return predict_theta(self, model, float(h))
