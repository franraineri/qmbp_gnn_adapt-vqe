"""E3: Active Learning for Optimal h-Grid Selection.

Hypothesis: An ensemble-based active learning strategy selects the next
h-point based on MPNN prediction variance, reducing VQE runs by 30-50%
while maintaining ΔE/gap < 5% at deployment.

Method:
    1. Start with 5 seed points
    2. Train ensemble of 5 MPNNs
    3. Select next h via max-variance acquisition
    4. Run VQE, add to training set, retrain
    5. Repeat until convergence or max budget

Expected outcome: ΔE/gap < 5% with 10-12 points (vs 17 baseline).

Reference: Miao et al. (2024) PRApplied 21, 014053.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.experiments_v8.core.base_experiment import BaseExperiment
from scripts.experiments_v8.core.config import ExperimentConfig, MPNNConfig, SystemConfig
from scripts.experiments_v8.core.metrics import V8Metrics
from scripts.experiments_v8.techniques.active_learning import (
    compute_ensemble_uncertainty,
    select_next_point,
)


class ExperimentE3(BaseExperiment):
    """Active learning for optimal h-grid selection."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="E3",
            category="E",
            description="Active learning reduces VQE data collection by 30-50%",
            hypothesis=(
                "Ensemble-based active learning achieves ΔE/gap < 5% with "
                "10-12 training points instead of 17 (30% reduction)."
            ),
            system=SystemConfig(
                n_qubits=6,
                p_layers=2,
                h_values=[],  # Dynamically selected
                h_test=[1.5],
            ),
            mpnn=MPNNConfig(
                hidden_dim=64,
                n_layers=3,
                n_epochs=2000,
                lr=1e-3,
                n_ensemble=5,
                use_active_learning=True,
                acquisition="max_variance",
            ),
            seeds=[42, 43, 44],
            verbose=True,
        )

    def setup(self) -> None:
        """Standard setup."""
        from src.poc.v6 import (
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
        self.logger.info(f"E3 setup: N={N}, p={p}, ensemble_size={self.config.mpnn.n_ensemble}")

    def run_single(self, seed: int) -> list[V8Metrics]:
        """Run active learning loop and compare with fixed grid."""
        import torch
        from qiskit.primitives import StatevectorEstimator

        np.random.seed(seed)
        torch.manual_seed(seed)
        StatevectorEstimator()
        N = self.config.system.n_qubits
        n_params = self.circuit.num_parameters
        h_test = self.config.system.h_test[0]

        # Candidate h-points (pool to select from)
        h_candidates = np.linspace(0.8, 2.0, 25)

        # Seed points (initial training set)
        seed_h = [0.8, 1.0, 1.25, 1.5, 2.0]
        training_h = list(seed_h)
        training_theta = {}

        # Generate VQE data for seed points
        prev_theta = np.random.uniform(-0.01, 0.01, n_params)
        for h in sorted(seed_h, reverse=True):
            theta = self._run_vqe_at_h(h, prev_theta)
            training_theta[h] = theta
            prev_theta = theta.copy()

        # Active learning loop
        max_budget = 20
        iteration_log = []

        for iteration in range(max_budget - len(seed_h)):
            # Train ensemble
            ensemble = self._train_ensemble(training_h, training_theta, seed + iteration)

            # Compute uncertainty at candidates (excluding already-selected)
            remaining = [h for h in h_candidates if h not in training_h]
            if not remaining:
                break

            uncertainties = []
            for h_cand in remaining:
                preds = [self._predict_mlp(model, h_cand) for model in ensemble]
                unc = compute_ensemble_uncertainty(preds)
                uncertainties.append(unc["variance"])

            # Check stopping criterion at test point
            test_preds = [self._predict_mlp(model, h_test) for model in ensemble]
            test_unc = compute_ensemble_uncertainty(test_preds)

            # Deploy best prediction at test point
            test_theta = test_unc["mean"]
            sol = self.get_exact_solution(h_test)
            H = sol["hamiltonian"]
            exact_energy = sol["exact"].ground_energy
            gap = (
                sol["exact"].gap
                if sol["exact"].gap > 1e-10
                else max(2 * abs(1 - h_test), 2 * np.pi / N)
            )
            e_pred = self.evaluate_energy(test_theta, H)
            de_gap = abs(e_pred - exact_energy) / gap

            iteration_log.append(
                {
                    "iteration": iteration,
                    "n_training_points": len(training_h),
                    "test_de_gap": float(de_gap),
                    "test_uncertainty": float(test_unc["variance"]),
                    "max_candidate_uncertainty": float(max(uncertainties)) if uncertainties else 0,
                }
            )

            # Stop if good enough
            if de_gap < 0.05 and test_unc["variance"] < 0.01:
                self.logger.info(
                    f"    Converged at iteration {iteration} with {len(training_h)} points"
                )
                break

            # Select next point
            remaining_arr = np.array(remaining)
            idx, h_next = select_next_point(
                remaining_arr,
                uncertainties,
                acquisition=self.config.mpnn.acquisition,
            )

            # Run VQE at selected point
            nearest_h = min(training_h, key=lambda x: abs(x - h_next))
            init_theta = training_theta[nearest_h]
            theta_next = self._run_vqe_at_h(h_next, init_theta)
            training_h.append(h_next)
            training_theta[h_next] = theta_next

            self.logger.info(
                f"    Iter {iteration}: selected h={h_next:.3f}, "
                f"ΔE/gap={de_gap:.4f}, n_points={len(training_h)}"
            )

        # Final evaluation
        final_ensemble = self._train_ensemble(training_h, training_theta, seed + 999)
        final_preds = [self._predict_mlp(model, h_test) for model in final_ensemble]
        final_unc = compute_ensemble_uncertainty(final_preds)
        final_theta = final_unc["mean"]

        sol = self.get_exact_solution(h_test)
        H = sol["hamiltonian"]
        exact_energy = sol["exact"].ground_energy
        gap = (
            sol["exact"].gap
            if sol["exact"].gap > 1e-10
            else max(2 * abs(1 - h_test), 2 * np.pi / N)
        )
        e_final = self.evaluate_energy(final_theta, H)
        de_gap_final = abs(e_final - exact_energy) / gap

        # Cold-start comparison
        cold_theta = np.random.uniform(-0.5, 0.5, n_params)
        e_cold = self.evaluate_energy(cold_theta, H)
        de_gap_cold = abs(e_cold - exact_energy) / gap

        # Fixed grid comparison (use all 17 points from uniform grid)
        fixed_h = np.linspace(0.8, 2.0, 17).tolist()
        fixed_theta = {}
        prev = np.random.uniform(-0.01, 0.01, n_params)
        for h in sorted(fixed_h, reverse=True):
            prev = self._run_vqe_at_h(h, prev)
            fixed_theta[h] = prev.copy()

        fixed_ensemble = self._train_ensemble(fixed_h, fixed_theta, seed + 2000)
        fixed_preds = [self._predict_mlp(model, h_test) for model in fixed_ensemble]
        fixed_unc = compute_ensemble_uncertainty(fixed_preds)
        e_fixed = self.evaluate_energy(fixed_unc["mean"], H)
        de_gap_fixed = abs(e_fixed - exact_energy) / gap

        m = V8Metrics(
            h_value=h_test,
            energy=e_final,
            exact_energy=exact_energy,
            energy_error=abs(e_final - exact_energy),
            gap=gap,
            relative_error=de_gap_final,
            seed=seed,
            wall_time_s=0.0,
            prediction_uncertainty=float(final_unc["variance"]),
            technique_metadata={
                "n_active_points": len(training_h),
                "n_fixed_points": len(fixed_h),
                "de_gap_active": float(de_gap_final),
                "de_gap_fixed_17pts": float(de_gap_fixed),
                "de_gap_cold": float(de_gap_cold),
                "data_reduction_pct": float((len(fixed_h) - len(training_h)) / len(fixed_h) * 100),
                "selected_h_points": sorted(training_h),
                "iteration_log": iteration_log,
                "warm_vs_cold_gain_pct": float(
                    (de_gap_cold - de_gap_final) / max(de_gap_cold, 1e-10) * 100
                ),
            },
        )

        return [m]

    def _run_vqe_at_h(self, h: float, init_theta: np.ndarray) -> np.ndarray:
        """Run VQE at a single h-point with warm-start."""
        from qiskit.primitives import StatevectorEstimator
        from scipy.optimize import minimize

        estimator = StatevectorEstimator()
        sol = self.get_exact_solution(h)
        H = sol["hamiltonian"]
        n_params = self.circuit.num_parameters

        def cost_fn(params):
            bound = self.circuit.assign_parameters(params)
            job = estimator.run([(bound, H)])
            return float(job.result()[0].data.evs)

        best_energy = float("inf")
        best_theta = init_theta.copy()

        for restart in range(3):
            x0 = (
                init_theta + np.random.normal(0, 0.1, n_params)
                if restart > 0
                else init_theta.copy()
            )
            x0 = np.clip(x0, -np.pi, np.pi)
            result = minimize(
                cost_fn,
                x0,
                method="L-BFGS-B",
                bounds=[(-np.pi, np.pi)] * n_params,
                options={"maxiter": 300, "ftol": 1e-14},
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_theta = result.x.copy()

        return best_theta

    def _train_ensemble(self, h_list, theta_dict, seed):
        """Train ensemble of simple MLPs."""
        import torch
        import torch.nn as nn

        n_ensemble = self.config.mpnn.n_ensemble
        n_params = self.circuit.num_parameters
        ensemble = []

        h_arr = np.array(sorted(h_list))
        theta_arr = np.array([theta_dict[h] for h in h_arr])

        X = torch.tensor(h_arr.reshape(-1, 1), dtype=torch.float32)
        Y = torch.tensor(theta_arr, dtype=torch.float32)

        for i in range(n_ensemble):
            torch.manual_seed(seed + i * 100)
            model = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, n_params),
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            for _epoch in range(self.config.mpnn.n_epochs):
                optimizer.zero_grad()
                pred = model(X)
                loss = loss_fn(pred, Y)
                loss.backward()
                optimizer.step()

            ensemble.append(model)

        return ensemble

    def _predict_mlp(self, model, h: float) -> np.ndarray:
        """Predict theta from a single MLP."""
        import torch

        with torch.no_grad():
            x = torch.tensor([[h]], dtype=torch.float32)
            pred = model(x).numpy().flatten()
        return pred

    def analyze(self, results: dict[int, list[V8Metrics]]) -> dict:
        analysis = super().analyze(results)

        active_points = []
        reductions = []
        for _seed, metrics in results.items():
            for m in metrics:
                md = m.technique_metadata
                active_points.append(md["n_active_points"])
                reductions.append(md["data_reduction_pct"])

        analysis["active_learning_summary"] = {
            "mean_points_needed": float(np.mean(active_points)),
            "baseline_points": 17,
            "mean_data_reduction_pct": float(np.mean(reductions)),
            "hypothesis_confirmed": float(np.mean(active_points)) <= 12,
        }

        return analysis

    def report(self, analysis: dict) -> str:
        base = super().report(analysis)
        lines = [base, "", "Active Learning Summary:"]

        al_summary = analysis.get("active_learning_summary", {})
        lines.extend(
            [
                f"  Points needed (active):  {al_summary.get('mean_points_needed', 0):.1f}",
                f"  Points needed (fixed):   {al_summary.get('baseline_points', 17)}",
                f"  Data reduction:          {al_summary.get('mean_data_reduction_pct', 0):.1f}%",
                f"  Hypothesis (<=12 pts):   {al_summary.get('hypothesis_confirmed', False)}",
            ]
        )

        return "\n".join(lines)
