"""C1: Physics-Informed MPNN Loss.

Hypothesis: Adding an energy-validation term to the MPNN loss every K epochs
prevents the MPNN from learning parameters with low MSE but high energy error,
improving ΔE/gap by 10-30% at the valid regime boundary.

Method:
    1. Generate VQE training data at N=6 (17 h-points, descending sweep)
    2. Train MPNN-baseline (MSE-only, 6000 epochs)
    3. Train MPNN-physics (MSE + λ·|E(θ_pred)-E_exact|, λ=0.1, start epoch 1000)
    4. Deploy both at boundary h-values (h=1.0, 1.25, 1.5)
    5. Compare ΔE/gap

IMPORTANT: We are NOT changing the VQE cost function (V5.x lesson).
The θ targets remain pure-energy VQE optima. The energy term is a
MPNN training regularizer only.

References:
    - Miao et al. (2024) PRApplied 21, 014053
    - Zhang et al. (2025) arXiv:2505.01236 (Qracle)
    - Lee et al. (2026) arXiv:2602.19752
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.experiments_v8.core.base_experiment import BaseExperiment
from scripts.experiments_v8.core.config import (
    ExperimentConfig,
    MPNNConfig,
    SystemConfig,
    VQEConfig,
)
from scripts.experiments_v8.core.metrics import V8Metrics

logger = logging.getLogger(__name__)


class ExperimentC1(BaseExperiment):
    """Physics-informed MPNN loss experiment."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="C1",
            category="C",
            description=(
                "Physics-informed MPNN loss: MSE + λ·|E(θ_pred)-E_exact| "
                "improves deployment at boundary h-values"
            ),
            hypothesis=(
                "Adding energy validation to MPNN training improves ΔE/gap "
                "by 10-30% at h=1.25 (boundary) without regression at h=1.5."
            ),
            system=SystemConfig(
                n_qubits=6,
                p_layers=2,
                # Training h-grid (17 points, valid regime)
                h_values=[
                    0.8,
                    0.9,
                    1.0,
                    1.1,
                    1.2,
                    1.25,
                    1.3,
                    1.4,
                    1.5,
                    1.6,
                    1.7,
                    1.8,
                    1.9,
                    2.0,
                    2.25,
                    2.5,
                    3.0,
                ],
                # Test at boundary + safe points
                h_test=[1.0, 1.25, 1.5, 1.75, 2.0],
            ),
            vqe=VQEConfig(n_restarts=5, maxiter=500),
            mpnn=MPNNConfig(
                hidden_dim=64,
                n_layers=3,
                n_epochs=6000,
                lr=1e-3,
                patience=300,
                use_physics_loss=True,
                physics_loss_weight=0.1,
                physics_loss_start_epoch=1000,
                physics_loss_eval_every=100,
            ),
            seeds=[42, 43, 44],
            verbose=True,
        )

    def setup(self) -> None:
        """Setup shared infrastructure."""
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

        N = self.config.system.n_qubits
        p = self.config.system.p_layers
        base_lattice = make_lattice("chain_1d", N, J=1.0, h=1.0)
        self.circuit, _ = self.hva.create(N, p, base_lattice)
        logger.info(f"C1 setup: N={N}, p={p}, n_params={self.circuit.num_parameters}")

    def run_single(self, seed: int) -> list[V8Metrics]:
        """Train baseline vs physics-informed MPNN, deploy both."""
        from qiskit.primitives import StatevectorEstimator
        from scipy.optimize import minimize

        from src.poc.v6.mpnn_predictor import (
            MPNNPredictor,
            train_mpnn,
        )

        np.random.seed(seed)
        torch.manual_seed(seed)
        N = self.config.system.n_qubits
        n_params = self.circuit.num_parameters
        estimator = StatevectorEstimator()

        h_train = self.config.system.h_values
        h_test = self.config.system.h_test

        # ── Phase 2: Generate VQE training data ──────────────────────
        logger.info(f"    Generating VQE data ({len(h_train)} h-points)...")
        vqe_data = {}  # h -> (theta_opt, energy, exact_energy, gap, lattice)
        prev_theta = np.random.uniform(-0.01, 0.01, n_params)

        for h in sorted(h_train, reverse=True):
            sol = self.get_exact_solution(h, N)
            H = sol["hamiltonian"]

            def cost_fn(params, _H=H):
                bound = self.circuit.assign_parameters(params)
                job = estimator.run([(bound, _H)])
                return float(job.result()[0].data.evs)

            best_e = float("inf")
            best_theta = prev_theta.copy()
            for r in range(self.config.vqe.n_restarts):
                x0 = (
                    prev_theta.copy()
                    if r == 0
                    else (best_theta + np.random.normal(0, 0.1, n_params))
                )
                x0 = np.clip(x0, -np.pi, np.pi)
                res = minimize(
                    cost_fn,
                    x0,
                    method="L-BFGS-B",
                    bounds=[(-np.pi, np.pi)] * n_params,
                    options={"maxiter": self.config.vqe.maxiter, "ftol": 1e-14},
                )
                if res.fun < best_e:
                    best_e = res.fun
                    best_theta = res.x.copy()

            vqe_data[h] = {
                "theta": best_theta.copy(),
                "energy": best_e,
                "exact_energy": sol["exact"].ground_energy,
                "gap": sol["exact"].gap,
                "lattice": sol["lattice"],
            }
            prev_theta = best_theta.copy()

        # ── Phase 3: Build graph dataset ─────────────────────────────
        logger.info("    Building graph dataset...")
        dataset = []
        hamiltonians_map = {}  # idx -> (H, exact_energy)

        for idx, h in enumerate(sorted(vqe_data.keys())):
            d = vqe_data[h]
            lattice = d["lattice"]
            # Build graph data manually
            from torch_geometric.data import Data as PyGData

            # Node features: [h_i, coordination_number]
            coord = 2 if N > 2 else 1  # interior nodes
            x = torch.tensor([[h, coord]] * N, dtype=torch.float32)
            # Edge index (chain_1d)
            edges = []
            for i in range(N - 1):
                edges.append([i, i + 1])
                edges.append([i + 1, i])
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            y = torch.tensor(d["theta"], dtype=torch.float32)

            graph = PyGData(x=x, edge_index=edge_index, y=y)
            dataset.append(graph)

            H = self.builder.build(lattice)
            hamiltonians_map[idx] = (H, d["exact_energy"])

        # ── Train MPNN-baseline (MSE only) ───────────────────────────
        logger.info("    Training MPNN-baseline (MSE only)...")
        torch.manual_seed(seed)
        model_baseline = MPNNPredictor(
            node_features=2,
            hidden_dim=self.config.mpnn.hidden_dim,
            n_layers=self.config.mpnn.n_layers,
            output_dim=n_params,
        )
        history_baseline = train_mpnn(
            model_baseline,
            dataset,
            n_epochs=self.config.mpnn.n_epochs,
            lr=self.config.mpnn.lr,
            patience=self.config.mpnn.patience,
        )
        logger.info(f"    Baseline: final MSE={history_baseline['final_mse']:.6f}")

        # ── Train MPNN-physics (MSE + energy validation) ─────────────
        logger.info("    Training MPNN-physics (MSE + energy)...")
        torch.manual_seed(seed)
        model_physics = MPNNPredictor(
            node_features=2,
            hidden_dim=self.config.mpnn.hidden_dim,
            n_layers=self.config.mpnn.n_layers,
            output_dim=n_params,
        )

        def energy_val_fn(pred_batch, batch):
            """Evaluate |E(θ_pred) - E_exact| for physics loss."""
            pred_np = pred_batch.detach().cpu().numpy()
            errors = []
            n_eval = min(5, len(pred_np))
            indices = np.random.choice(len(pred_np), n_eval, replace=False)
            for i in indices:
                if i in hamiltonians_map:
                    H, e_exact = hamiltonians_map[i]
                    theta = pred_np[i]
                    bound = self.circuit.assign_parameters(theta)
                    job = estimator.run([(bound, H)])
                    e_pred = float(job.result()[0].data.evs)
                    errors.append(abs(e_pred - e_exact))
            return errors if errors else [0.0]

        history_physics = train_mpnn(
            model_physics,
            dataset,
            n_epochs=self.config.mpnn.n_epochs,
            lr=self.config.mpnn.lr,
            patience=self.config.mpnn.patience,
            energy_val_interval=self.config.mpnn.physics_loss_eval_every,
            energy_val_fn=energy_val_fn,
            divergence_threshold=0.01,
        )
        logger.info(
            f"    Physics: final MSE={history_physics['final_mse']:.6f}, "
            f"energy_evals={len(history_physics['energy_val_history'])}"
        )

        # ── Phase 4: Deploy both models at test h-values ─────────────
        logger.info("    Deploying at test points...")
        metrics = []

        for h_t in h_test:
            t0 = time.time()
            sol = self.get_exact_solution(h_t, N)
            H = sol["hamiltonian"]
            e_exact = sol["exact"].ground_energy
            gap = sol["exact"].gap
            if gap < 1e-10:
                gap = max(2 * abs(1.0 - h_t), 2 * np.pi / N)

            # Build test graph
            from torch_geometric.data import Batch
            from torch_geometric.data import Data as PyGData

            coord = 2 if N > 2 else 1
            x = torch.tensor([[h_t, coord]] * N, dtype=torch.float32)
            edges = []
            for i in range(N - 1):
                edges.append([i, i + 1])
                edges.append([i + 1, i])
            edge_index = torch.tensor(edges, dtype=torch.long).t()
            test_graph = PyGData(x=x, edge_index=edge_index)
            test_batch = Batch.from_data_list([test_graph])

            # Predict with baseline
            model_baseline.eval()
            with torch.no_grad():
                theta_baseline = model_baseline(test_batch).numpy().flatten()
            e_baseline = self.evaluate_energy(theta_baseline, H)
            de_gap_baseline = abs(e_baseline - e_exact) / gap

            # Predict with physics model
            model_physics.eval()
            with torch.no_grad():
                theta_physics = model_physics(test_batch).numpy().flatten()
            e_physics = self.evaluate_energy(theta_physics, H)
            de_gap_physics = abs(e_physics - e_exact) / gap

            elapsed = time.time() - t0

            m = V8Metrics(
                h_value=h_t,
                energy=e_physics,
                exact_energy=e_exact,
                energy_error=abs(e_physics - e_exact),
                gap=gap,
                relative_error=de_gap_physics,
                seed=seed,
                wall_time_s=elapsed,
                theta_opt=theta_physics.tolist(),
                technique_metadata={
                    "de_gap_baseline": float(de_gap_baseline),
                    "de_gap_physics": float(de_gap_physics),
                    "improvement_pct": float(
                        (de_gap_baseline - de_gap_physics) / max(de_gap_baseline, 1e-10) * 100
                    ),
                    "baseline_mse": history_baseline["final_mse"],
                    "physics_mse": history_physics["final_mse"],
                    "physics_stopped_early": history_physics["stopped_early"],
                    "physics_stop_reason": history_physics["stop_reason"],
                },
            )
            metrics.append(m)

        return metrics

    def analyze(self, results: dict[int, list[V8Metrics]]) -> dict:
        """Compare baseline vs physics-informed across all seeds."""
        analysis = super().analyze(results)

        baseline_errors = []
        physics_errors = []
        per_h = {}

        for _seed, metrics in results.items():
            for m in metrics:
                md = m.technique_metadata
                baseline_errors.append(md["de_gap_baseline"])
                physics_errors.append(md["de_gap_physics"])

                h = m.h_value
                if h not in per_h:
                    per_h[h] = {"baseline": [], "physics": []}
                per_h[h]["baseline"].append(md["de_gap_baseline"])
                per_h[h]["physics"].append(md["de_gap_physics"])

        analysis["physics_loss_comparison"] = {
            "mean_de_gap_baseline": float(np.mean(baseline_errors)),
            "mean_de_gap_physics": float(np.mean(physics_errors)),
            "improvement_pct": float(
                (np.mean(baseline_errors) - np.mean(physics_errors))
                / max(np.mean(baseline_errors), 1e-10)
                * 100
            ),
            "pass_rate_baseline": float(np.mean([e < 0.05 for e in baseline_errors])),
            "pass_rate_physics": float(np.mean([e < 0.05 for e in physics_errors])),
            "per_h": {
                str(h): {
                    "baseline_mean": float(np.mean(v["baseline"])),
                    "physics_mean": float(np.mean(v["physics"])),
                    "improvement_pct": float(
                        (np.mean(v["baseline"]) - np.mean(v["physics"]))
                        / max(np.mean(v["baseline"]), 1e-10)
                        * 100
                    ),
                }
                for h, v in sorted(per_h.items())
            },
        }

        return analysis

    def report(self, analysis: dict) -> str:
        base = super().report(analysis)
        lines = [base, "", "Physics-Informed Loss Comparison:"]

        comp = analysis.get("physics_loss_comparison", {})
        lines.extend(
            [
                f"  Baseline (MSE only): mean ΔE/gap = "
                f"{comp.get('mean_de_gap_baseline', 0):.4f}, "
                f"pass rate = {comp.get('pass_rate_baseline', 0) * 100:.0f}%",
                f"  Physics-informed:    mean ΔE/gap = "
                f"{comp.get('mean_de_gap_physics', 0):.4f}, "
                f"pass rate = {comp.get('pass_rate_physics', 0) * 100:.0f}%",
                f"  Improvement:         {comp.get('improvement_pct', 0):.1f}%",
                "",
                "  Per-h breakdown:",
            ]
        )

        for h_str, data in comp.get("per_h", {}).items():
            imp = data["improvement_pct"]
            marker = "✅" if imp > 0 else "❌"
            lines.append(
                f"    h={h_str}: baseline={data['baseline_mean']:.4f}, "
                f"physics={data['physics_mean']:.4f} "
                f"({imp:+.1f}%) {marker}"
            )

        return "\n".join(lines)
