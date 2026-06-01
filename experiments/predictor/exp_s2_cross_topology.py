"""S2: Cross-Topology Transfer Learning.

Hypothesis: An MPNN trained on chain_1d N=10 can predict θ for ladder and
triangular topologies without re-training, demonstrating that the GNN learns
the physics (h→θ mapping) rather than topology-specific patterns.

Method:
    1. Run VQE on chain_1d, ladder, and triangular at N=10 p=1
    2. Train MPNN on chain_1d data only
    3. Deploy on ladder and triangular (zero-shot transfer)
    4. Measure ΔE/gap degradation
    5. Test fine-tuning with 2-3 points from target topology

Expected outcome:
    Zero-shot transfer likely fails (topologies have different θ_opt).
    Fine-tuning with 3 points may recover performance.
    Either way, quantifies the topology-specificity of learned representations.

Thesis value: MEDIUM-HIGH — validates/invalidates topology-agnosticism claim.

References:
    - Findings Index #1: Framework topology-agnostic (64% pass rate, 5 topologies)
    - Huang et al. (2022): ML can efficiently predict within a phase
    - Lee et al. (2026): GNN generalizes across Hamiltonians
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import ExperimentConfig, MPNNConfig, SystemConfig, VQEConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics

logger = logging.getLogger(__name__)

# Valid regime boundaries for p=1 at N=10
VALID_REGIMES_P1_N10: dict[str, float] = {
    "chain_1d": 1.9,
    "ladder": 3.0,
    "triangular": 3.5,
}

# Training h-grids (all within valid regime for each topology)
H_GRIDS: dict[str, list[float]] = {
    "chain_1d": [4.0, 3.5, 3.0, 2.5, 2.0],
    "ladder": [5.0, 4.5, 4.0, 3.5, 3.0],
    "triangular": [6.0, 5.5, 5.0, 4.5, 4.0],
}

# Test points (within valid regime, not in training)
H_TEST: dict[str, float] = {
    "chain_1d": 2.75,
    "ladder": 3.75,
    "triangular": 4.75,
}


class ExperimentS2(BaseExperiment):
    """Cross-topology transfer learning for MPNN predictor."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        return ExperimentConfig(
            experiment_id="S2",
            category="S",
            description="Cross-topology MPNN transfer: train on chain, deploy on ladder/triangular",
            hypothesis=(
                "MPNN trained on chain_1d generalizes to other topologies "
                "with ΔE/gap < 10% (zero-shot) or < 5% (3-point fine-tune)."
            ),
            system=SystemConfig(
                n_qubits=10,
                p_layers=1,
                topology="chain_1d",
                h_values=[4.0, 3.5, 3.0, 2.5, 2.0],
                h_test=[2.75],
            ),
            vqe=VQEConfig(n_restarts=3, maxiter=100, sigma=0.3),
            mpnn=MPNNConfig(hidden_dim=128, n_layers=3, n_epochs=6000, lr=1e-3, patience=500),
            seeds=[42, 43, 44],
            verbose=True,
        )

    def setup(self) -> None:
        """Setup builders for multi-topology VQE."""
        from qmbp_simulation import ClassicalSolver, HamiltonianBuilder, HVACircuitBuilder

        self._setup_logging()
        self.builder = HamiltonianBuilder()
        self.solver = ClassicalSolver()
        self.hva = HVACircuitBuilder()
        logger.info("S2 setup: cross-topology transfer at N=10, p=1")

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Run VQE on all topologies, train on source, deploy on targets."""

        from experiments.helpers.graph_utils import build_experiment_dataset, predict_theta
        from qmbp_simulation import make_lattice
        from qmbp_simulation.predictors import MPNNPredictor, train_mpnn

        np.random.seed(seed)
        torch.manual_seed(seed)

        N = self.config.system.n_qubits
        p = self.config.system.p_layers
        topologies = ["chain_1d", "ladder", "triangular"]
        source_topo = "chain_1d"
        metrics = []

        # Phase 2: VQE on all topologies
        vqe_data: dict[str, dict[float, np.ndarray]] = {}
        for topo in topologies:
            t0 = time.time()
            vqe_data[topo] = self._run_vqe_sweep(topo, N, p, seed)
            logger.info("  VQE %s: %d points (%.1fs)", topo, len(vqe_data[topo]), time.time() - t0)

        # Phase 3: Train MPNN on source topology only
        source_lattice = make_lattice(source_topo, N, J=1.0, h=1.0)
        circuit, _ = self.hva.create(N, p, source_lattice)
        n_params = circuit.num_parameters

        h_sorted = sorted(vqe_data[source_topo].keys())
        theta_array = np.array([vqe_data[source_topo][h] for h in h_sorted])

        # Build graph dataset for source topology
        # Temporarily set topology to source for build_experiment_dataset
        original_topo = self.config.system.topology
        self.config.system.topology = source_topo
        dataset = build_experiment_dataset(self, np.array(h_sorted), theta_array)
        self.config.system.topology = original_topo

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
        logger.info("  MPNN trained on %s (%d points)", source_topo, len(h_sorted))

        # Phase 4: Deploy on all topologies (including source as baseline)
        for topo in topologies:
            h_test = H_TEST[topo]
            lattice = make_lattice(topo, N, J=1.0, h=h_test)
            H = self.builder.build(lattice)
            result = self.solver.solve(H, lattice)
            exact_energy = result.ground_energy
            gap = result.gap if result.gap > 1e-10 else max(2 * abs(1 - h_test), 2 * np.pi / N)

            # Predict using source-trained MPNN with TARGET topology graph
            self.config.system.topology = topo
            theta_pred = predict_theta(self, model, h_test)
            self.config.system.topology = original_topo

            # Evaluate on target topology circuit
            target_lattice = make_lattice(topo, N, J=1.0, h=h_test)
            target_circuit, _ = self.hva.create(N, p, target_lattice)
            e_pred = self._evaluate_energy(target_circuit, theta_pred, H)
            de_gap = abs(e_pred - exact_energy) / gap

            is_source = topo == source_topo
            m = ExperimentMetrics(
                h_value=h_test,
                energy=e_pred,
                exact_energy=exact_energy,
                energy_error=abs(e_pred - exact_energy),
                gap=gap,
                relative_error=de_gap,
                seed=seed,
                wall_time_s=0.0,
                converged=de_gap < 0.05,
                technique_metadata={
                    "topology": topo,
                    "source_topology": source_topo,
                    "is_source": is_source,
                    "transfer_type": "self" if is_source else "zero-shot",
                    "de_gap": float(de_gap),
                    "pass_5pct": de_gap < 0.05,
                    "pass_10pct": de_gap < 0.10,
                },
            )
            metrics.append(m)
            status = "✅" if de_gap < 0.05 else ("⚠️" if de_gap < 0.10 else "❌")
            logger.info(
                "  Deploy %s→%s: ΔE/gap=%.4f %s",
                source_topo,
                topo,
                de_gap,
                status,
            )

        return metrics

    def _run_vqe_sweep(self, topology: str, N: int, p: int, seed: int) -> dict[float, np.ndarray]:
        """Run descending VQE sweep on a given topology."""
        from scipy.optimize import minimize

        from qmbp_simulation import make_lattice

        h_values = H_GRIDS[topology]
        lattice = make_lattice(topology, N, J=1.0, h=1.0)
        circuit, _ = self.hva.create(N, p, lattice)
        n_params = circuit.num_parameters

        theta_data: dict[float, np.ndarray] = {}
        prev_theta = np.random.uniform(-0.01, 0.01, n_params)

        for h in sorted(h_values, reverse=True):
            target_lattice = make_lattice(topology, N, J=1.0, h=h)
            H = self.builder.build(target_lattice)
            target_circuit, _ = self.hva.create(N, p, target_lattice)

            best_e, best_theta = float("inf"), prev_theta.copy()
            for r in range(self.config.vqe.n_restarts):
                x0 = (
                    prev_theta.copy()
                    if r == 0
                    else best_theta + np.random.normal(0, self.config.vqe.sigma, n_params)
                )
                x0 = np.clip(x0, -np.pi, np.pi)

                res = minimize(
                    lambda params, _H=H, _qc=target_circuit: self._evaluate_energy(_qc, params, _H),
                    x0,
                    method="L-BFGS-B",
                    bounds=[(-np.pi, np.pi)] * n_params,
                    options={"maxiter": self.config.vqe.maxiter, "ftol": 1e-14},
                )
                if res.fun < best_e:
                    best_e = res.fun
                    best_theta = res.x.copy()

            theta_data[h] = best_theta.copy()
            prev_theta = best_theta.copy()

        return theta_data

    def _evaluate_energy(self, circuit, params, H) -> float:
        """Evaluate energy using StatevectorEstimator."""
        from qiskit.primitives import StatevectorEstimator

        estimator = StatevectorEstimator()
        bound = circuit.assign_parameters(params)
        job = estimator.run([(bound, H)])
        return float(job.result()[0].data.evs)

    def _build_dataset(self, topology: str, N: int, h_values: list, theta_array: np.ndarray):
        """Build PyG dataset for a given topology."""
        from qmbp_simulation import make_lattice
        from qmbp_simulation.predictors import build_graph_dataset

        lattice = make_lattice(topology, N, J=1.0, h=1.0)
        return build_graph_dataset(
            lattice=lattice,
            h_values=np.array(h_values),
            theta_targets=theta_array,
        )

    def _predict_with_model(self, model, topology: str, N: int, h: float) -> np.ndarray:
        """Predict θ using model with a specific topology's graph structure."""
        from qmbp_simulation import make_lattice
        from qmbp_simulation.predictors import build_graph_dataset

        lattice = make_lattice(topology, N, J=1.0, h=h)
        # Build a single-point dataset for prediction
        dataset = build_graph_dataset(
            lattice=lattice,
            h_values=np.array([h]),
            theta_targets=np.zeros((1, model.output_dim)),
        )
        model.eval()
        with torch.no_grad():
            pred = model(dataset[0])
        return pred.numpy().flatten()
