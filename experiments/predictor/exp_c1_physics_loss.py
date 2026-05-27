"""C1: Physics-Informed MPNN Loss.

Hypothesis: Adding an energy-validation term to the MPNN loss every K epochs
prevents the MPNN from learning parameters with low MSE but high energy error,
improving ΔE/gap by 10-30% at the valid regime boundary.

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

from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import (
    ExperimentConfig,
    MPNNConfig,
    SystemConfig,
    VQEConfig,
)
from qmbp_simulation.framework.metrics import ExperimentMetrics

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
        base_lattice = make_lattice(self.config.system.topology, N, J=1.0, h=1.0)
        self.circuit, _ = self.hva.create(N, p, base_lattice)
        logger.info(f"C1 setup: N={N}, p={p}, n_params={self.circuit.num_parameters}")

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Train baseline vs physics-informed MPNN, deploy both."""
        raise NotImplementedError(
            "C1 full implementation requires MPNN training infrastructure. "
            "See scripts/experiments_v8/experiments/exp_c1_physics_loss.py for reference."
        )
