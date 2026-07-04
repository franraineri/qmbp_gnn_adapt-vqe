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

from qmbp_simulation.framework import BaseExperiment
from qmbp_simulation.framework.config import ExperimentConfig, MPNNConfig, SystemConfig
from qmbp_simulation.framework.metrics import ExperimentMetrics


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
            system=SystemConfig(n_qubits=6, p_layers=2, h_values=[], h_test=[1.5]),
            mpnn=MPNNConfig(
                hidden_dim=64,
                n_layers=3,
                n_epochs=2000,
                lr=1e-3,
                n_ensemble=5,
                use_active_learning=True,
                acquisition="max_variance",
            ),
            seeds=DEFAULT_SEEDS,
            verbose=True,
        )

    def setup(self) -> None:
        """Standard setup."""
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
        base_lattice = make_lattice(self.config.system.topology, N, J=1.0, h=1.0)
        self.circuit, _ = self.hva.create(N, p, base_lattice)
        self.logger.info(f"E3 setup: N={N}, p={p}")

    def run_single(self, seed: int) -> list[ExperimentMetrics]:
        """Run active learning loop and compare with fixed grid."""
        raise NotImplementedError(
            "E3 full implementation requires ensemble MPNN training. "
            "See scripts/experiments_v8/experiments/exp_e3_active.py for reference."
        )
