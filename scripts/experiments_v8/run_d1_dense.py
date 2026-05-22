#!/usr/bin/env python3
"""D1 variant: Dense h-grid near h_c to resolve peak location.

Tests whether the ||dθ/dh|| peak at h≈0.68 is a grid-resolution
artifact or a genuine physical result.

Uses 50 points in [0.5, 1.5] instead of 40 points in [0.5, 2.5].
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.experiments_v8.core.config import ExperimentConfig
from scripts.experiments_v8.experiments.exp_d1_weight_space import (
    ExperimentD1,
)


class ExperimentD1Dense(ExperimentD1):
    """D1 with dense h-grid focused on [0.5, 1.5]."""

    @classmethod
    def default_config(cls) -> ExperimentConfig:
        cfg = ExperimentD1.default_config()
        cfg.experiment_id = "D1_dense"
        cfg.description = "D1 variant: dense h-grid [0.5,1.5] to resolve peak location near h_c"
        return cfg

    def run_single(self, seed: int) -> list:
        """Override h-ranges to focus near h_c."""
        import logging

        from scripts.experiments_v8.core.metrics import V8Metrics

        logger = logging.getLogger(__name__)
        np.random.seed(seed)
        N = self.config.system.n_qubits

        # Dense grid focused on critical region
        h_full = np.linspace(0.5, 1.5, 50)
        h_valid = np.linspace(1.25, 2.5, 25)

        theta_full = self._generate_vqe_data(h_full, seed)
        theta_valid = self._generate_vqe_data(h_valid, seed)

        mpnn_a, _ = self._train_mpnn(h_full, theta_full, seed, tag="full_dense")
        mpnn_b, _ = self._train_mpnn(h_valid, theta_valid, seed, tag="valid")

        # Probe on dense grid near h_c
        h_probe = np.linspace(0.5, 1.5, 50)
        grad_a = self._compute_weight_gradients(mpnn_a, h_probe, N)
        grad_b = self._compute_weight_gradients(mpnn_b, h_probe, N)

        peak_a_idx = int(np.argmax(grad_a))
        peak_b_idx = int(np.argmax(grad_b))
        peak_a_h = float(h_probe[peak_a_idx])
        peak_b_h = float(h_probe[peak_b_idx])

        logger.info(f"  Seed {seed}: DENSE peak_A={peak_a_h:.3f}, peak_B={peak_b_h:.3f}, h_c=1.0")

        metrics = []
        for i, h in enumerate(h_probe):
            m = V8Metrics(
                h_value=float(h),
                energy=0.0,
                exact_energy=0.0,
                energy_error=0.0,
                gap=1.0,
                relative_error=float(grad_a[i]),
                seed=seed,
                wall_time_s=0.0,
                technique_metadata={
                    "grad_norm_full_dense": float(grad_a[i]),
                    "grad_norm_valid": float(grad_b[i]),
                    "peak_h_full": peak_a_h,
                    "peak_h_valid": peak_b_h,
                    "peak_h_full_dense": peak_a_h,
                    "known_h_c": 1.0,
                },
            )
            metrics.append(m)
        return metrics


def main():
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = ExperimentD1Dense.default_config()
    cfg.verbose = True
    exp = ExperimentD1Dense(cfg)
    exp.execute()


if __name__ == "__main__":
    main()
