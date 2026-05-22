#!/usr/bin/env python
"""C1 at N=10: Test if physics-informed loss improvement scales with N.

Uses the actual V8 experiment framework (ExperimentC1) with N=10 config override.

Hypothesis: At N=10, MSE-ΔE/gap decorrelation is worse, so physics loss
should show >10% improvement (vs 3.9% at N=6).

Expected: ~10-15 min execution.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.experiments_v8.core.config import (
    ExperimentConfig,
    MPNNConfig,
    SystemConfig,
    VQEConfig,
)
from scripts.experiments_v8.experiments.exp_c1_physics_loss import ExperimentC1

# Override config for N=10
config = ExperimentConfig(
    experiment_id="C1-N10",
    category="C",
    description="Physics-informed MPNN loss at N=10",
    hypothesis=(
        "Physics loss improvement is larger at N=10 due to worse "
        "MSE-ΔE/gap correlation with sparser data."
    ),
    system=SystemConfig(
        n_qubits=10,
        p_layers=2,
        # 11 training points in valid regime for N=10
        h_values=[1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
        # Test at boundary + safe
        h_test=[1.5, 1.75, 2.0],
    ),
    vqe=VQEConfig(n_restarts=3, maxiter=500),
    mpnn=MPNNConfig(
        hidden_dim=128,
        n_layers=3,
        n_epochs=3000,  # Reduced for speed
        lr=1e-3,
        patience=200,
        use_physics_loss=True,
        physics_loss_weight=0.1,
        physics_loss_start_epoch=500,
        physics_loss_eval_every=200,  # Less frequent for N=10 (expensive)
    ),
    seeds=[42, 43, 44],
    verbose=True,
)

# Run experiment
exp = ExperimentC1(config)
exp.execute()
