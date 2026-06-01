"""Quick smoke test for S8 — single seed, reduced epochs."""

import sys

sys.path.insert(0, ".")

import numpy as np

from experiments.scaling.exp_s8_d1_finite_size_scaling import ExperimentS8
from qmbp_simulation.framework.config import (
    AnalysisConfig,
    ExperimentConfig,
    MPNNConfig,
    SystemConfig,
    VQEConfig,
)

# Minimal config for smoke test
cfg = ExperimentConfig(
    experiment_id="S8-smoke",
    category="S",
    description="Smoke test",
    hypothesis="test",
    system=SystemConfig(
        n_qubits=4,
        p_layers=2,
        topology="chain_1d",
        J=1.0,
        h_values=np.arange(0.5, 2.55, 0.1).tolist(),
    ),
    vqe=VQEConfig(n_restarts=2, maxiter=100, ftol=1e-12),
    mpnn=MPNNConfig(hidden_dim=32, n_layers=3, n_epochs=500, lr=1e-3, dropout=0.1, patience=100),
    analysis=AnalysisConfig(scaling_n_values=[4, 6]),
    seeds=[42],
    verbose=True,
    auto_warm_cold_comparison=False,
)

exp = ExperimentS8(config=cfg)
exp.setup()
metrics = exp.run_single(42)

print(f"\nGot {len(metrics)} metrics")
for m in metrics:
    meta = m.technique_metadata or {}
    if meta.get("type") == "scaling_fit":
        print(
            f"  SCALING FIT: nu={meta.get('nu')}, a={meta.get('a')}, success={meta.get('fit_success')}"
        )
    elif "N" in meta:
        print(f"  N={meta['N']}: h_peak={meta['h_peak']:.3f}")

print("\nSMOKE TEST PASSED")
