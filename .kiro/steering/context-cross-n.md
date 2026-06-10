---
inclusion: fileMatch
fileMatchPattern: "**/cross_n*,**/zero_shot*,**/19_cross_n*,**/binnacle-cross-n*"
---

# Cross-N Zero-Shot GNN Context (invoke with #context-cross-n)

> Pre-digested context for cross-system-size generalization of the MPNN predictor.

## What's Done

- Train N=40+80 (14 pts) → predict N=50,60,70,100: **30/30 PASS**, mean ΔE/gap=0.15%.
- Multi-seed confirmed: seeds 42/43/44 all 5/5 PASS (std=0.074%).
- Extrapolation to N=100 (beyond training): 0.18% ΔE/gap — GNN beats scipy 2.6×.
- BatchNorm discovery: harmful for cross-N on chain_1d. Fix: `norm_type="none"`.
- VQE warm-start useless for 2 params (COBYLA always finds global min).
- Bond-resolved (79D) pending — this is where cross-N truly matters.

## Key Discovery: BatchNorm Failure

**Problem**: All nodes in chain_1d are identical post-GINConv → zero intra-graph variance.
BatchNorm `running_stats` capture graph-SIZE artifact, not feature variation.
- With BN: 18.5% error (θ_x underpredicted by 25-40%).
- Without BN: 0.13% error.

**Fix**:
```python
from qmbp_simulation.predictors import MPNNPredictor

# For cross-N tasks:
model = MPNNPredictor(
    node_features=2, hidden_dim=128, n_layers=3,
    output_dim=2,  # p=1: [θ_zz, θ_x]
    norm_type="none",  # CRITICAL for cross-N
)

# For fixed-N tasks (unchanged default):
model = MPNNPredictor(..., norm_type="batch")
```

## Configuration

| Training set | Target | Result |
|-------------|--------|--------|
| N=40 + N=80 (14 pts) | N=50 | 0.13% ΔE/gap |
| N=40 + N=80 (14 pts) | N=60 | 0.13% |
| N=40 + N=80 (14 pts) | N=70 | 0.16% |
| N=40 + N=80 (14 pts) | N=100 | 0.18% (extrapolation!) |

## Why GNN Over Interpolation?

- scipy interpolation achieves 0.11% on scalar (2-param) cross-N → comparable.
- BUT interpolation CANNOT scale to bond-resolved (79D at N=40).
- GNN handles high-dimensional output natively via graph structure.
- 79D necessity proof is the pending validation.

## Constraints

- Only validated on chain_1d topology (homogeneous nodes).
- For topologies with heterogeneous nodes (ladder, triangular): BN may be fine.
- Training requires ≥2 system sizes (ideally bracketing the target).
- Interpolation between training sizes works. Extrapolation degrades gracefully.
- h-values must be within valid regime for BOTH training sizes.

## DO NOT

- Use `norm_type="batch"` for cross-N on chain_1d (25-40% underprediction).
- Claim VQE warm-start helps at 2 params (proven useless — identical results).
- Skip the `norm_type` argument when creating MPNNPredictor for cross-N.
- Extrapolate beyond 2× training size gap without validation.
- Re-run N=40→N=60 zero-shot (already 30/30 PASS with 3 seeds).

## Pending Work

1. **Bond-resolved cross-N** (79D output at N=40): proves GNN necessity over scipy.
2. **Heterogeneous topologies**: test if BN is fine when nodes differ (ladder, triangular).
3. **Scaling to N=200+**: theoretical limit unknown, MPS VQE becomes the bottleneck.

## Source Files

- #[[file:src/qmbp_simulation/predictors/mpnn.py]] (MPNNPredictor with norm_type param)
- #[[file:scripts/experiment_runners/scaling/run_zero_shot_cross_n_v3.py]]
- #[[file:scripts/experiment_runners/scaling/run_cross_n_ablation_suite.py]]
- #[[file:scripts/experiment_runners/scaling/run_bond_resolved_cross_n.py]]
- #[[file:documentation/binnacles/binnacle-cross-n-zero-shot.md]]
- #[[file:documentation/analysis/19_cross_n_validation_plan.md]]
- #[[file:results/scaling/zero_shot/]]
