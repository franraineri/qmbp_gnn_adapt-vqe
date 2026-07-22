# EXP-S8/S8b: Finite-Size Scaling of h_c via Weight Gradients — NEGATIVE RESULT

**Date**: 2026-06-01
**Status**: REJECTED (both variants)
**Thesis value**: Documented negative result — establishes limits of weight-space phase detection

---

## Hypothesis

The peak of ||dW/dh|| from a neural network trained on VQE data converges to
h_c=1.0 as N→∞, following the finite-size scaling law:

```
h_peak(N) = h_c + a·N^(-1/ν)    with ν = 1 (TFIM 1D, Ising class)
```

## Variants Tested

| Variant | Architecture | Input | Peak metric |
|---------|-------------|-------|-------------|
| S8 | MLP (h=128, 2 layers, dropout=0.1) | Scalar h ∈ ℝ | argmax \|\|dθ/dh\|\| |
| S8b | MPNN GINConv (h=128, L=3, dropout=0.1) | Graph (N nodes) | argmax \|\|dW/dh\|\| |

## Configuration (both variants)

```
N ∈ {4, 6, 8, 10}, p=2, chain_1d, J=1.0
h_train: [0.5, 2.5] (21 points, step 0.1)
Seeds: [42, 43, 44, 45, 46]
Training: 6000 epochs, lr=1e-3, dropout=0.1
VQE: 5 restarts, maxiter=300, L-BFGS-B, descending sweep
```

## Results

### S8 (MLP proxy)

| Seed | N=4 | N=6 | N=8 | N=10 |
|------|-----|-----|-----|------|
| 42 | 0.704 | 0.704 | **2.255** ⚠️ | 0.663 |
| 43 | 0.663 | 0.704 | 0.704 | 0.704 |
| 44 | **2.337** ⚠️ | **2.337** ⚠️ | **1.929** ⚠️ | **2.337** ⚠️ |
| 45 | 0.704 | 0.704 | 0.704 | 0.663 |
| 46 | 0.663 | 0.704 | 0.704 | 0.704 |

**Median h_peak (excluding seed 44 outlier)**: 0.704 for ALL N values.
**N-dependence**: NONE. Peak position is constant across system sizes.
**Scaling fit**: ν = 5.0 (hits upper bound) — meaningless.

### S8b (MPNN GINConv)

| Seed | N=4 | N=6 | N=8 | N=10 |
|------|-----|-----|-----|------|
| 42 | 0.500 | 0.500 | **2.214** ⚠️ | 0.500 |
| 43 | 0.500 | 0.500 | 0.500 | 0.500 |
| 44 | **2.337** ⚠️ | **2.337** ⚠️ | **1.929** ⚠️ | **2.337** ⚠️ |
| 45 | 0.500 | 0.500 | 0.500 | 0.500 |
| 46 | 0.500 | 0.500 | 0.500 | 0.500 |

**Median h_peak**: 0.500 for ALL N values (boundary of h-range).
**N-dependence**: NONE. Gradient norm is monotonically decreasing.
**Internal peaks**: Occasionally detected (h≈1.07-1.36) but NOT the global max.
**Scaling fit**: ν = 5.0 (hits upper bound) — meaningless.

### Seed 44 Pathology

Seed 44 produces catastrophic training failures in BOTH architectures:
- MLP: gradient norms ~80× normal (h_peak at boundary h=2.34)
- MPNN: MSE > 1e-2 (vs normal ~1e-4), gradient norms ~500-1200×

This is consistent with known seed 44 sensitivity (documented for triangular topology).
The VQE data at seed 44 likely has discontinuities that prevent smooth MLP/MPNN fitting.

## Root Cause Analysis

### Why S8 (MLP) fails

The MLP receives scalar h as input. The only thing that changes between N=4 and N=10
is the output dimension (n_params). The MLP's interpolation behavior in input space
is architecture-dependent, not physics-dependent. The peak at h≈0.7 reflects where
the VQE data transitions from "good" (h>1.0) to "noisy" (h<1.0) — this boundary
is the same for all N.

### Why S8b (MPNN) fails

Despite receiving a graph with N nodes, the MPNN's weight gradient norm ||dW/dh|| is
**dominated by the boundary effect**. The gradient is largest at h=0.5 (lowest h in
training range) because:

1. VQE data quality degrades sharply below h≈1.0 (outside valid regime)
2. The θ_opt values at h<1.0 are noisy/inconsistent across the h-sweep
3. The MPNN's loss landscape has the steepest gradient where data is most irregular
4. This boundary effect overwhelms any physical signal from the phase transition

### Why Hernandes et al. (2025) works but we don't

Key differences:
1. **Training data**: They use exact ground states (NQS), we use VQE-optimized θ
2. **Metric**: They measure weight-space *distance* between models trained at
   adjacent h-values, not gradient norm of a single model
3. **Architecture**: Their NQS directly represents ψ(h), ours maps h→θ (indirect)
4. **Data quality**: Their data is uniformly high-quality across all h; ours degrades
   below h_c due to HVA expressibility limits

## Conclusions

1. **Weight gradient norm ||dW/dh|| cannot extract ν** from VQE-trained networks
   because the boundary effect (data quality transition) dominates the physical signal.

2. **The D1 result (peak at h≈0.7) is qualitative, not quantitative.** It detects
   that "something happens" near h_c but the peak position is an artifact of:
   - Training data quality boundary (S8b: peak at h=0.5)
   - MLP interpolation behavior (S8: peak at h=0.7)

3. **Graph structure (MPNN vs MLP) does not help** because the problem is in the
   metric (gradient norm) and data quality, not the architecture's ability to
   encode N.

4. **Seed 44 is pathological** — should be excluded from future experiments or
   investigated separately.

## Thesis Contribution

This negative result contributes to the thesis by:

- **Establishing limits**: D1 weight-space detection is qualitative (detects
  transition exists) but not quantitative (cannot extract critical exponents)
- **Differentiating from literature**: Hernandes et al. (2025) succeeds because
  they use exact states + distance metric, not VQE data + gradient norm
- **Motivating future work**: Quantitative ν extraction requires either:
  (a) exact ground state data (not VQE), or
  (b) a different metric (weight-space distance, Fisher information), or
  (c) restricting analysis to h > h_min (valid regime only)

## Files

- `experiments/scaling/exp_s8_d1_finite_size_scaling.py` — S8 (MLP variant)
- `experiments/scaling/exp_s8b_mpnn_finite_size_scaling.py` — S8b (MPNN variant)
- `results/experiments/exp_s8/run_20260601_141053.json` — S8 raw results
- `results/experiments/exp_s8b/run_20260601_142514.json` — S8b raw results
