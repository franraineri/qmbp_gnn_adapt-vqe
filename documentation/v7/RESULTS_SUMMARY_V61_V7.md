# Results Summary: V6.1 vs V7 Techniques

> GNN-HVA Framework — Hybrid GNN-HVA for Topological Phase Characterization
> Date: 2026-05-19 | 75+ experiments across N=6, N=10, N=20

---

## V6.1 Pipeline (Baseline)

The production pipeline: Phase 1 (exact diag/DMRG) → Phase 2 (VQE descending sweep) → Phase 3 (MPNN prediction) → Phase 4 (deployment + error mitigation).

| System | h_test | ΔE/gap | Checklist | Status |
|--------|--------|--------|-----------|--------|
| N=6 | 1.25 | 3.5% | 2-3/6 | ✅ Physics limit |
| N=6 | 1.5 | 1.4% | 5/6 | ✅ Best result |
| N=10 | 1.5 | 2.7% | 4/4 | ✅ Fully solved |
| N=20 | 2.0 | 1.75% | — | ✅ Scaling demonstrated |

**Core config:** L-BFGS-B (5 restarts), GINConv MPNN (h=64/128, L=3), fidelity filter ≥0.93, descending warm-start sweep.

---

## V7 Techniques Tested (12 experiments)

### Technique 1: Nevergrad (Gradient-Free Optimizers)

| Method | How | Result | Verdict |
|--------|-----|--------|---------|
| CMA-ES | Population-based evolutionary | 31% worse than L-BFGS-B | ❌ Rejected |
| OnePlusOne | Single-point mutation | 39% worse | ❌ Rejected |
| DE | Differential evolution | 95% worse | ❌ Rejected |
| TwoPointsDE | Two-point DE variant | 79% worse | ❌ Rejected |

**Why they fail:** No barren plateaus in HVA p≤2 (Mele et al. 2026) → gradients are informative → L-BFGS-B exploits them efficiently. Evolutionary methods waste budget on exploration that isn't needed.

---

### Technique 2: QRC (Quantum Reservoir Computing)

| Method | How | Result | Verdict |
|--------|-----|--------|---------|
| QRC→MLP (N=6) | Fixed HVA reservoir + Rx(h) encoding → features → MLP | Slightly better than MPNN at N=6 | ⚠️ Doesn't scale |
| QRC→MLP (N=10) | Same at N=10 | **Identical to MPNN** (<1% diff) | ❌ No advantage |
| 4 reservoir designs | Random, near-identity, entangling, optimized | Not tested (predictor is solved) | — Skipped |
| Hybrid QRC+MPNN | Concatenate features | Not tested (ceiling-limited) | — Skipped |

**Why it doesn't help:** At N≥10, both QRC and MPNN hit the HVA expressibility ceiling. The predictor is NOT the bottleneck — error_from_mpnn = 0.000.

---

### Technique 3: MPS Circuit Simulation

| Method | How | Result | Verdict |
|--------|-----|--------|---------|
| MPS accuracy (N=6) | Same θ on SV vs MPS | |MPS-SV| = 1e-14 (exact) | ✅ Validated |
| MPS accuracy (N=10) | Same θ on SV vs MPS | |MPS-SV| = 1e-14 (exact) | ✅ Validated |
| MPS VQE (N=20) | L-BFGS-B + warm-start via MPS | ΔE=0.020 at h=2.0 | ✅ Enables scaling |
| Full pipeline (N=20) | DMRG + VQE + MPNN + deploy | **ΔE/gap = 1.75%** | ✅ **Best N=20 result** |

**Key finding:** MPS is exact for 1D HVA circuits (chi=64 sufficient). Enables N=20 VQE where statevector is too slow.

---

### Technique 4: SPSA (Hardware Optimizer)

| Method | How | Result | Verdict |
|--------|-----|--------|---------|
| SPSA grid search (4A) | 36 configs × 10 seeds | Best: a=0.1, c=0.05, A=10 | ✅ Config found |
| SPSA vs COBYLA (4C) | FakeTorino noise model | SPSA 3× better | ✅ **Use for hardware** |
| SPSA warm-start (4B) | Refine MPNN prediction | **Makes things WORSE** (-146% to -356%) | ❌ Don't refine |
| SPSA + ZNE (4E) | ZNE per SPSA evaluation | Marginal (~10% gain in sim) | ⚠️ Real value on hardware |

**Key finding:** SPSA is the correct hardware optimizer (3× better than COBYLA under noise). But do NOT use it to refine already-good predictions — shot noise pushes away from the optimum.

---

### Technique 5: Noise-Aware Training

| Method | How | Result | Verdict |
|--------|-----|--------|---------|
| Noisy VQE data (5A) | SPSA under 8192-shot noise | θ_noisy differs by 1.25 rad from θ_clean | Characterization |
| Noise-aware MPNN (5B) | Train on noisy θ | **6× worse** than noiseless | ❌ Fails under shot noise |
| Iterative refinement (5E) | Train→deploy→collect→retrain | 9% improvement, saturates in 2 rounds | ⚠️ Modest |

**Why it fails:** Shot noise makes VQE find scattered local minima (not systematically shifted). The MPNN can't learn a smooth mapping from noisy targets. Would only help with coherent gate errors (real hardware).

---

### Technique 6: Transfer Learning (N=6→N=10)

| Method | How | Result | Verdict |
|--------|-----|--------|---------|
| Pre-train N=6, fine-tune N=10 | 3000ep pre-train + 3000ep fine-tune | 7% worse than baseline | ❌ Rejected |
| Combined (N=6+N=10 data) | Train on mixed dataset | 4% worse than baseline | ❌ Rejected |

**Why it fails:** N=6 and N=10 have different optimal θ landscapes. Pre-training biases weights toward N=6 patterns that don't transfer.

---

### Technique 7: N=20 Scaling (Valid Regime Discovery)

| Method | How | Result | Verdict |
|--------|-----|--------|---------|
| Full h-grid (h∈[0.8,2.0]) | Include all h-values | ΔE/gap = 6.0% ❌ | Bad data poisons MPNN |
| Energy filter (ΔE<0.05) | Remove bad VQE points | ΔE/gap = 7.4% ❌ | Removes too much coverage |
| **Valid regime only (h∈[1.5,2.0])** | Train where VQE works | **ΔE/gap = 1.75% ✅** | **Correct approach** |

**Key finding:** At N=20, only train on h≥1.5 (where HVA p=2 can express the GS). Including physics-limited h-values actively poisons the MPNN.

---

## Best Combination of Techniques

### For Simulation (thesis results)

| Phase | Best Technique | Config |
|-------|---------------|--------|
| Phase 1 | Exact diag (N≤14) / DMRG (N≥15) | Standard |
| Phase 2 | **L-BFGS-B + restarts + descending warm-start** | N=6: 5 rst σ=0.1; N=20: 7 rst σ=0.3 |
| Phase 3 | **GINConv MPNN** (not QRC, not GAT, not transfer) | h=64(N=6)/128(N≥10), L=3, 6000ep |
| Phase 4 | **StatevectorEstimator** (simulation) | Direct prediction, no SPSA refinement |
| H-grid | **Valid regime only** | N=6: h≥0.8; N=10: h≥0.8; N=20: h≥1.5 |
| Filter | Fidelity≥0.93 (N≤14) / None (N≥15) | At N≥15, restrict h-grid instead |

### For Hardware (IBM Torino deployment)

| Phase | Best Technique | Config |
|-------|---------------|--------|
| Phase 2 | Same as simulation (generate training data offline) | L-BFGS-B noiseless |
| Phase 3 | Same MPNN (trained on noiseless data) | NOT noise-aware |
| Phase 4 | **MPNN prediction → SPSA refinement (only if initial ΔE is poor)** | a=0.1, c=0.05, A=10 |
| Mitigation | **DD + Twirling + TREX + ZNE** (Resilience Level 2) | N=6: 3 layouts; N=10: O(n) layouts |
| Shots | ≥8192 | Shot noise < observable signal |

### For Future Scaling (N=30+)

| Component | Recommendation |
|-----------|---------------|
| Ground truth | DMRG (TeNPy) |
| VQE | MPS backend (chi=64) + L-BFGS-B + 10 restarts σ=0.5 |
| H-grid | h≥1.8 (projected valid regime for N=30) |
| MPNN | h=128, L=3, train on valid regime only |
| Test point | h≥2.5 (deep paramagnetic) |

---

## What Works vs What Doesn't

### ✅ Works

| Technique | Where | Impact |
|-----------|-------|--------|
| L-BFGS-B with restarts | Phase 2 (all N) | Optimal noiseless optimizer |
| GINConv MPNN | Phase 3 (all N) | Lattice-agnostic, scales to N=20 |
| Descending warm-start | Phase 2 (all N) | Smooth θ landscape, no wrapping needed |
| MPS simulation | N≥20 VQE | Exact, chi=64 sufficient for 1D |
| SPSA (a=0.1, c=0.05) | Hardware Phase 4 | 3× better than COBYLA under noise |
| Valid-regime-only training | N≥15 | Prevents data poisoning |
| Fidelity filter ≥0.93 | N≤14 | Removes bad VQE points automatically |

### ❌ Doesn't Work

| Technique | Why | Evidence |
|-----------|-----|----------|
| Nevergrad/evolutionary optimizers | No barren plateaus → gradients work | V7 1A: 31-95% worse |
| QRC at N≥10 | Predictor is ceiling-limited | V7 2B: identical to MPNN |
| Noise-aware MPNN training | Shot noise → scattered θ (unlearnable) | V7 5B: 6× worse |
| Transfer learning N→N' | Different θ landscapes per N | V7 TL: 7% worse |
| SPSA refinement of good predictions | Noise pushes away from optimum | V7 4B: -146% to -356% |
| Energy-error filtering at N≥15 | Removes coverage, MPNN can't interpolate | N=20 Run 2: 7.4% |
| Training on invalid h-regime | Bad θ poisons MPNN | N=20 Run 1: 6.0% |
| GATConv for 1D chains | Attention has nothing to attend to | V6 binnacle: instability |
| Data augmentation at N≥10 | Interpolated θ inaccurate | V6 binnacle: hurts |
| Larger MPNN (h=128 at N=6) | Overfits on 17 points | V6 sweep: worse |
| Higher LR (3e-3) | ΔE/gap instability | V6 sweep: exceeds 5% |

---

## Summary Table: Pipeline Performance

| N | h_test | ΔE/gap | Method | Time |
|---|--------|--------|--------|------|
| 6 | 1.25 | 3.5% ✅ | V6.1 standard | 25s |
| 6 | 1.5 | 1.4% ✅ | V6.1 standard | 25s |
| 10 | 1.5 | 2.7% ✅ | V6.1 (h=128, seed=43) | 50s |
| 20 | 2.0 | **1.75% ✅** | V7 (valid regime, 7 rst, σ=0.3) | 50 min |

The pipeline resolves the quantum phase (ΔE/gap < 5%) at every system size tested.
The remaining frontier is real hardware deployment on IBM Torino.
