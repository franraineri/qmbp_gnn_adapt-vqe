# Plan 06v2: Noise-Aware MPNN with Coherent Errors (BackendEstimatorV2 + FakeTorino)

**Paper:** Karim et al. (2025) — Fast and Noise-aware ML VQE Optimiser (arXiv:2503.20210)
**Priority:** MEDIUM — depends on #03 (Flow) and #04 (Qracle) being ready
**Effort:** 1.5 weeks (90 min/sweep × multiple configs + analysis)
**Prerequisites:** #03 Flow-VQE integrated, #04 Qracle unified graph integrated

## Previous Result: F18 (FAILED — shot noise only)

V7 Experiment 5B already tested noise-aware MPNN and **failed 6-10× worse**:
- Config: N=6, SPSA/COBYLA + shot noise (4096 shots, no coherent errors)
- Root cause: shot noise produces **scattered** θ (random, unlearnable)
- Conclusion: pure stochastic noise corrupts training targets → F18 confirmed

## What's Different This Time

| V7 5B (FAILED) | This plan (v8 2.2) |
|---|---|
| Shot noise only (Gaussian) | Full FakeTorino noise model |
| No coherent errors | T1/T2 decay + crosstalk + gate over-rotation |
| θ_noisy = random scatter | θ_noisy = systematic shift (hypothesis) |
| COBYLA/SPSA gradient-free | SPSA only (confirmed 3× better under noise) |
| N=6, global HVA (2 params) | N=10, bond-resolved (19 params) + unified graph |
| Standalone experiment | Stacked on top of #03 + #04 improvements |

**Key hypothesis:** Coherent gate errors create a *learnable, smooth* shift in
the energy landscape. θ_opt(coherent_noise) ≠ θ_opt(noiseless), but the mapping
h → θ_opt(coherent) is still a smooth function that the MPNN can learn.

## Design: 5-Way Comparison Matrix

Designed to run AFTER #03 and #04 are integrated, producing a full ablation:

```
         Training data source
         ┌─────────────────────────────────┐
         │  Noiseless θ    │  FakeTorino θ  │
─────────┼─────────────────┼────────────────┤
Graph A  │  (A1) Baseline   │  (A2) Noise-aware │  ← current BondResolvedMPNN
(H only) │  [already have]   │  [new: this plan]  │
─────────┼─────────────────┼────────────────────┤
Graph B  │  (B1) Qracle     │  (B2) Qracle+Noise │  ← unified graph (#04)
(H+Circ) │  [from #04]       │  [new: this plan]  │
─────────┼─────────────────┼────────────────────┤
+Flow    │  (C1) Flow K=5   │  (C2) Flow+Noise   │  ← multi-shot (#03)
         │  [from #03]       │  [new: this plan]  │
         └─────────────────┴────────────────────┘

Deploy target: FakeTorino (BackendEstimatorV2, 8192 shots)
```

## Execution Plan

### Phase 0 — Confirm Prerequisites (Day 0)

- [ ] #03 Flow-VQE: `FlowMultiShotPredictor` works on noiseless data
- [ ] #04 Qracle: `build_unified_bond_resolved_graph()` + masked BondResolvedMPNN works
- [ ] Existing baseline A1 has bond-resolved results at N=10 chain_1d

### Phase 1 — Noisy Training Data Collection (Days 1-3)

**Goal:** Generate θ_opt(FakeTorino) training dataset at N=10, chain_1d, p=1.

**Config:**
```python
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit.primitives import BackendEstimatorV2

backend = BackendEstimatorV2(FakeTorino())
# SPSA optimizer (validated: 3× better than COBYLA under coherent noise, V7 4C)
# shots=8192 (minimum for SNR>1 per observable)
# n_restarts=15 (noisy landscape is rougher)
# maxiter=3000 per restart (SPSA needs more iterations)
```

**h-grid:** 35 points, non-uniform (denser near h_min ≈ 0.83 for bond-resolved):
- h ∈ [0.8, 1.5]: Δh=0.05 (15 points — critical region)
- h ∈ [1.5, 3.0]: Δh=0.1 (15 points — paramagnetic)
- h ∈ [3.0, 4.0]: Δh=0.2 (5 points — deep paramagnetic)

**Seeds:** 5 independent seeds (select best θ per h-point by energy)

**Expected wall-clock:** ~90 min per sweep × 5 seeds = ~7.5 hours total
(parallelizable across seeds)

**Quality gates before accepting:**
- ΔE/gap < 20% (noiseless evaluation of θ_noisy) → keeps ~70% of points
- θ smoothness check: max jump < 1.0 rad between consecutive h-points
- At least 20/35 points pass → sufficient for MPNN training

**Key diagnostic:** Plot θ_opt(FakeTorino) vs θ_opt(noiseless) per-bond.
If the difference is smooth and small (<0.3 rad), the shift is coherent/learnable.
If scattered (>1.0 rad variance across seeds), abort — same failure mode as V7 5B.

### Phase 2 — Train All Variants (Days 4-5)

Train 6 MPNN variants using the comparison matrix:

| ID | Graph | θ_target | Model | Notes |
|---|---|---|---|---|
| A1 | H-only | noiseless | BondResolvedMPNN | Already have (baseline) |
| A2 | H-only | FakeTorino | BondResolvedMPNN | New |
| B1 | H+Circuit | noiseless | BondResolvedMPNN+mask | From #04 |
| B2 | H+Circuit | FakeTorino | BondResolvedMPNN+mask | New |
| C1 | H-only | noiseless | Flow K=5 | From #03 |
| C2 | H-only | FakeTorino | Flow K=5 | New (train flow on noisy θ) |

Training hyperparams (same for all):
- hidden_dim=256, n_layers=3, norm_type="none"
- lr=1e-3, patience=300, epochs=8000
- Canonical θ (canonicalize_theta mandatory)

### Phase 3 — Deploy & Compare (Days 6-7)

Deploy all 6 models on FakeTorino at **test** h-points (held out from training):
- h_test = {1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0} (7 points)
- 3 seeds per test point
- Measure: ΔE/gap (vs exact E₀), |ΔE| absolute, pass rate (<5%)

**Additionally test with ZNE on top:**
- Each model's predictions + PEA-ZNE (3 noise factors)
- Compare: noise-aware raw vs noiseless+ZNE vs noise-aware+ZNE

### Phase 4 — Analysis & Decision (Days 8-9)

Generate comparison table:

```
| Model | Deploy (raw) | Deploy (+ZNE) | Improvement vs A1 |
|-------|:---:|:---:|:---:|
| A1 (baseline) | X% | Y% | — |
| A2 (noise-aware) | ? | ? | ? |
| B1 (Qracle) | ? | ? | ? |
| B2 (Qracle+noise) | ? | ? | ? |
| C1 (Flow K=5) | ? | ? | ? |
| C2 (Flow+noise) | ? | ? | ? |
```

**Decision criteria:**
- If A2 > A1 by ≥30%: noise-aware training works with coherent errors ✅
- If B2 > B1 by ≥20%: noise-aware + unified graph compound ✅
- If any noise-aware (raw) ≈ noiseless+ZNE: eliminates need for ZNE → major finding
- If ALL noise-aware variants ≤ noiseless: confirms F18 even with coherent errors → negative result (still publishable as "coherent shift is too small to learn")

## Modules to Reuse

| Existing module | Role in this plan |
|---|---|
| `execution.NoisyBackend` | NOT used — FakeTorino needs BackendEstimatorV2 directly |
| `optimizers.vqe.VQEOptimizer` | Phase 2 VQE (configure for SPSA) |
| `predictors.mpnn.BondResolvedMPNN` | Model A1/A2 |
| `predictors.mpnn.train_bond_resolved_mpnn` | Training loop (identical) |
| `predictors.mpnn.build_bond_resolved_graph` | Graph A variants |
| `predictors.unified_graph` (from #04) | Graph B variants |
| `analysis.flow_warmstart.FlowWarmstartManager` | Variants C1/C2 |
| `analysis.flow_multishot.FlowMultiShotPredictor` (from #03) | Deploy C variants |
| `execution.noisy_utils.run_pea_zne` | ZNE comparison layer |
| `utils.canonicalize_theta` | Mandatory preprocessing |
| `framework.runner_base.ValidationRunner` | Script structure |

## New Code Required

Only ONE new script (no new library modules):

```
scripts/experiment_runners/noise_aware/
└── run_noise_aware_comparison.py    # Orchestrates Phases 1-4
```

Uses `ValidationRunner` base class. Configurable via flags:
```bash
# Phase 1: collect noisy data
python scripts/experiment_runners/noise_aware/run_noise_aware_comparison.py \
    --phase collect --n-qubits 10 --topology chain_1d --p-layers 1 --seeds 5

# Phase 2-3: train + deploy all variants  
python scripts/experiment_runners/noise_aware/run_noise_aware_comparison.py \
    --phase evaluate --variants A1,A2,B1,B2,C1,C2

# Phase 4: generate comparison report
python scripts/experiment_runners/noise_aware/run_noise_aware_comparison.py \
    --phase report --output results/noise_aware_comparison/
```

## Early Abort Conditions

Stop and report negative result if:
1. Phase 1 quality: < 15/35 h-points pass ΔE/gap < 20% → SPSA can't converge under FakeTorino
2. Phase 1 diagnostic: θ_noisy variance across seeds > 0.8 rad → scattered (same as V7 5B)
3. Phase 3 result: ALL noise-aware variants ≤ noiseless baseline by > 20% → coherent shift not learnable

## Expected Timeline

```
Prerequisites: #03 + #04 complete
Day 0:    Confirm prerequisites pass
Days 1-3: Phase 1 — noisy data collection (can run overnight)
Days 4-5: Phase 2 — train 6 variants (~2 hours each on GPU)
Days 6-7: Phase 3 — deploy on FakeTorino (90 min per variant × 6)
Days 8-9: Phase 4 — analysis + decision
Total: ~9 working days (after prerequisites)
```

## Risks & Mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| FakeTorino SPSA won't converge (19 dims) | MEDIUM | Use 15 restarts, maxiter=3000, warm-start from noiseless θ |
| θ shift is too small to learn | MEDIUM | Diagnostic in Phase 1; if < 0.05 rad shift → abort early |
| 7.5h data collection too slow | LOW | Parallelize 5 seeds; use only 3 seeds as minimum |
| Overfitting on 20-25 noisy points | MEDIUM | Dropout=0.2, validation split (80/20), early stopping |
| FakeTorino ≠ real hardware | HIGH | This is accepted: FakeTorino is proxy. If results are positive, follow up with ibm_torino |

## Relationship to Thesis

- **If positive (noise-aware helps):** New contribution post-F18 — "coherent errors create learnable structure that pure shot noise does not" + practical implication for hardware deployment
- **If negative (noise-aware still fails):** Strengthens F18 from "fails under shot noise" to "fails under ANY noise" — closes the question definitively
- **Either way:** The 5-way comparison (#03 × #04 × #06) is a valuable ablation for the paper
