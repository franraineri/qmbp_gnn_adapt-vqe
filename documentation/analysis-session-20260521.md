# Analysis Session — 2026-05-21

## Auto-Review: What We Have

### Pipeline Status

| Component | Status | Quality |
|---|---|---|
| Phase 1 (ED/DMRG) | ✅ Production | Exact, validated at N=6/10/20 |
| Phase 2 (VQE) | ✅ Production | L-BFGS-B optimal, 5-7 restarts |
| Phase 3 (MPNN) | ✅ Production | GINConv, MSE < 0.005 |
| Phase 4 (Deploy) | ✅ Production + **Baseline** | ΔE/gap < 3%, gain metric added |
| Heisenberg extension | ✅ Implemented | Finding: p=2 insufficient |
| Random baseline | ✅ Implemented | Default ON, gain=83%+ |
| Diagnostics | ✅ Always-on | Checkpoints, SNR, smoothness |

### Results Inventory

| Experiment set | Runs | Key finding |
|---|---|---|
| V6.1 TFIM (N=6) | 40+ | ΔE/gap=1.4% at h=1.5 |
| V6.1 TFIM (N=10) | 15 | ΔE/gap=2.7% at h=1.5 |
| V6.1 TFIM (N=20) | 3 | ΔE/gap=1.75% at h=2.0 |
| V7 Nevergrad | 4 methods | L-BFGS-B wins by 31-95% |
| V7 QRC | 2 sizes | MPNN = QRC at N≥10 |
| V7 MPS | 3 experiments | Exact for 1D, chi=64 sufficient |
| V7 SPSA | 4 experiments | a=0.1, c=0.05 optimal for hardware |
| V7 Noise-aware | 3 experiments | Fails under shot noise |
| V7 Transfer learning | 2 experiments | N→N' doesn't transfer |
| Heisenberg (new) | 3 experiments | HVA p=2 insufficient |
| Noisy simulation | 6+ sweeps | ZNE works at N=6, fails at N=10 |

### What's Missing (Gaps Identified)

1. **No formal baseline comparison in existing results** — the 75+ TFIM runs don't have warm-vs-cold data. Only the new runs will have it.
2. **No ablation study** — we know the full pipeline works, but we haven't systematically removed components to measure their individual contribution.
3. **No scaling analysis plot** — we have N=6, 10, 20 results but no formal analysis of how ΔE/gap scales with N.
4. **No computational cost comparison** — we know the pipeline is fast, but haven't compared wall-clock time vs alternative approaches.
5. **No sensitivity analysis** — how robust is the pipeline to hyperparameter changes?

---

## Proposed Comparisons Before Scaling

### Comparison 1: Ablation Study (Component Contribution)

**Hypothesis:** Each pipeline component contributes measurably to the final ΔE/gap.

| Ablation | What we remove | Expected effect |
|---|---|---|
| No warm-start (random θ) | MPNN prediction | ΔE/gap >> 5% (already shown: ~5.14) |
| No descending sweep | Warm-start between h-points | VQE converges slower, some points fail |
| No fidelity filter | Train on all points | MPNN learns bad θ → worse prediction |
| No multi-restart | Single VQE start | More local minima → noisier θ_opt |
| Fewer h-points | 10 instead of 27 | Less training data → worse generalization |

**Implementation:** Run the pipeline 5 times, each time disabling one component. Compare ΔE/gap.

**Value:** Quantifies the contribution of each design decision. Essential for thesis Chapter 4 (Results).

### Comparison 2: Scaling Law Analysis

**Hypothesis:** ΔE/gap scales predictably with N, allowing extrapolation to N=30+.

| N | ΔE/gap (best) | Valid regime | Time |
|---|---|---|---|
| 6 | 1.4% | h ≥ 1.25 | 25s |
| 10 | 2.7% | h ≥ 1.5 | 50s |
| 20 | 1.75% | h ≥ 2.0 | 50 min |

**Analysis needed:**
- Plot ΔE/gap vs N (is it linear? sublinear? saturating?)
- Plot valid regime boundary vs N (h_min scales as ~0.25*N?)
- Plot computational time vs N (exponential in Phase 1, polynomial in Phase 3?)
- Extrapolate: at what N does the pipeline become impractical classically?

**Value:** Predicts where quantum hardware becomes necessary. Core thesis argument.

### Comparison 3: MPNN Prediction Quality Decomposition

**Hypothesis:** The total error decomposes into circuit error + prediction error, and we can measure each.

```
ΔE_total = ΔE_circuit(θ_opt) + ΔE_prediction(θ_pred - θ_opt)
```

Where:
- `ΔE_circuit(θ_opt)` = error even with perfect θ (HVA expressibility limit)
- `ΔE_prediction` = additional error from MPNN imperfection

**Implementation:** For each h_test, compute:
1. Deploy with θ_opt (from VQE) → ΔE_circuit
2. Deploy with θ_pred (from MPNN) → ΔE_total
3. ΔE_prediction = ΔE_total - ΔE_circuit

**Value:** Shows that the MPNN adds negligible error (prediction error ≈ 0). Proves the bottleneck is the circuit, not the ML.

### Comparison 4: Training Efficiency (Data Points vs Accuracy)

**Hypothesis:** The MPNN achieves good predictions with remarkably few training points.

| Training points | Expected ΔE/gap | Context |
|---|---|---|
| 5 | ? | Minimal viable dataset |
| 10 | ? | Half the standard grid |
| 17 | 1.4% | Current (fidelity-filtered) |
| 27 | ? | Full grid (no filter) |

**Implementation:** Train MPNN with progressively fewer points and measure ΔE/gap at h_test=1.5.

**Value:** Demonstrates data efficiency of the approach. Important for scaling (at N=20, each VQE point costs minutes).

### Comparison 5: Warm-Start Gain vs h-value (Phase Diagram)

**Hypothesis:** The warm-start gain varies across the phase diagram — maximum near the critical point.

| h_test | Warm ΔE/gap | Cold ΔE/gap | Gain |
|---|---|---|---|
| 0.5 | ? | ? | ? (ferromagnetic — both fail?) |
| 1.0 | ? | ? | ? (critical — hardest) |
| 1.25 | 3.5% | ? | ? |
| 1.5 | 1.4% | ? | ? |
| 2.0 | ~0.1% | ? | ? (easy — both succeed?) |

**Implementation:** Run `deploy_with_baseline()` at 5-7 h-values across the phase diagram.

**Value:** Shows WHERE the MPNN warm-start matters most. If gain is highest near h_c, it proves the ML is solving the hardest part of the problem.

### Comparison 6: Cross-Model Expressibility Map

**Hypothesis:** Different models have different expressibility profiles with HVA p=2.

| Model | Best fidelity | Valid regime | Reason |
|---|---|---|---|
| TFIM | 99.5% | h ≥ 1.25 | Low entanglement in paramagnetic phase |
| Heisenberg XXZ | 48% | None | Too much entanglement |
| XY | 23% | None | Same issue |
| TFIM (periodic) | ? | ? | Slightly more entanglement |
| TFIM (ladder) | ~0% | None | Coordination 3 too complex |

**Already have:** TFIM chain, Heisenberg, XY, ladder results.

**Missing:** TFIM with periodic boundaries (adds one edge, slightly more entanglement).

**Value:** Maps the "expressibility frontier" of HVA p=2. Shows exactly which physics is accessible with shallow circuits.

---

## Priority Ranking

| # | Comparison | Effort | Value for thesis | Recommendation |
|---|---|---|---|---|
| 1 | **Warm-start gain vs h** | Low (1 run) | Very high | **Do first** — uses new baseline |
| 2 | **Error decomposition** | Low (1 run) | Very high | **Do second** — proves ML isn't bottleneck |
| 3 | **Ablation study** | Medium (5 runs) | High | Do third — quantifies each component |
| 4 | **Training efficiency** | Medium (4 runs) | High | Do fourth — data efficiency argument |
| 5 | **Scaling law** | Low (analysis only) | Very high | Do fifth — uses existing data |
| 6 | **Cross-model map** | Already done | Medium | Document only — results exist |

### Estimated total time: ~15-20 minutes of computation (N=6 noiseless)

---

## Out-of-the-Box Ideas

### Idea A: "Zero-Shot" Phase Classification

Instead of predicting θ and then measuring observables, what if the MPNN directly predicts the phase label?

```
Input: graph(h, topology) → MPNN → "paramagnetic" / "ferromagnetic"
```

No VQE, no circuit, no quantum hardware needed. Just a classifier trained on Phase 1 data.

**Why it's interesting:** If this works (and it should — the phase boundary is smooth), it proves that for *classification* tasks, the quantum computer is unnecessary. The quantum advantage only appears for *quantitative* predictions (exact energy, correlations).

**Implementation:** Add a classification head to the MPNN, train on Phase 1 labels.

### Idea B: "How Few Qubits Do You Need?"

Run the full pipeline at N=4 (trivial), N=6 (current), N=8, N=10. Plot ΔE/gap vs N. The question: at what N does the MPNN prediction become non-trivial? At N=4, even random θ might work. At N=10, the MPNN is essential.

**Why it's interesting:** Identifies the "quantum advantage threshold" — the system size where classical brute-force fails but ML+quantum succeeds.

### Idea C: "Reverse Engineering the MPNN"

The MPNN has learned a mapping h → θ_opt. Can we extract what it learned?

1. Plot θ_pred(h) as a continuous curve (the MPNN's "understanding" of the physics)
2. Compare against θ_opt(h) from VQE (the ground truth)
3. Compute the Jacobian ∂θ/∂h — where does the MPNN predict the steepest change?
4. Does the Jacobian peak at h_c? (Would prove the MPNN "knows" about the phase transition)

**Why it's interesting:** Interpretability. Shows the MPNN isn't a black box — it has learned physically meaningful structure.

### Idea D: "Adversarial Test Points"

Instead of testing at h=1.5 (middle of training range), test at:
- h=1.05 (just above critical — hardest interpolation)
- h=0.95 (just below critical — extrapolation into invalid regime)
- h=3.0 (far extrapolation — should still work, it's deep paramagnetic)

**Why it's interesting:** Tests generalization limits. Where does the MPNN break?

### Idea E: "Time-to-Solution Comparison"

Compare total wall-clock time for:
1. Full VQE from scratch at h_test (no MPNN) — how long?
2. MPNN prediction + 0 iterations — how long?
3. MPNN prediction + 2 ADAPT iterations — how long?

Plot: accuracy vs time. The MPNN should dominate at any time budget.

**Why it's interesting:** The most practical metric for real users. "How much faster is ML warm-start?"
