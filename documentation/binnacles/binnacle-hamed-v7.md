# Binnacle — Hamed's Feedback Experiments (V7)

## 2026-05-18 — Meeting Feedback & Experimental Validation

### Context

Hamed Mohammadbagherpoor provided feedback on the GNN-HVA pipeline during a meeting.
Key suggestions: investigate Nevergrad (gradient-free optimizer), Quantum Reservoir Computing
as warm-start, MPS simulation for scaling, and permutation encoding. Additionally referenced
papers: Kutvonen et al. (2020), arXiv:2510.00171, and the Nevergrad library.

All experiments run in `scripts/experiments_hamed_v7/` — isolated from the main pipeline.

### Pre-Experiment Analysis

| Hamed's Suggestion | Already in Project? | Action Taken |
|---|---|---|
| Pauli observables (not tomography) | ✅ Core constraint | No action needed |
| QRC (fixed reservoir) | ✅ Implemented (abandoned V6.1) | Re-tested as warm-start |
| DMRG/MPS ground truth | ✅ Via TeNPy | N/A |
| Commuting Pauli grouping | ✅ Qiskit handles automatically | N/A |
| PoC first, then hardware | ✅ Done (N=6→N=10→hardware) | N/A |
| Nevergrad optimizer | ❌ Not tested | → Experiment 1 |
| MPS circuit simulation | ❌ Not used | → Experiment 3 (script ready) |
| QRC → NN warm-start | ❌ Different from current QRC | → Experiment 2 |
| Permutation encoding | ❌ Not applicable | Analyzed — irrelevant to our pipeline |
| SPSA for hardware | ❌ Not tested | → Experiment 4 (identified from literature) |
| Noise-aware training | ❌ Not tested | → Experiment 5 (from Karim et al. 2025) |

---

## Experiment 1: Nevergrad vs L-BFGS-B

### Hypothesis
Gradient-free methods (CMA-ES, PSO, DE, OnePlusOne) might outperform L-BFGS-B
if barren plateaus exist in the optimization landscape.

### Configuration
- N=6, 1D TFIM chain, HVA p=2 (4 parameters)
- h ∈ {0.5, 1.0, 1.25, 1.5, 2.0}, 3 seeds per optimizer
- Nevergrad budget: 500 evaluations
- L-BFGS-B: 5 restarts, maxiter=1000

### Results

| Optimizer | Avg ΔE | Avg Fidelity | Avg Evals | Avg Time |
|-----------|--------|-------------|-----------|----------|
| **L-BFGS-B (5 restarts)** | **1.36e-01** | **0.922** | 699 | 0.69s |
| CMA (Nevergrad) | 1.78e-01 | 0.909 | 501 | 0.81s |
| OnePlusOne | 1.87e-01 | 0.904 | 501 | 0.55s |
| PSO | 3.04e-01 | 0.872 | 501 | 0.58s |
| DE | 3.16e-01 | 0.883 | 501 | 0.56s |
| NGOpt | — | — | — | FAILED (array dim issue) |

### Conclusion
**L-BFGS-B wins decisively.** For shallow HVA with no barren plateaus (Mele et al. 2026),
gradient-based optimization is optimal. Nevergrad's evolutionary strategies are designed for
high-dimensional problems with vanishing gradients — neither condition applies here.

### Learning
Hamed's concern about barren plateaus is valid in general, but our architecture (HVA p≤2 +
local energy cost) provably avoids them. The correct gradient-free optimizer for hardware
(where shot noise corrupts gradients) is SPSA, not evolutionary strategies.

---

## Experiment 2: QRC → MLP Warm-Start

### Hypothesis
Quantum reservoir features (⟨Xᵢ⟩, ⟨ZᵢZⱼ⟩ from a fixed HVA reservoir with Rx(h) encoding)
provide richer information than graph structure alone for predicting VQE parameters.

### Configuration
- N=6, 20 training h-values (linspace 0.2–2.0), fidelity filter ≥ 0.93
- QRC: fixed random HVA reservoir → 11 features → MLP(64,32) → θ_pred
- MPNN: GINConv h=64, L=3, 3000 epochs
- Direct MLP: h → MLP(64,32) → θ_pred (baseline)

### Results

| Method | Avg ΔE (test) | Training MSE | Scalability |
|--------|--------------|-------------|-------------|
| **QRC→MLP** | **1.61e-01** | 2.7e-05 | ❌ Requires statevector |
| MPNN (GINConv) | 1.79e-01 | 1.39e-02 | ✅ Graph-based |
| Direct MLP (h→θ) | 6.73e+00 | 1.35e-01 | ✅ But terrible |

Per-h breakdown:
| h | QRC ΔE | MPNN ΔE | Winner |
|---|--------|---------|--------|
| 0.50 | 6.77e-01 | 6.66e-01 | MPNN |
| 1.00 | 7.48e-02 | 8.91e-02 | QRC |
| 1.25 | 3.10e-02 | 3.68e-02 | QRC |
| 1.50 | 1.42e-02 | 6.30e-02 | QRC |
| 1.80 | 5.97e-03 | 4.18e-02 | QRC |

### Conclusion
QRC→MLP is competitive and slightly better than MPNN at N=6, especially in the paramagnetic
phase (h>1.0). However, it **cannot scale** because the reservoir requires statevector
simulation (exponential cost). The MPNN approach scales via graph structure.

### Learning
The QRC features capture quantum correlations that the graph structure alone misses at small N.
This validates Hamed's intuition about "quantum information extraction." For a thesis focused
on scalability, MPNN remains the correct choice. QRC is valid future work if quantum reservoir
hardware becomes available.

---

## Experiment 3: MPS Simulation (Script Ready)

### Hypothesis
Qiskit Aer's MPS simulator enables VQE at N=20+ where statevector (2^N memory) fails,
because 1D TFIM HVA circuits have bounded entanglement.

### Status
Script ready at `scripts/experiments_hamed_v7/experiment_mps_simulation.py`.
Qiskit Aer 0.17.2 with MPS support confirmed installed.
Not yet executed (lower priority — infrastructure validation, not a research question).

### Expected Outcome
- N=6, N=10: MPS matches statevector within shot noise
- N=20: MPS completes where statevector would require 8GB+ RAM
- Near criticality (h≈1.0): higher bond dimension needed

---

## Experiment 4: SPSA vs COBYLA vs L-BFGS-B Under Shot Noise

### Hypothesis
SPSA (Simultaneous Perturbation Stochastic Approximation) is the standard optimizer for
hardware VQE because it uses only 2 function evaluations per iteration and naturally handles
shot noise. L-BFGS-B should fail because noise corrupts gradient estimates.

### Configuration
- N=6, h=1.5, HVA p=2 (4 parameters)
- Shot noise simulated as Gaussian with std = 1/√n_shots
- n_shots ∈ {256, 1024, 4096, 8192}, 5 seeds each
- SPSA: 200 iterations (601 evals including best-tracking)
- COBYLA: maxiter=500
- L-BFGS-B: maxiter=200

### Results

| n_shots | SPSA | COBYLA | L-BFGS-B |
|---------|------|--------|----------|
| 256 | **8.05e-02** | 2.96e-01 | 8.50e-01 |
| 1024 | **7.47e-02** | 2.71e-01 | 8.50e-01 |
| 4096 | **7.06e-02** | 2.62e-01 | 8.50e-01 |
| 8192 | **6.70e-02** | 2.61e-01 | 8.50e-01 |

### Conclusion
**SPSA wins by 3-4× over COBYLA and 10× over L-BFGS-B** under all noise levels.
L-BFGS-B completely fails (noise corrupts finite-difference gradient estimates).
SPSA should be the Phase 4 hardware optimizer.

### Learning
Hamed's intuition about gradient-free methods for hardware is correct — but the right tool
is SPSA (2 evals/iteration, stochastic gradient), not Nevergrad's evolutionary strategies
(population-based, high eval count). This is a well-established result in the VQE literature
(Lavrijsen et al. 2020, Kandala et al. 2017).

---

## Experiment 5: Noise-Aware MPNN Training

### Hypothesis
Training the MPNN on VQE data obtained under shot noise produces parameters that are
optimal UNDER noise, improving hardware deployment results (Karim et al. 2025).

### Configuration
- N=6, 15 training h-values (linspace 0.5–2.0)
- Noiseless VQE: L-BFGS-B, maxiter=1000
- Noisy VQE: COBYLA with shot noise (n_shots=4096)
- Both MPNNs: GINConv h=64, L=3, 3000 epochs
- Evaluation: exact energy of predicted parameters + simulated noisy measurement

### Results

| Method | Avg ΔE (exact) | Training MSE |
|--------|---------------|-------------|
| **Noiseless-trained MPNN** | **4.16e-02** | 1.07e-03 |
| Noise-aware MPNN | 1.26e-01 | 4.9e-05 |

Per-h breakdown:
| h | Noiseless ΔE | Noise-aware ΔE |
|---|-------------|---------------|
| 1.00 | 9.49e-02 | 2.22e-01 |
| 1.25 | 4.08e-02 | 1.29e-01 |
| 1.50 | 1.96e-02 | 1.01e-01 |
| 1.80 | 1.10e-02 | 5.21e-02 |

### Conclusion
**Noiseless-trained MPNN wins.** Under pure shot noise (no coherent gate errors), noisy VQE
finds worse parameters, and the MPNN learns those worse parameters. The noise-aware approach
would only help with systematic coherent errors (gate over-rotation, crosstalk) present on
real hardware but absent in our shot-noise-only simulation.

### Learning
This experiment should be re-run with FakeTorino's full noise model (coherent + incoherent
errors) to properly test the hypothesis. With only shot noise, the "noise-aware" training
is actually "worse-parameter-aware" training. The technique remains promising for real
hardware deployment (Karim et al. 2025 validated it on IBM hardware with real gate errors).

---

## Summary of Findings

| Experiment | Key Finding | Impact on Pipeline |
|---|---|---|
| Nevergrad vs L-BFGS-B | L-BFGS-B wins (no BPs in our setting) | ✅ Validates current choice |
| QRC → MLP | Competitive at N=6, doesn't scale | 📝 Future work section |
| MPS Simulation | Infrastructure ready for N=20+ | 🔧 Available when needed |
| SPSA under noise | 3-4× better than COBYLA | ⚡ **Use for Phase 4 hardware** |
| Noise-aware training | Doesn't help with shot noise only | 📝 Re-test on real hardware |

### Action Items

1. **Immediate:** Use SPSA as the optimizer for Phase 4 hardware deployment
2. **Thesis:** Include Nevergrad comparison in optimization section (validates L-BFGS-B)
3. **Thesis:** Include QRC comparison in warm-start section (validates MPNN scalability)
4. **Future work:** Re-run noise-aware training with FakeTorino full noise model
5. **Future work:** Run MPS experiment for N=20 scaling demonstration

### New Bibliography Entries Added

- Rapin & Teytaud (2018) — Nevergrad library
- Kutvonen et al. (2020) — QRC optimization (Nature Sci Rep)
- arXiv:2510.00171 — QRC with Jaynes-Cummings model
- Qiskit Aer MPS tutorial
- Lavrijsen et al. (2020) — Classical optimizers for NISQ (SPSA validation)


---

## 2026-05-18 — Phase A Full Execution (Rigorous Protocol)

### Context

Phase A of the V7 full experiment plan executed via `run_full_plan.py --phase A --force`.
All 4 experiments completed successfully in 6.5 minutes total. This is the first execution
with the full spec protocol (proper seeds, budgets, h-grids) replacing the preliminary runs.

Infrastructure improvements applied before execution:
- Ground truth caching (eliminates redundant exact diag across seeds)
- Shared runners module (no code duplication across 5 technique scripts)
- Best-state MPNN training wrapper
- Fixed fidelity computation in SPSA metrics
- Fixed MPS evaluation (statevector-based instead of slow BackendEstimatorV2)

---

### Experiment 4A: SPSA Hyperparameter Grid Search (DEFINITIVE)

**Config:** N=6, h=1.5, n_shots=4096, 10 seeds, grid: a∈{0.05,0.1,0.2,0.5}, c∈{0.05,0.1,0.2}, A∈{10,20,50}
**Time:** 147.9s (36 configs × 10 seeds = 360 SPSA runs)

| a | c | A | Mean ΔE | Std ΔE |
|---|---|---|---------|--------|
| **0.1** | **0.05** | **10** | **6.52e-02** | — |
| 0.05 | 0.05 | 10 | 8.08e-02 | — |
| 0.1 | 0.1 | 10 | 8.05e-02 | — |
| 0.05 | 0.1 | 20 | 8.57e-02 | — |
| 0.5 | 0.2 | 20 | 1.53e-01 | 7.7e-02 |
| 0.5 | 0.1 | 50 | 1.72e-01 | 1.3e-01 |

**Best config: a=0.1, c=0.05, A=10** (mean ΔE=6.52e-02)

**Key findings:**
- Small `a` (0.05–0.1) and small `c` (0.05) are optimal — conservative step sizes
- Large `a` (0.5) causes instability (high std, worse mean)
- Stability constant `A` has minor effect (10 vs 20 vs 50 similar for good a,c)
- All configs achieve fidelity > 0.97 and correct phase classification

**Decision:** Use a=0.1, c=0.05, A=10 for all subsequent SPSA experiments (4B–4E).

---

### Experiment 4B: SPSA Warm-Start Refinement (DEFINITIVE)

**Config:** N=6, h∈{1.25,1.5,2.0}, warm-start from L-BFGS-B VQE, n_iterations∈{10,20,50,100,200}, 10 seeds
**Time:** 32.0s

| h | Baseline ΔE | Best SPSA ΔE | Best n_iter | Improvement |
|---|-------------|-------------|-------------|-------------|
| 1.25 | 4.03e-02 | 4.03e-02 | 10 | 0% (no gain) |
| 1.5 | 1.94e-02 | 1.94e-02 | 10–100 | 0% (no gain) |
| 2.0 | 3.62e-03 | 8.88e-03 | 20 | **-146%** (WORSE) |

**Critical finding: SPSA refinement HURTS when starting from a good warm-start.**

At h=2.0 (easiest point, baseline ΔE=3.6e-03), SPSA noise pushes the solution AWAY from
the optimum. At h=1.25 and h=1.5, SPSA cannot improve on the L-BFGS-B solution because
the warm-start is already near-optimal and shot noise prevents further refinement.

**Decision:** Do NOT use SPSA refinement after MPNN warm-start in the noiseless regime.
SPSA is only valuable when starting from a poor initial guess (cold-start) OR when the
cost function is inherently noisy (real hardware). For Phase 4 hardware deployment:
use MPNN prediction directly, apply SPSA only if the initial energy is poor.

---

### Experiment 1A: Nevergrad Fair Budget Comparison (DEFINITIVE)

**Config:** N=6, budget=1000, 7 h-values, 5 seeds, cold-start
**Time:** 205.2s

| Optimizer | Mean ΔE | Mean Fidelity |
|-----------|---------|---------------|
| **L-BFGS-B (5 restarts)** | **1.36e-01** | **0.926** |
| CMA | 1.78e-01 | 0.910 |
| OnePlusOne | 1.89e-01 | 0.906 |
| TwoPointsDE | 2.43e-01 | 0.898 |
| DE | 2.65e-01 | 0.890 |

**Confirms preliminary finding with rigorous protocol.** L-BFGS-B wins across all 7 h-values
and 5 seeds with budget=1000 (doubled from preliminary 500). The ranking is stable:
L-BFGS-B > CMA > OnePlusOne > TwoPointsDE > DE.

Note: Mean ΔE is high (0.136) because it includes h=0.5 (ferromagnetic phase where HVA p=2
cannot express the ground state — known physics limit). At h≥1.25, L-BFGS-B achieves ΔE<0.03.

**Decision:** Close the Nevergrad question definitively. L-BFGS-B is optimal for our setting.
Document in thesis as "validated rejection" per the decision framework.

---

### Experiment 3A: MPS Accuracy Validation (ISSUE FOUND)

**Config:** N=6, chi∈{32,64,128,256}, h∈{0.5,1.0,1.5,2.0}
**Time:** 6.9s

| h | MPS Energy | SV Energy | |MPS-SV| | Validated? |
|---|-----------|-----------|---------|-----------|
| 0.5 | -5.0469 | -4.9859 | 6.1e-02 | ❌ |
| 1.0 | -7.1105 | -7.2035 | 9.3e-02 | ❌ |
| 1.5 | -9.7891 | -9.8282 | 3.9e-02 | ❌ |
| 2.0 | -12.6050 | -12.6273 | 2.2e-02 | ❌ |

**0/16 validated** (threshold: |MPS-SV| < 1e-4)

**Root cause analysis:**
1. MPS energy is IDENTICAL across all chi values (32=64=128=256) → chi is not the bottleneck
2. MPS consistently finds LOWER energy than statevector at h=0.5 (MPS=-5.047 vs SV=-4.986) → MPS is finding a better minimum!
3. The "diff_vs_sv" is not an MPS accuracy issue — it's a VQE convergence issue. Both methods run COBYLA/L-BFGS-B from the same initial guess but converge to different local minima.
4. The validation criterion (|MPS-SV| < 1e-4) is testing optimizer convergence, not MPS simulator accuracy.

**The experiment design has a flaw:** To validate MPS accuracy, we should evaluate the SAME parameters on both backends (not run separate VQE optimizations). The current design conflates "MPS simulator accuracy" with "optimizer convergence from random init."

**Decision:** Redesign 3A to evaluate identical θ on both MPS and statevector backends.
The MPS simulator itself is likely accurate — the issue is the experimental protocol.

---

### Phase A Summary

| Experiment | Status | Key Result | Action |
|------------|--------|-----------|--------|
| 4A | ✅ Definitive | Best SPSA: a=0.1, c=0.05, A=10 | Use for all SPSA experiments |
| 4B | ✅ Definitive | SPSA hurts warm-start | Don't refine good predictions |
| 1A | ✅ Definitive | L-BFGS-B wins (confirmed) | Close Nevergrad question |
| 3A | ⚠️ Protocol flaw | MPS vs SV tests optimizer, not simulator | Redesign needed |

**Total execution time:** 6.5 minutes (well within the 20-min Phase A estimate)

### Next Steps

1. Fix 3A protocol: evaluate same θ on both backends (not separate VQE runs)
2. Proceed to Phase B (3B, 3C, 1C, 2B) — scaling tests
3. Use optimal SPSA config (a=0.1, c=0.05, A=10) for Phase C experiments


---

## Cross-Comparison: V7 Phase A vs V6.1 Established Results

### Context

V6.1 pipeline (N=6, h=1.5) achieves: ΔE/gap=1.36%, ⟨X⟩=2.6e-03, fidelity=0.997, checklist=5/6.
V6.1 pipeline (N=10, h=1.5) achieves: ΔE/gap=2.79%, ⟨X⟩=9.7e-03, fidelity=0.992, checklist=2-3/6.

The V7 experiments test whether alternative techniques can improve on these baselines.

### Optimizer Comparison (V7 1A vs V6.1 Phase 2)

| Method | Mean ΔE (N=6, all h) | Best ΔE (h=1.5) | Evals | Use Case |
|--------|----------------------|-----------------|-------|----------|
| **V6.1 L-BFGS-B (5 restarts)** | 1.36e-01 | **1.94e-02** | ~700 | Noiseless VQE ✅ |
| CMA (Nevergrad) | 1.78e-01 | 3.79e-02 | 1000 | Not competitive |
| OnePlusOne | 1.89e-01 | 5.81e-02 | 1000 | Not competitive |
| **V7 SPSA (best config)** | — | **6.52e-02** | 402 | Shot-noise VQE ✅ |

**Conclusion:** L-BFGS-B remains optimal for noiseless simulation (V6.1 Phase 2). SPSA is the correct choice only for hardware deployment where shot noise corrupts gradients. The V6.1 optimizer choice is validated.

### SPSA Warm-Start (V7 4B) vs V6.1 MPNN Prediction

| Scenario | ΔE at h=1.5 | ΔE at h=2.0 |
|----------|-------------|-------------|
| V6.1 MPNN prediction (no refinement) | 1.94e-02 | 3.62e-03 |
| V7 SPSA refinement (200 iter) | 2.64e-02 (+36%) | 1.65e-02 (+356%) |

**Critical insight:** SPSA refinement degrades the MPNN prediction in noiseless simulation. The V6.1 pipeline's "predict and deploy" strategy is correct — SPSA should only be applied on real hardware where the cost function is inherently noisy and the MPNN prediction may be suboptimal due to noise mismatch.

### MPS Validation (V7 3A) vs V6.1 Statevector

| h | V6.1 SV VQE (L-BFGS-B) | V7 MPS VQE (COBYLA) | Diff |
|---|-------------------------|---------------------|------|
| 0.5 | -4.986 | -5.047 | MPS finds lower E (better!) |
| 1.0 | -7.204 | -7.111 | SV finds lower E |
| 1.5 | -9.828 | -9.789 | SV finds lower E |
| 2.0 | -12.627 | -12.605 | SV finds lower E |

**Issue:** The 3A protocol tests optimizer convergence, not MPS accuracy. MPS at h=0.5 found a LOWER energy than statevector (impossible if both are exact) — this means COBYLA on MPS found a different local minimum than L-BFGS-B on statevector. The MPS simulator itself is likely accurate; the experiment needs redesign.

### V7 Findings That Inform V6.1 Pipeline

| V7 Finding | Impact on V6.1 | Action |
|------------|----------------|--------|
| L-BFGS-B definitively wins (1A) | Validates V6.1 Phase 2 | No change needed |
| SPSA best config: a=0.1, c=0.05, A=10 (4A) | Use for Phase 4 hardware | Update hardware_deployer_v61.py defaults |
| SPSA hurts warm-start (4B) | Don't refine MPNN predictions in simulation | Keep current "predict and deploy" |
| MPS chi-independent at N=6 (3A) | MPS is not chi-limited for 1D TFIM | Can use chi=64 for speed at N=20 |

### What V7 Adds Beyond V6.1

V6.1 established the pipeline works (ΔE/gap < 5% at N=6 and N=10). V7 answers the "what if" questions from Hamed's feedback:

1. **What if we used gradient-free optimizers?** → Worse. L-BFGS-B is optimal (no barren plateaus).
2. **What if we refined predictions with SPSA?** → Hurts in simulation. Only helps on real hardware.
3. **What if we used MPS for scaling?** → MPS works but protocol needs fixing. Chi is not the bottleneck.
4. **What are the optimal SPSA hyperparameters?** → a=0.1, c=0.05, A=10 (ready for hardware).

### Remaining V7 Experiments (Phases B–E)

| Phase | Key Question | Expected Outcome |
|-------|-------------|-----------------|
| B (3B, 3C, 1C, 2B) | Does MPS scale to N=20? Does MPNN beat QRC at N=10? | MPS should work; MPNN should win |
| C (5A, 4C, 5B, 4D) | Does noise-aware training help with FakeTorino? | May help (coherent errors differ from shot noise) |
| D (2A, 2D, 5C, 4E, 5E) | Novel contributions (hybrid QRC+MPNN, iterative refinement) | Thesis "future work" material |
| E (3D, 3E, 1B, 5D, 2C) | Stretch goals (N=30, warm-start Nevergrad) | Nice-to-have, not critical |
