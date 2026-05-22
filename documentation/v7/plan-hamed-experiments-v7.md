# Experiment Plan — Hamed's Techniques (V7 Full Test Suite)

> Comprehensive plan for rigorous testing of each technique from Hamed's feedback.
> Each technique gets a full experimental protocol: hypothesis, controls, metrics,
> system sizes, seeds, and success criteria.

---

## Overview

| # | Technique | Status | Priority | Est. Time |
|---|-----------|--------|----------|-----------|
| 1 | Nevergrad (gradient-free VQE) | Preliminary done | Low | 15 min |
| 2 | QRC → NN warm-start | Preliminary done | Medium | 20 min |
| 3 | MPS circuit simulation | Script ready | Medium | 30 min |
| 4 | SPSA for hardware VQE | Preliminary done | **High** | 10 min |
| 5 | Noise-aware MPNN training | Preliminary done | Medium | 40 min |

**Total estimated time for full suite: ~2 hours**

---

## Technique 1: Nevergrad (Gradient-Free Optimization)

### Goal
Definitively establish whether gradient-free optimizers offer any advantage over L-BFGS-B
for our HVA VQE, across system sizes and landscape difficulty.

### Preliminary Finding
L-BFGS-B wins at N=6 with 500-eval budget. But the preliminary test was limited:
- Only N=6 (4 params — too easy for evolutionary methods)
- Budget of 500 may be too low for population-based methods
- Did not test with warm-start initialization (our actual use case)

### Full Test Protocol

**Experiment 1A: Fair budget comparison at N=6**
- Optimizers: L-BFGS-B (5 restarts), CMA-ES, OnePlusOne, DE, TwoPointsDE
- Budget: 1000 evaluations for all (matching L-BFGS-B total with restarts)
- h-values: {0.5, 0.8, 1.0, 1.1, 1.25, 1.5, 2.0} (7 points including critical region)
- Seeds: 5 per optimizer per h-value
- Initial guess: uniform(-0.01, 0.01) — cold start
- Metrics: ΔE, fidelity, wall time, n_evaluations to reach ΔE<0.01

**Experiment 1B: Warm-start scenario (our actual use case)**
- Same optimizers, but initial guess = θ_opt from adjacent h-point (warm-start)
- This tests whether gradient-free methods help when already near the optimum
- Budget: 200 evaluations (refinement, not global search)
- Expected: L-BFGS-B should dominate even more with warm-start

**Experiment 1C: Scaling to N=10 (8 parameters)**
- Same protocol as 1A but N=10
- Budget: 2000 evaluations (more params need more budget)
- h-values: {1.0, 1.25, 1.5, 2.0}
- Tests whether higher dimensionality changes the ranking

### Success Criteria
- If any gradient-free method achieves ΔE < L-BFGS-B at same eval budget → reconsider
- If L-BFGS-B wins across all settings → definitively validated, close the question

### Controls
- Same initial guess for all optimizers (per seed)
- Same random seed for reproducibility
- Report both "best energy found" and "evaluations to reach threshold"

---

## Technique 2: QRC → NN Warm-Start

### Goal
Rigorously compare QRC-based parameter prediction against MPNN across system sizes,
reservoir designs, and training set sizes to understand when each approach wins.

### Preliminary Finding
QRC→MLP slightly outperforms MPNN at N=6 (avg ΔE 1.61e-01 vs 1.79e-01), especially
in the paramagnetic phase. But the test was limited:
- Only one reservoir design (random HVA params)
- Only N=6 (where QRC features are cheap to compute)
- Only 13 training points (after fidelity filter)
- MPNN only trained 3000 epochs (vs 6000 in production)

### Full Test Protocol

**Experiment 2A: Reservoir design comparison**
- Reservoir types:
  - (a) Random uniform params in [-π, π] (current)
  - (b) Small random params in [-0.1, 0.1] (near identity)
  - (c) Structured params: θ_ZZ = π/4, θ_X = π/4 (maximally entangling)
  - (d) Optimized reservoir: params from VQE at h=1.0 (critical point)
- N=6, 20 training points, same MLP architecture
- Metrics: test ΔE at h ∈ {0.5, 1.0, 1.25, 1.5, 1.8}
- Reference: Kutvonen et al. (2020) — reservoir design matters

**Experiment 2B: Scaling comparison at N=10**
- QRC→MLP vs MPNN at N=10 (8 params, 19 features for QRC)
- 27 training points (standard h-grid), fidelity filter ≥ 0.93
- MPNN: h=128, L=3, 6000 epochs, patience=500 (production config)
- QRC MLP: (128, 64) hidden layers, 2000 epochs
- Test at h ∈ {1.25, 1.4, 1.5}
- Expected: MPNN should win at N=10 (more structure to exploit)

**Experiment 2C: Training set size sensitivity**
- N=6, vary training points: {8, 12, 16, 20, 27}
- Both QRC→MLP and MPNN
- Tests which method is more data-efficient
- Reference: Miao et al. (2024) — 20 points suffice for NN-VQE

**Experiment 2D: Hybrid QRC+MPNN**
- Concatenate QRC features with MPNN graph features
- Feed combined features into MLP head
- Tests whether quantum features ADD information beyond graph structure
- Architecture: [QRC_features || MPNN_embedding] → MLP(128, 64) → θ_pred

### Success Criteria
- If QRC wins at N=10 → serious consideration for pipeline integration
- If MPNN wins at N=10 → QRC is future work only
- If hybrid outperforms both → novel contribution for thesis

### Controls
- Same VQE training data for all methods
- Same test points and evaluation protocol
- Same random seeds

---

## Technique 3: MPS Circuit Simulation

### Goal
Validate that Qiskit Aer's MPS simulator produces correct VQE results and enables
scaling to N=20 where statevector is infeasible.

### Preliminary Finding
Script ready, not yet executed. Qiskit Aer 0.17.2 confirmed installed.

### Full Test Protocol

**Experiment 3A: Accuracy validation at N=6**
- Run VQE with both StatevectorEstimator and MPS BackendEstimatorV2
- h-values: {0.5, 1.0, 1.5, 2.0}
- Compare: energy difference between methods (should be < 1e-4)
- MPS bond dimensions: {32, 64, 128, 256}
- Identify minimum chi for convergence

**Experiment 3B: Accuracy validation at N=10**
- Same as 3A but N=10
- Both methods should still be feasible
- Quantify MPS approximation error vs exact statevector

**Experiment 3C: Scaling to N=20**
- MPS-only VQE (statevector feasible but slow: 2^20 = 1M amplitudes, ~16MB per complex state)
- h-values: {0.5, 1.0, 1.5, 2.0}
- Bond dimensions: {64, 128, 256, 512}
- Validate against TeNPy DMRG ground truth (already available via ClassicalSolver)
- Measure: wall time, memory usage, energy accuracy vs DMRG
- Note: statevector IS feasible at N=20 (~16MB) but VQE would be very slow due to
  repeated matrix-vector products. MPS should be much faster for 1D circuits.

**Experiment 3D: Scaling to N=30 (if 3C succeeds)**
- MPS VQE at N=30 (2^30 = 1B amplitudes — impossible for statevector)
- Only h ∈ {1.5, 2.0} (paramagnetic, low entanglement)
- Bond dimension: 256
- Validate against DMRG
- This demonstrates Hamed's point: "scale to hardware (>30 qubits)"

**Experiment 3E: Critical region stress test**
- N=20, h ∈ {0.9, 0.95, 1.0, 1.05, 1.1} (near critical point)
- MPS bond dimension sweep: {128, 256, 512, 1024}
- Measure: at what chi does MPS break down near criticality?
- Expected: entanglement grows logarithmically at criticality for 1D TFIM,
  so MPS should still work but need higher chi

### Success Criteria
- 3A/3B: MPS energy within 1e-4 of statevector → validated
- 3C: N=20 VQE completes in < 10 min with ΔE/gap < 5% → scaling demonstrated
- 3D: N=30 VQE completes → Hamed's "above 30 qubits" target met
- 3E: Identify chi threshold for critical region → practical guidance

### Controls
- Same initial parameters for MPS and statevector comparisons
- DMRG ground truth as independent reference
- Report wall time and peak memory for each configuration

---

## Technique 4: SPSA for Hardware VQE

### Goal
Establish SPSA as the optimal hardware optimizer with tuned hyperparameters,
and integrate it into the Phase 4 deployment pipeline.

### Preliminary Finding
SPSA wins 3-4× over COBYLA under shot noise at N=6, h=1.5. But the test was limited:
- Only one h-value
- Only Gaussian shot noise (no coherent errors)
- SPSA hyperparameters not tuned (used textbook defaults)
- Did not test with warm-start (our actual use case)

### Full Test Protocol

**Experiment 4A: SPSA hyperparameter tuning**
- N=6, h=1.5, n_shots=4096
- Grid search over SPSA parameters:
  - a ∈ {0.05, 0.1, 0.2, 0.5}
  - c ∈ {0.05, 0.1, 0.2}
  - α ∈ {0.602} (standard)
  - γ ∈ {0.101} (standard)
  - A ∈ {10, 20, 50} (stability constant)
- 10 seeds per configuration
- Metric: avg ΔE after 200 iterations

**Experiment 4B: SPSA with warm-start (our use case)**
- Initial guess = MPNN-predicted θ (not random)
- This is the actual Phase 4 scenario: MPNN predicts θ, then SPSA refines on hardware
- Compare: SPSA refinement vs no refinement (direct MPNN prediction)
- n_iterations ∈ {10, 20, 50, 100, 200}
- Identify minimum iterations needed for meaningful improvement

**Experiment 4C: Full noise model (FakeTorino)**
- Replace Gaussian shot noise with FakeTorino BackendEstimatorV2
- This includes: gate errors, readout errors, T1/T2 decoherence, crosstalk
- Compare SPSA vs COBYLA under realistic noise
- N=6, h ∈ {1.25, 1.5, 2.0}
- n_shots = 8192 (our planned hardware budget)

**Experiment 4D: Scaling to N=10 under noise**
- Same as 4C but N=10
- Tests whether SPSA advantage holds at larger system size
- Expected: SPSA should still win, but convergence may be slower

**Experiment 4E: SPSA + ZNE integration**
- Run SPSA optimization where each energy evaluation uses ZNE (3 layouts)
- This is the full Phase 4 stack: MPNN → SPSA refinement → ZNE per evaluation
- N=6 only (ZNE works at N=6)
- Compare: SPSA+ZNE vs SPSA-only vs no-refinement

### Success Criteria
- 4A: Identify optimal (a, c, A) for our setting
- 4B: If SPSA improves MPNN prediction by >10% in <50 iterations → integrate into pipeline
- 4C: SPSA still wins under full noise model → confirmed for hardware
- 4E: SPSA+ZNE achieves ΔE/gap < 3% → thesis-ready result

### Controls
- Same initial parameters across optimizers
- Same noise model for fair comparison
- Report both "best found" and "final iterate" (SPSA can oscillate)

---

## Technique 5: Noise-Aware MPNN Training

### Goal
Determine whether training the MPNN on noisy VQE data improves hardware predictions
when realistic noise (not just shot noise) is present.

### Preliminary Finding
Under pure shot noise, noiseless-trained MPNN wins (4.16e-02 vs 1.26e-01).
This is because shot noise makes VQE find worse parameters, and the MPNN learns those.
The hypothesis requires coherent gate errors to be meaningful.

### Full Test Protocol

**Experiment 5A: FakeTorino noisy VQE training data**
- Generate VQE training data using FakeTorino BackendEstimatorV2
- N=6, 27 h-values (standard grid), descending sweep
- Optimizer: COBYLA (gradient-free, appropriate for noisy backend)
- n_shots = 8192, maxiter = 300
- Compare θ_opt(noisy) vs θ_opt(noiseless) — how different are they?

**Experiment 5B: Train MPNN on FakeTorino data**
- Train MPNN on noisy θ_opt from 5A
- Train MPNN on noiseless θ_opt (control)
- Evaluate both on FakeTorino deployment (simulating hardware)
- Metrics: energy under noise, phase classification accuracy
- N=6, test at h ∈ {1.25, 1.4, 1.5}

**Experiment 5C: Mixed training (noiseless + noisy)**
- Train MPNN on combined dataset: noiseless θ_opt + noisy θ_opt
- This gives the model both the "ideal target" and "noise-adapted target"
- Architecture: add noise-level as input feature (0 for noiseless, 1 for noisy)
- At inference: set noise-level=1 for hardware deployment

**Experiment 5D: Scaling to N=10 with noise**
- Same as 5B but N=10
- FakeTorino noisy VQE at N=10 (much noisier — higher CES)
- Expected: noise-aware training may help MORE at N=10 where noise is dominant

**Experiment 5E: Iterative noise-aware refinement**
- Round 1: Train MPNN on noiseless data, deploy on FakeTorino, collect results
- Round 2: Use Round 1 hardware results as additional training data
- Round 3: Retrain MPNN on combined (noiseless + hardware) data
- Tests whether iterative refinement converges to better hardware predictions
- Reference: active learning (Miao et al. 2024)

### Success Criteria
- 5B: If noise-aware MPNN achieves lower ΔE on FakeTorino → adopt for hardware
- 5C: If mixed training beats both → novel contribution
- 5D: If noise-aware helps at N=10 → critical for thesis hardware results
- 5E: If iterative refinement converges in ≤3 rounds → practical methodology

### Controls
- Same h-grid and test points across all variants
- Same MPNN architecture and training hyperparameters
- Evaluate on BOTH noiseless (ceiling) and noisy (realistic) backends
- Report phase classification accuracy (not just energy)

---

## Execution Order (Recommended)

### Phase A: Quick wins (validate preliminary results rigorously)

1. **Technique 4A** — SPSA hyperparameter tuning (~5 min)
2. **Technique 4B** — SPSA with warm-start (~5 min)
3. **Technique 1A** — Nevergrad fair budget (~10 min)
4. **Technique 3A** — MPS accuracy at N=6 (~5 min)

### Phase B: Scaling tests

5. **Technique 3B** — MPS accuracy at N=10 (~10 min)
6. **Technique 3C** — MPS scaling to N=20 (~15 min)
7. **Technique 1C** — Nevergrad at N=10 (~15 min)
8. **Technique 2B** — QRC vs MPNN at N=10 (~20 min)

### Phase C: Noise-aware experiments (require FakeTorino, slower)

9. **Technique 5A** — Generate FakeTorino training data (~20 min)
10. **Technique 4C** — SPSA under FakeTorino noise (~10 min)
11. **Technique 5B** — Noise-aware MPNN training (~15 min)
12. **Technique 4D** — SPSA at N=10 under noise (~15 min)

### Phase D: Advanced / novel contributions

13. **Technique 2A** — Reservoir design comparison (~15 min)
14. **Technique 2D** — Hybrid QRC+MPNN (~20 min)
15. **Technique 5C** — Mixed training (~15 min)
16. **Technique 4E** — SPSA + ZNE integration (~10 min)
17. **Technique 5E** — Iterative refinement (~30 min)

### Phase E: Stretch goals (if time permits)

18. **Technique 3D** — MPS at N=30 (~30 min)
19. **Technique 3E** — MPS critical region stress test (~20 min)
20. **Technique 1B** — Nevergrad with warm-start (~10 min)
21. **Technique 5D** — Noise-aware at N=10 (~30 min)
22. **Technique 2C** — Training set size sensitivity (~15 min)

---

## Global Metrics & Reporting

For every experiment, report:

1. **Energy error** (ΔE = |E_method - E_exact|)
2. **Relative error** (ΔE/gap)
3. **Phase classification** (correct/incorrect based on ⟨X⟩ vs ⟨ZZ⟩)
4. **Wall time** (seconds)
5. **Function evaluations** (quantum circuit executions)
6. **Reproducibility** (std across seeds)

All results saved as JSON in `scripts/experiments_hamed_v7/results/`.
Summary tables appended to `documentation/binnacles/binnacle-hamed-v7.md`.

---

## Decision Framework

After all experiments, use this framework to decide what enters the main pipeline:

| Condition | Action |
|-----------|--------|
| Method beats current pipeline at N=10 | Integrate into main pipeline |
| Method is competitive but doesn't scale | Document in thesis "Future Work" |
| Method loses across all settings | Document as "validated rejection" |
| Method wins only in specific regime | Document regime and conditions |

---

## Dependencies & Prerequisites

- `nevergrad` — already installed (v1.0.12)
- `qiskit-aer` — already installed (v0.17.2)
- `scikit-learn` — already installed
- `torch`, `torch_geometric` — already installed
- FakeTorino — available via `qiskit_ibm_runtime.fake_provider`
- No additional installations needed

---

## Script Improvements Applied (2026-05-18)

Issues identified and fixed in the preliminary scripts:

| Script | Issue | Fix |
|--------|-------|-----|
| All | Usage docstrings referenced old path `experiments_hamed/` | Updated to `experiments_hamed_v7/` |
| `experiment_nevergrad.py` | NGOpt fails on 4-param arrays | Removed from default list, added TwoPointsDE |
| `experiment_nevergrad.py` | Unused imports (VQEOptimizer, VQEConfig, asdict) | Removed |
| `experiment_spsa_hardware.py` | Extra `cost_fn(theta)` eval per iteration inflated SPSA count | Refactored to track best from perturbation evals only |
| `experiment_noise_aware_training.py` | Different random initial guess for noiseless vs noisy sweep | Both now use same `common_initial_guess` |
| `experiment_noise_aware_training.py` | Unused VQEConfig import | Removed |
| `experiment_qrc_warmstart.py` | Unused SparsePauliOp import in build_reservoir | Removed |
| `experiment_mps_simulation.py` | Unused `bounds` variable in `run_vqe_mps` | Removed |
| `experiment_mps_simulation.py` | Closure captures loop variable in `cost_sv` | Explicit parameter binding |
| All | No CLI arguments for running plan sub-experiments | Added argparse to all scripts |

### CLI Usage (for plan sub-experiments)

```bash
# Experiment 1A: Fair budget at N=6
python scripts/experiments_hamed_v7/experiment_nevergrad.py --N 6 --budget 1000 --seeds 5

# Experiment 1C: Scaling to N=10
python scripts/experiments_hamed_v7/experiment_nevergrad.py --N 10 --budget 2000 --seeds 5 --h-values 1.0 1.25 1.5 2.0

# Experiment 2B: QRC at N=10
python scripts/experiments_hamed_v7/experiment_qrc_warmstart.py --N 10 --n-train 27

# Experiment 4 with more shots
python scripts/experiments_hamed_v7/experiment_spsa_hardware.py --N 10 --shots 4096 8192 16384

# Experiment 5 at N=10
python scripts/experiments_hamed_v7/experiment_noise_aware_training.py --N 10 --n-train 20 --n-shots 8192
```
