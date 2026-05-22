# Plan V8: Noiseless Simulation Experiments — Final Selection

> Cross-referenced with curated bibliography, V6.1/V7 results, and existing V8 infrastructure.
> Date: 2026-05-22
> Scope: Only experiments that produce NEW learning, are noiseless, and are locally executable.

---

## Current Implementation Status (as of 2026-05-22)

| Component | Implemented | Executed | Notes |
|-----------|:-----------:|:--------:|-------|
| **Core framework** | ✅ | — | `base_experiment.py`, `config.py`, `metrics.py`, `landscape.py`, `result_store.py` |
| **CLI runner** | ✅ | — | `run_experiment.py`, `compare_results.py` |
| **Technique: sign_equivariant** | ✅ | — | C3 ready |
| **Technique: hessian_restart** | ✅ | — | B4 ready |
| **Technique: parameter_freezing** | ✅ | — | B2 ready |
| **Technique: active_learning** | ✅ | — | E3 ready |
| **Technique: physics_loss** | ✅ | — | C1 ready |
| **Technique: dypp** | ✅ | — | F1 ready |
| **Technique: analytical_init** | ✅ | — | B1 ready |
| **Exp A3 (scaling law)** | ✅ | ✅ | Results available |
| **Exp B1 (analytical init)** | ✅ | ✅ | Results available |
| **Exp B4 (Hessian)** | ✅ | ✅ | Results available |
| **Exp F1 (DyPP)** | ✅ | ✅ | Results available |
| **Exp F3 (fluctuation)** | ✅ | ✅ | Results available |
| **Exp B2 (freezing)** | ✅ | ⬜ | Script exists, not yet run |
| **Exp C3 (sign)** | ✅ | ⬜ | Script exists, not yet run |
| **Exp C1 (physics loss)** | ❌ | ⬜ | Registered but script missing → ImportError |
| **Exp D1 (weight space)** | ❌ | ⬜ | Registered but script missing → ImportError |
| **Exp E3 (active learning)** | ❌ | ⬜ | Registered but script missing → ImportError |
| **Exp E4 (longitudinal)** | ❌ | ⬜ | Registered but script missing → ImportError |
| **Baseline generation** | ❌ | ⬜ | No `generate_baselines.py` script yet |

---

## Selection Criteria Applied

1. **Not already answered** by V6.1 or V7 (checked binnacles + RESULTS_SUMMARY)
2. **Supported by literature** (cross-referenced with bibliography_curated.md)
3. **Noiseless only** — no FakeTorino, no shot noise, pure statevector/MPS
4. **Locally executable** — no IBM credentials, no N=12+ statevector (too slow)
5. **Produces thesis-quality output** — figures, tables, or validated claims
6. **Includes warm-start vs cold-start comparison** in every VQE-based experiment

---

## Experiments Already Run (V8 infrastructure)

| ID | Status | Result File |
|----|--------|-------------|
| A3 | ✅ Run | `results/exp_a3/run_20260522_110943.json` |
| B1 | ✅ Run | `results/exp_b1/run_20260522_110911.json` |
| B4 | ✅ Run | `results/exp_b4/` |
| F1 | ✅ Run | `results/exp_f1/` |
| F3 | ✅ Run | `results/exp_f3/run_20260522_110855.json` |

---

## Final Experiment Selection (8 experiments, ~20h total)

### Priority Order

| # | ID | Name | Time | Thesis Value | Bibliography Support |
|---|-----|------|------|-------------|---------------------|
| 1 | C3 | Sign canonicalization (p=1 N=20) | 1h | HIGH | Wiersema et al. 2020 (Z₂ symmetry) |
| 2 | D1 | Weight-space phase detection | 3h | HIGH | Hernandes et al. 2025 (arXiv:2503.17140) |
| 3 | B4 | Hessian-guided restarts | 3h | MEDIUM-HIGH | Landscape analysis (Cerezo et al. 2021) |
| 4 | E4 | TFIM + longitudinal field | 4h | HIGH | Dutta et al. 2015 (TFIM monograph) |
| 5 | B2 | TITAN parameter freezing | 3h | MEDIUM | Peng et al. 2025 (arXiv:2509.15193) |
| 6 | E3 | Active learning h-grid | 4h | HIGH | Miao et al. 2024 (PRApplied 21, 014053) |
| 7 | C1 | Physics-informed MPNN loss | 4h | MEDIUM-HIGH | Miao et al. 2024, Zhang et al. 2025 |
| 8 | F1 | DyPP extrapolation | 2h | MEDIUM | arXiv:2307.12449 (DyPP 2023) |

### Excluded from this round (with justification)

| ID | Why excluded |
|----|-------------|
| A1 | DMRG gap — useful but not a simulation *experiment*, just infrastructure |
| A2 | TCI landscape — needs external library (xfac), high implementation effort |
| B3 | LCC — high implementation effort (8h), better as separate sprint |
| D3 | Tensor completion — needs tensor library, medium-high effort |
| E1 | N=30 pipeline — depends on C3 + B1 being validated first |
| E2 | Topology generalization — HVA expressibility is the bottleneck (Heisenberg showed this) |
| F2 | Flow-VQE — needs normalizing flow implementation, tangential |

---

## Detailed Experiment Specifications


### Experiment C3: Sign Canonicalization for p=1 N=20

**Hypothesis:** Enforcing θ_x > 0 (Strategy A) resolves the Z₂ sign ambiguity
at N=20 p=1, enabling the MPNN to learn a consistent mapping and deploy
successfully at all h ≥ 2.25 (not just h=3.0).

**Bibliography:**
- Wiersema et al. (2020) PRX Quantum 1, 020319 — documents HVA Z₂ symmetry
- Mele et al. (2022) PRA 106, L060401 — parameter transferability requires smooth θ(h)

**What's already implemented:**
- `scripts/experiments_v8/techniques/sign_equivariant.py` — `canonicalize_sign()`, `canonicalize_dataset()`, `SignInvariantLoss`, `detect_sign_inconsistency()`
- `scripts/experiments_v8/experiments/__init__.py` — C3 registered

**What needs to be created:**
- `scripts/experiments_v8/experiments/exp_c3_sign.py` — the experiment script

**Method:**
1. Run VQE descending sweep at N=20, p=1, h ∈ [2.25, 2.5, 2.75, 3.0, 3.5, 4.0], 3 seeds
2. Detect sign inconsistencies via `detect_sign_inconsistency()`
3. Apply `canonicalize_dataset()` (Strategy A: enforce θ_x > 0)
4. Train MPNN on canonicalized data (h=128, L=3, 6000 epochs)
5. Deploy at h_test = [2.5, 3.0, 3.5] — compare with/without canonicalization
6. Also test Strategy B (`SignInvariantLoss`) as alternative

**Warm-start comparison:**
- VQE with descending warm-start (standard) vs cold-start (random init per h)
- MPNN prediction (warm) vs random θ (cold) at deployment

**Metrics:**
- ΔE/gap at each h_test (threshold: < 5%)
- Number of seeds that pass (target: 3/3 with canonicalization vs 1/3 without)
- MPNN MSE with vs without canonicalization

**Success criterion:** All 3 seeds pass ΔE/gap < 5% at h ≥ 2.5 after canonicalization.

**Reuses:**
- `src/poc/v6/vqe_optimizer.py` — VQEOptimizer.descending_sweep()
- `src/poc/v6/mpnn_predictor.py` — MPNNPredictor, train_mpnn()
- `src/poc/v6/classical_solver.py` — DMRG ground truth at N=20
- `scripts/experiments_v8/techniques/sign_equivariant.py` — all strategies
- `scripts/experiments_v8/core/base_experiment.py` — BaseExperiment lifecycle

**Estimated time:** 1h (VQE at N=20 p=1 is fast: 2 params, ~30s/point)

---

### Experiment D1: Weight-Space Phase Detection

**Hypothesis:** The trained MPNN's weight gradient norm ||dW/dh|| peaks near
the quantum phase transition, enabling zero-QPU phase detection. The peak
location shifts predictably with N following finite-size scaling.

**Bibliography:**
- Hernandes et al. (2025) arXiv:2503.17140 — phase transitions in NN weight space
- Huang et al. (2022) Science 377, eabk3333 — ML can efficiently predict GS properties within a phase

**What's already implemented:**
- `src/poc/v6/analysis_utils.py` — `WeightGradientAnalyzer` (computes ||dW/dh||)
- `scripts/experiments_v8/experiments/__init__.py` — D1 registered
- Comparison A in binnacle-comparative-analysis: single Jacobian peak at h=1.77

**What needs to be created:**
- `scripts/experiments_v8/experiments/exp_d1_weight_space.py`

**Method:**
1. For N ∈ {6, 10}: train MPNN on dense h-grid (40 points, h ∈ [0.5, 2.5])
   - Two variants: (A) full h-range including invalid regime, (B) valid-regime only
2. Compute ||dW/dh|| at each h using finite differences on MPNN weights
3. Also compute: Fisher information matrix trace, singular value spectrum of weight matrices
4. Locate peaks in all three quantities
5. Compare peak location with known h_c = 1.0 and finite-size h_c(N)
6. Test: does MPNN-A (full range) detect h_c while MPNN-B (valid only) detects training boundary?

**Warm-start comparison:**
- VQE training data: warm-start sweep vs independent cold-start per h-point
- Compare: does warm-start produce smoother θ(h) → smoother weight gradients?

**Metrics:**
- Peak location h_peak for each N and training variant
- Peak sharpness (FWHM of ||dW/dh|| curve)
- Correlation between h_peak and known h_c(N)
- Fisher information trace vs h (alternative phase indicator)

**Success criterion:** Peak within ±0.2 of h_c for MPNN-A; peak at training boundary for MPNN-B.

**Reuses:**
- `src/poc/v6/mpnn_predictor.py` — full MPNN training pipeline
- `src/poc/v6/analysis_utils.py` — WeightGradientAnalyzer
- `src/poc/v6/vqe_optimizer.py` — VQE for training data generation
- `scripts/experiments_v8/core/base_experiment.py` — BaseExperiment

**Estimated time:** 3-4h (2× MPNN training at N=6 + N=10, ~30 min each + VQE data
generation with 40 dense h-points if not already available + analysis).
Note: if VQE data at 40 h-points needs to be generated from scratch, add ~1h for N=10.

---

### Experiment B4: Hessian-Guided Adaptive Restarts

**Hypothesis:** Computing the 4×4 Hessian at VQE convergence (8 extra evaluations)
identifies saddle points and provides escape directions, reducing the number of
restarts needed from 5 to 2-3 while maintaining accuracy.

**Bibliography:**
- Cerezo et al. (2021) Nature Comms 12, 1791 — landscape structure of shallow circuits
- Wiersema et al. (2020) PRX Quantum 1, 020319 — HVA has mild/absent barren plateaus

**What's already implemented:**
- `scripts/experiments_v8/core/landscape.py` — `compute_hessian()`, `analyze_critical_point()`
- `scripts/experiments_v8/core/config.py` — `VQEConfig.use_hessian_check`, `hessian_escape_threshold`
- `scripts/experiments_v8/experiments/__init__.py` — B4 registered

**What needs to be created:**
- `scripts/experiments_v8/experiments/exp_b4_hessian.py`
- `scripts/experiments_v8/techniques/hessian_restart.py`

**Method:**
1. For N ∈ {6, 10}, h ∈ {1.0, 1.25, 1.5, 2.0}, 3 seeds:
   - Standard VQE: 5 random restarts, take best (baseline)
   - Hessian-guided VQE:
     a. Run L-BFGS-B from initial guess
     b. At convergence, compute Hessian (8 extra evals for 4 params)
     c. If all eigenvalues > 0: accept (true minimum), STOP
     d. If any eigenvalue < 0: escape along negative eigenvector, restart from there
     e. Repeat until true minimum found or max 5 restarts
2. Compare: accuracy (ΔE/gap), number of restarts used, total evaluations, wall time

**Warm-start comparison:**
- Hessian-guided with warm-start init (from previous h) vs cold-start (random)
- Measure: how often does warm-start land directly in a true minimum (0 extra restarts)?

**Metrics:**
- ΔE/gap (must match 5-restart baseline within 1%)
- Mean restarts used (target: 2-3 vs 5)
- Total function evaluations (target: 40-60% reduction)
- Fraction of initial convergences that are true minima vs saddle points
- Hessian condition number (landscape curvature characterization)

**Success criterion:** Same accuracy as 5-restart with ≤3 restarts on average.

**Reuses:**
- `scripts/experiments_v8/core/landscape.py` — compute_hessian, analyze_critical_point
- `src/poc/v6/vqe_optimizer.py` — VQEOptimizer (for baseline comparison)
- `scripts/experiments_v8/core/base_experiment.py` — BaseExperiment

**Estimated time:** 3h (N=6 fast, N=10 moderate; Hessian is cheap for 4 params)


---

### Experiment E4: TFIM with Longitudinal Field (2D Phase Diagram)

**Hypothesis:** The MPNN trained on standard TFIM can predict parameters for
TFIM + longitudinal field (H = -J·ΣZZ - h·ΣX - g·ΣZ) with g as additional
node feature, demonstrating cross-model generalization within a continuous
Hamiltonian family.

**Bibliography:**
- Dutta et al. (2015) *Quantum Phase Transitions in Transverse Field Spin Models* — canonical TFIM reference, covers longitudinal field perturbation
- Huang et al. (2022) Science 377, eabk3333 — ML predicts GS properties within a phase
- Lee et al. (2026) arXiv:2602.19752 — GNN generalizes across Hamiltonians

**What's already implemented:**
- `src/poc/v6/hamiltonian_builder.py` — HamiltonianBuilder (needs extension for g·Z term)
- `scripts/experiments_v8/core/config.py` — `SystemConfig.g_longitudinal`, `SystemConfig.model`
- `scripts/experiments_v8/experiments/__init__.py` — E4 registered

**What needs to be created:**
- `scripts/experiments_v8/experiments/exp_e4_longitudinal.py`
- Small extension to `HamiltonianBuilder` (or local helper) for longitudinal field

**Method:**
1. Extend Hamiltonian: H = -J·Σ(ZᵢZᵢ₊₁) - h·Σ(Xᵢ) - g·Σ(Zᵢ)
   - g breaks Z₂ symmetry → crossover instead of sharp transition
   - g ∈ {0, 0.1, 0.2, 0.3, 0.5}, h ∈ {1.0, 1.25, 1.5, 1.75, 2.0}
2. For each (h, g): exact diag + VQE (L-BFGS-B, 5 restarts, descending h-sweep per g)
3. Verify HVA p=2 expressibility: fidelity > 0.93 for g ≤ 0.3
4. Train MPNN with extended node features: [h_i, coord_number, g_i]
   - Training: all (h, g) combinations in valid regime
   - Test: held-out (h, g) pairs (interpolation + extrapolation)
5. Deploy and measure ΔE/gap at test points

**Warm-start comparison:**
- VQE: descending h-sweep with warm-start vs cold-start per (h,g) point
- MPNN: prediction (warm) vs random θ (cold) at deployment
- Additional: does warm-start across g (fixed h, varying g) work?

**Metrics:**
- ΔE/gap at each (h, g) test point
- Fidelity vs g (at what g does HVA p=2 break down?)
- MPNN generalization: interpolation error vs extrapolation error
- Cross-g warm-start effectiveness (gain metric)

**Success criterion:**
- HVA p=2 works for g ≤ 0.3 (fidelity > 0.93)
- MPNN predicts unseen (h, g) with ΔE/gap < 5% within valid regime
- Demonstrates 2D phase diagram capability (novel for GNN-VQE literature)

**Reuses:**
- `src/poc/v6/hamiltonian_builder.py` — base builder (extend locally)
- `src/poc/v6/classical_solver.py` — exact diag (works for any SparsePauliOp)
- `src/poc/v6/hva_builder.py` — same HVA circuit (Hamiltonian-agnostic)
- `src/poc/v6/mpnn_predictor.py` — MPNN (add g as node feature)
- `src/poc/v6/vqe_optimizer.py` — VQEOptimizer

**Estimated time:** 4h (N=6 only; 25 (h,g) combinations × 5 restarts)

---

### Experiment B2: TITAN-Style Parameter Freezing

**Hypothesis:** In HVA p=2, the second-layer parameters (θ_zz2, θ_x2) have
|dθ/dh| < threshold for h ≥ 1.5, meaning they can be frozen after initial
convergence, reducing VQE cost by ~40% with < 1% accuracy loss.

**Bibliography:**
- Peng et al. (2025) TITAN, NeurIPS, arXiv:2509.15193 — trajectory-informed parameter freezing, 40-60% fewer evaluations on TFIM
- Wiersema et al. (2020) PRX Quantum 1, 020319 — HVA parameter structure

**What's already implemented:**
- `scripts/experiments_v8/core/config.py` — `VQEConfig.freeze_params`, `VQEConfig.freeze_after_h`
- `scripts/experiments_v8/experiments/__init__.py` — B2 registered

**What needs to be created:**
- `scripts/experiments_v8/experiments/exp_b2_freezing.py`
- `scripts/experiments_v8/techniques/parameter_freezing.py`

**Method:**
1. Analyze existing VQE data: compute |dθᵢ/dh| for each of 4 parameters across h-sweep
   - Use N=6 and N=10 data from V6.1 runs (or regenerate with 3 seeds)
2. Identify "frozen" parameters: |dθ/dh| < 0.05 rad per Δh=0.25
3. Run VQE with freezing strategy:
   - Phase A (h ≥ h_freeze): optimize all 4 params normally
   - Phase B (h < h_freeze): freeze identified params, optimize only active ones
4. Compare: full 4-param VQE vs frozen VQE (accuracy, time, evaluations)
5. Test at N=6 and N=10 with h_freeze ∈ {1.5, 1.75, 2.0}

**Warm-start comparison:**
- Full VQE with warm-start (baseline) vs frozen VQE with warm-start
- Cold-start frozen VQE (does freezing help even without warm-start?)

**Metrics:**
- ΔE/gap difference: frozen vs full (target: < 0.5% degradation)
- Function evaluations saved (target: 30-50% reduction)
- Wall time reduction
- Per-parameter |dθ/dh| trajectory (visualization for thesis)

**Success criterion:** < 1% accuracy loss with ≥ 30% fewer evaluations.

**Reuses:**
- `src/poc/v6/vqe_optimizer.py` — VQEOptimizer (modify bounds for frozen params)
- `scripts/experiments_v8/core/landscape.py` — compute_theta_smoothness()
- Existing V6.1 VQE data (if available in results/)

**Estimated time:** 3h (analysis of existing data + new VQE runs with freezing)

---

### Experiment E3: Active Learning for Optimal h-Grid Selection

**Hypothesis:** An ensemble-based active learning strategy selects the next
h-point based on MPNN prediction variance, reducing VQE runs by 30-50%
while maintaining ΔE/gap < 5% at deployment.

**Bibliography:**
- Miao et al. (2024) PRApplied 21, 014053 — active learning for NN-VQE, dropout uncertainty
- Zhang et al. (2025) arXiv:2505.01236 (Qracle) — GNN parameter initializer with data efficiency

**What's already implemented:**
- `scripts/experiments_v8/core/config.py` — `MPNNConfig.use_active_learning`, `n_ensemble`, `acquisition`
- `scripts/experiments_v8/experiments/__init__.py` — E3 registered

**What needs to be created:**
- `scripts/experiments_v8/experiments/exp_e3_active.py`
- `scripts/experiments_v8/techniques/active_learning.py`

**Method:**
1. Setup: N=6, p=2, h ∈ [0.8, 2.0] (valid regime), target: ΔE/gap < 5% at h_test=1.5
2. Active learning loop:
   a. Start with 5 seed points: h = {0.8, 1.0, 1.25, 1.5, 2.0}
      (Note: h=0.8 and h=1.0 are deliberately in the invalid regime for N=6.
      This tests whether active learning discovers they are uninformative.)
   b. Run VQE at seed points (with warm-start descending sweep)
   c. Train ensemble of 5 MPNNs (different random seeds)
   d. Compute prediction variance at candidate h-points
   e. Select h with max variance (exploration) → run VQE there
   f. Add to training set, retrain ensemble
   g. Repeat until ensemble variance < threshold at h_test OR max 20 points
   h. Early stopping: if variance does not decrease for 3 consecutive iterations,
      the method is not converging — abort and report partial results.
3. Compare: active learning (adaptive) vs fixed uniform grid vs random selection
4. Metric: number of VQE runs to achieve ΔE/gap < 5%

**Reproducibility:** Fix ensemble seeds to [100, 101, 102, 103, 104]. Use
`torch.use_deterministic_algorithms(True)` during MPNN training.

**Warm-start comparison:**
- Active learning with warm-start VQE (descending from nearest known point)
- Active learning with cold-start VQE (random init at each selected h)
- Fixed grid with warm-start (V6.1 baseline)

**Metrics:**
- VQE runs to reach ΔE/gap < 5% (target: 10-12 vs 17 baseline)
- Data efficiency: accuracy vs number of training points curve
- Where does active learning place points? (expect: near boundary h≈1.25)
- Ensemble calibration: does high variance correlate with high deployment error?

**Success criterion:** Achieves ΔE/gap < 5% with ≤ 12 points (vs 17 baseline = 30% reduction).

**Reuses:**
- `src/poc/v6/mpnn_predictor.py` — MPNNPredictor (train 5 instances)
- `src/poc/v6/vqe_optimizer.py` — VQEOptimizer
- `src/poc/v6/hardware_deployer_v61.py` — deploy_with_baseline() for evaluation
- `scripts/experiments_v8/core/base_experiment.py` — BaseExperiment

**Estimated time:** 4h (iterative loop with multiple MPNN trainings)

---

### Experiment C1: Physics-Informed MPNN Loss

**Hypothesis:** Adding an energy-validation term to the MPNN loss every K epochs
(evaluate E(θ_pred) on the actual Hamiltonian) prevents the MPNN from learning
parameters with low MSE but high energy error, improving ΔE/gap by 10-30% at
the valid regime boundary.

**Bibliography:**
- Miao et al. (2024) PRApplied 21, 014053 — energy-aware NN training for VQE
- Zhang et al. (2025) arXiv:2505.01236 (Qracle) — GNN with energy feedback
- Lee et al. (2026) arXiv:2602.19752 — energy-based loss for VQE parameter prediction

**What's already implemented:**
- `scripts/experiments_v8/core/config.py` — `MPNNConfig.use_physics_loss`, `physics_loss_weight`, `physics_loss_start_epoch`, `physics_loss_eval_every`
- `scripts/experiments_v8/experiments/__init__.py` — C1 registered
- Comparison 4 (binnacle-comparative-analysis): proved MSE ≠ deployment quality

**What needs to be created:**
- `scripts/experiments_v8/experiments/exp_c1_physics_loss.py`
- `scripts/experiments_v8/techniques/physics_loss.py`

**Method:**
1. Baseline: standard MPNN training (MSE-only loss, 6000 epochs) at N=6
2. Physics-informed variant:
   - First 1000 epochs: MSE-only (let weights converge)
   - Epochs 1000-6000: loss = MSE(θ) + λ · mean(|E(θ_pred) - E_exact|)
   - λ = 0.1 (tunable), evaluate energy every 100 epochs on 5 random training points
   - Overhead estimate: 50 energy evaluations total (5000 epochs / 100 × 5 points).
     At N=6: ~1s total. At N=10: ~30s total. Negligible vs 6000 epochs of MPNN training.
3. Compare at h_test = {1.0, 1.25, 1.5} (boundary region where MSE≠ΔE/gap)
4. Also test at N=10 to verify scaling

**Important constraint:** We are NOT changing the VQE cost function (V5.x lesson).
The θ targets remain pure-energy VQE optima. The energy term is a MPNN training
regularizer only.

**Warm-start comparison:**
- Deploy with MPNN prediction (warm) vs random θ (cold)
- Compare: physics-informed MPNN vs standard MPNN vs no MPNN (cold)

**Metrics:**
- ΔE/gap at boundary h-values (target: 10-30% improvement over MSE-only)
- Training MSE (should be slightly higher — tradeoff)
- Energy error during training (should decrease with physics loss)
- Correlation: MSE vs ΔE/gap (should improve with physics loss)

**Success criterion:** ΔE/gap improves by > 10% at h=1.25 (N=6) without regression at h=1.5.

**Reuses:**
- `src/poc/v6/mpnn_predictor.py` — MPNNPredictor (extend training loop)
- `src/poc/v6/vqe_optimizer.py` — for energy evaluation during training
- `scripts/experiments_v8/core/base_experiment.py` — BaseExperiment

**Estimated time:** 4h (multiple MPNN trainings with energy evaluation overhead)

---

### Experiment F1: DyPP (Dynamic Parameter Prediction) Extrapolation

**Hypothesis:** During the descending h-sweep, linear/quadratic extrapolation
from the last 2-3 converged θ(h) points predicts θ(h_{i+1}) accurately enough
to replace random restarts, reducing VQE iterations by 30-50% in the smooth regime.

**Bibliography:**
- arXiv:2307.12449 — Dynamic Parameter Prediction for VQA (2023)
- Mele et al. (2022) PRA 106, L060401 — parameter transferability in HVA
- Skogh et al. (2023) Electronic Structure 5, 035002 — parameter transfer between related Hamiltonians

**What's already implemented:**
- `scripts/experiments_v8/core/config.py` — `VQEConfig.use_dypp`, `VQEConfig.dypp_order`
- `scripts/experiments_v8/experiments/__init__.py` — F1 registered ✅
- `scripts/experiments_v8/techniques/dypp.py` — DyPP extrapolation logic ✅

**What needs to be created:**
- `scripts/experiments_v8/experiments/exp_f1_dypp.py`
- `scripts/experiments_v8/techniques/dypp.py`

**Method:**
1. Standard descending sweep at N=6, h from 2.0 to 0.8 (Δh=0.1, 13 points), 3 seeds
2. At each h_i (after converging h_{i-1} and h_{i-2}):
   - Linear DyPP: θ_pred = θ(h_i) + (θ(h_i) - θ(h_{i-1})) · Δh/Δh_prev
   - Quadratic DyPP: fit parabola through last 3 points, extrapolate
   - Standard warm-start: θ_pred = θ(h_{i-1}) (current approach)
3. Use each prediction as VQE initial guess (single L-BFGS-B, no restarts)
4. Compare: iterations to converge, final energy, accuracy

**Warm-start comparison (core of this experiment):**
- DyPP linear extrapolation vs standard warm-start (previous h only) vs cold-start
- Measure iteration savings at each h-point
- Identify where DyPP fails (near h_c where θ changes non-linearly)

**Metrics:**
- VQE iterations to converge (per h-point)
- Total function evaluations across full sweep
- ΔE/gap at each point (must match baseline)
- DyPP prediction error: |θ_dypp - θ_opt| (should be < |θ_warmstart - θ_opt|)
- Failure detection: where does DyPP extrapolation break down?

**Success criterion:** 30-50% fewer iterations in smooth regime (h > 1.5) with same accuracy.

**Reuses:**
- `src/poc/v6/vqe_optimizer.py` — VQEOptimizer (modify initial guess logic)
- `scripts/experiments_v8/core/landscape.py` — compute_theta_smoothness()
- `scripts/experiments_v8/core/base_experiment.py` — BaseExperiment

**Estimated time:** 2h (N=6 is fast; main work is implementing extrapolation logic)


---

## Cross-Experiment Comparison Framework

Every experiment automatically produces:

1. **Warm-start vs cold-start gain** — measured at every VQE point
2. **ΔE/gap vs V6.1 baseline** — via `ComparisonResult.compute()`
3. **Wall time comparison** — technique vs baseline
4. **Statistical significance** — Welch's t-test across 3 seeds

### Comparison Matrix (auto-generated after all experiments run)

```
python scripts/experiments_v8/compare_results.py --all
```

Produces:

| Experiment | ΔE/gap | vs V6.1 | Speedup | Warm-start Gain | Verdict |
|------------|--------|---------|---------|-----------------|---------|
| C3 (sign) | ? | ? | ? | ? | ? |
| D1 (weight) | N/A | N/A | N/A | N/A | Analysis |
| B4 (hessian) | ? | ? | ? | ? | ? |
| E4 (longit.) | ? | ? | ? | ? | ? |
| B2 (freeze) | ? | ? | ? | ? | ? |
| E3 (active) | ? | ? | ? | ? | ? |
| C1 (physics) | ? | ? | ? | ? | ? |
| F1 (dypp) | ? | ? | ? | ? | ? |

### Baseline Generation

Before running experiments, generate and cache V6.1 baselines via a dedicated script
(not embedded in each experiment's `setup()` to avoid redundant re-generation):

```bash
# Generate all baselines once (creates cached results for all experiments to use)
python scripts/experiments_v8/generate_baselines.py
```

Implementation in each experiment's `setup()` should only READ cached baselines:

```python
# In each experiment's setup(), load baseline (do NOT regenerate):
baseline = self.result_store.get_baseline(f"n{N}_h{h}")
if baseline is None:
    raise RuntimeError(
        f"Baseline not found for n={N}, h={h}. "
        "Run 'python scripts/experiments_v8/generate_baselines.py' first."
    )
```

---

## Auto-Logging Architecture

### Per-Experiment JSON Output

```json
{
  "config": { /* full ExperimentConfig serialized */ },
  "analysis": {
    "experiment_id": "C3",
    "hypothesis": "...",
    "summary": { "mean_de_gap": 0.023, "pass_rate": 1.0, ... },
    "per_seed": { "42": {...}, "43": {...}, "44": {...} },
    "warm_vs_cold": {
      "warm_start_mean_de_gap": 0.023,
      "cold_start_mean_de_gap": 4.12,
      "gain_pct": 99.4
    },
    "technique_specific": { /* experiment-dependent */ }
  },
  "results": {
    "42": [{ "h_value": 2.5, "energy": -12.3, ... }],
    "43": [...],
    "44": [...]
  },
  "comparison_with_baseline": {
    "experiment_id": "C3",
    "baseline_id": "v61",
    "improvement_pct": 15.2,
    "verdict": "improvement"
  },
  "environment": { "python": "3.12", "qiskit": "1.4.2", ... }
}
```

### Warm-Start Logging (added to BaseExperiment)

Every VQE-based experiment logs:

```python
warm_cold_comparison = {
    "h": h,
    "warm_start": {
        "init_theta": theta_warm.tolist(),
        "init_energy": e_warm_init,  # Energy BEFORE optimization (measures init quality)
        "final_energy": e_warm,
        "de_gap": de_gap_warm,
        "n_iterations": nit_warm,
    },
    "cold_start": {
        "init_theta": theta_cold.tolist(),
        "init_energy": e_cold_init,  # Energy BEFORE optimization
        "final_energy": e_cold,
        "de_gap": de_gap_cold,
        "n_iterations": nit_cold,
    },
    "gain_pct": (de_gap_cold - de_gap_warm) / de_gap_cold * 100,
}
```

---

## Implementation Order & Dependencies

```
C3 (sign) ──────────────────────────────────────────→ [standalone, unblocks E1 later]
     │
D1 (weight space) ─────────────────────────────────→ [standalone, uses existing MPNN]
     │
B4 (hessian) ──────────────────────────────────────→ [standalone, uses landscape.py]
     │
E4 (longitudinal) ─────────────────────────────────→ [needs small HamiltonianBuilder extension]
     │
B2 (freezing) ─────────────────────────────────────→ [uses B4's Hessian analysis as input]
     │
E3 (active learning) ──────────────────────────────→ [needs ensemble MPNN training]
     │
C1 (physics loss) ─────────────────────────────────→ [needs energy eval in training loop]
     │
F1 (dypp) ─────────────────────────────────────────→ [standalone, simple extrapolation]
```

No hard dependencies between experiments (all can run independently), but:
- B2 benefits from B4's Hessian characterization (knows which params are "flat")
- E3 benefits from C1's physics loss (better uncertainty calibration)
- C3 is prerequisite for future E1 (N=30 pipeline)

---

## Files to Create

> **Updated 2026-05-22:** Reflects actual filesystem state. Items already created are marked.

### New Experiment Scripts (4 remaining)

| File | Experiment | Lines (est.) | Status |
|------|-----------|-------------|--------|
| `experiments/exp_c3_sign.py` | C3: Sign canonicalization | ~180 | ✅ Already exists |
| `experiments/exp_d1_weight_space.py` | D1: Weight-space phase detection | ~220 | ❌ Needs creation |
| `experiments/exp_b4_hessian.py` | B4: Hessian-guided restarts | ~200 | ✅ Already exists |
| `experiments/exp_e4_longitudinal.py` | E4: TFIM + longitudinal field | ~250 | ❌ Needs creation |
| `experiments/exp_b2_freezing.py` | B2: Parameter freezing | ~180 | ✅ Already exists |
| `experiments/exp_e3_active.py` | E3: Active learning | ~250 | ❌ Needs creation |
| `experiments/exp_c1_physics_loss.py` | C1: Physics-informed loss | ~220 | ❌ Needs creation |
| `experiments/exp_f1_dypp.py` | F1: DyPP extrapolation | ~160 | ✅ Already exists |

### New Technique Modules (all already created)

| File | Technique | Lines (est.) | Status |
|------|-----------|-------------|--------|
| `techniques/hessian_restart.py` | B4: Hessian escape + adaptive restart | ~100 | ✅ Already exists |
| `techniques/parameter_freezing.py` | B2: Trajectory analysis + freezing | ~120 | ✅ Already exists |
| `techniques/active_learning.py` | E3: Ensemble uncertainty + acquisition | ~150 | ✅ Already exists |
| `techniques/physics_loss.py` | C1: Energy-validated training loss | ~100 | ✅ Already exists |
| `techniques/dypp.py` | F1: Linear/quadratic extrapolation | ~80 | ✅ Already exists |

### Modifications to Existing Files

| File | Change | Status |
|------|--------|--------|
| `experiments/__init__.py` | F1 already registered; no change needed | ✅ Done |
| `core/base_experiment.py` | Add `_run_warm_cold_comparison()` helper method | ⬜ Pending |
| `core/metrics.py` | Add `WarmColdComparison` dataclass | ⬜ Pending |
| `generate_baselines.py` | New script for baseline caching | ⬜ Pending |

### No Modifications to Stable V6 Code

All V6 modules (`hamiltonian_builder.py`, `vqe_optimizer.py`, `mpnn_predictor.py`, etc.)
are used as-is via imports. The longitudinal field extension (E4) is implemented as a
local helper in the experiment script, not by modifying `hamiltonian_builder.py`.

---

## Execution Commands

```bash
# Run individual experiments
python scripts/experiments_v8/run_experiment.py --exp C3 --verbose
python scripts/experiments_v8/run_experiment.py --exp D1 --verbose
python scripts/experiments_v8/run_experiment.py --exp B4 --verbose
python scripts/experiments_v8/run_experiment.py --exp E4 --verbose
python scripts/experiments_v8/run_experiment.py --exp B2 --verbose
python scripts/experiments_v8/run_experiment.py --exp E3 --verbose
python scripts/experiments_v8/run_experiment.py --exp C1 --verbose
python scripts/experiments_v8/run_experiment.py --exp F1 --verbose

# Run all new experiments
python scripts/experiments_v8/run_experiment.py --exp C3 D1 B4 E4 B2 E3 C1 F1

# Compare all results
python scripts/experiments_v8/compare_results.py --all

# Compare specific category
python scripts/experiments_v8/compare_results.py --category B
```

---

## Expected Thesis Contributions

| Experiment | Thesis Section | Contribution Type |
|-----------|---------------|-------------------|
| C3 | §4.6 (p=1 scaling) | Practical fix enabling N=20 deployment |
| D1 | §5.1 (Discussion) | Novel: zero-QPU phase detection from ML weights |
| B4 | §3.3 (Methodology) | Optimization improvement with theoretical backing |
| E4 | §5.5 (Generalization) | Novel: 2D phase diagram capability |
| B2 | §3.3 (Methodology) | Practical speedup validated on quantum circuits |
| E3 | §3.4 (Data efficiency) | Novel: active learning for VQE data collection |
| C1 | §3.4 (MPNN training) | Addresses known MSE≠ΔE/gap weakness |
| F1 | §3.3 (VQE optimization) | Validates DyPP principle on spin systems |

---

## Risk Assessment

| Experiment | Risk | Mitigation |
|-----------|------|-----------|
| C3 | Low — technique already implemented | Just needs experiment script |
| D1 | Low — WeightGradientAnalyzer exists | May not find h_c (finds boundary instead) |
| B4 | Low — Hessian code exists | May show HVA has no saddle points (still learning) |
| E4 | Medium — HVA may fail at g>0.3 | Document failure as expressibility limit |
| B2 | Low — analysis of existing data | May show all params are active (negative result = learning) |
| E3 | Medium — ensemble training is slow | Limit to N=6 only |
| C1 | Medium — energy eval adds overhead | May not improve if MSE≈ΔE/gap in valid regime |
| F1 | Low — simple extrapolation | May fail near h_c (expected, document boundary) |

All experiments produce learning even with negative results (validated rejection = thesis value).
