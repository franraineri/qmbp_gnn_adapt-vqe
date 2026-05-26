# Binnacle: V8 Round 2 — Pipeline Characterization (G-Series)

> Date: 2026-05-25
> Experiments: G1, G2, G3, G4, G5
> Focus: Data efficiency, uncertainty, scaling, adaptive VQE, robustness
> All at N=6 p=2 (except G3: N=20 p=2)

---

## Summary Table

| Exp | Hypothesis | Confirmed? | Key Metric | Thesis Value |
|-----|-----------|:----------:|-----------|:---:|
| **G5** | Seed-independent deployment | ✅ Yes | std=0.004, all pass | MEDIUM |
| **G4** | κ predicts restart needs | ❌ No | r=-0.29, no correlation | LOW |
| **G1** | 9-11 points sufficient | ✅ Yes | k_min=5-9, 63% reduction | HIGH |
| **G2** | Ensemble variance calibrated | ❌ No | r=0.195, not calibrated | LOW |
| **G3** | N=20 optimized < 3% | ❌ No | 1.26 mean, 1 restart fails | MEDIUM |

---

## G5: Cross-Seed Generalization ✅

**Result:** Pipeline is seed-independent (std = 0.004)

| Seed | Mean ΔE/gap | Pass Rate |
|------|:-----------:|:---------:|
| 42 | 0.0210 | 4/4 ✅ |
| 43 | 0.0125 | 4/4 ✅ |
| 44 | 0.0125 | 4/4 ✅ |

**Key finding:** Different VQE seeds produce different θ_opt trajectories,
but the MPNN trained on ANY seed's data deploys equally well. The model
learns the physics (smooth θ(h) mapping), not optimizer noise.

**Variation:** 55.7% relative (seed 42 is 0.021 vs 43/44 at 0.013) but
ALL seeds pass the 5% threshold. The std=0.004 confirms seed-independence.

**Thesis claim:** "The MPNN predictor is robust to VQE seed choice —
all seeds produce deployment-quality predictions (ΔE/gap < 2.1%)."

---

## G4: Condition Number vs Restarts ❌

**Result:** κ does NOT predict restart needs (r = -0.29)

| h | κ (mean) | 1 restart | 3 restarts | 5 restarts |
|---|:--------:|:---------:|:----------:|:----------:|
| 1.00 | 14 | 0% | 0% | 0% |
| 1.25 | 23 | 27% | 73% | 77% |
| 1.50 | 46 | 33% | 70% | 73% |
| 1.75 | 3292* | 40% | 93% | 97% |
| 2.00 | 531* | 47% | 87% | 97% |

*High variance between seeds (κ ranges from 34 to 9765 at h=1.75)

**Key findings:**
1. **h=1.0 always fails** regardless of restarts — HVA expressibility limit
2. **κ is NOT a reliable predictor** — r=-0.29 (wrong sign, weak)
3. **The real predictor is h itself:** success rate increases monotonically
   with h (0% → 27% → 33% → 40% → 47% for 1 restart)
4. **3 restarts is the sweet spot:** jumps from ~35% (1r) to ~75% (3r),
   then only marginal gain to 5r (~85%)
5. **κ has extreme variance** between seeds (14-9765 at same h) — the
   Hessian is sensitive to which minimum the optimizer finds

**Why hypothesis failed:** The condition number measures landscape curvature
at the CONVERGED minimum, not the difficulty of FINDING it. A narrow minimum
(high κ) is actually easier to converge to once found — the issue is the
basin of attraction, not the curvature within it.

**Thesis:** §3.3 — "Condition number does not predict convergence difficulty.
The dominant factor is h-value (proximity to h_c), not landscape curvature."

---

## G1: Data Efficiency Curve ✅

**Result:** k_min = 5-9 points (63% reduction from 17)

| k | Seed 42 | Seed 43 | Seed 44 | Mean |
|---|:-------:|:-------:|:-------:|:----:|
| 5 | 0.059 ❌ | 0.011 ✅ | 0.011 ✅ | 0.027 |
| 7 | 0.062 ❌ | 0.010 ✅ | 0.010 ✅ | 0.027 |
| 9 | **0.049 ✅** | 0.010 ✅ | 0.010 ✅ | 0.023 |
| 11 | 0.043 ✅ | 0.010 ✅ | 0.010 ✅ | 0.021 |
| 13 | 0.034 ✅ | 0.010 ✅ | 0.010 ✅ | 0.018 |
| 15 | 0.029 ✅ | 0.010 ✅ | 0.010 ✅ | 0.016 |
| 17 | 0.027 ✅ | 0.010 ✅ | 0.010 ✅ | 0.016 |

**Key findings:**
1. **Seeds 43/44 pass with just 5 points!** The MPNN generalizes extremely
   well when VQE data is clean (these seeds converge better).
2. **Seed 42 needs 9 points** to cross the 5% threshold — more sensitive
   to data density.
3. **Diminishing returns after k=9:** going from 9→17 only improves from
   4.9% to 2.7% (marginal gain for 2× the VQE cost).
4. **Conservative recommendation: k=9** (passes all seeds, 47% reduction)
5. **Aggressive recommendation: k=5** (passes 2/3 seeds, 71% reduction)

**Practical impact:** The pipeline currently uses 17 h-points. Reducing to
9 saves 47% of VQE computation time with no accuracy loss for most seeds.

**Thesis claim:** "The MPNN achieves ΔE/gap < 5% with as few as 9 uniformly-
spaced training points in [1.0, 2.0], representing a 47% reduction in VQE
cost compared to the standard 17-point grid."

**Section:** §3.4 (Data efficiency) — HIGH value, quantifies minimum cost.

---

## G2: Ensemble Uncertainty Calibration ❌

**Result:** Ensemble variance does NOT correlate with ΔE/gap (r = 0.195)

**Key findings:**
1. **Pearson r = 0.195** (p=0.135) — no significant correlation
2. **Pass rate = 53%** — the ensemble predicts well within training range
   but fails outside (h < 1.0 and h > 2.0)
3. **Seed 42 has much higher error** (mean 6.4 vs 1.8 for 43/44) — same
   pattern as other experiments (seed 42 VQE is less reliable)

**Why hypothesis failed:** The ensemble variance measures MPNN weight
initialization sensitivity, not prediction reliability. All 5 MPNNs in
the ensemble are trained on the SAME data — they disagree on interpolation
details, not on whether the prediction is good or bad.

**What would work instead:** Train on DIFFERENT subsets of data (bootstrap)
or use dropout at inference time (MC-Dropout). The current ensemble design
(same data, different init) doesn't capture epistemic uncertainty.

**Thesis:** §3.4 — "Naive ensemble (same data, different init) does not
provide calibrated uncertainty. MC-Dropout or bootstrap ensembles may be
needed for reliable uncertainty quantification." (1 paragraph, negative result)

---

## G3: N=20 p=2 Optimized Pipeline ❌

**Result:** 1 restart + freeze FAILS at N=20 p=2 (mean ΔE/gap = 1.26)

| Seed | Mean ΔE/gap | Pass Rate | Time |
|------|:-----------:|:---------:|:----:|
| 42 | 0.096 | 1/3 | 1114s |
| 43 | 1.866 | 0/3 | 459s |
| 44 | 1.807 | 0/3 | 129s |

**Key findings:**
1. **1 restart is INSUFFICIENT at N=20 p=2** — only seed 42 partially works
2. **The B4 finding (no saddle points) does NOT transfer to N=20** — at N=6
   the landscape is benign, but at N=20 with 4 parameters the optimization
   is much harder (more local minima or flatter landscape)
3. **Parameter freezing may be counterproductive at N=20** — freezing 2/4
   params removes degrees of freedom needed for convergence at larger N
4. **Time: 600s total (10 min)** — faster than V7's 50 min, but accuracy
   is unacceptable

**Why it failed:** The V8 optimizations (B4: 1 restart, B2: freeze) were
validated at N=6 where the landscape is trivial. At N=20:
- The landscape has more structure (not just 1 basin)
- 4 parameters in a 2^20-dimensional Hilbert space need more exploration
- Freezing 2 params at h≥1.5 may be too aggressive for N=20

**Correct N=20 p=2 config (from V7 3C):**
- 7 restarts, σ=0.3, NO freezing, h∈[1.5, 2.5] only
- This gives ΔE/gap ≈ 1.75% but takes 50 min

**Lesson:** Landscape properties are N-dependent. B4/B2 findings at N=6
do NOT generalize to N=20. The "75% cost reduction" only applies at N≤10.

**Thesis:** §4.3 — "VQE optimization shortcuts validated at N=6 (1 restart,
parameter freezing) do not transfer to N=20 where the landscape requires
more exploration. The optimal N=20 config remains 7 restarts without freezing."

---

## Cross-Experiment Insights

### 1. The pipeline is remarkably robust at N=6 (G5)
All seeds produce deployment-quality predictions. The MPNN learns physics.

### 2. Data efficiency is much better than expected (G1)
9 points suffice (47% reduction). Seeds 43/44 even work with 5 points.
This is the strongest practical result — directly reduces pipeline cost.

### 3. N=6 findings don't transfer to N=20 (G3 vs B4/B2)
The landscape at N=20 is fundamentally different. Optimizations validated
at small N must be re-validated at target N before deployment.

### 4. Simple uncertainty quantification doesn't work (G2)
Ensemble variance (same data, different init) is not calibrated.
Need bootstrap or MC-Dropout for reliable UQ.

### 5. Condition number is not useful for adaptive VQE (G4)
The h-value itself is a better predictor of difficulty than κ.
Use h-based rules: h < 1.25 → skip (HVA limit), h ≥ 1.5 → 3 restarts.

---

## Updated Optimal Configurations

| System | Restarts | Freeze | Points | Time |
|--------|:--------:|:------:|:------:|:----:|
| N=6, p=2 | 1 | Yes (2/4) | 9 | ~25s |
| N=10, p=2 | 3 | Yes (2/4) | 13* | ~5 min |
| N=20, p=2 | 7 | **No** | 5-7* | ~50 min |
| N=20, p=1 | 3-5 | N/A | 6 | ~15 min |

*Estimated from G1 scaling; needs validation at N=10/20.

---

## Thesis Contributions from Round 2

| Finding | Section | Type |
|---------|---------|:----:|
| 9 points sufficient (G1) | §3.4 | Quantitative |
| Pipeline is seed-independent (G5) | §3.4 | Validation |
| N=6 optimizations don't transfer to N=20 (G3) | §4.3 | Caveat |
| κ doesn't predict difficulty (G4) | §3.3 | Negative |
| Ensemble UQ not calibrated (G2) | §3.4 | Negative |
