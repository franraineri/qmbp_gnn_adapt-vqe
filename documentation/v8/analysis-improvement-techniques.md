# Analysis: Improvement Techniques for Simulation & Hardware

> Based on V7 experiment results (12 experiments), V6.1 baselines (60+ runs),
> and literature review. Identifies actionable techniques ranked by expected impact.

---

## Current Performance Ceiling

| System | Best ΔE/gap | Bottleneck | What limits further improvement |
|--------|-------------|------------|--------------------------------|
| N=6, h=1.5 | 1.4% | HVA p=2 expressibility | Circuit can't reach exact GS (fid=0.997) |
| N=10, h=1.5 | 2.7% | HVA p=2 expressibility | error_from_mpnn=0.000, all error is circuit |
| N=20, h=2.0 | ~1.0% | VQE convergence | L-BFGS-B finds good minimum in easy regime |
| N=20, h=1.5 | ~7.7% | HVA p=2 + optimizer | Both expressibility and convergence |
| Hardware N=6 | ~5% (projected) | Shot noise + coherent errors | ZNE helps (+40%), DD untested |
| Hardware N=10 | Unknown | ZNE fails (non-perturbative) | Need DD + O(n) layouts |

---

## Tier 1: High-Impact Techniques (Actionable Now)

### 1.1 Dynamical Decoupling (DD) Pre-Mitigation — Hardware

**What:** Insert DD sequences (XY4, CPMG, or learned sequences) during idle periods
to suppress decoherence before ZNE extrapolation.

**Why it helps:** At N=10, CES is too large for ZNE to work (non-perturbative regime).
DD reduces effective CES back into the perturbative regime where ZNE's linear
extrapolation is valid. Pokharel et al. (2025) demonstrated this on 100 IBM qubits.

**Expected impact:** Could enable ZNE at N=10 (currently fails with R²<0.05).
If DD reduces CES from 6.29 to <1.0, ZNE should recover R²>0.8.

**Implementation:** `PadDynamicalDecoupling` pass in Qiskit's transpilation pipeline.
Already available in `generate_preset_pass_manager`. Cannot test locally (YGate not
in FakeTorino basis gates) — requires real IBM Torino.

**Status:** Blocked on hardware access. First priority when IBM Torino is available.

---

### 1.2 Multi-Restart MPS VQE at N=20 — Simulation

**What:** Run the full V6.1 pipeline (Phase 1→4) at N=20 using MPS as the backend,
with the production VQE config (5 restarts, maxiter=1000, descending sweep).

**Why it helps:** 3C showed single L-BFGS-B from warm-start achieves ΔE/gap≈7.7% at h=1.5.
The V6.1 pipeline uses 5 restarts which was the highest-impact change at N=6 (30% improvement).
At N=20, restarts should push h=1.5 closer to 5%.

**Expected impact:** h=1.5 from ~7.7% → ~5-6% (borderline pass). h=2.0 already passes (1%).

**Implementation:** Already partially done (3C uses 3 restarts). Increase to 5 restarts
and use the full `VQEOptimizer.descending_sweep` with MPS backend.

**Status:** Ready to implement. Requires modifying 3C to use `VQEOptimizer` directly.

---

### 1.3 SPSA with Optimal Config on Real Hardware — Hardware

**What:** Deploy MPNN prediction → SPSA refinement (a=0.1, c=0.05, A=10) on IBM Torino
at N=6, h=1.5 with 8192 shots.

**Why it helps:** 4A found optimal SPSA config. 4C confirmed SPSA 3× better than COBYLA
under FakeTorino noise. On real hardware, SPSA refinement may help (unlike simulation
where 4B showed it hurts — because real hardware has systematic errors that SPSA can
correct, unlike random shot noise).

**Expected impact:** ΔE/gap < 5% at N=6 on real hardware (currently projected ~5%).

**Implementation:** Use `HardwareDeployerV61(mode="hardware")` with SPSA post-processing.
Config ready from 4A results.

**Status:** Blocked on IBM Torino access.

---

## Tier 2: Medium-Impact Techniques (Worth Investigating)

### 2.1 CLP-ZNE with O(n) Layouts — Hardware

**What:** Instead of 3 random layouts, use n cyclic-permutation layouts (Rabinovich et al. 2025).
For N=10, this means 10 layouts instead of 3.

**Why it helps:** 3 layouts fail at N=10 (R²<0.05) because the CES range is too narrow
for reliable extrapolation. O(n) layouts provide more data points spanning a wider CES range,
enabling better linear fit even in the non-perturbative regime.

**Expected impact:** May recover ZNE at N=10 (R²>0.8) without DD. Cost: 10× more circuit executions.

**Implementation:** Modify `LayoutSelector` to generate cyclic permutations instead of random layouts.

**Status:** Can be implemented locally but validated only on hardware.

---

### 2.2 Noise-Aware Training with Coherent Errors — Simulation/Hardware

**What:** Re-run 5B but generate noisy training data through FakeTorino's `BackendEstimatorV2`
(full coherent + incoherent noise model) instead of Gaussian shot noise.

**Why it helps:** 5B failed because shot noise produces scattered θ (unlearnable).
Coherent gate errors produce systematically shifted θ (learnable). The MPNN could learn
to predict θ that are optimal UNDER the specific noise profile of the hardware.

**Expected impact:** Unknown — this is the genuinely open question. If coherent errors
create a learnable shift, noise-aware MPNN could outperform noiseless MPNN on hardware.

**Implementation:** Replace `noisy_cost_function` with `BackendEstimatorV2(FakeTorino())`
in the VQE sweep. Very slow (~90 min for 27 h-points) but feasible.

**Status:** Implementable but time-intensive. Lower priority than DD.

---

### 2.3 Adaptive h-Grid Near Valid Regime Boundary — Simulation

**What:** Instead of uniform 27-point grid, concentrate training points near the
valid regime boundary (h≈1.25 for N=6, h≈1.5 for N=10, h≈2.0 for N=20).

**Why it helps:** The MPNN's prediction error is highest at the regime boundary
(where θ changes most rapidly). More training data in this region improves
interpolation accuracy exactly where it matters.

**Expected impact:** Could push the valid regime boundary lower by 0.1-0.2 in h
(e.g., N=10 from h≥1.5 to h≥1.4). Modest but meaningful for thesis.

**Implementation:** Modify `generate_h_grid` to use non-uniform spacing with
higher density near the estimated boundary.

**Status:** Easy to implement. Low risk.

---

### 2.4 Transfer Learning Across System Sizes — Simulation

**What:** Pre-train MPNN on N=6 data (cheap, many points), then fine-tune on N=10 data
(expensive, fewer points). The graph structure changes but the physics is the same.

**Why it DOESN'T help:** [VERIFIED, 2026-05-18] Tested with 3 seeds. Baseline wins by 7%.
N=6 and N=10 have different optimal θ landscapes — pre-training biases weights toward
N=6 patterns that don't transfer. Combined training (N=6+N=10 together) is also 4% worse
due to conflicting targets.

**Status:** Definitively rejected. Do not revisit.

---

## Tier 3: Low-Impact / Speculative Techniques

### 3.1 Higher HVA Depth (p=3) — Simulation

**What:** Increase circuit depth from p=2 to p=3 (6 parameters instead of 4).

**Why it would help:** The expressibility ceiling is the dominant bottleneck.
p=3 can express more entanglement, potentially reaching the exact GS at h=1.25.
Sumeet et al. (2025) showed N/2 layers needed for thermodynamic limit.

**Why NOT to do it:** Violates Mele et al. (2026) constraint — deeper circuits
have exponentially worse noise on hardware. The thesis architecture is p≤2.
Also, p=3 at N=10 means 6 parameters → harder optimization landscape.

**Status:** Explicitly excluded by project constraints. Document as "future work."

---

### 3.2 ADAPT-VQE with More Iterations — Simulation

**What:** Allow ADAPT-VQE to add more than 2 layers during deployment refinement.

**Why NOT:** Violates the ADAPT iterations ≤ 2 constraint. Each ADAPT iteration
adds circuit depth, worsening hardware noise. The current 2-iteration limit
is already at the boundary of what's feasible on NISQ hardware.

**Status:** Excluded by constraint.

---

### 3.3 Variational Quantum Eigensolver with Natural Gradients — Simulation

**What:** Replace L-BFGS-B with quantum natural gradient (QNG) optimizer.

**Why NOT:** QNG requires computing the quantum Fisher information matrix (O(p²) circuit
evaluations per step). For p=2 (4 params), this is 16 extra evaluations per step.
L-BFGS-B already converges in ~100 steps with 5 restarts. QNG would be slower
with no benefit for our small parameter space.

**Status:** Not worth the complexity for 4-8 parameters.

---

## Recommended Execution Order

| Priority | Technique | Where | Time | Blocked? |
|----------|-----------|-------|------|----------|
| 1 | DD + ZNE on IBM Torino (N=6) | Hardware | 30 min | IBM access |
| 2 | SPSA refinement on IBM Torino (N=6) | Hardware | 15 min | IBM access |
| 3 | Full V6.1 pipeline at N=20 via MPS | Simulation | 30 min | No |
| 4 | CLP-ZNE with 10 layouts (N=10) | Hardware | 60 min | IBM access |
| 5 | Noise-aware with FakeTorino coherent errors | Simulation | 90 min | No |
| 6 | Adaptive h-grid near boundary | Simulation | 20 min | No |
| 7 | Transfer learning N=6→N=10 | Simulation | 30 min | No |

---

## Key Insight: The Bottleneck Has Shifted

The V7 experiments definitively showed that the pipeline's ML components (Phase 3)
are NOT the bottleneck. The bottleneck is:

1. **In simulation:** HVA p=2 expressibility (physics limit, cannot be fixed without
   violating depth constraint)
2. **On hardware:** Noise mitigation effectiveness (DD + ZNE + TREX stack)

The path forward is NOT better ML — it's better error mitigation on real hardware.
This aligns with the project's active priority: IBM Torino deployment.
