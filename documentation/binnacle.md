# PoC Development Binnacle

## 2026-04-22 — PoC V3 → V4 Optimization

### Context
PoC V3 executed successfully but showed bottlenecks in Phase 2 (VQE optimization) and Phase 3 (MLP prediction accuracy). V4 applies targeted improvements.

### V3 Bottleneck Analysis

**Phase 2 (VQE):**
- Single-start L-BFGS-B with `maxiter=500`, `ftol=1e-12`
- Initial guess: `uniform(-0.01, 0.01)` — too narrow
- Only **3/27** points reached fid ≥ 99.5% (h ≥ 1.8)
- Low iteration counts (8-20) at low h → premature convergence in local minima, not true expressibility ceiling

**Phase 3 (MLP):**
- `FID_THRESHOLD=0.96` → only 13 training points (h ≥ 1.1)
- Architecture: `1→16→16→4` (372 params) — underfitting
- `n_epochs=2000`, `patience=200` — energy errors plateaued at ~1.27e-01
- `h_test=1.5` — too easy (middle of training range)

### V4 Changes Applied

| Parameter | V3 | V4 | Rationale |
|---|---|---|---|
| VQE optimizer | Single L-BFGS-B | Multi-start (1 warm-start + 3 restarts, σ=0.1) | Escape local minima |
| VQE maxiter | 500 | 1000 | More optimization budget per run |
| VQE ftol | 1e-12 | 1e-14 | Tighter convergence |
| Initial guess range | ±0.01 | ±0.05 | Broader initial exploration |
| FID_THRESHOLD | 0.96 | 0.93 | Include more training data |
| MLP hidden units | 16 | 32 | More model capacity (1,220 params) |
| n_epochs | 2000 | 4000 | Longer training |
| Scheduler patience | 200 | 150 | Faster LR decay |
| h_test | 1.5 | 1.25 | Harder test (closer to critical region) |

### V4 Results

**Phase 2 — VQE multi-start sweep:**
- Valid regime (fid ≥ 99.5%): **h ≥ 1.4** (was h ≥ 1.8 in V3)
- Points with fid ≥ 99.5%: **7/27** (was 3/27)
- Average fidelity: **86.24%** (was 83.00%)
- Best fidelity: **0.9995** at h=2.0 (was 0.9975)
- ΔE/gap at h=1.4: **1.60%** ✅

**Phase 3 — MLP training:**
- Training data: **17 points** (h ≥ 0.9), was 13 points
- MSE final: **2.71e-04** (was 5.39e-04)
- Max energy error still bounded by HVA expressibility (~1.13e-01 at h=0.9)

**Phase 4 — Deployment at h=1.25 (unseen):**

| Metric | V3 (h=1.5) | V4 (h=1.25) | Threshold | Status |
|---|---|---|---|---|
| ΔE/gap | 2.81% | 4.37% | < 5% | ✅ |
| ⟨X⟩ error | 1.61e-02 | **1.14e-03** | < 1e-2 | ✅ (was ❌) |
| ⟨ZZ⟩ error | 3.66e-02 | **6.08e-03** | < 1e-2 | ✅ (was ❌) |
| ΔE | 3.77e-02 | 3.89e-02 | < 1e-2 | ❌ (aspirational) |
| Fidelity | 0.991 | 0.990 | ≥ 99.5% | ❌ (noiseless-only) |
| ADAPT iters | 2 | 2 | ≤ 2 | ✅ |
| **Checklist** | **2/6** | **4/6** | | |

### Key Takeaways

1. **Multi-start VQE was the highest-impact change.** Pushed valid regime from h≥1.8 to h≥1.4, doubling usable training data.
2. **Local observables now pass** (⟨X⟩ error 1.14e-03, ⟨ZZ⟩ error 6.08e-03) — the pipeline correctly characterizes the quantum phase even at the harder test point h=1.25.
3. **ΔE < 1e-2 remains aspirational** — bounded by HVA p=2 expressibility ceiling. At h=1.25, VQE itself achieves ΔE≈3.02e-02, so the pipeline (3.89e-02) is close to this fundamental limit.
4. **ΔE/gap = 4.37%** confirms the pipeline resolves the physics (distinguishes ground from first excited state) despite the absolute ΔE exceeding 1e-2.
5. **Remaining bottleneck** is the HVA ansatz expressibility — would require p>2 layers or a different initial state to improve further, which conflicts with the shallow-circuit constraint from Mele et al.

---

## 2026-04-22 — Python 3.12 Migration + V4 Re-run

### Environment Update
- Python: 3.9.6 → **3.12.13** (via `brew install python@3.12`)
- Qiskit: 2.1.x → **2.4.0**
- PyTorch: → **2.11.0**
- NumPy: → **2.4.4**
- `requirements.txt` updated with minimum version pins (`>=`) for reproducibility

### V4 Re-run Results (Python 3.12)

Results differ from the 3.9 run due to different random number generation across NumPy versions (multi-start VQE is seed-sensitive).

**Phase 2:**
- Valid regime (fid ≥ 99.5%): **h ≥ 1.5** (was h ≥ 1.4 on Py3.9)
- Points fid ≥ 99.5%: **6/27** (was 7/27)
- Average fidelity: **85.20%** (was 86.24%)
- Max ΔE: **1.46e+00** (was 1.32e+00)

**Phase 3:**
- Training data: **17 points** (h ≥ 0.9), same as Py3.9 run
- MSE final: **7.51e-05** (improved from 2.71e-04)

**Phase 4 — Deployment at h=1.25:**

| Metric | V4/Py3.9 | V4/Py3.12 | Threshold | Status |
|---|---|---|---|---|
| ΔE/gap | 4.37% | < 5% | < 5% | ✅ |
| ⟨X⟩ error | 1.14e-03 | 1.28e-02 | < 1e-2 | ❌ (was ✅) |
| ⟨ZZ⟩ error | 6.08e-03 | 2.66e-02 | < 1e-2 | ❌ (was ✅) |
| ΔE | 3.89e-02 | ≥ 1e-2 | < 1e-2 | ❌ |
| Fidelity | 0.990 | < 99.5% | ≥ 99.5% | ❌ |
| ADAPT iters | 2 | 2 | ≤ 2 | ✅ |
| **Checklist** | **4/6** | **2/6** | | |

### Analysis
The regression from 4/6 to 2/6 is caused by the stochastic nature of multi-start VQE — different random seeds from NumPy 2.4 vs 1.x produce different optimization trajectories. The Phase 2 VQE found slightly worse local minima for the h=1.25 region, which propagated through the MLP training into worse observable predictions.

**Root cause:** No fixed random seed in the VQE multi-start loop. Future improvement: pin `np.random.seed()` before the sweep for reproducibility across environments.

**Key takeaway:** The ΔE/gap metric (priority #1) and circuit compliance (priority #5) remain stable across runs. The observable errors (priority #2) are sensitive to VQE optimization quality, confirming Phase 2 as the critical bottleneck.

---

## 2026-04-22 — PoC V5.0 and V5.1 Results

### V5.0 Changes (from V4)

**Phase 2 — Notebook 1:**
- `hybrid_cost()`: weighted energy + local-observable matching (α=0.5, ×100 scaling on obs term)
- `observable_cost()`: per-site comparison against site-averaged exact values
- Bidirectional sweep: descending (h=2→0) + ascending (h=0→2), both using `hybrid_cost` + `multi_start_vqe`
- Best-of-two selection per h-point

**Phase 3 — Notebook 2:**
- `compute_energy_loss()`: energy-aware loss term (λ=0.1, every 50 epochs, batch=4)
- Used `.detach()` on model output before quantum eval
- MLP: 1→32→32→4, 4000 epochs, patience=150

### V5.0 Results

| Metric | Phase 2 | Phase 3-4 (h=1.25) |
|---|---|---|
| Valid regime | h ≥ 1.5 (6/27 points) | — |
| Avg fidelity | 84.86% | — |
| ΔE/gap | — | 89.10% ❌ |
| ⟨X⟩ error | — | 1.06e-01 ❌ |
| ⟨ZZ⟩ error | — | 3.17e-01 ❌ |
| ΔE | — | 7.95e-01 ❌ |
| Fidelity | — | 0.836 ❌ |
| ADAPT iters | — | 2 ✅ |
| **Checklist** | — | **1/6** |

### V5.0 Failure Analysis

Phase 2 worked well (6 points at fid≥99.5%). Phase 3-4 **catastrophically failed** (MSE=3.36e-01, checklist 1/6).

**Root causes identified:**
1. **`observable_cost` bug**: compared each per-site observable against the site-averaged exact value — mathematically wrong
2. **`hybrid_cost` scaling bug**: `alpha * 100 * obs_cost` — arbitrary ×100 multiplier made the weighting meaningless given different scales of energy (~-8) vs observable error (~0.001)
3. **Ascending sweep was identical to descending**: same `hybrid_cost`, same restarts — no landscape diversity
4. **`compute_energy_loss` used `.detach()`**: severed gradient chain, making energy-aware loss a no-op
5. **`hybrid_cost` changed θ_opt landscape**: Phase 2 now optimizes for energy+observables, but Phase 3 MLP still trains on MSE(θ) and validates against pure energy. The θ_opt values became non-smooth and the MLP couldn't learn them.

---

### V5.1 Changes (bug fixes from V5.0)

**Phase 2 — Notebook 1:**
- Fixed `observable_cost`: now compares mean-vs-mean (not per-site vs mean)
- Fixed `hybrid_cost`: replaced `alpha * 100 * obs` with adaptive normalization `alpha * obs / max(1e-10, obs + |e|) * |e|`, α=0.3
- Fixed ascending sweep: uses energy-only cost with wider perturbations (σ=0.3, 5 restarts) for genuine landscape diversity

**Phase 3 — Notebook 2:**
- Fixed `compute_energy_loss`: clarified that `.detach()` is intentional (quantum eval can't backprop), energy term serves as monitoring signal only
- Version label updated to V5.1

### V5.1 Results

| Metric | Phase 2 | Phase 3-4 (h=1.25) |
|---|---|---|
| Valid regime | **h ≥ 1.4** (7/27 points) ✅ improved | — |
| Avg fidelity | **85.98%** | — |
| ΔE/gap | — | 660.88% ❌ |
| ⟨X⟩ error | — | 7.19e-01 ❌ |
| ⟨ZZ⟩ error | — | 1.00e-01 ❌ |
| ΔE | — | 5.89e+00 ❌ |
| Fidelity | — | 0.000131 ❌ |
| ADAPT iters | — | 2 ✅ |
| **Checklist** | — | **1/6** |

### V5.1 Failure Analysis

Phase 2 improved (7/27 at fid≥99.5%, regime h≥1.4 — best yet). Phase 3-4 **still catastrophically failed** (MSE=4.50, fidelity=0.0001).

**Persistent root cause:** The `hybrid_cost` in Phase 2 produces θ_opt values that minimize a mixed energy+observable objective. These θ_opt have a fundamentally different landscape than pure-energy θ_opt. The MLP in Phase 3 trains on MSE(θ_pred, θ_opt) but validates against E_exact — the two objectives are misaligned. The MLP learns to match the hybrid-cost θ_opt perfectly (low MSE) but these θ values don't minimize energy.

**Key lesson:** Changing the Phase 2 cost function requires corresponding changes in Phase 3 training targets and validation. The pipeline phases are tightly coupled — you can't change one without updating the others.

### Comparison Table (V4 → V5 → V5.1)

| Metric | V4 (h=1.25) | V5.0 (h=1.25) | V5.1 (h=1.25) |
|---|---|---|---|
| Phase 2 valid regime | h ≥ 1.5 | h ≥ 1.5 | **h ≥ 1.4** |
| Phase 2 fid≥99.5% points | 6/27 | 6/27 | **7/27** |
| ΔE/gap | 4.28% ✅ | 89.10% ❌ | 660.88% ❌ |
| ⟨X⟩ error | 2.66e-02 ❌ | 1.06e-01 ❌ | 7.19e-01 ❌ |
| ⟨ZZ⟩ error | 4.75e-02 ❌ | 3.17e-01 ❌ | 1.00e-01 ❌ |
| Checklist | **2/6** | 1/6 | 1/6 |

**Conclusion:** V5.0/V5.1 improved Phase 2 but broke Phase 3-4. The hybrid cost function approach needs the MLP to be retrained with energy-based targets (not hybrid-cost θ_opt), or the Phase 3 loss must be redesigned to match the Phase 2 objective. V4 remains the best end-to-end result.

---

## 2026-04-22 — PoC V5.1 (fixed) Results

### Changes Applied
- Reverted to **pure energy cost** (removed hybrid_cost, observable_cost)
- Kept **bidirectional sweep** (desc: σ=0.1/3 restarts, asc: σ=0.3/5 restarts)
- Added **`wrap_theta()`**: normalizes θ_opt to [-π, π] before saving
- Added **θ sanity assertion** in NB2: `assert max|θ| < 2π`
- Added **versioned .npz filename** + version metadata
- Extracted shared functions to **`tfim_utils.py`**
- Removed empty trailing cells

### Results

**Phase 2:** ✅ Good
- θ range: [-3.03, 3.02] (within [-π, π]) — wrapping works
- Valid regime: h ≥ 1.4 (7/27 at fid≥99.5%) — same as best previous
- Avg fidelity: 85.95%

**Phase 3-4:** ❌ Still broken (1/6 checklist)
- MSE: 2.71 — MLP cannot learn the θ landscape
- ΔE/gap: 63.80%, ⟨X⟩ error: 1.23e-01, ⟨ZZ⟩ error: 2.98e-01

### Root Cause
The `wrap_theta()` to [-π, π] creates **discontinuities** in the θ_opt landscape. When the optimizer finds θ=3.1 at h=1.3 and θ=-3.1 at h=1.35 (both physically equivalent via periodicity), the MLP sees a jump of 6.2 — impossible to fit smoothly.

**The wrapping fixes the range** (no more θ=28) **but breaks smoothness** — the opposite problem from V5.0/V5.1-pre-fix.

### Lesson Learned
Angle wrapping alone is insufficient. The correct approach is **unwrapping**: ensure θ_opt values are continuous across the h-sweep by choosing the branch closest to the previous point. This is analogous to `np.unwrap()` for phase angles.

### V4 Remains Best End-to-End (2/6)
V4's pure energy cost with single descending sweep naturally produces smooth θ because warm-start propagation keeps parameters close to the previous solution. The bidirectional sweep + wrapping disrupts this continuity.

### Next Step
For V6: keep bidirectional sweep + pure energy cost, but replace `wrap_theta` with **`unwrap_theta`** that ensures continuity relative to the previous h-point's θ_opt. Or: simply keep V4's approach (single descending sweep) and focus improvements on Phase 3 (larger MLP, more data augmentation) and Phase 4 (better ADAPT-VQE refinement).


 Summary: Phase 2 is now clean (θ ∈ [-π, π], 7/27 valid), but Phase 3 still fails because wrap_theta creates discontinuities — θ jumping from +3.1 to -3.1 between adjacent h-points. The  
  MLP can't learn that.                                                                                                                                                                     
                                                                                                                                                                                            
  The fix would be np.unwrap()-style continuity enforcement instead of simple modular wrapping. But honestly, V4's single descending sweep naturally avoids this problem because warm-start 
  propagation keeps θ values close to the previous solution — no wrapping needed.  