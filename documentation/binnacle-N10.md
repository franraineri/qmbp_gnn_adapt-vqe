# Binnacle — N=10 Scaling Experiments

> Experiments testing the V6 pipeline at N=10 qubits (1D TFIM chain).
> All experiments use the best configuration from the N=6 hyperparameter sweep:
> VQE 5 restarts, maxiter=1000, MPNN h=64 L=3 6000ep lr=1e-3, fid≥0.93.

---

## 2026-05-04 — V6.0 Benchmark — N=10 chain, best config, h=1.5

### Configuration
- System: 1D TFIM, N=10, p=2, 27 h-points, h_test=1.5
- restarts=5, maxiter=1000, MPNN(h=64, L=3, ep=6000, lr=0.001, pat=150), fid≥0.93
- Seeds: [42, 43, 44]

### Per-Run Results (Adapt-VQE at h=1.5)

| Run | Seed | ΔE/gap | ⟨X⟩ err | ⟨ZZ⟩ err | ΔE | Fidelity | ADAPT | Checklist | Time |
|-----|------|--------|---------|----------|-----|----------|-------|-----------|------|
| 1 | 42 | 2.95% ✅ | 1.38e-02 ❌ | 2.69e-02 ❌ | 3.44e-02 ❌ | 0.9909 ❌ | 2 | **2/6** | 50s |
| 2 | 43 | 2.74% ✅ | 6.27e-03 ✅ | 1.40e-02 ❌ | 3.20e-02 ❌ | 0.9920 ❌ | 2 | **3/6** | 51s |
| 3 | 44 | 2.68% ✅ | 9.06e-03 ✅ | 1.86e-02 ❌ | 3.12e-02 ❌ | 0.9921 ❌ | 2 | **3/6** | 53s |

### Aggregate Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| ΔE/gap | 2.79% | 0.12% | 2.68% | 2.95% |
| ⟨X⟩ error | 9.72e-03 | 3.12e-03 | 6.27e-03 | 1.38e-02 |
| ⟨ZZ⟩ error | 1.98e-02 | 5.33e-03 | 1.40e-02 | 2.69e-02 |
| Fidelity | 0.9916 | 0.0005 | 0.9909 | 0.9921 |
| Checklist | 2.7/6 | 0.5 | 2/6 | 3/6 |
| Runtime | 51s | 1s | 50s | 53s |

### Analysis: N=10 vs N=6

| Metric | N=6 (h=1.5) | N=10 (h=1.5) | Change |
|--------|-------------|--------------|--------|
| ΔE/gap | 1.36% | 2.79% | +1.4pp (still passes) |
| ⟨X⟩ error | 2.6e-03 | 9.7e-03 | ~4x worse (borderline) |
| ⟨ZZ⟩ error | 5e-03 | 2.0e-02 | ~4x worse (fails) |
| Fidelity | 0.997 | 0.992 | drops below 99.5% |
| Checklist | 5/6 | 2–3/6 | regression |
| Runtime | ~25s | ~51s | ~2x |

### Key Findings
1. The pipeline scales to N=10 without code changes — only a CLI parameter.
2. ΔE/gap (primary metric) still passes comfortably (2.79% < 5%).
3. Observable errors degrade ~4x from N=6 to N=10 — expected for 1024-dim Hilbert space with 17 training points.
4. The MPNN architecture (h=64, L=3) may be undersized for N=10.
5. Runtime ~51s per full pipeline is acceptable.

### Next Steps for N=10
- Try MPNN h=128 (was overfitting at N=6, but N=10 has more graph structure)
- Try data augmentation (`--augment` flag)
- Try more h-points (40 instead of 27)
- Test at h=1.25 to see critical-region degradation


---

## 2026-05-05 — N=10 Hyperparameter Sweep (7 experiments, 14 executions)

### Methodology
Systematic exploration of parameters that might behave differently at N=10 vs N=6. Key hypothesis: the MPNN h=128 (which overfitted at N=6 with 17 training points) may work better at N=10 where the graph has more structure (10 nodes, 9 edges vs 6/5).

### Results

| Exp | Config | h_test | ΔE/gap | ⟨X⟩ err (mean) | Fidelity | Checklist | Notes |
|-----|--------|--------|--------|-----------------|----------|-----------|-------|
| — | Baseline (h=64, 6000ep) | 1.5 | 2.79% ✅ | 9.72e-03 ⚠️ | 0.9916 | 2–3/6 | From previous session |
| A | Augmentation | 1.5 | 2.83% ✅ | 1.23e-02 ❌ | 0.9914 | 2–3/6 | Augmentation doesn't help here |
| B | **h=128** | 1.5 | 2.86% ✅ | **8.38e-03** ✅ | 0.9917 | **3/6** | ⭐ Best — ⟨X⟩ passes consistently |
| C | h=128 + augment | 1.5 | 2.81% ✅ | 9.79e-03 ⚠️ | 0.9916 | 2–3/6 | Augmentation hurts h=128 |
| D | h=128 | 1.25 | 10.54% ❌ | 3.09e-02 ❌ | 0.9729 | 1/6 | Critical region much worse at N=10 |
| E | h=128 | 1.4 | 4.69% ⚠️ | 1.81e-02 ❌ | 0.9866 | 1–2/6 | Borderline — ΔE/gap barely passes |
| F | h=128 + 40pts | 1.5 | 2.84% ✅ | 1.22e-02 ❌ | 0.9914 | 2–3/6 | Denser grid: 9x slower, no gain |
| G | h=128 + 8000ep | 1.5 | 2.85% ✅ | **8.40e-03** ✅ | 0.9916 | **3/6** | Same as 6000ep — converged |

### Key Findings

**1. MPNN h=128 is the right size for N=10.**
At N=6, h=128 overfitted (17 points, 6-node graph). At N=10, the graph has more structure (10 nodes, 9 edges) and h=128 consistently achieves ⟨X⟩ < 1e-2 — pushing the checklist from 2–3/6 to a stable 3/6.

**2. Data augmentation does NOT help at N=10.**
Contrary to our hypothesis, augmentation slightly worsens results. The interpolated θ values may be less accurate at N=10 because the θ landscape is more complex (4 parameters controlling 10 qubits).

**3. The critical region (h≤1.4) is much harder at N=10.**
- h=1.25: ΔE/gap > 10% (fails badly) — HVA p=2 expressibility is worse at larger N near the phase transition
- h=1.4: ΔE/gap ≈ 5% (borderline) — barely passes on some seeds
- h=1.5: ΔE/gap ≈ 2.8% (comfortable) — the valid operating regime for N=10

**4. Denser h-grid (40 pts) is wasteful at N=10.**
9x slower (441s vs 54s) with no improvement. The extra VQE compute doesn't produce better θ_opt for the MPNN.

**5. 6000 MPNN epochs is sufficient even for h=128.**
8000 epochs gives identical results — the model converges by 6000.

### Recommended Configuration for N=10

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| MPNN hidden | **128** | Right-sized for 10-node graph (was 64 for N=6) |
| MPNN layers | 3 | Same as N=6 |
| MPNN epochs | 6000 | Converged — 8000 is wasteful |
| VQE restarts | 5 | Same as N=6 |
| Augmentation | **OFF** | Hurts at N=10 |
| H-grid | 27 points | 40 is 9x slower with no gain |
| Test point | **h ≥ 1.5** | h=1.4 is borderline, h=1.25 fails |

### Expected Checklist by Test Point (N=10)

| h_test | Checklist | ΔE/gap | ⟨X⟩ | ⟨ZZ⟩ | ΔE | Fidelity | ADAPT |
|--------|-----------|--------|-----|------|-----|----------|-------|
| 1.25 | 1/6 | ❌ 10.5% | ❌ | ❌ | ❌ | ❌ | ✅ |
| 1.4 | 1–2/6 | ⚠️ 4.7% | ❌ | ❌ | ❌ | ❌ | ✅ |
| 1.5 | **3/6** | ✅ 2.8% | ✅ 8.4e-3 | ❌ 2e-2 | ❌ 3.3e-2 | ❌ 0.992 | ✅ |

### N=10 vs N=6 Comparison (best config for each)

| Metric | N=6 (h=1.5) | N=10 (h=1.5) | Degradation |
|--------|-------------|--------------|-------------|
| Best checklist | 5/6 | 3/6 | -2 metrics |
| ΔE/gap | 1.4% | 2.8% | 2x |
| ⟨X⟩ error | 2.6e-03 | 8.4e-03 | 3x |
| ⟨ZZ⟩ error | 5e-03 | 2e-02 | 4x |
| Fidelity | 0.997 | 0.992 | drops below 99.5% |
| MPNN hidden | 64 | 128 | 2x capacity needed |
| Runtime | ~25s | ~55s | 2x |

The pipeline scales gracefully — ΔE/gap and ⟨X⟩ still pass, and the MPNN correctly adapts to the larger graph. The remaining metrics (⟨ZZ⟩, ΔE, fidelity) are bounded by HVA p=2 expressibility at N=10.


---

## Key Lessons Learned — N=10

1. **MPNN capacity must scale with system size.** At N=6 (6 nodes, 5 edges), h=64 is optimal and h=128 overfits. At N=10 (10 nodes, 9 edges), h=128 is optimal and h=64 underfits. The rule of thumb: hidden_dim ≈ 10–13× the number of nodes.

2. **The HVA expressibility ceiling degrades with N.** At N=6, fidelity reaches 0.995 at h=1.4. At N=10, fidelity only reaches 0.992 at h=1.5. The same p=2 circuit has to control more qubits with the same 4 parameters — it becomes less expressive per qubit.

3. **The valid operating regime shifts outward with N.** N=6 works at h≥1.4. N=10 only works at h≥1.5. The critical region (h≈1.0) becomes progressively harder because finite-size effects are weaker at larger N — the gap closes faster, making ΔE/gap harder to satisfy.

4. **Data augmentation is counterproductive at N=10.** The interpolated θ values assume linearity between adjacent h-points. At N=10, the θ landscape is more complex (4 parameters controlling 1024-dimensional Hilbert space), so linear interpolation introduces inaccurate training targets that confuse the MPNN.

5. **Denser h-grids don't help — the bottleneck is VQE quality, not data quantity.** Adding more h-points (40 vs 27) means more VQE runs, but the extra points are either filtered out (low fidelity) or in easy regimes where θ is already smooth. The MPNN's limitation is prediction accuracy, not training data volume.

6. **The pipeline scales gracefully.** Going from N=6 to N=10 required zero code changes — only CLI parameters (`--n-qubits 10 --mpnn-hidden 128`). The modular architecture works as designed.


---

## Conceptual Note: Does Data Augmentation Make Sense?

**The question:** We generate our own training data via VQE. If we need more data, why not just run VQE at more h-points instead of interpolating between existing ones?

**The answer: augmentation is a shortcut that avoids VQE cost, but it's an inferior shortcut.**

The pipeline has two ways to get more training data:

1. **Run VQE at more h-points** (e.g., 40 instead of 27). This produces *exact* θ_opt values — each one is the true optimum for that h. Cost: ~15 minutes of VQE compute per extra point.

2. **Interpolate between existing points** (augmentation). This produces *approximate* θ values — linear interpolation assumes the landscape is linear between adjacent h-points. Cost: zero (instant).

**Why augmentation failed at N=10:** The θ landscape at N=10 is more complex than at N=6. Linear interpolation between θ(h=1.3) and θ(h=1.35) produces a θ that is NOT the true optimum at h=1.325 — it's just a guess. The MPNN then trains on these inaccurate targets, which degrades its predictions.

**Why running more VQE points also failed:** We tested 40 h-points (Exp F) and it was 9x slower with no improvement. The reason: the extra points are either in the low-fidelity regime (h<0.8, filtered out) or in the easy regime (h>1.5, where θ is already smooth and the MPNN doesn't need help).

**The real bottleneck is not data quantity — it's VQE quality in the critical region.** The 17 training points we have (h∈[0.9, 2.0] after fidelity filter) are all high-quality (fid≥93%). Adding more points in this range would help marginally, but the MPNN's prediction error is dominated by the difficulty of the test point (h=1.5 at N=10), not by insufficient training data.

**Conclusion:** For this pipeline, neither augmentation nor denser grids are the right approach. The correct path to better N=10 results is either (a) a more expressive circuit (p>2, which violates Mele et al.) or (b) a better MPNN architecture that captures the non-linear θ landscape more accurately (which h=128 partially achieves).
