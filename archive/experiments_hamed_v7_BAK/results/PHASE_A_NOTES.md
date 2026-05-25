# Phase A Results — Hamed V7 Full Experiments

> Execution: 2026-05-18 | Total: 6.5 min | All 4 experiments: SUCCESS
> Protocol: full spec (proper seeds, budgets, h-grids)
> MPS re-validated with corrected same-θ protocol

---

## Experiment 4A: SPSA Hyperparameter Grid Search

**Config:** N=6, h=1.5, n_shots=4096, 10 seeds per config
**Grid:** a∈{0.05,0.1,0.2,0.5} × c∈{0.05,0.1,0.2} × A∈{10,20,50} = 36 configs × 10 seeds = 360 runs
**Time:** 147.9s

### Best Configurations (top 5)

| a | c | A | Mean ΔE | Relative ΔE/gap |
|---|---|---|---------|-----------------|
| **0.10** | **0.05** | **10** | **6.52e-02** | **4.9%** |
| 0.05 | 0.05 | 10 | 8.08e-02 | 6.0% |
| 0.10 | 0.05 | 20 | 8.05e-02 | 6.0% |
| 0.10 | 0.10 | 10 | 8.05e-02 | 6.0% |
| 0.05 | 0.05 | 20 | 8.57e-02 | 6.4% |

### Pattern Analysis
- **a (step size):** 0.05-0.1 optimal. a=0.2 is 20% worse. a=0.5 is 2-3× worse with high variance.
- **c (perturbation):** 0.05 consistently best. c=0.1 is close. c=0.2 slightly worse.
- **A (stability):** Minor effect. A=10 marginally better than A=20,50 for good (a,c).
- **All configs:** correct phase classification (100%), fidelity > 0.97.

### Decision
**Use a=0.1, c=0.05, A=10 for all hardware SPSA experiments.**
At this config, SPSA achieves ΔE/gap ≈ 5% under 4096-shot noise — close to the 5% threshold.
On real hardware with 8192 shots, expect ~3-4% (noise scales as 1/√shots).

---

## Experiment 4B: SPSA Warm-Start Refinement

**Config:** N=6, h∈{1.25,1.5,2.0}, warm-start from L-BFGS-B VQE, n_iter∈{10,20,50,100,200}, 10 seeds
**Time:** 32.0s

### Results

| h | Baseline ΔE | Best SPSA ΔE | Worst SPSA ΔE | Verdict |
|---|-------------|-------------|---------------|---------|
| 1.25 | 4.03e-02 | 4.03e-02 (n=10) | 4.31e-02 (n=200) | No gain, slight degradation |
| 1.5 | 1.94e-02 | 1.94e-02 (n=10) | 2.64e-02 (n=200) | No gain, -36% at high iter |
| 2.0 | 3.62e-03 | — | 1.65e-02 (n=200) | **-356% worse** |

### Key Insight
SPSA refinement is **counterproductive** when starting from a good warm-start:
1. At h=2.0 (easiest, baseline ΔE=3.6e-03), shot noise dominates the signal — SPSA wanders away.
2. More iterations = more damage. n=200 is always worse than n=10.
3. The warm-start is already near-optimal; noise prevents further improvement.

### Decision
**Do NOT apply SPSA after MPNN prediction in noiseless simulation.**
SPSA is only valuable for: (a) cold-start under noise, or (b) real hardware where the
cost function is inherently noisy and the MPNN prediction may be suboptimal.

---

## Experiment 1A: Nevergrad Fair Budget Comparison

**Config:** N=6, budget=1000, h∈{0.5,0.8,1.0,1.1,1.25,1.5,2.0}, 5 seeds, cold-start
**Time:** 205.2s (175 optimizer runs)

### Overall Ranking

| Optimizer | Mean ΔE (all h) | Mean Fidelity |
|-----------|-----------------|---------------|
| **L-BFGS-B (5 restarts)** | **0.136** | **0.926** |
| CMA | 0.178 (+31%) | 0.910 |
| OnePlusOne | 0.189 (+39%) | 0.906 |
| TwoPointsDE | 0.243 (+79%) | 0.898 |
| DE | 0.265 (+95%) | 0.890 |

### Regime-Split Analysis

**Paramagnetic regime (h≥1.0, where HVA works):**

| Optimizer | Mean ΔE (h≥1.0) | Improvement over NG best |
|-----------|------------------|--------------------------|
| L-BFGS-B | ~0.03 | baseline |
| CMA | ~0.07 | 2.3× worse |
| OnePlusOne | ~0.08 | 2.7× worse |

**Ferromagnetic regime (h<1.0, physics-limited):**

All optimizers perform poorly (ΔE > 0.4) because HVA p=2 + |+⟩^N cannot express
the ferromagnetic ground state. This is a physics limit, not an optimizer failure.
The high overall mean (0.136) is dominated by these h<1.0 points.

### Decision
**L-BFGS-B is definitively optimal for our setting.** Close the Nevergrad question.
- No barren plateaus (Mele et al. 2026) → gradients are informative
- Only 4 parameters → gradient computation is cheap
- Smooth landscape for HVA on TFIM → no need for global search

---

## Experiment 3A/3B: MPS Accuracy Validation

**Protocol:** Run VQE on statevector → evaluate SAME θ on MPS (tests simulator, not optimizer)
**Time:** <1s each

### N=6 Results

| h | |MPS - Statevector| | chi=64 = chi=256? |
|---|---------------------|-------------------|
| 1.0 | 1.1e-14 | Yes (identical) |
| 1.5 | 2.8e-14 | Yes (identical) |
| 2.0 | 6.2e-14 | Yes (identical) |

### N=10 Results

| h | |MPS - Statevector| | chi=64 = chi=256? |
|---|---------------------|-------------------|
| 1.0 | 5.0e-14 | Yes (identical) |
| 1.5 | 2.1e-14 | Yes (identical) |
| 2.0 | 2.5e-13 | Yes (identical) |

### Conclusions
1. **MPS is exact** for 1D HVA p=2 circuits (error = machine epsilon ≈ 1e-14)
2. **chi=64 is sufficient** — bond dimension never saturates (area-law entanglement)
3. **chi-independent** — HVA p=2 on 1D chain produces so little entanglement that even chi=32 would work
4. **Safe to use chi=64 for N=20 and N=30** (3C, 3D experiments)

---

## Result Files (definitive, clean)

| File | Size | Content |
|------|------|---------|
| `spsa_4A_20260518_202845.json` | 124 KB | 360 metrics (36 configs × 10 seeds) |
| `spsa_4B_20260518_202917.json` | 51 KB | 150 metrics (3h × 5 iters × 10 seeds) |
| `nevergrad_1A_20260518_203242.json` | 58 KB | 175 metrics (7h × 5 seeds × 5 optimizers) |
| `mps_3A_20260518_204626.json` | 10 KB | 6 metrics (3h × 2 chi, same-θ protocol) |
| `mps_3B_20260518_204653.json` | — | 6 metrics (3h × 2 chi, N=10) |

---

## Decisions for Subsequent Phases

| Decision | Source | Applied to |
|----------|--------|-----------|
| SPSA config: a=0.1, c=0.05, A=10 | 4A | Phases C, D (4C, 4D, 4E) |
| No SPSA refinement after warm-start | 4B | Phase 4 hardware deployment |
| L-BFGS-B for noiseless VQE | 1A | All training data generation |
| MPS chi=64 sufficient | 3A/3B | Phases B, E (3C, 3D, 3E) |
| Skip h<1.0 for MPS VQE | 3C fix | All N≥20 experiments |
