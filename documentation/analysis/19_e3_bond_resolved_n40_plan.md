# E3: Bond-Resolved HVA at N=40 — Task Plan

> **Experiment ID**: E3_BR_SCALING
> **Date**: 2026-06-08
> **Status**: Planning
> **Hypothesis**: Bond-resolved HVA at N=40 demonstrates GNN necessity in high-dimensional
> variational spaces, establishing the scaling path toward classical intractability.

---

## Objective

Scale bond-resolved HVA (79 parameters, p=1) to N=40 using existing MPS infrastructure.
Demonstrate that:
1. VQE converges with warm-start descending sweep (COBYLA or SPSA)
2. MPNN predicts 79-dim θ_opt from graph-encoded Hamiltonian
3. GNN prediction is significantly faster than random search (quantum advantage argument)

---

## Architecture Decisions

### Topology Choice: chain_1d N=40 (Phase 1) + heavy_hex N=20 (Phase 2)

**Rationale**: chain_1d N=40 validates infrastructure (MPS + bond-resolved + COBYLA at 79
params). However, chain_1d has translational symmetry — bond-resolved θ converges to
quasi-uniform values. The thesis argument (GNN necessity) requires heavy_hex where spatial
asymmetry makes bond-resolved genuinely high-dimensional.

**Plan**: Execute chain_1d N=40 first (proven MPS path), then heavy_hex N=20 (44 params,
still within StatevectorEstimator if needed as fallback).

### Optimizer Strategy: SPSA primary, COBYLA fallback

**Rationale**: SPSA uses 2 evals/iteration regardless of N_params. COBYLA builds N+1=80
vertex simplex (80 base evals). At 7-10s/eval (aer_mps N=40), SPSA is more predictable.
Both already implemented and validated.

### MPNN Config: h=256, asymmetric heads (n_edges, n_qubits)

**Rationale**: Current `per_parameter_heads` splits output_dim//2 evenly — incorrect for
bond-resolved where n_edges ≠ n_qubits (39 edges + 40 sites = 79). Need to fix the head
split to accept (n_edges, n_qubits) explicitly.

---

## Dependencies on Existing Code

| Component | File | Status | Changes Needed |
|-----------|------|:------:|----------------|
| `create_bond_resolved()` | `circuits/hva.py` | ✅ | None |
| `MPSBackend` (aer_mps) | `execution/mps_backend.py` | ✅ | None |
| `VQEOptimizer` (COBYLA) | `optimizers/vqe.py` | ✅ | None |
| `SPSAOptimizer` | `optimizers/spsa.py` | ✅ | None |
| `ClassicalSolver` (DMRG) | `solvers/classical.py` | ✅ | None |
| `ValidationRunner` base | `framework/runner_base.py` | ✅ | None |
| `run_scaling_validation.py` | `scripts/.../scaling/` | ✅ | Pattern reuse |
| `run_bond_resolved_validation.py` | `scripts/.../bond_resolved/` | ✅ | Pattern reuse |
| `MPNNPredictor` | `predictors/mpnn.py` | ⚠️ | Fix asymmetric head split |
| `build_graph_dataset` | `predictors/mpnn.py` | ✅ | None (any output_dim) |
| `model_registry` (tfim_bond_resolved) | `models/model_registry.py` | ✅ | None |

---

## Tasks

### Task 0: Fix MPNN per_parameter_heads for asymmetric splits — ✅ DONE

**Problem**: `per_parameter_heads=True` currently splits `output_dim // 2` for both heads.
For bond-resolved with 39 edges + 40 sites = 79, this gives head_zz=39, head_x=39 (loses 1 param).

**Fix**: Added `n_edges` parameter to `MPNNPredictor.__init__()`. When `per_parameter_heads=True`
and `n_edges` is provided, uses `n_edges` for head_zz and `output_dim - n_edges` for head_x.
Falls back to `output_dim // 2` when `n_edges=None` (backward compat).

**Files**: `src/qmbp_simulation/predictors/mpnn.py`
**Verified**: 22 MPNN tests pass, backward compat confirmed, asymmetric split tested.

---

### Task 1: Runner Script — E3 Bond-Resolved Scaling — ✅ DONE

**Pattern**: Extends `ValidationRunner` (same as `run_bond_resolved_validation.py`).
Reuses Phase 1 (DMRG) from `run_scaling_validation.py` and bond-resolved circuit from
the existing bond-resolved runner.

**File**: `scripts/experiment_runners/bond_resolved/run_e3_bond_resolved_scaling.py`

**Sections**:

| Section | Name | Go/No-Go | Estimated Time |
|:-------:|------|----------|:--------------:|
| 0 | Sanity Check (circuit + 1 eval) | Finite energy, correct n_params=79 | 10 min |
| 1 | VQE Convergence (1 h-point, COBYLA vs SPSA) | ΔE/gap < 5% in < 3h wall | 2-4h |
| 2 | Descending Sweep (9 h-points, best optimizer) | ≥7/9 pass ΔE/gap < 5% | 12-18h |
| 3 | MPNN Training (h=256, 9+ training points) | Deploy ΔE/gap < 10% | 1h |
| 4 | GNN Necessity (random search baseline) | MPNN beats random by > 5× evals | 2h |

**CLI**:
```bash
# Sanity only
python scripts/.../run_e3_bond_resolved_scaling.py --section 0

# Convergence test (single h-point)
python scripts/.../run_e3_bond_resolved_scaling.py --section 1

# Full sweep (batch job)
python scripts/.../run_e3_bond_resolved_scaling.py --section 2

# MPNN training + deploy (after sweep completes)
python scripts/.../run_e3_bond_resolved_scaling.py --section 3 4
```

---

### Task 2: Section 0 — Sanity Check

**What it does**:
1. `create_bond_resolved(N=40, p=1, chain_1d)` → verify 79 params
2. Single eval with MPSBackend at θ=0 → finite energy
3. Single eval with θ_uniform (same angle for all bonds/sites) → compare with global HVA
4. Verify identical energy when bond-resolved params are set uniformly

**Go condition**: Energies are finite and bond-resolved at uniform θ = global HVA energy.
**No-go**: Circuit construction fails or energies diverge.

---

### Task 3: Section 1 — VQE Convergence Test

**What it does**:
1. DMRG ground truth at h=5.0 (deep paramagnetic, easy landscape)
2. Run COBYLA: maxiter=2000, 1 restart, warm_start from zeros
3. Run SPSA: 500 iterations (1000 evals), a=0.1, c=0.05
4. Compare ΔE/gap and wall time

**Config**:
- N=40, p=1, chain_1d, seed=42
- MPSBackend(strategy="aer_mps", chi_max=64, precision=0.005)
- h=H_MAX≈6.5 (deep paramagnetic, well above h_min≈4.5 for N=40)

**Go condition**: At least one optimizer achieves ΔE/gap < 5% within 3h wall time.
**No-go**: Neither converges below 5% in 5000+ evaluations.

**Fallback if no-go**: Try h=8.0 (even easier landscape). If that also fails → abort E3.

---

### Task 4: Section 2 — Descending Sweep

**What it does**:
1. DMRG ground truth for 9 h-points in valid regime
2. Descending sweep with warm-start (winning optimizer from Section 1)
3. Save θ_opt[79] for each h-point → training data for Phase 3

**Config**:
- h_values: np.linspace(h_max, h_min_safe, 9) where h_min_safe = 1.5 + 0.020·40^1.31 + 0.5 ≈ 4.5
- Estimated h: [6.0, 5.8, 5.6, 5.4, 5.2, 5.0, 4.8, 4.6, 4.5]
- Seed 42 only (multi-seed optional if time permits)
- maxiter: 2000 (COBYLA) or 800 iterations (SPSA)

**Go condition**: ≥7/9 h-points pass ΔE/gap < 5%.
**No-go**: <5/9 pass.

**Output**: `results/bond_resolved_scaling/sweep_N40_chain1d_{timestamp}.json`
Contains `theta_opt_by_h` array for Phase 3.

---

### Task 5: Section 3 — MPNN Training

**What it does**:
1. Build graph dataset from Section 2 θ_opt data (9 graphs × 79 targets)
2. Train MPNN: GINConv, h=256, L=3, output_dim=79, per_parameter_heads=True, n_edges=39
3. Train config: n_epochs=6000, lr=1e-3, patience=800 (model has built-in dropout=0.1)
4. Deploy on 4 midpoint h-values (interpolation test)
5. Evaluate ΔE/gap for each deployed prediction

**Go condition**: Deploy ΔE/gap < 10% for ≥3/4 midpoints.
**No-go**: MSE diverges or deploy error > 50%.

**Risk mitigation**: With only 9 training points, overfitting is likely. Use:
- Model built-in dropout=0.1 (in head MLPs)
- patience=800 (early stopping via ReduceLROnPlateau)
- Multi-seed augmentation if available (3 seeds × 9 h = 27 points)

---

### Task 6: Section 4 — GNN Necessity (Quantum Advantage Argument)

**What it does**:
1. Random search baseline: sample 200 random θ[79] from uniform(-π, π)
2. Evaluate each on MPSBackend at a single h-point (mid-sweep)
3. Find best ΔE/gap from random search
4. Compare: random needs N_random evals to match VQE warm-start (1 forward pass)
5. If best_random > VQE after 200 evals: GNN necessity ratio = 200+ (lower bound)

**Go condition**: MPNN/VQE beats random search by >5× in eval count.
**No-go**: MPNN no better than 3× random.

**Caveat for chain_1d**: Random search in 79D will likely find quasi-uniform solutions
because the optimal IS quasi-uniform (translational symmetry). The "necessity"
argument is weakened here — the effective search space is ~2D, not 79D.
The real GNN necessity test requires heavy_hex (Plan E) where spatial structure
makes the effective dimensionality match the parameter count.

---

### Task 7: Experiment Criteria Registration — ✅ DONE

**File**: `src/qmbp_simulation/framework/criteria.py`

Added `E3_BR_SCALING` entry to `EXPERIMENT_CRITERIA`:
```python
"E3_BR_SCALING": {
    "metric": "pass_rate",
    "threshold": 0.78,
    "desc": "Bond-resolved N=40 VQE converges (>=7/9 h-points ΔE/gap < 5%)",
},
```

Also added chain_1d N=40/50/80 entries to `P1_VALID_REGIME` in `preflight.py` for
regime boundary validation.

---

### Task 8: Binnacle Documentation

**File**: `documentation/binnacles/binnacle-e3-bond-resolved-scaling.md`

Create after execution with:
- Hypothesis statement
- Config used
- Results table (per-h ΔE/gap, timing, convergence)
- Comparison: COBYLA vs SPSA timing
- MPNN training curves + deploy results
- Key findings (positive or negative)
- Thesis value statement

---

## Execution Order

```
Task 0 (MPNN fix)          ✅ DONE — n_edges param, asymmetric head split
Task 1 (runner script)     ✅ DONE — 5 sections, ValidationRunner, preflight passes
Task 7 (criteria)          ✅ DONE — E3_BR_SCALING + P1_VALID_REGIME entries
Task 8 (binnacle skeleton) ✅ DONE — placeholders ready for results
Unit tests                 ✅ DONE — 15/15 pass (test_mpnn_bond_resolved.py)
─── Implementation complete ───
Task 2 (Section 0)         ← Run: 10 min       [READY TO EXECUTE]
Task 3 (Section 1)         ← Run: 2-4h (go/no-go gate)
Task 4 (Section 2)         ← Run: 12-18h batch (conditional on Section 1)
Task 5 (Section 3)         ← Run: 1h (conditional on Section 2)
Task 6 (Section 4)         ← Run: 2h (low value on chain_1d)
Task 8 (binnacle fill)     ← Write after results
Plan E (heavy_hex N=20)    ← High-value follow-up (requires minor CLI adaptation)
```

---

## Execution Decision Tree

The key insight: **not all sections have equal thesis value**. Execute sequentially
with go/no-go gates. STOP when the marginal value drops below the time cost.

```
Section 0: Sanity Check (10 min)
├── FAIL → STOP. Fix circuit/MPS integration issue.
│   (No thesis impact — infrastructure bug)
│
└── PASS → Section 1: VQE Convergence (2-4h)
    │
    ├── Cold-start ΔE/gap > 15% (both optimizers) → ★ STOP ★
    │   THESIS CLAIM (Chapter 6, strongest):
    │   "Without MPNN initialization, VQE cannot converge in 79-dimensional
    │    bond-resolved parameter space. GNN warm-start is a NECESSARY condition."
    │   → Extends N=16 square finding (8-14%) to N=40 chain_1d at higher dim.
    │     Note: N=16 was on asymmetric topology (square) — if chain_1d ALSO fails,
    │     the finding is even stronger (symmetry doesn't save you at 79 params).
    │   → Direct evidence that GNN transitions from "useful" to "essential".
    │   → Document in binnacle + thesis. No further execution needed.
    │
    ├── Cold-start ΔE/gap 5-15% → ★ CONDITIONAL STOP ★
    │   THESIS CLAIM (moderate):
    │   "MPNN warm-start reduces VQE convergence cost by N_evals at 79 params."
    │   → If time permits: Section 2 overnight to quantify warm-start benefit
    │   → Otherwise: document partial convergence finding. Sufficient for thesis.
    │
    └── Cold-start ΔE/gap < 5% → landscape is easy (expected for chain_1d)
        │   THESIS CLAIM (weakest for chain_1d):
        │   "79D VQE converges easily on symmetric topologies — GNN saves time
        │    but is not strictly necessary. The necessity emerges on asymmetric
        │    topologies (heavy_hex) where spatial structure breaks uniformity."
        │
        ├── If time-constrained → STOP. Sufficient for infrastructure validation.
        │
        ├── If batch time available → Section 2 overnight (Table 5.24 data)
        │   → Section 3+4 only if sweep data passes
        │
        └── ★ PIVOT to Plan E ★: heavy_hex N=20 (44 params)
            (The GNN necessity proof that chain_1d cannot provide)
```

---

## Plan E: Heavy-Hex N=20 Bond-Resolved (highest thesis value)

**Why this is the key experiment**:
- heavy_hex N=10 bond-resolved gives +49.7% vs global (binnacle proven)
- N=16 square bond-resolved FAILS without warm-start (8-14% error)
- heavy_hex N=20 combines both: non-uniform topology + high dim (39 params)
- Uses NoiselessBackend (StatevectorEstimator): 2^20 = 1M amplitudes, ~1s/eval

**Technical feasibility check**:
- N=20 StatevectorEstimator: confirmed viable (V7/V8 used N=20 routinely)
- heavy_hex N=20: 19 edges → 19+20 = 39 bond-resolved params
- L-BFGS-B viable (exact gradients from StatevectorEstimator, no shot noise)
- Estimated timing: ~5-10s per h-point with L-BFGS-B + 3 restarts

**What's needed to execute**:
1. Add `--topology` CLI arg to the runner (currently `TOPOLOGY` is a module constant)
2. Compute h_min_safe from `P1_VALID_REGIME[("heavy_hex", 20)]` or fallback to
   scaling law. NOTE: heavy_hex N=20 is NOT in P1_VALID_REGIME — need to add or
   use empirical estimate (~3.5 based on N=10 threshold of 3.0).
3. Conditional backend: NoiselessBackend for N≤22, MPSBackend for N>22
4. Adjust n_restarts (5 for statevector — no shot noise means more restarts are cheap)

**Estimated changes**: ~30 lines in the runner (CLI arg + backend dispatch + h_min logic)

**Expected outcome**:
- Bond-resolved θ_opt shows SPATIAL STRUCTURE (std_θ_zz > 0.01 across bonds)
- Cold-start VQE at 44 params either fails or converges slowly (N=16 precedent)
- MPNN is necessary (interpolation of 44-dim spatially-structured θ fails)
- +30-50% energy improvement vs global HVA at N=20 heavy_hex

**Thesis value**: This is the "Chapter 6 main result" — GNN necessity at scale
on the hardware-deployment target topology. It connects:
- Bond-resolved expressibility (Section 3 of existing binnacle: +49.7%)
- GNN at high dimension (Section 5 of existing binnacle: 19-dim prediction works)
- Scaling path (this experiment: 44 params at N=20, toward hardware-scale N=50+)

**Execution command** (after CLI adaptation):
```bash
python scripts/.../run_e3_bond_resolved_scaling.py \
    --section 0 1 2 3 4 \
    --n-qubits 20 --topology heavy_hex --optimizer cobyla \
    --cobyla-maxiter 1500
```

**Timing**: ~30-60 min total with StatevectorEstimator at N=20.

**Comparison with chain_1d N=40** (justifying why both matter):

| Aspect | chain_1d N=40 (79 params) | heavy_hex N=20 (39 params) |
|--------|:-------------------------:|:--------------------------:|
| Topology symmetry | Uniform (all bonds equiv) | Non-uniform (degree 2 vs 3) |
| θ_opt structure | Quasi-uniform | Spatially structured |
| GNN necessity | Unlikely (landscape is ~2D) | Likely (39D with structure) |
| Compute cost | 12-18h (MPS) | 30-60 min (statevector) |
| Thesis chapter | Ch.5 (scaling demo) | Ch.6 (necessity proof) |
| Hardware relevance | Low (chain not native) | High (heavy_hex = IBM Heron) |

---

## Value Assessment Summary

| Execution | Time | Thesis Contribution | Priority |
|-----------|:----:|:-------------------:|:--------:|
| Section 0 (chain_1d N=40) | 10 min | Infrastructure validation | 1 (must do) |
| Section 1 (chain_1d N=40) | 2-4h | Convergence characterization at 79D | 2 (must do) |
| Section 2 (chain_1d N=40) | 12-18h | Training data for Table 5.24 | 4 (optional) |
| Section 3+4 (chain_1d N=40) | 3h | Low (quasi-uniform landscape) | 5 (skip unless sweep done) |
| **Plan E: heavy_hex N=20** | **30-60 min** | **GNN necessity proof (Ch.6 main result)** | **3 (high value, do after S1)** |

**Key principle**: The most valuable result from E3 is likely a STOP after Section 1.
Either VQE fails (proving GNN necessity) or it converges easily (proving chain_1d
symmetry protects the landscape). Both are thesis-worthy findings with 4h investment.

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| COBYLA fails to converge at 79 params (cold-start) | Medium-High | Low (this IS a valid finding) | SPSA fallback; if both fail → thesis claim is "GNN necessary" |
| chain_1d landscape is quasi-2D (no real advantage for GNN) | High | Medium | Acknowledge; pivot to Plan E (heavy_hex N=20) for thesis argument |
| MPNN overfits with 9 training points | High | Low | Built-in dropout=0.1, patience=800, early stopping |
| MPS eval too slow (>15s/eval at N=40) | Low | Medium | Reduce precision to 0.01; or use fewer h-points |
| Sweep takes >24h (timing underestimate) | Low | Medium | --section allows partial execution; sweep saves per-point to JSON |
| heavy_hex N=20 not in P1_VALID_REGIME | Certain | Low | Add entry or use empirical ~3.5 threshold |
| StatevectorEstimator OOM at N=20 | Very Low | Low | Confirmed viable from V7/V8 runs (N=20 used extensively) |

---

## Expected Thesis Contribution

**If successful (chain_1d)**:
- Table 5.24: "Bond-Resolved N=40 Pipeline Performance" (ΔE/gap, timing, scaling)
- Demonstrates pipeline scales to 79-parameter variational space
- MPNN interpolation viable at high output_dim (proof of concept)

**If successful (future heavy_hex N=20)**:
- Definitive "GNN is necessary" argument (interpolation fails, GNN succeeds)
- +30-50% energy improvement over global HVA (free lunch on hardware topology)
- Thesis Chapter 6: "Scaling Bond-Resolved HVA Beyond Classical Optimization"

**If negative result (convergence fails)**:
- Valid finding: documents the N_params boundary where warm-start VQE breaks down
- Motivates future work: differentiable MPS, transfer learning, active-learning VQE
- Still publishable as characterization of the convergence landscape

---

## References

- Existing bond-resolved validation: `documentation/binnacles/binnacle-bond-resolved-hva.md`
- MPS scaling results: `documentation/binnacles/binnacle-mps-scaling.md`
- Scaling runner (reuse pattern): `scripts/experiment_runners/scaling/run_scaling_validation.py`
- Bond-resolved runner (reuse pattern): `scripts/experiment_runners/bond_resolved/run_bond_resolved_validation.py`
- SPSA validation: V7 experiment 4A/4C
- Fusco et al. (2026, arXiv:2604.11688): Bond-resolved HVA theory
