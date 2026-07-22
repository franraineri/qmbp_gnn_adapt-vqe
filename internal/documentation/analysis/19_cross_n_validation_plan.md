# Cross-N Zero-Shot Generalization — Validation Plan & Next Steps

**Date**: 2026-06-08
**Status**: Discovery validated (v3), package fix applied, scaling plan defined.

## Key Finding

BatchNorm1d is harmful for cross-N generalization on topologies with nodal
symmetry (chain_1d). Removing it enables 5/5 PASS at N=60 (unseen) trained
only on N=40+N=80 data (14 points). Mean ΔE/gap = 0.13%.

**Package fix applied**: `MPNNPredictor(norm_type="none")` — backward compatible.

---

## 1. Validation Protocol (Ensuring Claim Reliability)

### 1.1 Statistical Robustness (multi-seed)

The current result uses seed=42 for both training and VQE data. To claim
generalization with confidence, we need:

| Test | Command | Expected |
|------|---------|----------|
| Seed sweep (GNN train) | `--source-seed 42/43/44` × 3 runs | All 5/5 PASS |
| Target N sweep | `--target-n 50,55,65,70` | All PASS (interpolation range) |
| Extrapolation test | `--target-n 100` (beyond N=80) | Measure degradation |

**Success criterion**: Mean ΔE/gap < 5% across ALL seeds AND target-N values.

### 1.2 Ablation Matrix (confirming causality)

| Ablation | norm_type | N-feature | Sources | Expected |
|----------|:---------:|:---------:|:-------:|:--------:|
| A (original) | batch | no | N=40 | ❌ 324% |
| B | batch | yes | N=40+80 | ❌ 18.5% |
| C | batch | no | N=40+80 | ⚠️ 9.5% |
| **D (fix)** | **none** | **yes** | **N=40+80** | **✅ 0.13%** |
| E | none | no | N=40+80 | Test (isolate N-feature contribution) |
| F | layer | yes | N=40+80 | Test (alternative to none) |

Already have A, B, C, D. Need E and F to fully separate contributions.

### 1.3 Leave-One-Out Cross-Validation

Train on 2 of {N=40, N=50*, N=80}, test on the held-out N.
(*N=50 needs theta_opt — run `run_scaling_validation.py` with updated runner)

| Train | Test | Status |
|-------|------|--------|
| N=40+N=80 | N=60 | ✅ Done (0.13%) |
| N=40+N=80 | N=50 | Pending (needs N=50 theta_opt) |
| N=40+N=50 | N=80 | Pending (extrapolation test) |
| N=50+N=80 | N=40 | Pending |

### 1.4 Physics Consistency Checks

| Check | Method | Pass if |
|-------|--------|---------|
| Variational principle | E_pred ≤ E_exact + ε | Always (ε < 1e-3) |
| θ monotonicity | θ_zz increases as h decreases | Consistent with training data |
| θ_x near-constant | std(θ_x) / mean(θ_x) < 5% | Physics-consistent |
| Energy vs optimal VQE | Compare with VQE at N_target | ΔE/gap difference < 1% |

</text>
</invoke>

---

## 2. Scaling Strategy (From Experiment → Thesis Contribution)

### 2.1 Immediate (no new compute needed)

| Action | Files | Impact |
|--------|-------|--------|
| ✅ Package fix applied | `src/.../predictors/mpnn.py` | `norm_type="none"` option |
| Run N=50 with updated runner | `run_scaling_validation.py --n 50` | Fills theta_opt gap |
| Ablations E, F | v3 script variations | Completes causality table |
| Thesis figures from results | `generate_scaling_figures.py` | 5 figures done |

### 2.2 Short-term (1-2 compute sessions)

| Experiment | Purpose | Compute |
|------------|---------|---------|
| Multi-N sweep (target=50,55,65,70,100) | Validate interpolation range | ~50 min |
| Multi-seed (seeds 42,43,44 × GNN train) | Statistical significance | ~30 min |
| LayerNorm vs None ablation | Choose best norm for production | ~10 min |
| **Bond-resolved N=40 (79 params)** | GNN ESSENTIAL proof | ~30 min |

### 2.3 Bond-resolved: The Killer Experiment

The current chain_1d p=1 result has 2 parameters — trivial for interpolation.
The **thesis differentiator** is bond-resolved HVA where:
- N=40 → 79 parameters (each bond has its own θ_zz)
- Scipy interpolation CANNOT handle 79D → GNN is ESSENTIAL
- This is where `norm_type="none"` + N-feature shows true value

```bash
# Bond-resolved validation (already exists)
python scripts/experiment_runners/bond_resolved/run_bond_resolved_validation.py \
    --n 40 --norm-type none --n-features 3
```

### 2.4 Thesis Claim Hierarchy

| Claim | Evidence | Status |
|-------|----------|--------|
| "GNN generalizes cross-N for TFIM" | v3 result (5/5 at N=60) | ✅ Validated |
| "BatchNorm harmful for nodal symmetry" | v2 vs v3 comparison | ✅ Validated |
| "N/100 feature enables size-aware prediction" | Need ablation E (v3 no-N-feat) | 🔲 Pending |
| "GNN essential for bond-resolved" | Need bond-resolved experiment | 🔲 Pending |
| "Framework scales to N=100+" | Need N=100 extrapolation test | 🔲 Pending |

---

## 3. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| N=60 result is seed-specific | Low | Medium | Multi-seed sweep |
| Fails at N=100 (extrapolation) | Medium | Low | Claim "interpolation" not "extrapolation" |
| Bond-resolved doesn't generalize | Medium | High | Is the key differentiator. If fails, use as negative result. |
| LayerNorm required for 2D topologies | Unknown | Medium | Test on ladder/triangular separately |

---

## 4. Commands to Execute (Priority Order)

```bash
# ── Priority 1: Complete ablation matrix ─────────────────────────
# E: norm_type=none WITHOUT N-feature (isolate N contribution)
.venv/bin/python scripts/experiment_runners/scaling/run_zero_shot_cross_n.py \
    --source-file results/scaling/scaling_N40_aer_mps_20260608_001053.json \
                  results/scaling/scaling_N80_aer_mps_20260607_211634.json \
    --target-n 60 --no-n-feature
# (This already exists as v2 ablation with BN. Need v3 without BN.)

# ── Priority 2: Multi-target validation ──────────────────────────
# Test at multiple unseen N values
for N in 50 55 65 70; do
    .venv/bin/python scripts/experiment_runners/scaling/run_zero_shot_cross_n_v3.py \
        --source-file results/scaling/scaling_N40_aer_mps_20260608_001053.json \
                      results/scaling/scaling_N80_aer_mps_20260607_211634.json \
        --target-n $N
done

# ── Priority 3: Extrapolation boundary ──────────────────────────
.venv/bin/python scripts/experiment_runners/scaling/run_zero_shot_cross_n_v3.py \
    --source-file results/scaling/scaling_N40_aer_mps_20260608_001053.json \
                  results/scaling/scaling_N80_aer_mps_20260607_211634.json \
    --target-n 100

# ── Priority 4: Generate N=50 theta_opt (fills training data gap) ─
.venv/bin/python scripts/experiment_runners/scaling/run_scaling_validation.py \
    --n 50 --topology chain_1d --strategy aer_mps --precision 0.005 --seeds 42

# ── Priority 5: Bond-resolved (thesis differentiator) ────────────
.venv/bin/python scripts/experiment_runners/bond_resolved/run_bond_resolved_validation.py \
    --n 40 --p-layers 1 --bond-resolved
```

---

## References

| Item | Path |
|------|------|
| v3 script | `scripts/experiment_runners/scaling/run_zero_shot_cross_n_v3.py` |
| v3 results | `results/scaling/zero_shot/zero_shot_v3_N40_80_to_N60_20260608_110212.json` |
| Package fix | `src/qmbp_simulation/predictors/mpnn.py` (`norm_type` param) |
| Scaling figures | `scripts/generate_scaling_figures.py` |
| v2 results (BN comparison) | `results/scaling/zero_shot/zero_shot_N40_80_to_N60_*.json` |
