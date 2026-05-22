# V8 Experiment Status — Single Source of Truth

> Authoritative reference for V8 experiment status.
> Last updated: 2026-05-22 (post Round 1 + C1 completion)

---

## Experiment Results Summary

| ID | Name | Result | ΔE/gap | Thesis Value |
|----|------|:------:|:------:|:------------:|
| A3 | Finite-size scaling | ⚠️ Needs re-run with N=20 | — | HIGH |
| B1 | Analytical init | ❌ Only works at h≥4.0 | 0.244 | LOW |
| B2 | Parameter freezing | ✅ 2/4 frozen, 0% loss | 0.168 | MEDIUM |
| B4 | Hessian restarts | ✅ 0 saddle points, 73% savings | 0.050 | HIGH |
| C1 | Physics-informed loss | ✅ +3.9% improvement, no regression | 0.044 | MEDIUM |
| C3 | Sign canonicalization | ✅ Not needed; N=20 p=1 works | 0.016 | HIGH |
| D1 | Weight-space detection | ✅ Peak=0.99 when loss≈0.002 | — | HIGH |
| D1-dense | Dense grid validation | ✅ Confirms regularization effect | — | HIGH |
| E4 | Longitudinal field | ❌ HVA fails at g>0 | 0.246 | MEDIUM |
| F1 | DyPP extrapolation | ❌ Only 8-13% savings | 0.109 | LOW |
| F3 | Landscape fluctuation | ❌ Doesn't predict boundary | — | LOW |

---

## Validated Decisions

| Decision | Source | Confidence |
|----------|--------|:----------:|
| 1 restart sufficient at N=6 (no saddle points) | B4 | HIGH |
| Freeze θ_zz2, θ_x2 at h≥1.5 (0% loss) | B2 | HIGH |
| Sign canonicalization unnecessary | C3 (3 runs) | HIGH |
| N=20 p=1: 3 restarts, 100 maxiter, MPS chi=64 | C3 | HIGH |
| N=20 p=1 has local min at ΔE/gap=0.437 (needs ≥5 restarts) | C3 | HIGH |
| DyPP rejected (8-13% vs 30-50% hypothesized) | F1 | HIGH |
| HVA p=2 is TFIM-specific (g>0 fails) | E4 | HIGH |
| Physics loss: safe +3.9%, not transformative at N=6 | C1 | HIGH |
| Weight-space detection needs loss≈0.002 (not 0) | D1-dense | HIGH |

---

## Infrastructure Status

| Component | Status |
|-----------|--------|
| Core framework (base, config, metrics, landscape, store) | ✅ |
| CLI (run_experiment.py, compare_results.py) | ✅ |
| StructuredLogger + RunSummary | ✅ |
| Auto-registration decorator | ✅ |
| Constraint enforcement (ValueError for p>2) | ✅ |
| Cached StatevectorEstimator | ✅ |
| Checkpoint recovery (handles corruption) | ✅ |
| Division-by-zero protection | ✅ |
| Baseline generation script | ❌ Not created |

---

## Registry (10 experiments registered)

| ID | Status | Notes |
|----|:------:|-------|
| A3 | ✅ | Needs re-run with fixed fit + N=20 MPS |
| B1 | ✅ | Complete — negative result documented |
| B2 | ✅ | Complete |
| B4 | ✅ | Complete |
| C1 | ✅ | Complete (newly implemented) |
| C3 | ✅ | Complete (3 runs) |
| D1 | ✅ | Complete (N=6 + N=10 + dense variant) |
| E4 | ✅ | Complete — negative result documented |
| F1 | ✅ | Complete — negative result documented |
| F3 | ✅ | Complete — negative result documented |

### Planned but not registered

| ID | Reason |
|----|--------|
| A1 | Infrastructure (DMRG gap), not in final plan |
| A2 | Needs xfac library |
| B3 | High effort (8h), separate sprint |
| D3 | Needs tensorly |
| E1 | Depends on A3 N=20 validation |
| E3 | Active learning — technique exists, script pending |

---

## Remaining Work

### Must-do (for thesis)

1. **A3 re-run with N=20 MPS** (~60-75 min)
   - Fixed: scipy.optimize.curve_fit (replaces broken log-transform)
   - Fixed: N=[4,6,8,10,14,20] (adds N=14 and N=20 with MPS)
   - Fixed: relative_error no longer stores h_min
   - Validates scaling law against known N=20→h_min=2.0

### Nice-to-have (if time permits)

2. **C3 with 5 restarts** — confirm 100% seed reliability at N=20 p=1
3. **D1 with controlled regularization** — dropout=0.1 or early stopping at loss≈0.002
4. **C1 at N=10** — test if physics loss improvement is larger (MSE-ΔE/gap decorrelation)
5. **E3 (Active Learning)** — implement script, test data efficiency

### Not pursuing

- F1 (DyPP) — rejected, 8-13% savings not worth complexity
- E4 extensions — HVA is model-specific, no fix possible without ansatz change
- F3 extensions — fluctuation doesn't predict boundary

---

## Key Thesis Contributions from V8

| Contribution | Section | Novelty |
|-------------|---------|:-------:|
| HVA landscape is saddle-free (B4) | §3.3 | Confirms Wiersema 2020 |
| Weight-space phase detection (D1) | §5.1 | **NOVEL** |
| Regularization sweet-spot for detection (D1-dense) | §5.1 | **NOVEL** |
| N=20 p=1 validated, sign problem resolved (C3) | §4.6 | Extends V7 |
| Parameter freezing validated on quantum circuits (B2) | §3.3 | Validates TITAN |
| Physics loss: safe but modest at N=6 (C1) | §3.4 | Validates Miao 2024 |
| HVA is TFIM-specific, not model-agnostic (E4) | §5.5 | Defines scope |
| Scaling law h_min(N) (A3, pending re-run) | §4.4 | Connects to physics |

---

## Binnacles

- `binnacle-v8-experiments-round1.md` — All Round 1 results (F1, B4, D1, B2, E4, C3, C1)
- `binnacle-d1-weight-space-phase-detection.md` — Deep analysis of D1 + D1-dense
- `binnacle-v8-experiments.md` — Earlier A3, B1, F3 results

---

## Cross-References

- Experiment plans: `documentation/plan-new-simulation-experiments-v8.md`
- Infrastructure review: `documentation/review-v8-infrastructure-gaps.md`
- Final selection: `documentation/plan-v8-noiseless-experiments-final.md`
- V6.1/V7 results: `documentation/v7/RESULTS_SUMMARY_V61_V7.md`
- Project status: `.kiro/steering/project-status.md`
