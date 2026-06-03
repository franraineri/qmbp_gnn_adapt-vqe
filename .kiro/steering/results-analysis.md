---
inclusion: fileMatch
fileMatchPattern: "results/**,scripts/digest/**,scripts/compare*,analysis/**,documentation/analysis/**"
---

# Results Analysis — Interpretation & Decision Guide

## Canonical Analysis Sources (ALWAYS USE THESE)

When writing the thesis, citing analysis results, or making claims about the framework:

| Purpose | Canonical File | Why |
|---------|---------------|-----|
| **Summary & thesis statements** | `documentation/analysis/08_summary.md` | Final corrected summary with all 14 studies |
| **Definitive tables (5.1–5.6)** | `documentation/analysis/09_thesis_tables.md` | Top-15 per topology, cross-topology, ZNE boundary |
| **Verified findings (131 variants)** | `analysis/10_key_findings_corrected.md` | Post-verification data with gen_gap/smoothness rules |
| **p=1 ZNE confirmation** | `documentation/analysis/14_p1_zne_validation.md` | 9 runs, 3 topologies × 3 seeds |
| **Cross-topology diagnostics** | `analysis/09_diagnostics_deep_dive.md` | Corrected table (131 variants), correlations |
| **Findings master index** | `analysis/FINDINGS_INDEX.md` | All 35 findings with confidence levels |

**DO NOT cite** files in `documentation/analysis/worklog/` or `analysis/worklog/` — these contain
superseded data (old 120-variant counts, incorrect ladder N=6 pass rates) or session worklogs.

**Key corrections applied** (why worklog files were moved):
- Ladder N=6 pass rate: ~~23%~~ → **50%** (digest missed 9 variants)
- Global pass rate: ~~59%~~ → **64%** (131 vs 120 variants)
- "Hyperparams irrelevant" → only at N=10; h=128 critical at N=6
- Triangular N=10 "seed-dependent" → outlier-driven (without outlier, std=0.003)

---

## Where Results Live

```
results/
├── experiments/           # BaseExperiment hypothesis tests (V8)
│   └── exp_<id>/          # e.g., exp_b4/, exp_g1/
│       ├── run_*.json     # Timestamped results (latest = most recent)
│       └── log_*.json     # Execution logs
├── thesis/                # Definitive thesis variant runs
│   ├── n6_noiseless/      # Baseline N=6 pipeline
│   ├── n6_noisy/          # Baseline N=6 ZNE
│   ├── variants_N6_N10_1D_linnear/   # Chain 1D variants
│   ├── variants_N6_ladder/            # N=6 ladder
│   ├── variants_N6_triangular/        # N=6 triangular
│   ├── variants_N10_ladder/           # N=10 ladder
│   ├── variants_N10_triangular/       # N=10 triangular
│   ├── p1_variants_N10/              # p=1 multi-topology (R1)
│   ├── p1_variants_N10_r2/           # p=1 corrected h_test (R2)
│   ├── p1_variants_N16_r2/           # p=1 N=16 scaling
│   ├── p1_variants_N24_r2/           # p=1 N=24 scaling
│   ├── verification_r1/              # Systematic verification (2026-05-30)
│   ├── analysis_p1_zne/              # p=1 ZNE validation (9 runs)
│   └── variants_N10_multi/           # Multi-topology batch runs
│       └── <variant_name>/
│           ├── pipeline_run_*.json    # Noiseless 4-phase result
│           ├── noisy_*.json           # ZNE result
│           ├── diagnostics.json       # Phase-by-phase diagnostics
│           └── checkpoints/           # MPNN/phase12 data
└── benchmarks/            # Performance benchmarks (not analysis-relevant)
```

**Rule**: Always use the LATEST file (sorted reverse by timestamp). Multiple runs in the same folder = re-runs with improved config.

## Three Result Kinds & Their Key Metrics

### Noiseless Pipeline (pipeline_run_*.json)
Primary question: "Does the MPNN predict well enough for phase characterization?"

| Metric | Good | Marginal | Bad | Meaning |
|--------|------|----------|-----|---------|
| `delta_e_over_gap` | <0.05 | 0.05–0.10 | >0.10 | Energy error relative to spectral gap |
| `convergence_rate` | 1.0 | 0.8–1.0 | <0.8 | Fraction of h-points where VQE converged |
| `theta_smoothness` | <0.05 | 0.05–1.0 | >1.0 | Parameter variation between adjacent h |
| `generalization_gap` | <1e-4 | 1e-4–1e-2 | >1e-2 | Train vs test MSE difference (overfitting) |
| `phase_correct` | true | — | false | Correct paramagnetic/ferromagnetic label |

### Noisy/ZNE (noisy_*.json)
Primary question: "Does ZNE improve the noisy estimate?"

| Metric | Good | Marginal | Bad | Meaning |
|--------|------|----------|-----|---------|
| `mean_r2` | >0.95 | 0.8–0.95 | <0.8 | Linear fit quality of E vs CES |
| `mean_gain_pct` | >+30% | 0–30% | <0% | Error reduction from ZNE (negative = ZNE hurts) |
| `n_mitigated_wins` | =n_total | >0 | 0 | How many h-points ZNE beats raw noisy |

**Critical insight**: High R² + negative gain = ZNE fits well but extrapolates wrong direction. This happens at N=10 p=2 where the linear E(CES) assumption breaks down. **Solution**: Use p=1 at N=10 (same CX budget as p=2 N=6) — confirmed with 9 runs, mean gain=+49%.

### Experiment (run_*.json in exp_<id>/)
Primary question: "Was the hypothesis confirmed or disproved?"

| Verdict | Meaning | Action |
|---------|---------|--------|
| `confirmed` | Hypothesis holds — technique works | Use in pipeline |
| `rejected` | Hypothesis disproved — valid finding | Document as negative result |
| `failed` | Did not meet strict threshold | Check if threshold is too strict |

Verdicts are computed by `compute_verdict(exp_id, summary)` from `framework/criteria.py`.
Per-experiment thresholds live in `EXPERIMENT_CRITERIA` (same module).

## Interpretation Rules

1. **ΔE/gap is the ONLY hard criterion** — everything else is informational.
2. **Rejected ≠ broken** — E4, F1, G2, G3, G4 are experiments where rejection IS the finding.
3. **N=6 vs N=10 vs N=20**: Valid regime shifts with N (h≥1.25, h≥1.5, h≥2.0 respectively).
4. **ZNE at N=10 always fails** — this is a known physics limit (Tsubouchi et al. 2023), not a bug.
5. **theta_smoothness > 1.0** usually means the warm-start chain broke (VQE found a different basin).
6. **generalization_gap > 0.01** means MPNN is overfitting — reduce epochs or increase training data.

## Diagnostic Tools

```bash
# Coverage scan — what data exists, what's missing
python analysis/scan_coverage.py
python analysis/scan_coverage.py --discover --extended
python analysis/scan_coverage.py --topology ladder --p 1 --n-qubits 10

# Failure diagnosis — automated root cause analysis
python analysis/diagnose.py --all
python analysis/diagnose.py results/thesis/p1_variants_N10_r2 --severity fail

# Quick overview of all results (digest)
python scripts/digest/run_digest.py
python scripts/digest/run_digest.py --kind noiseless --p-layers 1

# Compare topologies
python scripts/digest/run_digest.py --kind noiseless --group-by topology

# Verification plan — systematic claim validation
python scripts/experiment_runners/run_verification_plan.py --list
python scripts/experiment_runners/run_verification_plan.py --noiseless-only  # Tier 1
python scripts/experiment_runners/verify_results.py                          # Analyze results

# Existing compare tool (experiment verdicts)
python scripts/compare.py --all

# Export coverage data
python analysis/scan_coverage.py --discover --json coverage.json --csv coverage.csv --markdown coverage.md
```

## Decision Flowchart

**If ΔE/gap > 5%:**
1. Check `convergence_rate` — if <1.0, VQE didn't converge → increase restarts
2. Check `theta_smoothness` — if >1.0, warm-start broke → check h-grid density
3. Check `generalization_gap` — if >0.01, MPNN overfitting → reduce epochs/increase data
4. Check h_test value — if near h_c, it's a physics limit, not a bug

**If ZNE gain is negative:**
1. Check N and p — if N≥10 AND p=2, this is expected (known failure at ~36 CX gates)
2. **Solution**: Use p=1 (reduces CX to ~18, recovers +49% gain)
3. Check n_layouts — more layouts doesn't help at N≥10 p=2
4. Check topology — triangular has worst ZNE performance but p=1 still works (+50%)

For detailed JSON schemas, see #[[file:.kiro/knowledge/result-schemas.md]]
