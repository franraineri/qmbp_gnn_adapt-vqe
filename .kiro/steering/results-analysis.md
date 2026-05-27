---
inclusion: fileMatch
fileMatchPattern: "results/**,scripts/digest/**,scripts/compare*"
---

# Results Analysis — Interpretation & Decision Guide

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
│   └── variants_N10_triangular/       # N=10 triangular
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

**Critical insight**: High R² + negative gain = ZNE fits well but extrapolates wrong direction. This happens at N=10 where the linear E(CES) assumption breaks down.

### Experiment (run_*.json in exp_<id>/)
Primary question: "Was the hypothesis confirmed or disproved?"

| Verdict | Meaning | Action |
|---------|---------|--------|
| `confirmed` | Hypothesis holds — technique works | Use in pipeline |
| `rejected` | Hypothesis disproved — valid finding | Document as negative result |
| `failed` | Did not meet strict threshold | Check if threshold is too strict |

## Interpretation Rules

1. **ΔE/gap is the ONLY hard criterion** — everything else is informational.
2. **Rejected ≠ broken** — E4, F1, G2, G3, G4 are experiments where rejection IS the finding.
3. **N=6 vs N=10 vs N=20**: Valid regime shifts with N (h≥1.25, h≥1.5, h≥2.0 respectively).
4. **ZNE at N=10 always fails** — this is a known physics limit (Tsubouchi et al. 2023), not a bug.
5. **theta_smoothness > 1.0** usually means the warm-start chain broke (VQE found a different basin).
6. **generalization_gap > 0.01** means MPNN is overfitting — reduce epochs or increase training data.

## Diagnostic Tools

```bash
# Quick overview of all results
python -m scripts.digest

# Compare topologies
python -m scripts.digest --kind noiseless --group-by topology

# Find outliers
python -m scripts.digest --kind noiseless --outliers

# Statistical summary
python -m scripts.digest --kind noiseless --stats --topology ladder

# Compare two configurations
python -m scripts.digest --compare variants_N10_ladder variants_N10_triangular

# Existing compare tool (experiment verdicts)
python scripts/compare.py --all
```

## Decision Flowchart

**If ΔE/gap > 5%:**
1. Check `convergence_rate` — if <1.0, VQE didn't converge → increase restarts
2. Check `theta_smoothness` — if >1.0, warm-start broke → check h-grid density
3. Check `generalization_gap` — if >0.01, MPNN overfitting → reduce epochs/increase data
4. Check h_test value — if near h_c, it's a physics limit, not a bug

**If ZNE gain is negative:**
1. Check N — if N≥10, this is expected (known failure)
2. Check n_layouts — more layouts doesn't help at N≥10
3. Check topology — triangular has worst ZNE performance

For detailed JSON schemas, see #[[file:.kiro/knowledge/result-schemas.md]]
