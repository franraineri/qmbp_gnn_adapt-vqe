---
inclusion: fileMatch
fileMatchPattern: "project_health/**,analysis/**,scripts/run_*,experiments/**,results/**"
---

# Diagnostics Guide — Detecting Problems & Using Tools

## Detecting Broken or Corrupted Results

### Required Fields (if missing → result is broken)

**Noiseless pipeline_run_*.json MUST have:**
- `config.n_qubits` or `system.n_qubits` (int > 0)
- `config.h_values` (non-empty list, descending)
- `phase4_results` (non-empty list with at least one entry)
- `phase4_results[0].delta_e_over_gap` (float, not null)
- `diagnostics.phase2.convergence_rate` (float 0–1)
- `elapsed_s` (float > 0)

**Noisy noisy_*.json MUST have:**
- `config.n_qubits` or `system.n_qubits` (int > 0)
- `summary.mean_r2` (float 0–1)
- `summary.n_total` (int > 0)
- `results_per_h` (non-empty list)

**Experiment run_*.json MUST have:**
- `config.experiment_id` (non-empty string)
- `analysis.summary` (dict without "error" key)
- `analysis.summary.pass_rate` or `analysis.summary.mean_de_gap`

### Sanity Checks (if violated → result is suspect)

| Check | Valid range | If violated |
|-------|------------|-------------|
| `delta_e_over_gap` | 0–50 | >50 means VQE completely failed |
| `convergence_rate` | 0–1 | Must be exactly in [0,1] |
| `mean_r2` | 0–1 | >1 or <0 means computation error |
| `elapsed_s` | >0 | 0 means the run didn't actually execute |
| `n_qubits` | 4,6,8,10,14,20 | Other values are non-standard |
| `p_layers` | 1 or 2 | >2 violates Mele et al. constraint |
| `theta_smoothness` | 0–10 | >10 means catastrophic warm-start failure |
| `generalization_gap` | 0–1 | >1 means MPNN is completely broken |
| `h_values` order | descending | Ascending = wrong sweep direction |

### Common Corruption Patterns

1. **Empty phase4_results**: Run crashed during Phase 4 → re-run needed
2. **delta_e_over_gap = null**: Phase 4 prediction failed → check MPNN checkpoint
3. **convergence_rate = 0**: All VQE runs failed → check h-grid (too close to h_c?)
4. **elapsed_s = 0**: Timer not started → file was created but run didn't execute
5. **Multiple run files, latest has worse results**: Regression → compare configs

## Using the Digest Tool

### When to use which mode:

| Situation | Command |
|-----------|---------|
| "What's the overall status?" | `python -m project_health.digest` |
| "How does ladder compare to chain?" | `--kind noiseless --group-by topology` |
| "What's failing and why?" | `--kind noiseless --outliers` |
| "Is this result statistically significant?" | `--kind noiseless --stats --topology X` |
| "Did config A improve over config B?" | `--compare folder_A folder_B` |
| "Show me only the best results" | `--sort delta_e --top 10` |
| "What's the ZNE situation?" | `--kind noisy --group-by n_qubits` |
| "Which experiments are confirmed?" | `--kind experiment --sort verdict` |
| "Export for thesis table" | `--markdown -o thesis_table.md` |
| "Feed to another script" | `--json results.json` |

### Reading the Progress Output (stderr)

```
[digest] Results dir: results              ← confirms correct directory
[digest] Scanning all result areas...      ← scan started
Scanning 18 experiment directories...      ← from scanner module
  → 15 experiments parsed                  ← how many had valid results
Scanning 8 thesis folders...               ← thesis scan
  → 130 noiseless, 58 noisy results       ← total discovered
[digest] Scanned: 130 noiseless, ...       ← summary before filters
[digest] Applying filters: topology=ladder ← active filters shown
[digest] After filters: 45 noiseless, ...  ← post-filter counts
[digest] Done — saved to output.txt        ← completion
```

If counts are unexpectedly low → check folder names, topology spelling, or n_qubits values.

### Interpreting Group-By Output

```
Group           Count   Pass   Marg   Fail   Med ΔE/gap   Mean ΔE/gap  Best       Worst
chain_1d        38      26     7      5      0.0293       0.1357       0.0010     2.4208
ladder          45      28     8      9      0.0345       0.0992       0.0004     1.8892
triangular      46      27     4      15     0.0404       0.4143       0.0010     14.4009
```

- **Median vs Mean**: Large difference = outliers pulling the mean (triangular: 0.04 vs 0.41)
- **Pass/Fail ratio**: Quick health check per group
- **Worst**: If >1.0, there's a catastrophic failure in that group → use `--outliers` to find it

### Interpreting Outlier Output

```
nl_seed_42    14.4009    triangular 10   high gen.gap
ext_extrapolation  2.4208  chain_1d  10   rough θ-sweep
```

The "Why?" column gives automatic diagnosis:
- `high gen.gap` → MPNN overfitting (reduce epochs or add data)
- `rough θ-sweep` → warm-start chain broke (check h-grid density)
- `only N restart(s)` → insufficient VQE exploration
- `small hidden=X` → MPNN capacity too low
- `investigate manually` → no obvious automated diagnosis

## Failure Triage Order

When a result fails (ΔE/gap > 5%), check in this order:

1. **Is h_test in the valid regime?** (N=6: h≥1.25, N=10: h≥1.5, N=20: h≥2.0)
   - If not → physics limit, not a bug
2. **Did VQE converge?** (convergence_rate = 1.0?)
   - If not → increase restarts or maxiter
3. **Is θ-sweep smooth?** (theta_smoothness < 0.1?)
   - If not → h-grid too sparse or warm-start broke
4. **Is MPNN generalizing?** (generalization_gap < 1e-3?)
   - If not → overfitting (reduce epochs, increase patience, add data)
5. **Is the topology supported?** (chain_1d, ladder, triangular at tested N?)
   - If new topology → may need different hyperparameters

## Cross-Reference

- Result interpretation thresholds: this file + `results-analysis.md`
- JSON field reference: #[[file:.kiro/knowledge/result-schemas.md]]
- Known error patterns: #[[file:.kiro/knowledge/error-patterns.md]]
- Experiment protocol: #[[file:.kiro/steering/experiment-protocol.md]]
- Tool invocation reference: #[[file:.kiro/steering/analysis-tooling.md]]

## MPNN Evaluation Diagnostics

When MPNN prediction quality is suspect, use the dedicated analyzer:

```bash
# Full report across all V3 runs
python -m project_health.analysis.mpnn_eval_analyzer

# Thesis-ready table (latest run)
python -m project_health.analysis.mpnn_eval_analyzer --thesis-table

# JSON for programmatic analysis
python -m project_health.analysis.mpnn_eval_analyzer --json report.json
```

**Diagnostic indicators:**

| Metric | Good | Marginal | Action |
|--------|------|---------|--------|
| S10 speedup_vs_random | ≥ 1.5x | 1.0-1.5x | More training data / epochs |
| S11 LOO pass_rate | ≥ 80% | 60-80% | Extend h_train grid |
| S12 ML fraction | < 30% | 30-60% | More training pts near h_test |
| S13 interp pass_rate | ≥ 80% | 60-80% | Check h_test inside h_train range |
| S19 |r| κ-noise | ≥ 0.70 | 0.50-0.70 | Only valid for chain_1d topology |

Auto-generated warnings flag:
- speedup < 1x → GNN HURTS, retrain before QPU
- LOO pass_rate < 60% → insufficient data, extend h_train  
- topology_transfer ratio > 3x → GNN is topology-specific
- LOO std > 15% → result seed-sensitive, more epochs needed

## Additional Diagnostic Tools

### Sanity Check (`python -m project_health.analysis.sanity_check`)

24 automated checks on analysis outputs. Run after any bulk analysis.

```bash
python -m project_health.analysis.sanity_check              # All 24 checks
python -m project_health.analysis.sanity_check --only physics  # Physics subset
python -m project_health.analysis.sanity_check --json out.json  # Machine output
```

Checks: valid regime consistency, scaling law fit, claim contradictions, statistical test validity.

### Scaling Analyzer (`python -m project_health.analysis.scaling_analyzer`)

MPS scaling law analysis — validates h_min(N) formula and timing predictions.

```bash
python -m project_health.analysis.scaling_analyzer
```

### Failure Diagnosis Priority (`analysis/diagnose.py`)

When multiple results fail, triage by root cause:

| Root Cause | % | Detection |
|-----------|---|-----------|
| CHAIN_BREAK (θ>1.0) | 45% | `theta_smoothness > 1.0` in Phase 2 |
| MPNN_OVERFIT (gen_gap>0.01) | 25% | `generalization_gap > 0.01` in Phase 3 |
| BOUNDARY_EFFECT | 14% | h_test near valid regime boundary |
| OUTSIDE_REGIME | 9% | h_test below h_min_safe |
| VQE_DIVERGENCE | 7% | `convergence_rate < 0.8` |

```bash
python analysis/diagnose.py --all --severity fail   # Only failures
python analysis/diagnose.py results/thesis/variants_N10_ladder/  # Specific folder
```
