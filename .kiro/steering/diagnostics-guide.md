---
inclusion: fileMatch
fileMatchPattern: "project_health/digest/**,scripts/run_*,experiments/**"
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
