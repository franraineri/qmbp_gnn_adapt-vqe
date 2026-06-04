# Project Health Checker

Unified orchestration of Phase 4 analysis tools. Produces a single
health report covering experiments, coverage, quality diagnostics,
timing, distribution analysis, and actionable items.

## Usage

```bash
# Full text report
python -m project_health

# JSON output
python -m project_health --json

# Markdown output (for documentation/reports)
python -m project_health --markdown

# Summary only (no per-experiment table)
python -m project_health --compact

# Save to file (explicit name)
python -m project_health -o health_report.txt
python -m project_health --json -o health_report.json

# Save to directory (auto-timestamped filename)
python -m project_health -o reports/
python -m project_health --markdown -o reports/
# → produces e.g. reports/health_report_20260603_195022.md

# Only show what changed since last run
python -m project_health --diff-only

# CI mode (exit code 1 if CRITICAL gaps exist)
python -m project_health --ci

# Skip state persistence (no delta tracking)
python -m project_health --no-state

# Verbose logging
python -m project_health -v
```

## Architecture

```
project_health/
├── __init__.py       Public API (all typed models exported)
├── __main__.py       CLI entry point (--json, --markdown, --compact, -o dir/)
├── engine.py         Core orchestration (scan → analyze → report)
├── coverage.py       Coverage gaps + VQE/MPNN/timing/distribution analytics
├── models.py         Data models (all typed, serializable)
├── reporter.py       Output formatters (text, JSON, Markdown)
├── state.py          Persistence for delta tracking
├── figures.py        Matplotlib figure generation (optional)
├── compare.py        Cross-experiment comparison CLI
├── analysis/         Analysis scripts (diagnose, scan_coverage, verify, etc.)
│   ├── diagnose.py           Automated failure root cause analysis
│   ├── scan_coverage.py      Coverage scanner + gap analysis + extended analytics
│   ├── verify_claims.py      Thesis claim verification against data
│   ├── verify_results.py     Pipeline result verification against specs (generic)
│   ├── validate_s_series.py  S-series experiment validation
│   └── heisenberg_summary.py Heisenberg XXZ cross-N comparison
├── digest/           Result scanning and formatting
│   ├── scanner.py            ResultScanner (parse all results)
│   ├── formatters.py         Output formatting
│   └── models.py             Result data models
└── README.md         This file
```

## Design Principles

1. **No re-parsing** — uses `ResultScanner` from `project_health/digest/` as the
   single data source. No duplicate JSON parsing.

2. **No re-evaluation** — experiment verdicts come pre-computed from the
   scanner (which uses `framework/criteria.py`). We read, not recalculate.

3. **Pure logic** — `engine.py` produces a `HealthReport` dataclass with
   zero I/O. Formatting is a separate concern in `reporter.py`.

4. **Delta tracking** — persists seen files in `.project_health_state.json`.
   On subsequent runs, new AND removed results are highlighted.

5. **CI-friendly** — `--ci` flag exits non-zero on CRITICAL gaps.
   `--json` output is machine-parseable.

6. **Timestamped outputs** — when saving to a directory with `-o reports/`,
   filenames include ISO timestamps for versioned report history.

7. **Multi-source actionable items** — actions are derived not just from
   coverage gaps, but also from VQE quality diagnostics, MPNN overfitting,
   distribution imbalances, energy decomposition, and experiment failures.

8. **Doesn't replace tools** — each Phase 4 tool (digest, scan_coverage,
   verify_claims) remains usable independently for deep-dive analysis.
   All live under `project_health/` now.

## Dependencies

- `project_health/digest/scanner.py` → `ResultScanner` (scan + parse)
- `project_health/digest/models.py` → `NoiselessResult`, `NoisyResult`, `ExperimentResult`
- `qmbp_simulation.framework.criteria` → `EXPERIMENT_CRITERIA`, `compute_verdict`

## What It Reports

| Section | Source | New in v2 |
|---------|--------|:---------:|
| Experiment verdicts | `ResultScanner` → `ExperimentResult.verdict` | |
| Noiseless quality (global + per-topology) | `ResultScanner` → `NoiselessResult.delta_e_over_gap` | |
| Noisy/ZNE quality | `ResultScanner` → `NoisyResult.success_criteria_met` | |
| VQE convergence quality | `NoiselessResult.convergence_rate`, `.theta_smoothness` | ✅ |
| MPNN training quality | `NoiselessResult.generalization_gap`, `.theta_zz_mse` | ✅ |
| Timing breakdown | `.elapsed_s` across all result types | ✅ |
| Result distribution | By model, topology, N-qubits, p-layers | ✅ |
| Energy error decomposition | Circuit vs MPNN error attribution | ✅ |
| Coverage gaps | Logic in `coverage.py` over scan results | |
| Delta since last run (new + removed) | File-set diff via `state.py` | |
| Actionable items (multi-source) | Gaps + VQE + MPNN + distribution + experiments | ✅ |

## Gap Types Detected

| Gap | Priority | Description |
|-----|----------|-------------|
| `MISSING_ZNE` (heavy_hex N≥10) | CRITICAL | Blocks hardware deployment |
| `MISSING_EXPERIMENT` | HIGH | Defined in criteria.py but no results |
| `MISSING_P1_NOISELESS` | MEDIUM | p=2 exists but p=1 missing |
| `INSUFFICIENT_SEEDS` | MEDIUM | < 3 seeds for reproducibility |
| `MISSING_ZNE` (other) | MEDIUM | p=1 noiseless without ZNE validation |
| `INVALID_REGIME` | LOW | h_test below valid boundary (expected failures) |

## ZNE Technique Comparison (`compare.py --zne`)

Consolidated analysis of all GF-ZNE and PEA-ZNE experiment results:

```bash
python project_health/compare.py --zne          # Full cross-method comparison
python project_health/compare.py --zne --json results.json  # Machine-readable
```

Scans directories: `exp_gf_zne_cmp/`, `exp_zne_3way/`, `exp_pea_zne_val/`,
`exp_pea_hw_ready/`, `exp_pea_pipeline/`, `exp_zne_cross_topo/`.

Produces per-method statistics (gain, R², robustness), per-topology breakdown,
coverage matrix, and gap analysis. Current state (2026-06-04):

| Method | Mean Gain | Evaluations | Always Positive |
|--------|:---------:|:-----------:|:---------------:|
| PEA-ZNE | +83% | 48 | 48/48 (100%) |
| GF-ZNE | +12% | 60 | 54/60 (90%) |
| CES-ZNE | +3% | 18 | 14/18 (78%) |

## Actionable Item Sources

Actions are derived from multiple analysis dimensions, not just coverage gaps:

| Source | What it detects | Category |
|--------|----------------|----------|
| Coverage gaps | Missing configs, seeds, ZNE, experiments | `hardware`, `coverage`, `reproducibility` |
| VQE quality | Chain breaks (θ>1.0), low convergence rate | `vqe_quality` |
| MPNN quality | Overfitting (gen_gap>0.01), high θ-MSE | `mpnn_quality` |
| Distribution | p-layer imbalance, under-represented topologies | `distribution` |
| Energy decomposition | Dominant error source (circuit vs MPNN) | `diagnostics` |
| Experiment failures | Zero-pass experiments, near-threshold partial passes | `experiments` |

### Failure Mode Correlation

The actions map directly to the project's known failure modes:

| Root Cause | % of Failures | Detected By |
|-----------|:---:|-------------|
| CHAIN_BREAK (θ>1.0) | 45% | VQE quality → `n_chain_break_warnings` |
| MPNN_OVERFIT (gen_gap>0.01) | 25% | MPNN quality → `n_overfit_warnings` |
| BOUNDARY_EFFECT | 14% | Coverage gaps → `INVALID_REGIME` |
| OUTSIDE_REGIME | 9% | Coverage gaps → `INVALID_REGIME` |
| VQE_DIVERGENCE | 7% | VQE quality → low `convergence_rate_min` |

## Output Formats

| Format | Flag | Use Case |
|--------|------|----------|
| Text | *(default)* | Console inspection, quick checks |
| JSON | `--json` | Machine parsing, CI pipelines, further analysis |
| Markdown | `--markdown` | Documentation, reports, git-tracked history |

## Programmatic Usage

```python
from project_health.engine import run_health_check
from project_health.models import Priority
from project_health.reporter import format_markdown, generate_timestamped_filename

report = run_health_check()

# Check if deployment-ready
critical = [a for a in report.actions if a.priority == Priority.CRITICAL]
if critical:
    print("NOT ready for hardware deployment")

# Access per-topology stats
for topo, stats in report.noiseless_by_topology.items():
    print(f"{topo}: {stats['pass_rate']:.0%} pass rate")

# Check VQE health
if report.vqe_quality.n_chain_break_warnings > 0:
    print(f"⚠️ {report.vqe_quality.n_chain_break_warnings} chain breaks detected")

# Check MPNN health
if report.mpnn_quality.n_overfit_warnings > 0:
    print(f"⚠️ {report.mpnn_quality.n_overfit_warnings} overfit warnings")

# Generate timestamped report
filename = generate_timestamped_filename("health_report", "md")
output = format_markdown(report)
Path(f"reports/{filename}").write_text(output)

# Filter actions by category
hw_actions = [a for a in report.actions if a.category == "hardware"]
quality_actions = [a for a in report.actions if a.category in ("vqe_quality", "mpnn_quality")]
```
