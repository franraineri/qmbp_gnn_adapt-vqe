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
│   ├── audit_findings.py              ★ Deep raw-data audit (23 checks, Level 1+2)
│   ├── diagnose.py                    Automated failure root cause analysis
│   ├── scan_coverage.py               Coverage scanner + gap analysis
│   ├── verify_claims.py               Thesis claim verification (legacy)
│   ├── verify_results.py              Pipeline result verification against specs
│   ├── validate_s_series.py           S-series experiment validation
│   ├── heisenberg_summary.py          Heisenberg XXZ cross-N comparison
│   ├── sanity_check.py                24 automated sanity checks
│   ├── scaling_analyzer.py            MPS scaling law validation (N=40-120)
│   ├── scaling_extensions_analyzer.py E5 extensions (bond-dim, HE, NLCE)
│   ├── statistical_tests.py           Shared statistical test utilities
│   ├── flow_warmstart_analyzer.py     ★ Flow/σ_flow/bond-resolved analysis
│   ├── thesis_findings_validator.py   ★ Corroborate ALL thesis findings
│   ├── thesis_tables_compiler.py      ★ Auto-generate thesis tables (MD+LaTeX)
│   └── thesis_figures.py              ★ Global thesis figures (PDF/PNG)
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

## Thesis Compilation Tools

Three dedicated modules for thesis writing support, each building on the same
`ResultScanner` data pipeline:

### Thesis Findings Validator (`analysis/thesis_findings_validator.py`)

Systematically validates ALL key findings against raw experimental data.
Produces statistical evidence (t-tests, effect sizes, confidence intervals)
and classifies findings as CORROBORATED / QUALIFIED / UNSUPPORTED / CONTRADICTED.

```bash
python -m project_health.analysis.thesis_findings_validator --verbose
python -m project_health.analysis.thesis_findings_validator --only scaling,zne
python -m project_health.analysis.thesis_findings_validator --json report.json
python -m project_health.analysis.thesis_findings_validator --latex findings.tex
make validate-findings
```

15 findings validated across categories: `scaling`, `zne`, `gnn`, `topology`, `global`, `physics`.

```python
from project_health.analysis.thesis_findings_validator import run_validation

report = run_validation(categories=["scaling"], verbose=False)
print(f"Corroboration rate: {report.overall_corroboration_rate:.0%}")
for f in report.findings:
    print(f"  {f.finding_id}: {f.verdict} ({f.strength})")
```

### Thesis Tables Compiler (`analysis/thesis_tables_compiler.py`)

Auto-generates 10 publication-ready tables in Markdown and LaTeX from live data:

| ID | Table | Content |
|----|-------|---------|
| T1 | Global Pipeline Performance | All topology × N aggregated |
| T2 | ZNE Strategy Comparison | PEA vs GF vs CES |
| T3 | Scaling Law Validation | N=6 → N=80 |
| T4 | GNN-QEM Results | Correction + transfer + ablation |
| T5 | Experiment Verdicts | By category (confirmed/rejected/failed) |
| T6 | Cross-Topology Transfer | GNN generalization |
| T7 | Failure Mode Distribution | Root cause classification |
| T8 | Hyperparameter Sensitivity | hidden_dim, restarts, topology, seed |
| T9 | MPS Backend Performance | N=40-80 timing + accuracy |
| T10 | Timing Breakdown | Phase-by-phase by system size |

```bash
python -m project_health.analysis.thesis_tables_compiler --verbose
python -m project_health.analysis.thesis_tables_compiler --latex documentation/thesis_tables/
python -m project_health.analysis.thesis_tables_compiler --markdown tables.md
python -m project_health.analysis.thesis_tables_compiler --only T1,T3
make thesis-tables
```

### Thesis Global Figures (`analysis/thesis_figures.py`)

Publication-ready global figures (PDF 300dpi, no titles) aggregating data
across ALL experiments:

| Figure | Description |
|--------|-------------|
| `global_de_gap_distribution` | Histogram of ΔE/gap across 430+ runs |
| `scaling_law_comprehensive` | N vs ΔE/gap + scaling law overlay (2-panel) |
| `topology_performance_violin` | Violin plots per topology at N=10 |
| `pea_vs_gf_comparison` | Bar chart PEA vs GF per topology |
| `gnn_qem_summary_panel` | 3-panel: correction, ablation, composability |
| `experiment_verdicts_overview` | Stacked bar by category |
| `pipeline_timing_stacked` | Stacked area time vs N |
| `cross_n_performance_heatmap` | Heatmap N × h |
| `findings_corroboration_summary` | Corroboration status of all findings |
| `zne_gain_by_topology_and_strategy` | Gain heatmap: topology × strategy |

```bash
python -m project_health.analysis.thesis_figures --format pdf --dpi 300
python -m project_health.analysis.thesis_figures --list
python -m project_health.analysis.thesis_figures --only global_de_gap_distribution
python -m project_health.analysis.thesis_figures --with-titles  # for presentations
make thesis-figures
```

### Full Compilation (`make thesis-all`)

Runs everything in sequence: validate → tables → figures.

```bash
make thesis-all
# Output:
#   documentation/thesis_tables/findings_report.json
#   documentation/thesis_tables/all_tables.md
#   documentation/thesis_tables/*.tex
#   documentation/thesis_figures/*.pdf
```

### Deep Findings Audit (`analysis/audit_findings.py`)

Verifies ALL quantitative claims against raw JSON data files. 23 checks across
two levels: Level 1 reads result files directly, Level 2 uses `ResultScanner`
for diagnostics data.

```bash
PYTHONPATH=. python project_health/analysis/audit_findings.py              # Full audit (29 checks)
PYTHONPATH=. python project_health/analysis/audit_findings.py --only F2,F5 # Selective
PYTHONPATH=. python project_health/analysis/audit_findings.py --only ERR_DECOMP,CONV_RATE  # Level 2 only
PYTHONPATH=. python project_health/analysis/audit_findings.py --only N120_SWEEP,MPS_MODE,E5_EXT  # Level 4 (new)
```

Checks: F2 (PEA 18/18), F3 (scaling law), F4 (GNN-QEM), F5 (cross-N 30/30),
F8 (PEA triangular), F9 (not composable), F10 (experiment verdicts), F11 (affine),
F14 (circuit selection), F16 (cross-topo fails), F21 (DyPP), F22 (warm-start),
HEISENBERG, D1, PEA_TOPO, ABLATION, F13 (run count), MPS_CHI,
ERR_DECOMP, CONV_RATE, THETA_SMOOTH, GEN_GAP, TIMING,
NOISY_GAINS, DATA_COV, N120_SWEEP, MPS_MODE, E5_EXT, MULTI_SEED.

## MPS Scaling Digest (`--kind scaling`)

Dedicated digest mode for MPS scaling validation results:

```bash
python -m project_health.digest --kind scaling            # Summary table
python -m project_health.digest --kind scaling --verbose  # With h-values and files
python -m project_health.digest --kind scaling --json scaling.json  # Machine-readable
```

Scans three data sources from `results/scaling/`:
- `scaling_N*_*.json` — standard validation runs (N=40/50/80, multi-seed)
- `scaling_N120_full_sweep.json` — rigorous N=120 boundary sweep (3 seeds × 5 h-points)
- `mps_mode_comparison.json` — deterministic vs stochastic evaluation comparison

Output sections:
1. **Per-N summary table**: pass rate, ΔE/gap statistics, timing
2. **Scaling law validation**: predicted vs tested h_min per N
3. **N=120 sweep**: bootstrap CI, pass/fail, scaling law extrapolation
4. **Mode comparison**: speedup factor, energy consistency between modes
5. **Thesis summary**: validated system sizes, scaling law statement

New data models: `ScalingResult`, `ModeComparisonResult`, `N120SweepResult`
(exported from `project_health.digest`).

```python
from project_health.digest import ResultScanner, ScalingResult

scanner = ResultScanner(Path("results"))
scaling = scanner.scan_scaling()          # list[ScalingResult]
mc = scanner.scan_mode_comparison()       # ModeComparisonResult | None
n120 = scanner.scan_n120_sweep()          # N120SweepResult | None
```

## MPS Backend Evaluation Modes (since 2026-06-10)

The `MPSBackend(strategy="aer_mps")` now supports two evaluation modes:

| Mode | Flag | Per-eval time | Accuracy | Use case |
|------|------|:---:|:---:|---------|
| **Deterministic** (default) | `deterministic=True` | ~12ms | Machine epsilon | VQE loops, scaling validation |
| Stochastic (legacy) | `deterministic=False` | ~6s | σ ≈ precision | Noise-tolerance testing |

Results generated before 2026-06-10 used stochastic mode implicitly.
New results include `metadata.mps_evaluation_mode = "deterministic"` for traceability.

The improvement is purely technical: stochastic results had σ≈0.005 noise per
evaluation. The energy difference between modes is ≤ 2.5×10⁻⁵ — negligible
for all thesis claims (ΔE/gap threshold is 5%).

```python
# New default (exact, fast):
backend = MPSBackend(strategy="aer_mps", chi_max=64, seed=42)

# Legacy (stochastic, slow — backward compatibility):
backend = MPSBackend(strategy="aer_mps", deterministic=False, precision=0.005)
```

## Testing

```bash
# Fast tests only (~20s, no full scan)
pytest tests/test_thesis_tools.py -v -m "not slow"

# All tests including slow (~90s, full scan + figure generation)
pytest tests/test_thesis_tools.py -v
```

Tests validate: imports, crash-free execution, output structure, JSON serialization,
schema integrity of result files, and cross-tool consistency.

---

## MPNN Evaluation Suite Analyzer (`analysis/mpnn_eval_analyzer.py`)

Dedicated analyzer for `HW_REHEARSAL_V3` MPNN evaluation results (sections 10-19).

### Usage

```bash
# Full report (all runs)
python -m project_health.analysis.mpnn_eval_analyzer

# With thesis table (latest run)
python -m project_health.analysis.mpnn_eval_analyzer --thesis-table

# JSON output for further analysis
python -m project_health.analysis.mpnn_eval_analyzer --json report.json

# Verbose (shows learning curve per-k details)
python -m project_health.analysis.mpnn_eval_analyzer --verbose
```

### What It Analyzes

Parses `results/experiments/exp_hw_rehearsal_v3/run_*.json` and produces:

| Section | Metric | Status (N=6 chain) | Status (N=10 heavy_hex) |
|---------|--------|--------------------|------------------------|
| S10 Warm-start | Speedup vs random | 2.81 ± 0.23x ✅ | 2.45x ✅ |
| S11 LOO-CV | Pass rate | 100% (8 pts) ✅ | 100% (7 pts) ✅ |
| S12 Landscape | ML frac of error | 13% (circuit-limited) ✅ | — |
| S13 Interp/Extrap | Interpolation pass | 100% ✅ | — |
| S14 Noisy eval | noisy_raw ΔE/gap | 113% ❌ (gate-fold only) | 106% ❌ |
| S15 Scaling | Speedup trend | flat (-0.03/N) ✅ | — |
| S17 Transfer | chain→ladder | 200x FAIL ❌ | — |
| S19 κ-noise | Pearson \|r\| | 0.84 ✅ (chain_1d) | 0.52 ❌ (heavy_hex) |

### Key Findings (2026-06-15)

1. **MPNN warm-start confirmed**: 2.72 ± 0.26x speedup across all configs
2. **LOO-CV reliable with 7+ training pts**: 100% pass rate at both N=6 and N=10
3. **S14 noisy fails for both configs**: gate-folding ZNE alone insufficient; use full V2 pipeline (PEA+DD+twirling)
4. **κ-noise correlation only valid for chain_1d**: |r|=0.84 (chain), |r|=0.52 (heavy_hex) — different noise physics
5. **Topology transfer fails**: chain→ladder ratio=200x; GNN is NOT topology-agnostic for parameter prediction
6. **N=10 heavy_hex**: better than N=6 on most metrics (LOO mean ΔE/gap=0.38% vs 1.34%)

### κ Thresholds (calibrated at N=6 chain_1d only)

| κ | Risk | Applies to |
|---|------|-----------|
| ≥ 50 | LOW | chain_1d |
| 45-50 | MEDIUM | chain_1d |
| < 45 | HIGH | chain_1d |

**For heavy_hex N=10**: κ ∈ [111, 174] — scale is completely different. κ thresholds
don't apply. Use V2 hardware rehearsal (sections 1-9) for go/no-go decisions.
