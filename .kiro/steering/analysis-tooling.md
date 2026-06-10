---
inclusion: fileMatch
fileMatchPattern: "analysis/**,scripts/**,experiments/**,results/**,project_health/**"
---

# Analysis Tooling — ALWAYS Use Existing Scripts

## Rule (MANDATORY)

When analyzing data, inspecting results, or producing output files:

**ALWAYS use the existing analysis and digest scripts first.**
Do NOT write ad-hoc one-off scripts or inline Python when an existing tool covers the need.
If the existing tool is close but not sufficient, **extend it** rather than creating a new file.

## Quick Decision Tree

```
Need to...
├── See overall project status?     → python -m project_health [--compact]
├── Inspect pipeline results?       → python -m project_health.digest --kind noiseless
├── Inspect ZNE results?            → python -m project_health.digest --kind noisy
├── Inspect experiment verdicts?    → python -m project_health.digest --kind experiment
├── Inspect cross-topology results? → python -m project_health.digest --kind cross_topology
├── Compare topologies/configs?     → python -m project_health.digest --group-by topology
├── Check what data exists?         → analysis/scan_coverage.py --discover
├── Find coverage gaps?             → analysis/scan_coverage.py --extended
├── Understand a failure?           → analysis/diagnose.py [path] [--all]
├── Compare experiments?            → scripts/compare.py [--all] [--category X]
├── Compare ZNE methods?            → project_health/compare.py --zne
├── Validate runner script?         → python scripts/preflight.py --from-script <path>
├── Validate a thesis claim?        → python -m project_health.analysis.verify_claims
├── Verify pipeline correctness?    → python -m project_health.analysis.verify_results
├── Check analysis sanity?          → python -m project_health.analysis.sanity_check
├── Analyze MPS scaling?            → python -m project_health.analysis.scaling_analyzer
├── Analyze E5 extensions?          → python -m project_health.analysis.scaling_extensions_analyzer
├── Analyze cross-topology transfer?→ make cross-topology
├── Generate figures?               → make figures (PNG) or make figures-thesis (PDF)
├── PCA phase detection?            → python scripts/analysis/theta_pca_phase_detection.py
├── θ-derivative vs D1?            → python scripts/analysis/theta_derivative_analysis.py
├── Full analysis pipeline?         → python analysis/run_analysis.py
├── Deep raw-data audit (23 checks)?→ PYTHONPATH=. python project_health/analysis/audit_findings.py
│
│ ─── THESIS COMPILATION (global aggregation) ─────────────────
├── Corroborate ALL findings?       → python -m project_health.analysis.thesis_findings_validator
├── Generate thesis tables?         → python -m project_health.analysis.thesis_tables_compiler
├── Generate thesis global figures? → python -m project_health.analysis.thesis_figures
├── Full thesis compilation?        → make thesis-all
├── Validate + LaTeX export?        → make validate-findings-latex
└── Something tools don't cover?    → Extend the closest existing tool
```

## Tool Reference (complete flags)

### 1. Project Health Report (`python -m project_health`)

Full health report: experiments, coverage, VQE/MPNN quality, timing, actions.

```bash
python -m project_health                    # Full text report
python -m project_health --compact          # Summary only
python -m project_health --json             # Machine-parseable
python -m project_health --markdown         # For documentation
python -m project_health -o reports/        # Auto-timestamped save
python -m project_health --diff-only        # Only show changes since last run
python -m project_health --ci              # Exit 1 on CRITICAL gaps
python -m project_health --no-state        # Skip delta tracking
```

**Programmatic:**
```python
from project_health.engine import run_health_check
from project_health.models import Priority

report = run_health_check()
critical = [a for a in report.actions if a.priority == Priority.CRITICAL]
for topo, stats in report.noiseless_by_topology.items():
    print(f"{topo}: {stats['pass_rate']:.0%}")
```

### 2. Result Digest (`python -m project_health.digest`)

Quick inspection of all results by kind: noiseless, noisy, experiment.

```bash
# By kind
python -m project_health.digest --kind noiseless
python -m project_health.digest --kind noisy
python -m project_health.digest --kind experiment

# Filters
python -m project_health.digest --kind noiseless --topology ladder --n-qubits 10 --p-layers 1
python -m project_health.digest --kind noiseless --folder variants_N10_ladder
python -m project_health.digest --kind noiseless --model tfim_longitudinal

# Sorting + limiting
python -m project_health.digest --kind noiseless --sort delta_e --top 10
python -m project_health.digest --kind noisy --sort gain --top 5
python -m project_health.digest --kind experiment --sort verdict

# Grouped comparisons
python -m project_health.digest --kind noiseless --group-by topology
python -m project_health.digest --kind noiseless --group-by n_restarts
python -m project_health.digest --kind noisy --group-by n_qubits

# Statistical analysis
python -m project_health.digest --kind noiseless --stats
python -m project_health.digest --kind noiseless --outliers

# Side-by-side comparison
python -m project_health.digest --compare variants_N10_ladder variants_N10_triangular

# Output formats
python -m project_health.digest --markdown -o digest.md
python -m project_health.digest --json digest.json
```

**Programmatic:**
```python
from project_health.digest import ResultScanner, NoiselessResult
scanner = ResultScanner(Path("results"))
noiseless, noisy, experiments = scanner.scan_all()
ladder_n10 = [r for r in noiseless if r.topology == "ladder" and r.n_qubits == 10]
```

### 3. Coverage Scanner (`analysis/scan_coverage.py`)

Discover data, find gaps, extended analytics.

```bash
python analysis/scan_coverage.py --discover           # What data exists
python analysis/scan_coverage.py --extended           # Full analytics
python analysis/scan_coverage.py --topology chain_1d  # Filter by topology
python analysis/scan_coverage.py --p 1                # Filter by depth
```

### 4. Failure Diagnosis (`analysis/diagnose.py`)

Automated root cause analysis for failed results.

```bash
python analysis/diagnose.py --all                     # All results
python analysis/diagnose.py results/experiments/exp_B4/  # Specific folder
python analysis/diagnose.py --severity fail           # Only failures
```

### 5. Experiment Comparison (`scripts/compare.py`)

Cross-experiment verdict comparison against criteria.

```bash
python scripts/compare.py --all                       # All experiments
python scripts/compare.py --category optimization     # By category
python scripts/compare.py --noisy                     # ZNE experiments
```

### 6. ZNE Method Comparison (`project_health/compare.py`)

Cross-method ZNE analysis (PEA vs GF vs CES).

```bash
python project_health/compare.py --zne                # Full comparison
python project_health/compare.py --zne --json out.json  # Machine-readable
```

### 7. Sanity Check (`python -m project_health.analysis.sanity_check`)

24 automated checks on analysis outputs.

```bash
python -m project_health.analysis.sanity_check         # All checks
python -m project_health.analysis.sanity_check --only physics  # Physics subset
python -m project_health.analysis.sanity_check --json out.json  # JSON output
```

### 8. MPS Scaling Analyzer (`python -m project_health.analysis.scaling_analyzer`)

Scaling law analysis for N>30 results.

```bash
python -m project_health.analysis.scaling_analyzer     # Full analysis
```

### 9. Scaling Extensions Analyzer (`python -m project_health.analysis.scaling_extensions_analyzer`)

E5 results: bond dimension, HE comparison, NLCE convergence, thesis tables.

```bash
python -m project_health.analysis.scaling_extensions_analyzer              # Full report
python -m project_health.analysis.scaling_extensions_analyzer --verbose     # Per-h detail
python -m project_health.analysis.scaling_extensions_analyzer --thesis-tables  # Tables 5.25/5.26
python -m project_health.analysis.scaling_extensions_analyzer --cross-check    # Cross-section validation
python -m project_health.analysis.scaling_extensions_analyzer --convergence-data plot.json  # NLCE L vs E/N
python -m project_health.analysis.scaling_extensions_analyzer --json report.json  # Full JSON export
```

### 10. Cross-Topology Transfer (`python -m project_health.digest --kind cross_topology`)

Cross-topology GNN transfer results: within-topology cross-N, cross-topology, ablation.

```bash
python -m project_health.digest --kind cross_topology           # Full report
python -m project_health.digest --kind cross_topology --verbose # Per-direction detail
python -m project_health.digest --kind cross_topology --json cross_topo.json  # JSON export
make cross-topology                                             # Quick report
```

Result files scanned (from `results/scaling/cross_topology/`):
- `cross_n_validation_*.json` — within-topology cross-N (sanity check)
- `cross_topology_transfer_*.json` — tri→hex, hex→tri transfer
- `ablation_study_*.json` — GNN vs MLP vs Scipy + norm_type comparison
- `orchestrator_summary_*.json` — full run summary + verdicts

### 11. Preflight Validation (`python scripts/preflight.py`)

Validate variant runner scripts before execution (9 checks).

```bash
python scripts/preflight.py --from-script <path>       # Standard
python scripts/preflight.py --from-script <path> --strict  # Warnings=errors (CI)
python scripts/preflight.py --from-json variants.json  # From JSON
make preflight SCRIPT=<path>                           # Via Makefile
```

### 10. Figures (`make figures` / `make figures-thesis`)

Generate all analysis + thesis figures.

```bash
make figures            # PNG, all figures
make figures-thesis     # PDF 300dpi, thesis-ready
```

### 11. Specialized Analysis Scripts

```bash
# Unsupervised phase detection from θ_opt
python scripts/analysis/theta_pca_phase_detection.py [--format pdf] [--theme thesis]

# θ-derivative vs D1 weight gradient comparison
python scripts/analysis/theta_derivative_analysis.py [--format pdf]

# Extract θ trajectories from pipeline results
python scripts/analysis/extract_theta_trajectories.py

# Systematic claim validation (multi-section runner)
python scripts/experiment_runners/run_verification_plan.py [--list] [--noiseless-only]

# Full analysis pipeline
python analysis/run_analysis.py
```

### 12. Thesis Findings Validator (`python -m project_health.analysis.thesis_findings_validator`)

Corroborates ALL key thesis findings against raw data with statistical tests.

```bash
python -m project_health.analysis.thesis_findings_validator            # Full report
python -m project_health.analysis.thesis_findings_validator --verbose   # Per-finding details
python -m project_health.analysis.thesis_findings_validator --only scaling,zne  # Filter by category
python -m project_health.analysis.thesis_findings_validator --json report.json  # Machine-readable
python -m project_health.analysis.thesis_findings_validator --latex findings.tex # LaTeX table
make validate-findings                                                 # Quick run
make validate-findings-latex                                           # With LaTeX export
```

Categories: `scaling`, `zne`, `gnn`, `topology`, `global`.
Verdicts: `CORROBORATED` (p<0.01, strong effect), `QUALIFIED` (p<0.05 or caveats), `UNSUPPORTED`, `CONTRADICTED`.

**Programmatic:**
```python
from project_health.analysis.thesis_findings_validator import run_validation
report = run_validation(categories=["scaling", "zne"], verbose=True)
print(f"Corroboration rate: {report.overall_corroboration_rate:.0%}")
for f in report.findings:
    print(f"  {f.finding_id}: {f.verdict} ({f.strength})")
```

### 13. Thesis Tables Compiler (`python -m project_health.analysis.thesis_tables_compiler`)

Auto-generates global thesis tables from live data in Markdown and LaTeX.

```bash
python -m project_health.analysis.thesis_tables_compiler               # Print to stdout (Markdown)
python -m project_health.analysis.thesis_tables_compiler --verbose      # Show progress
python -m project_health.analysis.thesis_tables_compiler --markdown tables.md  # Save Markdown
python -m project_health.analysis.thesis_tables_compiler --latex tables/       # Save LaTeX (per-table + combined)
python -m project_health.analysis.thesis_tables_compiler --only T1,T3,T5       # Specific tables
python -m project_health.analysis.thesis_tables_compiler --json tables.json    # JSON export
make thesis-tables                                                     # Quick run (Markdown + LaTeX)
```

Tables: T1 (Global Performance), T2 (ZNE Comparison), T3 (Scaling Law), T4 (GNN-QEM),
T5 (Experiment Verdicts), T6 (Cross-Topology), T7 (Failure Modes), T8 (Hyperparameter Sensitivity),
T9 (MPS Performance), T10 (Timing Breakdown).

**Programmatic:**
```python
from project_health.analysis.thesis_tables_compiler import compile_tables
report = compile_tables(only=["T1", "T2"], verbose=True)
for table in report.tables:
    print(f"{table.table_id}: {table.title} ({len(table.rows)} rows)")
```

### 14. Thesis Global Figures (`python -m project_health.analysis.thesis_figures`)

Publication-ready global figures aggregating data across ALL experiments.

```bash
python -m project_health.analysis.thesis_figures                       # All figures (PDF)
python -m project_health.analysis.thesis_figures --list                 # List available
python -m project_health.analysis.thesis_figures --only global_de_gap_distribution  # Specific
python -m project_health.analysis.thesis_figures --format png --dpi 150 # PNG for slides
python -m project_health.analysis.thesis_figures --with-titles          # Include titles
python -m project_health.analysis.thesis_figures --output-dir figs/     # Custom dir
make thesis-figures                                                     # Quick run
```

Available figures: `global_de_gap_distribution`, `scaling_law_comprehensive`,
`topology_performance_violin`, `pea_vs_gf_comparison`, `gnn_qem_summary_panel`,
`experiment_verdicts_overview`, `pipeline_timing_stacked`, `cross_n_performance_heatmap`,
`findings_corroboration_summary`, `zne_gain_by_topology_and_strategy`.

### 15. Full Thesis Compilation (`make thesis-all`)

Runs ALL thesis compilation steps in order: validate → tables → figures.

```bash
make thesis-all   # Everything: findings validation + tables (md+tex) + all figures
```

Output:
- `documentation/thesis_tables/findings_report.json` — Corroboration report
- `documentation/thesis_tables/all_tables.md` — All tables in Markdown
- `documentation/thesis_tables/*.tex` — Per-table LaTeX + `all_tables.tex`
- `documentation/thesis_figures/*.pdf` — All global + registry figures

## Data Architecture

### Single Source of Truth: Valid Regime Boundaries

`P1_VALID_REGIME` and `P2_VALID_REGIME` defined ONLY in:
```
src/qmbp_simulation/framework/preflight.py
```
All consumers MUST import from there. Test `TestRegimeBoundaryConsistency` enforces via identity checks.

### NoiselessResult Key Fields

| Phase | Field | Source in JSON | Use |
|-------|-------|----------------|-----|
| 1 | `gap_min` | `diagnostics.phase1.gap_min` | Criticality indicator |
| 2 | `theta_smoothness` | `diagnostics.phase2.theta_smoothness` | Chain break detection |
| 2 | `convergence_rate` | `diagnostics.phase2.convergence_rate` | VQE health |
| 3 | `generalization_gap` | `diagnostics.phase3.generalization_gap` | MPNN overfit |
| 4 | `error_from_circuit` | `diagnostics.phase4.energy_decomposition.error_from_circuit` | Error attribution |
| 4 | `error_from_mpnn` | `diagnostics.phase4.energy_decomposition.error_from_mpnn` | Error attribution |

### NoisyResult: ZNE Strategy Detection

Auto-populated from:
1. `config.zne_strategy` or `config.amplifier` (if present)
2. Filename heuristic: "pea" → `"pea"`, "gf" → `"gate_folding"`, "ces" → `"ces"`

## Anti-Patterns (DO NOT)

- ❌ Writing `analysis/_tmp_*.py` throwaway scripts for one-off analysis
- ❌ Using `python -c "..."` for multi-line data inspection
- ❌ Creating new scripts when `digest/` or `analysis/` already covers it
- ❌ Manually parsing JSON result files (use scanner/diagnose)
- ❌ Duplicating formatting logic from `project_health/digest/formatters.py`
- ❌ Defining local valid-regime dicts (import from `preflight.py`)
- ❌ Running `cat results/...json | python -c "import json..."` — use digest instead

## When a New Script IS Justified

A new standalone script is appropriate ONLY when:
1. The analysis is fundamentally different from all existing tools
2. It will be reused multiple times (not a one-off)
3. It belongs to a new category not covered
4. It's registered in the experiment framework

Placement:
- Experiment scripts → `experiments/<category>/`
- Reusable analysis → extend `analysis/scan_coverage.py` or `analysis/diagnose.py`
- Result formatting → extend `project_health/digest/`

## Extension Guidelines

When extending an existing script:
- Add new flags/options rather than changing existing behavior
- Keep backward compatibility
- Add the new capability to `--help` output
- Update this steering file if a new major capability is added

### Extending Thesis Tools

When adding new findings to validate:
1. Add a `@register_finding(id, category, claim)` decorated function in `thesis_findings_validator.py`
2. Categories: `scaling`, `zne`, `gnn`, `topology`, `global`, `physics`
3. Return `FindingValidation` with verdict + evidence list
4. Load data from `_load_scan_results()` kwargs or use direct file reading for custom sources

When adding new thesis tables:
1. Add a `@register_table("T<N>")` decorated function in `thesis_tables_compiler.py`
2. Return `TableSpec(table_id, title, caption, columns, rows, notes)`
3. Data comes from `data` dict (keys: noiseless, noisy, experiments, scaling, cross_topo, gnn_qem)

When adding new thesis figures:
1. Add a `@register_thesis_figure(name, description)` decorated function in `thesis_figures.py`
2. Signature: `func(data: dict, cfg: FigureConfig) -> bool`
3. Save to `cfg.output_dir / f"fig_{name}.{cfg.fmt}"`

### GNN-QEM JSON Schema Reference

| File | Top-level keys | Metrics location |
|------|---------------|------------------|
| `cross_topology_results.json` | `zero_shot`, `fine_tuned`, `verdict` | `zero_shot.improvement_rate`, `.reduction_pct`, `.n_samples` |
| `ablation_no_enoisy_results.json` | `gnn_no_enoisy`, `mlp_no_enoisy`, `linear_no_enoisy`, `verdict` | `*.improvement_rate`, `*.n_total` |
| `post_zne_validation.json` | `summary`, `per_point` | `summary.n_gnn_regresses`, `.n_evaluations` |
| `vqe_realistic_results.json` | `data_stats`, `exp1_*`, `exp2_*`, `circuit_selection` | `circuit_selection.spearman_rho`, `.binary_accuracy_pct` |
| `affine_overshoot_audit.json` | `summary`, `records` | `summary.n_zne_records`, `summary.n_overshoot` |

### Zero-Shot Cross-N JSON Schema

```
results/scaling/zero_shot/zero_shot_v3_*.json
├── experiment, version, metadata
├── strategy_a_gnn_no_bn
│   ├── description, training_mse
│   └── results: [{h, e_pred, e_dmrg, gap, de_gap, theta_pred, passed}, ...]
├── strategy_b_interpolation
│   └── results: [{h, e_pred, e_dmrg, gap, de_gap, passed}, ...]
└── comparison
```

### ZNE Cross-Topology Definitive Schema

```
results/experiments/exp_zne_cross_topo/run_*.json
├── results
│   ├── section_1..3: {name, success, data: {pass, results, summary}}
│   └── section_4: {data: {comparison: [{topology,h,de_pea,de_gf,pea_gain,...}], summary: {paired_t_stat, paired_p_value, pea_wins_total, ...}}}
└── summary: {n_sections, n_passed, all_passed}
```

## Thesis Compilation Workflow (Full Pipeline)

### Quick Start
```bash
make thesis-all   # validate + tables + figures (full pipeline)
```

### Step-by-Step
```bash
# 1. Corroborate ALL findings against raw data
python -m project_health.analysis.thesis_findings_validator --verbose

# 2. Generate global tables (Markdown + LaTeX)
python -m project_health.analysis.thesis_tables_compiler --latex documentation/thesis_tables/

# 3. Generate global figures (PDF 300dpi)
python -m project_health.analysis.thesis_figures --output-dir documentation/thesis_figures/

# 4. Verify LaTeX thesis claims
python -m project_health.analysis.verify_claims

# 5. Sanity check (24 automated checks)
python -m project_health.analysis.sanity_check
```

### Finding → Thesis Table → Figure Mapping

| Finding ID | Thesis Section | Table | Figure |
|-----------|---------------|-------|--------|
| PIPELINE_UNIVERSALITY | 5.2 Cross-Topology | tab:cross_topo | topology_performance_violin |
| PEA_SUPERIORITY | 5.4 PEA-ZNE | tab:pea_zne | pea_vs_gf_comparison |
| SCALING_LAW | 5.3 Escalamiento | tab:scaling | scaling_law_comprehensive |
| GNN_QEM_CROSS_TOPO | 5.5 GNN-QEM | tab:gnn_qem_composability | gnn_qem_summary_panel |
| CROSS_N_ZERO_SHOT | 5.3.1 Cross-N | tab:cross_n | cross_n_performance_heatmap |
| TOPOLOGY_AGNOSTIC | 5.2 Cross-Topology | tab:cross_topo | topology_performance_violin |
| BATCHNORM_FINDING | 5.3.1 Cross-N | tab:cross_n | — |
| GNN_NOT_COMPOSABLE | 5.6.2 GNN-QEM | tab:gnn_qem_composability | gnn_qem_summary_panel |
| CX_BUDGET_RULE | 5.4.1 CX Budget | tab:cx_budget | zne_gain_heatmap |
| EXPERIMENT_SUCCESS_RATE | 5.8 Resumen | tab:experiment_summary | experiment_verdicts_overview |
| FAILURE_PREVENTION | 5.7 Pruebas Sistemáticas | tab:root_cause | — |
| S8_CRITICAL_EXPONENT | 5.6.3 S8/S8b | tab:s8_scaling | — |
| NOISE_AWARE_FAILS | 5.6.1 Noise-Aware | — (inline) | — |
| KITAEV_INCOMPATIBLE | 5.5.4 Kitaev | tab:kitaev | — |
| CROSS_TOPOLOGY_TRANSFER_FAILS | 5.2 (hallazgo) | — (inline) | — |
| PAULI_EVOLUTION_GATE | 5.6.5 Transpilación | tab:transpilation | — |
| DYPP_REDUNDANT | 5.6.4 DyPP | — (inline) | — |

### Registered Findings (for `thesis_findings_validator.py`)

Current: 22 findings validated and corroborated (21 CORROBORATED + 1 QUALIFIED). All implemented.

All findings are now registered as F16-F22 in `thesis_findings_validator.py`.
Deep audit: `PYTHONPATH=. python project_health/analysis/audit_findings.py` (23 checks, all VERIFIED).

### Verification Plan

Full plan documented in: `documentation/analysis/21_thesis_compilation_verification_plan.md`

Steps:
1. Run `make thesis-all` → expect 21/22 findings CORROBORATED + 1 QUALIFIED
2. Run `PYTHONPATH=. python project_health/analysis/audit_findings.py` → 23/23 VERIFIED
3. Verify 10/10 tables generated without errors
4. Verify 10/10 figures generated (PDF output)
4. Check `sanity_check` → 23/24+ pass
5. Check LaTeX bibliography balance: all \citep matched by \bibitem
6. Cross-reference: every thesis table number referenced in prose

### Adding New Findings to the Validator

```python
# In project_health/analysis/thesis_findings_validator.py:

@register_finding("S8_CRITICAL_EXPONENT", "physics",
    "Weight-gradient peak position shows no N-dependence (ν=5.0 → extraction fails)")
def _validate_s8_critical_exponent(**_) -> FindingValidation:
    # Load S8/S8b results from results/experiments/exp_s8/
    # Check: h_peak is constant across N=4,6,8,10
    # Check: ν_fit hits upper bound (5.0)
    # Verdict: CORROBORATED (the negative result IS the finding)
    ...
```
