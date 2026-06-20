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
├── Inspect MPS scaling results?   → python -m project_health.digest --kind scaling
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
├── Deep raw-data audit (29 checks)?→ PYTHONPATH=. python project_health/analysis/audit_findings.py
├── Analyze MPNN eval suite (S10-19)?→ python -m project_health.analysis.mpnn_eval_analyzer
├── Analyze flow warmstart results? → python -m project_health.analysis.flow_warmstart_analyzer
├── Analyze AQC-Tensor compression?→ python -m project_health.analysis.aqc_tensor_analyzer
├── Analyze layout optimizer (VF2)?→ python -m project_health.analysis.layout_optimizer_analyzer
├── Analyze Mitiq comparisons?     → python -m project_health.analysis.mitiq_analyzer
├── Analyze mitigation benchmark?  → python -m project_health.analysis.mitigation_benchmark_analyzer
├── Analyze transpilation metrics? → python scripts/analyze_transpilation.py
├── Run full flow→deployment?       → make hw-flow-full
│
│ ─── THESIS COMPILATION ─────────────────
├── Corroborate ALL findings?       → python -m project_health.analysis.thesis_findings_validator
├── Generate thesis tables?         → python -m project_health.analysis.thesis_tables_compiler
├── Generate thesis global figures? → python -m project_health.analysis.thesis_figures
├── Full thesis compilation?        → make thesis-all
├── Validate + LaTeX export?        → make validate-findings-latex
└── Something tools don't cover?    → Extend the closest existing tool
```

## Tool Reference (complete flags)

### 1. Project Health Report (`python -m project_health`)

```bash
python -m project_health                    # Full text report
python -m project_health --compact          # Summary only
python -m project_health --json             # Machine-parseable
python -m project_health --markdown         # For documentation
python -m project_health -o reports/        # Auto-timestamped save
python -m project_health --diff-only        # Only show changes since last run
python -m project_health --ci              # Exit 1 on CRITICAL gaps
```

**Programmatic:**
```python
from project_health.engine import run_health_check
from project_health.models import Priority
report = run_health_check()
critical = [a for a in report.actions if a.priority == Priority.CRITICAL]
```

### 2. Result Digest (`python -m project_health.digest`)

```bash
# By kind
python -m project_health.digest --kind noiseless|noisy|experiment|cross_topology|scaling

# Filters
python -m project_health.digest --kind noiseless --topology ladder --n-qubits 10 --p-layers 1
python -m project_health.digest --kind noiseless --folder variants_N10_ladder

# Sorting + limiting
python -m project_health.digest --kind noiseless --sort delta_e --top 10

# Grouped comparisons
python -m project_health.digest --kind noiseless --group-by topology

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
```

### 3. Coverage Scanner (`analysis/scan_coverage.py`)

```bash
python analysis/scan_coverage.py --discover           # What data exists
python analysis/scan_coverage.py --extended           # Full analytics
python analysis/scan_coverage.py --topology chain_1d  # Filter by topology
```

### 4. Failure Diagnosis (`analysis/diagnose.py`)

```bash
python analysis/diagnose.py --all                     # All results
python analysis/diagnose.py results/experiments/exp_B4/  # Specific folder
python analysis/diagnose.py --severity fail           # Only failures
```

### 5. Experiment Comparison (`scripts/compare.py`)

```bash
python scripts/compare.py --all                       # All experiments
python scripts/compare.py --category optimization     # By category
python scripts/compare.py --noisy                     # ZNE experiments
```

### 6. ZNE Method Comparison (`project_health/compare.py`)

```bash
python project_health/compare.py --zne                # Full comparison
python project_health/compare.py --zne --json out.json  # Machine-readable
```

### 7. Sanity Check

```bash
python -m project_health.analysis.sanity_check         # All 24 checks
python -m project_health.analysis.sanity_check --only physics  # Physics subset
python -m project_health.analysis.sanity_check --json out.json
```

### 8-9. Scaling Analyzers

```bash
python -m project_health.analysis.scaling_analyzer                     # MPS scaling law
python -m project_health.analysis.scaling_extensions_analyzer          # E5 extensions
python -m project_health.analysis.scaling_extensions_analyzer --thesis-tables  # Tables 5.25/5.26
python -m project_health.analysis.scaling_extensions_analyzer --json report.json
```

### 10. Cross-Topology Transfer

```bash
python -m project_health.digest --kind cross_topology           # Full report
make cross-topology                                             # Quick report
```

### 11. Preflight Validation

```bash
python scripts/preflight.py --from-script <path>       # Standard
python scripts/preflight.py --from-script <path> --strict  # Warnings=errors (CI)
make preflight SCRIPT=<path>
```

### 12. Figures

```bash
make figures            # PNG, all figures
make figures-thesis     # PDF 300dpi, thesis-ready
```

### 13. Specialized Analysis Scripts

```bash
python scripts/analysis/theta_pca_phase_detection.py [--format pdf] [--theme thesis]
python scripts/analysis/theta_derivative_analysis.py [--format pdf]
python scripts/analysis/extract_theta_trajectories.py [--only-scaling]
python scripts/experiment_runners/run_verification_plan.py [--list] [--noiseless-only]
python analysis/run_analysis.py
```

### 14. Thesis Findings Validator

```bash
python -m project_health.analysis.thesis_findings_validator --verbose
python -m project_health.analysis.thesis_findings_validator --only scaling,zne
python -m project_health.analysis.thesis_findings_validator --json report.json
python -m project_health.analysis.thesis_findings_validator --latex findings.tex
make validate-findings-latex
```

### 15. Thesis Tables Compiler

```bash
python -m project_health.analysis.thesis_tables_compiler --latex tables/
python -m project_health.analysis.thesis_tables_compiler --only T1,T3,T5
python -m project_health.analysis.thesis_tables_compiler --json tables.json
make thesis-tables
```

Tables: T1-T10 (Global Performance, ZNE, Scaling, GNN-QEM, Verdicts, Cross-Topology,
Failure Modes, Hyperparameters, MPS Performance, Timing).

### 16. Thesis Global Figures

```bash
python -m project_health.analysis.thesis_figures                       # All (PDF)
python -m project_health.analysis.thesis_figures --list                 # List available
python -m project_health.analysis.thesis_figures --only global_de_gap_distribution
python -m project_health.analysis.thesis_figures --format png --dpi 150
make thesis-figures
```

### 17. Full Thesis Compilation

```bash
make thesis-all   # validate + tables + figures
```

### 18. Flow Warmstart Analyzer

```bash
python -m project_health.analysis.flow_warmstart_analyzer [--verbose] [--json out.json]
```

## Data Architecture & JSON Schemas

For all JSON schemas, key fields, and validation rules, see:
#[[file:.kiro/knowledge/result-schemas.md]]

### Single Source of Truth: Valid Regime Boundaries

`P1_VALID_REGIME` and `P2_VALID_REGIME` defined ONLY in:
```
src/qmbp_simulation/framework/preflight.py
```
All consumers MUST import from there. Test `TestRegimeBoundaryConsistency` enforces via identity checks.

### Scaling Law Formula (Two Regimes)

| Regime | Formula | Valid range | Used by |
|--------|---------|------------|---------|
| Exact diag | `h_min = 1.0 + 0.020*N^1.31` | N=4-20 | `exp_a3_scaling_law.py` |
| MPS (corrected) | `h_min = 1.5 + 0.020*N^1.31` | N=40-120 | Runner scripts, digest |

**Rule**: N>30 MUST use the corrected formula (`1.5 + ...`). The +0.50 offset is validated at N=40/50/80/120.

## Anti-Patterns (DO NOT)

- Writing `analysis/_tmp_*.py` throwaway scripts for one-off analysis
- Using `python -c "..."` for multi-line data inspection
- Creating new scripts when `digest/` or `analysis/` already covers it
- Manually parsing JSON result files (use scanner/diagnose)
- Duplicating formatting logic from `project_health/digest/formatters.py`
- Defining local valid-regime dicts (import from `preflight.py`)
- Running `cat results/...json | python -c "import json..."` -- use digest instead

## When a New Script IS Justified

Only when:
1. The analysis is fundamentally different from all existing tools
2. It will be reused multiple times (not a one-off)
3. It belongs to a new category not covered
4. It is registered in the experiment framework

Placement:
- Experiment scripts: `experiments/<category>/`
- Reusable analysis: extend `analysis/scan_coverage.py` or `analysis/diagnose.py`
- Result formatting: extend `project_health/digest/`

## Extension Guidelines

When extending an existing script:
- Add new flags/options rather than changing existing behavior
- Keep backward compatibility
- Add the new capability to `--help` output
- Update this steering file if a new major capability is added

### Extending Thesis Tools

When adding new findings:
1. `@register_finding(id, category, claim)` in `thesis_findings_validator.py`
2. Return `FindingValidation` with verdict + evidence list

When adding new tables:
1. `@register_table("T<N>")` in `thesis_tables_compiler.py`
2. Return `TableSpec(table_id, title, caption, columns, rows, notes)`

When adding new figures:
1. `@register_thesis_figure(name, desc)` in `thesis_figures.py`
2. Signature: `func(data: dict, cfg: FigureConfig) -> bool`
3. Save to `cfg.output_dir / f"fig_{name}.{cfg.fmt}"`

## Thesis Compilation Workflow

```bash
make thesis-all   # Full pipeline: validate -> tables -> figures
```

Step-by-step:
```bash
python -m project_health.analysis.thesis_findings_validator --verbose
python -m project_health.analysis.thesis_tables_compiler --latex documentation/thesis_tables/
python -m project_health.analysis.thesis_figures --output-dir documentation/thesis_figures/
python -m project_health.analysis.verify_claims
python -m project_health.analysis.sanity_check
```

Status: 23 findings (22 CORROBORATED + 1 QUALIFIED), 10 tables, 18 figures.
Verification plan: `documentation/analysis/21_thesis_compilation_verification_plan.md`
