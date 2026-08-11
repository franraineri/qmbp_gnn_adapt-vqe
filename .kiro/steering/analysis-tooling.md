---
inclusion: fileMatch
fileMatchPattern: "analysis/**,scripts/**,experiments/**,results/**,project_health/**"
---

# Analysis Tooling — ALWAYS Use Existing Scripts

## Rule (MANDATORY)

When analyzing data, inspecting results, or producing output files:

**ALWAYS use the existing analysis and digest scripts first.**
Do NOT write ad-hoc one-off scripts or inline Python when an existing tool covers the need
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
├── Diagnose by group (quick)?     → python -m project_health --diagnose [--model X]
├── Detect regressions?            → python project_health/cli/query_index.py --regressions
├── Temporal drift (date-correlated)?→ python project_health/cli/query_index.py --temporal-drift
├── Compare experiments?            → project_health/compare.py [--all] [--category X]
├── Compare ZNE methods?            → project_health/compare.py --zne
├── Validate runner script?         → python src/qmbp_simulation/framework/preflight.py --from-script <path>
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
├── Deep raw-data audit (29 checks)?→ PYTHONPATH=. python project_health/analysis/validation/audit_findings.py
├── Analyze MPNN eval suite (S10-19)?→ python -m project_health.analysis.mpnn_eval_analyzer
├── Analyze flow warmstart results? → python -m project_health.analysis.flow_warmstart_analyzer
├── Analyze AQC-Tensor compression?→ python -m project_health.analysis.aqc_tensor_analyzer
├── Analyze layout optimizer (VF2)?→ python -m project_health.analysis.layout_optimizer_analyzer
├── Analyze Mitiq comparisons?     → python -m project_health.analysis.mitiq_analyzer
├── Analyze mitigation benchmark?  → python -m project_health.analysis.mitigation_benchmark_analyzer
├── Analyze noiseless pipeline?   → python -m project_health.analysis.noiseless_pipeline_analyzer
├── Validate hw run post-execution?→ .venv/bin/python scripts/verify_affine_bug.py --validate <run_dir>
├── Quick post-exec check (1 file)?→ python -m project_health.analysis.hardware.post_execution_validator <path>
├── Batch validate all hw results? → python -m project_health.analysis.hardware.post_execution_validator results/hardware/ --batch
├── Check pipeline correction bugs?→ .venv/bin/python scripts/verify_affine_bug.py
├── Ready for IBM hardware?         → .venv/bin/python scripts/hardware/preflight_hw.py
├── Audit code-path consistency?  → .venv/bin/python scripts/verify_affine_bug.py --audit
├── Run full flow→deployment?       → make hw-flow-full
│
│ ─── NEW METRICS & DIAGNOSTICS (2026-07-13) ──────
├── Inspect simulation backend used? → check simulation_diagnostics in JSON (auto-present)
├── Filter runs by backend type?    → python project_health/cli/query_index.py --backend mps
├── Variational violations summary? → python scripts/maintenance/scan_new_runs.py --verbose (shows ⚠️viol=N)
├── Per-point violation detail?     → python project_health/cli/inspect_noiseless_run.py --latest <dir> --vqe-detail
├── Chi-convergence (scaling 2D)?   → run_scaling_validation.py --verify-chi (MANDATORY for 2D N>16)
├── Verify thesis run integrity?    → python scripts/verify_thesis_runs.py (12 checks + chi + violations)
├── Cross-check E_exact consistency?→ scripts/verify_thesis_runs.py (Check 8b: S1 vs S2 mismatch)
├── Error budget decomposition?     → see .kiro/steering/error-budget-decomposition.md
├── Energy variance per-h analysis? → inspect_noiseless_run.py --latest <dir> --vqe-detail
├── Fragile passes (Var>0.5+pass)?  → scripts/verify_thesis_runs.py (auto-checks)
├── Go/No-Go for hardware?          → energy_variance(θ_pred) < 0.2 at all h_test
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

### 5. Experiment Comparison (`project_health/compare.py`)

```bash
python project_health/compare.py --all                       # All experiments
python project_health/compare.py --category optimization     # By category
python project_health/compare.py --noisy                     # ZNE experiments
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
python -m project_health.analysis.scaling_analyzer                     # MPS frontier analysis
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
python src/qmbp_simulation/framework/preflight.py --from-script <path>       # Standard
python src/qmbp_simulation/framework/preflight.py --from-script <path> --strict  # Warnings=errors (CI)
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

### 19. Pipeline Correction Verifier & Post-Execution Validator (`scripts/verify_affine_bug.py`)

Unified script for verifying the energy correction pipeline (affine clipping, ZNE extrapolation, bounds consistency) and validating individual hardware run results post-execution.

**Three modes:**

```bash
# Mode 1: Quick invariant check (Parts 1-4, ~instant, for CI/hooks)
.venv/bin/python scripts/verify_affine_bug.py --quick

# Mode 2: Full pipeline verification (Parts 1-9, ~2s, after code changes)
.venv/bin/python scripts/verify_affine_bug.py

# Mode 3: Post-execution validation of a specific run (after every QPU execution)
.venv/bin/python scripts/verify_affine_bug.py --validate results/hardware/run_XXXXXXXX_XXXXXX
.venv/bin/python scripts/verify_affine_bug.py --validate results/mitigation_benchmark/...
```

**Parts covered:**

| Part | What | When to run |
|------|------|-------------|
| 1 | Bug reproduction (old formula) | After code changes |
| 2 | Fixed affine_correct_energy() | After code changes |
| 3 | Hardware runs audit (all 18+) | After code changes |
| 4 | Monotonicity (never worsens) | After code changes |
| 5 | Bounds consistency (3 impls) | After code changes |
| 6 | ZNE extrapolation sanity | After code changes |
| 7 | H8 invariant (never amplifies) | After code changes |
| 8 | Edge-case detection (float64) | After code changes |
| 9 | Benchmark regression scan | After code changes |
| 10 | **Post-execution validation** | **After every QPU run** |
| 11 | **Circuit metrics & QPU time** | **After every QPU run** |

**Part 10 checks (per-run, via --validate):**
- Energy finiteness and physics bounds (via `VQEValidator.compute_energy_bounds`)
- Affine correction consistency (stale data detection)
- Observable dimensions and Pauli bounds (|⟨O⟩| ≤ 1)
- Energy-observable cross-validation (TFIM H reconstruction)
- ZNE R² quality gate (≥ 0.80)
- Variational principle (e_zne vs e_exact)
- Measurement SNR (via `compute_snr`)
- Phase classification confidence (via `compute_classification_confidence`)
- Verdict consistency (via `classify_de_gap`)

**Part 11 checks (circuit & timing, via --validate):**
- CES spread ratio (>0.3 = CES-ZNE viable, <0.3 = PEA required)
- Transpiled circuit depth, depth_2q, n_2q_gates, fidelity estimate
- ZNE CX threshold viability (via `preflight._ZNE_CX_THRESHOLD_GF/PEA`)
- QPU time estimate vs actual (depth-aware CLOPS model)
- Routing overhead (transpiled_vs_logical_ratio)

**Reuses (imports, no duplication):**
- `qmbp_simulation.analysis.VQEValidator` — physics bounds
- `qmbp_simulation.analysis.metrics.compute_snr` — SNR
- `qmbp_simulation.analysis.metrics.compute_classification_confidence` — phase confidence
- `qmbp_simulation.execution.affine_correct_energy` — canonical correction
- `qmbp_simulation.execution.hardware.preflight` — ZNE CX thresholds (18 GF / 50 PEA)
- `qmbp_simulation.framework.criteria.compute_verdict` — verdict evaluation
- `project_health.analysis.validation.verify_results.classify_de_gap` — threshold classification
- QPU time constants from `project_health.cli.qpu_time_estimator` — CLOPS model

**Exit codes:** 0 = all checks pass, N>0 = N violations found.

### 20. Post-Execution Validator (standalone module)

Lightweight library module with 12 automated checks. Callable from CLI, runners, or hooks.

```bash
# Single file/directory
python -m project_health.analysis.hardware.post_execution_validator <path>

# JSON output (machine-readable, for CI)
python -m project_health.analysis.hardware.post_execution_validator <path> --json

# Batch mode (all results in a directory)
python -m project_health.analysis.hardware.post_execution_validator results/hardware/ --batch
```

**Checks (C1–C12):**

| ID | Check | Severity |
|----|-------|----------|
| C1 | QPU time estimate vs actual (CLOPS model) | WARNING/INFO |
| C2 | Fidelity estimate vs ΔE/gap (optimistic prediction) | WARNING/INFO |
| C3 | Error budget vs ΔE/gap correlation | WARNING/INFO |
| C4 | Observable bounds (\|⟨O⟩\| ≤ 1) | ERROR |
| C5 | Energy-observable cross-validation (TFIM) | WARNING |
| C6 | Variational principle (E ≥ E_exact) | WARNING/INFO |
| C7 | ZNE R² quality ([0,1], ≥ 0.80) | ERROR/WARNING |
| C8 | Verdict consistency (stored vs recomputed) | WARNING |
| C9 | Stale affine correction (2026-06-22 bug) | WARNING |
| C10 | Phase label vs ⟨X⟩ consistency | WARNING |
| C11 | Circuit depth vs ZNE viability (18/50 CX) | IMPROVEMENT/WARNING |
| C12 | Shot noise floor / SNR sufficiency | INFO/IMPROVEMENT |

**Programmatic API:**
```python
from project_health.analysis.hardware.post_execution_validator import (
    validate_run, validate_envelope, validate_hardware_summary, print_report,
)
report = validate_run(Path("results/hardware/run_20260617_141440/"))
print_report(report)        # Human-readable
report.to_dict()            # JSON-serializable
report.passed               # True if 0 ERRORs
```

**Relationship to verify_affine_bug.py:**
- `verify_affine_bug.py --validate` calls this module as Part 11 (after its own 10 checks)
- The standalone module is useful for quick validation, batch operations, or JSON export without the full 900-line regression suite

## Data Architecture & JSON Schemas

For all JSON schemas, key fields, and validation rules, see:
#[[file:.kiro/knowledge/result-schemas.md]]

### Single Source of Truth: Valid Regime Boundaries

`P1_VALID_REGIME` and `P2_VALID_REGIME` defined ONLY in:
```
src/qmbp_simulation/framework/preflight.py
```
All consumers MUST import from there. Test `TestRegimeBoundaryConsistency` enforces via identity checks.

### Frontier Fits (Canonical)

| p | Formula | Valid range | Source |
|---|---------|------------|--------|
| 1 | `h_min = 2.36 + 0.0073*N` | N=20-250 (R²=0.91) | H_EXPR_MATRIX |
| 2 | `h_min = 1.57 + 0.005*N` | N=20-120 (R²=0.95) | H_EXPR_MATRIX |
| ≥3 | `h_min ≈ 1.6 ± 0.1` (constant) | N=20-120 | H_EXPR_MATRIX |
| ≥4 | `h_min ≈ 1.4 ± 0.1` (constant) | N=20-80 | H_EXPR_MATRIX |

**DO NOT use** the old power-law `h_min = 1.5 + 0.020·N^1.31` — it overestimates by 1.9× at N=60, 2.7× at N=100. Some scripts still use it as a conservative h-grid estimator; this is safe (always overestimates → never underestimates valid regime) but not accurate.

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
