# project_health/

Unified orchestration of analysis, diagnostics, figures, and reporting for the GNN-HVA pipeline results. Produces health reports, thesis artifacts, and coverage analysis across 476+ pipeline runs.

## Entry Points (python -m)

| Command | Purpose |
|---------|---------|
| `python -m project_health` | Full health report (text/json/markdown/compact) |
| `python -m project_health.digest` | Result scanner by kind (noiseless/noisy/experiment/scaling/cross_topology) |
| `python -m project_health.figures` | Figure generation (health + scaling figures, PNG/PDF) |
| `python -m project_health.analysis.sanity_check` | Physics + data integrity checks |
| `python -m project_health.analysis.scaling_analyzer` | MPS scaling law validation (N=40-200) |
| `python -m project_health.analysis.scaling_extensions_analyzer` | E5 bond-dim/HE/NLCE extensions |
| `python -m project_health.analysis.flow_warmstart_analyzer` | Flow warmstart σ_flow analysis |
| `python -m project_health.analysis.mitigation_benchmark_analyzer` | Mitigation benchmark (21 configs) |
| `python -m project_health.analysis.thesis_findings_validator` | Validate thesis claims vs raw data |
| `python -m project_health.analysis.thesis_tables_compiler` | Compile LaTeX/Markdown thesis tables |
| `python -m project_health.analysis.thesis_figures` | Thesis-level global figures (cross-experiment) |
| `python -m project_health.analysis.diagnose` | Automated failure root cause analysis |

## Package Structure

```
project_health/
├── __init__.py               Public API: HealthReport, ActionItem, CoverageGap, etc.
├── __main__.py               CLI: python -m project_health [--json|--markdown|--compact|--ci]
├── compare.py                Re-export alias → cli/compare.py
│
├── core/                     Health report engine
│   ├── engine.py             Orchestration: scan → analyze → report
│   ├── coverage.py           Coverage gap detection + VQE/MPNN/timing quality metrics
│   ├── models.py             Typed dataclasses (HealthReport, ActionItem, CoverageGap, Priority)
│   ├── reporter.py           Output formatters (text, JSON, Markdown)
│   └── state.py              Delta tracking (.project_health_state.json persistence)
│
├── digest/                   Result scanning and knowledge extraction
│   ├── __init__.py           Public API: ResultScanner, NoiselessResult, NoisyResult, etc.
│   ├── __main__.py           CLI: python -m project_health.digest [--kind X] [--sort Y]
│   ├── scanner.py            ResultScanner — parses all result JSON files
│   ├── models.py             Dataclasses for each result kind
│   └── formatters.py         Text/markdown/grouped/stats/outliers/compare formatters
│
├── analysis/                 Domain-organized analyzers (each is runnable via python -m)
│   ├── __init__.py           Package docstring listing subpackages
│   ├── diagnose.py           Standalone: failure root cause classifier (RootCause enum)
│   ├── statistical_tests.py  Shared: paired_ttest, improvement_rate, effect_size_cohens_d
│   │
│   ├── validation/           Verification & sanity checks
│   │   ├── sanity_check.py           Physics + data integrity (CheckResult, SanityReport)
│   │   ├── verify_results.py         Pipeline result verification (scan_results_directory, evaluate_criteria)
│   │   ├── validate_s_series.py      S-series entanglement validation (_compute_entropy)
│   │   ├── audit_findings.py         Structured audit finding records
│   │   ├── thesis_findings_validator.py  Validate thesis claims (run_validation, _VALIDATORS)
│   │   └── affine_overshoot_auditor.py   ZNE affine overshoot check (102 records, 0% overshoot)
│   │
│   ├── hardware/             Hardware & error mitigation analysis
│   │   ├── hw_rehearsal_analyzer.py         Hardware rehearsal sections analysis
│   │   ├── hw_results_analyzer.py           Hardware result envelope parsing
│   │   ├── mitigation_benchmark_analyzer.py MitigationBenchmarkAnalyzer (21 configs × 15 h-points)
│   │   ├── mitiq_analyzer.py                Mitiq integration health (get_mitiq_health_summary)
│   │   ├── layout_optimizer_analyzer.py     Mapomatic VF2 layout analysis (analyze())
│   │   └── transpilation_analyzer.py        Transpiled circuit properties audit
│   │
│   ├── scaling/              MPS scaling & extensions
│   │   ├── scaling_analyzer.py              N=40-200 scaling law validation
│   │   ├── scaling_extensions_analyzer.py   E5: bond-dim, HE ansatz, NLCE
│   │   └── flow_warmstart_analyzer.py       FlowWarmstartManager σ_flow analysis
│   │
│   ├── models/               GNN/MPNN/AQC model analyzers
│   │   ├── aqc_tensor_analyzer.py    AQC-Tensor compression health (get_aqc_health_summary)
│   │   ├── gnn_qem_analyzer.py       GNN-QEM error correction analysis
│   │   └── mpnn_eval_analyzer.py     MPNN evaluation suite (warmstart, LOO-CV, scaling)
│   │
│   ├── thesis/               Thesis compilation tools
│   │   ├── thesis_tables_compiler.py  All thesis tables (compile_tables, Markdown + LaTeX)
│   │   ├── thesis_figures.py          Cross-experiment aggregated figures (generate_all)
│   │   └── heisenberg_summary.py      V9 Heisenberg extension summary
│   │
│   ├── coverage/             Coverage scanning
│   │   └── scan_coverage.py  Standalone coverage gap scanner
│   │
│   └── [16 alias files]      Import aliases at root (e.g., analysis/sanity_check.py → analysis/validation/sanity_check.py)
│                              These are permanent re-exports used by Makefile and tests.
│
├── cli/                      Auxiliary CLI tools
│   ├── compare.py            Cross-experiment + ZNE technique comparison (--all/--noisy/--zne)
│   ├── inspect_results.py    Mitigation benchmark circuit audit (per-config, per-h)
│   └── qpu_time_estimator.py QPU time estimation per circuit (IBM Kingston CLOPS model)
│
├── figures/                  Figure generation
│   ├── __init__.py
│   ├── health_figures.py     23 diagnostic figures (FigureConfig, registry, generate_figures)
│   └── scaling_figures.py    MPS scaling-specific figures
│
└── _deprecated/              Dead code (can be deleted — all have confirmed replacements)
    ├── verify_claims.py           → replaced by thesis_findings_validator
    ├── hw_results_checker.py      → replaced by hw_results_analyzer
    └── boundary_100_analyzer.py   → one-off script, results in scaling_analyzer
```

## Import Aliases (analysis/ root)

All modules in `analysis/` subpackages have a 1-line alias at the `analysis/` root for convenience:

```python
# Both paths work identically:
from project_health.analysis.sanity_check import CheckResult, SanityReport
from project_health.analysis.validation.sanity_check import CheckResult, SanityReport

# Canonical (subpackage) path preferred for new code.
# Alias (root) path used by Makefile targets and existing tests.
```

Full alias mapping:
- `analysis/sanity_check.py` → `analysis/validation/sanity_check.py`
- `analysis/verify_results.py` → `analysis/validation/verify_results.py`
- `analysis/validate_s_series.py` → `analysis/validation/validate_s_series.py`
- `analysis/audit_findings.py` → `analysis/validation/audit_findings.py`
- `analysis/thesis_findings_validator.py` → `analysis/validation/thesis_findings_validator.py`
- `analysis/affine_overshoot_auditor.py` → `analysis/validation/affine_overshoot_auditor.py`
- `analysis/hw_rehearsal_analyzer.py` → `analysis/hardware/hw_rehearsal_analyzer.py`
- `analysis/mitigation_benchmark_analyzer.py` → `analysis/hardware/mitigation_benchmark_analyzer.py`
- `analysis/mitiq_analyzer.py` → `analysis/hardware/mitiq_analyzer.py`
- `analysis/layout_optimizer_analyzer.py` → `analysis/hardware/layout_optimizer_analyzer.py`
- `analysis/transpilation_analyzer.py` → `analysis/hardware/transpilation_analyzer.py`
- `analysis/scaling_analyzer.py` → `analysis/scaling/scaling_analyzer.py`
- `analysis/scaling_extensions_analyzer.py` → `analysis/scaling/scaling_extensions_analyzer.py`
- `analysis/flow_warmstart_analyzer.py` → `analysis/scaling/flow_warmstart_analyzer.py`
- `analysis/mpnn_eval_analyzer.py` → `analysis/models/mpnn_eval_analyzer.py`
- `analysis/gnn_qem_analyzer.py` → `analysis/models/gnn_qem_analyzer.py`
- `analysis/aqc_tensor_analyzer.py` → `analysis/models/aqc_tensor_analyzer.py`
- `analysis/thesis_figures.py` → `analysis/thesis/thesis_figures.py`
- `analysis/thesis_tables_compiler.py` → `analysis/thesis/thesis_tables_compiler.py`
- `analysis/heisenberg_summary.py` → `analysis/thesis/heisenberg_summary.py`
- `analysis/scan_coverage.py` → `analysis/coverage/scan_coverage.py`

## Makefile Targets

```bash
make health              # python -m project_health --compact
make health-full         # python -m project_health --markdown -o reports/
make sanity              # python -m project_health.analysis.sanity_check
make scaling             # python -m project_health.analysis.scaling_analyzer
make extensions          # python -m project_health.analysis.scaling_extensions_analyzer --verbose --cross-check
make cross-topology      # python -m project_health.digest --kind cross_topology
make figures             # python -m project_health.figures --source both
make figures-thesis      # PDF 300dpi + thesis_figures + copy to tesis-figures/
make validate-findings   # python -m project_health.analysis.thesis_findings_validator --verbose
make mitigation-analyze  # python -m project_health.analysis.mitigation_benchmark_analyzer --thesis-table --figures
make thesis-tables       # python -m project_health.analysis.thesis_tables_compiler --verbose --markdown/--latex
make thesis-all          # validate-findings + tables + figures (full thesis compilation)
make hw-flow-analyze     # python -m project_health.analysis.flow_warmstart_analyzer --verbose
```

## Programmatic API

```python
# Health report
from project_health import HealthReport, ActionItem, CoverageGap
from project_health.core.engine import run_health_check
report = run_health_check(results_dir=Path("results"))

# Result scanning
from project_health.digest import ResultScanner, NoiselessResult, NoisyResult
scanner = ResultScanner(Path("results"))
noiseless, noisy, experiments = scanner.scan_all()
scaling = scanner.scan_scaling()

# Statistical tests
from project_health.analysis.statistical_tests import paired_ttest, improvement_rate, effect_size_cohens_d
result = paired_ttest(before=[0.15, 0.12], after=[0.02, 0.01])

# Failure diagnosis
from project_health.analysis.diagnose import scan_all_thesis, classify_root_causes

# Coverage analysis
from project_health.core.coverage import detect_coverage_gaps, derive_actions
```

## Key Dependencies

- `core/` and `digest/` — lightweight, no heavy deps (stdlib + project's own `qmbp_simulation.framework`)
- `analysis/` modules — may import `numpy`, `qmbp_simulation.framework.preflight`
- `figures/` — requires `matplotlib`, `numpy`
- `cli/inspect_results.py`, `cli/qpu_time_estimator.py` — require `numpy`
