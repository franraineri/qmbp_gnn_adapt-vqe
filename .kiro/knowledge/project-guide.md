# Project Guide — GNN-HVA Framework

> This is the entry point for AI agents working on this project. Read this first.

## What This Project Is

A Master's thesis implementing a hybrid classical-quantum pipeline for characterizing
topological phases of matter. The pipeline uses a Graph Neural Network (MPNN) to predict
optimal parameters for a shallow quantum circuit (HVA), which is then deployed on IBM
quantum hardware to classify quantum phases via local observable measurements.

## Repository Map

```
├── src/qmbp_simulation/     ← INSTALLABLE PACKAGE (the framework)
│   ├── __init__.py          ← Package-level re-exports
│   ├── utils/               ← Seed, JSON, timing (no internal deps)
│   ├── models/              ← LatticeConfig, Hamiltonians, data models, constants
│   ├── solvers/             ← Exact diag + DMRG ground truth
│   ├── circuits/            ← HVA circuit construction (p≤2 enforced)
│   ├── execution/           ← Backend ABC + noiseless/noisy/hardware + ZNE utils
│   ├── optimizers/          ← Multi-start VQE + SPSA with warm-start
│   ├── predictors/          ← MPNN model, training, checkpoints
│   ├── pipeline/            ← Dataset save/load, pipeline orchestration, helpers
│   ├── framework/           ← Experiment engine, CLI, benchmarking, result I/O, logging
│   └── analysis/            ← Gradient analysis, diagnostics, landscape, metrics
│
├── experiments/              ← EXPERIMENT SCRIPTS (consumers of the package)
│   ├── helpers/             ← Reusable technique modules (dypp, freezing, etc.)
│   ├── optimization/        ← B1, B2, B4, C3, G4
│   ├── scaling/             ← A3, G3
│   ├── landscape/           ← F1, F3
│   ├── predictor/           ← C1, D1, E3, G1, G2, G5
│   ├── hardware/            ← Hardware-specific experiments
│   └── generalization/      ← E4
│
├── scripts/                 ← CLI ENTRY POINTS (thin wrappers over framework)
│   ├── run_experiment.py    ← Unified CLI for running experiments by ID
│   ├── run_pipeline.py      ← Full 4-phase pipeline CLI
│   ├── compare.py           ← Cross-experiment result comparison
│   ├── smoke_test.py        ← Package smoke test (N=4, p=1, <30s)
│   └── benchmark.py         ← Performance benchmarking
│
├── tests/                   ← Pytest test suite
│   ├── conftest.py          ← Shared fixtures
│   ├── unit/                ← Per-module unit + property tests
│   └── integration/         ← Smoke, pipeline e2e, backward compat
│
├── results/                 ← Experiment outputs (gitignored)
│   ├── experiments/         ← Auto-generated JSON results
│   ├── benchmarks/          ← Benchmark results
│   └── thesis/              ← Committed definitive results
│
├── documentation/           ← Human-readable docs
│   ├── binnacles/           ← Experiment logs
│   └── bibliography/        ← Literature references
│
├── .kiro/                   ← AI agent configuration
│   ├── skills/quantum/SKILL.md    ← Core rules and constraints
│   ├── knowledge/                 ← Domain knowledge and recipes
│   ├── steering/                  ← Active guidance (status, code style, protocols)
│   └── hooks/                     ← Automated checks
│
├── Makefile                 ← SINGLE ENTRY POINT for all operations
└── pyproject.toml           ← Build config, ruff, pytest
```

## How to Run Things

| Command | What it does | Time |
|---------|-------------|------|
| `make help` | Show all targets | instant |
| `make test` | Fast tests (`-m "not slow"`) | ~12s |
| `make test-full` | All tests including slow | ~60s |
| `make lint` | Ruff linter | ~1s |
| `make typecheck` | Mypy type checking (full) | ~5s |
| `make coverage` | Tests with coverage report | ~15s |
| `make health` | Project health report (compact) | ~5s |
| `make figures` | Generate all analysis figures | ~10s |
| `make check-full` | lint + test + smoke-test | ~15s |
| `make preflight SCRIPT=<path>` | Validate variant script before running | ~3s |
| `python tests/smoke_test.py` | Package smoke test (N=4, p=1) | ~30s |
| `python src/qmbp_simulation/framework/preflight.py --from-script <path>` | Preflight validation | ~3s |
| `python project_health/compare.py --all` | Compare all results | ~5s |
| `python scripts/benchmarks/benchmark.py` | Performance benchmarks | ~30s |

### Installation

```bash
pip install -e ".[dev,test]"        # Install with dev + test extras
python -c "import qmbp_simulation" # Verify import works
python tests/smoke_test.py         # Verify pipeline works
make check-full                    # Full quality gate
```

## Framework Module Reference

The `framework/` subpackage provides reusable infrastructure for scripts and experiments:

### `framework/cli.py` — Shared CLI Argument Groups

Eliminates argparse boilerplate across scripts. Provides:
- `create_base_parser(description, epilog)` — Standardized parser with RawDescriptionHelpFormatter
- `add_system_args(parser)` — Adds --n-qubits, --topology, --J, --periodic, --p
- `add_sweep_args(parser)` — Adds --h-values, --h-test
- `add_vqe_args(parser)` — Adds --n-restarts, --maxiter, --sigma, --seed
- `add_mpnn_args(parser)` — Adds --hidden-dim, --n-layers, --n-epochs, --lr, --patience
- `add_output_args(parser)` — Adds --output-dir, --verbose, --debug
- `add_result_filter_args(parser)` — Adds --topology, --n-qubits, --p-layers, --model, --folder
- `add_format_args(parser)` — Adds --markdown, --json, --output, --sort, --top, --group-by
- `add_variant_runner_args(parser)` — Adds --dry-run, --variant, --start-from, --list
- `validate_descending_sweep(h_values)` — Validates/normalizes h-values to descending order
- `validate_system_size(n_qubits, p_layers)` — Checks constraints (p≤2, N=12 warning)
- `configure_logging(verbose, debug)` — Sets up logging level
- `build_mpnn_config_dict(args)` — Extracts MPNN config from parsed args
- `resolve_output_dir(path)` — Creates and returns output directory Path

### `framework/result_io.py` — Standardized Result Saving

- `build_result_envelope(config, results, summary, elapsed_s)` — Standard JSON structure
- `save_experiment_result(data, experiment_id)` — Saves to `results/experiments/exp_{id}/run_{ts}.json`
- `save_pipeline_result(data, output_dir)` — Saves to `{dir}/pipeline_run_{ts}.json`
- `save_benchmark_result(data, output_path)` — Saves benchmark results
- `load_result(path)` — Loads any JSON result file
- `generate_timestamp()` — Returns YYYYMMDD_HHMMSS string

### `framework/result_store.py` — Result Querying & Comparison

- `ResultStore` class with methods:
  - `list_experiments()` — Discover available experiment IDs
  - `resolve_category(category)` — Map category name/letter to experiment IDs
  - `load_latest(experiment_id)` — Load most recent result
  - `compare_experiments(exp_ids)` — Evaluate against per-experiment criteria
  - `load_noisy_results()` — Load ZNE experiment results
  - `analyze_noisy_correlations(results)` — Compute R², gain stats
  - `analyze_noisy_by_group(results, key)` — Group-by analysis
  - `format_experiment_table(comparisons)` — Text table output
- `CATEGORY_MAP` — Dict mapping category names to experiment ID prefixes

### `framework/criteria.py` — Experiment Success Criteria (single source)

Single source of truth for experiment evaluation. Never duplicate elsewhere.
- `EXPERIMENT_CRITERIA` — Dict mapping exp_id → {metric, threshold, desc}
- `REJECTION_IS_FINDING` — Set of exp_ids where rejection = valid finding
- `compute_verdict(exp_id, summary)` — Returns (verdict, criteria_desc)
- `Verdict` — Type alias: `Literal["confirmed", "rejected", "failed"]`

### `framework/benchmarking.py` — Performance Regression Suite

- `BenchmarkSuite(n_qubits, n_repeats)` — Configurable benchmark runner
  - `.run(components)` — Run benchmarks for solver/circuit/vqe/mpnn
  - `.print_summary(results)` — Formatted table output
  - `.to_dict(results)` — JSON-serializable output
- `BenchmarkResult` — Dataclass with component, n_qubits, elapsed_s, details

### `framework/logging.py` — Structured Events + Progress

- `StructuredLogger(experiment_id)` — Machine-parseable event logging
  - `.log(event_type, seed, h_value, data)` — Record event
  - `.start_timer(label)` / `.stop_timer(label)` — Timing
  - `.save(path)` — Persist to JSON
- `ProgressReporter(title)` — Console progress reporting
  - `.phase(num, description)` — Context manager for phase timing
  - `.checkpoint(label, value)` — Print checkpoint
  - `.summary(metrics)` — Final summary with timing breakdown
  - `.total_elapsed_s` — Property for total time

### `pipeline/runner.py` — Pipeline Orchestration

- `PipelineRunner` — Full Phase 1→2→3→4 orchestration with diagnostics
- `run_exact_diag_sweep(h_values, n_qubits, ...)` — Standalone Phase 1 helper

### `framework/preflight.py` — Pre-flight Validation

Validates variant runner configurations before execution. Use **always** before running a variant script for the first time or after editing it.

- `PreflightChecker(specs, project_root, strict)` — Main checker class
  - `.run_all(verbose)` → `PreflightReport`
  - `.check_h_test_unseen()` — Data leakage detection
  - `.check_h_test_valid_regime()` — Regime violation detection
  - `.check_descending_sweep()` — Warm-start order validation
  - `.check_interpolation()` — Extrapolation risk detection
  - `.check_output_fresh()` — Output collision detection
- `VariantSpec` — Lightweight variant specification (decoupled from PipelineVariant)
  - `.from_pipeline_variant(variant)` — Parse from PipelineVariant command
  - `.from_dict(d)` — Parse from JSON dict
- `specs_from_pipeline_variants(variants)` — Batch convert PipelineVariant → VariantSpec
- `specs_from_variant_runner(build_fn, build_fn, build_fn, N)` — From builder functions
- `specs_from_json(path)` — From JSON file
- `P1_VALID_REGIME` / `P2_VALID_REGIME` — Canonical valid regime dicts
- `get_regime_threshold(topology, n_qubits, p)` — Lookup threshold

**CLI**: `python src/qmbp_simulation/framework/preflight.py --from-script <path> [--strict] [--quiet]`
**Makefile**: `make preflight SCRIPT=<path>`

## Where to Find What

| I need to... | Look at... |
|---|---|
| Understand physics constraints | `.kiro/skills/quantum/SKILL.md` |
| See code templates and patterns | `.kiro/knowledge/workflow-recipes.md` |
| Check what's stable vs active | `.kiro/steering/project-status.md` |
| Follow code conventions | `.kiro/steering/code-style.md` |
| See known failure modes | `.kiro/knowledge/error-patterns.md` |
| Check numerical baselines | `.kiro/knowledge/validation-targets.md` |
| See analysis of results | `.kiro/knowledge/poc-results.md` |
| Understand MPNN architecture | `.kiro/knowledge/gnn-architecture.md` |
| See hardware deployment strategy | `.kiro/steering/hardware-deployment.md` |
| Review literature insights | `.kiro/knowledge/literature-synthesis.md` |
| See experiment framework guide | `.kiro/steering/v8-experiments.md` |
| See full experiment history | `documentation/binnacles/` |
| Validate a script before running | `src/qmbp_simulation/framework/preflight.py --from-script <path>` |
| Validate package | `tests/smoke_test.py` |
| Check knowledge freshness | `.kiro/knowledge/changelog.md` |

## Repository Zones (CRITICAL for AI agents)

1. **Framework** (`src/qmbp_simulation/`) — The installable package. Primary development target.
2. **Consumers** (`experiments/` + `scripts/`) — Use the framework via `from qmbp_simulation import ...`. Not part of the package.

**Rule**: No module in `src/qmbp_simulation/` imports from `experiments/` or `scripts/`.
Scripts are thin CLI wrappers that delegate to framework classes.

## Key Workflow Rules

1. Never modify stable modules (listed in `project-status.md`) without explicit request.
2. Always use `make_lattice()` to create lattice configs — never `copy.copy()`.
3. Always validate changes with `make test`.
4. Document results in the binnacle with configuration, metrics, and analysis.
5. Phase 2 MUST use pure energy cost (V5.x lesson).
6. All scripts live in `scripts/` — never put executable scripts in `src/`.
7. Use `from qmbp_simulation import ...` for all imports — never `sys.path` hacks.
8. Experiments inherit from `BaseExperiment` and use `ExperimentMetrics` for results.
9. New scripts should use `framework/cli.py` for argument parsing.
10. Result saving should use `framework/result_io.py` for consistent structure.
11. **Always run preflight before executing variant runner scripts** — `python src/qmbp_simulation/framework/preflight.py --from-script <path>`.
12. **JSON serialization**: use `json_serialize` from `utils/helpers.py` — never create local `_json_default`.
13. **Experiment criteria**: import from `framework/criteria.py` — never duplicate threshold dicts.

## Current Best Configuration (from 60+ benchmark runs)

### N=6 (1D TFIM chain)

| Parameter | Value |
|-----------|-------|
| VQE restarts | 5 (p=2) / 1 (p=1) |
| VQE maxiter | 1000 |
| VQE σ (restart) | 0.1 |
| MPNN hidden | 64 |
| MPNN layers | 3 |
| MPNN epochs | 6000 |
| MPNN lr | 1e-3 |
| MPNN patience | 300 |
| Fid threshold | 0.93 |
| GINConv | yes (GATConv rejected) |

### N=10 (1D TFIM chain)

| Parameter | Value |
|-----------|-------|
| VQE restarts | 5 (p=2) / 1 (p=1) |
| VQE maxiter | 1000 |
| VQE σ (restart) | 0.1 |
| MPNN hidden | **128** |
| MPNN layers | 3 |
| MPNN epochs | 6000 |
| MPNN lr | 1e-3 |
| MPNN patience | **500** |
| Fid threshold | 0.93 |
| Seeds | 42, 43, 44 (use median — seed 43 problematic for ladder, seed 44 for triangular) |

### N=20 (MPS-based)

| Parameter | Value |
|-----------|-------|
| MPS chi | 64 (sufficient for 1D HVA) |
| VQE restarts | 7 (p=2) / 5 (p=1) |
| VQE maxiter | 100 |
| Valid regime | h ≥ 2.0 (p=2), h ≥ 2.25 (p=1) |
| MPNN hidden | 128 |
| Training points | 11 (h∈[1.5,2.0]) for p=2; 15 (h∈[2.25,4.0]) for p=1 |
