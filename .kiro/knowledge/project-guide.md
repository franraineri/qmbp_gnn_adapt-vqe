# Project Guide — GNN-HVA Framework

> This is the entry point for AI agents working on this project. Read this first.

## What This Project Is

A Master's thesis implementing a hybrid classical-quantum pipeline for characterizing topological phases of matter. The pipeline uses a Graph Neural Network (MPNN) to predict optimal parameters for a shallow quantum circuit (HVA), which is then deployed on IBM quantum hardware to classify quantum phases via local observable measurements.

## Repository Map

```
├── src/qmbp_simulation/     ← INSTALLABLE PACKAGE (the framework)
│   ├── __init__.py          ← Package-level re-exports
│   ├── utils/               ← Seed, JSON, timing (no internal deps)
│   ├── models/              ← LatticeConfig, Hamiltonians, data models, constants
│   ├── solvers/             ← Exact diag + DMRG ground truth
│   ├── circuits/            ← HVA circuit construction (p≤2 enforced)
│   ├── execution/           ← Backend ABC + noiseless/noisy/hardware implementations
│   ├── optimizers/          ← Multi-start VQE + SPSA with warm-start
│   ├── predictors/          ← MPNN model, training, checkpoints
│   ├── pipeline/            ← Dataset save/load, pipeline orchestration
│   ├── framework/           ← BaseExperiment lifecycle, config, metrics, logging
│   └── analysis/            ← Gradient analysis, diagnostics, landscape, metrics
│
├── experiments/              ← EXPERIMENT SCRIPTS (consumers of the package)
│   ├── helpers/             ← Reusable technique modules (dypp, freezing, etc.)
│   ├── optimization/        ← B1, B2, B4, C3
│   ├── scaling/             ← A3
│   ├── landscape/           ← F1, F3
│   ├── predictor/           ← C1, D1, E3
│   ├── hardware/            ← Hardware-specific experiments
│   └── generalization/      ← E4
│
├── scripts/                 ← CLI ENTRY POINTS
│   ├── run_experiment.py    ← Unified CLI for running experiments by ID
│   ├── run_pipeline.py      ← Full 4-phase pipeline CLI
│   ├── compare.py           ← Cross-experiment result comparison
│   ├── smoke_test.py        ← Package smoke test (N=4, p=1, <30s)
│   ├── benchmark.py         ← Performance benchmarking
│   ├── run_thesis_results.py ← Thesis table generation
│   └── hooks/               ← Pre-commit hook scripts
│
├── tests/                   ← Pytest test suite
│   ├── conftest.py          ← Shared fixtures
│   ├── unit/                ← Per-module unit + property tests
│   └── integration/         ← Smoke, pipeline e2e, backward compat
│
├── results/                 ← Experiment outputs
│   ├── experiments/         ← Auto-generated JSON results (gitignored)
│   ├── benchmarks/          ← Benchmark results (gitignored)
│   └── thesis/              ← Committed definitive results
│
├── documentation/           ← Human-readable docs
│   ├── binnacles/           ← Experiment logs
│   └── bibliography/        ← Literature references
│
├── .kiro/                   ← AI agent configuration
│   ├── skills/quantum/SKILL.md    ← Core rules and constraints
│   ├── knowledge/                 ← Domain knowledge
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
| `make check-full` | lint + test + smoke-test | ~15s |
| `python scripts/smoke_test.py` | Package smoke test (N=4, p=1) | ~30s |
| `python scripts/run_experiment.py --list` | List all experiments | instant |
| `python scripts/run_experiment.py --exp A3` | Run experiment A3 | varies |
| `python scripts/run_pipeline.py --n 6 --p 2` | Full pipeline | ~5min |
| `python scripts/compare.py --all` | Compare all results | ~5s |

### Installation

```bash
pip install -e .                    # Install package in editable mode
python -c "import qmbp_simulation" # Verify import works
python scripts/smoke_test.py       # Verify pipeline works
```

## Where to Find What

| I need to... | Look at... |
|---|---|
| Understand physics constraints | `.kiro/skills/quantum/SKILL.md` |
| See Qiskit 2.x code patterns | `.kiro/knowledge/workflow-recipes.md` |
| Check what's stable vs active | `.kiro/steering/project-status.md` |
| Follow code conventions | `.kiro/steering/code-style.md` |
| See known failure modes | `.kiro/knowledge/error-patterns.md` |
| Check numerical baselines & tables | `.kiro/knowledge/validation-targets.md` |
| See analysis & interpretation of results | `.kiro/knowledge/poc-results.md` |
| Understand MPNN architecture | `.kiro/knowledge/gnn-architecture.md` |
| See hardware deployment strategy | `.kiro/steering/hardware-deployment.md` |
| Review literature insights | `.kiro/knowledge/literature-synthesis.md` |
| See experiment framework guide | `.kiro/steering/v8-experiments.md` |
| See full experiment history | `documentation/binnacles/` |
| Run experiments | `scripts/run_experiment.py --list` |
| Validate package | `scripts/smoke_test.py` |
| Check knowledge freshness | `.kiro/knowledge/changelog.md` |

## Repository Zones (CRITICAL for AI agents)

1. **Framework** (`src/qmbp_simulation/`) — The installable package. Primary development target.
2. **Consumers** (`experiments/` + `scripts/`) — Use the framework via `from qmbp_simulation import ...`. Not part of the package.

## Key Workflow Rules

1. Never modify stable modules (listed in `project-status.md`) without explicit request.
2. Always use `make_lattice()` to create lattice configs — never `copy.copy()`.
3. Always validate changes with `make test`.
4. Document results in the binnacle with configuration, metrics, and analysis.
5. Phase 2 MUST use pure energy cost (V5.x lesson).
6. All scripts live in `scripts/` — never put executable scripts in `src/`.
7. Use `from qmbp_simulation import ...` for all imports — never `sys.path` hacks.
8. Experiments inherit from `BaseExperiment` and use `ExperimentMetrics` for results.

## Current Best Configuration (from 60+ benchmark runs)

### N=6 (1D TFIM chain)

| Parameter | Value |
|-----------|-------|
| VQE restarts | 5 |
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
| VQE restarts | 5 |
| VQE maxiter | 1000 |
| VQE σ (restart) | 0.1 |
| MPNN hidden | **128** |
| MPNN layers | 3 |
| MPNN epochs | 6000 |
| MPNN lr | 1e-3 |
| MPNN patience | **500** |
| Fid threshold | 0.93 |
| Preferred seed | **43** (10x better MSE) |
