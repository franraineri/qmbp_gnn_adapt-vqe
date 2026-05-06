# Project Guide — GNN-HVA Framework

> This is the entry point for AI agents working on this project. Read this first.

## What This Project Is

A Master's thesis implementing a hybrid classical-quantum pipeline for characterizing topological phases of matter. The pipeline uses a Graph Neural Network (MPNN) to predict optimal parameters for a shallow quantum circuit (HVA), which is then deployed on IBM quantum hardware to classify quantum phases via local observable measurements.

## Repository Map

```
├── src/poc/v6/              ← ACTIVE CODE (9 Python modules + 2 notebooks)
│   ├── config.py            ← Shared dataclasses (LatticeConfig, VQEConfig, etc.)
│   ├── hamiltonian_builder.py ← Phase 1: Hamiltonian + lattice generators
│   ├── classical_solver.py  ← Phase 1: Exact diag + DMRG ground truth
│   ├── hva_builder.py       ← Phase 2: HVA circuit construction
│   ├── vqe_optimizer.py     ← Phase 2: Multi-start VQE + callbacks
│   ├── mpnn_predictor.py    ← Phase 3: MPNN model + training loop
│   ├── qrc_pipeline.py      ← Phase 4: QRC fallback route
│   ├── hardware_deployer.py ← Phase 4: Adapt-VQE + QRC deployment
│   ├── pipeline_utils.py    ← Cross-phase: dataset save/load, integrity checks
│   ├── poc_v6_phases1_2.ipynb ← Full Phase 1-2 notebook
│   └── poc_v6_phases3_4.ipynb ← Full Phase 3-4 notebook
│
├── scripts/                 ← ALL executable scripts (single location)
│   ├── smoke_test.py        ← Quick end-to-end validation (~7s)
│   ├── benchmark_v6.py      ← Multi-run benchmark with configurable params
│   ├── run_notebooks.py     ← Notebook executor with validation
│   ├── hooks/               ← Pre-commit hook scripts
│   │   └── check_hva_depth.py
│   ├── benchmark_results/   ← Raw JSON results (gitignored)
│   └── notebook_results/    ← Executed notebooks (gitignored)
│
├── tests/                   ← Pytest test suite
│   ├── conftest.py          ← Shared fixtures
│   └── test_v6_pipeline.py  ← 18 unit tests
│
├── documentation/           ← Human-readable docs
│   ├── binnacle-N6.md       ← Experiment log: N=6 (complete, 40+ experiments)
│   ├── binnacle-N10.md      ← Experiment log: N=10 scaling (active)
│   └── architectural_doc_es_en.md ← Architecture justification
│
├── .kiro/                   ← AI agent configuration
│   ├── skills/quantum/SKILL.md    ← Core rules and constraints
│   ├── knowledge/                 ← Domain knowledge (7 files)
│   ├── steering/                  ← Active guidance (status + code style)
│   ├── hooks/                     ← Automated checks (Qiskit compliance)
│   └── specs/                     ← V6 spec (requirements, design, tasks)
│
├── Makefile                 ← SINGLE ENTRY POINT for all operations
├── pyproject.toml           ← Build config, ruff, mypy, pytest
└── requirements.txt         ← Python dependencies
```

## How to Run Things

The Makefile is the single entry point. Use `make` for everything.

| Command | What it does | Time |
|---------|-------------|------|
| `make help` | Show all targets | instant |
| `make test` | Pytest (18 tests) | ~5s |
| `make smoke-test` | End-to-end pipeline | ~7s |
| `make lint` | Ruff linter | ~1s |
| `make check-full` | lint + test + smoke-test | ~15s |
| `make benchmark` | 3-run benchmark (N=6) | ~50s |
| `make benchmark ARGS="--n-qubits 10"` | Custom benchmark | varies |
| `make benchmark-n10` | N=10 chain shortcut | ~2.5min |
| `make run-notebooks` | Execute both notebooks | ~15min |
| `make check` | All pre-commit hooks | ~5s |

## Where to Find What

| I need to... | Look at... |
|---|---|
| Understand physics constraints | `.kiro/skills/quantum/SKILL.md` |
| See Qiskit 2.x code patterns | `.kiro/knowledge/workflow-recipes.md` |
| Check what's stable vs active | `.kiro/steering/project-status.md` |
| Follow code conventions | `.kiro/steering/code-style.md` |
| See known failure modes | `.kiro/knowledge/error-patterns.md` |
| Check numerical baselines | `.kiro/knowledge/poc-results.md` |
| Understand MPNN architecture | `.kiro/knowledge/gnn-architecture.md` |
| See hardware deployment strategy | `.kiro/knowledge/optimization-hardware.md` |
| Review literature insights & improvements | `.kiro/knowledge/literature-synthesis.md` |
| See full experiment history | `documentation/binnacle-N6.md` (N=6) or `binnacle-N10.md` (N=10) |
| Find alternative techniques | `documentation/alternative_bibliography.md` |

## Key Workflow Rules
1. Never modify stable modules (listed in `project-status.md`) without explicit request.
2. Always use `make_lattice()` to create lattice configs — never `copy.copy()`.
3. Always validate changes with `make check-full`.
4. Document results in the binnacle with configuration, metrics, and analysis.
5. Phase 2 MUST use pure energy cost (V5.x lesson).
6. All scripts live in `scripts/` — never put executable scripts in `src/`.

## Current Best Configuration (from 28+ benchmark runs)

| Parameter | Value |
|-----------|-------|
| VQE restarts | 5 |
| VQE maxiter | 1000 |
| MPNN hidden | 64 |
| MPNN layers | 3 |
| MPNN epochs | 6000 |
| MPNN lr | 1e-3 |
| Fid threshold | 0.93 |

Expected: 5/6 at h=1.5 (N=6), 4-5/6 at h=1.4 (N=6), 2-3/6 at h=1.5 (N=10).
