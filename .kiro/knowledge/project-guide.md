# Project Guide — GNN-HVA Framework

> This is the entry point for AI agents working on this project. Read this first.

## What This Project Is

A Master's thesis implementing a hybrid classical-quantum pipeline for characterizing topological phases of matter. The pipeline uses a Graph Neural Network (MPNN) to predict optimal parameters for a shallow quantum circuit (HVA), which is then deployed on IBM quantum hardware to classify quantum phases via local observable measurements.

## Repository Map

```
├── src/poc/v6/              ← ACTIVE CODE (9 Python modules + 2 notebooks)
│   ├── config.py            ← Shared dataclasses (LatticeConfig, VQEConfig, etc.)
│   ├── hamiltonian_builder.py ← Phase 1: Hamiltonian construction + lattice generators
│   ├── classical_solver.py  ← Phase 1: Exact diag + DMRG ground truth
│   ├── hva_builder.py       ← Phase 2: HVA circuit construction
│   ├── vqe_optimizer.py     ← Phase 2: Multi-start VQE + callbacks
│   ├── mpnn_predictor.py    ← Phase 3: MPNN model + training loop
│   ├── qrc_pipeline.py      ← Phase 4: QRC fallback route
│   ├── hardware_deployer.py ← Phase 4: Adapt-VQE + QRC deployment
│   ├── pipeline_utils.py    ← Cross-phase: dataset save/load, integrity checks
│   ├── smoke_test.py        ← Quick end-to-end validation (7s)
│   ├── poc_v6_phases1_2.ipynb ← Full Phase 1-2 notebook
│   └── poc_v6_phases3_4.ipynb ← Full Phase 3-4 notebook
│
├── src/poc/v4/              ← Previous best PoC (reference only, do not modify)
├── src/poc/v5/, v5.1/       ← Failed experiments (reference only)
│
├── scripts/
│   ├── benchmark_v6.py      ← Multi-run benchmark with configurable parameters
│   └── benchmark_results/   ← Raw JSON results from benchmark runs
│
├── tests/
│   ├── conftest.py          ← Shared pytest fixtures
│   └── test_v6_pipeline.py  ← 18 unit tests (pytest)
│
├── documentation/
│   ├── binnacle.md          ← Chronological development log with all results
│   ├── architectural_doc_es_en.md ← Architecture justification (bilingual)
│   └── bibliography.md      ← All paper references
│
├── .kiro/
│   ├── skills/quantum/SKILL.md    ← Core rules and constraints (READ THIS)
│   ├── knowledge/                 ← Domain knowledge files (7 files)
│   ├── steering/                  ← Active guidance (project status + code style)
│   ├── hooks/                     ← Automated checks (Qiskit compliance)
│   └── specs/gnn-hva-v6-architecture/ ← V6 spec (requirements, design, tasks)
│
├── run_v6_smoke_test.sh     ← Quick validation script
├── run_v6_benchmark.sh      ← Multi-run benchmark script
├── pyproject.toml           ← Build config, ruff, mypy, pytest
└── requirements.txt         ← Python dependencies
```

## Where to Find What

| I need to... | Look at... |
|---|---|
| Understand the physics constraints | `.kiro/skills/quantum/SKILL.md` |
| See correct Qiskit 2.x patterns | `.kiro/knowledge/workflow-recipes.md` |
| Check what's stable vs active | `.kiro/steering/project-status.md` |
| Follow code conventions | `.kiro/steering/code-style.md` |
| See known failure modes | `.kiro/knowledge/error-patterns.md` |
| Check numerical baselines | `.kiro/knowledge/poc-results.md` |
| Understand the MPNN architecture | `.kiro/knowledge/gnn-architecture.md` |
| See full experiment history | `documentation/binnacle.md` |
| Run tests | `pytest tests/ -v` |
| Quick validation | `./run_v6_smoke_test.sh` |
| Benchmark with parameters | `python scripts/benchmark_v6.py --help` |

## Development Methodology

### Spec-Driven Development
Major features start as a spec in `.kiro/specs/` with three documents: requirements → design → tasks. The V6 architecture was built this way. Tasks reference specific requirements and are executed sequentially.

### Benchmark-Driven Tuning
Hyperparameter changes are validated via `scripts/benchmark_v6.py` with multiple seeds. Results are appended to `documentation/binnacle.md` with full configuration, per-run metrics, and aggregate statistics. This creates a searchable history of what was tried and what worked.

### Key Workflow Rules
1. Never modify stable modules (listed in `project-status.md`) without explicit request.
2. Always use `make_lattice()` to create lattice configs — never `copy.copy()`.
3. Always validate changes with `pytest tests/ -v` and `./run_v6_smoke_test.sh`.
4. Document results in the binnacle with configuration, metrics, and analysis.
5. Phase 2 MUST use pure energy cost. This is the V5.x lesson — changing it without updating Phase 3 breaks everything.

## Current Best Configuration (from 28 benchmark runs)

| Parameter | Value |
|-----------|-------|
| VQE restarts | 5 |
| VQE maxiter | 1000 |
| MPNN hidden | 64 |
| MPNN layers | 3 |
| MPNN epochs | 6000 |
| MPNN lr | 1e-3 |
| Fid threshold | 0.93 |

Expected results: 5/6 at h=1.5, 4-5/6 at h=1.4, 2-3/6 at h=1.25. The only metric that never passes is ΔE < 1e-2 (HVA expressibility ceiling).
