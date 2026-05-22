# Project Guide — GNN-HVA Framework

> This is the entry point for AI agents working on this project. Read this first.

## What This Project Is

A Master's thesis implementing a hybrid classical-quantum pipeline for characterizing topological phases of matter. The pipeline uses a Graph Neural Network (MPNN) to predict optimal parameters for a shallow quantum circuit (HVA), which is then deployed on IBM quantum hardware to classify quantum phases via local observable measurements.

## Repository Map

```
├── src/poc/v6/              ← ACTIVE CODE (14 Python modules + 2 notebooks + experimental/)
│   ├── __init__.py          ← Public API (core modules only, lazy-loads heavy deps)
│   ├── config.py            ← Shared dataclasses (LatticeConfig, VQEConfig, etc.) [STABLE]
│   ├── config_v61.py        ← V6.1 constants + dataclasses (DeployResultV61, LayoutResult, etc.)
│   ├── hamiltonian_builder.py ← Phase 1: Hamiltonian + lattice generators [STABLE]
│   ├── classical_solver.py  ← Phase 1: Exact diag + DMRG ground truth [STABLE]
│   ├── hva_builder.py       ← Phase 2: HVA circuit construction [STABLE]
│   ├── vqe_optimizer.py     ← Phase 2: Multi-start VQE + callbacks [STABLE]
│   ├── mpnn_predictor.py    ← Phase 3: MPNN model + training + checkpoint save/load
│   ├── pipeline_utils.py    ← Cross-phase: dataset save/load, integrity checks [STABLE]
│   ├── qrc_pipeline.py      ← Phase 4: QRC fallback route
│   ├── hardware_deployer_v61.py ← Phase 4: V6.1 full hardware path (ZNE, DD, twirling)
│   ├── analysis_utils.py    ← Post-training: WeightGradientAnalyzer (zero QPU cost)
│   ├── diagnostics.py       ← Pipeline observability (DiagnosticCollector, logging)
│   ├── poc_v6_phases1_2.ipynb ← Full Phase 1-2 notebook
│   └── poc_v6_phases3_4.ipynb ← Full Phase 3-4 notebook
│
├── scripts/                 ← ALL executable scripts (single location)
│   ├── smoke_test.py        ← LEGACY V6.0 end-to-end (superseded by smoke_test_v61.py)
│   ├── smoke_test_v61.py    ← V6.1 smoke test: deployer + gradient analysis (~16s)
│   ├── benchmark_v6.py      ← Multi-run benchmark with configurable params
│   ├── run_notebooks.py     ← Notebook executor with auto-registry + binnacle generation
│   ├── run_v61_parametric.py ← Parametric pipeline runner
│   ├── run_v61_noisy.py     ← Noisy simulation sweep
│   ├── experiments_hamed_v7/ ← V7 full experiment suite (22 sub-experiments)
│   ├── experiments_v8/      ← V8 noiseless experiment framework (10 experiments)
│   │   ├── core/            ← Infrastructure (base_experiment, config, metrics, landscape)
│   │   ├── experiments/     ← One file per experiment (A3, B1, B2, B4, C1, C3, D1, E4, F1, F3)
│   │   ├── techniques/      ← Reusable building blocks
│   │   ├── results/         ← Auto-generated JSON results
│   │   ├── run_experiment.py ← CLI entry point
│   │   └── run_*.py         ← Standalone scripts (B4@N=10, F3@p=1, D1-reg, C1@N=10)
│   ├── hooks/               ← Pre-commit hook scripts
│   │   └── check_hva_depth.py
│   ├── benchmark_results/   ← Raw JSON results (gitignored)
│   └── notebook_results/    ← Executed notebooks + JSON summaries (gitignored)
│
├── tests/                   ← Pytest test suite
│   ├── conftest.py          ← Shared fixtures
│   └── test_v6_pipeline.py  ← 33 tests (18 V6.0 + 15 V6.1)
│
├── documentation/           ← Human-readable docs
│   ├── binnacles/           ← Experiment logs (auto-populated by run_notebooks.py)
│   │   ├── binnacle-N6.md   ← N=6 experiments (complete, 40+)
│   │   └── binnacle-N10.md  ← N=10 scaling (active)
│   └── architectural_doc_es_en.md ← Architecture justification
│
├── .kiro/                   ← AI agent configuration
│   ├── skills/quantum/SKILL.md    ← Core rules and constraints
│   ├── knowledge/                 ← Domain knowledge (8 files)
│   ├── steering/                  ← Active guidance (status + code style + hardware)
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
| `make test` | Pytest (33 tests) | ~8s |
| `make smoke-test` | V6.0 end-to-end pipeline | ~7s |
| `make lint` | Ruff linter | ~1s |
| `make check-full` | lint + test + smoke-test | ~15s |
| `make benchmark` | 3-run benchmark (N=6) | ~50s |
| `make benchmark ARGS="--n-qubits 10"` | Custom benchmark | varies |
| `make benchmark-n10` | N=10 chain shortcut | ~2.5min |
| `make run-notebooks` | Execute both notebooks with auto-registry | ~15min |
| `make check` | All pre-commit hooks | ~5s |

### Direct Script Usage (beyond Makefile)

**V6.1 Smoke Test** — validates deployer + gradient analysis:
```bash
python scripts/smoke_test_v61.py    # ~16s, exercises all V6.1 modules
```

**Notebook Executor** — full auto-registry with structured metrics:
```bash
python scripts/run_notebooks.py                         # both notebooks
python scripts/run_notebooks.py --phase 1-2             # only phases 1-2
python scripts/run_notebooks.py --phase 3-4             # only phases 3-4
python scripts/run_notebooks.py --timeout 600           # 10 min wall-clock timeout
python scripts/run_notebooks.py --binnacle             # append binnacle entry to docs
python scripts/run_notebooks.py --label "N10 test"     # label for the run
python scripts/run_notebooks.py --binnacle-file binnacle-N10.md  # explicit target
python scripts/run_notebooks.py --dry-run              # pre-flight only, no execution
python scripts/run_notebooks.py --keep-last 10         # prune old results
```

Exit codes: `0` = all pass, `1` = execution failure, `2` = validation failure, `3` = pre-flight failure.

The notebook executor auto-extracts metrics (fidelity, MSE, ΔE/gap, checklist, phase label, gradient peaks, ZNE R², etc.) and generates binnacle-ready markdown with environment info, run-to-run comparison, and auto-generated observations.

## Where to Find What

| I need to... | Look at... |
|---|---|
| Understand physics constraints | `.kiro/skills/quantum/SKILL.md` |
| See Qiskit 2.x code patterns | `.kiro/knowledge/workflow-recipes.md` |
| Check what's stable vs active | `.kiro/steering/project-status.md` |
| Follow code conventions | `.kiro/steering/code-style.md` |
| See known failure modes | `.kiro/knowledge/error-patterns.md` |
| Check numerical baselines & tables | `.kiro/knowledge/validation-targets.md` (source of truth for numbers) |
| See analysis & interpretation of results | `.kiro/knowledge/poc-results.md` |
| Understand MPNN architecture | `.kiro/knowledge/gnn-architecture.md` |
| See hardware deployment strategy | `.kiro/knowledge/optimization-hardware.md` + `.kiro/steering/hardware-deployment.md` |
| Review literature insights & improvements | `.kiro/knowledge/literature-synthesis.md` |
| See V8 experiment results | `documentation/v8/STATUS.md` (master doc) |
| See V8 experiment framework guide | `.kiro/steering/v8-experiments.md` |
| See full experiment history | `documentation/binnacles/` (multiple binnacle files) |
| Find alternative techniques | `documentation/bibliography/alternative_bibliography.md` |
| Use shared pipeline execution | V8 framework: `scripts/experiments_v8/` |
| Find deprecated approaches | Removed (GATPredictor, augmentation, pipeline_core — all deleted) |
| Run V8 experiments | `scripts/experiments_v8/run_experiment.py --list` |
| Validate V6.1 deployer | `scripts/smoke_test_v61.py` |
| Check knowledge freshness | `.kiro/knowledge/changelog.md` |

## Key Workflow Rules
1. Never modify stable modules (listed in `project-status.md`) without explicit request.
2. Always use `make_lattice()` to create lattice configs — never `copy.copy()`.
3. Always validate changes with `make check-full`.
4. Document results in the binnacle with configuration, metrics, and analysis.
5. Phase 2 MUST use pure energy cost (V5.x lesson).
6. All scripts live in `scripts/` — never put executable scripts in `src/`.
7. Use `run_notebooks.py --binnacle --label "description"` for automated experiment logging.
8. V6.1 modules (`hardware_deployer_v61.py`, `config_v61.py`, `analysis_utils.py`) are the active Phase 4 code.
9. All scripts live in `scripts/` — never put executable scripts in `src/`.

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
| Data augmentation | no |

Expected (V6.1 4-metric checklist): 4/4 at h=1.25, 4/4 at h=1.4, 4/4 at h=1.5.

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
| GINConv | yes |
| Preferred seed | **43** (10x better MSE) |
| Data augmentation | no (hurts at N=10) |

Expected (V6.1 4-metric checklist): 4/4 at h=1.5 (all seeds), 4/4 at h=1.4 (seeds 43/44), 3/4 at h=1.4 (seed 42).

## Auto-Extracted Metrics (from `run_notebooks.py`)

The notebook executor automatically extracts these metrics from cell outputs:

| Metric key | Source | Threshold |
|------------|--------|-----------|
| `avg_fidelity` | Phase 1-2 VQE | ≥93% |
| `final_mse` | Phase 3 MPNN | <0.1 (warn), <0.005 (excellent) |
| `delta_e_over_gap` | Phase 4 deploy | <5% pass, <10% marginal |
| `checklist_pass/total` | Phase 4 validation | 5/6 at h=1.5 |
| `critical_region_detected` | Gradient analysis | True = phase transition in weights |
| `gradient_peak_h` | Gradient analysis | Expected near h≈1.0–1.2 |
| `zne_r_squared` | Hardware ZNE | >0.8 (warn if lower) |
| `shots` | Hardware deploy | ≥8192 for N≤6 |
| `deploy_mode` | Hardware deploy | "simulation" or "hardware" |
