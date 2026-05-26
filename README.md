# Hybrid GNN-HVA Framework for Topological Phase Characterization

## Overview

Master's Thesis (TFM) in Quantum Computing and Condensed Matter Physics. The project
accelerates Variational Quantum Eigensolvers (VQE) for quantum phase characterization
using a predictive hybrid architecture: a classical Graph Neural Network (GNN) trained
on Tensor Network data provides "Intelligent Warm-Start" initialization for a shallow,
physics-informed quantum circuit (Hamiltonian Variational Ansatz - HVA).

**Key constraint** (Mele et al., Nature Physics 2026): Non-unital noise truncates
circuits to O(log n). All HVA circuits are shallow (p ≤ 2 layers), enforced by
pre-commit hooks.

## Quick Start

```bash
# Clone and setup
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# Verify installation
python -c "from qmbp_simulation import HamiltonianBuilder, make_lattice; print('OK')"

# Run smoke test (N=4, p=1, <30s)
python scripts/smoke_test.py

# Run full test suite
pytest tests/ -x
```

## Package Structure

```
project-root/
├── src/
│   └── qmbp_simulation/           # Installable package (Zone 1: Framework)
│       ├── __init__.py             # Package-level re-exports
│       ├── utils/                  # Seed, JSON, timing (no internal deps)
│       ├── models/                 # LatticeConfig, Hamiltonians, data models
│       ├── solvers/                # ExactDiag, DMRG
│       ├── circuits/               # HVA builder
│       ├── execution/              # Backend ABC + implementations
│       ├── optimizers/             # VQE, SPSA
│       ├── predictors/             # MPNN model, training, checkpoints
│       ├── pipeline/               # Orchestration, dataset I/O
│       ├── framework/              # BaseExperiment, config, metrics, logging
│       └── analysis/               # Gradient analysis, diagnostics, comparison
├── experiments/                    # Experiment scripts (Zone 2: Consumers)
│   ├── optimization/               # VQE technique experiments
│   ├── scaling/                    # Finite-size scaling
│   ├── landscape/                  # Hessian, fluctuation
│   ├── predictor/                  # MPNN enhancements
│   ├── hardware/                   # Hardware deployment
│   ├── generalization/             # Model-agnostic tests
│   └── helpers/                    # DyPP, sign canon, freezing, etc.
├── scripts/                        # CLI entry points (Zone 2: Consumers)
│   ├── run_experiment.py           # Run experiments by ID
│   ├── run_pipeline.py             # Full 4-phase pipeline
│   ├── compare.py                  # Cross-experiment comparison
│   ├── smoke_test.py               # Quick validation (<30s)
│   └── benchmark.py                # Performance benchmarking
├── tests/                          # pytest suite
│   ├── unit/                       # Per-module unit tests
│   ├── integration/                # End-to-end pipeline tests
│   └── conftest.py                 # Shared fixtures
├── results/                        # Experiment outputs
├── documentation/                  # Thesis docs, binnacles, bibliography
└── pyproject.toml                  # Package config, Ruff, pytest
```

## Import Examples

```python
# Core imports (from package top-level)
from qmbp_simulation import (
    HamiltonianBuilder, make_lattice, ClassicalSolver,
    HVACircuitBuilder, VQEOptimizer,
    LatticeConfig, VQEConfig, GroundTruthResult, VQEResult,
    save_phase12_dataset, load_phase12_dataset,
)

# Submodule imports
from qmbp_simulation.models import SUPPORTED_TOPOLOGIES, MAX_P_LAYERS
from qmbp_simulation.execution import NoiselessBackend, NoisyBackend, MitigationOptions
from qmbp_simulation.optimizers import SPSAOptimizer
from qmbp_simulation.predictors import MPNNPredictor, build_graph_dataset, train_mpnn
from qmbp_simulation.framework import BaseExperiment, ExperimentConfig, ExperimentMetrics
from qmbp_simulation.analysis import (
    WeightGradientAnalyzer, DiagnosticCollector,
    compute_snr, compute_hessian, landscape_fluctuation,
    compute_fraction_near_gs, compute_theta_smoothness,
)
from qmbp_simulation.pipeline import PipelineRunner
```

## The 4-Phase Pipeline

1. **Phase 1 — Classical Ground Truth**: Exact diag (N<15) or DMRG/TeNPy (N≤40)
2. **Phase 2 — HVA VQE**: Descending h-sweep with warm-start, L-BFGS-B, p≤2
3. **Phase 3 — MPNN Predictor**: GINConv + global pooling, fidelity-filtered data
4. **Phase 4 — Deployment**: MPNN warm-start → hardware VQE with error mitigation

The `PipelineRunner` includes always-on diagnostics via `DiagnosticCollector` —
every run captures timing, convergence, θ-smoothness, per-h MSE, and energy
decomposition metrics automatically.

## Running Experiments

```bash
# List available experiments
python scripts/run_experiment.py --list

# Run a single experiment
python scripts/run_experiment.py --exp B4 --verbose

# Run multiple experiments
python scripts/run_experiment.py --exp B4 D1 F1

# Compare results against baseline

# Run full pipeline (N=6, default h-sweep)
python scripts/run_pipeline.py --n-qubits 6 --p 2

# Run full pipeline with diagnostics output
python scripts/run_pipeline.py --n-qubits 6 --p 2 --verbose --output-dir results/my_run
```

## Running Tests

```bash
# Full test suite
pytest tests/

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# With coverage
pytest tests/ --cov=qmbp_simulation --cov-report=term-missing

# Skip slow tests
pytest tests/ -m "not slow"
```

## Key Results

| System | ΔE/gap | Status |
|--------|:------:|--------|
| N=6, h≥1.25 | < 5% | ✅ Thesis-ready |
| N=10, h≥1.5 | < 5% | ✅ Thesis-ready |
| N=20 p=1, h≥2.25 | 1.58% | ✅ Validated |
| N=20 p=2, h≥2.0 | 1.75% | ✅ Validated (MPS) |

## Tech Stack

| Component | Tool | Version |
|-----------|------|---------|
| Quantum circuits | Qiskit | 2.4.x |
| Hardware runtime | qiskit-ibm-runtime | 0.46.x |
| Noisy simulation | qiskit-aer (MPS) | 0.17.x |
| ML predictor | PyTorch + PyTorch Geometric | 2.11 + 2.7 |
| Tensor networks | TeNPy | 1.1.x |
| Linting | Ruff | 0.11.x |
| Testing | pytest + Hypothesis | 9.x + 6.x |
| Git hooks | pre-commit (12 hooks) | 4.2.x |

## Repository Zones

The codebase is organized into three clearly separated zones:

1. **Framework** (`src/qmbp_simulation/`) — The installable package. Primary development target.
   Can grow and evolve freely. All reusable logic lives here.

2. **Consumers** (`experiments/` + `scripts/`) — Use the framework via
   `from qmbp_simulation import ...`. Not part of the installable package.

## Constraints (enforced by pre-commit)

- HVA only, never HEA. p ≤ 2 layers.
- Primitives V2 only (no deprecated Qiskit APIs)
- Fidelity threshold ≥ 0.93 in training data
- No secrets in commits (gitleaks)
- Conventional commits (commitizen)

## Documentation

- **[V8 Status](documentation/v8/STATUS.md)** — Single source of truth for V8 experiments
- **[V7 Results](documentation/v7/RESULTS_SUMMARY_V61_V7.md)** — Complete V6.1/V7 summary
- **[Architecture](documentation/architectural_doc_es_en.md)** — System design (ES/EN)
- **[Thesis Guide](documentation/thesis-structure-guide.md)** — Chapter outline
- **[Bibliography](documentation/bibliography/bibliography_curated.md)** — Curated references
- **[Binnacles](documentation/binnacles/)** — All experiment logs

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev,test]"

# Lint
ruff check src/ experiments/ scripts/ tests/

# Format
ruff format src/ experiments/ scripts/ tests/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

---

*Franco Raineri — Universidad de Buenos Aires, 2026*
