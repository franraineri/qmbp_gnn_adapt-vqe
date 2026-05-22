# Hybrid GNN-HVA Framework for Topological Phase Characterization

## Project Overview

Master's Thesis (TFM) in Quantum Computing and Condensed Matter Physics. The project
accelerates Variational Quantum Eigensolvers (VQE) for quantum phase characterization
using a predictive hybrid architecture: a classical Graph Neural Network (GNN) trained
on Tensor Network data provides "Intelligent Warm-Start" initialization for a shallow,
physics-informed quantum circuit (Hamiltonian Variational Ansatz - HVA).

**Key constraint** (Mele et al., Nature Physics 2026): Non-unital noise truncates
circuits to O(log n). All HVA circuits are shallow (p ≤ 2 layers), enforced by
pre-commit hooks.

## Current Status

- **V6.1 pipeline**: Complete and thesis-ready (N=6, N=10 validated)
- **V7 experiments**: Complete (12/22 run, 10 skipped with justification)
- **V8 experiments**: Complete — 13 runs across 10 experiments (landscape, scaling, methodology)
- **Hardware**: Pending IBM Torino deployment (local simulation exhausted)

### Key Results

| System | ΔE/gap | Status |
|--------|:------:|--------|
| N=6, h≥1.25 | < 5% | ✅ Thesis-ready |
| N=10, h≥1.5 | < 5% | ✅ Thesis-ready |
| N=20 p=1, h≥2.25 | 1.58% | ✅ Validated (V8 C3) |
| N=20 p=2, h≥2.0 | 1.75% | ✅ Validated (V7 3C, MPS) |

## The 4-Phase Pipeline

1. **Phase 1 — Classical Ground Truth**: Exact diag (N<15) or DMRG/TeNPy (N≤40)
2. **Phase 2 — HVA VQE**: Descending h-sweep with warm-start, L-BFGS-B, p≤2
3. **Phase 3 — MPNN Predictor**: GINConv + global pooling, fidelity-filtered data
4. **Phase 4 — Deployment**: MPNN warm-start → hardware VQE with error mitigation

## Project Structure

```
qmbp_gnn_adapt-vqe/
├── src/poc/v6/                         # Core pipeline (STABLE — do not modify)
│   ├── config.py                       # Shared dataclasses
│   ├── hamiltonian_builder.py          # Lattice + SparsePauliOp construction
│   ├── classical_solver.py             # Exact diag + DMRG
│   ├── hva_builder.py                  # HVA circuit (p≤2 enforced)
│   ├── vqe_optimizer.py                # Multi-start L-BFGS-B
│   ├── mpnn_predictor.py               # GINConv MPNN + training
│   ├── hardware_deployer_v61.py        # Production deployer (ZNE, DD, TREX)
│   ├── analysis_utils.py               # Weight gradient analysis
│   ├── diagnostics.py                  # Pipeline observability
│   └── pipeline_utils.py               # Dataset save/load
│
├── scripts/
│   ├── experiments_v8/                 # V8 experiment framework (COMPLETE)
│   │   ├── core/                       # BaseExperiment, config, metrics, logging
│   │   ├── techniques/                 # Reusable: hessian, freezing, physics_loss...
│   │   ├── experiments/                # 10 experiment scripts (A3-F3)
│   │   ├── results/                    # JSON results (auto-generated)
│   │   ├── run_experiment.py           # CLI: --exp A3 --verbose
│   │   ├── run_b4_n10.py              # Standalone: Hessian at N=10
│   │   ├── run_f3_p1.py              # Standalone: Landscape p=1 vs p=2
│   │   ├── run_d1_regularized.py      # Standalone: D1 with dropout
│   │   ├── run_c1_n10.py             # Standalone: Physics loss at N=10
│   │   └── compare_results.py         # Cross-experiment comparison
│   ├── experiments_hamed_v7/           # V7 experiments (complete)
│   └── hooks/                          # Pre-commit hook scripts
│
├── tests/                              # pytest suite (126 tests)
│
├── documentation/
│   ├── v8/                             # V8 status + improvement techniques
│   ├── v7/                             # V7 results summary + quantum utility plan
│   ├── binnacles/                      # Experiment logs (11 files: N6, N10, V7, V8)
│   └── bibliography/                   # Curated + full + alternative references
│
├── .kiro/
│   ├── steering/                       # AI agent guidance files
│   ├── skills/quantum/                 # Quantum computing skill context
│   └── knowledge/                      # Project knowledge base
│
├── Makefile                            # Unified workflow targets
├── pyproject.toml                      # Ruff config + project metadata
├── .pre-commit-config.yaml             # 12 hooks (ruff, HVA guard, secrets...)
└── requirements.txt                    # Dependencies
```

## Quick Start

```bash
# Setup
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install

# Verify
make check-full              # lint + 126 tests + smoke test

# Run V8 experiments
python scripts/experiments_v8/run_experiment.py --list
python scripts/experiments_v8/run_experiment.py --exp B4 --verbose
python scripts/experiments_v8/compare_results.py --all

# Run full pipeline (notebooks)
make run-notebooks
```

## V8 Experiment Framework

The V8 framework (`scripts/experiments_v8/`) provides a modular system for running
noiseless simulation experiments with automatic logging, comparison, and reporting.

```bash
# Available experiments
python scripts/experiments_v8/run_experiment.py --list

# Run single or multiple
python scripts/experiments_v8/run_experiment.py --exp C1 --verbose
python scripts/experiments_v8/run_experiment.py --exp B4 D1 F1

# Compare against V6.1 baseline
python scripts/experiments_v8/compare_results.py --all
```

### V8 Key Findings

| Experiment | Finding | Thesis Impact |
|-----------|---------|:---:|
| A3 (Scaling) | h_min = 1.0 + 0.020·N^1.31 (R²=1.00); p=1 scales better | HIGH |
| B4 (Hessian) | HVA landscape has 0 saddle points at N=6 AND N=10 | HIGH |
| D1 (Weight-space) | MPNN gradient peaks detect h_c; dropout=0.1 makes it robust | HIGH |
| C3 (Sign canon.) | N=20 p=1 works; sign canonicalization unnecessary | HIGH |
| B2 (Freezing) | 2/4 params frozen at h≥1.5, 0% accuracy loss | MEDIUM |
| C1 (Physics loss) | +3.9% at N=6, -12.3% at N=10 (context-dependent) | MEDIUM |
| E4 (Longitudinal) | HVA p=2 is TFIM-specific (g>0 fails) | MEDIUM |
| F3 (Fluctuation) | No BPs; p=1 landscape simpler; fraction_near_gs predicts boundary | MEDIUM |
| F1 (DyPP) | Rejected: only 8-13% savings | LOW |
| B1 (Analytical) | Rejected: converges to wrong basin | LOW |

## Tech Stack

| Component | Tool | Version |
|-----------|------|---------|
| Quantum circuits | Qiskit | 2.4.x |
| Hardware runtime | qiskit-ibm-runtime | 0.46.x |
| Noisy simulation | qiskit-aer (MPS) | 0.17.x |
| ML predictor | PyTorch + PyTorch Geometric | 2.11 + 2.7 |
| Tensor networks | TeNPy | 1.1.x |
| Linting | Ruff | 0.11.x |
| Testing | pytest | 9.x |
| Git hooks | pre-commit (12 hooks) | 4.2.x |

## Validation Metrics

| Priority | Metric | Threshold | Hardware? |
|----------|--------|-----------|:---------:|
| 1 | ΔE / gap | < 5% | ✅ |
| 2 | ⟨Xᵢ⟩, ⟨ZᵢZᵢ₊₁⟩ errors | < 1e-2 | ✅ |
| 3 | Fidelity | ≥ 99.5% (noiseless only) | ❌ |
| 4 | ADAPT iterations | ≤ 2 | ✅ |

## Documentation

- **[V8 Status](documentation/v8/STATUS.md)** — Single source of truth for V8 experiments
- **[V7 Results](documentation/v7/RESULTS_SUMMARY_V61_V7.md)** — Complete V6.1/V7 summary
- **[Architecture](documentation/architectural_doc_es_en.md)** — System design (ES/EN)
- **[Thesis Guide](documentation/thesis-structure-guide.md)** — Chapter outline
- **[Bibliography](documentation/bibliography/bibliography_curated.md)** — Curated references
- **[Binnacles](documentation/binnacles/)** — All experiment logs

## Constraints (enforced by pre-commit)

- HVA only, never HEA. p ≤ 2 layers.
- Primitives V2 only (no deprecated Qiskit APIs)
- Fidelity threshold ≥ 0.93 in training data
- No secrets in commits (gitleaks)
- Conventional commits (commitizen)

---

*Franco Raineri — Universidad de Buenos Aires, 2026*
