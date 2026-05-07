# Hybrid GNN-HVA Framework for Topological Phase Characterization

## 🔬 Project Overview

This repository contains the codebase and theoretical framework for a Master's Thesis (TFM) in Quantum Computing and Condensed Matter Physics. The project aims to accelerate Variational Quantum Eigensolvers (VQE) to characterize Quantum Topological Phases (like Quantum Spin Liquids) in frustrated many-body systems.

To overcome the limitations of Noisy Intermediate-Scale Quantum (NISQ) devices—specifically the "barren plateau" problem and noise-induced truncation—we propose a **predictive hybrid architecture**: A classical Graph Neural Network (GNN) trained on Tensor Network data acts as an "Intelligent Warm-Start" to initialize a shallow, physics-informed quantum circuit (Hamiltonian Variational Ansatz - HVA).

## 📄 Theoretical Paradigm Shift

> *Mele, A. A., et al. "Noise-induced shallow circuits and the absence of barren plateaus" (Nature Physics, 2026).*

1. **Depth Truncation:** Non-unital noise truncates circuits to $\mathcal{O}(\log n)$. All HVA circuits MUST be shallow ($p \le 2$).
2. **Local Observables Only:** Global cost functions suffer barren plateaus under noise. We use $\langle X_i \rangle$, $\langle Z_i Z_{i+1} \rangle$ for phase characterization.
3. **Stable Gradients:** Shallow circuits + local costs = no barren plateaus. The GNN warm-start exploits this for instantaneous convergence.

## 🗺️ The 4-Phase Pipeline

### Phase 1: Classical Ground Truth Generation
- Exact Diagonalization (N < 15) or DMRG/TeNPy (N ≤ 40)
- Supports arbitrary lattice topologies: chain_1d, ladder, triangular, Kagome
- Output: $(h, J) \to \{\psi_{gs}, E_0, \text{gap}, \langle X_i \rangle, \langle Z_i Z_j \rangle\}$

### Phase 2: HVA Warm-Start VQE
- Hamiltonian Variational Ansatz (HVA), never HEA. $p \le 2$ layers.
- Descending sweep $h=2 \to 0$ with warm-start propagation
- Multi-start L-BFGS-B with diagnostic callbacks and trajectory logging

### Phase 3: MPNN Predictive Model
- Message Passing Neural Network (GINConv + global pooling) via PyTorch Geometric
- Lattice-agnostic: same model handles different topologies and system sizes
- Fidelity-filtered training data (≥ 93%) with energy-driven validation callbacks

### Phase 4: Dual-Route Deployment
- **Main route:** MPNN → θ_pred → HVA warm-start → AdaptVQE (max 2 iterations)
- **Fallback route:** Quantum Reservoir Computing (QRC) — fixed random HVA as reservoir, classical linear readout
- Phase classification via data-driven ⟨X⟩ = ⟨ZZ⟩ crossover (not hardcoded $h_c = 1.0$)

## 💻 Tech Stack

| Component | Tool | Version |
|-----------|------|---------|
| Quantum circuits | Qiskit | 2.4.x |
| Algorithms | qiskit-algorithms | 0.4.x |
| Hardware | qiskit-ibm-runtime | 0.46.x |
| ML predictor | PyTorch + PyTorch Geometric | 2.11 + 2.7 |
| Tensor networks | TeNPy | 1.1.x |
| Classical ML | scikit-learn | 1.8.x |
| Linting | Ruff | 0.11.x |
| Testing | pytest | 9.x |
| Git hooks | pre-commit | 4.2.x |

### Qiskit 2.x Rules (enforced by ruff + pre-commit)

| ✅ Do | ❌ Don't |
|-------|---------|
| `SparsePauliOp.from_sparse_list(...)` | `PauliSumOp`, `opflow` |
| `StatevectorEstimator` / `EstimatorV2` | `qiskit.execute()`, `Aer.get_backend()` |
| `from qiskit_algorithms import ...` | `from qiskit.algorithms import ...` |
| `generate_preset_pass_manager()` | `transpile()` |

## 📊 Validation Metrics

| Priority | Metric | Threshold | Hardware? |
|----------|--------|-----------|-----------|
| 1 | ΔE / gap | < 5% | ✅ |
| 2 | ⟨Xᵢ⟩, ⟨ZᵢZᵢ₊₁⟩ errors | < 1e-2 | ✅ |
| 3 | ΔE | < 1e-2 (aspirational) | ✅ |
| 4 | Fidelity | ≥ 99.5% (noiseless only) | ❌ |
| 5 | ADAPT iterations | ≤ 2 | ✅ |

### Achieved Results (40+ benchmark runs)

| Test Point | N=6 Checklist | N=10 Checklist | Status |
|------------|---------------|----------------|--------|
| h = 1.5 | **5/6** | **3/6** | ✅ Valid operating regime |
| h = 1.4 | 4–5/6 | 1–2/6 | ✅ N=6 valid, N=10 borderline |
| h = 1.25 | 2–3/6 | 1/6 | ⚠️ Physics limit (HVA p=2 ceiling) |

The h=1.25 ceiling is independently confirmed as a physics limit by Tripathi et al. (2026) — not a pipeline deficiency.

## 🔬 Error Mitigation Strategy (Phase 4 — IBM Torino)

| Technique | Method | Overhead | Source |
|-----------|--------|----------|--------|
| Inhomogeneous ZNE | Multiple qubit mappings → CES extrapolation | 3-5× circuits | Uvarov et al. 2024 |
| Learned DD | Optimized pulse sequences via Qiskit DD pass | Free | Pokharel et al. 2025 |
| TREX | Twirled readout error extinction | ~2× shots | IBM native |
| NN-enhanced ZNE | MLP fit instead of polynomial extrapolation | Minimal | Sun et al. 2025 |

## 🏗️ Project Structure

```
qmbp_gnn_adapt-vqe/
├── src/poc/v6/                    # Active codebase (modular V6.0)
│   ├── config.py                  # Shared dataclasses & constants
│   ├── hamiltonian_builder.py     # Lattice generators + SparsePauliOp construction
│   ├── classical_solver.py        # Exact diag + DMRG/TeNPy
│   ├── hva_builder.py             # HVA circuit construction (p ≤ 2 enforced)
│   ├── vqe_optimizer.py           # Multi-start L-BFGS-B with callbacks
│   ├── mpnn_predictor.py          # GINConv MPNN + training loop
│   ├── qrc_pipeline.py            # Quantum Reservoir Computing fallback
│   ├── hardware_deployer.py       # AdaptVQE + QRC dual-route deployment
│   ├── pipeline_utils.py          # Dataset save/load, metadata, locality checks
│   ├── poc_v6_phases1_2.ipynb     # Notebook: Phase 1-2 orchestration
│   └── poc_v6_phases3_4.ipynb     # Notebook: Phase 3-4 orchestration
├── scripts/                       # ALL executable scripts
│   ├── smoke_test.py              # Quick end-to-end validation (~7s)
│   ├── benchmark_v6.py            # Multi-run benchmark with binnacle logging
│   ├── run_notebooks.py           # Automated notebook execution with validation
│   └── hooks/check_hva_depth.py   # Pre-commit: p ≤ 2 enforcement
├── tests/                         # pytest suite (18 tests)
├── documentation/                 # Thesis docs, binnacle, bibliography
├── .pre-commit-config.yaml        # 12 hooks (ruff, nbstripout, HVA guard, etc.)
├── pyproject.toml                 # Ruff config + banned Qiskit APIs
├── Makefile                       # Unified workflow targets
└── requirements.txt               # Dependencies
```

## 🚀 Quick Start

```bash
# Setup
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install

# Verify everything works
make check-full          # lint + 18 tests + smoke test

# Run the PoC pipeline
make run-notebooks       # execute both notebooks with validation

# Or step by step:
make run-nb-12           # Phase 1-2: exact diag + VQE sweep
make run-nb-34           # Phase 3-4: MPNN training + deployment

# Benchmark
make benchmark           # 3 runs with different seeds

# Other targets
make help                # show all available targets
```

Data flows between notebooks via `phase1_phase2_tfim_N6_p2_v6.npz`. Run Phase 1-2 first.

## 📚 Documentation

* **[Project Summary (English)](documentation/qmbp_doc_summary_en.md)** — Physics problem, hybrid solution, implementation by phase, bibliography.
* **[Resumen del Proyecto (Español)](documentation/qmbp_doc_summary_es.md)** — Versión completa en español.
* **[Architectural Document (ES/EN)](documentation/architectural_doc_es_en.md)** — GNN data strategy, noise resilience, spin systems rationale, QPU execution analysis, computational scaling.
* **[Thesis Structure Guide](documentation/thesis-structure-guide.md)** — Chapter outline, framing guidelines, reviewer Q&A.
* **[Bibliography](documentation/bibliography.md)** — Complete APA reference list (24 sections, 70+ papers).
* **[Alternative Bibliography](documentation/alternative_bibliography.md)** — Alternative techniques and methodologies to consider for future work.
* **[V6 Changes](src/poc/v6/CHANGES_V6.md)** — What changed from V4/V5 and why.
* **[Binnacle N=6](documentation/binnacles/binnacle-N6.md)** — Complete N=6 experiment log (40+ runs, definitive).
* **[Binnacle N=10](documentation/binnacles/binnacle-N10.md)** — N=10 scaling experiments (active).

---

# 🇪🇸 Versión en Español

## Descripción

Arquitectura híbrida predictiva para caracterización de fases topológicas cuánticas. Una Red Neuronal de Paso de Mensajes (MPNN) clásica predice parámetros óptimos para un circuito cuántico superficial (HVA, $p \le 2$), permitiendo convergencia instantánea en dispositivos NISQ antes de que el ruido destruya la señal.

## Inicio Rápido

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pre-commit install
make check-full          # lint + tests + smoke test
make run-notebooks       # ejecutar notebooks con validación
make benchmark           # benchmark multi-seed
```

## Estructura del Pipeline

1. **Fase 1:** Diagonalización exacta / DMRG → ground truth clásico
2. **Fase 2:** VQE con warm-start descendente → θ_opt dataset
3. **Fase 3:** MPNN (PyTorch Geometric) → predictor h → θ_pred
4. **Fase 4:** Despliegue dual: AdaptVQE (principal) + QRC (fallback)

Documentación completa en [`documentation/`](documentation/).
