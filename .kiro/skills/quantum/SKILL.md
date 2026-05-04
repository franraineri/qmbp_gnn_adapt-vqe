---
name: quantum-hva-thesis
description: Core rules and constraints for the Hybrid GNN-HVA Framework for Topological Phase Characterization thesis. Use when writing quantum circuits, building Hamiltonians, running VQE, or making architectural decisions.
---

# Quantum Computing & Condensed Matter — Core Rules

Expert in quantum computing, variational quantum algorithms, condensed matter physics, and hybrid classical-quantum architectures for the Master's Thesis: **Hybrid GNN-HVA Framework for Topological Phase Characterization**.

## Governing Principle (Mele et al., Nature Physics 2026)

1. **Depth truncation**: Non-unital noise truncates circuits to O(log n). ALL HVA circuits MUST be shallow: p ≤ 2 layers.
2. **Local observables only**: Characterize phases via ⟨Xᵢ⟩, ⟨ZᵢZᵢ₊₁⟩, local energy density. NEVER global state fidelity on hardware.
3. **Stable gradients**: Shallow circuits + local costs = no barren plateaus. GNN warm-start exploits this.

## Architectural Constraints (Non-Negotiable)

- **Ansatz**: ONLY HVA. NEVER HEA.
- **Depth**: p ≤ 2 layers.
- **Initial state**: |+⟩^N (`qc.h(range(n))`). MANDATORY.
- **Warm-start**: θ_opt(Hᵢ) seeds Hᵢ₊₁. Sweep DESCENDING h=2→0. Init with `np.random.uniform(-0.01, 0.01)`, never zeros.
- **AdaptVQE**: max_iterations ≤ 2.
- **Observables**: `SparsePauliOp`, local quantities only on hardware.
- **Fallbacks**: 2D → quasi-1D spin ladders. Noise → SPT phases.

## Qiskit 2.x Rules

| Do ✅ | Don't ❌ |
|-------|---------|
| `SparsePauliOp.from_sparse_list(...)` | `PauliSumOp`, `opflow` |
| `qiskit.primitives.StatevectorEstimator` | `qiskit.execute()`, `Aer.get_backend()` |
| `qiskit_ibm_runtime.EstimatorV2` | Primitives V1, `backend.run()` |
| `from qiskit_algorithms import ...` | `from qiskit.algorithms import ...` |
| `circuit.assign_parameters(theta)` | Manual parameter substitution |
| `generate_preset_pass_manager(optimization_level=2)` | `transpile()` |
| `result[0].data.evs` | Legacy result formats |

Forbidden: `qiskit.opflow`, `qiskit.algorithms` (old path), `PauliSumOp`, `WeightedPauliOperator`, `qiskit.execute()`, `Aer.get_backend()`, `backend.run()`, `transpile()`, Primitives V1.

## Pipeline

| Phase | Goal | Output |
|-------|------|--------|
| 1 | Classical ground truth (Exact Diag / DMRG) | (h,J) → ψ, local observables |
| 2 | HVA θ_opt via warm-start VQE | θ_opt dataset |
| 3 | GNN/MLP predictor (h → θ_pred) | Trained model |
| 4 | Hardware deployment (EstimatorV2) | Mitigated VQE results |

## Validation (pass/fail order)

1. ΔE/gap < 5% (primary)
2. ⟨Xᵢ⟩, ⟨ZᵢZᵢ₊₁⟩ errors < 1e-2
3. Fidelity ≥ 99.5% (noiseless only, never hardware)
4. ADAPT iterations ≤ 2

## Literature Validation (Phase 3 Architecture)

Our GNN/MLP warm-start approach is validated by three independent 2024-2026 papers:

- **NN-VQE** (Miao et al., PRApplied 2024): MLP h→θ for parameterized spin Hamiltonians. 20 training points, dropout regularization, active learning. Directly validates our PoC MLP design.
- **Qracle** (Zhang et al., 2025): GNN-based VQE parameter initializer. Unified Hamiltonian+ansatz graph encoding. Up to 64% fewer optimization steps. Validates our GNN scaling path.
- **Flow-VQE** (Zou et al., npj QI 2026): Generative normalizing flows for warm-start. Up to 50x acceleration. Alternative approach — we chose deterministic mapping (simpler, sufficient for smooth TFIM landscape).

Key takeaway: GNN-based initialization works best on physically structured Hamiltonians (spin systems), poorly on random circuits. Our spin-system focus is optimal.

## Current PoC (V6.0)

- 1D TFIM, N=6, HVA p=2, |+⟩^N
- **Modular architecture**: 9 Python modules under `src/poc/v6/`
- Phases 1-2: `src/poc/v6/poc_v6_phases1_2.ipynb` / Phases 3-4: `src/poc/v6/poc_v6_phases3_4.ipynb`
- Non-uniform h-grid: Δh=0.05 near critical region h∈[0.8,1.4], Δh=0.1 elsewhere (27 points)
- **MPNN predictor** (PyTorch Geometric GINConv + global_mean_pool) — replaces V4 MLP
- Fidelity filter ≥ 0.93, dropout=0.1
- **QRC fallback route**: fixed HVA reservoir + Rx(h) encoding + linear regression readout
- Dataset metadata: `cost_function="energy"`, `version="v6.0"` (prevents V5.x phase coupling failure)
- **Known limit**: HVA p=2 + |+⟩^N cannot express ferromagnetic ground state (h<1.0). Validated for h≥1.0.

### V6 Module Imports

```python
from src.poc.v6 import (
    HamiltonianBuilder, make_lattice, ClassicalSolver,
    HVACircuitBuilder, VQEOptimizer, HardwareDeployer,
    LatticeConfig, VQEConfig, GroundTruthResult, VQEResult, DeployResult,
    save_phase12_dataset, load_phase12_dataset,
)
from src.poc.v6.mpnn_predictor import MPNNPredictor, build_graph_dataset, train_mpnn
from src.poc.v6.qrc_pipeline import QRCPipeline
```

## IBM Connection Pattern

```python
import os
from qiskit_ibm_catalog import QiskitFunctionsCatalog
from qiskit_ibm_runtime import QiskitRuntimeService

ibm_token = os.environ.get("IBM_KEY")
ibm_instance = os.environ.get("IBM_INSTANCE_CRN")
backend_name = "ibm_torino"

service = QiskitRuntimeService(channel="ibm_quantum_platform", token=ibm_token, instance=ibm_instance)
catalog = QiskitFunctionsCatalog(instance=ibm_instance, token=ibm_token)
```
