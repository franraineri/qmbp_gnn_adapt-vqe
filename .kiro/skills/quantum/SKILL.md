---
inclusion: always
name: quantum-hva-thesis
description: Core rules and constraints for the Hybrid GNN-HVA Framework for Topological Phase Characterization thesis. Use when writing quantum circuits, building Hamiltonians, running VQE, or making architectural decisions.
---

# GNN-HVA Thesis — Physics & Design Rules

Master's Thesis: **Hybrid GNN-HVA Framework for Topological Phase Characterization**.
Scope: noiseless ideal simulation (StatevectorEstimator). Hardware deployment = future work (Ch6).

## Thesis Objective (Canonical)

Demonstrate that GNN prediction + shallow HVA in a unified pipeline reduces quantum cost of phase classification by 29-500×, maintaining ΔE/gap < 5% within the valid operating regime, and formally document that regime's limits.

## Governing Principles

1. **Depth truncation** (Mele 2026): Non-unital noise truncates to O(log n). Hardware: p ≤ 3. Noiseless: p unrestricted (thesis uses p=1-4).
2. **Local observables only**: ⟨Xᵢ⟩, ⟨ZᵢZᵢ₊₁⟩, local energy density. NEVER global fidelity on hardware.
3. **Stable gradients**: Shallow HVA + local costs = no barren plateaus. GNN warm-start exploits this.

## Architectural Constraints

- **Ansatz**: ONLY HVA. NEVER HEA (Tripathi 2026 confirms HVA > EfficientSU2).
- **Initial state**: |+⟩^N (`qc.h(range(n))`). MANDATORY.
- **Warm-start**: θ_opt(Hᵢ) seeds Hᵢ₊₁. Sweep DESCENDING h=2→0. Init `np.random.uniform(-0.01, 0.01)`.
- **Known limit**: HVA p=2 + |+⟩^N cannot express ferromagnetic ground state (h<1.0).
- **h_min formula**: h_min(N,p) = 1.5 + 0.020·N^{1.31} (empirical, R²=1.0, N=4-200). NOT physics — circuit expressibility frontier. Exponent β=1.31 ≠ ν=1.

## Metrics (Section 5.1)

| Metric | Role | Threshold |
|--------|------|-----------|
| ΔE/gap | Primary success | < 5% = correct phase |
| \|ΔE\| | Absolute error | Stable ~0.003-0.033 across N |
| Fidelity | State quality (N≤20 only) | ≥ 0.96 |
| Pass rate (tasa de aprobación) | Fraction passing | — |
| Speedup | vs random-init VQE | 29-500× |

- "Deploy" is DEPRECATED → use "PassRate"
- ΔE is stable across N; ΔE/gap "improvement" at large N is artifact (gap grows as denominator)
- Fidelity unavailable for N>22 (MPS backend)

## Registered Models

| Name | params_per_layer | Status | CX budget (p=1 N=6) |
|------|:---:|---|:---:|
| tfim | 2 | Production | 10 CZ ✅ |
| tfim_longitudinal | 3 | Validated | 10 CZ ✅ |
| tfim_frustrated | 3 | Noiseless only | 27 CZ ❌ |
| tfim_bond_resolved | n_edges+n_qubits | Per-bond params | varies |
| heisenberg | 4 | HVA-limited (works h>3.5) | 30+ CZ ❌ |
| heisenberg_transverse | 4 | HVA-limited | 30+ CZ ❌ |
| xy | 4 | HVA-limited | 30+ CZ ❌ |
| kitaev | 3 | Implemented | 20 CZ ❌ |

**CX Budget Rule**: ZNE-viable ↔ new Hamiltonian term maps to single-qubit gate (RX, RZ).

## MPNN Architecture (UnifiedMPNN)

- GINConv (maximally expressive MPNN, Xu ICLR 2019)
- `norm_type="none"` MANDATORY (batch/layer norm breaks cross-N generalization)
- Residual skip connections + FiLM conditioning by h
- `gate_readout=True` (predict θ_zz from gate nodes directly)
- Bond-resolved mode: per-bond θ_ij parameters (79D for N=10 ladder)

## Qiskit 2.x Rules

| Do ✅ | Don't ❌ |
|-------|---------|
| `SparsePauliOp.from_sparse_list(...)` | `PauliSumOp`, `opflow` |
| `qiskit.primitives.StatevectorEstimator` | `qiskit.execute()`, `Aer.get_backend()` |
| `qiskit_ibm_runtime.EstimatorV2` | Primitives V1, `backend.run()` |
| `from qiskit_algorithms import ...` | `from qiskit.algorithms import ...` |
| `circuit.assign_parameters(theta)` | Manual param substitution |
| `generate_preset_pass_manager(optimization_level=2)` | `transpile()` |
| `result[0].data.evs` | Legacy result formats |

Forbidden: `qiskit.opflow`, `PauliSumOp`, `WeightedPauliOperator`, `qiskit.execute()`, `Aer.get_backend()`, `backend.run()`, `transpile()`, Primitives V1.

## Pipeline Phases

| Phase | Goal | Output |
|-------|------|--------|
| 1 | Classical ground truth (ED / DMRG) | (h,J) → E₀, gap, ψ |
| 2 | HVA θ_opt via warm-start VQE (descending h-sweep) | θ_opt dataset |
| 3 | MPNN predictor (graph → θ_pred) | Trained model |
| 4 | Zero-shot prediction at unseen h/N | ΔE/gap < 5% within h_min |

## Module Dependency DAG

```
utils → models → solvers, circuits → execution → optimizers
models, execution, solvers, circuits → predictors
solvers, optimizers, predictors, analysis → pipeline
pipeline, analysis → framework
```

## Key Physics Decisions

1. **Non-uniform h-grid**: Δh=0.05 near critical region h∈[0.8,1.4], Δh=0.1 elsewhere
2. **Fidelity filter ≥ 0.93** for VQE training data quality
3. **Bond-resolved** mode: per-edge ZZ parameters allow topology-aware prediction (vs 2 global θ)
4. **Cross-N generalization**: MPNN trained on N=4-10 predicts N=20-100 (zero-shot)
5. **Multi-topology (MT)**: single model trained on all topologies (FiLM + residual)
6. **h=1.25 ceiling is physics**: HVA p=2 struggles with entanglement entropy at criticality (Tripathi 2026)

## Hardware Deployment (Future Work — Chapter 6)

### Validated Techniques
- **PEA-ZNE**: Primary amplifier (R²=0.998, ΔE/gap=2.07%). Gate-folding inferior.
- **QESEM**: First run SUCCESS — 0.71% ΔE/gap on ibm_kingston (N=10 heavy_hex h=4.0, 428s QPU)
- **Affine correction**: Simple clip to [E_ground, E_upper]. Soft interpolation was buggy (614× amplification).
- **Inhomogeneous ZNE**: Multiple layouts → different CES → linear extrapolation
- **GNN-QEM**: ML-based post-ZNE correction (complementary to QESEM)

### Hardware Rules
- ZNE budget: p=1 N=10 ≈ 18 CX → viable. N=14 → 69 CX → non-perturbative.
- Shots: ≥8192 (SNR > 1 for ⟨X⟩ signal ~8e-3 at N=10)
- PEA is ONLY viable ZNE for circuits with SWAP overhead
- Layout selection: VF2 SWAP-free preferred, BFS fallback

### IBM Connection
```python
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_catalog import QiskitFunctionsCatalog
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=os.environ["IBM_KEY"], instance=os.environ["IBM_INSTANCE_CRN"])
```

## Quench Dynamics (Extension — Quantum Advantage Direction)

GNN prepares |ψ₀(h₁)⟩ → Trotter evolution under H(h₂) → volume-law entanglement → classical fails → QPU needed.

Key insight: preparation is cheap (GNN). Evolution is where QPU adds value. Complementary to IBM+Qedma paradigm (Kicked TFIM, July 2026).

Target: heavy-hex N=50+, quench crossing h_c → unexplored territory.

## Literature Validation

| Finding | Source |
|---------|--------|
| GNN > CNN by 36% for circuits | Meng 2025 |
| GINConv = WL-maximal | Xu ICLR 2019 |
| HVA > HEA on TFIM | Tripathi 2026 |
| h=1.25 ceiling = physics | Tripathi 2026 |
| N/2 layers for thermo limit | Sumeet 2025 |
| Noise broadens crossover | Sharma 2026 |
| QESEM > ZNE accuracy | Qedma 2025 |
| IBM quantum advantage (Kicked TFIM) | arXiv:2607.24937 |

## State of the Art: Classical & Quantum Limits (Aug 2026)

Full reference: `internal/documentation/thesis/estado_del_arte_materia_condensada.md`

### Classical Method Limits

| Method | Max Size | Where it fails |
|--------|----------|----------------|
| ED (Lanczos) | ~40-50 spins | Exponential 2^N memory |
| DMRG 1D | ~1000+ sites (D~4000) | Polynomial if gapped; fails at criticality/dynamics |
| DMRG 2D (cylinder) | ~100-200 (width ≤12) | D grows exponentially with cylinder width |
| fPEPS 2D | D=28 > DMRG m=32k | D^10 contraction cost; limited to equilibrium |
| QMC (no sign) | 10,368 sites | Only non-frustrated / half-filling |
| QMC (sign problem) | ~100-300 sites | Signal decays exponentially with β·N |
| NQS/VMC (frustrated 2D) | 42×42 = 1,764 sites | Training instability, validation hard |
| AFQMC (solids) | Thermodynamic limit O(N³) | Phaseless bias |

Key takeaway: **ground states satisfy area-law** → classically tractable for gapped 1D and modest 2D. Dynamics and frustrated 2D at scale remain hard.

### Quantum Hardware Results (2026)

| System | Qubits | Result | Ref |
|--------|:---:|--------|-----|
| IBM+Qedma Kicked TFIM | 74 | Prethermal oscillations (quantum advantage) | arXiv:2607.24937 |
| IBM Kagome thermal states | 139 | Frustrated spin thermal preparation | arXiv:2605.26245 |
| Quantinuum 2D Fermi-Hubbard quench | ~60 | Volume-law dynamics | arXiv:2510.26300 |
| Q-CTRL 1D Fermi-Hubbard | 120 | 3000× claim (partially refuted by GPU TDVP) | arXiv:2605.04025 |

### Conditions for Quantum Advantage

Three necessary conditions (from evidence 2024-2026):
1. **Volume-law entanglement** — ground states (area-law) are always classically solvable; dynamics/quench generate volume-law
2. **Dimensionality >1D** — 1D always vulnerable to GPU tensor networks (TDVP χ=60k on H200)
3. **Observable sensitivity** — measured quantity must depend on components that classical methods truncate

### Where Our Project Sits

- Our regime (ground states, N=4-200, area-law) is classically tractable — we demonstrate methodology and efficiency, not quantum advantage
- Heavy-hex is quasi-2D (has loops) → BP loses guarantees → potential sweet spot
- The GNN enables O(1)-cost state preparation that could feed into protocols where advantage exists (quench dynamics, Floquet evolution)
- Possible value directions: (a) quench post-preparation, (b) QPU verification of DMRG in grade-F regimes, (c) frustrated 2D where classical methods disagree

### IBM+Qedma vs This Project

| | IBM+Qedma | GNN-HVA |
|-|-----------|---------|
| Problem | Dynamics (time evolution) | Ground states (static) |
| Entanglement | Volume law (grows with t) | Area law (bounded) |
| Circuit | Fixed (Floquet) | Parametrized (HVA, optimized) |
| ML role | None | Central (29-500× speedup) |
| Complementarity | Needs expensive state preparation | Provides it at zero cost |

The bridge: GNN-HVA prepares |ψ₀(h)⟩ instantly → Floquet/Trotter protocol evolves → QPU explores dynamics where classical fails. Preparation cost = bottleneck for others; free for us.

## Code Rules

- ALWAYS import and reuse existing code. NEVER copy-paste or duplicate.
- Dataset metadata: `cost_function="energy"`, `version="v6.0"`
- Non-uniform h-grid via `generate_h_grid()` (dense near h_c)
- Descending sweep h=2→0 for warm-start continuity
