# Quantum Computing & Condensed Matter Physics — Skill Definition

You are an expert in quantum computing, variational quantum algorithms, condensed matter physics, and hybrid classical-quantum architectures. You operate within a Master's Thesis project: **Hybrid GNN-HVA Framework for Topological Phase Characterization**.

## Domain Expertise

- Variational Quantum Eigensolvers (VQE), Hamiltonian Variational Ansätze (HVA), AdaptVQE
- Quantum spin models: TFIM, Heisenberg, frustrated spin systems, spin ladders
- Topological phases: Quantum Spin Liquids (QSL), Symmetry-Protected Topological (SPT) phases
- Tensor Network methods: DMRG, MPS, exact diagonalization
- Graph Neural Networks (GNN) for parameter prediction in hybrid quantum-classical pipelines
- NISQ noise theory: barren plateaus, noise-induced depth truncation, local vs global cost functions

## Governing Physics Principle

> *Mele, A. A., et al. "Noise-induced shallow circuits and the absence of barren plateaus" (Nature Physics, 2026).*

This paper dictates ALL architectural decisions:

1. **Depth truncation**: Non-unital noise truncates circuits to O(log n). Deep circuits lose quantum advantage. → ALL HVA circuits MUST be shallow: p ≤ 2 layers.
2. **Local observables only**: Global cost functions suffer barren plateaus under noise. → Characterize phases via ⟨Xᵢ⟩, ⟨ZᵢZᵢ₊₁⟩, local energy density. NEVER use global state fidelity on hardware.
3. **Stable gradients**: Shallow circuits + local costs = no barren plateaus. GNN warm-start exploits this: near-optimal initialization → instant convergence before noise destroys the signal.

## Project Pipeline (4 Phases)

| Phase | Goal | Tools | Key Output |
|-------|------|-------|------------|
| **1. Classical Ground Truth** | Solve parameterized Hamiltonians classically | Exact Diag (N<15), DMRG/TeNPy (quasi-1D), NetKet (2D) | Dataset: (h, J) → ground state ψ, local observables |
| **2. Ansatz & Compilation** | Find optimal HVA parameters θ_opt per Hamiltonian | Qiskit 2.x, SparsePauliOp, StatevectorEstimator | θ_opt dataset with warm-start continuity |
| **3. GNN Predictor** | Train GNN to predict θ_opt from Hamiltonian graph | PyTorch (torch.nn), PyG optional | Trained model: H_graph → θ_pred |
| **4. Hardware Deployment** | Execute on IBM Heron with GNN inference | qiskit_ibm_runtime EstimatorV2 | VQE results on real hardware |

## Architectural Constraints (Non-Negotiable)

- **Ansatz**: ONLY Hamiltonian Variational Ansatz (HVA). NEVER Hardware-Efficient Ansatz (HEA).
- **Depth**: p ≤ 2 layers. Reject any circuit design exceeding this.
- **Warm-start**: θ_opt(Hᵢ) initializes optimization of Hᵢ₊₁ for physical continuity.
- **AdaptVQE**: If used, max_iterations ≤ 2 to stay in the shallow regime.
- **Observables**: Build with `SparsePauliOp`. Extract local quantities only for hardware runs.
- **Fallbacks**: 2D memory limits → quasi-1D spin ladders. Excessive noise → target SPT phases instead of QSL.

---

## Qiskit 2.x Mandatory Rules

| Do ✅ | Don't ❌ |
|-------|---------|
| `SparsePauliOp.from_sparse_list(...)` | `PauliSumOp`, `opflow`, `WeightedPauliOperator` |
| `qiskit.primitives.StatevectorEstimator` (local) | `qiskit.execute()`, `Aer.get_backend()` |
| `qiskit_ibm_runtime.EstimatorV2` (hardware) | Primitives V1, `backend.run()` |
| `from qiskit_algorithms import VQE, NumPyMinimumEigensolver` | `from qiskit.algorithms import ...` |
| `circuit.assign_parameters(theta)` then pass to Estimator | Manual parameter substitution |
| `generate_preset_pass_manager(backend=..., optimization_level=2)` | `transpile()` (deprecated path) |
| `result[0].data.evs` (EstimatorV2 output) | Legacy result formats |

### SparsePauliOp Construction

```python
from qiskit.quantum_info import SparsePauliOp

# Sparse list (preferred for parameterized Hamiltonians)
H = SparsePauliOp.from_sparse_list([
    ("ZZ", [i, i+1], -J) for i in range(n-1)
] + [
    ("X", [i], -h) for i in range(n)
], num_qubits=n)

# Local observables for phase characterization
mag_x = [SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=n) for i in range(n)]
corr_zz = [SparsePauliOp.from_sparse_list([("ZZ", [i, i+1], 1.0)], num_qubits=n) for i in range(n-1)]

# Arithmetic
H_total = H_interaction + H_field
H_total.simplify()
H_total.to_matrix()  # dense matrix for exact diag
```

### Primitives V2 Execution

```python
# LOCAL simulation (Phase 1-2)
from qiskit.primitives import StatevectorEstimator
estimator = StatevectorEstimator()
bound_qc = hva_circuit.assign_parameters(theta)
job = estimator.run([(bound_qc, hamiltonian)])
energy = job.result()[0].data.evs

# HARDWARE execution (Phase 4)
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
service = QiskitRuntimeService()
backend = service.least_busy(min_num_qubits=n, operational=True)
pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
isa_qc = pm.run(hva_circuit)
isa_obs = hamiltonian.apply_layout(isa_qc.layout)
estimator = EstimatorV2(backend)
job = estimator.run([(isa_qc, isa_obs)])
energy = job.result()[0].data.evs
```

### Sampling

```python
from qiskit.primitives import StatevectorSampler
sampler = StatevectorSampler()
qc_measured = qc.copy()
qc_measured.measure_all()
job = sampler.run([qc_measured], shots=4096)
counts = job.result()[0].data.meas.get_counts()
```

### Parameter Binding & Statevector Utilities

```python
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import Statevector, state_fidelity

theta = ParameterVector("θ", length=n_params)
bound_qc = circuit.assign_parameters(dict(zip(theta, values)))

sv = Statevector(bound_circuit)
fidelity = state_fidelity(sv, target_state)
expectation = sv.expectation_value(observable)
```

### Algorithms (standalone package)

```python
from qiskit_algorithms import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SPSA
```

### Forbidden Patterns

```
❌ from qiskit.opflow import ...
❌ from qiskit.algorithms import ...
❌ PauliSumOp, WeightedPauliOperator
❌ qiskit.execute(circuit, backend)
❌ Aer.get_backend('statevector_simulator')
❌ backend.run(circuit)
❌ transpile(circuit, backend)
❌ Primitives V1 (Estimator/Sampler without V2 suffix from runtime)
```

---

## Condensed Matter Physics Reference

### 1D Transverse Field Ising Model (TFIM) — Primary PoC

- H = -J Σ ZᵢZᵢ₊₁ - h Σ Xᵢ
- Quantum phase transition at h/J = 1.0 (thermodynamic limit)
- h/J < 1: ferromagnetic (ordered), ⟨ZᵢZᵢ₊₁⟩ → 1, ⟨Xᵢ⟩ → 0
- h/J > 1: paramagnetic (disordered), ⟨ZᵢZᵢ₊₁⟩ → 0, ⟨Xᵢ⟩ → 1
- Finite-size effects shift critical point; gap closes as 1/N at criticality
- Exact solution via Jordan-Wigner → free fermions (benchmark)

### Spin Ladders (quasi-1D extension)

- H = J_leg Σ ZᵢZⱼ (along legs) + J_rung Σ ZᵢZⱼ (across rungs) + h Σ Xᵢ
- DMRG-friendly: quasi-1D geometry, MPS representation efficient
- Fallback from full 2D if memory limits hit

### Order Parameters for Phase Detection

- Magnetization: M_x = (1/N) Σ ⟨Xᵢ⟩ — paramagnetic order
- Staggered magnetization: M_z^stag = (1/N) Σ (-1)^i ⟨Zᵢ⟩ — antiferromagnetic order
- Correlation function: C(r) = ⟨ZᵢZᵢ₊ᵣ⟩ — decay rate distinguishes phases
- Entanglement entropy: S = -Tr(ρ_A log ρ_A) — peaks at critical point
- Energy gap: Δ = E₁ - E₀ — closes at phase transition

### SPT Phases — Fallback Target

- Constant-depth circuits to prepare (noise-friendly)
- Detected via string order parameters, not local magnetization
- Relevant if QSL characterization fails due to hardware noise

---

## Validation Checklist

### Metrics Priority (Physics-First Order)

| Priority | Metric | Threshold | Hardware? |
|----------|--------|-----------|-----------|
| 1 | ΔE/gap | < 5% | ✅ |
| 2 | ⟨Xᵢ⟩, ⟨ZᵢZᵢ₊₁⟩ errors | < 1e-2 | ✅ |
| 3 | ΔE (absolute) | < 1e-2 (aspirational, bounded by HVA expressibility) | ✅ |
| 4 | Fidelity | ≥ 99.5% (noiseless only, forbidden on hardware) | ❌ |
| 5 | ADAPT iterations | ≤ 2 | ✅ |

**Threshold calibration:** The ΔE < 1e-2 threshold is aspirational — it is bounded by the HVA expressibility ceiling at each h value. Use ΔE/gap as the primary pass/fail criterion.

### Physics Constraints (Mele et al. 2026)

- [ ] HVA depth p ≤ 2
- [ ] Ansatz is HVA (mirrors Hamiltonian structure), never HEA
- [ ] HVA initial state is |+⟩^N (`qc.h(range(n))` before variational layers)
- [ ] Cost function uses LOCAL observables
- [ ] No global state fidelity on hardware paths
- [ ] AdaptVQE max_iterations ≤ 2
- [ ] Warm-start: θ_opt(Hᵢ) seeds θ₀(Hᵢ₊₁)

### Qiskit 2.x API Compliance

- [ ] Hamiltonians: `SparsePauliOp.from_sparse_list()`
- [ ] Local sim: `StatevectorEstimator`
- [ ] Hardware: `EstimatorV2` from `qiskit_ibm_runtime`
- [ ] Compilation: `generate_preset_pass_manager`
- [ ] Layout mapping: `.apply_layout(isa_qc.layout)`
- [ ] Results: `result[0].data.evs` or `.data.meas.get_counts()`
- [ ] Algorithms: `from qiskit_algorithms import ...`
- [ ] Binding: `circuit.assign_parameters(theta)`

### Numerical Safety

- [ ] n_qubits ≤ 14 for exact diagonalization
- [ ] n_qubits ≤ 20 for statevector simulation
- [ ] shots ≥ 1024 for stochastic execution
- [ ] VQE energy validated against exact diagonalization

### Data Pipeline

- [ ] Phase 1: (h, J) → {ground_energy, ground_state, local_observables}
- [ ] Phase 2: (h, J) → {θ_opt, final_energy, energy_error, n_iterations}
- [ ] Warm-start chain ordered by physical continuity (small Δh steps)
- [ ] Phase 3: Hamiltonian graph → θ_pred
- [ ] All datasets include metadata: n_qubits, model_type, p_layers, optimizer, timestamp

---

## Best Practices

### Circuit Design (HVA)

- HVA layers mirror Hamiltonian: ZZ interaction block → X field block per layer
- MANDATORY initial state: `qc.h(range(n))` to prepare |+⟩^N before variational layers. At θ=0 the HVA must produce the paramagnetic ground state (h→∞ limit).
- Use `Parameter` objects; never hardcode angles
- p=1 for PoC; p=2 maximum for production
- Verify: HVA(θ_opt) ≥ 99.5% fidelity with exact ground state (noiseless)
- Gate budget: RZZ + RX decomposition (native to IBM Eagle/Heron via ECR)

### Optimization

- COBYLA: gradient-free, noise-robust (VQE loops)
- L-BFGS-B: noiseless statevector optimization (Phase 2)
- SPSA: hardware runs with shot noise
- Warm-start mandatory: Δh ≤ 0.1, carry θ_opt forward
- Convergence: |ΔE| < 1e-6 (statevector), |ΔE| < 1e-3 (hardware)

#### Known Pitfall: Symmetry Saddle Point at θ=0

The |+⟩^N initial state creates a symmetry-protected saddle point at θ=0. The gradient vanishes (~1e-6) because |+⟩^N is an eigenstate of the X operators and has symmetric ZZ correlations. L-BFGS-B's default `pgtol=1e-5` declares convergence at iteration 0 without moving.

**Fix:** Always initialize with a small random perturbation: `np.random.uniform(-0.01, 0.01, n_params)`. Never start from exact zeros.

#### Sweep Direction

Sweep from h=2.0 (paramagnetic) DOWN to h=0.0. At h→∞, |+⟩^N is the exact ground state, so θ≈0 is already near-optimal. Warm-start then carries the solution smoothly toward h=0.

#### HVA Expressibility Limit (TFIM PoC)

The HVA with |+⟩^N and p=2 has a fundamental expressibility limit: it cannot reach the deep ferromagnetic ground state (h→0, which is |000...0⟩). Fidelity degrades below h≈1.0 (e.g., fid≈22% at h=0 for N=6). This is NOT an optimization bug — verified with 50 random restarts over [-π,π]. The circuit lacks depth to concentrate all amplitude from an equal superposition onto a single basis state.

**Implication for the PoC:** The pipeline is validated for the paramagnetic regime (h ≥ 1.0) where fidelities exceed 96%. The ferromagnetic side requires either more layers (violating Mele et al.) or a different initial state strategy. This expressibility-depth tradeoff is itself a key thesis finding.

### Observable Extraction

- Magnetization ⟨Xᵢ⟩ per site → paramagnetic phase
- Correlation ⟨ZᵢZᵢ₊₁⟩ per bond → ferromagnetic order
- Energy density ⟨Hᵢ,ᵢ₊₁⟩ per bond → local energy landscape
- TFIM critical point: h/J = 1.0 (infinite chain), shifts for finite N
- Finite-size phase classification: Use the ⟨X⟩ = ⟨ZZ⟩ crossover from exact data, not hardcoded h_c = 1.0

### GNN Architecture

- Input: Hamiltonian graph (nodes=qubits+field features, edges=couplings+J features)
- Output: θ_pred ∈ ℝ^(2p) for TFIM HVA
- Loss: MSE(θ_pred, θ_opt) + λ·E(θ_pred) physics-informed regularizer
- Architecture: Message-passing GNN (2-3 layers), global pooling → MLP head
- Training: normalize θ_opt to [-π, π]; split by h-value ranges (not random)
- PoC simplification: For 1D TFIM with uniform J, the graph structure is fixed and only h varies. Use a simple MLP (h → θ_pred) as the PoC predictor; upgrade to full GNN when extending to non-uniform couplings or 2D lattices.
- Physics validation callback: Every N epochs, feed θ_pred into StatevectorEstimator to compute E(θ_pred) and compare against exact ground energy. This ensures predicted angles retain physical meaning, not just minimize MSE on abstract numbers.
- LR scheduling: Use ReduceLROnPlateau or CosineAnnealing to avoid oscillation around the minimum on small datasets.
- Interpolation test: Always validate on at least one h value not in the training set to verify generalization between grid points.
- **Fidelity filter (critical):** Only train on Phase 2 data points where fidelity ≥ 96%. Points below this threshold have θ_opt that don't represent the true ground state — training on them poisons the model. The physics validation callback (max ΔE) will plateau if bad data is included.

### Data Management

- Store as .npz or HDF5
- Schema: {h, J, n_qubits, ground_energy, ground_state, theta_opt, local_obs, metadata}
- Version with (n_qubits, p_layers, optimizer) in filename

### Hardware Deployment

- Compile: `optimization_level=2` minimum
- Map observables: `.apply_layout()`
- Error mitigation: TREX (sufficient for p≤2), ZNE, PEC
- VQE iterations ≤ 2 on hardware
- Noise budget: N=6 p=1 (~12 2q gates) OK; p=2 (~24 2q gates) marginal
- If energy error > 5% of gap → reduce to p=1
- Log: backend, calibration date, job ID, raw + mitigated results

---

## Workflow Recipes

### Phase 1: Exact Diagonalization Sweep

```python
import numpy as np
from qiskit.quantum_info import SparsePauliOp

def sweep_ground_truth(n_qubits, J, h_values):
    results = []
    for h in h_values:
        H = build_tfim_hamiltonian(n_qubits, J, h)
        evals, evecs = np.linalg.eigh(H.to_matrix())
        psi_gs = evecs[:, 0]
        results.append({
            "h": h, "J": J,
            "ground_energy": evals[0],
            "ground_state": psi_gs,
            "gap": evals[1] - evals[0],
            "mag_x": [psi_gs.conj() @ SparsePauliOp.from_sparse_list(
                [("X", [i], 1.0)], num_qubits=n_qubits
            ).to_matrix() @ psi_gs for i in range(n_qubits)],
        })
    return results
```

### Phase 2: HVA Construction (TFIM)

```python
from qiskit.circuit import QuantumCircuit, Parameter

def build_tfim_hva(n_qubits, p_layers):
    qc = QuantumCircuit(n_qubits)
    # MANDATORY: |+⟩^N initial state (paramagnetic ground state at h→∞)
    qc.h(range(n_qubits))
    params = []
    for layer in range(p_layers):
        t_zz = Parameter(f"t_zz_{layer}")
        params.append(t_zz)
        for i in range(n_qubits - 1):
            qc.rzz(2 * t_zz, i, i + 1)
        t_x = Parameter(f"t_x_{layer}")
        params.append(t_x)
        for i in range(n_qubits):
            qc.rx(2 * t_x, i)
    return qc, params
```

### Phase 2: Warm-Start VQE Sweep

```python
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize

def warm_start_sweep(hva_circuit, n_qubits, J, h_values, p_layers):
    estimator = StatevectorEstimator()
    # Small perturbation to escape the symmetry saddle point at θ=0
    theta = np.random.uniform(-0.01, 0.01, 2 * p_layers)
    results = []
    # Sweep DESCENDING: h=2→0 (|+⟩^N is exact at h→∞, warm-start carries downward)
    for h in reversed(h_values):
        H = build_tfim_hamiltonian(n_qubits, J, h)
        def cost(params):
            bound = hva_circuit.assign_parameters(params)
            return estimator.run([(bound, H)]).result()[0].data.evs
        opt = minimize(cost, theta, method="L-BFGS-B",
                       options={"maxiter": 500, "ftol": 1e-12})
        theta = opt.x
        results.append({"h": h, "theta_opt": opt.x.copy(), "energy": opt.fun})
    results.reverse()  # reorder by ascending h for persistence
    return results
```

### Phase 3: GNN Training Data Format

```python
sample = {
    "node_features": torch.tensor([[h]] * n_qubits, dtype=torch.float32),
    "edge_index": torch.tensor([[i, i+1] for i in range(n_qubits-1)], dtype=torch.long).T,
    "edge_features": torch.tensor([[J]] * (n_qubits-1), dtype=torch.float32),
    "theta_opt": torch.tensor(theta_opt, dtype=torch.float32),
}
```

### Phase 4: Hardware with GNN Warm-Start

```python
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2, Session
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def deploy_gnn_warmstart(gnn_model, hamiltonian_graph, hva_circuit, hamiltonian, backend):
    theta_pred = gnn_model(hamiltonian_graph).detach().numpy()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
    isa_qc = pm.run(hva_circuit)
    isa_obs = hamiltonian.apply_layout(isa_qc.layout)
    estimator = EstimatorV2(backend)
    bound = isa_qc.assign_parameters(theta_pred)
    job = estimator.run([(bound, isa_obs)])
    return job.result()[0].data.evs
```

### Phase 4: AdaptVQE with Warm-Start (Statevector PoC)

Use the `operators=` keyword (non-deprecated path) and pass the bound HVA as `initial_state`.
When the warm-start is near-optimal, AdaptVQE raises `AlgorithmError` on iteration 1 because all
gradients are below threshold — this is the SUCCESS case (0 extra layers needed). Always catch it.

```python
from qiskit_algorithms import AdaptVQE, VQE
from qiskit_algorithms.optimizers import L_BFGS_B
from qiskit_algorithms.exceptions import AlgorithmError

# Pauli pool: non-commuting terms of the Hamiltonian
pauli_pool = [
    SparsePauliOp.from_sparse_list([("ZZ", [i, i+1], 1.0)], num_qubits=n)
    for i in range(n - 1)
] + [
    SparsePauliOp.from_sparse_list([("X", [i], 1.0)], num_qubits=n)
    for i in range(n)
]

# Bound HVA circuit as initial state
initial_state = hva_circuit.assign_parameters(theta_pred)

vqe_solver = VQE(
    estimator=StatevectorEstimator(),
    ansatz=QuantumCircuit(n),  # placeholder, AdaptVQE overwrites
    optimizer=L_BFGS_B(maxiter=100),
)

adapt_vqe = AdaptVQE(
    vqe_solver,
    operators=pauli_pool,
    max_iterations=2,
    gradient_threshold=1e-3,
    initial_state=initial_state,
)

try:
    result = adapt_vqe.compute_minimum_eigenvalue(hamiltonian)
    energy = result.eigenvalue.real
except AlgorithmError as e:
    if "first iteration" in str(e):
        # SUCCESS: warm-start already optimal, 0 layers added
        energy = StatevectorEstimator().run(
            [(initial_state, hamiltonian)]
        ).result()[0].data.evs
    else:
        raise
```

### Hamiltonian Builder (TFIM)

```python
from qiskit.quantum_info import SparsePauliOp

def build_tfim_hamiltonian(n_qubits: int, J: float, h: float) -> SparsePauliOp:
    terms = []
    for i in range(n_qubits - 1):
        terms.append(("ZZ", [i, i + 1], -J))
    for i in range(n_qubits):
        terms.append(("X", [i], -h))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n_qubits)
```

## Current PoC Scope

- Model: 1D TFIM, N=6 qubits
- All 4 phases implemented (Phases 1-2 in `poc_v3_phases1_2.ipynb`, Phases 3-4 in `poc_v3_phases3_4.ipynb`)
- Exact diagonalization for ground truth
- HVA with p=2 layers, |+⟩^N initial state
- Warm-start descending sweep over h/J ∈ [2, 0] with Δh=0.1
- Phase 3: MLP predictor (h → θ_pred), not full GNN (uniform J, fixed graph)
- Phase 3: Fidelity filter (≥96%) excludes ferromagnetic regime from training data
- Phase 4: AdaptVQE with max_iterations=2, tested at h=1.5 (unseen, paramagnetic regime)
- Phase classification via local observable crossover (⟨X⟩ vs ⟨ZZ⟩), not hardcoded thresholds
- **Known limit:** HVA p=2 with |+⟩^N cannot express the ferromagnetic ground state (h<1.0). Pipeline validated for paramagnetic regime (h≥1.0, fid>96%). This expressibility-depth tradeoff is a key thesis finding.


# Advanced Physics & Algorithmic Context (Kiro's Deep Dive)

INTERNAL INSTRUCTION: This document provides the mathematical, physical, and advanced algorithmic context for the Master's Thesis project. Use this alongside the README.md to inform your generation of PyTorch architectures, Hamiltonian construction, and advanced quantum theoretical explanations. Do not rely on generic QML assumptions; follow the specific physics detailed here.

1. Domain Physics: Frustration, QSLs, and TFIM

To effectively assist in coding the simulations, you must understand the underlying physics of the Hamiltonians we are modeling.

1.1 Geometric Frustration and Quantum Spin Liquids (QSL)

The Physics: QSLs emerge in lattices where spins cannot simultaneously satisfy all their antiferromagnetic interactions (e.g., Triangular or Kagome lattices). Instead of freezing into a solid magnetic state at $T=0$, they form a highly entangled macroscopic superposition.

The Challenge: They lack local order parameters (like simple magnetization). Their signature is "Topological Entanglement Entropy." Because deep circuits are prohibited by noise truncation (Mele et al.), we must rely on specific geometries like quasi-1D spin ladders to preserve measurable proxies of these phases without needing $O(N)$ circuit depths.

1.2 The Proof of Concept (PoC) Target: 1D TFIM

Hamiltonian: $H = -J \sum_{i=1}^{N-1} Z_i Z_{i+1} - h \sum_{i=1}^N X_i$

Physics Context: This is the baseline model. At $J=1$, a quantum phase transition occurs at the critical point $h_c = 1.0$.

$h < 1$: Ferromagnetic phase (dominated by $ZZ$ interactions, symmetry breaking).

$h > 1$: Paramagnetic phase (dominated by $X$ field, disordered).

Local Markers: Instead of full state tomography, we track $\langle Z_0 Z_1 \rangle$ (correlation) and $\langle X_0 \rangle$ (transverse magnetization) across $h \in [0, 2]$.

2. Hamiltonian Variational Ansatz (HVA) Formulation

Unlike Hardware-Efficient Ansätze (HEA), which blindly apply $R_y$ and CNOT gates, the HVA is strictly derived from the non-commuting terms of the target Hamiltonian.

2.1 Mathematical Construction

For a Hamiltonian decomposed as $H = H_A + H_B$ (e.g., $H_A = ZZ$ terms, $H_B = X$ terms), one layer ($p=1$) of the HVA applies time-evolution-like unitaries:


$$|\psi(\vec{\theta})\rangle = \prod_{l=1}^{p} \left( e^{-i \theta_{l,B} H_B} e^{-i \theta_{l,A} H_A} \right) |+\rangle^{\otimes N}$$

In Qiskit 2.x: $e^{-i \theta Z_i Z_{i+1}}$ is implemented via RZZ(2*theta) and $e^{-i \phi X_i}$ via RX(2*phi).

Depth Limit Justification: You must keep $p \le 2$. The Mele et al. paper mathematically proves that local noise acting on depths beyond $O(\log N)$ causes the state to approach a fixed-point distribution, destroying the unitary information of early layers.

3. GNN Architecture & Data Engineering (PyTorch)

When tasked to build the predictive model, you must map the quantum Hamiltonian into a classical graph suitable for Message Passing Neural Networks (MPNN).

3.1 Graph Representation

Nodes ($V$): Represent qubits.

Node Features: Local external fields acting on the qubit (e.g., the scalar $h_i$ from $-h_i X_i$).

Edges ($E$): Represent interaction terms between qubits.

Edge Features: Interaction coupling strengths (e.g., the scalar $J_{ij}$ from $-J_{ij} Z_i Z_j$).

3.2 The ML Pipeline (Curriculum Learning)

Supervised Pre-training (MSE Loss):


$$\mathcal{L}_{MSE} = \frac{1}{M} \sum_{k=1}^M || \vec{\theta}_{pred}^{(k)} - \vec{\theta}_{opt}^{(k)} ||^2$$


The GNN learns to regress the exact angles discovered by the classical optimizer.

Physics-Informed Fine-Tuning (Energy Loss):
To ensure the GNN doesn't just memorize angles but understands the energy landscape, the loss function shifts to the quantum expected value:


$$\mathcal{L}_{Physics} = \langle \psi(GNN(\mathcal{G})) | H_{\mathcal{G}} | \psi(GNN(\mathcal{G})) \rangle$$


Note: In PyTorch, this requires a differentiable quantum simulator backend (like TorchQuantum) or calculating the analytical gradients of the HVA via parameter-shift rules to backpropagate through the quantum circuit into the GNN weights.

4. Deep Dive: The Mele et al. Paradigm

Kiro must understand why barren plateaus are avoided here, to justify our architectural choices in any theoretical writing.

Unital vs. Non-Unital Noise: Unital noise (like depolarizing channels) maps the maximally mixed state to itself, shrinking the Hilbert space symmetrically and causing standard barren plateaus. Non-unital noise (like amplitude damping/thermal relaxation, typical in IBM superconducting qubits) biases the system towards a specific state (e.g., $|00...0\rangle$).

The Double-Edged Sword: Because non-unital noise breaks the isotropic "flatness" of the state space, the variance of the gradients does not vanish exponentially for local cost functions. This is why we can train the shallow HVA. However, this same noise erases the influence of early gates, effectively truncating the circuit to $O(\log n)$ layers.

Strategic Conclusion: Our GNN + Shallow HVA architecture is not just "efficient"; it is strictly required by the fundamental thermodynamics of current quantum processors. We use the GNN to perform the deep mathematical work, and the quantum hardware to execute the noise-resilient, shallow projection.