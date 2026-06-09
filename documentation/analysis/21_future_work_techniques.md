# Future Work: Advanced Techniques for GNN-HVA Scaling

> Techniques identified as promising extensions beyond the scope of this thesis.
> Included for Chapter 7 (Future Work) discussion with literature references.
>
> **Date**: 2026-06-08

---

## 1. Neighbor-Informed Learning (NIL) for Error Mitigation

**What it is**: A unified QEM framework that generalizes ZNE and PEC by using
"neighbor circuits" (small perturbations of the target circuit) as training data
to predict the ideal observable value. Instead of extrapolating along a single
noise axis (ZNE), NIL learns from a neighborhood of circuits in parameter space.

**Why it matters for GNN-HVA**: The GNN already generates θ_opt predictions that
form a natural "neighborhood" around the deployed circuit. NIL could use GNN-
predicted circuits at nearby h-values as the neighbor set, eliminating the need
for explicit noise amplification (no PEA overhead).

**Integration path**: Replace `run_adaptive_zne()` with NIL using GNN predictions
at h ± δh as neighbor circuits. Zero additional QPU cost if GNN predictions are
available for multiple h-values.

**Reference**: Wei et al., "Scalable Quantum Error Mitigation with Neighbor-Informed
Learning," arXiv:2512.12578 (2024).

---

## 2. ML-QEM: Machine Learning for Practical QEM

**What it is**: Train a classical ML model on (noisy_observable, true_observable)
pairs from calibration circuits, then apply to correct new circuits. Demonstrated
at 100 qubits on IBM hardware with 100× cost reduction vs standard mitigation.

**Relationship to current work**: Our GNN-QEM module is conceptually the same
approach but specialized for energy correction with graph structure. The generic
ML-QEM additionally handles arbitrary observables and uses transfer learning
across circuit families.

**Integration path**: The GNN-QEM `correct_energy()` function already implements
this pattern. Future extension: train on multiple observable types (not just
energy) and enable cross-topology transfer via shared latent space.

**Reference**: Czarnik et al., "Machine Learning for Practical Quantum Error
Mitigation," arXiv:2309.17368 (2023). Experiments on IBM 100-qubit hardware.

---

## 3. Telemetry-Driven Adaptive Error Mitigation (GSC-QEMit)

**What it is**: A hierarchical framework that monitors QPU telemetry (T1, T2, gate
errors) in real-time and adaptively selects the mitigation strategy during execution.
Uses a forecast model to predict noise evolution and a multi-armed bandit to optimize
the mitigation/overhead trade-off.

**Current coverage in our pipeline**:
- ✅ `take_calibration_snapshot()` — telemetry capture
- ✅ `check_calibration_drift()` — drift detection (abort if T1 drift > 20%)
- ✅ `run_adaptive_zne()` — automatic PEA → GF fallback based on R²
- ❌ Predictive drift model (not implemented)
- ❌ Multi-armed bandit for strategy selection (not implemented)
- ❌ Mid-run strategy switching (current: decision at run start only)

**When it would help**: Long hardware runs (>1h) where noise characteristics
shift during execution. For our typical runs (~30 min VQE sweep), the static
adaptive_zne is sufficient.

**Reference**: "A Telemetry-Driven Hierarchical Forecast-and-Bandit Framework
for Adaptive Quantum Error Mitigation," arXiv:2604.24551 (2024).

---

## 4. SC-ADAPT-VQE: Scalable Circuits for Translationally Invariant Systems

**What it is**: An algorithm that determines HVA-like circuit structure CLASSICALLY
(on small systems) and then tiles the result to arbitrary system sizes. Demonstrated
on the Schwinger model vacuum at 100 qubits on IBM Eagle.

**Relevance**: Our HVA p=1 for 1D TFIM is already translationally invariant —
the global HVA circuit IS a "scalable circuit" by definition. SC-ADAPT-VQE would
matter if we needed p>2 or non-trivial ansatz structure near criticality.

**Integration path**: Low priority. Our p≤2 constraint (Mele et al.) already ensures
circuits are scalable. SC-ADAPT-VQE is more relevant for non-HVA ansätze.

**Reference**: Farrell et al., "Scalable Circuits for Preparing Ground States on
Digital Quantum Computers: The Schwinger Model Vacuum on 100 Qubits,"
arXiv:2308.04481 (2024). PRX Quantum 5(2), 020315.

---

## 5. Utility-Scale Hamiltonian Engineering (103 qubits Kagome)

**What it is**: Split VQE into local (per-site, classically optimizable) and global
(entanglement, quantum) components. Allows single-layer ansatz at 100+ qubits by
pre-computing the local part analytically.

**Our version**: Bond-resolved HVA with θ_x (local) vs θ_zz (global) is
structurally equivalent. The Kagome paper additionally uses "Hamiltonian engineering"
to modify the physical Hamiltonian to simplify the required ansatz — making defect
triangles couple more strongly to mimic the dynamics.

**Key difference**: They CHANGE the Hamiltonian to fit the hardware. We CHANGE the
parametrization to fit the GNN. Both achieve utility-scale from shallow circuits.

**Reference**: "Utility-Scale Quantum Computation of Ground-State Energy in a 100+
Site Planar Kagome Antiferromagnet via Hamiltonian Engineering,"
arXiv:2507.06361 (2025). IBM Heron r1/r2 processors.

---

## 6. Parameter Freezing (TITAN)

**What it is**: During VQE optimization, identify parameters that converge early
and freeze them — reducing the effective dimension of the landscape. Uses
trajectory analysis to detect convergence.

**Relevance to bond-resolved**: At N=40 with 79 params, many θ_zz bonds in chain_1d
converge to nearly identical values (translational symmetry). TITAN would detect this
and freeze them, reducing to ~2-5 effective parameters. This would make cold-start
VQE viable even at 79 nominal params.

**Key insight for thesis**: "TITAN-style freezing would recover the quasi-2D structure
that makes chain_1d easy. On heavy_hex (non-uniform), fewer parameters freeze →
GNN remains necessary."

**Reference**: "A Trajectory-Informed Technique for Adaptive Parameter Freezing
in Large-Scale VQE," arXiv:2509.15193 (2025).

---

## 7. GNN for Quantum Chip Parameter Design

**What it is**: Use GNN to design parameters of superconducting quantum chips
(junction frequencies, coupling strengths). Achieves 51% fewer errors than
state-of-the-art on 870-qubit chips, 200× faster.

**Parallel to our work**: Same insight (graph structure encodes spatial relationships)
applied to a different level of the stack. Our GNN maps graph→circuit_params;
their GNN maps graph→chip_params.

**Thesis connection**: "Graph neural networks are proving essential across the full
quantum computing stack — from chip design (870 qubits) to error mitigation
(GNN-QEM) to variational parameter prediction (this work)."

**Reference**: "Scalable Parameter Design for Superconducting Quantum Circuits
with Graph Neural Networks," arXiv:2411.16354 (2024).
