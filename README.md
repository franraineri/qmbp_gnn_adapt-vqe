# Hybrid GNN-HVA Framework for Topological Phase Characterization

## 🔬 Project Overview

This repository contains the codebase and theoretical framework for a Master's Thesis (TFM) in Quantum Computing and Condensed Matter Physics. The project aims to accelerate Variational Quantum Eigensolvers (VQE) to characterize Quantum Topological Phases (like Quantum Spin Liquids) in frustrated many-body systems.

To overcome the limitations of Noisy Intermediate-Scale Quantum (NISQ) devices—specifically the "barren plateau" problem and noise-induced truncation—we abandon pure Quantum Neural Networks (QCNN) and standard deep VQEs. Instead, we propose a  **predictive hybrid architecture** : A classical Graph Neural Network (GNN) trained on Tensor Network data acts as an "Intelligent Warm-Start" to initialize a shallow, physics-informed quantum circuit (Hamiltonian Variational Ansatz - HVA).

## 📄 Theoretical Paradigm Shift (Critical Context for AI Agents)

**ATTENTION KIRO (AI Agent):** All code generation and architectural decisions MUST comply with the findings of the recent literature:

> *Mele, A. A., et al. "Noise-induced shallow circuits and the absence of barren plateaus" (Nature Physics, 2026).*

**Key takeaways from this paper dictating our codebase:**

1. **Depth Truncation:** Non-unital noise effectively truncates quantum circuits to logarithmic depth $\mathcal{O}(\log n)$. Deep circuits are classically simulable and lose quantum advantage. **Rule:** All our quantum circuits (HVAs) MUST be strictly shallow (e.g., $p=1$ or $p=2$ layers).
2. **Local Observables Only:** Global cost functions still suffer from barren plateaus under noise. **Rule:** We must extract and monitor *local observables* (e.g., local magnetization $\langle X_i \rangle$, local correlation $\langle Z_i Z_{i+1} \rangle$) to characterize phases, rather than relying on global state fidelity in the quantum hardware execution.
3. **Absence of Barren Plateaus for Local Costs:** By using shallow circuits and local observables, we guarantee stable gradients. Our GNN exploits this by providing the perfect starting seed, allowing instantaneous convergence before noise destroys the signal.

## 🗺️ The 4-Phase Roadmap

The project is strictly divided into four operational phases. Kiro must contextualize any task within this specific pipeline:

### PHASE 1: Classical Ground Truth Generation

* **Goal:** Solve parameterized Hamiltonians (e.g., 1D Transverse Field Ising Model, 2D Spin Ladders) classically.
* **Tools:** Exact Diagonalization (for PoC < 15 qubits), DMRG / TeNPy (for quasi-1D), NetKet (Neural Quantum States for 2D).
* **Output:** Dataset mapping Hamiltonian parameters (e.g., $h, J$) to exact ground state vectors and local observable expectation values.

### PHASE 2: Symmetry-Aware Ansatz & Compilation

* **Goal:** Translate the classical ground states into optimal parameters ($\theta_{opt}$) for a quantum circuit.
* **Architecture:** Use a  **Hamiltonian Variational Ansatz (HVA)** . Never use Hardware-Efficient Ansätze (HEA).
* **Constraint:** The HVA must be shallow ($p \le 2$).
* **Optimization Strategy:** Use  **Warm Start** . The optimized $\theta$ for Hamiltonian $H_i$ must be used as the initial guess for $H_{i+1}$ to ensure physical continuity and fast convergence.

### PHASE 3: Graph Neural Network (GNN) Predictive Model

* **Goal:** Train a classical model to predict $\theta_{opt}$ from the Hamiltonian graph.
* **Tools:** PyTorch (`torch.nn`).
* **Strategy:** Map the physical interactions (edges) and qubit parameters (nodes) to the HVA angles. Optional fine-tuning using physics-informed loss (minimizing local energy).

### PHASE 4: Deployment & Restricted Adaptive Refinement

* **Goal:** Execute on real IBM Hardware (e.g., IBM Heron) using the trained GNN for inference.
* **Workflow:** Unseen Hamiltonian -> GNN predicts $\theta_{pred}$ -> Initialize HVA (Warm-Start) -> Execute VQE.
* **Adaptive Step:** If using `AdaptVQE`, strictly limit to `max_iterations=2` to prevent the circuit from growing into the noise-truncation regime.

## 💻 Tech Stack & Code Practices (Qiskit 2.x Standard)

**KIRO INSTRUCTIONS:** You must write code adhering to the **Qiskit 2.x ecosystem** (and modern 1.x). Deprecated Qiskit 0.4x syntax is strictly forbidden.

### Mandatory Coding Rules:

1. **Operators:** ALWAYS use `qiskit.quantum_info.SparsePauliOp` for building Hamiltonians and observables. NEVER use `PauliSumOp` or `opflow` (they are deprecated).
   * *Correct:* `SparsePauliOp.from_sparse_list([("ZZ", [0, 1], 1.0)], num_qubits=N)`
2. **Execution/Primitives:** ALWAYS use  **Qiskit Primitives V2** .
   * Use `qiskit.primitives.StatevectorEstimator` for exact local simulations (PoC).
   * Use `qiskit_ibm_runtime.EstimatorV2` for hardware execution.
   * *Never* use `qiskit.execute`, `Aer.get_backend()`, or Primitives V1.
3. **Algorithms:** Import algorithms from the standalone package `qiskit_algorithms`, NOT from `qiskit.algorithms` (deprecated).
4. **Data Binding:** Use `circuit.assign_parameters()` to bind predicted angles before passing them to the Estimator.

## 🚧 Contingencies & Scope

* If 2D Tensor Network simulations hit memory limits, fallback to quasi-1D cylindrical Spin Ladders.
* If hardware noise is too high even for shallow HVAs, target Symmetry-Protected Topological (SPT) phases (which require constant-depth circuits) instead of pure QSLs.

## 🚀 Quick Start (PoC)

The current Proof of Concept (PoC V2.0) focuses on Phases 1 and 2 using the 1D Transverse Field Ising Model (TFIM) for $N=6$ qubits. Refer to the corresponding Jupyter Notebooks in the repository for the baseline implementation.
