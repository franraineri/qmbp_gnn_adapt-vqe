# Alternative Bibliography — Techniques & Methodologies to Consider

Papers proposing alternative approaches, different architectures, or complementary techniques that could enhance or provide alternatives to the GNN-HVA framework. These are rigorous, high-performance results from confident sources that may inspire future directions.

---

## 1. Alternative VQE Initialization Strategies

### Generative Models (Non-GNN)

Nakaji, K., Kristensen, L. B., Kemmoku, R., Campos-Gonzalez-Angulo, J. A., Vakili, M. G., Huang, H., Bagherimehrab, M., Gorgulla, C., Wong, F., McCaskey, A., Kim, J.-S., Nguyen, T., Rao, P., Gao, Q., Sugawara, M., Yamamoto, N., & Aspuru-Guzik, A. (2025). The generative quantum eigensolver (GQE) and its application for ground state search. *arXiv preprint arXiv:2401.09253v2*. https://arxiv.org/abs/2401.09253

**Relevance:** Transformer-based (GPT) generative model that produces quantum circuits with desired properties. Operates outside the VQE paradigm entirely — generates circuits rather than optimizing parameters. Surpasses CCSD for N₂ bond dissociation. Could replace our MPNN predictor with a generative circuit designer.

---

### SpinGQE: Generative Eigensolver for Spin Hamiltonians

Dade, N. O. O. et al. (2026). SpinGQE: A generative quantum eigensolver for spin Hamiltonians. *arXiv preprint arXiv:2603.24298*. https://arxiv.org/abs/2603.24298

**Relevance:** Extends GQE specifically to spin Hamiltonians (our domain). Transformer decoder learns distributions over quantum circuits for low-energy states. Validated on 4-qubit Heisenberg model. Navigates energy landscapes without problem-specific symmetries. Open-source implementation available. Directly applicable alternative to our MPNN + VQE approach.

---

### Tensor-Train Hypernetworks

Nature Publishing Group. (2025). TensorHyper-VQC: A tensor-train-guided hypernetwork for robust and scalable variational quantum computing. *npj Quantum Information*. https://www.nature.com/articles/s41534-025-01157-z

**Relevance:** Fully delegates VQE parameter generation to a classical tensor-train network, decoupling optimization from quantum hardware. Alternative to our GNN approach — uses tensor network structure instead of graph structure to predict parameters.

---

### Reinforcement Learning Initialization

arXiv:2508.18514. (2025). Reinforcement learning initializations for deep variational quantum circuits. https://arxiv.org/abs/2508.18514

**Relevance:** RL-based initialization strategy that reshapes the parameter landscape to avoid barren plateau regions. Alternative to our warm-start sweep — could provide better initialization for difficult parameter regions (e.g., near h=1.0 critical point).

---

### Neural Network + Autoencoder VQE

Mesman, K. et al. (2025). NN-AE-VQE: Neural network parameter prediction on autoencoded variational quantum eigensolvers. *arXiv preprint arXiv:2411.15667v2*. https://arxiv.org/abs/2411.15667

**Relevance:** Quantum autoencoder compresses state representation + neural network predicts circuit parameters. Reduces parameter count while maintaining accuracy. Alternative compression strategy to our direct MPNN regression.

---

### Attention + VAE for Unsupervised Phase Detection

Li, X. et al. (2026). Learning variational quantum circuit parameters with classical artificial intelligence for quantum phase transition detection. *arXiv preprint arXiv:2506.06678*. https://arxiv.org/abs/2506.06678

**Relevance:** LLM-style attention mechanism + VAE captures hidden correlations in VQE circuit parameters. Detects phase transitions in an unsupervised manner without measuring observables. Could complement our supervised MPNN approach with unsupervised phase boundary detection.

---

## 2. Alternative Phase Classification Methods

### Shadow Tomography + Time-Series ML

Ye, W. et al. (2025). Universal quantum phase classification on quantum computers from machine learning. *arXiv preprint arXiv:2508.04774*. https://arxiv.org/abs/2508.04774

**Relevance:** Combines shadow tomography with time-series ML models for phase classification. Does not rely on local order parameters — achieves universal classification. Validated on 1D Ising and ANNNI models. Alternative to our observable-based classification.

---

### Quantum Reservoir for Topological Phase Detection

Li, X. et al. (2025). Unsupervised detection of topological phase transitions with a quantum reservoir. *arXiv preprint arXiv:2509.25825*. https://arxiv.org/abs/2509.25825

**Relevance:** Many-body localized evolution as quantum reservoir for unsupervised topological phase detection. Requires only local measurements — no full density matrix reconstruction. Validated on extended SSH model. Alternative to our QRC pipeline with different reservoir dynamics.

---

### QCNN for 2D Phase Recognition

Sander, L. et al. (2025). Quantum convolutional neural network for phase recognition in two dimensions. *arXiv preprint arXiv:2407.04114v2*. https://arxiv.org/abs/2407.04114

**Relevance:** QCNN that identifies Z₂ topological order → paramagnetic phase transition in 2D. Exhibits noise threshold for topological order recognition. Captures correlations inaccessible to direct measurements. Alternative quantum ML approach for our 2D scaling targets (Kagome, triangular).

---

### Variational Autoencoder for Phase Discovery

Yee, B. et al. (2026). From classical to quantum: Extending Prometheus for unsupervised discovery of phase transitions in three dimensions and quantum systems. *arXiv preprint arXiv:2602.14928v4*. https://arxiv.org/abs/2602.14928

**Relevance:** Quantum-aware VAE (Q-VAE) operating on complex wavefunctions with fidelity-based loss. Achieves 2% accuracy in quantum critical point detection for TFIM. Extracts critical exponents unsupervised. Could provide independent validation of our phase boundaries.

---

### SHAP-Driven Quantum Phase Classification

Mahlow, F. et al. (2025). Quantum phases classification using quantum machine learning with SHAP-driven feature selection. *arXiv preprint arXiv:2504.10673*. https://arxiv.org/abs/2504.10673

**Relevance:** QSVM + VQC with SHAP interpretability for ANNNI model phase classification. Identifies most informative features (5-6 key features sufficient). Could inform feature selection for our MPNN input encoding.

---

### Neural Quantum States for Phase Detection

Hernandes, V. et al. (2025). Adiabatic fine-tuning of neural quantum states enables detection of phase transitions in weight space. *arXiv preprint arXiv:2503.17140v2*. https://arxiv.org/abs/2503.17140

**Relevance:** Trains NQS across phase diagram; phase transitions manifest as structures in neural network weight space. Validated on TFIM and J₁-J₂ Heisenberg model. Novel perspective — could detect phase transitions from our MPNN's trained weights without explicit observable measurement.

---

## 3. Alternative VQE Architectures & Improvements

### Rigorous Neural Post-Processing (U-VQNHE)

Kim, M. et al. (2026). A rigorous hybridization of variational quantum eigensolver and classical neural network. *arXiv preprint arXiv:2602.17295*. https://arxiv.org/abs/2602.17295

**Relevance:** Unitary variational quantum-neural hybrid eigensolver (U-VQNHE). Learnable diagonal post-processing layer with variational safety guarantees. Tested on TFIM. Could enhance our Phase 4 hardware results by post-processing measurement outcomes.

---

### ADAPT-VQE with Coupled Exchange Operators

Cortes, C. L. et al. (2025). Reducing the resources required by ADAPT-VQE using coupled exchange operators and improved subroutines. *npj Quantum Information*, *11*, 56. https://www.nature.com/articles/s41534-025-01039-4

**Relevance:** Novel CEO operator pool for ADAPT-VQE that reduces measurement counts and circuit depth. State-of-the-art resource assessment for hardware ADAPT-VQE. Could improve our Phase 4 AdaptVQE efficiency if we need more than 0-2 iterations.

---

### Transferable ML for Circuit Parameters

Bincoletto, D. et al. (2025). A transferable machine learning approach to predict quantum circuit parameters for electronic structure problems. *arXiv preprint arXiv:2511.03726*. https://arxiv.org/abs/2511.03726

**Relevance:** ML parameter prediction that transfers between different system sizes (trained on small, predicts for large). Validates the transferability principle that our MPNN exploits for scaling from N=6 to N=10+.

---

## 4. Alternative Reservoir Computing Approaches

### Discrete Time Crystal Reservoir

Yin, Z.-Q. et al. (2025). Robust and efficient quantum reservoir computing with discrete time crystal. *arXiv preprint arXiv:2508.15230*. https://arxiv.org/abs/2508.15230

**Relevance:** Gradient-free, noise-robust QRC using discrete time crystal dynamics. First experimental demonstration of QRC for classification on superconducting processors. Establishes correlation between non-equilibrium phase transitions and QRC performance. Alternative reservoir dynamics to our fixed-HVA reservoir.

---

## 5. Tensor Network + Quantum Hybrid Approaches

### 2D Tensor Network Pre-optimization

Martin, B. A. et al. (2026). Pre-optimization of quantum circuits, barren plateaus and classical simulability: Tensor networks to unlock the variational quantum eigensolver. *arXiv preprint arXiv:2602.04676*. https://arxiv.org/abs/2602.04676

**Relevance:** Differentiable 2D tensor networks optimize VQE circuits for TFIM ground states. Identifies regimes where quantum hardware offers better scaling than TN simulations. Could replace our MPNN warm-start with TN-based warm-start for 2D systems.

---

### Infinite Tensor Networks for Circuit Learning

arXiv:2506.02105. (2025). Learning circuits with infinite tensor networks. https://arxiv.org/abs/2506.02105

**Relevance:** Uses translation-invariant infinite MPS to learn efficient quantum circuits. Achieves 5.2× reduction in T-count. Exploits translation invariance to reduce optimization complexity. Could provide better circuit designs for our HVA on periodic chains.

---

## 6. Scalability & Barren Plateau Solutions

### Hardware-Efficient Ansatz Without Barren Plateaus

arXiv:2403.04844. (2024). Hardware-efficient ansatz without barren plateaus in any depth. https://arxiv.org/abs/2403.04844

**Relevance:** Constructs HEA that provably avoids barren plateaus at any depth. Challenges the assumption that only physics-inspired ansätze (like HVA) avoid BPs. If validated, could relax our strict HVA-only constraint for hardware deployment.

---

### Barren Plateaus Beyond Observable Concentration

arXiv:2603.18479. (2026). Barren plateaus beyond observable concentration. https://arxiv.org/abs/2603.18479

**Relevance:** Extends barren plateau theory beyond the observable concentration framework. New mechanisms for gradient vanishing in PQCs. Important for understanding the theoretical limits of our shallow HVA approach.

---

## Summary: Priority Techniques to Investigate

| Technique | Paper | Potential Impact | Effort |
|-----------|-------|-----------------|--------|
| GQE (transformer circuit generation) | Nakaji et al. 2025 | High — paradigm shift from optimization to generation | High |
| SpinGQE (spin-specific generative) | Dade et al. 2026 | High — directly targets our domain | Medium |
| U-VQNHE (neural post-processing) | Kim et al. 2026 | Medium — improves hardware results | Low |
| TN pre-optimization for 2D | Martin et al. 2026 | High — enables 2D scaling | Medium |
| TITAN (parameter freezing) | Peng et al. 2025 | Medium — reduces VQE cost 40-60% | Low |
| QESEM (utility-scale mitigation) | Aharonov et al. 2026 | High — better than ZNE on IBM Heron | Medium |
| Shadow tomography + time-series ML | Ye et al. 2025 | Medium — alternative classification | Medium |
| DTC quantum reservoir | Yin et al. 2025 | Medium — better reservoir dynamics | Medium |
| Attention + VAE unsupervised | Li et al. 2026 | Low-Medium — complementary detection | Low |
| VQEzy dataset | Chen et al. 2025 | Low — pre-training data / benchmark | Low |

---

> **Note:** These papers represent alternative or complementary approaches. The current GNN-HVA architecture remains well-validated by the core bibliography. These alternatives should be considered for future work, scaling challenges, or if specific pipeline components underperform.
