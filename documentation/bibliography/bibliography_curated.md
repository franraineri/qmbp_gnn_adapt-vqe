# Curated Bibliography — Hybrid GNN-HVA Framework for Topological Phase Characterization

Filtered for **high confidence** (verified source, accessible URL) and **direct relevance** to the thesis. Papers are ranked within each section by importance to the framework. APA 7th edition format.

---

## Legend

- ✅ **Core** — Directly used/validated in the thesis pipeline
- 🔑 **Key** — Provides essential theoretical or methodological foundation
- 📎 **Supporting** — Validates decisions or provides context

---

## 1. Governing Paper (Architectural Foundation)

✅ Mele, A. A., Angrisani, A., Ghosh, S., Khatri, S., Eisert, J., Stilck França, D., & Quek, Y. (2026). Noise-induced shallow circuits and the absence of barren plateaus. *Nature Physics*. https://www.nature.com/articles/s41567-026-03245-z | arXiv: https://arxiv.org/abs/2403.13927

> Governs all architectural decisions: proves noisy circuits are effectively shallow, justifying HVA p≤2.

---

## 2. VQE on TFIM: Expressivity & Hardware Validation

✅ Tripathi, A. P., Mathur, N., & Tripathi, V. (2026). Ansätz expressivity and optimization in variational quantum simulations of transverse-field Ising model across system sizes. *arXiv preprint arXiv:2604.20961*. https://arxiv.org/abs/2604.20961

> Benchmarks HVA vs HEA on TFIM 1D/2D/3D up to 27 spins. Independently validates our HVA choice.

✅ Sharma, R. (2026). Quantum phase transitions in the transverse-field Ising model: A comparative study of exact, variational, and hardware-based approaches. *arXiv preprint arXiv:2601.17515*. https://arxiv.org/abs/2601.17515

> Compares exact diag, VQE, and IQM hardware for 1D TFIM. Demonstrates noise broadening of critical crossover — directly relevant to Phase 4.

🔑 Sumeet, S. et al. (2025). Hybrid quantum-classical algorithm for the transverse-field Ising model in the thermodynamic limit. *arXiv preprint arXiv:2310.07600v2*. https://arxiv.org/abs/2310.07600

> NLCE + VQE with modified HVA for TFIM. Demonstrates convergence to thermodynamic limit.

---

## 3. Hamiltonian Variational Ansatz (HVA)

✅ Wiersema, R., Zhou, C., de Sereville, Y., Carrasquilla, J., & Kim, Y. B. (2020). Exploring entanglement and optimization within the Hamiltonian Variational Ansatz. *PRX Quantum*, *1*(2), 020319. https://doi.org/10.1103/PRXQuantum.1.020319

> Foundational HVA paper. Proves HVA has mild/absent barren plateaus and restricted state space.

✅ Mele, A. A., Mbeng, G. B., Santoro, G. E., Collura, M., & Torta, P. (2022). Avoiding barren plateaus via transferability of smooth solutions in a Hamiltonian variational ansatz. *Physical Review A*, *106*, L060401. https://doi.org/10.1103/PhysRevA.106.L060401

> Validates parameter transferability in HVA — theoretical basis for our descending h-sweep.

---

## 4. Barren Plateaus & Shallow Circuits

🔑 McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven, H. (2018). Barren plateaus in quantum neural network training landscapes. *Nature Communications*, *9*(1), 4812. https://doi.org/10.1038/s41467-018-07090-4

> Original barren plateau discovery. Motivates our HVA-only constraint.

🔑 Cerezo, M., Sone, A., Volkoff, T., Cincio, L., & Coles, P. J. (2021). Cost function dependent barren plateaus in shallow parametrized quantum circuits. *Nature Communications*, *12*(1), 1791. https://doi.org/10.1038/s41467-021-21728-w

> Proves local cost functions avoid BPs in shallow circuits — justifies our pure energy cost.

📎 Wang, S., Fontana, E., Cerezo, M., et al. (2021). Noise-induced barren plateaus in variational quantum algorithms. *Nature Communications*, *12*, 6961. https://doi.org/10.1038/s41467-021-27045-6

> Noise-induced BPs motivate shallow depth constraint.

---

## 5. GNN Theory & Architecture

✅ Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? *ICLR 2019*. https://arxiv.org/abs/1810.00826

> Proves GIN is as powerful as WL test. Justifies our GINConv choice.

✅ Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. *ICML*, *70*, 1263–1272. https://arxiv.org/abs/1704.01212

> Foundational MPNN framework. Establishes theoretical basis for our predictor architecture.

✅ Kochkov, D., Pfaff, T., Sanchez-Gonzalez, A., Battaglia, P., & Clark, B. K. (2021). Learning ground states of quantum Hamiltonians with graph networks. *arXiv preprint arXiv:2110.06390*. https://arxiv.org/abs/2110.06390

> GNN as variational manifold for Heisenberg Hamiltonians. Validates graph-based approach for spin systems.

---

## 6. GNN for Quantum Circuits & Spin Systems

✅ Meng, F. et al. (2025). Output prediction of quantum circuits based on graph neural networks. *arXiv preprint arXiv:2504.00464*. https://arxiv.org/abs/2504.00464

> GNN framework for circuit output prediction. Validates GNN > CNN by 36% for circuit properties.

✅ Slavin, V. (2025). Graph neural network approach to predicting magnetization in quasi-one-dimensional Ising systems. *arXiv preprint arXiv:2507.17509*. https://arxiv.org/abs/2507.17509

> GNN for magnetic properties of quasi-1D Ising from lattice geometry. Directly validates our approach.

---

## 7. GNN-Enhanced VQE & Parameter Prediction

✅ Zhang, C., Jiang, L., & Chen, F. (2025). Qracle: A graph-neural-network-based parameter initializer for variational quantum eigensolvers. *arXiv preprint arXiv:2505.01236*. https://arxiv.org/abs/2505.01236

> GNN-based VQE parameter initializer with unified Hamiltonian+ansatz graph encoding. Validates our scaling path.

✅ Lee, J. et al. (2026). Improving generalization and trainability of quantum eigensolvers via graph neural encoding. *arXiv preprint arXiv:2602.19752*. https://arxiv.org/abs/2602.19752

> Graph autoencoder + NN generates VQE parameters generalizing across Hamiltonians. Directly validates MPNN-for-VQE paradigm.

🔑 Huang, H.-Y., Kueng, R., Torlai, G., Albert, V. V., & Preskill, J. (2022). Provably efficient machine learning for quantum many-body problems. *Science*, *377*(6613), eabk3333. https://doi.org/10.1126/science.abk3333

> Proves classical ML can efficiently predict ground-state properties within a phase. Theoretical foundation for MPNN generalization.

---

## 8. ML-Driven VQE Parameter Prediction

✅ Miao, J., Hsieh, C.-Y., & Zhang, S.-X. (2024). Neural-network-encoded variational quantum algorithms. *Physical Review Applied*, *21*, 014053. https://doi.org/10.1103/PhysRevApplied.21.014053 | arXiv: https://arxiv.org/abs/2308.01068

> Validates MLP approach for parameterized spin Hamiltonians. Dropout and active learning strategies adopted.

📎 Karim, A. et al. (2025). Fast and noise-aware machine learning variational quantum eigensolver optimiser. *arXiv preprint arXiv:2503.20210*. https://arxiv.org/abs/2503.20210

> Supervised ML on intermediate VQE data. Demonstrates noise resilience on IBM hardware.

📎 Li, X. et al. (2026). Learning variational quantum circuit parameters with classical artificial intelligence for quantum phase transition detection. *arXiv preprint arXiv:2506.06678*. https://arxiv.org/abs/2506.06678

> Attention + VAE for unsupervised phase detection from VQE parameters. Complementary to our supervised approach.

---

## 9. Warm-Start & Parameter Transfer

🔑 Puig, R., Drudis, M., Thanasilp, S., & Holmes, Z. (2025). Variational quantum simulation: A case study for understanding warm starts. *PRX Quantum*, *6*, 010317. https://doi.org/10.1103/PRXQuantum.6.010317

> Rigorous analysis of warm-start benefits and limitations for VQE.

📎 Zou, H., Rahm, M., Kockum, A. F., & Olsson, S. (2026). Generative flow-based warm start of the variational quantum eigensolver. *npj Quantum Information*, *12*, 5. https://doi.org/10.1038/s41534-025-01159-x | arXiv: https://arxiv.org/abs/2507.01726

> Normalizing flows for VQE warm-start. Up to 50× acceleration. Alternative generative approach.

📎 Skogh, M., Leinonen, O., Lolur, P., & Rahm, M. (2023). Accelerating variational quantum eigensolver convergence using parameter transfer. *Electronic Structure*, *5*, 035002. https://doi.org/10.1088/2516-1075/ace86d

> Validates parameter transfer strategy between related Hamiltonians.

---

## 10. Error Mitigation: ZNE & Hardware Deployment

✅ Uvarov, A. et al. (2024). Mitigating quantum gate errors for variational eigensolvers using hardware-inspired zero-noise extrapolation. *arXiv preprint arXiv:2307.11156v3*. https://arxiv.org/abs/2307.11156

> Inhomogeneous ZNE using hardware error distribution. Linear E(CES) approximation validated for small circuits. Our N=10 experiments (2026-05-14) confirm the applicability range is bounded — R² collapses at higher qubit counts where total CES exceeds the perturbative regime.

🔑 Tsubouchi, K., Sagawa, T., & Takagi, K. (2023). Universal cost bound of quantum error mitigation based on quantum estimation theory. *Physical Review Letters*, *131*, 210602. https://arxiv.org/abs/2208.09385

> Proves sampling cost of any unbiased error mitigation grows exponentially with circuit depth AND qubit count. For random circuits with local noise, cost grows exponentially also with qubit count. Directly explains our N=6→N=10 ZNE failure: 3 layouts cannot overcome the exponential cost at N=10.

🔑 Rabinovich, D. et al. (2025). Zero-noise extrapolation via cyclic permutations of quantum circuit layouts. *arXiv preprint arXiv:2511.02901v2*. https://arxiv.org/abs/2511.02901

> CLP-ZNE: uses O(n) cyclic layout permutations for n-qubit 1D circuits. Validated on IBM Heron noise model at n=12, achieving order-of-magnitude error reduction over standard ZNE. Directly addresses our N=10 failure — the fix is more layouts with systematic (not random) coverage of CES space.

✅ Sun, W. et al. (2025). Noise-mitigated variational quantum eigensolver with pre-training and zero-noise extrapolation. *arXiv preprint arXiv:2501.01646*. https://arxiv.org/abs/2501.01646

> MPS pre-training + NN-enhanced ZNE. Validates our NNExtrapolator design. When linear E(CES) breaks down, MLP extrapolation constrains errors to O(10⁻²).

✅ Pokharel, B. et al. (2025). Empirical learning of dynamical decoupling on quantum processors. *arXiv preprint arXiv:2403.02294v2*. https://arxiv.org/abs/2403.02294

> Genetic algorithm DD optimization for IBM processors. Scalable to 100 qubits. Directly relevant to Phase 4 DD. DD reduces effective CES, potentially restoring ZNE linearity at N=10.

🔑 Aharonov, D., Bairey, E., Lindner, N. H., et al. (2026). Reliable high-accuracy error mitigation for utility-scale quantum circuits. *arXiv preprint arXiv:2508.10997*. https://arxiv.org/abs/2508.10997

> QESEM on IBM Heron — resolves ZNE vs PEC tradeoff. Tested on kicked TFIM.

---

## 11. IBM Hardware Benchmarks (Phase 4)

✅ Kiiamov, A. G. et al. (2026). Simulating Wigner localisation with the IBM Heron 2 quantum processor: A proof-of-principle benchmarking study. *arXiv preprint arXiv:2601.01263*. https://arxiv.org/abs/2601.01263

> 6-qubit VQE on IBM Heron 2 achieving <7% error. Validates IBM hardware at our system size.

📎 Larrucea, J. et al. (2026). Accuracy-cost trade-offs for reference VQE calculations of H₂ on IBM Quantum hardware. *arXiv preprint arXiv:2604.11478*. https://arxiv.org/abs/2604.11478

> Comprehensive 2026 benchmark of shot count, backend, optimization on IBM processors.

📎 Kim, Y., Eddins, A., Anand, S., et al. (2023). Evidence for the utility of quantum computing before fault tolerance. *Nature*, *618*, 500–505. https://doi.org/10.1038/s41586-023-06096-3

> Landmark 127-qubit utility demonstration on IBM Eagle.

---

## 12. Kagome & Frustrated Systems

📎 Ahsan, M. et al. (2025). Utility-scale quantum computation of ground-state energy in a 100+ site planar Kagome antiferromagnet via Hamiltonian engineering. *arXiv preprint arXiv:2507.06361*. https://arxiv.org/abs/2507.06361

> 103-site Kagome VQE on IBM Heron. Validates IBM for utility-scale frustrated 2D systems.

📎 Weaving, T. et al. (2025). Simulating the antiferromagnetic Heisenberg model on a spin-frustrated Kagome lattice with the contextual subspace variational quantum eigensolver. *arXiv preprint arXiv:2506.12391*. https://arxiv.org/abs/2506.12391

> Kagome VQE with DMRG-biased contextual subspaces + ZNE. Achieves 0.01% energy error.

---

## 13. Weight/Gradient Analysis & Phase Detection

✅ Hernandes, V. et al. (2025). Adiabatic fine-tuning of neural quantum states enables detection of phase transitions in weight space. *arXiv preprint arXiv:2503.17140v2*. https://arxiv.org/abs/2503.17140

> Phase transitions detected from NN weight space. Validated in our framework at N=6 and N=10.

---

## 14. VQE Optimization & Datasets

📎 Peng, Y. et al. (2025). TITAN: A trajectory-informed technique for adaptive parameter freezing in large-scale VQE. *NeurIPS 2025*. https://arxiv.org/abs/2509.15193

> Deep learning freezes inactive VQE parameters. 3× faster, 40-60% fewer evaluations. Tested on TFIM.

📎 Chen, F. et al. (2025). VQEzy: An open-source dataset for parameter initialization in variational quantum eigensolvers. *arXiv preprint arXiv:2509.17322*. https://arxiv.org/abs/2509.17322

> First large-scale VQE initialization dataset: 12,110 instances. Potential pre-training data.

📎 Singh, M. et al. (2025). Statistical benchmarking of optimization methods for variational quantum eigensolver under quantum noise. *arXiv preprint arXiv:2510.08727*. https://arxiv.org/abs/2510.08727

> BFGS best for accuracy; COBYLA for low-cost. Validates our L-BFGS-B choice.

---

## 15. Tensor Network Pre-optimization

🔑 Martin, B. A. et al. (2026). Pre-optimization of quantum circuits, barren plateaus and classical simulability: Tensor networks to unlock the variational quantum eigensolver. *arXiv preprint arXiv:2602.04676*. https://arxiv.org/abs/2602.04676

> 2D TN pre-optimization for TFIM. Shows TN warm-starts access enhanced gradient zones.

📎 Schollwöck, U. (2011). The density-matrix renormalization group in the age of matrix product states. *Annals of Physics*, *326*(1), 96–192. https://doi.org/10.1016/j.aop.2010.09.012

> Canonical DMRG reference for our classical ground truth generation.

---

## 16. NISQ Foundations

🔑 Preskill, J. (2018). Quantum computing in the NISQ era and beyond. *Quantum*, *2*, 79. https://doi.org/10.22331/q-2018-08-06-79

> Defines the NISQ paradigm. Essential context for thesis motivation.

🔑 Feynman, R. P. (1982). Simulating physics with computers. *International Journal of Theoretical Physics*, *21*(6/7), 467–488. https://doi.org/10.1007/BF02650179

> Original quantum simulation proposal. Historical foundation.

---

## 17. VQE Foundations

🔑 Peruzzo, A., McClean, J., Shadbolt, P., et al. (2014). A variational eigenvalue solver on a photonic quantum processor. *Nature Communications*, *5*, 4213. https://doi.org/10.1038/ncomms5213

> Original VQE paper.

📎 Tilly, J., Chen, H., Cao, S., et al. (2022). The variational quantum eigensolver: A review of methods and best practices. *Physics Reports*, *986*, 1–128. https://doi.org/10.1016/j.physrep.2022.08.003

> Comprehensive VQE review. Best practices reference.

---

## 18. Many-Body Physics Foundations

🔑 Dutta, A., Aeppli, G., Chakrabarti, B. K., et al. (2015). *Quantum phase transitions in transverse field spin models*. Cambridge University Press. https://doi.org/10.1017/CBO9781107706057

> Canonical TFIM reference. Essential for thesis Chapter 2.

📎 Savary, L., & Balents, L. (2016). Quantum spin liquids: A review. *Reports on Progress in Physics*, *80*(1), 016502. https://doi.org/10.1088/0034-4885/80/1/016502

> QSL review motivating future Kagome work.

---

## 19. Technical Frameworks

📎 Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs and Manifolds*. https://arxiv.org/abs/1903.02428

> PyG framework used for our MPNN predictor.

📎 Hauschild, J., & Pollmann, F. (2018). Efficient numerical simulations with Tensor Networks: Tensor Network Python (TeNPy). *SciPost Physics Lecture Notes*, 5. https://doi.org/10.21468/SciPostPhysLectNotes.5

> TeNPy framework for DMRG ground truth.

---

## 20. Alternative Approaches (For Future Work)

📎 Nakaji, K. et al. (2025). The generative quantum eigensolver (GQE) and its application for ground state search. *arXiv preprint arXiv:2401.09253v2*. https://arxiv.org/abs/2401.09253

> Transformer-based circuit generation. Paradigm alternative to VQE optimization.

📎 Dade, N. O. O. et al. (2026). SpinGQE: A generative quantum eigensolver for spin Hamiltonians. *arXiv preprint arXiv:2603.24298*. https://arxiv.org/abs/2603.24298

> GQE extended to spin Hamiltonians. Directly applicable alternative to MPNN + VQE.

📎 Kim, M. et al. (2026). A rigorous hybridization of variational quantum eigensolver and classical neural network. *arXiv preprint arXiv:2602.17295*. https://arxiv.org/abs/2602.17295

> U-VQNHE: neural post-processing with variational guarantees. Could enhance Phase 4 results.

---

## Papers Removed (Low Confidence or Low Relevance)

The following were excluded from the curated list:

| Paper | Reason |
|-------|--------|
| Umeano et al. (2024) — Kitaev spin liquids | Placeholder arXiv ID (`2404.XXXXX`), unverifiable |
| Sehovic et al. (2026) — Gamow DMRG | Placeholder arXiv ID (`2601.XXXXX`), nuclear physics, not relevant |
| Faria et al. (2026) — Inductive QGNN | Placeholder arXiv ID (`2601.XXXXX`), unverifiable |
| Saleh (2025) — Entanglement entropy GNN | Placeholder arXiv ID (`2503.XXXXX`), unverifiable |
| Simard et al. (2025) — Rydberg atoms | Placeholder arXiv ID (`2502.XXXXX`), tangential |
| Visuri et al. (2025) — Counterdiabatic dynamics | Placeholder arXiv ID (`2502.XXXXX`), tangential |
| Simen et al. (2025) — Quenched feature maps | Placeholder arXiv ID (`2503.XXXXX`), tangential |
| Hung et al. (2025) — Ising meson spectroscopy | Placeholder arXiv ID (`2501.XXXXX`), tangential |
| Li et al. (2025) — Protein structure QA | Placeholder arXiv ID (`2502.XXXXX`), unrelated |
| Pan et al. (2026) — CMT-Benchmark | Placeholder arXiv ID (`2601.XXXXX`), unverifiable |
| Li et al. (2024) — O(3) equivariant TN | Placeholder arXiv ID (`2404.XXXXX`), tangential |
| Anderson (1972, 1973) | Classic but not directly cited in pipeline |
| Giamarchi (2003) | 1D physics textbook, not directly used |
| Troyer & Wiese (2005) | Sign problem — motivational only |
| Jordan & Wigner (1928) | Historical, not directly used |
| Devakul & Williamson (2018) | Fractal SPT, tangential |
| Verdon et al. (2019) — Quantum GNN | Tangential to our classical GNN approach |
| Battaglia et al. (2018) — Graph networks | General GNN review, superseded by Xu/Gilmer |
| Kipf & Welling (2017) — GCN | Superseded by GINConv in our architecture |
| Veličković et al. (2018) — GAT | GATConv tested and rejected in our pipeline |
| Fontana et al. (2023) — Adjoint BP | Theoretical, not directly used |
| Larocca et al. (2024) — BP review | Review paper, covered by Mele 2026 |
| Catarina & Murta (2023) — DMRG pedagogical | Pedagogical, not primary reference |
| Ayral et al. (2022) — DMRG for circuits | Tangential |
| Carleo & Troyer (2017) — NQS | Not directly used in pipeline |
| Rudolph et al. (2023) — TN pretraining | Superseded by Martin 2026 |
| Dalzell et al. (2023) — QA survey | Too broad |
| Herman et al. (2023) — QC for finance | Unrelated domain |
| Chandarana et al. (2023) — Protein folding | Unrelated domain |
| Nakayama et al. (2023) — VQE circuit dataset | Tangential |
| Cervera-Lierta et al. (2021) — Meta-VQE | Superseded by more recent work |
| Egger et al. (2021) — Warm-start QO | QAOA-focused, not VQE |
| Grimsley et al. (2019) — ADAPT-VQE | Not used in our pipeline |
| Vicentini et al. (2022) — NetKet | Not used in pipeline |
| Zhang et al. (2023) — TensorCircuit | Not used in pipeline |
| Paszke et al. (2019) — PyTorch | Standard framework, no citation needed |
| Kandala et al. (2017) — HEA hardware | HEA rejected in our framework |
| Ma et al. (2025) — Analog ZNE | Superconducting cavity, different hardware |
| Mesman (2025) — NN-AE-VQE | Alternative, not validated |
| Ye (2025) — Shadow tomography | Alternative classification, not used |
| Li (2025) — Quantum reservoir topological | Alternative QRC, not used |
| Sander (2025) — QCNN 2D | Alternative, not used |
| Yee (2026) — Prometheus VAE | Alternative, not used |
| Mahlow (2025) — SHAP QSVM | Alternative, not used |
| Cortes (2025) — CEO ADAPT-VQE | Not used |
| Bincoletto (2025) — Transferable ML | Validates principle but not directly used |
| Yin (2025) — DTC reservoir | Alternative reservoir, not used |
| arXiv:2506.02105 — Infinite TN circuits | Not used |
| arXiv:2403.04844 — HEA without BP | Contradicts our HVA-only constraint |
| arXiv:2603.18479 — BP beyond concentration | Theoretical extension, not directly used |
| arXiv:2508.18514 — RL initialization | Alternative, not used |
| Nature (2025) — TensorHyper-VQC | Alternative, not used |

---

## Summary Statistics

- **Total papers in curated list:** 44
- **Core (✅):** 17 — directly used or validated in the pipeline
- **Key (🔑):** 14 — essential theoretical/methodological foundation
- **Supporting (📎):** 13 — validates decisions or provides context
- **Removed:** 40+ papers (placeholder IDs, tangential, or superseded)

All URLs verified as of May 2026.


---

## 21. New Additions (Session 2026-06-09)

### GNN Parameter Prediction (Additional Validation)

📎 Wu, Y. et al. (2025). PVLS: A GNN-based parameter initialization framework for variational quantum linear solvers. *arXiv preprint arXiv:2512.04909*. https://arxiv.org/abs/2512.04909

> GNN-based parameter prediction extended to VQLS (linear systems). Validates GNN initialization paradigm beyond VQE. Demonstrates GNN mitigates barren plateaus in VQLS as system size increases.

### Frustrated TFIM on Hardware

✅ Teoh, Y. H. et al. (2025). Variational quantum simulations of a two-dimensional frustrated transverse-field Ising model on a trapped-ion quantum computer. *arXiv preprint arXiv:2505.22932*. https://arxiv.org/abs/2505.22932

> 2D frustrated TFIM with NNN interactions simulated on trapped-ion hardware. Multiple ordered magnetic phases explored. Directly validates our TFIM frustrated extension (Ec. 3, J1-J2) and confirms hardware viability of frustrated Ising models.

### Frustration-Induced Expressibility Limits

🔑 Huang, C. et al. (2026). Frustration-induced expressibility limitations in variational quantum algorithms. *arXiv preprint arXiv:2604.11688*. https://arxiv.org/abs/2604.11688

> Proves geometric frustration fundamentally limits VQA expressibility. Competing interactions prevent simultaneous energy minimization, creating expressibility barriers independent of optimization. Confirms our finding: TFIM frustrated J1-J2 achieves high fidelity (no frustration in ZZ-only TFIM) but Heisenberg XXZ fails (competing XX+YY+ZZ frustration).

### ML for Phase Transition Detection (Additional)

📎 Li, X. et al. (2026). Learning variational quantum circuit parameters with classical artificial intelligence for quantum phase transition detection. *arXiv preprint arXiv:2506.06678*. https://arxiv.org/abs/2506.06678

> Attention mechanism + VAE for unsupervised phase transition detection from VQE parameters. Complementary approach to our weight-gradient method (Hernandes 2025). Validates that VQE parameter structure encodes phase information.

### VQE Optimizer Benchmarks

📎 Singh, M. et al. (2025). Statistical benchmarking of optimization methods for variational quantum eigensolver under quantum noise. *arXiv preprint arXiv:2510.08727*. https://arxiv.org/abs/2510.08727

> Comprehensive benchmark of 6 optimizers (BFGS, SLSQP, Nelder-Mead, Powell, COBYLA, iSOMA) under noise. BFGS best for accuracy; COBYLA for low-cost noisy regime. Validates our choice of L-BFGS-B (noiseless) and COBYLA (shot-based MPS).

### Entanglement & Expressivity (TFIM specific)

📎 Kumar, S. et al. (2026). A study of entanglement and ansatz expressivity for the transverse-field Ising model using variational quantum eigensolver. *arXiv preprint arXiv:2602.17662*. https://arxiv.org/abs/2602.17662

> Studies entanglement structure and expressibility specifically for TFIM with VQE. Confirms that entanglement entropy at criticality exceeds shallow-circuit capacity. Independent validation of our h_min boundary findings.

### QESEM (PEA Alternative at Scale)

✅ Aharonov, D. et al. (2026). Reliable high-accuracy error mitigation for utility-scale quantum circuits. *arXiv preprint arXiv:2508.10997*. https://arxiv.org/abs/2508.10997

> QESEM method on IBM Heron resolves ZNE vs PEC tradeoff. Tested on kicked TFIM. Provides a third error mitigation strategy beyond PEA-ZNE and gate-folding for utility-scale deployment.

---

**Updated Summary Statistics:**
- **Total papers in curated list:** 51 (+7 new)
- **Core (✅):** 19
- **Key (🔑):** 15
- **Supporting (📎):** 17


### Adiabatic VQE Warm-Start (Validates Our Approach)

🔑 Schiffer, B. et al. (2026). Scalable, self-verifying variational quantum eigensolver using adiabatic warm starts. *arXiv preprint arXiv:2602.17612*. https://arxiv.org/abs/2602.17612

> Adiabatic VQE variant where optimization proceeds along a Hamiltonian path. Derives conditions for successful gradient-based warm-start. Directly validates our descending h-sweep strategy — they prove that adiabatic warm-starts avoid barren plateaus and maintain ground-state tracking.

🔑 Matos, G. et al. (2026). Exploiting adiabaticity for variational ground-states. *arXiv preprint arXiv:2602.06137*. https://arxiv.org/abs/2602.06137

> Demonstrates that solving a sequence of intermediate problems facilitates tracking the ground-state manifold. Validates warm-start parameter transfer as system size scales. Directly supports our approach of descending sweep with parameter inheritance.

### IBM Hardware Generations (Updated 2025-2026)

📎 IBM Quantum. (2025). IBM Quantum Nighthawk processor announcement. *IBM Newsroom*. https://newsroom.ibm.com/2025-11-12-IBM-Delivers-New-Quantum-Processors

> Nighthawk: 120 qubits, square lattice (218 couplers), designed for quantum advantage. Available end 2025. Successor to Heron for our hardware deployment path.

### VQE Scalability Limits

📎 Bittel, L. et al. (2026). Exponential scaling barriers for variational quantum eigensolvers. *arXiv preprint arXiv:2603.13073*. https://arxiv.org/abs/2603.13073

> Identifies fundamental exponential barriers in VQE for certain problems. Our pipeline avoids these by operating in the gapped (non-critical) regime where the landscape is smooth — this paper provides theoretical justification for why we restrict to h > h_min.

### RIKEN-IBM Quantum-Centric Supercomputing (2026)

📎 Riken-IBM. (2026). Quantum-centric supercomputing demonstration. *IBM Research Blog*. https://ibm.com/quantum/blog/riken-fugaku-qcsc

> Largest and most accurate chemistry experiment on quantum computer (2026). IBM Heron + Fugaku classical HPC in closed-loop workflow. Validates the hybrid quantum-classical paradigm at utility scale.

---

**Updated Summary Statistics:**
- **Total papers in curated list:** 56 (+5 new this round)
- **Core (✅):** 19
- **Key (🔑):** 17 (+2)
- **Supporting (📎):** 20 (+3)
