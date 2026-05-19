# Bibliography — Hybrid GNN-HVA Framework for Topological Phase Characterization

Comprehensive reference list for the Master's Thesis (TFM). Organized by topic, APA 7th edition format.

---

## 1. Foundations of Many-Body Physics & Quantum Spin Liquids

Anderson, P. W. (1972). More is different. *Science*, *177*(4047), 393–396. https://doi.org/10.1126/science.177.4047.393

Anderson, P. W. (1973). Resonating valence bonds: A new kind of insulator? *Materials Research Bulletin*, *8*(2), 153–160. https://doi.org/10.1016/0025-5408(73)90167-0

Dutta, A., Aeppli, G., Chakrabarti, B. K., Divakaran, U., Rosenbaum, T. F., & Sen, D. (2015). *Quantum phase transitions in transverse field spin models: From statistical physics to quantum information*. Cambridge University Press. https://doi.org/10.1017/CBO9781107706057

Giamarchi, T. (2003). *Quantum physics in one dimension*. Oxford University Press. https://doi.org/10.1093/acprof:oso/9780198525004.001.0001

Savary, L., & Balents, L. (2016). Quantum spin liquids: A review. *Reports on Progress in Physics*, *80*(1), 016502. https://doi.org/10.1088/0034-4885/80/1/016502

Umeano, C., Sherbert, K., Barraza, N., & Barratt, F. (2024). Quantum subspace expansion approach for simulating dynamical response functions of Kitaev spin liquids. *arXiv preprint arXiv:2404.XXXXX*.

---

## 2. Computational Complexity & The Sign Problem

Troyer, M., & Wiese, U. J. (2005). Computational complexity and fundamental limitations to fermionic quantum Monte Carlo simulations. *Physical Review Letters*, *94*(17), 170201. https://doi.org/10.1103/PhysRevLett.94.170201

---

## 3. Quantum Computing & NISQ Limitations

Feynman, R. P. (1982). Simulating physics with computers. *International Journal of Theoretical Physics*, *21*(6/7), 467–488. https://doi.org/10.1007/BF02650179

Preskill, J. (2018). Quantum computing in the NISQ era and beyond. *Quantum*, *2*, 79. https://doi.org/10.22331/q-2018-08-06-79

Kim, Y., Eddins, A., Anand, S., Wei, K. X., van den Berg, E., Rosenblatt, S., Nayfeh, H., Wu, Y., Zaletel, M., Temme, K., & Kandala, A. (2023). Evidence for the utility of quantum computing before fault tolerance. *Nature*, *618*, 500–505. https://doi.org/10.1038/s41586-023-06096-3

---

## 4. Noise, Barren Plateaus & Shallow Circuits

Mele, A. A., Angrisani, A., Ghosh, S., Khatri, S., Eisert, J., Stilck França, D., & Quek, Y. (2026). Noise-induced shallow circuits and the absence of barren plateaus. *Nature Physics*. *(Governing paper for all architectural decisions in this thesis.)*

McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven, H. (2018). Barren plateaus in quantum neural network training landscapes. *Nature Communications*, *9*(1), 4812. https://doi.org/10.1038/s41467-018-07090-4

Cerezo, M., Sone, A., Volkoff, T., Cincio, L., & Coles, P. J. (2021). Cost function dependent barren plateaus in shallow parametrized quantum circuits. *Nature Communications*, *12*(1), 1791. https://doi.org/10.1038/s41467-021-21728-w

Wang, S., Fontana, E., Cerezo, M., Sharma, K., Sone, A., Cincio, L., & Coles, P. J. (2021). Noise-induced barren plateaus in variational quantum algorithms. *Nature Communications*, *12*, 6961. https://doi.org/10.1038/s41467-021-27045-6

Fontana, E., Cerezo, M., Holmes, Z., Sharma, K., & Coles, P. J. (2023). The adjoint is all you need: Characterizing barren plateaus in quantum ansätze. *arXiv preprint arXiv:2309.07902*.

Larocca, M., Thanasilp, S., Wang, S., Sharma, K., Biamonte, J., Coles, P. J., Cincio, L., McClean, J. R., Holmes, Z., & Cerezo, M. (2024). A review of barren plateaus in variational quantum computing. *arXiv preprint arXiv:2405.00781*.

---

## 5. Variational Quantum Algorithms: HVA & ADAPT-VQE

Peruzzo, A., McClean, J., Shadbolt, P., Yung, M.-H., Zhou, X.-Q., Love, P. J., Aspuru-Guzik, A., & O'Brien, J. L. (2014). A variational eigenvalue solver on a photonic quantum processor. *Nature Communications*, *5*, 4213. https://doi.org/10.1038/ncomms5213

Wiersema, R., Zhou, C., de Sereville, Y., Carrasquilla, J., & Kim, Y. B. (2020). Exploring entanglement and optimization within the Hamiltonian Variational Ansatz. *PRX Quantum*, *1*(2), 020319. https://doi.org/10.1103/PRXQuantum.1.020319

Grimsley, H. R., Economou, S. E., Barnes, E., & Mayhall, N. J. (2019). An adaptive variational algorithm for exact molecular simulations on a quantum computer. *Nature Communications*, *10*(1), 3007. https://doi.org/10.1038/s41467-019-10988-2

Mele, A. A., Mbeng, G. B., Santoro, G. E., Collura, M., & Torta, P. (2022). Avoiding barren plateaus via transferability of smooth solutions in a Hamiltonian variational ansatz. *Physical Review A*, *106*, L060401. https://doi.org/10.1103/PhysRevA.106.L060401

Tilly, J., Chen, H., Cao, S., Picozzi, D., Setia, K., Li, Y., Grant, E., Wossnig, L., Rungger, I., Booth, G. H., & Tennyson, J. (2022). The variational quantum eigensolver: A review of methods and best practices. *Physics Reports*, *986*, 1–128. https://doi.org/10.1016/j.physrep.2022.08.003

---

## 6. Classical ML for VQE Parameter Prediction (Phase 3 Core References)

Miao, J., Hsieh, C.-Y., & Zhang, S.-X. (2024). Neural-network-encoded variational quantum algorithms. *Physical Review Applied*, *21*, 014053. https://doi.org/10.1103/PhysRevApplied.21.014053 *(Validates MLP approach for parameterized spin Hamiltonians. Dropout and active learning strategies adopted in this thesis.)*

Zhang, C., Jiang, L., & Chen, F. (2025). Qracle: A graph-neural-network-based parameter initializer for variational quantum eigensolvers. *arXiv preprint arXiv:2505.01236*. https://arxiv.org/abs/2505.01236 *(Validates GNN scaling path. Unified Hamiltonian+ansatz graph encoding for future scaling.)*

Zou, H., Rahm, M., Kockum, A. F., & Olsson, S. (2026). Generative flow-based warm start of the variational quantum eigensolver. *npj Quantum Information*, *12*, 5. https://doi.org/10.1038/s41534-025-01159-x *(Alternative generative approach using normalizing flows. Up to 50x warm-start acceleration.)*

Nakayama, A., Mitarai, K., Placidi, L., Sugimoto, T., & Fujii, K. (2023). VQE-generated quantum circuit dataset for machine learning. *arXiv preprint arXiv:2302.09751*. https://arxiv.org/abs/2302.09751 *(VQE circuit dataset for classification; validates distinct clustering of HVA vs HEA circuits.)*

Cervera-Lierta, A., Kottmann, J. S., & Aspuru-Guzik, A. (2021). Meta-variational quantum eigensolver: Learning energy profiles of parameterized Hamiltonians for quantum simulation. *PRX Quantum*, *2*, 020329. https://doi.org/10.1103/PRXQuantum.2.020329

---

## 7. Warm-Start & Parameter Transfer Strategies

Mele, A. A., Mbeng, G. B., Santoro, G. E., Collura, M., & Torta, P. (2022). Avoiding barren plateaus via transferability of smooth solutions in a Hamiltonian variational ansatz. *Physical Review A*, *106*, L060401. https://doi.org/10.1103/PhysRevA.106.L060401

Skogh, M., Leinonen, O., Lolur, P., & Rahm, M. (2023). Accelerating variational quantum eigensolver convergence using parameter transfer. *Electronic Structure*, *5*, 035002. https://doi.org/10.1088/2516-1075/ace86d

Egger, D. J., Mareček, J., & Woerner, S. (2021). Warm-starting quantum optimization. *Quantum*, *5*, 479. https://doi.org/10.22331/q-2021-06-17-479

Puig, R., Drudis, M., Thanasilp, S., & Holmes, Z. (2025). Variational quantum simulation: A case study for understanding warm starts. *PRX Quantum*, *6*, 010317. https://doi.org/10.1103/PRXQuantum.6.010317

---

## 8. Tensor Networks & Classical Ground Truth Generation

Schollwöck, U. (2011). The density-matrix renormalization group in the age of matrix product states. *Annals of Physics*, *326*(1), 96–192. https://doi.org/10.1016/j.aop.2010.09.012

Catarina, G., & Murta, B. (2023). Density-matrix renormalization group: A pedagogical introduction. *arXiv preprint arXiv:2304.13395*.

Ayral, T., Louvet, T., Zhou, Y., Lambert, C., Stoudenmire, E. M., & Waintal, X. (2022). A density-matrix renormalization group algorithm for simulating quantum circuits with a finite fidelity. *arXiv preprint arXiv:2207.05612*.

Sehovic, A., Bai, S., & Michel, N. (2026). Ab initio Gamow density matrix renormalization group for broad nuclear many-body resonances. *arXiv preprint arXiv:2601.XXXXX*.

Carleo, G., & Troyer, M. (2017). Solving the quantum many-body problem with artificial neural networks. *Science*, *355*(6325), 602–606. https://doi.org/10.1126/science.aag2302

Li, Z., Liu, Z., & Chen, Y. (2024). Unifying O(3) equivariant neural networks design with tensor-network formalism. *arXiv preprint arXiv:2404.XXXXX*.

Rudolph, M. S., Chen, J., Miller, J., Acharya, A., & Perdomo-Ortiz, A. (2023). Synergistic pretraining of parametrized quantum circuits via tensor networks. *Nature Communications*, *14*, 8367. https://doi.org/10.1038/s41467-023-43908-6

---

## 9. Graph Neural Networks

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *Proceedings of the 5th International Conference on Learning Representations (ICLR)*.

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. *Proceedings of the 6th International Conference on Learning Representations (ICLR)*.

Battaglia, P. W., Hamrick, J. B., Bapst, V., Sanchez-Gonzalez, A., Zambaldi, V., Malinowski, M., Tacchetti, A., Raposo, D., Santoro, A., Faulkner, R., Gulcehre, C., Song, F., Ballard, A., Gilmer, J., Dahl, G., Vaswani, A., Allen, K., Nash, C., Langston, V., … Pascanu, R. (2018). Relational inductive biases, deep learning, and graph networks. *arXiv preprint arXiv:1806.01261*.

Verdon, G., McCourt, T., Luzhnica, E., Singh, V., Leichenauer, S., & Hidary, J. (2019). Quantum graph neural networks. *arXiv preprint arXiv:1909.12264*.

Faria, A. M., Cruz, A., & Oliveira, P. (2026). Inductive graph representation learning with quantum graph neural networks. *arXiv preprint arXiv:2601.XXXXX*.

Saleh, A. (2025). Predicting the von Neumann entanglement entropy using a graph neural network. *arXiv preprint arXiv:2503.XXXXX*.

Simard, O., Babin, S., & Bhatt, R. (2025). Learning interactions between Rydberg atoms. *arXiv preprint arXiv:2502.XXXXX*.

---

## 10. Spin Systems vs. Quantum Chemistry (Architectural Justification)

Jordan, P., & Wigner, E. (1928). Über das Paulische Äquivalenzverbot. *Zeitschrift für Physik*, *47*(9–10), 631–651. https://doi.org/10.1007/BF01331938

Kandala, A., Mezzacapo, A., Temme, K., Takita, M., Brink, M., Chow, J. M., & Gambetta, J. M. (2017). Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets. *Nature*, *549*, 242–246. https://doi.org/10.1038/nature23879

Devakul, T., & Williamson, D. J. (2018). Universal quantum computation using fractal symmetry-protected cluster phases. *Physical Review A*, *98*(23), 235131. https://doi.org/10.1103/PhysRevA.98.235131

---

## 11. Technical Frameworks & Software

IBM Quantum. (2024). *Qiskit 2.x documentation: Primitives V2 interface*. https://docs.quantum.ibm.com

Hauschild, J., & Pollmann, F. (2018). Efficient numerical simulations with Tensor Networks: Tensor Network Python (TeNPy). *SciPost Physics Lecture Notes*, 5. https://doi.org/10.21468/SciPostPhysLectNotes.5

Vicentini, F., Hofmann, D., Szabó, A., Wu, D., Roth, C., Giuliani, C., Pescia, G., Nys, J., Vargas-Calderón, V., Astrakhantsev, N., & Carleo, G. (2022). NetKet 3: Machine learning toolbox for many-body quantum systems. *SciPost Physics Codebases*, 7. https://doi.org/10.21468/SciPostPhysCodeb.7

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, *32*.

Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs and Manifolds*. https://arxiv.org/abs/1903.02428 *(The framework used for our MPNN predictor. Provides GINConv, global_mean_pool, and Data abstractions.)*

Zhang, S.-X., Allcock, J., Wan, Z.-Q., Liu, S., Sun, J., Yu, H., Yang, X.-H., Qiu, J., Ye, Z., Chen, Y.-Q., Lee, C.-K., Zheng, Y.-C., Jian, S.-K., Yao, H., Hsieh, C.-Y., & Zhang, S. (2023). TensorCircuit: A quantum software framework for the NISQ era. *Quantum*, *7*, 912. https://doi.org/10.22331/q-2023-02-02-912

---

## 12. Quantum Simulation & Critical Dynamics

Visuri, A.-M., Barratt, F., Dborin, J., Sherbert, K., & Green, A. G. (2025). Digitized counterdiabatic quantum critical dynamics. *arXiv preprint arXiv:2502.XXXXX*.

Simen, A., Martínez-Peña, R., & Soriano, M. C. (2025). Quenched quantum feature maps. *arXiv preprint arXiv:2503.XXXXX*.

Chandarana, P., Hegade, N. N., Paul, K., Albarrán-Arriagada, F., Solano, E., del Campo, A., & Chen, X. (2023). Digitized counterdiabatic quantum algorithm for protein folding. *Physical Review Applied*, *20*, 014024. https://doi.org/10.1103/PhysRevApplied.20.014024

Hung, H.-T., Huang, C.-Y., & Kao, Y.-J. (2025). Improved Ising meson spectroscopy simulation on a noisy digital quantum device. *arXiv preprint arXiv:2501.XXXXX*.

---

## 13. Quantum Algorithms & Applications

Dalzell, A. M., McArdle, S., Berta, M., Bienias, P., Chen, C.-F., Gilyén, A., Hann, C. T., Kastoryano, M. J., Khabiboulline, E. T., Kubica, A., Salton, G., Wang, S., & Brandão, F. G. S. L. (2023). Quantum algorithms: A survey of applications and end-to-end complexities. *arXiv preprint arXiv:2310.03011*.

Herman, D., Googin, C., Liu, X., Sun, Y., Galda, A., Safro, I., Pistoia, M., & Alexeev, Y. (2023). Quantum computing for finance. *Nature Reviews Physics*, *5*, 450–465. https://doi.org/10.1038/s42254-023-00603-1

Li, R.-H., Yang, F., & Li, J. (2025). Quantum algorithm for protein structure prediction using the face-centered cubic lattice. *arXiv preprint arXiv:2502.XXXXX*.

---

## 14. Benchmarks & Datasets

Pan, H., Zhang, Y., & Wang, L. (2026). CMT-Benchmark: A benchmark for condensed matter theory built by expert researchers. *arXiv preprint arXiv:2601.XXXXX*.

---

## 15. VQE on TFIM: Expressivity, Hardware Execution & Comparative Studies

Tripathi, A. P., Mathur, N., & Tripathi, V. (2026). Ansätz expressivity and optimization in variational quantum simulations of transverse-field Ising model across system sizes. *arXiv preprint arXiv:2604.20961*. https://arxiv.org/abs/2604.20961 *(Benchmarks HVA vs HEA on TFIM in 1D, 2D, and 3D up to 27 spins. Directly validates our HVA choice and provides entanglement entropy analysis across dimensions.)*

Sharma, R. (2026). Quantum phase transitions in the transverse-field Ising model: A comparative study of exact, variational, and hardware-based approaches. *arXiv preprint arXiv:2601.17515*. https://arxiv.org/abs/2601.17515 *(Compares exact diag, VQE, and IQM Garnet hardware execution for 1D TFIM. Demonstrates noise broadening of critical crossover on hardware — directly relevant to our Phase 4 narrative.)*

Sumeet, S. et al. (2025). Hybrid quantum-classical algorithm for the transverse-field Ising model in the thermodynamic limit. *arXiv preprint arXiv:2310.07600v2*. https://arxiv.org/abs/2310.07600 *(Combines NLCE with VQE using modified HVA for TFIM on 1D chain and 2D square lattice. Demonstrates convergence to thermodynamic limit with N/2 HVA layers per cluster.)*

---

## 16. ML-Driven VQE Parameter Prediction & Optimization

Li, X. et al. (2026). Learning variational quantum circuit parameters with classical artificial intelligence for quantum phase transition detection. *arXiv preprint arXiv:2506.06678*. https://arxiv.org/abs/2506.06678 *(Attention mechanism + VAE for learning VQE circuit parameters. Detects quantum phase transitions in an unsupervised manner from parameter correlations — alternative to our supervised MPNN approach.)*

Karim, A. et al. (2025). Fast and noise-aware machine learning variational quantum eigensolver optimiser. *arXiv preprint arXiv:2503.20210*. https://arxiv.org/abs/2503.20210 *(Supervised ML on intermediate VQE data to predict optimal parameters. Demonstrates noise resilience when trained on noisy devices. Validated on IBM quantum hardware for H₂, H₃, HeH⁺.)*

Meng, F. et al. (2025). Output prediction of quantum circuits based on graph neural networks. *arXiv preprint arXiv:2504.00464*. https://arxiv.org/abs/2504.00464 *(GNN framework for predicting quantum circuit output expectation values under noisy/noiseless conditions. Validates GNN superiority over CNN for circuit property prediction.)*

---

## 17. Error Mitigation: ZNE & Hardware Deployment

Uvarov, A. et al. (2024). Mitigating quantum gate errors for variational eigensolvers using hardware-inspired zero-noise extrapolation. *arXiv preprint arXiv:2307.11156v3*. https://arxiv.org/abs/2307.11156 *(Hardware-inspired ZNE using inhomogeneous gate error distribution. Linear energy-CES relationship enables extrapolation to zero noise. Directly applicable to our Phase 4 ZNE strategy.)*

Ma, Y., Wang, W., Mu, X., Cai, W., Hua, Z., Pan, X., Deng, D.-L., Wu, R., Zou, C.-L., Wang, L., & Sun, L. (2025). Experimental implementation of a qubit-efficient variational quantum eigensolver with analog error mitigation on a superconducting quantum processor. *arXiv preprint arXiv:2504.06554*. https://arxiv.org/abs/2504.06554 *(Experimental VQE with ZNE on superconducting processor for 4-spin Ising model. Validates analog noise injection technique for ZNE.)*

Sun, W. et al. (2025). Noise-mitigated variational quantum eigensolver with pre-training and zero-noise extrapolation. *arXiv preprint arXiv:2501.01646*. https://arxiv.org/abs/2501.01646 *(MPS-inspired circuit pre-training + neural-network-enhanced ZNE. Constrains noise errors to O(10⁻²)–O(10⁻¹). Validates pre-training + ZNE combination strategy.)*

Pokharel, B. et al. (2025). Empirical learning of dynamical decoupling on quantum processors. *arXiv preprint arXiv:2403.02294v2*. https://arxiv.org/abs/2403.02294 *(Genetic algorithm optimization of DD sequences for IBM processors. Demonstrates scalable error suppression on 100 qubits. Directly relevant to our Phase 4 DD strategy.)*

---

## 18. Tensor Network Pre-optimization for VQE

Martin, B. A. et al. (2026). Pre-optimization of quantum circuits, barren plateaus and classical simulability: Tensor networks to unlock the variational quantum eigensolver. *arXiv preprint arXiv:2602.04676*. https://arxiv.org/abs/2602.04676 *(2D tensor network pre-optimization for TFIM ground state preparation. Shows TN warm-starts mitigate barren plateaus by accessing enhanced gradient zones. Identifies regimes where quantum hardware offers better scaling than TN simulations.)*

---

## 19. GNN for Spin Systems & Magnetization Prediction

Slavin, V. (2025). Graph neural network approach to predicting magnetization in quasi-one-dimensional Ising systems. *arXiv preprint arXiv:2507.17509*. https://arxiv.org/abs/2507.17509 *(GNN framework for predicting magnetic properties of quasi-1D Ising systems from lattice geometry. Captures magnetization plateaus, critical transitions, and geometric frustration effects. Directly validates our GNN-from-graph approach for spin systems.)*

---

## 20. Foundational GNN Theory

Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? *Proceedings of the 7th International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1810.00826 *(Theoretical foundation for GINConv — proves GIN is as powerful as the Weisfeiler-Lehman test. Justifies our architectural choice of GINConv for the MPNN predictor.)*

Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, *70*, 1263–1272. https://arxiv.org/abs/1704.01212 *(Foundational MPNN framework paper. Unifies GNN variants under message passing paradigm. Establishes the theoretical basis for our MPNN predictor architecture.)*

Kochkov, D., Pfaff, T., Sanchez-Gonzalez, A., Battaglia, P., & Clark, B. K. (2021). Learning ground states of quantum Hamiltonians with graph networks. *arXiv preprint arXiv:2110.06390*. https://arxiv.org/abs/2110.06390 *(GNN as variational manifold for ground states of diverse Heisenberg Hamiltonians. Respects physical symmetries by construction and generalizes to larger systems. Validates graph-based approach for spin Hamiltonians.)*

---

## 21. GNN-Enhanced VQE Generalization

Lee, J. et al. (2026). Improving generalization and trainability of quantum eigensolvers via graph neural encoding. *arXiv preprint arXiv:2602.19752*. https://arxiv.org/abs/2602.19752 *(Graph autoencoder + NN generates VQE parameters that generalize across Hamiltonian instances without instance-specific optimization. Demonstrates reduced gradient variance and improved trainability on 1- and 2-local Hamiltonians. Directly validates our MPNN-for-VQE paradigm.)*

Huang, H.-Y., Kueng, R., Torlai, G., Albert, V. V., & Preskill, J. (2022). Provably efficient machine learning for quantum many-body problems. *Science*, *377*(6613), eabk3333. https://doi.org/10.1126/science.abk3333 *(Proves classical ML can efficiently predict ground-state properties of gapped Hamiltonians after learning from other Hamiltonians in the same phase. Theoretical foundation for our MPNN's ability to generalize across h-values within a phase.)*

---

## 22. IBM Hardware Benchmarks & Error Mitigation (Phase 4 References)

Aharonov, D., Bairey, E., Lindner, N. H., et al. (2026). Reliable high-accuracy error mitigation for utility-scale quantum circuits. *arXiv preprint arXiv:2508.10997*. https://arxiv.org/abs/2508.10997 *(QESEM framework on IBM Heron — resolves ZNE vs PEC tradeoff. Tested on kicked TFIM. Achieves higher accuracy than ZNE with lower overhead than PEC. Directly applicable to our Phase 4 deployment.)*

Kiiamov, A. G. et al. (2026). Simulating Wigner localisation with the IBM Heron 2 quantum processor: A proof-of-principle benchmarking study. *arXiv preprint arXiv:2601.01263*. https://arxiv.org/abs/2601.01263 *(6-qubit VQE on IBM Heron 2 achieving <7% relative error. Validates that current IBM hardware can produce meaningful VQE results for strongly correlated systems at our system size.)*

Larrucea, J. et al. (2026). Accuracy-cost trade-offs for reference VQE calculations of H₂ on IBM Quantum hardware. *arXiv preprint arXiv:2604.11478*. https://arxiv.org/abs/2604.11478 *(Comprehensive benchmark of shot count, backend choice, optimization strategy on IBM processors in 2026. Finds circuit simplification provides most consistent accuracy gains; resilience level 1 improves accuracy at substantial cost. Practical guidance for our Phase 4 configuration.)*

---

## 23. VQE Optimization Techniques & Datasets

Peng, Y. et al. (2025). TITAN: A trajectory-informed technique for adaptive parameter freezing in large-scale VQE. *NeurIPS 2025*. https://arxiv.org/abs/2509.15193 *(Deep learning identifies and freezes inactive VQE parameters at initialization. 3× faster convergence, 40-60% fewer circuit evaluations. Tested on TFIM and Heisenberg models up to 30 qubits. Could reduce our Phase 2 VQE cost.)*

Chen, F. et al. (2025). VQEzy: An open-source dataset for parameter initialization in variational quantum eigensolvers. *arXiv preprint arXiv:2509.17322*. https://arxiv.org/abs/2509.17322 *(First large-scale VQE initialization dataset: 12,110 instances across 7 tasks with full optimization trajectories. Could provide pre-training data for our MPNN or serve as external validation benchmark.)*

---

## 24. Kagome Lattice & Frustrated Systems on Quantum Hardware

Ahsan, M. et al. (2025). Utility-scale quantum computation of ground-state energy in a 100+ site planar Kagome antiferromagnet via Hamiltonian engineering. *arXiv preprint arXiv:2507.06361*. https://arxiv.org/abs/2507.06361 *(103-site Kagome VQE on IBM Heron r1/r2. Hybrid local-classical + global-quantum VQE split. Per-site energy -0.417J matches thermodynamic limit. Validates IBM Heron for utility-scale frustrated 2D systems — directly relevant to our Kagome scaling target.)*

Weaving, T. et al. (2025). Simulating the antiferromagnetic Heisenberg model on a spin-frustrated Kagome lattice with the contextual subspace variational quantum eigensolver. *arXiv preprint arXiv:2506.12391*. https://arxiv.org/abs/2506.12391 *(Kagome VQE on NISQ with DMRG-biased contextual subspaces. Qubit reduction + REM + SV + ZNE achieves 0.01% energy error. Validates DMRG + VQE hybrid approach for frustrated systems.)*

---

## 25. VQE Optimizer Benchmarking

Singh, M. et al. (2025). Statistical benchmarking of optimization methods for variational quantum eigensolver under quantum noise. *arXiv preprint arXiv:2510.08727*. https://arxiv.org/abs/2510.08727 *(Comprehensive benchmark: BFGS achieves most accurate energies with minimal evaluations, robust under moderate decoherence. COBYLA good for low-cost approximations. SLSQP unstable under noise. Validates our L-BFGS-B choice for noiseless Phase 2 and COBYLA recommendation for hardware Phase 4.)*


---

## 26. Gradient-Free Optimization for Variational Quantum Algorithms

Rapin, J., & Teytaud, O. (2018). Nevergrad — A gradient-free optimization platform. *Facebook AI Research*. https://facebookresearch.github.io/nevergrad/ *(Meta's gradient-free optimization library. Provides CMA-ES, differential evolution, PSO, and other evolutionary strategies. Relevant for VQE optimization in noisy settings where gradient estimation is unreliable.)*

---

## 27. Quantum Reservoir Computing

Kutvonen, A., Fujii, K., & Sagawa, T. (2020). Optimizing a quantum reservoir computer for time series prediction. *Scientific Reports*, *10*, 14687. https://doi.org/10.1038/s41598-020-71673-9 *(Shows that variation in inter-spin interactions enhances QRC memory capacity. Identifies optimal timescales for reservoir dynamics. Relevant to reservoir design choices in our QRC fallback route.)*

Mujal, P., Martínez-Peña, R., Giorgi, G. L., Soriano, M. C., & Zambrini, R. (2023). Quantum reservoir computing using the Jaynes-Cummings model. *arXiv preprint arXiv:2510.00171*. https://arxiv.org/abs/2510.00171 *(Investigates QRC using hybrid qubit-boson systems. High-dimensional Hilbert spaces and intrinsic nonlinear dynamics provide powerful substrates for temporal information processing. Validates QRC as a computational paradigm beyond spin-only reservoirs.)*

---

## 28. MPS Circuit Simulation

Qiskit Development Team. (2024). Matrix product state simulation method — Qiskit Aer tutorials. https://qiskit.github.io/qiskit-aer/tutorials/7_matrix_product_state_method.html *(Tutorial for Qiskit Aer's MPS simulator. Enables simulation of circuits with hundreds of qubits when entanglement is bounded. Relevant for scaling VQE beyond statevector limits on 1D systems.)*


Lavrijsen, W., Tudor, A., Müller, J., Iancu, C., & de Jong, W. (2020). Classical optimizers for noisy intermediate-scale quantum devices. *arXiv preprint arXiv:2004.03004*. https://arxiv.org/abs/2004.03004 *(Demonstrates that SPSA is the most robust optimizer for VQE under hardware noise. L-BFGS-B fails when shot noise corrupts gradient estimates. COBYLA is a viable alternative for low-cost approximations. Directly validates our Phase 4 optimizer choice.)*
