# Bibliography — Hybrid GNN-HVA Framework for Topological Phase Characterization

Comprehensive reference list for the Master's Thesis (TFM). Organized by topic, APA 7th edition format.

---

## 1. Foundations of Many-Body Physics & Quantum Spin Liquids

Anderson, P. W. (1972). More is different. *Science*, *177*(4047), 393–396. https://doi.org/10.1126/science.177.4047.393

Anderson, P. W. (1973). Resonating valence bonds: A new kind of insulator? *Materials Research Bulletin*, *8*(2), 153–160. https://doi.org/10.1016/0025-5408(73)90167-0

Savary, L., & Balents, L. (2016). Quantum spin liquids: A review. *Reports on Progress in Physics*, *80*(1), 016502. https://doi.org/10.1088/0034-4885/80/1/016502

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

Carleo, G., & Troyer, M. (2017). Solving the quantum many-body problem with artificial neural networks. *Science*, *355*(6325), 602–606. https://doi.org/10.1126/science.aag2302

Rudolph, M. S., Chen, J., Miller, J., Acharya, A., & Perdomo-Ortiz, A. (2023). Synergistic pretraining of parametrized quantum circuits via tensor networks. *Nature Communications*, *14*, 8367. https://doi.org/10.1038/s41467-023-43908-6

---

## 9. Graph Neural Networks

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *Proceedings of the 5th International Conference on Learning Representations (ICLR)*.

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. *Proceedings of the 6th International Conference on Learning Representations (ICLR)*.

Battaglia, P. W., Hamrick, J. B., Bapst, V., Sanchez-Gonzalez, A., Zambaldi, V., Malinowski, M., Tacchetti, A., Raposo, D., Santoro, A., Faulkner, R., Gulcehre, C., Song, F., Ballard, A., Gilmer, J., Dahl, G., Vaswani, A., Allen, K., Nash, C., Langston, V., … Pascanu, R. (2018). Relational inductive biases, deep learning, and graph networks. *arXiv preprint arXiv:1806.01261*.

---

## 10. Spin Systems vs. Quantum Chemistry (Architectural Justification)

Jordan, P., & Wigner, E. (1928). Über das Paulische Äquivalenzverbot. *Zeitschrift für Physik*, *47*(9–10), 631–651. https://doi.org/10.1007/BF01331938

Kandala, A., Mezzacapo, A., Temme, K., Takita, M., Brink, M., Chow, J. M., & Gambetta, J. M. (2017). Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets. *Nature*, *549*, 242–246. https://doi.org/10.1038/nature23879

---

## 11. Technical Frameworks & Software

IBM Quantum. (2024). *Qiskit 2.x documentation: Primitives V2 interface*. https://docs.quantum.ibm.com

Hauschild, J., & Pollmann, F. (2018). Efficient numerical simulations with Tensor Networks: Tensor Network Python (TeNPy). *SciPost Physics Lecture Notes*, 5. https://doi.org/10.21468/SciPostPhysLectNotes.5

Vicentini, F., Hofmann, D., Szabó, A., Wu, D., Roth, C., Giuliani, C., Pescia, G., Nys, J., Vargas-Calderón, V., Astrakhantsev, N., & Carleo, G. (2022). NetKet 3: Machine learning toolbox for many-body quantum systems. *SciPost Physics Codebases*, 7. https://doi.org/10.21468/SciPostPhysCodeb.7

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, *32*.

Zhang, S.-X., Allcock, J., Wan, Z.-Q., Liu, S., Sun, J., Yu, H., Yang, X.-H., Qiu, J., Ye, Z., Chen, Y.-Q., Lee, C.-K., Zheng, Y.-C., Jian, S.-K., Yao, H., Hsieh, C.-Y., & Zhang, S. (2023). TensorCircuit: A quantum software framework for the NISQ era. *Quantum*, *7*, 912. https://doi.org/10.22331/q-2023-02-02-912
