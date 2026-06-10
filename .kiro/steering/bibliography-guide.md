---
inclusion: fileMatch
fileMatchPattern: "**/bibliography*.md,**/*.tex,**/tesis*"
---

# Bibliography Guide — Quick Reference for Literature Decisions

## Purpose
This file helps Kiro quickly identify which papers to cite for specific claims, find the right reference for a new assertion, avoid introducing papers that have been deliberately excluded, and PREVENT false novelty claims.

## CRITICAL: Attribution Rule (ALWAYS ENFORCE)

**Our contribution is INTEGRATION + SYSTEMATIC VALIDATION, not the individual techniques.**

### What IS Our Contribution (can say "se propone/valida/demuestra"):
1. Integration of GNN prediction + HVA + warm-start + PEA-ZNE into unified 4-phase pipeline
2. Systematic validation across 5 topologies + N=6-80 (430+ executions)
3. Cross-N generalization finding (BatchNorm harmful on regular graphs)
4. Extensibility to Ising variants documented (longitudinal OK, Heisenberg negative, Kitaev negative)
5. Diagnostic/early-stopping system (69% failure prevention)
6. Scaling law h_min = 1.5 + 0.020·N^1.31
7. GNN-QEM non-composability with PEA-ZNE (alternatives, not complements) — design rule for mitigation pipelines
8. S8/S8b: weight-gradient phase detection is qualitative only (cannot extract ν from VQE data)
9. Noise-aware MPNN training fails (V7 5B): shot noise corrupts training targets → noiseless training is necessary
10. Kitaev chain: 3-barrier incompatibility proof (CX budget + initial state + expressibility)
11. DyPP fails (F1): warm-start is already near-optimal for 4-param HVA — no room for adaptive improvement
12. Cross-N warm-start useless at p=1 (2 params): landscape is trivially convex → init irrelevant
13. PauliEvolutionGate gives -11% 2Q-depth: adopted as standard for hardware deployment
14. 6 Hamiltonian candidates systematically evaluated (Table with viability criteria) → only longitudinal viable
15. Cross-topology transfer fails (S2): same architecture, different learned representations per topology
16. Unsupervised phase detection via PCA of θ_opt: PC1 explains 99.96% variance, peaks at h≈1.25
17. N=20 landscape has 2-3 local minima (qualitative change from N≤10) → requires ≥7 restarts

### What is NOT Ours (MUST ALWAYS attribute with \citep{}):
| Technique | Original Authors | Citation Key |
|-----------|-----------------|--------------|
| Warm-start VQE | Mele 2022, Puig 2025 | `mele2022`, `puig2025` |
| Adiabatic warm-start guarantees | Schiffer 2026 | `schiffer2026` |
| GNN parameter prediction paradigm | Miao 2024, Zhang 2025 | `miao2024`, `zhang2025` |
| PEA-ZNE | Kim 2023 | `kim2023` |
| Gate-folding ZNE | Uvarov 2024 | `uvarov2024` |
| GINConv architecture | Xu 2019 | `xu2019` |
| HVA design | Wiersema 2020 | `wiersema2020` |
| Depth truncation theorem | Mele 2026 | `mele2026` |
| Local cost → no BP | Cerezo 2021 | `cerezo2021` |
| DMRG/MPS ground truth | Hauschild 2018 | `hauschild2018` |
| MC-Dropout uncertainty | Gal 2016 | `gal2016` |
| Weight-space phase detection | Hernandes 2025 | `hernandes2025` |
| Affine correction | Wang 2024 | `wang2024` |
| TFIM physics | Dutta 2015, Sachdev 2011 | `dutta2015`, `sachdev2011` |

### Correct Language Patterns:
- ✅ "siguiendo el paradigma propuesto por \citet{miao2024}"
- ✅ "la técnica PEA-ZNE, propuesta por \citet{kim2023}, se integra en el pipeline"
- ✅ "la estrategia de warm-start ---formalizada por \citet{puig2025}--- se aplica aquí"
- ✅ "nuestra contribución es la validación cruzada sobre 4 topologías"
- ❌ "nuestro pipeline elimina el 100% de la optimización" (sounds like we invented it)
- ❌ "ningún trabajo previo demuestra X" (may be false; use "no se ha identificado en la literatura revisada")
- ❌ "se propone por primera vez" (unless truly novel — only items in "What IS Ours" above)

## Canonical References by Claim Type

### Architecture & Design Decisions
| Claim | Primary Reference | Key |
|-------|-------------------|-----|
| "HVA over HEA" | Wiersema 2020 (PRX Quantum) + Tripathi 2026 | `wiersema2020`, `tripathi2026` |
| "p ≤ 2 depth constraint" | Mele 2026 (Nature Physics) | `mele2026` |
| "GINConv is maximally expressive" | Xu 2019 (ICLR) | `xu2019` |
| "GNN > CNN for circuits" | Meng 2025 | `meng2025` |
| "Local cost → no barren plateaus" | Cerezo 2021 (Nat Commun) | `cerezo2021` |
| "Warm-start provides larger gradients" | Puig 2025 (PRX Quantum) | `puig2025` |
| "Adiabatic warm-start convergence" | Schiffer 2026 | `schiffer2026` |
| "Parameter transferability in HVA" | Mele 2022 (PRA) | `mele2022` |
| "COBYLA for noisy/shot-based VQE" | Singh 2025 + Tilly 2022 | `singh2025`, `tilly2022` |
| "PEA-ZNE as primary mitigation" | Kim 2023 (Nature 618) | `kim2023` |
| "GNN for spin system properties" | Slavin 2025 + Huang 2022 (Science) | `slavin2025`, `huang2022` |

### Physics Claims
| Claim | Primary Reference | Key |
|-------|-------------------|-----|
| "TFIM critical point h_c = 1" | Dutta 2015 (Cambridge) | `dutta2015` |
| "Finite-size scaling ∝ N^{-1/ν}" | Dutta 2015 + Sachdev 2011 | `dutta2015`, `sachdev2011` |
| "HVA p=2 expressibility limit at h≈1.25" | Tripathi 2026 (independent) | `tripathi2026` |
| "Heisenberg requires p∝N layers" | Wiersema 2020 | `wiersema2020` |
| "Frustration limits VQA expressibility" | Huang 2026b | `huang2026frustration` |
| "Entanglement exceeds circuit at h_c" | Kumar 2026 | `kumar2026` |
| "N/2 layers for thermodynamic limit" | Sumeet 2025 | `sumeet2025` |
| "Noise broadens critical crossover" | Sharma 2026 | `sharma2026` |

### Comparative Claims
| Claim | Reference for compared work | Key |
|-------|----------------------------|-----|
| "Qracle reduces 64% iterations" | Zhang 2025 | `zhang2025` |
| "NN-VQE uses 20 training points" | Miao 2024 (PRA) | `miao2024` |
| "Flow-VQE achieves 50× acceleration" | Zou 2026 (npj QI) | `zou2026` |
| "Lee: graph autoencoder for VQE" | Lee 2026 | `lee2026` |
| "VQE needs 500-1000 evaluations" | Tilly 2022 (Physics Reports) | `tilly2022` |
| "Frustrated TFIM on trapped-ion" | Teoh 2025 | `teoh2025` |
| "103-site Kagome on IBM Heron" | Ahsan 2025 | `ahsan2025` |
| "6-qubit Wigner on IBM Heron 2" | Kiiamov 2026 | `kiiamov2026` |
| "VQE+ZNE on Ising hardware" | Ma 2025 | `ma2025` |

### Error Mitigation
| Claim | Primary Reference | Key |
|-------|-------------------|-----|
| "PEA learns noise model" | Kim 2023 (Nature 618) | `kim2023` |
| "Inhomogeneous ZNE via layouts" | Uvarov 2024 | `uvarov2024` |
| "NN-enhanced ZNE" | Sun 2025 | `sun2025` |
| "DD on 100 qubits" | Pokharel 2025 | `pokharel2025` |
| "TREX readout correction" | van den Berg 2022 (Nat Phys) | `vandenberg2022` |
| "Affine energy correction" | Wang 2024 | `wang2024` |
| "~18 CX threshold for ZNE" | Own results + Kim 2023 | `kim2023` |

### Tools & Frameworks
| Tool | Reference | Key |
|------|-----------|-----|
| PyTorch Geometric | Fey 2019 | `fey2019` |
| TeNPy (DMRG) | Hauschild 2018 | `hauschild2018` |
| Qiskit | — (no citation needed) | — |
| MC-Dropout | Gal 2016 (ICML) | `gal2016` |
| VQE (original) | Peruzzo 2014 (Nat Commun) | `peruzzo2014` |
| MPNN framework | Gilmer 2017 (ICML) | `gilmer2017` |

## Papers Deliberately Excluded (DO NOT cite)
- **HEA papers** (Kandala 2017) — HEA rejected in our framework
- **ADAPT-VQE** (Grimsley 2019) — not used in pipeline
- **Quantum chemistry UCCSD** — wrong domain, cited only via Tilly 2022
- **GATConv** (Veličković 2018) — tested and rejected
- **Placeholder arXiv IDs** (2404.XXXXX patterns) — unverifiable
- **Pre-2020 VQE papers** (except Peruzzo 2014) — superseded by Tilly 2022
- **QRC papers** — route not taken in final pipeline
- **Kipf GCN 2017** — superseded by GINConv

## Where to Find Full Bibliography
- **Curated list (56 papers):** `documentation/bibliography/bibliography_curated.md`
- **Alternative approaches:** `documentation/bibliography/alternative_bibliography.md`
- **Raw collection:** `documentation/bibliography/bibliography.md`

## Adding New Papers — Checklist
1. Verify arXiv ID exists (not placeholder)
2. Check publication date (prefer 2023+ for current state claims)
3. Classify: ✅ Core / 🔑 Key / 📎 Supporting
4. Add to `bibliography_curated.md` in correct section
5. If citing in thesis: add `\bibitem` AND at least one `\citep{}`/`\citet{}`
6. Run balance check: `grep -oE 'cite[pt]?\{[^}]+\}' ... | sort -u` vs `grep bibitem ...`
7. NEVER claim the cited technique as our own invention
