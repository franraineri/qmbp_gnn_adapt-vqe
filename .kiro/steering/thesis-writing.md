---
inclusion: fileMatch
fileMatchPattern: "**/*.tex"
---

# Thesis Writing — LaTeX Conventions & Quality Rules

## Thesis General Objective (CANONICAL)

"Demostrar que la integración de predicción GNN con HVA shallow en un pipeline unificado permite reducir el costo cuántico de la clasificación de fases entre 29× y 500×, manteniendo ΔE/gap < 5% dentro del régimen operativo válido, y documentar formalmente los límites de dicho régimen."

Every chapter, section, and claim must serve this objective.

## Scope: Noiseless Ideal Simulation ONLY

- ALL results are from ideal simulation (StatevectorEstimator, no noise)
- Noise/hardware is ONLY mentioned as context (Ch1 motivation, Ch2 state of art)
- NO own noisy results, NO PEA-ZNE results, NO GNN-QEM results
- Hardware deployment = future work line in Ch6
- p is NOT restricted to ≤2 — we use p=1-4 freely
- The p≤2 constraint is mentioned ONLY in Ch2 as a noise-related consideration from literature

## CRITICAL: Every Claim Must Be Grounded

**Rule #1 (ALWAYS ENFORCE):** Every assertion MUST be grounded by EITHER:
- A `\citep{}` or `\citet{}` reference to published literature, OR
- A `Tabla~\ref{}` / `Ec.~\ref{}` reference to own results in Chapter 5, OR
- An explicit "experiment X, N seeds, condition Y" inline reference to own data.

**Rule #2 (ALWAYS ENFORCE): No False Novelty Claims.**
- Our contribution is the INTEGRATION and SYSTEMATIC VALIDATION, not the individual techniques
- Use "se integra", "se valida", "se extiende", NOT "se propone" or "se descubre"
- Techniques that are NOT ours: warm-start (Mele2022, Puig2025), GNN prediction (Miao2024, Zhang2025), GINConv (Xu2019), HVA (Wiersema2020), DMRG (Hauschild2018)
- What IS ours: (1) integration into unified pipeline, (2) systematic validation across 5 topologies + p=1-4 + N=4-20, (3) cross-N generalization finding (BatchNorm), (4) extensibility documented, (5) formal limits documentation

## Document Structure (v3.0 Refactored — 2026-07-15)

| Chapter | Content | Rule |
|---------|---------|------|
| 1. Introducción | Problem, motivation, pipeline overview (3 phases) | NO results. Noise as context only |
| 2. Marco Teórico | Physics + VQE + HVA + GNN + mitigation as literature | NO own results. Mitigation = state of art |
| 3. Desarrollo del Trabajo | Objectives (OE1-OE6), methodology (3 phases), implementation | Describes HOW |
| 4. Descripción Detallada | Context, prior work, requirements | Problem detail + constraints |
| 5. Evaluación | ALL noiseless results + extensibility + comparisons + limits + code | ALL numbers here |
| 6. Conclusiones | Summary + contributions + objectives + future (hardware as line 1) | Refs to Ch.5 tables |

### Pipeline = 3 Phases (NOT 4)
1. **Phase 1**: Ground Truth (ExactDiag / DMRG)
2. **Phase 2**: VQE Warm-Start (descending sweep, HVA p=1-4)
3. **Phase 3**: MPNN Predictor + Deployment evaluation (ideal simulation)

### Key Structural Rules
- All own numerical data in Chapter 5 only
- Chapter 5 includes "Ventaja de la Solución" + "Código Desarrollado" sections
- Conclusions reference specific tables from Ch.5
- Extensibility section: ALL models (tfim, tfim_longitudinal, heisenberg, heisenberg_transverse, kitaev) — concise summary with key findings, not exhaustive data dumps
- Topologies: cadena 1D, escalera, triangular, cuadrada, heavy-hex (NOT kagomé)

## Metrics (Canonical Names — 2026-07-16)

| Metric | Definition | Use |
|--------|-----------|-----|
| **ΔE/gap** | \|E_pred - E_exact\| / (E₁ - E₀) | PRIMARY success criterion (< 5%) |
| **PassRate** | Fraction of h-points with ΔE/gap < 5% | Pipeline effectiveness. In tables: "PassRate" column |
| **Speedup** | VQE iterations (random) / evaluations (pipeline) | Acceleration factor (29-500×) |
| **Fidelidad** | \|⟨ψ(θ)\|ψ_exact⟩\|² | State quality (≥ 0.93 filter) |
| **θ_smooth** | max‖θ*(hᵢ) - θ*(hᵢ₋₁)‖∞ | Chain-break detector |

**DEPRECATED (do not use in thesis):** "Deploy", "deploy pass rate", "checklist n/4", "gen_gap", "labels accuracy".
The term "Deploy %" was replaced by "PassRate" on 2026-07-16.

## Citation Style

- `\citep{authorYear}` for parenthetical, `\citet{authorYear}` for textual
- BibTeX keys: `authorYear` lowercase (e.g., `mele2026`, `tripathi2026`)
- Always cite the specific result, not just the paper
- Every \bibitem must be \cited; every \cite must have a \bibitem

## Key References by Topic

| Topic | Primary | Backup |
|-------|---------|--------|
| Depth truncation / noise | Mele 2026 (Nature Physics) | — |
| Barren plateaus | McClean 2018 + Cerezo 2021 | — |
| HVA design | Wiersema 2020 (PRX Quantum) | Tripathi 2026 |
| Warm-start | Puig 2025 (PRX Quantum) | Mele 2022 (PRA) |
| VQE review | Tilly 2022 (Physics Reports) | Peruzzo 2014 |
| GNN expressivity | Xu 2019 (ICLR) | Meng 2025 |
| TFIM physics | Dutta 2015 (Cambridge UP) | Sachdev 2011 |
| GNN for spins | Huang 2022 (Science) | Kochkov 2021 |
| Hardware validation (context) | Kim 2023, Sharma 2026 | Ma 2025 |
| Quantum advantage boundary | Martin 2026 | — |

## Narrative Framing

- h≈1.3 ceiling → "expressibility boundary of HVA at moderate depth" (cite Tripathi2026 + own Table)
- Pipeline value → "methodology for phase classification, not quantum advantage" (cite Martin2026)
- GNN choice → "maximally expressive MPNN" (cite Xu2019 + Meng2025)
- Warm-start → "provably larger gradients, smooth landscape" (cite Puig2025 + Mele2022)
- Heisenberg failure → "expressibility limit confirmed p=1-4, requires p∝N" (cite Wiersema2020 + own Table)
- Topology ranking → "connectivity determines difficulty" (cite own Table + noiseless_v2_analysis)
- Bond-resolved → "necessity boundary of MPNN" — explained first (what/why), then result (4414×, cross-N fails). Ref via §\ref{subsec:bond_resolved}
- Noise context only → "motivates shallow circuits" (cite Mele2026) — NO own noise results

## Numerical Precision

- ΔE/gap: 2 decimal places (e.g., 3.77%)
- Fidelities: 3 decimal places (e.g., 0.997)
- Speedup: integer or 1 decimal (e.g., 44×, 492×)
- Always ± std for multi-seed results
- Frontier fits: report R² value and slope

## Language

- Spanish for main text
- Technical terms in English when no standard translation exists
- Third-person impersonal for results ("se demuestra", "se observa")
- First-person plural sparingly for own contributions ("nuestro pipeline")

## Common Errors to Avoid

1. Putting results in Chapter 2
2. Ungrounded strong claims without citation
3. Missing table references in conclusions
4. Using "preliminar" — thesis contains only definitive results
5. Mentioning own noisy/hardware results (REMOVED from thesis)
6. Using "p ≤ 2" as a constraint of this work (it's only context)
7. Saying "kagomé" as a topology we tested (we didn't — it's future work)
8. Exhaustive data dumps in extensibility — show concise findings, not all runs
9. Re-explaining bond-resolved, ν/S8, or Hamiltonianos Candidatos in multiple places — each has ONE canonical location
10. Introducing PCA/PC1 without explaining what it is and what the dataset is
11. Mentioning Schwinger lattice or TFIM boundary fields (removed from thesis)

## Results Data Sources

| Data | Source |
|------|--------|
| Cross-topology N=10 p=1-4 | `internal/documentation/analysis/noiseless_v2_analysis.md` (Sections 1-4) |
| Scaling N=4-20 | `noiseless_v2_analysis.md` (Sections 7, 9, 11) |
| TFIM longitudinal multi-topo | `noiseless_v2_analysis.md` (Section 2) |
| Heisenberg exhaustive (26 runs) | `noiseless_v2_analysis.md` (Section 3) |
| Heisenberg transverse (20 runs) | `noiseless_v2_analysis.md` (Section 7.3) |
| Expressibility p=5 study | `noiseless_v2_analysis.md` (Section 5) |
| Physics loss experiment | `noiseless_v2_analysis.md` (Section 10) |
| N=6/N=10 baseline | `internal/documentation/analysis/09_thesis_tables.md` |
| Cross-N zero-shot | `binnacle-cross-n-zero-shot.md` |
| Cross-N UnifiedMPNN + large-N | `internal/documentation/analysis/accelerated_cross_n_coverage.md` |
| GitHub repository | `https://github.com/franraineri/qmbp_gnn_adapt-vqe` |

## Current Document Status (internal/tesis-v3.0.tex — Updated 2026-07-21)

### Stats
- Lines: ~1750 | Chapters: 10 | Environments: balanced
- Scope: Noiseless ideal simulation only
- Models: tfim, tfim_longitudinal, heisenberg, heisenberg_transverse, kitaev (neg. results)
- Topologies: chain_1d, ladder, triangular, square, heavy_hex
- N range: 4-20 (primary), 40-200 (scaling with p=1, secondary)
- p range: 1-4

### Structural Decisions (2026-07-16)
- **Bond-Resolved HVA** (§5): Has \label{subsec:bond_resolved}. Explains WHAT it is (per-bond params vs shared), WHY tested (MPNN necessity boundary), THEN result.
- **Extensibilidad section** (§5): Ends with "Resumen de Viabilidad por Modelo" subsection (table with 4 models: TFIM-long, frustrado, Heisenberg, Kitaev). Schwinger/boundary fields REMOVED.
- **"Aproximaciones Alternativas"** (§5): Contains ONLY S8/S8b (ν extraction) and DyPP (F1). "Hamiltonianos Candidatos" subsection was moved to Extensibilidad.
- **PCA/PC1** (§ Detección de Fases): Fully introduced — what PCA is, what PC1 means, why 99.96% → 1D manifold.
- **Discusión "¿Por qué Funciona?"** (§5): Now 5 properties (was 4):
  1. Smooth landscape (+ θ_smooth values 0.01-0.05, monotonicity)
  2. Physics restricts space + minimal output_dim (2 numbers for p=1, 99K→2 scalars)
  3. Graph structure informs prediction (GINConv, Slavin2025)
  4. Warm-start + multi-seed data multiplication (9h×3seeds=27pts)
  5. Moderate depth compatible with hardware (Mele2026)

### Pending Work
- Review Anexo A for deprecated noise content
- Compile LaTeX: `pdflatex internal/tesis-v3.0.tex` (requires `estilo_unir-1.sty`)
- Verify all \ref targets resolve (no `??` in PDF)
