---
inclusion: fileMatch
fileMatchPattern: "**/*.tex"
---

# Thesis Writing — LaTeX Conventions & Quality Rules

## CRITICAL: Every Claim Must Be Grounded

**Rule #1 (ALWAYS ENFORCE):** Every assertion, finding, decision, or claim in the thesis MUST be grounded by EITHER:
- A `\citep{}` or `\citet{}` reference to published literature, OR
- A `Tabla~\ref{}` / `Ec.~\ref{}` reference to own results in the Results chapter, OR
- An explicit "experiment X, N seeds, condition Y" inline reference to own data.

**No ungrounded claims.** If a statement cannot be referenced, either add the reference or rewrite as a hypothesis/proposal (using subjunctive mood).

**Rule #2 (ALWAYS ENFORCE): No False Novelty Claims.**
- If a technique, method, or finding was previously published by others, ALWAYS attribute it: "siguiendo el paradigma de X propuesto por \citet{Author}"
- Our contribution is the INTEGRATION and SYSTEMATIC VALIDATION, not the individual techniques
- Use language like "se integra", "se valida", "se extiende", NOT "se propone" or "se descubre" for pre-existing techniques
- Specific techniques that are NOT ours (must always attribute): warm-start (Mele2022, Puig2025), GNN parameter prediction (Miao2024, Zhang2025), PEA-ZNE (Kim2023), GINConv (Xu2019), HVA design (Wiersema2020), DMRG (Hauschild2018), MC-Dropout (Gal2016)
- What IS our contribution: (1) integration into unified pipeline, (2) systematic validation across 5 topologies + N=6-80, (3) cross-N generalization finding (BatchNorm issue), (4) extensibility to Ising variants documented, (5) diagnostic/early-stopping system

### Grounding Checklist (apply to every paragraph)
- [ ] Strong verbs (demuestra, confirma, establece, garantiza) → MUST have \citep{} or \ref{}
- [ ] Numerical claims (reduces 50×, achieves 95%) → MUST cite table/figure or source
- [ ] Design decisions (chose X over Y) → MUST cite theoretical justification
- [ ] Negative results (X fails, X is impossible) → MUST cite independent confirmation or own systematic evidence
- [ ] Comparison claims (better than X by Y%) → MUST cite both own data AND the compared work

## Document Structure (v3.0 — Director Approved)

| Chapter | Content | Rule |
|---------|---------|------|
| 1. Introducción | Problem, motivation, pipeline overview | NO numerical results |
| 2. Marco Teórico | Physics + algorithms + GNN + mitigation | NO results, NO methodology |
| 3. Desarrollo del Trabajo | Objectives, methodology, implementation | Describes HOW, not WHAT was found |
| 4. Descripción Detallada e Identificación de Requisitos | Context, prior work, functional/non-functional requirements | Problem detail + requirements |
| 5. Evaluación | ALL results + comparisons + discussion + ventaja + "Código desarrollado" | ALL numbers here + applicability |
| 6. Conclusiones y Trabajo Futuro | Summary + validity + contributions vs objectives + future lines | References back to Ch.5 tables |

### Structural Rules
- **No results in Chapter 2.** State of the art presents only published results from others.
- **No methodology in Chapter 2.** Methodology belongs in Chapter 3.
- **Chapter 4 establishes WHAT must be solved** — requirements, constraints, prior iterations.
- **All own numerical data in Chapter 5.** Any number ΔE/gap, fidelity, timing → Chapter 5 only.
- **Chapter 5 must include "Ventaja de la Solución y Aplicabilidad"** section.
- **"Código desarrollado" section** at end of Chapter 5 with GitHub URL.
- **Chapter 6 structure:** (1) Summary of problem+approach+validity, (2) Contributions list with table refs, (3) Cumplimiento de objetivos, (4) Líneas de trabajo futuro (with application fields).
- **Conclusions reference tables.** Each conclusion must cite the specific table/figure from Ch.5.

## Citation Style

- Use `\citep{AuthorYear}` for parenthetical: "...confirmed independently \citep{tripathi2026}."
- Use `\citet{AuthorYear}` for textual: "\citet{mele2026} demonstrated that..."
- BibTeX keys: `authorYear` format lowercase (e.g., `mele2026`, `tripathi2026`)
- Always cite the **specific result**, not just the paper
- For multiple citations: `\citep{author1Year, author2Year}`
- Every \bibitem must be cited at least once in the text
- Every citation key must have a corresponding \bibitem

## Reference Quality Hierarchy (prefer top → bottom)

1. **Nature / Science / Nature Physics** (Kim2023, Mele2026) — maximum authority
2. **Physical Review journals** (PRX Quantum, PRA, PRApplied) — (Puig2025, Mele2022, Miao2024)
3. **npj Quantum Information** (Zou2026) — Nature group
4. **Peer-reviewed journals** (Skogh2023 Electronic Structure, Tilly2022 Physics Reports)
5. **ICLR/ICML proceedings** (Xu2019, Gilmer2017, Gal2016) — top-tier ML venues
6. **arXiv preprints** (Tripathi2026, Zhang2025, etc.) — acceptable if 2024+, flag if sole evidence

### Key Authoritative References by Topic
| Topic | Primary Reference | Backup |
|-------|-------------------|--------|
| Depth truncation | Mele et al. 2026 (Nature Physics) | — |
| Barren plateaus | McClean 2018 + Cerezo 2021 (Nat Commun) | — |
| HVA design | Wiersema 2020 (PRX Quantum) | Tripathi 2026 |
| Warm-start theory | Puig 2025 (PRX Quantum) | Mele 2022 (PRA) |
| VQE review | Tilly 2022 (Physics Reports) | Peruzzo 2014 |
| GNN expressivity | Xu 2019 (ICLR) | Meng 2025 |
| TFIM physics | Dutta 2015 (Cambridge UP) | Sachdev 2011 |
| PEA-ZNE | Kim 2023 (Nature 618) | Uvarov 2024 |
| GNN for spins | Huang 2022 (Science) | Kochkov 2021 |
| MC-Dropout | Gal & Ghahramani 2016 (ICML) | — |
| Hardware validation | Kim 2023, Sharma 2026, Kiiamov 2026 | Ma 2025 |
| Quantum advantage boundary | Martin 2026 | — |

## Results References (definitive data locations)

| Data | Source File | Thesis Table |
|------|-------------|-------------|
| Cross-topology N=10 (chain/ladder/tri/heavy-hex) | `09_thesis_tables.md` Tables 5.1-5.4 | Table 4.1 |
| Scaling N=6 to N=80 | `09_thesis_tables.md` Table 5.23 | Table 4.2 |
| N=6 p=2 baseline (3 seeds × 3 h) | `09_thesis_tables.md` | Table 4.3 |
| N=10 p=2 baseline (3 seeds × 2 h) | `09_thesis_tables.md` | Table 4.4 |
| PEA-ZNE cross-topology (18/18 wins) | `binnacle-gate-folding-zne.md` | Table 4.5 |
| CX budget rule | `09_thesis_tables.md` Table 5.5 | Table 4.6 |
| TFIM + longitudinal (E4b) | `09_thesis_tables.md` Table 5.22 | Table 4.7 |
| Heisenberg negative (V9) | `09_thesis_tables.md` Table 5.14 | Table 4.8 |
| Root cause analysis (174 runs) | `09_thesis_tables.md` Table 5.9 | Table 4.9 |
| Data efficiency (k=5-17) | `09_thesis_tables.md` Table 5.17 | Table 4.10 |
| Literature comparison | Own analysis | Table 4.11 |
| Cross-N zero-shot (30/30 PASS) | `binnacle-cross-n-zero-shot.md` | Table 4.12 |
| MC-Dropout (r=0.82) | `09_thesis_tables.md` Table 5.20 | Table 4.13 |
| Source JSON for all | `results/thesis/` | — |
| GitHub repository | `https://github.com/franraineri/qmbp_gnn_adapt-vqe` | Section 4.7 |

## Narrative Framing (each with mandatory citation)

- h=1.25 ceiling → "physics limit of HVA p=2" (MUST cite Tripathi2026 + own Table)
- Hardware noise broadening → "expected behavior" (MUST cite Sharma2026)
- Pipeline value → "methodology demonstration, not quantum advantage" (MUST cite Martin2026)
- GNN choice → "maximally expressive MPNN" (MUST cite Xu2019 + Meng2025)
- Warm-start effectiveness → "provably larger gradients" (MUST cite Puig2025)
- Triple synergy → depth + cost + ansatz (MUST cite Mele2026 + Cerezo2021 + Wiersema2020)
- Heisenberg failure → "expressibility limit, requires p∝N" (MUST cite Wiersema2020 + own Table)
- Cross-N BatchNorm issue → "regular graph destroys variance" (MUST cite Xu2019 + own experiment)
- PEA > GF-ZNE → "4.6× gain, t=46.3" (MUST cite Kim2023 + own Table)

## Numerical Precision

- ΔE/gap: 2 decimal places (e.g., 3.77%)
- Energies: 4 significant figures (e.g., -7.2734)
- Fidelities: 3 decimal places (e.g., 0.997)
- MSE: scientific notation (e.g., 3.2×10⁻³)
- Always include ± std when reporting multi-seed results
- Statistics: report t-statistic and p-value for comparative claims
- Scaling laws: report R² value

## Figure Conventions

- Phase diagram plots: h on x-axis, observable on y-axis
- Always mark h_c ≈ 1.0 with vertical dashed line
- Use consistent colors: blue=paramagnetic, red=ferromagnetic, gray=critical region
- Error bars from multi-seed runs (3 seeds minimum)
- All figures generated via `make figures-thesis` (PDF 300dpi)

## Language

- Write in Spanish (matching thesis document language)
- Technical terms in English when no standard Spanish translation exists
- Acronyms: define on first use, then use freely (VQE, MPNN, ZNE, HVA, TFIM, PEA)
- Use third-person impersonal ("se demuestra", "se observa") for results
- Use first-person plural ("proponemos", "nuestro pipeline") sparingly, for own contributions only

## Common Errors to Avoid

1. **Putting results in Chapter 2** — Estado del Arte must only contain published literature results
2. **Ungrounded strong claims** — "X is the only/best/unique" without citation
3. **Missing table references in conclusions** — every conclusion must point to evidence
4. **Citing the paper without the specific result** — say WHAT was demonstrated
5. **Using "preliminar"** — the final thesis contains only definitive results
6. **Orphaned bibliography entries** — every \bibitem must be \cited somewhere
7. **Methodology in wrong chapter** — development/methodology goes in Chapter 3 only
8. **Unreferenced numerical claims** — any number (50×, 95%, 430+ runs) needs source

## Bibliography Management

- **Quick reference for which paper to cite:** Use `#bibliography-guide` steering (manual inclusion)
- **Full curated list (56 papers):** `documentation/bibliography/bibliography_curated.md`
- **Alternative approaches:** `documentation/bibliography/alternative_bibliography.md`
- **Before citing a new paper:** verify arXiv ID, check date (2023+), classify (Core/Key/Supporting), add to curated list

## Appendices (Mandatory)

### Anexo A: Resultados y Mediciones Detalladas
- Contains extended tables NOT shown in Chapter 5 (full variant lists, per-seed breakdowns)
- Transpilation audit tables (CX counts, SWAP counts, depth, estimated fidelity)
- Heisenberg depth-scaling data (p=1 to p=6)
- Hardware pre-deployment parameter optimization
- Reference from Chapter 5: "véase Anexo A para tablas detalladas"
- Data source: `documentation/analysis/09_thesis_tables.md` (Tables 5.1-5.23)

### Anexo B: Artículo de Investigación (Obligatorio)
- 6-8 pages, English or Spanish (currently English)
- Follows provided template structure: Abstract, Introduction, Method, Results, Discussion, Conclusions
- Self-contained summary of the full thesis work
- Must include: problem statement, methodology overview, key numerical results, comparison with literature, conclusions
- All claims must be referenced (same bibliography as main thesis)
- GitHub URL included in conclusions
- Target venue style: arXiv/conference paper (concise, quantitative)

## Current Document Status (tesis-v3.0.tex — Updated 2026-06-09)

### Stats
- Lines: 1790 | Chapters: 10 | Tables: 31 | Table refs: 23
- Citations: 49 unique keys | Bibitems: 49 (balanced)
- Environments: 88/88 (balanced)
- Verification: 22 findings, 95% corroboration rate (21 CORROBORATED, 1 QUALIFIED, 0 CONTRADICTED)

### Pending Actions for Final Submission
1. **Insert figures**: Run `make figures-thesis` → insert `\includegraphics` in appropriate sections
2. **Hardware execution (OE6)**: Run on IBM Torino when credentials available → add results to Ch.5
3. **Compile LaTeX**: Verify with `pdflatex tesis-v3.0.tex` (requires `estilo_unir-1.sty`)

### Known Verification Issues (RESOLVED 2026-06-09)
- **F1**: Fixed — now uses valid-regime filter (ΔE/gap<20%), CORROBORATED (STRONG)
- **F6**: Reformulated from "topology ranking" to "topology-agnostic" — CORROBORATED (STRONG)
- **F10**: Updated to 84% (41/49) — CORROBORATED (STRONG)
- **F11**: Fixed path lookup (summary.n_zne_records) — CORROBORATED (STRONG)
- **F14**: Fixed path lookup (circuit_selection.spearman_rho) — CORROBORATED (STRONG)
- **F16**: QUALIFIED (MODERATE) — qualitative result from binnacle S2, valid as-is

### Experiments NOT in Thesis (documented but excluded for brevity)
- Per-observable site-selective ZNE (V3 noisy variants) — novel but superseded by PEA
- Non-linear ZNE decision rule (quadratic only when R²_lin < 0.5) — implementation detail
- K-means rejected for phase detection — PCA/derivative sufficient
- seed_transpiler CES diversity failure — infrastructure fix, not thesis content
- GNN-QEM ablation with E_noisy (correction is 99.96% linear) — detail in GNN-QEM binnacle
- SKQD alternatives analysis (5 methods evaluated, all rejected) — in doc 14

### Key Files for Context
| Purpose | File |
|---------|------|
| Thesis document | `tesis-v3.0.tex` |
| Definitive results data | `documentation/analysis/09_thesis_tables.md` |
| Project status (comprehensive) | `.kiro/steering/project-status.md` |
| Bibliography curated (56 papers) | `documentation/bibliography/bibliography_curated.md` |
| Verification plan | `documentation/analysis/21_thesis_compilation_verification_plan.md` |
| Findings validator (22 findings) | `project_health/analysis/thesis_findings_validator.py` |
| Tables compiler (10 tables) | `project_health/analysis/thesis_tables_compiler.py` |
| Figures generator (10 figs) | `project_health/analysis/thesis_figures.py` |
