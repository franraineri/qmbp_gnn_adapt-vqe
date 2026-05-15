---
inclusion: fileMatch
fileMatchPattern: "**/*.tex"
---

# Thesis Writing — LaTeX Conventions

## Citation Style
- Use `\cite{AuthorYear}` for parenthetical, `\citet{AuthorYear}` for textual
- BibTeX keys: `AuthorYear` format (e.g., `Mele2026`, `Tripathi2026`)
- Always cite the specific result, not just the paper
- For multiple citations: `\cite{Author1Year, Author2Year}`

## Results References

| Data | Location | Thesis Section |
|------|----------|----------------|
| N=6 results (3 seeds × 3 h_test) | Table 4.2 | Results chapter |
| N=10 results (3 seeds × 2 h_test) | Table 4.3 | Results chapter |
| Noisy simulation (3-mode comparison) | Section 4.5 | Hardware validation |
| Gradient analysis peaks | Section 4.4 | MPNN analysis |
| Source JSON data | `scripts/notebook_results/` | — |

## Narrative Framing (literature-backed)
- h=1.25 ceiling → "physics limit of HVA p=2" (cite Tripathi et al. 2026)
- Hardware noise broadening → "expected behavior, not failure" (cite Sharma 2026)
- Pipeline value → "methodology demonstration, not quantum advantage" (cite Martin et al. 2026)
- GNN choice → "36% improvement over CNN for circuit prediction" (cite Meng et al. 2025)
- Warm-start effectiveness → "provably larger gradients" (cite Puig et al. 2025)
- Three-way synergy → noise resilience + trainability + expressibility (cite Mele 2026, Cerezo 2021, Wiersema 2020)

## Numerical Precision
- ΔE/gap: 2 decimal places (e.g., 3.77%)
- Energies: 4 significant figures (e.g., -7.2734)
- Fidelities: 3 decimal places (e.g., 0.997)
- MSE: scientific notation (e.g., 3.2×10⁻³)
- Always include ± std when reporting multi-seed results
- Shot noise: σ ≈ 1/√shots (report to 1 decimal in scientific notation)

## Figure Conventions
- Phase diagram plots: h on x-axis, observable on y-axis
- Always mark h_c ≈ 1.0 with vertical dashed line
- Use consistent colors: blue=paramagnetic, red=ferromagnetic, gray=critical region
- Error bars from multi-seed runs (3 seeds minimum)

## Language
- Write in the language matching the thesis document (Spanish for tesis-v2.1.tex)
- Technical terms in English when no standard Spanish translation exists
- Acronyms: define on first use, then use freely (VQE, MPNN, ZNE, HVA, TFIM)

## Key Claims to Support with Data
1. "HVA p=2 resolves the physics" → ΔE/gap < 5% at h≥1.25 (Table 4.2)
2. "MPNN eliminates quantum optimization" → 0 ADAPT iterations needed (warm-start ideal)
3. "Pipeline scales to N=10" → Table 4.3 with h=1.4 passing
4. "ZNE reduces errors" → noisy simulation 3-mode comparison (Section 4.5)
5. "Weight gradients detect phase transitions" → peaks at h≈1.0-1.2 (Section 4.4)
