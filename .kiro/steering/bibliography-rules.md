---
inclusion: fileMatch
fileMatchPattern: "documentation/bibliography*"
---

# Bibliography Rules — Citation Management

## Format

APA 7th edition. All entries in `documentation/bibliography.md` and `documentation/bibliography_curated.md`.

## Structure

Bibliography is organized by topic section:
1. Foundations of Many-Body Physics & Quantum Spin Liquids
2. Computational Complexity & The Sign Problem
3. Quantum Computing & NISQ Limitations
4. Variational Quantum Algorithms (VQE, QAOA, ADAPT)
5. Ansatz Design (HVA, HEA, UCCSD)
6. Barren Plateaus & Trainability
7. Error Mitigation (ZNE, PEC, DD, TREX)
8. Graph Neural Networks for Quantum Systems
9. Classical Methods (DMRG, Monte Carlo, Tensor Networks)
10. Hardware Experiments & Benchmarks

## Adding New References

1. Place in the correct topic section (alphabetical within section)
2. Use full APA 7th format: `Author, A. B., & Author, C. D. (Year). Title. *Journal*, *Volume*(Issue), Pages. https://doi.org/...`
3. For arXiv preprints: `Author, A. B. (Year). Title. *arXiv preprint arXiv:XXXX.XXXXX*.`
4. Include DOI or arXiv link — never leave a reference without a URL
5. If the paper informs a design decision, also add a note to `.kiro/knowledge/literature-synthesis.md`

## Alternative Bibliography

`documentation/alternative_bibliography.md` contains techniques we evaluated but did NOT adopt. When rejecting a technique, document:
- What it is (1 sentence)
- Why we considered it
- Why we rejected it (with reference to our constraints)
- Under what conditions it might be reconsidered

## Cross-References

When citing in code comments or documentation:
- Use short form: `(Author et al., Year)` or `Author (Year)`
- In pre-commit hooks and SKILL.md: `(Mele et al. 2026)` style (no comma before year)
- Always include the arXiv ID for papers not yet published in journals

## Key Papers (always cite)

| Short form | Full reference |
|-----------|---------------|
| Mele et al. 2026 | Mele, A. A., et al. Nature Physics (2026) — noise depth truncation |
| Cerezo et al. 2021 | Cerezo, M., et al. Nature Communications 12, 1791 — barren plateaus |
| Xu et al. 2019 | Xu, K., et al. ICLR 2019 — GIN/WL-test equivalence |
| Wiersema et al. 2020 | Wiersema, R., et al. PRX Quantum 1, 020319 — HVA expressibility |
| Tripathi et al. 2026 | Tripathi, V., et al. arXiv:2604.20961 — HVA benchmarks |
| Sharma 2026 | Sharma, K. arXiv:2601.17515 — hardware noise broadening |
| Uvarov et al. 2024 | Uvarov, A., et al. arXiv:2307.11156 — inhomogeneous ZNE |
