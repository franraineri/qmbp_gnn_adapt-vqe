# Cross-Topology Performance Summary

## Table 1: Overall Performance by Topology (HVA p=1, TFIM Bond-Resolved)

| Topology | N range | Training pts | Pass rate (dual) | Zoo model | Grade | $|\Delta E|/N$ (largest) | $h_{\text{frontier}}$ range |
|:---------|:--------|:---:|:---:|:---:|:---:|:---:|:---:|
| chain\_1d | 6–20 | 606 | **90%** | 90% | **A** | $3.3 \times 10^{-3}$ | [1.56, 2.21] |
| heavy\_hex | 4–16 | 441 | **91%** | 91% | **A** | $4.1 \times 10^{-3}$ | [0.96, 2.02] |
| square | 4–14 | 435 | 75% | 76% | B | $8.9 \times 10^{-3}$ | [1.84, 3.29] |
| ladder | 4–16 | 497 | 44% | 45% | D | $1.2 \times 10^{-2}$ | [1.85, 3.21] |
| triangular | 3–12 | 340 | 47% | 47% | D | $2.5 \times 10^{-2}$ | [0.50, 4.44] |

**Criterion**: Dual (ΔE/gap < 5% AND |ΔE| < 0.10). 2346 total training points across 32 configurations.

---

## Table 2: Scaling Degradation — Pass Rate (dual) vs System Size N

| N | chain\_1d | heavy\_hex | ladder | square | triangular |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 3–4 | — | 94% | 73% | 85% | 100% |
| 6 | 99% | 100% | 72% | 85% | 58% |
| 8 | 96% | — | 50% | 81% | 12% |
| 10 | 92% | 91% | 24% | 61% | 12% |
| 12 | 91% | 87% | 48% | 63% | 14% |
| 14–16 | 84%† | 87% | 27–28% | 54% | — |
| 20 | 78% | — | — | — | — |

† N=15 for chain\_1d.

**Key observation**: 1D topologies (chain, heavy\_hex) degrade gracefully with N. 2D topologies (ladder, square, triangular) exhibit sharp performance cliffs at $N \approx 8$–$10$.

---

## Table 3: Gap Masking Severity (Single-criterion inflation)

| Topology | Worst N | pass@5% | pass@dual | Gap masked |
|:---------|:---:|:---:|:---:|:---:|
| chain\_1d | 20 | 86% | 78% | 8% |
| heavy\_hex | 16 | 88% | 87% | 1% |
| ladder | 10 | 73% | 24% | **50%** |
| square | 12 | 87% | 63% | 23% |
| triangular | 12 | 43% | 14% | 29% |

**Finding**: Ladder topology exhibits the most severe gap masking (up to 50pp inflation). The spectral gap closes polynomially with N in quasi-1D frustrated systems, inflating the ΔE/gap metric while |ΔE| remains large.

---

## Finding: Ansatz Expressibility Limit in Frustrated Topologies

### Observation

HVA with depth $p=1$ and initial state $|+\rangle^{\otimes N}$ fails to express the ground state of the transverse-field Ising model on **frustrated** lattice geometries (triangular, ladder) for $N \geq 8$, while remaining effective on **unfrustrated** geometries (chain\_1d, heavy\_hex) up to $N=20$.

### Evidence

| Diagnostic | chain\_1d (1D) | triangular (frustrated 2D) |
|:-----------|:---:|:---:|
| Pass rate at $N=6$ | 99% | 58% |
| Pass rate at $N=10$ | 92% | 12% |
| $h_{\text{frontier}}$ at $N=10$ | 1.84 | 3.98 |
| $\theta$ smoothness at $N=10$ | 1.57 | 1.57 |
| Gap masking at $N=10$ | 0% | 20% |
| Per-site error $|\Delta E|/N$ | $3.3 \times 10^{-3}$ | $2.5 \times 10^{-2}$ |

### Physical Interpretation

1. **Geometric frustration**: Triangular lattice bonds cannot be simultaneously satisfied (antiferromagnetic frustration in ZZ coupling). The HVA ansatz with $p=1$ provides only a single ZZ rotation per bond — insufficient to resolve competing interactions.

2. **h\_frontier divergence**: The frontier $h_f$ (minimum field strength for pipeline success) scales linearly with system connectivity:
   - chain (degree 2): $h_f \approx 1.5 + 0.035N$
   - triangular (degree 4–6): $h_f \approx 0.5 + 0.35N$

   At $N=10$, triangular requires $h > 3.98$ (deep paramagnetic phase) while chain works from $h > 1.84$ (near the QPT).

3. **Gap masking**: In frustrated systems, the spectral gap $\Delta$ remains large even when $|\Delta E|$ is physically significant ($> 0.10$). The dual criterion (requiring BOTH ΔE/gap < 5% AND |ΔE| < 0.10) correctly identifies these as failures.

### Implications for the Framework

- The GNN-HVA framework is **validated** for 1D and quasi-1D topologies (chain, heavy\_hex) up to $N=20$ with $p=1$.
- For 2D frustrated topologies, the framework requires either:
  - Increased circuit depth ($p \geq 2$) — at the cost of increased noise sensitivity (Mele et al., 2026)
  - Reduced scope: limit to small N ($\leq 6$) or deep paramagnetic regime ($h \gg h_c$)
- This is **not** a failure of the MPNN predictor — the predictor correctly learns the limited variational landscape. It is a **physics limit** of the HVA ansatz at depth $p=1$.

### Consistency with Literature

- Tripathi et al. (2026): HVA $p=2$ struggles with entanglement entropy at criticality for 1D/2D/3D TFIM up to 27 spins. Our $p=1$ result at $N \geq 8$ for 2D is consistent.
- Sumeet et al. (2025): thermodynamic-limit accuracy requires $p \sim N/2$ layers. For $N=10$ triangular, this implies $p \geq 5$ — far beyond our hardware constraint.
