# HVA (GNN) vs Vector Loading — circuit resources (N=10, chain_1d)

Preparing the TFIM ground state |ψ₀(h)⟩. HVA is a constant-depth ansatz (fidelity-limited); VL loads an MPS approximation of bond dimension χ (fidelity → 1, depth grows with χ). 2q = transpiled CX to basis ['rz', 'sx', 'x', 'cx'].

## h = 1.0

HVA reference: fidelity=0.7830, ΔE/gap=1.7956, 2q=18, depth_2q=18, NISQ-fid≈0.859 (19 params, constant in χ).

| method | χ | fidelity | ΔE/gap | 2q (CX) | depth_2q | NISQ-fid |
|---|---:|---:|---:|---:|---:|---:|
| HVA (GNN) | — | 0.7830 | 1.7956 | 18 | 18 | 0.859 |
| VL approx (Ran) | 2 | 1.0000 | 0.0408 | 27 | 27 | 0.791 |
| VL approx (Ran) | 4 | 0.9983 | 0.0320 | 108 | 45 | 0.397 |
| VL approx (Ran) | 8 | 0.9981 | 0.0331 | 216 | 69 | 0.159 |
| VL approx (Ran) | 16 | 0.9981 | 0.0336 | 432 | 117 | 0.025 |
| VL exact (Schön) (skipped) | 2 | — | — | — | — | — |
| VL exact (Schön) | 4 | 1.0000 | 0.0000 | 136 | 136 | 0.313 |
| VL exact (Schön) | 8 | 1.0000 | 0.0000 | 515 | 510 | 0.013 |
| VL exact (Schön) | 16 | 1.0000 | 0.0000 | 515 | 510 | 0.013 |

## h = 2.5

HVA reference: fidelity=0.9930, ΔE/gap=0.0206, 2q=18, depth_2q=18, NISQ-fid≈0.859 (19 params, constant in χ).

| method | χ | fidelity | ΔE/gap | 2q (CX) | depth_2q | NISQ-fid |
|---|---:|---:|---:|---:|---:|---:|
| HVA (GNN) | — | 0.9930 | 0.0206 | 18 | 18 | 0.859 |
| VL approx (Ran) | 2 | 1.0000 | 0.0000 | 27 | 27 | 0.791 |
| VL approx (Ran) | 4 | 0.9999 | 0.0002 | 81 | 33 | 0.500 |
| VL approx (Ran) | 8 | 1.0000 | 0.0001 | 108 | 42 | 0.397 |
| VL approx (Ran) | 16 | 0.9999 | 0.0003 | 108 | 42 | 0.397 |
| VL exact (Schön) (skipped) | 2 | — | — | — | — | — |
| VL exact (Schön) | 4 | 1.0000 | 0.0000 | 131 | 131 | 0.325 |
| VL exact (Schön) | 8 | 1.0000 | 0.0000 | 361 | 358 | 0.046 |
| VL exact (Schön) | 16 | 1.0000 | 0.0000 | 361 | 358 | 0.046 |

## Verdict

A three-way trade-off, not just cost: (1) *theoretical accuracy* — VL reaches ΔE/gap ≈ 0 while HVA has an ansatz ceiling that fails the 5% criterion near h_c (ΔE/gap ≈ 1.8 at h=1.0) but passes deep in the paramagnetic phase; (2) *circuit cost* — VL's 2q count grows with χ and with the state's entanglement, 1–2 orders above HVA's constant ~18 CX; (3) *hardware executability* — the estimated NISQ fidelity collapses for VL as χ grows (VL exact χ=8 ≈ 1% at h=1.0: theoretically exact but unexecutable), while HVA stays the most executable (~0.86). So VL wins on accuracy but loses on executability; HVA wins on executability but is accuracy-limited near criticality.

Figures: `figures/fidelity_vs_chi.png`, `figures/n2q_vs_chi.png`
