# Project Status (Auto-Generated)

**Last updated**: 2026-08-28 20:35
**Total runs**: 525 | Pass: 149 | Fail: 376 | Rate: 28%
**Total compute**: 347.3 hours
**Models**: heisenberg, heisenberg_transverse, kitaev, tfim, tfim_bond_resolved, tfim_frustrated, tfim_longitudinal, xy
**Topologies**: chain_1d, heavy_hex, kagome, ladder, square, triangular
**N values**: [4, 6, 8, 10, 16, 20, 100]

## Coverage Matrix (latest quality per config)

| Model | chain_1d |
|---|---|
| tfim | A (N=6) |
| tfim_bond_resolved | A (N=6) |

## Suggested Next Experiments

- NO DATA: tfim chain_1d N=10 — never tested
- NO DATA: tfim chain_1d N=16 — never tested
- NO DATA: tfim chain_1d N=20 — never tested
- NO DATA: tfim heavy_hex N=10 — never tested
- NO DATA: tfim heavy_hex N=16 — never tested
- NO DATA: tfim heavy_hex N=20 — never tested
- NO DATA: tfim_longitudinal chain_1d N=10 — never tested
- NO DATA: tfim_longitudinal chain_1d N=16 — never tested

## Large-N Extrapolation (Zero-Shot MPNN)

| Topology | N | Pts | ΔE/gap | |ΔE|/N | Grade |
|----------|---|-----|--------|--------|-------|
| chain_1d | 10 | 41 | 74.091 | 6.94e-02 | F |
| chain_1d | 12 | 2 | 0.050 | 1.42e-02 | D |
| chain_1d | 16 | 10 | 0.058 | 2.09e-02 | C |
| chain_1d | 20 | 36 | 2.103 | 5.93e-02 | F |
| chain_1d | 30 | 28 | 0.035 | 7.10e-03 | B |
| chain_1d | 40 | 24 | 0.041 | 5.90e-03 | B |
| chain_1d | 60 | 22 | 0.072 | 7.45e-03 | C |
| chain_1d | 80 | 8 | 0.166 | 8.85e-03 | D |
| chain_1d | 100 | 19 | 0.137 | 7.98e-03 | D |
| chain_1d | 150 | 3 | 0.784 | 3.58e-02 | F |
| chain_1d | 200 | 3 | 1.047 | 3.59e-02 | F |
| heavy_hex | 8 | 47 | 0.030 | 2.34e-02 | B |
| heavy_hex | 10 | 57 | 0.014 | 7.88e-03 | A |
| heavy_hex | 12 | 25 | 0.125 | 5.07e-02 | F |
| heavy_hex | 14 | 54 | 1.999 | 5.82e-01 | F |
| heavy_hex | 16 | 47 | 0.048 | 1.40e-02 | C |
| heavy_hex | 18 | 43 | 0.232 | 6.28e-02 | F |
| heavy_hex | 20 | 96 | 0.177 | 1.46e-02 | F |
| heavy_hex | 21 | 14 | 0.341 | 4.44e-02 | F |
| heavy_hex | 22 | 31 | 0.203 | 2.54e-02 | F |
| heavy_hex | 24 | 47 | 0.286 | 3.35e-02 | F |
| heavy_hex | 26 | 27 | 0.251 | 2.65e-02 | F |
| heavy_hex | 30 | 51 | 0.337 | 1.13e-02 | F |
| heavy_hex | 32 | 10 | 0.384 | 2.02e-02 | F |
| heavy_hex | 40 | 33 | 0.201 | 1.45e-02 | F |
| heavy_hex | 50 | 6 | 0.286 | 1.84e-02 | F |
| heavy_hex | 60 | 6 | 0.498 | 2.92e-02 | F |
| ladder | 16 | 6 | 0.078 | 1.40e-02 | D |
| ladder | 20 | 24 | 0.259 | 7.24e-03 | F |
| ladder | 26 | 14 | 0.517 | 8.95e-03 | F |
| ladder | 30 | 14 | 0.513 | 7.44e-03 | F |
| ladder | 40 | 6 | 1.558 | 9.88e-03 | F |
| square | 16 | 26 | 0.081 | 1.65e-02 | D |
| square | 20 | 26 | 0.791 | 1.75e-02 | F |
| square | 30 | 13 | 2.406 | 2.97e-02 | F |
| triangular | 12 | 10 | 1.785 | 7.33e-02 | F |
| triangular | 16 | 10 | 28.703 | 1.63e-01 | F |
| triangular | 24 | 10 | 23.474 | 2.56e-01 | F |

## Best Model per Topology (Auto-Tracked)

| Topology | p | Checkpoint | Arch | Pass% | N-range | Best N | Worst N |
|----------|---|-----------|------|-------|---------|--------|---------|
| chain_1d | 1 | unified_tfim_br_chain_1d_multiN_6+8... | baseline | 71% | — | — | — |
| chain_1d | 1 | unified_tfim_br_chain_1d_multiN_6+8... | baseline | 45% | N=6-31 | N6=100% | N31=17% |
| chain_1d | 1 | unified_tfim_br_chain_1d_multiN_6+8... | baseline | 36% | N=6-31 | N6=100% | N31=17% |
| chain_1d | 1 | unifMPNN__chain_1d_p1_h_0p5_1p5.pt | baseline | 0% | — | — | — |
| chain_1d | 1 | unifMPNN__chain_1d_p1_h_0p5_1p5_v2.pt | baseline | 0% | — | — | — |
| heavy_hex | 1 | unified_tfim_br_heavy_hex_multiN_4+... | baseline | 10% | N=4-29 | N4=0% | all 0% |
| heavy_hex | 1 | unified_tfim_br_heavy_hex_multiN_4+... | baseline | 0% | N=4-29 | N4=0% | all 0% |
| heavy_hex | 1 | unified_tfim_br_heavy_hex_fromMT_4+... | baseline | 44% | — | — | — |
| heavy_hex | 1 | unified_tfim_br_heavy_hex_multiN_4+... | baseline | 75% | — | — | — |
| heavy_hex | 1 | unifMPNN__heavy_hex_p1_res_mse.pt | residual | 0% | — | — | — |
| heavy_hex | 1 | unifMPNN__heavy_hex_p1_res_film_mse.pt | residual+film | 0% | — | — | — |
| heavy_hex | 1 | unified_tfim_br_heavy_hex_fromMT_4+... | baseline | 0% | — | — | — |
| heavy_hex | 1 | unified_multiN_heavyhex_p1.pt | baseline | 0% | — | — | — |
| heavy_hex | 2 | unified_tfim_br_heavy_hex_multiN_4+... | baseline | 0% | — | — | — |
| heavy_hex | 2 | unified_tfim_br_heavy_hex_multiN_4+... | baseline | 32% | — | — | — |
| heavy_hex | 2 | unified_tfim_br_heavy_hex_multiN_4+... | baseline | 36% | — | — | — |
| heavy_hex | 2 | unified_tfim_br_heavy_hex_multiN_4+... | baseline | 85% | — | — | — |
| heavy_hex | 2 | unified_tfim_br_heavy_hex_multiN_4+... | baseline | 60% | — | — | — |
| heavy_hex | 2 | unified_tfim_br_heavy_hex_multiN_4+... | baseline | 20% | — | — | — |
| heavy_hex | 2 | unified_tfim_br_heavy_hex_multiN_4+... | baseline | 67% | — | — | — |
| ladder | 1 | unified_tfim_br_ladder_multiN_4+6+8... | baseline | 45% | N=4-24 | N4=83% | N10=47% |
| ladder | 1 | unified_tfim_br_ladder_fromMT_4+6+8... | baseline | 0% | — | — | — |
| ladder | 1 | unified_tfim_br_ladder_multiN_4+6+8... | baseline | 0% | — | — | — |
| multi_topology | 1 | unified_tfim_br_MT_residual+film_p1.pt | residual+film | 6% | N=3-31 | N4=50% | N3=17% |
| multi_topology | 1 | unifMPNN__MT_p1_res_film_base.pt | residual+film | 0% | — | — | — |
| square | 1 | unified_tfim_br_square_multiN_4+6+8... | baseline | 33% | N=4-21 | N4=83% | N10=33% |
| square | 1 | unified_tfim_br_square_multiN_4+6+8... | baseline | 33% | N=4-21 | N4=100% | N6=83% |
| triangular | 1 | unified_tfim_br_triangular_multiN_3... | baseline | 25% | N=3-16 | N6=50% | N4=33% |
| **multi_topo** | unified_tfim_br_MT_residual+film_p1.pt | residual+film | 6% | N=3-31 | N4=50% | N3=17% |

---
*Generated by `ResultIndex.refresh_status()` from ResultIndex*