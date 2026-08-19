# Zoo Model Evaluation Report

**Generated**: 2026-08-19 04:05 UTC
**Elapsed**: 0.3s
**Models evaluated**: 6

---

## square — `unified_tfim_br_square_multiN_4+6+8+10+12+14_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 3.3876e-03 | 0.0588 | 1.47e-02 | 0.0340 | B |
| 6 | IN | 8 | 4.6972e-02 | 0.0850 | 1.42e-02 | 0.0599 | C |
| 8 | IN | 8 | 6.4160e-02 | 0.0616 | 7.70e-03 | 0.0180 | A |
| 10 | IN | 8 | 9.0171e-02 | 0.1240 | 1.24e-02 | 0.0612 | C |
| 12 | IN | 8 | 1.1446e-01 | 0.2249 | 1.87e-02 | 0.3563 | F |
| 14 | IN | 8 | 1.3439e-01 | 0.1294 | 9.25e-03 | 0.0537 | C |
| 16 | EXT | 26 | — | — | 1.64e-02 | 0.0809 | D |
| 20 | EXT | 26 | — | — | 1.74e-02 | 0.7903 | F |
| 30 | EXT | 12 | — | — | 2.89e-02 | 2.5784 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0617 (possible stale e_exact or gap) N=4
> - ⚠️ Outlier: max ΔE/gap=0.360 is 6× the mean — median may be more representative N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0580 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0714 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1156 (possible stale e_exact or gap) N=10
> - ⚠️ Outlier: max ΔE/gap=2.587 is 7× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.5825 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1378 (possible stale e_exact or gap) N=14

## chain_1d — `unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20+60_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 6 | IN | 8 | 4.5883e-02 | 0.0406 | 6.77e-03 | 0.0153 | A |
| 8 | IN | 8 | 4.1163e-02 | 0.0521 | 6.52e-03 | 0.0203 | A |
| 10 | IN | 8 | 3.1066e-02 | 0.0561 | 5.61e-03 | 0.0294 | B |
| 12 | IN | 8 | 2.7463e-02 | 0.0461 | 3.84e-03 | 0.0218 | A |
| 15 | IN | 8 | 1.4725e-01 | 0.0620 | 4.13e-03 | 0.0162 | A |
| 16 | EXT | 6 | — | — | 8.95e-03 | 0.0206 | B |
| 20 | IN | 8 | 1.5024e-01 | 0.0718 | 3.59e-03 | 0.0243 | A |
| 30 | EXT | 20 | — | — | 7.71e-03 | 0.0358 | B |
| 40 | EXT | 16 | — | — | 7.35e-03 | 0.0480 | C |
| 60 | EXT | 16 | — | — | 9.23e-03 | 0.0862 | D |
| 100 | EXT | 12 | — | — | 8.15e-03 | 0.1111 | D |
| 150 | EXT | 3 | — | — | 3.58e-02 | 0.7841 | F |
| 200 | EXT | 3 | — | — | 3.59e-02 | 1.0472 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0385 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0779 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0570 (possible stale e_exact or gap) N=10
> - ⚠️ Outlier: max ΔE/gap=0.120 is 6× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0597 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1271 (possible stale e_exact or gap) N=15
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0781 (possible stale e_exact or gap) N=20

## ladder — `unified_tfim_br_ladder_fromMT_4+6+8+10+12+16+20_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 1.3789e-01 | 0.0893 | 2.23e-02 | 0.0593 | C |
| 6 | IN | 8 | 2.1023e-02 | 0.0688 | 1.15e-02 | 0.0196 | A |
| 8 | IN | 8 | 3.8838e-02 | 0.1319 | 1.65e-02 | 0.0666 | D |
| 10 | IN | 8 | 3.9320e-02 | 0.1422 | 1.42e-02 | 0.0604 | C |
| 12 | IN | 8 | 9.1870e-03 | 0.1124 | 9.37e-03 | 0.0319 | B |
| 14 | IN | 8 | 7.0008e-02 | 0.2594 | 1.85e-02 | 0.1866 | F |
| 16 | IN | 8 | 5.1751e-02 | 0.3067 | 1.92e-02 | 0.3716 | F |
| 20 | IN | 8 | 3.7168e-03 | 0.2011 | 1.01e-02 | 0.4113 | F |
| 26 | EXT | 14 | — | — | 8.95e-03 | 0.5173 | F |
| 30 | EXT | 14 | — | — | 7.44e-03 | 0.5130 | F |
| 40 | EXT | 6 | — | — | 9.88e-03 | 1.5579 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0897 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1160 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0857 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1134 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1990 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1447 (possible stale e_exact or gap) N=14
> - ⚠️ Outlier: max ΔE/gap=2.326 is 6× the mean — median may be more representative N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.2666 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0156 (possible stale e_exact or gap) N=20

## heavy_hex — `unified_tfim_br_heavy_hex_multiN_4+6+10+12+16+20+40_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 3.2462e-02 | 0.0446 | 1.12e-02 | 0.0234 | B |
| 6 | IN | 8 | 1.6590e-01 | 0.0245 | 4.08e-03 | 0.0086 | A |
| 10 | IN | 8 | 1.1558e-01 | 0.0496 | 4.96e-03 | 0.0181 | A |
| 12 | IN | 8 | 2.3249e-01 | 0.0716 | 5.97e-03 | 0.0568 | C |
| 16 | IN | 8 | 4.9103e-02 | 0.0501 | 3.13e-03 | 0.0217 | A |
| 20 | EXT | 33 | — | — | 7.07e-03 | 0.1498 | D |
| 30 | EXT | 23 | — | — | 1.39e-02 | 0.7444 | F |
| 40 | EXT | 6 | — | — | 1.79e-03 | 0.0390 | C |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0499 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0467 (possible stale e_exact or gap) N=16

## multi_topology — `unified_tfim_br_multi_topology_multiN_ablation_film_p1.pt`

*No evaluation data available.*

## triangular — `unified_tfim_br_triangular_multiN_3+4+6_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 3 | IN | 8 | 3.1413e-01 | 0.0037 | 1.24e-03 | 0.0012 | A |
| 4 | IN | 8 | 2.3382e-01 | 0.0727 | 1.82e-02 | 0.1682 | F |
| 6 | IN | 8 | 3.1334e-04 | 0.1153 | 1.92e-02 | 0.0486 | C |
| 8 | IN | 8 | 8.1856e-03 | 0.5464 | 6.83e-02 | 3.0954 | F |
| 10 | IN | 8 | 1.0479e-01 | 0.4476 | 4.48e-02 | 0.7619 | F |
| 12 | IN | 8 | 3.1783e-02 | 0.5694 | 4.75e-02 | 1.0070 | F |
| 16 | EXT | 10 | — | — | 1.63e-01 | 28.7030 | F |
| 24 | EXT | 10 | — | — | 2.56e-01 | 23.4736 | F |

> **Metric Warnings:**
> - ⚠️ Outlier: max ΔE/gap=0.007 is 6× the mean — median may be more representative N=3
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0149 (possible stale e_exact or gap) N=3
> - ⚠️ Outlier: max ΔE/gap=1.087 is 6× the mean — median may be more representative N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.7688 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0950 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=23.500 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 21.1681 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 2.3036 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.2810 (possible stale e_exact or gap) N=12

---

## Summary Ranking

| Topology | Checkpoint | In-dist θ MSE | Out-dist |ΔE|/N | Grade |
|----------|-----------|:---:|:---:|:---:|
| heavy_hex | unified_tfim_br_heavy_hex_multiN_4+6+10+12+16+20+4 | 1.1911e-01 | 7.58e-03 | D (poor) |
| chain_1d | unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20 | 7.3845e-02 | 1.62e-02 | F (failing) |
| ladder | unified_tfim_br_ladder_fromMT_4+6+8+10+12+16+20_p1 | 4.6467e-02 | 8.76e-03 | F (failing) |
| square | unified_tfim_br_square_multiN_4+6+8+10+12+14_p1.pt | 7.5589e-02 | 2.09e-02 | F (failing) |
| triangular | unified_tfim_br_triangular_multiN_3+4+6_p1.pt | 1.1550e-01 | 2.09e-01 | F (failing) |
| multi_topology | unified_tfim_br_multi_topology_multiN_ablation_fil | — | — | F (failing) |

---
*Generated by `scripts/analysis/evaluate_zoo_models.py`*