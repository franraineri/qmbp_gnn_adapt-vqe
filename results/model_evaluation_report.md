# Zoo Model Evaluation Report

**Generated**: 2026-08-19 17:27 UTC
**Elapsed**: 0.8s
**Models evaluated**: 6

---

## chain_1d — `unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20+60_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 6 | IN | 8 | 4.5883e-02 | 0.0406 | 6.77e-03 | 0.0153 | A |
| 8 | IN | 8 | 4.1163e-02 | 0.0521 | 6.52e-03 | 0.0203 | A |
| 10 | IN | 8 | 2.9364e-02 | 0.0434 | 4.34e-03 | 0.0145 | A |
| 12 | IN | 8 | 2.7463e-02 | 0.0461 | 3.84e-03 | 0.0218 | A |
| 15 | IN | 8 | 1.4725e-01 | 0.0620 | 4.13e-03 | 0.0162 | A |
| 16 | IN | 8 | 7.0022e-02 | 0.0866 | 5.41e-03 | 0.0150 | A |
| 20 | IN | 8 | 1.1787e-01 | 0.0776 | 3.88e-03 | 0.0293 | B |
| 26 | IN | 6 | 1.7453e-02 | 0.0997 | 3.83e-03 | 0.0154 | B |
| 30 | IN | 7 | 1.3843e-03 | 0.1220 | 4.07e-03 | 0.0200 | B |
| 40 | EXT | 24 | — | — | 5.90e-03 | 0.0408 | B |
| 60 | EXT | 22 | — | — | 7.45e-03 | 0.0725 | C |
| 100 | EXT | 19 | — | — | 7.98e-03 | 0.1369 | D |
| 150 | EXT | 3 | — | — | 3.58e-02 | 0.7841 | F |
| 200 | EXT | 3 | — | — | 3.59e-02 | 1.0472 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0385 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0779 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0540 (possible stale e_exact or gap) N=10
> - ⚠️ Outlier: max ΔE/gap=0.120 is 6× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0597 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1271 (possible stale e_exact or gap) N=15
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1342 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0854 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1208 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1569 (possible stale e_exact or gap) N=30

## square — `unified_tfim_br_square_multiN_4+6+8+10+12+14_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 3.9957e-03 | 0.0588 | 1.47e-02 | 0.0340 | B |
| 6 | IN | 8 | 4.6947e-02 | 0.0850 | 1.42e-02 | 0.0599 | C |
| 8 | IN | 8 | 9.9857e-02 | 0.0616 | 7.70e-03 | 0.0180 | A |
| 10 | IN | 8 | 7.5461e-02 | 0.0789 | 7.89e-03 | 0.0192 | A |
| 12 | IN | 8 | 6.3995e-02 | 0.2249 | 1.87e-02 | 0.3563 | F |
| 14 | IN | 8 | 6.5658e-02 | 0.1294 | 9.25e-03 | 0.0537 | C |
| 16 | IN | 8 | 3.3029e-03 | 0.1867 | 1.17e-02 | 0.0355 | B |
| 20 | EXT | 26 | — | — | 1.74e-02 | 0.7903 | F |
| 30 | EXT | 12 | — | — | 2.89e-02 | 2.5784 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0617 (possible stale e_exact or gap) N=4
> - ⚠️ Outlier: max ΔE/gap=0.360 is 6× the mean — median may be more representative N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0580 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0714 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1110 (possible stale e_exact or gap) N=10
> - ⚠️ Outlier: max ΔE/gap=2.587 is 7× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.5825 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1378 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1665 (possible stale e_exact or gap) N=16

## triangular — `unified_tfim_br_triangular_multiN_3+4+6_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 3 | IN | 8 | 3.0566e-01 | 0.0037 | 1.24e-03 | 0.0012 | A |
| 4 | IN | 8 | 1.8114e-01 | 0.0727 | 1.82e-02 | 0.1682 | F |
| 6 | IN | 8 | 3.4395e-04 | 0.0959 | 1.60e-02 | 0.0304 | B |
| 8 | IN | 8 | 1.0681e-03 | 0.5464 | 6.83e-02 | 3.0954 | F |
| 10 | IN | 8 | 8.7132e-02 | 0.4476 | 4.48e-02 | 0.7619 | F |
| 12 | IN | 8 | 1.8148e-03 | 0.5694 | 4.75e-02 | 1.0070 | F |
| 16 | EXT | 10 | — | — | 1.63e-01 | 28.7030 | F |
| 24 | EXT | 10 | — | — | 2.56e-01 | 23.4736 | F |

> **Metric Warnings:**
> - ⚠️ Outlier: max ΔE/gap=0.007 is 6× the mean — median may be more representative N=3
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0149 (possible stale e_exact or gap) N=3
> - ⚠️ Outlier: max ΔE/gap=1.087 is 6× the mean — median may be more representative N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.7688 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1024 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=23.500 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 21.1681 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 2.3036 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.2810 (possible stale e_exact or gap) N=12

## ladder — `unified_tfim_br_ladder_multiN_4+6+8+10+12+16+20+26+40_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 3.4981e-03 | 0.0893 | 2.23e-02 | 0.0593 | C |
| 6 | IN | 8 | 3.0965e-02 | 0.0688 | 1.15e-02 | 0.0196 | A |
| 8 | IN | 8 | 3.8699e-02 | 0.1319 | 1.65e-02 | 0.0666 | D |
| 10 | IN | 8 | 2.5735e-02 | 0.1349 | 1.35e-02 | 0.0333 | B |
| 12 | IN | 8 | 9.3013e-03 | 0.1124 | 9.37e-03 | 0.0319 | B |
| 14 | IN | 8 | 6.6112e-02 | 0.2594 | 1.85e-02 | 0.1866 | F |
| 16 | IN | 8 | 2.5237e-02 | 0.1396 | 8.73e-03 | 0.0357 | B |
| 20 | IN | 8 | 1.7116e-03 | 0.1828 | 9.14e-03 | 0.2948 | F |
| 26 | EXT | 14 | — | — | 8.95e-03 | 0.5173 | F |
| 30 | EXT | 14 | — | — | 7.44e-03 | 0.5130 | F |
| 40 | EXT | 6 | — | — | 9.88e-03 | 1.5579 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0897 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1160 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0857 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1866 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1990 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1447 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1400 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=1.493 is 5× the mean — median may be more representative N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0241 (possible stale e_exact or gap) N=20

## multi_topology — `unified_tfim_br_MT_residual+film_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 3 | IN | 8 | 7.2555e-02 | 0.0037 | 1.24e-03 | 0.0012 | A |
| 4 | IN | 8 | 1.5265e-01 | 0.0446 | 1.12e-02 | 0.0234 | B |
| 4 | IN | 8 | 5.2139e-02 | 0.0893 | 2.23e-02 | 0.0593 | C |
| 4 | IN | 8 | 4.2338e-02 | 0.0588 | 1.47e-02 | 0.0340 | B |
| 4 | IN | 8 | 9.0019e-03 | 0.0727 | 1.82e-02 | 0.1682 | F |
| 6 | IN | 8 | 7.5763e-02 | 0.0406 | 6.77e-03 | 0.0153 | A |
| 6 | IN | 8 | 1.5045e-01 | 0.0245 | 4.08e-03 | 0.0086 | A |
| 6 | IN | 8 | 4.9228e-02 | 0.0688 | 1.15e-02 | 0.0196 | A |
| 6 | IN | 8 | 6.5454e-02 | 0.0850 | 1.42e-02 | 0.0599 | C |
| 6 | IN | 8 | 4.4852e-04 | 0.0959 | 1.60e-02 | 0.0304 | B |
| 8 | IN | 8 | 6.8978e-02 | 0.0521 | 6.52e-03 | 0.0203 | A |
| 8 | IN | 8 | 5.6688e-02 | 0.1319 | 1.65e-02 | 0.0666 | D |
| 8 | IN | 8 | 9.0818e-02 | 0.0616 | 7.70e-03 | 0.0180 | A |
| 8 | IN | 8 | 1.4074e-02 | 0.5464 | 6.83e-02 | 3.0954 | F |
| 10 | IN | 8 | 5.7367e-02 | 0.0434 | 4.34e-03 | 0.0145 | A |
| 10 | IN | 8 | 1.2978e-01 | 0.0556 | 5.56e-03 | 0.0268 | B |
| 10 | IN | 8 | 3.8025e-02 | 0.1349 | 1.35e-02 | 0.0333 | B |
| 10 | IN | 8 | 7.4144e-02 | 0.0789 | 7.89e-03 | 0.0192 | A |
| 10 | IN | 8 | 1.0866e-01 | 0.4476 | 4.48e-02 | 0.7619 | F |
| 12 | IN | 8 | 6.1980e-02 | 0.0461 | 3.84e-03 | 0.0218 | A |
| 12 | IN | 8 | 2.2751e-01 | 0.0716 | 5.97e-03 | 0.0568 | C |
| 12 | IN | 8 | 1.7446e-02 | 0.1124 | 9.37e-03 | 0.0319 | B |
| 12 | IN | 8 | 6.5174e-02 | 0.2249 | 1.87e-02 | 0.3563 | F |
| 12 | IN | 8 | 4.1030e-02 | 0.5694 | 4.75e-02 | 1.0070 | F |
| 14 | IN | 8 | 7.6054e-02 | 0.2594 | 1.85e-02 | 0.1866 | F |
| 14 | IN | 8 | 6.5477e-02 | 0.1294 | 9.25e-03 | 0.0537 | C |
| 15 | IN | 8 | 1.4929e-01 | 0.0620 | 4.13e-03 | 0.0162 | A |
| 16 | IN | 8 | 2.2290e-02 | 0.0866 | 5.41e-03 | 0.0150 | A |
| 16 | IN | 8 | 1.1159e-01 | 0.0850 | 5.31e-03 | 0.0510 | C |
| 16 | IN | 8 | 3.2227e-02 | 0.1396 | 8.73e-03 | 0.0357 | B |
| 16 | IN | 8 | 1.9754e-03 | 0.1867 | 1.17e-02 | 0.0355 | B |
| 20 | IN | 8 | 6.8379e-02 | 0.0776 | 3.88e-03 | 0.0293 | B |
| 20 | IN | 8 | 8.4891e-03 | 0.1074 | 5.37e-03 | 0.0222 | A |
| 20 | IN | 8 | 5.6657e-03 | 0.1828 | 9.14e-03 | 0.2948 | F |
| 24 | EXT | 10 | — | — | 2.56e-01 | 23.4736 | F |
| 26 | IN | 6 | 1.9970e-02 | 0.0997 | 3.83e-03 | 0.0154 | B |
| 26 | IN | 5 | 7.7625e-03 | 0.1229 | 4.73e-03 | 0.0247 | B |
| 30 | IN | 7 | 2.2352e-02 | 0.1220 | 4.07e-03 | 0.0200 | B |
| 30 | IN | 6 | 1.9733e-02 | 0.1149 | 3.83e-03 | 0.0251 | B |
| 40 | EXT | 24 | — | — | 5.90e-03 | 0.0408 | B |
| 40 | EXT | 6 | — | — | 1.79e-03 | 0.0390 | C |
| 40 | EXT | 6 | — | — | 9.88e-03 | 1.5579 | F |
| 60 | EXT | 22 | — | — | 7.45e-03 | 0.0725 | C |
| 100 | EXT | 19 | — | — | 7.98e-03 | 0.1369 | D |
| 150 | EXT | 3 | — | — | 3.58e-02 | 0.7841 | F |
| 200 | EXT | 3 | — | — | 3.59e-02 | 1.0472 | F |

> **Metric Warnings:**
> - ⚠️ Outlier: max ΔE/gap=0.007 is 6× the mean — median may be more representative N=3
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0149 (possible stale e_exact or gap) N=3
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0499 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0897 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0617 (possible stale e_exact or gap) N=4
> - ⚠️ Outlier: max ΔE/gap=1.087 is 6× the mean — median may be more representative N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.7688 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0385 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1160 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=0.360 is 6× the mean — median may be more representative N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0580 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1024 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0779 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0857 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0714 (possible stale e_exact or gap) N=8
> - ⚠️ Outlier: max ΔE/gap=23.500 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 21.1681 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0540 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1866 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1110 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 2.3036 (possible stale e_exact or gap) N=10
> - ⚠️ Outlier: max ΔE/gap=0.120 is 6× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0597 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1990 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=2.587 is 7× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.5825 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.2810 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1447 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1378 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1271 (possible stale e_exact or gap) N=15
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1342 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0839 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1400 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1665 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0854 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1181 (possible stale e_exact or gap) N=20
> - ⚠️ Outlier: max ΔE/gap=1.493 is 5× the mean — median may be more representative N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0241 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1208 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1569 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1228 (possible stale e_exact or gap) N=30

## heavy_hex — `unified_tfim_br_heavy_hex_fromMT_4+6+10+12+16+20_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 2.3509e-03 | 0.0446 | 1.12e-02 | 0.0234 | B |
| 6 | IN | 8 | 7.0841e-02 | 0.0245 | 4.08e-03 | 0.0086 | A |
| 10 | IN | 8 | 8.2772e-02 | 0.0556 | 5.56e-03 | 0.0268 | B |
| 12 | IN | 8 | 1.9457e-01 | 0.0716 | 5.97e-03 | 0.0568 | C |
| 16 | IN | 8 | 5.6939e-02 | 0.0850 | 5.31e-03 | 0.0510 | C |
| 20 | IN | 8 | 1.5650e-01 | 0.1074 | 5.37e-03 | 0.0222 | A |
| 26 | IN | 5 | 1.4886e-01 | 0.1229 | 4.73e-03 | 0.0247 | B |
| 30 | IN | 6 | 1.1799e-01 | 0.1149 | 3.83e-03 | 0.0251 | B |
| 40 | EXT | 6 | — | — | 1.79e-03 | 0.0390 | C |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0499 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0839 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1181 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1228 (possible stale e_exact or gap) N=30

---

## Summary Ranking

| Topology | Checkpoint | In-dist θ MSE | Out-dist |ΔE|/N | Grade |
|----------|-----------|:---:|:---:|:---:|
| heavy_hex | unified_tfim_br_heavy_hex_fromMT_4+6+10+12+16+20_p | 1.0385e-01 | 1.79e-03 | B (good) |
| chain_1d | unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20 | 5.5317e-02 | 1.86e-02 | D (poor) |
| ladder | unified_tfim_br_ladder_multiN_4+6+8+10+12+16+20+26 | 2.5157e-02 | 8.76e-03 | F (failing) |
| square | unified_tfim_br_square_multiN_4+6+8+10+12+14_p1.pt | 5.1317e-02 | 2.32e-02 | F (failing) |
| multi_topology | unified_tfim_br_MT_residual+film_p1.pt | 6.1394e-02 | 4.51e-02 | F (failing) |
| triangular | unified_tfim_br_triangular_multiN_3+4+6_p1.pt | 9.6193e-02 | 2.09e-01 | F (failing) |

---

# MT vs ST Head-to-Head Comparison

**Generated**: 2026-08-19 17:27 UTC
**Score**: MT **7** — ST **1** — Ties **7**
**MT avg pass_rate**: 22% | **ST avg pass_rate**: 7%

## Per-Topology Summary

| Topology | MT pass% | ST pass% | Winner | Δ | MT wins | ST wins |
|----------|:--------:|:--------:|:------:|:-:|:-------:|:-------:|
| chain_1d | 40% | 18% | 🟢 MT | +22% | 2 | 1 |
| heavy_hex | 36% | 0% | 🟢 MT | +36% | 3 | 0 |
| ladder | 22% | 16% | 🟢 MT | +7% | 1 | 0 |
| square | 20% | 0% | 🟢 MT | +20% | 1 | 0 |
| triangular | 0% | 0% | ⚪ tie | +0% | 0 | 0 |

## Per-N Breakdown

| Topology | N | MT pass% | MT grade | ST pass% | ST grade | Winner |
|----------|:-:|:--------:|:--------:|:--------:|:--------:|:------:|
| chain_1d | 10 | 47% | D | 53% | B | ❌ ST |
| chain_1d | 16 | 40% | F | 0% | F | ✅ MT |
| chain_1d | 20 | 33% | F | 0% | F | ✅ MT |
| heavy_hex | 10 | 47% | F | 0% | F | ✅ MT |
| heavy_hex | 16 | 33% | F | 0% | F | ✅ MT |
| heavy_hex | 20 | 27% | F | 0% | F | ✅ MT |
| ladder | 10 | 47% | F | 47% | F | — tie |
| ladder | 16 | 20% | F | 0% | F | ✅ MT |
| ladder | 20 | 0% | F | 0% | F | — tie |
| square | 10 | 40% | F | 0% | — | ✅ MT |
| square | 16 | 0% | F | 0% | — | — tie |
| triangular | 8 | 0% | F | 0% | F | — tie |
| triangular | 11 | 0% | F | 0% | F | — tie |
| triangular | 12 | 0% | F | 0% | F | — tie |
| triangular | 13 | 0% | F | 0% | F | — tie |

---
*Auto-generated from model_comparison/ (15 comparisons)*

*Generated by `scripts/analysis/evaluate_zoo_models.py`*