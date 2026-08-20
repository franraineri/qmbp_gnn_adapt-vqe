# Zoo Model Evaluation Report

**Generated**: 2026-08-20 19:10 UTC
**Elapsed**: 6.0s
**Models evaluated**: 11

---

## square — `unified_tfim_br_square_multiN_4+6+8+10+12+14_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 3.9957e-03 | 0.0588 | 1.47e-02 | 0.0340 | B |
| 6 | IN | 8 | 4.6947e-02 | 0.0850 | 1.42e-02 | 0.0599 | C |
| 8 | IN | 8 | 9.9857e-02 | 0.0616 | 7.70e-03 | 0.0180 | A |
| 10 | IN | 8 | 7.5461e-02 | 0.0799 | 7.99e-03 | 0.0194 | A |
| 12 | IN | 8 | 6.3995e-02 | 0.2249 | 1.87e-02 | 0.3563 | F |
| 14 | IN | 8 | 6.5658e-02 | 0.1294 | 9.25e-03 | 0.0537 | C |
| 16 | IN | 8 | 1.8704e-03 | 0.2401 | 1.50e-02 | 0.0597 | C |
| 20 | EXT | 26 | — | — | 1.75e-02 | 0.7914 | F |
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
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.4327 (possible stale e_exact or gap) N=16

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
| 20 | IN | 8 | 1.7116e-03 | 0.1820 | 9.10e-03 | 0.2928 | F |
| 26 | IN | 8 | 9.8858e-04 | 0.2629 | 1.01e-02 | 0.6368 | F |
| 30 | IN | 8 | 1.5169e-03 | 0.2166 | 7.22e-03 | 0.4806 | F |
| 40 | EXT | 6 | — | — | 9.88e-03 | 1.5579 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0897 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1160 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0857 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1866 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1990 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1447 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1400 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=1.481 is 5× the mean — median may be more representative N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0156 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.9788 (possible stale e_exact or gap) N=26
> - ⚠️ Outlier: max ΔE/gap=3.509 is 7× the mean — median may be more representative N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 2.7743 (possible stale e_exact or gap) N=30

## multi_topology — `unified_tfim_br_MT_residual+film_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 3 | IN | 8 | 2.6504e-02 | 0.0037 | 1.24e-03 | 0.0012 | A |
| 4 | IN | 8 | 5.6611e-01 | 0.0446 | 1.12e-02 | 0.0234 | B |
| 4 | IN | 8 | 5.8442e-03 | 0.0893 | 2.23e-02 | 0.0593 | C |
| 4 | IN | 8 | 5.3376e-03 | 0.0588 | 1.47e-02 | 0.0340 | B |
| 4 | IN | 8 | 1.5788e-02 | 0.0727 | 1.82e-02 | 0.1682 | F |
| 6 | IN | 8 | 4.7086e-02 | 0.0406 | 6.77e-03 | 0.0153 | A |
| 6 | IN | 8 | 1.7510e-01 | 0.0245 | 4.08e-03 | 0.0086 | A |
| 6 | IN | 8 | 3.4116e-02 | 0.0688 | 1.15e-02 | 0.0196 | A |
| 6 | IN | 8 | 4.8475e-02 | 0.0850 | 1.42e-02 | 0.0599 | C |
| 6 | IN | 8 | 5.9424e-03 | 0.0959 | 1.60e-02 | 0.0304 | B |
| 8 | IN | 8 | 3.9982e-02 | 0.0498 | 6.23e-03 | 0.0198 | A |
| 8 | IN | 8 | 4.2210e-02 | 0.1319 | 1.65e-02 | 0.0666 | D |
| 8 | IN | 8 | 9.9604e-02 | 0.0616 | 7.70e-03 | 0.0180 | A |
| 8 | IN | 8 | 6.2548e-03 | 0.5464 | 6.83e-02 | 3.0954 | F |
| 10 | IN | 8 | 3.2752e-02 | 0.0434 | 4.34e-03 | 0.0145 | A |
| 10 | IN | 8 | 2.0032e-01 | 0.0552 | 5.52e-03 | 0.0266 | B |
| 10 | IN | 8 | 2.8739e-02 | 0.1349 | 1.35e-02 | 0.0333 | B |
| 10 | IN | 8 | 7.5632e-02 | 0.0799 | 7.99e-03 | 0.0194 | A |
| 10 | IN | 8 | 7.6611e-02 | 0.4476 | 4.48e-02 | 0.7619 | F |
| 12 | IN | 8 | 2.7495e-02 | 0.0461 | 3.84e-03 | 0.0218 | A |
| 12 | IN | 8 | 3.1521e-01 | 0.0716 | 5.97e-03 | 0.0568 | C |
| 12 | IN | 8 | 1.2328e-02 | 0.1124 | 9.37e-03 | 0.0319 | B |
| 12 | IN | 8 | 6.4590e-02 | 0.2249 | 1.87e-02 | 0.3563 | F |
| 12 | IN | 8 | 5.1422e-03 | 0.5694 | 4.75e-02 | 1.0070 | F |
| 14 | IN | 8 | 6.9487e-02 | 0.2594 | 1.85e-02 | 0.1866 | F |
| 14 | IN | 8 | 6.7317e-02 | 0.1294 | 9.25e-03 | 0.0537 | C |
| 15 | IN | 8 | 1.2478e-01 | 0.0590 | 3.93e-03 | 0.0153 | A |
| 16 | IN | 8 | 2.1574e-03 | 0.1008 | 6.30e-03 | 0.0197 | A |
| 16 | IN | 8 | 1.5821e-01 | 0.0850 | 5.31e-03 | 0.0510 | C |
| 16 | IN | 8 | 2.8122e-02 | 0.1396 | 8.73e-03 | 0.0357 | B |
| 16 | IN | 8 | 2.8282e-03 | 0.2401 | 1.50e-02 | 0.0597 | C |
| 20 | IN | 8 | 3.9612e-02 | 0.0895 | 4.47e-03 | 0.0333 | B |
| 20 | IN | 8 | 1.5656e-01 | 0.1243 | 6.21e-03 | 0.0449 | B |
| 20 | IN | 8 | 4.6206e-03 | 0.1820 | 9.10e-03 | 0.2928 | F |
| 24 | EXT | 10 | — | — | 2.56e-01 | 23.4736 | F |
| 26 | IN | 6 | 2.1006e-03 | 0.0997 | 3.83e-03 | 0.0154 | B |
| 26 | IN | 5 | 3.4549e-03 | 0.1229 | 4.73e-03 | 0.0247 | B |
| 26 | IN | 8 | 3.7482e-03 | 0.2629 | 1.01e-02 | 0.6368 | F |
| 30 | IN | 8 | 2.4913e-03 | 0.1960 | 6.53e-03 | 0.0331 | B |
| 30 | IN | 8 | 1.6709e-01 | 0.5553 | 1.85e-02 | 0.7698 | F |
| 30 | IN | 8 | 3.4232e-03 | 0.2166 | 7.22e-03 | 0.4806 | F |
| 40 | IN | 8 | 3.6480e-03 | 0.2139 | 5.35e-03 | 0.0424 | B |
| 40 | IN | 6 | 3.2097e-01 | 0.0716 | 1.79e-03 | 0.0390 | C |
| 60 | IN | 8 | 4.6270e-03 | 0.3242 | 5.40e-03 | 0.0594 | C |
| 80 | EXT | 8 | — | — | 8.85e-03 | 0.1658 | D |
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
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1506 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0839 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1400 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.4327 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1099 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1864 (possible stale e_exact or gap) N=20
> - ⚠️ Outlier: max ΔE/gap=1.481 is 5× the mean — median may be more representative N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0156 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1208 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.9788 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.3921 (possible stale e_exact or gap) N=30
> - ⚠️ Outlier: max ΔE/gap=4.958 is 6× the mean — median may be more representative N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.9199 (possible stale e_exact or gap) N=30
> - ⚠️ Outlier: max ΔE/gap=3.509 is 7× the mean — median may be more representative N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 2.7743 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5344 (possible stale e_exact or gap) N=40
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.8478 (possible stale e_exact or gap) N=60

## heavy_hex — `unified_tfim_br_heavy_hex_multiN_4+6+10+12+16+20+40_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 4.3954e-03 | 0.0446 | 1.12e-02 | 0.0234 | B |
| 6 | IN | 8 | 2.0972e-01 | 0.0245 | 4.08e-03 | 0.0086 | A |
| 10 | IN | 8 | 7.5982e-02 | 0.0552 | 5.52e-03 | 0.0266 | B |
| 12 | IN | 8 | 2.0096e-01 | 0.0716 | 5.97e-03 | 0.0568 | C |
| 16 | IN | 8 | 4.8022e-02 | 0.0850 | 5.31e-03 | 0.0510 | C |
| 20 | IN | 8 | 8.5909e-02 | 0.1243 | 6.21e-03 | 0.0449 | B |
| 26 | IN | 5 | 4.8332e-02 | 0.1229 | 4.73e-03 | 0.0247 | B |
| 30 | IN | 8 | 2.5327e-01 | 0.5553 | 1.85e-02 | 0.7698 | F |
| 40 | IN | 6 | 1.8169e-02 | 0.0716 | 1.79e-03 | 0.0390 | C |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0499 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0839 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1864 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Outlier: max ΔE/gap=4.958 is 6× the mean — median may be more representative N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.9199 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## chain_1d — `unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20+26+60_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 6 | IN | 8 | 4.7533e-02 | 0.0406 | 6.77e-03 | 0.0153 | A |
| 8 | IN | 8 | 4.1203e-02 | 0.0498 | 6.23e-03 | 0.0198 | A |
| 10 | IN | 8 | 3.3033e-02 | 0.0434 | 4.34e-03 | 0.0145 | A |
| 12 | IN | 8 | 2.7171e-02 | 0.0461 | 3.84e-03 | 0.0218 | A |
| 15 | IN | 8 | 1.2750e-01 | 0.0590 | 3.93e-03 | 0.0153 | A |
| 16 | IN | 8 | 1.0541e-03 | 0.1008 | 6.30e-03 | 0.0197 | A |
| 20 | IN | 8 | 3.9527e-02 | 0.0895 | 4.47e-03 | 0.0333 | B |
| 26 | IN | 6 | 7.4895e-04 | 0.0997 | 3.83e-03 | 0.0154 | B |
| 30 | IN | 8 | 1.5753e-03 | 0.1960 | 6.53e-03 | 0.0331 | B |
| 40 | IN | 8 | 2.1457e-03 | 0.2139 | 5.35e-03 | 0.0424 | B |
| 60 | IN | 8 | 2.8674e-03 | 0.3242 | 5.40e-03 | 0.0594 | C |
| 80 | EXT | 8 | — | — | 8.85e-03 | 0.1658 | D |
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
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1506 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1099 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1208 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.3921 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5344 (possible stale e_exact or gap) N=40
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.8478 (possible stale e_exact or gap) N=60

## chain_1d — `unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20+60_p1_v4.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 6 | IN | 8 | 4.5883e-02 | 0.0406 | 6.77e-03 | 0.0153 | A |
| 8 | IN | 8 | 4.1163e-02 | 0.0498 | 6.23e-03 | 0.0198 | A |
| 10 | IN | 8 | 2.9364e-02 | 0.0434 | 4.34e-03 | 0.0145 | A |
| 12 | IN | 8 | 2.7463e-02 | 0.0461 | 3.84e-03 | 0.0218 | A |
| 15 | IN | 8 | 1.4725e-01 | 0.0590 | 3.93e-03 | 0.0153 | A |
| 16 | IN | 8 | 7.6865e-02 | 0.1008 | 6.30e-03 | 0.0197 | A |
| 20 | IN | 8 | 1.1787e-01 | 0.0895 | 4.47e-03 | 0.0333 | B |
| 26 | IN | 6 | 1.7453e-02 | 0.0997 | 3.83e-03 | 0.0154 | B |
| 30 | IN | 8 | 2.4716e-03 | 0.1960 | 6.53e-03 | 0.0331 | B |
| 40 | IN | 8 | 2.0272e-03 | 0.2139 | 5.35e-03 | 0.0424 | B |
| 60 | IN | 8 | 2.1585e-03 | 0.3242 | 5.40e-03 | 0.0594 | C |
| 80 | EXT | 8 | — | — | 8.85e-03 | 0.1658 | D |
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
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1506 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1099 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1208 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.3921 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5344 (possible stale e_exact or gap) N=40
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.8478 (possible stale e_exact or gap) N=60

## chain_1d — `unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20+60_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 6 | IN | 8 | 4.5883e-02 | 0.0406 | 6.77e-03 | 0.0153 | A |
| 8 | IN | 8 | 4.1163e-02 | 0.0498 | 6.23e-03 | 0.0198 | A |
| 10 | IN | 8 | 2.9364e-02 | 0.0434 | 4.34e-03 | 0.0145 | A |
| 12 | IN | 8 | 2.7463e-02 | 0.0461 | 3.84e-03 | 0.0218 | A |
| 15 | IN | 8 | 1.4725e-01 | 0.0590 | 3.93e-03 | 0.0153 | A |
| 16 | IN | 8 | 7.6865e-02 | 0.1008 | 6.30e-03 | 0.0197 | A |
| 20 | IN | 8 | 1.1787e-01 | 0.0895 | 4.47e-03 | 0.0333 | B |
| 26 | IN | 6 | 1.7453e-02 | 0.0997 | 3.83e-03 | 0.0154 | B |
| 30 | IN | 8 | 2.4716e-03 | 0.1960 | 6.53e-03 | 0.0331 | B |
| 40 | IN | 8 | 2.0272e-03 | 0.2139 | 5.35e-03 | 0.0424 | B |
| 60 | IN | 8 | 2.1585e-03 | 0.3242 | 5.40e-03 | 0.0594 | C |
| 80 | EXT | 8 | — | — | 8.85e-03 | 0.1658 | D |
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
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1506 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1099 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1208 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.3921 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5344 (possible stale e_exact or gap) N=40
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.8478 (possible stale e_exact or gap) N=60

## heavy_hex — `unified_tfim_br_heavy_hex_multiN_4+6+10+12+16+20+40_p1_v1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 3.2462e-02 | 0.0446 | 1.12e-02 | 0.0234 | B |
| 6 | IN | 8 | 1.6590e-01 | 0.0245 | 4.08e-03 | 0.0086 | A |
| 10 | IN | 8 | 1.1371e-01 | 0.0552 | 5.52e-03 | 0.0266 | B |
| 12 | IN | 8 | 2.3249e-01 | 0.0716 | 5.97e-03 | 0.0568 | C |
| 16 | IN | 8 | 4.7231e-02 | 0.0850 | 5.31e-03 | 0.0510 | C |
| 20 | IN | 8 | 8.3843e-02 | 0.1243 | 6.21e-03 | 0.0449 | B |
| 26 | IN | 5 | 1.0463e-01 | 0.1229 | 4.73e-03 | 0.0247 | B |
| 30 | IN | 8 | 2.6020e-01 | 0.5553 | 1.85e-02 | 0.7698 | F |
| 40 | IN | 6 | 1.0041e-01 | 0.0716 | 1.79e-03 | 0.0390 | C |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0499 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0839 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1864 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Outlier: max ΔE/gap=4.958 is 6× the mean — median may be more representative N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.9199 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unified_tfim_br_heavy_hex_fromMT_4+6+10+12+16+20_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 2.3509e-03 | 0.0446 | 1.12e-02 | 0.0234 | B |
| 6 | IN | 8 | 7.0841e-02 | 0.0245 | 4.08e-03 | 0.0086 | A |
| 10 | IN | 8 | 8.2772e-02 | 0.0552 | 5.52e-03 | 0.0266 | B |
| 12 | IN | 8 | 1.9457e-01 | 0.0716 | 5.97e-03 | 0.0568 | C |
| 16 | IN | 8 | 5.6939e-02 | 0.0850 | 5.31e-03 | 0.0510 | C |
| 20 | IN | 8 | 7.5634e-02 | 0.1243 | 6.21e-03 | 0.0449 | B |
| 26 | IN | 5 | 1.4886e-01 | 0.1229 | 4.73e-03 | 0.0247 | B |
| 30 | IN | 8 | 2.8967e-01 | 0.5553 | 1.85e-02 | 0.7698 | F |
| 40 | IN | 6 | 4.2320e-01 | 0.0716 | 1.79e-03 | 0.0390 | C |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0499 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0839 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1864 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Outlier: max ΔE/gap=4.958 is 6× the mean — median may be more representative N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.9199 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## square — `unified_tfim_br_square_multiN_4+6+8+10+12+14_p1_v4.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 3.3876e-03 | 0.0588 | 1.47e-02 | 0.0340 | B |
| 6 | IN | 8 | 4.6972e-02 | 0.0850 | 1.42e-02 | 0.0599 | C |
| 8 | IN | 8 | 6.4160e-02 | 0.0616 | 7.70e-03 | 0.0180 | A |
| 10 | IN | 8 | 7.8291e-02 | 0.0799 | 7.99e-03 | 0.0194 | A |
| 12 | IN | 8 | 1.1446e-01 | 0.2249 | 1.87e-02 | 0.3563 | F |
| 14 | IN | 8 | 1.3439e-01 | 0.1294 | 9.25e-03 | 0.0537 | C |
| 16 | IN | 8 | 4.7264e-02 | 0.2401 | 1.50e-02 | 0.0597 | C |
| 20 | EXT | 26 | — | — | 1.75e-02 | 0.7914 | F |
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
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.4327 (possible stale e_exact or gap) N=16

---

## Summary Ranking

| Topology | Checkpoint | In-dist θ MSE | Out-dist |ΔE|/N | Grade |
|----------|-----------|:---:|:---:|:---:|
| heavy_hex | unified_tfim_br_heavy_hex_multiN_4+6+10+12+16+20+4 | 1.0497e-01 | — | D (poor) |
| heavy_hex | unified_tfim_br_heavy_hex_multiN_4+6+10+12+16+20+4 | 1.2676e-01 | — | D (poor) |
| heavy_hex | unified_tfim_br_heavy_hex_fromMT_4+6+10+12+16+20_p | 1.4943e-01 | — | D (poor) |
| chain_1d | unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20 | 2.9487e-02 | 2.21e-02 | F (failing) |
| chain_1d | unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20 | 4.6361e-02 | 2.21e-02 | F (failing) |
| chain_1d | unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20 | 4.6361e-02 | 2.21e-02 | F (failing) |
| ladder | unified_tfim_br_ladder_multiN_4+6+8+10+12+16+20+26 | 2.0376e-02 | 9.88e-03 | F (failing) |
| square | unified_tfim_br_square_multiN_4+6+8+10+12+14_p1.pt | 5.1112e-02 | 2.32e-02 | F (failing) |
| square | unified_tfim_br_square_multiN_4+6+8+10+12+14_p1_v4 | 6.9846e-02 | 2.32e-02 | F (failing) |
| multi_topology | unified_tfim_br_MT_residual+film_p1.pt | 7.2615e-02 | 6.89e-02 | F (failing) |
| triangular | unified_tfim_br_triangular_multiN_3+4+6_p1.pt | 9.6193e-02 | 2.09e-01 | F (failing) |

---

# MT vs ST Head-to-Head Comparison

**Generated**: 2026-08-20 19:10 UTC
**Score**: MT **7** — ST **1** — Ties **7**
**MT avg quality_score**: 0.091 | **ST avg quality_score**: 0.079

## Per-Topology Summary

| Topology | MT score | ST score | Winner | Δ | MT wins | ST wins |
|----------|:--------:|:--------:|:------:|:-:|:-------:|:-------:|
| chain_1d | 0.202 | 0.263 | 🔴 ST | -0.061 | 2 | 1 |
| heavy_hex | 0.126 | 0.053 | 🟢 MT | +0.073 | 3 | 0 |
| ladder | 0.074 | 0.055 | ⚪ tie | +0.019 | 0 | 0 |
| square | 0.059 | 0.000 | 🟢 MT | +0.059 | 2 | 0 |
| triangular | 0.008 | 0.017 | ⚪ tie | -0.009 | 0 | 0 |

## Per-N Breakdown

| Topology | N | MT score | MT ΔE/gap | MT grade | ST score | ST ΔE/gap | ST grade | Winner |
|----------|:-:|:--------:|:---------:|:--------:|:--------:|:---------:|:--------:|:------:|
| chain_1d | 10 | 0.271 | 9.0% | D | 0.686 | 3.8% | B | ❌ ST |
| chain_1d | 16 | 0.179 | 14.9% | F | 0.068 | 23.5% | F | ✅ MT |
| chain_1d | 20 | 0.158 | 18.7% | F | 0.034 | 33.2% | F | ✅ MT |
| heavy_hex | 10 | 0.164 | 13.3% | F | 0.077 | 18.7% | F | ✅ MT |
| heavy_hex | 16 | 0.122 | 20.9% | F | 0.055 | 29.3% | F | ✅ MT |
| heavy_hex | 20 | 0.093 | 83.5% | F | 0.026 | 161.6% | F | ✅ MT |
| ladder | 10 | 0.091 | 37.9% | F | 0.068 | 48.0% | F | — tie |
| ladder | 16 | 0.068 | 181.1% | F | 0.050 | 230.1% | F | — tie |
| ladder | 20 | 0.064 | 162.5% | F | 0.047 | 203.9% | F | — tie |
| square | 10 | 0.085 | 41.1% | F | 0.000 | — | — | ✅ MT |
| square | 16 | 0.033 | — | F | 0.000 | — | — | ✅ MT |
| triangular | 8 | 0.020 | 42.2% | F | 0.029 | 40.9% | F | — tie |
| triangular | 11 | 0.006 | 129.7% | F | 0.018 | 116.0% | F | — tie |
| triangular | 12 | 0.003 | 281.2% | F | 0.012 | 234.7% | F | — tie |
| triangular | 13 | 0.002 | 573.8% | F | 0.009 | 459.7% | F | — tie |

---
*Auto-generated from model_comparison/ (15 comparisons)*
*Decision metric: quality_score (continuous 0-1, sigmoid-based on mean ΔE/gap + P90 + |ΔE|/N)*

*Generated by `scripts/analysis/evaluate_zoo_models.py`*
