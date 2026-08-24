# Zoo Model Evaluation Report

**Generated**: 2026-08-24 01:44 UTC
**Elapsed**: 3.8s
**Models evaluated**: 11

---

## square — `unified_tfim_br_square_multiN_4+6+8+10+12+14_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 3.9545e-03 | 0.0588 | 1.47e-02 | 0.0340 | B |
| 6 | IN | 8 | 4.6900e-02 | 0.0850 | 1.42e-02 | 0.0599 | C |
| 8 | IN | 8 | 9.9884e-02 | 0.0616 | 7.70e-03 | 0.0180 | A |
| 10 | IN | 8 | 7.5470e-02 | 0.0799 | 7.99e-03 | 0.0194 | A |
| 12 | IN | 8 | 6.3982e-02 | 0.2249 | 1.87e-02 | 0.3563 | F |
| 14 | IN | 8 | 6.5605e-02 | 0.1294 | 9.25e-03 | 0.0537 | C |
| 16 | IN | 8 | 1.8440e-03 | 0.2401 | 1.50e-02 | 0.0597 | C |
| 20 | IN | 8 | 8.6110e-03 | 4.3143 | 2.16e-01 | 13.6147 | F |
| 30 | EXT | 13 | — | — | 2.97e-02 | 2.4057 | F |

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
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 15.3549 (possible stale e_exact or gap) N=20

## triangular — `unified_tfim_br_triangular_multiN_3+4+6_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 3 | IN | 8 | 3.0614e-01 | 0.0037 | 1.24e-03 | 0.0012 | A |
| 4 | IN | 8 | 1.6334e-01 | 0.0727 | 1.82e-02 | 0.1682 | F |
| 6 | IN | 8 | 3.0297e-04 | 0.0959 | 1.60e-02 | 0.0304 | B |
| 8 | IN | 8 | 2.0394e-03 | 0.5464 | 6.83e-02 | 3.0954 | F |
| 10 | IN | 8 | 9.2920e-02 | 0.4476 | 4.48e-02 | 0.7619 | F |
| 12 | IN | 8 | 6.4746e-03 | 0.5694 | 4.75e-02 | 1.0070 | F |
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
| 4 | IN | 8 | 1.7903e-03 | 0.0893 | 2.23e-02 | 0.0593 | C |
| 6 | IN | 8 | 3.0440e-02 | 0.0688 | 1.15e-02 | 0.0196 | A |
| 8 | IN | 8 | 3.8279e-02 | 0.1319 | 1.65e-02 | 0.0666 | D |
| 10 | IN | 8 | 2.5378e-02 | 0.1349 | 1.35e-02 | 0.0333 | B |
| 12 | IN | 8 | 8.8079e-03 | 0.1124 | 9.37e-03 | 0.0319 | B |
| 14 | IN | 8 | 6.5576e-02 | 0.2594 | 1.85e-02 | 0.1866 | F |
| 16 | IN | 8 | 2.4874e-02 | 0.1396 | 8.73e-03 | 0.0357 | B |
| 20 | IN | 8 | 1.0654e-03 | 0.1820 | 9.10e-03 | 0.2928 | F |
| 26 | IN | 8 | 2.2192e-04 | 0.2629 | 1.01e-02 | 0.6368 | F |
| 30 | IN | 8 | 3.2046e-04 | 0.2166 | 7.22e-03 | 0.4806 | F |
| 40 | IN | 2 | 4.4692e-04 | 0.1659 | 4.15e-03 | 0.0465 | D |

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
> - ⚠️ Only 2 points — means have low statistical confidence N=40
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1298 (possible stale e_exact or gap) N=40

## multi_topology — `unified_tfim_br_MT_residual+film_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 3 | IN | 8 | 2.6348e-02 | 0.0037 | 1.24e-03 | 0.0012 | A |
| 4 | IN | 6 | 1.1482e-03 | 0.0241 | 6.02e-03 | 0.0050 | B |
| 4 | IN | 8 | 4.9060e-01 | 0.0446 | 1.12e-02 | 0.0234 | B |
| 4 | IN | 8 | 5.6863e-03 | 0.0893 | 2.23e-02 | 0.0593 | C |
| 4 | IN | 8 | 5.2425e-03 | 0.0588 | 1.47e-02 | 0.0340 | B |
| 4 | IN | 8 | 1.5231e-02 | 0.0727 | 1.82e-02 | 0.1682 | F |
| 6 | IN | 8 | 4.7953e-02 | 0.0406 | 6.77e-03 | 0.0153 | A |
| 6 | IN | 8 | 1.7775e-01 | 0.0245 | 4.08e-03 | 0.0086 | A |
| 6 | IN | 8 | 3.4038e-02 | 0.0688 | 1.15e-02 | 0.0196 | A |
| 6 | IN | 8 | 4.8388e-02 | 0.0850 | 1.42e-02 | 0.0599 | C |
| 6 | IN | 8 | 5.8968e-03 | 0.0959 | 1.60e-02 | 0.0304 | B |
| 8 | IN | 8 | 4.1364e-02 | 0.0498 | 6.23e-03 | 0.0198 | A |
| 8 | IN | 8 | 3.2350e-01 | 0.2490 | 3.11e-02 | 478.3574 | F |
| 8 | IN | 8 | 4.2081e-02 | 0.1319 | 1.65e-02 | 0.0666 | D |
| 8 | IN | 8 | 9.9763e-02 | 0.0616 | 7.70e-03 | 0.0180 | A |
| 8 | IN | 8 | 6.2856e-03 | 0.5464 | 6.83e-02 | 3.0954 | F |
| 10 | IN | 8 | 3.3738e-02 | 0.0434 | 4.34e-03 | 0.0145 | A |
| 10 | IN | 8 | 1.9627e-01 | 0.0552 | 5.52e-03 | 0.0266 | B |
| 10 | IN | 8 | 2.8675e-02 | 0.1349 | 1.35e-02 | 0.0333 | B |
| 10 | IN | 8 | 7.5771e-02 | 0.0799 | 7.99e-03 | 0.0194 | A |
| 10 | IN | 8 | 7.6892e-02 | 0.4476 | 4.48e-02 | 0.7619 | F |
| 12 | IN | 8 | 2.8633e-02 | 0.0461 | 3.84e-03 | 0.0218 | A |
| 12 | IN | 8 | 3.0902e-01 | 0.0716 | 5.97e-03 | 0.0568 | C |
| 12 | IN | 8 | 1.2309e-02 | 0.1124 | 9.37e-03 | 0.0319 | B |
| 12 | IN | 8 | 6.4605e-02 | 0.2249 | 1.87e-02 | 0.3563 | F |
| 12 | IN | 8 | 5.4536e-03 | 0.5694 | 4.75e-02 | 1.0070 | F |
| 14 | IN | 5 | 5.9632e-04 | 0.0911 | 6.51e-03 | 0.0167 | B |
| 14 | IN | 8 | 1.7882e-01 | 0.6801 | 4.86e-02 | 24.6514 | F |
| 14 | IN | 8 | 6.9374e-02 | 0.2594 | 1.85e-02 | 0.1866 | F |
| 14 | IN | 8 | 6.7382e-02 | 0.1294 | 9.25e-03 | 0.0537 | C |
| 15 | IN | 8 | 1.2712e-01 | 0.0590 | 3.93e-03 | 0.0153 | A |
| 16 | IN | 8 | 1.6839e-03 | 0.1008 | 6.30e-03 | 0.0197 | A |
| 16 | IN | 8 | 5.0286e-01 | 0.0633 | 3.95e-03 | 0.0214 | A |
| 16 | IN | 8 | 2.8107e-02 | 0.1396 | 8.73e-03 | 0.0357 | B |
| 16 | IN | 8 | 2.9272e-03 | 0.2401 | 1.50e-02 | 0.0597 | C |
| 18 | IN | 8 | 2.5298e-01 | 0.0296 | 1.65e-03 | 0.0071 | A |
| 20 | IN | 8 | 4.0382e-02 | 0.0895 | 4.47e-03 | 0.0333 | B |
| 20 | IN | 8 | 1.5656e-01 | 0.1243 | 6.21e-03 | 0.0449 | B |
| 20 | IN | 8 | 4.6148e-03 | 0.1820 | 9.10e-03 | 0.2928 | F |
| 20 | IN | 8 | 8.9337e-03 | 4.3143 | 2.16e-01 | 13.6147 | F |
| 21 | IN | 8 | 3.9067e-01 | 0.0153 | 7.30e-04 | 0.0030 | A |
| 22 | IN | 8 | 1.5634e-01 | 0.0289 | 1.31e-03 | 0.0145 | A |
| 24 | IN | 8 | 8.4468e-02 | 0.0411 | 1.71e-03 | 0.0207 | A |
| 26 | IN | 6 | 1.8271e-03 | 0.0997 | 3.83e-03 | 0.0154 | B |
| 26 | IN | 5 | 2.9529e-03 | 0.1229 | 4.73e-03 | 0.0247 | B |
| 26 | IN | 8 | 3.7306e-03 | 0.2629 | 1.01e-02 | 0.6368 | F |
| 30 | IN | 8 | 2.6089e-03 | 0.1960 | 6.53e-03 | 0.0331 | B |
| 30 | IN | 8 | 1.6653e-01 | 0.5553 | 1.85e-02 | 0.7698 | F |
| 30 | IN | 8 | 3.4685e-03 | 0.2166 | 7.22e-03 | 0.4806 | F |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 8 | 3.7375e-03 | 0.2139 | 5.35e-03 | 0.0424 | B |
| 40 | IN | 6 | 3.3313e-01 | 0.0716 | 1.79e-03 | 0.0390 | C |
| 40 | IN | 2 | 3.2678e-03 | 0.1659 | 4.15e-03 | 0.0465 | D |
| 50 | EXT | 6 | — | — | 1.23e-01 | 2.5883 | F |
| 60 | IN | 8 | 4.7969e-03 | 0.3242 | 5.40e-03 | 0.0594 | C |
| 80 | EXT | 8 | — | — | 8.85e-03 | 0.1658 | D |
| 100 | EXT | 19 | — | — | 7.98e-03 | 0.1369 | D |
| 150 | EXT | 3 | — | — | 3.58e-02 | 0.7841 | F |
| 200 | EXT | 3 | — | — | 3.59e-02 | 1.0472 | F |

> **Metric Warnings:**
> - ⚠️ Outlier: max ΔE/gap=0.007 is 6× the mean — median may be more representative N=3
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0149 (possible stale e_exact or gap) N=3
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=4
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
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
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
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1055 (possible stale e_exact or gap) N=14
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1447 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1378 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1271 (possible stale e_exact or gap) N=15
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1506 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0866 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1400 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.4327 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0506 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1099 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1864 (possible stale e_exact or gap) N=20
> - ⚠️ Outlier: max ΔE/gap=1.481 is 5× the mean — median may be more representative N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0156 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 15.3549 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
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
> - ⚠️ Only 2 points — means have low statistical confidence N=40
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1298 (possible stale e_exact or gap) N=40
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.8478 (possible stale e_exact or gap) N=60

## heavy_hex — `unified_tfim_br_heavy_hex_multiN_4+6+10+12+16+20+40_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 2.3658e+01 | 0.0446 | 1.12e-02 | 0.0234 | B |
| 6 | IN | 8 | 6.6958e+01 | 0.0245 | 4.08e-03 | 0.0086 | A |
| 8 | IN | 8 | 5.2519e+01 | 0.2490 | 3.11e-02 | 478.3574 | F |
| 10 | IN | 8 | 1.1856e+02 | 0.0552 | 5.52e-03 | 0.0266 | B |
| 12 | IN | 8 | 1.3452e+02 | 0.0716 | 5.97e-03 | 0.0568 | C |
| 14 | IN | 8 | 9.3138e+01 | 0.6801 | 4.86e-02 | 24.6514 | F |
| 16 | IN | 8 | 1.9814e+02 | 0.0633 | 3.95e-03 | 0.0214 | A |
| 18 | IN | 8 | 3.8906e+02 | 0.0296 | 1.65e-03 | 0.0071 | A |
| 20 | IN | 8 | 4.4908e+02 | 0.1243 | 6.21e-03 | 0.0449 | B |
| 21 | IN | 8 | 7.4560e+02 | 0.0153 | 7.30e-04 | 0.0030 | A |
| 22 | IN | 8 | 5.5427e+02 | 0.0289 | 1.31e-03 | 0.0145 | A |
| 24 | IN | 8 | 6.1607e+02 | 0.0411 | 1.71e-03 | 0.0207 | A |
| 26 | IN | 5 | 1.0491e+03 | 0.1229 | 4.73e-03 | 0.0247 | B |
| 30 | IN | 8 | 7.1095e+02 | 0.5553 | 1.85e-02 | 0.7698 | F |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 6.9620e+02 | 0.0716 | 1.79e-03 | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.23e-01 | 2.5883 | F |
| 60 | EXT | 6 | — | — | 6.97e-02 | 1.9004 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0499 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0866 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0506 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1864 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Outlier: max ΔE/gap=4.958 is 6× the mean — median may be more representative N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.9199 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## chain_1d — `unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20+26+60_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 6 | 4.2892e-04 | 0.0241 | 6.02e-03 | 0.0050 | B |
| 6 | IN | 8 | 4.7533e-02 | 0.0406 | 6.77e-03 | 0.0153 | A |
| 8 | IN | 8 | 4.1203e-02 | 0.0498 | 6.23e-03 | 0.0198 | A |
| 10 | IN | 8 | 3.3033e-02 | 0.0434 | 4.34e-03 | 0.0145 | A |
| 12 | IN | 8 | 2.7171e-02 | 0.0461 | 3.84e-03 | 0.0218 | A |
| 14 | IN | 5 | 0.0000e+00 | 0.0911 | 6.51e-03 | 0.0167 | B |
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
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0385 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0779 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0540 (possible stale e_exact or gap) N=10
> - ⚠️ Outlier: max ΔE/gap=0.120 is 6× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0597 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1055 (possible stale e_exact or gap) N=14
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
| 4 | IN | 6 | 5.4325e-04 | 0.0241 | 6.02e-03 | 0.0050 | B |
| 6 | IN | 8 | 4.7338e-02 | 0.0406 | 6.77e-03 | 0.0153 | A |
| 8 | IN | 8 | 4.3363e-02 | 0.0498 | 6.23e-03 | 0.0198 | A |
| 10 | IN | 8 | 5.0710e-02 | 0.0434 | 4.34e-03 | 0.0145 | A |
| 12 | IN | 8 | 6.0847e-02 | 0.0461 | 3.84e-03 | 0.0218 | A |
| 14 | IN | 5 | 6.9282e-02 | 0.0911 | 6.51e-03 | 0.0167 | B |
| 15 | IN | 8 | 2.0189e-01 | 0.0590 | 3.93e-03 | 0.0153 | A |
| 16 | IN | 8 | 8.4249e-02 | 0.1008 | 6.30e-03 | 0.0197 | A |
| 20 | IN | 8 | 1.2937e-01 | 0.0895 | 4.47e-03 | 0.0333 | B |
| 26 | IN | 6 | 3.2276e-02 | 0.0997 | 3.83e-03 | 0.0154 | B |
| 30 | IN | 8 | 5.7665e-02 | 0.1960 | 6.53e-03 | 0.0331 | B |
| 40 | IN | 8 | 2.7299e-01 | 0.2139 | 5.35e-03 | 0.0424 | B |
| 60 | IN | 8 | 5.4843e-01 | 0.3242 | 5.40e-03 | 0.0594 | C |
| 80 | EXT | 8 | — | — | 8.85e-03 | 0.1658 | D |
| 100 | EXT | 19 | — | — | 7.98e-03 | 0.1369 | D |
| 150 | EXT | 3 | — | — | 3.58e-02 | 0.7841 | F |
| 200 | EXT | 3 | — | — | 3.59e-02 | 1.0472 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0385 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0779 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0540 (possible stale e_exact or gap) N=10
> - ⚠️ Outlier: max ΔE/gap=0.120 is 6× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0597 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1055 (possible stale e_exact or gap) N=14
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
| 4 | IN | 6 | 5.4325e-04 | 0.0241 | 6.02e-03 | 0.0050 | B |
| 6 | IN | 8 | 4.7338e-02 | 0.0406 | 6.77e-03 | 0.0153 | A |
| 8 | IN | 8 | 4.3363e-02 | 0.0498 | 6.23e-03 | 0.0198 | A |
| 10 | IN | 8 | 5.0710e-02 | 0.0434 | 4.34e-03 | 0.0145 | A |
| 12 | IN | 8 | 6.0847e-02 | 0.0461 | 3.84e-03 | 0.0218 | A |
| 14 | IN | 5 | 6.9282e-02 | 0.0911 | 6.51e-03 | 0.0167 | B |
| 15 | IN | 8 | 2.0189e-01 | 0.0590 | 3.93e-03 | 0.0153 | A |
| 16 | IN | 8 | 8.4249e-02 | 0.1008 | 6.30e-03 | 0.0197 | A |
| 20 | IN | 8 | 1.2937e-01 | 0.0895 | 4.47e-03 | 0.0333 | B |
| 26 | IN | 6 | 3.2276e-02 | 0.0997 | 3.83e-03 | 0.0154 | B |
| 30 | IN | 8 | 5.7665e-02 | 0.1960 | 6.53e-03 | 0.0331 | B |
| 40 | IN | 8 | 2.7299e-01 | 0.2139 | 5.35e-03 | 0.0424 | B |
| 60 | IN | 8 | 5.4843e-01 | 0.3242 | 5.40e-03 | 0.0594 | C |
| 80 | EXT | 8 | — | — | 8.85e-03 | 0.1658 | D |
| 100 | EXT | 19 | — | — | 7.98e-03 | 0.1369 | D |
| 150 | EXT | 3 | — | — | 3.58e-02 | 0.7841 | F |
| 200 | EXT | 3 | — | — | 3.59e-02 | 1.0472 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0385 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0779 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0540 (possible stale e_exact or gap) N=10
> - ⚠️ Outlier: max ΔE/gap=0.120 is 6× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0597 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1055 (possible stale e_exact or gap) N=14
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
| 4 | IN | 8 | 7.9081e-01 | 0.0446 | 1.12e-02 | 0.0234 | B |
| 6 | IN | 8 | 5.2152e-01 | 0.0245 | 4.08e-03 | 0.0086 | A |
| 8 | IN | 8 | 8.5178e-01 | 0.2490 | 3.11e-02 | 478.3574 | F |
| 10 | IN | 8 | 7.3458e-01 | 0.0552 | 5.52e-03 | 0.0266 | B |
| 12 | IN | 8 | 1.2015e+00 | 0.0716 | 5.97e-03 | 0.0568 | C |
| 14 | IN | 8 | 9.5015e-01 | 0.6801 | 4.86e-02 | 24.6514 | F |
| 16 | IN | 8 | 1.0403e+00 | 0.0633 | 3.95e-03 | 0.0214 | A |
| 18 | IN | 8 | 1.3231e+00 | 0.0296 | 1.65e-03 | 0.0071 | A |
| 20 | IN | 8 | 1.3645e+00 | 0.1243 | 6.21e-03 | 0.0449 | B |
| 21 | IN | 8 | 1.4325e+00 | 0.0153 | 7.30e-04 | 0.0030 | A |
| 22 | IN | 8 | 1.4048e+00 | 0.0289 | 1.31e-03 | 0.0145 | A |
| 24 | IN | 8 | 1.2350e+00 | 0.0411 | 1.71e-03 | 0.0207 | A |
| 26 | IN | 5 | 1.5099e+00 | 0.1229 | 4.73e-03 | 0.0247 | B |
| 30 | IN | 8 | 2.1755e+00 | 0.5553 | 1.85e-02 | 0.7698 | F |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 2.1194e+00 | 0.0716 | 1.79e-03 | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.23e-01 | 2.5883 | F |
| 60 | EXT | 6 | — | — | 6.97e-02 | 1.9004 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0499 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0866 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0506 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1864 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Outlier: max ΔE/gap=4.958 is 6× the mean — median may be more representative N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.9199 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unified_tfim_br_heavy_hex_fromMT_4+6+10+12+16+20_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 5.4594e-01 | 0.0446 | 1.12e-02 | 0.0234 | B |
| 6 | IN | 8 | 3.1405e-01 | 0.0245 | 4.08e-03 | 0.0086 | A |
| 8 | IN | 8 | 1.0813e+00 | 0.2490 | 3.11e-02 | 478.3574 | F |
| 10 | IN | 8 | 1.1041e+00 | 0.0552 | 5.52e-03 | 0.0266 | B |
| 12 | IN | 8 | 3.8039e-01 | 0.0716 | 5.97e-03 | 0.0568 | C |
| 14 | IN | 8 | 2.3707e+00 | 0.6801 | 4.86e-02 | 24.6514 | F |
| 16 | IN | 8 | 2.4281e+00 | 0.0633 | 3.95e-03 | 0.0214 | A |
| 18 | IN | 8 | 5.5705e-01 | 0.0296 | 1.65e-03 | 0.0071 | A |
| 20 | IN | 8 | 1.2127e+00 | 0.1243 | 6.21e-03 | 0.0449 | B |
| 21 | IN | 8 | 6.2047e-01 | 0.0153 | 7.30e-04 | 0.0030 | A |
| 22 | IN | 8 | 1.3749e+00 | 0.0289 | 1.31e-03 | 0.0145 | A |
| 24 | IN | 8 | 7.4073e-01 | 0.0411 | 1.71e-03 | 0.0207 | A |
| 26 | IN | 5 | 6.2846e-01 | 0.1229 | 4.73e-03 | 0.0247 | B |
| 30 | IN | 8 | 8.2041e-01 | 0.5553 | 1.85e-02 | 0.7698 | F |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 3.2228e+00 | 0.0716 | 1.79e-03 | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.23e-01 | 2.5883 | F |
| 60 | EXT | 6 | — | — | 6.97e-02 | 1.9004 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0499 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0866 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0506 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1864 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Outlier: max ΔE/gap=4.958 is 6× the mean — median may be more representative N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.9199 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## square — `unified_tfim_br_square_multiN_4+6+8+10+12+14_p1_v4.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 5.1386e-03 | 0.0588 | 1.47e-02 | 0.0340 | B |
| 6 | IN | 8 | 6.7428e-02 | 0.0850 | 1.42e-02 | 0.0599 | C |
| 8 | IN | 8 | 1.4349e-01 | 0.0616 | 7.70e-03 | 0.0180 | A |
| 10 | IN | 8 | 1.1198e-01 | 0.0799 | 7.99e-03 | 0.0194 | A |
| 12 | IN | 8 | 1.0516e-01 | 0.2249 | 1.87e-02 | 0.3563 | F |
| 14 | IN | 8 | 1.2233e-01 | 0.1294 | 9.25e-03 | 0.0537 | C |
| 16 | IN | 8 | 3.6094e-02 | 0.2401 | 1.50e-02 | 0.0597 | C |
| 20 | IN | 8 | 4.0019e-02 | 4.3143 | 2.16e-01 | 13.6147 | F |
| 30 | EXT | 13 | — | — | 2.97e-02 | 2.4057 | F |

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
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 15.3549 (possible stale e_exact or gap) N=20

---

## Summary Ranking

| Topology | Checkpoint | In-dist θ MSE | Out-dist |ΔE|/N | Grade |
|----------|-----------|:---:|:---:|:---:|
| chain_1d | unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20 | 2.4984e-02 | 2.21e-02 | D (poor) |
| chain_1d | unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20 | 1.2300e-01 | 2.21e-02 | D (poor) |
| chain_1d | unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20 | 1.2300e-01 | 2.21e-02 | D (poor) |
| ladder | unified_tfim_br_ladder_multiN_4+6+8+10+12+16+20+26 | 1.7927e-02 | — | F (failing) |
| multi_topology | unified_tfim_br_MT_residual+film_p1.pt | 9.0614e-02 | 3.86e-02 | F (failing) |
| heavy_hex | unified_tfim_br_heavy_hex_multiN_4+6+10+12+16+20+4 | 3.9318e+02 | 7.09e-02 | F (failing) |
| heavy_hex | unified_tfim_br_heavy_hex_multiN_4+6+10+12+16+20+4 | 1.2437e+00 | 7.09e-02 | F (failing) |
| heavy_hex | unified_tfim_br_heavy_hex_fromMT_4+6+10+12+16+20_p | 1.1601e+00 | 7.09e-02 | F (failing) |
| square | unified_tfim_br_square_multiN_4+6+8+10+12+14_p1.pt | 4.5781e-02 | 2.97e-02 | F (failing) |
| square | unified_tfim_br_square_multiN_4+6+8+10+12+14_p1_v4 | 7.8955e-02 | 2.97e-02 | F (failing) |
| triangular | unified_tfim_br_triangular_multiN_3+4+6_p1.pt | 9.5203e-02 | 2.09e-01 | F (failing) |

---

# MT vs ST Head-to-Head Comparison

**Generated**: 2026-08-24 01:44 UTC
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