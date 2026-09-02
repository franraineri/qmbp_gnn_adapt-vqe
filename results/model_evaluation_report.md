# Zoo Model Evaluation Report

**Generated**: 2026-08-29 18:13 UTC
**Elapsed**: 75.6s
**Models evaluated**: 39

---

## square — `unified_tfim_br_square_multiN_4+6+8+10+12+14_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 3.9545e-03 | — | — | 0.0340 | B |
| 6 | IN | 8 | 4.6900e-02 | — | — | 0.0599 | C |
| 8 | IN | 8 | 9.9884e-02 | — | — | 0.0180 | A |
| 10 | IN | 8 | 7.5470e-02 | — | — | 0.0194 | A |
| 12 | IN | 8 | 6.3982e-02 | — | — | 0.3563 | F |
| 14 | IN | 8 | 6.5605e-02 | — | — | 0.0537 | C |
| 16 | IN | 8 | 1.8440e-03 | — | — | 0.0597 | C |
| 20 | IN | 8 | 8.6110e-03 | — | — | 13.6147 | F |
| 30 | EXT | 13 | — | — | 2.97e-02 | 2.2370 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0617 (possible stale e_exact or gap) N=4
> - ⚠️ Outlier: max ΔE/gap=0.360 is 6× the mean — median may be more representative N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0580 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0714 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1110 (possible stale e_exact or gap) N=10
> - ⚠️ Outlier: max ΔE/gap=2.587 is 7× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.5825 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1378 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.4329 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 15.3549 (possible stale e_exact or gap) N=20

## triangular — `unified_tfim_br_triangular_multiN_3+4+6_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 3 | IN | 8 | 3.0614e-01 | — | — | 0.0012 | A |
| 4 | IN | 8 | 1.6334e-01 | — | — | 0.1682 | D |
| 6 | IN | 8 | 3.0297e-04 | — | — | 0.0304 | B |
| 8 | IN | 8 | 2.0394e-03 | — | — | 3.0954 | F |
| 10 | IN | 8 | 9.2920e-02 | — | — | 0.7619 | F |
| 12 | IN | 8 | 6.4746e-03 | — | — | 1.0070 | F |
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
| 4 | IN | 8 | 1.7903e-03 | — | — | 0.0593 | C |
| 6 | IN | 8 | 3.0440e-02 | — | — | 0.0196 | A |
| 8 | IN | 8 | 3.8279e-02 | — | — | 0.0666 | C |
| 10 | IN | 8 | 2.5378e-02 | — | — | 0.0333 | B |
| 12 | IN | 8 | 8.8079e-03 | — | — | 0.0319 | B |
| 14 | IN | 8 | 6.5576e-02 | — | — | 0.1866 | D |
| 16 | IN | 8 | 2.4874e-02 | — | — | 0.0357 | B |
| 20 | IN | 8 | 1.0654e-03 | — | — | 0.2928 | F |
| 26 | IN | 8 | 2.2192e-04 | — | — | 0.6368 | F |
| 30 | IN | 8 | 3.2046e-04 | — | — | 0.4806 | F |
| 40 | IN | 2 | 4.4692e-04 | — | — | 0.0465 | D |

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
| 3 | IN | 8 | 2.6348e-02 | — | — | 0.0012 | A |
| 4 | IN | 8 | 1.9541e-01 | — | — | 0.1444 | D |
| 4 | IN | 8 | 4.9060e-01 | — | — | 0.0205 | A |
| 4 | IN | 8 | 5.6863e-03 | — | — | 0.0593 | C |
| 4 | IN | 8 | 5.2425e-03 | — | — | 0.0340 | B |
| 4 | IN | 8 | 1.5231e-02 | — | — | 0.1682 | D |
| 6 | IN | 8 | 3.9571e-02 | — | — | 0.4339 | F |
| 6 | IN | 8 | 1.7775e-01 | — | — | 0.0086 | A |
| 6 | IN | 8 | 3.4038e-02 | — | — | 0.0196 | A |
| 6 | IN | 8 | 4.8388e-02 | — | — | 0.0599 | C |
| 6 | IN | 8 | 5.8968e-03 | — | — | 0.0304 | B |
| 8 | IN | 8 | 5.0137e-02 | — | — | 9.2092 | F |
| 8 | IN | 8 | 3.2350e-01 | — | — | 478.3574 | F |
| 8 | IN | 8 | 4.2081e-02 | — | — | 0.0666 | C |
| 8 | IN | 8 | 9.9763e-02 | — | — | 0.0180 | A |
| 8 | IN | 8 | 6.2856e-03 | — | — | 3.0954 | F |
| 10 | IN | 8 | 1.2319e-02 | — | — | 39.1834 | F |
| 10 | IN | 8 | 1.9627e-01 | — | — | 0.0286 | B |
| 10 | IN | 8 | 2.8675e-02 | — | — | 0.0333 | B |
| 10 | IN | 8 | 7.5771e-02 | — | — | 0.0194 | A |
| 10 | IN | 8 | 7.6892e-02 | — | — | 0.7619 | F |
| 12 | IN | 8 | 2.6440e-02 | — | — | 154.7274 | F |
| 12 | IN | 8 | 3.0902e-01 | — | — | 0.0571 | C |
| 12 | IN | 8 | 1.2309e-02 | — | — | 0.0319 | B |
| 12 | IN | 8 | 6.4605e-02 | — | — | 0.3563 | F |
| 12 | IN | 8 | 5.4597e-03 | — | — | 1.0070 | F |
| 14 | IN | 8 | 2.1407e-02 | — | — | 3806.6317 | F |
| 14 | IN | 8 | 1.7882e-01 | — | — | 24.6514 | F |
| 14 | IN | 8 | 6.9374e-02 | — | — | 0.1866 | D |
| 14 | IN | 8 | 6.7382e-02 | — | — | 0.0537 | C |
| 15 | IN | 8 | 1.2712e-01 | — | — | 0.0154 | A |
| 16 | IN | 8 | 1.2521e-02 | — | — | 49.0765 | F |
| 16 | IN | 8 | 5.4352e-01 | — | — | 0.0452 | B |
| 16 | IN | 8 | 2.8107e-02 | — | — | 0.0357 | B |
| 16 | IN | 8 | 2.9272e-03 | — | — | 0.0597 | C |
| 18 | IN | 8 | 1.1483e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 3.2148e-02 | — | — | 0.7112 | F |
| 20 | IN | 8 | 1.1257e+00 | — | — | 5.3181 | F |
| 20 | IN | 8 | 4.6148e-03 | — | — | 0.2928 | F |
| 20 | IN | 8 | 8.9337e-03 | — | — | 13.6147 | F |
| 21 | IN | 8 | 3.9067e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 1.5634e-01 | — | — | 0.0145 | A |
| 24 | IN | 8 | 8.4468e-02 | — | — | 0.0207 | A |
| 26 | IN | 6 | 1.8271e-03 | — | — | 0.0154 | B |
| 26 | IN | 5 | 2.9529e-03 | — | — | 0.0247 | B |
| 26 | IN | 8 | 3.7306e-03 | — | — | 0.6368 | F |
| 30 | IN | 8 | 2.6089e-03 | — | — | 0.0331 | B |
| 30 | IN | 8 | 1.6653e-01 | — | — | 0.2108 | D |
| 30 | IN | 8 | 3.4685e-03 | — | — | 0.4806 | F |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 8 | 3.7375e-03 | — | — | 0.0424 | B |
| 40 | IN | 6 | 3.3313e-01 | — | — | 0.0390 | C |
| 40 | IN | 2 | 3.2678e-03 | — | — | 0.0465 | D |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | IN | 8 | 4.7969e-03 | — | — | 0.0594 | C |
| 80 | EXT | 8 | — | — | 8.85e-03 | 0.1658 | D |
| 100 | EXT | 19 | — | — | 7.98e-03 | 0.1369 | D |
| 150 | EXT | 3 | — | — | 3.58e-02 | 0.7841 | F |
| 200 | EXT | 3 | — | — | 3.59e-02 | 1.0472 | F |

> **Metric Warnings:**
> - ⚠️ Outlier: max ΔE/gap=0.007 is 6× the mean — median may be more representative N=3
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0149 (possible stale e_exact or gap) N=3
> - ⚠️ Outlier: max ΔE/gap=0.814 is 6× the mean — median may be more representative N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5898 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0897 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0617 (possible stale e_exact or gap) N=4
> - ⚠️ Outlier: max ΔE/gap=1.087 is 6× the mean — median may be more representative N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.7688 (possible stale e_exact or gap) N=4
> - ⚠️ Outlier: max ΔE/gap=3.353 is 8× the mean — median may be more representative N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 2.8794 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1160 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=0.360 is 6× the mean — median may be more representative N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0580 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1024 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=72.951 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 71.8524 (possible stale e_exact or gap) N=8
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0857 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0714 (possible stale e_exact or gap) N=8
> - ⚠️ Outlier: max ΔE/gap=23.500 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 21.1681 (possible stale e_exact or gap) N=8
> - ⚠️ Outlier: max ΔE/gap=312.317 is 8× the mean — median may be more representative N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 310.7901 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1866 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1110 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 2.3036 (possible stale e_exact or gap) N=10
> - ⚠️ Outlier: max ΔE/gap=1235.945 is 8× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1233.9820 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1990 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=2.587 is 7× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.5825 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.2810 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=30381.980 is 8× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 30379.1988 (possible stale e_exact or gap) N=14
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1447 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1378 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1271 (possible stale e_exact or gap) N=15
> - ⚠️ Outlier: max ΔE/gap=390.746 is 8× the mean — median may be more representative N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 388.7826 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1400 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.4329 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Outlier: max ΔE/gap=5.153 is 7× the mean — median may be more representative N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.2738 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
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
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
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
| 4 | IN | 8 | 2.3658e+01 | — | — | 0.0205 | A |
| 6 | IN | 8 | 6.6958e+01 | — | — | 0.0086 | A |
| 8 | IN | 8 | 5.2519e+01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 1.1856e+02 | — | — | 0.0286 | B |
| 12 | IN | 8 | 1.3452e+02 | — | — | 0.0571 | C |
| 14 | IN | 8 | 9.3138e+01 | — | — | 24.6514 | F |
| 16 | IN | 8 | 1.4675e+02 | — | — | 0.0452 | B |
| 18 | IN | 8 | 2.0340e+02 | — | — | 20.4816 | F |
| 20 | IN | 8 | 2.1100e+02 | — | — | 5.3181 | F |
| 21 | IN | 8 | 7.4560e+02 | — | — | 0.0030 | A |
| 22 | IN | 8 | 5.5427e+02 | — | — | 0.0145 | A |
| 24 | IN | 8 | 6.1607e+02 | — | — | 0.0207 | A |
| 26 | IN | 5 | 1.0491e+03 | — | — | 0.0247 | B |
| 30 | IN | 8 | 7.1095e+02 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 6.9620e+02 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## chain_1d — `unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20+26+60_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 2.0312e-01 | — | — | 0.1444 | D |
| 6 | IN | 8 | 3.9338e-02 | — | — | 0.4339 | F |
| 8 | IN | 8 | 5.0125e-02 | — | — | 9.2092 | F |
| 10 | IN | 8 | 1.1206e-02 | — | — | 39.1834 | F |
| 12 | IN | 8 | 2.4774e-02 | — | — | 154.7274 | F |
| 14 | IN | 8 | 2.0176e-02 | — | — | 3806.6317 | F |
| 15 | IN | 8 | 1.2750e-01 | — | — | 0.0154 | A |
| 16 | IN | 8 | 1.1291e-02 | — | — | 49.0765 | F |
| 20 | IN | 8 | 3.1025e-02 | — | — | 0.7112 | F |
| 26 | IN | 6 | 7.4895e-04 | — | — | 0.0154 | B |
| 30 | IN | 8 | 1.5753e-03 | — | — | 0.0331 | B |
| 40 | IN | 8 | 2.1457e-03 | — | — | 0.0424 | B |
| 60 | IN | 8 | 2.8674e-03 | — | — | 0.0594 | C |
| 80 | EXT | 8 | — | — | 8.85e-03 | 0.1658 | D |
| 100 | EXT | 19 | — | — | 7.98e-03 | 0.1369 | D |
| 150 | EXT | 3 | — | — | 3.58e-02 | 0.7841 | F |
| 200 | EXT | 3 | — | — | 3.59e-02 | 1.0472 | F |

> **Metric Warnings:**
> - ⚠️ Outlier: max ΔE/gap=0.814 is 6× the mean — median may be more representative N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5898 (possible stale e_exact or gap) N=4
> - ⚠️ Outlier: max ΔE/gap=3.353 is 8× the mean — median may be more representative N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 2.8794 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=72.951 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 71.8524 (possible stale e_exact or gap) N=8
> - ⚠️ Outlier: max ΔE/gap=312.317 is 8× the mean — median may be more representative N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 310.7901 (possible stale e_exact or gap) N=10
> - ⚠️ Outlier: max ΔE/gap=1235.945 is 8× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1233.9820 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=30381.980 is 8× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 30379.1988 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1271 (possible stale e_exact or gap) N=15
> - ⚠️ Outlier: max ΔE/gap=390.746 is 8× the mean — median may be more representative N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 388.7826 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=5.153 is 7× the mean — median may be more representative N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.2738 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1208 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.3921 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5344 (possible stale e_exact or gap) N=40
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.8478 (possible stale e_exact or gap) N=60

## chain_1d — `unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20+60_p1_v4.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 2.0390e-01 | — | — | 0.1444 | D |
| 6 | IN | 8 | 4.0025e-02 | — | — | 0.4339 | F |
| 8 | IN | 8 | 5.3616e-02 | — | — | 9.2092 | F |
| 10 | IN | 8 | 2.6620e-02 | — | — | 39.1834 | F |
| 12 | IN | 8 | 7.4206e-02 | — | — | 154.7274 | F |
| 14 | IN | 8 | 8.2610e-02 | — | — | 3806.6317 | F |
| 15 | IN | 8 | 2.0189e-01 | — | — | 0.0154 | A |
| 16 | IN | 8 | 6.5399e-02 | — | — | 49.0765 | F |
| 20 | IN | 8 | 1.1158e-01 | — | — | 0.7112 | F |
| 26 | IN | 6 | 3.2276e-02 | — | — | 0.0154 | B |
| 30 | IN | 8 | 5.7665e-02 | — | — | 0.0331 | B |
| 40 | IN | 8 | 2.7299e-01 | — | — | 0.0424 | B |
| 60 | IN | 8 | 5.4843e-01 | — | — | 0.0594 | C |
| 80 | EXT | 8 | — | — | 8.85e-03 | 0.1658 | D |
| 100 | EXT | 19 | — | — | 7.98e-03 | 0.1369 | D |
| 150 | EXT | 3 | — | — | 3.58e-02 | 0.7841 | F |
| 200 | EXT | 3 | — | — | 3.59e-02 | 1.0472 | F |

> **Metric Warnings:**
> - ⚠️ Outlier: max ΔE/gap=0.814 is 6× the mean — median may be more representative N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5898 (possible stale e_exact or gap) N=4
> - ⚠️ Outlier: max ΔE/gap=3.353 is 8× the mean — median may be more representative N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 2.8794 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=72.951 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 71.8524 (possible stale e_exact or gap) N=8
> - ⚠️ Outlier: max ΔE/gap=312.317 is 8× the mean — median may be more representative N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 310.7901 (possible stale e_exact or gap) N=10
> - ⚠️ Outlier: max ΔE/gap=1235.945 is 8× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1233.9820 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=30381.980 is 8× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 30379.1988 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1271 (possible stale e_exact or gap) N=15
> - ⚠️ Outlier: max ΔE/gap=390.746 is 8× the mean — median may be more representative N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 388.7826 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=5.153 is 7× the mean — median may be more representative N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.2738 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1208 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.3921 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5344 (possible stale e_exact or gap) N=40
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.8478 (possible stale e_exact or gap) N=60

## chain_1d — `unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20+60_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 2.0390e-01 | — | — | 0.1444 | D |
| 6 | IN | 8 | 4.0025e-02 | — | — | 0.4339 | F |
| 8 | IN | 8 | 5.3616e-02 | — | — | 9.2092 | F |
| 10 | IN | 8 | 2.6620e-02 | — | — | 39.1834 | F |
| 12 | IN | 8 | 7.4206e-02 | — | — | 154.7274 | F |
| 14 | IN | 8 | 8.2610e-02 | — | — | 3806.6317 | F |
| 15 | IN | 8 | 2.0189e-01 | — | — | 0.0154 | A |
| 16 | IN | 8 | 6.5399e-02 | — | — | 49.0765 | F |
| 20 | IN | 8 | 1.1158e-01 | — | — | 0.7112 | F |
| 26 | IN | 6 | 3.2276e-02 | — | — | 0.0154 | B |
| 30 | IN | 8 | 5.7665e-02 | — | — | 0.0331 | B |
| 40 | IN | 8 | 2.7299e-01 | — | — | 0.0424 | B |
| 60 | IN | 8 | 5.4843e-01 | — | — | 0.0594 | C |
| 80 | EXT | 8 | — | — | 8.85e-03 | 0.1658 | D |
| 100 | EXT | 19 | — | — | 7.98e-03 | 0.1369 | D |
| 150 | EXT | 3 | — | — | 3.58e-02 | 0.7841 | F |
| 200 | EXT | 3 | — | — | 3.59e-02 | 1.0472 | F |

> **Metric Warnings:**
> - ⚠️ Outlier: max ΔE/gap=0.814 is 6× the mean — median may be more representative N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5898 (possible stale e_exact or gap) N=4
> - ⚠️ Outlier: max ΔE/gap=3.353 is 8× the mean — median may be more representative N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 2.8794 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=72.951 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 71.8524 (possible stale e_exact or gap) N=8
> - ⚠️ Outlier: max ΔE/gap=312.317 is 8× the mean — median may be more representative N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 310.7901 (possible stale e_exact or gap) N=10
> - ⚠️ Outlier: max ΔE/gap=1235.945 is 8× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1233.9820 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=30381.980 is 8× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 30379.1988 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1271 (possible stale e_exact or gap) N=15
> - ⚠️ Outlier: max ΔE/gap=390.746 is 8× the mean — median may be more representative N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 388.7826 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=5.153 is 7× the mean — median may be more representative N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.2738 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1208 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.3921 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5344 (possible stale e_exact or gap) N=40
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.8478 (possible stale e_exact or gap) N=60

## heavy_hex — `unified_tfim_br_heavy_hex_multiN_4+6+10+12+16+20+40_p1_v1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 7.9081e-01 | — | — | 0.0205 | A |
| 6 | IN | 8 | 5.2152e-01 | — | — | 0.0086 | A |
| 8 | IN | 8 | 8.5178e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 7.3458e-01 | — | — | 0.0286 | B |
| 12 | IN | 8 | 1.2015e+00 | — | — | 0.0571 | C |
| 14 | IN | 8 | 9.5015e-01 | — | — | 24.6514 | F |
| 16 | IN | 8 | 1.1274e+00 | — | — | 0.0452 | B |
| 18 | IN | 8 | 2.5321e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 3.1449e+00 | — | — | 5.3181 | F |
| 21 | IN | 8 | 1.4325e+00 | — | — | 0.0030 | A |
| 22 | IN | 8 | 1.4048e+00 | — | — | 0.0145 | A |
| 24 | IN | 8 | 1.2350e+00 | — | — | 0.0207 | A |
| 26 | IN | 5 | 1.5099e+00 | — | — | 0.0247 | B |
| 30 | IN | 8 | 2.1755e+00 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 2.1194e+00 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unified_tfim_br_heavy_hex_fromMT_4+6+10+12+16+20_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 5.4594e-01 | — | — | 0.0205 | A |
| 6 | IN | 8 | 3.1405e-01 | — | — | 0.0086 | A |
| 8 | IN | 8 | 1.0813e+00 | — | — | 478.3574 | F |
| 10 | IN | 8 | 1.1041e+00 | — | — | 0.0286 | B |
| 12 | IN | 8 | 3.8039e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 2.3707e+00 | — | — | 24.6514 | F |
| 16 | IN | 8 | 2.7094e+00 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.6077e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 2.0342e+00 | — | — | 5.3181 | F |
| 21 | IN | 8 | 6.2047e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 1.3749e+00 | — | — | 0.0145 | A |
| 24 | IN | 8 | 7.4073e-01 | — | — | 0.0207 | A |
| 26 | IN | 5 | 6.2846e-01 | — | — | 0.0247 | B |
| 30 | IN | 8 | 8.2041e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 3.2228e+00 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## square — `unified_tfim_br_square_multiN_4+6+8+10+12+14_p1_v4.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 5.1386e-03 | — | — | 0.0340 | B |
| 6 | IN | 8 | 6.7428e-02 | — | — | 0.0599 | C |
| 8 | IN | 8 | 1.4349e-01 | — | — | 0.0180 | A |
| 10 | IN | 8 | 1.1198e-01 | — | — | 0.0194 | A |
| 12 | IN | 8 | 1.0516e-01 | — | — | 0.3563 | F |
| 14 | IN | 8 | 1.2233e-01 | — | — | 0.0537 | C |
| 16 | IN | 8 | 3.6094e-02 | — | — | 0.0597 | C |
| 20 | IN | 8 | 4.0019e-02 | — | — | 13.6147 | F |
| 30 | EXT | 13 | — | — | 2.97e-02 | 2.2370 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0617 (possible stale e_exact or gap) N=4
> - ⚠️ Outlier: max ΔE/gap=0.360 is 6× the mean — median may be more representative N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0580 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0714 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1110 (possible stale e_exact or gap) N=10
> - ⚠️ Outlier: max ΔE/gap=2.587 is 7× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.5825 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1378 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.4329 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 15.3549 (possible stale e_exact or gap) N=20

## heavy_hex — `unified_tfim_br_heavy_hex_multiN_4+6+8+10+12+14+18+20+22+24+30+32+40_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 9.5069e-01 | — | — | 0.0205 | A |
| 6 | IN | 8 | 3.2443e-02 | — | — | 0.0086 | A |
| 8 | IN | 8 | 6.4991e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 5.1574e-02 | — | — | 0.0286 | B |
| 12 | IN | 8 | 2.5691e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 2.5024e-03 | — | — | 24.6514 | F |
| 16 | IN | 8 | 5.9099e-01 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.1822e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 1.2319e+00 | — | — | 5.3181 | F |
| 21 | IN | 8 | 4.5898e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 2.9034e-02 | — | — | 0.0145 | A |
| 24 | IN | 8 | 2.0050e-01 | — | — | 0.0207 | A |
| 26 | IN | 5 | 5.3981e-02 | — | — | 0.0247 | B |
| 30 | IN | 8 | 1.8248e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 7.0268e-02 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unified_tfim_br_heavy_hex_multiN_4+6+8+10+14+16+18+20+22+24+30+32+40_p2.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 5.4358e-01 | — | — | 0.0205 | A |
| 6 | IN | 8 | 1.5616e-01 | — | — | 0.0086 | A |
| 8 | IN | 8 | 4.0131e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 2.3990e-01 | — | — | 0.0286 | B |
| 12 | IN | 8 | 4.0024e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 2.0538e-01 | — | — | 24.6514 | F |
| 16 | IN | 8 | 5.2769e-01 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.2431e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 1.2340e+00 | — | — | 5.3181 | F |
| 21 | IN | 8 | 4.3560e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 1.6851e-01 | — | — | 0.0145 | A |
| 24 | IN | 8 | 9.7112e-02 | — | — | 0.0207 | A |
| 26 | IN | 5 | 2.2379e-02 | — | — | 0.0247 | B |
| 30 | IN | 8 | 2.1484e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 2.0074e-01 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unified_tfim_br_heavy_hex_multiN_4+6+8+10+14+16+18+20+22+24+30+32+40_p2_v2.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 9.4555e-01 | — | — | 0.0205 | A |
| 6 | IN | 8 | 2.4055e-02 | — | — | 0.0086 | A |
| 8 | IN | 8 | 8.0120e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 4.5052e-02 | — | — | 0.0286 | B |
| 12 | IN | 8 | 2.6282e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 2.5771e-01 | — | — | 24.6514 | F |
| 16 | IN | 8 | 1.2791e+00 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.1732e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 9.2495e-01 | — | — | 5.3181 | F |
| 21 | IN | 8 | 5.2900e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 2.6606e-02 | — | — | 0.0145 | A |
| 24 | IN | 8 | 2.3988e-01 | — | — | 0.0207 | A |
| 26 | IN | 5 | 8.7960e-02 | — | — | 0.0247 | B |
| 30 | IN | 8 | 1.9297e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 6.0977e-02 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unified_tfim_br_heavy_hex_multiN_4+6+8+10+14+16+18+20+22+24+30+32+40_p2_v3.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 9.6221e-01 | — | — | 0.0205 | A |
| 6 | IN | 8 | 2.6880e-02 | — | — | 0.0086 | A |
| 8 | IN | 8 | 8.0242e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 4.3111e-02 | — | — | 0.0286 | B |
| 12 | IN | 8 | 2.4538e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 3.0742e-03 | — | — | 24.6514 | F |
| 16 | IN | 8 | 1.2440e+00 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.2030e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 9.6317e-01 | — | — | 5.3181 | F |
| 21 | IN | 8 | 5.6297e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 2.7004e-02 | — | — | 0.0145 | A |
| 24 | IN | 8 | 2.6218e-01 | — | — | 0.0207 | A |
| 26 | IN | 5 | 8.7437e-02 | — | — | 0.0247 | B |
| 30 | IN | 8 | 1.9469e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 5.1851e-02 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unified_tfim_br_heavy_hex_multiN_4+6+8+10+14+16+18+20+22+24+30+32+40_p2_v4.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 9.5941e-01 | — | — | 0.0205 | A |
| 6 | IN | 8 | 4.9899e-02 | — | — | 0.0086 | A |
| 8 | IN | 8 | 8.3856e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 4.2392e-02 | — | — | 0.0286 | B |
| 12 | IN | 8 | 2.6340e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 3.8595e-03 | — | — | 24.6514 | F |
| 16 | IN | 8 | 1.2774e+00 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.2335e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 1.0190e+00 | — | — | 5.3181 | F |
| 21 | IN | 8 | 5.2111e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 3.7560e-02 | — | — | 0.0145 | A |
| 24 | IN | 8 | 2.4184e-01 | — | — | 0.0207 | A |
| 26 | IN | 5 | 7.3511e-02 | — | — | 0.0247 | B |
| 30 | IN | 8 | 1.8184e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 5.5682e-02 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unified_tfim_br_heavy_hex_multiN_4+6+8+10+14+16+18+20+22+24+30+32+40_p2_v5.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 3.5222e-01 | — | — | 0.0205 | A |
| 6 | IN | 8 | 1.5735e-01 | — | — | 0.0086 | A |
| 8 | IN | 8 | 2.8403e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 1.8757e-01 | — | — | 0.0286 | B |
| 12 | IN | 8 | 2.4695e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 1.9162e-01 | — | — | 24.6514 | F |
| 16 | IN | 8 | 5.7504e-01 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.0783e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 1.0562e+00 | — | — | 5.3181 | F |
| 21 | IN | 8 | 3.8629e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 1.7025e-01 | — | — | 0.0145 | A |
| 24 | IN | 8 | 1.0206e-01 | — | — | 0.0207 | A |
| 26 | IN | 5 | 1.4715e-02 | — | — | 0.0247 | B |
| 30 | IN | 8 | 1.6208e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 3.9903e-01 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unifMPNN__heavy_hex_p1_res_mse.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 2.9976e-01 | — | — | 0.0205 | A |
| 6 | IN | 8 | 1.6224e-01 | — | — | 0.0086 | A |
| 8 | IN | 8 | 2.7137e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 1.9086e-01 | — | — | 0.0286 | B |
| 12 | IN | 8 | 2.2609e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 2.0255e-01 | — | — | 24.6514 | F |
| 16 | IN | 8 | 6.0542e-01 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.0604e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 1.0277e+00 | — | — | 5.3181 | F |
| 21 | IN | 8 | 3.9349e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 1.9120e-01 | — | — | 0.0145 | A |
| 24 | IN | 8 | 1.2285e-01 | — | — | 0.0207 | A |
| 26 | IN | 5 | 3.5182e-02 | — | — | 0.0247 | B |
| 30 | IN | 8 | 1.6923e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 4.9105e-01 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unifMPNN__heavy_hex_p1_res_film_mse.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 3.0974e-01 | — | — | 0.0205 | A |
| 6 | IN | 8 | 1.6281e-01 | — | — | 0.0086 | A |
| 8 | IN | 8 | 2.7752e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 1.9076e-01 | — | — | 0.0286 | B |
| 12 | IN | 8 | 2.2663e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 1.9573e-01 | — | — | 24.6514 | F |
| 16 | IN | 8 | 6.1339e-01 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.0560e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 1.0249e+00 | — | — | 5.3181 | F |
| 21 | IN | 8 | 3.9290e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 1.9000e-01 | — | — | 0.0145 | A |
| 24 | IN | 8 | 1.2180e-01 | — | — | 0.0207 | A |
| 26 | IN | 5 | 3.4221e-02 | — | — | 0.0247 | B |
| 30 | IN | 8 | 1.6842e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 4.8565e-01 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## chain_1d — `unifMPNN__chain_1d_p1_h_0p5_1p5.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 2.1148e-01 | — | — | 0.1444 | D |
| 6 | IN | 8 | 4.2540e-02 | — | — | 0.4339 | F |
| 8 | IN | 8 | 5.0934e-02 | — | — | 9.2092 | F |
| 10 | IN | 8 | 1.1774e-02 | — | — | 39.1834 | F |
| 12 | IN | 8 | 2.4962e-02 | — | — | 154.7274 | F |
| 14 | IN | 8 | 1.5538e-02 | — | — | 3806.6317 | F |
| 15 | IN | 8 | 1.3606e-01 | — | — | 0.0154 | A |
| 16 | IN | 8 | 1.1363e-02 | — | — | 49.0765 | F |
| 20 | IN | 8 | 3.3934e-02 | — | — | 0.7112 | F |
| 26 | IN | 6 | 7.4185e-03 | — | — | 0.0154 | B |
| 30 | IN | 8 | 7.5837e-03 | — | — | 0.0331 | B |
| 40 | IN | 8 | 8.2430e-03 | — | — | 0.0424 | B |
| 60 | IN | 8 | 8.3018e-03 | — | — | 0.0594 | C |
| 80 | EXT | 8 | — | — | 8.85e-03 | 0.1658 | D |
| 100 | EXT | 19 | — | — | 7.98e-03 | 0.1369 | D |
| 150 | EXT | 3 | — | — | 3.58e-02 | 0.7841 | F |
| 200 | EXT | 3 | — | — | 3.59e-02 | 1.0472 | F |

> **Metric Warnings:**
> - ⚠️ Outlier: max ΔE/gap=0.814 is 6× the mean — median may be more representative N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5898 (possible stale e_exact or gap) N=4
> - ⚠️ Outlier: max ΔE/gap=3.353 is 8× the mean — median may be more representative N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 2.8794 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=72.951 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 71.8524 (possible stale e_exact or gap) N=8
> - ⚠️ Outlier: max ΔE/gap=312.317 is 8× the mean — median may be more representative N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 310.7901 (possible stale e_exact or gap) N=10
> - ⚠️ Outlier: max ΔE/gap=1235.945 is 8× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1233.9820 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=30381.980 is 8× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 30379.1988 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1271 (possible stale e_exact or gap) N=15
> - ⚠️ Outlier: max ΔE/gap=390.746 is 8× the mean — median may be more representative N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 388.7826 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=5.153 is 7× the mean — median may be more representative N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.2738 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1208 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.3921 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5344 (possible stale e_exact or gap) N=40
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.8478 (possible stale e_exact or gap) N=60

## chain_1d — `unifMPNN__chain_1d_p1_h_0p5_1p5_v2.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 2.3961e-01 | — | — | 0.1444 | D |
| 6 | IN | 8 | 6.5744e-02 | — | — | 0.4339 | F |
| 8 | IN | 8 | 5.9434e-02 | — | — | 9.2092 | F |
| 10 | IN | 8 | 3.5540e-02 | — | — | 39.1834 | F |
| 12 | IN | 8 | 4.6056e-02 | — | — | 154.7274 | F |
| 14 | IN | 8 | 2.2666e-02 | — | — | 3806.6317 | F |
| 15 | IN | 8 | 1.8320e-01 | — | — | 0.0154 | A |
| 16 | IN | 8 | 4.6039e-02 | — | — | 49.0765 | F |
| 20 | IN | 8 | 9.2891e-02 | — | — | 0.7112 | F |
| 26 | IN | 6 | 1.1017e-01 | — | — | 0.0154 | B |
| 30 | IN | 8 | 8.8713e-02 | — | — | 0.0331 | B |
| 40 | IN | 8 | 1.0657e-01 | — | — | 0.0424 | B |
| 60 | IN | 8 | 1.2237e-01 | — | — | 0.0594 | C |
| 80 | EXT | 8 | — | — | 8.85e-03 | 0.1658 | D |
| 100 | EXT | 19 | — | — | 7.98e-03 | 0.1369 | D |
| 150 | EXT | 3 | — | — | 3.58e-02 | 0.7841 | F |
| 200 | EXT | 3 | — | — | 3.59e-02 | 1.0472 | F |

> **Metric Warnings:**
> - ⚠️ Outlier: max ΔE/gap=0.814 is 6× the mean — median may be more representative N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5898 (possible stale e_exact or gap) N=4
> - ⚠️ Outlier: max ΔE/gap=3.353 is 8× the mean — median may be more representative N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 2.8794 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=72.951 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 71.8524 (possible stale e_exact or gap) N=8
> - ⚠️ Outlier: max ΔE/gap=312.317 is 8× the mean — median may be more representative N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 310.7901 (possible stale e_exact or gap) N=10
> - ⚠️ Outlier: max ΔE/gap=1235.945 is 8× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1233.9820 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=30381.980 is 8× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 30379.1988 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1271 (possible stale e_exact or gap) N=15
> - ⚠️ Outlier: max ΔE/gap=390.746 is 8× the mean — median may be more representative N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 388.7826 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=5.153 is 7× the mean — median may be more representative N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.2738 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1208 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.3921 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5344 (possible stale e_exact or gap) N=40
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.8478 (possible stale e_exact or gap) N=60

## heavy_hex — `unified_tfim_br_heavy_hex_fromMT_4+6+8+10+12+14+18+20+21+26+30+40_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 2.4768e+00 | — | — | 0.0205 | A |
| 6 | IN | 8 | 1.2076e+00 | — | — | 0.0086 | A |
| 8 | IN | 8 | 3.9972e+00 | — | — | 478.3574 | F |
| 10 | IN | 8 | 4.6485e+00 | — | — | 0.0286 | B |
| 12 | IN | 8 | 2.8572e+00 | — | — | 0.0571 | C |
| 14 | IN | 8 | 4.8697e+00 | — | — | 24.6514 | F |
| 16 | IN | 8 | 4.9867e+00 | — | — | 0.0452 | B |
| 18 | IN | 8 | 4.6711e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 7.6569e+00 | — | — | 5.3181 | F |
| 21 | IN | 8 | 4.7187e+00 | — | — | 0.0030 | A |
| 22 | IN | 8 | 7.7837e+00 | — | — | 0.0145 | A |
| 24 | IN | 8 | 4.5681e+00 | — | — | 0.0207 | A |
| 26 | IN | 5 | 1.2388e+01 | — | — | 0.0247 | B |
| 30 | IN | 8 | 7.4704e+00 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 1.2393e+01 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unified_multiN_heavyhex_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 5.5722e-01 | — | — | 0.0205 | A |
| 6 | IN | 8 | 1.3359e-01 | — | — | 0.0086 | A |
| 8 | IN | 8 | 4.7228e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 3.3005e-01 | — | — | 0.0286 | B |
| 12 | IN | 8 | 4.9396e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 6.3222e-01 | — | — | 24.6514 | F |
| 16 | IN | 8 | 9.0727e-01 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.4130e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 1.4472e+00 | — | — | 5.3181 | F |
| 21 | IN | 8 | 8.6490e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 9.5031e-01 | — | — | 0.0145 | A |
| 24 | IN | 8 | 1.4335e+00 | — | — | 0.0207 | A |
| 26 | IN | 5 | 1.6953e+00 | — | — | 0.0247 | B |
| 30 | IN | 8 | 3.2713e+00 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 1.8466e+01 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## ladder — `unified_tfim_br_ladder_fromMT_4+6+8+10+12+14+20+26+30_p1.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 5.7197e-02 | — | — | 0.0593 | C |
| 6 | IN | 8 | 7.0391e-02 | — | — | 0.0196 | A |
| 8 | IN | 8 | 1.1768e-01 | — | — | 0.0666 | C |
| 10 | IN | 8 | 4.7882e-02 | — | — | 0.0333 | B |
| 12 | IN | 8 | 1.6533e-02 | — | — | 0.0319 | B |
| 14 | IN | 8 | 9.2352e-02 | — | — | 0.1866 | D |
| 16 | IN | 8 | 3.1760e-02 | — | — | 0.0357 | B |
| 20 | IN | 8 | 7.2421e-03 | — | — | 0.2928 | F |
| 26 | IN | 8 | 4.9908e-03 | — | — | 0.6368 | F |
| 30 | IN | 8 | 4.6824e-03 | — | — | 0.4806 | F |
| 40 | IN | 2 | 7.8832e-03 | — | — | 0.0465 | D |

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

## ladder — `unified_tfim_br_ladder_multiN_4+6+8+10+12+14+20+26+30_p1_v4.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 8.4446e-03 | — | — | 0.0593 | C |
| 6 | IN | 8 | 3.1543e-02 | — | — | 0.0196 | A |
| 8 | IN | 8 | 4.3556e-02 | — | — | 0.0666 | C |
| 10 | IN | 8 | 3.6837e-02 | — | — | 0.0333 | B |
| 12 | IN | 8 | 2.9067e-02 | — | — | 0.0319 | B |
| 14 | IN | 8 | 1.3050e-01 | — | — | 0.1866 | D |
| 16 | IN | 8 | 7.0342e-02 | — | — | 0.0357 | B |
| 20 | IN | 8 | 7.4781e-02 | — | — | 0.2928 | F |
| 26 | IN | 8 | 1.1893e-01 | — | — | 0.6368 | F |
| 30 | IN | 8 | 1.3722e-01 | — | — | 0.4806 | F |
| 40 | IN | 2 | 1.9940e-01 | — | — | 0.0465 | D |

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

## heavy_hex — `unifMPNN__heavy_hex_p1_res_mse_v2.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 9.5749e-01 | — | — | 0.0205 | A |
| 6 | IN | 8 | 3.5612e-02 | — | — | 0.0086 | A |
| 8 | IN | 8 | 7.3282e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 4.4597e-02 | — | — | 0.0286 | B |
| 12 | IN | 8 | 2.9094e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 2.4864e-01 | — | — | 24.6514 | F |
| 16 | IN | 8 | 1.0965e+00 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.2109e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 1.0626e+00 | — | — | 5.3181 | F |
| 21 | IN | 8 | 4.7902e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 1.8530e-02 | — | — | 0.0145 | A |
| 24 | IN | 8 | 1.8885e-01 | — | — | 0.0207 | A |
| 26 | IN | 5 | 6.3282e-02 | — | — | 0.0247 | B |
| 30 | IN | 8 | 1.8249e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 7.3921e-02 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unified_tfim_br_heavy_hex_multiN_4+6+8+10+14+16+18+20+22+24+30+32+40_p2_v6.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 1.1694e+00 | — | — | 0.0205 | A |
| 6 | IN | 8 | 2.4593e-01 | — | — | 0.0086 | A |
| 8 | IN | 8 | 5.5115e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 3.6461e-01 | — | — | 0.0286 | B |
| 12 | IN | 8 | 4.3650e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 2.4924e-01 | — | — | 24.6514 | F |
| 16 | IN | 8 | 5.3777e-01 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.2803e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 1.5125e+00 | — | — | 5.3181 | F |
| 21 | IN | 8 | 4.3358e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 2.2000e-01 | — | — | 0.0145 | A |
| 24 | IN | 8 | 1.0857e-01 | — | — | 0.0207 | A |
| 26 | IN | 5 | 1.3737e-01 | — | — | 0.0247 | B |
| 30 | IN | 8 | 2.0983e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 1.1481e-01 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unifMPNN__heavy_hex_p1_res_film_mse_v2.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 1.0266e+00 | — | — | 0.0205 | A |
| 6 | IN | 8 | 1.7971e-02 | — | — | 0.0086 | A |
| 8 | IN | 8 | 7.4669e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 5.6466e-02 | — | — | 0.0286 | B |
| 12 | IN | 8 | 2.6882e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 2.2181e-02 | — | — | 24.6514 | F |
| 16 | IN | 8 | 1.0699e+00 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.0397e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 9.9106e-01 | — | — | 5.3181 | F |
| 21 | IN | 8 | 6.1480e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 9.6693e-02 | — | — | 0.0145 | A |
| 24 | IN | 8 | 2.4897e-01 | — | — | 0.0207 | A |
| 26 | IN | 5 | 3.0197e-02 | — | — | 0.0247 | B |
| 30 | IN | 8 | 1.7204e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 6.5130e-02 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unified_tfim_br_heavy_hex_multiN_4+6+8+10+14+16+18+20+22+24+30+32+40_p2_v7.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 9.3932e-01 | — | — | 0.0205 | A |
| 6 | IN | 8 | 2.6494e-02 | — | — | 0.0086 | A |
| 8 | IN | 8 | 7.7406e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 4.5742e-02 | — | — | 0.0286 | B |
| 12 | IN | 8 | 2.8959e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 2.5853e-03 | — | — | 24.6514 | F |
| 16 | IN | 8 | 1.2438e+00 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.1652e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 9.7648e-01 | — | — | 5.3181 | F |
| 21 | IN | 8 | 5.1343e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 4.9090e-02 | — | — | 0.0145 | A |
| 24 | IN | 8 | 2.2419e-01 | — | — | 0.0207 | A |
| 26 | IN | 5 | 6.2854e-02 | — | — | 0.0247 | B |
| 30 | IN | 8 | 1.8268e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 5.7653e-02 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unifMPNN__heavy_hex_p1_plain_energyw.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 8.8001e-01 | — | — | 0.0205 | A |
| 6 | IN | 8 | 2.8699e-02 | — | — | 0.0086 | A |
| 8 | IN | 8 | 6.1294e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 4.7524e-02 | — | — | 0.0286 | B |
| 12 | IN | 8 | 2.9679e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 4.2560e-01 | — | — | 24.6514 | F |
| 16 | IN | 8 | 1.0943e+00 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.1396e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 8.3812e-01 | — | — | 5.3181 | F |
| 21 | IN | 8 | 4.0501e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 9.8767e-02 | — | — | 0.0145 | A |
| 24 | IN | 8 | 2.1724e-01 | — | — | 0.0207 | A |
| 26 | IN | 5 | 1.7358e-01 | — | — | 0.0247 | B |
| 30 | IN | 8 | 2.2122e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 9.8842e-02 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unifMPNN__heavy_hex_p1_res_energyw.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 1.0377e+00 | — | — | 0.0205 | A |
| 6 | IN | 8 | 3.2063e-01 | — | — | 0.0086 | A |
| 8 | IN | 8 | 5.6055e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 3.6088e-01 | — | — | 0.0286 | B |
| 12 | IN | 8 | 5.9901e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 2.8149e-01 | — | — | 24.6514 | F |
| 16 | IN | 8 | 5.6049e-01 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.4368e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 1.5117e+00 | — | — | 5.3181 | F |
| 21 | IN | 8 | 5.7396e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 2.5301e-01 | — | — | 0.0145 | A |
| 24 | IN | 8 | 1.7533e-01 | — | — | 0.0207 | A |
| 26 | IN | 5 | 1.6161e-01 | — | — | 0.0247 | B |
| 30 | IN | 8 | 3.4796e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 1.1637e-01 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unifMPNN__heavy_hex_p1_res_film_energyw.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 8.0989e-01 | — | — | 0.0205 | A |
| 6 | IN | 8 | 5.3433e-01 | — | — | 0.0086 | A |
| 8 | IN | 8 | 5.1868e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 4.1763e-01 | — | — | 0.0286 | B |
| 12 | IN | 8 | 7.6775e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 3.1579e-01 | — | — | 24.6514 | F |
| 16 | IN | 8 | 5.3220e-01 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.5038e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 1.3586e+00 | — | — | 5.3181 | F |
| 21 | IN | 8 | 9.3225e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 4.0978e-01 | — | — | 0.0145 | A |
| 24 | IN | 8 | 3.6345e-01 | — | — | 0.0207 | A |
| 26 | IN | 5 | 4.4426e-01 | — | — | 0.0247 | B |
| 30 | IN | 8 | 5.4148e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 1.4135e-01 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unifMPNN__heavy_hex_p1_res_physics05.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 4.0546e-01 | — | — | 0.0205 | A |
| 6 | IN | 8 | 1.5727e-01 | — | — | 0.0086 | A |
| 8 | IN | 8 | 2.9458e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 1.7882e-01 | — | — | 0.0286 | B |
| 12 | IN | 8 | 2.6348e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 1.5705e-01 | — | — | 24.6514 | F |
| 16 | IN | 8 | 5.5425e-01 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.1257e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 1.1209e+00 | — | — | 5.3181 | F |
| 21 | IN | 8 | 4.0410e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 1.2193e-01 | — | — | 0.0145 | A |
| 24 | IN | 8 | 8.3542e-02 | — | — | 0.0207 | A |
| 26 | IN | 5 | 1.4522e-02 | — | — | 0.0247 | B |
| 30 | IN | 8 | 2.0579e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 1.2749e-01 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unifMPNN__heavy_hex_p1_full_stack.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 7.3262e-01 | — | — | 0.0205 | A |
| 6 | IN | 8 | 1.8553e-01 | — | — | 0.0086 | A |
| 8 | IN | 8 | 5.0441e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 2.9980e-01 | — | — | 0.0286 | B |
| 12 | IN | 8 | 3.8548e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 3.0352e-01 | — | — | 24.6514 | F |
| 16 | IN | 8 | 5.1417e-01 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.2065e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 1.1019e+00 | — | — | 5.3181 | F |
| 21 | IN | 8 | 6.5132e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 1.8851e-01 | — | — | 0.0145 | A |
| 24 | IN | 8 | 1.6519e-01 | — | — | 0.0207 | A |
| 26 | IN | 5 | 2.1005e-01 | — | — | 0.0247 | B |
| 30 | IN | 8 | 3.2505e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 1.8549e-01 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unifMPNN__heavy_hex_p1_res_film_energyw_critical.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 4.1812e-01 | — | — | 0.0205 | A |
| 6 | IN | 8 | 2.8712e-01 | — | — | 0.0086 | A |
| 8 | IN | 8 | 3.6899e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 1.8633e-01 | — | — | 0.0286 | B |
| 12 | IN | 8 | 4.3566e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 1.9317e-01 | — | — | 24.6514 | F |
| 16 | IN | 8 | 4.5474e-01 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.2467e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 1.2749e+00 | — | — | 5.3181 | F |
| 21 | IN | 8 | 4.3971e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 1.6798e-01 | — | — | 0.0145 | A |
| 24 | IN | 8 | 1.0780e-01 | — | — | 0.0207 | A |
| 26 | IN | 5 | 1.2856e-01 | — | — | 0.0247 | B |
| 30 | IN | 8 | 2.3974e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 1.1553e-01 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## heavy_hex — `unifMPNN__heavy_hex_p1_res_film_deploy.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 1.0028e+00 | — | — | 0.0205 | A |
| 6 | IN | 8 | 2.2866e-02 | — | — | 0.0086 | A |
| 8 | IN | 8 | 7.1988e-01 | — | — | 478.3574 | F |
| 10 | IN | 8 | 4.8030e-02 | — | — | 0.0286 | B |
| 12 | IN | 8 | 2.1618e-01 | — | — | 0.0571 | C |
| 14 | IN | 8 | 3.7640e-02 | — | — | 24.6514 | F |
| 16 | IN | 8 | 1.2064e+00 | — | — | 0.0452 | B |
| 18 | IN | 8 | 1.1100e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 8.8609e-01 | — | — | 5.3181 | F |
| 21 | IN | 8 | 4.8088e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 5.2284e-02 | — | — | 0.0145 | A |
| 24 | IN | 8 | 1.9903e-01 | — | — | 0.0207 | A |
| 26 | IN | 5 | 5.3178e-02 | — | — | 0.0247 | B |
| 30 | IN | 8 | 1.8487e-01 | — | — | 0.2108 | D |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 6 | 5.7554e-02 | — | — | 0.0390 | C |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | EXT | 6 | — | — | 2.92e-02 | 0.4976 | F |

> **Metric Warnings:**
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0158 (possible stale e_exact or gap) N=21
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0236 (possible stale e_exact or gap) N=22
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0332 (possible stale e_exact or gap) N=24
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1205 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40

## multi_topology — `unifMPNN__MT_p1_res_film_base.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 3 | IN | 8 | 1.5229e-02 | — | — | 0.0012 | A |
| 4 | IN | 8 | 1.7277e-01 | — | — | 0.1444 | D |
| 4 | IN | 8 | 7.0319e-01 | — | — | 0.0205 | A |
| 4 | IN | 8 | 6.9455e-03 | — | — | 0.0593 | C |
| 4 | IN | 8 | 8.2979e-03 | — | — | 0.0340 | B |
| 4 | IN | 8 | 6.5389e-03 | — | — | 0.1682 | D |
| 6 | IN | 8 | 3.8473e-02 | — | — | 0.4339 | F |
| 6 | IN | 8 | 1.4145e-01 | — | — | 0.0086 | A |
| 6 | IN | 8 | 3.5911e-02 | — | — | 0.0196 | A |
| 6 | IN | 8 | 4.9769e-02 | — | — | 0.0599 | C |
| 6 | IN | 8 | 2.0576e-03 | — | — | 0.0304 | B |
| 8 | IN | 8 | 5.1127e-02 | — | — | 9.2092 | F |
| 8 | IN | 8 | 3.5947e-01 | — | — | 478.3574 | F |
| 8 | IN | 8 | 4.2420e-02 | — | — | 0.0666 | C |
| 8 | IN | 8 | 9.8171e-02 | — | — | 0.0180 | A |
| 8 | IN | 8 | 1.8812e-02 | — | — | 3.0954 | F |
| 10 | IN | 8 | 1.8884e-02 | — | — | 39.1834 | F |
| 10 | IN | 8 | 1.7675e-01 | — | — | 0.0286 | B |
| 10 | IN | 8 | 2.8990e-02 | — | — | 0.0333 | B |
| 10 | IN | 8 | 7.4338e-02 | — | — | 0.0194 | A |
| 10 | IN | 8 | 8.6936e-02 | — | — | 0.7619 | F |
| 12 | IN | 8 | 3.5308e-02 | — | — | 154.7274 | F |
| 12 | IN | 8 | 2.8193e-01 | — | — | 0.0571 | C |
| 12 | IN | 8 | 1.1957e-02 | — | — | 0.0319 | B |
| 12 | IN | 8 | 6.2793e-02 | — | — | 0.3563 | F |
| 12 | IN | 8 | 2.6329e-02 | — | — | 1.0070 | F |
| 14 | IN | 8 | 3.3819e-02 | — | — | 3806.6317 | F |
| 14 | IN | 8 | 1.5492e-01 | — | — | 24.6514 | F |
| 14 | IN | 8 | 6.6818e-02 | — | — | 0.1866 | D |
| 14 | IN | 8 | 5.6263e-02 | — | — | 0.0537 | C |
| 15 | IN | 8 | 1.2629e-01 | — | — | 0.0154 | A |
| 16 | IN | 8 | 2.5591e-02 | — | — | 49.0765 | F |
| 16 | IN | 8 | 6.2021e-01 | — | — | 0.0452 | B |
| 16 | IN | 8 | 2.9890e-02 | — | — | 0.0357 | B |
| 16 | IN | 8 | 2.6851e-03 | — | — | 0.0597 | C |
| 18 | IN | 8 | 1.0950e+00 | — | — | 20.4816 | F |
| 20 | IN | 8 | 4.5600e-02 | — | — | 0.7112 | F |
| 20 | IN | 8 | 1.0443e+00 | — | — | 5.3181 | F |
| 20 | IN | 8 | 7.0701e-03 | — | — | 0.2928 | F |
| 20 | IN | 8 | 1.6336e-02 | — | — | 13.6147 | F |
| 21 | IN | 8 | 3.9486e-01 | — | — | 0.0030 | A |
| 22 | IN | 8 | 1.5928e-01 | — | — | 0.0145 | A |
| 24 | IN | 8 | 9.3759e-02 | — | — | 0.0207 | A |
| 26 | IN | 6 | 9.0475e-03 | — | — | 0.0154 | B |
| 26 | IN | 5 | 1.5710e-02 | — | — | 0.0247 | B |
| 26 | IN | 8 | 1.1163e-02 | — | — | 0.6368 | F |
| 30 | IN | 8 | 1.2523e-02 | — | — | 0.0331 | B |
| 30 | IN | 8 | 1.6223e-01 | — | — | 0.2108 | D |
| 30 | IN | 8 | 1.3925e-02 | — | — | 0.4806 | F |
| 32 | EXT | 10 | — | — | 2.02e-02 | 0.3840 | F |
| 40 | IN | 8 | 1.6497e-02 | — | — | 0.0424 | B |
| 40 | IN | 6 | 4.3330e-01 | — | — | 0.0390 | C |
| 40 | IN | 2 | 9.5081e-03 | — | — | 0.0465 | D |
| 50 | EXT | 6 | — | — | 1.84e-02 | 0.2856 | F |
| 60 | IN | 8 | 1.7260e-02 | — | — | 0.0594 | C |
| 80 | EXT | 8 | — | — | 8.85e-03 | 0.1658 | D |
| 100 | EXT | 19 | — | — | 7.98e-03 | 0.1369 | D |
| 150 | EXT | 3 | — | — | 3.58e-02 | 0.7841 | F |
| 200 | EXT | 3 | — | — | 3.59e-02 | 1.0472 | F |

> **Metric Warnings:**
> - ⚠️ Outlier: max ΔE/gap=0.007 is 6× the mean — median may be more representative N=3
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0149 (possible stale e_exact or gap) N=3
> - ⚠️ Outlier: max ΔE/gap=0.814 is 6× the mean — median may be more representative N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5898 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0480 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0897 (possible stale e_exact or gap) N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0617 (possible stale e_exact or gap) N=4
> - ⚠️ Outlier: max ΔE/gap=1.087 is 6× the mean — median may be more representative N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.7688 (possible stale e_exact or gap) N=4
> - ⚠️ Outlier: max ΔE/gap=3.353 is 8× the mean — median may be more representative N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 2.8794 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0548 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1160 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=0.360 is 6× the mean — median may be more representative N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0580 (possible stale e_exact or gap) N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1024 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=72.951 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 71.8524 (possible stale e_exact or gap) N=8
> - ⚠️ Outlier: max ΔE/gap=3825.547 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3824.1708 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0857 (possible stale e_exact or gap) N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0714 (possible stale e_exact or gap) N=8
> - ⚠️ Outlier: max ΔE/gap=23.500 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 21.1681 (possible stale e_exact or gap) N=8
> - ⚠️ Outlier: max ΔE/gap=312.317 is 8× the mean — median may be more representative N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 310.7901 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0924 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1866 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1110 (possible stale e_exact or gap) N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 2.3036 (possible stale e_exact or gap) N=10
> - ⚠️ Outlier: max ΔE/gap=1235.945 is 8× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1233.9820 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0346 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1990 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=2.587 is 7× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.5825 (possible stale e_exact or gap) N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.2810 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=30381.980 is 8× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 30379.1988 (possible stale e_exact or gap) N=14
> - ⚠️ Outlier: max ΔE/gap=170.795 is 7× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 169.3679 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1447 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1378 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1271 (possible stale e_exact or gap) N=15
> - ⚠️ Outlier: max ΔE/gap=390.746 is 8× the mean — median may be more representative N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 388.7826 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0631 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1400 (possible stale e_exact or gap) N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.4329 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=139.153 is 7× the mean — median may be more representative N=18
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 137.3449 (possible stale e_exact or gap) N=18
> - ⚠️ Outlier: max ΔE/gap=5.153 is 7× the mean — median may be more representative N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.2738 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 14.0321 (possible stale e_exact or gap) N=20
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
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1.0355 (possible stale e_exact or gap) N=30
> - ⚠️ Outlier: max ΔE/gap=3.509 is 7× the mean — median may be more representative N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 2.7743 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5344 (possible stale e_exact or gap) N=40
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.0448 (possible stale e_exact or gap) N=40
> - ⚠️ Only 2 points — means have low statistical confidence N=40
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1298 (possible stale e_exact or gap) N=40
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.8478 (possible stale e_exact or gap) N=60

## chain_1d — `unified_tfim_br_chain_1d_multiN_4_p2.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 2.1002e-01 | — | — | 0.1444 | D |
| 6 | IN | 8 | 6.9065e-02 | — | — | 0.4339 | F |
| 8 | IN | 8 | 7.2095e-02 | — | — | 9.2092 | F |
| 10 | IN | 8 | 6.0121e-02 | — | — | 39.1834 | F |
| 12 | IN | 8 | 7.7790e-02 | — | — | 154.7274 | F |
| 14 | IN | 8 | 3.9187e-02 | — | — | 3806.6317 | F |
| 15 | IN | 8 | 2.0048e-01 | — | — | 0.0154 | A |
| 16 | IN | 8 | 7.0097e-02 | — | — | 49.0765 | F |
| 20 | IN | 8 | 1.2421e-01 | — | — | 0.7112 | F |
| 26 | IN | 6 | 1.9773e-01 | — | — | 0.0154 | B |
| 30 | IN | 8 | 1.9785e-01 | — | — | 0.0331 | B |
| 40 | IN | 8 | 2.8968e-01 | — | — | 0.0424 | B |
| 60 | IN | 8 | 5.0152e-01 | — | — | 0.0594 | C |
| 80 | EXT | 8 | — | — | 8.85e-03 | 0.1658 | D |
| 100 | EXT | 19 | — | — | 7.98e-03 | 0.1369 | D |
| 150 | EXT | 3 | — | — | 3.58e-02 | 0.7841 | F |
| 200 | EXT | 3 | — | — | 3.59e-02 | 1.0472 | F |

> **Metric Warnings:**
> - ⚠️ Outlier: max ΔE/gap=0.814 is 6× the mean — median may be more representative N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5898 (possible stale e_exact or gap) N=4
> - ⚠️ Outlier: max ΔE/gap=3.353 is 8× the mean — median may be more representative N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 2.8794 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=72.951 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 71.8524 (possible stale e_exact or gap) N=8
> - ⚠️ Outlier: max ΔE/gap=312.317 is 8× the mean — median may be more representative N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 310.7901 (possible stale e_exact or gap) N=10
> - ⚠️ Outlier: max ΔE/gap=1235.945 is 8× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1233.9820 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=30381.980 is 8× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 30379.1988 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1271 (possible stale e_exact or gap) N=15
> - ⚠️ Outlier: max ΔE/gap=390.746 is 8× the mean — median may be more representative N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 388.7826 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=5.153 is 7× the mean — median may be more representative N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.2738 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1208 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.3921 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5344 (possible stale e_exact or gap) N=40
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.8478 (possible stale e_exact or gap) N=60

## chain_1d — `unifMPNN__chain_1d_p1_h_0p5_1p5_v3.pt`

| N | Source | Pts | θ MSE | |ΔE| | |ΔE|/N | ΔE/gap | Grade |
|---|--------|-----|-------|------|--------|--------|-------|
| 4 | IN | 8 | 3.8665e-01 | — | — | 0.1444 | D |
| 6 | IN | 8 | 6.1737e-02 | — | — | 0.4339 | F |
| 8 | IN | 8 | 4.5665e-02 | — | — | 9.2092 | F |
| 10 | IN | 8 | 5.5750e-03 | — | — | 39.1834 | F |
| 12 | IN | 8 | 4.7172e-02 | — | — | 154.7274 | F |
| 14 | IN | 8 | 8.4263e-02 | — | — | 3806.6317 | F |
| 15 | IN | 8 | 1.2544e-01 | — | — | 0.0153 | A |
| 16 | IN | 8 | 4.4333e-02 | — | — | 49.0765 | F |
| 20 | IN | 8 | 6.5845e-02 | — | — | 0.7112 | F |
| 26 | IN | 6 | 2.6124e-03 | — | — | 0.0154 | B |
| 30 | IN | 8 | 4.1534e-03 | — | — | 0.0331 | B |
| 40 | IN | 8 | 5.4888e-03 | — | — | 0.0424 | B |
| 60 | IN | 8 | 4.4920e-03 | — | — | 0.0594 | C |
| 80 | EXT | 8 | — | — | 8.85e-03 | 0.1658 | D |
| 100 | EXT | 19 | — | — | 7.98e-03 | 0.1369 | D |
| 150 | EXT | 3 | — | — | 3.58e-02 | 0.7841 | F |
| 200 | EXT | 3 | — | — | 3.59e-02 | 1.0472 | F |

> **Metric Warnings:**
> - ⚠️ Outlier: max ΔE/gap=0.814 is 6× the mean — median may be more representative N=4
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5898 (possible stale e_exact or gap) N=4
> - ⚠️ Outlier: max ΔE/gap=3.353 is 8× the mean — median may be more representative N=6
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 2.8794 (possible stale e_exact or gap) N=6
> - ⚠️ Outlier: max ΔE/gap=72.951 is 8× the mean — median may be more representative N=8
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 71.8524 (possible stale e_exact or gap) N=8
> - ⚠️ Outlier: max ΔE/gap=312.317 is 8× the mean — median may be more representative N=10
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 310.7901 (possible stale e_exact or gap) N=10
> - ⚠️ Outlier: max ΔE/gap=1235.945 is 8× the mean — median may be more representative N=12
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 1233.9820 (possible stale e_exact or gap) N=12
> - ⚠️ Outlier: max ΔE/gap=30381.980 is 8× the mean — median may be more representative N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 30379.1988 (possible stale e_exact or gap) N=14
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1271 (possible stale e_exact or gap) N=15
> - ⚠️ Outlier: max ΔE/gap=390.746 is 8× the mean — median may be more representative N=16
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 388.7826 (possible stale e_exact or gap) N=16
> - ⚠️ Outlier: max ΔE/gap=5.153 is 7× the mean — median may be more representative N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 3.2738 (possible stale e_exact or gap) N=20
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.1208 (possible stale e_exact or gap) N=26
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.3921 (possible stale e_exact or gap) N=30
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.5344 (possible stale e_exact or gap) N=40
> - ⚠️ Metric inconsistency: max |reconstructed - stored| ΔE/gap = 0.8478 (possible stale e_exact or gap) N=60

---

## Summary Ranking

| Topology | Checkpoint | In-dist θ MSE | Out-dist |ΔE|/N | Grade |
|----------|-----------|:---:|:---:|:---:|
| ladder | unified_tfim_br_ladder_multiN_4+6+8+10+12+16+20+26 | 1.7927e-02 | — | B (good) |
| ladder | unified_tfim_br_ladder_fromMT_4+6+8+10+12+14+20+26 | 4.1691e-02 | — | C (acceptable) |
| ladder | unified_tfim_br_ladder_multiN_4+6+8+10+12+14+20+26 | 8.0056e-02 | — | F (failing) |
| multi_topology | unified_tfim_br_MT_residual+film_p1.pt | 1.3023e-01 | 2.12e-02 | F (failing) |
| multi_topology | unifMPNN__MT_p1_res_film_base.pt | 1.3639e-01 | 2.12e-02 | F (failing) |
| heavy_hex | unified_tfim_br_heavy_hex_multiN_4+6+10+12+16+20+4 | 3.6151e+02 | 2.26e-02 | F (failing) |
| heavy_hex | unified_tfim_br_heavy_hex_multiN_4+6+10+12+16+20+4 | 1.4488e+00 | 2.26e-02 | F (failing) |
| heavy_hex | unified_tfim_br_heavy_hex_fromMT_4+6+10+12+16+20_p | 1.3037e+00 | 2.26e-02 | F (failing) |
| heavy_hex | unified_tfim_br_heavy_hex_multiN_4+6+8+10+12+14+18 | 3.9629e-01 | 2.26e-02 | F (failing) |
| heavy_hex | unified_tfim_br_heavy_hex_multiN_4+6+8+10+14+16+18 | 4.0603e-01 | 2.26e-02 | F (failing) |
| heavy_hex | unified_tfim_br_heavy_hex_multiN_4+6+8+10+14+16+18 | 4.5674e-01 | 2.26e-02 | F (failing) |
| heavy_hex | unified_tfim_br_heavy_hex_multiN_4+6+8+10+14+16+18 | 4.4529e-01 | 2.26e-02 | F (failing) |
| heavy_hex | unified_tfim_br_heavy_hex_multiN_4+6+8+10+14+16+18 | 4.5326e-01 | 2.26e-02 | F (failing) |
| heavy_hex | unified_tfim_br_heavy_hex_multiN_4+6+8+10+14+16+18 | 3.5758e-01 | 2.26e-02 | F (failing) |
| heavy_hex | unifMPNN__heavy_hex_p1_res_mse.pt | 3.6329e-01 | 2.26e-02 | F (failing) |
| heavy_hex | unifMPNN__heavy_hex_p1_res_film_mse.pt | 3.6336e-01 | 2.26e-02 | F (failing) |
| heavy_hex | unified_tfim_br_heavy_hex_fromMT_4+6+8+10+12+14+18 | 5.7796e+00 | 2.26e-02 | F (failing) |
| heavy_hex | unified_multiN_heavyhex_p1.pt | 2.2045e+00 | 2.26e-02 | F (failing) |
| heavy_hex | unifMPNN__heavy_hex_p1_res_mse_v2.pt | 4.4574e-01 | 2.26e-02 | F (failing) |
| heavy_hex | unified_tfim_br_heavy_hex_multiN_4+6+8+10+14+16+18 | 5.0477e-01 | 2.26e-02 | F (failing) |
| heavy_hex | unifMPNN__heavy_hex_p1_res_film_mse_v2.pt | 4.3115e-01 | 2.26e-02 | F (failing) |
| heavy_hex | unified_tfim_br_heavy_hex_multiN_4+6+8+10+14+16+18 | 4.3688e-01 | 2.26e-02 | F (failing) |
| heavy_hex | unifMPNN__heavy_hex_p1_plain_energyw.pt | 4.3855e-01 | 2.26e-02 | F (failing) |
| heavy_hex | unifMPNN__heavy_hex_p1_res_energyw.pt | 5.5317e-01 | 2.26e-02 | F (failing) |
| heavy_hex | unifMPNN__heavy_hex_p1_res_film_energyw.pt | 6.3942e-01 | 2.26e-02 | F (failing) |
| heavy_hex | unifMPNN__heavy_hex_p1_res_physics05.pt | 3.4766e-01 | 2.26e-02 | F (failing) |
| heavy_hex | unifMPNN__heavy_hex_p1_full_stack.pt | 4.6397e-01 | 2.26e-02 | F (failing) |
| heavy_hex | unifMPNN__heavy_hex_p1_res_film_energyw_critical.p | 4.0434e-01 | 2.26e-02 | F (failing) |
| heavy_hex | unifMPNN__heavy_hex_p1_res_film_deploy.pt | 4.1852e-01 | 2.26e-02 | F (failing) |
| chain_1d | unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20 | 4.0454e-02 | 2.21e-02 | F (failing) |
| chain_1d | unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20 | 1.3625e-01 | 2.21e-02 | F (failing) |
| chain_1d | unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20 | 1.3625e-01 | 2.21e-02 | F (failing) |
| chain_1d | unifMPNN__chain_1d_p1_h_0p5_1p5.pt | 4.3857e-02 | 2.21e-02 | F (failing) |
| chain_1d | unifMPNN__chain_1d_p1_h_0p5_1p5_v2.pt | 9.3769e-02 | 2.21e-02 | F (failing) |
| chain_1d | unified_tfim_br_chain_1d_multiN_4_p2.pt | 1.6230e-01 | 2.21e-02 | F (failing) |
| chain_1d | unifMPNN__chain_1d_p1_h_0p5_1p5_v3.pt | 6.7956e-02 | 2.21e-02 | F (failing) |
| square | unified_tfim_br_square_multiN_4+6+8+10+12+14_p1.pt | 4.5781e-02 | 2.97e-02 | F (failing) |
| square | unified_tfim_br_square_multiN_4+6+8+10+12+14_p1_v4 | 7.8955e-02 | 2.97e-02 | F (failing) |
| triangular | unified_tfim_br_triangular_multiN_3+4+6_p1.pt | 9.5203e-02 | 2.09e-01 | F (failing) |

---

# MT vs ST Head-to-Head Comparison

**Generated**: 2026-08-29 18:13 UTC
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
*Decision metric: quality_score (continuous 0-1, sigmoid-based on mean ΔE/gap + P90 )*

*Generated by `scripts/analysis/evaluate_zoo_models.py`*