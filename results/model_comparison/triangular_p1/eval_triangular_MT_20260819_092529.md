# Model Evaluation: triangular

> **🌐 MULTI-TOPOLOGY MODEL** — This evaluation uses a model trained on multiple topologies simultaneously. Results reflect cross-topology transfer capability.

**Date**: 2026-08-19 09:25 UTC
**Model**: unified_tfim_br_multitopo_chain_1d+heavy_hex+ladder+square+triangular_maxN20_residual+film_p1.pt
**Multi-topology**: YES
**h-range**: [2.5, 5.0] (6 pts)
**Target N**: [8, 11, 12, 13]

---

## N = 8 (24 params)

**Quality: F (score=0.02)
ΔE/gap: 0.4216 ± 0.5281 | P90=0.9995 | max=1.5770
|ΔE|/N=7.63e-02
Distribution: [P25=0.116 | P50=0.156 | P75=0.364 | P90=1.000]**

| h | E_pred | E_exact | |ΔE| | |ΔE|/N | gap | ΔE/gap | Category | Action | Note |
|---|--------|---------|------|--------|-----|--------|----------|--------|------|
| 2.500 | -21.8353 | -22.9041 | 1.0689 | 1.34e-01 | 0.6778 | 1.5770 | severe_error(1.00) | increase_p |  |
| 3.000 | -25.4817 | -26.1021 | 0.6204 | 7.76e-02 | 1.4702 | 0.4220 | moderate_error(0.39) | refine_vqe |  |
| 3.500 | -29.1719 | -29.6342 | 0.4623 | 5.78e-02 | 2.4225 | 0.1908 | moderate_error(0.15) | refine_vqe |  |
| 4.000 | -32.9222 | -33.3383 | 0.4160 | 5.20e-02 | 3.4337 | 0.1212 | moderate_error(0.07) | refine_vqe |  |
| 4.500 | -36.6707 | -37.1355 | 0.4648 | 5.81e-02 | 4.4629 | 0.1042 | moderate_error(0.06) | refine_vqe |  |
| 5.000 | -40.3576 | -40.9878 | 0.6301 | 7.88e-02 | 5.4957 | 0.1147 | moderate_error(0.07) | refine_vqe |  |

## N = 11 (34 params)

**Quality: F (score=0.01)
ΔE/gap: 1.2972 ± 1.8605 | P90=3.2381 | max=5.4112
|ΔE|/N=1.11e-01
Distribution: [P25=0.293 | P50=0.377 | P75=0.912 | P90=3.238]**

| h | E_pred | E_exact | |ΔE| | |ΔE|/N | gap | ΔE/gap | Category | Action | Note |
|---|--------|---------|------|--------|-----|--------|----------|--------|------|
| 2.500 | -30.0088 | -31.8554 | 1.8466 | 1.68e-01 | 0.3413 | 5.4112 | ansatz_limited(1.00) | restrict_h_range |  |
| 3.000 | -34.9518 | -36.0578 | 1.1060 | 1.01e-01 | 1.0386 | 1.0650 | severe_error(1.00) | increase_p |  |
| 3.500 | -39.9346 | -40.8426 | 0.9080 | 8.25e-02 | 1.9991 | 0.4542 | moderate_error(0.43) | refine_vqe |  |
| 4.000 | -44.9988 | -45.9109 | 0.9121 | 8.29e-02 | 3.0428 | 0.2998 | moderate_error(0.26) | refine_vqe |  |
| 4.500 | -50.0425 | -51.1211 | 1.0786 | 9.81e-02 | 4.1016 | 0.2630 | moderate_error(0.22) | refine_vqe |  |
| 5.000 | -54.9144 | -56.4115 | 1.4971 | 1.36e-01 | 5.1572 | 0.2903 | moderate_error(0.25) | refine_vqe |  |

## N = 12 (38 params)

**Quality: F (score=0.00)
ΔE/gap: 2.8118 ± 4.3028 | P90=7.2414 | max=12.3464
|ΔE|/N=1.61e-01
Distribution: [P25=0.520 | P50=0.706 | P75=1.817 | P90=7.241]**

| h | E_pred | E_exact | |ΔE| | |ΔE|/N | gap | ΔE/gap | Category | Action | Note |
|---|--------|---------|------|--------|-----|--------|----------|--------|------|
| 2.500 | -32.6364 | -35.2650 | 2.6286 | 2.19e-01 | 0.2129 | 12.3464 | ansatz_limited(1.00) | restrict_h_range |  |
| 3.000 | -37.9498 | -39.6368 | 1.6869 | 1.41e-01 | 0.7897 | 2.1363 | ansatz_limited(1.00) | restrict_h_range |  |
| 3.500 | -43.2707 | -44.7352 | 1.4644 | 1.22e-01 | 1.7048 | 0.8590 | severe_error(0.85) | increase_p |  |
| 4.000 | -48.6891 | -50.2071 | 1.5179 | 1.26e-01 | 2.7465 | 0.5527 | severe_error(0.53) | increase_p |  |
| 4.500 | -54.0778 | -55.8612 | 1.7834 | 1.49e-01 | 3.8147 | 0.4675 | moderate_error(0.44) | refine_vqe |  |
| 5.000 | -59.1322 | -61.6148 | 2.4825 | 2.07e-01 | 4.8809 | 0.5086 | severe_error(0.48) | increase_p |  |

## N = 13 (42 params)

**Quality: F (score=0.00)
ΔE/gap: 5.7383 ± 9.4013 | P90=15.2907 | max=26.6098
|ΔE|/N=2.11e-01
Distribution: [P25=0.804 | P50=1.165 | P75=3.340 | P90=15.291]**

| h | E_pred | E_exact | |ΔE| | |ΔE|/N | gap | ΔE/gap | Category | Action | Note |
|---|--------|---------|------|--------|-----|--------|----------|--------|------|
| 2.500 | -35.1995 | -38.7113 | 3.5118 | 2.70e-01 | 0.1320 | 26.6098 | ansatz_limited(1.00) | restrict_h_range |  |
| 3.000 | -40.8789 | -43.2470 | 2.3681 | 1.82e-01 | 0.5963 | 3.9717 | ansatz_limited(1.00) | restrict_h_range |  |
| 3.500 | -46.5366 | -48.6400 | 2.1033 | 1.62e-01 | 1.4572 | 1.4434 | severe_error(1.00) | increase_p |  |
| 4.000 | -52.2974 | -54.5075 | 2.2102 | 1.70e-01 | 2.4949 | 0.8859 | severe_error(0.88) | increase_p |  |
| 4.500 | -57.9492 | -60.6030 | 2.6538 | 2.04e-01 | 3.5728 | 0.7428 | severe_error(0.73) | increase_p |  |
| 5.000 | -63.2080 | -66.8187 | 3.6107 | 2.78e-01 | 4.6502 | 0.7765 | severe_error(0.76) | increase_p |  |
