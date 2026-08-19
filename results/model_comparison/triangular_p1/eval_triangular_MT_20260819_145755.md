# Model Evaluation: triangular

> **🌐 MULTI-TOPOLOGY MODEL** — This evaluation uses a model trained on multiple topologies simultaneously. Results reflect cross-topology transfer capability.

**Date**: 2026-08-19 14:57 UTC
**Model**: unified_tfim_br_multitopo_chain_1d+heavy_hex+ladder+square+triangular_maxN20_residual+film_p1.pt
**Multi-topology**: YES
**h-range**: [1.5, 5.0] (15 pts)
**Target N**: [6, 10, 12]

---

## N = 6 (16 params)

**Quality: F (score=0.03)
ΔE/gap: 0.7490 ± 1.7480 | P90=1.8091 | max=6.8858
|ΔE|/N=5.33e-02
Distribution: [P25=0.013 | P50=0.046 | P75=0.333 | P90=1.809]**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=6.886 is 9× the mean — median may be more representative N=6

| h | E_pred | E_exact | |ΔE| | |ΔE|/N | gap | ΔE/gap | Category | Action | Note |
|---|--------|---------|------|--------|-----|--------|----------|--------|------|
| 1.500 | -10.8490 | -12.2014 | 1.3524 | 2.25e-01 | 0.1964 | 6.8858 | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -12.1209 | -13.0748 | 0.9539 | 1.59e-01 | 0.4018 | 2.3743 | ansatz_limited(1.00) | restrict_h_range |  |
| 2.000 | -13.4301 | -14.0995 | 0.6694 | 1.12e-01 | 0.6963 | 0.9614 | severe_error(0.96) | increase_p |  |
| 2.250 | -14.7768 | -15.2472 | 0.4704 | 7.84e-02 | 1.0643 | 0.4419 | moderate_error(0.41) | refine_vqe |  |
| 2.500 | -16.1524 | -16.4858 | 0.3334 | 5.56e-02 | 1.4844 | 0.2246 | moderate_error(0.18) | refine_vqe |  |
| 2.750 | -17.5496 | -17.7886 | 0.2390 | 3.98e-02 | 1.9382 | 0.1233 | moderate_error(0.08) | refine_vqe |  |
| 3.000 | -18.9602 | -19.1363 | 0.1762 | 2.94e-02 | 2.4127 | 0.0730 | near_pass(0.02) | refine_vqe |  |
| 3.250 | -20.3832 | -20.5160 | 0.1328 | 2.21e-02 | 2.9001 | 0.0458 | gap_masked(0.44) | refine_vqe |  |
| 3.500 | -21.8154 | -21.9187 | 0.1033 | 1.72e-02 | 3.3952 | 0.0304 | gap_masked(0.34) | refine_vqe |  |
| 3.750 | -23.2579 | -23.3386 | 0.0807 | 1.34e-02 | 3.8950 | 0.0207 | pass(0.41) | none |  |
| 4.000 | -24.7065 | -24.7715 | 0.0650 | 1.08e-02 | 4.3976 | 0.0148 | pass(0.30) | none |  |
| 4.250 | -26.1587 | -26.2145 | 0.0558 | 9.29e-03 | 4.9019 | 0.0114 | pass(0.23) | none |  |
| 4.500 | -27.6147 | -27.6655 | 0.0507 | 8.45e-03 | 5.4071 | 0.0094 | pass(0.19) | none |  |
| 4.750 | -29.0701 | -29.1228 | 0.0528 | 8.79e-03 | 5.9128 | 0.0089 | pass(0.18) | none |  |
| 5.000 | -30.5254 | -30.5854 | 0.0600 | 1.00e-02 | 6.4188 | 0.0093 | pass(0.19) | none |  |

## N = 10 (30 params)

**Quality: F (score=0.00)
ΔE/gap: 38.9852 ± 115.3512 | P90=62.3859 | max=462.4354
|ΔE|/N=1.25e-01
Distribution: [P25=0.131 | P50=0.293 | P75=4.257 | P90=62.386]**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=462.435 is 12× the mean — median may be more representative N=10

| h | E_pred | E_exact | |ΔE| | |ΔE|/N | gap | ΔE/gap | Category | Action | Note |
|---|--------|---------|------|--------|-----|--------|----------|--------|------|
| 1.500 | -18.4560 | -22.9134 | 4.4574 | 4.46e-01 | 0.0096 | 462.4354 | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -20.6458 | -23.9918 | 3.3459 | 3.35e-01 | 0.0373 | 89.6034 | ansatz_limited(1.00) | restrict_h_range |  |
| 2.000 | -22.8495 | -25.2700 | 2.4205 | 2.42e-01 | 0.1123 | 21.5597 | ansatz_limited(1.00) | restrict_h_range |  |
| 2.250 | -25.0575 | -26.7709 | 1.7134 | 1.71e-01 | 0.2713 | 6.3161 | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -27.3152 | -28.5006 | 1.1853 | 1.19e-01 | 0.5395 | 2.1972 | ansatz_limited(1.00) | restrict_h_range |  |
| 2.750 | -29.5918 | -30.4316 | 0.8397 | 8.40e-02 | 0.9101 | 0.9227 | severe_error(0.92) | increase_p |  |
| 3.000 | -31.8705 | -32.5148 | 0.6443 | 6.44e-02 | 1.3536 | 0.4760 | moderate_error(0.45) | refine_vqe |  |
| 3.250 | -34.1651 | -34.7036 | 0.5385 | 5.38e-02 | 1.8399 | 0.2927 | moderate_error(0.26) | refine_vqe |  |
| 3.500 | -36.4853 | -36.9642 | 0.4789 | 4.79e-02 | 2.3486 | 0.2039 | moderate_error(0.16) | refine_vqe |  |
| 3.750 | -38.8251 | -39.2745 | 0.4493 | 4.49e-02 | 2.8680 | 0.1567 | moderate_error(0.11) | refine_vqe |  |
| 4.000 | -41.1773 | -41.6201 | 0.4429 | 4.43e-02 | 3.3919 | 0.1306 | moderate_error(0.08) | refine_vqe |  |
| 4.250 | -43.5320 | -43.9919 | 0.4599 | 4.60e-02 | 3.9170 | 0.1174 | moderate_error(0.07) | refine_vqe |  |
| 4.500 | -45.8703 | -46.3833 | 0.5131 | 5.13e-02 | 4.4416 | 0.1155 | moderate_error(0.07) | refine_vqe |  |
| 4.750 | -48.1941 | -48.7902 | 0.5961 | 5.96e-02 | 4.9652 | 0.1201 | moderate_error(0.07) | refine_vqe |  |
| 5.000 | -50.4928 | -51.2093 | 0.7165 | 7.16e-02 | 5.4873 | 0.1306 | moderate_error(0.08) | refine_vqe |  |

## N = 12 (38 params)

**Quality: F (score=0.00)
ΔE/gap: 497.7524 ± 1568.6849 | P90=622.5936 | max=6304.5679
|ΔE|/N=2.39e-01
Distribution: [P25=0.531 | P50=1.250 | P75=26.801 | P90=622.594]**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=6304.568 is 13× the mean — median may be more representative N=12

| h | E_pred | E_exact | |ΔE| | |ΔE|/N | gap | ΔE/gap | Category | Action | Note |
|---|--------|---------|------|--------|-----|--------|----------|--------|------|
| 1.500 | -21.0301 | -29.2709 | 8.2408 | 6.87e-01 | 0.0013 | 6304.5679 | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -24.1507 | -30.4600 | 6.3093 | 5.26e-01 | 0.0069 | 920.7145 | ansatz_limited(1.00) | restrict_h_range |  |
| 2.000 | -27.1091 | -31.8441 | 4.7350 | 3.95e-01 | 0.0270 | 175.4124 | ansatz_limited(1.00) | restrict_h_range |  |
| 2.250 | -29.9705 | -33.4386 | 3.4682 | 2.89e-01 | 0.0841 | 41.2555 | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -32.6364 | -35.2650 | 2.6286 | 2.19e-01 | 0.2129 | 12.3464 | ansatz_limited(1.00) | restrict_h_range |  |
| 2.750 | -35.2939 | -37.3364 | 2.0426 | 1.70e-01 | 0.4461 | 4.5791 | ansatz_limited(1.00) | restrict_h_range |  |
| 3.000 | -37.9498 | -39.6368 | 1.6869 | 1.41e-01 | 0.7897 | 2.1363 | ansatz_limited(1.00) | restrict_h_range |  |
| 3.250 | -40.5950 | -42.1203 | 1.5254 | 1.27e-01 | 1.2201 | 1.2502 | severe_error(1.00) | increase_p |  |
| 3.500 | -43.2707 | -44.7352 | 1.4644 | 1.22e-01 | 1.7048 | 0.8590 | severe_error(0.85) | increase_p |  |
| 3.750 | -45.9741 | -47.4401 | 1.4660 | 1.22e-01 | 2.2187 | 0.6607 | severe_error(0.64) | increase_p |  |
| 4.000 | -48.6891 | -50.2071 | 1.5179 | 1.26e-01 | 2.7465 | 0.5527 | severe_error(0.53) | increase_p |  |
| 4.250 | -51.4082 | -53.0181 | 1.6099 | 1.34e-01 | 3.2800 | 0.4908 | moderate_error(0.46) | refine_vqe |  |
| 4.500 | -54.0778 | -55.8612 | 1.7834 | 1.49e-01 | 3.8147 | 0.4675 | moderate_error(0.44) | refine_vqe |  |
| 4.750 | -56.6235 | -58.7286 | 2.1051 | 1.75e-01 | 4.3487 | 0.4841 | moderate_error(0.46) | refine_vqe |  |
| 5.000 | -59.1322 | -61.6148 | 2.4825 | 2.07e-01 | 4.8809 | 0.5086 | severe_error(0.48) | increase_p |  |
