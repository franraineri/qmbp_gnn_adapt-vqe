# Model Evaluation: heavy_hex

**Date**: 2026-09-01 16:35 UTC
**Model**: data/model_zoo/checkpoints/unified_tfim_br_heavy_hex_multiN_4+6+10+12+16+20+40_p1_v1.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 2.5] (15 pts)
**Target N**: [8, 10, 20]

---

## N = 8 (15 params)

**ΔE/gap: 0.4204 ± 0.5910 | P90=1.0260 | max=2.3159
|ΔE|/N: 3.94e-02
Fidelity: mean=0.9305 min=0.7493 (exact)
Distribution: [P25=0.095 | P50=0.148 | P75=0.395 | P90=1.026]
Regions: critical=1.0128 | ordered=0.0887**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=2.316 is 6× the mean — median may be more representative N=8

**Fidelity: mean F=0.9305, min F=0.7493** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -9.1977 | -9.8947 | 0.6970 | 0.3010 | 2.3159 | 0.7493 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -10.0151 | -10.5644 | 0.5493 | 0.4521 | 1.2149 | 0.8244 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.210 | -10.7592 | -11.2098 | 0.4506 | 0.6067 | 0.7428 | 0.8715 | N/A | — | severe_error(0.73) | increase_p |  |
| 1.320 | -11.5784 | -11.9500 | 0.3715 | 0.7897 | 0.4705 | 0.9066 | N/A | — | moderate_error(0.44) | refine_vqe |  |
| 1.430 | -12.3998 | -12.7141 | 0.3142 | 0.9819 | 0.3200 | 0.9301 | N/A | — | moderate_error(0.28) | refine_vqe |  |
| 1.540 | -13.2219 | -13.4964 | 0.2745 | 1.1807 | 0.2325 | 0.9458 | N/A | — | moderate_error(0.19) | refine_vqe |  |
| 1.640 | -13.9702 | -14.2201 | 0.2500 | 1.3653 | 0.1831 | 0.9554 | N/A | — | moderate_error(0.14) | refine_vqe |  |
| 1.750 | -14.7943 | -15.0271 | 0.2328 | 1.5716 | 0.1481 | 0.9627 | N/A | — | moderate_error(0.10) | refine_vqe |  |
| 1.860 | -15.6209 | -15.8433 | 0.2224 | 1.7804 | 0.1249 | 0.9676 | N/A | — | moderate_error(0.08) | refine_vqe |  |
| 1.960 | -16.3742 | -16.5920 | 0.2178 | 1.9719 | 0.1105 | 0.9707 | N/A | — | moderate_error(0.06) | refine_vqe |  |
| 2.070 | -17.2055 | -17.4216 | 0.2161 | 2.1840 | 0.0990 | 0.9731 | N/A | — | near_pass(0.05) | refine_vqe |  |
| 2.180 | -18.0379 | -18.2566 | 0.2187 | 2.3972 | 0.0912 | 0.9746 | N/A | — | near_pass(0.04) | refine_vqe |  |
| 2.290 | -18.8703 | -19.0961 | 0.2258 | 2.6114 | 0.0865 | 0.9754 | N/A | — | near_pass(0.04) | refine_vqe |  |
| 2.390 | -19.6263 | -19.8627 | 0.2364 | 2.8068 | 0.0842 | 0.9755 | N/A | — | near_pass(0.04) | refine_vqe |  |
| 2.500 | -20.4595 | -20.7091 | 0.2497 | 3.0223 | 0.0826 | 0.9754 | N/A | — | near_pass(0.03) | refine_vqe |  |

## N = 10 (19 params)

**ΔE/gap: 1.2188 ± 1.5528 | P90=2.7544 | max=6.2315
|ΔE|/N: 8.99e-02
Fidelity: mean=0.8431 min=0.5873 (exact)
Distribution: [P25=0.360 | P50=0.532 | P75=1.175 | P90=2.754]
Regions: critical=2.7701 | ordered=0.3246**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=6.231 is 5× the mean — median may be more representative N=10

**Fidelity: mean F=0.8431, min F=0.5873** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -11.1004 | -12.4722 | 1.3717 | 0.2201 | 6.2315 | 0.5873 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -12.1145 | -13.2894 | 1.1749 | 0.3632 | 3.2346 | 0.6872 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -13.0360 | -14.0836 | 1.0477 | 0.5151 | 2.0340 | 0.7519 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -14.0495 | -14.9990 | 0.9495 | 0.6978 | 1.3608 | 0.8012 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.430 | -15.0649 | -15.9469 | 0.8820 | 0.8912 | 0.9897 | 0.8350 | N/A | — | severe_error(0.99) | increase_p |  |
| 1.540 | -16.0822 | -16.9194 | 0.8371 | 1.0915 | 0.7669 | 0.8584 | N/A | — | severe_error(0.75) | increase_p |  |
| 1.640 | -17.0101 | -17.8199 | 0.8099 | 1.2777 | 0.6338 | 0.8739 | N/A | — | severe_error(0.61) | increase_p |  |
| 1.750 | -18.0342 | -18.8250 | 0.7908 | 1.4858 | 0.5322 | 0.8865 | N/A | — | severe_error(0.51) | increase_p |  |
| 1.860 | -19.0595 | -19.8422 | 0.7827 | 1.6962 | 0.4614 | 0.8956 | N/A | — | moderate_error(0.43) | refine_vqe |  |
| 1.960 | -19.9929 | -20.7757 | 0.7828 | 1.8891 | 0.4144 | 0.9017 | N/A | — | moderate_error(0.38) | refine_vqe |  |
| 2.070 | -21.0219 | -21.8104 | 0.7884 | 2.1026 | 0.3750 | 0.9069 | N/A | — | moderate_error(0.34) | refine_vqe |  |
| 2.180 | -22.0544 | -22.8521 | 0.7977 | 2.3171 | 0.3443 | 0.9109 | N/A | — | moderate_error(0.31) | refine_vqe |  |
| 2.290 | -23.0898 | -23.8998 | 0.8099 | 2.5324 | 0.3198 | 0.9141 | N/A | — | moderate_error(0.28) | refine_vqe |  |
| 2.390 | -24.0349 | -24.8565 | 0.8217 | 2.7287 | 0.3011 | 0.9167 | N/A | — | moderate_error(0.26) | refine_vqe |  |
| 2.500 | -25.0797 | -25.9132 | 0.8335 | 2.9452 | 0.2830 | 0.9193 | N/A | — | moderate_error(0.25) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 37.7328 ± 12.3304 | P90=45.7736 | max=46.9653
|ΔE|/N: 7.20e-01
Var(H): 73.2272
Distribution: [P25=33.255 | P50=44.919 | P75=45.318 | P90=45.774]
Regions: critical=45.5595 | ordered=22.4665**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=492.5074.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -10.6149 | -25.3695 | 14.7546 | 0.3142 | 46.9653 | N/A | 39.0463 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -12.4982 | -26.9058 | 14.4076 | 0.3142 | 45.8607 | N/A | 43.1697 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -14.2176 | -28.4372 | 14.2196 | 0.3142 | 45.2624 | N/A | 47.1247 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -16.1180 | -30.2298 | 14.1118 | 0.3142 | 44.9193 | N/A | 51.6915 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -18.0271 | -32.0983 | 14.0712 | 0.3142 | 44.7900 | N/A | 56.4774 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -19.9461 | -34.0227 | 14.0766 | 0.3142 | 44.8072 | N/A | 61.4683 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -21.6972 | -35.8088 | 14.1116 | 0.3142 | 44.9187 | N/A | 66.1867 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -23.6314 | -37.8057 | 14.1742 | 0.3142 | 45.1180 | N/A | 71.5592 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -25.5744 | -39.8292 | 14.2548 | 0.3142 | 45.3746 | N/A | 77.1144 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -27.3483 | -41.6875 | 14.3392 | 0.3142 | 45.6430 | N/A | 82.3151 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -29.3073 | -43.7486 | 14.4413 | 0.3519 | 41.0357 | N/A | 88.1994 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -31.2707 | -45.8248 | 14.5541 | 0.5713 | 25.4746 | N/A | 94.2712 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -33.2377 | -47.9136 | 14.6759 | 0.7908 | 18.5589 | N/A | 100.5312 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | -35.0326 | -49.8220 | 14.7894 | 0.9903 | 14.9339 | N/A | 106.3530 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -37.0130 | -51.9300 | 14.9170 | 1.2099 | 12.3295 | N/A | 112.8998 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
