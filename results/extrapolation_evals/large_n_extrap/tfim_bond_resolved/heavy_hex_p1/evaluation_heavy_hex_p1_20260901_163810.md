# Model Evaluation: heavy_hex

**Date**: 2026-09-01 16:38 UTC
**Model**: data/model_zoo/checkpoints/unified_multiN_heavyhex_p1.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 2.5] (15 pts)
**Target N**: [8, 10, 20]

---

## N = 8 (15 params)

**ΔE/gap: 1.1751 ± 1.5663 | P90=2.9689 | max=5.9672
|ΔE|/N: 1.08e-01
Fidelity: mean=0.8418 min=0.5479 (exact)
Distribution: [P25=0.223 | P50=0.454 | P75=1.287 | P90=2.969]
Regions: critical=2.8486 | ordered=0.1833**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=5.967 is 5× the mean — median may be more representative N=8

**Fidelity: mean F=0.8418, min F=0.5479** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -8.0987 | -9.8947 | 1.7960 | 0.3010 | 5.9672 | 0.5479 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -9.0089 | -10.5644 | 1.5555 | 0.4521 | 3.4403 | 0.6383 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -9.8376 | -11.2098 | 1.3722 | 0.6067 | 2.2617 | 0.7039 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -10.7547 | -11.9500 | 1.1953 | 0.7897 | 1.5137 | 0.7618 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.430 | -11.6729 | -12.7141 | 1.0411 | 0.9819 | 1.0603 | 0.8076 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.540 | -12.5877 | -13.4964 | 0.9087 | 1.1807 | 0.7697 | 0.8436 | N/A | — | severe_error(0.76) | increase_p |  |
| 1.640 | -13.4143 | -14.2201 | 0.8059 | 1.3653 | 0.5902 | 0.8696 | N/A | — | severe_error(0.57) | increase_p |  |
| 1.750 | -14.3131 | -15.0271 | 0.7140 | 1.5716 | 0.4543 | 0.8919 | N/A | — | moderate_error(0.43) | refine_vqe |  |
| 1.860 | -15.2058 | -15.8433 | 0.6375 | 1.7804 | 0.3581 | 0.9093 | N/A | — | moderate_error(0.32) | refine_vqe |  |
| 1.960 | -16.0104 | -16.5920 | 0.5816 | 1.9719 | 0.2949 | 0.9218 | N/A | — | moderate_error(0.26) | refine_vqe |  |
| 2.070 | -16.8893 | -17.4216 | 0.5323 | 2.1840 | 0.2437 | 0.9325 | N/A | — | moderate_error(0.20) | refine_vqe |  |
| 2.180 | -17.7700 | -18.2566 | 0.4866 | 2.3972 | 0.2030 | 0.9416 | N/A | — | moderate_error(0.16) | refine_vqe |  |
| 2.290 | -18.6252 | -19.0961 | 0.4709 | 2.6114 | 0.1803 | 0.9466 | N/A | — | moderate_error(0.14) | refine_vqe |  |
| 2.390 | -19.4217 | -19.8627 | 0.4410 | 2.8068 | 0.1571 | 0.9523 | N/A | — | moderate_error(0.11) | refine_vqe |  |
| 2.500 | -20.3086 | -20.7091 | 0.4005 | 3.0223 | 0.1325 | 0.9588 | N/A | — | moderate_error(0.09) | refine_vqe |  |

## N = 10 (19 params)

**ΔE/gap: 2.1110 ± 3.0563 | P90=5.3734 | max=11.7315
|ΔE|/N: 1.27e-01
Fidelity: mean=0.7796 min=0.4078 (exact)
Distribution: [P25=0.327 | P50=0.724 | P75=2.184 | P90=5.373]
Regions: critical=5.2759 | ordered=0.2710**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=11.731 is 6× the mean — median may be more representative N=10

**Fidelity: mean F=0.7796, min F=0.4078** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -9.8897 | -12.4722 | 2.5825 | 0.2201 | 11.7315 | 0.4078 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -10.9985 | -13.2894 | 2.2909 | 0.3632 | 6.3071 | 0.5098 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -12.0373 | -14.0836 | 2.0463 | 0.5151 | 3.9729 | 0.5896 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -13.1930 | -14.9990 | 1.8060 | 0.6978 | 2.5882 | 0.6622 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -14.3608 | -15.9469 | 1.5861 | 0.8912 | 1.7798 | 0.7222 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.540 | -15.5421 | -16.9194 | 1.3772 | 1.0915 | 1.2618 | 0.7730 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.640 | -16.6014 | -17.8199 | 1.2185 | 1.2777 | 0.9536 | 0.8098 | N/A | — | severe_error(0.95) | increase_p |  |
| 1.750 | -17.7492 | -18.8250 | 1.0758 | 1.4858 | 0.7241 | 0.8417 | N/A | — | severe_error(0.71) | increase_p |  |
| 1.860 | -18.9064 | -19.8422 | 0.9358 | 1.6962 | 0.5517 | 0.8696 | N/A | — | severe_error(0.53) | increase_p |  |
| 1.960 | -19.9455 | -20.7757 | 0.8302 | 1.8891 | 0.4394 | 0.8898 | N/A | — | moderate_error(0.41) | refine_vqe |  |
| 2.070 | -21.0642 | -21.8104 | 0.7462 | 2.1026 | 0.3549 | 0.9062 | N/A | — | moderate_error(0.32) | refine_vqe |  |
| 2.180 | -22.1605 | -22.8521 | 0.6916 | 2.3171 | 0.2985 | 0.9178 | N/A | — | moderate_error(0.26) | refine_vqe |  |
| 2.290 | -23.2480 | -23.8998 | 0.6517 | 2.5324 | 0.2574 | 0.9266 | N/A | — | moderate_error(0.22) | refine_vqe |  |
| 2.390 | -24.2223 | -24.8565 | 0.6342 | 2.7287 | 0.2324 | 0.9320 | N/A | — | moderate_error(0.19) | refine_vqe |  |
| 2.500 | -25.2900 | -25.9132 | 0.6232 | 2.9452 | 0.2116 | 0.9365 | N/A | — | moderate_error(0.17) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 13.4402 ± 6.4323 | P90=19.9893 | max=20.3697
|ΔE|/N: 2.35e-01
Var(H): 29.9512
Distribution: [P25=8.751 | P50=15.174 | P75=18.763 | P90=19.989]
Regions: critical=16.8495 | ordered=5.1742**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=239.4907.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -21.0230 | -25.3695 | 4.3466 | 0.3142 | 13.8355 | N/A | 14.6723 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -22.1389 | -26.9058 | 4.7669 | 0.3142 | 15.1736 | N/A | 19.3048 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -23.1615 | -28.4372 | 5.2757 | 0.3142 | 16.7930 | N/A | 23.9926 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -24.3933 | -30.2298 | 5.8365 | 0.3142 | 18.5782 | N/A | 29.1551 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -25.8569 | -32.0983 | 6.2414 | 0.3142 | 19.8671 | N/A | 33.6917 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -27.6234 | -34.0227 | 6.3993 | 0.3142 | 20.3697 | N/A | 37.1263 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -29.5034 | -35.8088 | 6.3054 | 0.3142 | 20.0707 | N/A | 39.0418 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -31.8533 | -37.8057 | 5.9524 | 0.3142 | 18.9470 | N/A | 39.6182 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -34.4573 | -39.8292 | 5.3719 | 0.3142 | 17.0993 | N/A | 38.4392 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -36.9758 | -41.6875 | 4.7116 | 0.3142 | 14.9976 | N/A | 35.9488 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -39.7284 | -43.7486 | 4.0201 | 0.3519 | 11.4234 | N/A | 32.7379 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -42.3523 | -45.8248 | 3.4725 | 0.5713 | 6.0781 | N/A | 29.9842 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -44.9070 | -47.9136 | 3.0066 | 0.7908 | 3.8021 | N/A | 27.3975 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | -47.1879 | -49.8220 | 2.6341 | 0.9903 | 2.6598 | N/A | 25.1133 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -49.6221 | -51.9300 | 2.3080 | 1.2099 | 1.9076 | N/A | 23.0443 | dirty_state | severe_error(1.00) | increase_p |  |
