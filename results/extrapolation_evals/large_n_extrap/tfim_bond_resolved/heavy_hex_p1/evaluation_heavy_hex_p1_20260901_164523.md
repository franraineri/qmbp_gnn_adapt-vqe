# Model Evaluation: heavy_hex

**Date**: 2026-09-01 16:45 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_res_film_energyw_critical.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 2.5] (15 pts)
**Target N**: [8, 10, 20]

---

## N = 8 (15 params)

**ΔE/gap: 1.4028 ± 1.5826 | P90=3.2707 | max=6.1424
|ΔE|/N: 1.48e-01
Fidelity: mean=0.8030 min=0.5407 (exact)
Distribution: [P25=0.399 | P50=0.702 | P75=1.603 | P90=3.271]
Regions: critical=3.1325 | ordered=0.3339**

**Fidelity: mean F=0.8030, min F=0.5407** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -8.0460 | -9.8947 | 1.8487 | 0.3010 | 6.1424 | 0.5407 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -8.8806 | -10.5644 | 1.6838 | 0.4521 | 3.7239 | 0.6193 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -9.6379 | -11.2098 | 1.5719 | 0.6067 | 2.5909 | 0.6742 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -10.4975 | -11.9500 | 1.4524 | 0.7897 | 1.8393 | 0.7241 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.430 | -11.3726 | -12.7141 | 1.3414 | 0.9819 | 1.3661 | 0.7647 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.540 | -12.2413 | -13.4964 | 1.2551 | 1.1807 | 1.0631 | 0.7959 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.640 | -13.0422 | -14.2201 | 1.1779 | 1.3653 | 0.8628 | 0.8201 | N/A | — | severe_error(0.86) | increase_p |  |
| 1.750 | -13.9246 | -15.0271 | 1.1025 | 1.5716 | 0.7015 | 0.8422 | N/A | — | severe_error(0.69) | increase_p |  |
| 1.860 | -14.8053 | -15.8433 | 1.0380 | 1.7804 | 0.5830 | 0.8603 | N/A | — | severe_error(0.56) | increase_p |  |
| 1.960 | -15.6058 | -16.5920 | 0.9862 | 1.9719 | 0.5001 | 0.8740 | N/A | — | severe_error(0.47) | increase_p |  |
| 2.070 | -16.4869 | -17.4216 | 0.9347 | 2.1840 | 0.4280 | 0.8870 | N/A | — | moderate_error(0.40) | refine_vqe |  |
| 2.180 | -17.3682 | -18.2566 | 0.8885 | 2.3972 | 0.3706 | 0.8980 | N/A | — | moderate_error(0.34) | refine_vqe |  |
| 2.290 | -18.2494 | -19.0961 | 0.8467 | 2.6114 | 0.3242 | 0.9075 | N/A | — | moderate_error(0.29) | refine_vqe |  |
| 2.390 | -19.0505 | -19.8627 | 0.8122 | 2.8068 | 0.2894 | 0.9150 | N/A | — | moderate_error(0.25) | refine_vqe |  |
| 2.500 | -19.9315 | -20.7091 | 0.7776 | 3.0223 | 0.2573 | 0.9222 | N/A | — | moderate_error(0.22) | refine_vqe |  |

## N = 10 (19 params)

**ΔE/gap: 2.2156 ± 2.8428 | P90=5.2596 | max=11.1265
|ΔE|/N: 1.54e-01
Fidelity: mean=0.7503 min=0.4230 (exact)
Distribution: [P25=0.536 | P50=0.960 | P75=2.335 | P90=5.260]
Regions: critical=5.1787 | ordered=0.4467**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=11.126 is 5× the mean — median may be more representative N=10

**Fidelity: mean F=0.7503, min F=0.4230** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -10.0229 | -12.4722 | 2.4493 | 0.2201 | 11.1265 | 0.4230 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -11.0722 | -13.2894 | 2.2172 | 0.3632 | 6.1042 | 0.5192 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -12.0271 | -14.0836 | 2.0565 | 0.5151 | 3.9927 | 0.5885 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -13.1083 | -14.9990 | 1.8907 | 0.6978 | 2.7096 | 0.6513 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -14.1998 | -15.9469 | 1.7471 | 0.8912 | 1.9605 | 0.7014 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.540 | -15.2943 | -16.9194 | 1.6250 | 1.0915 | 1.4888 | 0.7414 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.640 | -16.2964 | -17.8199 | 1.5236 | 1.2777 | 1.1924 | 0.7717 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.750 | -17.3989 | -18.8250 | 1.4261 | 1.4858 | 0.9598 | 0.7993 | N/A | — | severe_error(0.96) | increase_p |  |
| 1.860 | -18.4999 | -19.8422 | 1.3423 | 1.6962 | 0.7913 | 0.8220 | N/A | — | severe_error(0.78) | increase_p |  |
| 1.960 | -19.5002 | -20.7757 | 1.2754 | 1.8891 | 0.6752 | 0.8393 | N/A | — | severe_error(0.66) | increase_p |  |
| 2.070 | -20.6008 | -21.8104 | 1.2096 | 2.1026 | 0.5753 | 0.8555 | N/A | — | severe_error(0.55) | increase_p |  |
| 2.180 | -21.7017 | -22.8521 | 1.1504 | 2.3171 | 0.4965 | 0.8694 | N/A | — | moderate_error(0.47) | refine_vqe |  |
| 2.290 | -22.8024 | -23.8998 | 1.0974 | 2.5324 | 0.4333 | 0.8813 | N/A | — | moderate_error(0.40) | refine_vqe |  |
| 2.390 | -23.8032 | -24.8565 | 1.0533 | 2.7287 | 0.3860 | 0.8908 | N/A | — | moderate_error(0.35) | refine_vqe |  |
| 2.500 | -24.9045 | -25.9132 | 1.0087 | 2.9452 | 0.3425 | 0.9000 | N/A | — | moderate_error(0.31) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 9.1861 ± 4.7200 | P90=14.9803 | max=17.5432
|ΔE|/N: 1.63e-01
Var(H): 20.1540
Distribution: [P25=5.642 | P50=9.511 | P75=12.297 | P90=14.980]
Regions: critical=14.3636 | ordered=3.6019**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=155.2354.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -19.8582 | -25.3695 | 5.5114 | 0.3142 | 17.5432 | N/A | 19.5574 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -22.0236 | -26.9058 | 4.8822 | 0.3142 | 15.5404 | N/A | 19.7768 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -23.9949 | -28.4372 | 4.4423 | 0.3142 | 14.1402 | N/A | 19.9876 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -26.1981 | -30.2298 | 4.0318 | 0.3142 | 12.8336 | N/A | 20.0624 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -28.4036 | -32.0983 | 3.6947 | 0.3142 | 11.7605 | N/A | 20.1203 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -30.6017 | -34.0227 | 3.4211 | 0.3142 | 10.8896 | N/A | 20.2193 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -32.6104 | -35.8088 | 3.1984 | 0.3142 | 10.1809 | N/A | 20.2420 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -34.8179 | -37.8057 | 2.9878 | 0.3142 | 9.5105 | N/A | 20.2740 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -37.0244 | -39.8292 | 2.8049 | 0.3142 | 8.9281 | N/A | 20.3062 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -39.0310 | -41.6875 | 2.6564 | 0.3142 | 8.4557 | N/A | 20.3248 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -41.2401 | -43.7486 | 2.5084 | 0.3519 | 7.1278 | N/A | 20.3242 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -43.4499 | -45.8248 | 2.3749 | 0.5713 | 4.1569 | N/A | 20.3104 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -45.6585 | -47.9136 | 2.2551 | 0.7908 | 2.8518 | N/A | 20.2978 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | -47.6670 | -49.8220 | 2.1550 | 0.9903 | 2.1760 | N/A | 20.2736 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -49.8769 | -51.9300 | 2.0531 | 1.2099 | 1.6970 | N/A | 20.2328 | dirty_state | severe_error(1.00) | increase_p |  |
