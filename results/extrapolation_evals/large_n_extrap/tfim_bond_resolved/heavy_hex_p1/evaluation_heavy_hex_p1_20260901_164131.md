# Model Evaluation: heavy_hex

**Date**: 2026-09-01 16:41 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_res_energyw.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 2.5] (15 pts)
**Target N**: [8, 10, 20]

---

## N = 8 (15 params)

**ΔE/gap: 1.5579 ± 1.7849 | P90=3.6311 | max=6.9547
|ΔE|/N: 1.63e-01
Fidelity: mean=0.7880 min=0.5075 (exact)
Distribution: [P25=0.444 | P50=0.769 | P75=1.746 | P90=3.631]
Regions: critical=3.4889 | ordered=0.3734**

**Fidelity: mean F=0.7880, min F=0.5075** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -7.8016 | -9.8947 | 2.0932 | 0.3010 | 6.9547 | 0.5075 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -8.6842 | -10.5644 | 1.8802 | 0.4521 | 4.1584 | 0.5920 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -9.4866 | -11.2098 | 1.7232 | 0.6067 | 2.8402 | 0.6534 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -10.3691 | -11.9500 | 1.5809 | 0.7897 | 2.0020 | 0.7069 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -11.2515 | -12.7141 | 1.4626 | 0.9819 | 1.4895 | 0.7489 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.540 | -12.1339 | -13.4964 | 1.3625 | 1.1807 | 1.1540 | 0.7824 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.640 | -12.9361 | -14.2201 | 1.2840 | 1.3653 | 0.9404 | 0.8073 | N/A | — | severe_error(0.94) | increase_p |  |
| 1.750 | -13.8187 | -15.0271 | 1.2084 | 1.5716 | 0.7689 | 0.8299 | N/A | — | severe_error(0.76) | increase_p |  |
| 1.860 | -14.7015 | -15.8433 | 1.1419 | 1.7804 | 0.6413 | 0.8486 | N/A | — | severe_error(0.62) | increase_p |  |
| 1.960 | -15.5033 | -16.5920 | 1.0887 | 1.9719 | 0.5521 | 0.8630 | N/A | — | severe_error(0.53) | increase_p |  |
| 2.070 | -16.3845 | -17.4216 | 1.0371 | 2.1840 | 0.4749 | 0.8764 | N/A | — | moderate_error(0.45) | refine_vqe |  |
| 2.180 | -17.2663 | -18.2566 | 0.9903 | 2.3972 | 0.4131 | 0.8879 | N/A | — | moderate_error(0.38) | refine_vqe |  |
| 2.290 | -18.1482 | -19.0961 | 0.9479 | 2.6114 | 0.3630 | 0.8978 | N/A | — | moderate_error(0.33) | refine_vqe |  |
| 2.390 | -18.9497 | -19.8627 | 0.9130 | 2.8068 | 0.3253 | 0.9056 | N/A | — | moderate_error(0.29) | refine_vqe |  |
| 2.500 | -19.8309 | -20.7091 | 0.8783 | 3.0223 | 0.2906 | 0.9132 | N/A | — | moderate_error(0.25) | refine_vqe |  |

## N = 10 (19 params)

**ΔE/gap: 2.4051 ± 3.1266 | P90=5.7013 | max=12.2665
|ΔE|/N: 1.66e-01
Fidelity: mean=0.7373 min=0.3966 (exact)
Distribution: [P25=0.580 | P50=1.030 | P75=2.491 | P90=5.701]
Regions: critical=5.6357 | ordered=0.4851**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=12.267 is 5× the mean — median may be more representative N=10

**Fidelity: mean F=0.7373, min F=0.3966** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -9.7719 | -12.4722 | 2.7002 | 0.2201 | 12.2665 | 0.3966 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -10.8750 | -13.2894 | 2.4144 | 0.3632 | 6.6472 | 0.4966 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -11.8778 | -14.0836 | 2.2058 | 0.5151 | 4.2826 | 0.5710 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -12.9810 | -14.9990 | 2.0180 | 0.6978 | 2.8921 | 0.6363 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -14.0842 | -15.9469 | 1.8627 | 0.8912 | 2.0902 | 0.6880 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -15.1876 | -16.9194 | 1.7317 | 1.0915 | 1.5866 | 0.7293 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.640 | -16.1907 | -17.8199 | 1.6292 | 1.2777 | 1.2750 | 0.7600 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.750 | -17.2942 | -18.8250 | 1.5308 | 1.4858 | 1.0303 | 0.7880 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.860 | -18.3971 | -19.8422 | 1.4451 | 1.6962 | 0.8519 | 0.8112 | N/A | — | severe_error(0.84) | increase_p |  |
| 1.960 | -19.3987 | -20.7757 | 1.3770 | 1.8891 | 0.7289 | 0.8290 | N/A | — | severe_error(0.71) | increase_p |  |
| 2.070 | -20.5015 | -21.8104 | 1.3089 | 2.1026 | 0.6225 | 0.8458 | N/A | — | severe_error(0.60) | increase_p |  |
| 2.180 | -21.6051 | -22.8521 | 1.2470 | 2.3171 | 0.5382 | 0.8602 | N/A | — | severe_error(0.51) | increase_p |  |
| 2.290 | -22.7081 | -23.8998 | 1.1916 | 2.5324 | 0.4705 | 0.8727 | N/A | — | moderate_error(0.44) | refine_vqe |  |
| 2.390 | -23.7100 | -24.8565 | 1.1465 | 2.7287 | 0.4202 | 0.8825 | N/A | — | moderate_error(0.39) | refine_vqe |  |
| 2.500 | -24.8116 | -25.9132 | 1.1016 | 2.9452 | 0.3740 | 0.8920 | N/A | — | moderate_error(0.34) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 9.3744 ± 4.8172 | P90=15.2870 | max=18.0329
|ΔE|/N: 1.67e-01
Var(H): 20.7537
Distribution: [P25=5.773 | P50=9.677 | P75=12.513 | P90=15.287]
Regions: critical=14.6658 | ordered=3.6977**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=159.4586.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -19.7043 | -25.3695 | 5.6652 | 0.3142 | 18.0329 | N/A | 20.3033 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -21.9127 | -26.9058 | 4.9931 | 0.3142 | 15.8935 | N/A | 20.3977 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -23.9205 | -28.4372 | 4.5168 | 0.3142 | 14.3773 | N/A | 20.4764 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -26.1289 | -30.2298 | 4.1009 | 0.3142 | 13.0537 | N/A | 20.5562 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -28.3373 | -32.0983 | 3.7610 | 0.3142 | 11.9717 | N/A | 20.6287 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -30.5468 | -34.0227 | 3.4759 | 0.3142 | 11.0642 | N/A | 20.6868 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -32.5559 | -35.8088 | 3.2529 | 0.3142 | 10.3542 | N/A | 20.7292 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -34.7657 | -37.8057 | 3.0400 | 0.3142 | 9.6765 | N/A | 20.7692 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -36.9750 | -39.8292 | 2.8542 | 0.3142 | 9.0854 | N/A | 20.8040 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -38.9800 | -41.6875 | 2.7074 | 0.3142 | 8.6180 | N/A | 20.8541 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -41.1851 | -43.7486 | 2.5635 | 0.3519 | 7.2842 | N/A | 20.9083 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -43.3897 | -45.8248 | 2.4350 | 0.5713 | 4.2621 | N/A | 20.9618 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -45.5940 | -47.9136 | 2.3196 | 0.7908 | 2.9334 | N/A | 21.0150 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | -47.5968 | -49.8220 | 2.2252 | 0.9903 | 2.2469 | N/A | 21.0703 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -49.7984 | -51.9300 | 2.1317 | 1.2099 | 1.7619 | N/A | 21.1439 | dirty_state | severe_error(1.00) | increase_p |  |
