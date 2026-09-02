# Model Evaluation: heavy_hex

**Date**: 2026-09-01 16:44 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_full_stack.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 2.5] (15 pts)
**Target N**: [8, 10, 20]

---

## N = 8 (15 params)

**ΔE/gap: 1.3119 ± 1.5603 | P90=3.1375 | max=6.0086
|ΔE|/N: 1.32e-01
Fidelity: mean=0.8176 min=0.5461 (exact)
Distribution: [P25=0.328 | P50=0.618 | P75=1.496 | P90=3.138]
Regions: critical=3.0093 | ordered=0.2700**

**Fidelity: mean F=0.8176, min F=0.5461** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -8.0863 | -9.8947 | 1.8084 | 0.3010 | 6.0086 | 0.5461 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -8.9392 | -10.5644 | 1.6252 | 0.4521 | 3.5944 | 0.6278 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -9.7221 | -11.2098 | 1.4877 | 0.6067 | 2.4522 | 0.6865 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -10.5897 | -11.9500 | 1.3602 | 0.7897 | 1.7226 | 0.7374 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.430 | -11.4684 | -12.7141 | 1.2457 | 0.9819 | 1.2686 | 0.7781 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.540 | -12.3558 | -13.4964 | 1.1406 | 1.1807 | 0.9661 | 0.8113 | N/A | — | severe_error(0.96) | increase_p |  |
| 1.640 | -13.1668 | -14.2201 | 1.0533 | 1.3653 | 0.7715 | 0.8363 | N/A | — | severe_error(0.76) | increase_p |  |
| 1.750 | -14.0564 | -15.0271 | 0.9707 | 1.5716 | 0.6177 | 0.8587 | N/A | — | severe_error(0.60) | increase_p |  |
| 1.860 | -14.9470 | -15.8433 | 0.8964 | 1.7804 | 0.5035 | 0.8772 | N/A | — | severe_error(0.48) | increase_p |  |
| 1.960 | -15.7571 | -16.5920 | 0.8349 | 1.9719 | 0.4234 | 0.8914 | N/A | — | moderate_error(0.39) | refine_vqe |  |
| 2.070 | -16.6475 | -17.4216 | 0.7741 | 2.1840 | 0.3544 | 0.9047 | N/A | — | moderate_error(0.32) | refine_vqe |  |
| 2.180 | -17.5331 | -18.2566 | 0.7236 | 2.3972 | 0.3018 | 0.9155 | N/A | — | moderate_error(0.27) | refine_vqe |  |
| 2.290 | -18.4158 | -19.0961 | 0.6803 | 2.6114 | 0.2605 | 0.9244 | N/A | — | moderate_error(0.22) | refine_vqe |  |
| 2.390 | -19.2162 | -19.8627 | 0.6465 | 2.8068 | 0.2303 | 0.9312 | N/A | — | moderate_error(0.19) | refine_vqe |  |
| 2.500 | -20.0960 | -20.7091 | 0.6132 | 3.0223 | 0.2029 | 0.9377 | N/A | — | moderate_error(0.16) | refine_vqe |  |

## N = 10 (19 params)

**ΔE/gap: 2.0747 ± 2.7457 | P90=4.9937 | max=10.6966
|ΔE|/N: 1.39e-01
Fidelity: mean=0.7671 min=0.4340 (exact)
Distribution: [P25=0.448 | P50=0.869 | P75=2.192 | P90=4.994]
Regions: critical=4.9315 | ordered=0.3661**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=10.697 is 5× the mean — median may be more representative N=10

**Fidelity: mean F=0.7671, min F=0.4340** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -10.1175 | -12.4722 | 2.3546 | 0.2201 | 10.6966 | 0.4340 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -11.1772 | -13.2894 | 2.1123 | 0.3632 | 5.8152 | 0.5325 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -12.1463 | -14.0836 | 1.9374 | 0.5151 | 3.7614 | 0.6041 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -13.2218 | -14.9990 | 1.7772 | 0.6978 | 2.5469 | 0.6662 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -14.3097 | -15.9469 | 1.6372 | 0.8912 | 1.8372 | 0.7157 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.540 | -15.4086 | -16.9194 | 1.5107 | 1.0915 | 1.3841 | 0.7559 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.640 | -16.4172 | -17.8199 | 1.4027 | 1.2777 | 1.0978 | 0.7866 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.750 | -17.5333 | -18.8250 | 1.2917 | 1.4858 | 0.8693 | 0.8153 | N/A | — | severe_error(0.86) | increase_p |  |
| 1.860 | -18.6562 | -19.8422 | 1.1860 | 1.6962 | 0.6992 | 0.8399 | N/A | — | severe_error(0.68) | increase_p |  |
| 1.960 | -19.6753 | -20.7757 | 1.1004 | 1.8891 | 0.5825 | 0.8587 | N/A | — | severe_error(0.56) | increase_p |  |
| 2.070 | -20.7898 | -21.8104 | 1.0206 | 2.1026 | 0.4854 | 0.8757 | N/A | — | moderate_error(0.46) | refine_vqe |  |
| 2.180 | -21.9009 | -22.8521 | 0.9512 | 2.3171 | 0.4105 | 0.8899 | N/A | — | moderate_error(0.38) | refine_vqe |  |
| 2.290 | -23.0071 | -23.8998 | 0.8927 | 2.5324 | 0.3525 | 0.9016 | N/A | — | moderate_error(0.32) | refine_vqe |  |
| 2.390 | -24.0108 | -24.8565 | 0.8458 | 2.7287 | 0.3100 | 0.9106 | N/A | — | moderate_error(0.27) | refine_vqe |  |
| 2.500 | -25.1125 | -25.9132 | 0.8007 | 2.9452 | 0.2719 | 0.9191 | N/A | — | moderate_error(0.23) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 8.6844 ± 4.7077 | P90=14.5320 | max=16.8243
|ΔE|/N: 1.53e-01
Var(H): 18.5375
Distribution: [P25=4.915 | P50=8.882 | P75=12.092 | P90=14.532]
Regions: critical=13.9643 | ordered=3.1171**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=145.9243.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -20.0840 | -25.3695 | 5.2855 | 0.3142 | 16.8243 | N/A | 18.6808 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -22.1826 | -26.9058 | 4.7232 | 0.3142 | 15.0345 | N/A | 19.0833 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -24.1086 | -28.4372 | 4.3286 | 0.3142 | 13.7784 | N/A | 19.4396 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -26.2661 | -30.2298 | 3.9638 | 0.3142 | 12.6171 | N/A | 19.6999 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -28.4644 | -32.0983 | 3.6339 | 0.3142 | 11.5670 | N/A | 19.7681 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -30.6980 | -34.0227 | 3.3247 | 0.3142 | 10.5828 | N/A | 19.6226 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -32.7494 | -35.8088 | 3.0594 | 0.3142 | 9.7385 | N/A | 19.3284 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -35.0154 | -37.8057 | 2.7903 | 0.3142 | 8.8818 | N/A | 18.8895 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -37.2769 | -39.8292 | 2.5523 | 0.3142 | 8.1242 | N/A | 18.4247 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -39.3215 | -41.6875 | 2.3660 | 0.3142 | 7.5313 | N/A | 18.0444 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -41.5550 | -43.7486 | 2.1935 | 0.3519 | 6.2331 | N/A | 17.7124 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -43.7702 | -45.8248 | 2.0545 | 0.5713 | 3.5961 | N/A | 17.5115 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -45.9774 | -47.9136 | 1.9362 | 0.7908 | 2.4485 | N/A | 17.3712 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | -47.9797 | -49.8220 | 1.8423 | 0.9903 | 1.8603 | N/A | 17.2783 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.500 | -50.1789 | -51.9300 | 1.7511 | 1.2099 | 1.4474 | N/A | 17.2073 | dirty_state | severe_error(1.00) | increase_p |  |
