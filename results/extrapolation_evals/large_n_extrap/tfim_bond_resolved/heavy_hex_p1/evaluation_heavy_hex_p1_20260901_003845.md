# Model Evaluation: heavy_hex

**Date**: 2026-09-01 00:38 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_res_mse_v2.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 2.5] (15 pts)
**Target N**: [8, 10, 20]

---

## N = 8 (15 params)

**ΔE/gap: 0.4554 ± 0.6520 | P90=1.1663 | max=2.4959
|ΔE|/N: 3.96e-02
Fidelity: mean=0.9282 min=0.7483 (exact)
Distribution: [P25=0.075 | P50=0.163 | P75=0.466 | P90=1.166]
Regions: critical=1.1316 | ordered=0.0591**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=2.496 is 5× the mean — median may be more representative N=8

**Fidelity: mean F=0.9282, min F=0.7483** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -9.1435 | -9.8947 | 0.7512 | 0.3010 | 2.4959 | 0.7483 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -9.9450 | -10.5644 | 0.6194 | 0.4521 | 1.3698 | 0.8167 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.210 | -10.6875 | -11.2098 | 0.5223 | 0.6067 | 0.8609 | 0.8613 | N/A | — | severe_error(0.85) | increase_p |  |
| 1.320 | -11.5128 | -11.9500 | 0.4372 | 0.7897 | 0.5536 | 0.8960 | N/A | — | severe_error(0.53) | increase_p |  |
| 1.430 | -12.3433 | -12.7141 | 0.3707 | 0.9819 | 0.3775 | 0.9201 | N/A | — | moderate_error(0.34) | refine_vqe |  |
| 1.540 | -13.1761 | -13.4964 | 0.3203 | 1.1807 | 0.2713 | 0.9370 | N/A | — | moderate_error(0.23) | refine_vqe |  |
| 1.640 | -13.9340 | -14.2201 | 0.2861 | 1.3653 | 0.2095 | 0.9478 | N/A | — | moderate_error(0.17) | refine_vqe |  |
| 1.750 | -14.7712 | -15.0271 | 0.2559 | 1.5716 | 0.1628 | 0.9568 | N/A | — | moderate_error(0.12) | refine_vqe |  |
| 1.860 | -15.6135 | -15.8433 | 0.2298 | 1.7804 | 0.1291 | 0.9640 | N/A | — | moderate_error(0.08) | refine_vqe |  |
| 1.960 | -16.3855 | -16.5920 | 0.2066 | 1.9719 | 0.1048 | 0.9695 | N/A | — | moderate_error(0.06) | refine_vqe |  |
| 2.070 | -17.2409 | -17.4216 | 0.1807 | 2.1840 | 0.0827 | 0.9750 | N/A | — | near_pass(0.03) | refine_vqe |  |
| 2.180 | -18.0958 | -18.2566 | 0.1608 | 2.3972 | 0.0671 | 0.9790 | N/A | — | near_pass(0.02) | refine_vqe |  |
| 2.290 | -18.9501 | -19.0961 | 0.1461 | 2.6114 | 0.0559 | 0.9819 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 2.390 | -19.7276 | -19.8627 | 0.1351 | 2.8068 | 0.0481 | 0.9841 | N/A | — | gap_masked(0.45) | refine_vqe |  |
| 2.500 | -20.5834 | -20.7091 | 0.1257 | 3.0223 | 0.0416 | 0.9860 | N/A | — | gap_masked(0.42) | refine_vqe |  |

## N = 10 (19 params)

**ΔE/gap: 1.1569 ± 2.2191 | P90=3.3765 | max=8.3892
|ΔE|/N: 5.00e-02
Fidelity: mean=0.8881 min=0.5119 (exact)
Distribution: [P25=0.074 | P50=0.154 | P75=0.730 | P90=3.377]
Regions: critical=3.2402 | ordered=0.0585**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=8.389 is 7× the mean — median may be more representative N=10

**Fidelity: mean F=0.8881, min F=0.5119** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -10.6255 | -12.4722 | 1.8467 | 0.2201 | 8.3892 | 0.5119 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -11.7722 | -13.2894 | 1.5172 | 0.3632 | 4.1770 | 0.6289 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -12.9629 | -14.0836 | 1.1207 | 0.5151 | 2.1758 | 0.7375 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -14.3146 | -14.9990 | 0.6845 | 0.6978 | 0.9809 | 0.8432 | N/A | — | severe_error(0.98) | increase_p |  |
| 1.430 | -15.5207 | -15.9469 | 0.4262 | 0.8912 | 0.4783 | 0.9057 | N/A | — | moderate_error(0.45) | refine_vqe |  |
| 1.540 | -16.6100 | -16.9194 | 0.3093 | 1.0915 | 0.2834 | 0.9360 | N/A | — | moderate_error(0.25) | refine_vqe |  |
| 1.640 | -17.5581 | -17.8199 | 0.2618 | 1.2777 | 0.2049 | 0.9500 | N/A | — | moderate_error(0.16) | refine_vqe |  |
| 1.750 | -18.5959 | -18.8250 | 0.2290 | 1.4858 | 0.1542 | 0.9598 | N/A | — | moderate_error(0.11) | refine_vqe |  |
| 1.860 | -19.6397 | -19.8422 | 0.2025 | 1.6962 | 0.1194 | 0.9671 | N/A | — | moderate_error(0.07) | refine_vqe |  |
| 1.960 | -20.5911 | -20.7757 | 0.1845 | 1.8891 | 0.0977 | 0.9720 | N/A | — | near_pass(0.05) | refine_vqe |  |
| 2.070 | -21.6410 | -21.8104 | 0.1694 | 2.1026 | 0.0806 | 0.9760 | N/A | — | near_pass(0.03) | refine_vqe |  |
| 2.180 | -22.6980 | -22.8521 | 0.1542 | 2.3171 | 0.0665 | 0.9796 | N/A | — | near_pass(0.02) | refine_vqe |  |
| 2.290 | -23.7572 | -23.8998 | 0.1425 | 2.5324 | 0.0563 | 0.9822 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 2.390 | -24.7248 | -24.8565 | 0.1317 | 2.7287 | 0.0483 | 0.9844 | N/A | — | gap_masked(0.44) | refine_vqe |  |
| 2.500 | -25.7930 | -25.9132 | 0.1202 | 2.9452 | 0.0408 | 0.9865 | N/A | — | gap_masked(0.40) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 8.7813 ± 5.2699 | P90=14.2127 | max=19.0454
|ΔE|/N: 1.47e-01
Var(H): 17.9158
Distribution: [P25=4.905 | P50=9.800 | P75=11.642 | P90=14.213]
Regions: critical=10.8167 | ordered=2.4903**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=159.0022.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -20.8068 | -25.3695 | 4.5627 | 0.3142 | 14.5235 | N/A | 16.9897 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -23.7004 | -26.9058 | 3.2054 | 0.3142 | 10.2032 | N/A | 13.5723 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -25.2285 | -28.4372 | 3.2088 | 0.3142 | 10.2138 | N/A | 14.4702 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -27.3186 | -30.2298 | 2.9113 | 0.3142 | 9.2668 | N/A | 13.9068 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -28.9956 | -32.0983 | 3.1027 | 0.3142 | 9.8763 | N/A | 16.6209 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -31.0312 | -34.0227 | 2.9916 | 0.3142 | 9.5225 | N/A | 17.4301 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -32.7299 | -35.8088 | 3.0789 | 0.3142 | 9.8004 | N/A | 19.3768 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -33.6998 | -37.8057 | 4.1059 | 0.3142 | 13.0696 | N/A | 27.2698 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -33.8459 | -39.8292 | 5.9833 | 0.3142 | 19.0454 | N/A | 40.6295 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -37.3689 | -41.6875 | 4.3186 | 0.3142 | 13.7465 | N/A | 32.1906 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -41.2156 | -43.7486 | 2.5330 | 0.3519 | 7.1976 | N/A | 20.6379 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -44.3321 | -45.8248 | 1.4927 | 0.5713 | 2.6127 | N/A | 12.8509 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -46.8913 | -47.9136 | 1.0223 | 0.7908 | 1.2928 | N/A | 9.1397 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.390 | -49.0334 | -49.8220 | 0.7886 | 0.9903 | 0.7963 | N/A | 7.2629 | dirty_state | severe_error(0.79) | increase_p |  |
| 2.500 | -51.2620 | -51.9300 | 0.6680 | 1.2099 | 0.5521 | N/A | 6.3892 | dirty_state | severe_error(0.53) | increase_p |  |
