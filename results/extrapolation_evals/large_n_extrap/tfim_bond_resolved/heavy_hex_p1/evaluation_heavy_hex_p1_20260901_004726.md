# Model Evaluation: heavy_hex

**Date**: 2026-09-01 00:47 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_res_film_energyw.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 2.5] (15 pts)
**Target N**: [8, 10, 20]

---

## N = 8 (15 params)

**ΔE/gap: 1.2430 ± 1.5591 | P90=3.0038 | max=6.0378
|ΔE|/N: 1.22e-01
Fidelity: mean=0.8282 min=0.5454 (exact)
Distribution: [P25=0.301 | P50=0.541 | P75=1.342 | P90=3.004]
Regions: critical=2.8985 | ordered=0.2535**

**Fidelity: mean F=0.8282, min F=0.5454** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -8.0775 | -9.8947 | 1.8172 | 0.3010 | 6.0378 | 0.5454 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -8.9923 | -10.5644 | 1.5721 | 0.4521 | 3.4769 | 0.6362 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -9.8179 | -11.2098 | 1.3919 | 0.6067 | 2.2942 | 0.7013 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -10.7187 | -11.9500 | 1.2312 | 0.7897 | 1.5592 | 0.7566 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.430 | -11.6100 | -12.7141 | 1.1041 | 0.9819 | 1.1244 | 0.7985 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.540 | -12.4960 | -13.4964 | 1.0005 | 1.1807 | 0.8474 | 0.8307 | N/A | — | severe_error(0.84) | increase_p |  |
| 1.640 | -13.2974 | -14.2201 | 0.9227 | 1.3653 | 0.6758 | 0.8537 | N/A | — | severe_error(0.66) | increase_p |  |
| 1.750 | -14.1767 | -15.0271 | 0.8504 | 1.5716 | 0.5411 | 0.8739 | N/A | — | severe_error(0.52) | increase_p |  |
| 1.860 | -15.0536 | -15.8433 | 0.7897 | 1.7804 | 0.4436 | 0.8902 | N/A | — | moderate_error(0.41) | refine_vqe |  |
| 1.960 | -15.8476 | -16.5920 | 0.7444 | 1.9719 | 0.3775 | 0.9021 | N/A | — | moderate_error(0.34) | refine_vqe |  |
| 2.070 | -16.7182 | -17.4216 | 0.7034 | 2.1840 | 0.3221 | 0.9127 | N/A | — | moderate_error(0.29) | refine_vqe |  |
| 2.180 | -17.5871 | -18.2566 | 0.6695 | 2.3972 | 0.2793 | 0.9213 | N/A | — | moderate_error(0.24) | refine_vqe |  |
| 2.290 | -18.4546 | -19.0961 | 0.6415 | 2.6114 | 0.2456 | 0.9285 | N/A | — | moderate_error(0.21) | refine_vqe |  |
| 2.390 | -19.2418 | -19.8627 | 0.6209 | 2.8068 | 0.2212 | 0.9338 | N/A | — | moderate_error(0.18) | refine_vqe |  |
| 2.500 | -20.1063 | -20.7091 | 0.6029 | 3.0223 | 0.1995 | 0.9388 | N/A | — | moderate_error(0.16) | refine_vqe |  |

## N = 10 (19 params)

**ΔE/gap: 1.9672 ± 2.7675 | P90=4.7854 | max=10.8219
|ΔE|/N: 1.27e-01
Fidelity: mean=0.7819 min=0.4313 (exact)
Distribution: [P25=0.407 | P50=0.739 | P75=1.933 | P90=4.785]
Regions: critical=4.7663 | ordered=0.3426**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=10.822 is 6× the mean — median may be more representative N=10

**Fidelity: mean F=0.7819, min F=0.4313** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -10.0899 | -12.4722 | 2.3822 | 0.2201 | 10.8219 | 0.4313 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -11.2409 | -13.2894 | 2.0485 | 0.3632 | 5.6398 | 0.5412 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -12.2789 | -14.0836 | 1.8047 | 0.5151 | 3.5038 | 0.6222 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -13.4112 | -14.9990 | 1.5878 | 0.6978 | 2.2755 | 0.6920 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -14.5294 | -15.9469 | 1.4175 | 0.8912 | 1.5906 | 0.7450 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.540 | -15.6339 | -16.9194 | 1.2855 | 1.0915 | 1.1777 | 0.7851 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.640 | -16.6322 | -17.8199 | 1.1878 | 1.2777 | 0.9296 | 0.8137 | N/A | — | severe_error(0.93) | increase_p |  |
| 1.750 | -17.7265 | -18.8250 | 1.0985 | 1.4858 | 0.7393 | 0.8389 | N/A | — | severe_error(0.73) | increase_p |  |
| 1.860 | -18.8176 | -19.8422 | 1.0246 | 1.6962 | 0.6041 | 0.8589 | N/A | — | severe_error(0.58) | increase_p |  |
| 1.960 | -19.8068 | -20.7757 | 0.9689 | 1.8891 | 0.5129 | 0.8737 | N/A | — | severe_error(0.49) | increase_p |  |
| 2.070 | -20.8928 | -21.8104 | 0.9176 | 2.1026 | 0.4364 | 0.8870 | N/A | — | moderate_error(0.41) | refine_vqe |  |
| 2.180 | -21.9771 | -22.8521 | 0.8751 | 2.3171 | 0.3777 | 0.8980 | N/A | — | moderate_error(0.34) | refine_vqe |  |
| 2.290 | -23.0595 | -23.8998 | 0.8402 | 2.5324 | 0.3318 | 0.9070 | N/A | — | moderate_error(0.30) | refine_vqe |  |
| 2.390 | -24.0423 | -24.8565 | 0.8142 | 2.7287 | 0.2984 | 0.9138 | N/A | — | moderate_error(0.26) | refine_vqe |  |
| 2.500 | -25.1214 | -25.9132 | 0.7918 | 2.9452 | 0.2688 | 0.9201 | N/A | — | moderate_error(0.23) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 7.6564 ± 4.2797 | P90=13.1913 | max=16.3314
|ΔE|/N: 1.36e-01
Var(H): 16.4145
Distribution: [P25=4.461 | P50=7.447 | P75=10.137 | P90=13.191]
Regions: critical=12.5287 | ordered=2.8919**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=126.3343.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -20.2389 | -25.3695 | 5.1307 | 0.3142 | 16.3314 | N/A | 18.0346 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -22.5459 | -26.9058 | 4.3599 | 0.3142 | 13.8781 | N/A | 17.4468 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -24.6167 | -28.4372 | 3.8205 | 0.3142 | 12.1610 | N/A | 16.9605 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -26.8680 | -30.2298 | 3.3619 | 0.3142 | 10.7012 | N/A | 16.5037 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -29.0912 | -32.0983 | 3.0071 | 0.3142 | 9.5720 | N/A | 16.1656 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -31.2952 | -34.0227 | 2.7275 | 0.3142 | 8.6819 | N/A | 15.9273 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -33.2855 | -35.8088 | 2.5233 | 0.3142 | 8.0319 | N/A | 15.7969 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -35.4662 | -37.8057 | 2.3395 | 0.3142 | 7.4468 | N/A | 15.7239 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -37.6393 | -39.8292 | 2.1899 | 0.3142 | 6.9708 | N/A | 15.7248 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -39.6103 | -41.6875 | 2.0772 | 0.3142 | 6.6119 | N/A | 15.7822 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -41.7738 | -43.7486 | 1.9747 | 0.3519 | 5.6113 | N/A | 15.9099 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -43.9329 | -45.8248 | 1.8919 | 0.5713 | 3.3114 | N/A | 16.1087 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -46.0883 | -47.9136 | 1.8254 | 0.7908 | 2.3083 | N/A | 16.3768 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | -48.0453 | -49.8220 | 1.7767 | 0.9903 | 1.7940 | N/A | 16.6771 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.500 | -50.1947 | -51.9300 | 1.7353 | 1.2099 | 1.4343 | N/A | 17.0792 | dirty_state | severe_error(1.00) | increase_p |  |
