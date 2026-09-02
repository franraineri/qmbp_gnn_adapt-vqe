# Model Evaluation: heavy_hex

**Date**: 2026-09-01 16:39 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_res_film_mse_v2.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 2.5] (15 pts)
**Target N**: [8, 10, 20]

---

## N = 8 (15 params)

**ΔE/gap: 1.2741 ± 1.3007 | P90=2.7542 | max=5.2090
|ΔE|/N: 1.45e-01
Fidelity: mean=0.8079 min=0.5835 (exact)
Distribution: [P25=0.444 | P50=0.721 | P75=1.448 | P90=2.754]
Regions: critical=2.6836 | ordered=0.3846**

**Fidelity: mean F=0.8079, min F=0.5835** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -8.3270 | -9.8947 | 1.5678 | 0.3010 | 5.2090 | 0.5835 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -9.1426 | -10.5644 | 1.4218 | 0.4521 | 3.1445 | 0.6593 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -9.8939 | -11.2098 | 1.3159 | 0.6067 | 2.1689 | 0.7123 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -10.6786 | -11.9500 | 1.2713 | 0.7897 | 1.6100 | 0.7498 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.430 | -11.4516 | -12.7141 | 1.2624 | 0.9819 | 1.2856 | 0.7748 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.540 | -12.2739 | -13.4964 | 1.2226 | 1.1807 | 1.0355 | 0.7994 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.640 | -13.0391 | -14.2201 | 1.1810 | 1.3653 | 0.8650 | 0.8191 | N/A | — | severe_error(0.86) | increase_p |  |
| 1.750 | -13.8932 | -15.0271 | 1.1339 | 1.5716 | 0.7215 | 0.8379 | N/A | — | severe_error(0.71) | increase_p |  |
| 1.860 | -14.7518 | -15.8433 | 1.0916 | 1.7804 | 0.6131 | 0.8536 | N/A | — | severe_error(0.59) | increase_p |  |
| 1.960 | -15.5351 | -16.5920 | 1.0569 | 1.9719 | 0.5360 | 0.8657 | N/A | — | severe_error(0.51) | increase_p |  |
| 2.070 | -16.3952 | -17.4216 | 1.0264 | 2.1840 | 0.4700 | 0.8768 | N/A | — | moderate_error(0.44) | refine_vqe |  |
| 2.180 | -17.2535 | -18.2566 | 1.0031 | 2.3972 | 0.4185 | 0.8859 | N/A | — | moderate_error(0.39) | refine_vqe |  |
| 2.290 | -18.1156 | -19.0961 | 0.9806 | 2.6114 | 0.3755 | 0.8940 | N/A | — | moderate_error(0.34) | refine_vqe |  |
| 2.390 | -18.8972 | -19.8627 | 0.9655 | 2.8068 | 0.3440 | 0.9001 | N/A | — | moderate_error(0.31) | refine_vqe |  |
| 2.500 | -19.7571 | -20.7091 | 0.9521 | 3.0223 | 0.3150 | 0.9060 | N/A | — | moderate_error(0.28) | refine_vqe |  |

## N = 10 (19 params)

**ΔE/gap: 1.9728 ± 2.8898 | P90=5.0749 | max=11.0650
|ΔE|/N: 1.18e-01
Fidelity: mean=0.7915 min=0.4241 (exact)
Distribution: [P25=0.314 | P50=0.627 | P75=2.025 | P90=5.075]
Regions: critical=4.9644 | ordered=0.2537**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=11.065 is 6× the mean — median may be more representative N=10

**Fidelity: mean F=0.7915, min F=0.4241** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -10.0364 | -12.4722 | 2.4357 | 0.2201 | 11.0650 | 0.4241 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -11.1249 | -13.2894 | 2.1645 | 0.3632 | 5.9591 | 0.5255 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -12.1528 | -14.0836 | 1.9308 | 0.5151 | 3.7486 | 0.6042 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -13.3063 | -14.9990 | 1.6927 | 0.6978 | 2.4259 | 0.6766 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -14.5003 | -15.9469 | 1.4466 | 0.8912 | 1.6233 | 0.7405 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.540 | -15.6885 | -16.9194 | 1.2308 | 1.0915 | 1.1277 | 0.7924 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.640 | -16.7444 | -17.8199 | 1.0755 | 1.2777 | 0.8417 | 0.8287 | N/A | — | severe_error(0.83) | increase_p |  |
| 1.750 | -17.8930 | -18.8250 | 0.9320 | 1.4858 | 0.6273 | 0.8604 | N/A | — | severe_error(0.61) | increase_p |  |
| 1.860 | -19.0017 | -19.8422 | 0.8405 | 1.6962 | 0.4955 | 0.8817 | N/A | — | moderate_error(0.47) | refine_vqe |  |
| 1.960 | -20.0017 | -20.7757 | 0.7740 | 1.8891 | 0.4097 | 0.8968 | N/A | — | moderate_error(0.38) | refine_vqe |  |
| 2.070 | -21.0939 | -21.8104 | 0.7165 | 2.1026 | 0.3408 | 0.9098 | N/A | — | moderate_error(0.31) | refine_vqe |  |
| 2.180 | -22.1863 | -22.8521 | 0.6658 | 2.3171 | 0.2873 | 0.9206 | N/A | — | moderate_error(0.25) | refine_vqe |  |
| 2.290 | -23.2806 | -23.8998 | 0.6192 | 2.5324 | 0.2445 | 0.9298 | N/A | — | moderate_error(0.20) | refine_vqe |  |
| 2.390 | -24.2771 | -24.8565 | 0.5794 | 2.7287 | 0.2123 | 0.9372 | N/A | — | moderate_error(0.17) | refine_vqe |  |
| 2.500 | -25.3720 | -25.9132 | 0.5412 | 2.9452 | 0.1838 | 0.9441 | N/A | — | moderate_error(0.14) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 9.9610 ± 5.1822 | P90=17.1158 | max=19.1278
|ΔE|/N: 1.79e-01
Var(H): 21.8429
Distribution: [P25=6.170 | P50=9.779 | P75=12.985 | P90=17.116]
Regions: critical=15.7880 | ordered=4.0495**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=164.3271.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -19.3604 | -25.3695 | 6.0092 | 0.3142 | 19.1278 | N/A | 19.9797 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -21.2838 | -26.9058 | 5.6220 | 0.3142 | 17.8954 | N/A | 21.1258 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -23.4275 | -28.4372 | 5.0097 | 0.3142 | 15.9463 | N/A | 21.5739 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -26.0171 | -30.2298 | 4.2128 | 0.3142 | 13.4096 | N/A | 20.8550 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -28.1522 | -32.0983 | 3.9461 | 0.3142 | 12.5607 | N/A | 21.5754 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -30.3100 | -34.0227 | 3.7127 | 0.3142 | 11.8180 | N/A | 22.0155 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -32.4245 | -35.8088 | 3.3843 | 0.3142 | 10.7727 | N/A | 21.4727 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -34.7337 | -37.8057 | 3.0720 | 0.3142 | 9.7785 | N/A | 20.8657 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -36.9704 | -39.8292 | 2.8588 | 0.3142 | 9.0999 | N/A | 20.6873 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -38.9356 | -41.6875 | 2.7519 | 0.3142 | 8.7594 | N/A | 21.0399 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -41.0483 | -43.7486 | 2.7003 | 0.3519 | 7.6730 | N/A | 21.8709 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -43.1579 | -45.8248 | 2.6668 | 0.5713 | 4.6679 | N/A | 22.8109 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -45.3026 | -47.9136 | 2.6110 | 0.7908 | 3.3018 | N/A | 23.5127 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | -47.2821 | -49.8220 | 2.5398 | 0.9903 | 2.5647 | N/A | 23.9115 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -49.4619 | -51.9300 | 2.4682 | 1.2099 | 2.0400 | N/A | 24.3466 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
