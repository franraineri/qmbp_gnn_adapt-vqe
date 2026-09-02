# Model Evaluation: heavy_hex

**Date**: 2026-09-01 00:49 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_res_physics05.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 2.5] (15 pts)
**Target N**: [8, 10, 20]

---

## N = 8 (15 params)

**ΔE/gap: 1.1971 ± 1.4483 | P90=2.8558 | max=5.6135
|ΔE|/N: 1.20e-01
Fidelity: mean=0.8305 min=0.5645 (exact)
Distribution: [P25=0.309 | P50=0.549 | P75=1.322 | P90=2.856]
Regions: critical=2.7505 | ordered=0.2568**

**Fidelity: mean F=0.8305, min F=0.5645** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -8.2052 | -9.8947 | 1.6895 | 0.3010 | 5.6135 | 0.5645 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -9.0777 | -10.5644 | 1.4867 | 0.4521 | 3.2882 | 0.6494 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -9.8706 | -11.2098 | 1.3392 | 0.6067 | 2.2073 | 0.7094 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -10.7437 | -11.9500 | 1.2062 | 0.7897 | 1.5275 | 0.7604 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.430 | -11.6182 | -12.7141 | 1.0959 | 0.9819 | 1.1161 | 0.7996 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.540 | -12.4940 | -13.4964 | 1.0024 | 1.1807 | 0.8490 | 0.8304 | N/A | — | severe_error(0.84) | increase_p |  |
| 1.640 | -13.2900 | -14.2201 | 0.9302 | 1.3653 | 0.6813 | 0.8526 | N/A | — | severe_error(0.66) | increase_p |  |
| 1.750 | -14.1641 | -15.0271 | 0.8629 | 1.5716 | 0.5491 | 0.8723 | N/A | — | severe_error(0.53) | increase_p |  |
| 1.860 | -15.0368 | -15.8433 | 0.8065 | 1.7804 | 0.4530 | 0.8881 | N/A | — | moderate_error(0.42) | refine_vqe |  |
| 1.960 | -15.8290 | -16.5920 | 0.7630 | 1.9719 | 0.3869 | 0.8998 | N/A | — | moderate_error(0.35) | refine_vqe |  |
| 2.070 | -16.6996 | -17.4216 | 0.7220 | 2.1840 | 0.3306 | 0.9105 | N/A | — | moderate_error(0.30) | refine_vqe |  |
| 2.180 | -17.5692 | -18.2566 | 0.6874 | 2.3972 | 0.2867 | 0.9193 | N/A | — | moderate_error(0.25) | refine_vqe |  |
| 2.290 | -18.4385 | -19.0961 | 0.6576 | 2.6114 | 0.2518 | 0.9267 | N/A | — | moderate_error(0.21) | refine_vqe |  |
| 2.390 | -19.2402 | -19.8627 | 0.6225 | 2.8068 | 0.2218 | 0.9336 | N/A | — | moderate_error(0.18) | refine_vqe |  |
| 2.500 | -20.1257 | -20.7091 | 0.5834 | 3.0223 | 0.1930 | 0.9405 | N/A | — | moderate_error(0.15) | refine_vqe |  |

## N = 10 (19 params)

**ΔE/gap: 1.8379 ± 2.5372 | P90=4.4578 | max=9.9110
|ΔE|/N: 1.20e-01
Fidelity: mean=0.7902 min=0.4554 (exact)
Distribution: [P25=0.389 | P50=0.710 | P75=1.850 | P90=4.458]
Regions: critical=4.4275 | ordered=0.3246**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=9.911 is 5× the mean — median may be more representative N=10

**Fidelity: mean F=0.7902, min F=0.4554** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -10.2905 | -12.4722 | 2.1817 | 0.2201 | 9.9110 | 0.4554 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -11.3882 | -13.2894 | 1.9012 | 0.3632 | 5.2342 | 0.5608 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -12.3873 | -14.0836 | 1.6963 | 0.5151 | 3.2934 | 0.6372 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -13.4858 | -14.9990 | 1.5132 | 0.6978 | 2.1687 | 0.7024 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -14.5831 | -15.9469 | 1.3638 | 0.8912 | 1.5304 | 0.7524 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.540 | -15.6797 | -16.9194 | 1.2397 | 1.0915 | 1.1358 | 0.7912 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.640 | -16.6756 | -17.8199 | 1.1443 | 1.2777 | 0.8956 | 0.8193 | N/A | — | severe_error(0.89) | increase_p |  |
| 1.750 | -17.7705 | -18.8250 | 1.0544 | 1.4858 | 0.7097 | 0.8442 | N/A | — | severe_error(0.69) | increase_p |  |
| 1.860 | -18.8622 | -19.8422 | 0.9800 | 1.6962 | 0.5778 | 0.8641 | N/A | — | severe_error(0.56) | increase_p |  |
| 1.960 | -19.8507 | -20.7757 | 0.9250 | 1.8891 | 0.4896 | 0.8786 | N/A | — | moderate_error(0.46) | refine_vqe |  |
| 2.070 | -20.9334 | -21.8104 | 0.8770 | 2.1026 | 0.4171 | 0.8913 | N/A | — | moderate_error(0.39) | refine_vqe |  |
| 2.180 | -22.0147 | -22.8521 | 0.8374 | 2.3171 | 0.3614 | 0.9018 | N/A | — | moderate_error(0.33) | refine_vqe |  |
| 2.290 | -23.1038 | -23.8998 | 0.7959 | 2.5324 | 0.3143 | 0.9113 | N/A | — | moderate_error(0.28) | refine_vqe |  |
| 2.390 | -24.0921 | -24.8565 | 0.7644 | 2.7287 | 0.2801 | 0.9185 | N/A | — | moderate_error(0.24) | refine_vqe |  |
| 2.500 | -25.1772 | -25.9132 | 0.7360 | 2.9452 | 0.2499 | 0.9252 | N/A | — | moderate_error(0.21) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 6.8992 ± 3.9879 | P90=12.0578 | max=14.7349
|ΔE|/N: 1.21e-01
Var(H): 14.2862
Distribution: [P25=3.809 | P50=6.735 | P75=9.389 | P90=12.058]
Regions: critical=11.4667 | ordered=2.4094**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=112.7876.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -20.7404 | -25.3695 | 4.6291 | 0.3142 | 14.7349 | N/A | 15.9716 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -22.9328 | -26.9058 | 3.9730 | 0.3142 | 12.6465 | N/A | 15.6816 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -24.9265 | -28.4372 | 3.5107 | 0.3142 | 11.1748 | N/A | 15.4201 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -27.1184 | -30.2298 | 3.1114 | 0.3142 | 9.9040 | N/A | 15.1453 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -29.3107 | -32.0983 | 2.7876 | 0.3142 | 8.8731 | N/A | 14.8749 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -31.5038 | -34.0227 | 2.5190 | 0.3142 | 8.0181 | N/A | 14.6066 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -33.4975 | -35.8088 | 2.3113 | 0.3142 | 7.3570 | N/A | 14.3675 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -35.6897 | -37.8057 | 2.1159 | 0.3142 | 6.7352 | N/A | 14.1167 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -37.8797 | -39.8292 | 1.9496 | 0.3142 | 6.2057 | N/A | 13.8902 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -39.8681 | -41.6875 | 1.8194 | 0.3142 | 5.7913 | N/A | 13.7110 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -42.0503 | -43.7486 | 1.6982 | 0.3519 | 4.8256 | N/A | 13.5663 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -44.2291 | -45.8248 | 1.5957 | 0.5713 | 2.7930 | N/A | 13.4683 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -46.4125 | -47.9136 | 1.5011 | 0.7908 | 1.8983 | N/A | 13.3455 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.390 | -48.4028 | -49.8220 | 1.4191 | 0.9903 | 1.4330 | N/A | 13.1947 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.500 | -50.6025 | -51.9300 | 1.3275 | 1.2099 | 1.0972 | N/A | 12.9327 | dirty_state | severe_error(1.00) | increase_p |  |
