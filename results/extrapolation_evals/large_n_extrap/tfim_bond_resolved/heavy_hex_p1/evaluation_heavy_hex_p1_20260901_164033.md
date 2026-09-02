# Model Evaluation: heavy_hex

**Date**: 2026-09-01 16:40 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_plain_energyw.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 2.5] (15 pts)
**Target N**: [8, 10, 20]

---

## N = 8 (15 params)

**ΔE/gap: 0.7803 ± 0.7423 | P90=1.5035 | max=3.1520
|ΔE|/N: 9.62e-02
Fidelity: mean=0.8655 min=0.7008 (exact)
Distribution: [P25=0.346 | P50=0.521 | P75=0.794 | P90=1.504]
Regions: critical=1.5253 | ordered=0.2901**

**Fidelity: mean F=0.8655, min F=0.7008** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -8.9460 | -9.8947 | 0.9487 | 0.3010 | 3.1520 | 0.7008 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -9.7762 | -10.5644 | 0.7882 | 0.4521 | 1.7432 | 0.7763 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.210 | -10.5157 | -11.2098 | 0.6941 | 0.6067 | 1.1440 | 0.8220 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.320 | -11.2798 | -11.9500 | 0.6702 | 0.7897 | 0.8487 | 0.8479 | N/A | — | severe_error(0.84) | increase_p |  |
| 1.430 | -11.9887 | -12.7141 | 0.7253 | 0.9819 | 0.7387 | 0.8560 | N/A | — | severe_error(0.72) | increase_p |  |
| 1.540 | -12.7354 | -13.4964 | 0.7611 | 1.1807 | 0.6446 | 0.8648 | N/A | — | severe_error(0.63) | increase_p |  |
| 1.640 | -13.4195 | -14.2201 | 0.8006 | 1.3653 | 0.5864 | 0.8701 | N/A | — | severe_error(0.56) | increase_p |  |
| 1.750 | -14.2089 | -15.0271 | 0.8182 | 1.5716 | 0.5206 | 0.8779 | N/A | — | severe_error(0.50) | increase_p |  |
| 1.860 | -15.0257 | -15.8433 | 0.8176 | 1.7804 | 0.4592 | 0.8867 | N/A | — | moderate_error(0.43) | refine_vqe |  |
| 1.960 | -15.7710 | -16.5920 | 0.8210 | 1.9719 | 0.4164 | 0.8930 | N/A | — | moderate_error(0.39) | refine_vqe |  |
| 2.070 | -16.6208 | -17.4216 | 0.8009 | 2.1840 | 0.3667 | 0.9017 | N/A | — | moderate_error(0.33) | refine_vqe |  |
| 2.180 | -17.4775 | -18.2566 | 0.7791 | 2.3972 | 0.3250 | 0.9095 | N/A | — | moderate_error(0.29) | refine_vqe |  |
| 2.290 | -18.3484 | -19.0961 | 0.7477 | 2.6114 | 0.2863 | 0.9175 | N/A | — | moderate_error(0.25) | refine_vqe |  |
| 2.390 | -19.1553 | -19.8627 | 0.7074 | 2.8068 | 0.2520 | 0.9252 | N/A | — | moderate_error(0.21) | refine_vqe |  |
| 2.500 | -20.0438 | -20.7091 | 0.6654 | 3.0223 | 0.2202 | 0.9327 | N/A | — | moderate_error(0.18) | refine_vqe |  |

## N = 10 (19 params)

**ΔE/gap: 1.1307 ± 1.4591 | P90=3.4463 | max=4.8654
|ΔE|/N: 7.08e-02
Fidelity: mean=0.8547 min=0.6251 (exact)
Distribution: [P25=0.179 | P50=0.404 | P75=1.255 | P90=3.446]
Regions: critical=2.8156 | ordered=0.1399**

**Fidelity: mean F=0.8547, min F=0.6251** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -11.4012 | -12.4722 | 1.0710 | 0.2201 | 4.8654 | 0.6251 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -11.8991 | -13.2894 | 1.3903 | 0.3632 | 3.8277 | 0.6376 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -12.6032 | -14.0836 | 1.4804 | 0.5151 | 2.8743 | 0.6693 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -13.9069 | -14.9990 | 1.0921 | 0.6978 | 1.5651 | 0.7659 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.430 | -15.1043 | -15.9469 | 0.8426 | 0.8912 | 0.9456 | 0.8295 | N/A | — | severe_error(0.94) | increase_p |  |
| 1.540 | -16.1667 | -16.9194 | 0.7527 | 1.0915 | 0.6896 | 0.8605 | N/A | — | severe_error(0.67) | increase_p |  |
| 1.640 | -17.1405 | -17.8199 | 0.6794 | 1.2777 | 0.5317 | 0.8829 | N/A | — | severe_error(0.51) | increase_p |  |
| 1.750 | -18.2249 | -18.8250 | 0.6001 | 1.4858 | 0.4039 | 0.9037 | N/A | — | moderate_error(0.37) | refine_vqe |  |
| 1.860 | -19.3150 | -19.8422 | 0.5272 | 1.6962 | 0.3108 | 0.9208 | N/A | — | moderate_error(0.27) | refine_vqe |  |
| 1.960 | -20.3085 | -20.7757 | 0.4671 | 1.8891 | 0.2473 | 0.9337 | N/A | — | moderate_error(0.21) | refine_vqe |  |
| 2.070 | -21.3962 | -21.8104 | 0.4142 | 2.1026 | 0.1970 | 0.9446 | N/A | — | moderate_error(0.15) | refine_vqe |  |
| 2.180 | -22.4809 | -22.8521 | 0.3712 | 2.3171 | 0.1602 | 0.9531 | N/A | — | moderate_error(0.12) | refine_vqe |  |
| 2.290 | -23.5631 | -23.8998 | 0.3366 | 2.5324 | 0.1329 | 0.9597 | N/A | — | moderate_error(0.09) | refine_vqe |  |
| 2.390 | -24.5471 | -24.8565 | 0.3094 | 2.7287 | 0.1134 | 0.9647 | N/A | — | moderate_error(0.07) | refine_vqe |  |
| 2.500 | -25.6310 | -25.9132 | 0.2822 | 2.9452 | 0.0958 | 0.9694 | N/A | — | near_pass(0.05) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 5.3049 ± 3.4946 | P90=8.8507 | max=15.2804
|ΔE|/N: 1.00e-01
Var(H): 13.2878
Distribution: [P25=2.667 | P50=3.932 | P75=6.523 | P90=8.851]
Regions: critical=7.4941 | ordered=3.3048**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=91.0141.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -20.5691 | -25.3695 | 4.8005 | 0.3142 | 15.2804 | N/A | 19.9035 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -23.7569 | -26.9058 | 3.1489 | 0.3142 | 10.0232 | N/A | 14.8204 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -26.5367 | -28.4372 | 1.9005 | 0.3142 | 6.0494 | N/A | 9.4330 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -29.1276 | -30.2298 | 1.1022 | 0.3142 | 3.5084 | N/A | 5.3929 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -31.2787 | -32.0983 | 0.8196 | 0.3142 | 2.6089 | N/A | 4.0561 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -33.1736 | -34.0227 | 0.8492 | 0.3142 | 2.7030 | N/A | 4.4891 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -34.5793 | -35.8088 | 1.2295 | 0.3142 | 3.9137 | N/A | 7.3585 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -36.1006 | -37.8057 | 1.7051 | 0.3142 | 5.4274 | N/A | 11.3111 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -37.8051 | -39.8292 | 2.0241 | 0.3142 | 6.4430 | N/A | 14.4947 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -39.4595 | -41.6875 | 2.2280 | 0.3142 | 7.0919 | N/A | 16.9248 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -41.4247 | -43.7486 | 2.3238 | 0.3519 | 6.6033 | N/A | 18.7111 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -43.5782 | -45.8248 | 2.2466 | 0.5713 | 3.9323 | N/A | 19.0925 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -45.8330 | -47.9136 | 2.0806 | 0.7908 | 2.6311 | N/A | 18.6013 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | -47.9032 | -49.8220 | 1.9187 | 0.9903 | 1.9375 | N/A | 17.9244 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.500 | -50.2122 | -51.9300 | 1.7178 | 1.2099 | 1.4198 | N/A | 16.8036 | dirty_state | severe_error(1.00) | increase_p |  |
