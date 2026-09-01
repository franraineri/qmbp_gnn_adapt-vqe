# Model Evaluation: heavy_hex

**Date**: 2026-08-31 23:57 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_res_film_mse.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 2.5] (15 pts)
**Target N**: [8, 10, 20]

---

## N = 8 (15 params)

**ΔE/gap: 1.3597 ± 1.6082 | P90=3.2152 | max=6.2415
|ΔE|/N: 1.39e-01
Fidelity: mean=0.8113 min=0.5364 (exact)
Distribution: [P25=0.363 | P50=0.647 | P75=1.515 | P90=3.215]
Regions: critical=3.0925 | ordered=0.3018**

**Fidelity: mean F=0.8113, min F=0.5364** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -8.0162 | -9.8947 | 1.8785 | 0.3010 | 6.2415 | 0.5364 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -8.8948 | -10.5644 | 1.6696 | 0.4521 | 3.6926 | 0.6214 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -9.6937 | -11.2098 | 1.5161 | 0.6067 | 2.4989 | 0.6825 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -10.5728 | -11.9500 | 1.3772 | 0.7897 | 1.7440 | 0.7350 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.430 | -11.4521 | -12.7141 | 1.2620 | 0.9819 | 1.2852 | 0.7758 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.540 | -12.3318 | -13.4964 | 1.1647 | 1.1807 | 0.9864 | 0.8080 | N/A | — | severe_error(0.99) | increase_p |  |
| 1.640 | -13.1316 | -14.2201 | 1.0885 | 1.3653 | 0.7973 | 0.8317 | N/A | — | severe_error(0.79) | increase_p |  |
| 1.750 | -14.0105 | -15.0271 | 1.0165 | 1.5716 | 0.6468 | 0.8529 | N/A | — | severe_error(0.63) | increase_p |  |
| 1.860 | -14.8899 | -15.8433 | 0.9534 | 1.7804 | 0.5355 | 0.8703 | N/A | — | severe_error(0.51) | increase_p |  |
| 1.960 | -15.6896 | -16.5920 | 0.9024 | 1.9719 | 0.4576 | 0.8836 | N/A | — | moderate_error(0.43) | refine_vqe |  |
| 2.070 | -16.5696 | -17.4216 | 0.8520 | 2.1840 | 0.3901 | 0.8961 | N/A | — | moderate_error(0.36) | refine_vqe |  |
| 2.180 | -17.4504 | -18.2566 | 0.8062 | 2.3972 | 0.3363 | 0.9067 | N/A | — | moderate_error(0.30) | refine_vqe |  |
| 2.290 | -18.3316 | -19.0961 | 0.7646 | 2.6114 | 0.2928 | 0.9158 | N/A | — | moderate_error(0.26) | refine_vqe |  |
| 2.390 | -19.1327 | -19.8627 | 0.7300 | 2.8068 | 0.2601 | 0.9230 | N/A | — | moderate_error(0.22) | refine_vqe |  |
| 2.500 | -20.0141 | -20.7091 | 0.6951 | 3.0223 | 0.2300 | 0.9300 | N/A | — | moderate_error(0.19) | refine_vqe |  |

## N = 10 (19 params)

**ΔE/gap: 1.1657 ± 0.9472 | P90=2.2635 | max=3.8134
|ΔE|/N: 1.18e-01
Fidelity: mean=0.8096 min=0.6006 (exact)
Distribution: [P25=0.485 | P50=0.801 | P75=1.578 | P90=2.264]
Regions: critical=2.1551 | ordered=0.4018**

**Fidelity: mean F=0.8096, min F=0.6006** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -12.0806 | -12.4722 | 0.3916 | 0.2201 | 1.7788 | 0.7958 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.110 | -12.9985 | -13.2894 | 0.2909 | 0.3632 | 0.8008 | 0.8709 | N/A | — | severe_error(0.79) | increase_p |  |
| 1.210 | -12.1195 | -14.0836 | 1.9642 | 0.5151 | 3.8134 | 0.6006 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -13.2177 | -14.9990 | 1.7813 | 0.6978 | 2.5528 | 0.6657 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -14.3164 | -15.9469 | 1.6305 | 0.8912 | 1.8296 | 0.7165 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.540 | -15.4157 | -16.9194 | 1.5036 | 1.0915 | 1.3776 | 0.7567 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.640 | -16.4143 | -17.8199 | 1.4056 | 1.2777 | 1.1001 | 0.7862 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.750 | -17.5131 | -18.8250 | 1.3119 | 1.4858 | 0.8830 | 0.8129 | N/A | — | severe_error(0.88) | increase_p |  |
| 1.860 | -18.6122 | -19.8422 | 1.2300 | 1.6962 | 0.7252 | 0.8348 | N/A | — | severe_error(0.71) | increase_p |  |
| 1.960 | -19.6118 | -20.7757 | 1.1639 | 1.8891 | 0.6161 | 0.8516 | N/A | — | severe_error(0.60) | increase_p |  |
| 2.070 | -20.7119 | -21.8104 | 1.0985 | 2.1026 | 0.5224 | 0.8673 | N/A | — | severe_error(0.50) | increase_p |  |
| 2.180 | -21.8129 | -22.8521 | 1.0392 | 2.3171 | 0.4485 | 0.8808 | N/A | — | moderate_error(0.42) | refine_vqe |  |
| 2.290 | -22.9144 | -23.8998 | 0.9853 | 2.5324 | 0.3891 | 0.8924 | N/A | — | moderate_error(0.36) | refine_vqe |  |
| 2.390 | -23.9159 | -24.8565 | 0.9406 | 2.7287 | 0.3447 | 0.9015 | N/A | — | moderate_error(0.31) | refine_vqe |  |
| 2.500 | -25.0176 | -25.9132 | 0.8956 | 2.9452 | 0.3041 | 0.9103 | N/A | — | moderate_error(0.27) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 8.6315 ± 4.5361 | P90=14.2537 | max=16.9494
|ΔE|/N: 1.53e-01
Var(H): 18.7656
Distribution: [P25=5.217 | P50=8.835 | P75=11.552 | P90=14.254]
Regions: critical=13.6531 | ordered=3.3269**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=145.0358.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -20.0447 | -25.3695 | 5.3248 | 0.3142 | 16.9494 | N/A | 18.8208 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -22.2417 | -26.9058 | 4.6641 | 0.3142 | 14.8464 | N/A | 18.8148 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -24.2385 | -28.4372 | 4.1987 | 0.3142 | 13.3648 | N/A | 18.8133 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -26.4359 | -30.2298 | 3.7940 | 0.3142 | 12.0766 | N/A | 18.8105 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -28.6337 | -32.0983 | 3.4646 | 0.3142 | 11.0283 | N/A | 18.8072 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -30.8327 | -34.0227 | 3.1900 | 0.3142 | 10.1542 | N/A | 18.7983 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -32.8313 | -35.8088 | 2.9775 | 0.3142 | 9.4778 | N/A | 18.7946 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -35.0300 | -37.8057 | 2.7757 | 0.3142 | 8.8354 | N/A | 18.7902 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -37.2289 | -39.8292 | 2.6003 | 0.3142 | 8.2771 | N/A | 18.7849 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -39.2282 | -41.6875 | 2.4593 | 0.3142 | 7.8281 | N/A | 18.7788 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -41.4284 | -43.7486 | 2.3202 | 0.3519 | 6.5929 | N/A | 18.7648 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -43.6301 | -45.8248 | 2.1947 | 0.5713 | 3.8414 | N/A | 18.7377 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -45.8328 | -47.9136 | 2.0808 | 0.7908 | 2.6313 | N/A | 18.6993 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | -47.8356 | -49.8220 | 1.9863 | 0.9903 | 2.0057 | N/A | 18.6591 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -50.0390 | -51.9300 | 1.8911 | 1.2099 | 1.5630 | N/A | 18.6100 | dirty_state | severe_error(1.00) | increase_p |  |
