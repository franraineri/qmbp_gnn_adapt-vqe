# Model Evaluation: heavy_hex

**Date**: 2026-09-01 00:54 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_res_film_deploy.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 2.5] (15 pts)
**Target N**: [8, 10, 20]

---

## N = 8 (15 params)

**ΔE/gap: 0.8365 ± 0.8735 | P90=1.8167 | max=3.4846
|ΔE|/N: 9.38e-02
Fidelity: mean=0.8640 min=0.6756 (exact)
Distribution: [P25=0.280 | P50=0.445 | P75=0.973 | P90=1.817]
Regions: critical=1.7887 | ordered=0.2413**

**Fidelity: mean F=0.8640, min F=0.6756** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -8.8460 | -9.8947 | 1.0487 | 0.3010 | 3.4846 | 0.6756 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -9.6334 | -10.5644 | 0.9310 | 0.4521 | 2.0590 | 0.7454 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -10.3282 | -11.2098 | 0.8816 | 0.6067 | 1.4532 | 0.7862 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.320 | -11.0916 | -11.9500 | 0.8584 | 0.7897 | 1.0870 | 0.8161 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.430 | -11.8698 | -12.7141 | 0.8443 | 0.9819 | 0.8598 | 0.8380 | N/A | — | severe_error(0.85) | increase_p |  |
| 1.540 | -12.6877 | -13.4964 | 0.8087 | 1.1807 | 0.6850 | 0.8584 | N/A | — | severe_error(0.67) | increase_p |  |
| 1.640 | -13.4615 | -14.2201 | 0.7586 | 1.3653 | 0.5556 | 0.8765 | N/A | — | severe_error(0.53) | increase_p |  |
| 1.750 | -14.3282 | -15.0271 | 0.6989 | 1.5716 | 0.4447 | 0.8940 | N/A | — | moderate_error(0.42) | refine_vqe |  |
| 1.860 | -15.1697 | -15.8433 | 0.6737 | 1.7804 | 0.3784 | 0.9049 | N/A | — | moderate_error(0.35) | refine_vqe |  |
| 1.960 | -15.9337 | -16.5920 | 0.6584 | 1.9719 | 0.3339 | 0.9125 | N/A | — | moderate_error(0.30) | refine_vqe |  |
| 2.070 | -16.7765 | -17.4216 | 0.6452 | 2.1840 | 0.2954 | 0.9195 | N/A | — | moderate_error(0.26) | refine_vqe |  |
| 2.180 | -17.6231 | -18.2566 | 0.6335 | 2.3972 | 0.2643 | 0.9254 | N/A | — | moderate_error(0.23) | refine_vqe |  |
| 2.290 | -18.4770 | -19.0961 | 0.6191 | 2.6114 | 0.2371 | 0.9309 | N/A | — | moderate_error(0.20) | refine_vqe |  |
| 2.390 | -19.2599 | -19.8627 | 0.6028 | 2.8068 | 0.2148 | 0.9358 | N/A | — | moderate_error(0.17) | refine_vqe |  |
| 2.500 | -20.1198 | -20.7091 | 0.5893 | 3.0223 | 0.1950 | 0.9403 | N/A | — | moderate_error(0.15) | refine_vqe |  |

## N = 10 (19 params)

**ΔE/gap: 0.9582 ± 1.6253 | P90=2.7218 | max=6.1085
|ΔE|/N: 4.72e-02
Fidelity: mean=0.8918 min=0.5881 (exact)
Distribution: [P25=0.081 | P50=0.206 | P75=0.825 | P90=2.722]
Regions: critical=2.5861 | ordered=0.0601**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=6.108 is 6× the mean — median may be more representative N=10

**Fidelity: mean F=0.8918, min F=0.5881** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -11.1275 | -12.4722 | 1.3447 | 0.2201 | 6.1085 | 0.5881 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -12.1031 | -13.2894 | 1.1863 | 0.3632 | 3.2659 | 0.6770 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -13.1021 | -14.0836 | 0.9815 | 0.5151 | 1.9056 | 0.7541 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.320 | -14.2641 | -14.9990 | 0.7349 | 0.6978 | 1.0532 | 0.8295 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.430 | -15.4146 | -15.9469 | 0.5323 | 0.8912 | 0.5974 | 0.8854 | N/A | — | severe_error(0.58) | increase_p |  |
| 1.540 | -16.5079 | -16.9194 | 0.4115 | 1.0915 | 0.3770 | 0.9194 | N/A | — | moderate_error(0.34) | refine_vqe |  |
| 1.640 | -17.4663 | -17.8199 | 0.3536 | 1.2777 | 0.2768 | 0.9369 | N/A | — | moderate_error(0.24) | refine_vqe |  |
| 1.750 | -18.5181 | -18.8250 | 0.3068 | 1.4858 | 0.2065 | 0.9502 | N/A | — | moderate_error(0.16) | refine_vqe |  |
| 1.860 | -19.5719 | -19.8422 | 0.2703 | 1.6962 | 0.1594 | 0.9595 | N/A | — | moderate_error(0.12) | refine_vqe |  |
| 1.960 | -20.5449 | -20.7757 | 0.2308 | 1.8891 | 0.1222 | 0.9674 | N/A | — | moderate_error(0.08) | refine_vqe |  |
| 2.070 | -21.6142 | -21.8104 | 0.1962 | 2.1026 | 0.0933 | 0.9739 | N/A | — | near_pass(0.05) | refine_vqe |  |
| 2.180 | -22.6944 | -22.8521 | 0.1577 | 2.3171 | 0.0681 | 0.9800 | N/A | — | near_pass(0.02) | refine_vqe |  |
| 2.290 | -23.7662 | -23.8998 | 0.1336 | 2.5324 | 0.0528 | 0.9837 | N/A | — | near_pass(0.00) | refine_vqe |  |
| 2.390 | -24.7322 | -24.8565 | 0.1244 | 2.7287 | 0.0456 | 0.9855 | N/A | — | gap_masked(0.41) | refine_vqe |  |
| 2.500 | -25.7923 | -25.9132 | 0.1209 | 2.9452 | 0.0410 | 0.9866 | N/A | — | gap_masked(0.40) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 6.3037 ± 3.6713 | P90=11.1213 | max=11.9964
|ΔE|/N: 1.12e-01
Var(H): 14.7285
Distribution: [P25=2.897 | P50=6.217 | P75=9.155 | P90=11.121]
Regions: critical=5.7999 | ordered=2.7716**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=116.0839.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -24.2483 | -25.3695 | 1.1212 | 0.3142 | 3.5689 | N/A | 2.4937 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -26.0960 | -26.9058 | 0.8097 | 0.3142 | 2.5775 | N/A | 2.2113 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -25.6173 | -28.4372 | 2.8199 | 0.3142 | 8.9760 | N/A | 12.6947 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -28.3672 | -30.2298 | 1.8627 | 0.3142 | 5.9291 | N/A | 9.1433 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -29.6014 | -32.0983 | 2.4969 | 0.3142 | 7.9480 | N/A | 13.4299 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -30.4506 | -34.0227 | 3.5721 | 0.3142 | 11.3705 | N/A | 21.0171 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -32.0400 | -35.8088 | 3.7688 | 0.3142 | 11.9964 | N/A | 23.7388 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -34.4292 | -37.8057 | 3.3765 | 0.3142 | 10.7476 | N/A | 22.9039 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -36.8966 | -39.8292 | 2.9327 | 0.3142 | 9.3349 | N/A | 21.2774 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -39.0961 | -41.6875 | 2.5914 | 0.3142 | 8.2487 | N/A | 19.8233 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -41.5606 | -43.7486 | 2.1879 | 0.3519 | 6.2171 | N/A | 17.6489 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -43.9870 | -45.8248 | 1.8378 | 0.5713 | 3.2167 | N/A | 15.5850 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -46.3120 | -47.9136 | 1.6017 | 0.7908 | 2.0254 | N/A | 14.2824 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | -48.4410 | -49.8220 | 1.3810 | 0.9903 | 1.3945 | N/A | 12.8507 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.500 | -50.7151 | -51.9300 | 1.2149 | 1.2099 | 1.0042 | N/A | 11.8267 | dirty_state | severe_error(1.00) | increase_p |  |
