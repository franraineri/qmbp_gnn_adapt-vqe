# Model Evaluation: heavy_hex

> **🌐 MULTI-TOPOLOGY MODEL** — This evaluation uses a model trained on multiple topologies simultaneously. Results reflect cross-topology transfer capability.

**Date**: 2026-09-01 16:34 UTC
**Model**: data/model_zoo/checkpoints/unified_tfim_br_MT_residual+film_p1.pt
**p_layers**: 1
**Multi-topology**: YES
**h-range**: [1.0, 2.5] (15 pts)
**Target N**: [8, 10, 20]

---

## N = 8 (15 params)

**ΔE/gap: 0.6666 ± 0.9694 | P90=1.7293 | max=3.6910
|ΔE|/N: 5.64e-02
Fidelity: mean=0.8996 min=0.6579 (exact)
Distribution: [P25=0.097 | P50=0.225 | P75=0.696 | P90=1.729]
Regions: critical=1.6783 | ordered=0.0743**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=3.691 is 6× the mean — median may be more representative N=8

**Fidelity: mean F=0.8996, min F=0.6579** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -8.7838 | -9.8947 | 1.1109 | 0.3010 | 3.6910 | 0.6579 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -9.6474 | -10.5644 | 0.9170 | 0.4521 | 2.0281 | 0.7431 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -10.4326 | -11.2098 | 0.7772 | 0.6067 | 1.2811 | 0.8008 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.320 | -11.2964 | -11.9500 | 0.6535 | 0.7897 | 0.8276 | 0.8476 | N/A | — | severe_error(0.82) | increase_p |  |
| 1.430 | -12.1603 | -12.7141 | 0.5537 | 0.9819 | 0.5639 | 0.8819 | N/A | — | severe_error(0.54) | increase_p |  |
| 1.540 | -13.0241 | -13.4964 | 0.4723 | 1.1807 | 0.4001 | 0.9073 | N/A | — | moderate_error(0.37) | refine_vqe |  |
| 1.640 | -13.8093 | -14.2201 | 0.4108 | 1.3653 | 0.3009 | 0.9249 | N/A | — | moderate_error(0.26) | refine_vqe |  |
| 1.750 | -14.6731 | -15.0271 | 0.3540 | 1.5716 | 0.2253 | 0.9398 | N/A | — | moderate_error(0.18) | refine_vqe |  |
| 1.860 | -15.5368 | -15.8433 | 0.3065 | 1.7804 | 0.1722 | 0.9513 | N/A | — | moderate_error(0.13) | refine_vqe |  |
| 1.960 | -16.3221 | -16.5920 | 0.2699 | 1.9719 | 0.1369 | 0.9596 | N/A | — | moderate_error(0.09) | refine_vqe |  |
| 2.070 | -17.1858 | -17.4216 | 0.2358 | 2.1840 | 0.1080 | 0.9668 | N/A | — | moderate_error(0.06) | refine_vqe |  |
| 2.180 | -18.0495 | -18.2566 | 0.2071 | 2.3972 | 0.0864 | 0.9725 | N/A | — | near_pass(0.04) | refine_vqe |  |
| 2.290 | -18.9133 | -19.0961 | 0.1828 | 2.6114 | 0.0700 | 0.9770 | N/A | — | near_pass(0.02) | refine_vqe |  |
| 2.390 | -19.6986 | -19.8627 | 0.1641 | 2.8068 | 0.0585 | 0.9804 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 2.500 | -20.5624 | -20.7091 | 0.1467 | 3.0223 | 0.0485 | 0.9833 | N/A | — | gap_masked(0.49) | refine_vqe |  |

## N = 10 (19 params)

**ΔE/gap: 1.0778 ± 1.7227 | P90=2.7748 | max=6.6500
|ΔE|/N: 5.88e-02
Fidelity: mean=0.8686 min=0.5521 (exact)
Distribution: [P25=0.131 | P50=0.309 | P75=1.015 | P90=2.775]
Regions: critical=2.7922 | ordered=0.0995**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=6.650 is 6× the mean — median may be more representative N=10

**Fidelity: mean F=0.8686, min F=0.5521** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -11.0083 | -12.4722 | 1.4639 | 0.2201 | 6.6500 | 0.5521 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -12.0871 | -13.2894 | 1.2023 | 0.3632 | 3.3100 | 0.6627 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -13.0680 | -14.0836 | 1.0156 | 0.5151 | 1.9719 | 0.7387 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.320 | -14.1472 | -14.9990 | 0.8518 | 0.6978 | 1.2208 | 0.8005 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.430 | -15.2264 | -15.9469 | 0.7205 | 0.8912 | 0.8085 | 0.8456 | N/A | — | severe_error(0.80) | increase_p |  |
| 1.540 | -16.3055 | -16.9194 | 0.6138 | 1.0915 | 0.5624 | 0.8789 | N/A | — | severe_error(0.54) | increase_p |  |
| 1.640 | -17.2866 | -17.8199 | 0.5334 | 1.2777 | 0.4174 | 0.9019 | N/A | — | moderate_error(0.39) | refine_vqe |  |
| 1.750 | -18.3656 | -18.8250 | 0.4593 | 1.4858 | 0.3091 | 0.9214 | N/A | — | moderate_error(0.27) | refine_vqe |  |
| 1.860 | -19.4447 | -19.8422 | 0.3975 | 1.6962 | 0.2344 | 0.9365 | N/A | — | moderate_error(0.19) | refine_vqe |  |
| 1.960 | -20.4257 | -20.7757 | 0.3500 | 1.8891 | 0.1853 | 0.9472 | N/A | — | moderate_error(0.14) | refine_vqe |  |
| 2.070 | -21.5047 | -21.8104 | 0.3057 | 2.1026 | 0.1454 | 0.9567 | N/A | — | moderate_error(0.10) | refine_vqe |  |
| 2.180 | -22.5838 | -22.8521 | 0.2683 | 2.3171 | 0.1158 | 0.9641 | N/A | — | moderate_error(0.07) | refine_vqe |  |
| 2.290 | -23.6629 | -23.8998 | 0.2369 | 2.5324 | 0.0935 | 0.9700 | N/A | — | near_pass(0.05) | refine_vqe |  |
| 2.390 | -24.6439 | -24.8565 | 0.2127 | 2.7287 | 0.0779 | 0.9744 | N/A | — | near_pass(0.03) | refine_vqe |  |
| 2.500 | -25.7229 | -25.9132 | 0.1903 | 2.9452 | 0.0646 | 0.9783 | N/A | — | near_pass(0.02) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 3.7797 ± 2.9164 | P90=7.8082 | max=10.3255
|ΔE|/N: 6.35e-02
Var(H): 6.2542
Distribution: [P25=1.425 | P50=3.119 | P75=5.375 | P90=7.808]
Regions: critical=7.2838 | ordered=0.8558**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=53.7051.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -22.1257 | -25.3695 | 3.2438 | 0.3142 | 10.3255 | N/A | 9.9784 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -24.2814 | -26.9058 | 2.6244 | 0.3142 | 8.3537 | N/A | 9.2731 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -26.2413 | -28.4372 | 2.1959 | 0.3142 | 6.9898 | N/A | 8.6643 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -28.3974 | -30.2298 | 1.8325 | 0.3142 | 5.8330 | N/A | 8.0303 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -30.5535 | -32.0983 | 1.5448 | 0.3142 | 4.9173 | N/A | 7.4339 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -32.7096 | -34.0227 | 1.3132 | 0.3142 | 4.1799 | N/A | 6.8758 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -34.6697 | -35.8088 | 1.1391 | 0.3142 | 3.6259 | N/A | 6.4011 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -36.8257 | -37.8057 | 0.9800 | 0.3142 | 3.1195 | N/A | 5.9163 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -38.9816 | -39.8292 | 0.8477 | 0.3142 | 2.6982 | N/A | 5.4698 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -40.9415 | -41.6875 | 0.7459 | 0.3142 | 2.3744 | N/A | 5.0969 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -43.0975 | -43.7486 | 0.6511 | 0.3519 | 1.8501 | N/A | 4.7228 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.180 | -45.2535 | -45.8248 | 0.5713 | 0.5713 | 0.9999 | N/A | 4.3863 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.290 | -47.4094 | -47.9136 | 0.5042 | 0.7908 | 0.6376 | N/A | 4.0884 | dirty_state | severe_error(0.62) | increase_p |  |
| 2.390 | -49.3694 | -49.8220 | 0.4525 | 0.9903 | 0.4570 | N/A | 3.8505 | dirty_state | moderate_error(0.43) | refine_vqe |  |
| 2.500 | -51.5254 | -51.9300 | 0.4047 | 1.2099 | 0.3345 | N/A | 3.6249 | dirty_state | moderate_error(0.30) | refine_vqe |  |
