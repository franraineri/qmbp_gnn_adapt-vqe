# Model Evaluation: heavy_hex

> **🌐 MULTI-TOPOLOGY MODEL** — This evaluation uses a model trained on multiple topologies simultaneously. Results reflect cross-topology transfer capability.

**Date**: 2026-09-01 16:47 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__MT_p1_res_film_base.pt
**p_layers**: 1
**Multi-topology**: YES
**h-range**: [1.0, 2.5] (15 pts)
**Target N**: [8, 10, 20]

---

## N = 8 (15 params)

**ΔE/gap: 0.6872 ± 0.7796 | P90=1.4178 | max=3.1839
|ΔE|/N: 7.44e-02
Fidelity: mean=0.8846 min=0.6885 (exact)
Distribution: [P25=0.205 | P50=0.374 | P75=0.769 | P90=1.418]
Regions: critical=1.4895 | ordered=0.1773**

**Fidelity: mean F=0.8846, min F=0.6885** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -8.9365 | -9.8947 | 0.9583 | 0.3010 | 3.1839 | 0.6885 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -9.8248 | -10.5644 | 0.7396 | 0.4521 | 1.6358 | 0.7790 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.210 | -10.5480 | -11.2098 | 0.6618 | 0.6067 | 1.0908 | 0.8242 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.320 | -11.2904 | -11.9500 | 0.6595 | 0.7897 | 0.8352 | 0.8481 | N/A | — | severe_error(0.83) | increase_p |  |
| 1.430 | -12.0248 | -12.7141 | 0.6892 | 0.9819 | 0.7019 | 0.8613 | N/A | — | severe_error(0.69) | increase_p |  |
| 1.540 | -12.8177 | -13.4964 | 0.6788 | 1.1807 | 0.5749 | 0.8769 | N/A | — | severe_error(0.55) | increase_p |  |
| 1.640 | -13.5789 | -14.2201 | 0.6412 | 1.3653 | 0.4696 | 0.8923 | N/A | — | moderate_error(0.44) | refine_vqe |  |
| 1.750 | -14.4394 | -15.0271 | 0.5877 | 1.5716 | 0.3739 | 0.9083 | N/A | — | moderate_error(0.34) | refine_vqe |  |
| 1.860 | -15.3071 | -15.8433 | 0.5362 | 1.7804 | 0.3012 | 0.9218 | N/A | — | moderate_error(0.26) | refine_vqe |  |
| 1.960 | -16.0908 | -16.5920 | 0.5012 | 1.9719 | 0.2542 | 0.9312 | N/A | — | moderate_error(0.21) | refine_vqe |  |
| 2.070 | -16.9447 | -17.4216 | 0.4769 | 2.1840 | 0.2184 | 0.9385 | N/A | — | moderate_error(0.18) | refine_vqe |  |
| 2.180 | -17.7955 | -18.2566 | 0.4612 | 2.3972 | 0.1924 | 0.9440 | N/A | — | moderate_error(0.15) | refine_vqe |  |
| 2.290 | -18.6414 | -19.0961 | 0.4547 | 2.6114 | 0.1741 | 0.9478 | N/A | — | moderate_error(0.13) | refine_vqe |  |
| 2.390 | -19.4164 | -19.8627 | 0.4463 | 2.8068 | 0.1590 | 0.9512 | N/A | — | moderate_error(0.11) | refine_vqe |  |
| 2.500 | -20.2778 | -20.7091 | 0.4314 | 3.0223 | 0.1427 | 0.9551 | N/A | — | moderate_error(0.10) | refine_vqe |  |

## N = 10 (19 params)

**ΔE/gap: 1.2046 ± 1.5202 | P90=2.5072 | max=6.2077
|ΔE|/N: 8.81e-02
Fidelity: mean=0.8345 min=0.5683 (exact)
Distribution: [P25=0.323 | P50=0.579 | P75=1.273 | P90=2.507]
Regions: critical=2.7120 | ordered=0.2775**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=6.208 is 5× the mean — median may be more representative N=10

**Fidelity: mean F=0.8345, min F=0.5683** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -11.1057 | -12.4722 | 1.3665 | 0.2201 | 6.2077 | 0.5683 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -12.2274 | -13.2894 | 1.0621 | 0.3632 | 2.9240 | 0.6882 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -13.1142 | -14.0836 | 0.9694 | 0.5151 | 1.8822 | 0.7487 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.320 | -14.0185 | -14.9990 | 0.9805 | 0.6978 | 1.4051 | 0.7815 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.430 | -14.9302 | -15.9469 | 1.0167 | 0.8912 | 1.1409 | 0.8018 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.540 | -15.9008 | -16.9194 | 1.0186 | 1.0915 | 0.9332 | 0.8214 | N/A | — | severe_error(0.93) | increase_p |  |
| 1.640 | -16.8764 | -17.8199 | 0.9435 | 1.2777 | 0.7384 | 0.8458 | N/A | — | severe_error(0.72) | increase_p |  |
| 1.750 | -17.9643 | -18.8250 | 0.8607 | 1.4858 | 0.5793 | 0.8689 | N/A | — | severe_error(0.56) | increase_p |  |
| 1.860 | -19.0438 | -19.8422 | 0.7984 | 1.6962 | 0.4707 | 0.8863 | N/A | — | moderate_error(0.44) | refine_vqe |  |
| 1.960 | -20.0210 | -20.7757 | 0.7546 | 1.8891 | 0.3995 | 0.8986 | N/A | — | moderate_error(0.37) | refine_vqe |  |
| 2.070 | -21.0879 | -21.8104 | 0.7225 | 2.1026 | 0.3436 | 0.9087 | N/A | — | moderate_error(0.31) | refine_vqe |  |
| 2.180 | -22.1506 | -22.8521 | 0.7015 | 2.3171 | 0.3028 | 0.9164 | N/A | — | moderate_error(0.27) | refine_vqe |  |
| 2.290 | -23.2087 | -23.8998 | 0.6911 | 2.5324 | 0.2729 | 0.9221 | N/A | — | moderate_error(0.23) | refine_vqe |  |
| 2.390 | -24.1805 | -24.8565 | 0.6760 | 2.7287 | 0.2477 | 0.9273 | N/A | — | moderate_error(0.21) | refine_vqe |  |
| 2.500 | -25.2639 | -25.9132 | 0.6493 | 2.9452 | 0.2205 | 0.9334 | N/A | — | moderate_error(0.18) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 7.8137 ± 4.2358 | P90=13.0895 | max=15.9189
|ΔE|/N: 1.39e-01
Var(H): 16.8243
Distribution: [P25=4.585 | P50=7.706 | P75=10.662 | P90=13.090]
Regions: critical=12.6342 | ordered=2.9685**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=129.6055.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -20.3685 | -25.3695 | 5.0011 | 0.3142 | 15.9189 | N/A | 17.5078 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -22.6366 | -26.9058 | 4.2692 | 0.3142 | 13.5894 | N/A | 17.0356 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -24.5606 | -28.4372 | 3.8766 | 0.3142 | 12.3397 | N/A | 17.2337 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -26.7083 | -30.2298 | 3.5215 | 0.3142 | 11.2094 | N/A | 17.3561 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -28.9210 | -32.0983 | 3.1773 | 0.3142 | 10.1138 | N/A | 17.1474 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -31.1589 | -34.0227 | 2.8639 | 0.3142 | 9.1159 | N/A | 16.7685 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -33.1769 | -35.8088 | 2.6319 | 0.3142 | 8.3776 | N/A | 16.5047 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -35.3846 | -37.8057 | 2.4210 | 0.3142 | 7.7064 | N/A | 16.2846 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -37.5717 | -39.8292 | 2.2575 | 0.3142 | 7.1860 | N/A | 16.2141 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -39.5491 | -41.6875 | 2.1384 | 0.3142 | 6.8066 | N/A | 16.2456 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -41.7193 | -43.7486 | 2.0293 | 0.3519 | 5.7663 | N/A | 16.3418 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -43.8797 | -45.8248 | 1.9450 | 0.5713 | 3.4045 | N/A | 16.5495 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -46.0275 | -47.9136 | 1.8861 | 0.7908 | 2.3851 | N/A | 16.9082 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | -47.9955 | -49.8220 | 1.8265 | 0.9903 | 1.8443 | N/A | 17.1252 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.500 | -50.1852 | -51.9300 | 1.7448 | 1.2099 | 1.4421 | N/A | 17.1421 | dirty_state | severe_error(1.00) | increase_p |  |
