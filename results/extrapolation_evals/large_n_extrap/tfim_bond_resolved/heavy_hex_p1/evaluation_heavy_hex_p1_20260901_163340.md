# Model Evaluation: heavy_hex

**Date**: 2026-09-01 16:33 UTC
**Model**: data/model_zoo/checkpoints/unified_tfim_br_heavy_hex_multiN_4+6+10+12+16+20+40_p1.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 2.5] (15 pts)
**Target N**: [8, 10, 20]

---

## N = 8 (15 params)

**ΔE/gap: 7.0862 ± 3.2115 | P90=11.1771 | max=15.2051
|ΔE|/N: 1.19e+00
Fidelity: mean=0.1470 min=0.0001 (exact)
Distribution: [P25=5.213 | P50=6.564 | P75=7.760 | P90=11.177]
Regions: critical=9.9543 | ordered=5.6359**

**Fidelity: mean F=0.1470, min F=0.0001** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -5.3184 | -9.8947 | 4.5763 | 0.3010 | 15.2051 | 0.2694 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -5.0548 | -10.5644 | 5.5096 | 0.4521 | 12.1854 | 0.2209 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -5.3463 | -11.2098 | 5.8635 | 0.6067 | 9.6646 | 0.2143 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -6.2288 | -11.9500 | 5.7212 | 0.7897 | 7.2451 | 0.2372 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -7.3417 | -12.7141 | 5.3723 | 0.9819 | 5.4711 | 0.2896 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -7.4456 | -13.4964 | 6.0508 | 1.1807 | 5.1249 | 0.3154 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -6.9837 | -14.2201 | 7.2364 | 1.3653 | 5.3002 | 0.2711 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -7.3701 | -15.0271 | 7.6570 | 1.5716 | 4.8721 | 0.2573 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -4.3046 | -15.8433 | 11.5387 | 1.7804 | 6.4809 | 0.0763 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -3.6482 | -16.5920 | 12.9438 | 1.9719 | 6.5641 | 0.0435 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -0.0831 | -17.4216 | 17.3386 | 2.1840 | 7.9390 | 0.0078 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -0.4798 | -18.2566 | 17.7768 | 2.3972 | 7.4156 | 0.0015 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | 0.6988 | -19.0961 | 19.7949 | 2.6114 | 7.5802 | 0.0001 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | -12.4413 | -19.8627 | 7.4214 | 2.8068 | 2.6441 | 0.0006 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -12.8492 | -20.7091 | 7.8600 | 3.0223 | 2.6007 | 0.0004 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |

## N = 10 (19 params)

**ΔE/gap: 10.5639 ± 6.6114 | P90=15.3445 | max=31.3830
|ΔE|/N: 1.26e+00
Fidelity: mean=0.0837 min=0.0000 (exact)
Distribution: [P25=7.493 | P50=10.246 | P75=10.673 | P90=15.344]
Regions: critical=14.9777 | ordered=7.3254**

**Fidelity: mean F=0.0837, min F=0.0000** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -5.5638 | -12.4722 | 6.9083 | 0.2201 | 31.3830 | 0.0994 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -6.7209 | -13.2894 | 6.5685 | 0.3632 | 18.0836 | 0.1323 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -8.2965 | -14.0836 | 5.7872 | 0.5151 | 11.2358 | 0.1795 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -9.9526 | -14.9990 | 5.0464 | 0.6978 | 7.2321 | 0.2534 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -9.7498 | -15.9469 | 6.1971 | 0.8912 | 6.9540 | 0.2752 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -7.9473 | -16.9194 | 8.9720 | 1.0915 | 8.2199 | 0.1750 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -7.9136 | -17.8199 | 9.9063 | 1.2777 | 7.7530 | 0.1028 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -3.1009 | -18.8250 | 15.7240 | 1.4858 | 10.5830 | 0.0247 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -3.2083 | -19.8422 | 16.6339 | 1.6962 | 9.8064 | 0.0113 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -0.7862 | -20.7757 | 19.9894 | 1.8891 | 10.5814 | 0.0020 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -0.2685 | -21.8104 | 21.5419 | 2.1026 | 10.2456 | 0.0002 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | 1.3076 | -22.8521 | 24.1597 | 2.3171 | 10.4269 | 0.0000 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | 3.3540 | -23.8998 | 27.2538 | 2.5324 | 10.7621 | 0.0000 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | -17.7160 | -24.8565 | 7.1405 | 2.7287 | 2.6168 | 0.0000 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -18.3278 | -25.9132 | 7.5854 | 2.9452 | 2.5755 | 0.0000 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |

## N = 20 (39 params)

**ΔE/gap: 96.1122 ± 34.1984 | P90=144.1757 | max=148.6918
|ΔE|/N: 2.07e+00
Var(H): 60.0339
Distribution: [P25=70.078 | P50=90.959 | P75=125.231 | P90=144.176]
Regions: critical=67.7903 | ordered=90.9502**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=498.2334.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -13.0257 | -25.3695 | 12.3439 | 0.3142 | 39.2918 | N/A | 37.7435 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -10.4714 | -26.9058 | 16.4344 | 0.3142 | 52.3124 | N/A | 49.7992 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -6.0206 | -28.4372 | 22.4166 | 0.3142 | 71.3542 | N/A | 53.1753 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -3.5157 | -30.2298 | 26.7142 | 0.3142 | 85.0338 | N/A | 54.4300 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -3.5225 | -32.0983 | 28.5758 | 0.3142 | 90.9595 | N/A | 58.6833 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -0.0586 | -34.0227 | 33.9642 | 0.3142 | 108.1114 | N/A | 64.5531 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | 1.7678 | -35.8088 | 37.5767 | 0.3142 | 119.6102 | N/A | 68.8422 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | 3.3025 | -37.8057 | 41.1082 | 0.3142 | 130.8514 | N/A | 78.7725 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | 4.3778 | -39.8292 | 44.2070 | 0.3142 | 140.7153 | N/A | 84.8635 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | 5.0255 | -41.6875 | 46.7129 | 0.3142 | 148.6918 | N/A | 89.1898 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | 7.8016 | -43.7486 | 51.5502 | 0.3519 | 146.4827 | N/A | 82.5387 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | 11.6033 | -45.8248 | 57.4280 | 0.5713 | 100.5184 | N/A | 65.0656 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | 15.9555 | -47.9136 | 63.8691 | 0.7908 | 80.7678 | N/A | 39.4416 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | 18.3137 | -49.8220 | 68.1357 | 0.9903 | 68.8014 | N/A | 27.9213 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | 18.4613 | -51.9300 | 70.3914 | 1.2099 | 58.1809 | N/A | 45.4891 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
