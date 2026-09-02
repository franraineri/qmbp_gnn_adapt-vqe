# Model Evaluation: heavy_hex

**Date**: 2026-09-01 16:32 UTC
**Model**: data/model_zoo/checkpoints/unified_tfim_br_heavy_hex_fromMT_4+6+10+12+16+20_p1.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 2.5] (15 pts)
**Target N**: [8, 10, 20]

---

## N = 8 (15 params)

**ΔE/gap: 8.8931 ± 5.0918 | P90=15.2167 | max=23.4246
|ΔE|/N: 1.33e+00
Fidelity: mean=0.0106 min=0.0009 (exact)
Distribution: [P25=5.453 | P50=6.916 | P75=10.024 | P90=15.217]
Regions: critical=14.6388 | ordered=5.0261**

**Fidelity: mean F=0.0106, min F=0.0009** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -2.8446 | -9.8947 | 7.0501 | 0.3010 | 23.4246 | 0.0018 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -3.0413 | -10.5644 | 7.5231 | 0.4521 | 16.6385 | 0.0010 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -3.2718 | -11.2098 | 7.9380 | 0.6067 | 13.0840 | 0.0009 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -3.4488 | -11.9500 | 8.5012 | 0.7897 | 10.7656 | 0.0015 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -3.6002 | -12.7141 | 9.1138 | 0.9819 | 9.2814 | 0.0024 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -3.7479 | -13.4964 | 9.7485 | 1.1807 | 8.2568 | 0.0034 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -3.9167 | -14.2201 | 10.3034 | 1.3653 | 7.5466 | 0.0047 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -4.1576 | -15.0271 | 10.8695 | 1.5716 | 6.9161 | 0.0066 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -4.4886 | -15.8433 | 11.3547 | 1.7804 | 6.3776 | 0.0090 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -4.8104 | -16.5920 | 11.7816 | 1.9719 | 5.9747 | 0.0116 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -5.1680 | -17.4216 | 12.2536 | 2.1840 | 5.6107 | 0.0146 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -5.5610 | -18.2566 | 12.6957 | 2.3972 | 5.2960 | 0.0185 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -6.0633 | -19.0961 | 13.0329 | 2.6114 | 4.9907 | 0.0226 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | -6.5638 | -19.8627 | 13.2989 | 2.8068 | 4.7381 | 0.0275 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -7.1234 | -20.7091 | 13.5857 | 3.0223 | 4.4951 | 0.0329 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |

## N = 10 (19 params)

**ΔE/gap: 12.8806 ± 9.5917 | P90=23.6861 | max=41.9143
|ΔE|/N: 1.30e+00
Fidelity: mean=0.0069 min=0.0006 (exact)
Distribution: [P25=6.788 | P50=9.071 | P75=14.167 | P90=23.686]
Regions: critical=23.2438 | ordered=6.1673**

**Fidelity: mean F=0.0069, min F=0.0006** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -3.2456 | -12.4722 | 9.2266 | 0.2201 | 41.9143 | 0.0007 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -3.6678 | -13.2894 | 9.6216 | 0.3632 | 26.4891 | 0.0006 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -4.0493 | -14.0836 | 10.0343 | 0.5151 | 19.4817 | 0.0009 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -4.2289 | -14.9990 | 10.7701 | 0.6978 | 15.4349 | 0.0013 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -4.4520 | -15.9469 | 11.4949 | 0.8912 | 12.8988 | 0.0019 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -4.7045 | -16.9194 | 12.2149 | 1.0915 | 11.1909 | 0.0025 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -4.9753 | -17.8199 | 12.8446 | 1.2777 | 10.0526 | 0.0032 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -5.3474 | -18.8250 | 13.4775 | 1.4858 | 9.0710 | 0.0043 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -5.8876 | -19.8422 | 13.9546 | 1.6962 | 8.2268 | 0.0058 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -6.3950 | -20.7757 | 14.3806 | 1.8891 | 7.6124 | 0.0073 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -7.0008 | -21.8104 | 14.8096 | 2.1026 | 7.0436 | 0.0096 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -7.7150 | -22.8521 | 15.1371 | 2.3171 | 6.5329 | 0.0121 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -8.4783 | -23.8998 | 15.4214 | 2.5324 | 6.0897 | 0.0152 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | -9.1675 | -24.8565 | 15.6890 | 2.7287 | 5.7496 | 0.0178 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -9.9476 | -25.9132 | 15.9656 | 2.9452 | 5.4209 | 0.0211 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |

## N = 20 (39 params)

**ΔE/gap: 61.3274 ± 16.6871 | P90=80.5131 | max=85.0628
|ΔE|/N: 1.24e+00
Var(H): 63.9752
Distribution: [P25=53.933 | P50=62.924 | P75=74.070 | P90=80.513]
Regions: critical=60.5654 | ordered=46.8629**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=440.0473.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -7.9458 | -25.3695 | 17.4237 | 0.3142 | 55.4614 | N/A | 38.6579 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -8.6714 | -26.9058 | 18.2344 | 0.3142 | 58.0419 | N/A | 42.1202 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -9.3904 | -28.4372 | 19.0468 | 0.3142 | 60.6280 | N/A | 44.8616 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -10.4616 | -30.2298 | 19.7682 | 0.3142 | 62.9242 | N/A | 47.8970 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -11.4355 | -32.0983 | 20.6628 | 0.3142 | 65.7716 | N/A | 51.1864 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -12.3328 | -34.0227 | 21.6899 | 0.3142 | 69.0412 | N/A | 54.7949 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -13.1072 | -35.8088 | 22.7016 | 0.3142 | 72.2614 | N/A | 58.2472 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -13.9678 | -37.8057 | 23.8379 | 0.3142 | 75.8783 | N/A | 62.3843 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -14.5313 | -39.8292 | 25.2979 | 0.3142 | 80.5257 | N/A | 67.0587 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -14.9642 | -41.6875 | 26.7233 | 0.3142 | 85.0628 | N/A | 71.2788 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -15.4211 | -43.7486 | 28.3275 | 0.3519 | 80.4941 | N/A | 75.7103 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -15.8848 | -45.8248 | 29.9400 | 0.5713 | 52.4051 | N/A | 80.0303 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -16.3915 | -47.9136 | 31.5221 | 0.7908 | 39.8623 | N/A | 84.2628 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | -16.9445 | -49.8220 | 32.8775 | 0.9903 | 33.1987 | N/A | 88.2477 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -17.6249 | -51.9300 | 34.3051 | 1.2099 | 28.3544 | N/A | 92.8897 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
