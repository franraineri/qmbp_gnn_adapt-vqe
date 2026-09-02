# Model Evaluation: heavy_hex

**Date**: 2026-09-01 16:37 UTC
**Model**: data/model_zoo/checkpoints/unified_tfim_br_heavy_hex_fromMT_4+6+8+10+12+14+18+20+21+26+30+40_p1.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 2.5] (15 pts)
**Target N**: [8, 10, 20]

---

## N = 8 (15 params)

**ΔE/gap: 13.8147 ± 8.5649 | P90=24.8330 | max=37.6128
|ΔE|/N: 1.99e+00
Fidelity: mean=0.0014 min=0.0001 (exact)
Distribution: [P25=7.805 | P50=10.303 | P75=16.165 | P90=24.833]
Regions: critical=23.6971 | ordered=7.2005**

**Fidelity: mean F=0.0014, min F=0.0001** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | 1.4256 | -9.8947 | 11.3203 | 0.3010 | 37.6128 | 0.0001 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | 1.6793 | -10.5644 | 12.2437 | 0.4521 | 27.0790 | 0.0002 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | 1.8123 | -11.2098 | 13.0221 | 0.6067 | 21.4639 | 0.0003 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | 1.8797 | -11.9500 | 13.8296 | 0.7897 | 17.5134 | 0.0004 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | 1.8348 | -12.7141 | 14.5488 | 0.9819 | 14.8163 | 0.0005 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | 1.6984 | -13.4964 | 15.1948 | 1.1807 | 12.8697 | 0.0007 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | 1.4797 | -14.2201 | 15.6998 | 1.3653 | 11.4991 | 0.0010 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | 1.1658 | -15.0271 | 16.1928 | 1.5716 | 10.3033 | 0.0014 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | 0.8364 | -15.8433 | 16.6797 | 1.7804 | 9.3684 | 0.0016 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | 0.5492 | -16.5920 | 17.1413 | 1.9719 | 8.6927 | 0.0019 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | 0.2000 | -17.4216 | 17.6216 | 2.1840 | 8.0686 | 0.0022 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -0.1802 | -18.2566 | 18.0764 | 2.3972 | 7.5406 | 0.0025 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -0.5188 | -19.0961 | 18.5773 | 2.6114 | 7.1139 | 0.0027 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | -0.8070 | -19.8627 | 19.0557 | 2.8068 | 6.7891 | 0.0027 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -1.0937 | -20.7091 | 19.6155 | 3.0223 | 6.4902 | 0.0025 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |

## N = 10 (19 params)

**ΔE/gap: 22.6725 ± 15.8733 | P90=41.5410 | max=69.5344
|ΔE|/N: 2.34e+00
Fidelity: mean=0.0000 min=0.0000 (exact)
Distribution: [P25=12.264 | P50=15.958 | P75=25.450 | P90=41.541]
Regions: critical=40.2597 | ordered=11.3860**

**Fidelity: mean F=0.0000, min F=0.0000** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | 2.8345 | -12.4722 | 15.3066 | 0.2201 | 69.5344 | 0.0000 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | 3.4106 | -13.2894 | 16.7000 | 0.3632 | 45.9767 | 0.0000 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | 3.8858 | -14.0836 | 17.9694 | 0.5151 | 34.8876 | 0.0000 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | 4.3550 | -14.9990 | 19.3540 | 0.6978 | 27.7367 | 0.0000 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | 4.6950 | -15.9469 | 20.6419 | 0.8912 | 23.1630 | 0.0000 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | 4.8981 | -16.9194 | 21.8174 | 1.0915 | 19.9884 | 0.0000 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | 4.9277 | -17.8199 | 22.7476 | 1.2777 | 17.8030 | 0.0000 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | 4.8857 | -18.8250 | 23.7107 | 1.4858 | 15.9584 | 0.0000 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | 4.8557 | -19.8422 | 24.6979 | 1.6962 | 14.5604 | 0.0000 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | 4.8200 | -20.7757 | 25.5957 | 1.8891 | 13.5491 | 0.0000 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | 4.7660 | -21.8104 | 26.5764 | 2.1026 | 12.6400 | 0.0000 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | 4.6946 | -22.8521 | 27.5467 | 2.3171 | 11.8886 | 0.0000 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | 4.6158 | -23.8998 | 28.5155 | 2.5324 | 11.2603 | 0.0000 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | 4.5730 | -24.8565 | 29.4296 | 2.7287 | 10.7851 | 0.0000 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | 4.5875 | -25.9132 | 30.5007 | 2.9452 | 10.3561 | 0.0000 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |

## N = 20 (39 params)

**ΔE/gap: 123.3412 ± 36.9642 | P90=159.9806 | max=166.2611
|ΔE|/N: 2.43e+00
Var(H): 55.8916
Distribution: [P25=103.447 | P50=135.252 | P75=152.724 | P90=159.981]
Regions: critical=126.8285 | ordered=85.8675**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=369.3294.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | 9.2734 | -25.3695 | 34.6430 | 0.3142 | 110.2719 | N/A | 31.8619 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | 10.5400 | -26.9058 | 37.4458 | 0.3142 | 119.1937 | N/A | 33.3913 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | 11.5165 | -28.4372 | 39.9538 | 0.3142 | 127.1768 | N/A | 34.7103 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | 12.2610 | -30.2298 | 42.4908 | 0.3142 | 135.2525 | N/A | 36.8498 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | 12.5902 | -32.0983 | 44.6885 | 0.3142 | 142.2479 | N/A | 39.6778 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | 12.5892 | -34.0227 | 46.6119 | 0.3142 | 148.3704 | N/A | 43.1397 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | 12.2476 | -35.8088 | 48.0564 | 0.3142 | 152.9683 | N/A | 47.2268 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | 11.5834 | -37.8057 | 49.3891 | 0.3142 | 157.2102 | N/A | 52.7970 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | 11.0104 | -39.8292 | 50.8396 | 0.3142 | 161.8274 | N/A | 58.0733 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | 10.5450 | -41.6875 | 52.2325 | 0.3142 | 166.2611 | N/A | 63.4580 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | 9.9118 | -43.7486 | 53.6604 | 0.3519 | 152.4791 | N/A | 69.9981 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | 9.3775 | -45.8248 | 55.2022 | 0.5713 | 96.6225 | N/A | 75.4123 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | 8.8717 | -47.9136 | 56.7853 | 0.7908 | 71.8097 | N/A | 80.0270 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | 8.4466 | -49.8220 | 58.2685 | 0.9903 | 58.8379 | N/A | 83.9539 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | 8.0657 | -51.9300 | 59.9957 | 1.2099 | 49.5886 | N/A | 87.7962 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
