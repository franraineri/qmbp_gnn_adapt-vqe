# Model Evaluation: chain_1d

**Date**: 2026-09-02 18:18 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__chain_1d_p1_signinv_fid_dot5_v1.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 3.0] (15 pts)
**Target N**: [30, 40, 50, 60]

---

## N = 30 (59 params)

**ΔE/gap: 8.1421 ± 9.6542 | P90=17.5446 | max=39.1315
|ΔE|/N: 3.19e-01
Var(H): 68.2274
Distribution: [P25=2.527 | P50=4.208 | P75=8.640 | P90=17.545]
Regions: critical=18.1144 | ordered=2.5068**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=49.5724.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -36.0628 | -37.8381 | 1.7753 | 0.2094 | 8.4762 | N/A | 4.8024 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -39.6985 | -40.9173 | 1.2188 | 0.2896 | 4.2083 | N/A | 4.1102 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.290 | -21.5233 | -44.5522 | 23.0289 | 0.5885 | 39.1315 | N/A | 92.3717 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -30.2240 | -48.1342 | 17.9103 | 0.8677 | 20.6418 | N/A | 87.6352 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.570 | -37.0442 | -51.8389 | 14.7947 | 1.1470 | 12.8988 | N/A | 83.7968 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.710 | -43.0754 | -55.6318 | 12.5564 | 1.4264 | 8.8028 | N/A | 79.7511 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -49.4575 | -59.7683 | 10.3108 | 1.7259 | 5.9742 | N/A | 72.9812 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.000 | -54.7568 | -63.6811 | 8.9243 | 2.0055 | 4.4499 | N/A | 68.7767 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.140 | -59.5753 | -67.6333 | 8.0580 | 2.2851 | 3.5263 | N/A | 66.9066 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -64.1366 | -71.9027 | 7.7661 | 2.5848 | 3.0046 | N/A | 69.1498 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.430 | -68.2868 | -75.9142 | 7.6274 | 2.8645 | 2.6627 | N/A | 72.1362 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.570 | -72.4272 | -79.9468 | 7.5196 | 3.1443 | 2.3915 | N/A | 75.2728 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.710 | -76.5680 | -83.9972 | 7.4291 | 3.4240 | 2.1697 | N/A | 78.4708 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.860 | -81.0052 | -88.3535 | 7.3483 | 3.7238 | 1.9733 | N/A | 81.9640 | dirty_state | severe_error(1.00) | increase_p |  |
| 3.000 | -85.1471 | -92.4327 | 7.2856 | 4.0037 | 1.8197 | N/A | 85.2853 | dirty_state | severe_error(1.00) | increase_p |  |

## N = 40 (79 params)

**ΔE/gap: 11.7257 ± 14.0605 | P90=25.9963 | max=55.8947
|ΔE|/N: 3.30e-01
Var(H): 92.6444
Distribution: [P25=3.398 | P50=5.828 | P75=13.664 | P90=25.996]
Regions: critical=27.1626 | ordered=3.3696**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=76.6867.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -48.1366 | -50.5694 | 2.4328 | 0.1571 | 15.4876 | N/A | 6.5236 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -52.9831 | -54.6465 | 1.6634 | 0.2854 | 5.8281 | N/A | 5.5813 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.290 | -26.7915 | -59.4777 | 32.6862 | 0.5848 | 55.8947 | N/A | 127.4958 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -37.0702 | -64.2441 | 27.1739 | 0.8643 | 31.4399 | N/A | 127.6755 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.570 | -48.7793 | -69.1766 | 20.3973 | 1.1439 | 17.8309 | N/A | 114.7458 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.710 | -57.3711 | -74.2281 | 16.8570 | 1.4236 | 11.8411 | N/A | 107.0122 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -65.9475 | -79.7387 | 13.7911 | 1.7233 | 8.0027 | N/A | 97.6219 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.000 | -72.9884 | -84.9520 | 11.9636 | 2.0031 | 5.9726 | N/A | 92.1881 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.140 | -79.4078 | -90.2184 | 10.8106 | 2.2829 | 4.7355 | N/A | 89.7496 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -85.4748 | -95.9081 | 10.4334 | 2.5827 | 4.0397 | N/A | 92.8811 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.430 | -91.0043 | -101.2544 | 10.2501 | 2.8625 | 3.5808 | N/A | 96.9234 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.570 | -96.5244 | -106.6291 | 10.1047 | 3.1424 | 3.2156 | N/A | 101.1343 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.710 | -102.0450 | -112.0278 | 9.9828 | 3.4223 | 2.9170 | N/A | 105.4290 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.860 | -107.9606 | -117.8346 | 9.8739 | 3.7222 | 2.6527 | N/A | 110.1211 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 3.000 | -113.4824 | -123.2720 | 9.7896 | 4.0021 | 2.4461 | N/A | 114.5841 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |

## N = 50 (99 params)

**ΔE/gap: 15.6758 ± 18.5791 | P90=37.2810 | max=70.6829
|ΔE|/N: 3.43e-01
Var(H): 117.9827
Distribution: [P25=4.268 | P50=7.440 | P75=19.771 | P90=37.281]
Regions: critical=37.1027 | ordered=4.2304**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=111.2701.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -60.2094 | -63.3012 | 3.0918 | 0.1257 | 24.6035 | N/A | 8.2494 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -66.2668 | -68.3756 | 2.1089 | 0.2835 | 7.4396 | N/A | 7.0596 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.290 | -33.1907 | -74.4031 | 41.2124 | 0.5831 | 70.6829 | N/A | 160.2555 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -40.9390 | -80.3540 | 39.4150 | 0.8628 | 45.6848 | N/A | 175.1066 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.570 | -58.3222 | -86.5142 | 28.1920 | 1.1425 | 24.6754 | N/A | 154.6638 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.710 | -71.5763 | -92.8245 | 21.2482 | 1.4223 | 14.9392 | N/A | 134.7186 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -82.4675 | -99.7090 | 17.2416 | 1.7221 | 10.0118 | N/A | 122.0672 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.000 | -91.2345 | -106.2229 | 14.9883 | 2.0020 | 7.4868 | N/A | 115.4939 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.140 | -99.2491 | -112.8035 | 13.5545 | 2.2818 | 5.9401 | N/A | 112.5218 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -106.8164 | -119.9135 | 13.0971 | 2.5817 | 5.0730 | N/A | 116.5815 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.430 | -113.7255 | -126.5945 | 12.8690 | 2.8616 | 4.4971 | N/A | 121.6752 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.570 | -120.6259 | -133.3115 | 12.6856 | 3.1415 | 4.0380 | N/A | 126.9544 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.710 | -127.5267 | -140.0585 | 12.5318 | 3.4215 | 3.6627 | N/A | 132.3401 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.860 | -134.9209 | -147.3156 | 12.3946 | 3.7214 | 3.3307 | N/A | 138.2256 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 3.000 | -141.8226 | -154.1113 | 12.2887 | 4.0013 | 3.0712 | N/A | 143.8278 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |

## N = 60 (119 params)

**ΔE/gap: 20.0891 ± 23.2326 | P90=49.3340 | max=85.3261
|ΔE|/N: 3.58e-01
Var(H): 145.2724
Distribution: [P25=5.135 | P50=8.993 | P75=27.963 | P90=49.334]
Regions: critical=47.0308 | ordered=5.0893**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=155.5718.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -72.2814 | -76.0332 | 3.7518 | 0.1047 | 35.8270 | N/A | 10.0021 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -79.5496 | -82.1048 | 2.5552 | 0.2824 | 9.0480 | N/A | 8.5571 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.290 | -39.6581 | -89.3286 | 49.6705 | 0.5821 | 85.3261 | N/A | 192.8218 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -46.5397 | -96.4639 | 49.9241 | 0.8619 | 57.9222 | N/A | 216.5283 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.570 | -62.2332 | -103.8519 | 41.6187 | 1.1417 | 36.4518 | N/A | 214.7856 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.710 | -82.8488 | -111.4208 | 28.5721 | 1.4216 | 20.0985 | N/A | 177.8212 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -98.9455 | -119.6794 | 20.7339 | 1.7215 | 12.0443 | N/A | 146.7630 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.000 | -109.4954 | -127.4938 | 17.9984 | 2.0014 | 8.9930 | N/A | 138.6923 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.140 | -119.0996 | -135.3886 | 16.2891 | 2.2813 | 7.1403 | N/A | 135.2205 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -128.1621 | -143.9189 | 15.7568 | 2.5812 | 6.1045 | N/A | 140.2466 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.430 | -136.4510 | -151.9347 | 15.4838 | 2.8611 | 5.4118 | N/A | 146.3882 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.570 | -144.7317 | -159.9938 | 15.2621 | 3.1411 | 4.8589 | N/A | 152.7322 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.710 | -153.0126 | -168.0891 | 15.0765 | 3.4210 | 4.4070 | N/A | 159.2063 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.860 | -161.8851 | -176.7966 | 14.9115 | 3.7210 | 4.0074 | N/A | 166.2883 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 3.000 | -170.1663 | -184.9506 | 14.7843 | 4.0009 | 3.6952 | N/A | 173.0326 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
