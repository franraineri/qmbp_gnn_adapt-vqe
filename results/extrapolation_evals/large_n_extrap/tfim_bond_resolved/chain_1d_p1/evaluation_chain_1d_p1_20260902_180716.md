# Model Evaluation: chain_1d

**Date**: 2026-09-02 18:07 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__chain_1d_p1_signinv_fid_dot5_v1.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 3.0] (30 pts)
**Target N**: [8, 10, 12, 16, 20, 30, 40]

---

## N = 8 (15 params)

**ΔE/gap: 1.9539 ± 2.3073 | P90=4.9731 | max=9.8407
|ΔE|/N: 3.23e-01
Fidelity: mean=0.6932 min=0.2242 (exact)
Distribution: [P25=0.576 | P50=0.899 | P75=2.215 | P90=4.973]
Regions: critical=4.6509 | ordered=0.6068**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=9.841 is 5× the mean — median may be more representative N=8

**Fidelity: mean F=0.6932, min F=0.2242** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -9.4866 | -9.8380 | 0.3514 | 0.3691 | 0.9521 | 0.8551 | N/A | — | severe_error(0.95) | increase_p |  |
| 1.070 | -9.9717 | -10.2670 | 0.2953 | 0.4749 | 0.6218 | 0.8899 | N/A | — | severe_error(0.60) | increase_p |  |
| 1.140 | -4.9274 | -10.7137 | 5.7863 | 0.5880 | 9.8407 | 0.2242 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -5.6572 | -11.1749 | 5.5177 | 0.7065 | 7.8096 | 0.2668 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.280 | -6.6400 | -11.6479 | 5.0079 | 0.8291 | 6.0399 | 0.3299 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.340 | -7.5139 | -12.0612 | 4.5473 | 0.9367 | 4.8545 | 0.3887 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.410 | -8.4248 | -12.5512 | 4.1264 | 1.0645 | 3.8764 | 0.4482 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.480 | -9.2125 | -13.0482 | 3.8357 | 1.1942 | 3.2120 | 0.4952 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.550 | -9.9331 | -13.5514 | 3.6182 | 1.3254 | 2.7299 | 0.5338 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.620 | -10.7020 | -14.0597 | 3.3577 | 1.4579 | 2.3032 | 0.5753 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.690 | -11.4699 | -14.5726 | 3.1027 | 1.5913 | 1.9498 | 0.6150 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.760 | -12.2234 | -15.0894 | 2.8660 | 1.7255 | 1.6609 | 0.6514 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.830 | -12.9797 | -15.6096 | 2.6299 | 1.8604 | 1.4136 | 0.6865 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.900 | -13.7364 | -16.1329 | 2.3965 | 1.9959 | 1.2007 | 0.7199 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.970 | -14.4245 | -16.6589 | 2.2345 | 2.1319 | 1.0481 | 0.7450 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.030 | -14.9791 | -17.1118 | 2.1327 | 2.2487 | 0.9484 | 0.7618 | N/A | — | severe_error(0.95) | increase_p |  |
| 2.100 | -15.6176 | -17.6421 | 2.0244 | 2.3854 | 0.8487 | 0.7796 | N/A | — | severe_error(0.84) | increase_p |  |
| 2.170 | -16.2348 | -18.1743 | 1.9395 | 2.5223 | 0.7689 | 0.7943 | N/A | — | severe_error(0.76) | increase_p |  |
| 2.240 | -16.8061 | -18.7084 | 1.9023 | 2.6596 | 0.7153 | 0.8038 | N/A | — | severe_error(0.70) | increase_p |  |
| 2.310 | -17.3680 | -19.2441 | 1.8761 | 2.7970 | 0.6707 | 0.8117 | N/A | — | severe_error(0.65) | increase_p |  |
| 2.380 | -17.9273 | -19.7812 | 1.8540 | 2.9347 | 0.6317 | 0.8189 | N/A | — | severe_error(0.61) | increase_p |  |
| 2.450 | -18.4817 | -20.3197 | 1.8380 | 3.0726 | 0.5982 | 0.8251 | N/A | — | severe_error(0.58) | increase_p |  |
| 2.520 | -19.0345 | -20.8594 | 1.8249 | 3.2106 | 0.5684 | 0.8308 | N/A | — | severe_error(0.55) | increase_p |  |
| 2.590 | -19.5873 | -21.4003 | 1.8130 | 3.3488 | 0.5414 | 0.8361 | N/A | — | severe_error(0.52) | increase_p |  |
| 2.660 | -20.1401 | -21.9421 | 1.8020 | 3.4871 | 0.5168 | 0.8411 | N/A | — | severe_error(0.49) | increase_p |  |
| 2.720 | -20.6140 | -22.4073 | 1.7934 | 3.6057 | 0.4974 | 0.8451 | N/A | — | moderate_error(0.47) | refine_vqe |  |
| 2.790 | -21.1668 | -22.9509 | 1.7841 | 3.7442 | 0.4765 | 0.8495 | N/A | — | moderate_error(0.45) | refine_vqe |  |
| 2.860 | -21.7197 | -23.4953 | 1.7756 | 3.8828 | 0.4573 | 0.8536 | N/A | — | moderate_error(0.43) | refine_vqe |  |
| 2.930 | -22.2726 | -24.0404 | 1.7678 | 4.0215 | 0.4396 | 0.8575 | N/A | — | moderate_error(0.41) | refine_vqe |  |
| 3.000 | -22.8256 | -24.5863 | 1.7607 | 4.1602 | 0.4232 | 0.8612 | N/A | — | moderate_error(0.39) | refine_vqe |  |

## N = 10 (19 params)

**ΔE/gap: 2.7204 ± 3.3382 | P90=6.9555 | max=14.4166
|ΔE|/N: 3.33e-01
Fidelity: mean=0.6317 min=0.1428 (exact)
Distribution: [P25=0.753 | P50=1.181 | P75=2.980 | P90=6.956]
Regions: critical=6.6557 | ordered=0.7940**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=14.417 is 5× the mean — median may be more representative N=10

**Fidelity: mean F=0.6317, min F=0.1428** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -11.9046 | -12.3815 | 0.4769 | 0.2989 | 1.5954 | 0.7960 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.070 | -12.5092 | -12.9082 | 0.3990 | 0.4035 | 0.9887 | 0.8465 | N/A | — | severe_error(0.99) | increase_p |  |
| 1.140 | -6.0078 | -13.4593 | 7.4514 | 0.5169 | 14.4166 | 0.1428 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -6.9073 | -14.0299 | 7.1226 | 0.6364 | 11.1922 | 0.1788 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.280 | -8.1230 | -14.6165 | 6.4934 | 0.7603 | 8.5403 | 0.2345 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.340 | -9.2373 | -15.1297 | 5.8924 | 0.8692 | 6.7795 | 0.2911 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.410 | -10.4022 | -15.7387 | 5.3364 | 0.9984 | 5.3450 | 0.3520 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.480 | -11.4014 | -16.3571 | 4.9557 | 1.1295 | 4.3875 | 0.4015 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.550 | -12.3182 | -16.9834 | 4.6652 | 1.2620 | 3.6965 | 0.4436 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.620 | -13.2871 | -17.6165 | 4.3294 | 1.3957 | 3.1019 | 0.4889 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.690 | -14.2578 | -18.2556 | 3.9978 | 1.5303 | 2.6124 | 0.5334 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.760 | -15.2209 | -18.8997 | 3.6789 | 1.6656 | 2.2088 | 0.5760 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.830 | -16.1754 | -19.5484 | 3.3730 | 1.8014 | 1.8724 | 0.6165 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.900 | -17.1305 | -20.2010 | 3.0705 | 1.9378 | 1.5845 | 0.6557 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.970 | -17.9939 | -20.8571 | 2.8633 | 2.0746 | 1.3802 | 0.6852 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.030 | -18.6870 | -21.4220 | 2.7351 | 2.1921 | 1.2477 | 0.7050 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.100 | -19.4872 | -22.0837 | 2.5966 | 2.3295 | 1.1147 | 0.7262 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.170 | -20.2607 | -22.7480 | 2.4873 | 2.4671 | 1.0082 | 0.7439 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.240 | -20.9731 | -23.4146 | 2.4414 | 2.6049 | 0.9372 | 0.7552 | N/A | — | severe_error(0.93) | increase_p |  |
| 2.310 | -21.6729 | -24.0832 | 2.4103 | 2.7430 | 0.8787 | 0.7646 | N/A | — | severe_error(0.87) | increase_p |  |
| 2.380 | -22.3701 | -24.7538 | 2.3837 | 2.8812 | 0.8273 | 0.7731 | N/A | — | severe_error(0.82) | increase_p |  |
| 2.450 | -23.0625 | -25.4260 | 2.3635 | 3.0195 | 0.7828 | 0.7807 | N/A | — | severe_error(0.77) | increase_p |  |
| 2.520 | -23.7532 | -26.0999 | 2.3467 | 3.1580 | 0.7431 | 0.7877 | N/A | — | severe_error(0.73) | increase_p |  |
| 2.590 | -24.4440 | -26.7752 | 2.3312 | 3.2966 | 0.7072 | 0.7942 | N/A | — | severe_error(0.69) | increase_p |  |
| 2.660 | -25.1348 | -27.4518 | 2.3171 | 3.4353 | 0.6745 | 0.8003 | N/A | — | severe_error(0.66) | increase_p |  |
| 2.720 | -25.7269 | -28.0328 | 2.3059 | 3.5542 | 0.6488 | 0.8052 | N/A | — | severe_error(0.63) | increase_p |  |
| 2.790 | -26.4177 | -28.7116 | 2.2939 | 3.6931 | 0.6211 | 0.8106 | N/A | — | severe_error(0.60) | increase_p |  |
| 2.860 | -27.1086 | -29.3915 | 2.2829 | 3.8320 | 0.5957 | 0.8157 | N/A | — | severe_error(0.57) | increase_p |  |
| 2.930 | -27.7995 | -30.0724 | 2.2728 | 3.9710 | 0.5724 | 0.8205 | N/A | — | severe_error(0.55) | increase_p |  |
| 3.000 | -28.4905 | -30.7541 | 2.2637 | 4.1101 | 0.5508 | 0.8251 | N/A | — | severe_error(0.53) | increase_p |  |

## N = 12 (23 params)

**ΔE/gap: 3.5242 ± 4.4478 | P90=9.0025 | max=19.4257
|ΔE|/N: 3.40e-01
Fidelity: mean=0.5786 min=0.0909 (exact)
Distribution: [P25=0.930 | P50=1.482 | P75=3.751 | P90=9.002]
Regions: critical=8.7981 | ordered=0.9809**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=19.426 is 6× the mean — median may be more representative N=12

**Fidelity: mean F=0.5786, min F=0.0909** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -14.3220 | -14.9260 | 0.6040 | 0.2512 | 2.4048 | 0.7377 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.070 | -15.0463 | -15.5498 | 0.5035 | 0.3552 | 1.4174 | 0.8038 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.140 | -7.0891 | -16.2050 | 9.1159 | 0.4693 | 19.4257 | 0.0909 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -8.1624 | -16.8850 | 8.7226 | 0.5901 | 14.7825 | 0.1199 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.280 | -9.6047 | -17.5850 | 7.9803 | 0.7154 | 11.1544 | 0.1665 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.340 | -10.9639 | -18.1982 | 7.2342 | 0.8255 | 8.7634 | 0.2182 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.410 | -12.3732 | -18.9262 | 6.5530 | 0.9561 | 6.8538 | 0.2760 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.480 | -13.5894 | -19.6659 | 6.0765 | 1.0885 | 5.5826 | 0.3254 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.550 | -14.7069 | -20.4154 | 5.7085 | 1.2222 | 4.6708 | 0.3689 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.620 | -15.8688 | -21.1733 | 5.3045 | 1.3569 | 3.9094 | 0.4153 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.690 | -17.0499 | -21.9386 | 4.8887 | 1.4924 | 3.2758 | 0.4629 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.760 | -18.2138 | -22.7101 | 4.4963 | 1.6285 | 2.7610 | 0.5089 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.830 | -19.3717 | -23.4871 | 4.1155 | 1.7651 | 2.3315 | 0.5537 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.900 | -20.5249 | -24.2691 | 3.7441 | 1.9022 | 1.9683 | 0.5973 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.970 | -21.5635 | -25.0553 | 3.4919 | 2.0396 | 1.7121 | 0.6302 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.030 | -22.3951 | -25.7323 | 3.3373 | 2.1576 | 1.5468 | 0.6524 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.100 | -23.3569 | -26.5254 | 3.1685 | 2.2955 | 1.3803 | 0.6765 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.170 | -24.2867 | -27.3216 | 3.0350 | 2.4336 | 1.2471 | 0.6967 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.240 | -25.1402 | -28.1207 | 2.9805 | 2.5719 | 1.1589 | 0.7095 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.310 | -25.9779 | -28.9223 | 2.9444 | 2.7103 | 1.0864 | 0.7202 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.380 | -26.8129 | -29.7263 | 2.9133 | 2.8489 | 1.0226 | 0.7300 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.450 | -27.6433 | -30.5323 | 2.8891 | 2.9876 | 0.9670 | 0.7388 | N/A | — | severe_error(0.97) | increase_p |  |
| 2.520 | -28.4719 | -31.3403 | 2.8684 | 3.1264 | 0.9175 | 0.7468 | N/A | — | severe_error(0.91) | increase_p |  |
| 2.590 | -29.3006 | -32.1501 | 2.8495 | 3.2653 | 0.8727 | 0.7544 | N/A | — | severe_error(0.87) | increase_p |  |
| 2.660 | -30.1294 | -32.9615 | 2.8322 | 3.4043 | 0.8319 | 0.7615 | N/A | — | severe_error(0.82) | increase_p |  |
| 2.720 | -30.8398 | -33.6582 | 2.8185 | 3.5234 | 0.7999 | 0.7672 | N/A | — | severe_error(0.79) | increase_p |  |
| 2.790 | -31.6686 | -34.4723 | 2.8037 | 3.6626 | 0.7655 | 0.7735 | N/A | — | severe_error(0.75) | increase_p |  |
| 2.860 | -32.4975 | -35.2877 | 2.7902 | 3.8017 | 0.7339 | 0.7795 | N/A | — | severe_error(0.72) | increase_p |  |
| 2.930 | -33.3264 | -36.1043 | 2.7779 | 3.9409 | 0.7049 | 0.7851 | N/A | — | severe_error(0.69) | increase_p |  |
| 3.000 | -34.1553 | -36.9220 | 2.7667 | 4.0802 | 0.6781 | 0.7905 | N/A | — | severe_error(0.66) | increase_p |  |

## N = 16 (31 params)

**ΔE/gap: 5.2102 ± 6.8325 | P90=13.2049 | max=30.3464
|ΔE|/N: 3.48e-01
Fidelity: mean=0.4913 min=0.0367 (exact)
Distribution: [P25=1.282 | P50=2.258 | P75=5.292 | P90=13.205]
Regions: critical=13.3791 | ordered=1.3534**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=30.346 is 6× the mean — median may be more representative N=16

**Fidelity: mean F=0.4913, min F=0.0367** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -19.1552 | -20.0164 | 0.8612 | 0.1903 | 4.5250 | 0.6282 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.070 | -20.1194 | -20.8335 | 0.7141 | 0.2943 | 2.4268 | 0.7227 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -9.2447 | -21.6966 | 12.4519 | 0.4103 | 30.3464 | 0.0367 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -10.6390 | -22.5953 | 11.9562 | 0.5338 | 22.3995 | 0.0534 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.280 | -12.5319 | -23.5222 | 10.9903 | 0.6618 | 16.6060 | 0.0831 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.340 | -14.4069 | -24.3351 | 9.9283 | 0.7740 | 12.8270 | 0.1222 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.410 | -16.3128 | -25.3013 | 8.9884 | 0.9068 | 9.9119 | 0.1695 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.480 | -17.9649 | -26.2836 | 8.3187 | 1.0412 | 7.9899 | 0.2137 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.550 | -19.4788 | -27.2795 | 7.8007 | 1.1766 | 6.6301 | 0.2548 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.620 | -21.0398 | -28.2870 | 7.2472 | 1.3128 | 5.5205 | 0.2999 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.690 | -22.6295 | -29.3045 | 6.6750 | 1.4496 | 4.6048 | 0.3484 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.760 | -24.2006 | -30.3308 | 6.1302 | 1.5869 | 3.8630 | 0.3973 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.830 | -25.7650 | -31.3646 | 5.5996 | 1.7246 | 3.2470 | 0.4467 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.900 | -27.3148 | -32.4052 | 5.0904 | 1.8625 | 2.7331 | 0.4956 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.970 | -28.7029 | -33.4517 | 4.7488 | 2.0008 | 2.3735 | 0.5331 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.030 | -29.8117 | -34.3529 | 4.5413 | 2.1194 | 2.1427 | 0.5587 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.100 | -31.0968 | -35.4088 | 4.3120 | 2.2580 | 1.9097 | 0.5870 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.170 | -32.3387 | -36.4689 | 4.1302 | 2.3967 | 1.7233 | 0.6111 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.240 | -33.4745 | -37.5330 | 4.0585 | 2.5356 | 1.6006 | 0.6263 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.310 | -34.5880 | -38.6006 | 4.0125 | 2.6746 | 1.5003 | 0.6390 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.380 | -35.6989 | -39.6713 | 3.9724 | 2.8136 | 1.4119 | 0.6508 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.450 | -36.8051 | -40.7449 | 3.9399 | 2.9528 | 1.3343 | 0.6614 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.520 | -37.9096 | -41.8212 | 3.9117 | 3.0920 | 1.2651 | 0.6713 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.590 | -39.0142 | -42.9000 | 3.8858 | 3.2313 | 1.2026 | 0.6807 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.660 | -40.1189 | -43.9809 | 3.8620 | 3.3706 | 1.1458 | 0.6894 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.720 | -41.0658 | -44.9091 | 3.8433 | 3.4901 | 1.1012 | 0.6965 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.790 | -42.1706 | -45.9937 | 3.8231 | 3.6295 | 1.0533 | 0.7044 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.860 | -43.2755 | -47.0801 | 3.8047 | 3.7689 | 1.0095 | 0.7118 | N/A | — | severe_error(1.00) | increase_p |  |
| 2.930 | -44.3804 | -48.1681 | 3.7878 | 3.9084 | 0.9691 | 0.7189 | N/A | — | severe_error(0.97) | increase_p |  |
| 3.000 | -45.4853 | -49.2577 | 3.7724 | 4.0480 | 0.9319 | 0.7256 | N/A | — | severe_error(0.93) | increase_p |  |

## N = 20 (39 params)

**ΔE/gap: 7.4597 ± 11.0447 | P90=19.0392 | max=50.5467
|ΔE|/N: 3.55e-01
Var(H): 47.8584
Distribution: [P25=1.647 | P50=2.859 | P75=7.000 | P90=19.039]
Regions: critical=19.9445 | ordered=1.7400**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=50.547 is 7× the mean — median may be more representative N=20

**Infidelity decomposition:** 30 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=53.2699.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -23.9869 | -25.1078 | 1.1209 | 0.3142 | 3.5680 | N/A | 3.0870 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.070 | -25.1914 | -26.1175 | 0.9261 | 0.3142 | 2.9478 | N/A | 2.8564 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -11.3085 | -27.1882 | 15.8797 | 0.3142 | 50.5467 | N/A | 54.7046 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -12.9467 | -28.3055 | 15.3589 | 0.4404 | 34.8754 | N/A | 57.3155 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.280 | -15.1435 | -29.4594 | 14.3159 | 0.5793 | 24.7135 | N/A | 58.4789 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.340 | -17.6152 | -30.4721 | 12.8569 | 0.6984 | 18.4087 | N/A | 57.3448 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.410 | -20.2322 | -31.6763 | 11.4441 | 0.8375 | 13.6646 | N/A | 55.8241 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.480 | -22.3229 | -32.9012 | 10.5784 | 0.9767 | 10.8310 | N/A | 55.4415 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.550 | -24.2614 | -34.1435 | 9.8822 | 1.1159 | 8.8556 | N/A | 55.1452 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.620 | -26.2204 | -35.4006 | 9.1802 | 1.2552 | 7.3136 | N/A | 54.3232 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.690 | -28.2188 | -36.6705 | 8.4518 | 1.3946 | 6.0603 | N/A | 52.9046 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.760 | -30.1955 | -37.9515 | 7.7560 | 1.5340 | 5.0560 | N/A | 51.1832 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.830 | -32.1630 | -39.2422 | 7.0791 | 1.6735 | 4.2302 | N/A | 49.1051 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.900 | -34.1146 | -40.5413 | 6.4268 | 1.8130 | 3.5449 | N/A | 46.7275 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.970 | -35.8468 | -41.8481 | 6.0013 | 1.9525 | 3.0736 | N/A | 45.4985 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.030 | -37.2326 | -42.9735 | 5.7409 | 2.0722 | 2.7705 | N/A | 44.9960 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.100 | -38.8408 | -44.2921 | 5.4514 | 2.2117 | 2.4647 | N/A | 44.3541 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.170 | -40.3923 | -45.6162 | 5.2240 | 2.3514 | 2.2217 | N/A | 44.0350 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.240 | -41.8099 | -46.9453 | 5.1354 | 2.4910 | 2.0616 | N/A | 44.7269 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.310 | -43.1993 | -48.2788 | 5.0795 | 2.6307 | 1.9309 | N/A | 45.6486 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.380 | -44.5858 | -49.6163 | 5.0305 | 2.7704 | 1.8158 | N/A | 46.6010 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.450 | -45.9679 | -50.9575 | 4.9896 | 2.9101 | 1.7146 | N/A | 47.6023 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.520 | -47.3483 | -52.3021 | 4.9538 | 3.0498 | 1.6243 | N/A | 48.6291 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.590 | -48.7288 | -53.6498 | 4.9210 | 3.1895 | 1.5428 | N/A | 49.6660 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.660 | -50.1094 | -55.0003 | 4.8909 | 3.3293 | 1.4690 | N/A | 50.7129 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.720 | -51.2929 | -56.1599 | 4.8671 | 3.4491 | 1.4111 | N/A | 51.6180 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.790 | -52.6736 | -57.5151 | 4.8415 | 3.5888 | 1.3490 | N/A | 52.6831 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.860 | -54.0545 | -58.8725 | 4.8180 | 3.7286 | 1.2922 | N/A | 53.7581 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.930 | -55.4354 | -60.2320 | 4.7966 | 3.8684 | 1.2399 | N/A | 54.8429 | dirty_state | severe_error(1.00) | increase_p |  |
| 3.000 | -56.8164 | -61.5934 | 4.7771 | 4.0082 | 1.1918 | N/A | 55.9374 | dirty_state | severe_error(1.00) | increase_p |  |

## N = 30 (59 params)

**ΔE/gap: 12.2151 ± 18.3109 | P90=32.3340 | max=84.1486
|ΔE|/N: 3.67e-01
Var(H): 73.3767
Distribution: [P25=2.517 | P50=4.457 | P75=10.738 | P90=32.334]
Regions: critical=33.5203 | ordered=2.6571**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=84.149 is 7× the mean — median may be more representative N=30

**Infidelity decomposition:** 30 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=92.8384.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -36.0628 | -37.8381 | 1.7753 | 0.2094 | 8.4762 | N/A | 4.8024 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.070 | -37.8685 | -39.3276 | 1.4591 | 0.2094 | 6.9666 | N/A | 4.4393 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -16.5463 | -40.9173 | 24.3711 | 0.2896 | 84.1486 | N/A | 83.3423 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -18.2996 | -42.5812 | 24.2815 | 0.4291 | 56.5920 | N/A | 88.6334 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.280 | -20.9932 | -44.3024 | 23.3092 | 0.5686 | 40.9963 | N/A | 92.2121 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.340 | -24.2252 | -45.8145 | 21.5893 | 0.6882 | 31.3715 | N/A | 92.6289 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.410 | -28.8038 | -47.6140 | 18.8102 | 0.8278 | 22.7238 | N/A | 89.3529 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.480 | -33.1086 | -49.4454 | 16.3368 | 0.9674 | 16.8872 | N/A | 85.0820 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.550 | -36.1694 | -51.3036 | 15.1343 | 1.1071 | 13.6705 | N/A | 84.2054 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.620 | -39.1921 | -53.1847 | 13.9927 | 1.2468 | 11.2231 | N/A | 82.6862 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.690 | -42.2174 | -55.0855 | 12.8681 | 1.3865 | 9.2811 | N/A | 80.4717 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.760 | -45.2118 | -57.0033 | 11.7915 | 1.5262 | 7.7259 | N/A | 77.7654 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.830 | -48.1887 | -58.9359 | 10.7472 | 1.6660 | 6.4509 | N/A | 74.5201 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.900 | -51.1406 | -60.8817 | 9.7411 | 1.8058 | 5.3944 | N/A | 70.8113 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.970 | -53.7178 | -62.8390 | 9.1212 | 1.9456 | 4.6882 | N/A | 69.1252 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.030 | -55.7964 | -64.5250 | 8.7286 | 2.0654 | 4.2261 | N/A | 68.3875 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.100 | -58.2120 | -66.5005 | 8.2886 | 2.2052 | 3.7586 | N/A | 67.4157 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.170 | -60.5302 | -68.4845 | 7.9543 | 2.3451 | 3.3919 | N/A | 67.0234 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.240 | -62.6516 | -70.4760 | 7.8244 | 2.4849 | 3.1488 | N/A | 68.1197 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.310 | -64.7306 | -72.4743 | 7.7437 | 2.6247 | 2.9503 | N/A | 69.5633 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.380 | -66.8065 | -74.4789 | 7.6724 | 2.7646 | 2.7752 | N/A | 71.0463 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.450 | -68.8783 | -76.4890 | 7.6108 | 2.9045 | 2.6204 | N/A | 72.5804 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.520 | -70.9484 | -78.5044 | 7.5559 | 3.0444 | 2.4819 | N/A | 74.1454 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.590 | -73.0187 | -80.5244 | 7.5057 | 3.1842 | 2.3571 | N/A | 75.7260 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.660 | -75.0891 | -82.5488 | 7.4596 | 3.3241 | 2.2441 | N/A | 77.3217 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.720 | -76.8638 | -84.2871 | 7.4233 | 3.4440 | 2.1554 | N/A | 78.7015 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.790 | -78.9345 | -86.3186 | 7.3841 | 3.5839 | 2.0603 | N/A | 80.3252 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.860 | -81.0052 | -88.3535 | 7.3483 | 3.7238 | 1.9733 | N/A | 81.9640 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.930 | -83.0761 | -90.3916 | 7.3156 | 3.8637 | 1.8934 | N/A | 83.6175 | dirty_state | severe_error(1.00) | increase_p |  |
| 3.000 | -85.1471 | -92.4327 | 7.2856 | 4.0037 | 1.8197 | N/A | 85.2853 | dirty_state | severe_error(1.00) | increase_p |  |

## N = 40 (79 params)

**ΔE/gap: 17.2429 ± 25.3101 | P90=47.6832 | max=114.9818
|ΔE|/N: 3.79e-01
Var(H): 99.5358
Distribution: [P25=3.384 | P50=5.982 | P75=15.403 | P90=47.683]
Regions: critical=48.0454 | ordered=3.5711**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=114.982 is 7× the mean — median may be more representative N=40

**Infidelity decomposition:** 30 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=135.9171.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -48.1366 | -50.5694 | 2.4328 | 0.1571 | 15.4876 | N/A | 6.5236 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.070 | -50.5438 | -52.5378 | 1.9941 | 0.1571 | 12.6946 | N/A | 6.0285 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -21.8294 | -54.6465 | 32.8171 | 0.2854 | 114.9818 | N/A | 111.8555 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -24.1094 | -56.8569 | 32.7475 | 0.4251 | 77.0352 | N/A | 119.0986 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.280 | -26.4166 | -59.1454 | 32.7288 | 0.5648 | 57.9456 | N/A | 126.5020 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.340 | -29.2935 | -61.1569 | 31.8634 | 0.6846 | 46.5429 | N/A | 131.1543 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.410 | -35.0509 | -63.5517 | 28.5007 | 0.8244 | 34.5725 | N/A | 129.7178 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.480 | -41.7857 | -65.9896 | 24.2039 | 0.9642 | 25.1034 | N/A | 122.3116 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.550 | -47.2890 | -68.4638 | 21.1747 | 1.1040 | 19.1804 | N/A | 116.5404 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.620 | -52.1238 | -70.9689 | 18.8451 | 1.2438 | 15.1511 | N/A | 111.2022 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.690 | -56.2056 | -73.5005 | 17.2948 | 1.3837 | 12.4994 | N/A | 108.0736 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.760 | -60.2569 | -76.0550 | 15.7981 | 1.5235 | 10.3696 | N/A | 104.1758 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.830 | -64.2487 | -78.6297 | 14.3811 | 1.6634 | 8.6457 | N/A | 99.7197 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.900 | -68.2020 | -81.2220 | 13.0201 | 1.8032 | 7.2203 | N/A | 94.6579 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.970 | -71.6031 | -83.8300 | 12.2269 | 1.9431 | 6.2924 | N/A | 92.6500 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.030 | -74.3746 | -86.0764 | 11.7019 | 2.0630 | 5.6722 | N/A | 91.6720 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.100 | -77.5976 | -88.7089 | 11.1114 | 2.2029 | 5.0439 | N/A | 90.3658 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.170 | -80.6730 | -91.3528 | 10.6797 | 2.3428 | 4.5585 | N/A | 89.9719 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.240 | -83.4970 | -94.0067 | 10.5097 | 2.4828 | 4.2331 | N/A | 91.4805 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.310 | -86.2659 | -96.6699 | 10.4040 | 2.6227 | 3.9670 | N/A | 93.4435 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.380 | -89.0312 | -99.3414 | 10.3102 | 2.7626 | 3.7321 | N/A | 95.4549 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.450 | -91.7928 | -102.0206 | 10.2277 | 2.9025 | 3.5237 | N/A | 97.5198 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.520 | -94.5529 | -104.7066 | 10.1537 | 3.0424 | 3.3373 | N/A | 99.6207 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.590 | -97.3131 | -107.3990 | 10.0859 | 3.1824 | 3.1693 | N/A | 101.7427 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.660 | -100.0733 | -110.0972 | 10.0239 | 3.3223 | 3.0171 | N/A | 103.8858 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.720 | -102.4394 | -112.4143 | 9.9749 | 3.4423 | 2.8978 | N/A | 105.7389 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.790 | -105.1999 | -115.1221 | 9.9222 | 3.5822 | 2.7698 | N/A | 107.9199 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.860 | -107.9606 | -117.8346 | 9.8739 | 3.7222 | 2.6527 | N/A | 110.1211 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.930 | -110.7215 | -120.5513 | 9.8298 | 3.8621 | 2.5452 | N/A | 112.3422 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 3.000 | -113.4824 | -123.2720 | 9.7896 | 4.0021 | 2.4461 | N/A | 114.5841 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
