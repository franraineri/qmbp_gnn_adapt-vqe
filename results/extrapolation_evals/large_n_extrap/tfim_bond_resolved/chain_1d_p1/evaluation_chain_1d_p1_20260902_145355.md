# Model Evaluation: chain_1d

**Date**: 2026-09-02 14:53 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__chain_1d_p1_signinv_fid_v1.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 3.0] (30 pts)
**Target N**: [8, 10, 12, 16, 20, 30, 40]

---

## N = 8 (15 params)

**ΔE/gap: 0.2859 ± 0.3981 | P90=0.7371 | max=1.7518
|ΔE|/N: 3.48e-02
Fidelity: mean=0.9424 min=0.7718 (exact)
Distribution: [P25=0.038 | P50=0.110 | P75=0.354 | P90=0.737]
Regions: critical=0.8124 | ordered=0.0448**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=1.752 is 6× the mean — median may be more representative N=8

**Fidelity: mean F=0.9424, min F=0.7718** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -9.1914 | -9.8380 | 0.6465 | 0.3691 | 1.7518 | 0.7718 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.070 | -9.6817 | -10.2670 | 0.5853 | 0.4749 | 1.2325 | 0.8120 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.140 | -10.1740 | -10.7137 | 0.5397 | 0.5880 | 0.9178 | 0.8421 | N/A | — | severe_error(0.91) | increase_p |  |
| 1.210 | -10.6683 | -11.1749 | 0.5066 | 0.7065 | 0.7170 | 0.8646 | N/A | — | severe_error(0.70) | increase_p |  |
| 1.280 | -11.1641 | -11.6479 | 0.4838 | 0.8291 | 0.5835 | 0.8814 | N/A | — | severe_error(0.56) | increase_p |  |
| 1.340 | -11.5916 | -12.0612 | 0.4696 | 0.9367 | 0.5013 | 0.8927 | N/A | — | severe_error(0.48) | increase_p |  |
| 1.410 | -12.0961 | -12.5512 | 0.4551 | 1.0645 | 0.4275 | 0.9035 | N/A | — | moderate_error(0.40) | refine_vqe |  |
| 1.480 | -12.6093 | -13.0482 | 0.4389 | 1.1942 | 0.3675 | 0.9131 | N/A | — | moderate_error(0.33) | refine_vqe |  |
| 1.550 | -13.1367 | -13.5514 | 0.4147 | 1.3254 | 0.3129 | 0.9226 | N/A | — | moderate_error(0.28) | refine_vqe |  |
| 1.620 | -13.6750 | -14.0597 | 0.3847 | 1.4579 | 0.2639 | 0.9318 | N/A | — | moderate_error(0.23) | refine_vqe |  |
| 1.690 | -14.2201 | -14.5726 | 0.3525 | 1.5913 | 0.2215 | 0.9404 | N/A | — | moderate_error(0.18) | refine_vqe |  |
| 1.760 | -14.7627 | -15.0894 | 0.3266 | 1.7255 | 0.1893 | 0.9473 | N/A | — | moderate_error(0.15) | refine_vqe |  |
| 1.830 | -15.3083 | -15.6096 | 0.3014 | 1.8604 | 0.1620 | 0.9534 | N/A | — | moderate_error(0.12) | refine_vqe |  |
| 1.900 | -15.8578 | -16.1329 | 0.2751 | 1.9959 | 0.1379 | 0.9592 | N/A | — | moderate_error(0.09) | refine_vqe |  |
| 1.970 | -16.4089 | -16.6589 | 0.2500 | 2.1319 | 0.1173 | 0.9643 | N/A | — | moderate_error(0.07) | refine_vqe |  |
| 2.030 | -16.8814 | -17.1118 | 0.2304 | 2.2487 | 0.1025 | 0.9681 | N/A | — | moderate_error(0.06) | refine_vqe |  |
| 2.100 | -17.4326 | -17.6421 | 0.2095 | 2.3854 | 0.0878 | 0.9720 | N/A | — | near_pass(0.04) | refine_vqe |  |
| 2.170 | -17.9838 | -18.1743 | 0.1905 | 2.5223 | 0.0755 | 0.9754 | N/A | — | near_pass(0.03) | refine_vqe |  |
| 2.240 | -18.5351 | -18.7084 | 0.1734 | 2.6596 | 0.0652 | 0.9783 | N/A | — | near_pass(0.02) | refine_vqe |  |
| 2.310 | -19.0864 | -19.2441 | 0.1577 | 2.7970 | 0.0564 | 0.9809 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 2.380 | -19.6379 | -19.7812 | 0.1434 | 2.9347 | 0.0488 | 0.9831 | N/A | — | gap_masked(0.48) | refine_vqe |  |
| 2.450 | -20.1895 | -20.3197 | 0.1302 | 3.0726 | 0.0424 | 0.9851 | N/A | — | gap_masked(0.43) | refine_vqe |  |
| 2.520 | -20.7412 | -20.8594 | 0.1182 | 3.2106 | 0.0368 | 0.9869 | N/A | — | gap_masked(0.39) | refine_vqe |  |
| 2.590 | -21.2929 | -21.4003 | 0.1073 | 3.3488 | 0.0321 | 0.9884 | N/A | — | gap_masked(0.36) | refine_vqe |  |
| 2.660 | -21.8447 | -21.9421 | 0.0974 | 3.4871 | 0.0279 | 0.9898 | N/A | — | pass(0.56) | none |  |
| 2.720 | -22.3177 | -22.4073 | 0.0896 | 3.6057 | 0.0249 | 0.9908 | N/A | — | pass(0.50) | none |  |
| 2.790 | -22.8696 | -22.9509 | 0.0813 | 3.7442 | 0.0217 | 0.9919 | N/A | — | pass(0.43) | none |  |
| 2.860 | -23.4216 | -23.4953 | 0.0737 | 3.8828 | 0.0190 | 0.9928 | N/A | — | pass(0.38) | none |  |
| 2.930 | -23.9736 | -24.0404 | 0.0669 | 4.0215 | 0.0166 | 0.9936 | N/A | — | pass(0.33) | none |  |
| 3.000 | -24.5255 | -24.5863 | 0.0607 | 4.1602 | 0.0146 | 0.9944 | N/A | — | pass(0.29) | none |  |

## N = 10 (19 params)

**ΔE/gap: 0.4270 ± 0.6458 | P90=1.0947 | max=2.9323
|ΔE|/N: 3.67e-02
Fidelity: mean=0.9239 min=0.6921 (exact)
Distribution: [P25=0.051 | P50=0.147 | P75=0.489 | P90=1.095]
Regions: critical=1.2510 | ordered=0.0596**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=2.932 is 7× the mean — median may be more representative N=10

**Fidelity: mean F=0.9239, min F=0.6921** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -11.5050 | -12.3815 | 0.8765 | 0.2989 | 2.9323 | 0.6921 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.070 | -12.1197 | -12.9082 | 0.7884 | 0.4035 | 1.9538 | 0.7479 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.140 | -12.7362 | -13.4593 | 0.7231 | 0.5169 | 1.3989 | 0.7895 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.210 | -13.3548 | -14.0299 | 0.6751 | 0.6364 | 1.0609 | 0.8204 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.280 | -13.9758 | -14.6165 | 0.6407 | 0.7603 | 0.8426 | 0.8436 | N/A | — | severe_error(0.83) | increase_p |  |
| 1.340 | -14.5107 | -15.1297 | 0.6190 | 0.8692 | 0.7122 | 0.8591 | N/A | — | severe_error(0.70) | increase_p |  |
| 1.410 | -15.1411 | -15.7387 | 0.5975 | 0.9984 | 0.5985 | 0.8738 | N/A | — | severe_error(0.58) | increase_p |  |
| 1.480 | -15.7821 | -16.3571 | 0.5750 | 1.1295 | 0.5091 | 0.8865 | N/A | — | severe_error(0.48) | increase_p |  |
| 1.550 | -16.4403 | -16.9834 | 0.5431 | 1.2620 | 0.4304 | 0.8989 | N/A | — | moderate_error(0.40) | refine_vqe |  |
| 1.620 | -17.1126 | -17.6165 | 0.5039 | 1.3957 | 0.3610 | 0.9109 | N/A | — | moderate_error(0.33) | refine_vqe |  |
| 1.690 | -17.7943 | -18.2556 | 0.4613 | 1.5303 | 0.3014 | 0.9222 | N/A | — | moderate_error(0.26) | refine_vqe |  |
| 1.760 | -18.4744 | -18.8997 | 0.4253 | 1.6656 | 0.2553 | 0.9314 | N/A | — | moderate_error(0.22) | refine_vqe |  |
| 1.830 | -19.1565 | -19.5484 | 0.3919 | 1.8014 | 0.2175 | 0.9395 | N/A | — | moderate_error(0.18) | refine_vqe |  |
| 1.900 | -19.8430 | -20.2010 | 0.3580 | 1.9378 | 0.1848 | 0.9468 | N/A | — | moderate_error(0.14) | refine_vqe |  |
| 1.970 | -20.5315 | -20.8571 | 0.3256 | 2.0746 | 0.1569 | 0.9534 | N/A | — | moderate_error(0.11) | refine_vqe |  |
| 2.030 | -21.1218 | -21.4220 | 0.3002 | 2.1921 | 0.1370 | 0.9584 | N/A | — | moderate_error(0.09) | refine_vqe |  |
| 2.100 | -21.8106 | -22.0837 | 0.2732 | 2.3295 | 0.1173 | 0.9634 | N/A | — | moderate_error(0.07) | refine_vqe |  |
| 2.170 | -22.4995 | -22.7480 | 0.2485 | 2.4671 | 0.1007 | 0.9678 | N/A | — | moderate_error(0.05) | refine_vqe |  |
| 2.240 | -23.1885 | -23.4146 | 0.2261 | 2.6049 | 0.0868 | 0.9717 | N/A | — | near_pass(0.04) | refine_vqe |  |
| 2.310 | -23.8776 | -24.0832 | 0.2056 | 2.7430 | 0.0750 | 0.9750 | N/A | — | near_pass(0.03) | refine_vqe |  |
| 2.380 | -24.5669 | -24.7538 | 0.1868 | 2.8812 | 0.0648 | 0.9780 | N/A | — | near_pass(0.02) | refine_vqe |  |
| 2.450 | -25.2564 | -25.4260 | 0.1696 | 3.0195 | 0.0562 | 0.9806 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 2.520 | -25.9459 | -26.0999 | 0.1540 | 3.1580 | 0.0488 | 0.9829 | N/A | — | gap_masked(0.51) | refine_vqe |  |
| 2.590 | -26.6354 | -26.7752 | 0.1398 | 3.2966 | 0.0424 | 0.9849 | N/A | — | gap_masked(0.47) | refine_vqe |  |
| 2.660 | -27.3250 | -27.4518 | 0.1268 | 3.4353 | 0.0369 | 0.9866 | N/A | — | gap_masked(0.42) | refine_vqe |  |
| 2.720 | -27.9161 | -28.0328 | 0.1167 | 3.5542 | 0.0328 | 0.9880 | N/A | — | gap_masked(0.39) | refine_vqe |  |
| 2.790 | -28.6058 | -28.7116 | 0.1058 | 3.6931 | 0.0286 | 0.9894 | N/A | — | gap_masked(0.35) | refine_vqe |  |
| 2.860 | -29.2955 | -29.3915 | 0.0960 | 3.8320 | 0.0250 | 0.9906 | N/A | — | pass(0.50) | none |  |
| 2.930 | -29.9852 | -30.0724 | 0.0871 | 3.9710 | 0.0219 | 0.9917 | N/A | — | pass(0.44) | none |  |
| 3.000 | -30.6750 | -30.7541 | 0.0791 | 4.1101 | 0.0192 | 0.9926 | N/A | — | pass(0.38) | none |  |

## N = 12 (23 params)

**ΔE/gap: 0.5884 ± 0.9542 | P90=1.4894 | max=4.4444
|ΔE|/N: 3.81e-02
Fidelity: mean=0.9060 min=0.6168 (exact)
Distribution: [P25=0.063 | P50=0.184 | P75=0.628 | P90=1.489]
Regions: critical=1.7654 | ordered=0.0743**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=4.444 is 8× the mean — median may be more representative N=12

**Fidelity: mean F=0.9060, min F=0.6168** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -13.8097 | -14.9260 | 1.1163 | 0.2512 | 4.4444 | 0.6168 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.070 | -14.5512 | -15.5498 | 0.9986 | 0.3552 | 2.8111 | 0.6867 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -15.2931 | -16.2050 | 0.9118 | 0.4693 | 1.9431 | 0.7388 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.210 | -16.0359 | -16.8850 | 0.8491 | 0.5901 | 1.4390 | 0.7774 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.280 | -16.7823 | -17.5850 | 0.8028 | 0.7154 | 1.1220 | 0.8065 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.340 | -17.4264 | -18.1982 | 0.7718 | 0.8255 | 0.9349 | 0.8262 | N/A | — | severe_error(0.93) | increase_p |  |
| 1.410 | -18.1850 | -18.9262 | 0.7412 | 0.9561 | 0.7752 | 0.8449 | N/A | — | severe_error(0.76) | increase_p |  |
| 1.480 | -18.9545 | -19.6659 | 0.7114 | 1.0885 | 0.6536 | 0.8607 | N/A | — | severe_error(0.64) | increase_p |  |
| 1.550 | -19.7437 | -20.4154 | 0.6717 | 1.2222 | 0.5496 | 0.8759 | N/A | — | severe_error(0.53) | increase_p |  |
| 1.620 | -20.5506 | -21.1733 | 0.6228 | 1.3569 | 0.4590 | 0.8906 | N/A | — | moderate_error(0.43) | refine_vqe |  |
| 1.690 | -21.3680 | -21.9386 | 0.5705 | 1.4924 | 0.3823 | 0.9042 | N/A | — | moderate_error(0.35) | refine_vqe |  |
| 1.760 | -22.1863 | -22.7101 | 0.5238 | 1.6285 | 0.3217 | 0.9158 | N/A | — | moderate_error(0.29) | refine_vqe |  |
| 1.830 | -23.0049 | -23.4871 | 0.4822 | 1.7651 | 0.2732 | 0.9257 | N/A | — | moderate_error(0.23) | refine_vqe |  |
| 1.900 | -23.8283 | -24.2691 | 0.4408 | 1.9022 | 0.2317 | 0.9347 | N/A | — | moderate_error(0.19) | refine_vqe |  |
| 1.970 | -24.6543 | -25.0553 | 0.4010 | 2.0396 | 0.1966 | 0.9427 | N/A | — | moderate_error(0.15) | refine_vqe |  |
| 2.030 | -25.3625 | -25.7323 | 0.3698 | 2.1576 | 0.1714 | 0.9488 | N/A | — | moderate_error(0.13) | refine_vqe |  |
| 2.100 | -26.1888 | -26.5254 | 0.3366 | 2.2955 | 0.1466 | 0.9550 | N/A | — | moderate_error(0.10) | refine_vqe |  |
| 2.170 | -27.0155 | -27.3216 | 0.3062 | 2.4336 | 0.1258 | 0.9604 | N/A | — | moderate_error(0.08) | refine_vqe |  |
| 2.240 | -27.8422 | -28.1207 | 0.2785 | 2.5719 | 0.1083 | 0.9651 | N/A | — | moderate_error(0.06) | refine_vqe |  |
| 2.310 | -28.6688 | -28.9223 | 0.2535 | 2.7103 | 0.0935 | 0.9692 | N/A | — | near_pass(0.05) | refine_vqe |  |
| 2.380 | -29.4958 | -29.7263 | 0.2305 | 2.8489 | 0.0809 | 0.9728 | N/A | — | near_pass(0.03) | refine_vqe |  |
| 2.450 | -30.3230 | -30.5323 | 0.2094 | 2.9876 | 0.0701 | 0.9760 | N/A | — | near_pass(0.02) | refine_vqe |  |
| 2.520 | -31.1502 | -31.3403 | 0.1901 | 3.1264 | 0.0608 | 0.9788 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 2.590 | -31.9776 | -32.1501 | 0.1725 | 3.2653 | 0.0528 | 0.9813 | N/A | — | near_pass(0.00) | refine_vqe |  |
| 2.660 | -32.8051 | -32.9615 | 0.1564 | 3.4043 | 0.0459 | 0.9835 | N/A | — | gap_masked(0.52) | refine_vqe |  |
| 2.720 | -33.5144 | -33.6582 | 0.1438 | 3.5234 | 0.0408 | 0.9852 | N/A | — | gap_masked(0.48) | refine_vqe |  |
| 2.790 | -34.3420 | -34.4723 | 0.1303 | 3.6626 | 0.0356 | 0.9869 | N/A | — | gap_masked(0.43) | refine_vqe |  |
| 2.860 | -35.1697 | -35.2877 | 0.1180 | 3.8017 | 0.0310 | 0.9884 | N/A | — | gap_masked(0.39) | refine_vqe |  |
| 2.930 | -35.9973 | -36.1043 | 0.1070 | 3.9409 | 0.0271 | 0.9898 | N/A | — | gap_masked(0.36) | refine_vqe |  |
| 3.000 | -36.8250 | -36.9220 | 0.0970 | 4.0802 | 0.0238 | 0.9909 | N/A | — | pass(0.48) | none |  |

## N = 16 (31 params)

**ΔE/gap: 0.8272 ± 1.2564 | P90=2.3153 | max=4.8697
|ΔE|/N: 3.81e-02
Fidelity: mean=0.8777 min=0.5761 (exact)
Distribution: [P25=0.088 | P50=0.259 | P75=0.890 | P90=2.315]
Regions: critical=2.4808 | ordered=0.1038**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=4.870 is 6× the mean — median may be more representative N=16

**Fidelity: mean F=0.8777, min F=0.5761** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -19.1551 | -20.0164 | 0.8612 | 0.1903 | 4.5250 | 0.6282 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.070 | -19.4006 | -20.8335 | 1.4330 | 0.2943 | 4.8697 | 0.5761 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -20.4058 | -21.6966 | 1.2907 | 0.4103 | 3.1456 | 0.6469 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -21.4087 | -22.5953 | 1.1866 | 0.5338 | 2.2230 | 0.6999 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.280 | -22.4142 | -23.5222 | 1.1080 | 0.6618 | 1.6742 | 0.7403 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.340 | -23.2771 | -24.3351 | 1.0580 | 0.7740 | 1.3670 | 0.7672 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.410 | -24.2911 | -25.3013 | 1.0101 | 0.9068 | 1.1139 | 0.7926 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.480 | -25.3174 | -26.2836 | 0.9662 | 1.0412 | 0.9280 | 0.8139 | N/A | — | severe_error(0.92) | increase_p |  |
| 1.550 | -26.3647 | -27.2795 | 0.9148 | 1.1766 | 0.7775 | 0.8335 | N/A | — | severe_error(0.77) | increase_p |  |
| 1.620 | -27.4329 | -28.2870 | 0.8541 | 1.3128 | 0.6506 | 0.8521 | N/A | — | severe_error(0.63) | increase_p |  |
| 1.690 | -28.5163 | -29.3045 | 0.7882 | 1.4496 | 0.5438 | 0.8695 | N/A | — | severe_error(0.52) | increase_p |  |
| 1.760 | -29.6090 | -30.3308 | 0.7217 | 1.5869 | 0.4548 | 0.8853 | N/A | — | moderate_error(0.43) | refine_vqe |  |
| 1.830 | -30.7031 | -31.3646 | 0.6616 | 1.7246 | 0.3836 | 0.8990 | N/A | — | moderate_error(0.35) | refine_vqe |  |
| 1.900 | -31.8002 | -32.4052 | 0.6050 | 1.8625 | 0.3248 | 0.9111 | N/A | — | moderate_error(0.29) | refine_vqe |  |
| 1.970 | -32.8985 | -33.4517 | 0.5532 | 2.0008 | 0.2765 | 0.9216 | N/A | — | moderate_error(0.24) | refine_vqe |  |
| 2.030 | -33.8426 | -34.3529 | 0.5103 | 2.1194 | 0.2408 | 0.9297 | N/A | — | moderate_error(0.20) | refine_vqe |  |
| 2.100 | -34.9443 | -35.4088 | 0.4645 | 2.2580 | 0.2057 | 0.9382 | N/A | — | moderate_error(0.16) | refine_vqe |  |
| 2.170 | -36.0463 | -36.4689 | 0.4227 | 2.3967 | 0.1764 | 0.9455 | N/A | — | moderate_error(0.13) | refine_vqe |  |
| 2.240 | -37.1487 | -37.5330 | 0.3843 | 2.5356 | 0.1516 | 0.9520 | N/A | — | moderate_error(0.11) | refine_vqe |  |
| 2.310 | -38.2511 | -38.6006 | 0.3495 | 2.6746 | 0.1307 | 0.9576 | N/A | — | moderate_error(0.08) | refine_vqe |  |
| 2.380 | -39.3536 | -39.6713 | 0.3177 | 2.8136 | 0.1129 | 0.9626 | N/A | — | moderate_error(0.07) | refine_vqe |  |
| 2.450 | -40.4565 | -40.7449 | 0.2885 | 2.9528 | 0.0977 | 0.9670 | N/A | — | near_pass(0.05) | refine_vqe |  |
| 2.520 | -41.5595 | -41.8212 | 0.2617 | 3.0920 | 0.0847 | 0.9709 | N/A | — | near_pass(0.04) | refine_vqe |  |
| 2.590 | -42.6627 | -42.9000 | 0.2373 | 3.2313 | 0.0734 | 0.9743 | N/A | — | near_pass(0.02) | refine_vqe |  |
| 2.660 | -43.7659 | -43.9809 | 0.2150 | 3.3706 | 0.0638 | 0.9773 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 2.720 | -44.7115 | -44.9091 | 0.1976 | 3.4901 | 0.0566 | 0.9796 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 2.790 | -45.8148 | -45.9937 | 0.1789 | 3.6295 | 0.0493 | 0.9820 | N/A | — | gap_masked(0.60) | refine_vqe |  |
| 2.860 | -46.9182 | -47.0801 | 0.1619 | 3.7689 | 0.0430 | 0.9841 | N/A | — | gap_masked(0.54) | refine_vqe |  |
| 2.930 | -48.0216 | -48.1681 | 0.1465 | 3.9084 | 0.0375 | 0.9859 | N/A | — | gap_masked(0.49) | refine_vqe |  |
| 3.000 | -49.1250 | -49.2577 | 0.1327 | 4.0480 | 0.0328 | 0.9876 | N/A | — | gap_masked(0.44) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 0.9870 ± 1.3608 | P90=3.0099 | max=5.6943
|ΔE|/N: 3.78e-02
Fidelity: mean≥0.6188 min≥0.0768 (variance lower bound, 30 pts)
Var(H): 4.3876
Distribution: [P25=0.114 | P50=0.337 | P75=1.189 | P90=3.010]
Regions: critical=2.8877 | ordered=0.1346**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=5.694 is 6× the mean — median may be more representative N=20

**Fidelity: mean F≥0.6188, min F≥0.0768** (variance lower bound — N>16, exact statevector infeasible)
> 30/30 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=4.3876.

**Infidelity decomposition:** 30 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=7.7947.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -23.9869 | -25.1078 | 1.1209 | 0.3142 | 3.5680 | N/A | 3.0874 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.070 | -25.1914 | -26.1175 | 0.9261 | 0.3142 | 2.9478 | N/A | 2.8565 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -25.3993 | -27.1882 | 1.7889 | 0.3142 | 5.6943 | N/A | 6.6539 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -26.7014 | -28.3055 | 1.6042 | 0.4404 | 3.6426 | N/A | 6.4328 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.280 | -27.9991 | -29.4594 | 1.4603 | 0.5793 | 2.5209 | N/A | 6.2873 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.340 | -29.1065 | -30.4721 | 1.3655 | 0.6984 | 1.9552 | N/A | 6.2344 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.410 | -30.3946 | -31.6763 | 1.2817 | 0.8375 | 1.5304 | N/A | 6.2509 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.480 | -31.6878 | -32.9012 | 1.2134 | 0.9767 | 1.2424 | N/A | 6.2981 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.550 | -32.9950 | -34.1435 | 1.1486 | 1.1159 | 1.0292 | N/A | 6.3135 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.620 | -34.3228 | -35.4006 | 1.0779 | 1.2552 | 0.8587 | N/A | 6.2434 | dirty_state | severe_error(0.85) | increase_p |  |
| 1.690 | -35.6733 | -36.6705 | 0.9972 | 1.3946 | 0.7150 | N/A | 6.0556 | dirty_state | severe_error(0.70) | increase_p |  |
| 1.760 | -37.0332 | -37.9515 | 0.9183 | 1.5340 | 0.5986 | N/A | 5.8280 | dirty_state | severe_error(0.58) | increase_p |  |
| 1.830 | -38.4001 | -39.2422 | 0.8420 | 1.6735 | 0.5032 | N/A | 5.5696 | dirty_state | severe_error(0.48) | increase_p |  |
| 1.900 | -39.7695 | -40.5413 | 0.7719 | 1.8130 | 0.4257 | N/A | 5.3106 | dirty_state | moderate_error(0.40) | refine_vqe |  |
| 1.970 | -41.1445 | -41.8481 | 0.7036 | 1.9525 | 0.3604 | N/A | 5.0237 | dirty_state | moderate_error(0.33) | refine_vqe |  |
| 2.030 | -42.3221 | -42.9735 | 0.6514 | 2.0722 | 0.3144 | N/A | 4.7965 | dirty_state | moderate_error(0.28) | refine_vqe |  |
| 2.100 | -43.6993 | -44.2921 | 0.5928 | 2.2117 | 0.2680 | ≥0.0768 | 4.5164 | dirty_state | moderate_error(0.23) | refine_vqe |  |
| 2.170 | -45.0768 | -45.6162 | 0.5394 | 2.3514 | 0.2294 | ≥0.2319 | 4.2467 | dirty_state | moderate_error(0.19) | refine_vqe |  |
| 2.240 | -46.4545 | -46.9453 | 0.4907 | 2.4910 | 0.1970 | ≥0.3574 | 3.9876 | dirty_state | moderate_error(0.15) | refine_vqe |  |
| 2.310 | -47.8329 | -48.2788 | 0.4458 | 2.6307 | 0.1695 | ≥0.4604 | 3.7344 | dirty_state | moderate_error(0.13) | refine_vqe |  |
| 2.380 | -49.2114 | -49.6163 | 0.4049 | 2.7704 | 0.1462 | ≥0.5449 | 3.4928 | dirty_state | moderate_error(0.10) | refine_vqe |  |
| 2.450 | -50.5900 | -50.9575 | 0.3676 | 2.9101 | 0.1263 | ≥0.6148 | 3.2616 | dirty_state | moderate_error(0.08) | refine_vqe |  |
| 2.520 | -51.9687 | -52.3021 | 0.3334 | 3.0498 | 0.1093 | ≥0.6731 | 3.0407 | dirty_state | moderate_error(0.06) | refine_vqe |  |
| 2.590 | -53.3475 | -53.6498 | 0.3022 | 3.1895 | 0.0948 | ≥0.7218 | 2.8304 | dirty_state | near_pass(0.05) | refine_vqe |  |
| 2.660 | -54.7265 | -55.0003 | 0.2738 | 3.3293 | 0.0822 | ≥0.7626 | 2.6308 | dirty_state | near_pass(0.03) | refine_vqe |  |
| 2.720 | -55.9085 | -56.1599 | 0.2514 | 3.4491 | 0.0729 | ≥0.7925 | 2.4679 | dirty_state | near_pass(0.02) | refine_vqe |  |
| 2.790 | -57.2877 | -57.5151 | 0.2274 | 3.5888 | 0.0634 | ≥0.8224 | 2.2874 | dirty_state | near_pass(0.01) | refine_vqe |  |
| 2.860 | -58.6669 | -58.8725 | 0.2056 | 3.7286 | 0.0551 | ≥0.8477 | 2.1178 | dirty_state | near_pass(0.01) | refine_vqe |  |
| 2.930 | -60.0462 | -60.2320 | 0.1858 | 3.8684 | 0.0480 | ≥0.8691 | 1.9594 | dirty_state | gap_masked(0.62) | refine_vqe |  |
| 3.000 | -61.4255 | -61.5934 | 0.1679 | 4.0082 | 0.0419 | ≥0.8872 | 1.8115 | dirty_state | gap_masked(0.56) | refine_vqe |  |

## N = 30 (59 params)

**ΔE/gap: 1.9467 ± 3.0056 | P90=7.0055 | max=12.3481
|ΔE|/N: 4.17e-02
Fidelity: mean≥0.5344 min≥0.0065 (variance lower bound, 30 pts)
Var(H): 7.1941
Distribution: [P25=0.174 | P50=0.513 | P75=1.970 | P90=7.006]
Regions: critical=6.0682 | ordered=0.2061**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=12.348 is 6× the mean — median may be more representative N=30

**Fidelity: mean F≥0.5344, min F≥0.0065** (variance lower bound — N>16, exact statevector infeasible)
> 30/30 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=7.1941.

**Infidelity decomposition:** 30 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=19.5660.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -36.0629 | -37.8381 | 1.7752 | 0.2094 | 8.4761 | N/A | 4.8013 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.070 | -37.8686 | -39.3276 | 1.4591 | 0.2094 | 6.9665 | N/A | 4.4396 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -37.3411 | -40.9173 | 3.5763 | 0.2896 | 12.3481 | N/A | 14.3282 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -39.4247 | -42.5812 | 3.1565 | 0.4291 | 7.3566 | N/A | 13.5022 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.280 | -41.5008 | -44.3024 | 2.8016 | 0.5686 | 4.9275 | N/A | 12.7328 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.340 | -43.2862 | -45.8145 | 2.5283 | 0.6882 | 3.6739 | N/A | 12.0540 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.410 | -45.3733 | -47.6140 | 2.2407 | 0.8278 | 2.7069 | N/A | 11.2522 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.480 | -47.4235 | -49.4454 | 2.0219 | 0.9674 | 2.0900 | N/A | 10.6858 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.550 | -49.5213 | -51.3036 | 1.7823 | 1.1071 | 1.6100 | N/A | 9.8421 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.620 | -51.6075 | -53.1847 | 1.5773 | 1.2468 | 1.2651 | N/A | 9.0841 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.690 | -53.6258 | -55.0855 | 1.4597 | 1.3865 | 1.0528 | N/A | 8.8097 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.760 | -55.6400 | -57.0033 | 1.3633 | 1.5262 | 0.8932 | N/A | 8.6131 | dirty_state | severe_error(0.89) | increase_p |  |
| 1.830 | -57.6789 | -58.9359 | 1.2571 | 1.6660 | 0.7546 | N/A | 8.2819 | dirty_state | severe_error(0.74) | increase_p |  |
| 1.900 | -59.7256 | -60.8817 | 1.1561 | 1.8058 | 0.6402 | N/A | 7.9262 | dirty_state | severe_error(0.62) | increase_p |  |
| 1.970 | -61.7748 | -62.8390 | 1.0643 | 1.9456 | 0.5470 | N/A | 7.5802 | dirty_state | severe_error(0.52) | increase_p |  |
| 2.030 | -63.5363 | -64.5250 | 0.9887 | 2.0654 | 0.4787 | N/A | 7.2642 | dirty_state | moderate_error(0.45) | refine_vqe |  |
| 2.100 | -65.5919 | -66.5005 | 0.9086 | 2.2052 | 0.4120 | N/A | 6.9144 | dirty_state | moderate_error(0.38) | refine_vqe |  |
| 2.170 | -67.6559 | -68.4845 | 0.8286 | 2.3451 | 0.3533 | N/A | 6.5164 | dirty_state | moderate_error(0.32) | refine_vqe |  |
| 2.240 | -69.7204 | -70.4760 | 0.7556 | 2.4849 | 0.3041 | ≥0.0065 | 6.1344 | dirty_state | moderate_error(0.27) | refine_vqe |  |
| 2.310 | -71.7888 | -72.4743 | 0.6855 | 2.6247 | 0.2612 | ≥0.1675 | 5.7356 | dirty_state | moderate_error(0.22) | refine_vqe |  |
| 2.380 | -73.8568 | -74.4789 | 0.6221 | 2.7646 | 0.2250 | ≥0.2988 | 5.3591 | dirty_state | moderate_error(0.18) | refine_vqe |  |
| 2.450 | -75.9253 | -76.4890 | 0.5637 | 2.9045 | 0.1941 | ≥0.4079 | 4.9948 | dirty_state | moderate_error(0.15) | refine_vqe |  |
| 2.520 | -77.9938 | -78.5044 | 0.5106 | 3.0444 | 0.1677 | ≥0.4984 | 4.6486 | dirty_state | moderate_error(0.12) | refine_vqe |  |
| 2.590 | -80.0624 | -80.5244 | 0.4620 | 3.1842 | 0.1451 | ≥0.5741 | 4.3183 | dirty_state | moderate_error(0.10) | refine_vqe |  |
| 2.660 | -82.1313 | -82.5488 | 0.4174 | 3.3241 | 0.1256 | ≥0.6378 | 4.0022 | dirty_state | moderate_error(0.08) | refine_vqe |  |
| 2.720 | -83.9048 | -84.2871 | 0.3823 | 3.4440 | 0.1110 | ≥0.6844 | 3.7440 | dirty_state | moderate_error(0.06) | refine_vqe |  |
| 2.790 | -85.9739 | -86.3186 | 0.3447 | 3.5839 | 0.0962 | ≥0.7308 | 3.4572 | dirty_state | near_pass(0.05) | refine_vqe |  |
| 2.860 | -88.0432 | -88.3535 | 0.3103 | 3.7238 | 0.0833 | ≥0.7703 | 3.1855 | dirty_state | near_pass(0.04) | refine_vqe |  |
| 2.930 | -90.1128 | -90.3916 | 0.2789 | 3.8637 | 0.0722 | ≥0.8039 | 2.9281 | dirty_state | near_pass(0.02) | refine_vqe |  |
| 3.000 | -92.1824 | -92.4327 | 0.2503 | 4.0037 | 0.0625 | ≥0.8324 | 2.6867 | dirty_state | near_pass(0.01) | refine_vqe |  |

## N = 40 (79 params)

**ΔE/gap: 3.1052 ± 4.9884 | P90=11.5231 | max=19.2544
|ΔE|/N: 4.50e-02
Fidelity: mean≥0.5066 min≥0.0696 (variance lower bound, 30 pts)
Var(H): 10.3799
Distribution: [P25=0.231 | P50=0.669 | P75=3.011 | P90=11.523]
Regions: critical=9.9265 | ordered=0.2704**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=19.254 is 6× the mean — median may be more representative N=40

**Fidelity: mean F≥0.5066, min F≥0.0696** (variance lower bound — N>16, exact statevector infeasible)
> 30/30 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=10.3799.

**Infidelity decomposition:** 30 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=37.3189.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -48.1366 | -50.5694 | 2.4328 | 0.1571 | 15.4876 | N/A | 6.5218 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.070 | -50.5438 | -52.5378 | 1.9941 | 0.1571 | 12.6946 | N/A | 6.0290 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -49.1510 | -54.6465 | 5.4954 | 0.2854 | 19.2544 | N/A | 23.1861 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -52.0137 | -56.8569 | 4.8431 | 0.4251 | 11.3929 | N/A | 21.7570 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.280 | -54.8638 | -59.1454 | 4.2815 | 0.5648 | 7.5804 | N/A | 20.3615 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.340 | -57.2966 | -61.1569 | 3.8603 | 0.6846 | 5.6387 | N/A | 19.2001 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.410 | -60.1094 | -63.5517 | 3.4422 | 0.8244 | 4.1756 | N/A | 17.9801 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.480 | -62.9159 | -65.9896 | 3.0737 | 0.9642 | 3.1879 | N/A | 16.8081 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.550 | -65.7249 | -68.4638 | 2.7388 | 1.1040 | 2.4809 | N/A | 15.6290 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.620 | -68.5668 | -70.9689 | 2.4021 | 1.2438 | 1.9312 | N/A | 14.2352 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.690 | -71.3847 | -73.5005 | 2.1158 | 1.3837 | 1.5291 | N/A | 13.0132 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.760 | -74.1977 | -76.0550 | 1.8573 | 1.5235 | 1.2191 | N/A | 11.8316 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.830 | -77.0012 | -78.6297 | 1.6285 | 1.6634 | 0.9790 | N/A | 10.7288 | dirty_state | severe_error(0.98) | increase_p |  |
| 1.900 | -79.7164 | -81.2220 | 1.5056 | 1.8032 | 0.8350 | N/A | 10.3141 | dirty_state | severe_error(0.83) | increase_p |  |
| 1.970 | -82.4445 | -83.8300 | 1.3855 | 1.9431 | 0.7130 | N/A | 9.8510 | dirty_state | severe_error(0.70) | increase_p |  |
| 2.030 | -84.7877 | -86.0764 | 1.2887 | 2.0630 | 0.6247 | N/A | 9.4502 | dirty_state | severe_error(0.60) | increase_p |  |
| 2.100 | -87.5272 | -88.7089 | 1.1818 | 2.2029 | 0.5365 | N/A | 8.9716 | dirty_state | severe_error(0.51) | increase_p |  |
| 2.170 | -90.2692 | -91.3528 | 1.0836 | 2.3428 | 0.4625 | N/A | 8.5044 | dirty_state | moderate_error(0.43) | refine_vqe |  |
| 2.240 | -93.0137 | -94.0067 | 0.9930 | 2.4828 | 0.4000 | N/A | 8.0468 | dirty_state | moderate_error(0.37) | refine_vqe |  |
| 2.310 | -95.7600 | -96.6699 | 0.9099 | 2.6227 | 0.3469 | N/A | 7.6046 | dirty_state | moderate_error(0.31) | refine_vqe |  |
| 2.380 | -98.5162 | -99.3414 | 0.8253 | 2.7626 | 0.2987 | ≥0.0696 | 7.1008 | dirty_state | moderate_error(0.26) | refine_vqe |  |
| 2.450 | -101.2727 | -102.0206 | 0.7478 | 2.9025 | 0.2576 | ≥0.2145 | 6.6174 | dirty_state | moderate_error(0.22) | refine_vqe |  |
| 2.520 | -104.0310 | -104.7066 | 0.6757 | 3.0424 | 0.2221 | ≥0.3365 | 6.1418 | dirty_state | moderate_error(0.18) | refine_vqe |  |
| 2.590 | -106.7897 | -107.3990 | 0.6093 | 3.1824 | 0.1915 | ≥0.4387 | 5.6845 | dirty_state | moderate_error(0.15) | refine_vqe |  |
| 2.660 | -109.5492 | -110.0972 | 0.5480 | 3.3223 | 0.1650 | ≥0.5251 | 5.2419 | dirty_state | moderate_error(0.12) | refine_vqe |  |
| 2.720 | -111.9146 | -112.4143 | 0.4996 | 3.4423 | 0.1451 | ≥0.5883 | 4.8788 | dirty_state | moderate_error(0.10) | refine_vqe |  |
| 2.790 | -114.6743 | -115.1221 | 0.4478 | 3.5822 | 0.1250 | ≥0.6512 | 4.4759 | dirty_state | moderate_error(0.08) | refine_vqe |  |
| 2.860 | -117.4341 | -117.8346 | 0.4005 | 3.7222 | 0.1076 | ≥0.7044 | 4.0949 | dirty_state | moderate_error(0.06) | refine_vqe |  |
| 2.930 | -120.1938 | -120.5513 | 0.3575 | 3.8621 | 0.0926 | ≥0.7495 | 3.7361 | dirty_state | near_pass(0.04) | refine_vqe |  |
| 3.000 | -122.9535 | -123.2720 | 0.3185 | 4.0021 | 0.0796 | ≥0.7877 | 3.3998 | dirty_state | near_pass(0.03) | refine_vqe |  |
