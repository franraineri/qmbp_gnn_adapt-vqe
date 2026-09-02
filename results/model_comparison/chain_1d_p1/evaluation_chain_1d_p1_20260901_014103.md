# Model Evaluation: chain_1d

**Date**: 2026-09-01 01:41 UTC
**Model**: unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20+26+60_p1.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 3.0] (20 pts)
**Target N**: [8, 10, 12, 16, 20]

---

## N = 8 (15 params)

**ΔE/gap: 0.4428 ± 0.7277 | P90=1.1807 | max=3.0144
|ΔE|/N: 4.67e-02
Fidelity: mean=0.9254 min=0.6823 (exact)
Distribution: [P25=0.045 | P50=0.123 | P75=0.424 | P90=1.181]
Regions: critical=1.4275 | ordered=0.0505**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=3.014 is 7× the mean — median may be more representative N=8

**Fidelity: mean F=0.9254, min F=0.6823** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -8.7254 | -9.8380 | 1.1125 | 0.3691 | 3.0144 | 0.6823 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -9.5937 | -10.5203 | 0.9266 | 0.5388 | 1.7199 | 0.7611 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.210 | -10.3830 | -11.1749 | 0.7919 | 0.7065 | 1.1208 | 0.8132 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.320 | -11.2512 | -11.9227 | 0.6715 | 0.9006 | 0.7455 | 0.8555 | N/A | — | severe_error(0.73) | increase_p |  |
| 1.420 | -12.0406 | -12.6218 | 0.5812 | 1.0829 | 0.5367 | 0.8842 | N/A | — | severe_error(0.51) | increase_p |  |
| 1.530 | -12.9088 | -13.4070 | 0.4982 | 1.2878 | 0.3869 | 0.9083 | N/A | — | moderate_error(0.35) | refine_vqe |  |
| 1.630 | -13.6981 | -14.1327 | 0.4346 | 1.4769 | 0.2943 | 0.9252 | N/A | — | moderate_error(0.26) | refine_vqe |  |
| 1.740 | -14.5664 | -14.9413 | 0.3750 | 1.6871 | 0.2223 | 0.9397 | N/A | — | moderate_error(0.18) | refine_vqe |  |
| 1.840 | -15.3557 | -15.6842 | 0.3285 | 1.8797 | 0.1748 | 0.9502 | N/A | — | moderate_error(0.13) | refine_vqe |  |
| 1.950 | -16.2239 | -16.5084 | 0.2845 | 2.0930 | 0.1359 | 0.9594 | N/A | — | moderate_error(0.09) | refine_vqe |  |
| 2.050 | -17.0133 | -17.2631 | 0.2498 | 2.2877 | 0.1092 | 0.9661 | N/A | — | moderate_error(0.06) | refine_vqe |  |
| 2.160 | -17.8815 | -18.0982 | 0.2167 | 2.5028 | 0.0866 | 0.9722 | N/A | — | near_pass(0.04) | refine_vqe |  |
| 2.260 | -18.6708 | -18.8613 | 0.1905 | 2.6988 | 0.0706 | 0.9766 | N/A | — | near_pass(0.02) | refine_vqe |  |
| 2.370 | -19.5391 | -19.7044 | 0.1653 | 2.9150 | 0.0567 | 0.9807 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 2.470 | -20.3284 | -20.4738 | 0.1454 | 3.1120 | 0.0467 | 0.9837 | N/A | — | gap_masked(0.48) | refine_vqe |  |
| 2.580 | -21.1967 | -21.3229 | 0.1263 | 3.3290 | 0.0379 | 0.9865 | N/A | — | gap_masked(0.42) | refine_vqe |  |
| 2.680 | -21.9860 | -22.0971 | 0.1112 | 3.5266 | 0.0315 | 0.9886 | N/A | — | gap_masked(0.37) | refine_vqe |  |
| 2.790 | -22.8542 | -22.9509 | 0.0967 | 3.7442 | 0.0258 | 0.9904 | N/A | — | pass(0.52) | none |  |
| 2.890 | -23.6435 | -23.7288 | 0.0853 | 3.9422 | 0.0216 | 0.9919 | N/A | — | pass(0.43) | none |  |
| 3.000 | -24.5118 | -24.5863 | 0.0745 | 4.1602 | 0.0179 | 0.9932 | N/A | — | pass(0.36) | none |  |

## N = 10 (19 params)

**ΔE/gap: 0.6558 ± 1.1495 | P90=1.7088 | max=4.8476
|ΔE|/N: 4.83e-02
Fidelity: mean=0.9042 min=0.5939 (exact)
Distribution: [P25=0.058 | P50=0.163 | P75=0.579 | P90=1.709]
Regions: critical=2.1635 | ordered=0.0665**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=4.848 is 7× the mean — median may be more representative N=10

**Fidelity: mean F=0.9042, min F=0.5939** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -10.9324 | -12.3815 | 1.4491 | 0.2989 | 4.8476 | 0.5939 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -12.0173 | -13.2204 | 1.2031 | 0.4674 | 2.5740 | 0.6940 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -13.0036 | -14.0299 | 1.0263 | 0.6364 | 1.6127 | 0.7605 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.320 | -14.0885 | -14.9576 | 0.8691 | 0.8327 | 1.0438 | 0.8144 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.420 | -15.0748 | -15.8265 | 0.7517 | 1.0170 | 0.7391 | 0.8511 | N/A | — | severe_error(0.73) | increase_p |  |
| 1.530 | -16.1597 | -16.8037 | 0.6440 | 1.2240 | 0.5262 | 0.8819 | N/A | — | severe_error(0.50) | increase_p |  |
| 1.630 | -17.1459 | -17.7075 | 0.5615 | 1.4149 | 0.3969 | 0.9035 | N/A | — | moderate_error(0.37) | refine_vqe |  |
| 1.740 | -18.2308 | -18.7152 | 0.4844 | 1.6269 | 0.2977 | 0.9222 | N/A | — | moderate_error(0.26) | refine_vqe |  |
| 1.840 | -19.2171 | -19.6414 | 0.4243 | 1.8209 | 0.2330 | 0.9357 | N/A | — | moderate_error(0.19) | refine_vqe |  |
| 1.950 | -20.3020 | -20.6693 | 0.3673 | 2.0355 | 0.1805 | 0.9475 | N/A | — | moderate_error(0.14) | refine_vqe |  |
| 2.050 | -21.2883 | -21.6108 | 0.3226 | 2.2313 | 0.1446 | 0.9562 | N/A | — | moderate_error(0.10) | refine_vqe |  |
| 2.160 | -22.3731 | -22.6530 | 0.2798 | 2.4474 | 0.1143 | 0.9640 | N/A | — | moderate_error(0.07) | refine_vqe |  |
| 2.260 | -23.3594 | -23.6054 | 0.2460 | 2.6443 | 0.0930 | 0.9698 | N/A | — | near_pass(0.05) | refine_vqe |  |
| 2.370 | -24.4443 | -24.6579 | 0.2135 | 2.8614 | 0.0746 | 0.9750 | N/A | — | near_pass(0.03) | refine_vqe |  |
| 2.470 | -25.4306 | -25.6184 | 0.1878 | 3.0591 | 0.0614 | 0.9789 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 2.580 | -26.5155 | -26.6786 | 0.1632 | 3.2768 | 0.0498 | 0.9825 | N/A | — | gap_masked(0.54) | refine_vqe |  |
| 2.680 | -27.5017 | -27.6454 | 0.1436 | 3.4749 | 0.0413 | 0.9852 | N/A | — | gap_masked(0.48) | refine_vqe |  |
| 2.790 | -28.5866 | -28.7116 | 0.1250 | 3.6931 | 0.0338 | 0.9876 | N/A | — | gap_masked(0.42) | refine_vqe |  |
| 2.890 | -29.5729 | -29.6832 | 0.1103 | 3.8916 | 0.0283 | 0.9895 | N/A | — | gap_masked(0.37) | refine_vqe |  |
| 3.000 | -30.6578 | -30.7541 | 0.0963 | 4.1101 | 0.0234 | 0.9911 | N/A | — | pass(0.47) | none |  |

## N = 12 (23 params)

**ΔE/gap: 0.8980 ± 1.6624 | P90=2.2759 | max=7.1130
|ΔE|/N: 4.95e-02
Fidelity: mean=0.8842 min=0.5148 (exact)
Distribution: [P25=0.072 | P50=0.203 | P75=0.737 | P90=2.276]
Regions: critical=3.0154 | ordered=0.0825**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=7.113 is 8× the mean — median may be more representative N=12

**Fidelity: mean F=0.8842, min F=0.5148** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -13.1395 | -14.9260 | 1.7865 | 0.2512 | 7.1130 | 0.5148 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -14.4410 | -15.9208 | 1.4798 | 0.4194 | 3.5284 | 0.6322 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -15.6242 | -16.8850 | 1.2608 | 0.5901 | 2.1367 | 0.7109 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -16.9258 | -17.9926 | 1.0668 | 0.7886 | 1.3528 | 0.7752 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.420 | -18.1090 | -19.0312 | 0.9223 | 0.9749 | 0.9460 | 0.8192 | N/A | — | severe_error(0.94) | increase_p |  |
| 1.530 | -19.4105 | -20.2003 | 0.7898 | 1.1839 | 0.6672 | 0.8562 | N/A | — | severe_error(0.65) | increase_p |  |
| 1.630 | -20.5937 | -21.2822 | 0.6885 | 1.3762 | 0.5003 | 0.8824 | N/A | — | severe_error(0.47) | increase_p |  |
| 1.740 | -21.8953 | -22.4890 | 0.5938 | 1.5895 | 0.3736 | 0.9050 | N/A | — | moderate_error(0.34) | refine_vqe |  |
| 1.840 | -23.0785 | -23.5986 | 0.5201 | 1.7847 | 0.2914 | 0.9214 | N/A | — | moderate_error(0.25) | refine_vqe |  |
| 1.950 | -24.3800 | -24.8303 | 0.4502 | 2.0003 | 0.2251 | 0.9358 | N/A | — | moderate_error(0.18) | refine_vqe |  |
| 2.050 | -25.5632 | -25.9586 | 0.3954 | 2.1969 | 0.1800 | 0.9464 | N/A | — | moderate_error(0.14) | refine_vqe |  |
| 2.160 | -26.8648 | -27.2077 | 0.3429 | 2.4138 | 0.1421 | 0.9559 | N/A | — | moderate_error(0.10) | refine_vqe |  |
| 2.260 | -28.0480 | -28.3495 | 0.3015 | 2.6114 | 0.1154 | 0.9630 | N/A | — | moderate_error(0.07) | refine_vqe |  |
| 2.370 | -29.3495 | -29.6113 | 0.2617 | 2.8291 | 0.0925 | 0.9694 | N/A | — | near_pass(0.04) | refine_vqe |  |
| 2.470 | -30.5328 | -30.7630 | 0.2302 | 3.0272 | 0.0761 | 0.9741 | N/A | — | near_pass(0.03) | refine_vqe |  |
| 2.580 | -31.8343 | -32.0343 | 0.2000 | 3.2454 | 0.0616 | 0.9785 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 2.680 | -33.0175 | -33.1936 | 0.1761 | 3.4440 | 0.0511 | 0.9818 | N/A | — | near_pass(0.00) | refine_vqe |  |
| 2.790 | -34.3191 | -34.4723 | 0.1532 | 3.6626 | 0.0418 | 0.9848 | N/A | — | gap_masked(0.51) | refine_vqe |  |
| 2.890 | -35.5023 | -35.6375 | 0.1352 | 3.8614 | 0.0350 | 0.9871 | N/A | — | gap_masked(0.45) | refine_vqe |  |
| 3.000 | -36.8038 | -36.9220 | 0.1182 | 4.0802 | 0.0290 | 0.9891 | N/A | — | gap_masked(0.39) | refine_vqe |  |

## N = 16 (31 params)

**ΔE/gap: 1.4618 ± 2.9592 | P90=3.4824 | max=12.9403
|ΔE|/N: 5.09e-02
Fidelity: mean=0.8474 min=0.3836 (exact)
Distribution: [P25=0.100 | P50=0.282 | P75=1.054 | P90=3.482]
Regions: critical=5.0376 | ordered=0.1144**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=12.940 is 9× the mean — median may be more representative N=16

**Fidelity: mean F=0.8474, min F=0.3836** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -17.5535 | -20.0164 | 2.4629 | 0.1903 | 12.9403 | 0.3836 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -19.2883 | -21.3218 | 2.0335 | 0.3595 | 5.6571 | 0.5241 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -20.8654 | -22.5953 | 1.7298 | 0.5338 | 3.2408 | 0.6212 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -22.6003 | -24.0625 | 1.4622 | 0.7364 | 1.9856 | 0.7024 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.420 | -24.1774 | -25.4407 | 1.2633 | 0.9259 | 1.3643 | 0.7589 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.530 | -25.9122 | -26.9937 | 1.0814 | 1.1378 | 0.9505 | 0.8071 | N/A | — | severe_error(0.95) | increase_p |  |
| 1.630 | -27.4893 | -28.4318 | 0.9424 | 1.3323 | 0.7074 | 0.8416 | N/A | — | severe_error(0.69) | increase_p |  |
| 1.740 | -29.2242 | -30.0368 | 0.8126 | 1.5476 | 0.5251 | 0.8716 | N/A | — | severe_error(0.50) | increase_p |  |
| 1.840 | -30.8013 | -31.5129 | 0.7116 | 1.7442 | 0.4080 | 0.8934 | N/A | — | moderate_error(0.38) | refine_vqe |  |
| 1.950 | -32.5361 | -33.1521 | 0.6160 | 1.9612 | 0.3141 | 0.9128 | N/A | — | moderate_error(0.28) | refine_vqe |  |
| 2.050 | -34.1132 | -34.6541 | 0.5409 | 2.1590 | 0.2505 | 0.9271 | N/A | — | moderate_error(0.21) | refine_vqe |  |
| 2.160 | -35.8481 | -36.3172 | 0.4692 | 2.3769 | 0.1974 | 0.9399 | N/A | — | moderate_error(0.16) | refine_vqe |  |
| 2.260 | -37.4252 | -37.8377 | 0.4125 | 2.5753 | 0.1602 | 0.9495 | N/A | — | moderate_error(0.12) | refine_vqe |  |
| 2.370 | -39.1600 | -39.5181 | 0.3581 | 2.7937 | 0.1282 | 0.9582 | N/A | — | moderate_error(0.08) | refine_vqe |  |
| 2.470 | -40.7371 | -41.0522 | 0.3151 | 2.9925 | 0.1053 | 0.9647 | N/A | — | moderate_error(0.06) | refine_vqe |  |
| 2.580 | -42.4720 | -42.7457 | 0.2738 | 3.2114 | 0.0852 | 0.9706 | N/A | — | near_pass(0.04) | refine_vqe |  |
| 2.680 | -44.0491 | -44.2901 | 0.2411 | 3.4104 | 0.0707 | 0.9751 | N/A | — | near_pass(0.02) | refine_vqe |  |
| 2.790 | -45.7839 | -45.9937 | 0.2098 | 3.6295 | 0.0578 | 0.9792 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 2.890 | -47.3610 | -47.5462 | 0.1852 | 3.8287 | 0.0484 | 0.9823 | N/A | — | gap_masked(0.62) | refine_vqe |  |
| 3.000 | -49.0958 | -49.2577 | 0.1619 | 4.0480 | 0.0400 | 0.9851 | N/A | — | gap_masked(0.54) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 1.6619 ± 2.7624 | P90=5.3172 | max=9.9958
|ΔE|/N: 5.17e-02
Fidelity: mean≥0.6007 min≥0.1343 (variance lower bound, 9 pts)
Var(H): 3.3702
Distribution: [P25=0.129 | P50=0.367 | P75=1.425 | P90=5.317]
Regions: critical=5.5831 | ordered=0.1476**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=9.996 is 6× the mean — median may be more representative N=20

**Fidelity: mean F≥0.6007, min F≥0.1343** (variance lower bound — N>16, exact statevector infeasible)
> 9/20 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=3.3702.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -21.9675 | -25.1078 | 3.1403 | 0.3142 | 9.9958 | N/A | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -24.1356 | -26.7229 | 2.5872 | 0.3142 | 8.2355 | N/A | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -26.1067 | -28.3055 | 2.1989 | 0.4404 | 4.9930 | N/A | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -28.2748 | -30.1324 | 1.8576 | 0.6587 | 2.8201 | N/A | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.420 | -30.2458 | -31.8501 | 1.6044 | 0.8574 | 1.8712 | N/A | N/A | — | severe_error(1.00) | increase_p |  |
| 1.530 | -32.4139 | -33.7870 | 1.3730 | 1.0761 | 1.2759 | N/A | N/A | — | severe_error(1.00) | increase_p |  |
| 1.630 | -34.3849 | -35.5813 | 1.1964 | 1.2751 | 0.9382 | N/A | N/A | — | severe_error(0.93) | increase_p |  |
| 1.740 | -36.5531 | -37.5845 | 1.0314 | 1.4942 | 0.6903 | N/A | N/A | — | severe_error(0.67) | increase_p |  |
| 1.840 | -38.5241 | -39.4273 | 0.9032 | 1.6934 | 0.5334 | N/A | N/A | — | severe_error(0.51) | increase_p |  |
| 1.950 | -40.6922 | -41.4740 | 0.7818 | 1.9127 | 0.4088 | N/A | N/A | — | moderate_error(0.38) | refine_vqe |  |
| 2.050 | -42.6632 | -43.3497 | 0.6865 | 2.1120 | 0.3250 | N/A | N/A | — | moderate_error(0.29) | refine_vqe |  |
| 2.160 | -44.8313 | -45.4268 | 0.5954 | 2.3314 | 0.2554 | ≥0.1343 | 4.7057 | — | moderate_error(0.22) | refine_vqe |  |
| 2.260 | -46.8023 | -47.3258 | 0.5235 | 2.5309 | 0.2068 | ≥0.3241 | 4.3294 | — | moderate_error(0.17) | refine_vqe |  |
| 2.370 | -48.9705 | -49.4250 | 0.4545 | 2.7504 | 0.1653 | ≥0.4789 | 3.9420 | — | moderate_error(0.12) | refine_vqe |  |
| 2.470 | -50.9415 | -51.3414 | 0.3999 | 2.9500 | 0.1356 | ≥0.5847 | 3.6139 | — | moderate_error(0.09) | refine_vqe |  |
| 2.580 | -53.1096 | -53.4571 | 0.3475 | 3.1696 | 0.1096 | ≥0.6736 | 3.2795 | — | moderate_error(0.06) | refine_vqe |  |
| 2.680 | -55.0806 | -55.3866 | 0.3060 | 3.3692 | 0.0908 | ≥0.7358 | 2.9995 | — | near_pass(0.04) | refine_vqe |  |
| 2.790 | -57.2487 | -57.5151 | 0.2664 | 3.5888 | 0.0742 | ≥0.7890 | 2.7180 | — | near_pass(0.03) | refine_vqe |  |
| 2.890 | -59.2198 | -59.4549 | 0.2351 | 3.7885 | 0.0621 | ≥0.8268 | 2.4862 | — | near_pass(0.01) | refine_vqe |  |
| 3.000 | -61.3879 | -61.5934 | 0.2055 | 4.0082 | 0.0513 | ≥0.8595 | 2.2577 | — | near_pass(0.00) | refine_vqe |  |
