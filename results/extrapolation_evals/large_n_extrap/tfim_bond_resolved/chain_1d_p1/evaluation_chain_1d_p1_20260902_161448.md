# Model Evaluation: chain_1d

**Date**: 2026-09-02 16:14 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__chain_1d_p1_signinv_v2.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 3.0] (30 pts)
**Target N**: [8, 10, 12, 16, 20, 30, 40]

---

## N = 8 (15 params)

**ΔE/gap: 0.3975 ± 0.6586 | P90=1.1100 | max=2.9235
|ΔE|/N: 4.20e-02
Fidelity: mean=0.9316 min=0.6890 (exact)
Distribution: [P25=0.034 | P50=0.106 | P75=0.398 | P90=1.110]
Regions: critical=1.2335 | ordered=0.0407**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=2.923 is 7× the mean — median may be more representative N=8

**Fidelity: mean F=0.9316, min F=0.6890** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -8.7590 | -9.8380 | 1.0790 | 0.3691 | 2.9235 | 0.6890 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.070 | -9.3117 | -10.2670 | 0.9553 | 0.4749 | 2.0116 | 0.7421 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -9.8643 | -10.7137 | 0.8494 | 0.5880 | 1.4446 | 0.7850 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.210 | -10.4169 | -11.1749 | 0.7580 | 0.7065 | 1.0728 | 0.8197 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.280 | -10.9700 | -11.6479 | 0.6779 | 0.8291 | 0.8176 | 0.8480 | N/A | — | severe_error(0.81) | increase_p |  |
| 1.340 | -11.4439 | -12.0612 | 0.6173 | 0.9367 | 0.6590 | 0.8681 | N/A | — | severe_error(0.64) | increase_p |  |
| 1.410 | -11.9967 | -12.5512 | 0.5545 | 1.0645 | 0.5209 | 0.8878 | N/A | — | severe_error(0.50) | increase_p |  |
| 1.480 | -12.5494 | -13.0482 | 0.4989 | 1.1942 | 0.4177 | 0.9041 | N/A | — | moderate_error(0.39) | refine_vqe |  |
| 1.550 | -13.1020 | -13.5514 | 0.4494 | 1.3254 | 0.3390 | 0.9177 | N/A | — | moderate_error(0.30) | refine_vqe |  |
| 1.620 | -13.6546 | -14.0597 | 0.4051 | 1.4579 | 0.2779 | 0.9292 | N/A | — | moderate_error(0.24) | refine_vqe |  |
| 1.690 | -14.2068 | -14.5726 | 0.3657 | 1.5913 | 0.2298 | 0.9388 | N/A | — | moderate_error(0.19) | refine_vqe |  |
| 1.760 | -14.7591 | -15.0894 | 0.3302 | 1.7255 | 0.1914 | 0.9471 | N/A | — | moderate_error(0.15) | refine_vqe |  |
| 1.830 | -15.3114 | -15.6096 | 0.2982 | 1.8604 | 0.1603 | 0.9541 | N/A | — | moderate_error(0.12) | refine_vqe |  |
| 1.900 | -15.8637 | -16.1329 | 0.2692 | 1.9959 | 0.1349 | 0.9601 | N/A | — | moderate_error(0.09) | refine_vqe |  |
| 1.970 | -16.4159 | -16.6589 | 0.2430 | 2.1319 | 0.1140 | 0.9653 | N/A | — | moderate_error(0.07) | refine_vqe |  |
| 2.030 | -16.8893 | -17.1118 | 0.2224 | 2.2487 | 0.0989 | 0.9692 | N/A | — | near_pass(0.05) | refine_vqe |  |
| 2.100 | -17.4416 | -17.6421 | 0.2005 | 2.3854 | 0.0841 | 0.9732 | N/A | — | near_pass(0.04) | refine_vqe |  |
| 2.170 | -17.9938 | -18.1743 | 0.1805 | 2.5223 | 0.0716 | 0.9767 | N/A | — | near_pass(0.02) | refine_vqe |  |
| 2.240 | -18.5461 | -18.7084 | 0.1623 | 2.6596 | 0.0610 | 0.9797 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 2.310 | -19.0983 | -19.2441 | 0.1458 | 2.7970 | 0.0521 | 0.9823 | N/A | — | near_pass(0.00) | refine_vqe |  |
| 2.380 | -19.6505 | -19.7812 | 0.1307 | 2.9347 | 0.0445 | 0.9846 | N/A | — | gap_masked(0.44) | refine_vqe |  |
| 2.450 | -20.2027 | -20.3197 | 0.1170 | 3.0726 | 0.0381 | 0.9866 | N/A | — | gap_masked(0.39) | refine_vqe |  |
| 2.520 | -20.7550 | -20.8594 | 0.1045 | 3.2106 | 0.0325 | 0.9883 | N/A | — | gap_masked(0.35) | refine_vqe |  |
| 2.590 | -21.3072 | -21.4003 | 0.0931 | 3.3488 | 0.0278 | 0.9899 | N/A | — | pass(0.56) | none |  |
| 2.660 | -21.8594 | -21.9421 | 0.0828 | 3.4871 | 0.0237 | 0.9912 | N/A | — | pass(0.47) | none |  |
| 2.720 | -22.3327 | -22.4073 | 0.0747 | 3.6057 | 0.0207 | 0.9923 | N/A | — | pass(0.41) | none |  |
| 2.790 | -22.8849 | -22.9509 | 0.0660 | 3.7442 | 0.0176 | 0.9933 | N/A | — | pass(0.35) | none |  |
| 2.860 | -23.4371 | -23.4953 | 0.0582 | 3.8828 | 0.0150 | 0.9942 | N/A | — | pass(0.30) | none |  |
| 2.930 | -23.9893 | -24.0404 | 0.0512 | 4.0215 | 0.0127 | 0.9950 | N/A | — | pass(0.25) | none |  |
| 3.000 | -24.5415 | -24.5863 | 0.0448 | 4.1602 | 0.0108 | 0.9958 | N/A | — | pass(0.22) | none |  |

## N = 10 (19 params)

**ΔE/gap: 0.5863 ± 1.0323 | P90=1.6042 | max=4.7058
|ΔE|/N: 4.36e-02
Fidelity: mean=0.9119 min=0.6014 (exact)
Distribution: [P25=0.045 | P50=0.141 | P75=0.544 | P90=1.604]
Regions: critical=1.8540 | ordered=0.0539**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=4.706 is 8× the mean — median may be more representative N=10

**Fidelity: mean F=0.9119, min F=0.6014** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -10.9748 | -12.3815 | 1.4066 | 0.2989 | 4.7058 | 0.6014 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.070 | -11.6654 | -12.9082 | 1.2428 | 0.4035 | 3.0797 | 0.6690 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -12.3560 | -13.4593 | 1.1033 | 0.5169 | 2.1345 | 0.7238 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -13.0465 | -14.0299 | 0.9834 | 0.6364 | 1.5453 | 0.7682 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.280 | -13.7377 | -14.6165 | 0.8787 | 0.7603 | 1.1557 | 0.8044 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.340 | -14.3299 | -15.1297 | 0.7998 | 0.8692 | 0.9202 | 0.8301 | N/A | — | severe_error(0.92) | increase_p |  |
| 1.410 | -15.0207 | -15.7387 | 0.7180 | 0.9984 | 0.7191 | 0.8553 | N/A | — | severe_error(0.70) | increase_p |  |
| 1.480 | -15.7114 | -16.3571 | 0.6457 | 1.1295 | 0.5717 | 0.8762 | N/A | — | severe_error(0.55) | increase_p |  |
| 1.550 | -16.4020 | -16.9834 | 0.5813 | 1.2620 | 0.4606 | 0.8937 | N/A | — | moderate_error(0.43) | refine_vqe |  |
| 1.620 | -17.0923 | -17.6165 | 0.5242 | 1.3957 | 0.3756 | 0.9085 | N/A | — | moderate_error(0.34) | refine_vqe |  |
| 1.690 | -17.7824 | -18.2556 | 0.4732 | 1.5303 | 0.3092 | 0.9209 | N/A | — | moderate_error(0.27) | refine_vqe |  |
| 1.760 | -18.4725 | -18.8997 | 0.4272 | 1.6656 | 0.2565 | 0.9315 | N/A | — | moderate_error(0.22) | refine_vqe |  |
| 1.830 | -19.1625 | -19.5484 | 0.3858 | 1.8014 | 0.2142 | 0.9406 | N/A | — | moderate_error(0.17) | refine_vqe |  |
| 1.900 | -19.8526 | -20.2010 | 0.3484 | 1.9378 | 0.1798 | 0.9483 | N/A | — | moderate_error(0.14) | refine_vqe |  |
| 1.970 | -20.5427 | -20.8571 | 0.3144 | 2.0746 | 0.1516 | 0.9551 | N/A | — | moderate_error(0.11) | refine_vqe |  |
| 2.030 | -21.1342 | -21.4220 | 0.2879 | 2.1921 | 0.1313 | 0.9601 | N/A | — | moderate_error(0.09) | refine_vqe |  |
| 2.100 | -21.8242 | -22.0837 | 0.2595 | 2.3295 | 0.1114 | 0.9652 | N/A | — | moderate_error(0.06) | refine_vqe |  |
| 2.170 | -22.5143 | -22.7480 | 0.2337 | 2.4671 | 0.0947 | 0.9697 | N/A | — | near_pass(0.05) | refine_vqe |  |
| 2.240 | -23.2043 | -23.4146 | 0.2103 | 2.6049 | 0.0807 | 0.9736 | N/A | — | near_pass(0.03) | refine_vqe |  |
| 2.310 | -23.8943 | -24.0832 | 0.1889 | 2.7430 | 0.0689 | 0.9770 | N/A | — | near_pass(0.02) | refine_vqe |  |
| 2.380 | -24.5843 | -24.7538 | 0.1694 | 2.8812 | 0.0588 | 0.9800 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 2.450 | -25.2743 | -25.4260 | 0.1517 | 3.0195 | 0.0502 | 0.9826 | N/A | — | near_pass(0.00) | refine_vqe |  |
| 2.520 | -25.9644 | -26.0999 | 0.1355 | 3.1580 | 0.0429 | 0.9848 | N/A | — | gap_masked(0.45) | refine_vqe |  |
| 2.590 | -26.6544 | -26.7752 | 0.1208 | 3.2966 | 0.0367 | 0.9868 | N/A | — | gap_masked(0.40) | refine_vqe |  |
| 2.660 | -27.3444 | -27.4518 | 0.1075 | 3.4353 | 0.0313 | 0.9886 | N/A | — | gap_masked(0.36) | refine_vqe |  |
| 2.720 | -27.9358 | -28.0328 | 0.0970 | 3.5542 | 0.0273 | 0.9899 | N/A | — | pass(0.55) | none |  |
| 2.790 | -28.6258 | -28.7116 | 0.0858 | 3.6931 | 0.0232 | 0.9913 | N/A | — | pass(0.46) | none |  |
| 2.860 | -29.3157 | -29.3915 | 0.0757 | 3.8320 | 0.0198 | 0.9925 | N/A | — | pass(0.40) | none |  |
| 2.930 | -30.0057 | -30.0724 | 0.0666 | 3.9710 | 0.0168 | 0.9935 | N/A | — | pass(0.34) | none |  |
| 3.000 | -30.6957 | -30.7541 | 0.0584 | 4.1101 | 0.0142 | 0.9944 | N/A | — | pass(0.28) | none |  |

## N = 12 (23 params)

**ΔE/gap: 0.7990 ± 1.4821 | P90=2.1338 | max=6.9105
|ΔE|/N: 4.47e-02
Fidelity: mean=0.8932 min=0.5227 (exact)
Distribution: [P25=0.056 | P50=0.177 | P75=0.692 | P90=2.134]
Regions: critical=2.5642 | ordered=0.0671**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=6.910 is 9× the mean — median may be more representative N=12

**Fidelity: mean F=0.8932, min F=0.5227** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -13.1903 | -14.9260 | 1.7356 | 0.2512 | 6.9105 | 0.5227 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.070 | -14.0188 | -15.5498 | 1.5310 | 0.3552 | 4.3101 | 0.6021 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -14.8473 | -16.2050 | 1.3577 | 0.4693 | 2.8932 | 0.6669 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -15.6757 | -16.8850 | 1.2093 | 0.5901 | 2.0494 | 0.7197 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.280 | -16.5050 | -17.5850 | 1.0800 | 0.7154 | 1.5096 | 0.7629 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.340 | -17.2156 | -18.1982 | 0.9826 | 0.8255 | 1.1903 | 0.7937 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.410 | -18.0444 | -18.9262 | 0.8818 | 0.9561 | 0.9223 | 0.8240 | N/A | — | severe_error(0.92) | increase_p |  |
| 1.480 | -18.8732 | -19.6659 | 0.7927 | 1.0885 | 0.7283 | 0.8492 | N/A | — | severe_error(0.71) | increase_p |  |
| 1.550 | -19.7019 | -20.4154 | 0.7135 | 1.2222 | 0.5838 | 0.8704 | N/A | — | severe_error(0.56) | increase_p |  |
| 1.620 | -20.5298 | -21.1733 | 0.6435 | 1.3569 | 0.4743 | 0.8882 | N/A | — | moderate_error(0.45) | refine_vqe |  |
| 1.690 | -21.3577 | -21.9386 | 0.5808 | 1.4924 | 0.3892 | 0.9033 | N/A | — | moderate_error(0.36) | refine_vqe |  |
| 1.760 | -22.1856 | -22.7101 | 0.5245 | 1.6285 | 0.3221 | 0.9162 | N/A | — | moderate_error(0.29) | refine_vqe |  |
| 1.830 | -23.0135 | -23.4871 | 0.4736 | 1.7651 | 0.2683 | 0.9272 | N/A | — | moderate_error(0.23) | refine_vqe |  |
| 1.900 | -23.8414 | -24.2691 | 0.4277 | 1.9022 | 0.2249 | 0.9367 | N/A | — | moderate_error(0.18) | refine_vqe |  |
| 1.970 | -24.6692 | -25.0553 | 0.3861 | 2.0396 | 0.1893 | 0.9449 | N/A | — | moderate_error(0.15) | refine_vqe |  |
| 2.030 | -25.3788 | -25.7323 | 0.3535 | 2.1576 | 0.1639 | 0.9510 | N/A | — | moderate_error(0.12) | refine_vqe |  |
| 2.100 | -26.2067 | -26.5254 | 0.3188 | 2.2955 | 0.1389 | 0.9573 | N/A | — | moderate_error(0.09) | refine_vqe |  |
| 2.170 | -27.0345 | -27.3216 | 0.2871 | 2.4336 | 0.1180 | 0.9628 | N/A | — | moderate_error(0.07) | refine_vqe |  |
| 2.240 | -27.8624 | -28.1207 | 0.2583 | 2.5719 | 0.1005 | 0.9675 | N/A | — | moderate_error(0.05) | refine_vqe |  |
| 2.310 | -28.6902 | -28.9223 | 0.2321 | 2.7103 | 0.0856 | 0.9717 | N/A | — | near_pass(0.04) | refine_vqe |  |
| 2.380 | -29.5180 | -29.7263 | 0.2083 | 2.8489 | 0.0731 | 0.9753 | N/A | — | near_pass(0.02) | refine_vqe |  |
| 2.450 | -30.3458 | -30.5323 | 0.1865 | 2.9876 | 0.0624 | 0.9785 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 2.520 | -31.1736 | -31.3403 | 0.1667 | 3.1264 | 0.0533 | 0.9813 | N/A | — | near_pass(0.00) | refine_vqe |  |
| 2.590 | -32.0014 | -32.1501 | 0.1487 | 3.2653 | 0.0455 | 0.9838 | N/A | — | gap_masked(0.50) | refine_vqe |  |
| 2.660 | -32.8292 | -32.9615 | 0.1323 | 3.4043 | 0.0389 | 0.9859 | N/A | — | gap_masked(0.44) | refine_vqe |  |
| 2.720 | -33.5388 | -33.6582 | 0.1194 | 3.5234 | 0.0339 | 0.9875 | N/A | — | gap_masked(0.40) | refine_vqe |  |
| 2.790 | -34.3666 | -34.4723 | 0.1058 | 3.6626 | 0.0289 | 0.9892 | N/A | — | gap_masked(0.35) | refine_vqe |  |
| 2.860 | -35.1943 | -35.2877 | 0.0934 | 3.8017 | 0.0246 | 0.9907 | N/A | — | pass(0.49) | none |  |
| 2.930 | -36.0221 | -36.1043 | 0.0822 | 3.9409 | 0.0208 | 0.9920 | N/A | — | pass(0.42) | none |  |
| 3.000 | -36.8499 | -36.9220 | 0.0721 | 4.0802 | 0.0177 | 0.9931 | N/A | — | pass(0.35) | none |  |

## N = 16 (31 params)

**ΔE/gap: 1.2885 ± 2.6052 | P90=3.2580 | max=12.5909
|ΔE|/N: 4.61e-02
Fidelity: mean=0.8587 min=0.3914 (exact)
Distribution: [P25=0.077 | P50=0.247 | P75=0.991 | P90=3.258]
Regions: critical=4.2237 | ordered=0.0935**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=12.591 is 10× the mean — median may be more representative N=16

**Fidelity: mean F=0.8587, min F=0.3914** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -17.6200 | -20.0164 | 2.3964 | 0.1903 | 12.5909 | 0.3914 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.070 | -18.7243 | -20.8335 | 2.1092 | 0.2943 | 7.1679 | 0.4863 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -19.8287 | -21.6966 | 1.8679 | 0.4103 | 4.5522 | 0.5657 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -20.9330 | -22.5953 | 1.6623 | 0.5338 | 3.1142 | 0.6313 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.280 | -22.0385 | -23.5222 | 1.4838 | 0.6618 | 2.2419 | 0.6860 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.340 | -22.9859 | -24.3351 | 1.3492 | 0.7740 | 1.7432 | 0.7255 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.410 | -24.0911 | -25.3013 | 1.2102 | 0.9068 | 1.3345 | 0.7646 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.480 | -25.1962 | -26.2836 | 1.0874 | 1.0412 | 1.0444 | 0.7975 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.550 | -26.3007 | -27.2795 | 0.9787 | 1.1766 | 0.8318 | 0.8254 | N/A | — | severe_error(0.82) | increase_p |  |
| 1.620 | -27.4042 | -28.2870 | 0.8828 | 1.3128 | 0.6724 | 0.8489 | N/A | — | severe_error(0.66) | increase_p |  |
| 1.690 | -28.5077 | -29.3045 | 0.7968 | 1.4496 | 0.5497 | 0.8689 | N/A | — | severe_error(0.53) | increase_p |  |
| 1.760 | -29.6112 | -30.3308 | 0.7195 | 1.5869 | 0.4534 | 0.8861 | N/A | — | moderate_error(0.42) | refine_vqe |  |
| 1.830 | -30.7147 | -31.3646 | 0.6499 | 1.7246 | 0.3769 | 0.9009 | N/A | — | moderate_error(0.34) | refine_vqe |  |
| 1.900 | -31.8182 | -32.4052 | 0.5870 | 1.8625 | 0.3151 | 0.9137 | N/A | — | moderate_error(0.28) | refine_vqe |  |
| 1.970 | -32.9217 | -33.4517 | 0.5300 | 2.0008 | 0.2649 | 0.9248 | N/A | — | moderate_error(0.23) | refine_vqe |  |
| 2.030 | -33.8676 | -34.3529 | 0.4854 | 2.1194 | 0.2290 | 0.9331 | N/A | — | moderate_error(0.19) | refine_vqe |  |
| 2.100 | -34.9710 | -35.4088 | 0.4378 | 2.2580 | 0.1939 | 0.9416 | N/A | — | moderate_error(0.15) | refine_vqe |  |
| 2.170 | -36.0745 | -36.4689 | 0.3945 | 2.3967 | 0.1646 | 0.9490 | N/A | — | moderate_error(0.12) | refine_vqe |  |
| 2.240 | -37.1779 | -37.5330 | 0.3551 | 2.5356 | 0.1400 | 0.9555 | N/A | — | moderate_error(0.09) | refine_vqe |  |
| 2.310 | -38.2814 | -38.6006 | 0.3192 | 2.6746 | 0.1193 | 0.9611 | N/A | — | moderate_error(0.07) | refine_vqe |  |
| 2.380 | -39.3848 | -39.6713 | 0.2865 | 2.8136 | 0.1018 | 0.9661 | N/A | — | moderate_error(0.05) | refine_vqe |  |
| 2.450 | -40.4883 | -40.7449 | 0.2567 | 2.9528 | 0.0869 | 0.9705 | N/A | — | near_pass(0.04) | refine_vqe |  |
| 2.520 | -41.5917 | -41.8212 | 0.2295 | 3.0920 | 0.0742 | 0.9743 | N/A | — | near_pass(0.03) | refine_vqe |  |
| 2.590 | -42.6951 | -42.9000 | 0.2048 | 3.2313 | 0.0634 | 0.9776 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 2.660 | -43.7985 | -43.9809 | 0.1824 | 3.3706 | 0.0541 | 0.9806 | N/A | — | near_pass(0.00) | refine_vqe |  |
| 2.720 | -44.7443 | -44.9091 | 0.1648 | 3.4901 | 0.0472 | 0.9828 | N/A | — | gap_masked(0.55) | refine_vqe |  |
| 2.790 | -45.8477 | -45.9937 | 0.1460 | 3.6295 | 0.0402 | 0.9851 | N/A | — | gap_masked(0.49) | refine_vqe |  |
| 2.860 | -46.9511 | -47.0801 | 0.1290 | 3.7689 | 0.0342 | 0.9872 | N/A | — | gap_masked(0.43) | refine_vqe |  |
| 2.930 | -48.0546 | -48.1681 | 0.1136 | 3.9084 | 0.0291 | 0.9889 | N/A | — | gap_masked(0.38) | refine_vqe |  |
| 3.000 | -49.1580 | -49.2577 | 0.0997 | 4.0480 | 0.0246 | 0.9905 | N/A | — | pass(0.49) | none |  |

## N = 20 (39 params)

**ΔE/gap: 1.5355 ± 2.6063 | P90=5.0844 | max=9.7403
|ΔE|/N: 4.70e-02
Fidelity: mean≥0.6595 min≥0.1316 (variance lower bound, 30 pts)
Var(H): 5.1204
Distribution: [P25=0.100 | P50=0.322 | P75=1.341 | P90=5.084]
Regions: critical=4.9575 | ordered=0.1213**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=9.740 is 6× the mean — median may be more representative N=20

**Fidelity: mean F≥0.6595, min F≥0.1316** (variance lower bound — N>16, exact statevector infeasible)
> 30/30 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=5.1204.

**Infidelity decomposition:** 30 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=14.5716.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -22.0478 | -25.1078 | 3.0600 | 0.3142 | 9.7403 | N/A | 10.4455 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.070 | -23.4280 | -26.1175 | 2.6895 | 0.3142 | 8.5610 | N/A | 9.9695 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -24.8082 | -27.1882 | 2.3800 | 0.3142 | 7.5759 | N/A | 9.5042 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -26.1883 | -28.3055 | 2.1172 | 0.4404 | 4.8076 | N/A | 9.0503 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.280 | -27.5702 | -29.4594 | 1.8892 | 0.5793 | 3.2613 | N/A | 8.5977 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.340 | -28.7546 | -30.4721 | 1.7175 | 0.6984 | 2.4591 | N/A | 8.2182 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.410 | -30.1366 | -31.6763 | 1.5398 | 0.8375 | 1.8385 | N/A | 7.7842 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.480 | -31.5182 | -32.9012 | 1.3830 | 0.9767 | 1.4160 | N/A | 7.3619 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.550 | -32.8984 | -34.1435 | 1.2451 | 1.1159 | 1.1158 | N/A | 6.9595 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.620 | -34.2775 | -35.4006 | 1.1231 | 1.2552 | 0.8947 | N/A | 6.5752 | dirty_state | severe_error(0.89) | increase_p |  |
| 1.690 | -35.6567 | -36.6705 | 1.0138 | 1.3946 | 0.7270 | N/A | 6.2020 | dirty_state | severe_error(0.71) | increase_p |  |
| 1.760 | -37.0359 | -37.9515 | 0.9156 | 1.5340 | 0.5969 | N/A | 5.8402 | dirty_state | severe_error(0.58) | increase_p |  |
| 1.830 | -38.4150 | -39.2422 | 0.8271 | 1.6735 | 0.4942 | N/A | 5.4898 | dirty_state | moderate_error(0.47) | refine_vqe |  |
| 1.900 | -39.7942 | -40.5413 | 0.7472 | 1.8130 | 0.4121 | N/A | 5.1511 | dirty_state | moderate_error(0.38) | refine_vqe |  |
| 1.970 | -41.1733 | -41.8481 | 0.6748 | 1.9525 | 0.3456 | N/A | 4.8242 | dirty_state | moderate_error(0.31) | refine_vqe |  |
| 2.030 | -42.3554 | -42.9735 | 0.6181 | 2.0722 | 0.2983 | N/A | 4.5533 | dirty_state | moderate_error(0.26) | refine_vqe |  |
| 2.100 | -43.7345 | -44.2921 | 0.5576 | 2.2117 | 0.2521 | ≥0.1316 | 4.2480 | dirty_state | moderate_error(0.21) | refine_vqe |  |
| 2.170 | -45.1136 | -45.6162 | 0.5027 | 2.3514 | 0.2138 | ≥0.2848 | 3.9543 | dirty_state | moderate_error(0.17) | refine_vqe |  |
| 2.240 | -46.4927 | -46.9453 | 0.4526 | 2.4910 | 0.1817 | ≥0.4082 | 3.6720 | dirty_state | moderate_error(0.14) | refine_vqe |  |
| 2.310 | -47.8718 | -48.2788 | 0.4069 | 2.6307 | 0.1547 | ≥0.5085 | 3.4011 | dirty_state | moderate_error(0.11) | refine_vqe |  |
| 2.380 | -49.2510 | -49.6163 | 0.3654 | 2.7704 | 0.1319 | ≥0.5906 | 3.1420 | dirty_state | moderate_error(0.09) | refine_vqe |  |
| 2.450 | -50.6301 | -50.9575 | 0.3275 | 2.9101 | 0.1125 | ≥0.6582 | 2.8946 | dirty_state | moderate_error(0.07) | refine_vqe |  |
| 2.520 | -52.0092 | -52.3021 | 0.2930 | 3.0498 | 0.0961 | ≥0.7141 | 2.6590 | dirty_state | near_pass(0.05) | refine_vqe |  |
| 2.590 | -53.3882 | -53.6498 | 0.2616 | 3.1895 | 0.0820 | ≥0.7606 | 2.4350 | dirty_state | near_pass(0.03) | refine_vqe |  |
| 2.660 | -54.7673 | -55.0003 | 0.2330 | 3.3293 | 0.0700 | ≥0.7995 | 2.2226 | dirty_state | near_pass(0.02) | refine_vqe |  |
| 2.720 | -55.9494 | -56.1599 | 0.2106 | 3.4491 | 0.0610 | ≥0.8277 | 2.0498 | dirty_state | near_pass(0.01) | refine_vqe |  |
| 2.790 | -57.3285 | -57.5151 | 0.1866 | 3.5888 | 0.0520 | ≥0.8557 | 1.8590 | dirty_state | near_pass(0.00) | refine_vqe |  |
| 2.860 | -58.7075 | -58.8725 | 0.1650 | 3.7286 | 0.0442 | ≥0.8792 | 1.6798 | dirty_state | gap_masked(0.55) | refine_vqe |  |
| 2.930 | -60.0866 | -60.2320 | 0.1454 | 3.8684 | 0.0376 | ≥0.8989 | 1.5122 | dirty_state | gap_masked(0.48) | refine_vqe |  |
| 3.000 | -61.4657 | -61.5934 | 0.1278 | 4.0082 | 0.0319 | ≥0.9156 | 1.3562 | dirty_state | gap_masked(0.43) | refine_vqe |  |

## N = 30 (59 params)

**ΔE/gap: 2.8883 ± 5.5814 | P90=8.1143 | max=22.5924
|ΔE|/N: 4.83e-02
Fidelity: mean≥0.5873 min≥0.0814 (variance lower bound, 30 pts)
Var(H): 7.8807
Distribution: [P25=0.156 | P50=0.498 | P75=2.082 | P90=8.114]
Regions: critical=9.5899 | ordered=0.1883**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=22.592 is 8× the mean — median may be more representative N=30

**Fidelity: mean F≥0.5873, min F≥0.0814** (variance lower bound — N>16, exact statevector infeasible)
> 30/30 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=7.8807.

**Infidelity decomposition:** 30 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=36.6940.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -33.1064 | -37.8381 | 4.7317 | 0.2094 | 22.5924 | N/A | 16.0195 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.070 | -35.1767 | -39.3276 | 4.1510 | 0.2094 | 19.8194 | N/A | 15.2957 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -37.2469 | -40.9173 | 3.6705 | 0.2896 | 12.6735 | N/A | 14.5889 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -39.3170 | -42.5812 | 3.2642 | 0.4291 | 7.6077 | N/A | 13.8989 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.280 | -41.3913 | -44.3024 | 2.9111 | 0.5686 | 5.1201 | N/A | 13.2027 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.340 | -43.1689 | -45.8145 | 2.6456 | 0.6882 | 3.8443 | N/A | 12.6192 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.410 | -45.2438 | -47.6140 | 2.3702 | 0.8278 | 2.8633 | N/A | 11.9463 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.480 | -47.3183 | -49.4454 | 2.1271 | 0.9674 | 2.1988 | N/A | 11.2899 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.550 | -49.3878 | -51.3036 | 1.9158 | 1.1071 | 1.7305 | N/A | 10.6799 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.620 | -51.4562 | -53.1847 | 1.7286 | 1.2468 | 1.3864 | N/A | 10.0945 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.690 | -53.5245 | -55.0855 | 1.5609 | 1.3865 | 1.1258 | N/A | 9.5267 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.760 | -55.5929 | -57.0033 | 1.4103 | 1.5262 | 0.9241 | N/A | 8.9763 | dirty_state | severe_error(0.92) | increase_p |  |
| 1.830 | -57.6614 | -58.9359 | 1.2746 | 1.6660 | 0.7651 | N/A | 8.4428 | dirty_state | severe_error(0.75) | increase_p |  |
| 1.900 | -59.7297 | -60.8817 | 1.1520 | 1.8058 | 0.6379 | N/A | 7.9271 | dirty_state | severe_error(0.62) | increase_p |  |
| 1.970 | -61.7981 | -62.8390 | 1.0409 | 1.9456 | 0.5350 | N/A | 7.4291 | dirty_state | severe_error(0.51) | increase_p |  |
| 2.030 | -63.5710 | -64.5250 | 0.9540 | 2.0654 | 0.4619 | N/A | 7.0162 | dirty_state | moderate_error(0.43) | refine_vqe |  |
| 2.100 | -65.6393 | -66.5005 | 0.8612 | 2.2052 | 0.3905 | N/A | 6.5507 | dirty_state | moderate_error(0.36) | refine_vqe |  |
| 2.170 | -67.7077 | -68.4845 | 0.7768 | 2.3451 | 0.3313 | N/A | 6.1026 | dirty_state | moderate_error(0.30) | refine_vqe |  |
| 2.240 | -69.7761 | -70.4760 | 0.6999 | 2.4849 | 0.2817 | ≥0.0814 | 5.6720 | dirty_state | moderate_error(0.24) | refine_vqe |  |
| 2.310 | -71.8445 | -72.4743 | 0.6298 | 2.6247 | 0.2400 | ≥0.2368 | 5.2581 | dirty_state | moderate_error(0.20) | refine_vqe |  |
| 2.380 | -73.9129 | -74.4789 | 0.5659 | 2.7646 | 0.2047 | ≥0.3639 | 4.8620 | dirty_state | moderate_error(0.16) | refine_vqe |  |
| 2.450 | -75.9814 | -76.4890 | 0.5077 | 2.9045 | 0.1748 | ≥0.4685 | 4.4835 | dirty_state | moderate_error(0.13) | refine_vqe |  |
| 2.520 | -78.0498 | -78.5044 | 0.4546 | 3.0444 | 0.1493 | ≥0.5552 | 4.1226 | dirty_state | moderate_error(0.10) | refine_vqe |  |
| 2.590 | -80.1182 | -80.5244 | 0.4062 | 3.1842 | 0.1276 | ≥0.6273 | 3.7792 | dirty_state | moderate_error(0.08) | refine_vqe |  |
| 2.660 | -82.1866 | -82.5488 | 0.3622 | 3.3241 | 0.1090 | ≥0.6875 | 3.4534 | dirty_state | moderate_error(0.06) | refine_vqe |  |
| 2.720 | -83.9594 | -84.2871 | 0.3277 | 3.4440 | 0.0951 | ≥0.7312 | 3.1882 | dirty_state | near_pass(0.05) | refine_vqe |  |
| 2.790 | -86.0278 | -86.3186 | 0.2908 | 3.5839 | 0.0811 | ≥0.7746 | 2.8950 | dirty_state | near_pass(0.03) | refine_vqe |  |
| 2.860 | -88.0962 | -88.3535 | 0.2573 | 3.7238 | 0.0691 | ≥0.8111 | 2.6193 | dirty_state | near_pass(0.02) | refine_vqe |  |
| 2.930 | -90.1646 | -90.3916 | 0.2270 | 3.8637 | 0.0588 | ≥0.8419 | 2.3609 | dirty_state | near_pass(0.01) | refine_vqe |  |
| 3.000 | -92.2330 | -92.4327 | 0.1997 | 4.0037 | 0.0499 | ≥0.8677 | 2.1201 | dirty_state | gap_masked(0.67) | refine_vqe |  |

## N = 40 (79 params)

**ΔE/gap: 4.5739 ± 9.7626 | P90=11.1176 | max=40.8964
|ΔE|/N: 4.92e-02
Fidelity: mean≥0.5518 min≥0.1308 (variance lower bound, 30 pts)
Var(H): 10.6982
Distribution: [P25=0.213 | P50=0.678 | P75=2.829 | P90=11.118]
Regions: critical=15.4635 | ordered=0.2570**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=40.896 is 9× the mean — median may be more representative N=40

**Fidelity: mean F≥0.5518, min F≥0.1308** (variance lower bound — N>16, exact statevector infeasible)
> 30/30 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=10.6982.

**Infidelity decomposition:** 30 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=75.0777.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -44.1454 | -50.5694 | 6.4240 | 0.1571 | 40.8964 | N/A | 21.6791 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.070 | -46.9080 | -52.5378 | 5.6298 | 0.1571 | 35.8404 | N/A | 20.7022 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -49.6689 | -54.6465 | 4.9775 | 0.2854 | 17.4399 | N/A | 19.7547 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -52.4294 | -56.8569 | 4.4275 | 0.4251 | 10.4152 | N/A | 18.8311 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.280 | -55.1974 | -59.1454 | 3.9480 | 0.5648 | 6.9898 | N/A | 17.8882 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.340 | -57.5695 | -61.1569 | 3.5875 | 0.6846 | 5.2402 | N/A | 17.0972 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.410 | -60.3388 | -63.5517 | 3.2128 | 0.8244 | 3.8973 | N/A | 16.1798 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.480 | -63.1078 | -65.9896 | 2.8818 | 0.9642 | 2.9889 | N/A | 15.2823 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.550 | -65.8686 | -68.4638 | 2.5952 | 1.1040 | 2.3507 | N/A | 14.4546 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.620 | -68.6265 | -70.9689 | 2.3423 | 1.2438 | 1.8832 | N/A | 13.6684 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.690 | -71.3845 | -73.5005 | 2.1160 | 1.3837 | 1.5293 | N/A | 12.9055 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.760 | -74.1424 | -76.0550 | 1.9127 | 1.5235 | 1.2555 | N/A | 12.1664 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.830 | -76.9002 | -78.6297 | 1.7295 | 1.6634 | 1.0398 | N/A | 11.4506 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.900 | -79.6581 | -81.2220 | 1.5639 | 1.8032 | 0.8673 | N/A | 10.7578 | dirty_state | severe_error(0.86) | increase_p |  |
| 1.970 | -82.4160 | -83.8300 | 1.4140 | 1.9431 | 0.7277 | N/A | 10.0883 | dirty_state | severe_error(0.71) | increase_p |  |
| 2.030 | -84.7799 | -86.0764 | 1.2965 | 2.0630 | 0.6285 | N/A | 9.5330 | dirty_state | severe_error(0.61) | increase_p |  |
| 2.100 | -87.5378 | -88.7089 | 1.1712 | 2.2029 | 0.5316 | N/A | 8.9068 | dirty_state | severe_error(0.51) | increase_p |  |
| 2.170 | -90.2956 | -91.3528 | 1.0571 | 2.3428 | 0.4512 | N/A | 8.3039 | dirty_state | moderate_error(0.42) | refine_vqe |  |
| 2.240 | -93.0535 | -94.0067 | 0.9532 | 2.4828 | 0.3839 | N/A | 7.7246 | dirty_state | moderate_error(0.35) | refine_vqe |  |
| 2.310 | -95.8114 | -96.6699 | 0.8585 | 2.6227 | 0.3273 | N/A | 7.1679 | dirty_state | moderate_error(0.29) | refine_vqe |  |
| 2.380 | -98.5694 | -99.3414 | 0.7720 | 2.7626 | 0.2794 | ≥0.1308 | 6.6337 | dirty_state | moderate_error(0.24) | refine_vqe |  |
| 2.450 | -101.3275 | -102.0206 | 0.6931 | 2.9025 | 0.2388 | ≥0.2732 | 6.1227 | dirty_state | moderate_error(0.20) | refine_vqe |  |
| 2.520 | -104.0855 | -104.7066 | 0.6211 | 3.0424 | 0.2042 | ≥0.3912 | 5.6349 | dirty_state | moderate_error(0.16) | refine_vqe |  |
| 2.590 | -106.8435 | -107.3990 | 0.5555 | 3.1824 | 0.1746 | ≥0.4894 | 5.1708 | dirty_state | moderate_error(0.13) | refine_vqe |  |
| 2.660 | -109.6014 | -110.0972 | 0.4958 | 3.3223 | 0.1492 | ≥0.5715 | 4.7300 | dirty_state | moderate_error(0.10) | refine_vqe |  |
| 2.720 | -111.9654 | -112.4143 | 0.4489 | 3.4423 | 0.1304 | ≥0.6311 | 4.3708 | dirty_state | moderate_error(0.08) | refine_vqe |  |
| 2.790 | -114.7233 | -115.1221 | 0.3987 | 3.5822 | 0.1113 | ≥0.6904 | 3.9734 | dirty_state | moderate_error(0.06) | refine_vqe |  |
| 2.860 | -117.4813 | -117.8346 | 0.3532 | 3.7222 | 0.0949 | ≥0.7402 | 3.5991 | dirty_state | near_pass(0.05) | refine_vqe |  |
| 2.930 | -120.2393 | -120.5513 | 0.3120 | 3.8621 | 0.0808 | ≥0.7822 | 3.2480 | dirty_state | near_pass(0.03) | refine_vqe |  |
| 3.000 | -122.9973 | -123.2720 | 0.2747 | 4.0021 | 0.0686 | ≥0.8177 | 2.9201 | dirty_state | near_pass(0.02) | refine_vqe |  |
