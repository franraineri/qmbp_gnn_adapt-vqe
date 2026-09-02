# Model Evaluation: chain_1d

**Date**: 2026-09-02 18:30 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__chain_1d_p1_signinv_fid_v1.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 3.0] (15 pts)
**Target N**: [30, 40, 50, 60]

---

## N = 30 (59 params)

**ΔE/gap: 1.6567 ± 2.3223 | P90=4.4922 | max=8.4761
|ΔE|/N: 3.75e-02
Fidelity: mean≥0.5562 min≥0.1246 (variance lower bound, 15 pts)
Var(H): 6.5489
Distribution: [P25=0.177 | P50=0.512 | P75=2.002 | P90=4.492]
Regions: critical=4.9684 | ordered=0.1804**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=8.476 is 5× the mean — median may be more representative N=30

**Fidelity: mean F≥0.5562, min F≥0.1246** (variance lower bound — N>16, exact statevector infeasible)
> 15/15 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=6.5489.

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=15.3177.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -36.0629 | -37.8381 | 1.7752 | 0.2094 | 8.4761 | N/A | 4.8013 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -39.6986 | -40.9173 | 1.2188 | 0.2896 | 4.2081 | N/A | 4.1107 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.290 | -41.7971 | -44.5522 | 2.7551 | 0.5885 | 4.6816 | N/A | 12.6260 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -45.9583 | -48.1342 | 2.1759 | 0.8677 | 2.5078 | N/A | 11.0921 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.570 | -50.1224 | -51.8389 | 1.7165 | 1.1470 | 1.4965 | N/A | 9.5914 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.710 | -54.2007 | -55.6318 | 1.4311 | 1.4264 | 1.0033 | N/A | 8.7531 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.860 | -58.5552 | -59.7683 | 1.2131 | 1.7259 | 0.7029 | N/A | 8.1321 | dirty_state | severe_error(0.69) | increase_p |  |
| 2.000 | -62.6546 | -63.6811 | 1.0265 | 2.0055 | 0.5118 | N/A | 7.4272 | dirty_state | severe_error(0.49) | increase_p |  |
| 2.140 | -66.7714 | -67.6333 | 0.8619 | 2.2851 | 0.3772 | N/A | 6.6841 | dirty_state | moderate_error(0.34) | refine_vqe |  |
| 2.290 | -71.1977 | -71.9027 | 0.7050 | 2.5848 | 0.2728 | ≥0.1246 | 5.8489 | dirty_state | moderate_error(0.23) | refine_vqe |  |
| 2.430 | -75.3343 | -75.9142 | 0.5798 | 2.8645 | 0.2024 | ≥0.3788 | 5.0970 | dirty_state | moderate_error(0.16) | refine_vqe |  |
| 2.570 | -79.4714 | -79.9468 | 0.4754 | 3.1443 | 0.1512 | ≥0.5538 | 4.4111 | dirty_state | moderate_error(0.11) | refine_vqe |  |
| 2.710 | -83.6092 | -83.9972 | 0.3880 | 3.4240 | 0.1133 | ≥0.6771 | 3.7862 | dirty_state | moderate_error(0.07) | refine_vqe |  |
| 2.860 | -88.0432 | -88.3535 | 0.3103 | 3.7238 | 0.0833 | ≥0.7703 | 3.1855 | dirty_state | near_pass(0.04) | refine_vqe |  |
| 3.000 | -92.1824 | -92.4327 | 0.2503 | 4.0037 | 0.0625 | ≥0.8324 | 2.6867 | dirty_state | near_pass(0.01) | refine_vqe |  |

## N = 40 (79 params)

**ΔE/gap: 2.6232 ± 4.0533 | P90=6.6484 | max=15.4876
|ΔE|/N: 3.99e-02
Fidelity: mean≥0.5315 min≥0.1758 (variance lower bound, 15 pts)
Var(H): 9.2981
Distribution: [P25=0.234 | P50=0.667 | P75=3.083 | P90=6.648]
Regions: critical=8.0915 | ordered=0.2369**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=15.488 is 6× the mean — median may be more representative N=40

**Fidelity: mean F≥0.5315, min F≥0.1758** (variance lower bound — N>16, exact statevector infeasible)
> 15/15 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=9.2981.

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=29.6302.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -48.1366 | -50.5694 | 2.4328 | 0.1571 | 15.4876 | N/A | 6.5218 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -52.9831 | -54.6465 | 1.6634 | 0.2854 | 5.8281 | N/A | 5.5815 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.290 | -55.2701 | -59.4777 | 4.2076 | 0.5848 | 7.1952 | N/A | 20.1651 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -60.9121 | -64.2441 | 3.3320 | 0.8643 | 3.8551 | N/A | 17.6383 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.570 | -66.5335 | -69.1766 | 2.6430 | 1.1439 | 2.3105 | N/A | 15.2497 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.710 | -72.1873 | -74.2281 | 2.0408 | 1.4236 | 1.4336 | N/A | 12.6824 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.860 | -78.1663 | -79.7387 | 1.5724 | 1.7233 | 0.9124 | N/A | 10.5338 | dirty_state | severe_error(0.91) | increase_p |  |
| 2.000 | -83.6150 | -84.9520 | 1.3370 | 2.0031 | 0.6675 | N/A | 9.6556 | dirty_state | severe_error(0.65) | increase_p |  |
| 2.140 | -89.0935 | -90.2184 | 1.1248 | 2.2829 | 0.4927 | N/A | 8.7049 | dirty_state | moderate_error(0.47) | refine_vqe |  |
| 2.290 | -94.9743 | -95.9081 | 0.9338 | 2.5827 | 0.3616 | N/A | 7.7378 | dirty_state | moderate_error(0.33) | refine_vqe |  |
| 2.430 | -100.4850 | -101.2544 | 0.7693 | 2.8625 | 0.2688 | ≥0.1758 | 6.7540 | dirty_state | moderate_error(0.23) | refine_vqe |  |
| 2.570 | -106.0014 | -106.6291 | 0.6277 | 3.1424 | 0.1998 | ≥0.4113 | 5.8131 | dirty_state | moderate_error(0.16) | refine_vqe |  |
| 2.710 | -111.5204 | -112.0278 | 0.5074 | 3.4223 | 0.1483 | ≥0.5784 | 4.9381 | dirty_state | moderate_error(0.10) | refine_vqe |  |
| 2.860 | -117.4341 | -117.8346 | 0.4005 | 3.7222 | 0.1076 | ≥0.7044 | 4.0949 | dirty_state | moderate_error(0.06) | refine_vqe |  |
| 3.000 | -122.9535 | -123.2720 | 0.3185 | 4.0021 | 0.0796 | ≥0.7877 | 3.3998 | dirty_state | near_pass(0.03) | refine_vqe |  |

## N = 50 (99 params)

**ΔE/gap: 3.9856 ± 6.4258 | P90=10.0321 | max=24.6044
|ΔE|/N: 4.53e-02
Fidelity: mean≥0.5421 min≥0.2851 (variance lower bound, 15 pts)
Var(H): 13.4510
Distribution: [P25=0.285 | P50=0.822 | P75=4.882 | P90=10.032]
Regions: critical=12.4994 | ordered=0.2878**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=24.604 is 6× the mean — median may be more representative N=50

**Fidelity: mean F≥0.5421, min F≥0.2851** (variance lower bound — N>16, exact statevector infeasible)
> 15/15 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=13.4510.

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=53.1038.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -60.2093 | -63.3012 | 3.0919 | 0.1257 | 24.6044 | N/A | 8.2522 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -66.2666 | -68.3756 | 2.1090 | 0.2835 | 7.4400 | N/A | 7.0623 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.290 | -67.5462 | -74.4031 | 6.8569 | 0.5831 | 11.7602 | N/A | 34.9301 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -75.0108 | -80.3540 | 5.3432 | 0.8628 | 6.1931 | N/A | 30.0511 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.570 | -82.4342 | -86.5142 | 4.0800 | 1.1425 | 3.5711 | N/A | 24.8852 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.710 | -89.8413 | -92.8245 | 2.9832 | 1.4223 | 2.0974 | N/A | 19.2477 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -97.5032 | -99.7090 | 2.2058 | 1.7221 | 1.2809 | N/A | 15.1753 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.000 | -104.5767 | -106.2229 | 1.6462 | 2.0020 | 0.8223 | N/A | 11.9612 | dirty_state | severe_error(0.81) | increase_p |  |
| 2.140 | -111.4228 | -112.8035 | 1.3807 | 2.2818 | 0.6051 | N/A | 10.7078 | dirty_state | severe_error(0.58) | increase_p |  |
| 2.290 | -118.7870 | -119.9135 | 1.1265 | 2.5817 | 0.4363 | N/A | 9.3259 | dirty_state | moderate_error(0.41) | refine_vqe |  |
| 2.430 | -125.6604 | -126.5945 | 0.9341 | 2.8616 | 0.3264 | N/A | 8.1895 | dirty_state | moderate_error(0.29) | refine_vqe |  |
| 2.570 | -132.5485 | -133.3115 | 0.7629 | 3.1415 | 0.2429 | ≥0.2851 | 7.0557 | dirty_state | moderate_error(0.20) | refine_vqe |  |
| 2.710 | -139.4445 | -140.0585 | 0.6140 | 3.4215 | 0.1794 | ≥0.4905 | 5.9642 | dirty_state | moderate_error(0.14) | refine_vqe |  |
| 2.860 | -146.8338 | -147.3156 | 0.4817 | 3.7214 | 0.1294 | ≥0.6452 | 4.9139 | dirty_state | moderate_error(0.08) | refine_vqe |  |
| 3.000 | -153.7315 | -154.1113 | 0.3798 | 4.0013 | 0.0949 | ≥0.7475 | 4.0421 | dirty_state | near_pass(0.05) | refine_vqe |  |

## N = 60 (119 params)

**ΔE/gap: 6.1313 ± 9.6565 | P90=16.5681 | max=35.8475
|ΔE|/N: 5.87e-02
Fidelity: mean≥0.4620 min≥0.1598 (variance lower bound, 15 pts)
Var(H): 21.5704
Distribution: [P25=0.334 | P50=1.194 | P75=7.743 | P90=16.568]
Regions: critical=19.0334 | ordered=0.3423**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=35.848 is 6× the mean — median may be more representative N=60

**Fidelity: mean F≥0.4620, min F≥0.1598** (variance lower bound — N>16, exact statevector infeasible)
> 15/15 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=21.5704.

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=90.7718.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -72.2792 | -76.0332 | 3.7539 | 0.1047 | 35.8475 | N/A | 10.0841 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -79.5490 | -82.1048 | 2.5558 | 0.2824 | 9.0501 | N/A | 8.5802 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.290 | -77.4714 | -89.3286 | 11.8572 | 0.5821 | 20.3687 | N/A | 62.7975 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -87.0973 | -96.4639 | 9.3666 | 0.8619 | 10.8672 | N/A | 54.9474 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.570 | -96.5038 | -103.8519 | 7.3481 | 1.1417 | 6.4358 | N/A | 47.1374 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.710 | -105.9285 | -111.4208 | 5.4923 | 1.4216 | 3.8635 | N/A | 38.0142 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -116.3283 | -119.6794 | 3.3511 | 1.7215 | 1.9467 | N/A | 24.1367 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.000 | -125.1051 | -127.4938 | 2.3887 | 2.0014 | 1.1935 | N/A | 18.0286 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.140 | -133.7268 | -135.3886 | 1.6618 | 2.2813 | 0.7285 | N/A | 13.0610 | dirty_state | severe_error(0.71) | increase_p |  |
| 2.290 | -142.5630 | -143.9189 | 1.3559 | 2.5812 | 0.5253 | N/A | 11.3133 | dirty_state | severe_error(0.50) | increase_p |  |
| 2.430 | -150.8368 | -151.9347 | 1.0979 | 2.8611 | 0.3837 | N/A | 9.6579 | dirty_state | moderate_error(0.35) | refine_vqe |  |
| 2.570 | -159.0989 | -159.9938 | 0.8949 | 3.1411 | 0.2849 | ≥0.1598 | 8.2893 | dirty_state | moderate_error(0.25) | refine_vqe |  |
| 2.710 | -167.3657 | -168.0891 | 0.7234 | 3.4210 | 0.2115 | ≥0.3984 | 7.0405 | dirty_state | moderate_error(0.17) | refine_vqe |  |
| 2.860 | -176.2330 | -176.7966 | 0.5636 | 3.7210 | 0.1515 | ≥0.5841 | 5.7582 | dirty_state | moderate_error(0.11) | refine_vqe |  |
| 3.000 | -184.5087 | -184.9506 | 0.4419 | 4.0009 | 0.1104 | ≥0.7058 | 4.7096 | dirty_state | moderate_error(0.06) | refine_vqe |  |
