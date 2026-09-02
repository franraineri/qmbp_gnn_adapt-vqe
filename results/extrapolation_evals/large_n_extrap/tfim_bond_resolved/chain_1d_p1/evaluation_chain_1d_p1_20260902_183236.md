# Model Evaluation: chain_1d

**Date**: 2026-09-02 18:32 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__chain_1d_p1_signinv_v2.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 3.0] (15 pts)
**Target N**: [30, 40, 50, 60]

---

## N = 30 (59 params)

**ΔE/gap: 3.1873 ± 6.0779 | P90=9.5512 | max=22.5924
|ΔE|/N: 4.97e-02
Fidelity: mean≥0.6079 min≥0.1956 (variance lower bound, 15 pts)
Var(H): 7.9261
Distribution: [P25=0.158 | P50=0.497 | P75=2.135 | P90=9.551]
Regions: critical=10.6954 | ordered=0.1627**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=22.592 is 7× the mean — median may be more representative N=30

**Fidelity: mean F≥0.6079, min F≥0.1956** (variance lower bound — N>16, exact statevector infeasible)
> 15/15 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=7.9261.

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=40.8859.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -33.1064 | -37.8381 | 4.7317 | 0.2094 | 22.5924 | N/A | 16.0195 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -37.2469 | -40.9173 | 3.6705 | 0.2896 | 12.6735 | N/A | 14.5889 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.290 | -41.6876 | -44.5522 | 2.8647 | 0.5885 | 4.8678 | N/A | 13.1047 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -45.8366 | -48.1342 | 2.2976 | 0.8677 | 2.6480 | N/A | 11.7566 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.570 | -49.9788 | -51.8389 | 1.8602 | 1.1470 | 1.6218 | N/A | 10.5109 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.710 | -54.1155 | -55.6318 | 1.5163 | 1.4264 | 1.0630 | N/A | 9.3677 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.860 | -58.5478 | -59.7683 | 1.2205 | 1.7259 | 0.7072 | N/A | 8.2196 | dirty_state | severe_error(0.69) | increase_p |  |
| 2.000 | -62.6845 | -63.6811 | 0.9966 | 2.0055 | 0.4969 | N/A | 7.2210 | dirty_state | moderate_error(0.47) | refine_vqe |  |
| 2.140 | -66.8213 | -67.6333 | 0.8120 | 2.2851 | 0.3553 | N/A | 6.2925 | dirty_state | moderate_error(0.32) | refine_vqe |  |
| 2.290 | -71.2535 | -71.9027 | 0.6492 | 2.5848 | 0.2512 | ≥0.1956 | 5.3745 | dirty_state | moderate_error(0.21) | refine_vqe |  |
| 2.430 | -75.3904 | -75.9142 | 0.5238 | 2.8645 | 0.1829 | ≥0.4406 | 4.5898 | dirty_state | moderate_error(0.14) | refine_vqe |  |
| 2.570 | -79.5272 | -79.9468 | 0.4196 | 3.1443 | 0.1334 | ≥0.6080 | 3.8755 | dirty_state | moderate_error(0.09) | refine_vqe |  |
| 2.710 | -83.6640 | -83.9972 | 0.3332 | 3.4240 | 0.0973 | ≥0.7244 | 3.2315 | dirty_state | near_pass(0.05) | refine_vqe |  |
| 2.860 | -88.0962 | -88.3535 | 0.2573 | 3.7238 | 0.0691 | ≥0.8111 | 2.6193 | dirty_state | near_pass(0.02) | refine_vqe |  |
| 3.000 | -92.2330 | -92.4327 | 0.1997 | 4.0037 | 0.0499 | ≥0.8677 | 2.1201 | dirty_state | gap_masked(0.67) | refine_vqe |  |

## N = 40 (79 params)

**ΔE/gap: 5.0282 ± 10.5304 | P90=13.1213 | max=40.8964
|ΔE|/N: 5.06e-02
Fidelity: mean≥0.5756 min≥0.2353 (variance lower bound, 15 pts)
Var(H): 10.7593
Distribution: [P25=0.216 | P50=0.676 | P75=2.903 | P90=13.121]
Regions: critical=17.1457 | ordered=0.2222**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=40.896 is 8× the mean — median may be more representative N=40

**Fidelity: mean F≥0.5756, min F≥0.2353** (variance lower bound — N>16, exact statevector infeasible)
> 15/15 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=10.7593.

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=81.5041.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -44.1454 | -50.5694 | 6.4240 | 0.1571 | 40.8964 | N/A | 21.6791 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -49.6689 | -54.6465 | 4.9775 | 0.2854 | 17.4399 | N/A | 19.7547 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.290 | -55.5928 | -59.4777 | 3.8849 | 0.5848 | 6.6434 | N/A | 17.7553 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -61.1300 | -64.2441 | 3.1141 | 0.8643 | 3.6030 | N/A | 15.9210 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.570 | -66.6566 | -69.1766 | 2.5200 | 1.1439 | 2.2029 | N/A | 14.2275 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.710 | -72.1725 | -74.2281 | 2.0557 | 1.4236 | 1.4440 | N/A | 12.6918 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.860 | -78.0822 | -79.7387 | 1.6565 | 1.7233 | 0.9612 | N/A | 11.1508 | dirty_state | severe_error(0.96) | increase_p |  |
| 2.000 | -83.5979 | -84.9520 | 1.3540 | 2.0031 | 0.6760 | N/A | 9.8085 | dirty_state | severe_error(0.66) | increase_p |  |
| 2.140 | -89.1137 | -90.2184 | 1.1047 | 2.2829 | 0.4839 | N/A | 8.5594 | dirty_state | moderate_error(0.46) | refine_vqe |  |
| 2.290 | -95.0234 | -95.9081 | 0.8847 | 2.5827 | 0.3425 | N/A | 7.3247 | dirty_state | moderate_error(0.31) | refine_vqe |  |
| 2.430 | -100.5395 | -101.2544 | 0.7149 | 2.8625 | 0.2497 | ≥0.2353 | 6.2663 | dirty_state | moderate_error(0.21) | refine_vqe |  |
| 2.570 | -106.0555 | -106.6291 | 0.5737 | 3.1424 | 0.1826 | ≥0.4632 | 5.3010 | dirty_state | moderate_error(0.14) | refine_vqe |  |
| 2.710 | -111.5714 | -112.0278 | 0.4564 | 3.4223 | 0.1334 | ≥0.6218 | 4.4295 | dirty_state | moderate_error(0.09) | refine_vqe |  |
| 2.860 | -117.4813 | -117.8346 | 0.3532 | 3.7222 | 0.0949 | ≥0.7402 | 3.5991 | dirty_state | near_pass(0.05) | refine_vqe |  |
| 3.000 | -122.9973 | -123.2720 | 0.2747 | 4.0021 | 0.0686 | ≥0.8177 | 2.9201 | dirty_state | near_pass(0.02) | refine_vqe |  |

## N = 50 (99 params)

**ΔE/gap: 7.2483 ± 16.3507 | P90=16.7238 | max=64.7848
|ΔE|/N: 5.13e-02
Fidelity: mean≥0.4567 min≥0.0235 (variance lower bound, 15 pts)
Var(H): 13.6571
Distribution: [P25=0.276 | P50=0.859 | P75=3.680 | P90=16.724]
Regions: critical=25.0103 | ordered=0.2834**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=64.785 is 9× the mean — median may be more representative N=50

**Fidelity: mean F≥0.4567, min F≥0.0235** (variance lower bound — N>16, exact statevector infeasible)
> 15/15 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=13.6571.

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=145.2502.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -55.1601 | -63.3012 | 8.1411 | 0.1257 | 64.7848 | N/A | 27.4486 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -62.0695 | -68.3756 | 6.3061 | 0.2835 | 22.2468 | N/A | 25.0280 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.290 | -69.4826 | -74.4031 | 4.9206 | 0.5831 | 8.4392 | N/A | 22.4919 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -76.4110 | -80.3540 | 3.9430 | 0.8628 | 4.5702 | N/A | 20.1615 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.570 | -83.3258 | -86.5142 | 3.1884 | 1.1425 | 2.7907 | N/A | 18.0015 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.710 | -90.2210 | -92.8245 | 2.6035 | 1.4223 | 1.8305 | N/A | 16.0765 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.860 | -97.6087 | -99.7090 | 2.1004 | 1.7221 | 1.2196 | N/A | 14.1428 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.000 | -104.5040 | -106.2229 | 1.7189 | 2.0020 | 0.8586 | N/A | 12.4569 | dirty_state | severe_error(0.85) | increase_p |  |
| 2.140 | -111.3993 | -112.8035 | 1.4042 | 2.2818 | 0.6154 | N/A | 10.8864 | dirty_state | severe_error(0.60) | increase_p |  |
| 2.290 | -118.7872 | -119.9135 | 1.1263 | 2.5817 | 0.4363 | N/A | 9.3320 | dirty_state | moderate_error(0.41) | refine_vqe |  |
| 2.430 | -125.6830 | -126.5945 | 0.9115 | 2.8616 | 0.3185 | ≥0.0235 | 7.9965 | dirty_state | moderate_error(0.28) | refine_vqe |  |
| 2.570 | -132.5787 | -133.3115 | 0.7327 | 3.1415 | 0.2332 | ≥0.3133 | 6.7776 | dirty_state | moderate_error(0.19) | refine_vqe |  |
| 2.710 | -139.4744 | -140.0585 | 0.5841 | 3.4215 | 0.1707 | ≥0.5152 | 5.6752 | dirty_state | moderate_error(0.13) | refine_vqe |  |
| 2.860 | -146.8626 | -147.3156 | 0.4530 | 3.7214 | 0.1217 | ≥0.6663 | 4.6219 | dirty_state | moderate_error(0.08) | refine_vqe |  |
| 3.000 | -153.7582 | -154.1113 | 0.3531 | 4.0013 | 0.0882 | ≥0.7652 | 3.7587 | dirty_state | near_pass(0.04) | refine_vqe |  |

## N = 60 (119 params)

**ΔE/gap: 9.8698 ± 23.6173 | P90=20.3948 | max=94.5185
|ΔE|/N: 5.20e-02
Fidelity: mean≥0.4651 min≥0.1575 (variance lower bound, 15 pts)
Var(H): 16.6328
Distribution: [P25=0.338 | P50=1.045 | P75=4.470 | P90=20.395]
Regions: critical=34.3710 | ordered=0.3465**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=94.518 is 10× the mean — median may be more representative N=60

**Fidelity: mean F≥0.4651, min F≥0.1575** (variance lower bound — N>16, exact statevector infeasible)
> 15/15 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=16.6328.

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=238.9705.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -66.1352 | -76.0332 | 9.8979 | 0.1047 | 94.5185 | N/A | 33.3964 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -74.4375 | -82.1048 | 7.6673 | 0.2824 | 27.1500 | N/A | 30.4629 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.290 | -83.3548 | -89.3286 | 5.9738 | 0.5821 | 10.2620 | N/A | 27.3259 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -91.6773 | -96.4639 | 4.7866 | 0.8619 | 5.5534 | N/A | 24.4911 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.570 | -99.9853 | -103.8519 | 3.8666 | 1.1417 | 3.3865 | N/A | 21.8399 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.710 | -108.2606 | -111.4208 | 3.1602 | 1.4216 | 2.2230 | N/A | 19.5248 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -117.1269 | -119.6794 | 2.5525 | 1.7215 | 1.4827 | N/A | 17.1983 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.000 | -125.4024 | -127.4938 | 2.0914 | 2.0014 | 1.0450 | N/A | 15.1680 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.140 | -133.6778 | -135.3886 | 1.7108 | 2.2813 | 0.7499 | N/A | 13.2755 | dirty_state | severe_error(0.74) | increase_p |  |
| 2.290 | -142.5443 | -143.9189 | 1.3746 | 2.5812 | 0.5325 | N/A | 11.4010 | dirty_state | severe_error(0.51) | increase_p |  |
| 2.430 | -150.8204 | -151.9347 | 1.1144 | 2.8611 | 0.3895 | N/A | 9.7875 | dirty_state | moderate_error(0.36) | refine_vqe |  |
| 2.570 | -159.0964 | -159.9938 | 0.8974 | 3.1411 | 0.2857 | ≥0.1575 | 8.3121 | dirty_state | moderate_error(0.25) | refine_vqe |  |
| 2.710 | -167.3724 | -168.0891 | 0.7167 | 3.4210 | 0.2095 | ≥0.4041 | 6.9744 | dirty_state | moderate_error(0.17) | refine_vqe |  |
| 2.860 | -176.2395 | -176.7966 | 0.5571 | 3.7210 | 0.1497 | ≥0.5888 | 5.6937 | dirty_state | moderate_error(0.10) | refine_vqe |  |
| 3.000 | -184.5154 | -184.9506 | 0.4352 | 4.0009 | 0.1088 | ≥0.7101 | 4.6411 | dirty_state | moderate_error(0.06) | refine_vqe |  |
