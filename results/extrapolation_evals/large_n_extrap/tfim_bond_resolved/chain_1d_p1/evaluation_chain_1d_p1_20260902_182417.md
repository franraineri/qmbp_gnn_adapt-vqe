# Model Evaluation: chain_1d

**Date**: 2026-09-02 18:24 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__chain_1d_p1_signinv_fid_dot3_v1.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 3.0] (15 pts)
**Target N**: [30, 40, 50, 60]

---

## N = 30 (59 params)

**ΔE/gap: 2.1732 ± 3.0631 | P90=7.0213 | max=9.4483
|ΔE|/N: 4.74e-02
Fidelity: mean≥0.7273 min≥0.2700 (variance lower bound, 15 pts)
Var(H): 9.5148
Distribution: [P25=0.064 | P50=0.364 | P75=3.396 | P90=7.021]
Regions: critical=6.7430 | ordered=0.0801**

**Fidelity: mean F≥0.7273, min F≥0.2700** (variance lower bound — N>16, exact statevector infeasible)
> 15/15 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=9.5148.

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=21.3945.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -36.0626 | -37.8381 | 1.7755 | 0.2094 | 8.4773 | N/A | 4.7978 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -39.6983 | -40.9173 | 1.2191 | 0.2896 | 4.2092 | N/A | 4.1015 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.290 | -38.9919 | -44.5522 | 5.5603 | 0.5885 | 9.4483 | N/A | 32.1924 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -43.9371 | -48.1342 | 4.1971 | 0.8677 | 4.8372 | N/A | 27.8267 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.570 | -48.8762 | -51.8389 | 2.9627 | 1.1470 | 2.5831 | N/A | 22.2513 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.710 | -53.6377 | -55.6318 | 1.9940 | 1.4264 | 1.3979 | N/A | 16.7788 | dirty_state | severe_error(1.00) | increase_p |  |
| 1.860 | -58.5268 | -59.7683 | 1.2416 | 1.7259 | 0.7194 | N/A | 11.5333 | dirty_state | severe_error(0.70) | increase_p |  |
| 2.000 | -62.9505 | -63.6811 | 0.7306 | 2.0055 | 0.3643 | N/A | 7.1882 | dirty_state | moderate_error(0.33) | refine_vqe |  |
| 2.140 | -67.2486 | -67.6333 | 0.3846 | 2.2851 | 0.1683 | ≥0.2700 | 3.8120 | dirty_state | moderate_error(0.12) | refine_vqe |  |
| 2.290 | -71.6660 | -71.9027 | 0.2367 | 2.5848 | 0.0916 | ≥0.6718 | 2.1925 | dirty_state | near_pass(0.04) | refine_vqe |  |
| 2.430 | -75.7274 | -75.9142 | 0.1868 | 2.8645 | 0.0652 | ≥0.7960 | 1.6742 | dirty_state | near_pass(0.02) | refine_vqe |  |
| 2.570 | -79.7715 | -79.9468 | 0.1753 | 3.1443 | 0.0558 | ≥0.8417 | 1.5646 | dirty_state | near_pass(0.01) | refine_vqe |  |
| 2.710 | -83.8060 | -83.9972 | 0.1912 | 3.4240 | 0.0558 | ≥0.8474 | 1.7894 | dirty_state | near_pass(0.01) | refine_vqe |  |
| 2.860 | -88.1225 | -88.3535 | 0.2311 | 3.7238 | 0.0620 | ≥0.8314 | 2.3379 | dirty_state | near_pass(0.01) | refine_vqe |  |
| 3.000 | -92.1840 | -92.4327 | 0.2487 | 4.0037 | 0.0621 | ≥0.8327 | 2.6810 | dirty_state | near_pass(0.01) | refine_vqe |  |

## N = 40 (79 params)

**ΔE/gap: 4.6341 ± 6.5426 | P90=14.0265 | max=22.1017
|ΔE|/N: 8.04e-02
Fidelity: mean≥0.7373 min≥0.4260 (variance lower bound, 15 pts)
Var(H): 21.0705
Distribution: [P25=0.122 | P50=1.014 | P75=6.227 | P90=14.026]
Regions: critical=13.8132 | ordered=0.1658**

**Fidelity: mean F≥0.7373, min F≥0.4260** (variance lower bound — N>16, exact statevector infeasible)
> 15/15 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=21.0705.

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=46.0564.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -48.1354 | -50.5694 | 2.4341 | 0.1571 | 15.4958 | N/A | 6.4895 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -52.9817 | -54.6465 | 1.6647 | 0.2854 | 5.8328 | N/A | 5.5542 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.290 | -46.5530 | -59.4777 | 12.9246 | 0.5848 | 22.1017 | N/A | 67.3286 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -54.0259 | -64.2441 | 10.2182 | 0.8643 | 11.8224 | N/A | 61.9466 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.570 | -61.6018 | -69.1766 | 7.5747 | 1.1439 | 6.6217 | N/A | 52.8859 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.710 | -69.0497 | -74.2281 | 5.1784 | 1.4236 | 3.6375 | N/A | 41.0425 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -76.5923 | -79.7387 | 3.1463 | 1.7233 | 1.8258 | N/A | 28.3122 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.000 | -82.9216 | -84.9520 | 2.0303 | 2.0031 | 1.0136 | N/A | 19.9009 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.140 | -89.0274 | -90.2184 | 1.1909 | 2.2829 | 0.5217 | N/A | 12.2657 | dirty_state | severe_error(0.50) | increase_p |  |
| 2.290 | -95.2621 | -95.9081 | 0.6460 | 2.5827 | 0.2501 | N/A | 6.7931 | dirty_state | moderate_error(0.21) | refine_vqe |  |
| 2.430 | -100.8179 | -101.2544 | 0.4364 | 2.8625 | 0.1525 | ≥0.4260 | 4.7035 | dirty_state | moderate_error(0.11) | refine_vqe |  |
| 2.570 | -106.3444 | -106.6291 | 0.2848 | 3.1424 | 0.0906 | ≥0.6838 | 3.1228 | dirty_state | near_pass(0.04) | refine_vqe |  |
| 2.710 | -111.8267 | -112.0278 | 0.2012 | 3.4223 | 0.0588 | ≥0.8141 | 2.1774 | dirty_state | near_pass(0.01) | refine_vqe |  |
| 2.860 | -117.6718 | -117.8346 | 0.1627 | 3.7222 | 0.0437 | ≥0.8768 | 1.7069 | dirty_state | gap_masked(0.54) | refine_vqe |  |
| 3.000 | -123.0996 | -123.2720 | 0.1724 | 4.0021 | 0.0431 | ≥0.8859 | 1.8279 | dirty_state | gap_masked(0.57) | refine_vqe |  |

## N = 50 (99 params)

**ΔE/gap: 7.8630 ± 11.0994 | P90=23.1086 | max=38.4384
|ΔE|/N: 1.14e-01
Fidelity: mean≥0.6262 min≥0.3879 (variance lower bound, 15 pts)
Var(H): 36.6151
Distribution: [P25=0.369 | P50=1.811 | P75=9.652 | P90=23.109]
Regions: critical=22.8329 | ordered=0.4096**

**Fidelity: mean F≥0.6262, min F≥0.3879** (variance lower bound — N>16, exact statevector infeasible)
> 15/15 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=36.6151.

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=79.9949.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -60.2039 | -63.3012 | 3.0973 | 0.1257 | 24.6474 | N/A | 8.3516 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -66.2651 | -68.3756 | 2.1105 | 0.2835 | 7.4455 | N/A | 7.0141 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.290 | -51.9912 | -74.4031 | 22.4119 | 0.5831 | 38.4384 | N/A | 105.0171 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -62.4082 | -80.3540 | 17.9458 | 0.8628 | 20.8004 | N/A | 100.0207 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.570 | -72.9660 | -86.5142 | 13.5482 | 1.1425 | 11.8582 | N/A | 88.4601 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.710 | -83.2982 | -92.8245 | 9.5262 | 1.4223 | 6.6977 | N/A | 71.6576 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -93.8902 | -99.7090 | 5.8189 | 1.7221 | 3.3789 | N/A | 50.1641 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.000 | -102.5975 | -106.2229 | 3.6253 | 2.0020 | 1.8109 | N/A | 34.7299 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.140 | -110.1974 | -112.8035 | 2.6061 | 2.2818 | 1.1421 | N/A | 26.3662 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.290 | -118.1959 | -119.9135 | 1.7176 | 2.5817 | 0.6653 | N/A | 18.0944 | dirty_state | severe_error(0.65) | increase_p |  |
| 2.430 | -125.2903 | -126.5945 | 1.3043 | 2.8616 | 0.4558 | N/A | 14.3963 | dirty_state | moderate_error(0.43) | refine_vqe |  |
| 2.570 | -132.4252 | -133.3115 | 0.8862 | 3.1415 | 0.2821 | N/A | 10.3484 | dirty_state | moderate_error(0.24) | refine_vqe |  |
| 2.710 | -139.4757 | -140.0585 | 0.5828 | 3.4215 | 0.1703 | ≥0.3879 | 7.1654 | dirty_state | moderate_error(0.13) | refine_vqe |  |
| 2.860 | -146.9576 | -147.3156 | 0.3580 | 3.7214 | 0.0962 | ≥0.6695 | 4.5772 | dirty_state | near_pass(0.05) | refine_vqe |  |
| 3.000 | -153.8887 | -154.1113 | 0.2226 | 4.0013 | 0.0556 | ≥0.8211 | 2.8639 | dirty_state | near_pass(0.01) | refine_vqe |  |

## N = 60 (119 params)

**ΔE/gap: 11.5262 ± 16.2772 | P90=33.9721 | max=56.5999
|ΔE|/N: 1.44e-01
Fidelity: mean≥0.3092 min≥0.3092 (variance lower bound, 15 pts)
Var(H): 54.9163
Distribution: [P25=0.912 | P50=2.550 | P75=13.531 | P90=33.972]
Regions: critical=33.1333 | ordered=0.8174**

**Fidelity: mean F≥0.3092, min F≥0.3092** (variance lower bound — N>16, exact statevector infeasible)
> 15/15 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=54.9163.

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=123.1377.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -72.2366 | -76.0332 | 3.7966 | 0.1047 | 36.2548 | N/A | 10.1734 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.140 | -79.5263 | -82.1048 | 2.5784 | 0.2824 | 9.1303 | N/A | 8.7092 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.290 | -56.3803 | -89.3286 | 32.9483 | 0.5821 | 56.5999 | N/A | 142.3962 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -70.1340 | -96.4639 | 26.3299 | 0.8619 | 30.5480 | N/A | 138.2138 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.570 | -84.1066 | -103.8519 | 19.7453 | 1.1417 | 17.2939 | N/A | 123.6620 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.710 | -97.5346 | -111.4208 | 13.8862 | 1.4216 | 9.7680 | N/A | 101.3185 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -111.0272 | -119.6794 | 8.6522 | 1.7215 | 5.0260 | N/A | 72.4480 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.000 | -122.3903 | -127.4938 | 5.1034 | 2.0014 | 2.5500 | N/A | 47.8586 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.140 | -131.7212 | -135.3886 | 3.6675 | 2.2813 | 1.6076 | N/A | 36.5858 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.290 | -140.7415 | -143.9189 | 3.1774 | 2.5812 | 1.2310 | N/A | 32.9985 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.430 | -148.9123 | -151.9347 | 3.0224 | 2.8611 | 1.0564 | N/A | 32.8339 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.570 | -157.5839 | -159.9938 | 2.4099 | 3.1411 | 0.7672 | N/A | 27.7117 | dirty_state | severe_error(0.75) | increase_p |  |
| 2.710 | -166.2905 | -168.0891 | 1.7986 | 3.4210 | 0.5257 | N/A | 21.9068 | dirty_state | severe_error(0.50) | increase_p |  |
| 2.860 | -175.5671 | -176.7966 | 1.2295 | 3.7210 | 0.3304 | N/A | 15.8698 | dirty_state | moderate_error(0.30) | refine_vqe |  |
| 3.000 | -184.1362 | -184.9506 | 0.8143 | 4.0009 | 0.2035 | ≥0.3092 | 11.0585 | dirty_state | moderate_error(0.16) | refine_vqe |  |
