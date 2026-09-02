# Model Evaluation: heavy_hex

**Date**: 2026-09-01 16:35 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_res_mse.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 2.5] (15 pts)
**Target N**: [8, 10, 20]

---

## N = 8 (15 params)

**ΔE/gap: 1.3051 ± 1.5613 | P90=3.1015 | max=6.0528
|ΔE|/N: 1.32e-01
Fidelity: mean=0.8181 min=0.5447 (exact)
Distribution: [P25=0.340 | P50=0.612 | P75=1.449 | P90=3.101]
Regions: critical=2.9844 | ordered=0.2826**

**Fidelity: mean F=0.8181, min F=0.5447** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -8.0730 | -9.8947 | 1.8217 | 0.3010 | 6.0528 | 0.5447 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -8.9520 | -10.5644 | 1.6124 | 0.4521 | 3.5661 | 0.6300 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -9.7510 | -11.2098 | 1.4588 | 0.6067 | 2.4044 | 0.6911 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -10.6300 | -11.9500 | 1.3200 | 0.7897 | 1.6716 | 0.7434 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.430 | -11.5090 | -12.7141 | 1.2051 | 0.9819 | 1.2273 | 0.7839 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.540 | -12.3879 | -13.4964 | 1.1085 | 1.1807 | 0.9389 | 0.8157 | N/A | — | severe_error(0.94) | increase_p |  |
| 1.640 | -13.1870 | -14.2201 | 1.0331 | 1.3653 | 0.7567 | 0.8390 | N/A | — | severe_error(0.74) | increase_p |  |
| 1.750 | -14.0660 | -15.0271 | 0.9611 | 1.5716 | 0.6116 | 0.8598 | N/A | — | severe_error(0.59) | increase_p |  |
| 1.860 | -14.9449 | -15.8433 | 0.8984 | 1.7804 | 0.5046 | 0.8769 | N/A | — | severe_error(0.48) | increase_p |  |
| 1.960 | -15.7440 | -16.5920 | 0.8480 | 1.9719 | 0.4301 | 0.8899 | N/A | — | moderate_error(0.40) | refine_vqe |  |
| 2.070 | -16.6230 | -17.4216 | 0.7987 | 2.1840 | 0.3657 | 0.9020 | N/A | — | moderate_error(0.33) | refine_vqe |  |
| 2.180 | -17.5019 | -18.2566 | 0.7547 | 2.3972 | 0.3148 | 0.9121 | N/A | — | moderate_error(0.28) | refine_vqe |  |
| 2.290 | -18.3809 | -19.0961 | 0.7152 | 2.6114 | 0.2739 | 0.9208 | N/A | — | moderate_error(0.24) | refine_vqe |  |
| 2.390 | -19.1799 | -19.8627 | 0.6828 | 2.8068 | 0.2432 | 0.9276 | N/A | — | moderate_error(0.20) | refine_vqe |  |
| 2.500 | -20.0589 | -20.7091 | 0.6502 | 3.0223 | 0.2151 | 0.9342 | N/A | — | moderate_error(0.17) | refine_vqe |  |

## N = 10 (19 params)

**ΔE/gap: 2.0522 ± 2.7628 | P90=4.9336 | max=10.8047
|ΔE|/N: 1.37e-01
Fidelity: mean=0.7698 min=0.4315 (exact)
Distribution: [P25=0.455 | P50=0.835 | P75=2.099 | P90=4.934]
Regions: critical=4.8901 | ordered=0.3760**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=10.805 is 5× the mean — median may be more representative N=10

**Fidelity: mean F=0.7698, min F=0.4315** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -10.0937 | -12.4722 | 2.3784 | 0.2201 | 10.8047 | 0.4315 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -11.1924 | -13.2894 | 2.0970 | 0.3632 | 5.7733 | 0.5347 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -12.1912 | -14.0836 | 1.8924 | 0.5151 | 3.6742 | 0.6102 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -13.2899 | -14.9990 | 1.7091 | 0.6978 | 2.4494 | 0.6754 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -14.3885 | -15.9469 | 1.5584 | 0.8912 | 1.7487 | 0.7260 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.540 | -15.4872 | -16.9194 | 1.4321 | 1.0915 | 1.3121 | 0.7659 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.640 | -16.4860 | -17.8199 | 1.3339 | 1.2777 | 1.0440 | 0.7952 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.750 | -17.5847 | -18.8250 | 1.2403 | 1.4858 | 0.8348 | 0.8215 | N/A | — | severe_error(0.83) | increase_p |  |
| 1.860 | -18.6833 | -19.8422 | 1.1589 | 1.6962 | 0.6832 | 0.8430 | N/A | — | severe_error(0.67) | increase_p |  |
| 1.960 | -19.6821 | -20.7757 | 1.0935 | 1.8891 | 0.5789 | 0.8595 | N/A | — | severe_error(0.56) | increase_p |  |
| 2.070 | -20.7808 | -21.8104 | 1.0296 | 2.1026 | 0.4897 | 0.8747 | N/A | — | moderate_error(0.46) | refine_vqe |  |
| 2.180 | -21.8795 | -22.8521 | 0.9726 | 2.3171 | 0.4198 | 0.8877 | N/A | — | moderate_error(0.39) | refine_vqe |  |
| 2.290 | -22.9782 | -23.8998 | 0.9216 | 2.5324 | 0.3639 | 0.8987 | N/A | — | moderate_error(0.33) | refine_vqe |  |
| 2.390 | -23.9769 | -24.8565 | 0.8796 | 2.7287 | 0.3223 | 0.9074 | N/A | — | moderate_error(0.29) | refine_vqe |  |
| 2.500 | -25.0756 | -25.9132 | 0.8376 | 2.9452 | 0.2844 | 0.9157 | N/A | — | moderate_error(0.25) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 8.2419 ± 4.4129 | P90=13.7651 | max=16.4623
|ΔE|/N: 1.46e-01
Var(H): 17.7669
Distribution: [P25=4.895 | P50=8.362 | P75=11.064 | P90=13.765]
Regions: critical=13.1648 | ordered=3.1190**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=137.8357.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -20.1977 | -25.3695 | 5.1718 | 0.3142 | 16.4623 | N/A | 18.2035 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -22.3949 | -26.9058 | 4.5108 | 0.3142 | 14.3585 | N/A | 18.1291 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -24.3924 | -28.4372 | 4.0448 | 0.3142 | 12.8750 | N/A | 18.0636 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -26.5896 | -30.2298 | 3.6402 | 0.3142 | 11.5873 | N/A | 17.9940 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -28.7868 | -32.0983 | 3.3115 | 0.3142 | 10.5408 | N/A | 17.9267 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -30.9840 | -34.0227 | 3.0387 | 0.3142 | 9.6726 | N/A | 17.8620 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -32.9815 | -35.8088 | 2.8274 | 0.3142 | 8.9997 | N/A | 17.8052 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -35.1787 | -37.8057 | 2.6270 | 0.3142 | 8.3620 | N/A | 17.7451 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -37.3759 | -39.8292 | 2.4534 | 0.3142 | 7.8093 | N/A | 17.6875 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -39.3733 | -41.6875 | 2.3141 | 0.3142 | 7.3661 | N/A | 17.6373 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -41.5705 | -43.7486 | 2.1780 | 0.3519 | 6.1890 | N/A | 17.5843 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -43.7677 | -45.8248 | 2.0570 | 0.5713 | 3.6005 | N/A | 17.5339 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.290 | -45.9649 | -47.9136 | 1.9487 | 0.7908 | 2.4643 | N/A | 17.4859 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.390 | -47.9624 | -49.8220 | 1.8596 | 0.9903 | 1.8777 | N/A | 17.4443 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.500 | -50.1596 | -51.9300 | 1.7704 | 1.2099 | 1.4633 | N/A | 17.4010 | dirty_state | severe_error(1.00) | increase_p |  |
