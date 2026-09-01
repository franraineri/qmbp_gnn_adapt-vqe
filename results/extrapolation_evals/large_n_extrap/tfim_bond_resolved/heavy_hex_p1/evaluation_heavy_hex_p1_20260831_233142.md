# Model Evaluation: heavy_hex

**Date**: 2026-08-31 23:31 UTC
**Model**: data/model_zoo/checkpoints/unified_tfim_br_heavy_hex_multiN_4+6+8+10+12+14+18+20+22+24+30+32+40_p1.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 2.5] (15 pts)
**Target N**: [8, 10, 20]

---

## N = 8 (15 params)

**ΔE/gap: 0.5567 ± 0.6736 | P90=1.2536 | max=2.5720
|ΔE|/N: 5.30e-02
Fidelity: mean=0.9099 min=0.7406 (exact)
Distribution: [P25=0.091 | P50=0.254 | P75=0.783 | P90=1.254]
Regions: critical=1.3086 | ordered=0.0655**

**Fidelity: mean F=0.9099, min F=0.7406** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -9.1206 | -9.8947 | 0.7741 | 0.3010 | 2.5720 | 0.7406 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -9.9061 | -10.5644 | 0.6583 | 0.4521 | 1.4559 | 0.8056 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.210 | -10.6334 | -11.2098 | 0.5764 | 0.6067 | 0.9500 | 0.8475 | N/A | — | severe_error(0.95) | increase_p |  |
| 1.320 | -11.4203 | -11.9500 | 0.5296 | 0.7897 | 0.6707 | 0.8762 | N/A | — | severe_error(0.65) | increase_p |  |
| 1.430 | -11.8357 | -12.7141 | 0.8784 | 0.9819 | 0.8945 | 0.8362 | N/A | — | severe_error(0.89) | increase_p |  |
| 1.540 | -12.8710 | -13.4964 | 0.6254 | 1.1807 | 0.5297 | 0.8879 | N/A | — | severe_error(0.50) | increase_p |  |
| 1.640 | -13.6918 | -14.2201 | 0.5283 | 1.3653 | 0.3869 | 0.9104 | N/A | — | moderate_error(0.35) | refine_vqe |  |
| 1.750 | -14.6273 | -15.0271 | 0.3998 | 1.5716 | 0.2544 | 0.9353 | N/A | — | moderate_error(0.22) | refine_vqe |  |
| 1.860 | -15.5335 | -15.8433 | 0.3098 | 1.7804 | 0.1740 | 0.9522 | N/A | — | moderate_error(0.13) | refine_vqe |  |
| 1.960 | -16.3274 | -16.5920 | 0.2647 | 1.9719 | 0.1342 | 0.9612 | N/A | — | moderate_error(0.09) | refine_vqe |  |
| 2.070 | -17.1995 | -17.4216 | 0.2222 | 2.1840 | 0.1017 | 0.9692 | N/A | — | moderate_error(0.05) | refine_vqe |  |
| 2.180 | -18.0659 | -18.2566 | 0.1907 | 2.3972 | 0.0796 | 0.9750 | N/A | — | near_pass(0.03) | refine_vqe |  |
| 2.290 | -18.9369 | -19.0961 | 0.1592 | 2.6114 | 0.0610 | 0.9801 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 2.390 | -19.7275 | -19.8627 | 0.1352 | 2.8068 | 0.0482 | 0.9839 | N/A | — | gap_masked(0.45) | refine_vqe |  |
| 2.500 | -20.5967 | -20.7091 | 0.1124 | 3.0223 | 0.0372 | 0.9872 | N/A | — | gap_masked(0.37) | refine_vqe |  |

## N = 10 (19 params)

**ΔE/gap: 0.6535 ± 0.8473 | P90=1.7876 | max=2.9162
|ΔE|/N: 3.85e-02
Fidelity: mean=0.9072 min=0.7290 (exact)
Distribution: [P25=0.058 | P50=0.173 | P75=0.965 | P90=1.788]
Regions: critical=1.6747 | ordered=0.0473**

**Fidelity: mean F=0.9072, min F=0.7290** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -11.8302 | -12.4722 | 0.6419 | 0.2201 | 2.9162 | 0.7290 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -12.6053 | -13.2894 | 0.6841 | 0.3632 | 1.8833 | 0.7772 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.210 | -13.2368 | -14.0836 | 0.8468 | 0.5151 | 1.6441 | 0.7832 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.320 | -14.1656 | -14.9990 | 0.8334 | 0.6978 | 1.1943 | 0.8150 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.430 | -15.2914 | -15.9469 | 0.6556 | 0.8912 | 0.7356 | 0.8633 | N/A | — | severe_error(0.72) | increase_p |  |
| 1.540 | -16.3163 | -16.9194 | 0.6031 | 1.0915 | 0.5525 | 0.8855 | N/A | — | severe_error(0.53) | increase_p |  |
| 1.640 | -17.4299 | -17.8199 | 0.3900 | 1.2777 | 0.3052 | 0.9278 | N/A | — | moderate_error(0.27) | refine_vqe |  |
| 1.750 | -18.5682 | -18.8250 | 0.2568 | 1.4858 | 0.1728 | 0.9541 | N/A | — | moderate_error(0.13) | refine_vqe |  |
| 1.860 | -19.6822 | -19.8422 | 0.1600 | 1.6962 | 0.0943 | 0.9720 | N/A | — | near_pass(0.05) | refine_vqe |  |
| 1.960 | -20.6491 | -20.7757 | 0.1265 | 1.8891 | 0.0670 | 0.9790 | N/A | — | near_pass(0.02) | refine_vqe |  |
| 2.070 | -21.6885 | -21.8104 | 0.1219 | 2.1026 | 0.0580 | 0.9815 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 2.180 | -22.7184 | -22.8521 | 0.1337 | 2.3171 | 0.0577 | 0.9815 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 2.290 | -23.7791 | -23.8998 | 0.1206 | 2.5324 | 0.0476 | 0.9844 | N/A | — | gap_masked(0.40) | refine_vqe |  |
| 2.390 | -24.7476 | -24.8565 | 0.1089 | 2.7287 | 0.0399 | 0.9867 | N/A | — | gap_masked(0.36) | refine_vqe |  |
| 2.500 | -25.8147 | -25.9132 | 0.0985 | 2.9452 | 0.0334 | 0.9886 | N/A | — | pass(0.67) | none |  |

## N = 20 (39 params)

**ΔE/gap: 3.6124 ± 1.9419 | P90=5.5746 | max=6.8584
|ΔE|/N: 6.33e-02
Var(H): 7.0281
Distribution: [P25=1.895 | P50=4.103 | P75=5.062 | P90=5.575]
Regions: critical=5.5200 | ordered=1.2360**

**Infidelity decomposition:** 15 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=55.0236.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -23.2149 | -25.3695 | 2.1546 | 0.3142 | 6.8584 | N/A | 6.4995 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -25.0769 | -26.9058 | 1.8288 | 0.3142 | 5.8214 | N/A | 6.4433 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.210 | -26.8022 | -28.4372 | 1.6350 | 0.3142 | 5.2045 | N/A | 6.4522 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.320 | -28.7124 | -30.2298 | 1.5175 | 0.3142 | 4.8303 | N/A | 6.6281 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.430 | -30.5636 | -32.0983 | 1.5347 | 0.3142 | 4.8852 | N/A | 7.3108 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.540 | -32.4376 | -34.0227 | 1.5852 | 0.3142 | 5.0457 | N/A | 8.2257 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.640 | -34.2134 | -35.8088 | 1.5954 | 0.3142 | 5.0783 | N/A | 8.9857 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -36.5165 | -37.8057 | 1.2891 | 0.3142 | 4.1034 | N/A | 8.0344 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.860 | -38.7902 | -39.8292 | 1.0390 | 0.3142 | 3.3074 | N/A | 7.0143 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.960 | -40.7856 | -41.6875 | 0.9019 | 0.3142 | 2.8709 | N/A | 6.4703 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.070 | -42.8867 | -43.7486 | 0.8618 | 0.3519 | 2.4490 | N/A | 6.5847 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.180 | -45.0582 | -45.8248 | 0.7666 | 0.5713 | 1.3418 | N/A | 6.2011 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.290 | -47.0771 | -47.9136 | 0.8365 | 0.7908 | 1.0578 | N/A | 7.2321 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.390 | -49.0527 | -49.8220 | 0.7692 | 0.9903 | 0.7768 | N/A | 6.9681 | dirty_state | severe_error(0.77) | increase_p |  |
| 2.500 | -51.2588 | -51.9300 | 0.6712 | 1.2099 | 0.5548 | N/A | 6.3712 | dirty_state | severe_error(0.53) | increase_p |  |
