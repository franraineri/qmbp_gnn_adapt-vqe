# Model Evaluation: heavy_hex

**Date**: 2026-09-22 17:45 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_signinv_fid_v1.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [2.5, 5.0] (10 pts)
**Target N**: [10, 20]

---

## N = 10 (19 params)

**ΔE/gap: 0.3016 ± 0.1270 | P90=0.4698 | max=0.5698
|ΔE|/N: 1.44e-01
Fidelity: mean=0.9013 min=0.8418 (exact)
Distribution: [P25=0.203 | P50=0.265 | P75=0.367 | P90=0.470]**

**Fidelity: mean F=0.9013, min F=0.8418** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 2.500 | -24.2349 | -25.9132 | 1.6783 | 2.9452 | 0.5698 | 0.8418 | N/A | — | severe_error(0.55) | increase_p |  |
| 2.780 | -27.0146 | -28.6189 | 1.6043 | 3.4980 | 0.4586 | 0.8632 | N/A | — | moderate_error(0.43) | refine_vqe |  |
| 3.060 | -29.7975 | -31.3425 | 1.5450 | 4.0527 | 0.3812 | 0.8797 | N/A | — | moderate_error(0.35) | refine_vqe |  |
| 3.330 | -32.4842 | -33.9812 | 1.4970 | 4.5888 | 0.3262 | 0.8925 | N/A | — | moderate_error(0.29) | refine_vqe |  |
| 3.610 | -35.2736 | -36.7276 | 1.4540 | 5.1455 | 0.2826 | 0.9033 | N/A | — | moderate_error(0.24) | refine_vqe |  |
| 3.890 | -38.0664 | -39.4819 | 1.4154 | 5.7029 | 0.2482 | 0.9124 | N/A | — | moderate_error(0.21) | refine_vqe |  |
| 4.170 | -40.8633 | -42.2424 | 1.3791 | 6.2607 | 0.2203 | 0.9201 | N/A | — | moderate_error(0.18) | refine_vqe |  |
| 4.440 | -43.5716 | -44.9091 | 1.3375 | 6.7989 | 0.1967 | 0.9271 | N/A | — | moderate_error(0.15) | refine_vqe |  |
| 4.720 | -46.3872 | -47.6786 | 1.2914 | 7.3574 | 0.1755 | 0.9336 | N/A | — | moderate_error(0.13) | refine_vqe |  |
| 5.000 | -49.2096 | -50.4516 | 1.2420 | 7.9160 | 0.1569 | 0.9396 | N/A | — | moderate_error(0.11) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 1.1000 ± 0.7299 | P90=1.9815 | max=2.8804
|ΔE|/N: 1.52e-01
Var(H): 44.7206
Distribution: [P25=0.583 | P50=0.816 | P75=1.305 | P90=1.981]**

**Infidelity decomposition:** 10 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=6.1535.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 2.500 | -48.4452 | -51.9300 | 3.4849 | 1.2099 | 2.8804 | N/A | 34.5474 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.780 | -54.0020 | -57.3303 | 3.3283 | 1.7689 | 1.8816 | N/A | 36.7447 | dirty_state | severe_error(1.00) | increase_p |  |
| 3.060 | -59.5612 | -62.7685 | 3.2073 | 2.3281 | 1.3777 | N/A | 39.0097 | dirty_state | severe_error(1.00) | increase_p |  |
| 3.330 | -64.9239 | -68.0389 | 3.1150 | 2.8674 | 1.0863 | N/A | 41.2524 | dirty_state | severe_error(1.00) | increase_p |  |
| 3.610 | -70.4875 | -73.5256 | 3.0380 | 3.4268 | 0.8865 | N/A | 43.6311 | dirty_state | severe_error(0.88) | increase_p |  |
| 3.890 | -76.0542 | -79.0289 | 2.9747 | 3.9863 | 0.7462 | N/A | 46.0434 | dirty_state | severe_error(0.73) | increase_p |  |
| 4.170 | -81.6274 | -84.5454 | 2.9180 | 4.5459 | 0.6419 | N/A | 48.4230 | dirty_state | severe_error(0.62) | increase_p |  |
| 4.440 | -87.0105 | -89.8751 | 2.8645 | 5.0856 | 0.5633 | N/A | 50.6193 | dirty_state | severe_error(0.54) | increase_p |  |
| 4.720 | -92.6100 | -95.4107 | 2.8007 | 5.6452 | 0.4961 | N/A | 52.6185 | dirty_state | moderate_error(0.47) | refine_vqe |  |
| 5.000 | -98.2250 | -100.9537 | 2.7286 | 6.2049 | 0.4398 | N/A | 54.3164 | dirty_state | moderate_error(0.41) | refine_vqe |  |
