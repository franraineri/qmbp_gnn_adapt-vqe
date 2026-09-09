# Model Evaluation: heavy_hex

**Date**: 2026-09-09 14:18 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_signinv_fid_v1.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [2.5, 5.0] (10 pts)
**Target N**: [16, 20, 30]

### MPNN vs Random VQE Comparison

| N | MPNN |ΔE| | VQE |ΔE| | MPNN ΔE/gap | VQE ΔE/gap | Speedup (evals) | MPNN win rate |
|---|---------|---------|-------------|------------|-----------------|---------------|
| 16 | 2.4185 | 12.8259 | 0.5151 | 2.4393 | 3597× | 100% |
| 20 | 3.0460 | 12.7572 | 1.1000 | 3.8441 | 4320× | 100% |
| 30 | 4.7698 | 13.2082 | 1.7692 | 4.1453 | 6936× | 100% |
| **avg** | | | | | **4951×** | |

---

## N = 16 (31 params)

**ΔE/gap: 0.5151 ± 0.2169 | P90=0.8017 | max=0.9779
|ΔE|/N: 1.51e-01
Fidelity: mean=0.8410 min=0.7515 (exact)
Distribution: [P25=0.347 | P50=0.452 | P75=0.624 | P90=0.802]**

**Fidelity: mean F=0.8410, min F=0.7515** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 2.500 | -38.7454 | -41.5236 | 2.7783 | 2.8412 | 0.9779 | 0.7515 | N/A | — | severe_error(0.98) | increase_p |  |
| 2.780 | -43.1883 | -45.8460 | 2.6577 | 3.3978 | 0.7822 | 0.7835 | N/A | — | severe_error(0.77) | increase_p |  |
| 3.060 | -47.6354 | -50.1983 | 2.5629 | 3.9554 | 0.6479 | 0.8084 | N/A | — | severe_error(0.63) | increase_p |  |
| 3.330 | -51.9264 | -54.4160 | 2.4896 | 4.4938 | 0.5540 | 0.8276 | N/A | — | severe_error(0.53) | increase_p |  |
| 3.610 | -56.3793 | -58.8065 | 2.4272 | 5.0524 | 0.4804 | 0.8438 | N/A | — | moderate_error(0.45) | refine_vqe |  |
| 3.890 | -60.8379 | -63.2102 | 2.3722 | 5.6114 | 0.4228 | 0.8575 | N/A | — | moderate_error(0.39) | refine_vqe |  |
| 4.170 | -65.3067 | -67.6243 | 2.3175 | 6.1705 | 0.3756 | 0.8694 | N/A | — | moderate_error(0.34) | refine_vqe |  |
| 4.440 | -69.6287 | -71.8887 | 2.2600 | 6.7099 | 0.3368 | 0.8799 | N/A | — | moderate_error(0.30) | refine_vqe |  |
| 4.720 | -74.1206 | -76.3179 | 2.1973 | 7.2693 | 0.3023 | 0.8896 | N/A | — | moderate_error(0.27) | refine_vqe |  |
| 5.000 | -78.6305 | -80.7529 | 2.1224 | 7.8289 | 0.2711 | 0.8989 | N/A | — | moderate_error(0.23) | refine_vqe |  |

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

## N = 30 (59 params)

**ΔE/gap: 1.7692 ± 1.1888 | P90=3.1865 | max=4.7036
|ΔE|/N: 1.59e-01
Var(H): 70.2387
Distribution: [P25=0.934 | P50=1.294 | P75=2.075 | P90=3.186]**

**Infidelity decomposition:** 10 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=10.3307.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 2.500 | -72.5767 | -77.9281 | 5.3514 | 1.1377 | 4.7036 | N/A | 53.0661 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.780 | -80.9060 | -86.0282 | 5.1222 | 1.6973 | 3.0179 | N/A | 56.5299 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 3.060 | -89.2352 | -94.1844 | 4.9492 | 2.2569 | 2.1929 | N/A | 60.1514 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 3.330 | -97.2715 | -102.0888 | 4.8173 | 2.7966 | 1.7225 | N/A | 63.7337 | dirty_state | severe_error(1.00) | increase_p |  |
| 3.610 | -105.6055 | -110.3174 | 4.7120 | 3.3564 | 1.4039 | N/A | 67.5940 | dirty_state | severe_error(1.00) | increase_p |  |
| 3.890 | -113.9371 | -118.5710 | 4.6340 | 3.9162 | 1.1833 | N/A | 71.6366 | dirty_state | severe_error(1.00) | increase_p |  |
| 4.170 | -122.2682 | -126.8445 | 4.5763 | 4.4760 | 1.0224 | N/A | 75.8392 | dirty_state | severe_error(1.00) | increase_p |  |
| 4.440 | -130.3019 | -134.8378 | 4.5358 | 5.0158 | 0.9043 | N/A | 80.0340 | dirty_state | severe_error(0.90) | increase_p |  |
| 4.720 | -138.6304 | -143.1401 | 4.5096 | 5.5757 | 0.8088 | N/A | 84.5856 | dirty_state | severe_error(0.80) | increase_p |  |
| 5.000 | -146.9630 | -151.4534 | 4.4904 | 6.1355 | 0.7319 | N/A | 89.2171 | dirty_state | severe_error(0.72) | increase_p |  |
