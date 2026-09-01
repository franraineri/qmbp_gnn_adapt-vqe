# Model Evaluation: heavy_hex

**Date**: 2026-08-31 23:07 UTC
**Model**: data/model_zoo/checkpoints/unified_tfim_br_heavy_hex_multiN_4+6+8+10+12+14+18+20+22+24+30+32+40_p1.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 2.0] (10 pts)
**Target N**: [10, 20]

### MPNN vs Random VQE Comparison

| N | MPNN |ΔE| | VQE |ΔE| | MPNN ΔE/gap | VQE ΔE/gap | Speedup (evals) | MPNN win rate |
|---|---------|---------|-------------|------------|-----------------|---------------|
| 10 | 0.5053 | 2.7794 | 0.9329 | 4.0973 | 1756× | 80% |
| 20 | 1.4917 | 4.0879 | 4.7482 | 13.0123 | 3816× | 100% |
| **avg** | | | | | **2786×** | |

---

## N = 10 (19 params)

**ΔE/gap: 0.9329 ± 0.9034 | P90=1.9866 | max=2.9162
|ΔE|/N: 5.05e-02
Fidelity: mean=0.8721 min=0.7290 (exact)
Distribution: [P25=0.176 | P50=0.607 | P75=1.509 | P90=1.987]
Regions: critical=1.6578**

**Fidelity: mean F=0.8721, min F=0.7290** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -11.8302 | -12.4722 | 0.6419 | 0.2201 | 2.9162 | 0.7290 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -12.6053 | -13.2894 | 0.6841 | 0.3632 | 1.8833 | 0.7772 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.220 | -13.2959 | -14.1652 | 0.8694 | 0.5311 | 1.6369 | 0.7822 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.330 | -14.2795 | -15.0840 | 0.8045 | 0.7150 | 1.1252 | 0.8220 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.440 | -15.3733 | -16.0344 | 0.6611 | 0.9091 | 0.7272 | 0.8638 | N/A | — | severe_error(0.71) | increase_p |  |
| 1.560 | -16.5480 | -17.0983 | 0.5503 | 1.1285 | 0.4877 | 0.8960 | N/A | — | moderate_error(0.46) | refine_vqe |  |
| 1.670 | -17.7441 | -18.0927 | 0.3486 | 1.3342 | 0.2613 | 0.9361 | N/A | — | moderate_error(0.22) | refine_vqe |  |
| 1.780 | -18.8726 | -19.1013 | 0.2287 | 1.5430 | 0.1482 | 0.9595 | N/A | — | moderate_error(0.10) | refine_vqe |  |
| 1.890 | -19.9795 | -20.1214 | 0.1419 | 1.7540 | 0.0809 | 0.9753 | N/A | — | near_pass(0.03) | refine_vqe |  |
| 2.000 | -21.0287 | -21.1510 | 0.1224 | 1.9666 | 0.0622 | 0.9803 | N/A | — | near_pass(0.01) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 4.7482 ± 1.1335 | P90=5.9251 | max=6.8584
|ΔE|/N: 7.46e-02
Var(H): 7.2051
Distribution: [P25=4.063 | P50=4.911 | P75=5.158 | P90=5.925]
Regions: critical=5.5130**

**Infidelity decomposition:** 10 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=73.0027.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -23.2149 | -25.3695 | 2.1546 | 0.3142 | 6.8584 | N/A | 6.4995 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -25.0769 | -26.9058 | 1.8288 | 0.3142 | 5.8214 | N/A | 6.4433 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.220 | -26.9743 | -28.5975 | 1.6233 | 0.3142 | 5.1670 | N/A | 6.4627 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.330 | -28.8829 | -30.3969 | 1.5140 | 0.3142 | 4.8191 | N/A | 6.6676 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.440 | -30.7321 | -32.2712 | 1.5391 | 0.3142 | 4.8990 | N/A | 7.3892 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.560 | -32.7650 | -34.3775 | 1.6125 | 0.3142 | 5.1327 | N/A | 8.4853 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.670 | -34.8040 | -36.3507 | 1.5467 | 0.3142 | 4.9231 | N/A | 8.9550 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.780 | -37.1582 | -38.3553 | 1.1971 | 0.3142 | 3.8105 | N/A | 7.6371 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.890 | -39.3873 | -40.3849 | 0.9976 | 0.3142 | 3.1753 | N/A | 6.8687 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.000 | -41.5317 | -42.4350 | 0.9033 | 0.3142 | 2.8751 | N/A | 6.6423 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
