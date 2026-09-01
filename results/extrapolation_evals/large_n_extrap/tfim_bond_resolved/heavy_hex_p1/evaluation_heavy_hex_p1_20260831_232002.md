# Model Evaluation: heavy_hex

**Date**: 2026-08-31 23:20 UTC
**Model**: data/model_zoo/checkpoints/unified_tfim_br_heavy_hex_fromMT_4+6+10+12+16+20_p1.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 2.0] (10 pts)
**Target N**: [10, 20]

### MPNN vs Random VQE Comparison

| N | MPNN |ΔE| | VQE |ΔE| | MPNN ΔE/gap | VQE ΔE/gap | Speedup (evals) | MPNN win rate |
|---|---------|---------|-------------|------------|-----------------|---------------|
| 10 | 11.8957 | 2.7794 | 16.0239 | 4.0973 | 1756× | 0% |
| 20 | 21.7481 | 4.0879 | 69.2263 | 13.0123 | 3816× | 0% |
| **avg** | | | | | **2786×** | |

---

## N = 10 (19 params)

**ΔE/gap: 16.0239 ± 10.2747 | P90=28.0316 | max=41.9143
|ΔE|/N: 1.19e+00
Fidelity: mean=0.0031 min=0.0006 (exact)
Distribution: [P25=9.062 | P50=11.826 | P75=18.048 | P90=28.032]
Regions: critical=23.0576**

**Fidelity: mean F=0.0031, min F=0.0006** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -3.2456 | -12.4722 | 9.2266 | 0.2201 | 41.9143 | 0.0007 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -3.6678 | -13.2894 | 9.6216 | 0.3632 | 26.4891 | 0.0006 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.220 | -4.0679 | -14.1652 | 10.0973 | 0.5311 | 19.0122 | 0.0009 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.330 | -4.2473 | -15.0840 | 10.8367 | 0.7150 | 15.1565 | 0.0014 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.440 | -4.4739 | -16.0344 | 11.5605 | 0.9091 | 12.7161 | 0.0019 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.560 | -4.7568 | -17.0983 | 12.3415 | 1.1285 | 10.9363 | 0.0026 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.670 | -5.0688 | -18.0927 | 13.0238 | 1.3342 | 9.7615 | 0.0035 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.780 | -5.4786 | -19.1013 | 13.6226 | 1.5430 | 8.8288 | 0.0046 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 1.890 | -6.0379 | -20.1214 | 14.0835 | 1.7540 | 8.0296 | 0.0062 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.000 | -6.6086 | -21.1510 | 14.5424 | 1.9666 | 7.3948 | 0.0081 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |

## N = 20 (39 params)

**ΔE/gap: 69.2263 ± 9.9060 | P90=82.3588 | max=86.8940
|ΔE|/N: 1.09e+00
Var(H): 54.5258
Distribution: [P25=61.414 | P50=67.862 | P75=76.106 | P90=82.359]
Regions: critical=60.7074**

**Infidelity decomposition:** 10 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=552.4618.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -7.9458 | -25.3695 | 17.4237 | 0.3142 | 55.4614 | N/A | 38.6579 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.110 | -8.6714 | -26.9058 | 18.2344 | 0.3142 | 58.0419 | N/A | 42.1202 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.220 | -9.4863 | -28.5975 | 19.1112 | 0.3142 | 60.8328 | N/A | 45.1310 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.330 | -10.5555 | -30.3969 | 19.8414 | 0.3142 | 63.1572 | N/A | 48.1794 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.440 | -11.5229 | -32.2712 | 20.7482 | 0.3142 | 66.0437 | N/A | 51.4945 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.560 | -12.4870 | -34.3775 | 21.8905 | 0.3142 | 69.6798 | N/A | 55.4784 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.670 | -13.3432 | -36.3507 | 23.0075 | 0.3142 | 73.2353 | N/A | 59.3006 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.780 | -14.1454 | -38.3553 | 24.2099 | 0.3142 | 77.0624 | N/A | 63.6479 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.890 | -14.6694 | -40.3849 | 25.7155 | 0.3142 | 81.8549 | N/A | 68.3291 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 2.000 | -15.1364 | -42.4350 | 27.2986 | 0.3142 | 86.8940 | N/A | 72.9190 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
