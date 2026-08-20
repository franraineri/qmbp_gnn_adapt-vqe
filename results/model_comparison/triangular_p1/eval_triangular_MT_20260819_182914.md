# Model Evaluation: triangular

> **🌐 MULTI-TOPOLOGY MODEL** — This evaluation uses a model trained on multiple topologies simultaneously. Results reflect cross-topology transfer capability.

**Date**: 2026-08-19 18:29 UTC
**Model**: unified_tfim_br_multitopo_chain_1d+heavy_hex+ladder+square+triangular_maxN20_residual+film_p1.pt
**Multi-topology**: YES
**h-range**: [1.5, 5.0] (15 pts)
**Target N**: [16]

---

## N = 16 (54 params)

**Quality: F (score=0.00)
ΔE/gap: 53057.9728 ± 181757.4829 | P90=36745.6699 | max=731083.3721
|ΔE|/N=4.73e-01
Distribution: [P25=2.419 | P50=9.512 | P75=630.905 | P90=36745.670]**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=731083.372 is 14× the mean — median may be more representative N=16

| h | E_pred | E_exact | |ΔE| | |ΔE|/N | gap | ΔE/gap | Category | Action | Note |
|---|--------|---------|------|--------|-----|--------|----------|--------|------|
| 1.500 | -23.2100 | -42.0156 | 18.8056 | 1.18e+00 | 0.0000 | 731083.3721 | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -29.5381 | -43.4676 | 13.9295 | 8.71e-01 | 0.0002 | 56898.7457 | ansatz_limited(1.00) | restrict_h_range |  |
| 2.000 | -34.6349 | -45.1451 | 10.5103 | 6.57e-01 | 0.0016 | 6516.0562 | ansatz_limited(1.00) | restrict_h_range |  |
| 2.250 | -38.7666 | -47.0514 | 8.2848 | 5.18e-01 | 0.0079 | 1044.9660 | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -42.5792 | -49.1943 | 6.6151 | 4.13e-01 | 0.0305 | 216.8432 | ansatz_limited(1.00) | restrict_h_range |  |
| 2.750 | -45.9493 | -51.5902 | 5.6409 | 3.53e-01 | 0.0946 | 59.6163 | ansatz_limited(1.00) | restrict_h_range |  |
| 3.000 | -49.1829 | -54.2623 | 5.0794 | 3.17e-01 | 0.2401 | 21.1580 | ansatz_limited(1.00) | restrict_h_range |  |
| 3.250 | -52.4417 | -57.2223 | 4.7806 | 2.99e-01 | 0.5026 | 9.5119 | ansatz_limited(1.00) | restrict_h_range |  |
| 3.500 | -55.7818 | -60.4450 | 4.6632 | 2.91e-01 | 0.8846 | 5.2717 | ansatz_limited(1.00) | restrict_h_range |  |
| 3.750 | -59.1454 | -63.8718 | 4.7264 | 2.95e-01 | 1.3546 | 3.4891 | ansatz_limited(1.00) | restrict_h_range |  |
| 4.000 | -62.4700 | -67.4413 | 4.9713 | 3.11e-01 | 1.8752 | 2.6512 | ansatz_limited(1.00) | restrict_h_range |  |
| 4.250 | -65.8163 | -71.1076 | 5.2913 | 3.31e-01 | 2.4196 | 2.1868 | ansatz_limited(1.00) | restrict_h_range |  |
| 4.500 | -69.0557 | -74.8405 | 5.7848 | 3.62e-01 | 2.9731 | 1.9457 | severe_error(1.00) | increase_p |  |
| 4.750 | -71.9992 | -78.6208 | 6.6216 | 4.14e-01 | 3.5281 | 1.8768 | severe_error(1.00) | increase_p |  |
| 5.000 | -74.6735 | -82.4360 | 7.7625 | 4.85e-01 | 4.0812 | 1.9020 | severe_error(1.00) | increase_p |  |
