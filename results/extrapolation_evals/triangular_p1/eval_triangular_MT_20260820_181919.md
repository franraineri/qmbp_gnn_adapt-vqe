# Model Evaluation: triangular

> **🌐 MULTI-TOPOLOGY MODEL** — This evaluation uses a model trained on multiple topologies simultaneously. Results reflect cross-topology transfer capability.

**Date**: 2026-08-20 18:19 UTC
**Model**: data/model_zoo/checkpoints/unified_tfim_br_MT_residual+film_p1.pt
**Multi-topology**: YES
**h-range**: [2.5, 5.0] (6 pts)
**Target N**: [12, 16]

---

## N = 12 (38 params)

**Quality: F (score=0.01)
ΔE/gap: 2.9203 ± 5.0943 | P90=8.2237 | max=14.1921
|ΔE|/N=1.09e-01
Distribution: [P25=0.306 | P50=0.376 | P75=1.798 | P90=8.224]**

| h | E_pred | E_exact | |ΔE| | |ΔE|/N | gap | ΔE/gap | Category | Action | Note |
|---|--------|---------|------|--------|-----|--------|----------|--------|------|
| 2.500 | -32.2434 | -35.2650 | 3.0215 | 2.52e-01 | 0.2129 | 14.1921 | ansatz_limited(1.00) | restrict_h_range | cached |
| 3.000 | -37.8559 | -39.6368 | 1.7809 | 1.48e-01 | 0.7897 | 2.2553 | ansatz_limited(1.00) | restrict_h_range | cached |
| 3.500 | -44.2261 | -44.7352 | 0.5091 | 4.24e-02 | 1.7048 | 0.2986 | moderate_error(0.26) | refine_vqe | cached |
| 4.000 | -49.0401 | -50.2071 | 1.1670 | 9.72e-02 | 2.7465 | 0.4249 | moderate_error(0.39) | refine_vqe | cached |
| 4.500 | -54.6121 | -55.8612 | 1.2491 | 1.04e-01 | 3.8147 | 0.3274 | moderate_error(0.29) | refine_vqe | cached |
| 5.000 | -61.4998 | -61.6148 | 0.1149 | 9.58e-03 | 4.8809 | 0.0235 | gap_masked(0.38) | refine_vqe | cached |

## N = 16 (54 params)

**Quality: F (score=0.00)
ΔE/gap: 47.6676 ± 93.9980 | P90=140.3443 | max=257.0578
|ΔE|/N=2.50e-01
Distribution: [P25=1.384 | P50=1.970 | P75=18.315 | P90=140.344]**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=257.058 is 5× the mean — median may be more representative N=16

| h | E_pred | E_exact | |ΔE| | |ΔE|/N | gap | ΔE/gap | Category | Action | Note |
|---|--------|---------|------|--------|-----|--------|----------|--------|------|
| 2.500 | -41.3524 | -49.1943 | 7.8419 | 4.90e-01 | 0.0305 | 257.0578 | ansatz_limited(1.00) | restrict_h_range | cached |
| 3.000 | -48.5893 | -54.2623 | 5.6730 | 3.55e-01 | 0.2401 | 23.6307 | ansatz_limited(1.00) | restrict_h_range | cached |
| 3.500 | -59.2762 | -60.4450 | 1.1688 | 7.31e-02 | 0.8846 | 1.3213 | severe_error(1.00) | increase_p | cached |
| 4.000 | -62.9976 | -67.4413 | 4.4438 | 2.78e-01 | 1.8752 | 2.3698 | ansatz_limited(1.00) | restrict_h_range | cached |
| 4.500 | -70.1711 | -74.8405 | 4.6694 | 2.92e-01 | 2.9731 | 1.5706 | severe_error(1.00) | increase_p | cached |
| 5.000 | -82.2105 | -82.4360 | 0.2256 | 1.41e-02 | 4.0812 | 0.0553 | near_pass(0.01) | refine_vqe | cached |
