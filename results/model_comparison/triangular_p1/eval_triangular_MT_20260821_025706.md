# Model Evaluation: triangular

> **🌐 MULTI-TOPOLOGY MODEL** — This evaluation uses a model trained on multiple topologies simultaneously. Results reflect cross-topology transfer capability.

**Date**: 2026-08-21 02:57 UTC
**Model**: unified_tfim_br_MT_residual+film_p1_v2.pt
**Multi-topology**: YES
**h-range**: [2.5, 5.0] (6 pts)
**Target N**: [10]

---

## N = 10 (30 params)

**Quality: F (score=0.02)
ΔE/gap: 0.6057 ± 0.9657 | P90=1.6482 | max=2.7245
|ΔE|/N=5.46e-02
Distribution: [P25=0.048 | P50=0.138 | P75=0.478 | P90=1.648]**

| h | E_pred | E_exact | |ΔE| | |ΔE|/N | gap | ΔE/gap | Category | Action | Note |
|---|--------|---------|------|--------|-----|--------|----------|--------|------|
| 2.500 | -27.0308 | -28.5006 | 1.4698 | 1.47e-01 | 0.5395 | 2.7245 | ansatz_limited(1.00) | restrict_h_range |  |
| 3.000 | -31.7407 | -32.5148 | 0.7741 | 7.74e-02 | 1.3536 | 0.5719 | severe_error(0.55) | increase_p |  |
| 3.500 | -36.5013 | -36.9642 | 0.4629 | 4.63e-02 | 2.3486 | 0.1971 | moderate_error(0.15) | refine_vqe |  |
| 4.000 | -41.3506 | -41.6201 | 0.2695 | 2.69e-02 | 3.3919 | 0.0795 | near_pass(0.03) | refine_vqe |  |
| 4.500 | -46.2145 | -46.3833 | 0.1688 | 1.69e-02 | 4.4416 | 0.0380 | gap_masked(0.56) | refine_vqe |  |
| 5.000 | -51.0812 | -51.2093 | 0.1281 | 1.28e-02 | 5.4873 | 0.0233 | gap_masked(0.43) | refine_vqe |  |
