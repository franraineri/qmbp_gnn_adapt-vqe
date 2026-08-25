# Model Evaluation: triangular

> **🌐 MULTI-TOPOLOGY MODEL** — This evaluation uses a model trained on multiple topologies simultaneously. Results reflect cross-topology transfer capability.

**Date**: 2026-08-20 14:14 UTC
**Model**: unified_tfim_br_MT_residual+film_p1.pt
**Multi-topology**: YES
**h-range**: [2.5, 5.0] (6 pts)
**Target N**: [10]

---

## N = 10 (30 params)

**Quality: F (score=0.01)
ΔE/gap: 0.7657 ± 1.1591 | P90=2.0174 | max=3.3095
|ΔE|/N=7.74e-02
Distribution: [P25=0.101 | P50=0.197 | P75=0.609 | P90=2.017]**

| h | E_pred | E_exact | |ΔE| | |ΔE|/N | gap | ΔE/gap | Category | Action | Note |
|---|--------|---------|------|--------|-----|--------|----------|--------|------|
| 2.500 | -26.7151 | -28.5006 | 1.7854 | 1.79e-01 | 0.5395 | 3.3095 | ansatz_limited(1.00) | restrict_h_range |  |
| 3.000 | -31.5329 | -32.5148 | 0.9819 | 9.82e-02 | 1.3536 | 0.7254 | severe_error(0.71) | increase_p |  |
| 3.500 | -36.3503 | -36.9642 | 0.6139 | 6.14e-02 | 2.3486 | 0.2614 | moderate_error(0.22) | refine_vqe |  |
| 4.000 | -41.1672 | -41.6201 | 0.4529 | 4.53e-02 | 3.3919 | 0.1335 | moderate_error(0.09) | refine_vqe |  |
| 4.500 | -45.9837 | -46.3833 | 0.3996 | 4.00e-02 | 4.4416 | 0.0900 | near_pass(0.04) | refine_vqe |  |
| 5.000 | -50.7997 | -51.2093 | 0.4095 | 4.10e-02 | 5.4873 | 0.0746 | near_pass(0.03) | refine_vqe |  |
