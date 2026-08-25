# Model Evaluation: triangular

> **🌐 MULTI-TOPOLOGY MODEL** — This evaluation uses a model trained on multiple topologies simultaneously. Results reflect cross-topology transfer capability.

**Date**: 2026-08-19 15:59 UTC
**Model**: unified_tfim_br_multitopo_chain_1d+heavy_hex+ladder+square+triangular_maxN20_residual+film_p1.pt
**Multi-topology**: YES
**h-range**: [2.5, 5.0] (3 pts)
**Target N**: [6]

---

## N = 6 (16 params)

**Quality: F (score=0.17)
ΔE/gap: 0.0849 ± 0.0989 | P90=0.1838 | max=0.2246
|ΔE|/N=2.63e-02
Distribution: [P25=0.015 | P50=0.021 | P75=0.123 | P90=0.184]**

> **Metric Reliability Warnings:**
> - ⚠️ Only 3 points — means have low statistical confidence N=6

| h | E_pred | E_exact | |ΔE| | |ΔE|/N | gap | ΔE/gap | Category | Action | Note |
|---|--------|---------|------|--------|-----|--------|----------|--------|------|
| 2.500 | -16.1524 | -16.4858 | 0.3334 | 5.56e-02 | 1.4844 | 0.2246 | moderate_error(0.18) | refine_vqe |  |
| 3.750 | -23.2579 | -23.3386 | 0.0807 | 1.34e-02 | 3.8950 | 0.0207 | pass(0.41) | none |  |
| 5.000 | -30.5254 | -30.5854 | 0.0600 | 1.00e-02 | 6.4188 | 0.0093 | pass(0.19) | none |  |
