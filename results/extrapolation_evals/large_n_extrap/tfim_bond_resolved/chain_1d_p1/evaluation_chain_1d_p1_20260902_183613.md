# Model Evaluation: chain_1d

**Date**: 2026-09-02 18:36 UTC
**Model**: data/model_zoo/checkpoints/unifMPNN__chain_1d_p1_signinv_fid_dot5_v1.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 3.0] (1 pts)
**Target N**: [4]

---

## N = 4 (7 params)

**ΔE/gap: 3.6722 ± 0.0000 | P90=3.6722 | max=3.6722
|ΔE|/N: 6.38e-01
Fidelity: mean=0.4866 min=0.4866 (exact)
Distribution: [P25=3.672 | P50=3.672 | P75=3.672 | P90=3.672]
Regions: critical=3.6722**

> **Metric Reliability Warnings:**
> - ⚠️ Only 1 points — means have low statistical confidence N=4

**Fidelity: mean F=0.4866, min F=0.4866** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -2.2081 | -4.7588 | 2.5507 | 0.6946 | 3.6722 | 0.4866 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
