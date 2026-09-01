# Model Evaluation: chain_1d

**Date**: 2026-08-31 23:03 UTC
**Model**: unified_tfim_br_chain_1d_multiN_6+8+10+12+15+16+20+26+60_p1.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 4.0] (3 pts)
**Target N**: [8, 10, 12, 16, 20]

---

## N = 8 (15 params)

**ΔE/gap: 1.0213 ± 1.4094 | P90=2.4203 | max=3.0144
|ΔE|/N: 5.36e-02
Fidelity: mean=0.8882 min=0.6823 (exact)
Distribution: [P25=0.025 | P50=0.044 | P75=1.529 | P90=2.420]
Regions: critical=3.0144 | ordered=0.0248**

> **Metric Reliability Warnings:**
> - ⚠️ Only 3 points — means have low statistical confidence N=8

**Fidelity: mean F=0.8882, min F=0.6823** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -8.7254 | -9.8380 | 1.1125 | 0.3691 | 3.0144 | 0.6823 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -20.5652 | -20.7051 | 0.1399 | 3.1711 | 0.0441 | 0.9845 | N/A | — | gap_masked(0.47) | refine_vqe |  |
| 4.000 | -32.4050 | -32.4387 | 0.0338 | 6.1482 | 0.0055 | 0.9978 | N/A | — | pass(0.11) | none |  |

## N = 10 (19 params)

**ΔE/gap: 1.6376 ± 2.2699 | P90=3.8897 | max=4.8476
|ΔE|/N: 5.58e-02
Fidelity: mean=0.8570 min=0.5939 (exact)
Distribution: [P25=0.033 | P50=0.058 | P75=2.453 | P90=3.890]
Regions: critical=4.8476 | ordered=0.0326**

> **Metric Reliability Warnings:**
> - ⚠️ Only 3 points — means have low statistical confidence N=10

**Fidelity: mean F=0.8570, min F=0.5939** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -10.9324 | -12.3815 | 1.4491 | 0.2989 | 4.8476 | 0.5939 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -25.7265 | -25.9072 | 0.1807 | 3.1184 | 0.0580 | 0.9800 | N/A | — | near_pass(0.01) | refine_vqe |  |
| 4.000 | -40.5205 | -40.5642 | 0.0437 | 6.1011 | 0.0072 | 0.9971 | N/A | — | pass(0.14) | none |  |

## N = 12 (23 params)

**ΔE/gap: 2.3979 ± 3.3342 | P90=5.7048 | max=7.1130
|ΔE|/N: 5.73e-02
Fidelity: mean=0.8289 min=0.5148 (exact)
Distribution: [P25=0.040 | P50=0.072 | P75=3.592 | P90=5.705]
Regions: critical=7.1130 | ordered=0.0403**

> **Metric Reliability Warnings:**
> - ⚠️ Only 3 points — means have low statistical confidence N=12

**Fidelity: mean F=0.8289, min F=0.5148** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -13.1395 | -14.9260 | 1.7865 | 0.2512 | 7.1130 | 0.5148 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -30.8877 | -31.1093 | 0.2216 | 3.0867 | 0.0718 | 0.9754 | N/A | — | near_pass(0.02) | refine_vqe |  |
| 4.000 | -48.6360 | -48.6897 | 0.0537 | 6.0733 | 0.0088 | 0.9965 | N/A | — | pass(0.18) | none |  |

## N = 16 (31 params)

**ΔE/gap: 4.3506 ± 6.0739 | P90=10.3721 | max=12.9403
|ΔE|/N: 5.92e-02
Fidelity: mean=0.7817 min=0.3836 (exact)
Distribution: [P25=0.056 | P50=0.099 | P75=6.520 | P90=10.372]
Regions: critical=12.9403 | ordered=0.0558**

> **Metric Reliability Warnings:**
> - ⚠️ Only 3 points — means have low statistical confidence N=16

**Fidelity: mean F=0.7817, min F=0.3836** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -17.5535 | -20.0164 | 2.4629 | 0.1903 | 12.9403 | 0.3836 | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -41.2103 | -41.5135 | 0.3032 | 3.0522 | 0.0993 | 0.9664 | N/A | — | near_pass(0.05) | refine_vqe |  |
| 4.000 | -64.8670 | -64.9407 | 0.0737 | 6.0435 | 0.0122 | 0.9952 | N/A | — | pass(0.24) | none |  |

## N = 20 (39 params)

**ΔE/gap: 3.3798 ± 4.6785 | P90=8.0222 | max=9.9958
|ΔE|/N: 6.03e-02
Fidelity: mean≥0.7856 min≥0.6115 (variance lower bound, 2 pts)
Var(H): 2.4865
Distribution: [P25=0.072 | P50=0.128 | P75=5.062 | P90=8.022]
Regions: critical=9.9958 | ordered=0.0717**

> **Metric Reliability Warnings:**
> - ⚠️ Only 3 points — means have low statistical confidence N=20

**Fidelity: mean F≥0.7856, min F≥0.6115** (variance lower bound — N>16, exact statevector infeasible)
> 2/3 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=2.4865.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -21.9675 | -25.1078 | 3.1403 | 0.3142 | 9.9958 | N/A | N/A | — | ansatz_limited(1.00) | restrict_h_range |  |
| 2.500 | -51.5328 | -51.9176 | 0.3848 | 3.0099 | 0.1279 | ≥0.6115 | 3.5199 | — | moderate_error(0.08) | refine_vqe |  |
| 4.000 | -81.0981 | -81.1917 | 0.0936 | 6.0062 | 0.0156 | ≥0.9597 | 1.4531 | — | pass(0.31) | none |  |
