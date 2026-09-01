# Model Evaluation: chain_1d

**Date**: 2026-09-01 00:03 UTC
**Model**: unified_tfim_frustrated_chain_1d_n8_p1_20260831T205934.pt
**p_layers**: 1
**Multi-topology**: no
**h-range**: [1.0, 3.5] (11 pts)
**Target N**: [6, 10, 20]

---

## N = 6 (11 params)

**ΔE/gap: 0.2729 ± 0.2571 | P90=0.5704 | max=0.9401
|ΔE|/N: 1.00e-01
Fidelity: mean=0.9010 min=0.6820 (exact)
Distribution: [P25=0.097 | P50=0.162 | P75=0.329 | P90=0.570]
Regions: critical=0.7553 | ordered=0.1043**

**Fidelity: mean F=0.9010, min F=0.6820** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -6.0001 | -7.0074 | 1.0072 | 1.0714 | 0.9401 | 0.6820 | N/A | — | severe_error(0.94) | increase_p |  |
| 1.250 | -7.5001 | -8.3616 | 0.8615 | 1.5102 | 0.5704 | 0.7959 | N/A | — | severe_error(0.55) | increase_p |  |
| 1.500 | -9.0001 | -9.7529 | 0.7527 | 1.9653 | 0.3830 | 0.8589 | N/A | — | moderate_error(0.35) | refine_vqe |  |
| 1.750 | -10.5001 | -11.1686 | 0.6685 | 2.4308 | 0.2750 | 0.8967 | N/A | — | moderate_error(0.24) | refine_vqe |  |
| 2.000 | -12.0001 | -12.6014 | 0.6013 | 2.9035 | 0.2071 | 0.9212 | N/A | — | moderate_error(0.17) | refine_vqe |  |
| 2.250 | -13.5001 | -14.0465 | 0.5463 | 3.3813 | 0.1616 | 0.9378 | N/A | — | moderate_error(0.12) | refine_vqe |  |
| 2.500 | -15.0001 | -15.5008 | 0.5006 | 3.8629 | 0.1296 | 0.9497 | N/A | — | moderate_error(0.08) | refine_vqe |  |
| 2.750 | -16.5001 | -16.9621 | 0.4620 | 4.3473 | 0.1063 | 0.9585 | N/A | — | moderate_error(0.06) | refine_vqe |  |
| 3.000 | -18.0001 | -18.4290 | 0.4289 | 4.8341 | 0.0887 | 0.9651 | N/A | — | near_pass(0.04) | refine_vqe |  |
| 3.250 | -19.5001 | -19.9004 | 0.4002 | 5.3226 | 0.0752 | 0.9703 | N/A | — | near_pass(0.03) | refine_vqe |  |
| 3.500 | -21.0001 | -21.3753 | 0.3752 | 5.8126 | 0.0645 | 0.9744 | N/A | — | near_pass(0.02) | refine_vqe |  |

## N = 10 (19 params)

**ΔE/gap: 0.4972 ± 0.4671 | P90=1.0390 | max=1.7081
|ΔE|/N: 1.08e-01
Fidelity: mean=0.8288 min=0.4655 (exact)
Distribution: [P25=0.178 | P50=0.295 | P75=0.600 | P90=1.039]
Regions: critical=1.3735 | ordered=0.1906**

**Fidelity: mean F=0.8288, min F=0.4655** (exact statevector overlap)

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -10.0005 | -11.7997 | 1.7992 | 1.0534 | 1.7081 | 0.4655 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.250 | -12.5005 | -14.0447 | 1.5442 | 1.4862 | 1.0390 | 0.6503 | N/A | — | severe_error(1.00) | increase_p |  |
| 1.500 | -15.0005 | -16.3533 | 1.3528 | 1.9367 | 0.6985 | 0.7551 | N/A | — | severe_error(0.68) | increase_p |  |
| 1.750 | -17.5005 | -18.7043 | 1.2038 | 2.3985 | 0.5019 | 0.8192 | N/A | — | severe_error(0.48) | increase_p |  |
| 2.000 | -20.0005 | -21.0850 | 1.0844 | 2.8681 | 0.3781 | 0.8611 | N/A | — | moderate_error(0.35) | refine_vqe |  |
| 2.250 | -22.5005 | -23.4872 | 0.9867 | 3.3433 | 0.2951 | 0.8900 | N/A | — | moderate_error(0.26) | refine_vqe |  |
| 2.500 | -25.0005 | -25.9057 | 0.9052 | 3.8227 | 0.2368 | 0.9107 | N/A | — | moderate_error(0.20) | refine_vqe |  |
| 2.750 | -27.5005 | -28.3366 | 0.8361 | 4.3052 | 0.1942 | 0.9261 | N/A | — | moderate_error(0.15) | refine_vqe |  |
| 3.000 | -30.0005 | -30.7773 | 0.7768 | 4.7903 | 0.1622 | 0.9379 | N/A | — | moderate_error(0.12) | refine_vqe |  |
| 3.250 | -32.5005 | -33.2259 | 0.7254 | 5.2773 | 0.1375 | 0.9470 | N/A | — | moderate_error(0.09) | refine_vqe |  |
| 3.500 | -35.0005 | -35.6809 | 0.6804 | 5.7660 | 0.1180 | 0.9543 | N/A | — | moderate_error(0.07) | refine_vqe |  |

## N = 20 (39 params)

**ΔE/gap: 3.0067 ± 4.6663 | P90=7.6182 | max=16.2541
|ΔE|/N: 1.26e-01
Fidelity: mean≥0.1544 min≥0.0658 (variance lower bound, 11 pts)
Var(H): 18.9872
Distribution: [P25=0.446 | P50=0.850 | P75=2.514 | P90=7.618]
Regions: critical=11.9362 | ordered=0.4962**

> **Metric Reliability Warnings:**
> - ⚠️ Outlier: max ΔE/gap=16.254 is 5× the mean — median may be more representative N=20

**Fidelity: mean F≥0.1544, min F≥0.0658** (variance lower bound — N>16, exact statevector infeasible)
> 11/11 points use the Eckart bound F ≥ 1 − Var(H)/gap², mean Var(H)=18.9872.

**Infidelity decomposition:** 11 dirty-state (attackable via optimization), 0 small-gap (physics ceiling near h_c), mean Var(H)/gap²=27.6036.

| h | E_pred | E_exact | |ΔE| | gap | ΔE/gap | Fidelity | Var(H) | Factor | Category | Action | Note |
|---|--------|---------|------|--------|-----|----------|--------|--------|----------|--------|------|
| 1.000 | -20.0014 | -25.1078 | 5.1064 | 0.3142 | 16.2541 | N/A | 18.9943 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.250 | -25.0014 | -28.9609 | 3.9595 | 0.5197 | 7.6182 | N/A | 18.9929 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.500 | -30.0014 | -33.2545 | 3.2531 | 1.0164 | 3.2004 | N/A | 18.9915 | dirty_state | ansatz_limited(1.00) | restrict_h_range |  |
| 1.750 | -35.0014 | -37.7679 | 2.7665 | 1.5141 | 1.8271 | N/A | 18.9900 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.000 | -40.0014 | -42.4102 | 2.4088 | 2.0123 | 1.1970 | N/A | 18.9886 | dirty_state | severe_error(1.00) | increase_p |  |
| 2.250 | -45.0014 | -47.1355 | 2.1341 | 2.5110 | 0.8499 | N/A | 18.9872 | dirty_state | severe_error(0.84) | increase_p |  |
| 2.500 | -50.0014 | -51.9176 | 1.9162 | 3.0099 | 0.6366 | N/A | 18.9858 | dirty_state | severe_error(0.62) | increase_p |  |
| 2.750 | -55.0014 | -56.7404 | 1.7390 | 3.5090 | 0.4956 | N/A | 18.9844 | dirty_state | moderate_error(0.47) | refine_vqe |  |
| 3.000 | -60.0014 | -61.5934 | 1.5920 | 4.0082 | 0.3972 | N/A | 18.9830 | dirty_state | moderate_error(0.37) | refine_vqe |  |
| 3.250 | -65.0014 | -66.4694 | 1.4680 | 4.5076 | 0.3257 | ≥0.0658 | 18.9815 | dirty_state | moderate_error(0.29) | refine_vqe |  |
| 3.500 | -70.0014 | -71.3635 | 1.3620 | 5.0070 | 0.2720 | ≥0.2429 | 18.9801 | dirty_state | moderate_error(0.23) | refine_vqe |  |
