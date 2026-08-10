# Comparison: QESEM (Qedma) vs PEA-ZNE (IBM Runtime)

---

## System Under Study (identical in both)

| Parameter | Value |
|-----------|-------|
| N qubits | 10 |
| Topology | heavy_hex |
| Model | TFIM |
| Ansatz | HVA p=1 |
| Native 2Q gate | CZ (median error ~0.18%) |

---

## 1. Error Mitigation Techniques

### QESEM (Qedma — Qiskit Function)

| Technique | Configuration |
|-----------|---------------|
| **Main method** | Quasi-probabilistic error mitigation (QESEM unbiased) |
| **Noise scaling** | 3 internal scale factors: 0.0 (extrapolated), 1.0 (hardware), 2.0 (amplified) |
| **Extrapolation** | Native QESEM (scale → 0) + heuristic exponential [1.0, 2.0] |
| **Dynamical Decoupling** | No (handled internally by Qedma protocol) |
| **Twirling** | Not explicit (incorporated in QESEM protocol) |
| **Readout Error Mitigation** | Included (implicit REM, reports `results_with_REM`) |
| **Layer Noise Learning** | No (uses Qedma's own noise model) |
| **Transpilation** | "standard" (handled by Qedma server-side) |
| **Optimization level** | Controlled by Qedma |

### PEA-ZNE (IBM Runtime Estimator)

| Technique | Configuration |
|-----------|---------------|
| **Main method** | PEA (Probabilistic Error Amplification) + ZNE extrapolation |
| **Noise factors** | [1.0, 1.5, 3.0] |
| **Extrapolation** | Exponential + Linear (dual, best R² wins) |
| **Dynamical Decoupling** | Yes — XpXm sequence, skip_reset_qubits=true |
| **Twirling** | Yes — active-circuit strategy, 48 randomizations × 192 shots/rand |
| **Readout Error Mitigation** | measure_mitigation = true (M3/TREX) |
| **Layer Noise Learning** | Yes — 48 randomizations × 192 shots/rand |
| **Transpilation** | User-side (Qiskit transpile, optimization_level=2) |
| **Layout selection** | Mapomatic VF2, lowest_cost strategy, CES-aware |
| **Affine correction** | Post-ZNE affine linear correction (optional) |

---

## 2. Shots and Circuit Configuration

### QESEM

| Parameter | Tier-0 (h=4.0, job 82aa) | Tier-1 (h=4.0, job 4f16) | Tier-1 (h=3.5, job d628) |
|-----------|--------------------------|--------------------------|--------------------------|
| **Total shots** | 756,048 | 772,848 | 779,448 |
| **Mitigation shots** | 266,000 (35%) | 282,800 (37%) | 289,400 (37%) |
| **Signal shots** | ~490,048 (65%) | ~490,048 (63%) | ~490,048 (63%) |
| **N observables** | 20 | 20 | 20 |
| **Precision target** | 0.01 | 0.01 | 0.01 |
| **N layouts (circuits)** | 1 | 1 | 1 |
| **Circuit time** | 83 μs | 83 μs | 83 μs |

### PEA-ZNE

| Parameter | Run 2026-06-14 (Tier 0) | Run 2026-06-23 (recovered) |
|-----------|-------------------------|---------------------------|
| **Shots per circuit** | 16,384 | 16,384 |
| **N layouts** | 1 | 1 (implicit) |
| **Noise factors** | [1, 1.5, 3] → 3 circuits/obs | [1, 1.5, 3] → 3 circuits/obs |
| **LNL overhead** | 48 × 192 = 9,216 shots | 48 × 192 = 9,216 shots |
| **Twirling overhead** | 48 × 192 = 9,216 shots | 48 × 192 = 9,216 shots |
| **Total per PUB** | ~16,384 + 9,216 + 9,216 = ~34,816 | ~49,152 (shots_per_eval) |
| **N observables** | 19 (Energy in 1 PUB) | 19 |

---

## 3. QPU Resources

| Metric | QESEM (average 3 jobs) | PEA-ZNE (Jun 14) | PEA-ZNE (Jun 23) |
|--------|------------------------|-------------------|-------------------|
| **QPU time (reported)** | 428 s | 284 s | 572 s |
| **QPU executing (actual)** | 278 s (tier-0) / 803 s (tier-1 h=3.5) | 284 s | 572 s |
| **Total wall-clock** | N/A (Qiskit Function) | 746.7 s (12.4 min) | ~17 min (created→finished) |
| **Billed seconds** | N/A (within Qiskit Function) | ~284 s | 572 s |
| **Queue wait** | 880 s (tier-0) / 38,886 s (tier-1!) | ~1.2 s | ~1.2 s |
| **T_one_job** | N/A | 322 s | — |

### QESEM resource_usage Breakdown (Tier-0, job 82aa)

| Phase | CPU Time | QPU Time |
|-------|----------|----------|
| MAPPING | 366.4 s | 0 |
| OPTIMIZING_FOR_HARDWARE | 104.7 s | 0 |
| WAITING_FOR_QPU | 880.5 s | 0 |
| EXECUTING_QPU | 0 | **278.4 s** |
| POST_PROCESSING | 408.2 s | 0 |
| **Total server-side** | **1,760 s** | **278 s** |

### QESEM resource_usage Breakdown (Tier-1 h=3.5, job d628)

| Phase | CPU Time | QPU Time |
|-------|----------|----------|
| MAPPING | 432.9 s | 0 |
| OPTIMIZING_FOR_HARDWARE | 114.1 s | 0 |
| WAITING_FOR_QPU | 38,886.2 s (10.8 h!) | 0 |
| EXECUTING_QPU | 0 | **803.5 s** |
| POST_PROCESSING | 348.4 s | 0 |
| **Total server-side** | **39,782 s** | **803 s** |

---

## 4. Reported Gate Fidelities

| Metric | QESEM Tier-0 (82aa) | QESEM Tier-1 (4f16) | QESEM Tier-1 h=3.5 (d628) | PEA-ZNE |
|--------|---------------------|---------------------|---------------------------|---------|
| **ID1Q fidelity** | 0.99903 | 0.99882 | 0.99828 | Not reported* |
| **RZZ (2Q) fidelity** | 0.99724 | 0.99779 | 0.99660 | Not reported* |
| **2Q infidelity** | 0.28% | 0.22% | 0.34% | ~0.18% (chip median) |

*PEA-ZNE does not report gate fidelities in the result JSON — it uses IBM backend calibration data.*

---

## 5. Error Bars and Estimation Quality

### QESEM — Error bars by observable type (Tier-0, h=4.0)

| Observable type | σ range | Average σ |
|-----------------|---------|-----------|
| **Total energy** | 0.288 | 0.288 |
| **⟨X⟩ per site (10)** | 0.007 – 0.025 | ±0.015 |
| **⟨ZZ⟩ per bond (6)** | 0.003 – 0.005 | ±0.004 |
| **⟨ZZ⟩ cross-chain (3)** | 0.003 – 0.004 | ±0.004 |

### PEA-ZNE — Error bars by observable type (Jun 23, h=4.0)

| Observable type | σ range | Average σ |
|-----------------|---------|-----------|
| **Total energy** | — (not reported as such) | — |
| **⟨X⟩ per site (10)** | 0.001 – 0.050 | ±0.017 |
| **⟨ZZ⟩ per bond (6)** | 0.009 – 0.046 | ±0.023 |
| **⟨ZZ⟩ cross-chain (3)** | 0.003 – 0.418 | ±0.141 |

### Uniformity Comparison

| Aspect | QESEM | PEA-ZNE |
|--------|-------|---------|
| **Inter-site variance ⟨X⟩** | Low (σ_max/σ_min ≈ 3.6×) | High (σ_max/σ_min ≈ 50×) |
| **Inter-bond variance ⟨ZZ⟩** | Very low (σ_max/σ_min ≈ 1.5×) | High (σ_max/σ_min ≈ 5×) |
| **Outliers** | None | ZZ cross-chain σ=0.42 (unstable) |

---

## 6. Unmitigated vs Mitigated Values (only QESEM reports both)

### Tier-0 (h=4.0, job 82aa)

| Observable | Unmitigated | Mitigated (QESEM) | Correction |
|-----------|------------|------------------|------------|
| **Energy** | -38.471 | -40.524 | +5.3% |
| **⟨X⟩ average** | 0.937 | 0.984 | +5.0% |
| **⟨ZZ⟩ average** | 0.105 | 0.124 | +18.1% |

### Tier-1 (h=3.5, job d628)

| Observable | Unmitigated | Mitigated (QESEM) | Correction |
|-----------|------------|------------------|------------|
| **Energy** | -33.641 | -35.359 | +5.1% |
| **⟨X⟩ average** | 0.929 | 0.975 | +4.9% |

*PEA-ZNE does not report separate unmitigated values — only the final extrapolated result.*
*They can be inferred from `evs_noise_factors[i][0]` (noise_factor=1.0) as a proxy.*

### PEA-ZNE — "Unmitigated" Proxy (noise_factor=1.0 vs extrapolated)

For ⟨X⟩ site 0: raw(NF=1) = 0.945 → extrapolated = 0.956 (correction +1.2%)
For ⟨ZZ⟩ bond 0: raw(NF=1) = 0.098 → extrapolated = 0.111 (correction +13%)

---

## 7. Comparative Summary of Resources and Overhead

| Dimension | QESEM | PEA-ZNE | Winner |
|-----------|-------|---------|--------|
| **QPU time (h=4.0)** | 278 s (actual exec) | 284–572 s | QESEM (~0–50% less) |
| **Total shots** | ~756K | ~49K/PUB | PEA-ZNE (15× fewer) |
| **Error bars ⟨X⟩** | ±0.015 (homogeneous) | ±0.017 (variable) | QESEM (slightly) |
| **Error bars ⟨ZZ⟩** | ±0.004 | ±0.023 | **QESEM (5.7× better)** |
| **Server-side overhead** | ~1760s CPU total | ~0 (all client-side) | PEA-ZNE |
| **Queue wait** | 880–38,886 s | 1.2 s | **PEA-ZNE** |
| **Transparency** | Reports unmitigated + 3 scales | Reports 3 noise factors raw | Tie |
| **User control** | Minimal (black-box Qiskit Function) | Full (transpilation, layout, DD, etc.) | PEA-ZNE |
| **Access requirement** | Qiskit Functions (specific plan) | Runtime Estimator (Open Plan OK) | PEA-ZNE |

---

## 8. Technical Notes

1. **QESEM reports `total_qpu_time=428s`** for all jobs, but `resource_usage.EXECUTING_QPU`
   shows different values (278s for tier-0, 803s for tier-1 h=3.5). The difference suggests
   that `total_qpu_time` is a budget/cap while `EXECUTING_QPU` is the actual value.

2. **PEA-ZNE Tier 0 (Jun 14)** used only 1 layout and the "balanced" config (48×192).
   The Jun 23 run also used 1 effective layout.

3. **Extreme queue wait in QESEM**: Job d628 waited ~10.8h in the Qedma server queue.
   This is NOT QPU time — it is internal queue time of the Qiskit Functions service.

4. **QESEM vs PEA-ZNE shots**: QESEM uses ~15× more shots, but produces comparable
   error bars for ⟨X⟩ and significantly better ones for ⟨ZZ⟩. The difference is explained
   by QESEM needing extra shots to construct the quasi-probability distribution.

5. **PEA-ZNE has `evs_noise_factors`**: A 19×3 array with raw values at each noise factor.
   This allows manually reconstructing the extrapolation curve and computing R².
