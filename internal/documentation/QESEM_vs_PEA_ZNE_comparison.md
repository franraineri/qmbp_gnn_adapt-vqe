# Comparación: QESEM (Qedma) vs PEA-ZNE (IBM Runtime)

**Fecha de análisis**: 2026-07-28
**Objetivo**: Comparar las técnicas de mitigación, configuración, y recursos QPU
(sin considerar métricas de resultado como ΔE/gap).

---

## Sistema bajo estudio (idéntico en ambos)

| Parámetro | Valor |
|-----------|-------|
| N qubits | 10 |
| Topología | heavy_hex |
| Modelo | TFIM |
| Ansatz | HVA p=1 |
| Backend | ibm_kingston (Heron R2, 156 qubits) |
| Gate nativa 2Q | CZ (median error ~0.18%) |

---

## 1. Técnicas de Mitigación de Error

### QESEM (Qedma — Qiskit Function)

| Técnica | Configuración |
|---------|---------------|
| **Método principal** | Quasi-probabilistic error mitigation (QESEM unbiased) |
| **Noise scaling** | 3 scale factors internos: 0.0 (extrapolado), 1.0 (hardware), 2.0 (amplificado) |
| **Extrapolación** | QESEM nativo (scale → 0) + heuristic exponential [1.0, 2.0] |
| **Dynamical Decoupling** | No (manejado internamente por protocolo Qedma) |
| **Twirling** | No explícito (incorporado en el protocolo QESEM) |
| **Readout Error Mitigation** | Incluido (REM implícito, reporta `results_with_REM`) |
| **Layer Noise Learning** | No (usa modelo de ruido propio de Qedma) |
| **Transpilación** | "standard" (manejada por Qedma server-side) |
| **Optimization level** | Controlado por Qedma |

### PEA-ZNE (IBM Runtime Estimator)

| Técnica | Configuración |
|---------|---------------|
| **Método principal** | PEA (Probabilistic Error Amplification) + ZNE extrapolation |
| **Noise factors** | [1.0, 1.5, 3.0] |
| **Extrapolación** | Exponential + Linear (dual, mejor R² gana) |
| **Dynamical Decoupling** | Sí — XpXm sequence, skip_reset_qubits=true |
| **Twirling** | Sí — active-circuit strategy, 48 randomizations × 192 shots/rand |
| **Readout Error Mitigation** | measure_mitigation = true (M3/TREX) |
| **Layer Noise Learning** | Sí — 48 randomizations × 192 shots/rand |
| **Transpilación** | User-side (Qiskit transpile, optimization_level=2) |
| **Layout selection** | Mapomatic VF2, lowest_cost strategy, CES-aware |
| **Affine correction** | Post-ZNE affine linear correction (optional) |

---

## 2. Configuración de Shots y Circuitos

### QESEM

| Parámetro | Tier-0 (h=4.0, job 82aa) | Tier-1 (h=4.0, job 4f16) | Tier-1 (h=3.5, job d628) |
|-----------|--------------------------|--------------------------|--------------------------|
| **Shots totales** | 756,048 | 772,848 | 779,448 |
| **Shots mitigación** | 266,000 (35%) | 282,800 (37%) | 289,400 (37%) |
| **Shots señal** | ~490,048 (65%) | ~490,048 (63%) | ~490,048 (63%) |
| **N observables** | 20 | 20 | 20 |
| **Precision target** | 0.01 | 0.01 | 0.01 |
| **N layouts (circuitos)** | 1 | 1 | 1 |
| **Circuit time** | 83 μs | 83 μs | 83 μs |

### PEA-ZNE

| Parámetro | Run 2026-06-14 (Tier 0) | Run 2026-06-23 (recovered) |
|-----------|-------------------------|---------------------------|
| **Shots por circuito** | 16,384 | 16,384 |
| **N layouts** | 1 | 1 (implicit) |
| **Noise factors** | [1, 1.5, 3] → 3 circuits/obs | [1, 1.5, 3] → 3 circuits/obs |
| **LNL overhead** | 48 × 192 = 9,216 shots | 48 × 192 = 9,216 shots |
| **Twirling overhead** | 48 × 192 = 9,216 shots | 48 × 192 = 9,216 shots |
| **Total por PUB** | ~16,384 + 9,216 + 9,216 = ~34,816 | ~49,152 (shots_per_eval) |
| **N observables** | 19 (Energía en 1 PUB) | 19 |

---

## 3. Recursos QPU

| Métrica | QESEM (promedio 3 jobs) | PEA-ZNE (14 jun) | PEA-ZNE (23 jun) |
|---------|------------------------|-------------------|-------------------|
| **QPU time (reportado)** | 428 s | 284 s | 572 s |
| **QPU executing (real)** | 278 s (tier-0) / 803 s (tier-1 h=3.5) | 284 s | 572 s |
| **Wall-clock total** | N/A (Qiskit Function) | 746.7 s (12.4 min) | ~17 min (created→finished) |
| **Billed seconds** | N/A (dentro de Qiskit Function) | ~284 s | 572 s |
| **Queue wait** | 880 s (tier-0) / 38,886 s (tier-1!) | ~1.2 s | ~1.2 s |
| **T_one_job** | N/A | 322 s | — |

### Desglose resource_usage QESEM (Tier-0, job 82aa)

| Fase | CPU Time | QPU Time |
|------|----------|----------|
| MAPPING | 366.4 s | 0 |
| OPTIMIZING_FOR_HARDWARE | 104.7 s | 0 |
| WAITING_FOR_QPU | 880.5 s | 0 |
| EXECUTING_QPU | 0 | **278.4 s** |
| POST_PROCESSING | 408.2 s | 0 |
| **Total server-side** | **1,760 s** | **278 s** |

### Desglose resource_usage QESEM (Tier-1 h=3.5, job d628)

| Fase | CPU Time | QPU Time |
|------|----------|----------|
| MAPPING | 432.9 s | 0 |
| OPTIMIZING_FOR_HARDWARE | 114.1 s | 0 |
| WAITING_FOR_QPU | 38,886.2 s (10.8 h!) | 0 |
| EXECUTING_QPU | 0 | **803.5 s** |
| POST_PROCESSING | 348.4 s | 0 |
| **Total server-side** | **39,782 s** | **803 s** |

---

## 4. Gate Fidelities Reportadas

| Métrica | QESEM Tier-0 (82aa) | QESEM Tier-1 (4f16) | QESEM Tier-1 h=3.5 (d628) | PEA-ZNE |
|---------|---------------------|---------------------|---------------------------|---------|
| **ID1Q fidelity** | 0.99903 | 0.99882 | 0.99828 | No reportado* |
| **RZZ (2Q) fidelity** | 0.99724 | 0.99779 | 0.99660 | No reportado* |
| **2Q infidelity** | 0.28% | 0.22% | 0.34% | ~0.18% (chip median) |

*PEA-ZNE no reporta gate fidelities en el resultado JSON — usa las calibraciones de IBM backend.

---

## 5. Error Bars y Calidad de Estimación

### QESEM — Error bars por tipo de observable (Tier-0, h=4.0)

| Tipo observable | Rango σ | Promedio σ |
|-----------------|---------|-----------|
| **Energía total** | 0.288 | 0.288 |
| **⟨X⟩ por sitio (10)** | 0.007 – 0.025 | ±0.015 |
| **⟨ZZ⟩ por bond (6)** | 0.003 – 0.005 | ±0.004 |
| **⟨ZZ⟩ cross-chain (3)** | 0.003 – 0.004 | ±0.004 |

### PEA-ZNE — Error bars por tipo de observable (23 jun, h=4.0)

| Tipo observable | Rango σ | Promedio σ |
|-----------------|---------|-----------|
| **Energía total** | — (no reportado como tal) | — |
| **⟨X⟩ por sitio (10)** | 0.001 – 0.050 | ±0.017 |
| **⟨ZZ⟩ por bond (6)** | 0.009 – 0.046 | ±0.023 |
| **⟨ZZ⟩ cross-chain (3)** | 0.003 – 0.418 | ±0.141 |

### Comparación de uniformidad

| Aspecto | QESEM | PEA-ZNE |
|---------|-------|---------|
| **Varianza inter-sitio ⟨X⟩** | Baja (σ_max/σ_min ≈ 3.6×) | Alta (σ_max/σ_min ≈ 50×) |
| **Varianza inter-bond ⟨ZZ⟩** | Muy baja (σ_max/σ_min ≈ 1.5×) | Alta (σ_max/σ_min ≈ 5×) |
| **Outliers** | Ninguno | ZZ cross-chain σ=0.42 (inestable) |

---

## 6. Valor Sin Mitigar vs Mitigado (solo QESEM reporta ambos)

### Tier-0 (h=4.0, job 82aa)

| Observable | Sin mitigar | Mitigado (QESEM) | Corrección |
|-----------|------------|------------------|------------|
| **Energía** | -38.471 | -40.524 | +5.3% |
| **⟨X⟩ promedio** | 0.937 | 0.984 | +5.0% |
| **⟨ZZ⟩ promedio** | 0.105 | 0.124 | +18.1% |

### Tier-1 (h=3.5, job d628)

| Observable | Sin mitigar | Mitigado (QESEM) | Corrección |
|-----------|------------|------------------|------------|
| **Energía** | -33.641 | -35.359 | +5.1% |
| **⟨X⟩ promedio** | 0.929 | 0.975 | +4.9% |

*PEA-ZNE no reporta valores sin mitigar separados — solo el resultado extrapolado final.*
*Se pueden inferir de `evs_noise_factors[i][0]` (noise_factor=1.0) como proxy.*

### PEA-ZNE — Proxy de "sin mitigar" (noise_factor=1.0 vs extrapolado)

Para ⟨X⟩ sitio 0: raw(NF=1) = 0.945 → extrapolado = 0.956 (corrección +1.2%)
Para ⟨ZZ⟩ bond 0: raw(NF=1) = 0.098 → extrapolado = 0.111 (corrección +13%)

---

## 7. Resumen Comparativo de Recursos y Overhead

| Dimensión | QESEM | PEA-ZNE | Ganador |
|-----------|-------|---------|---------|
| **QPU time (h=4.0)** | 278 s (real exec) | 284–572 s | QESEM (~0–50% menos) |
| **Shots totales** | ~756K | ~49K/PUB | PEA-ZNE (15× menos) |
| **Error bars ⟨X⟩** | ±0.015 (homogéneo) | ±0.017 (variable) | QESEM (ligeramente) |
| **Error bars ⟨ZZ⟩** | ±0.004 | ±0.023 | **QESEM (5.7× mejor)** |
| **Server-side overhead** | ~1760s CPU total | ~0 (todo client-side) | PEA-ZNE |
| **Queue wait** | 880–38,886 s | 1.2 s | **PEA-ZNE** |
| **Transparencia** | Reporta unmitigated + 3 scales | Reporta 3 noise factors raw | Empate |
| **Control del usuario** | Mínimo (black-box Qiskit Function) | Total (transpilation, layout, DD, etc.) | PEA-ZNE |
| **Requisito de acceso** | Qiskit Functions (plan específico) | Runtime Estimator (Open Plan OK) | PEA-ZNE |

---

## 8. Archivos Fuente

### QESEM Results

| Archivo | Job ID | Tier | h |
|---------|--------|------|---|
| `results/recovered/qesem/qesem_recovered_82aa33cc-862c-4ba1-8017-6ab61eb7054e.json` | 82aa33cc | Tier-0 | 4.0 |
| `results/recovered/qesem/qesem_recovered_4f16e846-9af2-4ee8-a78d-6f829766eefe.json` | 4f16e846 | Tier-1 | 4.0 |
| `results/recovered/qesem/qesem_recovered_d628a502-677a-4610-a78c-3d5266c0cdbf.json` | d628a502 | Tier-1 | 3.5 |

### PEA-ZNE Results

| Archivo | Job ID | Fecha | h |
|---------|--------|-------|---|
| `results/recovered/recovered_d8tche5bh0os73epdphg.json` | d8tche5bh0os73epdphg | 2026-06-23 | 4.0 |
| `scripts/hardware/complete_tier0_from_qpu.py` (datos) | d8nihtj2d42s73cdtit0 | 2026-06-14 | 4.0 |

### Scripts de Análisis

- `scripts/hardware/analyze_qesem_tier1.py`
- `scripts/hardware/analyze_qesem_error_detail.py`
- `scripts/hardware/convert_qesem_to_hwresult.py`
- `scripts/hardware/complete_tier0_from_qpu.py`
- `scripts/hardware/recover_job_result.py`

---

## 9. Notas Técnicas

1. **QESEM reporta `total_qpu_time=428s`** para todos los jobs, pero `resource_usage.EXECUTING_QPU`
   muestra valores diferentes (278s para tier-0, 803s para tier-1 h=3.5). La diferencia sugiere
   que `total_qpu_time` es un presupuesto/cap mientras que `EXECUTING_QPU` es el real.

2. **PEA-ZNE Tier 0 (14 jun)** usó solo 1 layout y la config "balanced" (48×192).
   El run del 23 jun también usó 1 layout efectivo.

3. **Queue wait extremo en QESEM**: El job d628 esperó ~10.8h en cola de Qedma server.
   Esto NO es tiempo de QPU — es cola interna del servicio Qiskit Functions.

4. **Shots QESEM vs PEA-ZNE**: QESEM usa ~15× más shots, pero produce error bars
   comparables en ⟨X⟩ y significativamente mejores en ⟨ZZ⟩. La diferencia se explica
   porque QESEM necesita shots extra para construir el quasi-probability distribution.

5. **PEA-ZNE tiene `evs_noise_factors`**: Array 19×3 con los valores raw a cada noise factor.
   Esto permite reconstruir la curva de extrapolación manualmente y calcular R².
