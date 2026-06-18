# Plan de Benchmarking Sistemático — Técnicas de Mitigación y Supresión de Ruido

**Fecha**: 2026-06-17
**Objetivo**: Evaluar sistemáticamente combinaciones de técnicas de error suppression/mitigation
con distintos parámetros, comparando simulación vs hardware real, para determinar
la configuración óptima del pipeline GNN-HVA en IBM Torino.

---

## 1. Diseño Experimental

### 1.1 Circuito Base (constante en todos los benchmarks)

| Parámetro | Valor |
|-----------|-------|
| Modelo | TFIM |
| N | 10 |
| p | 1 |
| Topología | heavy_hex |
| h_test | [3.25, 3.5, 3.75, 4.0] |
| θ_init | MPNN warm-start (pre-entrenado) |
| Transpiler level | 2 (nativo) / 0 (Mitiq) — variable del benchmark |

### 1.2 Modos de Ejecución

| Modo | Backend | Noise Model | Costo |
|------|---------|-------------|-------|
| `noiseless` | StatevectorEstimator | Ninguno | 0 QPU |
| `fake_backend` | FakeTorino (AerSimulator) | Calibración real | 0 QPU |
| `hardware` | ibm_torino | Ruido real | QPU time |

### 1.3 Baseline (referencia obligatoria)

Antes de cualquier técnica, establecer dos puntos de referencia:
- **Baseline noiseless**: Energía exacta via `ClassicalSolver` (techo de precisión).
  **NO re-ejecutar** — ya disponible en Phase 2 results para cada h_value.
- **Baseline noisy-raw**: FakeTorino SIN ninguna mitigación (piso de precisión) = config C0_raw.

Todas las métricas de mejora se calculan respecto a estos dos puntos.
El runner obtiene `e_exact` directamente de `ClassicalSolver.solve()` sin ejecutar
VQE noiseless — es un cálculo O(1) vía diagonalización exacta para N=10.

---

## 2. Técnicas a Evaluar

### 2.1 Error Suppression (prevención — no agregan overhead de shots)

| ID | Técnica | Parámetros a variar |
|----|---------|---------------------|
| DD-1 | Dynamical Decoupling | `sequence_type`: XX, XpXm, XY4 |
| DD-OFF | Sin DD | (referencia) |

**Nota (2026-06-17)**: Los parámetros `skip_reset_qubits` y `dd_scheduling` (alap/asap)
NO son configurables a través de IBM Runtime EstimatorV2 options. IBM Runtime aplica DD
internamente con scheduling ALAP por defecto. Solo `sequence_type` es parametrizable
vía `options.dynamical_decoupling.sequence_type`. Se eliminaron DD-2 y DD-3 del scope.

### 2.2 Error Mitigation (post-procesamiento — agregan overhead)

| ID | Técnica | Parámetros a variar |
|----|---------|---------------------|
| TW-1 | Pauli Twirling | num_randomizations: 16, 32, 64, 128 |
| TW-OFF | Sin Twirling | (referencia) |
| TREX-1 | TREX readout mitigation | ON / OFF |
| ZNE-GF | ZNE Gate Folding (nativo) | noise_factors: [1,2,3], [1,1.5,2,3], [1,1.5,3] |
| ZNE-PEA | ZNE + PEA amplifier (nativo) | learning budget: 32×128, 48×192, 64×256 |
| ZNE-MITIQ | Mitiq ZNE random fold (opt_level=0) | factors: [1,1.5,2,2.5,3], factory: linear/richardson |
| CDR-MITIQ | Mitiq CDR (Clifford Data Regression) | n_training: 5, 10, 15 |
| DDD-ZNE-MITIQ | Mitiq DDD+ZNE composition | ddd_rule: xx/xyxy, factory: linear |
| ZNE-OFF | Sin ZNE | (referencia) |
| AFF | Affine correction | ON / OFF (post-ZNE) |
| GNN-QEM | GNN error correction | ON / OFF (solo sin PEA) |

**Nota sobre transpilación y Mitiq**: Mitiq opera con `optimization_level=0`
(Qiskit 2.x cancela gates foldeados a nivel ≥ 1). Esto produce circuitos más
profundos pero con ZNE más preciso. Las configs Mitiq (ZNE-MITIQ, CDR-MITIQ,
DDD-ZNE-MITIQ) transpilan a level 0 internamente — no confundir con el
`transpiler_level=2` usado para PEA/GF nativo.

### 2.3 Combinaciones Prioritarias (evitar explosión combinatoria)

En lugar de probar todas las combinaciones (>500), definimos **configuraciones**:

| Config ID | DD | Twirling | TREX | ZNE | Extras | Prio |
|-----------|-------|----------|------|-----|--------|:----:|
| `C0_raw` | OFF | OFF | OFF | OFF | Ninguno | P0 |
| `C1_dd_only` | XpXm | OFF | OFF | OFF | — | P2 |
| `C2_dd_tw` | XpXm | 32 rand | ON | OFF | — | P2 |
| `C3_full_gf` | XpXm | 32 rand | ON | GF [1,1.5,3] | Affine | P1 |
| `C4_full_pea_light` | XpXm | auto | ON | PEA 32×128 | Affine | P1 |
| `C5_full_pea_balanced` | XpXm | 48 rand | ON | PEA 48×192 | Affine | P0 |
| `C6_full_pea_aggressive` | XpXm | 64 rand | ON | PEA 64×256 | Affine | P4 |
| `C7_xy4_pea` | XY4 | 48 rand | ON | PEA 48×192 | Affine | P3 |
| `C8_no_dd_pea` | OFF | 48 rand | ON | PEA 48×192 | Affine | P3 |
| `C9_gnn_qem` | XpXm | 32 rand | ON | GF [1,1.5,3] | GNN-QEM | P4 |
| `C10_kitchen_sink` | XpXm | 64 rand | ON | PEA 64×256 | Affine + GNN-QEM (control: should NOT help) | P4 |
| `C11_mitiq_zne` | OFF | OFF | OFF | Mitiq ZNE random (opt=0) | Affine | P1 |
| `C12_mitiq_cdr` | OFF | OFF | OFF | Mitiq CDR (10 circuits) | Affine | P0 |
| `C13_mitiq_ddd_zne` | OFF | OFF | OFF | Mitiq DDD(XX)+ZNE | Affine | P3 |
| `C14_dd_mitiq_cdr` | XpXm | 32 rand | ON | Mitiq CDR (10 circuits) | Affine | P2 |
| `C15_pea_ibm_canonical` | XpXm | 48 rand | ON | PEA factors=[1,1.3,1.6] | Affine | P1 |
| `C16_aqc_pea` | XpXm | 48 rand | ON | PEA 48×192 (AQC p=2→shallow) | Affine + AQC | P2 |
| `C17_aqc_mitiq_cdr` | OFF | OFF | OFF | Mitiq CDR (AQC p=2→shallow) | Affine + AQC | P3 |
| `C18_aqc_raw` | XpXm | 32 rand | ON | OFF (AQC p=2→shallow, sin ZNE) | AQC only | P4 |

**Total**: 19 configuraciones × 4 h-points = 76 ejecuciones por modo.

**Nota sobre configs C16-C18 (AQC-Tensor)**: Estas usan el circuito AQC-comprimido
(p=2 optimizado → shallow via MPS decomposition, ~9 CZ en heavy_hex N=10).
El objetivo es determinar si la expresividad extra de p=2 compensa bajo ruido:
- C16: AQC + PEA-ZNE completo (¿supera a C5 con p=1 directo?)
- C17: AQC + Mitiq CDR (¿CDR funciona mejor con circuito expresivo?)
- C18: AQC sin ZNE (¿la compresión sola da circuito tan shallow que no necesita ZNE?)

El beneficio AQC es **h-dependent**: máximo near h_c (3.0-3.25) donde expresividad
domina, mínimo en deep paramagnetic (h>4.0) donde p=1 ya basta. El benchmark
debe capturar este crossover — por eso incluimos h=3.0 y h=4.0 en h_test.

**Nota sobre configs C11-C14 (Mitiq)**: Estas usan `optimization_level=0` para
transpilación (obligatorio para preservar gate folding en Mitiq). Las configs
C11-C13 ejecutan sin DD/Twirling/TREX porque Mitiq maneja la mitigación
completamente a nivel software. C14 combina IBM suppression (DD+Tw+TREX)
con Mitiq CDR como mitigation — testa si la composición suppression+CDR
supera a PEA standalone.

**Hipótesis de transpilación**: `opt_level=0` produce circuitos ~30% más profundos
que `opt_level=2` pero permite folding preciso. La pregunta clave es si el ruido
adicional por profundidad extra se compensa con la mejor extrapolación ZNE.
Configs C11 vs C3 testan exactamente esto.

---

## 3. Métricas a Capturar

### 3.1 Métricas de Precisión (ya existentes en summary.json — NO duplicar)

Estas ya se guardan en el flujo normal. Solo verificar que estén presentes:
- `e_zne` (energía mitigada)
- `e_exact` (energía exacta de referencia)
- `delta_e_gap` (|E_zne - E_exact| / gap)
- `zne_r2` (calidad de extrapolación)
- `phase_label` vs `expected_label`
- `mag_x_mean`, `corr_zz_mean`
- `mitigation_config` (snapshot completo — ya implementado)

### 3.2 Métricas Adicionales del Benchmark (nuevas, específicas de este estudio)

| Métrica | Descripción | Cómo se obtiene |
|---------|-------------|-----------------|
| `execution_mode` | "noiseless" / "fake_backend" / "hardware" | Config |
| `config_id` | Identificador de la configuración (C0..C18) | Asignado |
| `wall_time_s` | Tiempo total de ejecución (wall clock) | `time.time()` |
| `qpu_seconds` | Tiempo QPU consumido (solo hardware) | `job.metrics()` |
| `circuit_depth_logical` | Profundidad del circuito lógico | `circuit.depth()` |
| `circuit_depth_transpiled` | Profundidad post-transpilación | `transpiled.depth()` |
| `circuit_depth_with_dd` | Profundidad estimada post-DD (≈ depth_transpiled + max_idle_stretch) | Estimación: `depth_transpiled + max_idle_stretch` (DD llena idle slots) |
| `n_2q_gates` | Gates 2Q en circuito transpilado | `count_ops()` |
| `n_1q_gates` | Gates 1Q en circuito transpilado | `count_ops()` |
| `depth_2q` | Profundidad 2Q (scheduling-aware) | `transpiled_circuit_stats()` |
| `total_shots_consumed` | Shots totales (incluyendo noise learning) | Calculado |
| `improvement_vs_raw` | (E_raw - E_mitigated) / (E_raw - E_exact) | Calculado |
| `overhead_factor` | shots_total / shots_baseline | Calculado |
| `precision_per_shot` | |ΔE/gap| / total_shots | Eficiencia |
| `optimization_level` | Nivel de transpilación usado | Config (0 para Mitiq, 2 para nativo) |
| `transpiled_vs_logical_ratio` | depth_transpiled / depth_logical | Indica overhead de routing |
| `noise_learning_budget` | num_rand × shots_per_rand (PEA) | Config |
| `layer_pair_depths` | Profundidades usadas para noise learning | Config (default o explícito) |

### 3.3 Métricas de Transpilación (clave para objetivo: mejores transpilaciones)

Estas métricas permiten comparar dos transpilaciones del MISMO circuito lógico
y determinar cuál producirá menor error en hardware. La decisión de transpilación
afecta ANTES de que cualquier mitigación actúe.

**Métricas primarias (dominan el error):**

| Métrica | Descripción | Por qué importa |
|---------|-------------|-----------------|
| `depth_2q` | Critical path a través de gates 2Q | **#1 predictor de error hardware.** Determina cuántos ciclos de decoherence sufren los qubits durante las gates 2Q secuenciales |
| `n_2q_gates` | Total de gates 2Q | Cada CZ/ECR/CX aporta ~0.3-1% error. Dominan completamente el error budget |
| `fidelity_estimate` | exp(-Σ n_gate×ε_gate) | Predicción directa del output quality |

**Métricas secundarias (refinas la decisión):**

| Métrica | Descripción | Cómo interpretar |
|---------|-------------|-----------------|
| `n_swap_gates` | SWAPs insertados por routing | Cada SWAP = 3 CX → routing costoso. `n_swap = (n_2q_transpiled - n_2q_logical) / 3` |
| `routing_overhead_pct` | (n_2q_transpiled - n_2q_logical) / n_2q_logical × 100 | 0% = routing perfecto. >50% = layout subóptimo |
| `depth_ratio` | depth_transpiled / depth_logical | <2 = transpilación eficiente. >3 = transpilación costosa |
| `idle_cycles_per_qubit` | Media de ciclos idle por qubit | Determina si DD tendrá efecto. >5 idle cycles → DD puede ayudar |
| `max_idle_stretch` | Mayor tramo consecutivo de idle en un qubit | Si <3 ciclos → DD inútil. Si >10 → DD valioso |
| `parallelism_ratio` | n_2q_gates / depth_2q | =1 → completamente secuencial. >1 → gates parallelizadas |
| `gate_density_2q` | n_2q_gates / (n_active_qubits × depth_2q) | Qué fracción del espacio qubit×tiempo está ocupado por 2Q |

**Métricas de layout/mapping:**

| Métrica | Descripción | Impacto en mitigación |
|---------|-------------|----------------------|
| `layout_2q_error_sum` | Σ ε(CZ) sobre bonds usados | Menor = mejor layout |
| `layout_readout_error_sum` | Σ ε(readout) sobre qubits usados | Afecta TREX effectiveness |
| `layout_T1_min` | Peor T1 en el layout | Si <50μs → decoherence domina |
| `layout_T2_min` | Peor T2 en el layout | Si <30μs → dephasing domina |
| `layout_connectivity_score` | Fracción de bonds lógicos que son nativos | 1.0 = no routing needed |
| `CES` | Circuit Error Score (post-transpilation) | Composite metric, ya implementado |

**Métricas de tiempo/decoherence (hardware real):**

| Métrica | Descripción | Fórmula |
|---------|-------------|---------|
| `circuit_duration_ns` | Duración total del circuito en nanosegundos | sum(gate_durations) en critical path |
| `decoherence_limit` | Fracción de T1 consumida por el circuito | circuit_duration / min(T1) |
| `dephasing_limit` | Fracción de T2 consumida por el circuito | circuit_duration / min(T2) |
| `zne_viable` | bool: n_2q × max_fold < 18 CX threshold | Determina si gate-folding ZNE es lineal |
| `pea_overhead_ns` | Tiempo extra por noise learning | n_randomizations × shots × circuit_duration |

**Métricas para comparar opt_level=0 vs opt_level=2:**

| Métrica | opt=0 (Mitiq) | opt=2 (PEA nativo) | Qué revela |
|---------|:---:|:---:|---|
| `depth_2q` | Mayor (no cancela) | Menor (cancela inversas) | Costo de no-cancelación |
| `n_2q_gates` | Mayor | Menor | Gates extra por no-optimizar |
| `folding_effective` | ✅ (folds preservados) | N/A | Prerequisito para Mitiq ZNE |
| `fidelity_estimate` | Menor (más gates) | Mayor (menos gates) | Trade-off: precision_ZNE vs gate_noise |
| `net_benefit` | fidelity × mitigation_gain | fidelity × mitigation_gain | El número que decide |

La métrica **`net_benefit`** es la que resuelve la pregunta de transpilación:
```
net_benefit = fidelity_estimate(transpilation) × mitigation_gain(technique)

Si net_benefit(opt=0 + Mitiq) > net_benefit(opt=2 + PEA) → usar Mitiq
Si net_benefit(opt=2 + PEA) > net_benefit(opt=0 + Mitiq) → usar PEA
```

### 3.4 Métricas AQC-Tensor (cuando `--aqc-compress` activo)

| Métrica | Descripción | Cómo interpretar |
|---------|-------------|-----------------|
| `aqc_compression_used` | bool — si se usó AQC p=2→shallow | Flag para filtrado |
| `aqc_fidelity` | F = |⟨ψ_compressed|ψ_p2⟩|² | ≥0.998 → aceptado, <0.998 → fallback a p=1 |
| `aqc_bond_dim` | χ usado en MPS decomposition | Mayor χ → mejor fidelity pero más tiempo |
| `aqc_n_2q_original` | Gates 2Q del circuito p=2 (36 para N=10) | Referencia |
| `aqc_n_2q_compressed` | Gates 2Q post-compresión (~9-18) | Output clave |
| `aqc_2q_reduction_pct` | (1 - compressed/original) × 100 | Target: ≥50% |
| `aqc_compression_time_s` | Wall-clock de la compresión | Overhead offline (~1-10s) |
| `aqc_expressivity_benefit` | ΔE/gap(p=1 directo) - ΔE/gap(AQC) | >0 = AQC es mejor |
| `aqc_fidelity_vs_h` | Array: F(h) para cada h_test | Revela degradación near h_c |
| `aqc_fallback_triggered` | bool: fidelity < threshold → usó p=1 | Indica h donde AQC falla |
| `aqc_n_params_optimized` | Params del circuito comprimido | Para reproducibilidad |

### 3.5 Métricas de Robustez y Confiabilidad

| Métrica | Descripción | Cómo interpretar |
|---------|-------------|-----------------|
| `zne_r2` | R² de la extrapolación | <0.80 = extrapolación poco confiable |
| `extrapolation_residuals` | Residuos del fit (per noise factor) | Detectar non-linearidad |
| `per_site_magnetization_std` | σ(⟨Xᵢ⟩) across sites | Inhomogeneidad = noise not mitigated |
| `shot_noise_estimate` | 1/√(shots) × observable_range | Piso de precision |
| `mitiq_cdr_raw_vs_exact` | |E_raw - E_exact| (pre-CDR) | Input quality for CDR |
| `mitiq_cdr_correction_magnitude` | |E_mitigated - E_raw| | CDR correction signal |
| `phase_label_confidence` | |⟨X⟩ - ⟨ZZ⟩| × √shots | Confiabilidad clasificación |
| `energy_within_physical_bounds` | bool: E_ground ≤ E_result ≤ E_upper | Sanity check básico |

### 3.6 Métricas IBM-Specific (Hardware Real)

Capturadas SOLO en modo `hardware`, de `job.metrics()` y calibration data:

| Métrica | Descripción | Para qué sirve |
|---------|-------------|----------------|
| `t1_mean_layout` | T1 promedio de qubits en layout | Predice decoherence |
| `t2_mean_layout` | T2 promedio de qubits en layout | Predice dephasing |
| `cx_error_mean_layout` | Error medio de 2Q gates en layout | Input para PEC overhead calc |
| `readout_error_mean` | Error medio de readout | Predice TREX benefit |
| `calibration_age_hours` | Horas desde última calibración | Freshness indicator |
| `job_execution_time_s` | Tiempo real en QPU | Scheduling overhead vs compute |

### 3.7 Métricas de Comparación (derivadas post-ejecución)

Calculadas en el análisis posterior, NO durante ejecución:
- **Ganancia relativa** por técnica individual (ablation)
- **Costo-beneficio**: Δprecisión / Δtiempo (Pareto frontier)
- **Ranking** de configuraciones por ΔE/gap
- **Correlación** entre parámetros y resultados (Spearman ρ)
- **Transfer ratio** simulación → hardware (¿rankings se mantienen?)
- **Sensitivity analysis**: ∂(ΔE/gap)/∂(parámetro) para cada parámetro continuo
- **Confidence interval** (95% CI) para cada método (multi-seed o bootstrap)

---

## 4. Estructura de Resultados

```
results/mitigation_benchmark/
├── manifest.json                    # Lista de todas las ejecuciones + metadata
├── configs/
│   ├── C0_raw.json                 # Definición de cada configuración
│   ├── C1_dd_only.json
│   └── ...
├── fake_backend/                    # Resultados en simulación
│   ├── C0_raw/
│   │   ├── h3p25_run_20260618_*.json
│   │   ├── h3p50_run_20260618_*.json
│   │   └── ...
│   ├── C5_full_pea_balanced/
│   │   └── ...
│   └── ...
├── hardware/                        # Resultados en hardware real
│   ├── C5_full_pea_balanced/
│   │   └── ...
│   └── ...
└── analysis/
    ├── comparison_table.json        # Tabla consolidada
    ├── ablation_study.json          # Impacto individual de cada técnica
    └── figures/
        ├── precision_vs_config.png
        ├── cost_benefit.png
        └── technique_ablation.png
```

### 4.1 Formato de Resultado por Ejecución

Cada ejecución guarda UN JSON con:
```json
{
  "benchmark_metadata": {
    "config_id": "C5_full_pea_balanced",
    "execution_mode": "fake_backend",
    "h_value": 3.5,
    "timestamp": "2026-06-18T10:30:00Z",
    "benchmark_version": "1.0"
  },
  "circuit_stats": {
    "depth_logical": 22,
    "depth_transpiled": 34,
    "depth_with_dd_estimate": 38,
    "n_2q_gates": 18,
    "n_1q_gates": 30,
    "total_ops": 48
  },
  "timing": {
    "wall_time_s": 45.2,
    "qpu_seconds": null,
    "noise_learning_time_s": 12.3,
    "zne_extrapolation_time_s": 0.01
  },
  "results": {
    "e_mitigated": -11.234,
    "e_raw": -10.89,
    "e_exact": -11.25,
    "delta_e_gap": 0.008,
    "improvement_vs_raw": 0.955,
    "zne_r2": 0.998,
    "phase_label": "paramagnetic",
    "correct_label": true
  },
  "shots": {
    "per_layout": 16384,
    "n_layouts": 3,
    "noise_learning_shots": 9216,
    "total_consumed": 58368
  },
  "mitigation_config": { ... }
}
```

---

## 5. Plan de Ejecución

### Fase A — Simulación completa (FakeTorino, 0 QPU cost)

**Objetivo**: Identificar las mejores 3-4 configuraciones antes de gastar QPU.

1. Ejecutar 19 configs × 4 h-points = 76 runs en `fake_backend`
2. Calcular métricas de comparación
3. Ranking por ΔE/gap promedio
4. Seleccionar top-4 + baseline para hardware

**Tiempo estimado**: ~4-5 horas (FakeTorino local, parallelizable)
Mitiq configs (C11-C14) son más rápidas que PEA (~0.5s vs ~2s por run).
AQC configs (C16-C18) añaden ~5s de compresión por h-point (offline, cached).

### Fase B — Validación en hardware (IBM Torino)

**Objetivo**: Confirmar que los rankings de simulación se mantienen en hardware real.

1. Ejecutar top-3 configs + C0_raw + C5_balanced = 5 configs × 4 h-points = 20 runs
2. Comparar con resultados de Fase A
3. Calcular correlación sim↔hardware

**Tiempo estimado**: ~1-2 horas QPU (con Batch mode, queue time adicional)

### Fase C — Análisis y Tablas de Tesis

1. Generar tabla comparativa (Chapter 5 material)
2. Figuras: barplot ΔE/gap por config, cost-benefit scatter, ablation heatmap
3. Conclusiones sobre configuración óptima
4. Documentar en binnacle

---

## 6. Implementación

### 6.1 Script Principal

```
scripts/experiment_runners/hardware/run_mitigation_benchmark.py
```

Responsabilidades:
- Cargar configs desde `results/mitigation_benchmark/configs/`
- Para cada config: construir `MitigationOptions`, ejecutar, guardar
- Añadir entrada al `manifest.json`
- Calcular `circuit_stats` y `timing` automáticamente

### 6.2 Analyzer

```
python -m project_health.analysis.mitigation_benchmark_analyzer
```

Responsabilidades:
- Leer todos los JSONs del benchmark
- Consolidar en `comparison_table.json`
- Calcular métricas derivadas (improvement_vs_raw, overhead_factor)
- Generar figuras

### 6.3 Integración con Pipeline Existente

- Usa `HardwareBackend.run_deployment()` internamente (métricas existentes gratis)
- `mitigation_config` snapshot ya se guarda en summary.json
- Solo agrega `benchmark_metadata`, `circuit_stats`, `timing` como wrapper

---

## 7. Prioridad de Ejecución (si QPU time es limitado)

| Prioridad | Configs | Razón |
|-----------|---------|-------|
| P0 (obligatorio) | C0_raw, C5_balanced, C12_mitiq_cdr | Baseline + mejor candidato + cross-check independiente |
| P1 (alta) | C3_full_gf, C4_pea_light, C11_mitiq_zne, C15_ibm_canonical | Comparar GF vs PEA vs Mitiq vs IBM-canonical |
| P2 (media) | C1_dd_only, C2_dd_tw, C14_dd_mitiq_cdr, C16_aqc_pea | Ablation DD/twirling + CDR + AQC+PEA |
| P3 (baja) | C7_xy4, C8_no_dd, C13_mitiq_ddd_zne, C17_aqc_mitiq_cdr | Variantes de DD + DDD+ZNE + AQC+CDR |
| P4 (opcional) | C6_aggressive, C9_gnn, C10_kitchen, C18_aqc_raw | Extremos y controles |

---

## 8. Criterios de Éxito

| Criterio | Threshold |
|----------|-----------|
| Mejor config: ΔE/gap | < 3% en hardware |
| Correlación sim↔hardware | Spearman ρ > 0.8 |
| Overhead justificado | Δprecisión/Δcosto > 0 para toda técnica incluida |
| Ablation claro | Cada técnica ON→OFF causa degradación medible |
| Reproducibilidad | 3 seeds × best config → std < 1% |

---

## 9. Metodología de Análisis — Cómo Encontrar la Configuración Óptima

### 9.1 Principio: Análisis por Capas (de simple a complejo)

El análisis sigue un embudo de 4 niveles. Cada nivel filtra configuraciones y
refina la comprensión:

```
Nivel 1: Ranking bruto           → ¿Quién gana en ΔE/gap?
Nivel 2: Ablation study          → ¿Qué técnica individual aporta más?
Nivel 3: Análisis costo-beneficio → ¿El overhead se justifica?
Nivel 4: Robustez y correlación   → ¿Es estable? ¿Se transfiere a hardware?
```

### 9.2 Nivel 1 — Ranking Bruto

**Input**: Todos los resultados de Fase A (79 runs: 16 configs × 4h + 3 AQC × 5h).
**Método**: Ordenar configuraciones por ΔE/gap promedio (media sobre h-points).

| Métrica principal | Cómo se calcula |
|-------------------|-----------------|
| `mean_delta_e_gap` | media(ΔE/gap) sobre los 4 h-points |
| `max_delta_e_gap` | peor caso entre los 4 h-points |
| `success_rate` | % de h-points con ΔE/gap < 5% |
| `correct_phase_rate` | % de h-points con label correcto |

**Output**: Tabla ordenada. Las configs con `success_rate < 50%` se descartan
inmediatamente (no vale la pena analizar más).

### 9.3 Nivel 2 — Ablation Study (contribución individual)

**Pregunta**: Si quito UNA técnica de la mejor configuración, ¿cuánto pierde?

**Método**: Comparación pareada. Para la mejor config (digamos C5), calcular:

```
Δ_DD      = ΔE/gap(C5) - ΔE/gap(C8_sin_DD)        → contribución de DD
Δ_Twirl   = ΔE/gap(C5) - ΔE/gap(C5_sin_twirl)     → contribución de Twirling
Δ_TREX    = ΔE/gap(C5) - ΔE/gap(C5_sin_trex)      → contribución de TREX
Δ_ZNE     = ΔE/gap(C5) - ΔE/gap(C2_sin_zne)       → contribución de ZNE
Δ_Affine  = ΔE/gap(con_affine) - ΔE/gap(sin_affine) → contribución de Affine
```

**Interpretación**:
- Δ > 0: La técnica empeora (debería quitarse)
- Δ < 0: La técnica mejora (debe mantenerse)
- |Δ| < ruido estadístico: La técnica no tiene efecto medible

**Visualización**: Heatmap de ablation (filas: técnicas, columnas: h-points,
color: Δ contribución). Permite ver si una técnica ayuda en ciertos regímenes
pero no en otros.

### 9.4 Nivel 3 — Análisis Costo-Beneficio

**Pregunta**: ¿El overhead adicional de QPU/tiempo se justifica por la mejora?

**Métricas de eficiencia**:

```
efficiency = |Δ_precision| / overhead_factor

donde:
  Δ_precision = ΔE/gap(baseline) - ΔE/gap(config)   [mejora absoluta]
  overhead_factor = total_shots(config) / total_shots(C0_raw)

precision_per_second = (1 - ΔE/gap) / wall_time_s    [para simulación]
precision_per_qpu_s  = (1 - ΔE/gap) / qpu_seconds    [para hardware]
```

**Frontera de Pareto**: Graficar ΔE/gap (eje Y, menor=mejor) vs overhead_factor
(eje X, menor=mejor). Las configuraciones en la frontera de Pareto son las
únicas que vale la pena considerar — todo lo demás está dominado.

```
   ΔE/gap
    5% |  x C0_raw
       |
    3% |     x C3_gf            x C9_gnn
       |
    1% |        * C5_balanced ←── Frontera de Pareto
       |              * C6_aggressive
    0% |________________________
       1x    2x    4x    8x   overhead
```

**Decisión**: Si C5 y C6 tienen ΔE/gap similar pero C6 cuesta 2× más → elegir C5.

### 9.5 Nivel 4 — Robustez y Transferencia Sim→Hardware

**Robustez** (multi-seed, solo para top-3 configs):
- Ejecutar con seeds 42, 43, 44
- Calcular `std_delta_e_gap` y `coefficient_of_variation`
- Config estable: CV < 0.2 (std es <20% de la media)

**Transferencia Simulación → Hardware**:
- Para cada config ejecutada en ambos modos, calcular:
  ```
  transfer_ratio = ΔE/gap(hardware) / ΔE/gap(fake_backend)
  ```
- Si `transfer_ratio ≈ 1.0`: la simulación predice bien el hardware
- Si `transfer_ratio >> 1.0`: la simulación es optimista (hardware peor)
- Si `transfer_ratio < 1.0`: el hardware es MEJOR que simulación (posible si
  calibración mejoró vs el snapshot de FakeTorino)

**Correlación de rankings**:
- Spearman ρ entre ranking_sim y ranking_hw
- Si ρ > 0.8: podemos confiar en simulación para futuros benchmarks
- Si ρ < 0.5: la simulación no es predictiva y debemos hacer todo en hardware

### 9.6 Análisis de Sensibilidad por Parámetro

Para técnicas con parámetros continuos (PEA budget, twirling randomizations),
generar curvas de respuesta:

**PEA learning budget vs precisión**:
```
Budget (shots)    |  4K (32×128)  |  9K (48×192)  |  16K (64×256)
ΔE/gap           |    2.1%       |    0.8%       |    0.7%
Wall time        |    30s        |    45s         |    90s
```

Si la curva satura (el salto 9K→16K mejora poco), la configuración media es óptima.

**Twirling randomizations vs precisión**:
```
Randomizations   |  16   |  32   |  64   |  128
ΔE/gap           |  3.2% |  1.8% |  1.7% |  1.7%
```

Punto de saturación = valor óptimo (mínimo costo con máxima precisión).

---

## 10. Tablas de Tesis (output final)

### Tabla 10.1 — Comparación de Técnicas de Supresión (Chapter 5)

| Técnica | ΔE/gap (sim) | ΔE/gap (hw) | Overhead | Transfer ratio |
|---------|:------------:|:-----------:|:--------:|:--------------:|
| Raw (C0) | X.XX% | X.XX% | 1.0× | — |
| +DD XpXm (C1) | X.XX% | — | 1.0× | — |
| +DD+Tw+TREX (C2) | X.XX% | — | 1.0× | — |
| Full GF (C3) | X.XX% | X.XX% | ~3× | X.XX |
| Full PEA balanced (C5) | X.XX% | X.XX% | ~4× | X.XX |
| Full PEA aggressive (C6) | X.XX% | — | ~8× | — |

### Tabla 10.2 — Ablation Study (Chapter 5)

| Técnica removida | Δ ΔE/gap | p-value | Conclusión |
|------------------|:--------:|:-------:|------------|
| -DD | +X.XX% | 0.XXX | Necesaria / No necesaria |
| -Twirling | +X.XX% | 0.XXX | ... |
| -TREX | +X.XX% | 0.XXX | ... |
| -ZNE | +X.XX% | 0.XXX | ... |
| -Affine | +X.XX% | 0.XXX | ... |

### Tabla 10.3 — Configuración Óptima Recomendada

```
┌─────────────────────────────────────────────┐
│ CONFIGURACIÓN ÓPTIMA: C?_???                │
├─────────────────────────────────────────────┤
│ DD:       XpXm, skip_reset=True, alap      │
│ Twirling: NN randomizations               │
│ TREX:     ON                               │
│ ZNE:      PEA/GF, factors=[...]            │
│ Affine:   ON/OFF                           │
│ Shots:    NNNNN × N layouts                │
├─────────────────────────────────────────────┤
│ ΔE/gap (sim):  X.XX% ± X.XX%              │
│ ΔE/gap (hw):   X.XX% ± X.XX%              │
│ Overhead:      X.X× vs raw                 │
│ Transfer:      X.XX (sim→hw)               │
└─────────────────────────────────────────────┘
```

---

## 11. Flujo de Decisiones (Diagrama)

```
                    ┌──────────────┐
                    │  Fase A: Sim │
                    │  (44 runs)   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Nivel 1:     │
                    │ Ranking bruto│──── descartar success_rate < 50%
                    └──────┬───────┘
                           │ top-6 configs
                    ┌──────▼───────┐
                    │ Nivel 2:     │
                    │ Ablation     │──── identificar técnicas sin efecto
                    └──────┬───────┘
                           │ técnicas confirmadas
                    ┌──────▼───────┐
                    │ Nivel 3:     │
                    │ Costo-benefi │──── descartar dominadas
                    └──────┬───────┘
                           │ top-3 en frontera Pareto
                    ┌──────▼───────┐
                    │  Fase B: HW  │
                    │  (20 runs)   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Nivel 4:     │
                    │ Robustez +   │
                    │ Transferencia│
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Config Óptima│
                    │ + Tablas Tesis│
                    └──────────────┘
```

---

## 12. Hipótesis a Validar

Cada benchmark tiene una hipótesis explícita. Sin hipótesis → no ejecutar.

| # | Hipótesis | Configs que la testan |
|---|-----------|----------------------|
| H1 | DD XpXm mejora vs no-DD en ibm_torino (gates rápidos, 84ns) | C1 vs C0, C5 vs C8 |
| H2 | XY4 NO mejora sobre XpXm en ibm_torino (idle time insuficiente) | C7 vs C5 |
| H3 | PEA supera a GF por >50% en ΔE/gap (replicando nuestros resultados previos) | C5 vs C3 |
| H4 | PEA budget 48×192 es suficiente (saturación de la curva de aprendizaje) | C4 vs C5 vs C6 |
| H5 | Twirling aporta >10% de mejora relativa vs no-twirling | C2 vs C1 |
| H6 | GNN-QEM NO ayuda después de PEA (confirmado en exp anterior) | C10 vs C5 |
| H7 | GNN-QEM SÍ ayuda después de GF-ZNE (alternativa viable) | C9 vs C3 |
| H8 | Affine correction tiene costo cero y nunca empeora | Todos con/sin affine |
| H9 | Rankings de simulación correlacionan con hardware (ρ > 0.8) | Fase A vs Fase B |
| H10 | La config óptima logra ΔE/gap < 3% en hardware real | Fase B, best config |
| H11 | Mitiq ZNE random (opt=0) supera a nuestro GF-ZNE (opt=2) en simulación | C11 vs C3 |
| H12 | Mitiq CDR produce mejora comparable a PEA sin noise model | C12 vs C5 |
| H13 | DD + Mitiq CDR (C14) es competitivo con PEA completo (C5) | C14 vs C5 |
| H14 | opt_level=0 + Mitiq compensa profundidad extra con mejor extrapolación | C11 depth vs C3 depth vs ΔE/gap |
| H15 | IBM canonical factors [1,1.3,1.6] son suficientes (PEA preciso, no necesita factores altos) | C15 vs C5 |
| H16 | AQC p=2→shallow + PEA supera p=1 directo + PEA (expresividad near h_c) | C16 vs C5 (h=3.0,3.25) |
| H17 | AQC + Mitiq CDR produce circuito expresivo + mitigation sin noise model | C17 vs C12 |
| H18 | AQC sin ZNE: circuito tan shallow (9 CZ) que mitigación es innecesaria | C18 vs C16, C18 vs C0 |
| H19 | El beneficio AQC es h-dependent: máximo en [3.0,3.25], nulo en h≥4.0 | C16 per-h breakdown |

---

## 13. Condiciones de Aborto

| Condición | Acción |
|-----------|--------|
| FakeTorino OOM en alguna config | Reducir shots o layouts para esa config |
| ZNE R² < 0.5 en alguna config | Registrar como "ZNE failed", no descartar datos |
| >50% de configs tienen success_rate=0 | Revisar θ_init (posible bug en MPNN) |
| Wall time > 30 min para 1 run en sim | Revisar config (posible PEA budget excesivo) |
| Hardware queue > 4h entre submits | Batch más runs por sesión |

---

## 14. Notas Importantes

- **NO re-ejecutar noiseless** — ya tenemos esos resultados en Phase 2 (VQE converged)
- **NO duplicar métricas** — `run_deployment()` ya guarda todo en summary.json; el benchmark solo agrega `benchmark_metadata` + `circuit_stats` + `timing`
- **Distinguir siempre** el `execution_mode` en cada resultado
- **Seeds**: Usar seed=42 para comparabilidad. Multi-seed (42/43/44) solo para la config ganadora
- **No optimizar durante benchmark** — θ_init es fijo (MPNN prediction). Solo medimos E(θ_pred)
- **Versionado**: El plan es v1.0. Si descubrimos que necesitamos configs adicionales, crear v1.1 con justificación

### 14.1 Notas Técnicas Adicionales (correcciones 2026-06-17)

- **DD scheduling/skip_reset NO configurable**: IBM Runtime EstimatorV2 maneja DD
  internamente (ALAP default). Solo `sequence_type` es configurable. DD-2 y DD-3
  eliminados del scope experimental.
- **`circuit_depth_with_dd` es una estimación**: DD no se aplica localmente en FakeTorino.
  Se calcula como `depth_transpiled + max_idle_stretch` (DD llena slots idle con pulsos).
- **`noise_learning_shots` = 0 en fake_backend**: PEA en simulación local no ejecuta
  noise learning real (estima directamente). Solo se llena en modo hardware.
- **Affine sobre raw**: Para validar H8 completamente, `affine_enabled=True` con
  `zne_method=None` aplica affine clipping directamente sobre `e_raw` (energía cruda).
  Esto testea si affine nunca empeora incluso sin ZNE previa.
- **`n_layouts` por config**: Default=3, pero C0_raw puede usar 1 layout (sin averaging)
  para establecer el piso de costo más bajo posible.
- **Transfer ratio**: Se calcula post-hoc como `ΔE/gap(hw) / ΔE/gap(sim)` para configs
  ejecutadas en ambos modos. Solo disponible después de Fase B.
- **Correlación Spearman**: Se computa entre ranking_sim y ranking_hw (requiere ≥5 configs
  en ambos modos). Threshold de éxito: ρ > 0.8.
- **Sensitivity curves**: Para PEA budget (C4→C5→C6) y twirling (variando randomizations),
  se generan curvas de respuesta {parámetro → ΔE/gap} para identificar punto de saturación.
- **Hypothesis mapping**: El analyzer vincula pares de configs a las hipótesis H1-H19
  para reportar validación/refutación automática en la tabla de comparación.

---

## 15. Referencias

| Tema | Ubicación |
|------|-----------|
| PEA validation (nuestros resultados previos) | `documentation/binnacles/binnacle-gate-folding-zne.md` |
| ZNE cross-topology (PEA 18/18 wins) | `results/experiments/exp_zne_cross_topo/` |
| GNN-QEM + PEA incompatibility | `results/gnn_qem/post_zne_validation.json` |
| DD investigation (XpXm vs XY4) | arXiv:2405.08689 (Rahman et al. 2024) |
| IBM Runtime options docs | `https://docs.quantum.ibm.com/guides/configure-error-suppression` |
| MitigationOptions dataclass | `src/qmbp_simulation/execution/backends.py` |
| Persistence (mitigation_config) | `src/qmbp_simulation/execution/hardware/persistence.py` |
| Hardware deployment script | `scripts/experiment_runners/hardware/run_ibm_deployment.py` |
| Mitiq integration module | `src/qmbp_simulation/execution/mitiq_utils.py` |
| Mitiq integration plan | `documentation/analysis/24_mitiq_integration_plan.md` |
| Mitiq steering (opt_level=0 critical) | `.kiro/steering/mitiq-integration.md` |
| Mitiq CDR paper | Czarnik et al., Quantum 5, 592 (2021) |
| Mitiq docs | https://mitiq.readthedocs.io/en/stable/ |
| Transpiler exploration findings | `documentation/analysis/15_transpiler_exploration.md` |

---

## 16. Integración con Mitiq — Consideraciones de Transpilación

### 16.1 El dilema opt_level=0 vs opt_level=2

Para ZNE, la transpilación determina cuánto ruido amplifica el folding:

| Transpilación | Profundidad circuito | Folding preservado | ZNE accuracy |
|:---:|:---:|:---:|:---:|
| `opt_level=2` + PEA nativo | Mínima (optimizada) | N/A (PEA no fold) | Excelente (+94%) |
| `opt_level=2` + GF nativo | Mínima (fold post-transpile) | Sí (fold en ISA) | Buena (+20%) |
| `opt_level=0` + Mitiq ZNE random | Mayor (~30% más gates) | Sí (Mitiq fold antes) | Muy buena (+76%) |
| `opt_level=0` + Mitiq CDR | Mayor | N/A (CDR no fold) | Buena (+58%) |

**Pregunta clave**: ¿El +56% de mejora ZNE de Mitiq (76% vs 20%) compensa el circuito
30% más profundo? En simulación (noise uniforme), sí. En hardware real (noise
no-uniforme, crosstalk, etc.), es una pregunta abierta que este benchmark resolverá.

### 16.2 Estrategia para el benchmark de transpilación

Las configs C11-C14 (Mitiq) usan `optimization_level=0` obligatoriamente.
Esto permite un análisis factorial:

```
Factor 1: Transpilation level      → 0 (Mitiq) vs 2 (nativo)
Factor 2: Mitigation technique     → ZNE random vs GF vs PEA vs CDR
Factor 3: Error suppression stack  → OFF vs DD+Tw+TREX
```

El resultado es una tabla 2×4×2 = 16 combinaciones que revela:
- Si `opt_level` importa más que la técnica de mitigación
- Si suppression + Mitiq es mejor que suppression + PEA nativo
- La transpilación óptima para CADA técnica de mitigación

### 16.3 Implementación (usa compare_mitigation_strategies)

Para configs C11-C14, el benchmark usa directamente nuestra integración:

```python
from qmbp_simulation.execution.mitiq_utils import (
    run_mitiq_zne,      # C11
    run_mitiq_cdr,      # C12, C14
    run_mitiq_ddd_zne,  # C13
    compare_mitigation_strategies,  # benchmark mode
)
```

Para configs C0-C10 (IBM nativo), usa `HardwareBackend.run_deployment()` como antes.

### 16.4 AQC × Transpilación × Mitigación: Análisis 3-factorial

AQC opera ANTES de la transpilación — reduce la complejidad del circuito lógico.
La interacción AQC × transpilación × mitigación crea un espacio 3D:

```
                      Mitigación
                    /           \
               PEA-ZNE       Mitiq CDR
              /       \       /       \
         [p=1]     [AQC]  [p=1]    [AQC]    ← Circuito input
            |         |      |         |
       opt=2       opt=2   opt=0     opt=0   ← Transpilación
```

Las preguntas factoriales clave:

| Factor A: Circuit | Factor B: Transpilation | Factor C: Mitigation | Config |
|:---:|:---:|:---:|:---:|
| p=1 directo | opt_level=2 | PEA | C5 |
| p=1 directo | opt_level=0 | Mitiq ZNE | C11 |
| p=1 directo | opt_level=0 | Mitiq CDR | C12 |
| AQC p=2→shallow | opt_level=2 | PEA | C16 |
| AQC p=2→shallow | opt_level=0 | Mitiq CDR | C17 |
| AQC p=2→shallow | opt_level=2 | None | C18 |

**Resultado esperado** (basado en datos existentes):
- Near h_c (3.0-3.25): C16 y C17 (AQC) deberían ganar por expresividad
- Deep paramagnetic (h>4.0): C5 debería ganar (p=1 ya suficiente, AQC overhead sin beneficio)
- El crossover h* define cuándo activar AQC en deployment

**Métricas clave para AQC×mitigación**:
- `aqc_expressivity_benefit` = ΔE/gap(p=1) - ΔE/gap(AQC) al mismo h
- `aqc_mitigation_interaction` = benefit(AQC+mitigación) - benefit(AQC alone) - benefit(mitigación alone)
  - Si > 0: interacción sinérgica (AQC + mitigación se amplifican mutuamente)
  - Si ≈ 0: efectos independientes (cada uno aporta su ganancia)
  - Si < 0: interferencia (AQC empeora la mitigación — ej. si circuito comprimido es menos "foldeable")


---

## 17. Guía de Implementación — Funciones Existentes a Reutilizar

**Principio**: La mayoría de los cálculos y métricas de este plan ya están
implementados en el codebase. El implementador debe REUTILIZAR, no reimplementar.

### 17.1 Métricas de circuito (Section 3.3) — YA EXISTEN

```python
# TODAS estas métricas se obtienen con UNA llamada:
from qmbp_simulation.analysis import transpiled_circuit_stats, compute_error_budget

stats = transpiled_circuit_stats(transpiled_circuit)
# → depth, depth_2q, n_2q_gates, n_1q_gates, total_gates, count_ops, width

budget = compute_error_budget(transpiled_circuit, backend=fake_backend, layout=layout)
# → error_budget, fidelity_estimate, per_gate_contribution, depth_2q
```

**Lo que falta agregar** (extensiones menores):

| Métrica nueva | Cómo implementar | Dónde agregar |
|---------------|-----------------|---------------|
| `n_swap_gates` | `(stats['n_2q_gates'] - n_2q_logical) // 3` | Cálculo trivial en el benchmark script |
| `routing_overhead_pct` | `(n_2q_transpiled - n_2q_logical) / n_2q_logical * 100` | Derivado, no necesita función |
| `idle_cycles_per_qubit` | Needs DAG analysis: `DAGCircuit.idle_wires()` | Extender `transpiled_circuit_stats()` |
| `max_idle_stretch` | DAG walk: longest consecutive identity on any qubit | Extender `transpiled_circuit_stats()` |
| `parallelism_ratio` | `stats['n_2q_gates'] / stats['depth_2q']` | Derivado |
| `circuit_duration_ns` | `transpiled.duration` (requiere scheduling pass) | Usar `generate_preset_pass_manager(scheduling_method='alap')` |
| `net_benefit` | `budget['fidelity_estimate'] * mitigation_gain` | Cálculo post-ejecución |

### 17.2 Layout selection — YA EXISTE

```python
from qmbp_simulation.analysis import rank_layouts_by_depth_2q, select_best_layout_for_zne
from qmbp_simulation.execution import compute_circuit_ces, select_layouts_low_ces

# Ranking de layouts por depth_2q:
rankings = rank_layouts_by_depth_2q(transpiled_circuits, layouts)

# CES (composite error score):
ces = compute_circuit_ces(transpiled_circuit, backend)

# Layout selection with mapomatic VF2 (si instalado):
from qmbp_simulation.execution.hardware.layout_optimizer import select_best_layouts
best = select_best_layouts(circuit, backend, n_layouts=3, strategy="lowest_cost")
```

**Lo que falta**: `layout_connectivity_score` (fracción de bonds nativos).
Implementar como utility en `circuit_visualizer.py`:
```python
def layout_connectivity_score(logical_edges, physical_coupling_map, layout):
    native = sum(1 for (a,b) in logical_edges if (layout[a], layout[b]) in physical_coupling_map)
    return native / len(logical_edges)
```

### 17.3 Error mitigation — YA EXISTE (configs C0-C10)

```python
# PEA-ZNE (configs C4-C8, C15):
from qmbp_simulation.execution import run_pea_zne, run_adaptive_zne
result = run_pea_zne(transpiled, H_mapped, backend, config)

# Gate-folding ZNE (configs C3, C9):
from qmbp_simulation.execution import run_gate_folding_zne
result = run_gate_folding_zne(transpiled, H_mapped, backend, config)

# Affine correction (todos):
from qmbp_simulation.execution import affine_correct_energy
corrected = affine_correct_energy(e_zne, e_ground, e_upper)

# GNN-QEM (config C9, C10):
from qmbp_simulation.predictors import correct_energy, load_qem_checkpoint

# Hardware backend (maneja todo internamente):
from qmbp_simulation.execution import HardwareBackend, HardwareConfig
backend = HardwareBackend(config=HardwareConfig(mitigation=MitigationOptions(...)))
result = backend.run_deployment(circuit, H, params, h_value=h, e_exact=e, gap=gap)
```

### 17.4 Mitiq mitigation — YA EXISTE (configs C11-C14)

```python
from qmbp_simulation.execution import (
    run_mitiq_zne,                   # C11
    run_mitiq_cdr,                   # C12, C14, C17
    run_mitiq_ddd_zne,               # C13
    compare_mitigation_strategies,    # Benchmark mode (todos a la vez)
    NoisyEstimatorConfig,
)

# Config para Mitiq (opt_level forced to 0 internally):
config = NoisyEstimatorConfig(shots=16384, seed_simulator=42)

# Comparison across methods (thesis table material):
result = compare_mitigation_strategies(
    circuit, H, backend, config,
    exact_energy=e_exact, gap=gap, h_value=h,
    strategies=["raw", "mitiq_zne_linear", "mitiq_cdr", "native_gf_zne", "native_pea_zne"],
)
```

### 17.5 AQC-Tensor compression — YA EXISTE (configs C16-C18)

```python
from qmbp_simulation.circuits.aqc_compression import (
    compress_circuit,        # Principal: p=2 bound → shallow
    AQCCompressionConfig,
)

config = AQCCompressionConfig(bond_dim=16, fidelity_threshold=0.998)
result = compress_circuit(bound_p2_circuit, config)
# result.compressed_circuit: QuantumCircuit (shallow)
# result.fidelity: float
# result.accepted: bool (fidelity >= threshold)
```

### 17.6 Benchmark execution — YA EXISTE (BenchmarkSuite)

```python
from qmbp_simulation.framework import BenchmarkSuite

# Benchmark mitiq performance:
suite = BenchmarkSuite(n_qubits=[4, 6, 10], n_repeats=3)
results = suite.run(components=["mitiq"])  # or ["solver", "vqe", "mitiq", "aqc"]
suite.print_summary(results)
```

### 17.7 Analysis & reporting — YA EXISTE

```python
# Mitiq analyzer (scans results/mitiq/ and section 21):
# python -m project_health.analysis.mitiq_analyzer --thesis-table --statistical

# AQC analyzer:
# python -m project_health.analysis.aqc_tensor_analyzer --thesis-table --statistical

# Health report (includes mitiq_status and aqc_status):
# python -m project_health --compact
```

### 17.8 Lo que SÍ hay que implementar nuevo (script principal del benchmark)

El único archivo nuevo necesario es el runner script:

```
scripts/experiment_runners/hardware/run_mitigation_benchmark.py
```

Responsabilidades:
1. Cargar configuración C0-C18 de un dict/JSON
2. Para cada config:
   - Build circuit (p=1 o AQC-compressed)
   - Set mitigation options según config
   - Execute via HardwareBackend.run_deployment() (C0-C10, C15-C18)
     OR via run_mitiq_*/compare_mitigation_strategies (C11-C14)
   - Collect `transpiled_circuit_stats()` + `compute_error_budget()` BEFORE execution
   - Save results with `benchmark_metadata` wrapper
3. Append to manifest.json
4. Print progress table

**Estructura interna sugerida**:
```python
@dataclass
class BenchmarkConfig:
    config_id: str
    dd_enabled: bool = False
    dd_sequence: str = "XpXm"
    twirling_randomizations: int = 0
    trex_enabled: bool = False
    zne_strategy: str = "off"  # "off"|"gf"|"pea"|"mitiq_zne"|"mitiq_cdr"|"mitiq_ddd"
    zne_noise_factors: list[float] | None = None
    pea_budget: tuple[int, int] = (48, 192)
    affine_enabled: bool = True
    gnn_qem_enabled: bool = False
    aqc_compress: bool = False
    mitiq_n_training: int = 10

BENCHMARK_CONFIGS: dict[str, BenchmarkConfig] = {
    "C0_raw": BenchmarkConfig("C0_raw"),
    "C5_full_pea_balanced": BenchmarkConfig(
        "C5_full_pea_balanced",
        dd_enabled=True, twirling_randomizations=48,
        trex_enabled=True, zne_strategy="pea",
    ),
    # ... etc
}
```

**Escalabilidad**: Agregar configs nuevas = agregar entrada al dict. Zero code changes.

---

## 18. Principios de Reutilización y Escalabilidad

### 18.1 NUNCA reimplementar lo que ya existe

| Necesitas... | USA esto | NO hagas esto |
|--------------|----------|---------------|
| Métricas de circuito | `transpiled_circuit_stats()` | `circuit.count_ops()` manual |
| Error budget | `compute_error_budget()` | Loops sobre gates × error_rates |
| ZNE native | `run_gate_folding_zne()` / `run_pea_zne()` | Implementar ZNE ad-hoc |
| ZNE Mitiq | `run_mitiq_zne()` | Importar mitiq directamente |
| CDR | `run_mitiq_cdr()` | Llamar `mitiq.cdr` raw |
| Comparar métodos | `compare_mitigation_strategies()` | Loops manuales |
| Layout selection | `select_best_layouts()` / `rank_layouts_by_depth_2q()` | BFS manual |
| JSON serialization | `json_dump()` / `json_serialize()` | `json.dump()` con default custom |
| Experiment result | `save_experiment_result()` | `json.dump()` directo |
| Experiment criteria | `EXPERIMENT_CRITERIA` + `compute_verdict()` | Dicts locales |

### 18.2 Extensión preferida sobre creación

Si necesitás una métrica nueva (ej: `idle_cycles_per_qubit`):

✅ **CORRECTO**: Extender `transpiled_circuit_stats()` en `circuit_visualizer.py`
```python
# En transpiled_circuit_stats(), agregar al final:
stats["idle_cycles_per_qubit"] = _compute_idle_cycles(circuit)
stats["max_idle_stretch"] = _compute_max_idle(circuit)
```

❌ **INCORRECTO**: Crear función separada en el benchmark script
```python
# NO: duplica lógica y no es reutilizable
def my_idle_cycles(circuit): ...
```

### 18.3 Formato de resultados: envelope pattern

Todos los resultados DEBEN usar el envelope pattern existente:

```python
from qmbp_simulation.framework.result_io import build_result_envelope

result = build_result_envelope(
    experiment_id="MITIGATION_BENCHMARK",
    config={...},
    results={...},
    metadata={"config_id": "C5", "execution_mode": "fake_backend"},
)
json_dump(result, output_path)
```

### 18.4 Principio de composición (no monolito)

El benchmark NO es un script monolítico de 2000 líneas.
Es una orquestación de funciones existentes:

```python
def run_single_config(config: BenchmarkConfig, h: float, ...) -> dict:
    """1 config × 1 h-point = 1 result."""
    # 1. Build circuit (reutiliza HVACircuitBuilder / aqc_compression)
    # 2. Transpile (reutiliza generate_preset_pass_manager)
    # 3. Collect pre-execution metrics (reutiliza transpiled_circuit_stats + compute_error_budget)
    # 4. Execute with mitigation (reutiliza run_pea_zne / run_mitiq_cdr / HardwareBackend)
    # 5. Collect results (reutiliza existing result fields)
    # 6. Return dict (JSON-serializable)
    ...

def run_benchmark(configs: list[str], h_values: list[float], mode: str) -> None:
    """Orchestrator: loops over configs × h_values, saves each result."""
    for config_id in configs:
        for h in h_values:
            result = run_single_config(BENCHMARK_CONFIGS[config_id], h, mode=mode)
            save_result(result)
    generate_manifest()
```

Cada `run_single_config` es <50 líneas porque DELEGA a funciones existentes.
