# Analysis Toolkit — GNN-HVA Framework

Herramientas de análisis para el proyecto GNN-HVA. Cada script tiene una
responsabilidad clara y se ejecuta en un orden lógico.

## Workflow de Análisis (orden recomendado)

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. INVENTARIO          →  scan_coverage.py                         │
│     "¿Qué datos tenemos y qué falta?"                              │
├─────────────────────────────────────────────────────────────────────┤
│  2. DIAGNÓSTICO         →  diagnose.py                              │
│     "¿Por qué fallaron estos runs?"                                 │
├─────────────────────────────────────────────────────────────────────┤
│  3. ANÁLISIS PROFUNDO   →  09_diagnostics_deep_dive.py              │
│     "¿Qué correlaciones existen en los datos?"                      │
├─────────────────────────────────────────────────────────────────────┤
│  4. VERIFICACIÓN        →  verify_claims.py                         │
│     "¿Los claims son robustos?"                                     │
├─────────────────────────────────────────────────────────────────────┤
│  5. FIGURAS             →  generate_figures.py                       │
│     "Generar visualizaciones para la tesis"                         │
└─────────────────────────────────────────────────────────────────────┘
```

## Scripts Principales

### 1. `scan_coverage.py` — Inventario y Gap Analysis

**Propósito**: Escanea TODOS los resultados (pipeline, noisy, experiments) y
produce un inventario completo con análisis de gaps.

```bash
# Scan básico (folders hardcoded)
python analysis/scan_coverage.py

# Auto-descubrir todos los folders bajo results/thesis/
python analysis/scan_coverage.py --discover

# Filtrar por topología, N, o p
python analysis/scan_coverage.py --topology chain_1d --n-qubits 10 --p 1

# Extended analytics (reproducibilidad, staleness, h-coverage, calidad)
python analysis/scan_coverage.py --extended

# Exportar a JSON/CSV/Markdown
python analysis/scan_coverage.py --json analysis/raw_data/coverage.json
python analysis/scan_coverage.py --csv analysis/raw_data/coverage.csv
python analysis/scan_coverage.py --markdown analysis/raw_data/coverage.md

# Modo silencioso (solo resumen)
python analysis/scan_coverage.py --quiet
```

**Output**: Identifica gaps en cobertura p=1 vs p=2, seed coverage, valid regime
violations, y produce recomendaciones priorizadas.

**Cuándo usarlo**: Antes de planificar nuevos experimentos. Después de ejecutar
un batch de runs para verificar que se guardaron correctamente.

---

### 2. `diagnose.py` — Diagnóstico Automatizado de Failures

**Propósito**: Lee los pipeline_run JSON en profundidad y clasifica la causa raíz
de cada failure en categorías accionables.

```bash
# Diagnosticar todos los failures en thesis results
python analysis/diagnose.py --all

# Diagnosticar un folder específico
python analysis/diagnose.py results/thesis/p1_variants_N10_r2

# Solo failures severos (skip MARGINAL)
python analysis/diagnose.py --all --severity fail

# Filtrar por topología o p
python analysis/diagnose.py --all --topology triangular --p 1

# Exportar diagnósticos a JSON
python analysis/diagnose.py --all --json analysis/raw_data/diagnoses.json

# Incluir runs que pasaron (para comparación)
python analysis/diagnose.py --all --show-passing
```

**Root causes clasificados**:
| Causa | Threshold | Detectable en |
|-------|-----------|---------------|
| CHAIN_BREAK | θ_smoothness > 1.0 | Phase 2 |
| MPNN_OVERFIT | gen_gap > 0.01 | Phase 3 |
| HVA_LIMIT | error_from_circuit > 0.01 | Phase 4 |
| OUTSIDE_REGIME | h_test < valid boundary | Pre-run |
| VQE_DIVERGENCE | convergence_rate < 1.0 | Phase 2 |
| BOUNDARY_EFFECT | h_test within 0.5 of boundary | Pre-run |

**Cuándo usarlo**: Después de que un batch de experimentos produce failures.
Para entender si re-ejecutar con diferentes parámetros o si es un límite físico.

---

### 3. `09_diagnostics_deep_dive.py` — Correlaciones y Tabla Definitiva

**Propósito**: Scan directo de TODOS los pipeline_run files para producir:
- Correlación θ_smoothness vs ΔE/gap
- Correlación gen_gap vs ΔE/gap
- Descomposición de error (circuit vs MPNN)
- Tabla cross-topología definitiva (131+ variants)

```bash
python analysis/09_diagnostics_deep_dive.py
```

**Output**: `analysis/09_diagnostics_deep_dive.md` + `raw_data/all_diagnostics.json`

**Cuándo usarlo**: Cuando se agregan nuevos datos y se necesita actualizar las
correlaciones y tablas definitivas.

---

### 4. `verify_claims.py` — Verificación de Robustez

**Propósito**: Cross-check de claims contra datos crudos. Verifica que las
afirmaciones en los documentos de análisis son consistentes con los datos.

```bash
python analysis/verify_claims.py
```

**Cuándo usarlo**: Antes de escribir la tesis. Después de correcciones mayores.

---

### 5. `generate_figures.py` — Figuras Thesis-Quality

**Propósito**: Genera las 4 figuras principales para el capítulo de resultados.

```bash
python analysis/generate_figures.py
```

**Output**: `analysis/figures/fig_01-04.png`

---

## Scripts Secundarios

| Script | Propósito | Cuándo usar |
|--------|-----------|-------------|
| `run_analysis.py` | Parsea execution logs → tablas markdown | Después de variant runner |
| `run_p1_zne_multiseed.py` | Ejecuta p=1 ZNE con 3 seeds | Ya completado (9 runs) |
| `step1a_p1_zne_validation.py` | Validación ZNE single-topology | Ya completado |
| `step2a_error_decomposition.py` | Error decomposition por topología | Ya completado |
| `step2c_smoothness_correlation.py` | Correlación θ vs ΔE/gap | Ya completado |

## Documentos de Resultados

### Fuentes Canónicas (usar para la tesis)

| Archivo | Contenido | Actualizado |
|---------|-----------|-------------|
| `../documentation/analysis/08_summary.md` | Resumen completo + thesis statements | 2026-05-30 |
| `../documentation/analysis/09_thesis_tables.md` | Tables 5.1–5.10 definitivas | 2026-05-30 |
| `10_key_findings_corrected.md` | 8 hallazgos clave verificados | 2026-05-30 |
| `FINDINGS_INDEX.md` | Índice maestro (36 hallazgos) | 2026-05-28 |

### Documentos de Análisis

| Archivo | Contenido |
|---------|-----------|
| `00_executive_summary.md` | Resumen ejecutivo (186 variants) |
| `02_reproducibility_analysis.md` | Cross-seed reproducibility |
| `03_hyperparameter_sensitivity.md` | Hidden dim, grid, restarts |
| `04_zne_failure_confirmation.md` | ZNE failure + p=1 finding |
| `05_negative_results_catalog.md` | 6 rechazos + anomalías |
| `08_lessons_learned.md` | Lecciones + restart paradox |
| `09_diagnostics_deep_dive.md` | Correlaciones (gen_gap, smoothness) |
| `11_p1_zne_verification.md` | Multi-seed triangular |
| `13_p1_zne_all_topologies.md` | p=1 ZNE cross-topology (8/9 seeds) |
| `thesis_chapter_results.md` | Draft del capítulo de resultados |

### Datos

| Archivo | Contenido | Registros |
|---------|-----------|-----------|
| `raw_data/all_variants.json` | Execution log data | 186 |
| `raw_data/all_diagnostics.json` | Pipeline diagnostics (scan directo) | 131+ |
| `raw_data/coverage.json` | Coverage scan structured data | 247+ |
| `raw_data/diagnoses.json` | Failure diagnoses structured | 76 |
| `verification/p1_zne_multiseed/` | p=1 ZNE multi-seed results | 3 |

### Figuras

| Archivo | Contenido | Uso en tesis |
|---------|-----------|--------------|
| `figures/fig_01_gen_gap_vs_de_gap.png` | Scatter: predictor de failure | Cap. 5 |
| `figures/fig_02_smoothness_histogram.png` | Chain breaks por topología | Cap. 4 |
| `figures/fig_03_cross_topology_bar.png` | Pass rate comparison | Cap. 5 |
| `figures/fig_04_smoothness_vs_de_gap.png` | Threshold effect | Cap. 5 |

## Nivel de Confianza (actualizado 2026-05-30)

| Claim | Confianza | Evidencia |
|-------|-----------|-----------|
| Framework topology-agnostic (64%) | ★★★ | 131+ variants, 5 topologías |
| Warm-start = contribución central | ★★★ | 5 evidencias independientes |
| ZNE falla N=10 p=2 | ★★★ | 33 runs consistentes |
| p=1 ZNE funciona (all topologies) | ★★★ | 8/9 seeds, 3 topologías |
| p=1 pipeline funciona N=10 | ★★★ | 9/9 PASS (R2, chain+ladder+tri) |
| gen_gap predice failure | ★★★ | 131 variants, 89% vs 15% |
| Error 100% MPNN (régimen válido) | ★★★ | 131 variants decomposition |
| CHAIN_BREAK = 45% de failures | ★★★ | 174 runs diagnosed |
| p=1 más consistente que p=2 | ★★☆ | COMP-4: std 0.002 vs 0.47 |
| N=16 requiere MPS | ★★★ | 13 runs, Phase 3 no completa |

## Ejemplo: Workflow Completo Post-Experimento

```bash
# 1. Ejecutar experimentos
python scripts/experiment_runners/run_p1_pipeline_variants_r2.py

# 2. Verificar qué se guardó
python analysis/scan_coverage.py --discover

# 3. Diagnosticar failures
python analysis/diagnose.py --all --severity fail

# 4. Actualizar correlaciones (si hay datos nuevos significativos)
python analysis/09_diagnostics_deep_dive.py

# 5. Verificar claims
python analysis/verify_claims.py

# 6. Regenerar figuras
python analysis/generate_figures.py

# 7. Exportar todo para la tesis
python analysis/scan_coverage.py --discover --extended --json analysis/raw_data/coverage.json
python analysis/diagnose.py --all --json analysis/raw_data/diagnoses.json
```
