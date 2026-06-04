# Analysis — GNN-HVA Framework

Resultados de análisis cross-experiment y figuras para la tesis.

## Ubicación de Scripts

> **Los scripts activos han sido movidos a `project_health/analysis/`.**
> Los shims en `analysis/scripts/` preservan backward compatibility con
> invocaciones existentes (steering files, protocolos, etc.).

| Script | Ubicación canónica | Shim (backward-compat) |
|--------|-------------------|------------------------|
| `scan_coverage.py` | `project_health/analysis/scan_coverage.py` | `analysis/scripts/scan_coverage.py` |
| `diagnose.py` | `project_health/analysis/diagnose.py` | `analysis/scripts/diagnose.py` |
| `verify_claims.py` | `project_health/analysis/verify_claims.py` | `analysis/scripts/verify_claims.py` |
| `validate_s_series.py` | `project_health/analysis/validate_s_series.py` | `analysis/scripts/validate_s_series.py` |
| `heisenberg_summary.py` | `project_health/analysis/heisenberg_summary.py` | `analysis/scripts/heisenberg_summary.py` |
| `compare.py` | `project_health/compare.py` | `scripts/compare.py` |

## Workflow de Análisis (orden recomendado)

```
┌─────────────────────────────────────────────────────────────────────┐
│  1. INVENTARIO          →  scan_coverage.py                         │
│     "¿Qué datos tenemos y qué falta?"                              │
├─────────────────────────────────────────────────────────────────────┤
│  2. DIAGNÓSTICO         →  diagnose.py                              │
│     "¿Por qué fallaron estos runs?"                                 │
├─────────────────────────────────────────────────────────────────────┤
│  3. VERIFICACIÓN        →  verify_claims.py                         │
│     "¿Los claims son robustos?"                                     │
├─────────────────────────────────────────────────────────────────────┤
│  4. FIGURAS             →  python -m project_health.figures          │
│     "Generar visualizaciones para la tesis"                         │
└─────────────────────────────────────────────────────────────────────┘
```

## Invocación (ambas rutas funcionan)

```bash
# Canonical (new)
python -m project_health.analysis.scan_coverage --discover --extended
python -m project_health.analysis.diagnose --all
python -m project_health.analysis.verify_claims
python -m project_health.compare --all

# Legacy (shims, still work)
python analysis/scripts/scan_coverage.py --discover --extended
python analysis/scripts/diagnose.py --all
python analysis/scripts/verify_claims.py
python scripts/compare.py --all
```

## Scripts Principales

### 1. `scan_coverage.py` — Inventario y Gap Analysis

**Propósito**: Escanea TODOS los resultados (pipeline, noisy, experiments) y
produce un inventario completo con análisis de gaps.

```bash
python analysis/scripts/scan_coverage.py --discover
python analysis/scripts/scan_coverage.py --topology chain_1d --n-qubits 10 --p 1
python analysis/scripts/scan_coverage.py --extended
python analysis/scripts/scan_coverage.py --json analysis/raw_data/coverage.json
```

### 2. `diagnose.py` — Diagnóstico Automatizado de Failures

**Propósito**: Clasifica la causa raíz de cada failure en categorías accionables.

```bash
python analysis/scripts/diagnose.py --all
python analysis/scripts/diagnose.py results/thesis/p1_variants_N10_r2
python analysis/scripts/diagnose.py --all --severity fail
python analysis/scripts/diagnose.py --all --json analysis/raw_data/diagnoses.json
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

### 3. `verify_claims.py` — Verificación de Robustez

Cross-check de claims contra datos crudos.

```bash
python analysis/scripts/verify_claims.py
```

### 4. Figuras (via project_health)

```bash
# Full generalized tool (recommended)
python -m project_health.figures --help
python -m project_health.figures --source analysis --only gen_gap_vs_de_gap smoothness_histogram

# Legacy shim (generates original 4 figures in analysis/figures/)
python analysis/scripts/generate_figures.py
```

## Deprecated Scripts (in `analysis/scripts/`)

Los siguientes scripts ya completaron su propósito y pueden ser removidos:

| Script | Estado | Resultado capturado en |
|--------|--------|----------------------|
| `09_diagnostics_deep_dive.py` | Completado | `analysis/09_diagnostics_deep_dive.md` |
| `run_analysis.py` | Superseded | `project_health/` + `digest/` |
| `run_p1_zne_multiseed.py` | Completado (9 runs) | `analysis/verification/` |
| `step1a_p1_zne_validation.py` | Completado | `analysis/11_p1_zne_verification.md` |
| `step2a_error_decomposition.py` | Completado | `documentation/analysis/` |
| `step2c_smoothness_correlation.py` | Completado | `documentation/analysis/` |
| `generate_figures.py` | Legacy shim | `project_health.figures` |
| `analyze_s_series.py` | Superseded | `validate_s_series.py` |
| `verify_depth_scaling.py` | Completado | Heisenberg p≤2 limit documented |
| `verify_heisenberg_sanity.py` | Completado | Expressibility limit confirmed |

## Datos y Documentos

### Datos

| Archivo | Contenido |
|---------|-----------|
| `raw_data/all_variants.json` | Execution log data (186 records) |
| `raw_data/all_diagnostics.json` | Pipeline diagnostics (131+ records) |
| `raw_data/coverage.json` | Coverage scan structured data |
| `raw_data/diagnoses.json` | Failure diagnoses structured |

### Figuras

| Archivo | Uso en tesis |
|---------|--------------|
| `figures/fig_01_gen_gap_vs_de_gap.png` | Cap. 5 |
| `figures/fig_02_smoothness_histogram.png` | Cap. 4 |
| `figures/fig_03_cross_topology_bar.png` | Cap. 5 |
| `figures/fig_04_smoothness_vs_de_gap.png` | Cap. 5 |

## Workflow Post-Experimento

```bash
# 1. Verificar qué se guardó
python analysis/scripts/scan_coverage.py --discover

# 2. Diagnosticar failures
python analysis/scripts/diagnose.py --all --severity fail

# 3. Verificar claims
python analysis/scripts/verify_claims.py

# 4. Report completo
python -m project_health

# 5. Exportar para la tesis
python analysis/scripts/scan_coverage.py --discover --extended --json analysis/raw_data/coverage.json
python analysis/scripts/diagnose.py --all --json analysis/raw_data/diagnoses.json
```
