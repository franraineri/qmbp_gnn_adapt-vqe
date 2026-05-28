# Índice Maestro de Hallazgos — GNN-HVA Framework

**Última actualización**: 2026-05-28
**Estado**: Todos los claims verificados, datos completos, p=1 ZNE confirmado multi-seed.

---

## Hallazgos Confirmados (★★★ — escribir con confianza)

| # | Hallazgo | Evidencia | Documentado en |
|---|----------|-----------|----------------|
| 1 | Framework topology-agnostic (64% pass rate, 5 topologías) | 131 variants, 5 configs | `thesis_chapter_results.md` §5.1 |
| 2 | Warm-start = contribución central (93-99.9% gain) | 5 evidencias independientes | `thesis_chapter_results.md` §5.2 |
| 3 | ZNE falla a N=10 p=2 (6/7 negative gain) | 7 resultados consistentes | `thesis_chapter_results.md` §5.7.1 |
| 4 | p=1 ZNE funciona a N=10 (8/9 seeds across 3 topologies) | Multi-seed, multi-topology | `13_p1_zne_all_topologies.md` |
| 5 | gen_gap > 1e-2 → 85% failure rate | 131 variants con diagnostics | `10_key_findings_corrected.md` #1 |
| 6 | θ_smoothness > 1.0 → chain break (76% fail/marginal) | 131 variants | `10_key_findings_corrected.md` #2 |
| 7 | Error es 100% MPNN en régimen válido (circuit=0) | 131 variants decomposition | `10_key_findings_corrected.md` #3 |
| 8 | chain_1d seed-independent (std=0.004) | 3 seeds | `02_reproducibility_analysis.md` |
| 9 | ladder N=10 seed-independent (std=0.012) | 3 seeds | `02_reproducibility_analysis.md` |
| 10 | hidden_dim irrelevante a N=10 (spread <2%) | 3 topologías | `03_hyperparameter_sensitivity.md` |
| 11 | 7 grid points suficiente (todas topologías) | 5 configs | `03_hyperparameter_sensitivity.md` |
| 12 | Implementación robusta (98.8%, 186 variants) | Execution logs | `06_implementation_metrics.md` |
| 13 | Restart paradox: mecanismo = basin switching → θ discontinuo | Diagnostics (smoothness+gen_gap) | `08_lessons_learned.md` §2.1 |
| 14 | Ladder N=6: 50% chain breaks (peor config) | 22 variants | `10_key_findings_corrected.md` #4 |
| 15 | Early-stopping detecta 69% de failures | 131 variants | `10_key_findings_corrected.md` #5 |
| 16 | No barren plateaus (F3: fluctuation >1.0) | V8 experiment | `project-status.md` |
| 17 | Hessian: 0 saddle points en HVA (B4) | V8 experiment, N=6+N=10 | `project-status.md` |
| 18 | Data efficiency: 9 points sufficient (G1) | V8 experiment | `project-status.md` |
| 19 | Pipeline seed-independent (G5: std=0.004) | V8 experiment | `project-status.md` |
| 20 | Scaling law: h_min = 1.0 + 0.020·N^1.31 (A3) | V8 experiment | `project-status.md` |

## Hallazgos con Calificación (★★☆ — escribir con caveat)

| # | Hallazgo | Caveat | Documentado en |
|---|----------|--------|----------------|
| 21 | Triangular N=10 seed-dependent (std=8.29) | Driven by 1 outlier (344× next) | `02_reproducibility_analysis.md` |
| 22 | Restart paradox es probabilístico | Depende del seed, no determinístico | `08_lessons_learned.md` §2.1 |
| 23 | h=128 crítico a N=6 | Menos datos que N=10 | `03_hyperparameter_sensitivity.md` |
| 24 | p=1 ZNE variabilidad es layout-dependent | 1/3 seeds falla por alto CES | `11_p1_zne_verification.md` |

## Resultados Negativos (★★★ — publicables)

| # | Hipótesis rechazada | Resultado | Documentado en |
|---|---------------------|-----------|----------------|
| 25 | HVA es model-agnostic (E4) | ❌ TFIM-specific | `05_negative_results_catalog.md` |
| 26 | DyPP ahorra 30-50% (F1) | ❌ Solo 8-13% | `05_negative_results_catalog.md` |
| 27 | Ensemble UQ calibrado (G2) | ❌ r=0.195 | `05_negative_results_catalog.md` |
| 28 | N=6 → N=20 transfer (G3) | ❌ ΔE/gap=1.26 | `05_negative_results_catalog.md` |
| 29 | κ predice restarts (G4) | ❌ r=-0.29 | `05_negative_results_catalog.md` |
| 30 | Physics loss @N=10 (C1) | ❌ -12.3% | `05_negative_results_catalog.md` |
| 31 | Analytical init (B1) | ❌ Wrong basin | `thesis_chapter_results.md` §5.9 |

## Implementaciones Realizadas

| # | Implementación | Archivo | Tests |
|---|----------------|---------|-------|
| 32 | Early-stopping (θ_smoothness + gen_gap) | `src/.../pipeline/runner.py` | 324 pass ✅ |
| 33 | `select_layouts_low_ces()` | `src/.../execution/noisy_utils.py` | 324 pass ✅ |
| 34 | 4 figuras thesis-quality | `analysis/figures/fig_01-04.png` | N/A |
| 35 | Diagnostics deep dive script | `analysis/09_diagnostics_deep_dive.py` | N/A |
| 36 | Multi-seed verification script | `analysis/run_p1_zne_multiseed.py` | N/A |

## Datos Disponibles

| Archivo | Contenido | Registros |
|---------|-----------|-----------|
| `raw_data/all_variants.json` | Execution log data | 186 |
| `raw_data/all_diagnostics.json` | Pipeline diagnostics (scan directo) | 131 |
| `verification/p1_zne_multiseed/multiseed_results.json` | p=1 ZNE 3-seed | 3 |

---

## Resumen en Una Frase

> El framework GNN-HVA es topology-agnostic (64% pass rate, 131 variants, 5 topologías),
> con warm-start como contribución central (93-99.9% gain), gen_gap como mejor predictor
> de failure (89% pass si <1e-4, 15% si >1e-2), y p=1 ZNE confirmado cross-topology para
> hardware deployment (8/9 seeds positivos, mean gain +49%, best +81%).
