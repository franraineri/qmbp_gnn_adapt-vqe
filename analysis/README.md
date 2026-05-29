# Análisis Comparativo — GNN-HVA Framework

Resultados del análisis sistemático de 186 variantes experimentales (131 con diagnostics completos).
Cada claim fue verificado contra datos crudos (`verify_claims.py`).

## Contenido

### Documentos principales
- `FINDINGS_INDEX.md` — **Índice maestro** (36 hallazgos con nivel de confianza)
- `00_executive_summary.md` — Resumen ejecutivo (actualizado 2026-05-28)
- `thesis_chapter_results.md` — **Draft del capítulo de resultados** (11 tablas)

### Análisis por eje
- `02_reproducibility_analysis.md` — Reproducibilidad cross-seed (Eje 1A)
- `03_hyperparameter_sensitivity.md` — Sensibilidad de hiperparámetros (Eje 3)
- `04_zne_failure_confirmation.md` — ZNE failure N=10 p=2 (Eje 4)
- `05_negative_results_catalog.md` — Catálogo de rechazos + anomalías (Eje 5)

### Hallazgos profundos
- `08_lessons_learned.md` — Lecciones, correcciones, next steps completados
- `09_diagnostics_deep_dive.md` — **Correlaciones** (gen_gap, smoothness vs ΔE/gap)
- `10_key_findings_corrected.md` — Hallazgos post-verificación (5 hallazgos clave)
- `11_p1_zne_verification.md` — Multi-seed triangular (2/3 seeds)
- `13_p1_zne_all_topologies.md` — **p=1 ZNE cross-topology** (8/9 seeds positivos)

### Worklog (archivos movidos — datos supersedidos o sesiones de trabajo)
- `worklog/01_cross_topology_table_SUPERSEDED.md` — Reemplazado por `09_diagnostics_deep_dive.md`
- `worklog/06_implementation_metrics_OLD120.md` — Datos de 120 variants (ahora son 131)
- `worklog/07_methodology_validation_OLD120.md` — Datos de 120 variants (ahora son 131)
- `worklog/12_session_summary_WORKLOG.md` — Log de sesión 2026-05-28

### Figuras
- `figures/fig_01_gen_gap_vs_de_gap.png` — Scatter: predictor de failure
- `figures/fig_02_smoothness_histogram.png` — Chain breaks por topología
- `figures/fig_03_cross_topology_bar.png` — Pass rate comparison
- `figures/fig_04_smoothness_vs_de_gap.png` — Threshold effect

### Scripts
- `scan_coverage.py` — **Coverage scanner** (identifies gaps in p=1 vs p=2 data)
- `run_analysis.py` — Generador principal (execution logs → tablas)
- `09_diagnostics_deep_dive.py` — Scan directo de pipeline files (131 variants)
- `verify_claims.py` — Verificación de robustez de claims
- `generate_figures.py` — Generador de figuras thesis-quality
- `run_p1_zne_multiseed.py` — Verificación experimental p=1 ZNE
- `step1a_p1_zne_validation.py` — Validación p=1 ZNE single-topology
- `step2a_error_decomposition.py` — Descomposición de error por topología
- `step2c_smoothness_correlation.py` — Correlación θ-smoothness vs ΔE/gap

### Datos
- `raw_data/all_variants.json` — 186 registros (execution logs)
- `raw_data/all_diagnostics.json` — 131 registros (pipeline files, scan directo)
- `raw_data/coverage.json` — Structured coverage data (all result types)
- `verification/p1_zne_multiseed/` — Resultados multi-seed

## Generación

```bash
# Coverage scan — identify what simulations are needed next
python analysis/scan_coverage.py
python analysis/scan_coverage.py --json analysis/raw_data/coverage.json

# Análisis completo (execution logs)
python analysis/run_analysis.py

# Diagnostics deep dive (scan directo de pipeline files)
python analysis/09_diagnostics_deep_dive.py

# Verificar robustez
python analysis/verify_claims.py

# Generar figuras
python analysis/generate_figures.py
```

## Nivel de Confianza (actualizado 2026-05-28)

| Claim | Confianza | Razón |
|-------|-----------|-------|
| Framework topology-agnostic (64%) | ★★★ | 131 variants, 5 topologías |
| Warm-start = contribución central | ★★★ | 5 evidencias independientes |
| ZNE falla N=10 p=2 | ★★★ | 7 resultados consistentes |
| **p=1 ZNE funciona (all topologies)** | **★★★** | **8/9 seeds, 3 topologías** |
| gen_gap predice failure | ★★★ | 131 variants, 89% vs 15% |
| Error 100% MPNN (régimen válido) | ★★★ | 131 variants decomposition |
| Hyperparams irrelevantes (N=10) | ★★★ | Spread <2% |
| Restart paradox (mecanismo) | ★★★ | Diagnostics confirman |
| Triangular seed-dependent | ★★☆ | Outlier-driven (1/3) |
| Ladder N=6: 50% chain breaks | ★★★ | 22 variants |
