# Plan de Análisis de Resultados — GNN-HVA Framework

**Fecha**: 2026-05-27
**Datos**: 135 noiseless, 60 noisy, 15 experimentos de hipótesis
**Herramienta**: `python -m scripts.digest`

## Estudios Realizados

| # | Estudio | Archivo | Conclusión principal |
|---|---------|---------|---------------------|
| 1 | Escalabilidad por topología | `01_topology_comparison.md` | Framework es topology-agnostic (median <5% en todas) |
| 2 | Robustez por semilla | `02_seed_robustness.md` | Varianza inter-seed < varianza inter-topología |
| 3 | Sensibilidad a hiperparámetros | `03_hyperparameter_sensitivity.md` | hidden=128 y restarts=5 son robustos cross-topology |
| 4 | Reconciliación de verdicts | `04_verdict_reconciliation.md` | 5 de 8 "failed" son en realidad passes con threshold estricto |
| 5 | Hallazgos negativos | `05_negative_findings.md` | 5 rejections = 5 contribuciones a la tesis |
| 6 | Frontera ZNE | `06_zne_boundary.md` | N=6 funciona (+48.5%), N=10 falla (-14.4%), p=1 es la clave |
| 7 | Outliers y casos extremos | `07_outliers.md` | 9 outliers, todos explicables (gen.gap o θ-sweep) |
