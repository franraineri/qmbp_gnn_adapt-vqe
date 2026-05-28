# Eje 5 — Resultados Negativos y Anomalías

## 5A. Catálogo de Rechazos Justificados (V8)

| Exp | Hipótesis | Resultado | Aprendizaje |
|-----|-----------|-----------|-------------|
| E4 | HVA es model-agnostic | ❌ Fid=0.89 con g=0.1 | HVA es TFIM-specific |
| F1 | DyPP ahorra 30-50% | ❌ Solo 8-13% | Warm-start ya near-optimal |
| G2 | Ensemble UQ calibrado | ❌ r=0.195 | Necesita bootstrap |
| G3 | N=6 findings → N=20 | ❌ ΔE/gap=1.26 | Landscape cambia con N |
| G4 | κ predice restarts | ❌ r=-0.29 | h-value es mejor predictor |
| C1@N=10 | Physics loss mejora | ❌ -12.3% | Solo ayuda con h-range completo |

## 5B. Anomalías Detectadas en Variant Runs

| Topología | N | Variant | ΔE/gap | Anomalía |
|-----------|---|---------|--------|----------|
| ladder | 6 | NL-A7 | 0.1186 | 7 restarts FALLA (más restarts debería ser mejor) |
| ladder | 6 | NL-C-sparse3 | 1.8892 | Catastrófico (>100%) |
| triangular | 6 | NL-A1 | 0.0030 | 1 restart excelente en topología compleja |
| triangular | 10 | NL-A7 | 0.9705 | 7 restarts FALLA (más restarts debería ser mejor) |
| triangular | 10 | NL-E-seed42 | 14.4009 | Catastrófico (>100%) |
| triangular | 10 | EXT-2-near-boundary | 1.0904 | Catastrófico (>100%) |

### Análisis de Anomalías

1. **N10_triangular seed=42 (ΔE/gap=14.4)**: Catastrófico. Probable warm-start chain break — el VQE encontró un mínimo local completamente incorrecto. Seed 43 y 44 funcionan bien → seed-dependent failure.
2. **N6_triangular NL-A5 FAIL pero NL-A1 PASS**: Contraintuitivo. Posible explicación: más restarts con σ grande pueden 'saltar' fuera del buen basin encontrado por el warm-start. El warm-start es tan bueno que restarts adicionales PERJUDICAN.
3. **N10_triangular NL-A7 FAIL (0.97)**: Mismo fenómeno que #2 amplificado. 7 restarts con σ grande destruyen la buena inicialización del warm-start.
