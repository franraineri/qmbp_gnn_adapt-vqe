# Eje 1 — Validación Metodológica

> **NOTA**: Estadísticas basadas en primer análisis (120 variants). Para datos definitivos (131 variants),
> ver `09_diagnostics_deep_dive.md` → "Tabla Completa Cross-Topología".

## 1B. Análisis del Criterio ΔE/gap < 5%

- **Total puntos noiseless**: 120
- **Mediana global**: 0.0375
- **Media global**: 0.2337
- **Percentil 25**: 0.0276
- **Percentil 75**: 0.1174
- **% que pasan (<0.05)**: 59%
- **% marginales (0.05-0.10)**: 14%
- **% que fallan (>0.10)**: 27%

### Por Topología

| Topología | N | % PASS | % MARGINAL | % FAIL | Mediana |
|-----------|---|--------|------------|--------|---------|
| chain_1d | 6 | 70% | 20% | 10% | 0.0293 |
| ladder | 6 | 23% | 38% | 38% | 0.0940 |
| ladder | 10 | 74% | 9% | 17% | 0.0359 |
| triangular | 6 | 52% | 4% | 44% | 0.0479 |
| triangular | 10 | 59% | 11% | 30% | 0.0384 |

## 1C. Validación del Warm-Start Descendente

### Evidencia del warm-start como contribución central

| Evidencia | Fuente | Resultado |
|-----------|--------|-----------|
| Gain 93-99.9% vs random init | Comparative Analysis #1 | Warm-start = toda la propuesta de valor |
| Sin warm-start → 843× peor | Comparative Analysis #3 (ablation) | Componente más importante |
| 1 restart suficiente en chain/ladder | Variant runs NL-A1 | Warm-start tan bueno que restarts son marginales |
| SPSA refinement HURTS warm-start | V7 4B: -146% | No refinar buenas predicciones |
| DyPP solo 8-13% mejora | V8 F1 | Warm-start ya near-optimal |

### Implicación

El warm-start descendente (h=2→0) con predicción MPNN es la contribución
metodológica central de la tesis. Todas las demás optimizaciones (restarts,
grid density, hidden dim) son marginales en comparación. El framework funciona
porque el MPNN aprende la estructura suave del landscape θ(h) y proporciona
inicializaciones que están dentro del basin of attraction del mínimo global.
