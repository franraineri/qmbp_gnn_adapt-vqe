# Estudio 2C — Correlación θ-smoothness vs ΔE/gap

**Pregunta**: ¿θ-smoothness predice la calidad del pipeline sin ejecutar Phase 4?

## Correlación Global

- **Pearson r (linear)**: 0.2779
- **Pearson r (log-log)**: 0.4550
- **n**: 134

→ **Correlación débil**: θ-smoothness NO es un buen predictor lineal de ΔE/gap.

## Análisis por Banda de θ-smoothness

| Banda | n | Median ΔE/gap | Mean ΔE/gap | Pass rate (<5%) |
|-------|---|---------------|-------------|-----------------|
| < 0.05 | 93 | 0.0286 | 0.0576 | 75/93 (81%) |
| 0.05–1.0 | 7 | 0.1570 | 0.1426 | 2/7 (29%) |
| ≥ 1.0 | 34 | 0.1112 | 0.2421 | 8/34 (24%) |

## Correlación por Topología

| Topología | n | Pearson r | Interpretación |
|-----------|---|-----------|----------------|
| chain_1d | 38 | 0.6158 | fuerte |
| ladder | 45 | -0.0101 | débil |
| triangular | 49 | 0.5113 | moderada |

## Conclusiones

1. θ-smoothness como **detector de problemas**: valores > 1.0 casi siempre
   indican warm-start roto (pass rate mucho menor en esa banda).
2. Como **predictor cuantitativo** de ΔE/gap: correlación débil-moderada.
   No reemplaza Phase 4, pero sirve como early-warning.
3. **Regla práctica**: Si θ-smoothness > 1.0, investigar antes de confiar en Phase 4.
