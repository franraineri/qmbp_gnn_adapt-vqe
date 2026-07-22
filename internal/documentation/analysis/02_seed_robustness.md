# Estudio 2 — Robustez por Semilla y Reproducibilidad

**Pregunta**: ¿Los resultados son seed-independent?

## Datos de Semillas (Ladder N=10)

De los resultados en `variants_N10_ladder`:

| Variante | Seed | ΔE/gap | Topology |
|----------|------|--------|----------|
| nl_seed_42 | 42 | 0.0402 | ladder |
| nl_seed_43 | 43 | 0.0166 | ladder |
| nl_seed_44 | 44 | 0.0297 | ladder |

- **Mean**: 0.0288
- **Std**: 0.0097
- **Max spread**: 0.0236 (factor 2.4× entre mejor y peor)
- **Todos pasan** (< 5%)

## Comparación: Varianza Inter-Seed vs Inter-Topología

| Fuente de variación | Spread (max-min) | Std |
|---------------------|------------------|-----|
| Seeds (ladder N=10) | 0.024 | 0.010 |
| Topologías (median) | 0.011 (0.029–0.040) | 0.005 |
| Hidden dim (ladder) | 0.032 (0.034–0.066) | — |
| Restarts (ladder) | 0.063 (0.034–0.097) | — |

## Conclusión

1. **El pipeline ES seed-independent**: Todos los seeds pasan el criterio de 5%.
2. **La varianza inter-seed (std=0.010) es comparable a la varianza inter-topología (std=0.005)**.
3. **Los hiperparámetros (restarts, hidden_dim) tienen más impacto que el seed**.
4. **G5 experiment (92% pass rate)** confirma esto a escala: solo 1/12 puntos falla marginalmente.

## Implicación para la Tesis

> "The pipeline is reproducible across random seeds (std(ΔE/gap) = 0.010 at N=10).
> Seed variance is smaller than the effect of hyperparameter choices (restarts,
> hidden_dim), confirming that results reflect systematic pipeline behavior
> rather than stochastic initialization luck."
