# Análisis de Correlación: theta_smoothness vs ΔE/gap

**Pregunta**: ¿Podemos predecir el fracaso del pipeline desde Phase 2 sin ejecutar Phase 3+4?

**Datos**: 131 variants con ambas métricas

## Distribución por Umbral de Smoothness

| Rango θ_smoothness | N | Pass (<5%) | Marginal | Fail (>10%) | Pass Rate | Mediana ΔE/gap |
|-------------------|---|------------|----------|-------------|-----------|----------------|
| < 0.05 (excelente) | 92 | 74 | 10 | 8 | 80% | 0.0286 |
| 0.05 - 0.10 (bueno) | 1 | 0 | 0 | 1 | 0% | 0.4290 |
| 0.10 - 1.0 (sospechoso) | 5 | 2 | 0 | 3 | 40% | 0.1570 |
| > 1.0 (chain break) | 33 | 8 | 8 | 17 | 24% | 0.1067 |

## Correlación

- **Pearson r** = -0.004 (débil)
- **Spearman ρ** = 0.415 (moderada)

## Regla de Decisión Propuesta

```
IF theta_smoothness > 1.0:
    ABORT Phase 3+4 (warm-start chain broke)
    ACTION: reduce restarts or increase h-grid density
ELIF theta_smoothness > 0.10:
    WARNING: elevated risk of MPNN failure
    ACTION: check gen_gap after Phase 3
ELSE:
    PROCEED normally
```

## Validación de la Regla

- θ_smoothness ≥ 1.0: 33 cases, 52% fail rate → regla es parcial
- θ_smoothness < 0.05: 92 cases, 80% pass rate → buen predictor de éxito
---

# Análisis de Correlación: generalization_gap vs ΔE/gap

**Pregunta**: ¿El gen_gap de Phase 3 predice el resultado de Phase 4?

**Datos**: 131 variants con ambas métricas

| Rango gen_gap | N | Pass | Marginal | Fail | Pass Rate |
|---------------|---|------|----------|------|-----------|
| < 1e-4 (excelente) | 55 | 49 | 4 | 2 | 89% |
| 1e-4 - 1e-3 (bueno) | 26 | 20 | 3 | 3 | 77% |
| 1e-3 - 1e-2 (sospechoso) | 30 | 12 | 8 | 10 | 40% |
| > 1e-2 (overfitting) | 20 | 3 | 3 | 14 | 15% |

**Pearson r** = 0.287
---

# Descomposición del Error: Circuito vs MPNN

**Pregunta**: ¿Cuánto error viene del HVA (expresibilidad) vs del MPNN (predicción)?

**Datos**: 131 variants con descomposición

| Topología | N | N pts | Mean Circuit Error | Mean MPNN Error | % Circuit | Bottleneck |
|-----------|---|-------|--------------------|-----------------|-----------|------------|
| chain_1d | 6 | 30 | 0.0000 | 0.0844 | 0% | MPNN |
| ladder | 6 | 22 | 0.0000 | 0.3867 | 0% | MPNN |
| ladder | 10 | 25 | 0.0000 | 0.0940 | 0% | MPNN |
| triangular | 6 | 27 | 0.0000 | 0.2423 | 0% | MPNN |
| triangular | 10 | 27 | 0.0000 | 1.5391 | 0% | MPNN |

## Interpretación

- Si Circuit >> MPNN: el HVA no puede expresar el ground state (límite físico, no mejorable con ML)
- Si MPNN >> Circuit: el predictor es el cuello de botella (mejorable con más datos/epochs/capacity)
- Si balanced: ambos contribuyen — mejora requiere ambos frentes
---

# Distribución de Diagnósticos por Topología

| Topología | N | N pts | Med. Smoothness | Med. Gen Gap | Med. ΔE/gap | Chain Breaks (>1.0) |
|-----------|---|-------|-----------------|--------------|------------|---------------------|
| chain_1d | 6 | 30 | 0.0325 | 7.14e-05 | 0.0293 | 2/30 |
| ladder | 6 | 22 | 1.5845 | 4.67e-03 | 0.0604 | 11/22 |
| ladder | 10 | 25 | 0.0358 | 6.70e-05 | 0.0341 | 4/25 |
| triangular | 6 | 27 | 0.0222 | 2.39e-03 | 0.0316 | 10/27 |
| triangular | 10 | 27 | 0.0210 | 7.62e-05 | 0.0376 | 6/27 |
---

# Tabla Completa Cross-Topología (Datos Corregidos)

Incluye TODOS los pipeline results encontrados (resuelve el problema de datos faltantes).

| Topología | N | Total Variants | PASS | MARGINAL | FAIL | Mejor | Mediana | Peor | Pass Rate |
|-----------|---|----------------|------|----------|------|-------|---------|------|-----------|
| chain_1d | 6 | 30 | 21 | 6 | 3 | 0.0148 | 0.0293 | 0.8999 | 70% |
| ladder | 6 | 22 | 11 | 5 | 6 | 0.0004 | 0.0812 | 1.8892 | 50% |
| ladder | 10 | 25 | 19 | 3 | 3 | 0.0010 | 0.0341 | 0.1787 | 76% |
| triangular | 6 | 27 | 16 | 2 | 9 | 0.0010 | 0.0316 | 0.3384 | 59% |
| triangular | 10 | 27 | 17 | 2 | 8 | 0.0002 | 0.0376 | 14.4009 | 63% |

**Total**: 131 variants con datos, 84/131 (64%) pasan ΔE/gap < 5%
