# Estudio 2A — Error Decomposition por Topología

**Pregunta**: ¿El error viene del HVA (circuit) o del MPNN (prediction)?

## Datos

### chain_1d (n=30)

| Metric | Value |
|--------|-------|
| Median error_from_circuit | 0.00000 |
| Median error_from_mpnn | 0.03387 |
| Median MPNN fraction | 100.0% |
| Mean error_from_circuit | 0.00000 |
| Mean error_from_mpnn | 0.08440 |

Worst MPNN errors:
- ext_extrapolation: circuit=0.0000, mpnn=1.0358
- nl_restarts_3: circuit=0.0000, mpnn=0.1436
- nl_p1: circuit=0.0000, mpnn=0.1169

### ladder (n=47)

| Metric | Value |
|--------|-------|
| Median error_from_circuit | 0.00000 |
| Median error_from_mpnn | 0.08264 |
| Median MPNN fraction | 100.0% |
| Mean error_from_circuit | 0.00000 |
| Mean error_from_mpnn | 0.23098 |

Worst MPNN errors:
- nl_grid_sparse3: circuit=0.0000, mpnn=4.8020
- ext_near_hc: circuit=0.0000, mpnn=0.6618
- nl_seed_44: circuit=0.0000, mpnn=0.5043

### triangular (n=54)

| Metric | Value |
|--------|-------|
| Median error_from_circuit | 0.00000 |
| Median error_from_mpnn | 0.10101 |
| Median MPNN fraction | 100.0% |
| Mean error_from_circuit | 0.00000 |
| Mean error_from_mpnn | 0.89068 |

Worst MPNN errors:
- nl_seed_42: circuit=0.0000, mpnn=33.8225
- nl_restarts_7: circuit=0.0000, mpnn=2.2793
- nl_p1_triangular: circuit=0.0000, mpnn=2.0455

### unknown (n=3)

| Metric | Value |
|--------|-------|
| Median error_from_circuit | 0.00000 |
| Median error_from_mpnn | 0.09074 |
| Median MPNN fraction | 100.0% |
| Mean error_from_circuit | 0.00000 |
| Mean error_from_mpnn | 1.16904 |

Worst MPNN errors:
- ext_extrapolation: circuit=0.0000, mpnn=3.3425
- ext_p1_boundary: circuit=0.0000, mpnn=0.0907
- ext_ultrasparse3: circuit=0.0000, mpnn=0.0739

## Resumen Comparativo

| Topología | n | Med circuit error | Med MPNN error | MPNN fraction |
|-----------|---|-------------------|----------------|---------------|
| chain_1d | 30 | 0.00000 | 0.03387 | 100.0% |
| ladder | 47 | 0.00000 | 0.08264 | 100.0% |
| triangular | 54 | 0.00000 | 0.10101 | 100.0% |
| unknown | 3 | 0.00000 | 0.09074 | 100.0% |

## Conclusiones

- **100% del error es MPNN prediction** en todas las topologías y todos los runs.
- `error_from_circuit = 0.0` significa que el VQE encontró el mínimo exacto del HVA
  (la energía VQE = energía exacta dentro del ansatz). El HVA p=2 NO es el bottleneck
  en el valid regime (h≥2.0 para ladder/triangular).
- **El MPNN es el único componente que puede mejorar** — mejor arquitectura, más datos,
  o mejor training podrían reducir ΔE/gap.
- **Ranking de dificultad MPNN**: chain_1d (med=0.034) < ladder (0.083) < triangular (0.101).
  El MPNN tiene más dificultad con grafos más complejos.

## Implicación para la Tesis

> "Energy decomposition analysis reveals that 100% of the deployment error
> originates from MPNN prediction, not HVA expressibility. The VQE consistently
> finds the global minimum within the ansatz (error_from_circuit = 0), confirming
> that the HVA p=2 is not the bottleneck in the valid regime. Future improvements
> should focus on MPNN architecture and training, not circuit depth."
