# Eje 1A — Reproducibilidad Cross-Seed por Topología

## Resultados de Seeds (42, 43, 44) por Topología

| Topología | N | Seed 42 ΔE/gap | Seed 43 ΔE/gap | Seed 44 ΔE/gap | Std | Seed-Independent? |
|-----------|---|----------------|----------------|----------------|-----|-------------------|
| chain_1d | 6 | 0.0286 | 0.0351 | 0.0282 | 0.0039 | ✅ |
| ladder | 6 | 0.0940 | 0.0812 | 0.1984 | 0.0643 | ❌ |
| triangular | 6 | 0.1541 | 0.0026 | 0.1433 | 0.0845 | ❌ |
| ladder | 10 | 0.0402 | 0.0166 | 0.0297 | 0.0118 | ✅ |
| triangular | 10 | 14.4009 | 0.0419 | 0.0372 | 8.2916 | ❌ |

## Análisis

- **chain_1d**: Esperado seed-independent (confirmado en G5: std=0.004)
- **ladder**: Verificar si la conectividad adicional introduce varianza
- **triangular**: Alta varianza esperada por frustración geométrica

## Implicación para la Tesis

La reproducibilidad degrada con la conectividad del grafo. Topologías frustradas (triangular) requieren más restarts para garantizar resultados seed-independent.
