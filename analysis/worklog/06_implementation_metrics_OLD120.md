# Eje 6 — Métricas de Implementación

> **NOTA**: Los pass rates en este archivo provienen del primer análisis (execution log matching, 120 variants).
> Para datos definitivos, ver `09_diagnostics_deep_dive.md` y `10_key_findings_corrected.md` (131 variants, scan directo).

## 6A. Costo Computacional

| Carpeta | Topología | N | Variants | Tiempo Total | Tiempo/Variant |
|---------|-----------|---|----------|--------------|----------------|
| variants_N6_N10_1D_linnear | chain_1d | 6 | 46 | 1533s (25.6min) | 33.3s |
| variants_N6_ladder | ladder | 6 | 33 | 11782s (196.4min) | 357.0s |
| variants_N6_triangular | triangular | 6 | 37 | 1205s (20.1min) | 32.6s |
| variants_N10_ladder | ladder | 10 | 33 | 2085s (34.7min) | 63.2s |
| variants_N10_triangular | triangular | 10 | 37 | 5616s (93.6min) | 151.8s |

**Total**: 186 variants, 22222s (6.2h)

## 6B. Tasa de Éxito del Framework

| Métrica | Valor |
|---------|-------|
| Total variants ejecutados | 186 |
| Errores de ejecución (crashes/timeouts) | 2 |
| Tasa de ejecución exitosa | 98.9% |
| Tiempo total de cómputo | 6.2 horas |

## 6C. Distribución de Veredictos (Noiseless)

| Topología | N | PASS | MARGINAL | FAIL | Pass Rate |
|-----------|---|------|----------|------|-----------|
| chain_1d | 6 | 21 | 6 | 3 | 70% |
| ladder | 6 | 3 | 5 | 5 | 23% |
| triangular | 6 | 14 | 1 | 12 | 52% |
| ladder | 10 | 17 | 2 | 4 | 74% |
| triangular | 10 | 16 | 3 | 8 | 59% |
