# Eje 2A — Tabla Comparativa Cross-Topología

## Resumen por Configuración (Topología × N)

**NOTA**: Esta tabla fue reemplazada por la versión corregida en `09_diagnostics_deep_dive.md`
que incluye 131 variants (vs 120 originales) al escanear directamente los pipeline_run files.

Ver `09_diagnostics_deep_dive.md` → "Tabla Completa Cross-Topología (Datos Corregidos)"

| Topología | N | Variants | PASS | MARGINAL | FAIL | Mejor ΔE/gap | Mediana ΔE/gap | Media ΔE/gap | Peor ΔE/gap | Pass Rate |
|-----------|---|----------|------|----------|------|-------------|---------------|--------------|-------------|-----------|
| chain_1d | 6 | 30 | 21 | 6 | 3 | 0.0148 | 0.0293 | 0.0841 | 0.8999 | 70% |
| ladder | 6 | 13 | 3 | 5 | 5 | 0.0040 | 0.0940 | 0.2276 | 1.8892 | 23% |
| ladder | 10 | 23 | 17 | 2 | 4 | 0.0010 | 0.0359 | 0.0541 | 0.1949 | 74% |
| triangular | 6 | 27 | 14 | 1 | 12 | 0.0024 | 0.0479 | 0.1109 | 0.7199 | 52% |
| triangular | 10 | 27 | 16 | 3 | 8 | 0.0002 | 0.0384 | 0.6786 | 14.4009 | 59% |

## Hallazgos Clave


### N=6

- **chain_1d**: mediana=0.0293, pass=21/30
- **ladder**: mediana=0.0940, pass=3/13
- **triangular**: mediana=0.0479, pass=14/27

### N=10

- **ladder**: mediana=0.0359, pass=17/23
- **triangular**: mediana=0.0384, pass=16/27
