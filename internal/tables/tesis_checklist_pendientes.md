# Chequeos pendientes de la tesis

Generado: 2026-09-30 14:34 UTC  
Fuente: `generate_thesis_tables.py --check-tex`  
Total pendientes: **7** (auto-arreglables 0 · manuales 7 · a verificar 0)

Orden recomendado: primero **AUTO-FIX** (banderas del script), luego **MANUAL** (criterio humano), por último **VERIFICAR** (confirmar que son falsos positivos y suprimirlos).

## Requieren criterio humano  (7)

### `cifra-epoca` · IMPORTANTE · 1
_Verificar que no se mezclen épocas de campaña._

- [ ] chain_1d: resultados de campañas de distintos meses (2026-08-17..2026-09-09); verificar que no se mezclen criterios/rangos de h en una misma tabla.

### `otros` · INFO · 6
_Revisar manualmente._

- [ ] heavy_hex N=14: la grilla canónica de extrapolación [2.5, 3.0, 3.5, 4.0, 4.5, 5.0] no cubre [3.5, 4.0, 4.5] en el régimen válido (h_frontier=1,99); tabla con menos puntos para ese N (INFO, no error).
- [ ] heavy_hex N=18: la grilla canónica de extrapolación [2.5, 3.0, 3.5, 4.0, 4.5, 5.0] no cubre [3.0, 3.5, 4.0, 4.5] en el régimen válido (h_frontier=2,37); tabla con menos puntos para ese N (INFO, no error).
- [ ] heavy_hex N=21: la grilla canónica de extrapolación [2.5, 3.0, 3.5, 4.0, 4.5, 5.0] no cubre [4.5] en el régimen válido (h_frontier=4,08); tabla con menos puntos para ese N (INFO, no error).
- [ ] heavy_hex N=22: la grilla canónica de extrapolación [2.5, 3.0, 3.5, 4.0, 4.5, 5.0] no cubre [4.5] en el régimen válido (h_frontier=4,08); tabla con menos puntos para ese N (INFO, no error).
- [ ] heavy_hex N=32: la grilla canónica de extrapolación [2.5, 3.0, 3.5, 4.0, 4.5, 5.0] no cubre [3.0, 3.5, 4.0, 4.5] en el régimen válido (h_frontier=N/A); tabla con menos puntos para ese N (INFO, no error).
- [ ] heavy_hex N=32 (tab:auto_heavy_hex_large_n): ΔE/gap medio=88.7\% es media de cocientes por punto, muy por encima del cociente de medias (27.8\%); dominado por puntos de gap estrecho. Verificar que el pie de tabla lo aclare (INFO, no error).
