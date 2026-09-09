# Chequeos pendientes de la tesis

Generado: 2026-09-08 23:34 UTC  
Fuente: `generate_thesis_tables.py --check-tex`  
Total pendientes: **11** (auto-arreglables 0 · manuales 11 · a verificar 0)

Orden recomendado: primero **AUTO-FIX** (banderas del script), luego **MANUAL** (criterio humano), por último **VERIFICAR** (confirmar que son falsos positivos y suprimirlos).

## Requieren criterio humano  (11)

### `orden-label` · EDITORIAL · 3
_Mover la primera \ref antes del objeto, o reordenar (cosmético)._

- [ ] [tesis-v4.0.tex:1047] tabla \label{tab:tfim_long} aparece (L1047) antes de su primera mención \ref (L1365).
- [ ] [tesis-v4.0.tex:1137] tabla \label{tab:unsupervised_detection} aparece (L1137) antes de su primera mención \ref (L1152).
- [ ] [tesis-v4.0.tex:1248] tabla \label{tab:comparison_literature} aparece (L1248) antes de su primera mención \ref (L1363).

### `otros` · INFO · 7
_Revisar manualmente._

- [ ] heavy_hex N=14: la grilla canónica de extrapolación [2.5, 3.0, 3.5, 4.0, 4.5, 5.0] no cubre [3.5, 4.0, 4.5] en el régimen válido (h_frontier=1,99); tabla con menos puntos para ese N (INFO, no error).
- [ ] heavy_hex N=18: la grilla canónica de extrapolación [2.5, 3.0, 3.5, 4.0, 4.5, 5.0] no cubre [3.0, 3.5, 4.0, 4.5] en el régimen válido (h_frontier=2,37); tabla con menos puntos para ese N (INFO, no error).
- [ ] heavy_hex N=21: la grilla canónica de extrapolación [2.5, 3.0, 3.5, 4.0, 4.5, 5.0] no cubre [4.5] en el régimen válido (h_frontier=4,08); tabla con menos puntos para ese N (INFO, no error).
- [ ] heavy_hex N=22: la grilla canónica de extrapolación [2.5, 3.0, 3.5, 4.0, 4.5, 5.0] no cubre [4.5] en el régimen válido (h_frontier=4,08); tabla con menos puntos para ese N (INFO, no error).
- [ ] heavy_hex N=32: la grilla canónica de extrapolación [2.5, 3.0, 3.5, 4.0, 4.5, 5.0] no cubre [3.0, 3.5, 4.0, 4.5] en el régimen válido (h_frontier=N/A); tabla con menos puntos para ese N (INFO, no error).
- [ ] campaña: modelo 'xy' (7 ejecuciones) excluido de la tabla de conteo por decisión de la tesis (base fina, fuera de la narrativa). Si se quiere incorporar, quitarlo de CAMPAIGN_EXCLUDED_MODELS y añadir párrafo + fila de viabilidad.
- [ ] heavy_hex N=32 (tab:auto_heavy_hex_large_n): ΔE/gap medio=88.7\% es media de cocientes por punto, muy por encima del cociente de medias (27.8\%); dominado por puntos de gap estrecho. Verificar que el pie de tabla lo aclare (INFO, no error).

### `revision-manual` · INFO · 1
_Aviso informativo; revisar a mano._

- [ ] [tesis-v4.0.tex] INFO: las 6 hipótesis (H1--H6) se mencionan en Conclusiones; revisar manualmente que cada respuesta sea concluyente (§22).
