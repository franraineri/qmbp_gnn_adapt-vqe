# Resumen general — square tfim_frustrated

*Auto-generado: 2026-10-13 03:47 UTC. Regenerar con `generate_ansatz_summary.py --overview`.*

Vista única de todas las configuraciones (N, h) con artefactos. Por cada config se muestra la **frontera de Pareto** (fid vs 2q). El detalle completo vive en los JSON/specs — este markdown es una vista, no una fuente de datos.

| N | h | régimen | # sols | mejor fid | 2q@mejor | mejor fid/2q (Pareto) | 2q@fid/2q |
|--:|--:|---------|:------:|----------:|---------:|----------------------:|----------:|
| 6 | 0.50 | no-degen | 4 | 0.9995 | 54 | 1.998e-02 | 50 |
| 8 | 0.50 | no-degen | 7 | 0.9984 | 146 | 1.816e-02 | 52 |
| 10 | 0.50 | no-degen | 12 | 0.9985 | 194 | 2.666e-02 | 36 |
| 10 | 1.00 | no-degen | 8 | 0.8723 | 50 | 2.967e-02 | 26 |
| 12 | 3.00 | no-degen | 1 | 0.9952 | 51 | 1.951e-02 | 51 |
| 14 | 0.30 | degen | 5 | 0.1824 | 86 | 2.120e-03 | 86 |
| 14 | 0.50 | no-degen | 21 | 0.9619 | 200 | 1.521e-02 | 60 |
| 14 | 0.70 | no-degen | 11 | 0.9476 | 228 | 1.116e-02 | 40 |
| 16 | 3.00 | no-degen | 1 | 0.9941 | 72 | 1.381e-02 | 72 |
| 18 | 0.50 | no-degen | 34 | 0.9516 | 450 | 6.408e-03 | 128 |
| 18 | 0.70 | no-degen | 6 | 0.8932 | 318 | 3.908e-03 | 138 |
| 18 | 3.00 | no-degen | 1 | 0.9912 | 83 | 1.194e-02 | 83 |
| 20 | 0.50 | degen | 17 | 0.9919 | 524 | 5.185e-03 | 172 |
| 20 | 1.00 | no-degen | 14 | 0.9652 | 524 | 8.561e-03 | 62 |

## Notas

- **mejor fid**: mayor fidelidad alcanzada en esa config (y su conteo 2q).
- **mejor fid/2q**: solución más eficiente de la frontera de Pareto.
- El detalle por-ansatz (todas las variantes, θ, edges) está en `bond_ablation/*.json` y `ansatz_specs/*.spec.json` — no en markdown.
