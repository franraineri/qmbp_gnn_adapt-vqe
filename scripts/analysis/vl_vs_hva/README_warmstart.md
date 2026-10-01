# Warm-start en los runners — patrón de integración canónico

**REGLA OBLIGATORIA (desde 2026-10)**: TODA convergencia de circuito en CUALQUIER
runner del estudio — la referencia/full Y cada variante masked/podada — debe
obtener su punto de arranque con **una sola llamada** a `prepare_warmstart`
(`qmbp_simulation.framework.study_core`). Nunca converger desde θ=0 (cold start)
ni desde un seed analítico pelado. No reimplementar la lógica de seeds,
micro-descent ni selección de candidatos en cada script. Persistir
`ws["provenance"]` como `seed_kind` de la fila para trazabilidad.

Runners ya conformes: `run_bond_topk_regime.py`, `run_variant_topk.py` (full +
masked), `run_bond_ablation.py` (ref + masked). Un runner nuevo DEBE seguir el
mismo patrón.

## La función

```python
from qmbp_simulation.framework.study_core import prepare_warmstart

ws = prepare_warmstart(
    qc, H, psi,                      # circuito, Hamiltoniano, ground state exacto
    n_nn=n_nn, n_nnn=n_nnn, n_qubits=N, p_layers=p, h=h, J2=j2,
    donors=donors,                   # opcional: θ convergidos previos (incl. cross-N)
    target_nnn_edges=nnn_edges,      # para alinear donors por bond
    strategy="combined",             # "combined" (cascada completa) | "regime"
    micro_descent=12,                # iters de L-BFGS para rankear candidatos
)
seed  = ws["seed"]            # el θ warm-start (ya refinado por micro-descent)
prov  = ws["provenance"]      # qué candidato ganó (p.ej. "calibrated+descent")
init  = ws["init_fidelity"]  # su fidelidad tras micro-descent
report = ws["report"]        # todos los candidatos + fids, para auditoría/row
# ws["_fid"], ws["_grad"], ws["_cost"] se reutilizan para la optimización
```

## Qué hace (los 4 pasos, encapsulados)

1. Arma `cost`/`fid`/`grad` del circuito (o los reutiliza si se los pasás).
2. Construye un micro-descent corto (L-BFGS) para rankear por **cuenca**, no por
   init-fid crudo.
3. Llama `best_combined_warmstart`: candidatos = seed **calibrado**
   (`calibrated_warmstart_theta`) + seed de **régimen** (`select_regime_seed`) +
   **donors transferidos** (incl. continuación cross-N vía `transfer_theta`) +
   `extra_candidates` que inyectes.
4. Devuelve el mejor seed + provenance + report.

## Estrategias

- `combined` (default): la cascada completa. Úsalo salvo que necesites un A/B.
- `regime`: sólo el seed de régimen (comportamiento pre-cascada), para comparar.

## Reglas para runners nuevos

- **No** llamar `first_order_warmstart_theta` / `second_order_*` / `select_regime_seed`
  directamente para armar el arranque: eso es trabajo de `prepare_warmstart`.
- Donors: cargalos como lista de dicts
  `{theta, n_nn, n_nnn, p, label, nnn_edges?, n_qubits?}`. El `n_qubits` habilita
  continuación cross-N (θ_x rellenado con la media N-invariante del donante).
- Persistir `ws["provenance"]` y `ws["report"]` en el row del resultado para
  trazabilidad (qué seed ganó y con qué fidelidad cada candidato).
- Escalar el presupuesto de optimización con `difficulty_index(N, h, gap)` /
  `budget_for_difficulty(...)` — más restarts sólo cerca de h_c.

## Referencia

Implementación de referencia ya migrada: `run_bond_topk_regime.py` (`_eval_regime`).
Núcleo: `qmbp_simulation.analysis.warmstart.best_combined_warmstart`.
