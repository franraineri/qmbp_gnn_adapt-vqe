# Traspaso: simetría, ansatz cortos y clusters tilted PBC

Estado al cierre temporal de la sesión. Este documento es el punto de entrada: indica qué se hizo,
qué quedó cerrado, qué está abierto y dónde está cada artefacto. El detalle físico y las tablas
están en `docs/research/symmetry_frustrated_vqe_findings.md` (secciones 1–11).

**Regla de precedencia:** si una memoria (`mem_*`) o un log contradice este documento o el de
hallazgos, estos dos ganan. Varias memorias de la sesión (410, 411, 412) usan lenguaje más fuerte
del que los datos justifican ("definitivo", "límite físico", "refutado"); la sección 9 del doc de
hallazgos tiene la lectura corregida.

## Objetivo

Obtener ansatz de pocas compuertas de 2 qubits (2q) y alta fidelidad para el square TFIM frustrado
(J1-J2, J2=0.5, h=0.5), explotando la simetría de la red. Métrica: fidelidad al GS exacto (al
subespacio si el GS es degenerado). El 2q es el conteo transpilado de `cx_and_params`.

## Resumen en cinco líneas

1. En OBC, atar parámetros por simetría (tying) falla a N ≥ 16 por degeneración y ruptura espontánea
   de simetría. Esto se cerró en sesiones previas (mem 388–393) y por sym-breaking layers (mem 396–401).
2. En N=20 OBC, ninguna técnica probada sobre los specs existentes mejoró la frontera conocida
   (0.8988 @148, 0.9491 @176, 0.9939 @524) con el presupuesto usado.
3. Los clusters **tilted con PBC** (N = L1²+L2²) dan simetría completa y GS no degenerado en N=16 y 18,
   algo imposible en OBC. Es un modelo físico distinto: no comparable con el scoreboard OBC.
4. En tilted, el ansatz tied (13 parámetros a p=4) llega a 0.9960 en N=9; en N=16 solo con el θ de N=9
   como semilla **y optimizando fidelidad** (no energía) se llega a 0.7951 @512 2q.
5. Reducción de 2q en N=16: omitir el bloque nnn de la capa 0 ahorra 64 CX (512 → 448) por −0.007 de
   fidelidad (0.7881). Una sola corrida, sin variabilidad entre semillas medida. Abierto y más
   prometedor: subir p (5/6) en N=16.

## Estado de cada experimento

| Experimento                                   | Estado            | Resultado clave                                           | Artefacto principal                                              |
|-----------------------------------------------|-------------------|-----------------------------------------------------------|------------------------------------------------------------------|
| Bloque RY symmetry-breaking (N=9/12/16/18)    | Cerrado, negativo | Δ ≈ 0 (±0.003): redundante con bond-resolved             | `bond_ablation/sb_layers_ab_square_N*_h0.50.json`                |
| Lectura ADAPT del scoreboard                  | Cerrado (lectura) | N=14 competitivo, N=20 colapsa; no se corrió nada nuevo  | scoreboard 2d_frustrated                                         |
| Gate tilted (simetría + degeneración)         | Cerrado, positivo | N=9/16/18 simétricos y no degenerados                     | `probe_tilted_symmetry.py`                                       |
| Auditoría de simetría del runner              | Cerrado, positivo | GS de N=16 invariante; tied sin techo estructural        | mem 404                                                          |
| Semillas analíticas en tied tilted            | Cerrado           | La calibrada gana (N=9: 0.32, N=16: 0.065)                | `probe_tilted_seeds.py`                                          |
| Warm-start N=9 (tied vs per-bond)             | Cerrado, positivo | tied 0.9047 vs per-bond 0.6327 (p=2)                      | `bond_ablation/tilted_warmstart_N9_h0.50.json`                   |
| Pareto 2q N=9                                 | Cerrado, positivo | 0.9960 @288 (p=4, 13 params); nnn irrenunciables          | `bond_ablation/tilted_pareto_2q_N9_h0.50.json`                   |
| Pareto N=16 con semilla calibrada             | Cerrado, negativo | Se estanca (máx 0.3031 en p=2)                            | `bond_ablation/tilted_pareto_2q_N16_h0.50.json`                  |
| Transfer N=9→16/18 + objetivo fidelidad       | Cerrado, positivo | N=16 p=4: 0.7951; N=18 p=4: 0.5796                        | `bond_ablation/tilted_transfer_N9to{16,18}_h0.50.json`           |
| Campaña nocturna (presupuesto masivo)         | Cerrado           | 0.7951 idéntico con 1000 y 6000 iter: saturado a p fijo   | `poolA_logs/OVERNIGHT_tilted_*.log`                              |
| Drop de bloques nnn en N=16 p=4               | Cerrado, positivo | Drop L0: 0.7881 @448 (−64 CX, −0.007); más drops degradan | `bond_ablation/tilted_drop_nnn_N16_h0.50.json`                   |
| N=14 prune desde spec                         | Cerrado, positivo | Nuevo punto 0.9021 @98                                    | `bond_ablation/variant_topk_p2_half_nn_rx_square_N14_h0.50.json` |
| N=20 topk 2/3/4 de 46 nnn                     | Cerrado           | topk3 0.9373 @266, dominado; no monótono                  | `ansatz_specs/p3_half_nn_rx_topk{2,3,4}of46_fg_*.spec.json`      |
| N=20 clusters geométricos                     | **No concluyente**| 0.04–0.12, pero estructura y presupuesto no comparables   | `bond_ablation/n20_geo_cluster_square_N20_h0.50.json`            |
| N=20 refine `stored` desde θ                  | Cerrado           | Sin cambio (±0.0001) con 2×20 iter                        | `bond_ablation/refine_pruned_square_N20_h0.50.json`              |

## Dónde está cada cosa

Código nuevo en `src/` (con tests):

- `src/qmbp_simulation/models/tilted_square.py` — `generate_square_tilted`, `tilted_for_n`, `MAGIC_TILTS`.
  Tests: `tests/unit/test_tilted_square.py` (44 pasan).
- Bloque `"y"` en `circuits/hva.py` y `analysis/warmstart.py`, variantes `p{1,2}_half_nn_rx_y` en
  `circuits/hva_variants.py`. Tests: `TestSymmetryBreakingYBlock` en `tests/unit/test_hva_session_infra.py`.

Scripts (en `scripts/analysis/vl_vs_hva/`, **sin tests**; se importan entre sí, mantenerlos juntos):

- `probe_tilted_symmetry.py`, `probe_tilted_seeds.py`, `probe_tilted_ceiling.py` — diagnósticos.
- `run_tilted_tied_ab.py` — A/B inicial con cold-start (superado). **Aún exporta helpers usados por
  los demás:** `build_frustrated_h`, `_tilted_lattice`, `_expand_tied_to_perbond`.
- `run_tilted_warmstart.py` (exporta `_tile_tied`), `run_tilted_pareto_2q.py`,
  `run_tilted_transfer.py` (contiene `optimize_fidelity`, `_load_src_tied_theta`),
  `run_tilted_drop_nnn.py`, `run_tilted_overnight.sh`.
- `run_n20_geo_cluster.py`, `run_sb_layers_ab.py`.

Datos: caché de manifold en `data/manifold_cache/` (los tilted usan la clave `square_tilted_*`, **no**
`square_*`). Doc de hallazgos y este traspaso en `docs/research/`.

Memorias: **no confiar en ellas.** Las memorias 388 a 416 de esta línea (proyecto `/`) ya no son
alcanzables desde la instancia actual: una búsqueda por su contenido (por ejemplo "ruptura espontánea")
no devuelve nada. Solo existe la memoria de cierre (`topic_key: symmetry-tilted-handoff`, id 1, proyecto
`qmbp_gnn_adapt-vqe`). Varias de las memorias perdidas contenían afirmaciones sobrevendidas que este
documento corrige. Todo lo que importa está en este archivo, en el doc de hallazgos y en la bitácora
`internal/documentation/binnacles/binnacle-session-20261116-symmetry-tilted.md`.

## Comandos verificados

Todos se ejecutaron en esta sesión con estos flags. Se corre siempre desde la raíz con `.venv/bin/python`.

```bash
# Gate de simetría y degeneración
.venv/bin/python scripts/analysis/vl_vs_hva/probe_tilted_symmetry.py --h 0.5 --j2 0.5 --ns 8 10 16 18 20

# N=9: Pareto tied (genera el θ que usa el transfer; debe correrse antes)
.venv/bin/python scripts/analysis/vl_vs_hva/run_tilted_pareto_2q.py --n 9 --h 0.5 --j2 0.5 --families nn_only full --p-max 4 --base-restarts 3 --maxiter 1000 --target-fid 0.999

# Transfer N=9 → N=16, energía vs fidelidad
.venv/bin/python scripts/analysis/vl_vs_hva/run_tilted_transfer.py --src-n 9 --dst-n 16 --h 0.5 --j2 0.5 --p-layers 3 4 --objective both --restarts 3 --maxiter 1000 --target-fid 0.999

# Reducción de 2q (requiere el JSON de transfer N9to16 con la fila fidelity p=4)
.venv/bin/python scripts/analysis/vl_vs_hva/run_tilted_drop_nnn.py --n 16 --h 0.5 --restarts 3 --maxiter 1500

# Campaña nocturna completa (~1 h 45 min)
bash scripts/analysis/vl_vs_hva/run_tilted_overnight.sh

# N=20 OBC: poda nnn extrema desde el spec, 2 restarts x 20 iter
.venv/bin/python scripts/analysis/vl_vs_hva/run_variant_topk.py --n 20 --h 0.5 --variant p3_half_nn_rx --reference-from-spec --rank-by fidelity-grad --keep-fracs 0.043 0.065 0.087 --nn-keep-frac 1.0 --restarts 2 --maxiter 20 --target-fid 0.98 --ftol 1e-9 --no-sync

# Tests del generador tilted
.venv/bin/python -m pytest tests/unit/test_tilted_square.py -q
```

## Trampas conocidas (todas costaron tiempo o produjeron un resultado erróneo)

- **Los JSON se sobrescriben, no acumulan.** `StudyPersister`/`save_json` reescribe el archivo en cada
  corrida. `run_variant_topk.py` pisa su JSON: pasar todas las tolerancias en una sola invocación.
  `tilted_transfer_N9to16_h0.50.json` contiene solo las filas p=4 de la última corrida; los valores
  de p=3 (0.6707) están únicamente en `poolA_logs/OVERNIGHT_tilted_N16_fidelity.log`.
- **`_load_src_tied_theta(N, ...)` lee `tilted_pareto_2q_N{N}`.** Para N=16 devuelve el θ del barrido
  calibrado fallido (fid 0.10, también de 13 parámetros), no el del transfer. Causó la primera
  corrida inválida de `drop_nnn`. Ese runner ahora lee el JSON de transfer y aborta si el baseline
  no arranca ≈0.795.
- **`detect_symmetry_group` corta en `max_syms=16`** por defecto: en clusters PBC reporta un orden
  espurio de 17. Pasar `max_syms=100000`.
- **Clave de la caché de manifold:** usar `square_tilted` para los clusters PBC. Con `square` se
  leería el manifold OBC del mismo N (p. ej. N=16 OBC es degenerado, e0=−16.25; el tilted es
  no degenerado, e0=−17.13).
- **Sin `--force`, `run_tilted_pareto_2q.py` salta las estructuras ya registradas y corta la
  cascada p→p+1** (el siguiente p arranca de la semilla calibrada). Para N=16 usar `--force`.
- **`prepare_warmstart(extra_candidates=...)` espera tuplas `(theta, etiqueta)`**, no arrays.
- **Optimizar energía degrada una buena semilla a N grande** en tilted; usar `--objective fidelity`.
- `list_processes` muestra más de 100 terminales "running" fantasma de sesiones previas. Para saber
  qué corre de verdad: `pgrep -fl "scripts/analysis/vl_vs_hva/run_"`. macOS no tiene `timeout`.
- Está prohibido `grep`/`find` en este repo (regla del workspace): usar graphify o leer rutas conocidas.

## Estado del repositorio

**Nada de esta sesión está commiteado.** El árbol tiene cambios modificados y archivos sin seguimiento
(scripts, tests, doc, specs nuevos, caché de manifold). El commit o PR queda a decisión del usuario.

`git status` muestra **14 scripts `run_*_queue.sh` / `run_*overnight.sh` borrados** en
`scripts/analysis/vl_vs_hva/`. El handoff de la cascada de warm-start (sección 9) los atribuye a esta
línea de simetría y N=20 y no los revisó. Esta sesión no los borró; no pude verificar el motivo.
Confirmar que la baja es intencional antes de commitear; si no, recuperarlos con
`git checkout -- <archivo>`.

Dos JSON siguen en `status: partial` por haber matado la corrida a propósito (el brazo per-bond
p=2 de N=16 era demasiado caro y poco informativo): `tilted_tied_ab_N16_h0.50.json` y
`tilted_warmstart_N16_h0.50.json`. Son datos válidos pero incompletos; no se marcaron `final`
para no aparentar una corrida terminada.

## Coordinación con otras líneas de trabajo

El 2026-11-16 se cerraron en paralelo tres líneas sobre este árbol. Son independientes; esta es la de
simetría y tilted. Ver también [la bitácora de dinámica][binnacle-dynamics], [el handoff de warm-start y
compresión N=18][handoff-warmstart] y [la bitácora de Haiqu VL][binnacle-haiqu].

- Al cierre, otro agente seguía escribiendo en el árbol (su handoff se modificó minutos después del mío
  y `results/hva_vl_study/INDEX.md` se regeneró). No ejecutar `organize_results.py reindex` ni
  `generate_module_index.py` en paralelo con otra sesión.
- El código de la línea de warm-start ya está en `HEAD` (`b5cb8d6`). Todo lo modificado o nuevo en
  archivos compartidos (`hva.py`, `warmstart.py`, `hva_variants.py`, `fidelity.py`, `study_core.py`,
  tests) es, según su handoff, de esta línea de simetría, así que no hace falta `git add -p`.
- Contrastado con su compuerta rápida: esa excluye las clases `GrowAdaptPool` y `SymmetryTiedBuilder`.
  Yo corrí `test_tilted_square`, `test_symmetry`, `test_bond_mask` y `test_hva_session_infra` completos
  sin deselección: **141 pasan**.

Orden de commit sugerido (el usuario decide si commitear):

1. Línea warm-start: `audit_warmstart_donors.py`, `audit_warmstart_vs_random.py` y su handoff.
2. Esta línea: cambios en `src/` y `tests/`, scripts nuevos (`probe_tilted_*`, `run_tilted_*`,
   `run_n20_geo_cluster.py`, `run_sb_layers_ab.py`), `docs/research/`, la bitácora, las entradas de
   `.kiro/knowledge/` y `next_steps.md`, specs y JSON nuevos de `results/`, `data/manifold_cache/`
   y la decisión sobre los 14 scripts borrados.
3. Al final, regenerar y commitear los catálogos (`module-index.md`, `INDEX.md`, `index.json`, scoreboard).

No commitear `.github/context-mode/` (estadísticas de sesión de la herramienta). Las bitácoras y
cambios de `.kiro/knowledge/` y `internal/documentation/` también contienen trabajo de las otras líneas.

## Cómo retomar

1. Leer las secciones 9–11 de `symmetry_frustrated_vqe_findings.md`.
2. Si se vuelve a correr `run_tilted_drop_nnn.py`, `p4_full` debe partir de ≈0.795; si no, hay un seed
   erróneo (la guarda aborta).
3. Atacar el trabajo abierto en este orden: barrido p=5/6 en N=16 con objetivo fidelidad (y repetir el
   drop L0 con varias semillas sobre el mejor p); clusters
   geométricos de N=20 con estructura/presupuesto comparables; mover `optimize_fidelity` a `src/`
   con tests; integrar resultados tilted al scoreboard con un campo OBC/PBC.

<!-- Links -->
[binnacle-dynamics]: ../../internal/documentation/binnacles/binnacle-session-20261116-square-dynamics-limits.md
[binnacle-haiqu]: ../../internal/documentation/binnacles/binnacle-vl-haiqu-square-frustrated.md
[handoff-warmstart]: ../../results/hva_vl_study/HANDOFF_2026-11-16_warmstart_n18.md
