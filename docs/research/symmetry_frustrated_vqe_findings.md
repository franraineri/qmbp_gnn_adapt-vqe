# Simetría, degeneración y ansatz cortos para el square TFIM frustrado — recorrido de investigación

**Sistema:** TFIM frustrado en red cuadrada, J2 = 0.5, h = 0.5 (fase frustrada).
**Objetivo:** construir ansatz de estado fundamental cortos (pocas compuertas de 2 qubits) y de alta fidelidad, explotando las propiedades de simetría de la red.
**Métrica:** `subspace_fidelity` sobre el manifold fundamental (la correcta cuando el gap → 0).

---

## 1. Punto de partida: la simetría ayuda a N pequeño

En N = 9 y N = 12 el parameter-tying por simetría puntual (un parámetro por órbita del grupo en vez de uno por bond) **sí** mejoró la eficiencia: fidelidad igual o mayor que el ansatz bond-resolved a **×2.2–3.3 menos parámetros** y el mismo número de 2q.

- N = 9 (3×3, grupo D4): orbit-tied converge a 0.8573 con 35 parámetros.
- N = 12 (3×4, grupo D2): loose ≈ tied ≈ orbit a ~0.97–0.985 con ×3 menos parámetros.

La regla empírica que extrajimos (helper `recommend_symmetry_strategy`): en la fase ordenada (h bajo) la simetría gana fidelidad; su utilidad decrece monótonamente con h.

---

## 2. El muro al escalar: la simetría tied deja de funcionar a N ≥ 16

Al llevar la técnica a N = 20 todos los intentos de sembrar o atar por simetría colapsaron. Mediciones de `init_fid` (overlap del seed con el estado, antes de optimizar):

| Técnica de seed (N = 20, h = 0.5) | init_fid |
|-----------------------------------|---------:|
| compress del θ por-bond guardado  | 0.0003 |
| analytic orbit-compressed         | 0.0095 |
| symmetrize del θ full convergido  | 0.0036 |
| symmetrize sign-canonicalized     | 0.0016 |
| transfer cross-N (N=10 → N=20)    | 0.0031 |
| transfer cross-N (N=18 → N=20)    | 0.0013 |
| seed analítico estructural (p3)   | 0.00006 |

**Ningún** seed informado arranca cerca del estado. El p3 full llega a 0.9939 **no** porque su seed sea bueno (init 5.7e-5) sino porque la optimización completa recorre el landscape hasta el basin. El costo es tiempo de optimización, no calidad de seed.

### 2.1 Causa raíz 1 — ruptura espontánea de simetría

Inspeccionando el θ full convergido (fid 0.9939), los ángulos **dentro de una misma órbita de simetría no son constantes** — varían radicalmente (p.ej. una órbita nn con valores [0.01, 0.18, 0.03, 0.22]; una nnn con [−0.01, 0, 0.39, 0]). Promediar por órbita (lo que hace el tying) destruye el estado.

Interpretación física: el ground state frustrado **degenerado** (doblete) rompe espontáneamente la simetría. El optimizador elige un representante concreto del manifold degenerado que individualmente **no** es invariante bajo el grupo puntual, aunque el manifold como conjunto sí lo sea. Imponer simetría en el ansatz le pide representar un estado que el óptimo no tiene.

### 2.2 Causa raíz 2 — simetría y no-degeneración son mutuamente excluyentes en nuestros clusters

Barrido conjunto de grupo de simetría y degeneración del GS (eigsh exacto) sobre nuestros grids rectangulares con **contorno abierto (OBC)**:

| N  | grupo (order) | manifold_dim | veredicto |
|----|--------------:|-------------:|-----------|
| 9  | D4 (8)        | 1 | simetría + no-degen → tying ideal ✓ |
| 10 | trivial (1)   | 1 | sin simetría — nada que atar |
| 12 | D2 (4)        | 1 | simetría + no-degen → tying ideal ✓ |
| 14 | trivial (1)   | 1 | sin simetría |
| 15 | order 2       | 1 | simetría + no-degen → tying ideal ✓ |
| 16 | D4 (8)        | 2 | simetría pero DEGEN → tying falla |
| 18 | trivial (1)   | 1 | no-degen pero sin simetría |
| 20 | D2 (4)        | 2 | simetría pero DEGEN → tying falla |
| 21 | —             | 2 | DEGEN |

Patrón: los grids "limpios" (cuadrados/rectangulares perfectos) tienen simetría rica pero el GS degenera a N ≥ 16; los grids truncados (N no factoriza) rompen la simetría puntual → grupo trivial. **Ningún N grande tiene ambas propiedades a la vez** en la construcción OBC. N = 9 y N = 12 fueron especiales por ser chicos (simetría + gap aún separa).

---

## 3. La construcción de la red era parte del problema

Nuestro `generate_square` usa **contorno abierto** y grid rectangular `ceil(sqrt(N))`. Esto induce los dos problemas:

- grids truncados (N = 18) → rompen la simetría puntual (grupo trivial);
- bordes abiertos (N = 20) → degeneración espuria inducida por el borde.

La física computacional evita ambos usando **clusters "tilted" con contorno periódico (PBC)**, en los "números mágicos" N = L₁² + L₂² (8, 10, 16, 18, 20, 26, 32, 36, 40…), que preservan la simetría puntual C4v completa del lattice infinito **y** evitan la degeneración de borde. Este camino de construcción **se exploró después**: ver la sección 10 (gate, N=9, transfer a N=16/18). Caveat: PBC cambia la física respecto del OBC que estudiamos, así que los resultados no son comparables con el scoreboard OBC.

---

## 4. La dirección correcta (según la literatura): romper, no imponer

El resultado más importante de la revisión bibliográfica: para estados con simetría rota —el régimen degenerado a N ≥ 16— **la solución no es más simetría en el ansatz, es menos**. El ansatz debe poder **romper** la simetría.

- Un HVA puro, al compartir la simetría del Hamiltoniano, no puede generar un GS con simetría rota en profundidad constante.
- Agregar **symmetry-breaking layers** al HVA permite alcanzar el GS en profundidad **constante** vs **lineal** del HVA puro — exactamente el objetivo de menos 2q / más fidelidad.
- Para frustrados, los **parámetros bond-resolved** (uno por bond, que ya usamos) recuperan precisión a menor profundidad; la casi-degeneración es un desafío documentado.

---

## 5. Hallazgos propios y las referencias que los respaldan

| Hallazgo propio (medido en este sistema) | Referencia que lo respalda |
|------------------------------------------|----------------------------|
| El ansatz simétrico-tied falla en el GS degenerado por ruptura espontánea de simetría | Park, *Efficient ground state preparation in VQE with symmetry-breaking layers*, arXiv:2106.02509 — demuestra que el HVA simétrico no alcanza GS con simetría rota en profundidad constante; la solución son capas de ruptura de simetría |
| El enfoque bond-resolved (un parámetro por bond) es el correcto para frustrados | *Frustration-Induced Expressibility Limitations in Variational Quantum Algorithms*, arXiv:2604.11688 — parámetros bond-resolved recuperan precisión a menor profundidad |
| La casi-degeneración del espectro frustrado es el muro (no barren plateaus) | arXiv:2604.11688 — "near-degenerate spectra in the frustrated regime further challenge variational methods"; la limitación es expresividad, no optimización |
| Simetría puntual + no-degeneración se pueden tener juntas en clusters tilted PBC (números mágicos) | Clusters de Betts / ED con simetría completa: cond-mat/9402061 (N=16,20,32,36), arXiv:0909.3723 (hasta N=40); Ghassemi Tabrizi & Kühne, *Magnetism* 5 (2025) 8 — point-group adaptation en ED |
| Incluir simetría en el diseño del circuito supera incluirla en la función de costo (régimen no degenerado) | Lyu, Xu, Yung, Bayat, *Symmetry enhanced variational quantum spin eigensolver*, arXiv:2203.02444 |
| Romper simetría en los estados de referencia alarga el ansatz de ADAPT-VQE | *Symmetry breaking slows convergence of the ADAPT-VQE*, arXiv:2207.03063 |

---

## 6. Resultados de referencia actuales (OBC, bond-resolved)

Mejores ansatz conocidos para N = 20, h = 0.5 (frontera de Pareto fidelidad vs 2q):

| Fidelidad | 2q  | Config |
|----------:|----:|--------|
| 0.9939 | 524 | p3_half_nn_rx full |
| 0.9800 | 290 | p3_half_nn_rx_topk7of46 |
| 0.9491 | 176 | p3_half_nn_rx_prune0.25_th_nn13_nnn12 |
| 0.8988 | 148 | p3_half_nn_rx_prune0.3_th_nn11_nnn10 |

Todos se obtuvieron con optimización completa (sin atajo de warm-start), consistente con el hallazgo de la sección 2.

> **Corrección (cierre temporal):** una versión previa de esta tabla indicaba 370 2q para
> `p3_half_nn_rx full`. El valor correcto es **524 2q** (scoreboard y spec guardado). El 370
> pertenece a `nnn2_half` (0.9647). Cualquier cifra "0.9939 @ 370" en notas o memorias
> anteriores es errónea.

---

## 7. Infraestructura construida (reutilizable)

- `cached_degenerate_ground_manifold` — cache NPZ del manifold degenerado entre corridas (×25 speedup; eigsh de N=20 pagado una sola vez). Integrado en 5 runners.
- Módulo de simetría: `detect_symmetry_group`, `bond_orbits`, `compress_seed_to_orbits`, `symmetrize_seed`, `generate_subgroups`, `recommend_symmetry_strategy`.
- `select_orbit_complete` — selección de bonds que respeta órbitas completas.
- `transfer_theta_for_blocks` con matching geométrico cross-N (coords de grilla).
- Runners: refinamiento desde specs podados, frontier orbit-complete, cross-N + geométrico.

Toda esta infraestructura queda válida para otros regímenes (N pequeño no degenerado, otras fases) aunque el tying no aplique al régimen frustrado degenerado.

---

## 8. Experimento: symmetry-breaking layers explícitas (resultado)

Siguiendo a Park (arXiv:2106.02509), implementamos un bloque de ruptura de
simetría — una capa de rotaciones **RY por sitio** (0 compuertas de 2 qubits,
rompe la estructura X-only / Z₂ del HVA) — y lo comparamos A/B contra el HVA puro
al **mismo número de 2q**, escalando N con análisis por serie. Métrica:
`subspace_fidelity`. Presupuesto reducido (2 restarts × 200 iter) a N ≥ 18 — el
A/B es justo porque ambos brazos usan el mismo presupuesto.

Δ = fidelidad(HVA + SB) − fidelidad(HVA puro), al mismo 2q:

| N  | degenerado | grupo    | Δ p1_half_nn_rx | Δ p2_half_nn_rx |
|----|:----------:|----------|----------------:|----------------:|
| 9  | no         | D4 (8)   | +0.0000         | −0.0000         |
| 12 | no         | D2 (4)   | −0.0005         | −0.0003         |
| 16 | **sí**     | D4 (8)   | +0.0029         | −0.0004         |
| 18 | no         | trivial  | +0.0001         | −0.0011         |

**Resultado: el bloque symmetry-breaking NO mejora nuestro HVA bond-resolved en
ningún N.** Todos los Δ son marginales (±0.003), dentro del ruido de
optimización. El único Δ positivo apreciable (N=16 p1, +0.0029) es sobre un p1
que colapsa a ~0.48 por falta de expresividad — no una mejora útil.

### Por qué la predicción de Park no se materializa aquí

Park compara sus symmetry-breaking layers contra el **HVA estándar** (un solo
parámetro por capa), que literalmente no puede romper la simetría del
Hamiltoniano. Nuestro ansatz es **bond-resolved** (un parámetro por bond +
`rx_final`), así que **ya tiene la capacidad de romper la simetría** en el
espacio per-bond. Agregar una capa RY explícita es **redundante**: la ruptura que
Park introduce con sus capas ya está disponible en nuestro ansatz por el
parámetro-por-bond. Esto cierra el círculo con la sección 5: el bond-resolved
(arXiv:2604.11688) ya era la implementación correcta de "romper la simetría" para
este sistema.

### Hallazgo colateral (importante)

El `p2_half_nn_rx` **puro** converge alto incluso en el régimen degenerado:
0.9857 a N=16, 0.8862 a N=18. Es decir, **la degeneración no es un muro absoluto
para la fidelidad** — con expresividad suficiente (p2) el ansatz puro llega. El
muro de N=20 descrito en las secciones 2–3 era la combinación de degeneración +
estructura demasiado shallow + cold-start (seed lejos del estado), no la
degeneración por sí sola.

### Estado del objetivo "menos 2q, más fidelidad"

Agotadas las rutas de simetría (tying, orbit-complete, symmetry-breaking layers,
transfer cross-N), la conclusión robusta es que **para este sistema el HVA
bond-resolved ya es la forma correcta del ansatz**, y la mejora de fidelidad a 2q
fijo a escala (N ≥ 16) no viene de la simetría. El único camino de la literatura
que quedaba sin explorar era **ADAPT-VQE** (construir el ansatz operador por operador según
el gradiente). Se lo revisó **solo leyendo las entradas ya existentes del scoreboard**, sin correr
nada nuevo (sección 9): competitivo a N=14, colapsa a N=20. Lo que siguió a este cierre fue la
línea tilted PBC (sección 10).

*Nota de implementación:* el bloque RY queda disponible en la gramática de
`blocks` (`"y"`) y el registro (`p1_half_nn_rx_y`, `p2_half_nn_rx_y`) para futuros
experimentos; es infraestructura válida aunque este experimento haya dado
negativo.

---

## 9. Cierre temporal de N=20 OBC: técnicas aplicadas sobre los mejores specs

Objetivo: mejorar fidelidad o reducir 2q partiendo de los specs ya construidos (sin reconstruir). Presupuesto de las corridas de esta sección: **2 restarts × 20 iteraciones** salvo que se indique otro.

| Técnica                                          | Resultado                                                              | Lectura                                                                 |
|--------------------------------------------------|------------------------------------------------------------------------|-------------------------------------------------------------------------|
| Poda nnn extrema (topk 2/3/4 de 46, fidelity-grad, nn completo) | 0.4299 @260, **0.9373 @266**, 0.4723 @272                | Ninguno entra en la frontera. topk3 está dominado por `diag_axial50_top20` (0.9749 @266) |
| Selección por clusters geométricos conectados    | k3/k4/k6/k8: 0.0818 / 0.0396 / 0.1228 / 0.0408 (198–218 2q)            | **No concluyente** (ver aviso abajo)                                    |
| Refine per-bond `stored` desde el θ guardado     | prune0.3_th 0.8988 → 0.8988; prune0.25_th 0.9491 → 0.9492              | Sin cambio medible; no probado con presupuesto alto                     |
| AQC compress (corridas previas)                  | ~0.49 con templates de bajo 2q                                         | Templates shallow no representan el GS                                  |
| ADAPT (lectura del scoreboard, sin correr nada)  | N=14: 0.9619 @200; N=20: 0.4538 @222                                   | Competitivo a N=14, colapsa a N=20                                      |

Hechos que sí son sólidos:

- **topk3 no es monótono con sus vecinos.** topk2 y topk4 colapsan y topk3 no; ambos restarts de cada
  punto coinciden exactamente, así que no es ruido de optimización. Los 3 nnn de topk3 son
  `[5,7]`, `[5,11]`, `[11,13]` (spec `p3_half_nn_rx_topk3of46_fg_square_N20_h0.50`).
- **Reducir los nnn apenas baja el 2q.** Los 31 nn aportan ~248 CX de los ~260–270, así que topk2/3/4 solo
  varían 260→272. La poda de nn colapsó en la corrida previa (`topk12of46_fg_nn20of31` → 0.1995).
- **Corrección de lectura del scoreboard:** `p3_half_nn_rx full` es 0.9939 @ **524** 2q (no 370, ver sección 6).

> **Aviso de validez (clusters geométricos).** El experimento de clusters usó la estructura
> `p2_half_nn` del runner geométrico (198–218 2q), distinta de la estructura `p3_half_nn_rx`
> del topk3 contra el que se lo comparó, y solo 20 iteraciones. Por eso **no demuestra** que la
> geometría no sirva: la comparación 0.08 vs 0.9373 está confundida por estructura y presupuesto.
> Una versión previa de las notas lo presentó como "refutado"; eso fue una sobreinterpretación.
> Para decidirlo habría que repetirlo sobre `p3_half_nn_rx` con presupuesto comparable.

Lo mismo aplica a la frase "el Pareto de N=20 es el límite físico": lo que se observó es que
**ninguna de las técnicas listadas mejoró la frontera con este presupuesto**. No se demostró un
límite físico. La frontera conocida sigue siendo la de la sección 6.

### 9.1 N=14: nuevo punto de Pareto por poda desde el spec

`run_variant_topk --variant p2_half_nn_rx --reference-from-spec --auto --restarts 4 --maxiter 800`
sobre el spec de 0.9500 @228:

| Selección      | Fidelidad | 2q  | fid/2q   |
|----------------|----------:|----:|---------:|
| full (spec)    | 0.9500    | 228 | 4.17e-03 |
| prune0.05      | 0.6270    | 158 | 3.97e-03 |
| prune0.1       | 0.9226    | 108 | 8.54e-03 |
| **prune0.15**  | **0.9021**| 98  | 9.20e-03 |
| prune0.2       | 0.7511    | 82  | 9.16e-03 |

`prune0.15` es un punto nuevo (no estaba registrado) y el de mejor eficiencia. El refine no mejoró
los puntos que ya existían (prune0.1: 0.9226 vs 0.9235 previo). No se sincronizó al scoreboard
(`--no-sync`).

---

## 10. Clusters tilted con PBC: la línea de investigación nueva

Esta sección contiene el resultado más original de la sesión. **PBC es un modelo físico distinto
del OBC**: las fidelidades de esta sección no son comparables con el scoreboard OBC (que además no
tiene ninguna entrada N=16 a h=0.5).

Implementación: `src/qmbp_simulation/models/tilted_square.py` (`generate_square_tilted`,
`tilted_for_n`, `MAGIC_TILTS`). Los nn son los desplazamientos `(1,0),(0,1)` y los nnn las diagonales
`(1,1),(1,-1)`, ambos con wrap sobre la celda inclinada `T1=(L1,L2)`, `T2=(-L2,L1)`. El Hamiltoniano se
arma con esos bonds geométricos (no con `_generate_nnn_edges`, que deriva nnn por 2-hop y solo
coincide con las diagonales en OBC).

### 10.1 Gate de simetría y degeneración (h=0.5, J2=0.5)

| N  | tilt  | orden del grupo | dim. manifold | gap físico | veredicto               |
|---:|-------|----------------:|--------------:|-----------:|-------------------------|
| 8  | (2,2) | 128             | 2             | 5.5002     | degenerado              |
| 9  | (3,0) | 72              | 1             | 0.1317     | simétrico, no degenerado|
| 10 | (3,1) | 240             | 2             | 1.9805     | degenerado              |
| 13 | (3,2) | 52              | 2             | 1.8956     | degenerado              |
| 16 | (4,0) | 128             | 1             | 0.0377     | simétrico, no degenerado|
| 18 | (3,3) | 144             | 1             | 0.0068     | simétrico, no degenerado|
| 20 | (4,2) | 80              | 2             | 0.0119     | degenerado              |

En OBC, N=16 era degenerado y N=18 tenía grupo trivial (sección 2.2); en tilted PBC, **N=16 y N=18
tienen a la vez el grupo completo y GS no degenerado**. N=20 no se rescata. En N=9/16/18 cada tipo
de bond forma **una sola órbita**, por lo que el ansatz tied tiene 3 parámetros por capa más 1
(`rx_final`), el mismo número a cualquier N (p=4 → 13 parámetros).

**Artefacto a conocer:** `detect_symmetry_group` corta en `max_syms=16`, lo que reporta un orden
espurio de 17 en clusters PBC. Hay que pasar `max_syms=100000` (los tests lo fijan).

### 10.2 Auditoría de que el runner explota la simetría

Verificado directamente en N=16: (1) las 127 permutaciones preservan nn y nnn; (2) las órbitas cubren
todos los bonds y sitios; (3) `orbit_index_map` particiona igual que `bond_orbits`; (4) el circuito
tied y el per-bond con seed expandido producen exactamente el mismo estado (solapamiento 1.0);
(5) el estado tied es invariante bajo el grupo; (6) **el GS exacto de N=16 es invariante** bajo todo
el grupo, así que el ansatz tied no tiene techo estructural. El punto 6 es lo que lo distingue del
régimen OBC degenerado, donde el GS rompe la simetría.

### 10.3 N=9: el método completo converge

Semilla analítica calibrada (`calibrated_zz_coef` / `calibrated_x_scale`, orbit-constant) +
`prepare_warmstart` + cascada p→p+1 (tiling del θ tied). Init de la semilla sin optimizar:

| N  | p | first_order | calibrada  | second_order |
|---:|--:|------------:|-----------:|-------------:|
| 9  | 1 | 0.0878      | **0.3156** | 0.0947       |
| 16 | 1 | 0.0013      | **0.0647** | 0.0020       |

Frontera de Pareto tied en N=9 (los nnn son irrenunciables: `nn_only` no mejora con la profundidad):

| 2q  | Fidelidad | Estructura  | Parámetros |
|----:|----------:|-------------|-----------:|
| 72  | 0.6832    | `[nn,nnn,x]`×1 | 4       |
| 144 | 0.9047    | ×2          | 7          |
| 216 | 0.9826    | ×3          | 10         |
| 288 | **0.9960**| ×4          | 13         |

El per-bond sembrado desde el óptimo tied **empeora** al refinar (N=9 p2: 0.9047 → 0.6327 con 99
parámetros): como el GS es simétrico, romper la simetría aleja del estado.

### 10.4 N=16: el p-sweep calibrado se estanca

| p | Fidelidad | 2q  | ΔE al GS (en gaps) |
|--:|----------:|----:|-------------------:|
| 1 | 0.1621    | 128 | 77                 |
| 2 | 0.3031    | 256 | 45                 |
| 3 | 0.2793    | 384 | 26                 |
| 4 | 0.1000    | 512 | 10                 |

La energía baja de forma monótona pero la fidelidad no, con manifold no degenerado.

### 10.5 Transfer cross-N y objetivo de optimización

Como el ansatz tied tiene los mismos parámetros a cualquier N, el θ tied convergido de N=9 se usa
**tal cual** como semilla de N=16/N=18 (sin remapear bonds). Optimizar la **energía** desde esa
semilla la degrada; optimizar la **fidelidad** directamente (`optimize_fidelity`) la mejora:

| N  | p | init (semilla N=9) | optim. energía | optim. fidelidad | 2q  |
|---:|--:|-------------------:|---------------:|-----------------:|----:|
| 16 | 3 | 0.5436             | 0.2793         | 0.6707           | 384 |
| 16 | 4 | 0.6237             | 0.3722         | **0.7951**       | 512 |
| 18 | 3 | 0.1896             | —              | 0.4041           | 432 |
| 18 | 4 | 0.2322             | 0.2434 (smoke) | 0.5796           | 576 |

Campaña nocturna (8 restarts × 4000 iter, y 6 × 6000 en un A/B): N=16 p=4 dio **0.7951 idéntico** con
1000 y con 6000 iteraciones. A profundidad fija la optimización está saturada; la fidelidad sube
con p (N=16: 0.6707 → 0.7951; N=18: 0.4041 → 0.5796).

**Interpretación (hipótesis, no demostrada):** a N grande el mínimo de energía del ansatz tied no
coincide con el máximo de fidelidad al GS. Lo observado es el efecto (energía baja, fidelidad baja
o sube según el objetivo); la causa no se aisló.

### 10.6 Reducción de 2q en N=16: omitir el bloque nnn de la capa 0

El θ de N=16 p=4 tiene |θ_nnn| de la capa 0 ≈ 0.025 (vs ~0.13–0.17 en las otras capas), lo que
sugirió que ese bloque aporta poco. Cada bloque nnn omitido quita una órbita completa (−64 CX) y el
ansatz sigue siendo simétrico. Se optimizó la fidelidad de cada variante desde el θ de p=4 con el
slot omitido (3 restarts × 1500 iter, una sola corrida por variante):

| Variante            | 2q  | Parámetros | Init de la semilla | Fidelidad | Δ2q  |
|---------------------|----:|-----------:|-------------------:|----------:|-----:|
| `p4_full`           | 512 | 13         | 0.7951             | 0.7951    | 0    |
| **`p4_drop_L0nnn`** | 448 | 12         | 0.7789             | **0.7881**| −64  |
| `p4_drop_L0L1nnn`   | 384 | 11         | 0.2880             | 0.6923    | −128 |
| `p4_drop_L0L1L2nnn` | 320 | 10         | —                  | 0.4982    | −192 |

Omitir solo el nnn de la capa 0 ahorra 64 CX (12.5 %) a un costo de 0.007 de fidelidad; la fidelidad
por 2q sube de 1.553e-3 a 1.759e-3. Omitir más capas degrada con fuerza. El orden es coherente con
la magnitud de θ: se pierde poco al quitar el bloque de menor |θ| y mucho al quitar los mayores.

Lectura prudente: la diferencia de 0.007 es de una sola corrida y no se midió la variabilidad entre
semillas aleatorias, así que "misma fidelidad" es aproximado, no demostrado. Tampoco se probó con p=3
ni con otro N.

**Corrida inválida previa (documentada para no repetirla):** la primera ejecución cargó el θ del
Pareto N=16 fallido (fid 0.10) en lugar del θ transfer+fidelidad (0.7951) por una colisión de nombres
de JSON, y su baseline `p4_full` terminó en 0.5565. Se corrigió el loader y se añadió una guarda que
aborta si el baseline no arranca ≈0.795. Resultado válido en
`results/hva_vl_study/bond_ablation/tilted_drop_nnn_N16_h0.50.json`.

---

## 11. Limitaciones y trabajo abierto

Limitaciones que condicionan cómo leer la sección 10:

- **PBC ≠ OBC.** No hay comparación cuantitativa con el scoreboard OBC.
- **`optimize_fidelity` necesita el GS exacto**, por lo que solo aplica donde se diagonaliza (N ≲ 20).
  Es una prueba de **expresividad** del ansatz, no un método para hardware.
- **Los ansatz tilted no están en el scoreboard ni como `ansatz_specs`.** Solo viven en los JSON
  (`theta_best`) de `bond_ablation/tilted_*`.
- `optimize_fidelity` vive en un script (`run_tilted_transfer.py`), no en `src/`, y los runners no
  tienen tests (el generador y la simetría sí: `tests/unit/test_tilted_square.py`).

Trabajo abierto, por orden de valor esperado:

1. **Barrer p=5/6 en N=16 con objetivo fidelidad.** La fidelidad no saturó en p=4. Una extrapolación
   ingenua sugiere que p=6 podría superar 0.95, pero **no está probado**. Requiere semillas de N=9
   a p=5/6 (barato) o tiling del óptimo de N=16 p=4.
2. Confirmar el resultado de 10.6 (omitir nnn de la capa 0) con varias semillas y probarlo en p=3 y
   p=5/6; si se sostiene, aplicarlo sobre el mejor ansatz del barrido de p.
3. Repetir los clusters geométricos de N=20 con estructura y presupuesto comparables (sección 9).
4. Mover `optimize_fidelity` a `src/` y cubrirlo con tests; integrar los resultados tilted al
   scoreboard con un campo de modelo (OBC/PBC) para no mezclarlos.
