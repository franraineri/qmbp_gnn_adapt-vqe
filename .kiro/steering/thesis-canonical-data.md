---
inclusion: fileMatch
fileMatchPattern: '**/*.tex'
---

# Datos canónicos de la tesis (v4.0)

Complemento de `thesis-style-and-process.md` (§1–§15 describen el *cómo*; este archivo fija el *qué*: objetivo, línea base, alcance, estructura, conteos, grillas de $h$, referencias y fuentes de datos). Ambos se cargan juntos al editar archivos `.tex`. Las secciones conservan su numeración original (§16, §17).

## 16. Datos canónicos de la tesis (v4.0)

Esta sección fija los datos de referencia del documento actual
(`internal/tesis-v4.0.tex`). Es la fuente de verdad para cifras, estructura y
citas; el resto del steering describe el *cómo*, esta sección el *qué*.

### Objetivo general (canónico)

"Demostrar que la integración de predicción GNN con HVA de profundidad acotada
en un pipeline unificado reduce el coste cuántico de la clasificación de fases
respecto a VQE con inicialización aleatoria (factor de aceleración $A$),
manteniendo $\Delta E/\text{gap} < 5\%$ dentro del régimen operativo válido, y
documentar formalmente los límites de dicho régimen."

No usar el rango antiguo "29×–500×". El factor de aceleración canónico es $A$
(Ec.~\ref{eq:speedup}, símbolo $A$ y no $S$, §11), un cociente de **evaluaciones de
circuito** $A = r \cdot \bar{k}$.

**El rango antiguo "$\sim 2{,}5\times$ a $4414\times$" era incorrecto y no debe
usarse:** mezclaba dos magnitudes distintas. El $4414\times$ es un valor de $A$
legítimo (parametrización por enlace, 79 dimensiones, §5.4.2); pero el $2{,}5\times$
NO es aceleración, es la razón de error $R(N)$ de las figuras de comparación (§5).
El valor canónico de $A$ debe recalcularse con el $\bar{k}$ medido (37–85 iteraciones
por punto), no con el de literatura (véase "Línea base del VQE aleatorio" más abajo),
y reportarse siempre declarando el $\bar{k}$ y el $r$ empleados. La degradación
creciente del VQE aleatorio con $N$ se reporta como razón de error $R(N)$, magnitud
separada de $A$.

### Línea base del VQE aleatorio (para calcular $A$)

El coste del VQE con inicialización aleatoria tiene tres cifras distintas que NO
deben confundirse:
- **500–1000 iteraciones por punto**: cifra de la literatura (Tilly 2022). Se usa
  solo como *referencia superior*, nunca como valor de trabajo.
- **37–85 iteraciones por punto** (heavy-hex, $p = 3$) y **19–38** (COBYLA, $p = 1$):
  valores *medidos en este trabajo*. Son los que se emplean para calcular $A$.

Regla: todo $A$ reportado declara explícitamente el $\bar{k}$ (iteraciones) y el $r$
(reinicios) con que se calcula, y usa el $\bar{k}$ medido. La cifra de literatura se
cita aparte como cota superior, no como valor de trabajo.

### Alcance

Simulación ideal (StatevectorEstimator para $N \leq 22$; MPS determinista
$\chi = 64$ para $N > 22$). Sin resultados propios de ruido/hardware; el ruido es
contexto (motivación, estado del arte) y el despliegue en hardware es trabajo
futuro.

**Profundidad:** campaña sistemática a $p = 1$–$4$, con sondeos puntuales a $p = 5$
(frontera $h_{\min}$, cadena 1D) y $p = 8$ (Heisenberg, resultado negativo). Al
declarar el alcance, decir "$p = 1$–$4$ en la campaña sistemática, con sondeos
puntuales a $p = 5$ y $p = 8$", no "$p = 1$–$4$" a secas. El límite $p \leq 2$ es
solo una consideración de ruido de la literatura (Cap. 2), no una restricción del
trabajo.

**$N$ máximo — tres escalas canónicas, usar SIEMPRE en este orden y con esta
distinción** (no dar un único "$N$ máximo"; hay tres cosas distintas):
- $N \leq 22$: validado contra vector de estado exacto (StatevectorEstimator).
- $N \leq 40$: pipeline completo Fases 1–3 con ground truth DMRG.
- $N \leq 250$: solo evaluación de circuito con MPS ($\chi = 64$), sin ground truth
  exacto.

En la tabla comparativa con la literatura, la celda "$N$ máx" del pipeline se escribe
"22 / 40 / 250" con nota, no un número suelto. Cualquier afirmación de escala debe
indicar a cuál de las tres escalas se refiere.

### Estructura (7 capítulos, orden lineal)

1. Introducción — problema, motivación, visión del pipeline. Sin resultados.
2. Contexto y estado de la cuestión — física + VQE + HVA + redes tensoriales +
   GNN. Sin resultados propios.
3. Objetivos e hipótesis — OE1–OE6, H1–H6, criterios de evaluación.
4. Desarrollo del trabajo — metodología (3 fases), implementación.
5. Resultados — todos los datos numéricos propios.
6. Discusión — interpretación, comparación con literatura, limitaciones, aplicabilidad.
7. Conclusiones y trabajo futuro — remite a tablas del Cap. 5; hardware como línea 1.
- Apéndice A: código y reproducibilidad. Apéndice B: resultados complementarios.

El pipeline son **3 fases** (ground truth → VQE warm-start → predictor MPNN);
la "definición del problema" es la entrada, no una fase.
Topologías: cadena 1D, escalera, triangular, cuadrada, heavy-hex (NO kagomé, que
es trabajo futuro). Modelos: TFIM, TFIM longitudinal, TFIM frustrado ($J_1$-$J_2$),
Heisenberg (XXZ y transversal, negativo), Kitaev (negativo).

**Modelo XY:** aparece en la tabla de conteo de ejecuciones (7 ejecuciones, 0\%),
pero NO forma parte de la narrativa canónica de la tesis (no está en objetivos,
hipótesis, viabilidad por modelo ni conclusiones). Decisión canónica: o se incorpora
como caso negativo XX+YY junto a Heisenberg/Kitaev (con párrafo en extensibilidad y
fila en la tabla de viabilidad), o se excluye de toda tabla de conteo. No dejarlo
como fila huérfana sin mención en el texto. Mientras no se decida incorporarlo, la
lista canónica de modelos es la de arriba (sin XY).

### Conteo canónico de la campaña

La unidad de recuento es la **ejecución** (§13): una corrida completa de las Fases
1–3 para una (configuración, semilla). El documento debe usar **una sola cifra
canónica** para el total de ejecuciones; hoy conviven tres números incompatibles
(1281, "más de 400", 174) que hay que reconciliar contra la fuente de verdad. Reglas:
- Fijar el total real de ejecuciones desde la tabla de conteo por modelo (fuente:
  ResultIndex / `project-status`) y usarlo de forma consistente en el preámbulo de
  Resultados y en la tabla de conteo.
- Si "174" se refiere a un meta-análisis (subconjunto para diagnóstico de fallos) y
  "más de 400" a otra época de la campaña, decirlo explícitamente: cada cifra declara
  qué mide y de qué subconjunto sale. No presentar tres totales para la misma unidad
  sin distinguirlos.
- El conteo por modelo de una configuración concreta (p. ej. "TFIM a $N=10$ son 79
  ejecuciones") debe ser coherente con el total agregado del mismo modelo en la tabla
  de conteo.

Total canónico actual (ResultIndex, sin XY): **1286 ejecuciones**. El "174" es un
subconjunto de meta-análisis de diagnóstico de fallos (76 no aprobadas), no un total
alternativo de campaña.

### Grillas de $h$ canónicas por régimen (consistencia intra-sección)

Cada régimen usa **una sola grilla de $h$**; todas las tablas de un mismo régimen
comparten denominador y declaran su grilla en el pie. No mezclar rangos entre tablas
del mismo régimen (era la causa de la confusión de denominadores del corrector, §1.9).

| Régimen | Grilla de $h$ | Uso | Tablas |
|---------|---------------|-----|--------|
| **Crítico / frontera** | denso en $[1{,}3;\, 5{,}0]$ (39 puntos de la malla de Fase 1) | comparar topologías y caracterizar dónde falla el HVA cerca de $h_c$ | cross_topo, cross_topo_depth, scaling, cross_topo_de, delta_e_vs_p, tfim_long_deploy |
| **Extrapolación / comparación** | $\{2{,}5; 3{,}0; 3{,}5; 4{,}0; 4{,}5; 5{,}0\}$ (grilla común a las 5 topologías) | comparar tamaños/topologías en el régimen de extrapolación | large_n_chain, auto_heavy_hex_intra_n, auto_heavy_hex_large_n |

- La malla de Fase 1 sobre $[0{,}5;\, 5{,}0]$ tiene 52 puntos; el régimen válido
  $[1{,}3;\, 5{,}0]$ comprende 39 (no "27" — esa cifra era de un barrido corto de otra
  época y se retiró).
- Toda tabla por-$h$ se restringe además al régimen válido de cada $N$ ($h \geq
  h_{\min}(N)$, leído del `model_quality_dashboard.json`).
- En columnas $\Delta E/\text{gap}$ agregadas: es la **media de los cocientes por
  punto** (no el cociente de las medias de $|\Delta E|$ y gap), en **%** con unidad
  explícita; declararlo en el pie.
- Intervalos de $h$ con punto y coma: `$h \in [1{,}3;\, 5{,}0]$` (nunca coma, que se
  lee como lista de cuatro números).

### Matriz canónica cross-topología (TFIM estándar, régimen crítico)

Fuente de verdad: Sección 1 (Summary Table) de
`internal/documentation/analysis/noiseless_v2_analysis.md` (TFIM estándar, $h \geq
1{,}3$, /39). Los valores de las Tablas cross_topo y cross_topo_depth deben coincidir
con ella. Configuraciones no evaluadas (cadena 1D $p=4$, escalera $p=2$) se marcan con
`---`, no se inventan.

### Referencias clave por tema

| Tema | Cita |
|------|------|
| Truncamiento por ruido | Mele 2026 (Nature Physics) |
| Barren plateaus | McClean 2018, Cerezo 2021 |
| Diseño HVA | Wiersema 2020 (PRX Quantum), Tripathi 2026 |
| Warm-start | Puig 2025 (PRX Quantum), Mele 2022 (PRA) |
| Revisión VQE | Tilly 2022 (Physics Reports), Peruzzo 2014 |
| Expresividad GNN | Xu 2019 (ICLR), Meng 2025 |
| Física del TFIM | Dutta 2015 (Cambridge UP), Sachdev 2011 |
| GNN para espines | Huang 2022 (Science), Kochkov 2021 |
| Emergencia (materia condensada) | Anderson 1972 |
| Redes tensoriales / DMRG | Schollwöck 2011, Hauschild 2018 |
| Cadena de Kitaev (solución exacta) | Kitaev 2001 |
| Frustrado en hardware de iones | Teoh 2025 |
| Coste de mitigación | Tsubouchi 2023 |
| Frontera de ventaja cuántica | martin2026 (verificar autoría; entrada por arXiv) |
| Hardware (contexto) | Kim 2023, Sharma 2026, Ma 2025 |

Regla: toda `\cite` con `\bibitem` y todo `\bibitem` citado (§9). Claves en
minúscula `autorYear`; sufijos a/b para mismo primer autor y año.

### Fuentes de datos por resultado

| Dato | Fuente de verdad |
|------|------------------|
| Entre topologías $N=10$, $p=1$–4 | `internal/documentation/analysis/noiseless_v2_analysis.md` |
| Escalado $N=4$–20 | idem (secciones de escalamiento) |
| TFIM longitudinal multi-topo | idem |
| Heisenberg (XXZ + transversal, 75 ejec.) | idem |
| Cross-N UnifiedMPNN + large-N | `internal/documentation/analysis/accelerated_cross_n_coverage.md` |
| Mejores resultados por topología×N | `results/best_results_scoreboard_p1.md` |
| Calificaciones por N | `results/model_evaluation_report.md` |
| Repositorio | `https://github.com/franraineri/qmbp_gnn_adapt-vqe` |

### Errores comunes a evitar

1. Poner resultados propios en el Cap. 2.
2. Afirmaciones fuertes sin anclaje (cita o `\ref`).
3. Conclusiones sin referencias a tablas del Cap. 5.
4. Usar "preliminar" — la tesis contiene solo resultados definitivos.
5. Mencionar resultados propios de ruido/hardware (no existen en la tesis).
6. Usar "$p \leq 2$" como restricción de este trabajo (es solo contexto).
7. Decir "kagomé" como topología evaluada (no lo es; es trabajo futuro).
8. Volcados exhaustivos de datos en extensibilidad — hallazgos concisos.
9. Reexplicar bond-resolved, extracción de $\nu$ o Hamiltonianos candidatos en
   varios sitios — cada uno tiene UNA ubicación canónica.
10. Introducir PCA/PC1 sin explicar qué es y cuál es el conjunto de datos.
11. Atribuir la fórmula $p \propto N$ como cota teórica citada (§5): la
    profundidad no es el eje del trabajo.
12. Confundir tres magnitudes distintas: la reducción de $|\Delta E|$ con $p$
    (15–55×), el factor de aceleración $A$ (cociente de evaluaciones), y la razón de
    error $R(N)$ (cociente de $\Delta E/\text{gap}$). Ver §5 y §11.
13. Usar el símbolo $S$ para la aceleración: $S$ es la entropía; la aceleración es
    $A$ (§11).
14. Dar el rango de aceleración como "$2{,}5\times$–$4400\times$": mezcla $A$ con
    $R(N)$ (§16, objetivo general).
15. Enunciar un criterio ($\Delta E/\text{gap} < 5\%$ "en el régimen válido") como
    universalmente cumplido cuando existe la excepción documentada de la frontera
    móvil (p. ej. a $N = 40$, $p = 1$, algunos puntos cerca de $h_{\min}(N)$ no
    pasan): enunciar el criterio con su excepción, no como cumplido sin matices.
16. Presentar una tasa de aprobación global agregada (toda la campaña) como si fuera
    la tasa por configuración en régimen válido, o compararlas entre sí (§5).
17. Dejar el modelo XY como fila de conteo sin mención en la narrativa (§16, Modelos).

## 17. Preguntas previsibles del tribunal (marco de respuesta)

Anticipar estas preguntas y tener la respuesta anclada (cita o `\ref`).

- **¿Por qué sistemas de espines y no química cuántica?** La transformación de
  Jordan-Wigner mapea operadores fermiónicos locales a cadenas de Pauli de
  longitud $O(N)$, con profundidad $O(N^4)$ para UCCSD; los sistemas de espines
  mapean isomórficamente a qubits con sobrecarga $O(1)$ por término (Tilly 2022).
- **¿No es esto simulación clásica con pasos extra?** A las escalas validadas la
  respuesta exacta es conocida; el valor es metodológico y de escalabilidad sin
  cambios de código a regímenes donde lo clásico falla (§\ref{sec:aplicabilidad},
  Ahsan 2025).
- **¿Por qué la MPNN y no VQE directo?** La MPNN predice $\theta$ en una única
  inferencia clásica, frente al bucle iterativo del VQE. Cuantificado por el factor
  de aceleración $A$ (Ec.~\ref{eq:speedup}, $A = r \cdot \bar{k}$), calculado con el
  $\bar{k}$ medido (37–85 iteraciones por punto en heavy-hex $p=3$), no con la cifra
  de literatura de 500–1000 (Tilly 2022), que se cita solo como referencia superior.
  Tener claro el valor de $\bar{k}$ y $r$ con que se reporta cada $A$.
- **¿$A$ o razón de error?** Distinguir el factor de aceleración $A$ (ahorro de
  evaluaciones) de la razón de error $R(N)$ (mejora de precisión a igual presupuesto).
  Las figuras de comparación MPNN vs VQE aleatorio muestran $R(N)$, no $A$ (§5, §11).
- **¿Y los barren plateaus?** Tres mecanismos concurrentes los evitan en el
  régimen usado: profundidad acotada (Mele 2026), funciones de coste locales
  (Cerezo 2021) y warm-start (Puig 2025).
- **¿Por qué falla la generalización entre tamaños en ciertas topologías?** Mayor
  coordinación $\Rightarrow$ más parámetros $\Rightarrow$ paisaje VQE más difícil;
  se discute en §\ref{subsec:frontera_expresividad} y §\ref{subsec:bond_resolved}.
