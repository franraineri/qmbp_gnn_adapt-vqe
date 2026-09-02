---
inclusion: fileMatch
fileMatchPattern: '**/*.tex'
---

# Estilo y proceso de escritura de la tesis (reglas generales)

Este steering abstrae **cómo** escribir y revisar la memoria. No contiene datos
numéricos: las cifras canónicas viven en `thesis-writing.md` y en las fuentes de
verdad del proyecto. Aquí van las reglas de comportamiento, lo que hay que
evitar, y la forma de trabajar.

## 1. Principio rector: fidelidad al dato

- **Nunca inventar ni interpolar números.** Toda cifra debe provenir de una
  fuente de verdad verificable (scoreboard, dashboard, eval reports, `project-status`).
  Antes de escribir un resultado, verificar contra esas fuentes.
- Si falta un dato para completar una tabla o texto, **no rellenar**: dejar un
  marcador visible en el lugar exacto con el formato
  `% TODO-DATOS: <qué falta y de dónde debería salir>`.
- Si un número de la tesis contradice el dato real (o hay warnings de tipo
  "stale e_exact"), no arrastrar el valor viejo: marcar la discrepancia con
  `% TODO-DATOS` y usar el dato real, o dejar el hueco si no existe.
- Distinguir siempre la **época de la campaña**: cifras de campañas distintas
  (fechas, rangos de $h$, target-$N$) no se mezclan en una misma tabla.
- No usar "exacto" para DMRG/MPS sin prueba de convergencia; decir "convergido
  dentro de la tolerancia numérica indicada".

## 2. Encuadre y alcance

- El eje del documento es **la GNN prediciendo los ángulos del HVA**: todo
  capítulo y afirmación sirve a ese objetivo.
- Alcance = **simulación ideal (sin ruido)**. El ruido/hardware es solo contexto
  (motivación y estado del arte); el despliegue en hardware es trabajo futuro.
- Toda afirmación fuerte va **anclada**: cita a literatura (`\citep`/`\citet`),
  o referencia a tabla/ecuación propia (`\ref`), o mención explícita al
  experimento (condición + nº de semillas).
- **No reclamar novedad falsa.** La contribución es la integración y la
  validación sistemática, no las técnicas individuales. Usar "se integra / se
  valida / se extiende", no "se propone / se descubre".

## 3. Idioma y anglicismos

- Español de España. Elegir un criterio y mantenerlo; si se conserva un término
  inglés, **definirlo la primera vez y ponerlo en cursiva**.
- Traducir por defecto los anglicismos con equivalente claro (ejecución, estado
  fundamental, datos de referencia, sin ruido, parada temprana, sobreajuste,
  brecha de generalización, etc.). Un término inglés puntual bien asentado puede
  conservarse, pero de forma consistente en todo el documento.
- No alternar como sinónimos términos que no lo son (p. ej. no mezclar
  "framework / flujo / procedimiento / arquitectura" indistintamente).

## 4. Ortografía, gramática y registro

- Revisar tildes, concordancia y nombres propios (portada, encabezados,
  agradecimientos incluidos).
- Tras dos puntos, minúscula (salvo nombre propio o cita).
- Sin coma entre sujeto y verbo.
- "cómo" con tilde cuando explica el modo; "para N=10" (no "a N=10");
  "en función de" (no "en función a"); evitar "en base a".
- Variar la redacción: no encadenar "Se observa que / Esto confirma que".
  Reservar "confirma" para evidencia realmente concluyente.
- Enumeraciones con estructura paralela (todos los ítems igual: o sustantivos, o
  frases completas).
- Reescribir frases rotas o redundantes en vez de parchearlas.

## 5. Jerarquía de métricas (regla central)

El eje de evaluación son las **métricas físicas limpias**, no las tasas derivadas.
Orden de prioridad al reportar y discutir resultados:

1. **$|\Delta E|$ (error energético absoluto)** — métrica primaria. Es la
   diferencia directa entre la energía predicha y la de referencia (ground truth),
   independiente de umbrales. Va siempre primero en tablas y texto.
2. **Cercanía al punto crítico $h_c$** — marco físico. Situar cada resultado
   respecto a $h_c$ y a la frontera de expresividad $h_{\min}$; el error se
   interpreta en función de dónde cae el punto, no de un umbral binario.
3. **Expresividad del ansatz** — explicación. Si $|\Delta E|$ crece, atribuirlo a
   la capacidad finita del HVA a profundidad acotada (no al predictor), con la
   evidencia correspondiente.

No usar $|\Delta E|/N$ (error por sitio): se retiró por no aportar información
adicional respecto a $|\Delta E|$.

**Sin énfasis en la profundidad acotada.** Se entiende que $p$ debe ser acotada
($p \leq 4$), pero no es el eje del trabajo: evitar fórmulas de "capas requeridas"
($p \propto N$, $p \approx N/2$, $p = N-1$) y no atribuirlas como cotas teóricas
citadas. Mencionar la profundidad como un parámetro fijo del experimento, no como
un resultado central.

Reglas derivadas:
- **PassRate es una métrica secundaria/derivada**, no el resultado principal. No
  abrir una discusión ni titular un hallazgo con el PassRate. Puede aparecer como
  columna de apoyo, siempre acompañada de $|\Delta E|$. Nunca reportar un PassRate
  sin el $|\Delta E|$ correspondiente.
- **No usar el PassRate para afirmar que "el pipeline funciona"**: decir que
  "$|\Delta E|$ se mantiene en el orden de $X$ dentro del régimen $h \geq h_{\min}$".
- **Speedup / factor de aceleración: eliminado** del cuerpo hasta que exista una
  definición única y reproducible (cociente de costes o de evaluaciones, con el
  cálculo explícito). No es un cociente de errores. Mientras no exista esa
  definición, no incluir columnas ni cifras de speedup.
- El umbral $\Delta E/\text{gap} < 5\%$ se conserva solo como criterio operativo
  de clasificación de fase, no como métrica de calidad en sí.

## 6. Tono académico

- "muestra", "sugiere", "aporta evidencia" en vez de "demuestra" (salvo prueba
  formal). "satisface el criterio utilizado" en vez de "garantiza".
- Evitar "límite fundamental" / "resultado exhaustivo": se probó un ansatz y un
  rango acotado de profundidades. En todo resultado negativo (Heisenberg, Kitaev)
  añadir **"dentro de las configuraciones evaluadas"**.
- Evitar "pipeline ampliamente validado" (la validación depende de un régimen
  seleccionado), "coste cuántico cero" (aún hay preparación y medición), y
  "reduce la QPU de días a minutos" sin un cálculo concreto de tiempos.
- Reducir adjetivos sin comparación cuantitativa: notable, excelente, potente,
  radicalmente, robusto, óptimo, madurez del procedimiento.

Automatización: el script marca estos términos (`%TODO-TONO`) y auto-corrige las
frases fijas seguras (`--fix-tone`: "el pipeline funciona", "coste cuántico cero",
"ampliamente validado", "límite fundamental", "resultado exhaustivo", "madurez del
procedimiento"). El resto ("demuestra"/"garantiza", adjetivos, añadir "dentro de
las configuraciones evaluadas") requiere **criterio humano**: a veces sí hay
demostración o comparación cuantitativa que justifica el término.

## 7. Tablas y figuras

- Tablas: solo valores exactos, configuraciones y detalle. Las **tendencias**
  (con $N$, $p$, $h$, topología) van en **figuras**. Reducir el número de tablas
  del cuerpo; los listados por semilla/config/topología van a apéndices.
- No repartir en varias tablas resultados que responden a una misma pregunta;
  fusionar o mover a apéndice.
- Declarar la incertidumbre de forma homogénea: indicar si la dispersión viene de
  semillas, de valores de $h$ o de ambos; mismas cantidades comparables → mismo
  tipo de incertidumbre. Explicar por qué media o mediana.
- Mismo número de decimales en cantidades comparables; alinear por la coma
  decimal; declarar unidades ($J = 1$).
- Definir en el pie toda notación no obvia ($\bar{F}_{\text{VQE}}$, etc.). El pie
  describe qué contiene la tabla; la interpretación va en el texto (pies breves).
- "Mejor configuración" debe definirse: seleccionada con validación, no tras ver
  el test. Encabezado "Ejecuciones/semillas aprobadas", no "Pasan".
- Detalles de implementación (p. ej. `norm_type`) van a Desarrollo o apéndice, no
  a una tabla principal.

## 8. Reproducibilidad

- Backend correcto y coherente: statevector para $N \leq 22$, MPS para $N > 22$.
  Nunca "statevector a $N = 40$". Para MPS indicar $\chi$, tolerancia, error de
  truncamiento y validación frente a tamaños menores.
- No llamar "exacto" a DMRG/MPS sin mostrar convergencia respecto a $\chi$; usar
  "convergido dentro de la tolerancia numérica indicada".
- Declarar versiones (Python, Qiskit, PyTorch, PyTorch Geometric, SciPy, TeNPy,
  Aer) y el hardware clásico (CPU/GPU/memoria/SO).
- Aclarar qué controla cada semilla (VQE, red, división de datos, simulación) y
  cómo se evita que un mismo $h$ caiga en entrenamiento y validación entre
  semillas. Un conjunto de test independiente, no solo 80/20.
- Relacionar cada figura y tabla con el script y el archivo de datos que la
  generan; guardar una configuración reproducible por experimento principal.

Automatización: el script (`--check-tex`) marca "statevector con $N > 22$" y
"exacto"+DMRG/MPS (`%TODO-REPRO`). El resto ---versiones de librerías, hardware
clásico, qué controla cada semilla, separación de $h$ train/val, conjunto de test
independiente--- se **escribe a mano** (no detectable mecánicamente).

## 9. Bibliografía

- Verificar autor, año, título y arXiv de cada entrada; que coincidan entre sí.
- Dos trabajos del mismo primer autor y año → sufijos 2025a, 2025b (texto y
  bibliografía). Indicar cuándo es preprint; si hay versión revisada, citar DOI y
  usar arXiv como complemento.
- Un único formato (autor, título, revista, volumen, páginas, DOI, arXiv); no
  mezclar versiones "vN" solo en algunas entradas; revisar capitalización inglesa.
- Toda cita con `\bibitem` y todo `\bibitem` citado. Cada referencia debe
  respaldar con precisión la afirmación que acompaña.

Automatización: el script (`--check-tex`) detecta arXiv duplicados, mismo
primer-autor+año sin sufijo a/b, versiones "vN" inconsistentes, y citas/bibitems
sin pareja. **Requiere verificación humana** (no automatizable sin acceso a arXiv):
que autor/título/año coincidan con el registro real, y que la afirmación citada
aparezca efectivamente en el artículo bajo las mismas condiciones (p. ej. la
frontera de ventaja cuántica en $N \approx 20$, o el resultado de $N/2$ capas).

## 10. Títulos y mayúsculas

- En títulos de capítulo/sección, tablas y pies de figura: solo mayúscula
  inicial y nombres propios (estilo español, no *Title Case* inglés). Aplicar el
  mismo criterio al índice.

## 11. Notación y consistencia terminológica

- Fijar y **definir una sola vez** cada sigla (VQE, HVA, GNN, MPNN, GIN, TFIM,
  MPS, DMRG, NISQ, PCA, etc.) y recordar al inicio de Resultados la definición
  de $p$ (profundidad) y $N$ (tamaño). No alternar $n$/$N$.
- Aclarar la jerarquía GNN (familia) → MPNN (paso de mensajes) → GINConv (capa) →
  predictor (red entrenada).
- Unificar nombres de topologías y modelos (una sola grafía por cada uno; nombres
  en español salvo identificadores internos de código).
- Elegir un término principal y reservar el técnico para variables/tablas
  (p. ej. "tasa de aprobación" en texto, columna abreviada en tablas).
- Todo porcentaje de aprobación lleva su caso absoluto: `95\% (37/39)`.
- Rangos numéricos con `--` (en-dash de LaTeX), nunca guion simple.

Automatización: el script (`--check-tex`) detecta siglas usadas sin definir en el
primer uso, variantes de grafía conviviendo (heavy-hex, coste/costo, gap/brecha
espectral, cadena 1D...), tasas de aprobación sin caso absoluto, y rangos con
guion simple. La elección del término canónico es criterio humano.

- Desambiguar métricas homónimas:
  - Fidelidad: del estado VQE / del estado predicho / media de entrenamiento.
  - Error: absoluto $\Delta E$ / por sitio $\Delta E/N$ / normalizado por gap /
    relativo a la energía total. Nunca decir "error relativo" sin denominador.
- Speedup/aceleración: ver §5 (eliminado del cuerpo hasta definición reproducible).
- Un solo término para "gap espectral" (definido una vez); un solo criterio
  entre "coste"/"costo" (usar **coste**) y entre "escalado"/"escalamiento".
- Coma decimal en texto y tablas; símbolo `×` para factores (no la letra x);
  formato uniforme para intervalos de $h$.

## 12. Estructura narrativa (orden canónico)

Secuencia lineal: introducción → marco teórico → objetivos e hipótesis →
desarrollo del trabajo → resultados → discusión → conclusiones y trabajo futuro.

- **Objetivos e hipótesis** son un capítulo propio, antes del desarrollo.
- **Desarrollo del trabajo** describe qué se implementó y cómo; empieza con un
  esquema del flujo completo y explica cada bloque en una subsección. Los
  requisitos van antes de la solución, no después.
- **Resultados** solo presentan experimentos y datos; la interpretación general,
  las limitaciones y la aplicabilidad van a **Discusión**.
- **Conclusiones** no introducen argumentos ni resultados nuevos; remiten a
  tablas del capítulo de resultados. Fusionar secciones que digan lo mismo.
- El código/reproducibilidad va al final de Desarrollo o en apéndice.
- El marco teórico debe explicar los conceptos con profundidad suficiente
  (transición de fase, gap, observables, VQE, HVA, barren plateaus, predicción
  de parámetros, por qué una GNN, y redes tensoriales conectadas al resto del
  trabajo), no solo enumerar referencias recientes.

## 13. Repeticiones

- Cada idea tiene **una única ubicación canónica**: presentar en su lugar y
  remitir con `\ref` desde el resto.
  - Límite de expresividad del HVA cerca de $h_c$: introducir en Desarrollo,
    demostrar en Resultados, discutir una sola vez.
  - Fases del pipeline, hardware IBM, conteos de campaña: definir una vez.
- Definir una sola vez la **taxonomía de conteo** (configuración / semilla /
  ejecución / experimento) y reportar todos los porcentajes con casos absolutos:
  `95% (37/39)`.

## 14. Orden canónico de revisión (prioridad)

Al revisar la memoria, seguir este orden (credibilidad → estructura → editorial):

**A. Credibilidad de los resultados (primero):**
1. Referencias rotas `??` (`\ref` sin `\label`).
2. Contradicciones entre cifras (contra las fuentes de verdad).
3. Descripción del backend: nunca "statevector a $N > 22$"; MPS con $\chi$/tolerancia.
4. Numeración de requisitos (si existe la sección).
5. Revisión completa de la bibliografía (§9).

**B. Reorganización (después):**
6. Objetivos e hipótesis como capítulo independiente.
7. Requisitos: integrarlos entre objetivos y desarrollo, no en capítulo suelto.
8. Separar resultados y discusión.
9. Eliminar repeticiones (una ubicación canónica por idea).

**C. Consistencia editorial (al final):**
10. Traducir/unificar anglicismos (§3).
11. Unificar términos, símbolos y métricas (§5, §11).
12. Corregir títulos y mayúsculas (§10).
13. Revisión ortográfica y gramatical (§4).

## 15. Forma de trabajar (proceso obligatorio)

1. **Verificar antes de escribir.** Contrastar cada cifra con la fuente de
   verdad correspondiente y su fecha de corte.
2. **Editar de forma quirúrgica**, preservando el resto del documento.
3. **Marcar huecos**, no inventarlos (`% TODO-DATOS: ...`).
4. **Validar la sintaxis LaTeX** tras editar: entornos `table`/`tabular`
   balanceados, sin `\begin` sin `\end`, `\label`/`\ref` consistentes, sin `??`,
   sin `#` sueltos, `$` balanceados, toda `\cite` con `\bibitem` y sin bibitems
   sin citar. Compilar si es posible; si falta el `.sty`, hacer validación
   estructural equivalente.
5. **Propagar el reencuadre** a resumen, introducción, objetivos, discusión y
   conclusiones cuando se cambia un resultado central.
6. **Entregar un resumen de cambios**: secciones/tablas eliminadas o relegadas,
   tablas nuevas con su fuente de datos, y lista completa de `TODO-DATOS` con su
   ubicación.
