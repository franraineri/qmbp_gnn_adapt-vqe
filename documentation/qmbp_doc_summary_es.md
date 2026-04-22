# Arquitectura Híbrida GNN-HVA para Caracterización de Fases Topológicas

## Documento de Contexto — Trabajo Final de Máster (TFM)

> Este documento explica, de forma progresiva y didáctica, el problema físico que motiva esta tesis, por qué las soluciones clásicas fallan, cómo la computación cuántica ofrece una salida, y qué arquitectura concreta proponemos para extraer utilidad de los procesadores cuánticos actuales (ruidosos y limitados).

---

# Parte I — El Problema Físico

## 1. El Problema Cuántico de Muchos Cuerpos

En 1972, el premio Nobel P.W. Anderson publicó su célebre ensayo *"More is Different"*, donde estableció que cuando miles de millones de partículas cuánticas (electrones, espines) interactúan, no se comportan como la suma de sus partes. Emergen leyes físicas completamente nuevas. Esto es la esencia del **Problema Cuántico de Muchos Cuerpos**.

La complejidad matemática surge porque las partículas no existen de forma aislada: sus funciones de onda se superponen e interactúan, lo que impide separar la ecuación de Schrödinger en piezas individuales resolubles. El sistema debe tratarse como una única entidad matemática masiva e inextricablemente conectada.

### El paradigma clásico: Landau y la ruptura de simetría

Durante décadas, los físicos clasificaron las fases de la materia usando la **Teoría de Ruptura de Simetría de Landau**:

- El agua se congela en hielo, el hierro se magnetiza → las partículas "rompen simetría" y caen en un patrón ordenado.
- Landau introdujo el **parámetro de orden local**: una cantidad medible (como la magnetización) que es cero en la fase desordenada y distinta de cero en la fase ordenada.
- Si miras un átomo en un imán, su orientación te dice el estado de todo el material.

### Lo que Landau no pudo explicar: Fases Topológicas

En los años 80, se descubrieron materiales que desafiaban completamente las reglas de Landau. A partir del **Efecto Hall Cuántico**, se observaron estados de la materia con transiciones de fase *sin ruptura de simetría* y *sin parámetro de orden local*.

A temperatura de cero absoluto (−273.15°C), donde toda fluctuación térmica se detiene, estos materiales se negaban a congelarse u ordenarse. Se los llamó **Fases Cuánticas Topológicas**. Su "identidad" no está definida por el arreglo local de sus átomos, sino por **invariantes topológicos globales** — propiedades de la función de onda que permanecen constantes aunque el sistema se deforme suavemente, de la misma manera que un donut y una taza de café comparten la propiedad topológica de tener exactamente un agujero.

---

## 2. Frustración Geométrica y Líquidos de Espín Cuánticos (QSL)

La fase topológica más buscada es el **Líquido de Espín Cuántico (QSL)**, propuesto por Anderson en 1973. Los QSL permanecen fundamentalmente desordenados, pero altamente correlacionados, hasta el cero absoluto.

### ¿Qué es la frustración geométrica?

Imaginemos un juego simple con tres átomos magnéticos en un triángulo. Las reglas de su interacción cuántica (antiferromagnetismo) dictan que vecinos deben apuntar en direcciones opuestas:

- Átomo 1 apunta ARRIBA ↑
- Átomo 2 apunta ABAJO ↓
- Átomo 3 está **frustrado**: conectado a un ↑ y un ↓, no puede satisfacer la regla para ambos vecinos simultáneamente.

Clásicamente, esta frustración genera una degeneración masiva del estado fundamental — millones de configuraciones igualmente válidas con la misma energía. Pero la mecánica cuántica no permite que el sistema simplemente elija una y se detenga.

### El estado RVB (Resonating Valence Bond)

Cuando escalamos el triángulo a una red 2D masiva (como una red de Kagome), todo el sistema se frustra intensamente. Como ningún arreglo "congelado" satisface los requisitos energéticos, las **fluctuaciones cuánticas** toman el control. Los espines permanecen en un estado líquido de movimiento constante incluso a cero absoluto.

En lugar de congelarse, los espines se entrelazan a distancias macroscópicas formando un **estado de Enlace de Valencia Resonante (RVB)**. Un enlace de valencia es un singlete cuántico — un par de espines inextricablemente entrelazados:

$$
\frac{|\uparrow\downarrow\rangle - |\downarrow\uparrow\rangle}{\sqrt{2}}
$$

En un QSL, estos enlaces no se fijan en un patrón estático. El estado "resuena": el verdadero estado fundamental es una superposición cuántica masiva de *todos los posibles* emparejamientos de singletes en toda la red.

### Consecuencias físicas

- **Cero orden local**: No puedes mirar un espín, ni un grupo pequeño, para conocer el estado del sistema. La fase se caracteriza por la **Entropía de Entrelazamiento Topológico**.
- **Excitaciones fraccionalizadas**: Voltear un espín en un QSL rompe un singlete, creando dos cuasipartículas independientes llamadas **espinones**, cada uno con espín fraccionario 1/2 pero sin carga eléctrica. Las propiedades fundamentales del electrón (espín y carga) se separan literalmente.
- **Potencial tecnológico**: Estas excitaciones fraccionalizadas actúan como **anyones**, con enorme potencial para memorias cuánticas topológicas tolerantes a fallos.

---

## 3. El Cuello de Botella Clásico: La Maldición de la Dimensionalidad

Para entender por qué simular un QSL es tan difícil, debemos mirar la matemática del **Espacio de Hilbert**.

| Sistema    | Amplitudes a rastrear | Memoria requerida |
| ---------- | --------------------- | ----------------- |
| 10 espines | 1.024                 | ~16 KB            |
| 30 espines | ~1.000 millones       | ~16 GB            |
| 50 espines | ~1,1 cuatrillones     | ~18 PB            |

En mecánica clásica, describir 50 monedas requiere 50 variables. En mecánica cuántica, 50 espines existen en superposición de todos los estados posibles simultáneamente, requiriendo $2^{50}$ amplitudes de probabilidad complejas.

Alrededor de 50 partículas cuánticas interactuantes, la capacidad de memoria del supercomputador más grande del mundo se agota completamente. Este escalado exponencial es la **Maldición de la Dimensionalidad**.

---

## 4. El Golpe Fatal: El "Problema del Signo"

* [ ] Para esquivar la Maldición de la Dimensionalidad, los físicos usan métodos de muestreo estadístico como **Quantum Monte Carlo (QMC)**: en lugar de rastrear cada estado posible, QMC muestrea un subconjunto aleatorio y calcula el promedio.

Sin embargo, QMC sufre un fallo catastrófico llamado **Problema del Signo** en sistemas frustrados (como los QSL):

- Las funciones de onda cuánticas pueden ser negativas o complejas (a diferencia de las probabilidades clásicas, siempre positivas).
- En sistemas frustrados, los caminos cuánticos tienen signos opuestos. Al sumar probabilidades, los caminos positivos y negativos se cancelan (interferencia destructiva).
- Para obtener una respuesta estadísticamente significativa, el computador clásico necesita exponencialmente más muestras.

En 2005, se demostró que el Problema del Signo es **NP-hard**: ningún algoritmo clásico podrá resolverlo eficientemente para casos generales. El camino clásico es un callejón sin salida.

---

## 5. La Promesa Cuántica y las Limitaciones NISQ

En 1982, Richard Feynman propuso la solución definitiva: *"La naturaleza no es clásica... y si quieres simular la naturaleza, más vale que lo hagas cuánticamente."*

Un computador cuántico posee nativamente un espacio de Hilbert exponencialmente grande. Un procesador de 50 qubits rastrea inherentemente $2^{50}$ amplitudes simplemente por existir. No hay Maldición de la Dimensionalidad ni Problema del Signo.

### La era NISQ (Noisy Intermediate-Scale Quantum)

Sin embargo, estamos en la era **NISQ**. El entorno interactúa constantemente con los frágiles qubits, causando:

- **Decoherencia**: pérdida del estado cuántico.
- **Errores de puerta**: operaciones imprecisas.

Como demostró *Mele et al. (2026)*, este ruido no solo añade imprecisiones — **trunca fundamentalmente la profundidad** del circuito. Si un algoritmo requiere 500 operaciones sucesivas, el ruido borrará la información de las primeras 450 puertas. El entrelazamiento nunca alcanza la escala macroscópica necesaria.

Esto obliga a inventar algoritmos **ultra-superficiales e híbridos** (como nuestra arquitectura GNN-HVA) para extraer utilidad antes de que el ruido destruya la simulación.

---

# Parte II — Nuestra Solución

## 6. La Arquitectura Híbrida GNN-HVA

Para sortear estas limitaciones del hardware, esta tesis propone una arquitectura híbrida que desplaza el trabajo pesado a un modelo clásico de Inteligencia Artificial:

### Los tres pilares

1. **Hamiltonian Variational Ansatz (HVA)**: Abandonamos circuitos cuánticos genéricos y profundos. Diseñamos un circuito estrictamente superficial ($p \leq 2$ capas) cuyas puertas derivan directamente de las ecuaciones físicas del material objetivo.
2. **Graph Neural Network (GNN)**: Entrenamos una GNN clásica con datos generados por Redes Tensoriales avanzadas. La GNN aprende la topografía compleja del sistema cuántico.
3. **"Warm-Start Inteligente"**: Ante un material nuevo de 40-50 qubits, la GNN predice instantáneamente los ángulos óptimos ($\theta_{opt}$) para las puertas cuánticas. Inyectamos estos ángulos directamente en el circuito HVA superficial en el hardware cuántico.

### ¿Por qué es innovador?

- **Respeta la física del hardware**: El límite estricto de profundidad previene la truncación de información por ruido.
- **Resuelve el problema de inicialización**: El "Warm-Start" coloca al algoritmo cuántico directamente en el fondo del valle energético, evitando completamente el problema de barren plateaus.
- **Camino hacia la utilidad cuántica**: Combina ML clásico para predicción de parámetros con hardware cuántico para la proyección final del estado.

---

## 7. Hoja de Ruta Operativa (4 Fases)

### Fase 1 — Generación de Ground Truth Clásico

- **Objetivo**: Resolver Hamiltonianos parametrizados clásicamente.
- **Herramientas**: Diagonalización Exacta ($N < 15$ para el PoC, actualmente $N=6$), DMRG/TeNPy (quasi-1D), NetKet (2D).
- **Salida**: Dataset $(h, J) \to$ {estado fundamental $\psi$, energía, gap espectral, observables locales}.

### Fase 2 — Compilación y Optimización del Ansatz

- **Objetivo**: Encontrar los parámetros óptimos $\theta_{opt}$ para el circuito HVA.
- **Método**: VQE con warm-start descendente ($h=2 \to 0$), optimizador L-BFGS-B.
- **Restricción**: HVA superficial ($p \leq 2$), estado inicial $|+\rangle^{\otimes N}$.

### Fase 3 — Entrenamiento del Modelo Predictivo (MLP/GNN)

- **Objetivo**: Entrenar un modelo clásico para predecir $\theta_{opt}$ a partir de los parámetros del Hamiltoniano.
- **PoC**: MLP simple ($h \to \theta_{pred}$) con filtro de fidelidad (≥96%) y validación energética.
- **Escalado**: GNN completa para acoplamientos no uniformes o redes 2D.

### Fase 4 — Despliegue en Hardware

- **Objetivo**: Ejecutar en hardware IBM real con inferencia de la GNN/MLP.
- **Flujo**: Hamiltoniano no visto → GNN predice $\theta_{pred}$ → Warm-Start del HVA → AdaptVQE restringido (≤2 iteraciones).
- **Validación**: Observables locales ($\langle X_i \rangle$, $\langle Z_i Z_{i+1} \rangle$) para clasificar la fase cuántica.

---

## 8. Métricas de Validación (Orden de Prioridad Física)

Las métricas están ordenadas por relevancia física. Las primeras son lo que importa en hardware real; las últimas son diagnósticos solo para simulación sin ruido.

| Prioridad   | Métrica                          | Qué nos dice                                                                                                                                                                               | Umbral                    | ¿Hardware? |
| ----------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ----------- |
| **1** | **ΔE / gap**               | ¿Resolvemos la física? El error energético relativo al gap espectral determina si el pipeline puede distinguir el estado fundamental del primer estado excitado.                         | < 5%                      | ✅          |
| **2** | **⟨Xᵢ⟩, ⟨ZᵢZᵢ₊₁⟩** | Caracterización de fase. Son los parámetros de orden que clasifican ferromagnético vs paramagnético. El cruce ⟨X⟩ = ⟨ZZ⟩ define el punto crítico de tamaño finito.                | error < 1e-2              | ✅          |
| **3** | **ΔE**                     | Precisión energética absoluta. Útil pero menos informativo que ΔE/gap — un ΔE de 0.01 no significa nada sin conocer la escala del gap.                                                | < 1e-2 (aspiracional)     | ✅          |
| **4** | **Fidelidad**               | Solapamiento total con el estado fundamental exacto. Potente para validación sin ruido pero**prohibido en hardware** (costo global → barren plateaus bajo ruido, Mele et al.).      | ≥ 99.5% (solo noiseless) | ❌          |
| **5** | **Iteraciones ADAPT**       | Compliance de profundidad del circuito. Debe mantenerse ≤ 2 para respetar el límite de truncación por ruido$\mathcal{O}(\log n)$. Terminación en 0 iteraciones es el resultado ideal. | ≤ 2                      | ✅          |

### Nota sobre calibración de umbrales

El umbral ΔE < 1e-2 es **aspiracional** — está acotado por el techo de expresibilidad del HVA en cada valor de $h$. A $h=1.5$ con $p=2$, el VQE mismo alcanza ΔE ≈ 1.9e-2, por lo que el pipeline MLP+AdaptVQE no puede superar esto. La métrica ΔE/gap (1.3% a $h=1.5$) muestra correctamente que el pipeline resuelve la física a pesar de que el ΔE absoluto exceda 1e-2.

---

# Parte III — Implementación Técnica

## 9. Stack Tecnológico

| Componente          | Herramienta                          | Uso                                  |
| ------------------- | ------------------------------------ | ------------------------------------ |
| Framework cuántico | **Qiskit 2.x** (ecosistema V2) | Hamiltonianos, circuitos, ejecución |
| Machine Learning    | **PyTorch** (`torch.nn`)     | MLP (PoC) y GNN (escalado)           |
| Solvers clásicos   | **NumPy**, **SciPy**     | Diagonalización exacta, L-BFGS-B    |
| Redes Tensoriales   | **TeNPy**, **NetKet**    | DMRG (quasi-1D), NQS (2D)            |

**Regla estricta**: Módulos deprecados (`qiskit.opflow`, `PauliSumOp`, `qiskit.algorithms`) están prohibidos en el código.

---

## 10. Técnicas de Implementación por Fase

### Fase 1: Diagonalización Exacta

- Usamos `np.linalg.eigh(H.to_matrix())` (densa, no sparse) para garantizar estabilidad numérica y obtener el **gap espectral** ($\Delta = E_1 - E_0$) directamente.
- **Observables bulk**: Promediamos sobre todos los sitios ($\frac{1}{N} \sum \langle X_i \rangle$) para evitar artefactos de borde.

### Fase 2: Diseño del HVA y Optimización

**Construcción del circuito:**

- El HVA replica la estructura del Hamiltoniano: bloque de interacción ZZ → bloque de campo X por capa.
- **Estado inicial obligatorio**: $|+\rangle^{\otimes N}$ (capa de Hadamard). A $\theta = 0$, el HVA produce el estado fundamental paramagnético ($h \to \infty$).
- Factor $2\theta$ en las puertas: `qc.rzz(2 * theta, i, i+1)` para implementar correctamente $e^{-i\theta H}$.

**Optimización (lecciones aprendidas del PoC):**

1. **Punto silla en θ=0**: El estado $|+\rangle^{\otimes N}$ crea un punto silla protegido por simetría. El gradiente se anula ($\sim 10^{-6}$) y L-BFGS-B declara convergencia en iteración 0 sin moverse. **Solución**: Inicializar siempre con perturbación aleatoria pequeña: `np.random.uniform(-0.01, 0.01, n_params)`.
2. **Dirección del sweep**: Barrer de $h=2.0$ (paramagnético) **descendiendo** hacia $h=0.0$. A $h \to \infty$, $|+\rangle^{\otimes N}$ ya es exacto, así que $\theta \approx 0$ es casi óptimo. El warm-start propaga la solución suavemente hacia $h=0$.
3. **Límite de expresibilidad (hallazgo clave de la tesis)**: El HVA con $|+\rangle^{\otimes N}$ y $p=2$ **no puede alcanzar** el estado fundamental ferromagnético profundo ($h \to 0$, que es $|000...0\rangle$). La fidelidad degrada por debajo de $h \approx 1.0$ (ej: 22% a $h=0$ para $N=6$). Verificado con 50 restarts aleatorios sobre $[-\pi, \pi]$ — no es un fallo de optimización sino una limitación estructural. El pipeline se valida para el **régimen paramagnético** ($h \geq 1.0$, fidelidades > 96%). Este compromiso expresibilidad-profundidad es en sí mismo un hallazgo significativo que ilustra las consecuencias prácticas del teorema de truncación por ruido.

**Optimización energética (compliance Mele et al.):**

- La función de costo es la **energía física** vía `StatevectorEstimator`, nunca la fidelidad global.
- La fidelidad se calcula como métrica de validación pasiva en simulaciones sin ruido, pero está estrictamente prohibida como función de costo y en paths de hardware.

### Fase 3: Modelo Predictivo (MLP/GNN)

- **PoC**: MLP simple ($h \to \theta_{pred}$) porque para el TFIM 1D con $J$ uniforme, la estructura del grafo es fija y solo varía $h$. La GNN completa se reserva para el escalado.
- **Validación física (callback)**: Cada N épocas, los ángulos predichos se alimentan a un `StatevectorEstimator` para calcular la energía cuántica resultante. Esto asegura que la red aprende el paisaje energético real, no solo interpola números abstractos.
- **Filtro de fidelidad (crítico)**: Solo entrenar con datos de Fase 2 donde fidelidad ≥ 96%. Puntos por debajo tienen $\theta_{opt}$ que no representan el verdadero estado fundamental — entrenar con ellos envenena el modelo. La firma diagnóstica de este modo de fallo es: MSE converge a casi cero mientras ΔE permanece constante (el MLP aprende fielmente datos basura).
- **Selección del punto de test**: El punto de test de Fase 4 debe estar dentro del régimen de alta fidelidad. Testear a $h=1.5$ (fid ≈ 99.6%) da resultados significativos; testear a $h=1.05$ (fid ≈ 97.6%) confunde limitaciones del ansatz con calidad del pipeline.
- **LR scheduling**: `ReduceLROnPlateau` para evitar oscilación en datasets pequeños.
- **Validación de interpolación**: Siempre validar en al menos un valor de $h$ no visto.
- **Quality gate**: Antes de entrenar, verificar que Fase 2 produjo datos válidos (fidelidad mínima > 99% en el régimen paramagnético).

### Fase 4: Despliegue

- **Ejecución**: `EstimatorV2` de `qiskit_ibm_runtime` para hardware.
- **AdaptVQE restringido**: `max_iterations=2` para respetar el límite de profundidad.
- **Convergencia en inicialización**: Cuando el warm-start es casi óptimo, AdaptVQE lanza `AlgorithmError` en la primera iteración (todos los gradientes bajo el umbral). Este es el **resultado ideal** — significa 0 capas extra necesarias. El código debe capturar esta excepción y tratarla como éxito.
- **Clasificación de fase**: Mediante observables locales y el cruce $\langle X \rangle = \langle ZZ \rangle$ de los datos exactos de Fase 1. Para sistemas de tamaño finito, el punto crítico se desplaza del límite termodinámico $h_c = 1.0$.

---

> 📚 **Bibliografía completa:** Todas las referencias citadas en este documento están consolidadas en [documentation/bibliography.md](bibliography.md).
