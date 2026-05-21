# Plan: Demostración de Utilidad Cuántica

## Contexto

Hemos demostrado que:
- El warm-start MPNN mejora 93-99.9% sobre random (Comparación 1)
- La clasificación de fase es trivialmente clásica (Análisis B)
- El cuello de botella es la expresividad del circuito, no el ML (Comparación 2)
- La calidad de datos importa más que la cantidad (N=20 con régimen válido)

**Pregunta central:** ¿Dónde está la utilidad cuántica real?

**Respuesta:** En predicciones CUANTITATIVAS (energía, correlaciones) para sistemas donde:
1. ED es imposible (N > 14)
2. DMRG falla (2D con ancho > 4-6)
3. Solo el hardware cuántico puede generar los datos

## Las 3 Pruebas de Valor

---

### Prueba 1: "Classical Cost Explosion" (ejecutable ahora)

**Hipótesis:** El costo clásico crece exponencialmente, pero la inferencia MPNN es O(1).

**Qué medir:**
| N | Tiempo Phase 1 (ED) | Tiempo Phase 2 (VQE) | Tiempo inferencia MPNN | Ratio |
|---|---|---|---|---|
| 6 | ~0.01s | ~15s | ~0.001s | 15,000× |
| 10 | ~0.1s | ~50s | ~0.001s | 50,000× |
| 14 | ~30s (estimado) | ~5min | ~0.001s | 300,000× |
| 20 | ~50min (DMRG) | ~50min | ~0.001s | 3,000,000× |

**Valor:** Demuestra que una vez entrenado, el MPNN reemplaza horas de cómputo clásico con milisegundos de inferencia. En hardware cuántico, esto se traduce en: "entrena offline con datos clásicos, despliega en QPU en un solo shot".

**Implementación:** Medir tiempos reales de cada fase y graficar la explosión exponencial vs la constante de inferencia.

---

### Prueba 2: "Warm-Start Under Noise" (ejecutable ahora con FakeTorino)

**Hipótesis:** El warm-start MPNN sigue siendo dramáticamente mejor que random incluso bajo ruido de hardware.

**Qué medir:**
- Deploy θ_pred (MPNN) en FakeTorino → ΔE/gap_warm_noisy
- Deploy θ_random en FakeTorino → ΔE/gap_cold_noisy
- Comparar gain bajo ruido vs gain noiseless

**Predicción:** El gain bajo ruido debería ser MAYOR que noiseless, porque:
- Con ruido, el VQE desde random tiene aún menos probabilidad de converger
- El warm-start te pone cerca del mínimo donde SPSA puede refinar con pocas iteraciones

**Valor:** Demuestra que el framework es robusto al ruido — la ventaja del ML no desaparece en hardware real.

**Implementación:** Usar `deploy_with_baseline()` con `mode="noisy_simulation"` y `n_layouts=1` (sin ZNE, solo ruido crudo).

---

### Prueba 3: "Prediction Beyond Classical Reach" (parcialmente ejecutable)

**Hipótesis:** El MPNN entrenado en N=6-10 puede predecir propiedades para N=14 (donde ED tarda 30s) con calidad comparable, en milisegundos.

**Qué medir:**
1. Entrenar MPNN en datos de N=6 y N=10 (ya tenemos)
2. Usar el MPNN para predecir θ a N=14 (inferencia directa — el MPNN es size-agnostic via global_mean_pool)
3. Verificar contra ED a N=14 (costoso pero posible como validación)
4. Medir: ¿la predicción cross-size funciona?

**Nota importante:** V7 mostró que transfer learning N=6→N=10 falla (7% peor). Pero eso fue con el MISMO MPNN. La pregunta aquí es diferente: ¿un MPNN entrenado en N=10 puede predecir para N=14 (mismo modelo, más qubits)?

**Si funciona:** Demuestra que el MPNN generaliza a sistemas más grandes sin reentrenamiento — el santo grial de la escalabilidad.

**Si falla:** Confirma que se necesita reentrenamiento por tamaño, pero el costo de reentrenamiento (50 min a N=20) sigue siendo mucho menor que VQE from scratch.

---

### Prueba 4: "SPSA Refinement Value" (ejecutable con FakeTorino)

**Hipótesis:** MPNN warm-start + 10-20 iteraciones SPSA en hardware ruidoso alcanza ΔE/gap < 5%.

**Qué medir:**
1. θ_pred (MPNN) → deploy en FakeTorino → ΔE/gap_initial
2. θ_pred → SPSA refinement (20 iters, a=0.1, c=0.05) → ΔE/gap_refined
3. θ_random → SPSA refinement (200 iters) → ΔE/gap_random_refined

**Predicción:**
- MPNN + 20 SPSA iters ≈ random + 200 SPSA iters (10× menos evaluaciones)
- Esto se traduce en 10× menos shots en hardware real = 10× menos costo

**Valor:** Cuantifica el ahorro de shots (= dinero) que el warm-start proporciona en hardware real.

---

## Priorización

| Prueba | Esfuerzo | Valor para tesis | Requiere hardware |
|---|---|---|---|
| 1. Cost explosion | Bajo (medir tiempos) | Alto (argumento de escalabilidad) | No |
| 2. Warm-start under noise | Medio (FakeTorino) | Muy alto (robustez) | No |
| 3. Cross-size prediction | Medio (N=14 VQE) | Muy alto (generalización) | No |
| 4. SPSA refinement value | Alto (SPSA + FakeTorino) | Alto (costo de hardware) | No (simulable) |

**Todas son ejecutables sin hardware cuántico real.** Pero sus resultados predicen directamente el comportamiento en hardware.

---

## Narrativa de Tesis

Estas pruebas construyen el argumento:

1. **"El cómputo clásico explota exponencialmente"** (Prueba 1)
   → Pero la inferencia MPNN es instantánea

2. **"El warm-start funciona bajo ruido"** (Prueba 2)
   → El framework es viable en hardware NISQ real

3. **"El MPNN puede predecir más allá de su entrenamiento"** (Prueba 3)
   → O confirma que necesita reentrenamiento (pero el costo es manejable)

4. **"El warm-start ahorra 10× shots en hardware"** (Prueba 4)
   → Traducción directa a ahorro de tiempo y dinero en QPU

**Conclusión:** El framework demuestra utilidad cuántica no porque el QPU sea "más rápido" que una laptop, sino porque:
- A N>20 (2D), ningún método clásico puede generar ground truth
- El MPNN entrenado en datos clásicos (N≤20) predice θ para N>20
- El QPU valida la predicción con un solo shot (no necesita VQE completo)
- El warm-start reduce el costo de validación en 10× vs random

Esto es **quantum utility**: usar el QPU para lo que solo él puede hacer (medir estados de N>20 qubits), mientras el ML clásico hace el trabajo pesado de predicción.
