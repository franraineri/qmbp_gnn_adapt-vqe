# Roadmap: Amortized Efficiency + Quench Dynamics

**Fecha**: 2026-08-21
**Prioridad**: Este plan es el principal. El plan QPT/DQPT (qpt_dqpt_implementation_plan.md) es complementario.

---

## Narrativa de la Tesis (dos paneles, un argumento)

**Panel A — Ground states son faciles (este plan, prioridad 1)**:
El GNN+HVA produce estados area-law evaluables con chi=64. DMRG tambien los resuelve, pero el GNN amortiza el costo: una vez entrenado, predice E_0(h) en 1ms para cualquier h. El resultado es ML-efficiency, no quantum advantage.

**Panel B — Dinamica es dificil (plan DQPT, complementario)**:
La evolucion temporal del mismo estado genera volume-law. chi=256 diverge a ~10-15 Trotter steps en heavy-hex. Mas alla, solo la QPU puede simular fielmente. El GNN elimina el bottleneck de preparacion de estado, habilitando la exploracion sistematica del regimen de quantum advantage.

**Narrativa unificada**: "Nuestro GNN prepara un estado eficientemente representable (area-law), cuya evolucion temporal genera entrelazamiento que excede cualquier representacion clasica. La preparacion es barata; la dinamica es donde la computacion cuantica es necesaria."

---

## Ruta 1 (PRIORIDAD): Amortized Efficiency

### Paso 1.1 — Wall-time comparison (datos ya existen, ~2h trabajo)

**Que hacer**: Extraer `time_s` del ground_truth_cache.json para todos los (topology, N, h) y comparar con inference time del MPNN.

**Datos disponibles**:
- GT cache: contiene `time_s` por cada ground truth computado (ED o DMRG)
- MPNN inference: ~1ms por prediccion (medible con timer simple)
- VQE wall-time: en los NPZ metadata o logs de training

**Script**:
```bash
.venv/bin/python -c "
import json, numpy as np
gt = json.load(open('data/ground_truth_cache.json'))
times_by_method = {'ed': [], 'dmrg': []}
for key, val in gt.items():
    t = val.get('time_s', 0)
    parts = key.split('|')
    n = int(parts[1]) if len(parts) > 1 else 0
    method = 'dmrg' if n > 14 else 'ed'
    if t > 0:
        times_by_method[method].append((n, t))

for method, data in times_by_method.items():
    if data:
        ns, ts = zip(*data)
        print(f'{method}: {len(data)} pts, mean={np.mean(ts):.2f}s, max={np.max(ts):.1f}s')
        for n_val in [6, 10, 14, 16, 20]:
            subset = [t for n, t in data if n == n_val]
            if subset:
                print(f'  N={n_val}: mean={np.mean(subset):.2f}s ({len(subset)} pts)')
"
```

**Output esperado**: Tabla que muestra DMRG tomando 1-300s por punto vs MPNN 0.001s.

### Paso 1.2 — Amortization plot (datos ya existen, ~3h trabajo)

**Que hacer**: Contar puntos DMRG usados para entrenamiento vs inferencias gratuitas post-training.

**Datos**:
- Zoo manifest: cada modelo tiene `n_training_points`
- Comparisons/extrapolations: cada evaluacion usa inference del MPNN sin costo adicional
- NPZ files: contienen los h_values usados en training

**Grafico**:
```
Eje X: Numero de puntos h evaluados (1, 10, 50, 100, 500, 1000)
Eje Y: Wall-time acumulado

Linea DMRG: slope = mean_dmrg_time_per_point (lineal, nunca amortiza)
Linea GNN: training_cost (fijo) + n_points * 0.001s (casi plano despues del training)

Crossover point: donde GNN es mas barato que DMRG
```

**Resultado esperado**: Crossover a ~20-50 puntos. Despues, GNN es 1000-10000x mas rapido.

### Paso 1.3 — chi-convergence del HVA (run_mps_precision_study.py, ~4h compute)

**Que hacer**: Correr evaluacion del circuito HVA(theta_pred) con multiples chi y mostrar convergencia rapida.

**Comando**:
```bash
# Para topologias 2D donde el argumento es mas fuerte
.venv/bin/python scripts/experiment_runners/scaling/run_mps_precision_study.py \
    --topology heavy_hex --n-qubits 20 --chi-values 32 64 128 256 512 -v

.venv/bin/python scripts/experiment_runners/scaling/run_mps_precision_study.py \
    --topology square --n-qubits 16 --chi-values 32 64 128 256 512 -v

.venv/bin/python scripts/experiment_runners/scaling/run_mps_precision_study.py \
    --topology triangular --n-qubits 12 --chi-values 32 64 128 256 512 -v
```

**Metrica**: |E(chi) - E(chi_max)| / |E(chi_max)| < 1% a partir de chi=64.

**Lo que demuestra**: El estado HVA tiene entrelazamiento bajo (area-law) → es un ansatz eficiente y correcto. La QPU no se necesita para evaluar ground states, pero si para evolucionarlos.

### Paso 1.4 — QPT detection como validacion MPNN (script de analisis, ~2h)

**Que hacer**: Calcular d^2E/dh^2 con datos existentes y mostrar h_c(MPNN) ~ h_c(exact).

**Comando**:
```bash
# Con ground truth (referencia)
.venv/bin/python scripts/analysis/qpt_detection.py --topology chain_1d --save

# Con predicciones MPNN (validacion del modelo)
.venv/bin/python scripts/analysis/qpt_detection.py --topology chain_1d --use-predicted --save

# Comparar ambos
.venv/bin/python scripts/analysis/qpt_detection.py --topology heavy_hex --use-predicted --save
```

**Lo que demuestra**: El MPNN no solo predice theta, sino que captura la fisica de la transicion de fase (h_c detectado correctamente). Es un consistency check, no un claim de QA.

---

## Ruta 2 (COMPLEMENTARIA): Quench Dynamics + Crossover Plot

### Paso 2.1 — DQPT baseline ED (N=10-20, ~4h compute)

**Que hacer**: Validar que el codigo detecta DQPTs conocidos del TFIM para sistemas chicos.

**Comando**:
```bash
for N in 8 10 12 14 16 18 20; do
    .venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py \
        --section 4 --n-qubits $N --topology chain_1d \
        --h1 0.5 --h2 2.0 --dt 0.05 --n-trotter 60
done
```

**Resultado esperado**: Loschmidt echo muestra minimos claros, rate function tiene picos, t* escala con N.

**Lo que demuestra**: La infraestructura funciona. El ground state preparado por GNN+HVA produce DQPTs correctos al ser quenched.

### Paso 2.2 — Crossover plot: chi-divergencia en dinamica (heavy-hex, ~8h compute)

**Que hacer**: Para heavy-hex N=10,16,20,28, evolucionar con multiples chi y detectar donde divergen.

**Comando**:
```bash
# Cada N con multiples chi — medir donde las curvas se separan
for N in 10 16 20 28; do
    .venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py \
        --section 2 --n-qubits $N --topology heavy_hex \
        --h1 0.5 --h2 2.0 --dt 0.1 --n-trotter 30 \
        --chi-values 64 128 256 512
done
```

**Metrica**: Trotter step en que |E(chi=256) - E(chi=512)| / |E(chi=512)| > 5%.

**Lo que demuestra**: El "grafico clave" de la tesis: "de este punto en adelante, la simulacion clasica (MPS) pierde precision, y solo la QPU puede continuar."

### Paso 2.3 — Contraste Panel A vs Panel B (analisis final)

**Que hacer**: Juntar resultados de 1.3 (chi converge rapido para ground states) y 2.2 (chi diverge para dinamica) en un unico grafico de dos paneles.

**Grafico final**:
```
Panel A: E_GS vs chi (converge a chi~64) — "preparacion es facil"
Panel B: E(t=15) vs chi (NO converge a chi=512) — "dinamica es dificil"

Flecha: "GNN prepara aqui (Panel A) → QPU evoluciona aqui (Panel B)"
```

---

## Prioridades y Timeline

| Paso | Esfuerzo | Datos nuevos? | Prioridad | Resultado para tesis |
|------|----------|:---:|:---:|---|
| 1.1 Wall-time | 2h | No | P0 | Tabla efficiency |
| 1.2 Amortization | 3h | No | P0 | Grafico crossover cost |
| 1.3 chi-convergence GS | 4h compute | Si (MPS runs) | P1 | Panel A |
| 1.4 QPT detection | 2h | No | P1 | Validacion MPNN |
| 2.1 DQPT baseline | 4h compute | Si (ED runs) | P2 | Baseline correcto |
| 2.2 Crossover plot | 8h compute | Si (MPS runs) | P2 | Panel B (clave!) |
| 2.3 Contraste final | 2h analisis | No | P3 | Argumento unificado |

**Total estimado**: ~25h (distribuido en ~1 semana si los computes corren en background).

**P0 se puede hacer HOY** con datos existentes — solo son scripts de analisis.

---

## Lo que NO hacer (anti-patterns)

- NO clamar que chain_1d a N=30-60 es inaccesible clasicamente (GPU TDVP lo resuelve).
- NO clamar quantum advantage para ground states (area-law → siempre resoluble clasicamente).
- NO presentar DQPT en MPS como resultado final (es infraestructura, no claim).
- NO comparar wall-time GNN vs DMRG como "quantum speedup" (es ML speedup, no quantum).
- NO correr DMRG con chi alto en 2D solo para "demostrar que falla" (eso ya esta en la literatura).

---

## Claims correctos para la tesis

1. "El GNN amortiza el costo de DMRG: O(100) puntos de training habilitan O(infinity) predicciones a 1ms cada una." (ML efficiency)

2. "El estado HVA(theta_MPNN) es evaluable clasicamente con chi=64 (area-law), confirmando que el ansatz es eficiente." (Ansatz validation)

3. "El MPNN captura h_c con error <5%, demostrando que la GNN aprende la fisica de la QPT." (Physics capture)

4. "La evolucion temporal del mismo estado supera la capacidad de MPS a ~10-15 Trotter steps en heavy-hex, estableciendo la frontera donde la QPU es necesaria." (Crossover identification)

5. "El GNN elimina el bottleneck de preparacion de estado (29-500x speedup), habilitando la exploracion sistematica del regimen de quantum advantage demostrada por IBM (arXiv:2607.24937)." (Enabling technology)

---

## Relacion con plan QPT/DQPT

| Este plan | Plan QPT/DQPT | Relacion |
|-----------|---------------|----------|
| Paso 1.1-1.2 (efficiency) | — | Independiente, ejecutar primero |
| Paso 1.3 (chi-conv GS) | — | Panel A del argumento |
| Paso 1.4 (QPT detection) | Fase 3 del otro plan | Mismo script, misma implementacion |
| Paso 2.1 (DQPT baseline) | Fase 2, Section 4 | Mismo codigo |
| Paso 2.2 (crossover plot) | Paso nuevo del otro plan | Panel B del argumento |
| Paso 2.3 (contraste) | — | Sintesis de ambos planes |

Los dos planes convergen en el **grafico de dos paneles** (chi-convergence GS vs chi-divergencia dinamica). Ese es el argumento central.

---

*Documento complementario a qpt_dqpt_implementation_plan.md. Ambos comparten infraestructura (trotter.py, observables.py) y datos (GT cache, NPZ, zoo models).*
