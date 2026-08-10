# Evaluación de Publicabilidad — Proyecto GNN-HVA

**Fecha**: 2026-07-21  
**Método**: Revisión contra literatura arxiv 2024-2026 actualizada

---

## Veredicto Rápido

| Paper propuesto | ¿Novedoso? | ¿Publicable? | Competencia directa |
|-----------------|:----------:|:------------:|---------------------|
| GNN warm-start para VQE | ❌ Saturado | No como paper standalone | Qracle, Flow-VQE, PVLS, PALQO, VQEzy, GNN-VQE |
| Atlas de expresividad HVA | ✅ Parcial | **Sí (nicho)** | Tripathi2026, Brozzi2026, 2604.11688 |
| GNN-QEM cross-topology | ✅ Diferenciado | **Sí (fuerte)** | GEM (2604.16815), pero ángulo distinto |
| Detección no supervisada θ→QPT | ⚠️ Competido | No como paper propio | arXiv:2506.06678 (VAE+atención, mismo concepto) |
| PEA-ZNE benchmarks | ❌ Incremental | No | PEA es técnica IBM existente |

---

## Paper 1 DESCARTADO: GNN Warm-Start para VQE

### Por qué NO es publicable

El espacio de "ML predice parámetros óptimos de VQE" está **saturado** a julio 2026:

- **Qracle** (Zhang et al., arXiv:2505.01236, mayo 2025): GNN que codifica Hamiltoniano+ansatz
  en grafo unificado → predice θ_opt. Reduce iteraciones 64%. Exactamente tu concepto.
- **Flow-VQE** (Zou et al., arXiv:2507.01726, npj QI 2026): Normalizing flows para warm-start.
  Aceleración 50×.
- **PVLS** (arXiv:2512.04909, dic 2024): GNN para VQLS, reduce costo 81%.
- **PALQO** (arXiv:2509.20733, sep 2025): Physics-informed ML para VQA, 90% reducción hasta
  40 qubits.
- **VQEzy** (arXiv:2509.17322, sep 2025): Dataset abierto de 12,110 instancias para
  inicialización VQE. El campo ya tiene su propio benchmark dataset.
- **GNN-VQE** (arXiv:2606.08794, junio 2026): GNN para selección de operadores en ADAPT-VQE.
- **Fast ML-VQE** (arXiv:2503.20210, marzo 2025): ML sobre datos intermedios de VQE.

Tu contribución (MPNN + warm-start + pipeline integrado) está subsumida por estos trabajos.
El speedup 29-500× no es suficiente diferenciación frente a Qracle (64% menos iteraciones)
o Flow-VQE (50× speedup) que ya están publicados.

### Salvable como

Componente menor dentro de un paper más amplio (pipeline completo), no como contribución
principal.

---

## Paper 2 DESCARTADO: Detección No Supervisada de QPT via θ_opt

### Por qué NO es publicable como paper independiente

**arXiv:2506.06678** (junio 2025, ya en v2) hace exactamente esto pero MEJOR:

> "Learning VQE Circuit Parameters with Classical AI for Quantum Phase Transition Detection"
> - Usan VAE + mecanismo de atención (de LLMs) sobre parámetros VQE
> - Detectan QPTs de forma no supervisada
> - Identifican "parámetro de orden generalizado" desde el espacio latente
> - Robusto incluso cuando VQE converge a mínimos locales

Tu método (PCA de θ_opt) es más simple, lo cual puede ser una ventaja pedagógica,
pero no supera al estado del arte. La ventaja de simpleza no justifica un paper cuando
ya existe uno más sofisticado publicado.

### Salvable como

Sección de un paper mayor: "sin necesidad de VAE, un simple PCA + |∂θ/∂h| ya detecta
la transición". Complementa 2506.06678 mostrando que el método más simple funciona
para TFIM. Buen resultado negativo: no necesitas LLMs para esto.

---

## ✅ Paper A: GNN-QEM — Corrección de Errores Cross-Topology via GNN

### Por qué SÍ es publicable y novedoso

**Competencia más cercana**: GEM (arXiv:2604.16815, abril 2026) — "Scalable Quantum Error
Mitigation with Physically Informed Graph Neural Networks"

**Diferenciación clara de tu trabajo vs GEM:**

| Aspecto | GEM (2604.16815) | Tu GNN-QEM |
|---------|-----------------|------------|
| Input graph | Circuito cuántico como grafo | **Hamiltoniano** como grafo |
| Qué predice | Corrección de expectation values | Corrección de energía VQE |
| Transfer | Mismo circuito, más qubits | **Cross-topology** (chain→heavy_hex) |
| Datos de training | Circuitos aleatorios | Datos de VQE con estructura física |
| Contexto | General (cualquier circuito) | Específico a sistemas de espín |
| Zero-shot | A circuitos más grandes | A **topologías no vistas** |

**Tu ángulo único**: GNN-QEM usa el grafo del *Hamiltoniano* (no del circuito) para predecir
correcciones. Esto explota la estructura física del problema, no solo la estructura del ruido.
El resultado de 100% improvement rate en heavy_hex (zero-shot desde chain+ladder) es
cualitativamente diferente a lo que hace GEM.

### Estructura propuesta

**Título**: "Physics-Informed Graph Neural Network for Zero-Shot Cross-Topology Quantum
Error Mitigation in Variational Eigensolvers"

**Contenido**:

1. **Motivación**: ZNE y CDR escalan mal con profundidad. GEM usa grafos de circuito.
   Nosotros proponemos usar el grafo del Hamiltoniano físico como prior inductivo.

2. **Método**:
   - Input: grafo de lattice G(V,E) con features {h_i, J_ij, E_noisy, topology_encoding}
   - Arquitectura: GINConv (3 layers, h=128) + global_mean_pool → ΔE_prediction
   - Training: datos de VQE ruidoso en topologías source (chain_1d, ladder)
   - Inference: zero-shot en topologías target (heavy_hex)

3. **Resultados clave**:
   - In-distribution: +99.4% reducción de error
   - Zero-shot heavy_hex: +72.3% (t=13.28, p<10⁻⁶), 100% improvement rate
   - Ablación: Graph esencial (GNN 100% vs MLP 67% vs Linear 0%)
   - Hallazgo negativo: NO composable con PEA (15/15 regresiones)

4. **Comparación con GEM**: Complementarios — GEM corrige errores de circuito genéricos,
   nosotros explotamos estructura Hamiltoniana para problemas de espín.

5. **Implicación**: Para VQE en modelos de espín, el grafo físico contiene información
   suficiente para mitigar errores sin conocer el modelo de ruido del hardware target.

### Venues sugeridos

- **Physical Review Research** (open access, bien para métodos cuánticos aplicados)
- **Quantum Science and Technology** (IOP, buen fit para QEM)
- **arXiv: quant-ph** primero como preprint

### Datos que ya tienes

- 69 evaluaciones PEA, 81 GF-ZNE
- Cross-topology: chain+ladder → heavy_hex (100% improvement)
- Ablaciones completas (GNN vs MLP vs Linear, con/sin E_noisy)
- Hallazgo negativo (incompatibilidad con PEA) como contribución
- 4 topologías × 3 seeds validadas

### Lo que falta

- [ ] Comparación explícita con GEM (si posible, implementar baseline GEM)
- [ ] Más topologías target (triangular, kagome) para robustecer claim
- [ ] Analizar qué features del grafo son más informativas (edge weights, node degree)
- [ ] Hardware real: al menos 1 run en IBM Heron para validar que funciona fuera de simulación

---

## ✅ Paper B: Atlas Empírico de Expresividad HVA con Conexión al Area Law

### Por qué SÍ es publicable

**Competencia**:

- **Tripathi et al. (2604.20961, abril 2026)**: Benchmarks HVA vs HEA en TFIM 1D/2D/3D
  hasta 27 spins. Confirma HVA > HEA pero NO da fits cuantitativos de h_min(N,p).
- **Brozzi et al. (2507.22550, julio 2025, publicado Springer 2026)**: Define "Hamiltonian
  expressibility" como métrica Monte Carlo. Analiza correlación con calidad VQE. Pero NO
  estudia dependencia en N ni conexión con area law.
- **arXiv:2604.11688 (abril 2026)**: Frustración y expresividad. Introduce bond-resolved
  parameters como fix. Pero solo estudia un modelo (J1-J2), no da atlas multi-topología.

**Tu ángulo único**: Ninguno de los anteriores proporciona:

1. **Fits cuantitativos** h_frontier(N, p) con coeficientes explícitos
2. **Evidencia del area law** como explicación: para p≥3, h_frontier es independiente de N
3. **Multi-topología** (5 topologías × 4 profundidades × N=4-250)
4. **Datos MPS a gran escala** (N=20-250, χ=64, determinista)
5. **Predictor práctico**: θ_smoothness como diagnóstico pre-ejecución

Esto es un **recurso** para la comunidad, no una demostración teórica. Su valor es empírico
y práctico: "antes de gastar QPU time, consulta el atlas".

### Estructura propuesta

**Título**: "Empirical Expressibility Atlas for Hardware-Efficient Variational Ansätze:
Depth Scaling, Topology Dependence, and the Area Law Connection"

**Contenido**:

1. **Problema**: ¿Cuánta profundidad HVA necesito para resolver TFIM a campo h en topología T
   con N qubits? No existe una respuesta cuantitativa en la literatura.

2. **Método**:
   - VQE noiseless (StatevectorEstimator + MPS χ=64) en grid denso (h,N,p,topology)
   - Criterio de éxito: ΔE/gap < 5% (equivalente a fidelidad >99.5%)
   - Fits de regresión: h_frontier = f(N, p) por topología
   - 503 pipeline runs, 5 topologías, p=1-5, N=4-250

3. **Resultados centrales**:

   ```
   p=1: h_frontier = 2.36 + 0.0073·N   (crece con N, R²=0.91)
   p=2: h_frontier = 1.57 + 0.0050·N   (crece más lento, R²=0.95)
   p≥3: h_frontier ≈ 1.6 (p=3), 1.4 (p=4)  (CONSTANTE, independiente de N)
   ```

   **Interpretación area law**: En 1D con gap finito, S(h) no depende de N.
   Si el HVA tiene suficientes capas para capturar ese entrelazamiento (p≥3),
   la frontera no se mueve al escalar N. Para p=1-2, el circuito es más superficial
   que la entropía requiere → frontera crece linealmente (1 ebit/capa < S requerido).

4. **Tabla de expresividad por topología**:
   - chain_1d: más fácil (z_max=2, propagación lineal)
   - heavy_hex: sorprendentemente buena (irregular ayuda a VQE)
   - triangular: 2× más difícil (frustración geométrica)
   - Heisenberg: incompatible a cualquier p (mismatch de simetría)

5. **Predictor de éxito**: θ_smoothness < 0.7 → >80% deploy success.
   Permite decidir si correr MPNN vale la pena ANTES de gastar QPU.

6. **Resultado confirmatorio p=5**: Al alcanzar p=N-1 (N=8), h_frontier→h_c exactamente.
   Confirma que la limitación es puramente de profundidad, no arquitectural.

### Venues sugeridos

- **Physical Review A** (fundamental quantum computing, buen fit para atlas empíricos)
- **New Journal of Physics** (open access, permite papers largos con datos extensos)
- **Quantum** (high impact, pero competencia fuerte)

### Datos que ya tienes

- H_EXPR_MATRIX.md: datos canónicos N=20-250, p=1-4, 3 seeds
- 503 pipeline runs con metadata completa
- Fits de regresión con R² y confidence intervals
- 5 topologías × 4+ profundidades validadas
- Resultado p=5 (confirmación de depth-limited expressibility)
- Conexión explícita con area law (entropy budget argument)
- Comparación con Tripathi2026 y Sumeet2025 como corroboración independiente

### Lo que falta

- [ ] 2D topologías con MPS (actualmente solo 1D chain/ladder son MPS-rigurosos)
- [ ] Comparar con Hamiltonian expressibility métrica de Brozzi2026
- [ ] Añadir error bars formales a los fits (bootstrap CI)
- [ ] Una predicción testeable: para N=30 p=3, h_frontier debería ser ~1.6 → verificar
- [ ] Discusión de cómo escala para otros modelos (longitudinal, frustrated) con fits propios

---

## Análisis de Competencia Detallado

### Landscape completo (julio 2026)

```
GNN + VQE Parameters:
├── Qracle (Zhang, 2505.01236) — GNN → θ_init, grafo unificado H+ansatz
├── Flow-VQE (Zou, 2507.01726) — Normalizing flows, 50× speedup
├── PVLS (2512.04909) — GNN para VQLS
├── VQEzy (2509.17322) — Dataset benchmark, 12K instancias
├── PALQO (2509.20733) — Physics-informed ML, 40 qubits
├── GNN-VQE (2606.08794) — GNN selección de operadores ADAPT
├── Fast ML-VQE (2503.20210) — ML sobre datos intermedios
└── Tu trabajo: MPNN pipeline end-to-end [SUBSUMIDO]

GNN + Quantum Error Mitigation:
├── GEM (2604.16815) — GNN sobre grafo de CIRCUITO, 16 qubits
├── QAGT-MLP (2511.03119) — Graph transformer para QEM
├── ML-QEM (2309.17368) — ML, 100 qubits
├── NN-ZNE (Sun, 2501.01646) — MLP extrapolación ZNE
└── Tu GNN-QEM: GNN sobre grafo de HAMILTONIANO [DIFERENCIADO ✅]

HVA Expressibility:
├── Tripathi2026 (2604.20961) — HVA vs HEA, TFIM 1-3D, N≤27
├── Brozzi2026 (2507.22550) — Hamiltonian expressibility métrica
├── 2604.11688 — Frustración + bond-resolved fix
├── Sumeet2025 (2310.07600) — N/2 layers needed
└── Tu atlas: fits h_min(N,p,topo), area law, N≤250 [DIFERENCIADO ✅]

Unsupervised QPT Detection:
├── 2506.06678 — VAE+attention sobre θ_VQE (DIRECTAMENTE competidor)
├── Prometheus (2602.14928) — VAE para QPT, 3D+quantum
├── 2106.07912 — Unsupervised QPT en QPU
├── 2402.18953 — Signatures of QPT in VQE (level crossings)
└── Tu PCA-θ: más simple pero menos potente [SUBSUMIDO]
```

---

## Resumen Estratégico Final

### Publicar (por orden de prioridad)

1. **Paper A (GNN-QEM)**: PRIORIDAD ALTA. Concepto diferenciado (grafo Hamiltoniano vs
   grafo circuito), resultados fuertes (100% improvement, zero-shot cross-topology),
   hallazgo negativo valioso (incompatibilidad con PEA). Competencia existe pero tu
   ángulo es claro y complementario a GEM.

2. **Paper B (Atlas Expresividad)**: PRIORIDAD MEDIA-ALTA. Recurso práctico para la
   comunidad, datos únicos (N=250, MPS), conexión teórica con area law no publicada
   cuantitativamente. Más "archival" que "breakthrough" pero publicable en PRA/NJP.

### No publicar como papers independientes

3. GNN warm-start: espacio saturado, tu contribución ya está cubierta por 6+ papers.
4. PCA-θ detection: arXiv:2506.06678 ya publicó un método más sofisticado.
5. PEA-ZNE benchmarks: técnica existente de IBM, tu comparación es útil pero incremental.

### Opción C: Paper integrado (pipeline completo)

Si Paper A y B se sienten demasiado "slice-and-dice", la alternativa es un **paper largo**
que presente el pipeline completo con los dos elementos novedosos como secciones:

**Título**: "A Systematic Framework for GNN-Accelerated Phase Characterization: From
Expressibility Limits to Cross-Topology Error Mitigation"

Esto tiene la ventaja de que la *integración* es tu contribución real — ningún otro grupo
ha publicado un pipeline DMRG→VQE→GNN→Deploy con validación a esta escala (503 runs,
5 topologías, N≤250). Pero un paper largo es más difícil de publicar en journals top.

**Venue ideal para paper largo**: Physical Review Research o Quantum Science and Technology.

---

## Referencias Clave para Citar/Diferenciar

| Ref | arXiv | Relación con tu trabajo |
|-----|-------|------------------------|
| GEM | 2604.16815 | Competidor QEM (usa grafo circuito, tú usas grafo H) |
| Qracle | 2505.01236 | Competidor warm-start (prior art que debes citar) |
| Tripathi | 2604.20961 | Corrobora tus hallazgos HVA>HEA en TFIM |
| 2506.06678 | 2506.06678 | Competidor detección QPT (VAE+atención > tu PCA) |
| 2604.11688 | 2604.11688 | Complementa tu atlas (frustración + bond-resolved) |
| Flow-VQE | 2507.01726 | Prior art warm-start (diferente enfoque: generativo) |
| Brozzi | 2507.22550 | Define métrica expresividad (puedes comparar con tu approach) |
| ML-QEM | 2309.17368 | Prior art ML para QEM (escala, pero no graph-based) |

---

*Generado tras búsqueda de literatura julio 2026. Revisar trimestralmente.*
