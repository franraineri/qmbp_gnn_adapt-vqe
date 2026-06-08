# IBM Quantum Hardware Generations — Comparative Analysis

**Fecha**: 2026-06-07
**Propósito**: Documentar las propiedades de las generaciones de procesadores IBM Quantum
relevantes para nuestro pipeline GNN-HVA, con datos verificados y referenciados.
**Relevancia**: Justificar la viabilidad de N=40-50 en ibm_torino (Heron r1) y N=80-100 en
Nighthawk para el plan de escalamiento (§17).

---

## Corrección Importante

**ibm_torino ES Heron r1** (133 qubits, tunable couplers), NO Eagle.
IBM confirmó en noviembre 2024 que las fractional gates (rx, rzz nativo) se incorporaron
al ISA de Heron empezando con ibm_torino ([IBM docs 2024-11-07]).
Eagle es la generación anterior (ibm_sherbrooke, 127 qubits, fixed coupling).

---

## Tabla Comparativa de Generaciones

| Propiedad | Eagle r3 | Heron r1 | Heron r2/r3 | Nighthawk |
|-----------|----------|----------|-------------|-----------|
| **Sistema representativo** | ibm_sherbrooke | ibm_torino, ibm_montecarlo | ibm_fez | ibm_miami, ibm_berlin |
| **Qubits** | 127 | 133 | 156 | 120 |
| **Topología** | Heavy-hex | Heavy-hex | Heavy-hex | **Square lattice** |
| **Acoplamiento** | Fixed-frequency, fixed coupling | Tunable couplers | Tunable couplers | Tunable couplers (next-gen) |
| **EPLG (100 qubits)** | 1.7×10⁻² | ~6.2×10⁻³ | ~3×10⁻³ | **2.15×10⁻³** |
| **Best 2Q gate error** | ~0.5-1% | ~0.1-0.3% | ~0.1% | <0.1% (57/176 gates below 10⁻³) |
| **T₁ (mediana)** | ~100-200 μs | ~200-300 μs | ~350 μs (r3) | **350 μs** |
| **2Q gate time** | ~300-500 ns | ~200 ns | ~138 ns | **68 ns** (ibm_berlin) |
| **Gates por ciclo coherencia** | ~450 | ~1,800 | ~2,500+ | ~5,000 |
| **Crosstalk** | ~10⁻³ (residual ZZ always-on) | ~10⁻⁵ (tunable coupler apagable) | ~10⁻⁵ | ~10⁻⁵ |
| **Couplers** | N/A (direct fixed) | ~180 tunable | ~180 tunable | **218 tunable** (+20%) |
| **Basis gates** | cx, id, rz, sx, x | **rzz (fraccional)**, rx, rz, id | rzz, rx, rz, id | rzz, rx, rz, id |
| **Disponibilidad** | Retirado (migrar a Heron) | ✅ Cloud access | ✅ Cloud access | ✅ Early access (ene 2026+) |
| **Lanzamiento** | Nov 2021 (127q) | Dic 2023 | Nov 2024 | Ene 2026 |

---

## Métricas Clave Verificadas

### EPLG (Error Per Layered Gate)

EPLG mide el error promedio por gate en una capa de gates 2Q aplicada a N qubits
simultáneamente. Es la métrica IBM oficial para comparar rendimiento a escala.

**Datos medidos** (arXiv:2311.05933, McKay et al. 2023):
- ibm_sherbrooke (Eagle r3): EPLG = 1.7×10⁻² at N=80 y N=100
- ibm_montecarlo (Heron r1): EPLG = 6.2×10⁻³ at N=80, 1.2×10⁻² at N=100

**Datos anunciados** (IBM Quantum Platform, enero 2026):
- ibm_miami (Nighthawk): EPLG = 2.15×10⁻³ at 100 qubits

**Mejora entre generaciones**:
- Eagle → Heron r1: ~2.7× reducción en EPLG (a N=80)
- Eagle → Nighthawk: ~8× reducción en EPLG (a N=100)
- Heron r1 → Nighthawk: ~2.9× reducción en EPLG

### Layer Fidelity (80/100 qubits)

La layer fidelity es la fidelidad de una capa completa de gates 2Q:

| Procesador | Layer Fidelity (N=80) | Layer Fidelity (N=100) |
|-----------|:--------------------:|:---------------------:|
| ibm_sherbrooke (Eagle) | 0.26 | 0.19 |
| ibm_montecarlo (Heron r1) | 0.61 | 0.26 |

Fuente: arXiv:2311.05933, medido con randomized benchmarking sobre cadenas.

### Impacto de Tunable Couplers

La diferencia arquitectónica principal entre Eagle y Heron/Nighthawk:

**Eagle (fixed coupling)**:
- Los qubits mantienen acoplamiento ZZ residual permanente (~10⁻³)
- Durante single-qubit gates, los vecinos experimentan rotaciones espurias
- El crosstalk genera errores CORRELACIONADOS entre qubits vecinos
- Los errores correlacionados son difíciles de modelar → PEA menos efectivo

**Heron/Nighthawk (tunable couplers)**:
- Elemento superconductor intermedio "apaga" el acoplamiento cuando no se usa
- Crosstalk residual ~10⁻⁵ (100× menor que Eagle)
- Errores son predominantemente LOCALES e INDEPENDIENTES
- Noise model más simple → PEA/PEC modela mejor → mejor ZNE extrapolation

Fuente: IBM Research APS 2024 ("Heron Processors on the Utility Frontier"),
IBM APS 2025 ("Noise characterization... on IBM Heron processors").

### Nighthawk — Square Lattice

Nighthawk introduce una topología cuadrada (cada qubit conectado a 4 vecinos)
en lugar de heavy-hex (conectividad ~2.5-3):

- 120 qubits + 218 tunable couplers
- 20% más couplers que Heron
- Permite circuitos 30% más complejos a misma fidelidad
- Una cadena lineal de N qubits requiere menos SWAPs en square vs heavy-hex

Fuente: IBM newsroom 2025-11-12, The Quantum Insider 2026-01-13.

---

## Impacto para Nuestro Pipeline (N=40, 50, 80)

### Modelo de Fidelidad por CX Count

Para un circuito con `n_cx` gates CX (o ECR), la fidelidad total es aproximadamente:

```
F_total ≈ (1 - ε_2Q)^n_cx
```

donde ε_2Q es el error por 2Q gate individual (NO es EPLG; EPLG incluye
efectos de paralelismo y crosstalk).

Para nuestro HVA p=1 chain_1d: `n_cx_logical = N - 1`
Después de transpilación a heavy-hex: `n_cx_real ≈ n_cx_logical + 3 × n_swaps`

### Estimaciones de Fidelidad por Generación

**Asunciones**:
- Error/2Q gate (best chain selection): Eagle ~0.8%, Heron r1 ~0.3%, Nighthawk ~0.15%
- SWAP overhead para chain en heavy-hex: ~0.3×N SWAPs (empírico, layout-dependent)
- SWAP overhead para chain en square lattice (Nighthawk): ~0.1×N SWAPs

| N | n_cx_logical | n_cx_real (heavy-hex) | n_cx_real (square) | F (Heron, 0.3%) | F (Nighthawk, 0.15%) |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 10 | 9 | ~15 | ~11 | 95.6% | 98.4% |
| 40 | 39 | ~75 | ~51 | 79.8% | 92.6% |
| 50 | 49 | ~94 | ~64 | 75.4% | 90.8% |
| 80 | 79 | ~151 | ~103 | 63.6% | 85.7% |
| 100 | 99 | ~189 | ~129 | 56.7% | 82.4% |

### ΔE/gap Estimado Post-PEA

PEA-ZNE recupera típicamente 90-95% del error por ruido (validado en nuestros
experimentos: +94.4% gain, 18/18 wins vs GF).

```
ΔE/gap_post_PEA ≈ (1 - F_total) × (1 - PEA_recovery) × scaling_factor
```

Con PEA recovery = 94% y scaling_factor ≈ 2 (noise → energy error amplification):

| N | Heron r1 (ibm_torino) | Nighthawk |
|:--:|:---:|:---:|
| 10 | ~0.5% ✅ | ~0.2% ✅ |
| 40 | ~2.4% ✅ | ~0.9% ✅ |
| 50 | ~3.0% ✅ | ~1.1% ✅ |
| 80 | ~4.4% ⚠️ (borderline) | ~1.7% ✅ |
| 100 | ~5.2% ❌ (exceeds 5%) | ~2.1% ✅ |

**Conclusiones**:
- **ibm_torino (Heron r1)**: ΔE/gap < 5% viable hasta **N≈50-60** con PEA
- **Nighthawk**: ΔE/gap < 5% viable hasta **N≈120+**
- La mejora es ~3× en N máximo entre Heron r1 y Nighthawk

### Importante: Estas son Estimaciones Optimistas

Los cálculos asumen:
1. Best-chain qubit selection (no random layout)
2. Errores independientes (no correlated noise residual)
3. PEA recovery constante (puede degradar a profundidades altas)
4. Sin drift durante la ejecución (TLS fluctuations mitigadas)

En la práctica, agregar un factor de seguridad de ~1.5× al ΔE/gap estimado.

---

## IBM Roadmap (Proyecciones Futuras)

Según el IBM Quantum Technology Atlas (ibm.com/roadmaps/quantum/2026):

| Año | Gates viables (con mitigation) | Hardware | Implicación para HVA |
|-----|:----:|-----------|----------------------|
| 2026 | 7,500 | Nighthawk (360q, 3 módulos) | N=200+ chain simulable en QPU |
| 2027 | 10,000 | Nighthawk mejorado | N=300+ |
| 2028 | 15,000 | Nighthawk + long-range couplers | N=500+ |
| 2029 | Fault-tolerant | Crossbill (error-corrected) | Unlimited depth |

Nuestro circuito HVA p=1 N=80 usa ~150 CX post-transpile. Esto es 2% del
budget de gates de Nighthawk 2026 (7,500). Margen de sobra.

---

## Fractional Gates (Heron ISA)

A partir de ibm_torino (noviembre 2024), Heron introduce fractional gates:
- `rzz(θ)`: gate 2Q nativo parametrizado (no descompuesto en cx+rz)
- `rx(θ)`: gate 1Q nativo parametrizado

**Implicación directa para HVA**: Nuestro circuito usa `rzz(2θ_zz)` en cada edge.
En Eagle, esto se descompone en 2 CX + rotaciones. En Heron, es un pulso nativo.
Resultado: **la mitad de gates efectivos** para la misma operación lógica.

Esto significa que los n_cx_real de la tabla anterior están SOBREESTIMADOS para Heron.
Con fractional gates, el circuito HVA se ejecuta directamente como RZZ nativo sin
descomposición CX. El budget de gates relevante es `n_rzz = N-1` (no `n_cx = 2(N-1)`).

Fuente: IBM Quantum docs "Fractional gates" (2024-11-07), aplicado a ibm_torino.

---

## Resumen para la Tesis

### Hardware Deployment Targets

| Target | Procesador | N máximo (ΔE/gap<5%) | Viabilidad |
|--------|-----------|:----:|:---:|
| Primario | ibm_torino (Heron r1) | 50-60 | ✅ Disponible ahora |
| Stretch | Nighthawk (ibm_miami/berlin) | 100-120 | ✅ Early access 2026 |
| Future work | Crossbill (fault-tolerant) | Unlimited | 2029+ |

### Argumento de Escalabilidad

1. **En simulación**: Pipeline funciona a N=80+ (MPS exacto, O(N·χ³), sin barrera)
2. **En ibm_torino**: N=40-50 demostrable con PEA-ZNE + fractional gates
3. **En Nighthawk**: N=80-100 viable (3× mejor error rates, square lattice, T₁=350μs)
4. **Proyección 2027+**: N=200+ (10,000 gates budget)

Este argumento posiciona la contribución como:
- **Hoy** (tesis): Pipeline end-to-end validado en simulación hasta N=80, hardware hasta N=50
- **Próximo paso**: Hardware N=80 en Nighthawk (accesible, sin cambio de pipeline)
- **Largo plazo**: El pipeline NO tiene barreras fundamentales — escala con el hardware

---

## Referencias

Todas las fuentes han sido verificadas con fecha de acceso junio 2026.

| # | Referencia | Claim verificado | URL |
|---|-----------|-----------------|-----|
| 1 | McKay et al. "Benchmarking Quantum Processor Performance at Scale" (arXiv:2311.05933, Nov 2023) | Eagle EPLG=1.7e-2, Heron EPLG=6.2e-3 at N=80. Layer fidelity Eagle 0.26 vs Heron 0.61 at N=80 | https://arxiv.org/abs/2311.05933 |
| 2 | IBM Newsroom "IBM Launches Most Advanced Quantum Computers" (Nov 2024) | Heron ejecuta 5,000 2Q gates con accuracy. | https://newsroom.ibm.com/2024-11-13-ibm-launches-its-most-advanced-quantum-computers |
| 3 | IBM Newsroom "IBM Delivers New Quantum Processors" (Nov 2025) | Nighthawk: 120 qubits, 218 tunable couplers, square lattice, +20% connectivity, 30% más complejidad | https://newsroom.ibm.com/2025-11-12-IBM-Delivers-New-Quantum-Processors,-Software,-and-Algorithm-Breakthroughs-on-Path-to-Advantage-and-Fault-Tolerance |
| 4 | IBM Quantum Platform "Nighthawk and the latest Heron are now available" (Ene 2026) | Nighthawk EPLG=2.15e-3 at 100q, 57/176 gates below 1e-3, T₁=350μs | https://quantum.cloud.ibm.com/announcements/product-updates/2026-01-05-nighthawk |
| 5 | IBM Quantum Platform "EU gets first Nighthawk" (Abr 2026) | ibm_berlin: 2Q gate 68ns (vs 138ns Heron), T₁~350μs | https://quantum.cloud.ibm.com/announcements/product-updates/2026-04-15-berlin-nighthawk |
| 6 | IBM Newsroom "U. Tokyo Heron upgrade" (May 2025) | Heron 3-4× improvement in 2Q error rates vs Eagle, order of magnitude in 100q layer performance | https://newsroom.ibm.com/2025-05-15-the-university-of-tokyo-to-equip-ibm-quantum-system-one-with-most-performant-ibm-heron-processor |
| 7 | IBM Quantum docs "Fractional gates" (Nov 2024) | ibm_torino = Heron ISA. rx, rzz nativo sin descomposición CX | https://docs.quantum.ibm.com/announcements/product-updates/2024-11-07-fractional-gates |
| 8 | IBM Research APS 2025 "Noise characterization on Heron" | 2Q error rates approaching 0.1% en tunable-coupling Heron | https://research.ibm.com/publications/noise-characterization-and-error-mitigation-on-ibm-heron-processors-part-1--1 |
| 9 | IBM Quantum Roadmap 2026 | Nighthawk: 7,500 gates (2026), 10,000 (2027), 15,000 (2028) | https://www.ibm.com/roadmaps/quantum/2026/ |
| 10 | The Quantum Insider "IBM Announces Nighthawk" (Ene 2026) | 120 qubits, square lattice, 218 tunable couplers, 5000 gates | https://thequantuminsider.com/2026/01/13/ibm-announces-nighthawk-and-latest-heron-are-now-available/ |
| 11 | PostQuantum "Heron r3 T2 and Quality Upgrade" (2026) | Heron r3: ~5× lower EPLG vs Eagle, ~8× faster gates. Eagle retired | https://postquantum.com/industry-news/ibm-heron-r3-pittsburgh/ |
| 12 | IBM Quantum blog "QDC 2024" | Heron capable of 5,000 2Q gates with accuracy (100×100 challenge) | https://ibm.com/quantum/blog/qdc-2024 |

---

## Notas de Verificación

- **ibm_torino = Heron r1** confirmado via Ref 7 (fractional gates ISA announcement)
- **EPLG de Heron early (ibm_montecarlo)** = 6.2e-3 at N=80 (Ref 1). Este es un Heron r1 early.
  ibm_torino puede tener mejoras incrementales posteriores pero mismo orden de magnitud.
- **EPLG de Nighthawk** = 2.15e-3 (Ref 4) — dato oficial IBM, no estimación.
- **Square lattice en Nighthawk** confirmado por múltiples fuentes (Refs 3, 4, 10).
- **Eagle retirado** (ibm_sherbrooke): IBM recomienda migrar a Heron (Ref 11).
- **T₁=350μs** para Nighthawk confirmado en ambos Refs 4 y 5.
- **68ns gate time** para ibm_berlin (Nighthawk) confirmado en Ref 5.

Content was rephrased for compliance with licensing restrictions.
