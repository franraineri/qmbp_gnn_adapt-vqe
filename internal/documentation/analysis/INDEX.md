# Índice Detallado — `internal/documentation/analysis`

**Archivos**: 42 documentos markdown
**Total**: 103,445 palabras, 14,794 líneas

---

## 1. [02_seed_robustness.md](02_seed_robustness.md)
**Estudio 2 — Robustez por Semilla y Reproducibilidad**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| — | 41 | 235 | 1.5 KB | 9 | 0 |

**Tags**: `tesis`, `topology`

> Pregunta: ¿Los resultados son seed-independent?

**Hallazgos / Aseveraciones** (5):

- The pipeline is reproducible across random seeds (std(ΔE/gap) = 0.010 at N=10). Seed variance is smaller than the effect of hyperparameter choices (restarts, hidden_dim), confirmin…
- *El pipeline ES seed-independent**: Todos los seeds pasan el criterio de 5%
- *La varianza inter-seed (std=0.010) es comparable a la varianza inter-topología (std=0.005)**
- *Los hiperparámetros (restarts, hidden_dim) tienen más impacto que el seed**
- *G5 experiment (92% pass rate)** confirma esto a escala: solo 1/12 puntos falla marginalmente

**Contenido** (4 secciones):

- **Datos de Semillas (Ladder N=10)** (L5)
- **Comparación: Varianza Inter-Seed vs Inter-Topología** (L20)
- **Conclusión** (L29)
- **Implicación para la Tesis** (L36)

---

## 2. [03_hyperparameter_sensitivity.md](03_hyperparameter_sensitivity.md)
**Estudio 3 — Sensibilidad a Hiperparámetros**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| — | 62 | 484 | 2.7 KB | 18 | 0 |

**Tags**: `tesis`, `topology`

> Pregunta: ¿Los hiperparámetros óptimos se mantienen across topologies?

**Métricas clave**: ΔE/gap < 5%

**Hallazgos / Aseveraciones** (5):

- The optimal hyperparameters (hidden_dim=128, n_restarts=5, patience=500) are topology-independent. The same configuration achieves ΔE/gap < 5% across chain_1d, ladder, and triangul…
- *hidden=128 es robusto**: Óptimo en chain_1d, ladder, y triangular
- *restarts=5 es el sweet spot**: Suficiente para ladder y triangular. 7 no aporta
- *Los hiperparámetros NO necesitan re-tuning por topología**: La misma config funciona
- *Caveat**: Los datos tienen confounding (runs con diferentes restarts también varían en h-grid). Para una comparación limpia se necesitarían runs controlados

**Contenido** (4 secciones):

- **Hidden Dimension** (L5)
  - Ladder (N=10)
  - Triangular (N=10)
  - Conclusión Hidden Dim
- **VQE Restarts** (L27)
  - Ladder
  - Triangular
  - Conclusión Restarts
- **Conclusiones Globales** (L50)
- **Implicación para la Tesis** (L57)

---

## 3. [04_verdict_reconciliation.md](04_verdict_reconciliation.md)
**Estudio 4 — Reconciliación de Verdicts**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| — | 72 | 737 | 4.4 KB | 16 | 0 |

**Tags**: `mpnn`, `tesis`, `pea`

> Pregunta: ¿Los 8 "failed" del digest son realmente fallos o thresholds demasiado estrictos?

**Métricas clave**: ΔE/gap > 5% · ΔE/gap < 5% · ΔE/gap = 5.2%

**Hallazgos / Aseveraciones** (8):

- Of 15 V8 experiments, 8 confirmed their hypotheses, 5 produced valid negative findings (contributing to understanding framework limitations), and only 2 genuinely failed (analytica…
- *Project-status dice**: "ALL VQE minima are genuine (0 saddle points). ✅
- *Project-status dice**: "Pipeline is seed-independent (std=0.004, all seeds pass). ✅
- *Project-status dice**: "9 points sufficient (47% reduction from 17). ✅
- *Project-status dice**: "No barren plateaus: Landscape fluctuation >1.0 everywhere. ✅
- *Project-status dice**: "Weight gradient peaks detect h_c when MPNN loss≈0.002. ✅
- *Confirmed: A3, A3_N20, B4, G5, G1, F3, D1, B2 = 8 confirmed** ✅
- *Failed: B1, C3 = 2 failed** ❌ (genuine failures)

**Contenido** (5 secciones):

- **Estado Actual del Digest** (L5)
- **Análisis Detallado** (L18)
  - B4 — "No saddle points" (75% pass, threshold=100%)
  - G5 — "Seed-independent" (92% pass, threshold=100%)
  - G1 — "9 points sufficient" (86% pass, threshold=90%)
  - F3 — "Fluctuation > 1.0" (0% pass, threshold=100%)
  - D1 — "Peak near h_c" (1% pass, threshold=80%)
- **Resumen de Correcciones Necesarias** (L45)
- **Verdicts Corregidos** (L56)
- **Implicación para la Tesis** (L66)

---

## 4. [05_negative_findings.md](05_negative_findings.md)
**Estudio 5 — Hallazgos Negativos como Contribución**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| — | 58 | 611 | 4.0 KB | 6 | 0 |

**Tags**: `mpnn`, `tesis`, `tfim`, `scaling`

> Pregunta: ¿Qué aprendimos de los 5 experimentos "rejected"?

**Métricas clave**: ΔE/gap=1.26 · ΔE/gap = 1.26

**Hallazgos / Aseveraciones** (5):

- *Model-specific** (no universal) — E4
- *Already near-optimal** (no room for DyPP) — F1
- *UQ requires proper methodology** — G2
- *Scaling is non-trivial** — G3
- *Simple heuristics beat complex metrics** — G4

**Contenido** (3 secciones):

- **Tabla de Hallazgos Negativos** (L7)
- **Análisis por Hallazgo** (L17)
  - E4 — HVA es Model-Specific ❌
  - F1 — Warm-Start Ya Es Óptimo ❌
  - G2 — Ensemble UQ No Calibrado ❌
  - G3 — N=6 Findings No Transfieren ❌
  - G4 — κ No Predice Dificultad ❌
- **Implicación Global** (L49)

---

## 5. [06_zne_boundary.md](06_zne_boundary.md)
**Estudio 6 — Frontera ZNE: N=6 vs N=10**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| — | 73 | 539 | 3.1 KB | 12 | 0 |

**Tags**: `zne`, `tesis`, `noise`

> Pregunta: ¿Dónde exactamente falla ZNE y por qué?

**Métricas clave**: R² = 0.991 · gain = +74.1% · R² > 0.95 · R²=0.98 · gain=+48.5% · R²=0.94 · gain=-14.4%

**Hallazgos / Aseveraciones** (7):

- ZNE effectiveness is governed by the total CX gate count, not system size alone. At ~18 CX gates (N=6 p=2 or N=10 p=1), linear ZNE achieves +48-74% error reduction. Above ~36 CX ga…
- `ext_noisy_p1` (ladder, N=10, p=1): gain = +74.1%, R² = 0.991
- *Frontera clara**: ZNE funciona a N=6 (gain +48.5%), falla a N=10 (gain -14.4%)
- *La frontera es por CX count, no por N**: p=1 a N=10 funciona (mismos CX que p=2 a N=6)
- *R² es necesario pero no suficiente**: R²>0.95 en ambos lados de la frontera
- *Topología importa**: chain_1d > ladder > triangular (más conectividad = más CX = peor ZNE)
- *Solo chain_1d a N=6 cumple success criteria** (9/23 = 39%). Ladder y triangular nunca

**Contenido** (7 secciones):

- **Resultado por System Size** (L5)
- **Resultado por Topología** (L12)
- **Estadísticas Detalladas** (L20)
  - N=6 ZNE
  - N=10 ZNE
- **Hallazgo Clave: p=1 como Solución** (L34)
- **Insight: R² es Engañoso** (L49)
- **Conclusiones** (L59)
- **Implicación para la Tesis** (L67)

---

## 6. [07_outliers.md](07_outliers.md)
**Estudio 7 — Outliers y Casos Extremos**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| — | 65 | 496 | 3.2 KB | 10 | 0 |

**Tags**: `mpnn`, `tesis`

> Pregunta: ¿Qué podemos aprender de los peores resultados?

**Métricas clave**: ΔE/gap > 0.2094 · ΔE/gap = 5.3%

**Hallazgos / Aseveraciones** (6):

- All detected outliers (9/135, 6.7%) have identifiable root causes: MPNN overfitting on sparse training grids (4 cases) or warm-start chain disruption near the phase boundary (5 cas…
- *Todos los outliers son explicables** — no hay bugs ocultos
- *Dos categorías dominantes**: overfitting (gen.gap) y warm-start roto (θ-sweep)
- *Los outliers extremos (>1.0) son todos variantes "ext_"** — diseñadas para probar límites
- *Sin outliers, el pipeline tiene mean ΔE/gap = 5.3%** — muy cerca del threshold de 5%
- *El outlier más extremo (14.4)** es un solo seed (42) en triangular N=10 — probablemente un local minimum catastrófico

**Contenido** (6 secciones):

- **Detección de Outliers (IQR Method)** (L5)
- **Outliers Identificados** (L11)
- **Análisis de Causas Raíz** (L25)
  - Categoría 1: High Generalization Gap (4 outliers)
  - Categoría 2: Rough θ-Sweep (5 outliers)
  - Categoría 3: Diseño Experimental (2 outliers)
- **Distribución Global (sin outliers)** (L44)
- **Conclusiones** (L52)
- **Implicación para la Tesis** (L60)

---

## 7. [08_summary.md](08_summary.md)
**Resumen del Análisis — GNN-HVA Framework**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-09 | 604 | 4,462 | 28.0 KB | 149 | 3 |

**Tags**: `zne`, `hardware`, `mpnn`, `tesis`, `heisenberg`, `tfim`

> > ⚠️ NOTA (2026-06-09): Las estadísticas globales de este encabezado reflejan el estado a 2026-05-28 (135 runs, 15 experiments).

**Métricas clave**: ΔE/gap < 5% · gain=+48.5% · gain=-14.4% · gain=+49% · ΔE/gap=0.56% · ΔE/gap=10.67

**Hallazgos / Aseveraciones** (8):

- *⚠️ NOTA (2026-06-09): Las estadísticas globales de este encabezado reflejan el estado a 2026-05-28 (135 runs, 15 experiments). El estado actual verificado es: 329 noiseless, 93 no…
- *Deleted (2026-06-09)**: `raw_all_results.json`, `raw_p1_zne_validation.json`, `table_experiments.md`, `table_topology_n10.md`, `table_zne_boundary.md`, `worklog/` (all entries). A…
- Automated root cause analysis of 76 non-passing pipeline runs reveals that 45% of failures are caused by warm-start chain breaks (θ_smoothness > 1.0), 25% by MPNN overfitting (gen_…
- Full details: `documentation/analysis/12_tier1_session_results.md` Hardware-specific: `documentation/analysis/11_hardware_rehearsal_findings.md`
- Valid regime for ladder p=1 N=10 is h≥3.0 (2/3 pass) with
- HVA p≤2 fundamentally cannot express Heisenberg XXZ ground states at N=6
- *Evidencia**: 4 topologías, todas con median ΔE/gap < 5%
- *Datos**: chain_1d (med=0.029, n=38), ladder (med=0.017, n=24 top-15), triangular (med=0.037, n=50)

**Contenido** (14 secciones):

- **1. Estado General del Framework** (L16)
- **2. Conclusiones de Alta Confianza** (L31)
  - 2.1 El framework es topology-agnostic
  - 2.2 ZNE tiene frontera clara en ~18 CX gates
  - 2.3 p=1 recupera ZNE a N=10 (NUEVO — confirmado)
  - 2.3b PEA-ZNE es universalmente superior a GF-ZNE (NUEVO — 2026-06-04)
  - 2.4 100% del error es MPNN prediction (no HVA)
  - 2.5 Los hiperparámetros son robustos cross-topology
  - … (+4 más)
- **3. Hallazgos Negativos (Contribuciones Válidas)** (L91)
- **4. Métricas Clave para la Tesis** (L104)
  - Cross-Topology (N=10, top-15 optimized)
  - ZNE Boundary (CX-budget hypothesis)
  - Experiment Verdicts
- **5. Thesis Implications — Índice para Compilación** (L130)
  - Para compilar Chapter 5
- **5b. p=1 Pipeline Results (N=10, Round 2) — 2026-05-30** (L159)
  - p=1 N=10 Definitive Results (R2, corrected h_test)
  - p=1 N=6 Verification Results (2026-05-30)
  - p=1 Ladder N=10 Boundary Verification (2026-05-30)
- **5c. Heavy-Hex Topology Results (IBM Heron Native, 2026-05-31)** (L203)
  - p=2 Heavy-Hex N=10 (14 runs, 7 min total)
  - p=1 Heavy-Hex N=10 — Hardware Deployment Candidate
  - N=16 Heavy-Hex (scaling limit)
  - Heavy-Hex Key Findings
  - N=16 Scaling Data (Phase 2 only — Phase 3/4 did not complete)
  - N=24 Scaling Data (partial — only 1 complete Phase 2)
- **6. Archivos Generados** (L301)
- **7. Status: ANALYSIS COMPLETE** (L332)
- **8. Automated Failure Diagnosis (2026-05-30)** (L339)
  - Root Cause Distribution (all failures + marginals)
  - Key Diagnostic Findings
  - Implicación para la Tesis
- **9. Heisenberg XXZ Model — Definitive Negative Result (2026-06-01)** (L392)
  - Summary
  - Why It Fails
  - Thesis Value
  - Cross-N Scaling (N=6, 10, 16)
  - Full Details
- **10. S-Series Scalability Experiments (2026-06-01)** (L443)
  - S-Series Verdicts (post-validation corrected)
  - Key Findings (post-validation)
  - Updated Experiment Verdicts (with S-series, post-validation)
- **Session 2026-06-03 — Hardware Readiness & MPS Validation** (L503)
  - New Experiments Executed
  - Updated Experiment Verdicts (with session additions)
  - Digest Cross-Reference (2026-06-03)
  - Key Takeaways for Hardware Deployment
  - References
- **Session 2026-06-03 (cont.) — Tier 1 Experiments & Hardware Rehearsal** (L558)
  - New Experiments Executed
  - Updated Statistics
  - Critical Finding: ZNE Strategy for Hardware
  - Preflight Enhancements

---

## 8. [09_thesis_tables.md](09_thesis_tables.md)
**Tablas Definitivas para la Tesis — Chapter 5**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-05-27 | 897 | 7,655 | 44.3 KB | 270 | 1 |

**Tags**: `zne`, `hardware`, `mpnn`, `tesis`, `heisenberg`, `tfim`

> Generadas: 2026-05-27

**Métricas clave**: ΔE/gap = 0.028 · ΔE/gap < 3% · ΔE/gap = 0.017 · ΔE/gap = 0.037 · ΔE/gap < 5% · ΔE/gap > 8

**Hallazgos / Aseveraciones** (4):

- *Context**: Analysis of 2026-06-15 session. Full detail: `documentation/binnacles/binnacle-hamiltonian-candidates.md` (Addenda 1 & 2), `documentation/analysis/21_future_work_techni…
- A Hamiltonian is compatible with the GNN-HVA framework if and only if its ground state entanglement entropy in the operating regime satisfies S(L/2) ≲ log(p+1). For p=2, this limit…
- The GNN-HVA pipeline operates at utility scale (N=40-50) with ΔE/gap < 1% across the valid operating regime. The MPS-based VQE evaluation scales polynomially with system size, and …
- The valid regime boundary for ladder p=1 N=10 is h≥3.0 (not h≥2.0

**Contenido** (25 secciones):

- **Table 5.1 — Chain 1D, N=10 (Baseline Topology)** (L9)
- **Table 5.2 — Ladder, N=10 (Higher Connectivity)** (L31)
- **Table 5.3 — Triangular, N=10 (Highest Connectivity)** (L54)
- **Table 5.4 — Cross-Topology Comparison (N=10, optimized configs only)** (L77)
- **Table 5.5 — ZNE Boundary (N=6 vs N=10)** (L97)
- **Table 5.6 — Experiment Verdicts Summary** (L112)
- **Table 5.7 — p=1 Pipeline Performance (N=10, 3 seeds each) — 2026-05-30** (L131)
- **Table 5.11 — p=1 Pipeline at N=6 (Verification R1, 2026-05-30)** (L158)
- **Table 5.12 — p=1 Ladder N=10 Boundary Verification (2026-05-30)** (L185)
- **Table 5.13 — Heavy-Hex Topology (IBM Heron Native, N=10, 2026-05-31)** (L209)
  - p=1 Heavy-Hex — Hardware Deployment Candidate
  - p=2 Heavy-Hex — Cross-Topology Comparison
  - p=1 Heavy-Hex ZNE (Noisy Simulation, 2026-05-31)
  - Pre-Hardware Parameter Optimization (2026-06-01)
- **Table 5.8 — p=1 Scaling Limits (N=16, N=24 — Phase 2 only)** (L275)
- **Table 5.9 — Failure Root Cause Analysis (174 pipeline runs, automated diagnosis)** (L310)
- **Table 5.10 — p=1 vs p=2 Direct Comparison (COMP-4, triangular N=10, matched config)** (L334)
- **Table 5.14 — Heisenberg XXZ Model: HVA Expressibility Limits (N=6, p=2, 2026-06-01)** (L356)
  - Anisotropy Sweep (chain_1d, seed=42, 10 restarts, 1500 maxiter)
  - Topology Comparison (Δ=1.0, seed=42)
  - Seed & Restart Independence (Δ=1.0, chain_1d)
  - Best Non-TFIM Case (XY on ladder, 3 seeds)
  - Cross-N Scaling (N=6, 10, 16)
  - Sanity Check Results (VQE Verification)
  - … (+1 más)
- **Table 5.15 — Model-Agnostic Framework Validation (2026-06-01)** (L471)
- **Table 5.16 — Entanglement Entropy at Valid Regime Boundary (S1, 2026-06-01)** (L494)
- **Table 5.17 — Data Efficiency at N=10 (S4, 2026-06-01)** (L516)
- **Table 5.18 — N=20 p=1 Full Pipeline with MPNN (S5, 2026-06-01)** (L542)
- **Table 5.19 — Landscape Analysis at N=20 (S3, 2026-06-01)** (L562)
- **Table 5.20 — MC-Dropout Uncertainty Quantification (S6, 2026-06-01)** (L580)
- **Table 5.21 — Cross-Topology Transfer (S2, 2026-06-01)** (L604)
- **Table 5.22 — Weight-Space Phase Detection: Finite-Size Scaling Attempt (S8/S8b, 2026-06-01)** (L623)
  - S8: MLP Proxy (h→θ, h=128, dropout=0.1, 6000 epochs)
  - S8b: MPNN GINConv (graph input, h=128, L=3, 6000 epochs)
  - Scaling Fit Results
- **Table 5.22 — Model Extensibility: TFIM + Longitudinal Field (E4b)** (L670)
  - Fidelity Comparison: Standard HVA vs Extended HVA
  - Cross-Topology at g=0.3
  - Hardware Viability (2Q Gate Count, FakeTorino)
- **Table 5.22b — Model Extensibility: Applicability Boundary of HVA p≤2** (L709)
  - Scalability of TFIM + Longitudinal Field
  - Why XX+YY Models Are Incompatible with the Framework
  - Boundary Diagram: Viable Operating Regimes
- **Table 5.23 — MPS Scaling: N=40/50 Pipeline Performance (2026-06-07)** (L818)
  - N=40, p=1, chain_1d (seed=42, 5 h-points)
  - N=50, p=1, chain_1d (seed=42, 5 h-points)
  - Scaling Law Validation
  - Transpilation Audit (FakeTorino, opt_level=2)
  - Cross-N Comparison (all at p=1, chain_1d)
  - Key Findings
  - … (+2 más)

---

## 9. [11_error_decomposition.md](11_error_decomposition.md)
**Estudio 2A — Error Decomposition por Topología**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| — | 93 | 467 | 3.1 KB | 29 | 0 |

**Tags**: `mpnn`, `tesis`

> Pregunta: ¿El error viene del HVA (circuit) o del MPNN (prediction)?

**Hallazgos / Aseveraciones** (5):

- Energy decomposition analysis reveals that 100% of the deployment error originates from MPNN prediction, not HVA expressibility. The VQE consistently finds the global minimum withi…
- *100% del error es MPNN prediction** en todas las topologías y todos los runs
- `error_from_circuit = 0.0` significa que el VQE encontró el mínimo exacto del HVA
- *El MPNN es el único componente que puede mejorar** — mejor arquitectura, más datos,
- *Ranking de dificultad MPNN**: chain_1d (med=0.034) < ladder (0.083) < triangular (0.101)

**Contenido** (4 secciones):

- **Datos** (L5)
  - chain_1d (n=30)
  - ladder (n=47)
  - triangular (n=54)
  - unknown (n=3)
- **Resumen Comparativo** (L67)
- **Conclusiones** (L76)
- **Implicación para la Tesis** (L87)

---

## 10. [11_hardware_rehearsal_findings.md](11_hardware_rehearsal_findings.md)
**Hardware Rehearsal Findings — 2026-06-03**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| — | 230 | 1,296 | 8.6 KB | 22 | 1 |

**Tags**: `zne`, `hardware`, `mpnn`, `thesis`, `pea`, `noise`

> Total time: 264s (~4.4 min)

**Métricas clave**: ΔE/gap ≤ 0.8% · R²=0.04 · R² < 0.90 · R² < 0.8

**Hallazgos / Aseveraciones** (8):

- select_layouts_low_ces(bound_circuit, backend, candidates, n_select=3, max_ces=0.5)
- MPNN predictions are excellent in-regime. The warm-start
- Phase classification will work on hardware even with
- Shot noise alone (std=1.2%) is well below the 3% threshold
- *Switch layout selection to `select_layouts_low_ces(max_ces=0.5)`**
- Current: `select_layouts_by_circuit_ces` (spread-maximizing, allows CES outliers)
- Correct: `select_layouts_low_ces` (filters out CES > 0.5)
- Location: `scripts/run_hardware_rehearsal.py` Section 2 + hardware deployment script

**Contenido** (6 secciones):

- **Execution Summary** (L3)
- **Critical Finding: CES Outlier in Layout Selection** (L17)
  - Problem
  - Root Cause
  - Solution
  - Impact on Hardware Deployment
- **Section-by-Section Analysis** (L63)
  - Section 1: MPNN Prediction Quality ✅
  - Section 2: End-to-End Noisy Pipeline ❌
  - Section 3: Observable SNR & Classification ✅
  - Section 4: Layout Stability ❌
  - Section 5: Shot Noise Sensitivity ✅
- **Action Items for Hardware Deployment** (L126)
  - Must Fix (Blocking)
  - Should Fix (Recommended)
  - Good to Know (No action needed)
- **Interpretation for Thesis** (L153)
- **Second Run: After `select_layouts_low_ces` Fix** (L170)
  - Results
  - Diagnosis: ZNE Requires CES Spread
  - Root Cause: heavy_hex + p=1 Has Uniform Noise
  - Implication for Hardware Deployment
  - Updated Recommendation

---

## 11. [12_smoothness_correlation.md](12_smoothness_correlation.md)
**Estudio 2C — Correlación θ-smoothness vs ΔE/gap**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| — | 35 | 202 | 1.2 KB | 8 | 0 |

**Tags**: `pea`

> Pregunta: ¿θ-smoothness predice la calidad del pipeline sin ejecutar Phase 4?

**Hallazgos / Aseveraciones** (3):

- θ-smoothness como detector de problemas: valores > 1.0 casi siempre
- Como predictor cuantitativo de ΔE/gap: correlación débil-moderada
- *Regla práctica**: Si θ-smoothness > 1.0, investigar antes de confiar en Phase 4

**Contenido** (4 secciones):

- **Correlación Global** (L5)
- **Análisis por Banda de θ-smoothness** (L13)
- **Correlación por Topología** (L21)
- **Conclusiones** (L29)

---

## 12. [12_tier1_session_results.md](12_tier1_session_results.md)
**Tier 1 & Hardware Rehearsal — Session Results (2026-06-03)**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-04 | 172 | 1,122 | 7.3 KB | 39 | 1 |

**Tags**: `zne`, `hardware`, `mpnn`, `thesis`, `tfim`, `pea`

> Experiments executed: T1a, T1b, T1c, HW_REHEARSAL

**Métricas clave**: ΔE/gap > 3 · R²=0.9999 · gain=+89.5% · R² = 0.9999 · ΔE/gap ≤ 0.8% · gain=16%

**Hallazgos / Aseveraciones** (3):

- The MPNN can interpolate in h (cross-validation at known J₂ achieves
- ZNE transfers perfectly to the longitudinal model:
- *D1 generalizes to frustrated TFIM.** The weight gradient peaks

**Contenido** (8 secciones):

- **Digest Summary (post-session)** (L10)
- **T1a Dense: MPNN 2D Predictor — Dense J₂ Grid (8 values)** (L20)
- **T1a: MPNN 2D Predictor (h × J₂ interpolation)** (L53)
- **T1b: ZNE on TFIM+longitudinal (FakeTorino, p=1)** (L76)
- **T1c: D1 Weight-Space Phase Detection for Frustrated TFIM** (L99)
- **HW_REHEARSAL: Hardware Deployment Rehearsal** (L123)
- **Updated Statistics (post-session)** (L152)
- **Coverage Gaps Remaining (from `scan_coverage.py`)** (L166)

---

## 13. [13_controlled_restarts.md](13_controlled_restarts.md)
**Estudio 1B — Controlled Restarts Comparison (Validated)**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| — | 49 | 369 | 2.3 KB | 10 | 0 |

**Tags**: `mpnn`, `tesis`

> Source: `scripts/experiment_runners/run_thesis_variants-ladder.py` (Group A: NL-A1/A3/A5/A7)

**Hallazgos / Aseveraciones** (5):

- At h=2.5 (deep paramagnetic regime), the VQE landscape on ladder N=10 is sufficiently benign that even 1 restart finds the global minimum. The recommendation of 5 restarts is a con…
- *En ladder N=10 a h_test=2.5**: El landscape es tan benigno que incluso 1 restart funciona
- *La variabilidad inter-run (seed MPNN)** es mayor que el efecto de restarts
- *restarts=5 es una elección segura** pero no es estrictamente necesario en este régimen
- *El claim "5 es el sweet spot" se mantiene** como recomendación conservadora, pero

**Contenido** (5 secciones):

- **Resultados (latest validated runs)** (L8)
- **Todas las ejecuciones (para ver variabilidad)** (L17)
- **Análisis** (L26)
- **Conclusión Revisada** (L33)
- **Implicación para la Tesis** (L43)

---

## 14. [13_hardware_zne_improvements.md](13_hardware_zne_improvements.md)
**Hardware ZNE Implementation — Improvements & Action Plan**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-04 | 551 | 2,569 | 19.5 KB | 40 | 18 |

**Tags**: `zne`, `hardware`, `thesis`, `pea`, `noise`

> Date: 2026-06-04

**Métricas clave**: R²=0.04 · R²=0.47 · R²=0.998 · R²=0.94 · R² < 0.5 · ΔE/gap=0.915

**Hallazgos / Aseveraciones** (1):

- No new dependencies required. All proposed patterns are

**Contenido** (14 secciones):

- **Executive Summary** (L9)
- **Issue 1: CES-ZNE Fails on Heavy_hex (CRITICAL)** (L27)
  - Problem
  - Evidence
  - Fix: Use IBM Runtime's Built-in ZNE (Server-Side)
- **Issue 2: No Dual-Amplifier Strategy (HIGH)** (L57)
  - Problem
  - Evidence
  - Fix: Tiered ZNE with R²-Based Fallback
- **Issue 3: Per-Site Observables Not ZNE-Mitigated (MEDIUM)** (L86)
  - Problem
  - Evidence
  - Assessment
  - Fix: Submit observables through the same ZNE-configured estimator
- **Issue 4: No R² Quality Gate Before Verdict (MEDIUM)** (L120)
  - Problem
  - Evidence
  - Fix: Add R² Quality Gate
- **Issue 5: SPSA Uses Full HardwareBackend.evaluate() (LOW)** (L153)
  - Problem
  - Assessment
  - Fix (Optional): Lightweight SPSA Evaluation
- **Implementation Plan** (L184)
  - Priority Order
- **Detailed Implementation Guide** (L198)
  - Improvement 1: Remove Client-Side CES-ZNE, Trust IBM Server-Side ZNE
  - Improvement 2: Dual-Amplifier Strategy with Automatic Fallback
  - Improvement 3: Observable ZNE Propagation
  - Improvement 4: R² Quality Gate
  - Improvement 5: Lightweight SPSA Evaluation (Optional)
- **Verification Checklist** (L445)
- **Consistency with Simulation Results** (L459)
- **Recommended Implementation Order** (L474)
- **File Changes Summary** (L496)
- **Non-Changes (Explicitly Preserved)** (L510)
- **Feasibility Verification (Automated, 2026-06-04)** (L524)

---

## 15. [13_next_steps_local_simulation.md](13_next_steps_local_simulation.md)
**Next Steps — Local Simulation Opportunities**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-03 (updated with deep ROI analysis) | 326 | 2,199 | 14.9 KB | 42 | 2 |

**Tags**: `zne`, `hardware`, `mpnn`, `tesis`, `heisenberg`, `tfim`

> Date: 2026-06-03 (updated with deep ROI analysis)

**Métricas clave**: R² > 0.95 · ΔE/gap > 5%

**Hallazgos / Aseveraciones** (1):

- Only ONE simulation has genuine thesis-blocking value: gate-folding ZNE

**Contenido** (11 secciones):

- **0. Deep ROI Analysis — What Actually Moves the Thesis Forward?** (L10)
  - Thesis Claims That Need Support (from tesis-v2.1.tex)
  - Honest Assessment: What Would NEW Simulations Accomplish?
- **1. BLOCKING: Gate-Folding ZNE Validation** (L41)
  - Why This Is The Only Critical Simulation
  - What To Validate
  - Success Criteria
  - Effort: 2-3 hours (implement gate folding + run 9 points)
  - Impact: **CRITICAL** — determines whether hardware deployment proceeds
- **2. HIGH-VALUE Simulations (Strengthen Existing Claims)** (L80)
  - 2.1 Unsupervised Phase Detection from θ Parameters (arXiv:2506.06678)
  - 2.2 Noise-Robust Phase Signatures (arXiv:2402.18953)
- **3. REVISED Priority-Ordered Action Plan** (L120)
  - Tier 1: BLOCKING (must do before hardware)
  - Tier 2: HIGH VALUE at ZERO compute cost (analysis of existing data)
  - Tier 3: MEDIUM VALUE (marginal thesis improvement)
  - Tier 4: LOW VALUE (only if hardware deployment is delayed)
- **4. What NOT To Simulate (Over-Engineering Traps)** (L153)
- **5. The Real Critical Path** (L166)
- **6. Recommended Execution Order (Total: ~5 hours)** (L188)
- **7. Hamiltonians for Hardware (Final Answer)** (L200)
- **8. What NOT To Simulate (Documented Dead Ends)** (L212)
- **9. Literature-Backed Novel Analysis Opportunities (Zero Compute Cost)** (L230)
  - 9.1 Unsupervised Phase Detection from θ_opt(h) — arXiv:2506.06678
  - 9.2 Noise-Robust Phase Signatures — arXiv:2402.18953
  - 9.3 Phase Diagram Sketching from Low-Depth VQE — arXiv:2301.09369
- **10. Post-Hamiltonian-Extension Next Steps (2026-06-03)** (L262)
  - 10.1 2D MPNN Predictor (train across J₂ grid) — HIGH VALUE
  - 10.2 Noisy Simulation for TFIM+Longitudinal (ZNE transfer) — MEDIUM VALUE
  - 10.3 Weight-Space Phase Detection for Frustrated TFIM (D1 analog) — MEDIUM VALUE
  - 10.4 Finite-Size Scaling for Extended Models — LOW-MEDIUM VALUE
  - 10.5 Cross-Model MPNN Transfer — EXPLORATORY
  - Priority Order

---

## 16. [14_p1_zne_validation.md](14_p1_zne_validation.md)
**Estudio 1A — p=1 ZNE Validation at N=10**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| — | 59 | 489 | 2.7 KB | 14 | 0 |

**Tags**: `zne`, `hardware`, `tesis`, `topology`

> Config: N=10, p=1, h=['4.0', '3.5', '3.0'], layouts=3, shots=16384

**Métricas clave**: gain=-14.4% · gain=+49%

**Hallazgos / Aseveraciones** (1):

- Reducing the HVA ansatz from p=2 to p=1 at N=10 recovers ZNE effectiveness across all tested topologies (chain_1d, ladder, triangular). Mean gain improves from -14.4% (p=2, 33 runs…

**Contenido** (5 secciones):

- **Resultados** (L6)
- **Agregado por Topología** (L20)
- **Veredicto Global** (L28)
- **Análisis** (L41)
- **Implicación para la Tesis** (L50)

---

## 17. [14_skqd_phase_detection_alternatives.md](14_skqd_phase_detection_alternatives.md)
**SKQD para Detección de Transiciones de Fase Cuánticas — Análisis de Viabilidad**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-05 | 1649 | 9,883 | 75.9 KB | 114 | 16 |

**Tags**: `zne`, `hardware`, `mpnn`, `tesis`, `heisenberg`, `tfim`

> Fecha: 2026-06-05

**Hallazgos / Aseveraciones** (1):

- SKQD para TFIM solo funciona bien para h ≤ h_c ≈ 1.0

**Contenido** (11 secciones):

- **Alternativa 1: Fidelity Susceptibility vía Overlap de Estados SQD** (L9)
  - ¿Qué es?
  - ¿Ya se hizo antes?
  - ¿Es factible?
  - Veredicto
- **Alternativa 2: Gap Espectral ΔE(h) = E₁ - E₀ vía SQD** (L61)
  - ¿Qué es?
  - ¿Ya se hizo antes?
  - ¿Es factible?
  - Veredicto
- **Alternativa 3: Ocupación Orbital / Correladores como Order Parameter** (L103)
  - ¿Qué es?
  - ¿Ya se hizo antes?
  - ¿Es factible?
  - Veredicto
- **Alternativa 4: Krylov Complexity como Order Parameter** (L157)
  - ¿Qué es?
  - ¿Ya se hizo antes?
  - ¿Es factible?
  - Veredicto
- **Alternativa 5: SKQD + GNN con Features Físicas (Framework Alternativo)** (L214)
  - ¿Qué es?
  - ¿Ya se hizo antes?
  - ¿Es factible?
  - Implementación Detallada
- **Apéndice: Análisis Detallado de Factibilidad — Phase 3 del SKQD-GNN** (L412)
  - El Problema Central: ¿Qué Aprende la GNN?
  - ¿Qué Cambiaría con SKQD?
  - Análisis Crítico: ¿Dónde Está el Valor Científico?
  - Propuesta Refinada: SKQD-GNN para Generalización Inter-Topología
  - Factibilidad Técnica del Paso Phase 3
  - Obstáculos Técnicos Identificados
  - … (+2 más)
- **Apéndice B: Investigación de Papers Relevantes al Phase 3** (L759)
  - Papers que atacan problemas iguales o similares
  - Paper 1: "Adiabatic Fine-Tuning of Neural Quantum States Enables Detection of Phase Transitions in Weight Space"
  - Paper 2: "Learning quantum phase transition in parametrized quantum circuits with an attention mechanism"
  - Paper 3: "Deep Learning of Phase Transitions for Quantum Spin Chains from Correlation Aspects"
  - Paper 4: "A Unsupervised Framework for Identifying Diverse Quantum Phase Transitions Using Classical Shadow Tomography"
  - Paper 5: "Fluctuation based interpretable analysis scheme for quantum many-body snapshots"
  - … (+6 más)
- **Resumen de Veredictos** (L979)
- **Conclusión** (L991)
- **Referencias Clave** (L1004)
- **Apéndice C: Pipelines Detallados — Approaches Noveles** (L1023)
  - Pipeline A: SKQD → Bitstring Correlation Graph → GNN
  - Pipeline B: SKQD Inter-Topología Transfer
  - Integración con el Proyecto Actual
  - Veredicto Final de Robustez
  - Referencias Adicionales (Appendix C)

---

## 18. [15_advanced_mitigation_techniques.md](15_advanced_mitigation_techniques.md)
**Advanced Error Mitigation Techniques — Implementation Plan**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-05 | 394 | 1,818 | 13.7 KB | 17 | 9 |

**Tags**: `zne`, `hardware`, `mpnn`, `tfim`, `pea`, `topology`

> Date: 2026-06-05

**Contenido** (7 secciones):

- **Overview** (L9)
- **Technique 5: Dual-Branch Affine Correction** (L27)
  - Paper
  - Concept
  - Implementation
  - When to apply
- **Technique 2: Block-Level ZNE (Layer-Wise Folding)** (L74)
  - Paper
  - Concept
  - Implementation
  - When to use
  - Integration with existing pipeline
- **Technique 4: TLS-Aware Scheduling** (L135)
  - Paper
  - Concept
  - Implementation
  - Thresholds (from literature)
  - When to use
- **Technique 1: GNN for Error Mitigation (IMPLEMENTED)** (L202)
  - Paper
  - Concept
  - Architecture (proposed)
  - Training data (already available)
  - Implementation plan
  - Implementation (completed 2026-06-05)
  - … (+1 más)
- **Integration Summary** (L312)
  - Post-deployment energy correction pipeline
  - For p≥2 circuits specifically
  - Hardware deployment wrapper
- **References** (L367)

---

## 19. [15_heisenberg_future_work.md](15_heisenberg_future_work.md)
**Heisenberg XXZ — Future Work & Alternative Ansätze**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-01 | 204 | 1,242 | 8.8 KB | 12 | 0 |

**Tags**: `zne`, `hardware`, `thesis`, `heisenberg`, `scaling`, `noise`

> Fecha: 2026-06-01

**Hallazgos / Aseveraciones** (2):

- The systematic evaluation of the Heisenberg XXZ model across 36 pipeline configurations and three system sizes (N=6, 10, 16) establishes a definitive negative result: HVA p≤2 with …
- No existe un p donde Heisenberg tenga AMBOS alta fidelidad Y

**Contenido** (5 secciones):

- **1. El Dilema Profundidad/Ruido** (L10)
- **2. Alternativas Identificadas en la Literatura** (L26)
  - 2.1 Ansatz con Preservación de Simetría (Recomendado)
  - 2.2 Ansatz Inspirado en MPS/DMRG
  - 2.3 TETRIS-ADAPT-VQE (Circuitos Adaptativos Compactos)
  - 2.4 Slice-Wise Initial State Optimization
  - 2.5 Generative Quantum Eigensolver (GQE)
- **3. Tabla Comparativa** (L157)
- **4. Recomendación para Trabajo Futuro** (L169)
- **5. Thesis Statement (para Chapter 6 — Conclusions & Future Work)** (L187)

---

## 20. [15_transpiler_exploration.md](15_transpiler_exploration.md)
**Transpiler Exploration — Findings (2026-06-05)**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| — | 124 | 669 | 4.8 KB | 15 | 2 |

**Tags**: `zne`, `hardware`, `thesis`, `pea`, `noise`

> Evaluate whether advanced transpilation options can reduce circuit depth/noise

**Métricas clave**: R²=0.998

**Contenido** (7 secciones):

- **Objective** (L3)
- **Methods Tested** (L8)
- **Key Results** (L16)
  - Transpiler Levels (same layout, Orig HVA)
  - PauliEvolutionGate Representation (same CES-optimized layout)
  - Rustiq Plugin
- **Implementation** (L52)
- **Noise Suppression Stack (unchanged)** (L58)
- **References** (L68)
- **Addendum: Validación Formal y Corrección de Bug (2026-06-15)** (L77)
  - Bug en `create_pauli_evolution()` — Corregido
  - Corrección de la Métrica "2Q-depth"
  - Estado de Producción (2026-06-15)

---

## 21. [16_noise_suppression_analysis.md](16_noise_suppression_analysis.md)
**Noise Suppression Experiments — Analysis & Next Steps**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-05 | 442 | 3,437 | 21.0 KB | 130 | 1 |

**Tags**: `zne`, `hardware`, `mpnn`, `thesis`, `pea`, `topology`

> Date: 2026-06-05

**Métricas clave**: R²=0.04 · R²=0.996 · ΔE/gap=89.8% · R²=0.9996 · R²=0.36 · ΔE/gap < 1%

**Hallazgos / Aseveraciones** (8):

- ✅ PASS (both sections)
- PEA dominance confirmed on triangular. Coverage gap G6 closed
- ✅ PASS (both zero-shot and fine-tuned)
- *0% overshoot rate** — ZNE NEVER extrapolates below ground state
- `affine_correct_energy()` is a zero-cost safety net that has
- *PEA-ZNE is universally superior** to GF-ZNE and CES-ZNE (p<10⁻¹⁹)
- *CES-ZNE is broken** on heavy_hex (uniform CES → no extrapolation leverage)
- *GF-ZNE is a reliable fallback** (always positive gain, works on all topologies)

**Contenido** (14 secciones):

- **1. Experiment Inventory & Outcomes** (L9)
  - Formal ZNE Experiments (6/6 confirmed ✅)
  - Supporting Experiments
  - Committed Thesis Data
- **2. Definitive Rankings (from 60+ h-point evaluations)** (L41)
- **3. What Is MISSING (Gap Analysis)** (L54)
  - 3.1 Critical Gaps (Block Hardware Deployment)
  - 3.2 High-Priority Gaps (Weaken Thesis Claims)
  - 3.3 Medium-Priority Gaps (Polish / Thesis Completeness)
  - 3.4 Coverage Matrix (what exists vs what's needed)
- **4. Technique Readiness Assessment** (L98)
- **5. GNN-QEM Detailed Status** (L113)
  - What's Done
  - Ablation Findings (Definitive)
  - Final Assessment
- **6. Next Steps (Prioritized)** (L146)
  - Phase A — Hardware Deployment Preparation (blocks everything else)
  - Phase B — Fill Coverage Gaps (thesis completeness)
  - Phase C — Hardware Execution (the actual goal)
  - Phase D — Thesis Documentation
- **7. Decision Points** (L185)
  - D1: Is GNN-QEM worth pursuing for the thesis?
  - D2: Should block-ZNE be formally validated?
  - D3: What if PEA amplifier is unavailable on IBM Heron?
- **8. Summary of Findings** (L219)
  - What We Know (established, no further simulation needed)
  - What We Don't Know (requires hardware or further work)
- **References** (L240)
- **9. Execution Results (2026-06-05)** (L256)
  - G6 Closed: PEA-ZNE on Triangular N=6 p=1
  - G4/G5 Closed: GNN-QEM Cross-Topology Generalization
  - G8 Closed: Affine Overshoot Audit
- **10. Updated Gap Status** (L313)
- **11. Post-ZNE GNN-QEM Validation (2026-06-05) — CRITICAL FINDING** (L335)
  - Experiment: Full Pipeline VQE → PEA-ZNE → GNN-QEM → Affine
  - Verdict: GNN-QEM is HARMFUL in the post-ZNE pipeline
  - What This Means
  - Updated Deployment Recommendation
  - Result File
- **12. Project Health Verification (2026-06-05)** (L379)
  - Tools Used
  - ZNE Consolidated Analysis (81 evaluations, 4 topologies)
  - Per-Topology Breakdown
  - Coverage Matrix
  - New Knowledge from Project Health Analysis
- **13. Final Next Steps** (L430)

---

## 22. [17_scaling_N30_research_plan.md](17_scaling_N30_research_plan.md)
**Plan de Investigación: Escalar el Pipeline GNN-HVA a N>30**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-06 | 696 | 5,450 | 35.2 KB | 109 | 8 |

**Tags**: `zne`, `hardware`, `mpnn`, `tesis`, `heisenberg`, `tfim`

> Fecha: 2026-06-06

**Métricas clave**: ΔE/gap=1.75% · ΔE/gap < 5% · ΔE/gap < 10% · ΔE/gap=3.33% · ΔE/gap < 15% · ΔE/gap<10%

**Hallazgos / Aseveraciones** (8):

- `n_cx_real < 150`: ✅ Proceed (PEA viable)
- `n_cx_real > 250`: ❌ Abort → reduce N o usar PauliEvolutionGate
- *`BackendEstimatorV2` es SHOT-BASED**: NO da expectation values exactos. Usa sampling con `precision` que determina ~shots. Con `precision=0.005` (~40k shots) el ruido estadístico …
- *COBYLA > L-BFGS-B para este régimen**: L-BFGS-B usa diferencias finitas que se confunden con shot noise. COBYLA (gradient-free) tolera el ruido y converge en 34 evaluaciones
- *VQE a N=40 CONVERGE al 3.33% ΔE/gap** (< 5% threshold) en solo 3.3 minutos con precision=0.005. Con restarts y mejor init (warm-start from h-sweep), el resultado mejorará
- *Timing estimado pipeline completo N=40**:
- Phase 1 (DMRG, 9 h-points): ~20s
- Phase 2 (VQE, 9 h-points × 5 restarts): ~9 × 5 × 3.3 min ≈ 2.5 horas

**Contenido** (18 secciones):

- **Estado Actual y Evidencia Existente** (L10)
  - Lo que ya sabemos (validado internamente)
  - Bottlenecks identificados
- **Revisión Bibliográfica** (L40)
  - A. Bibliografía interna ya indexada
  - B. Bibliografía externa nueva (no indexada previamente)
  - C. Lo que YA HICIERON otros (no repetir)
  - D. Nuestra contribución única (gap en la literatura)
- **Plan de Implementación (3 fases)** (L88)
  - Fase 1: Infrastructure (2-3 días)
  - Fase 2: Validation Experiments (3-5 días)
  - Fase 3: Analysis & Thesis Integration (2-3 días)
- **Decisiones de Diseño Explícitas** (L231)
  - ¿Por qué MPS y no Differentiable MPS (Liao 2023)?
  - ¿Por qué NO usar TITAN (parameter freezing)?
  - ¿Por qué NO SC-ADAPT-VQE?
  - ¿Simulación noisy a N>30?
- **Archivos a Crear/Modificar** (L263)
  - Crear
  - Modificar
  - NO tocar
- **Estimación de Tiempos** (L284)
- **Resultados de Viabilidad (2026-06-06)** (L300)
  - Ejecución experimental — PLAN VERIFICADO ✅
  - Hallazgos críticos
  - Decisión de diseño actualizada
- **Riesgos y Mitigaciones** (L340)
- **Success Criteria (para declarar "N>30 funciona")** (L352)
- **Valor para la Tesis** (L362)
- **Análisis de Escalabilidad a N=80 (por fase)** (L371)
  - Tabla resumen: Escalabilidad por fase
  - Análisis detallado por fase
- **Pipeline Revisado: Pasos Intermedios para Simulación → Hardware** (L476)
  - Fase 2b: Noisy-MPS Rehearsal (NUEVO)
  - Fase 2c: Transpilation Audit (NUEVO)
  - Fase 2d: Qubit Selection a N=40+ (NUEVO)
  - Pipeline final completo
- **Límites Teóricos de Escalabilidad (N→∞)** (L563)
  - ¿Hasta dónde puede escalar CADA componente?
  - Implicación para la tesis
- **Experimento adicional: SCALE-5 (N=80, simulación only)** (L594)
  - Hipótesis
  - Config
  - Predicciones
  - Success criteria
  - Valor incremental sobre N=50
- **Decisión: ¿Bond-resolved a N=80?** (L626)
- **Cambios al Plan Original: Resumen de Adiciones** (L638)
  - Cambios NO necesarios (confirmados)
- **Nota sobre shot noise en MPSBackend** (L659)
- **Timeline Revisado** (L679)

---

## 23. [18_ibm_hardware_generations.md](18_ibm_hardware_generations.md)
**IBM Quantum Hardware Generations — Comparative Analysis**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-07 | 261 | 1,929 | 12.7 KB | 52 | 2 |

**Tags**: `zne`, `hardware`, `tesis`, `pea`, `scaling`, `noise`

> Fecha: 2026-06-07

**Métricas clave**: ΔE/gap < 5% · ΔE/gap<5%

**Hallazgos / Aseveraciones** (7):

- *En simulación**: Pipeline funciona a N=80+ (MPS exacto, O(N·χ³), sin barrera)
- *En ibm_torino**: N=40-50 demostrable con PEA-ZNE + fractional gates
- *En Nighthawk**: N=80-100 viable (3× mejor error rates, square lattice, T₁=350μs)
- *Proyección 2027+**: N=200+ (10,000 gates budget)
- *Hoy** (tesis): Pipeline end-to-end validado en simulación hasta N=80, hardware hasta N=50
- *Próximo paso**: Hardware N=80 en Nighthawk (accesible, sin cambio de pipeline)
- *Largo plazo**: El pipeline NO tiene barreras fundamentales — escala con el hardware

**Contenido** (9 secciones):

- **Corrección Importante** (L11)
- **Tabla Comparativa de Generaciones** (L20)
- **Métricas Clave Verificadas** (L41)
  - EPLG (Error Per Layered Gate)
  - Layer Fidelity (80/100 qubits)
  - Impacto de Tunable Couplers
  - Nighthawk — Square Lattice
- **Impacto para Nuestro Pipeline (N=40, 50, 80)** (L104)
  - Modelo de Fidelidad por CX Count
  - Estimaciones de Fidelidad por Generación
  - ΔE/gap Estimado Post-PEA
  - Importante: Estas son Estimaciones Optimistas
- **IBM Roadmap (Proyecciones Futuras)** (L171)
- **Fractional Gates (Heron ISA)** (L187)
- **Resumen para la Tesis** (L205)
  - Hardware Deployment Targets
  - Argumento de Escalabilidad
- **Referencias** (L229)
- **Notas de Verificación** (L250)

---

## 24. [19_cross_n_validation_plan.md](19_cross_n_validation_plan.md)
**Cross-N Zero-Shot Generalization — Validation Plan & Next Steps**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-08 | 172 | 974 | 7.1 KB | 48 | 2 |

**Tags**: `mpnn`, `thesis`, `tfim`, `scaling`, `topology`

> Date: 2026-06-08

**Métricas clave**: ΔE/gap = 0.13% · ΔE/gap < 5%

**Contenido** (6 secciones):

- **Key Finding** (L6)
- **1. Validation Protocol (Ensuring Claim Reliability)** (L16)
  - 1.1 Statistical Robustness (multi-seed)
  - 1.2 Ablation Matrix (confirming causality)
  - 1.3 Leave-One-Out Cross-Validation
  - 1.4 Physics Consistency Checks
- **2. Scaling Strategy (From Experiment → Thesis Contribution)** (L70)
  - 2.1 Immediate (no new compute needed)
  - 2.2 Short-term (1-2 compute sessions)
  - 2.3 Bond-resolved: The Killer Experiment
  - 2.4 Thesis Claim Hierarchy
- **3. Risk Assessment** (L116)
- **4. Commands to Execute (Priority Order)** (L127)
- **References** (L164)

---

## 25. [19_e3_bond_resolved_n40_plan.md](19_e3_bond_resolved_n40_plan.md)
**E3: Bond-Resolved HVA at N=40 — Task Plan**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-08 | 422 | 2,609 | 18.1 KB | 39 | 5 |

**Tags**: `hardware`, `mpnn`, `thesis`, `tfim`, `scaling`, `topology`

> > Experiment ID: E3_BR_SCALING

**Métricas clave**: ΔE/gap < 5% · ΔE/gap < 10% · ΔE/gap > 15%

**Hallazgos / Aseveraciones** (1):

- *Experiment ID: E3_BR_SCALING Date: 2026-06-08 Status: Planning Hypothesis**: Bond-resolved HVA at N=40 demonstrates GNN necessity in high-dimensional variational spaces, establish…

**Contenido** (11 secciones):

- **Objective** (L11)
- **Architecture Decisions** (L21)
  - Topology Choice: chain_1d N=40 (Phase 1) + heavy_hex N=20 (Phase 2)
  - Optimizer Strategy: SPSA primary, COBYLA fallback
  - MPNN Config: h=256, asymmetric heads (n_edges, n_qubits)
- **Dependencies on Existing Code** (L47)
- **Tasks** (L65)
  - Task 0: Fix MPNN per_parameter_heads for asymmetric splits — ✅ DONE
  - Task 1: Runner Script — E3 Bond-Resolved Scaling — ✅ DONE
  - Task 2: Section 0 — Sanity Check
  - Task 3: Section 1 — VQE Convergence Test
  - Task 4: Section 2 — Descending Sweep
  - Task 5: Section 3 — MPNN Training
  - … (+3 más)
- **Execution Order** (L242)
- **Execution Decision Tree** (L262)
- **Plan E: Heavy-Hex N=20 Bond-Resolved (highest thesis value)** (L307)
- **Value Assessment Summary** (L366)
- **Risk Register** (L382)
- **Expected Thesis Contribution** (L396)
- **References** (L415)

---

## 26. [20_scaling_extensions_plan.md](20_scaling_extensions_plan.md)
**Scaling Extensions Plan — Beyond N=80**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-08 | 271 | 1,690 | 10.4 KB | 34 | 5 |

**Tags**: `zne`, `hardware`, `mpnn`, `thesis`, `tfim`, `pea`

> > Research plan for scaling the GNN-HVA framework beyond current limits.

**Hallazgos / Aseveraciones** (1):

- Research plan for scaling the GNN-HVA framework beyond current limits. Covers: MPS N=120, NLCE thermodynamic limit, and Hamiltonian engineering. Date: 2026-06-08 Status: Planning P…

**Contenido** (5 secciones):

- **1. MPS-VQE at N=120: Bond Dimension Validation** (L12)
  - What to prove
  - Method: Bond dimension scaling test (NOT full VQE sweep)
  - Cost estimate
  - Full VQE sweep at N=120 (optional, high cost)
  - Thesis claim
- **2. NLCE + GNN-HVA: Thermodynamic Limit** (L81)
  - Overview
  - Architecture
  - Implementation plan
  - What it proves (thesis value)
  - Combined with hardware
  - Execution cost
- **3. Hamiltonian Engineering: Local/Global Split Comparison** (L151)
  - Concept
  - Implementation as comparative experiment
  - What this shows
  - Cost: ~30 lines + 1 experiment run (~15 min with statevector at N=20)
- **4. Hamiltonian Engineering Comparative: Implementation Plan** (L199)
  - Objective
  - Implementation (30 lines, 1 script)
  - Expected results table (for thesis)
  - Thesis narrative
  - Execution plan
- **Summary: Priority-Ordered Execution** (L261)

---

## 27. [21_future_work_techniques.md](21_future_work_techniques.md)
**Future Work: Advanced Techniques for GNN-HVA Scaling**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-08 | 577 | 3,936 | 27.4 KB | 51 | 3 |

**Tags**: `zne`, `hardware`, `tesis`, `heisenberg`, `tfim`, `pea`

> > Techniques identified as promising extensions beyond the scope of this thesis.

**Métricas clave**: R²>0.9 · ΔE/gap < 5% · R²<0.90

**Hallazgos / Aseveraciones** (4):

- Techniques identified as promising extensions beyond the scope of this thesis. Included for Chapter 7 (Future Work) discussion with literature references. Date: 2026-06-08 Last sta…
- *Origen: Análisis de sesión 2026-06-15 — extensibilidad de modelos y límites del HVA. Contexto completo**: `documentation/binnacles/binnacle-hamiltonian-candidates.md` (Addendum 2)
- *Fecha: 2026-06-19 Contexto**: Evaluación completa del ecosistema Qiskit addons contra nuestro pipeline GNN-HVA. Técnicas que NO aplican hoy pero podrían ser útiles en escenarios d…
- El bias residual de esta medición es ≤ ε con probabilidad 1 − δ

**Contenido** (10 secciones):

- **Implementation Status Tracker** (L11)
  - Summary
- **1. Neighbor-Informed Learning (NIL) for Error Mitigation** (L33)
- **2. ML-QEM: Machine Learning for Practical QEM** (L54)
- **3. Telemetry-Driven Adaptive Error Mitigation (GSC-QEMit)** (L74)
- **4. SC-ADAPT-VQE: Scalable Circuits for Translationally Invariant Systems** (L98)
- **5. Utility-Scale Hamiltonian Engineering (103 qubits Kagome)** (L117)
- **6. Parameter Freezing (TITAN)** (L137)
- **7. Probing the p≤2 Expressibility Boundary for XX+YY Models** (L157)
  - El límite conocido
  - Qué estudiar y cómo (si tiene sentido ejecutarlo)
  - Estado actual de la evidencia
  - Conexión con literatura
  - Refs
- **8. GNN for Quantum Chip Parameter Design** (L272)
- **9. Qiskit Addons — Error Mitigation Techniques for Future Scaling** (L291)
  - Resumen de decisiones
  - 9.1 Propagated Noise Absorption (PNA) — `qiskit-addon-pna`
  - 9.2 Shaded Lightcones + PEC — `qiskit-addon-slc`
  - 9.3 Operator Backpropagation (OBP) — `qiskit-addon-obp`
  - 9.4 Circuit Cutting — `qiskit-addon-cutting`
  - 9.5 Técnicas descartadas definitivamente
  - … (+2 más)

---

## 28. [21_thesis_compilation_verification_plan.md](21_thesis_compilation_verification_plan.md)
**Plan de Verificación — Thesis Compilation & Finding Corroboration**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-09 | 369 | 2,477 | 17.3 KB | 34 | 6 |

**Tags**: `zne`, `hardware`, `mpnn`, `tesis`, `heisenberg`, `tfim`

> Fecha: 2026-06-09

**Métricas clave**: ΔE/gap < 5% · ΔE/gap=30.9% · gain: +17.7% · R²=0.999 · ΔE/gap=1.066 · ΔE/gap=0.038

**Hallazgos / Aseveraciones** (1):

- *CRITICAL STATEMENT for thesis: All performance optimizations applied in this session (noise pair filtering, MPS deterministic mode, parallel execution) are bit-exact** — they prod…

**Contenido** (10 secciones):

- **A. Findings a Corroborar** (L9)
  - Findings ya cubiertos por `thesis_findings_validator.py` (23 — todos implementados):
  - Estado actual (2026-06-10): 22/23 CORROBORATED, 1 QUALIFIED (F16)
- **B. Tablas Globales para la Tesis** (L43)
  - Tablas existentes (T1-T10):
  - Tablas NUEVAS a considerar:
- **C. Figuras Globales para la Tesis** (L64)
  - Figuras existentes (10):
  - Figuras NUEVAS a considerar:
- **D. Plan de Ejecución** (L84)
  - Paso 1: Ejecutar validador actual
  - Paso 1b: Deep raw-data audit
  - Paso 2: Ejecutar tablas
  - Paso 3: Ejecutar figuras
  - Paso 4: Verificar claims de la tesis contra datos
  - Paso 5: Full thesis compilation
- **E. Criterios de Aceptación del Plan** (L120)
- **F. Estado Actual (post-sesión 2026-06-09)** (L136)
- **G. Findings y Confirmaciones Sesión 2026-06-10** (L151)
  - Infraestructura: MPS Deterministic Mode (375× speedup)
  - Scaling Law — Nuevos Validated System Sizes
  - B4 Bond-Resolved Cross-N — GNN NECESSITY PROOF ✅
  - PEA-ZNE Coverage Complete (all 4 topologies)
  - GNN-QEM Circuit Selection — Bootstrap CI
  - Deep Audit Score: 23/25 VERIFIED + 2 PARTIAL
  - … (+1 más)
- **H. Pendiente Agregar en Tesis (★ = nueva contribución de esta sesión)** (L227)
- **I. Findings Sesión PEA Optimization (2026-06-10, continuación)** (L274)
  - PEA-ZNE Optimización de Rendimiento (Implemented & Validated)
  - Hardware Rehearsal V2: 9/9 PASS ✅ (post-fix)
  - ★ PEA Cross-Topology Dense: 90/90 wins, p=1.5×10⁻¹¹³
  - ★ Cross-Topology Transfer + Noisy PEA: 4/4 PASS, +96.2% noise reduction
  - PEA Scaling N=40/50 — Technical Limitation Documented
- **J. ★ Pendiente Agregar en Tesis (consolidado 2026-06-10)** (L335)
  - Notas sobre cache improvements y rigor científico:

---

## 29. [22_global_vision_audit.md](22_global_vision_audit.md)
**Auditoría de Visión Global — Project Health Pipeline (2026-06-09)**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| — | 366 | 2,931 | 21.6 KB | 81 | 1 |

**Tags**: `zne`, `hardware`, `mpnn`, `tesis`, `tfim`, `pea`

> Propósito: Verificación end-to-end de todas las herramientas de `project_health/`, validación de orden de ejecución, y consolidación de findings no documentados

**Métricas clave**: ΔE/gap=501% · ΔE/gap=719% · R²=0.999

**Hallazgos / Aseveraciones** (3):

- *Nota importante sobre evaluación MPS**: Todos los resultados generados a partir del 2026-06-10 usan `save_expectation_value` (evaluación exacta, sin shot noise). Esto NO invalida …
- *Estado actual**: ✅ CORROBORATED (STRONG) — "no statistically significant performance difference between topologies
- *Estado actual**: ✅ CORROBORATED (STRONG) — ρ=0.9452, binary_accuracy=100%

**Contenido** (10 secciones):

- **1. Herramientas Ejecutadas (en orden)** (L7)
- **2. Bug Corregido** (L25)
- **3. Discrepancias Detectadas (datos vs documentación)** (L34)
  - 3.1 Tasa de resultado útil: 84% ≠ 93%
  - 3.2 Pipeline runs: 430+ ≠ 210+
  - 3.3 F6 Topology Ranking → Reformulado a "Topology-Agnostic" (RESUELTO)
  - 3.4 T2 (ZNE Strategy) muestra "UNKNOWN"
  - 3.5 T4 (GNN-QEM) muestra zeros
  - 3.6 F14 (GNN Circuit Selection) — ARREGLADO
- **4. Orden de Ejecución Validado** (L80)
- **5. Nuevos Findings Documentados (no existían previamente)** (L116)
  - 5.1 Cross-Topology Transfer (spec reciente) FALLA en tri↔hex
  - 5.2 E5 Scaling Extensions: NLCE converge solo en fase gapped
  - 5.3 Distribución de resultados: p=1 under-represented
- **6. Recomendaciones** (L143)
- **7. Verificación de Integridad** (L153)
- **8. Análisis de Robustez — ¿Qué experimentos adicionales darían más confianza?** (L169)
  - Estado: Auditoría deep (23/23 VERIFIED, 2026-06-09)
  - Findings con robustez máxima (no requieren nada):
  - Findings con robustez buena (suficiente para tesis):
  - Findings donde un experimento adicional PODRÍA fortalecer la tesis:
  - Veredicto final: ¿Se necesita re-ejecutar algo?
  - Recomendación
- **9. Actualización Post-Sesión 2026-06-10** (L232)
  - Nuevos Hallazgos Confirmados
  - Scores Actualizados
  - ★ Pendiente Agregar en Tesis (actualizado)
  - Findings Definitivos NO Ejecutar (evaluados como bajo valor):
  - Scores Actualizados
  - ★ Pendiente Agregar en Tesis
- **10. Actualización Sesión PEA Optimization (2026-06-10, afternoon)** (L308)
  - Nuevos Resultados Confirmados
  - Datos Clave para la Tesis
  - Correcciones Aplicadas al Código
  - Verificación: Optimizaciones NO Afectan Resultados
  - Limitación Documentada: PEA a N≥20 requiere MPS

---

## 30. [23_noisy_simulation_scalability_limits.md](23_noisy_simulation_scalability_limits.md)
**Noisy Simulation Scalability Limits (2026-06-10)**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-10 | 112 | 719 | 4.7 KB | 12 | 1 |

**Tags**: `zne`, `hardware`, `pea`, `scaling`, `topology`, `noise`

> > Definitive findings on what CAN and CANNOT be validated locally for

**Métricas clave**: gain = 0% · R²=0.997

**Hallazgos / Aseveraciones** (4):

- Definitive findings on what CAN and CANNOT be validated locally for noise suppression at N≥20. Prevents future repetition of invalid experiments
- PEA scalability to N≥20 can ONLY be validated on real QPU
- *Memory limit (N≥20)**: FakeTorino (133 qubits) causes OOM when used with
- *Physics limit (N≥20 without routing)**: Native chain_1d circuits have only

**Contenido** (5 secciones):

- **Executive Summary** (L8)
- **Experiment: PEA_SCALING (2026-06-10)** (L23)
  - Configuration
  - Results
  - Root Cause Analysis
  - Implication for Hardware
- **Memory Constraints (Definitive)** (L73)
  - HardwareBackend Rehearsal Constraint
- **DO NOT REPEAT** (L95)
- **References** (L107)

---

## 31. [24_aqc_tensor_integration_plan.md](24_aqc_tensor_integration_plan.md)
**AQC-Tensor Integration Plan**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-17 | 278 | 1,461 | 10.8 KB | 28 | 4 |

**Tags**: `zne`, `hardware`, `mpnn`, `tesis`, `tfim`, `pea`

> Date: 2026-06-17

**Métricas clave**: ΔE/gap=0.24% · fidelity < 0.9995 · Fidelity ≥ 0.999 · Fidelity < 0.99 · ΔE/gap < 1% · ΔE/gap > 3%

**Hallazgos / Aseveraciones** (2):

- `p=1 N=10`: 34 CZ → ~18 CX post-transpilación → ZNE funciona ✅
- `p=2 N=10`: 68 CZ → ~36 CX post-transpilación → ZNE falla ❌

**Contenido** (8 secciones):

- **Motivación** (L10)
- **Arquitectura del módulo** (L26)
  - Ubicación
  - Dependencias nuevas
  - Interfaz pública (API)
- **Fases de Implementación** (L81)
  - Fase 1: Proof of Concept (POC standalone)
  - Fase 2: Módulo de producción (`aqc_compression.py`)
  - Fase 3: Validación cross-topology y comparativa
  - Fase 4: Integración en pipeline de hardware
  - Fase 5: Extensiones escalables (futuro)
- **Relación con componentes existentes** (L215)
- **Riesgos y mitigaciones** (L232)
- **Criterios de éxito (por fase)** (L244)
- **Timeline estimado** (L256)
- **Referencias** (L270)

---

## 32. [24_mitiq_integration_plan.md](24_mitiq_integration_plan.md)
**Mitiq Integration Plan — Módulo de Mitigación Complementario**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-17 | 455 | 2,684 | 19.1 KB | 42 | 8 |

**Tags**: `zne`, `hardware`, `thesis`, `tfim`, `pea`, `noise`

> Fecha: 2026-06-17

**Métricas clave**: R²>0.90 · R² ≥ 0.95 · ΔE/gap < 1%

**Contenido** (8 secciones):

- **1. Análisis: ¿Es útil Mitiq para simulaciones noisy vs. nuestros métodos actuales?** (L9)
  - 1.1 Comparación de enfoque noisy actual vs. Mitiq
  - 1.2 Valor de Mitiq para noisy simulations
  - 1.3 Cuándo Mitiq NO reemplaza nuestros métodos
  - 1.4 Estrategia recomendada
- **2. Fases de Implementación** (L65)
  - Fase 1: Foundation — Módulo base y executor adapter
  - Fase 2: Mitiq ZNE — Random folding + factories ricas
  - Fase 3: Mitiq CDR — La contribución principal
  - Fase 4: DDD+ZNE Composition — Stack de mitigación
  - Fase 5: PEC Benchmark — Reference offline
  - Fase 6: Noisy Simulation Pipeline via Mitiq
  - … (+3 más)
- **3. Diseño Técnico Detallado** (L270)
  - 3.1 Módulo DAG (sin circular imports)
  - 3.2 Dataclasses (output pattern)
  - 3.3 pyproject.toml changes
  - 3.4 Integration con `__init__.py`
- **4. Análisis de Valor por Método para Nuestro Caso Específico** (L362)
  - 4.1 N=10 p=1 heavy_hex (hardware deployment target)
  - 4.2 Simulación noisy (pre-hardware validation)
- **5. Riesgos y Mitigaciones** (L393)
- **6. Criterios de Éxito** (L406)
- **7. Timeline Estimado** (L425)
- **8. Referencias** (L444)

---

## 33. [24_mpnn_eval_suite_results.md](24_mpnn_eval_suite_results.md)
**MPNN Evaluation Suite — Results & Analysis**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-15 | 176 | 1,328 | 9.1 KB | 27 | 0 |

**Tags**: `zne`, `hardware`, `mpnn`, `thesis`, `tfim`, `pea`

> Date: 2026-06-15

**Métricas clave**: ΔE/gap = 0.42%

**Hallazgos / Aseveraciones** (8):

- speedup_vs_random = 2.98x, speedup_vs_prev_h = 0.87x
- ΔE_circuit = 0.24%, ΔE_MPNN = 0.04%, ΔE_total = 0.28%
- interp pass = 100% (3/3), extrap pass = 33%, degradation = 20.96x
- zero_shot ΔE/gap = 695%, in_dist = 0.035%, transfer_ratio = 200x
- noiseless=0.65%, noisy_raw=113%, noisy_zne=60%, ZNE improvement=+46.8%
- Use heavy_hex N=10 p=1 native topology for hardware (18 CX, already validated in V2)
- *GNN warm-start works**: 2.98x speedup vs random at N=6 ✅
- *3 training points minimum, 8 points optimal** for reliable LOO-CV (critical_size=3, but LOO requires 7+ for 80% pass rate)

**Contenido** (6 secciones):

- **Executive Summary (Final — 2026-06-15)** (L11)
- **Section-by-Section Analysis** (L28)
  - Section 10 — MPNN Warm-Start Benchmark ✅
  - Section 11 — LOO Cross-Validation
  - Section 12 — Landscape Quality ✅
  - Section 13 — Interpolation vs Extrapolation ✅
  - Section 17 — Zero-Shot Topology Transfer ❌ (NEW FINDING)
  - Section 18 — Multi-Seed LOO Robustness ✅
  - … (+1 más)
- **Updated Conclusions for Thesis** (L112)
- **Files** (L124)
- **New Results (2026-06-15, final batch)** (L138)
  - Section 14 — Noisy Eval (FakeTorino) ❌ (IMPORTANT FINDING)
  - Section 15 — Scaling with N (4,6,10 with correct p_layers) ✅
  - Section 19 — κ Extended Through Phase Transition ✅
  - Hardware Deployment Integration (New)
- **Final Bug Fixes Applied** (L167)

---

## 34. [25_mitigation_benchmark_plan.md](25_mitigation_benchmark_plan.md)
**Plan de Benchmarking Sistemático — Técnicas de Mitigación y Supresión de Ruido**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-17 | 1131 | 6,701 | 49.3 KB | 227 | 31 |

**Tags**: `zne`, `hardware`, `mpnn`, `tesis`, `tfim`, `pea`

> Fecha: 2026-06-17

**Métricas clave**: ΔE/gap < 5% · ΔE/gap < 3% · R² < 0.5

**Contenido** (18 secciones):

- **1. Diseño Experimental** (L10)
  - 1.1 Circuito Base (constante en todos los benchmarks)
  - 1.2 Modos de Ejecución
  - 1.3 Baseline (referencia obligatoria)
- **2. Técnicas a Evaluar** (L45)
  - 2.1 Error Suppression (prevención — no agregan overhead de shots)
  - 2.2 Error Mitigation (post-procesamiento — agregan overhead)
  - 2.3 Combinaciones Prioritarias (evitar explosión combinatoria)
- **3. Métricas a Capturar** (L134)
  - 3.1 Métricas de Precisión (ya existentes en summary.json — NO duplicar)
  - 3.2 Métricas Adicionales del Benchmark (nuevas, específicas de este estudio)
  - 3.3 Métricas de Transpilación (clave para objetivo: mejores transpilaciones)
  - 3.4 Métricas AQC-Tensor (cuando `--aqc-compress` activo)
  - 3.5 Métricas de Robustez y Confiabilidad
  - 3.6 Métricas IBM-Specific (Hardware Real)
  - … (+1 más)
- **4. Estructura de Resultados** (L290)
  - 4.1 Formato de Resultado por Ejecución
- **5. Plan de Ejecución** (L368)
  - Fase A — Simulación completa (FakeTorino, 0 QPU cost)
  - Fase B — Validación en hardware (IBM Torino)
  - Fase C — Análisis y Tablas de Tesis
- **6. Implementación** (L402)
  - 6.1 Script Principal
  - 6.2 Analyzer
  - 6.3 Integración con Pipeline Existente
- **7. Prioridad de Ejecución (si QPU time es limitado)** (L436)
- **8. Criterios de Éxito** (L448)
- **9. Metodología de Análisis — Cómo Encontrar la Configuración Óptima** (L460)
  - 9.1 Principio: Análisis por Capas (de simple a complejo)
  - 9.2 Nivel 1 — Ranking Bruto
  - 9.3 Nivel 2 — Ablation Study (contribución individual)
  - 9.4 Nivel 3 — Análisis Costo-Beneficio
  - 9.5 Nivel 4 — Robustez y Transferencia Sim→Hardware
  - 9.6 Análisis de Sensibilidad por Parámetro
- **10. Tablas de Tesis (output final)** (L593)
  - Tabla 10.1 — Comparación de Técnicas de Supresión (Chapter 5)
  - Tabla 10.2 — Ablation Study (Chapter 5)
  - Tabla 10.3 — Configuración Óptima Recomendada
- **11. Flujo de Decisiones (Diagrama)** (L638)
- **12. Hipótesis a Validar** (L680)
- **13. Condiciones de Aborto** (L708)
- **14. Notas Importantes** (L720)
  - 14.1 Notas Técnicas Adicionales (correcciones 2026-06-17)
- **15. Referencias** (L754)
- **16. Integración con Mitiq — Consideraciones de Transpilación** (L775)
  - 16.1 El dilema opt_level=0 vs opt_level=2
  - 16.2 Estrategia para el benchmark de transpilación
  - 16.3 Implementación (usa compare_mitigation_strategies)
  - 16.4 AQC × Transpilación × Mitigación: Análisis 3-factorial
- **17. Guía de Implementación — Funciones Existentes a Reutilizar** (L864)
  - 17.1 Métricas de circuito (Section 3.3) — YA EXISTEN
  - 17.2 Layout selection — YA EXISTE
  - 17.3 Error mitigation — YA EXISTE (configs C0-C10)
  - 17.4 Mitiq mitigation — YA EXISTE (configs C11-C14)
  - 17.5 AQC-Tensor compression — YA EXISTE (configs C16-C18)
  - 17.6 Benchmark execution — YA EXISTE (BenchmarkSuite)
  - … (+2 más)
- **18. Principios de Reutilización y Escalabilidad** (L1056)
  - 18.1 NUNCA reimplementar lo que ya existe
  - 18.2 Extensión preferida sobre creación
  - 18.3 Formato de resultados: envelope pattern
  - 18.4 Principio de composición (no monolito)

---

## 35. [26_bond_dimension_classical_simulability_review.md](26_bond_dimension_classical_simulability_review.md)
**Bond Dimension as the Classical Simulability Frontier — Literature Review (2026-09-01)**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| — | 155 | 1,031 | 8.4 KB | 8 | 0 |

**Tags**: `hardware`, `thesis`, `heisenberg`, `tfim`, `scaling`

> > External-literature review supporting the study objective:

**Hallazgos / Aseveraciones** (2):

- External-literature review supporting the study objective: "The GNN-prepared initial state shifts the entanglement barrier, requiring a larger χ than a trivial quench from |0⟩^N." …
- A quench from a non-trivial, correlated initial state |ψ₀(h₁)⟩ (prepared by the GNN-HVA pipeline) drives entanglement growth faster than a quench from a product state |0⟩^N or |+⟩^…

**Contenido** (5 secciones):

- **Thesis objective this review supports** (L13)
- **Two regimes (established physics)** (L26)
- **Key references** (L38)
  - The entanglement barrier (core mechanism)
  - Linear entanglement growth after a quench (why χ explodes)
  - IBM kicked-Ising was simulated classically (reframes our IBM diagram)
  - χ-vs-precision frontier (concrete numbers)
  - GNN/NN state preparation (competing methods — positioning)
- **Verdict** (L101)
- **Cross-refs (repo)** (L116)

---

## 36. [Completion_Plan-techniques.md](Completion_Plan-techniques.md)
**Completion_Plan-techniques**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-19 | 259 | 1,530 | 10.6 KB | 29 | 7 |

**Tags**: `zne`, `hardware`, `tesis`, `heisenberg`, `tfim`, `pea`
**Métricas clave**: R² < 0.85 · R² > 0.98

**Hallazgos / Aseveraciones** (4):

- *Fecha: 2026-06-19 Scope: Plan para completar las 2 técnicas parcialmente implementadas (#3 y #7). Prioridad**: Post-Kingston deployment. Ninguna bloquea el hardware run actual
- TFIM (h=2): S≈0.5, fid≥0.99 ✅
- TFIM+Long (g=0.3): S≈0.5, fid≥0.98 ✅
- Heisenberg (h=3): S≈2.2, fid=0.48 ❌

**Contenido** (1 secciones):

- **10. Completion Plan — Partially Implemented Techniques** (L3)
  - 10.1 Técnica #3: Adaptive Mitigation (GSC-QEMit) — de 60% a 100%
  - 10.2 Técnica #7 (FW-C): Entanglement Entropy como Predictor de Viabilidad — de 70% a 100%
  - 10.3 Priorización

---

## 37. [accelerated_cross_n_coverage.md](accelerated_cross_n_coverage.md)
**Accelerated Cross-N Coverage Analysis**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-10-04 (auto-updated by update_cross_n_coverage.py) | 391 | 4,413 | 20.3 KB | 243 | 0 |

**Tags**: `mpnn`, `tfim`, `topology`

> Fecha: 2026-10-04 (auto-updated by update_cross_n_coverage.py)

**Métricas clave**: ΔE/gap ≥ 5%

**Hallazgos / Aseveraciones** (1):

- *Nota**: Las secciones marcadas con `<!-- AUTO-GENERATED -->` se actualizan automáticamente. Las secciones de análisis físico y recomendaciones son manuales

**Contenido** (9 secciones):

- **Resumen Ejecutivo (auto-generated)** (L14)
- **Training Data Health** (L27)
  - Quality Tier Distribution (NPZ-level)
- **Gap Masking Analysis** (L106)
- **Detalle por Topología** (L144)
  - Chain 1D
  - Heavy Hex
  - Ladder
  - Square
  - Triangular
- **h_frontier per Topology** (L249)
- **Cross-N Transfer Summary** (L265)
- **Quality Tier Distribution** (L285)
- **Training Plan (auto-generated)** (L296)
  - ⚠️ IMPROVE — Insufficient signal (need more good points)
  - ✅ EXPAND — Useful configs (add more h-points for better generalization)
- **Model Zoo Health** (L379)

---

## 38. [classical_simulability_square_frustrated.md](classical_simulability_square_frustrated.md)
**Classical simulability of 2D square frustrated TFIM dynamics**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-11-16 | 332 | 2,610 | 17.5 KB | 34 | 3 |

**Tags**: `zne`, `hardware`, `thesis`, `heisenberg`, `tfim`, `pea`

> Status: **temporarily closed, 2026-11-16.** Supersedes the 2026-10-02 version, whose

**Métricas clave**: fidelity > 0.01

**Hallazgos / Aseveraciones** (8):

- In the 2D square frustrated TFIM, the exact microstate of a quench across h_c is followed by a 1D MPS of fixed χ only up to a step t\* that shrinks with N (measured to N=18). Entan…
- *Ground state is trivial for MPS** [MEASURED]: half-chain entropy 0.7–1.3 nats at
- *Dynamics across the transition is where classical cost grows** [MEASURED]: the
- *No practical classical limit exists at N ≤ 20** [MEASURED]: the exact statevector is
- *N=30** [EXTRAPOLATED]: exact statevector 17 GB (cluster, not laptop); MPS needs
- *N≈50** [EXTRAPOLATED]: exact statevector ≈ 18 PB (impossible); MPS needs χ ≈ 10⁶,
- *Quantum side** [MEASURED + EXTRAPOLATED]: outlasting a χ=10⁴ MPS at N=18 needs
- *No hardware quantum advantage is claimed.** This document bounds the classical side

**Contenido** (11 secciones):

- **1. Summary** (L18)
- **2. Setup and definitions** (L39)
- **3. Measured results** (L59)
  - 3.1 Critical point [MEASURED, N=10]
  - 3.2 Crossing quench, N = 10–20 (h_prep 3.0 → h₂ 0.3) [MEASURED]
  - 3.3 Saturation entropy is linear in N [MEASURED]
  - 3.4 Quench from the exact ground state, leaving the critical region [MEASURED]
  - 3.5 2D cut versus 1D half-chain [MEASURED, negative result]
  - 3.6 Ground-state entropy [MEASURED, N ≤ 18]
- **4. Projections beyond measured N [EXTRAPOLATED]** (L143)
- **5. Ground state versus dynamics** (L165)
- **6. Quantum-side budget for outlasting the classical limit** (L183)
  - 6.1 Routing overhead on the real square coupling map [MEASURED]
  - 6.2 Gate budget at N = 16 and 18 [MEASURED + EXTRAPOLATED]
- **7. Hypothesis (conservative)** (L231)
- **8. Open caveats and unfinished work** (L243)
- **9. Suggested next experiments** (L265)
- **10. Reproduction (flags verified with `--help` on 2026-11-16)** (L276)
- **11. Literature** (L310)

---

## 39. [cross_topology_report.md](cross_topology_report.md)
**Cross-Topology Unified Report**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-10-04 | 118 | 1,068 | 5.4 KB | 58 | 0 |

**Tags**: `mpnn`, `tfim`, `scaling`, `topology`

> Generated: 2026-10-04 23:58 UTC

**Métricas clave**: ΔE/gap < 5%

**Hallazgos / Aseveraciones** (1):

- All quality metrics use the dual criterion exclusively. `summary.pass_rate` in runner JSONs = execution health (sections completed), NOT quality

**Contenido** (7 secciones):

- **1. Scorecard** (L13)
- **2. Scaling: pass_rate_dual per (Topology, N)** (L25)
- **3. Gap Masking Severity** (L39)
- **4. Large-N Extrapolation (Zero-Shot)** (L64)
- **5. Training Data Quality** (L72)
- **6. Failure Mode Classification** (L84)
- **7. Recommended Actions** (L98)

---

## 40. [noiseless_comprehensive_status.md](noiseless_comprehensive_status.md)
**Análisis Completo de Resultados Noiseless — Estado al 2026-07-11**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-24 | 899 | 6,862 | 40.8 KB | 269 | 11 |

**Tags**: `zne`, `hardware`, `mpnn`, `tesis`, `heisenberg`, `tfim`

> Total runs noiseless: 320 (de ~1400 totales en el índice)

**Métricas clave**: ΔE/gap<0.03 · ΔE/gap=0.003 · ΔE/gap < 5% · ΔE/gap = 0.021 · ΔE/gap=0.55 · ΔE/gap=0.45

**Hallazgos / Aseveraciones** (8):

- For heavy_hex topology at N≥16, the spectral gap is computed using the analytical fallback Δ = 2π/N, which constitutes a lower bound on the true gap. Consequently, the ΔE/gap value…
- F1-F2, F6: `documentation/analysis/noiseless_v2_analysis.md` § 10 "Consolidated Empirical Findings
- F3: `noiseless_v2_analysis.md` § 9.3 (N=16 MSE=1.59e-2 con ratio 1.25:1)
- F4: `results/analysis/noiseless_final_scaling.json` (axis h_dependence, h_boundary≈1.3 en 8+ configs)
- F5: Sección 11.5 de este doc + `analyze_noiseless_scaling.py` output (100% wins en runs julio 11)
- F7: `exp_noiseless_tfim_4/run_20260702_184821.json` (N=16), `run_20260702_200440.json` (N=20)
- F8: `exp_noiseless_tfim_4/run_20260702_200440.json` (100%, 39/39, F̄=0.998)
- tfim ch N=20 p=4: `exp_noiseless_tfim_4/run_20260702_200440.json`

**Contenido** (16 secciones):

- **1. Inventario Completo de Datos** (L10)
  - 1.1 Modelos Testados
  - 1.2 Topologías Cubiertas
  - 1.3 Tamaños de Sistema (N)
  - 1.4 Profundidades (p)
- **2. Resultados Consolidados — Lo que SABEMOS (Thesis-Grade)** (L56)
  - 2.1 Findings Definitivos (multi-run validated)
  - 2.2 Findings de Expresividad (p=5, 2026-07-09)
  - 2.3 Resultados Definitivos (Thesis Table Material)
- **3. Lo que FALTA — Investigaciones Pendientes** (L115)
  - 3.1 Prioridad ALTA (necesario para tesis)
  - 3.2 Prioridad MEDIA (fortalece la tesis)
  - 3.3 Prioridad BAJA (nice-to-have, publicación extendida)
- **4. Parámetros Explorados vs No Explorados** (L149)
  - 4.1 Matriz de Cobertura (✅=pass, ⚠️=parcial, ❌=fail, —=no testado)
  - 4.2 Huecos Críticos en la Cobertura
- **5. Análisis de Regresiones y Anomalías** (L192)
  - 5.1 Regresiones Detectadas (project-status.md)
  - 5.2 Anomalías Conocidas
- **6. Parámetros Óptimos Establecidos** (L226)
  - 6.1 Configuración de Producción (recomendada para nuevos runs)
  - 6.2 Scaling Law para h_min
  - 6.3 Jerarquía de Modelos
  - 6.4 Jerarquía de Topologías
- **7. Preguntas Científicas Abiertas** (L277)
  - 7.1 Respondidas (cerradas)
  - 7.2 Parcialmente Respondidas
  - 7.3 No Respondidas (requieren experimentos nuevos)
- **8. Roadmap de Próximos Experimentos (priorizado)** (L316)
  - Fase A: Completar Tabla de Scaling (1-2 días)
  - Fase B: Multi-Seed Validation (1 día)
  - Fase C: Cross-Model Validation (1 día)
  - Fase D: Experimentos Opcionales (si hay tiempo)
- **9. Resumen Ejecutivo para Tesis** (L373)
  - Lo que está LISTO para escribir:
  - Lo que FALTA para tesis completa:
  - Esfuerzo estimado total: ~60-80 horas de compute (~3-4 días con nohup)
- **10. Métricas Globales del Programa Noiseless** (L395)
- **11. Resultados Finales del Estudio de Scaling (2026-07-10)** (L415)
  - 11.1 Global Summary
  - 11.2 Eje 1: N-Scaling (Tabla Definitiva)
  - 11.3 Eje 2: h-Dependence (Fronteras de Validez)
  - 11.4 Eje 3: p-Dependence (Profundidad Óptima)
  - 11.5 Eje 4: Topology Comparison (Ranking Final)
  - 11.6 Tabla de Scaling Final para Tesis
  - … (+1 más)
- **12. Known Artifact: DMRG Gap Floor in heavy_hex N≥16** (L652)
  - 12.1 Description
  - 12.2 Affected Runs
  - 12.3 Impact on Results
  - 12.4 Why chain_1d Works but heavy_hex Doesn't
  - 12.5 Resolution Path
  - 12.6 Statement for Thesis
  - … (+1 más)
- **13. Robustness Fixes Validation (2026-07-11)** (L737)
  - 13.1 Problem Statement
  - 13.2 Fixes Applied
  - 13.3 Before vs After
  - 13.4 Detailed Metrics (post-fix)
  - 13.5 Residual Outlier Analysis
  - 13.6 Updated Multi-Seed Summary (14/14 configs now pass)
  - … (+1 más)
- **14. Runs No Registrados Previamente (julio 10-11, 34 runs)** (L812)
- **15. Findings Nuevos Robustos (multi-run validated)** (L836)
  - F9: p=2 es viable a N=16/20 con h≥1.6 (hardware-optimal depth)
  - F10: MPNN wins = 100% es UNIVERSAL en todo el rango h
  - F11: Gradient clipping (max_norm=1.0) elimina MPNN training divergence
  - F12: Force-bidirectional es necesario para seeds difíciles a N≥16
  - F13: h_boundary escala con topología independientemente de N
- **16. Hipótesis Pendientes de Validación** (L891)

---

## 41. [noiseless_model_comparison_tables.md](noiseless_model_comparison_tables.md)
**noiseless_model_comparison_tables**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| — | 100 | 843 | 4.1 KB | 45 | 0 |

**Tags**: `mpnn`, `heisenberg`, `tfim`, `topology`, `noise`
**Métricas clave**: ΔE/gap < 5%

**Contenido** (7 secciones):

- **TN1 — Noiseless Model Ranking** (L1)
- **TN2 — Noiseless Topology Ranking** (L13)
- **TN3 — Model × Topology Cross-Table** (L27)
- **TN4 — Best Configurations Per Model** (L39)
- **TN5 — VQE Optimization Quality by Model** (L57)
- **TN6 — MPNN Training Quality by Model** (L69)
- **TN7 — Performance by Circuit Depth (p-layers)** (L81)

---

## 42. [noiseless_v2_analysis.md](noiseless_v2_analysis.md)
**Noiseless Pipeline v2 — Comprehensive Analysis**

| Fecha | Líneas | Palabras | Tamaño | Tablas | Código |
|-------|--------|----------|--------|--------|--------|
| 2026-06-27 | 1054 | 9,218 | 51.0 KB | 291 | 1 |

**Tags**: `zne`, `hardware`, `mpnn`, `thesis`, `heisenberg`, `tfim`

> Date: 2026-06-27

**Métricas clave**: ΔE/gap = 8.2 · ΔE/gap = 1.4 · ΔE/gap = 2.4 · ΔE/gap=0.024 · ΔE/gap=0.073 · ΔE/gap=0.225

**Hallazgos / Aseveraciones** (8):

- *Sections 8 and 9.5 are now OBSOLETE.** The DMRG gap floor issue has been permanently fixed in the solver. All new runs compute correct spectral gaps
- Heisenberg HVA p≤4 es completamente inviable para N=10
- The v3 pipeline improvements closed the gap between TFIM and TFIM Longitudinal. Both models now achieve 90%+ on the best topologies (chain_1d, heavy_hex) with p=3-4
- El grid denso penaliza a heavy_hex (−11% a −20%) más que a chain_1d. En chain_1d con p=4 la mejora es real (+17% en tfim_long) porque el ansatz tiene la expresibilidad suficiente p…
- ALL 4 SECTIONS PASS. First p=4 full-pipeline success
- ALL 4 SECTIONS PASS. 24/24 deploy points = 100%
- *heavy_hex** — Mejor en todos los modelos. Conectividad moderada (N=10 chain subgraph con 9 edges) favorece VQE + MPNN. Pipeline ALL PASS con tfim p=4 y tfim_long p=3
- *chain_1d** — Segundo mejor. Mínima conectividad, landscape simple pero h-crit cercano

**Contenido** (18 secciones):

- **1. TFIM Standard (`model=tfim`)** (L10)
  - Summary Table
  - p=1 Analysis
  - p=3 Analysis (NEW — 2026-06-27)
  - Best Result: chain_1d p=3 (NEW)
  - Previous Best: heavy_hex p=4
- **2. TFIM Longitudinal (`model=tfim_longitudinal`)** (L72)
  - Summary Table
  - Best Result: heavy_hex p=3
  - Critical Finding: p=4 DEGRADES performance
- **3. Heisenberg (`model=heisenberg`)** (L109)
  - Summary Table (COMPLETE — p=1 through p=4, all topologies)
  - Analysis by p-layer
  - Key Observation: MPNN "learns" but deploys garbage
- **4. Key Findings** (L154)
  - 4.1 Topology Ranking (consistent across models)
  - 4.2 Optimal Depth per Model
  - 4.3 θ_smoothness como Predictor de Éxito
  - 4.3b MSE como Indicador de Calidad MPNN
  - 4.4 Failure Modes Identified
  - 4.5 Hardware Deployment Implications
  - … (+1 más)
- **5. Experimental Conditions** (L227)
- **6. Data Summary** (L236)
  - Global Rankings (from 79 runs)
- **7. Pipeline v3 — Impact of Bidirectional Sweep + Energy Guard + Outlier Filter** (L268)
  - 7.1 v2 vs v3 Comparison (same topology, model, p — only pipeline improvements differ)
  - 7.2 Analysis of Improvements
  - 7.3 Heisenberg Transverse — Negative Result (v3)
  - 7.4 Updated Top-5 Configurations (v3)
  - 7.5 Updated Global Rankings (v3, 35 new runs)
- **8. Pipeline v4 — Dense h-grid (70 points) Validation** (L344)
  - 8.1 TFIM Standard — 4/18 PASS
  - 8.2 TFIM Longitudinal — 1/15 PASS
  - 8.3 Heisenberg Transverse — 0/10 PASS (confirmación definitiva)
  - 8.4 Hallazgos Concretos (v4)
  - 8.5 Top-5 Configuraciones Validadas (v4, 70 h-points)
  - 8.6 Comparación v3 vs v4 (mejores configuraciones)
  - … (+1 más)
- **9. Scaling to N=16 — TFIM Longitudinal p=4 heavy_hex (2026-07-02)** (L490)
  - 9.1 Context
  - 9.2 Configuration
  - 9.3 Results
  - 9.4 Reliable Findings (Consistent with N=10 results)
  - 9.5 Known Artifacts (still valid)
  - 9.6 Comparison with N=10 Equivalent (corrected gaps)
- **10. Consolidated Empirical Findings (v2–v4, 47 runs)** (L563)
  - Finding F1: Topology dominates over circuit depth (quantified)
  - Finding F2: Fidelity saturates for chain_1d/heavy_hex by p=2
  - Finding F3: MPNN training requires data:output ratio ≥ 2.5:1
  - Finding F4: Hard pass/fail boundary at h ≈ 1.3 for TFIM-class models
  - Finding F5: MPNN beats random init in 81-93% of test points
  - Finding F6: 2D topology failures are VQE-limited, not MPNN-limited
- **11. Optimized Runs — h-range Restricted (2026-07-02)** (L667)
  - 11.1 Results Summary
  - 11.2 Run 1: heavy_hex p=3 N=10 — 95% deploy (validates Finding F4)
  - 11.3 Run 2: chain_1d p=4 N=16 — 100% deploy (first N>10 full pass)
  - 11.4 Run 3: chain_1d p=4 N=20 — 100% deploy (largest N in project)
  - 11.5 New Findings from These Runs
- **5. Expressibility Depth Study: p=5, N=8 (2026-07-09)** (L750)
  - Motivation
  - Configuration
  - Results
  - Key Metrics
  - Analysis
  - Comparison: Expressibility Boundary vs p
  - … (+2 más)
- **6. p=4 N=8 with Restricted h-range — Full Pipeline PASS (2026-07-09)** (L830)
- **7. Definitive Result: N=10 p=3 chain_1d h=[1.3, 3.0] — 100% ALL PASS (2026-07-09)** (L863)
- **9. RESOLVED: Gap Floor Artifact Fixed (2026-07-13)** (L889)
  - 9.1 The Fix
  - 9.2 Impact: heavy_hex N=16 Now Passes at 100%
  - 9.3 Tracing: gap_method Field
  - 9.4 Conclusion
- **10. Physics Loss Experiment (2026-07-13)** (L943)
  - Hypothesis
  - Configuration
  - Results: TFIM
  - Results: TFIM Longitudinal
  - Conclusion
- **11. Expressibility Limit Investigation (2026-07-13)** (L995)
  - Question
  - Test Results
  - Root Cause
- **12. Updated Top Configurations (2026-07-13)** (L1035)
  - Key Changes from Previous Rankings

---
