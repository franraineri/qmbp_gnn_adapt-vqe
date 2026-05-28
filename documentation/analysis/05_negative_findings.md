# Estudio 5 — Hallazgos Negativos como Contribución

**Pregunta**: ¿Qué aprendimos de los 5 experimentos "rejected"?

Cada rejection es una contribución científica — demuestra qué NO funciona y por qué.

## Tabla de Hallazgos Negativos

| ID | Hipótesis | Resultado | Contribución a la Tesis |
|----|-----------|-----------|------------------------|
| E4 | HVA p=2 generaliza a campo longitudinal (g>0) | Fidelity cae a 0.89 con g=0.1 | HVA es model-specific, no model-agnostic |
| F1 | DyPP reduce iteraciones VQE 30-50% | Solo 8-13% de ahorro | Warm-start ya es near-optimal para 4 params |
| G2 | Ensemble variance correlaciona con ΔE/gap (r>0.7) | r=0.195 (no calibrado) | Naive ensemble no sirve para UQ; necesita bootstrap |
| G3 | Optimizaciones N=6 transfieren a N=20 | ΔE/gap=1.26 (falla) | Landscape findings son N-dependent |
| G4 | Condition number κ predice restarts necesarios | r=-0.29 (anti-correlación) | h-value es mejor predictor que κ |

## Análisis por Hallazgo

### E4 — HVA es Model-Specific ❌
**Hipótesis**: HVA p=2 funciona para TFIM + campo longitudinal (g≤0.3).
**Resultado**: Fidelity drops to 0.89 at g=0.1. Pass rate = 24%.
**Por qué importa**: Demuestra que el HVA está diseñado específicamente para TFIM. No es un ansatz universal. Esto es una limitación honesta del framework.
**Para la tesis**: "The HVA ansatz is tailored to the TFIM Hamiltonian structure. Adding a longitudinal field g·ΣZᵢ breaks the symmetry that HVA exploits, confirming that the framework is model-specific rather than model-agnostic."

### F1 — Warm-Start Ya Es Óptimo ❌
**Hipótesis**: DyPP (Dynamic Parameter Prediction) reduce iteraciones 30-50%.
**Resultado**: Solo 8-13% de ahorro. Pass rate = 64%.
**Por qué importa**: El descending warm-start con MPNN ya produce θ_init tan bueno que hay poco margen de mejora. DyPP es redundante.
**Para la tesis**: "The MPNN warm-start already provides near-optimal initialization (mean 14 iterations to convergence). DyPP's 8-13% iteration savings do not justify the added complexity for a 4-parameter HVA."

### G2 — Ensemble UQ No Calibrado ❌
**Hipótesis**: 5-MPNN ensemble variance correlaciona con error real (r>0.7).
**Resultado**: r=0.195. Pass rate = 52%.
**Por qué importa**: Naive ensemble (same data, different init) no produce uncertainty estimates calibradas. Se necesita bootstrap o MC-Dropout.
**Para la tesis**: "Naive ensemble variance (5 MPNNs with different initialization) does not correlate with prediction error (r=0.195). Calibrated uncertainty quantification requires bootstrap resampling or MC-Dropout, which we identify as future work."

### G3 — N=6 Findings No Transfieren ❌
**Hipótesis**: 1 restart + freeze funciona a N=20 (como a N=6).
**Resultado**: ΔE/gap = 1.26 (falla). Pass rate = 11%.
**Por qué importa**: El landscape a N=20 es fundamentalmente diferente. Los local minima que no existen a N=6 aparecen a N=20.
**Para la tesis**: "VQE landscape properties are N-dependent. The saddle-free landscape observed at N=6 (B4) does not persist at N=20, where local minima require ≥7 restarts for reliable convergence."

### G4 — κ No Predice Dificultad ❌
**Hipótesis**: Condition number κ predice cuántos restarts se necesitan.
**Resultado**: r=-0.29 (anti-correlación). Pass rate = 73%.
**Por qué importa**: El h-value (distancia al punto crítico) es mejor predictor de dificultad que cualquier métrica del landscape.
**Para la tesis**: "The Hessian condition number κ does not predict VQE restart requirements (r=-0.29). The transverse field strength h is a more reliable difficulty proxy: h≥2.0 requires 1 restart, h≈1.5 requires 3-5, h≈1.25 requires 7+."

## Implicación Global

Los 5 hallazgos negativos definen los **límites del framework**:
1. **Model-specific** (no universal) — E4
2. **Already near-optimal** (no room for DyPP) — F1
3. **UQ requires proper methodology** — G2
4. **Scaling is non-trivial** — G3
5. **Simple heuristics beat complex metrics** — G4

Estos no son fallos — son contribuciones que delimitan el espacio de aplicabilidad.
