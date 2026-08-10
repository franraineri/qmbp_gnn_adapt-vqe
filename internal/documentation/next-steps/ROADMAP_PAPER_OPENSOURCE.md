# Roadmap: Paper A (GNN-QEM) + Open-Source Release

**Fecha**: 2026-07-21  
**Objetivos mayores**:
1. Paper A (GNN-QEM cross-topology) → arXiv preprint
2. Open-source release con 3 notebooks de demo

---

## Estado Actual del Repositorio (Inventario Completo)

### Estadísticas del repo

| Métrica | Valor |
|---------|-------|
| Archivos en `.hypothesis/` | 587 (cache de tests — no debería estar commiteado) |
| Scripts sueltos en `scripts/` root | 34 (sin organizar) |
| Subdirs en `results/` | 14 (con 55+ experiment subdirs adentro) |
| Documentos en `documentation/` | 40+ analysis notes + 37 binnacles + 62 thesis figures |
| Archivos en `.kiro/` | 33 steering files + 13 knowledge + 15 specs |
| Módulos en `src/qmbp_simulation/` | ~60 .py files en 10 subdirs |
| Tests | ~110 test files (unit 46, integration 11, property 14, hardware 15, etc.) |
| Experiments library | 36+ experiment modules en `experiments/` |

### Problemas de arquitectura para open-source

| Problema | Ejemplos concretos | Severidad |
|----------|-------------------|:---------:|
| Cache/junk commiteado | `.hypothesis/` (587 files), `__pycache__/`, `.mypy_cache/`, `htmlcov/` | ALTA |
| Archivos temporales en raíz | `temporal.txt`, `test_output.log`, `.coverage`, `.DS_Store` | ALTA |
| Directorio con espacio en nombre | `workloads 17-07-25-677/` | ALTA |
| 34 scripts sueltos sin organizar | `analyze_qesem_*.py`, `recover_*.py`, `hardware.py` | MEDIA |
| Shell scripts personales en raíz | `run_fill_matrix_gaps.sh`, `run_hmin_exploration.sh`, etc. | MEDIA |
| Results pesados en git | 14 subdirs de JSONs, logs, checkpoints | ALTA |
| `src/poc/` vacío | Solo un `__init__.py` | BAJA |
| `figures/` con solo 2 archivos | `tfim_long_heavy_hex_scaling.{pdf,png}` | BAJA |
| `reports/` con solo 1 archivo | `health_report_20260617_013241.md` | BAJA |
| `thesis_plots/` con solo 3 PNGs | Deberían estar con la tesis | BAJA |
| `project_health/output/` generado | PNGs/PDFs reproducibles | MEDIA |

### Qué funciona bien (mantener intacto)

- `src/qmbp_simulation/` — paquete bien organizado (10 subdirs, ~60 módulos)
- `pyproject.toml` — configuración correcta de build con optional deps
- `tests/` — suite robusta (~110 test files, unit+integration+property)
- `.github/workflows/ci.yml` — CI existente
- `README.md` — decente como punto de partida
- `Makefile` — bien estructurado con targets claros
- `configs/presets/` — presets organizados por tipo
- `experiments/` — librería estructurada de experimentos (6 categorías)
- `scripts/analysis/` — 13 scripts canónicos (mantener)
- `scripts/experiment_runners/` — runners organizados (12 subdirs)
- `.pre-commit-config.yaml`, `.gitleaks.toml` — buenas prácticas

---

## GOAL 1: Paper A (GNN-QEM) → arXiv

### Qué ya tienes

- [x] Resultados completos en `results/gnn_qem/` (21 archivos)
- [x] Modelos entrenados (.pt files)
- [x] Ablaciones (con/sin E_noisy, GNN vs MLP vs Linear)
- [x] Cross-topology results (chain+ladder → heavy_hex, 100% improvement)
- [x] Post-ZNE validation (incompatibilidad PEA, 15/15 regresiones)
- [x] Finding documentado en `ESTADO_PROYECTO.md` (F4, F9)
- [x] Binnacle detallado: `documentation/binnacles/binnacle-gnn-qem-validation.md`

### Qué falta para el paper

| Task | Prioridad | Estimado |
|------|:---------:|:--------:|
| **P1**: Escribir el paper (LaTeX, 8-10 páginas) | CRÍTICA | 2-3 semanas |
| **P2**: Reproducibility script (1 comando genera todas las figuras) | ALTA | 2-3 días |
| **P3**: Comparación cuantitativa con GEM baseline (citar, no reimplementar) | ALTA | 1 día |
| **P4**: Feature importance analysis (qué features del grafo importan más) | MEDIA | 2-3 días |
| **P5**: Experimento adicional: ≥1 topología target más (triangular o kagome) | MEDIA | 1 semana |
| **P6**: Limpiar código GNN-QEM para repo companion del paper | ALTA | 2-3 días |

### Estructura del paper (propuesta)

```
paper/
├── main.tex                 # Paper principal
├── figures/
│   ├── architecture.pdf     # Diagrama del pipeline GNN-QEM
│   ├── cross_topo_results.pdf
│   ├── ablation.pdf
│   └── pea_incompatibility.pdf
├── supplementary.tex        # Material suplementario
└── reproduce.py             # Script que regenera figuras desde results/
```

### Next Steps (Paper A)

1. **Semana 1**: Extraer el código GNN-QEM a un módulo self-contained + script de reproducción
2. **Semana 2**: Escribir draft del paper (intro, método, resultados)
3. **Semana 3**: Feature importance + comparación con GEM + figuras finales
4. **Semana 4**: Revisión, suplementario, submit a arXiv

---

## GOAL 2: Open-Source Release con 3 Notebooks

### Audiencia target de los notebooks

1. **Notebook 1: Quick Start — Phase Classification Pipeline**
   - Input: modelo + topología + N
   - Corre Phase 1-3 en N=6 (~30s)
   - Muestra: ground truth, VQE convergence, MPNN prediction, PassRate
   - Colab-compatible

2. **Notebook 2: Expressibility Check — Pre-flight Diagnostics**
   - Usa el atlas h_min para diagnosticar viabilidad
   - Input: modelo, topología, N, p, h_range
   - Output: "viable/no viable", CX count, θ_smoothness prediction
   - Incluye visualización de la frontera

3. **Notebook 3: GNN-QEM — Error Correction Demo**
   - Carga modelo pre-entrenado
   - Simula resultados ruidosos
   - Aplica corrección GNN-QEM
   - Muestra: antes/después, cross-topology transfer
   - Requiere datos pre-computados (no QPU)

### Qué cambios de arquitectura necesitas

---

## Reorganización del Repositorio (Detallada)

### Estructura propuesta (post-reorganización)

```
qmbp_gnn_adapt-vqe/
│
├── src/qmbp_simulation/              # 🟢 INTACTO — paquete principal (60 módulos)
│
├── experiments/                       # 🟢 MANTENER — librería de experimentos
│   ├── generalization/
│   ├── hardware/
│   ├── helpers/
│   ├── landscape/
│   ├── optimization/
│   ├── predictor/
│   └── scaling/
│
├── scripts/                           # 🟡 REORGANIZAR
│   ├── analysis/                      # ✅ Mantener (13 canónicos + plot_july/)
│   ├── benchmarks/                    # ✅ Mantener (3 scripts)
│   ├── experiment_runners/            # ✅ Mantener (12 subdirs organizados)
│   ├── hooks/                         # ✅ Mantener (5 pre-commit hooks)
│   ├── runner_templates/              # ✅ Mantener (3 templates + README)
│   ├── validation/                    # ✅ Mantener (5 scripts)
│   └── maintenance/                   # 🟢 NUEVO — mover scripts útiles aquí
│       ├── organize_results.py
│       ├── scan_new_runs.py
│       ├── update_project_status.py
│       ├── generate_presets_from_index.py
│       └── md_index.py
│
├── tests/                             # 🟢 MANTENER intacto
│
├── configs/                           # 🟢 MANTENER
│   └── presets/ (hardware/, noiseless/, noisy/)
│
├── notebooks/                         # 🟢 NUEVO — 3 demos
│   ├── 01_pipeline_quickstart.ipynb
│   ├── 02_expressibility_check.ipynb
│   ├── 03_gnn_qem_demo.ipynb
│   └── data/                          # Modelos y datos pre-computados mínimos
│       ├── pretrained_mpnn_tfim_chain.pt
│       ├── pretrained_gnn_qem.pt
│       └── sample_results.json
│
├── paper/                             # 🟢 NUEVO — Paper A materials
│   ├── main.tex
│   ├── figures/
│   ├── supplementary.tex
│   └── reproduce.py
│
├── docs/                              # 🟢 NUEVO — documentación pública
│   ├── installation.md
│   ├── quickstart.md
│   ├── api_reference.md
│   ├── expressibility_atlas.md
│   └── architecture.md
│
├── internal/                          # 🟡 NUEVO — material de desarrollo
│   ├── thesis/
│   │   ├── tesis-v3.0.tex
│   │   ├── figures/ (← mover thesis_plots/ + documentation/thesis_figures/)
│   │   └── tables/ (← mover documentation/thesis_tables/)
│   ├── binnacles/ (← mover documentation/binnacles/, 37 archivos)
│   ├── analysis_notes/ (← mover documentation/analysis/, 40+ archivos)
│   ├── project_reports/
│   │   ├── ESTADO_PROYECTO.md
│   │   ├── POTENTIAL_PAPERS.md
│   │   ├── UTILIDAD_PRACTICA.md
│   │   ├── ROADMAP_PAPER_OPENSOURCE.md
│   │   └── PROJECT_REPORT_20260617.md
│   ├── bibliography/ (← mover documentation/bibliography/)
│   ├── legacy_scripts/ (← mover 34 scripts sueltos de scripts/)
│   │   ├── analyze_qesem_*.py (5 scripts)
│   │   ├── recover_*.py (3 scripts)
│   │   ├── hardware.py
│   │   ├── preflight_hw.py
│   │   ├── complete_tier0_from_qpu.py
│   │   ├── convert_qesem_to_hwresult.py
│   │   ├── estimate_qesem_budget.py
│   │   ├── print_tier0_circuit.py
│   │   ├── qesem_error_analysis.py
│   │   ├── analyze_run.py
│   │   ├── analyze_v3.py
│   │   ├── compare.py
│   │   ├── compare_gap_methods.py
│   │   ├── verify_affine_bug.py
│   │   ├── verify_thesis_runs.py
│   │   └── (resto de scripts ad-hoc)
│   ├── v7/ (← mover documentation/v7/)
│   └── v8/ (← mover documentation/v8/)
│
├── project_health/                    # 🟡 MANTENER pero marcar como dev-only
│   ├── analysis/
│   ├── cli/
│   ├── core/
│   ├── digest/
│   ├── figures/
│   └── output/ (← ADD TO .gitignore)
│
├── results/                           # 🟡 GITIGNORE la mayoría
│   ├── gnn_qem/                       # ✅ Mantener (datos para Paper A)
│   ├── H_EXPR_MATRIX.md              # ✅ Mantener (referencia del atlas)
│   ├── H_FRONTIER_MODELS.md          # ✅ Mantener
│   ├── H_FRONTIER_TOPOLOGIES.md      # ✅ Mantener
│   ├── HVA_EXPRESSIBILITY_ANALYSIS.md # ✅ Mantener
│   └── .gitkeep
│   # Todo lo demás → .gitignore (experiments/, thesis/, scaling/, etc.)
│
├── .github/workflows/ci.yml
├── .kiro/                             # Mantener (solo útil con Kiro)
├── pyproject.toml
├── Makefile                           # 🟡 ACTUALIZAR paths
├── README.md                          # 🟡 REESCRIBIR
├── LICENSE                            # 🟢 NUEVO (MIT)
├── CONTRIBUTING.md                    # 🟢 NUEVO
├── CHANGELOG.md                       # 🟢 NUEVO
├── requirements.txt
├── .pre-commit-config.yaml
├── .gitignore                         # 🟡 EXPANDIR
├── .gitleaks.toml
├── .markdownlintrc
└── .pymarkdown.json
```

### Detalle de archivos a ELIMINAR (no mover, borrar)

| Archivo/Dir | Razón |
|-------------|-------|
| `temporal.txt` | Archivo temporal |
| `test_output.log` | Output de test |
| `.coverage` | Coverage data (regenerable) |
| `.DS_Store` (todos) | macOS junk |
| `workloads 17-07-25-677/` | Workload temporal con espacio en nombre |
| `src/poc/` | Vacío (solo `__init__.py`) |
| `figures/` (2 files) | Mover a `internal/thesis/figures/` antes de eliminar dir |
| `reports/` (1 file) | Mover a `internal/project_reports/` antes de eliminar dir |
| `thesis_plots/` (3 PNGs) | Mover a `internal/thesis/figures/` antes de eliminar dir |
| `scripts/__pycache__/` | Cache |
| `results/flow_checkpoints/test_flow.pt` | Test artifact |
| `results/recovered/` | Job recovery artifacts |
| `results/analysis/*.txt` | Generated digests (reproducible) |
| `.project_health_state.json` | State file (regenerable) |

### Detalle de .gitignore expandido (agregar)

```gitignore
# === Already should be ignored ===
.hypothesis/
.mypy_cache/
.pytest_cache/
.ruff_cache/
htmlcov/
__pycache__/
*.pyc
.DS_Store
.coverage
*.log

# === New additions ===
# Results (heavy, reproducible)
results/experiments/
results/thesis/
results/thesis_extensions/
results/scaling/
results/bond_resolved_scaling/
results/aqc_tensor/
results/mitigation_benchmark/
results/mitigation_benchmark_v2/
results/benchmarks/
results/checkpoints/
results/flow_checkpoints/
results/recovered/
results/analysis/

# Generated outputs
project_health/output/

# Personal scripts
run_*.sh
temporal.txt
test_output.log
.project_health_state.json

# Workloads
workloads*/
```

### Detalle de scripts/ sueltos — destino de cada uno

| Script | Destino | Razón |
|--------|---------|-------|
| `analyze_noiseless_per_h.py` | `internal/legacy_scripts/` | Superseded por analysis/ |
| `analyze_noiseless_scaling.py` | `internal/legacy_scripts/` | Superseded |
| `analyze_qesem_error_detail.py` | `internal/legacy_scripts/` | Hardware/QESEM específico |
| `analyze_qesem_result.py` | `internal/legacy_scripts/` | Hardware/QESEM |
| `analyze_qesem_tier1.py` | `internal/legacy_scripts/` | Hardware/QESEM |
| `analyze_run.py` | `internal/legacy_scripts/` | Ad-hoc |
| `analyze_v3.py` | `internal/legacy_scripts/` | Outdated |
| `audit_pipeline_consistency.py` | `scripts/validation/` | Útil, mal ubicado |
| `compare_gap_methods.py` | `internal/legacy_scripts/` | One-off analysis |
| `compare.py` | `internal/legacy_scripts/` | Superseded por project_health/compare.py |
| `complete_tier0_from_qpu.py` | `internal/legacy_scripts/` | Hardware recovery |
| `convert_qesem_to_hwresult.py` | `internal/legacy_scripts/` | Hardware conversion |
| `estimate_qesem_budget.py` | `internal/legacy_scripts/` | Hardware budget |
| `generate_analysis_report.py` | `internal/legacy_scripts/` | Superseded |
| `generate_presets_from_index.py` | `scripts/maintenance/` | Útil para mantenimiento |
| `hardware.py` | `internal/legacy_scripts/` | Hardware runner |
| `investigate_vqe_convergence.py` | `internal/legacy_scripts/` | One-off |
| `list_available_backends.py` | `internal/legacy_scripts/` | Hardware utility |
| `md_index.py` | `scripts/maintenance/` | Útil |
| `organize_results.py` | `scripts/maintenance/` | Útil |
| `preflight_hw.py` | `internal/legacy_scripts/` | Hardware preflight |
| `preflight.py` | `scripts/validation/` | Útil, mal ubicado |
| `print_tier0_circuit.py` | `internal/legacy_scripts/` | Hardware debug |
| `qesem_error_analysis.py` | `internal/legacy_scripts/` | Hardware |
| `recover_job_result.py` | `internal/legacy_scripts/` | Hardware recovery |
| `recover_qesem_job.py` | `internal/legacy_scripts/` | Hardware recovery |
| `recover_qesem_jobs.py` | `internal/legacy_scripts/` | Hardware recovery |
| `report_noiseless_per_h.py` | `internal/legacy_scripts/` | Superseded |
| `run_n60_phase3_pipeline.sh` | `internal/legacy_scripts/` | One-off |
| `scan_new_runs.py` | `scripts/maintenance/` | Útil |
| `update_project_status.py` | `scripts/maintenance/` | Útil |
| `verify_affine_bug.py` | `internal/legacy_scripts/` | One-off debug |
| `verify_thesis_runs.py` | `internal/legacy_scripts/` | Thesis-specific |

### documentation/ — destino de cada item

| Item | Destino | Razón |
|------|---------|-------|
| `analysis/` (40+ files) | `internal/analysis_notes/` | Notas internas de investigación |
| `bibliography/` | `internal/bibliography/` | Ref interna |
| `binnacles/` (37 files) | `internal/binnacles/` | Journals de experimentos |
| `thesis_figures/` (62 files) | `internal/thesis/figures/` | Figuras de tesis |
| `thesis_tables/` | `internal/thesis/tables/` | Tablas de tesis |
| `v7/`, `v8/` | `internal/v7/`, `internal/v8/` | Historial de versiones |
| `tasks/` | `internal/tasks/` | Tareas internas |
| `next-steps/` | `internal/next-steps/` | Planificación interna |
| `ESTADO_PROYECTO.md` | `internal/project_reports/` | Report interno |
| `POTENTIAL_PAPERS.md` | `internal/project_reports/` | Análisis interno |
| `UTILIDAD_PRACTICA.md` | `internal/project_reports/` | Análisis interno |
| `ROADMAP_PAPER_OPENSOURCE.md` | `internal/project_reports/` | Este doc |
| `PROJECT_REPORT_20260617.md` | `internal/project_reports/` | Report |
| `architectural_doc_es_en.md` | `docs/architecture.md` (adaptar) | Públicamente útil |
| `qmbp_doc_summary_en.md` | `docs/` (adaptar) | Públicamente útil |
| `simulation_methods_guide.md` | `docs/` (adaptar) | Públicamente útil |
| `thesis-structure-guide.md` | `internal/thesis/` | Solo para tesis |
| `run_plan_with_new_metrics.md` | `internal/project_reports/` | Planificación |
| `qmbp_doc_summary_es.md` | `internal/project_reports/` | Redundante con EN |

---

## Plan de Ejecución (orden cronológico)

### Fase 0: Preparación (1-2 días)

- [ ] Crear branch `feature/open-source-prep`
- [ ] Hacer snapshot/tag del estado actual (`v1.0-thesis`)
- [ ] Definir qué datos mínimos necesitan los notebooks
- [ ] Decidir licencia (MIT recomendado para adopción máxima)

### Fase 1: Limpieza del repo (3-4 días)

**Día 1: Estructura de directorios**
- [ ] Crear `internal/`, `docs/`, `notebooks/`, `paper/`
- [ ] Mover thesis, binnacles, y project reports a `internal/`
- [ ] Mover 30 scripts sueltos de `scripts/` a `internal/legacy_scripts/`
- [ ] Eliminar `src/poc/`, `workloads*/`, `temporal.txt`, `test_output.log`
- [ ] Actualizar `.gitignore` (añadir `results/`, `.hypothesis/`, `*.log`)

**Día 2: Results y datos**
- [ ] Crear `notebooks/data/` con datos mínimos pre-computados
- [ ] Exportar modelos .pt esenciales (MPNN y GNN-QEM pre-trained)
- [ ] Mover `results/` pesados a `.gitignore` (los datos raw no se publican)
- [ ] Mantener solo `results/gnn_qem/` (necesario para paper)

**Día 3: Documentación pública**
- [ ] Reescribir `README.md` orientado a usuario externo (instalación + 3 use cases)
- [ ] Crear `docs/installation.md` (requisitos, venv, pip install)
- [ ] Crear `docs/quickstart.md` (3 comandos para un resultado)
- [ ] Crear `CONTRIBUTING.md` básico
- [ ] Crear `LICENSE` (MIT)

**Día 4: CI y calidad**
- [ ] Verificar que `pip install -e .` funciona limpiamente
- [ ] Verificar que `make test` pasa sin dependencias de archivos eliminados
- [ ] Actualizar CI para que solo corra tests del paquete (no project_health)
- [ ] Verificar que los 3 notebooks corren en un entorno limpio

### Fase 2: Notebooks (3-4 días)

**Notebook 1: Pipeline Quick Start** (~1 día)
```python
# Objetivo: en 30s, correr pipeline completo N=6
from qmbp_simulation import PipelineRunner, LatticeConfig

lattice = LatticeConfig(topology="chain_1d", N=6)
runner = PipelineRunner(model="tfim", p_layers=2)
results = runner.run(lattice, h_range=[1.5, 2.0, 2.5, 3.0, 4.0])
# Visualizar: energías, fidelidades, θ-smoothness, PassRate
```

**Notebook 2: Expressibility Check** (~1 día)
```python
# Objetivo: consultar atlas y diagnosticar viabilidad
from qmbp_simulation.analysis import expressibility_check

report = expressibility_check(
    model="tfim", topology="heavy_hex", N=10, p=3, h_values=[1.0, 1.5, 2.0]
)
report.plot_frontier()    # Visualizar frontera h_min(N) para este config
report.viable_range()     # → "h ≥ 1.4 es viable para esta configuración"
report.cx_budget()        # → "18 CX gates, ZNE viable con GF"
```

**Notebook 3: GNN-QEM Demo** (~2 días — más complejo)
```python
# Objetivo: demostrar corrección de errores cross-topology
from qmbp_simulation.predictors import GNNQEMCorrector

# Cargar modelo pre-entrenado (train: chain+ladder)
corrector = GNNQEMCorrector.load("notebooks/data/pretrained_gnn_qem.pt")

# Simular resultado ruidoso en heavy_hex (zero-shot)
noisy_results = load_sample_noisy_results("heavy_hex")
corrected = corrector.correct(noisy_results)

# Visualizar: antes/después, % mejora por h-value
plot_correction_comparison(noisy_results, corrected, exact_reference)
```

### Fase 3: Paper A draft (2-3 semanas)

**Semana 1: Código + figuras**
- [ ] Crear `paper/reproduce.py` que genera todas las figuras desde `results/gnn_qem/`
- [ ] Diseñar figura 1 (arquitectura GNN-QEM: Hamiltonian graph → GINConv → ΔE)
- [ ] Generar figura 2 (cross-topology results bar chart)
- [ ] Generar figura 3 (ablation: GNN vs MLP vs Linear)
- [ ] Generar figura 4 (PEA incompatibility scatter)

**Semana 2: Escritura**
- [ ] Intro + related work (posicionar vs GEM, ML-QEM)
- [ ] Method (architecture, training protocol, graph encoding)
- [ ] Results (in-distribution, zero-shot, ablation, negative)
- [ ] Discussion (why Hamiltonian graph > circuit graph for VQE-specific QEM)

**Semana 3: Pulido + submit**
- [ ] Supplementary material (hiperparámetros, training curves)
- [ ] Proofread + format check (arXiv requirements)
- [ ] Upload a arXiv (quant-ph)
- [ ] Linkar repo GitHub en el paper

### Fase 4: Release público (1-2 días)

- [ ] Merge branch a main
- [ ] Push tag `v2.0.0-oss`
- [ ] Verificar que notebooks corren en Google Colab
- [ ] (Opcional) Publicar en PyPI: `pip install qmbp-simulation`
- [ ] Anunciar en Twitter/X + comunidad Qiskit

---

## Decisiones de Arquitectura Pendientes

### 1. ¿Mono-repo o split?

**Opción A: Mono-repo (recomendado)**
- Todo en un repo, con `internal/` claramente separado
- Pro: simplicidad, un solo git clone para todo
- Con: repo pesado si results/ no se gitignore bien

**Opción B: Split**
- `qmbp-simulation` (paquete + notebooks) → repo público
- `qmbp-thesis` (tesis + binnacles + results) → repo privado
- Pro: repo público limpio
- Con: mantenimiento de dos repos, imports rotos

**Recomendación**: Mono-repo con `.gitignore` agresivo para `results/` y `internal/`
marcado como "development materials, not part of the package".

### 2. ¿Cómo empaquetar los modelos pre-trained?

**Opción A: En el repo (archivos .pt <10MB)**
- Pro: zero config, clone-and-run
- Con: bloat del repo

**Opción B: GitHub Releases / artifacts**
- Pro: repo ligero
- Con: requiere download step adicional

**Recomendación**: Opción A si total < 20MB (tus .pt son ~1-5MB cada uno).
Incluir en `notebooks/data/` con LFS si es necesario.

### 3. ¿project_health queda o se va?

**Recomendación**: Queda en `internal/project_health/` y se marca como
dev-only dependency. No se instala con `pip install qmbp-simulation`.
El pyproject.toml puede tener un optional dependency:
```toml
[project.optional-dependencies]
health = ["project_health @ ./internal/project_health"]
```

### 4. ¿GNN-QEM como módulo del paquete o repo separado?

**Recomendación**: Dentro del paquete en `src/qmbp_simulation/predictors/gnn_qem.py`
(probablemente ya existe). El paper simplemente referencia el paquete con un script
de reproducción en `paper/reproduce.py`.

---

## Métricas de Éxito

| Milestone | Criterio | Deadline |
|-----------|----------|:--------:|
| Repo limpio | `pip install -e .` + tests pass en fresh venv | Agosto 1 |
| Notebooks corren | 3 notebooks ejecutan en Colab sin error | Agosto 8 |
| Paper A draft | Borrador completo compartible | Agosto 22 |
| arXiv submit | Paper en arXiv | Septiembre 5 |
| Release público | Tag + notebooks verificados en Colab | Septiembre 12 |

---

## Riesgos y Mitigaciones

| Riesgo | Probabilidad | Mitigación |
|--------|:------------:|------------|
| Tests rotos tras reorganizar | Alta | Tag v1.0 antes de mover nada, CI en cada paso |
| Notebooks requieren datos que eliminaste | Media | Definir datos mínimos ANTES de limpiar |
| Paper rechazado por falta de hardware results | Baja | Posicionar como "simulation validated" explícitamente |
| Alguien publica algo similar a GNN-QEM antes | Media | Priorizar Paper A sobre todo lo demás |

---

*Este roadmap se actualizará conforme avancemos. Próxima revisión: después de Fase 0.*
