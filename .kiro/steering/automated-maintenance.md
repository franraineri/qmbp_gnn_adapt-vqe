inclusion: always

# Automated Maintenance (NO AI tokens needed)

## Post-Run Hooks (automáticos, 0 intervención)

| Hook | Trigger | Qué hace |
|------|---------|----------|
| `zoo-coherence-check` | agentStop | Diagnóstico: zoo↔dashboard↔GT coherencia |
| `validate-zoo-registration` | postTaskExecution | Previene n_qubits=10 bug |

## Auto-Integrations (built into runner_base.run())

| Qué | Cuándo | Cómo |
|-----|--------|------|
| **post_experiment_sync()** | Post-run, fire-and-forget subprocess | Consolidated: GT check → dashboard → best results scoreboard → auto-fix issues → eval report → coverage → ResultIndex → exclusions |
| Zoo pass_rate update | Post-run, if sections produced pass_rate_dual | `_log_data_quality_feedback` → `_extract_best_pass_rate_dual` → `auto_update_zoo_pass_rate` |
| Zoo pass_rate_by_n | Post model_comparison | `update_zoo_pass_rate_by_n()` per evaluated model |
| Exclusion-policy N-level filter | During `load_best_mpnn_for_cross_n` training branch | Removes N-values with `contaminated_training`/`gap_masking` failure modes |
| Auto-detect exclusions | Post-run (inside post_experiment_sync) | `auto_detect_exclusions()` |
| Dashboard regeneration | Post-run (inside post_experiment_sync) | `generate_model_quality_dashboard()` + GT coherence + MT vs ST |
| ResultIndex + project-status.md | Post-run (inside post_experiment_sync) | `ResultIndex.rebuild()` + `refresh_status()` |
| Eval report + coverage doc | Post-run (inside post_experiment_sync) | `evaluate_zoo_models.py` + `update_cross_n_coverage.py` |

## Scripts de Mantenimiento (correr manualmente, sin AI)

```bash
# Audit profundo + coherencia (subsume check_zoo_coherence checks)
.venv/bin/python scripts/maintenance/audit_and_fix_model_zoo.py --coherence

# ── Data Integrity & Consistency ──────────────────────────
# Full data consistency check (zoo ↔ comparison ↔ dashboard ↔ registry)
.venv/bin/python scripts/maintenance/query_model_registry.py consistency

# GT ↔ NPZ coherence (detect stale e_exact in training data)
.venv/bin/python -c "from qmbp_simulation.analysis.metrics import validate_gt_npz_coherence; r=validate_gt_npz_coherence(); print(r['summary'])"

# Fix stale e_exact (auto-correct NPZ from GT cache, creates .bak backups)
.venv/bin/python -c "from qmbp_simulation.analysis.metrics import validate_gt_npz_coherence; validate_gt_npz_coherence(fix=True)"

# Full post-experiment sync (all stores in correct order)
.venv/bin/python -c "from qmbp_simulation.analysis.metrics import post_experiment_sync; post_experiment_sync(verbose=True)"

# Backfill pass_rate_by_n from comparison history
.venv/bin/python -c "from qmbp_simulation.predictors.model_zoo import backfill_pass_rate_by_n_from_comparisons; print(f'Updated: {backfill_pass_rate_by_n_from_comparisons()} models')"

# ── MT vs ST Comparison ───────────────────────────────────
# Full comparison (all topologies)
.venv/bin/python scripts/maintenance/query_model_registry.py compare -v

# Filtered comparison
.venv/bin/python scripts/maintenance/query_model_registry.py compare -t chain_1d heavy_hex --n-min 10 -v

# Save report
.venv/bin/python scripts/maintenance/query_model_registry.py compare --save

# Aplicar fixes: tags + exclusion sync
.venv/bin/python scripts/maintenance/audit_and_fix_model_zoo.py --fix --sync-exclusions

# Limpieza completa (prune + archive)
.venv/bin/python scripts/maintenance/audit_and_fix_model_zoo.py --fix --prune-stale --archive-orphans

# Solo coherencia (rápido, hook-friendly)
.venv/bin/python scripts/maintenance/check_zoo_coherence.py

# Ver qué modelos necesitan retrain
.venv/bin/python scripts/maintenance/check_zoo_coherence.py --retrain-queue

# Evaluar modelos + actualizar zoo pass_rate
.venv/bin/python scripts/analysis/evaluate_zoo_models.py --update-zoo

# Regenerar coverage docs
.venv/bin/python scripts/maintenance/update_cross_n_coverage.py

# Actualizar project-status.md
.venv/bin/python scripts/maintenance/update_project_status.py

# ── Test maintenance ──
# Validar imports de tests (detecta paths rotos en <2s)
.venv/bin/python scripts/general_project_maintenance/validate_test_imports.py

# Auto-fix imports reubicados
.venv/bin/python scripts/general_project_maintenance/validate_test_imports.py --fix

# Correr test suite con timeout + parallel
.venv/bin/python scripts/general_project_maintenance/run_test_suite.py -j 4

# Re-correr solo los que fallaron antes
.venv/bin/python scripts/general_project_maintenance/run_test_suite.py --lf
```

## Cuándo Correr Qué

| Evento | Acción automática | Acción manual recomendada |
|--------|-------------------|---------------------------|
| Después de VQE/training | `post_experiment_sync()` via runner_base | `query_model_registry.py consistency` |
| Después de model_comparison | `update_zoo_pass_rate_by_n` + post_experiment_sync | `query_model_registry.py compare -v` |
| Antes de un retrain | — | `query_model_registry.py consistency` + `check_zoo_coherence.py --retrain-queue` |
| Después de agregar NPZ data | — | `validate_gt_npz_coherence(fix=True)` + `evaluate_zoo_models.py --update-zoo` |
| Después de actualizar GT cache | — | `validate_gt_npz_coherence(fix=True)` (corrige NPZ e_exact) |
| Limpieza periódica | — | `audit_and_fix_model_zoo.py --fix --archive-orphans --sync-exclusions` |
| Exclusiones desincronizadas | — | `audit_and_fix_model_zoo.py --coherence` (detecta drift) |
| Sospecha de data inconsistente | — | `query_model_registry.py consistency` |

## Funciones Clave (import desde src/, no duplicar)

| Necesidad | Import |
|-----------|--------|
| Evaluar calidad de NPZ | `from qmbp_simulation.analysis.metrics import get_usable_training_configs` |
| Dataset limpio para training | `from qmbp_simulation.framework.result_io import build_clean_training_dataset` |
| Validar antes de registrar | `from qmbp_simulation.predictors.model_zoo import _validate_zoo_entry` |
| Cola de retrain | `from qmbp_simulation.predictors.model_zoo import compute_retrain_queue` |
| Actualizar pass_rate post-eval | `runner.auto_update_zoo_pass_rate(pass_rate_dual)` (auto desde run()) |
| **Actualizar pass_rate_by_n** | `from qmbp_simulation.predictors.model_zoo import update_zoo_pass_rate_by_n` |
| **Backfill by_n desde history** | `from qmbp_simulation.predictors.model_zoo import backfill_pass_rate_by_n_from_comparisons` |
| Exclusion policy filter | `from qmbp_simulation.analysis.metrics import load_training_exclusions, get_excluded_files` |
| Sync exclusions → registry | `from qmbp_simulation.analysis.metrics import auto_detect_exclusions` |
| Zoo integrity check | `from qmbp_simulation.predictors.model_zoo import validate_zoo` |
| **GT↔NPZ coherencia** | `from qmbp_simulation.analysis.metrics import validate_gt_npz_coherence` |
| **Consistency cross-check** | `from qmbp_simulation.analysis.metrics import validate_data_consistency` |
| **Post-experiment full sync** | `from qmbp_simulation.analysis.metrics import post_experiment_sync` |
| **Auto-fix scoreboard issues** | `from qmbp_simulation.analysis.metrics import auto_fix_scoreboard_issues` |
| **MT vs ST query** | `from qmbp_simulation.analysis.metrics import query_mt_vs_st_comparison` |
