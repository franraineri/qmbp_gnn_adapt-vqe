## catálogo completo de herramientas de análisis organizadas por propósito:

1. Estado Global del Proyecto
# Project status (coverage matrix, regressions, large-N grades)
cat .kiro/steering/project-status.md

# Best-ever results per topology × N (at h=2.5)
cat results/best_results_scoreboard.md

# Regenerate best results scoreboard
.venv/bin/python scripts/analysis/generate_best_results_scoreboard.py

# Data integrity check (GT, NPZ, zoo, dashboard coherencia)
.venv/bin/python scripts/maintenance/inspect_data_stores.py

# Data integrity con validación cruzada del dashboard
.venv/bin/python scripts/maintenance/inspect_data_stores.py --validate-dashboard

# Data consistency (zoo↔comparison↔dashboard cross-check)
.venv/bin/python scripts/maintenance/query_model_registry.py consistency

# Full post-experiment sync (regenera todo en orden correcto)
.venv/bin/python -c "from qmbp_simulation.analysis.metrics import post_experiment_sync; post_experiment_sync(verbose=True)"
2. MT vs ST Comparison
# Comparación global MT vs ST (todas las topologías)
.venv/bin/python scripts/maintenance/query_model_registry.py compare -v

# Filtrado por topología
.venv/bin/python scripts/maintenance/query_model_registry.py compare -t chain_1d -v
.venv/bin/python scripts/maintenance/query_model_registry.py compare -t heavy_hex ladder -v

# Filtrado por N (solo extrapolation regime)
.venv/bin/python scripts/maintenance/query_model_registry.py compare --n-min 16 -v

# JSON output (para programmatic access)
.venv/bin/python scripts/maintenance/query_model_registry.py --json compare -t chain_1d

# Guardar reporte markdown
.venv/bin/python scripts/maintenance/query_model_registry.py compare --save

# Desde Python (más flexible):
.venv/bin/python -c "
from qmbp_simulation.analysis.metrics import query_mt_vs_st_comparison
r = query_mt_vs_st_comparison(topology='chain_1d', n_min=10)
print(f'MT {r[\"global\"][\"mt_wins\"]} - ST {r[\"global\"][\"st_wins\"]}')
for s in r['per_scenario']:
    print(f'  N={s[\"n_qubits\"]}: MT={s[\"mt_pass_rate\"]:.0%} ST={s[\"st_pass_rate\"]:.0%} → {s[\"winner\"]}')
"
3. Per-Topology Analysis
# Dashboard quality per topology (training data health)
.venv/bin/python -c "
import json
d = json.load(open('data/model_quality_dashboard.json'))
for topo, info in sorted(d['topology_summary'].items()):
    print(f'{topo:12s}: n_max_viable=N{info.get(\"n_max_viable\",\"?\")}, best_pass={info.get(\"best_pass_rate_5pct\",0):.0%}')
"

# Per-topology NPZ quality breakdown
.venv/bin/python -c "
import json
d = json.load(open('data/model_quality_dashboard.json'))
for c in sorted(d['configs'], key=lambda x: (x['topology'], x['n_qubits'])):
    print(f'{c[\"topology\"]:12s} N={c[\"n_qubits\"]:>2}: {c[\"n_points\"]:>3}pts pass_dual={c[\"pass_rate_dual_criterion\"]:.0%} h_frontier={c.get(\"h_frontier\",0):.2f} utility={c[\"training_utility\"]}')
"

# Zoo models per topology (with pass_rate_by_n)
.venv/bin/python -c "
from qmbp_simulation.predictors.model_zoo import _load_manifest
for e in _load_manifest():
    by_n = ', '.join(f'N{k}={float(v):.0%}' for k,v in sorted(e.pass_rate_by_n.items(), key=lambda x: int(x[0]))) if e.pass_rate_by_n else 'no per-N data'
    print(f'{e.topology:14s} pass={e.pass_rate:.0%} | {by_n}')
"

# Specific topology deep-dive (registry detail)
.venv/bin/python scripts/maintenance/query_model_registry.py list --topology chain_1d
.venv/bin/python scripts/maintenance/query_model_registry.py list --topology heavy_hex
.venv/bin/python scripts/maintenance/query_model_registry.py list --topology ladder
.venv/bin/python scripts/maintenance/query_model_registry.py list --topology square
.venv/bin/python scripts/maintenance/query_model_registry.py list --topology triangular
4. Model Quality & Health
# Eval report (grades A-F per model per N)
cat results/model_evaluation_report.md

# Registry health dashboard (all models)
.venv/bin/python scripts/maintenance/query_model_registry.py health-dashboard

# Specific model detail
.venv/bin/python scripts/maintenance/query_model_registry.py get "unified_tfim_br_chain*"

# Best model for deployment at specific N
.venv/bin/python scripts/maintenance/query_model_registry.py best -t chain_1d -n 20
.venv/bin/python scripts/maintenance/query_model_registry.py best -t heavy_hex -n 16

# Failure diagnostics (why is model failing?)
.venv/bin/python scripts/maintenance/query_model_registry.py diagnose "unified_tfim_br_chain_1d_multiN*"

# Retrain queue (what needs retraining and why)
.venv/bin/python -c "
from qmbp_simulation.predictors.model_zoo import compute_retrain_queue
queue = compute_retrain_queue()
for item in queue[:10]:
    print(f'{item[\"priority\"]}. {item[\"topology\"]:12s} {item[\"reason\"]}')
"

# Zoo coherence audit
.venv/bin/python scripts/maintenance/check_zoo_coherence.py
5. Pipeline Summary (MT pipeline completo)
# Consolidated report (ablation + training + MT vs ST + extrapolation)
.venv/bin/python scripts/analysis/pipeline_summary.py

# JSON output
.venv/bin/python scripts/analysis/pipeline_summary.py --json

# Save markdown
.venv/bin/python scripts/analysis/pipeline_summary.py --save
6. Model Comparison (run evaluations)
# Compare MT vs ST on specific topology (runs MPNN evaluation)
.venv/bin/python scripts/experiment_runners/cross_topology/run_model_comparison.py \
    --topology chain_1d --target-n 10 16 20 --auto-detect -v

# Compare on heavy_hex
.venv/bin/python scripts/experiment_runners/cross_topology/run_model_comparison.py \
    --topology heavy_hex --target-n 10 16 20 --auto-detect -v

# Compare with specific checkpoints
.venv/bin/python scripts/experiment_runners/cross_topology/run_model_comparison.py \
    --topology ladder --target-n 10 16 20 \
    --checkpoints data/model_zoo/checkpoints/unified_tfim_br_MT_residual+film_p1.pt \
                  data/model_zoo/checkpoints/unified_tfim_br_ladder_multiN_4+6+8+10+12+16+20+26_p1.pt \
    -v --save-report
7. GT & Data Integrity
# GT↔NPZ coherence check (read-only)
.venv/bin/python -c "from qmbp_simulation.analysis.metrics import validate_gt_npz_coherence; print(validate_gt_npz_coherence()['summary'])"

# Backfill pass_rate_by_n from comparison history
.venv/bin/python -c "from qmbp_simulation.predictors.model_zoo import backfill_pass_rate_by_n_from_comparisons; print(f'Updated: {backfill_pass_rate_by_n_from_comparisons()}')"
8. Coverage & Documentation
# Cross-N coverage documentation (auto-generated)
cat internal/documentation/analysis/accelerated_cross_n_coverage.md

# Cross-topology summary (thesis)
cat internal/documentation/thesis/cross_topology_summary.md
Quick Reference: "Dame el estado de X"
Quiero saber...	Comando
Estado global	cat .kiro/steering/project-status.md
Best-ever por topo×N	cat results/best_results_scoreboard.md
MT gana o pierde?	query_model_registry.py compare -v
chain_1d cómo va?	query_model_registry.py compare -t chain_1d -v
Qué modelo usar para N=20?	query_model_registry.py best -t chain_1d -n 20
Por qué falla en N=20?	query_model_registry.py diagnose "unified*chain*"
Data está limpia?	query_model_registry.py consistency
Qué reentrenar?	compute_retrain_queue()
Grades por N	cat results/model_evaluation_report.md
