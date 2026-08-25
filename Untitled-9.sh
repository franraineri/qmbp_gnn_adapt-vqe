.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology chain_1d --p-layers 2 --train-n 6 --target-n 6 10 --multi-n-train --n-anchors 10 --n-restarts 6 --h-min 1.5 --h-max 3.5 --h-points 16
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology chain_1d --p-layers 2 --train-n 8 --target-n 6 10 --multi-n-train --n-anchors 10 --n-restarts 6
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology chain_1d --p-layers 2 --train-n 10 --target-n 10 12 14 16 --multi-n-train --n-anchors 12 --n-restarts 6
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology ladder --p-layers 2 --train-n 6 --target-n 6 10 --multi-n-train --n-anchors 10 --n-restarts 6
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology ladder --p-layers 2 --train-n 8 --target-n 8 10 --multi-n-train --n-anchors 10 --n-restarts 6
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology ladder --p-layers 2 --train-n 10 --target-n 10 12 14 16 --multi-n-train --n-anchors 12 --n-restarts 6
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology square --p-layers 2 --train-n 4 --target-n 4 6 8 --multi-n-train --n-anchors 10 --n-restarts 6
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology square --p-layers 2 --train-n 6 --target-n 6 8 10 --multi-n-train --n-anchors 10 --n-restarts 6
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology heavy_hex --p-layers 2 --train-n 6 --target-n 6 8 10 --multi-n-train --n-anchors 10 --n-restarts 6
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology heavy_hex --p-layers 2 --train-n 10 --target-n 10 12 14 16 --multi-n-train --n-anchors 12 --n-restarts 6
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology chain_1d --p-layers 2 --train-n 10 --target-n 10 14 16 20 --multi-n-train --force-retrain --iterative-improve --max-iterations 2
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology ladder --p-layers 2 --train-n 10 --target-n 10 12 14 16 20 --multi-n-train --force-retrain --iterative-improve --max-iterations 2


.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology heavy_hex --p-layers 1 --multi-n-train --n-anchors 10 --force-retrain


Paso 1: Re-entrenar MT universal (con curriculum)
Runner:
run_multi_topology_training.py
 Datos: 2474 graphs (después de filtro de calidad) de 5 topologías Arquitectura: residual + FiLM (la misma que dio los mejores resultados la vez anterior)

.venv/bin/python scripts/experiment_runners/cross_topology/run_multi_topology_training.py --use-residual --film --curriculum --epochs 5000 --patience 300 --max-n 40 --max-de-gap 0.10 --hidden-dim 256 --n-layers 3
Por qué estos parámetros:

--curriculum: entrena primero en chain_1d + heavy_hex (mayor calidad), luego fine-tune en todas. Esto funcionó antes.
--epochs 5000 --patience 300: el modelo anterior hizo 2000 epochs con diverged status. Más epochs + más patience para que converja realmente.
--max-n 40: incluir los datos de N=30-40 que ahora tenemos para heavy_hex y ladder (antes no estaban)
--max-de-gap 0.10: filtro estricto, solo puntos con ΔE/gap < 10%
Output: unified_tfim_br_MT_residual+film_p1_v2.pt (gracias al nuevo versionado, el actual _p1.pt no se pierde)

Paso 2: Fine-tune para cada topología
Runner: run_finetune_from_mt.py

Una vez que el MT está entrenado, especializamos para cada topología:

# Heavy_hex (la que mejor respondió al fine-tune antes)
.venv/bin/python scripts/experiment_runners/cross_topology/run_finetune_from_mt.py --topology heavy_hex --epochs 1000 --lr 3e-4 --max-n 40

# Ladder
.venv/bin/python scripts/experiment_runners/cross_topology/run_finetune_from_mt.py --topology ladder --epochs 1000 --lr 3e-4 --max-n 30

# Square
.venv/bin/python scripts/experiment_runners/cross_topology/run_finetune_from_mt.py --topology square --epochs 1000 --lr 3e-4 --max-n 16
Output: unified_tfim_br_heavy_hex_fromMT_..._p1.pt, etc. (nuevos, sin pisar nada)

Paso 3: Validación (model_comparison)
# Evaluar MT vs ST vs fromMT en cada topología
for topo in heavy_hex ladder square; do
    .venv/bin/python scripts/experiment_runners/cross_topology/run_model_comparison.py     --topology $topo --target-n 10 16 20 --auto-detect --promote-best
done
Esto genera los eval reports que alimentan el scoreboard automáticamente.





.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology heavy_hex --p-layers 1  --target-n 30 40 50 60 --refine-failing --vqe-maxiter 200 --skip-random-baseline --h-points 6 --force-recompute
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology heavy_hex --p-layers 1  --target-n 8 10 20 30 40 --checkpoint data/model_zoo/checkpoints/unified_tfim_br_heavy_hex_multiN_4+6+8+10+12+14+18+20+22+24+30+32+40_p1.pt --skip-random-baseline --h-points 6 --force-recompute
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology heavy_hex --p-layers 1  --target-n 30 40 50 60 --checkpoint data/model_zoo/checkpoints/unified_multiN_heavyhex_p1.pt --refine-failing --vqe-maxiter 200 --skip-random-baseline --h-points 6 --force-recompute
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology heavy_hex --p-layers 1  --target-n 30 40 50 60 --checkpoint data/model_zoo/checkpoints/unified_tfim_br_MT_residual+film_p1.pt --refine-failing --vqe-maxiter 200 --skip-random-baseline --h-points 6 --force-recompute
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology heavy_hex --p-layers 1  --target-n 30 40 50 60 --checkpoint data/model_zoo/checkpoints/unified_tfim_br_heavy_hex_fromMT_4+6+8+10+12+14+18+20+21+26+30+40_p1.pt --refine-failing --vqe-maxiter 200 --skip-random-baseline --h-points 6 --force-recompute
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology heavy_hex --p-layers 1  --target-n 30 40 50 60 --checkpoint data/model_zoo/checkpoints/unified_multiN_heavyhex_p1_v2.pt --refine-failing --vqe-maxiter 200 --skip-random-baseline --h-points 6 --force-recompute






.venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py --section 4 --n-qubits 18 --topology heavy_hex --dqpt-h-pre 0.5 --dqpt-h-post 2.0 --dqpt-dt 0.05 --dqpt-steps 80


for N in 10 16 20; do
    .venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology heavy_hex --p-layers 2 --train-n $N --target-n $N --multi-n-train --h-min 1.85 --h-max 4.0 --iterative-improve --max-iteration 2 --n-restarts 2
done


.venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py  --section 4 --n-qubits 10 --topology heavy_hex  --h1 0.5 --h2 2.0 --dt 0.05 --n-trotter 80
.venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py  --section 4 --n-qubits 14 --topology heavy_hex  --h1 0.5 --h2 2.0 --dt 0.05 --n-trotter 80
.venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py  --section 4 --n-qubits 18 --topology heavy_hex  --h1 0.5 --h2 2.0 --dt 0.05 --n-trotter 80



.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py     --topology heavy_hex     --checkpoint data/model_zoo/checkpoints/unified_multiN_heavyhex_p1.pt     --force-recompute     --target-n 16 20 30 40 50 60 --refine-failing --vqe-maxiter 200 --skip-random-baseline

.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology heavy_hex --p-layers 1  --target-n 20 40 50 60 --refine-failing --vqe-maxiter 200 --skip-random-baseline --h-points 6 --force-recompute

.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py     --topology heavy_hex     --checkpoint data/model_zoo/checkpoints/unified_multiN_heavyhex_p1_v2.pt     --force-recompute     --target-n 16 20 30 40 50 60 --refine-failing --vqe-maxiter 200 --skip-random-baseline


# N=18 DQPT trajectory (10 min)

.venv/bin/python scripts/analysis/active_learning_advisor.py --topology heavy_hex --auto-report


.venv/bin/python scripts/analysis/validate_dqpt_results.py --topology heavy_hex --save


.venv/bin/python scripts/analysis/qpt_detection.py --topology heavy_hex --compare --save


.venv/bin/python scripts/analysis/circuit_cost_check.py --n-qubits 51 --trotter-steps 10 12 15 18 20 25 30 --save


.venv/bin/python scripts/analysis/dqpt_fidelity_threshold.py --topology heavy_hex --n-qubits 10 --h-pre 3.0 --h-post 0.5 --dt 0.05 --steps 60 --save


.venv/bin/python scripts/analysis/evaluate_gnn_fidelity.py --topology heavy_hex --n-qubits 10 12 14 16 20 --h-values 3.0 2.5 --save


.venv/bin/python scripts/analysis/evaluate_gnn_fidelity.py --topology heavy_hex --n-qubits 20 30 40 --from-extrapolation --save





.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology heavy_hex --target-n 10 --h-min 2.5 --h-max 5.0 --h-points 20 --p-layers 1  --refine-failing --vqe-maxiter 200 --skip-random-baseline
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology heavy_hex --target-n 12 --h-min 2.5 --h-max 5.0 --h-points 20 --p-layers 1   --refine-failing --vqe-maxiter 200 --skip-random-baseline
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology heavy_hex --target-n 14 --h-min 2.5 --h-max 5.0 --h-points 20 --p-layers 1    --refine-failing --vqe-maxiter 200 --skip-random-baseline
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology heavy_hex --target-n 16 --h-min 2.5 --h-max 5.0 --h-points 20 --p-layers 1     --refine-failing --vqe-maxiter 200 --skip-random-baseline
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology heavy_hex --target-n 18 --h-min 2.5 --h-max 5.0 --h-points 20 --p-layers 1     --refine-failing --vqe-maxiter 200 --skip-random-baseline
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology heavy_hex --target-n 20 --h-min 2.5 --h-max 5.0 --h-points 20 --p-layers 1     --refine-failing --vqe-maxiter 200 --skip-random-baseline
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology heavy_hex --target-n 24 --h-min 2.5 --h-max 5.0 --h-points 15 --p-layers 1     --refine-failing --vqe-maxiter 200 --skip-random-baseline
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology heavy_hex --target-n 26 --h-min 2.5 --h-max 5.0 --h-points 15 --p-layers 1     --refine-failing --vqe-maxiter 200 --skip-random-baseline
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology heavy_hex --target-n 30 --h-min 2.5 --h-max 5.0 --h-points 15 --p-layers 1     --refine-failing --vqe-maxiter 200 --skip-random-baseline

########################################################################
### ENTRENAMIENTO DE MODELOS MPNN — heavy_hex (multi-N, todos los datos N=4..40)
###
### Orden: (A) modelos ST directos → (B) MT base → (C) finetune desde MT
### Cada modelo tiene nombre unico. Correr en este orden respeta dependencias.
### Flags nuevos disponibles: --loss-type {theta_mse,energy_weighted}
###                           --physics-loss-weight FLOAT (0.0 = off)
###                           --use-residual  --film   (GINEConv activo por defecto)
########################################################################

# ===== (A) MODELOS ST — ejes de arquitectura y loss (mismos datos, aisla el metodo) =====

# A1. baseline: sin arquitectura extra, loss MSE (control base)
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology heavy_hex --multi-n-train --force-retrain --model-name baseline_mse

# A2. residual (GINEConv + residual, loss MSE)
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology heavy_hex --multi-n-train --force-retrain --use-residual --model-name res_mse

# A3. residual + FiLM (arquitectura completa, loss MSE)
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology heavy_hex --multi-n-train --force-retrain --use-residual --film --model-name res_film_mse

# A4. energy_weighted loss SIN residual (aisla efecto de la loss pura)
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology heavy_hex --multi-n-train --force-retrain --loss-type energy_weighted --model-name plain_energyw

# A5. residual + energy_weighted loss
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology heavy_hex --multi-n-train --force-retrain --use-residual --loss-type energy_weighted --model-name res_energyw

# A6. residual + FiLM + energy_weighted (stack + loss energetica)
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology heavy_hex --multi-n-train --force-retrain --use-residual --film --loss-type energy_weighted --model-name res_film_energyw

# A7. physics-informed loss (regularizacion por energia, lambda=0.05)
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology heavy_hex --multi-n-train --force-retrain --use-residual --physics-loss-weight 0.05 --model-name res_physics05

# A8. STACK COMPLETO: GINEConv + residual + FiLM + energy_weighted + physics
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology heavy_hex --multi-n-train --force-retrain --use-residual --film --loss-type energy_weighted --physics-loss-weight 0.05 --model-name full_stack

# A9. stack completo + iterative-improve (auto-refina puntos que fallan)
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology heavy_hex --multi-n-train --force-retrain --use-residual --film --loss-type energy_weighted --iterative-improve --max-iterations 3 --model-name full_stack_iter

# A10. regimen critico (h cerca de h_c) — energy_weighted brilla donde de_gap varia
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology heavy_hex --multi-n-train --force-retrain --use-residual --film --loss-type energy_weighted --h-min 0.5 --h-max 2.5 --model-name res_film_energyw_critical

# A11. regimen deploy (h en zona de interes 2.0-4.0)
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology heavy_hex --multi-n-train --force-retrain --use-residual --film --h-min 2.0 --h-max 4.0 --model-name res_film_deploy

# ===== (B) MODELO MULTI-TOPOLOGY BASE (necesario para el finetune de la seccion C) =====
# NOTA: el MT runner usa --use-residual (NO --residual)
.venv/bin/python scripts/experiment_runners/cross_topology/run_multi_topology_training.py --use-residual --film --curriculum --epochs 5000 --patience 300 --max-n 40 --max-de-gap 0.10 --hidden-dim 256 --n-layers 3 --model-name res_film_base
.venv/bin/python scripts/experiment_runners/cross_topology/run_multi_topology_training.py --use-residual --film --curriculum --epochs 5000 --patience 300 --max-n 40 --loss-type energy_weighted --model-name res_film_energyw_base

# ===== (C) FINETUNE DESDE MT hacia heavy_hex (depende de que exista el MT correspondiente) =====
.venv/bin/python scripts/experiment_runners/cross_topology/run_finetune_from_mt.py --topology heavy_hex --source-checkpoint data/model_zoo/checkpoints/unifMPNN__MT_p1_res_film_base.pt --epochs 1000 --lr 3e-4 --max-n 40 --model-name ft_from_MT
.venv/bin/python scripts/experiment_runners/cross_topology/run_finetune_from_mt.py --topology heavy_hex --source-checkpoint data/model_zoo/checkpoints/unifMPNN__MT_p1_res_film_energyw_base.pt --epochs 1000 --lr 3e-4 --max-n 40 --loss-type energy_weighted --model-name ft_energyw_from_MT

########################################################################
### EVALUACION — extrapolacion a N grande para TODOS los modelos entrenados
### (compara objetivamente cual predice mejores angulos theta)
### Corre despues de que TODOS los modelos de arriba esten entrenados.
########################################################################

for CKPT in \
    data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_baseline_mse.pt \
    data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_res_mse.pt \
    data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_res_film_mse.pt \
    data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_plain_energyw.pt \
    data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_res_energyw.pt \
    data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_res_film_energyw.pt \
    data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_res_physics05.pt \
    data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_full_stack.pt \
    data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_full_stack_iter.pt \
    data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_res_film_energyw_critical.pt \
    data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_res_film_deploy.pt \
    data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_ft_from_MT.pt \
    data/model_zoo/checkpoints/unifMPNN__heavy_hex_p1_ft_energyw_from_MT.pt \
; do
    .venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology heavy_hex --p-layers 1 --target-n 20 30 40 50 --checkpoint "$CKPT" --refine-failing --vqe-maxiter 200 --skip-random-baseline --h-points 6 --force-recompute
done

# ===== SINCRONIZACION + SCOREBOARD (que modelo gano) =====
.venv/bin/python -c "from qmbp_simulation.analysis.metrics import post_experiment_sync; post_experiment_sync(verbose=True)"
.venv/bin/python scripts/analysis/generate_best_results_scoreboard.py
.venv/bin/python scripts/maintenance/query_model_registry.py list --topology heavy_hex
cat results/best_results_scoreboard.md
########################################################################
