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

.venv/bin/python scripts/experiment_runners/cross_topology/run_multi_topology_training.py --use-residual --film --curriculum --epochs 5000 --patience 300 --max-n 40 --max-de-gap 0.10 --hidden-dim 256 --n-layers 3 -v
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
.venv/bin/python scripts/experiment_runners/cross_topology/run_finetune_from_mt.py --topology heavy_hex --epochs 1000 --lr 3e-4 --max-n 40 -v

# Ladder
.venv/bin/python scripts/experiment_runners/cross_topology/run_finetune_from_mt.py --topology ladder --epochs 1000 --lr 3e-4 --max-n 30 -v

# Square
.venv/bin/python scripts/experiment_runners/cross_topology/run_finetune_from_mt.py --topology square --epochs 1000 --lr 3e-4 --max-n 16 -v
Output: unified_tfim_br_heavy_hex_fromMT_..._p1.pt, etc. (nuevos, sin pisar nada)

Paso 3: Validación (model_comparison)
# Evaluar MT vs ST vs fromMT en cada topología
for topo in heavy_hex ladder square; do
    .venv/bin/python scripts/experiment_runners/cross_topology/run_model_comparison.py     --topology $topo --target-n 10 16 20 --auto-detect --promote-best -v
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
    .venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py     --section 2 --n-qubits $N --topology heavy_hex     --h1 3.0 --h2 0.5 --dt 0.1 --n-trotter 30 --chi-values 64 128 256
done


.venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py  --section 4 --n-qubits 10 --topology heavy_hex  --h1 0.5 --h2 2.0 --dt 0.05 --n-trotter 80
.venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py  --section 4 --n-qubits 14 --topology heavy_hex  --h1 0.5 --h2 2.0 --dt 0.05 --n-trotter 80
.venv/bin/python scripts/experiment_runners/scaling/run_quench_dynamics_study.py  --section 4 --n-qubits 18 --topology heavy_hex  --h1 0.5 --h2 2.0 --dt 0.05 --n-trotter 80



.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py     --topology heavy_hex     --checkpoint data/model_zoo/checkpoints/unified_multiN_heavyhex_p1.pt     --force-recompute     --target-n 16 20 30 40 50 60 --refine-failing --vqe-maxiter 200 --skip-random-baseline

.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology heavy_hex --p-layers 1  --target-n 20 40 50 60 --refine-failing --vqe-maxiter 200 --skip-random-baseline --h-points 6 --force-recompute

.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py     --topology heavy_hex     --checkpoint data/model_zoo/checkpoints/unified_multiN_heavyhex_p1_v2.pt     --force-recompute     --target-n 16 20 30 40 50 60 --refine-failing --vqe-maxiter 200 --skip-random-baseline


# N=18 DQPT trajectory (10 min)

.venv/bin/python scripts/analysis/active_learning_advisor.py --topology heavy_hex --auto-report


.venv/bin/python scripts/analysis/validate_dqpt_results.py --topology heavy_hex --save -v


.venv/bin/python scripts/analysis/qpt_detection.py --topology heavy_hex --compare --save


.venv/bin/python scripts/analysis/circuit_cost_check.py --n-qubits 51 --trotter-steps 10 12 15 18 20 25 30 --save


.venv/bin/python scripts/analysis/dqpt_fidelity_threshold.py --topology heavy_hex --n-qubits 10 --h-pre 3.0 --h-post 0.5 --dt 0.05 --steps 60 --save


.venv/bin/python scripts/analysis/evaluate_gnn_fidelity.py --topology heavy_hex --n-qubits 10 12 14 16 20 --h-values 3.0 2.5 --save


.venv/bin/python scripts/analysis/evaluate_gnn_fidelity.py --topology heavy_hex --n-qubits 20 30 40 --from-extrapolation --save
