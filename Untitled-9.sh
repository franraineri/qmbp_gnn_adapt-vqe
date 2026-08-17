.venv/bin/python scripts/maintenance/upgrade_npz_quality_tiers.py
.venv/bin/python -c "from qmbp_simulation.predictors.model_zoo import refresh_zoo_quality_scores; refresh_zoo_quality_scores()"

.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology ladder --train-n 4 --target-n 4 --h-min 2.0 --h-max 4.8 --h-points 15 --n-anchors 15 --maxiter 2000 --n-restarts 12 --iterative-improve --max-iterations 2 --refine-all --multi-n-train
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology ladder --train-n 6 --target-n 6 --h-min 2.0 --h-max 4.8 --h-points 15 --n-anchors 15 --maxiter 2500 --n-restarts 15 --iterative-improve --max-iterations 2 --refine-all --multi-n-train
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology ladder --train-n 8 --target-n 8 --h-min 2.0 --h-max 4.8 --h-points 15 --n-anchors 15 --maxiter 2500 --n-restarts 15 --iterative-improve --max-iterations 2 --refine-all --multi-n-train
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology ladder --train-n 10 --target-n 10 --h-min 2.0 --h-max 4.8 --h-points 20 --n-anchors 20 --maxiter 3000 --n-restarts 18 --iterative-improve --max-iterations 3 --refine-all --multi-n-train
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology ladder --train-n 12 --target-n 12 --h-min 2.5 --h-max 5.0 --h-points 15 --n-anchors 15 --maxiter 3000 --n-restarts 18 --iterative-improve --max-iterations 3 --refine-all --multi-n-train
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology ladder --train-n 16 --target-n 16 --h-min 3.0 --h-max 5.5 --h-points 15 --n-anchors 15 --maxiter 3500 --n-restarts 20 --iterative-improve --max-iterations 3 --refine-all --multi-n-train
.venv/bin/python scripts/maintenance/upgrade_npz_quality_tiers.py
.venv/bin/python -c "from qmbp_simulation.predictors.model_zoo import refresh_zoo_quality_scores; refresh_zoo_quality_scores()"
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology heavy_hex --train-n 4 --target-n 4 --h-min 2.0 --h-max 4.5 --h-points 15 --n-anchors 15 --maxiter 2000 --n-restarts 12 --iterative-improve --max-iterations 2 --refine-all --multi-n-train
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology chain_1d --train-n 10 --target-n 15 --h-min 2.0 --h-max 5.0 --h-points 20 --n-anchors 20 --maxiter 2500 --n-restarts 15 --iterative-improve --max-iterations 3 --refine-all --multi-n-train
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology chain_1d --train-n 10 --target-n 20 --h-min 2.0 --h-max 5.0 --h-points 15 --n-anchors 15 --maxiter 2500 --n-restarts 15 --iterative-improve --max-iterations 3 --refine-all --multi-n-train
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology ladder --train-n 10 --target-n 14 --h-min 2.0 --h-max 4.8 --h-points 20 --n-anchors 20 --maxiter 2500 --n-restarts 15 --iterative-improve --max-iterations 3 --refine-all --multi-n-train
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology ladder --train-n 10 --target-n 14 --h-min 2.0 --h-max 4.8 --h-points 20 --n-anchors 20 --maxiter 2500 --n-restarts 15 --iterative-improve --max-iterations 3 --refine-all --multi-n-train
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology ladder --train-n 10 --target-n 16 --h-min 2.0 --h-max 5.0 --h-points 18 --maxiter 3000 --n-restarts 15 --iterative-improve --max-iterations 3 --refine-all --multi-n-train
.venv/bin/python scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py --topology heavy_hex --train-n 6 --target-n 4 --h-min 2.0 --h-max 4.5 --h-points 14 --maxiter 2000 --n-restarts 10 --iterative-improve --max-iterations 2 --refine-all --multi-n-train
.venv/bin/python scripts/maintenance/upgrade_npz_quality_tiers.py
.venv/bin/python -c "from qmbp_simulation.predictors.model_zoo import refresh_zoo_quality_scores; refresh_zoo_quality_scores()"


## done until here

.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology chain_1d --target-n 20 30 40 60 --h-min 2.5 --h-max 5.0 --h-points 6 --refine-failing --max-refine 100
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology heavy_hex --target-n 20 30 40 --h-min 2.5 --h-max 4.5 --h-points 6 --refine-failing --max-refine 100
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology ladder --target-n 20 26 30 40 --h-min 2.5 --h-max 5.0 --h-points 6 --refine-failing --max-refine 100
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology square --target-n 16 20 30 --h-min 2.5 --h-max 4.5 --h-points 6 --refine-failing --max-refine 100


### the only pendings are above. triangular done
.venv/bin/python scripts/experiment_runners/scaling/run_large_n_extrapolation.py --topology triangular --target-n 12 16 24 --h-min 3.5 --h-max 5.0 --h-points 6 --refine-failing --max-refine 100
.venv/bin/python -c "from qmbp_simulation.predictors.model_zoo import refresh_zoo_quality_scores; refresh_zoo_quality_scores()"
.venv/bin/python scripts/maintenance/upgrade_npz_quality_tiers.py
