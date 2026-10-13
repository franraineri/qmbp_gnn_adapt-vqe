#!/usr/bin/env bash
# N18 ADAPT improved — the three useful h-points (0.3 excluded: gap ~3e-5, degenerate).
# Config: full improved levers + Pareto-stop (fpc_drop_min_fid 0.85) + lookahead
# (now gated on the one-step top-2 margin, so it's cheap when one action dominates)
# + the _fit_seed guard for the p2 anchor. Sequential so one process owns the CPU.
set -u
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
LOGDIR=results/hva_vl_study/poolA_logs
mkdir -p "$LOGDIR"

for h in 0.5 0.7 0.9; do
  log="$LOGDIR/N18_queue_h${h}.log"
  echo "[queue] ADAPT N18 h=$h -> $log"
  $PY -u scripts/analysis/vl_vs_hva/run_adapt_poolA.py \
    --n 18 --h-list "$h" --restarts 3 --maxiter 1500 --target-fid 0.95 \
    --max-layers 6 --prerank-micro --warm-seed --analytic-seed-new \
    --layer-growth-penalty 1.0 --tiebreak-abs-fid --lookahead \
    --stop-on-fpc-drop --fpc-drop-min-fid 0.85 --efficiency-patience 1 \
    --pareto-min-fid 0.85 --out-suffix queue_n18_h${h} --no-sync \
    > "$log" 2>&1
  echo "[queue] done ADAPT N18 h=$h (exit $?)"
done
echo "[queue] ALL N18 ADAPT COMPLETE"
