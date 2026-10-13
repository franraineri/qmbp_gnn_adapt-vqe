#!/usr/bin/env bash
# N=20 h=1.0 ansatz anchors — restarts=1 (warm-start only). h=1.0 has a large gap
# (0.227, easy regime) so the analytic warm-start converges without the expensive
# random restarts that made N18 (gap 0.006) crawl. Sequential; one process at a time.
set -u
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
LOGDIR=results/hva_vl_study/poolA_logs
mkdir -p "$LOGDIR"

for variant in p2_half_nn_rx p2_base; do
  log="$LOGDIR/N20_h10_${variant}.log"
  echo "[queue] $variant -> $log"
  $PY -u scripts/analysis/vl_vs_hva/run_variant_topk.py \
    --n 20 --h 1.0 --variant "$variant" \
    --prune-tols 0.1 0.2 --restarts 1 --maxiter 800 --no-sync \
    > "$log" 2>&1
  echo "[queue] done $variant (exit $?)"
done
echo "[queue] N20 h=1.0 ANCHORS COMPLETE"
