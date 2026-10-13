#!/usr/bin/env bash
# N18 comparison anchors: variant_topk prune sweep for p2_base and p2_half_nn_rx
# at h=0.5 and h=0.7 — the same reference set we compared ADAPT against at N14.
# Sequential so the four runs share CPU and the GT cache (one process at a time).
set -u
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
LOGDIR=results/hva_vl_study/n18_anchor_logs
mkdir -p "$LOGDIR"

TOLS="0.05 0.1 0.2"
for variant in p2_base p2_half_nn_rx; do
  for h in 0.5 0.7; do
    log="$LOGDIR/${variant}_h${h}.log"
    echo "[queue] $variant h=$h -> $log"
    $PY -u scripts/analysis/vl_vs_hva/run_variant_topk.py \
      --n 18 --h "$h" --variant "$variant" \
      --prune-tols $TOLS --restarts 3 --maxiter 800 --no-sync \
      > "$log" 2>&1
    echo "[queue] done $variant h=$h (exit $?)"
  done
done
echo "[queue] ALL N18 ANCHORS COMPLETE"
