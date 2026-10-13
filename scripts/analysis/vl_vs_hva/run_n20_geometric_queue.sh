#!/usr/bin/env bash
# N=20 geometry-aware sweep, CHAINED AFTER the running pareto queue.
# Waits for the pareto-queue bash (PID passed as $1) to finish, THEN runs the
# geometric nnn-selection sweep on both regimes — so it joins the same serial
# pipeline without interrupting the current process.
set -u
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
LOGDIR=results/hva_vl_study/poolA_logs
mkdir -p "$LOGDIR"

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[geomq] waiting for pareto queue PID $WAIT_PID to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[geomq] pareto queue finished — starting geometric sweep"
fi

for h in 0.5 1.0; do
  log="$LOGDIR/N20_geometric_h${h}.log"
  echo "[geomq] geometric sweep h=$h -> $log"
  $PY -u scripts/analysis/vl_vs_hva/run_n20_geometric.py \
    --n 20 --h "$h" --restarts 1 --maxiter 400 \
    > "$log" 2>&1
  echo "[geomq] done geometric h=$h (exit $?)"
done
echo "[geomq] N20 GEOMETRIC QUEUE COMPLETE"
