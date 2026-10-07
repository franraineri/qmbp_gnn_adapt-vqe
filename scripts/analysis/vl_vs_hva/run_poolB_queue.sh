#!/usr/bin/env bash
# ADAPT Option-B queue: canonical operator-list growth, N10 then N14, hard h.
#
# run_adapt_poolB.py persists partial+final itself (StudyPersister); this wrapper
# runs N10 then N14 sequentially with production budgets, logs each separately,
# and never lets one failure abort the other (set +e per command).
#
# Logs: results/hva_vl_study/poolB_logs/<timestamp>/<run>.log
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
GATE_DIR="${ROOT}/scripts/analysis/vl_vs_hva"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${ROOT}/results/hva_vl_study/poolB_logs/${TS}"
mkdir -p "${LOGDIR}"

HLIST="0.3 0.5 0.7 0.9"
RESTARTS=3           # >=2 required so a freshly appended operator can activate
MAXITER=1500
TARGET_FID=0.95
GROW_STEP=1          # canonical ADAPT: one operator at a time
MAX_OPS=200
MIN_DFID=0.01

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOGDIR}/queue.log"; }

run_step() {
  local name="$1"; shift
  log "START ${name}"
  "$@" >"${LOGDIR}/${name}.log" 2>&1
  local rc=$?
  if [ ${rc} -eq 0 ]; then log "DONE  ${name} (rc=0)"; else log "FAIL  ${name} (rc=${rc}) — continuing"; fi
  return 0
}

log "ADAPT Option-B queue — logs in ${LOGDIR}"
log "h_list=${HLIST} restarts=${RESTARTS} maxiter=${MAXITER} grow_step=${GROW_STEP} max_ops=${MAX_OPS}"

run_step poolB_N10 "${PY}" "${GATE_DIR}/run_adapt_poolB.py" \
  --n 10 --h-list ${HLIST} --restarts ${RESTARTS} --maxiter ${MAXITER} \
  --target-fid ${TARGET_FID} --grow-step ${GROW_STEP} --max-ops ${MAX_OPS} \
  --min-dfid ${MIN_DFID} --no-sync

run_step poolB_N14 "${PY}" "${GATE_DIR}/run_adapt_poolB.py" \
  --n 14 --h-list ${HLIST} --restarts ${RESTARTS} --maxiter ${MAXITER} \
  --target-fid ${TARGET_FID} --grow-step ${GROW_STEP} --max-ops ${MAX_OPS} \
  --min-dfid ${MIN_DFID} --no-sync

run_step scoreboard "${PY}" "${GATE_DIR}/update_scoreboard.py"

log "QUEUE COMPLETE"
