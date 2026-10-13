#!/usr/bin/env bash
# Overnight MASTER queue — two phases, sequential, crash-safe per runner.
#
# Phase 1 — Option B (canonical operator-list ADAPT), first REAL test, at N10 & N14,
#           several h, generous budget so it competes fairly (needs many ops).
# Phase 2 — Pool-A winners (warm-seed + layer-penalty + analytic-seed), the current
#           best config, at N18 across the ADAPT h-grid (0.3/0.5/0.7/0.9).
#
# Each runner persists partial+final itself (StudyPersister). set +e per command so
# one failure never aborts the rest. Logs per config.
#   Logs: results/hva_vl_study/night_master_logs/<timestamp>/<run>.log
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
GATE_DIR="${ROOT}/scripts/analysis/vl_vs_hva"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${ROOT}/results/hva_vl_study/night_master_logs/${TS}"
mkdir -p "${LOGDIR}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOGDIR}/queue.log"; }
run_step() {
  local name="$1"; shift
  log "START ${name}"
  "$@" >"${LOGDIR}/${name}.log" 2>&1
  local rc=$?
  if [ ${rc} -eq 0 ]; then log "DONE  ${name} (rc=0)"; else log "FAIL  ${name} (rc=${rc}) — continuing"; fi
  return 0
}

log "NIGHT MASTER queue — ${LOGDIR}"

# ── Phase 1: Option B (canonical operator-list ADAPT) — N10 then N14 ─────────
# Generous budget: max-ops high enough to build Pool-A-scale circuits (~90-200 CX
# = 45-100 ops), restarts>=3 so freshly appended operators activate, grow-step 2
# as a practical middle ground (pure 1-at-a-time is too slow at N14).
B_HLIST="0.3 0.5 0.7 0.9"
log "=== PHASE 1: Option B (operator-list ADAPT) ==="
run_step poolB_N10 "${PY}" "${GATE_DIR}/run_adapt_poolB.py" \
  --n 10 --h-list ${B_HLIST} --restarts 3 --maxiter 1500 \
  --grow-step 2 --max-ops 120 --target-fid 0.95 --no-sync
run_step poolB_N14 "${PY}" "${GATE_DIR}/run_adapt_poolB.py" \
  --n 14 --h-list ${B_HLIST} --restarts 3 --maxiter 1500 \
  --grow-step 2 --max-ops 160 --target-fid 0.95 --no-sync

# ── Phase 2: Pool-A winners at N18 (current best config) ─────────────────────
# warm-seed (B) + layer-growth-penalty (C) + analytic-seed-new (2c): the measured
# champions. N18 h-grid matching ADAPT. Penalty tuned for large N (helps there).
log "=== PHASE 2: Pool-A winners at N18 ==="
run_step poolA_best_N18 "${PY}" "${GATE_DIR}/run_adapt_poolA.py" \
  --n 18 --h-list 0.3 0.5 0.7 0.9 --restarts 3 --maxiter 1500 \
  --target-fid 0.95 --max-layers 6 --min-dfid 0.01 --layer-growth-penalty 1.0 \
  --warm-seed --analytic-seed-new --no-sync

run_step scoreboard "${PY}" "${GATE_DIR}/update_scoreboard.py"
log "QUEUE COMPLETE"
