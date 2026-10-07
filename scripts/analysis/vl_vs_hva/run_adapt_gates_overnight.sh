#!/usr/bin/env bash
# Overnight ADAPT gates queue: Gate 0 → Gate 1 → Gate 2, N10 then N14, near h_c.
#
# Each gate persists partial+final results on its own (StudyPersister), so a
# crash or an early morning Ctrl-C never loses completed points. This wrapper
# runs them sequentially with production budgets, logs each to its own file,
# and never lets one failure abort the rest (set +e per command).
#
# Logs: results/hva_vl_study/overnight_logs/<timestamp>/<gate>.log
# Run:  nohup bash scripts/analysis/vl_vs_hva/run_adapt_gates_overnight.sh &
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${ROOT}/.venv/bin/python"
GATE_DIR="${ROOT}/scripts/analysis/vl_vs_hva"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${ROOT}/results/hva_vl_study/overnight_logs/${TS}"
mkdir -p "${LOGDIR}"

HLIST="0.3 0.5 0.7 0.9 1.1 1.3 1.5"
RESTARTS=3
MAXITER=1500
G2_MAXITER=2000

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOGDIR}/queue.log"; }

run_step() {
  local name="$1"; shift
  log "START ${name}"
  "$@" >"${LOGDIR}/${name}.log" 2>&1
  local rc=$?
  if [ ${rc} -eq 0 ]; then
    log "DONE  ${name} (rc=0)"
  else
    log "FAIL  ${name} (rc=${rc}) — continuing queue"
  fi
  return 0
}

log "Overnight ADAPT gates queue — logs in ${LOGDIR}"
log "h_list=${HLIST} restarts=${RESTARTS} maxiter=${MAXITER}"

# ── Gate 0: gradient-ranking validator (N10, N14) ───────────────────────────
run_step gate0_N10 "${PY}" "${GATE_DIR}/run_adapt_gate0.py" \
  --n 10 --h-list ${HLIST} --topk 5 10 20 \
  --restarts ${RESTARTS} --maxiter ${MAXITER} --no-sync
run_step gate0_N14 "${PY}" "${GATE_DIR}/run_adapt_gate0.py" \
  --n 14 --h-list ${HLIST} --topk 5 10 20 \
  --restarts ${RESTARTS} --maxiter ${MAXITER} --no-sync

# ── Gate 1: ADAPT vs top-k at equal 2q (N10, N14) ───────────────────────────
run_step gate1_N10 "${PY}" "${GATE_DIR}/run_adapt_gate1.py" \
  --n 10 --h-list ${HLIST} --grow-step 2 \
  --restarts ${RESTARTS} --maxiter ${MAXITER} --no-sync
run_step gate1_N14 "${PY}" "${GATE_DIR}/run_adapt_gate1.py" \
  --n 14 --h-list ${HLIST} --grow-step 2 \
  --restarts ${RESTARTS} --maxiter ${MAXITER} --no-sync

# ── Gate 2: production growth at N14 (hard h), seeded from Gate 1 FULL θ ─────
G1_NPZ="${ROOT}/results/hva_vl_study/gate1_theta/gate1_full_theta_square_N14.npz"
for H in 0.5 0.3; do
  run_step "gate2_N14_h${H}" "${PY}" "${GATE_DIR}/run_adapt_gate2.py" \
    --n 14 --h ${H} --seed-k 20 --seed-npz "${G1_NPZ}" \
    --target-fid 0.95 --restarts ${RESTARTS} --maxiter ${G2_MAXITER} \
    --grow-coarse 4 --grow-fine 1 --grow-switch-frac 0.6 --no-sync
done

# ── Single scoreboard refresh at the end ────────────────────────────────────
run_step scoreboard "${PY}" "${GATE_DIR}/update_scoreboard.py"

log "QUEUE COMPLETE"
