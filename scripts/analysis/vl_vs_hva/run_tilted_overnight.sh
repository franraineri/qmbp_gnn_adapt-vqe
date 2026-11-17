#!/usr/bin/env bash
# Overnight tilted-PBC fidelity-objective campaign.
#
# The transfer study found the energy objective degrades a good transferred seed
# at N=16, while the FIDELITY objective lifts it (smoke: 0.62→0.80 in 100 iter).
# This runs the fidelity objective with a LARGE budget (many restarts, high iter)
# on the two non-degenerate tilted clusters N=16 (4,0) and N=18 (3,3), both seeded
# from the converged N=9 tied θ. Sequential, crash-safe (StudyPersister), no
# intervention needed. Each stage logs to results/hva_vl_study/poolA_logs/.
#
# Usage:  bash scripts/analysis/vl_vs_hva/run_tilted_overnight.sh

set -u
cd "$(dirname "$0")/../../.." || exit 1
PY=.venv/bin/python
R=scripts/analysis/vl_vs_hva/run_tilted_transfer.py
LOG=results/hva_vl_study/poolA_logs
mkdir -p "$LOG"

echo "=== TILTED OVERNIGHT CAMPAIGN START $(date) ==="

# Stage 1 — N=16, fidelity objective, large budget (8 restarts, 4000 iter), p3+p4.
echo "[stage 1] N=16 fidelity objective, large budget"
$PY -u "$R" --src-n 9 --dst-n 16 --h 0.5 --j2 0.5 --p-layers 3 4 \
    --objective fidelity --restarts 8 --maxiter 4000 --target-fid 0.999 \
    > "$LOG/OVERNIGHT_tilted_N16_fidelity.log" 2>&1
echo "[stage 1] done exit=$? $(date)"

# Stage 2 — N=18, fidelity objective, large budget. 2^18 so a touch slower.
echo "[stage 2] N=18 fidelity objective, large budget"
$PY -u "$R" --src-n 9 --dst-n 18 --h 0.5 --j2 0.5 --p-layers 3 4 \
    --objective fidelity --restarts 8 --maxiter 4000 --target-fid 0.999 \
    > "$LOG/OVERNIGHT_tilted_N18_fidelity.log" 2>&1
echo "[stage 2] done exit=$? $(date)"

# Stage 3 — N=16 energy-vs-fidelity A/B at even higher iter (confirm the gap holds
# with budget, p4 only — the best depth).
echo "[stage 3] N=16 A/B both objectives, very high iter (p4)"
$PY -u "$R" --src-n 9 --dst-n 16 --h 0.5 --j2 0.5 --p-layers 4 \
    --objective both --restarts 6 --maxiter 6000 --target-fid 0.999 \
    > "$LOG/OVERNIGHT_tilted_N16_ab_highiter.log" 2>&1
echo "[stage 3] done exit=$? $(date)"

echo "=== TILTED OVERNIGHT CAMPAIGN DONE $(date) ==="
echo "Results:"
echo "  results/hva_vl_study/bond_ablation/tilted_transfer_N9to16_h0.50.json"
echo "  results/hva_vl_study/bond_ablation/tilted_transfer_N9to18_h0.50.json"
echo "  logs in $LOG/OVERNIGHT_tilted_*.log"
