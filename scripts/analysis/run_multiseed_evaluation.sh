#!/bin/bash
# Multi-seed cross-topology evaluation
# Trains 3 seeds × 4 topologies = 12 fresh models, then evaluates each
# Results auto-appended to results/multiseed_evaluation.md

set -e
SCRIPT="scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py"
REPORT="results/multiseed_evaluation.md"

# Initialize report
cat > "$REPORT" << EOF
# Multi-Seed Cross-Topology Evaluation

**Date**: $(date +%Y-%m-%d)
**Model**: tfim_bond_resolved, p=1
**Method**: UnifiedMPNN retrained per seed (--multi-n-train --force-retrain)
**Seeds**: 42, 43, 44 (via QMBP_GLOBAL_SEED env var)

---

EOF

for TOPO in chain_1d heavy_hex square triangular; do
    case $TOPO in
        chain_1d)   HMIN=1.5; HMAX=5.5; N=10;;
        heavy_hex)  HMIN=1.4; HMAX=4.5; N=10;;
        square)     HMIN=2.0; HMAX=5.0; N=10;;
        triangular) HMIN=3.0; HMAX=5.5; N=6;;
    esac

    echo "## $TOPO (N=$N, h=[$HMIN, $HMAX])" >> "$REPORT"
    echo "" >> "$REPORT"
    echo "| Seed | Pass@5% | Pass@dual | Mean ΔE/gap | Mean |ΔE| | Result file |" >> "$REPORT"
    echo "|------|---------|-----------|-------------|-----------|-------------|" >> "$REPORT"

    for SEED in 42 43 44; do
        echo ">>> Running $TOPO N=$N seed=$SEED ..."
        QMBP_GLOBAL_SEED=$SEED .venv/bin/python "$SCRIPT" \
            --topology "$TOPO" --target-n "$N" \
            --multi-n-train --force-retrain \
            --h-min "$HMIN" --h-max "$HMAX" --h-points 25 \
            > "results/multiseed_${TOPO}_s${SEED}.log" 2>&1 || true

        # Extract metrics from the latest result JSON
        RESULT_FILE=$(ls -t results/experiments/exp_accel_cross_n/run_*.json 2>/dev/null | head -1)

        if [ -n "$RESULT_FILE" ]; then
            METRICS=$(.venv/bin/python -c "
import json, sys
try:
    with open('$RESULT_FILE') as f:
        d = json.load(f)
    sections = d.get('sections', d.get('results', {}))
    for key, sec in sections.items():
        if isinstance(sec, dict) and 'cross_n_results' in sec:
            for k, v in sec['cross_n_results'].items():
                if 'pass_rate_5pct' in v:
                    p5 = v['pass_rate_5pct']
                    pd = v.get('pass_rate_dual', v.get('pass_rate_5pct', 0))
                    de = v.get('mean_de_gap', 0)
                    ae = v.get('mean_abs_error', 0)
                    print(f'{p5:.0%}|{pd:.0%}|{de:.4f}|{ae:.4f}')
                    sys.exit(0)
    # Fallback: check if multi_n_train section has pass info
    for key, sec in sections.items():
        if isinstance(sec, dict) and 'n_training_points' in sec:
            print(f'train_only|—|—|—')
            sys.exit(0)
    print('no_data|—|—|—')
except Exception as e:
    print(f'error|—|—|{e}')
" 2>/dev/null)
            P5=$(echo "$METRICS" | cut -d'|' -f1)
            PD=$(echo "$METRICS" | cut -d'|' -f2)
            DE=$(echo "$METRICS" | cut -d'|' -f3)
            AE=$(echo "$METRICS" | cut -d'|' -f4)
            FNAME=$(basename "$RESULT_FILE")
            echo "| $SEED | $P5 | $PD | $DE | $AE | $FNAME |" >> "$REPORT"
        else
            echo "| $SEED | ERROR | — | — | — | no result |" >> "$REPORT"
        fi
    done

    echo "" >> "$REPORT"
done

# Append summary statistics
echo "---" >> "$REPORT"
echo "" >> "$REPORT"
echo "## Summary (mean ± std across seeds)" >> "$REPORT"
echo "" >> "$REPORT"

.venv/bin/python -c "
import json, numpy as np
from pathlib import Path
from collections import defaultdict

results_dir = Path('results/experiments/exp_accel_cross_n')
runs = sorted(results_dir.glob('run_*.json'), key=lambda p: p.stat().st_mtime)[-12:]

data = defaultdict(list)
for r in runs:
    with open(r) as f:
        d = json.load(f)
    config = d.get('config', {})
    topo = config.get('topology', '?')
    sections = d.get('sections', d.get('results', {}))
    for key, sec in sections.items():
        if isinstance(sec, dict) and 'cross_n_results' in sec:
            for k, v in sec['cross_n_results'].items():
                if 'pass_rate_5pct' in v:
                    data[topo].append({
                        'pass_5': v['pass_rate_5pct'],
                        'pass_dual': v.get('pass_rate_dual', v['pass_rate_5pct']),
                        'mean_de': v.get('mean_de_gap', 0),
                        'mean_abs': v.get('mean_abs_error', 0),
                    })

lines = ['| Topology | z | N | pass@5% (mean±std) | pass@dual (mean±std) | mean ΔE/gap |',
         '|----------|---|---|--------------------|--------------------|-------------|']
coord = {'chain_1d': 2, 'heavy_hex': 3, 'square': 4, 'triangular': 6}
target_n = {'chain_1d': 10, 'heavy_hex': 10, 'square': 10, 'triangular': 6}

for topo in ['chain_1d', 'heavy_hex', 'square', 'triangular']:
    if topo in data and data[topo]:
        runs_t = data[topo]
        p5 = [r['pass_5'] for r in runs_t]
        pd = [r['pass_dual'] for r in runs_t]
        de = [r['mean_de'] for r in runs_t]
        z = coord.get(topo, '?')
        n = target_n.get(topo, '?')
        lines.append(f'| {topo} | {z} | {n} | {np.mean(p5):.0%} +/- {np.std(p5):.0%} | {np.mean(pd):.0%} +/- {np.std(pd):.0%} | {np.mean(de):.4f} |')

print('\n'.join(lines))
" >> "$REPORT" 2>/dev/null

echo "" >> "$REPORT"
echo "*Generated by run_multiseed_evaluation.sh*" >> "$REPORT"

echo ">>> Done! Report: $REPORT"
