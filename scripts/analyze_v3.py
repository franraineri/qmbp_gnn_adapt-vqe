"""Analyze the successful v3 cross-N results to understand what was different."""

import glob
import json

files = sorted(glob.glob("results/scaling/zero_shot/zero_shot_v3_*.json"))
print(f"Found {len(files)} v3 results\n")

for path in files:
    with open(path) as f:
        d = json.load(f)
    meta = d["metadata"]
    sa = d["strategy_a_gnn_no_bn"]
    sb = d["strategy_b_interpolation"]

    tag = f"N={meta['n_source_sizes']} → N={meta['n_target']}"
    print(f"=== {tag} ({meta['topology']}) ===")
    print(
        f"  Training: {meta.get('n_training_points', '?')} pts, {meta.get('n_model_params', '?')} params, {meta.get('hidden_dim', '?')} hidden, {meta.get('n_epochs', '?')} epochs"
    )
    print(f"  Strategy: {meta.get('strategy', '?')}, precision: {meta.get('precision', '?')}")
    print(f"  GNN MSE: {sa['training_mse']:.6f}")
    print(
        f"  GNN:    mean ΔE/gap={sa['mean_de_gap'] * 100:.3f}%, pass={sa.get('n_pass', '?')}/{sa.get('n_total', len(sa.get('results', [])))}"
    )
    print(
        f"  Interp: mean ΔE/gap={sb['mean_de_gap'] * 100:.3f}%, pass={sb.get('n_pass', '?')}/{sb.get('n_total', len(sb.get('results', [])))}"
    )
    print(f"  Winner: {d['summary']['best_strategy']}")
    print()
