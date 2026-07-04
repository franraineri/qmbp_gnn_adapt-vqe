"""Analyze ALL cross-N results to find why v3 succeeded."""

import glob
import json

# Successful v3 runs
files = sorted(glob.glob("results/scaling/zero_shot/zero_shot_v3_*.json"))
print("=== SUCCESSFUL v3 runs (N=40,80 → various targets, chain_1d) ===")
for path in files:
    with open(path) as f:
        d = json.load(f)
    meta = d["metadata"]
    sa = d["strategy_a_gnn_no_bn"]
    sb = d["strategy_b_interpolation"]
    h_vals = [r["h"] for r in sa["results"]]
    h_range = f"[{min(h_vals):.1f}, {max(h_vals):.1f}]"
    pts = meta.get("n_training_points", "?")
    print(
        f"  N={meta['n_target']:>3}: MSE={sa['training_mse']:.2e}, GNN={sa['mean_de_gap'] * 100:.3f}%, Interp={sb['mean_de_gap'] * 100:.3f}%, h={h_range}, pts={pts}"
    )

# Our warmstart chain_1d
print("\n=== OUR warmstart (chain_1d N=6,8,10 → N=9) ===")
warmstart = sorted(
    glob.glob("results/experiments/exp_cross_n_warmstart_chain_1d_6_8_10_to_9/run_*.json")
)
for path in warmstart[-2:]:
    with open(path) as f:
        d = json.load(f)
    s2 = d["results"]["section_2"]["data"]
    cfg = d["config"]
    print(
        f"  {path.split('/')[-1]}: graphs={s2.get('n_graphs', '?')}, params={s2.get('n_model_params', '?')}, MSE={s2.get('final_mse', 0):.4f}, h=[{cfg['h_grid']['h_min']},{cfg['h_grid']['h_max']}]"
    )

# Bond-resolved cross-N
print("\n=== Bond-resolved cross-N (exp_b4_br_cross_n) ===")
br_files = sorted(glob.glob("results/experiments/exp_b4_br_cross_n/run_*.json"))
for path in br_files:
    with open(path) as f:
        d = json.load(f)
    if "results" in d and "section_2" in d.get("results", {}):
        s2 = d["results"]["section_2"].get("data", {})
        mse = s2.get("final_mse", s2.get("training_mse", "?"))
        print(f"  {path.split('/')[-1]}: MSE={mse}")
