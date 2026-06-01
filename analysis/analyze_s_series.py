"""Analyze S-series experiment results."""

import glob
import json

for exp_id in ["s1", "s2", "s3", "s4", "s5", "s6"]:
    files = sorted(glob.glob(f"results/experiments/exp_{exp_id}/run_*.json"))
    if not files:
        print(f"  {exp_id.upper()}: NO RESULTS")
        continue
    with open(files[-1]) as f:
        data = json.load(f)
    analysis = data.get("analysis", {})
    summary = analysis.get("summary", {})
    hypothesis = analysis.get("hypothesis", "")

    print(f"=== {exp_id.upper()} ===")
    print(f"  Hypothesis: {hypothesis[:100]}")
    mean_dg = summary.get("mean_de_gap", "N/A")
    pass_rate = summary.get("pass_rate", "N/A")
    total_time = summary.get("total_time_s", "N/A")
    print(f"  Mean dE/gap: {mean_dg}")
    print(f"  Pass rate: {pass_rate}")
    print(f"  Total time: {total_time}s")

    # Extract technique_metadata from all results
    results = data.get("results", {})
    for seed, metrics in results.items():
        if not metrics:
            continue
        for m in metrics:
            meta = m.get("technique_metadata", {})
            if not meta:
                continue
            # S1: entanglement
            if "S_at_boundary_p2" in meta:
                s_p2 = meta["S_at_boundary_p2"]
                s_p1 = meta.get("S_at_boundary_p1")
                s_p1_str = f"{s_p1:.4f}" if s_p1 is not None else "N/A"
                print(
                    f"  Seed {seed}, N={meta.get('N')}: S(h_min_p2)={s_p2:.4f}, S(h_min_p1)={s_p1_str}"
                )
            # S2: cross-topology
            elif "transfer_type" in meta:
                dg = meta.get("de_gap", 0)
                print(
                    f"  Seed {seed}, {meta.get('topology')}: transfer={meta['transfer_type']}, dE/gap={dg:.4f}, pass={meta.get('pass_5pct')}"
                )
            # S3: landscape
            elif "fluctuation" in meta:
                kappa = meta.get("condition_number")
                kappa_str = f"{kappa:.0f}" if kappa else "N/A"
                print(
                    f"  Seed {seed}, h={m.get('h_value')}: fluct={meta['fluctuation']:.3f}, kappa={kappa_str}, distinct={meta.get('n_distinct_minima')}"
                )
            # S4: data efficiency
            elif "k" in meta:
                print(
                    f"  Seed {seed}, k={meta['k']}: dE/gap={meta['de_gap']:.4f}, pass={meta['pass_5pct']}"
                )
            # S5: N=20 pipeline
            elif "de_gap_mpnn" in meta:
                print(
                    f"  Seed {seed}, h={m.get('h_value')}: MPNN={meta['de_gap_mpnn']:.4f}, interp={meta['de_gap_interpolation']:.4f}, {meta['mpnn_vs_interp']}"
                )
            # S6: MC-Dropout
            elif "mc_variance" in meta:
                r = meta.get("pearson_r", "")
                if r:
                    print(f"  Seed {seed}: Pearson r={r:.3f} (G2 baseline=0.195)")
    print()
