#!/usr/bin/env python3
"""Analyze topology training data: where does VQE converge?

Scans cross-N experiment results plus data stores (GT cache, eval cache,
model zoo, NPZ training data) for a given topology.

Usage:
    python scripts/maintenance/analyze_topology_results.py chain_1d
    python scripts/maintenance/analyze_topology_results.py heavy_hex
"""
import json
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <topology>")
    print("  e.g. chain_1d, ladder, heavy_hex, square, triangular")
    sys.exit(1)

_topology = sys.argv[1]

# ═══════════════════════════════════════════════════════════════
# Section A: Cross-N experiment results (exp_accel_cross_n)
# ═══════════════════════════════════════════════════════════════
print(f"{'═' * 60}")
print(f"TOPOLOGY ANALYSIS: {_topology}")
print(f"{'═' * 60}")

results_dir = ROOT / "results" / "experiments" / "exp_accel_cross_n"
topology_runs = [f for f in results_dir.glob("run_*.json")
                 if _topology in f.read_text()[:500]]

if not topology_runs:
    print(f"\n[Cross-N Results] No runs found in {results_dir.name}")
else:
    latest = sorted(topology_runs)[-1]
    with open(latest) as f:
        data = json.load(f)

    print(f"\n[Cross-N Results] Latest: {latest.name}")
    print(f"  Config: {data.get('config', {})}")
    print()

    # Find per-point training results
    for key, section in data.get("results", {}).items():
        if not isinstance(section, dict):
            continue
        sd = section.get("data", {})
        # Check cross-N self-eval (train-n == target-n)
        cross_n = sd.get("cross_n_results", {})
        for ck, result in cross_n.items():
            if "per_point" not in result:
                continue
            per_point = result["per_point"]
            print(f"  Section: {key}, Config: {ck}")
            print(f"  {'h':>6} {'ΔE/gap':>8} {'|ΔE|':>10} {'Fidelity':>9} {'Pass@5%':>8}")
            print("  " + "-" * 50)
            for r in per_point:
                h = r["h"]
                de_gap = r["de_gap"]
                abs_err = r["abs_error"]
                fid = r.get("fidelity", 0) or 0
                status = "✓" if de_gap < 0.05 else ("~" if de_gap < 0.10 else "✗")
                print(f"  {h:>6.2f} {de_gap:>8.4f} {abs_err:>10.2e} {fid:>9.4f} {status:>8}")

            # Summary by h-region
            easy = [r for r in per_point if r["h"] >= 2.7]
            mid = [r for r in per_point if 2.4 <= r["h"] < 2.7]
            hard = [r for r in per_point if r["h"] < 2.4]
            print()
            print(f"  h >= 2.7: {sum(1 for r in easy if r['de_gap']<0.05)}/{len(easy)} pass")
            print(f"  2.4<=h<2.7: {sum(1 for r in mid if r['de_gap']<0.05)}/{len(mid)} pass")
            print(f"  h < 2.4: {sum(1 for r in hard if r['de_gap']<0.05)}/{len(hard)} pass")
            print()
            if easy:
                passing = [r['h'] for r in easy if r['de_gap'] < 0.05]
                if passing:
                    print(f"  → Safe zone h>={min(passing):.2f}")

# ═══════════════════════════════════════════════════════════════
# Section B: Ground Truth Cache
# ═══════════════════════════════════════════════════════════════
gt_path = DATA / "ground_truth_cache.json"
if gt_path.exists():
    with open(gt_path) as f:
        raw = json.load(f)
    gt_entries = raw.get("entries", raw) if isinstance(raw, dict) else {}
    # Filter to this topology
    topo_gt = {k: v for k, v in gt_entries.items() if k.startswith(f"{_topology}|")}
    print(f"\n[Ground Truth Cache] {len(topo_gt)} entries for {_topology}")
    if topo_gt:
        # Group by N|model
        from collections import defaultdict
        gt_groups = defaultdict(list)
        for key in topo_gt:
            parts = key.split("|")
            gt_groups[f"N={parts[1]} {parts[2]}"].append(float(parts[3]))
        for gk in sorted(gt_groups.keys()):
            hs = sorted(gt_groups[gk])
            print(f"  {gk}: {len(hs)} pts, h=[{hs[0]:.2f}, {hs[-1]:.2f}]")
else:
    print(f"\n[Ground Truth Cache] NOT FOUND")

# ═══════════════════════════════════════════════════════════════
# Section C: Eval Cache
# ═══════════════════════════════════════════════════════════════
ec_path = DATA / "eval_cache.json"
if ec_path.exists():
    with open(ec_path) as f:
        ec_raw = json.load(f)
    ec_entries = ec_raw.get("entries", ec_raw) if isinstance(ec_raw, dict) else {}
    topo_ec = {k: v for k, v in ec_entries.items() if f"|{_topology}|" in k}
    print(f"\n[Eval Cache] {len(topo_ec)} entries for {_topology}")
    if topo_ec:
        from collections import defaultdict
        ec_groups = defaultdict(int)
        for key in topo_ec:
            parts = key.split("|")
            # key: model|topology|N|p|J|h|theta_hash
            if len(parts) >= 4:
                ec_groups[f"{parts[0]} N={parts[2]} p={parts[3]}"] += 1
        for gk in sorted(ec_groups.keys()):
            print(f"  {gk}: {ec_groups[gk]} cached evaluations")
else:
    print(f"\n[Eval Cache] NOT FOUND")

# ═══════════════════════════════════════════════════════════════
# Section D: Model Zoo
# ═══════════════════════════════════════════════════════════════
manifest_path = DATA / "model_zoo" / "manifest.json"
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
    zoo_entries = manifest if isinstance(manifest, list) else manifest.get("entries", [])
    topo_zoo = [e for e in zoo_entries if e.get("topology") == _topology]
    print(f"\n[Model Zoo] {len(topo_zoo)} models for {_topology}")
    for entry in sorted(topo_zoo, key=lambda e: (-e.get("pass_rate", 0), e.get("n_qubits", 0))):
        print(f"  {entry.get('checkpoint_file', '?')}: "
              f"{entry.get('model', '?')} N={entry.get('n_qubits', '?')} "
              f"p={entry.get('p_layers', '?')}, "
              f"pass={entry.get('pass_rate', 0):.0%}, "
              f"pts={entry.get('n_training_points', 0)}, "
              f"h={entry.get('h_range', [])}")
else:
    print(f"\n[Model Zoo] NOT FOUND")

# ═══════════════════════════════════════════════════════════════
# Section E: NPZ Training Data
# ═══════════════════════════════════════════════════════════════
npz_dir = DATA / "multi_n_training"
if npz_dir.exists():
    topo_npz = sorted(npz_dir.glob(f"{_topology}_N*.npz"))
    print(f"\n[NPZ Training Data] {len(topo_npz)} files for {_topology}")
    for npz_file in topo_npz:
        npz_data = np.load(npz_file, allow_pickle=True)
        h_vals = npz_data["h_values"]
        theta = npz_data["theta_opt"]
        n_pts = len(h_vals)
        n_params = theta.shape[1] if theta.ndim == 2 else 0
        n_nan = int(np.sum(~np.isfinite(theta)))

        # Quality stats
        e_key = "e_vqe" if "e_vqe" in npz_data else ("energies" if "energies" in npz_data else None)
        quality = ""
        if e_key and "e_exact" in npz_data:
            abs_err = np.abs(npz_data[e_key] - npz_data["e_exact"])
            if "de_gaps" in npz_data:
                de_gaps = npz_data["de_gaps"]
            elif "gaps" in npz_data:
                de_gaps = abs_err / np.maximum(npz_data["gaps"], 1e-10)
            else:
                de_gaps = abs_err  # proxy
            n_good = int(np.sum(de_gaps < 0.05))
            quality = f"  pass@5%={n_good}/{n_pts}"

        status = "✅" if n_nan == 0 else f"⚠️ {n_nan} NaN"
        print(f"  {npz_file.name}: {n_pts} pts, {n_params} params, "
              f"h=[{h_vals.min():.2f},{h_vals.max():.2f}] {status}{quality}")
else:
    print(f"\n[NPZ Training Data] NOT FOUND")

# Also check colab data
colab_dir = DATA / "vqe_colab"
if colab_dir.exists():
    colab_npz = sorted(colab_dir.glob(f"*_{_topology}_*.npz"))
    if colab_npz:
        print(f"\n[Colab VQE Data] {len(colab_npz)} files for {_topology}")
        for npz_file in colab_npz:
            npz_data = np.load(npz_file, allow_pickle=True)
            h_vals = npz_data["h_values"]
            n_pts = len(h_vals)
            print(f"  {npz_file.name}: {n_pts} pts, h=[{h_vals.min():.2f},{h_vals.max():.2f}]")
