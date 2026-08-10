#!/usr/bin/env python3
"""Analyze ladder training data: where does VQE converge?"""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Load latest ladder result
results_dir = ROOT / "results" / "experiments" / "exp_accel_cross_n"
ladder_runs = [f for f in results_dir.glob("run_*.json")
               if "ladder" in f.read_text()[:500]]
latest = sorted(ladder_runs)[-1]

with open(latest) as f:
    data = json.load(f)

print(f"File: {latest.name}")
print(f"Config: {data.get('config', {})}")
print()

# Find per-point training results from section 2
for key, section in data.get("results", {}).items():
    if not isinstance(section, dict):
        continue
    sd = section.get("data", {})
    # Check section 3 cross-N self-eval (train-n == target-n)
    cross_n = sd.get("cross_n_results", {})
    for ck, result in cross_n.items():
        if "per_point" not in result:
            continue
        per_point = result["per_point"]
        print(f"Section: {key}, Config: {ck}")
        print(f"{'h':>6} {'ΔE/gap':>8} {'|ΔE|':>10} {'Fidelity':>9} {'Pass@5%':>8}")
        print("-" * 50)
        for r in per_point:
            h = r["h"]
            de_gap = r["de_gap"]
            abs_err = r["abs_error"]
            fid = r.get("fidelity", 0) or 0
            status = "✓" if de_gap < 0.05 else ("~" if de_gap < 0.10 else "✗")
            print(f"{h:>6.2f} {de_gap:>8.4f} {abs_err:>10.2e} {fid:>9.4f} {status:>8}")

        # Summary by h-region
        easy = [r for r in per_point if r["h"] >= 2.7]
        mid = [r for r in per_point if 2.4 <= r["h"] < 2.7]
        hard = [r for r in per_point if r["h"] < 2.4]
        print()
        print(f"h >= 2.7: {sum(1 for r in easy if r['de_gap']<0.05)}/{len(easy)} pass")
        print(f"2.4<=h<2.7: {sum(1 for r in mid if r['de_gap']<0.05)}/{len(mid)} pass")
        print(f"h < 2.4: {sum(1 for r in hard if r['de_gap']<0.05)}/{len(hard)} pass")
        print()
        if easy:
            print(f"→ Safe zone h>={min(r['h'] for r in easy if r['de_gap']<0.05):.2f}")
