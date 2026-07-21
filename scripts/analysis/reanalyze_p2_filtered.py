"""Re-analyze p=1-4 with h >= 1.3 filter (same regime as definitive runs)."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

results = defaultdict(list)
base = Path("results/experiments")

for exp_dir in base.glob("exp_noiseless*"):
    for run_file in exp_dir.rglob("run_*.json"):
        fp = str(run_file)
        try:
            data = json.load(open(run_file))
        except:
            continue
        cfg = data.get("config", {})
        sys_c = cfg.get("system", {})
        model = sys_c.get("model", cfg.get("model", ""))
        n = sys_c.get("n_qubits", cfg.get("n_qubits"))
        p = sys_c.get("p_layers", cfg.get("p_layers"))
        topos = sys_c.get("topologies", [])
        topo = topos[0] if isinstance(topos, list) and len(topos) == 1 else ""
        if not topo:
            for t in ["chain_1d", "heavy_hex", "ladder", "square", "triangular"]:
                if t in fp:
                    topo = t
                    break
        if not model:
            if "tfim_longitudinal" in fp:
                model = "tfim_longitudinal"
            elif "tfim" in fp:
                model = "tfim"
        if model != "tfim" or n != 10 or p not in [1, 2, 3, 4] or not topo:
            continue

        res = data.get("results", {})
        points = None
        for sk in ["section_4", "section_5", "section_6"]:
            sec = res.get(sk, {})
            if not isinstance(sec, dict):
                continue
            if "deploy" not in sec.get("name", "").lower():
                continue
            sd = sec.get("data", {})
            points = sd.get("per_point", [])
            if not points:
                for td in sd.get("topologies", {}).values():
                    points = td.get("per_point", [])
                    if points:
                        break
            break

        if not points or not isinstance(points[0], dict) or "e_exact" not in points[0]:
            continue

        # Filter h >= 1.3
        filtered = []
        for pt in points:
            h = pt.get("h_test", pt.get("h", 0))
            if h < 1.3:
                continue
            e_pred, e_exact = pt.get("e_pred"), pt.get("e_exact")
            if e_pred is None or e_exact is None:
                continue
            filtered.append(
                {
                    "de": abs(e_pred - e_exact),
                    "de_gap": pt.get("de_gap"),
                    "fid": pt.get("fidelity"),
                }
            )

        if not filtered:
            continue

        des = [pt["de"] for pt in filtered]
        degs = [pt["de_gap"] for pt in filtered if pt["de_gap"] is not None]
        fids = [pt["fid"] for pt in filtered if pt["fid"] is not None]

        results[(topo, p)].append(
            {
                "de_mean": np.mean(des),
                "de_median": np.median(des),
                "deg_mean": np.mean(degs) if degs else None,
                "pass_rate": sum(1 for d in degs if d < 0.05) / len(degs) if degs else 0,
                "fid_mean": np.mean(fids) if fids else None,
                "n_pts": len(filtered),
            }
        )

# Print
topos = ["chain_1d", "heavy_hex", "ladder", "square", "triangular"]
print("\nPassRate (h >= 1.3) — comparable across all p")
print("=" * 65)
print(f"{'Topología':<12} {'p=1':>8} {'p=2':>8} {'p=3':>8} {'p=4':>8}")
print("-" * 65)
for topo in topos:
    row = []
    for p in [1, 2, 3, 4]:
        key = (topo, p)
        if key in results:
            best = max(results[key], key=lambda r: r["pass_rate"])
            row.append(f"{best['pass_rate'] * 100:.0f}%")
        else:
            row.append("—")
    print(f"{topo:<12} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8}")

print("\n\nΔE/gap mean (h >= 1.3)")
print("=" * 65)
print(f"{'Topología':<12} {'p=1':>8} {'p=2':>8} {'p=3':>8} {'p=4':>8}")
print("-" * 65)
for topo in topos:
    row = []
    for p in [1, 2, 3, 4]:
        key = (topo, p)
        if key in results:
            best = max(results[key], key=lambda r: r["pass_rate"])
            v = best["deg_mean"]
            row.append(f"{v:.4f}" if v else "N/A")
        else:
            row.append("—")
    print(f"{topo:<12} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8}")

print("\n\nΔE absoluto mean (h >= 1.3)")
print("=" * 65)
print(f"{'Topología':<12} {'p=1':>8} {'p=2':>8} {'p=3':>8} {'p=4':>8}")
print("-" * 65)
for topo in topos:
    row = []
    for p in [1, 2, 3, 4]:
        key = (topo, p)
        if key in results:
            best = max(results[key], key=lambda r: r["pass_rate"])
            row.append(f"{best['de_mean']:.5f}")
        else:
            row.append("—")
    print(f"{topo:<12} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8}")

print("\n\nFidelidad mean (h >= 1.3)")
print("=" * 65)
print(f"{'Topología':<12} {'p=1':>8} {'p=2':>8} {'p=3':>8} {'p=4':>8}")
print("-" * 65)
for topo in topos:
    row = []
    for p in [1, 2, 3, 4]:
        key = (topo, p)
        if key in results:
            best = max(results[key], key=lambda r: r["pass_rate"])
            v = best["fid_mean"]
            row.append(f"{v:.4f}" if v else "N/A")
        else:
            row.append("—")
    print(f"{topo:<12} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8}")
