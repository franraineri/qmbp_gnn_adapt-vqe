"""Analyze scaling results from cross-N transfer and multi-seed runs."""

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")


def analyze_cross_n():
    """Analyze cross-N transfer results."""
    path = Path("results/scaling/cross_n_N10_to_N40_20260607_230926.json")
    if not path.exists():
        print("Cross-N result not found")
        return

    with open(path) as f:
        data = json.load(f)

    print("=" * 60)
    print("CROSS-N TRANSFER: N=10 → N=40")
    print("=" * 60)
    s = data["summary"]
    print(f"Warm advantage: {s['warm_better_count']}/{s['total_h_points']}")
    print(f"Rate: {s['warm_advantage_rate']:.1%}")
    print()
    for w, c in zip(data["warm_results"], data["cold_results"], strict=False):
        winner = "WARM" if w["de_gap"] < c["de_gap"] else "COLD"
        print(f"  h={w['h']:.3f}: warm={w['de_gap']:.4f} cold={c['de_gap']:.4f} → {winner}")


def analyze_multi_seed():
    """Analyze multi-seed N=40 results."""
    path = Path("results/scaling/scaling_N40_aer_mps_20260608_001053.json")
    if not path.exists():
        print("Multi-seed result not found")
        return

    with open(path) as f:
        data = json.load(f)

    print()
    print("=" * 60)
    print(f"MULTI-SEED N=40 (version {data.get('version', '1.0')})")
    print("=" * 60)
    sm = data["summary"]
    print(f"Pass: {sm['n_pass']}/{sm['n_total']}, all_passed={sm['all_passed']}")
    if "mean_de_gap" in sm:
        print(f"Mean ΔE/gap: {sm['mean_de_gap']:.4f}")
        print(f"Max: {sm['max_de_gap']:.4f}, Std: {sm['std_de_gap']:.4f}")

    print()
    for sr in data["vqe_results"]:
        seed = sr["seed"]
        res = sr["results"]
        de_gaps = [r["de_gap"] for r in res]
        n_pass = sum(1 for r in res if r["passed"])
        has_theta = "theta_opt" in res[0] if res else False
        print(
            f"  Seed {seed}: {n_pass}/{len(res)} pass, "
            f"mean={sum(de_gaps) / len(de_gaps):.4f}, max={max(de_gaps):.4f}, "
            f"theta={has_theta}"
        )

    # Check failures
    failed = [
        (sr["seed"], r["h"], r["de_gap"])
        for sr in data["vqe_results"]
        for r in sr["results"]
        if not r["passed"]
    ]
    if failed:
        print(f"\n  ⚠ FAILURES ({len(failed)}):")
        for s, h, d in failed:
            print(f"    seed={s}, h={h:.3f}: ΔE/gap={d:.4f}")

    # Check smoothness
    has_smooth = "theta_smoothness" in data["vqe_results"][0]["results"][1]
    if has_smooth:
        print("\n  Theta smoothness:")
        for sr in data["vqe_results"]:
            vals = [
                r.get("theta_smoothness", 0)
                for r in sr["results"]
                if r.get("theta_smoothness", 0) > 0
            ]
            if vals:
                print(
                    f"    Seed {sr['seed']}: max={max(vals):.4f} mean={sum(vals) / len(vals):.4f}"
                )


if __name__ == "__main__":
    analyze_cross_n()
    analyze_multi_seed()
