#!/usr/bin/env python3
"""Deep statistical analysis of mitigation benchmark results.

Produces: per-config CIs, pairwise Mann-Whitney U tests, Cohen's d effect sizes,
h-value sensitivity, and validity checks.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))


def load_results():
    """Load all valid benchmark results."""
    results_dir = Path("results/mitigation_benchmark")
    by_config = defaultdict(list)

    for f in sorted(results_dir.rglob("h*_run_*.json")):
        try:
            r = json.loads(f.read_text())
            meta = r.get("benchmark_metadata", {})
            res = r.get("results", {})
            cid = meta.get("config_id", "")
            de = res.get("delta_e_gap")
            h = meta.get("h_value")
            e_mit = res.get("e_mitigated")
            e_raw = res.get("e_raw")
            e_exact = res.get("e_exact")
            zne_r2 = res.get("zne_r2")
            if de is not None and cid:
                by_config[cid].append(
                    {
                        "de": float(de),
                        "h": h,
                        "e_mit": e_mit,
                        "e_raw": e_raw,
                        "e_exact": e_exact,
                        "zne_r2": zne_r2,
                    }
                )
        except Exception:
            continue

    return by_config


def main():
    by_config = load_results()
    total = sum(len(v) for v in by_config.values())
    print(f"Loaded {total} points across {len(by_config)} configs\n")

    # === 1. Per-config statistics with 95% CI ===
    print("=" * 72)
    print("  PER-CONFIG STATISTICS (95% CI)")
    print("=" * 72)
    header = f"{'Config':<26} {'Mean%':>7} {'Std%':>6} {'CI_lo%':>7} {'CI_hi%':>7} {'N':>4}"
    print(header)
    print("-" * 62)

    ranked = sorted(by_config.items(), key=lambda x: np.mean([d["de"] for d in x[1]]))
    for cid, points in ranked:
        vals = np.array([d["de"] for d in points])
        m = np.mean(vals)
        s = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        n = len(vals)
        se = s / np.sqrt(n) if n > 1 else 0.0
        ci_lo = m - 1.96 * se
        ci_hi = m + 1.96 * se
        print(
            f"{cid:<26} {m * 100:>7.2f} {s * 100:>6.2f} {ci_lo * 100:>7.2f} {ci_hi * 100:>7.2f} {n:>4}"
        )

    # === 2. Pairwise comparisons ===
    print("\n" + "=" * 72)
    print("  PAIRWISE COMPARISONS (Mann-Whitney U, two-sided)")
    print("=" * 72)

    pairs = [
        ("C0_raw", "C3_full_gf", "H3: GF > raw?"),
        ("C3_full_gf", "C5_full_pea_balanced", "H4: PEA > GF?"),
        ("C4_full_pea_light", "C6_full_pea_heavy", "H5: More PEA budget helps?"),
        ("C5_full_pea_balanced", "C6_full_pea_heavy", "H14: PEA saturates?"),
        ("C0_raw", "C11_mitiq_zne", "H10: Mitiq vs raw"),
        ("C3_full_gf", "C11_mitiq_zne", "H10b: Mitiq vs GF"),
        ("C0_raw", "C18_aqc_raw", "H17: AQC helps raw?"),
        ("C5_full_pea_balanced", "C15_pea_no_affine", "H8: Affine helps PEA?"),
    ]

    for cid_a, cid_b, desc in pairs:
        if cid_a not in by_config or cid_b not in by_config:
            print(f"  {desc}: SKIP (missing data)")
            continue
        vals_a = np.array([d["de"] for d in by_config[cid_a]])
        vals_b = np.array([d["de"] for d in by_config[cid_b]])
        u_stat, p_val = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        mean_a, mean_b = np.mean(vals_a), np.mean(vals_b)
        delta = mean_b - mean_a
        direction = "B better" if delta < 0 else "A better" if delta > 0 else "equal"
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {desc}")
        print(f"    A={cid_a}: {mean_a * 100:.2f}%   B={cid_b}: {mean_b * 100:.2f}%")
        print(f"    Delta={delta * 100:+.2f}%, U={u_stat:.0f}, p={p_val:.4f} {sig} ({direction})")
        print()

    # === 3. Effect sizes ===
    print("=" * 72)
    print("  EFFECT SIZES (Cohen's d)")
    print("=" * 72)
    for cid_a, cid_b, desc in pairs:
        if cid_a not in by_config or cid_b not in by_config:
            continue
        vals_a = np.array([d["de"] for d in by_config[cid_a]])
        vals_b = np.array([d["de"] for d in by_config[cid_b]])
        pooled_std = np.sqrt((np.var(vals_a, ddof=1) + np.var(vals_b, ddof=1)) / 2)
        d = (np.mean(vals_a) - np.mean(vals_b)) / pooled_std if pooled_std > 0 else 0.0
        mag = (
            "LARGE"
            if abs(d) > 0.8
            else "medium"
            if abs(d) > 0.5
            else "small"
            if abs(d) > 0.2
            else "negligible"
        )
        print(f"  {cid_a} vs {cid_b}: d={d:+.3f} ({mag})")

    # === 4. h-value sensitivity ===
    print("\n" + "=" * 72)
    print("  H-VALUE SENSITIVITY (ΔE/gap by h for top configs)")
    print("=" * 72)
    for cid in ["C6_full_pea_heavy", "C5_full_pea_balanced", "C3_full_gf", "C0_raw"]:
        if cid not in by_config:
            continue
        by_h = defaultdict(list)
        for d in by_config[cid]:
            if d["h"] is not None:
                by_h[d["h"]].append(d["de"])
        print(f"  {cid}:")
        for h in sorted(by_h):
            vals = by_h[h]
            print(f"    h={h:.2f}: ΔE/gap={np.mean(vals) * 100:.2f}% (n={len(vals)})")
        print()

    # === 5. Validity checks ===
    print("=" * 72)
    print("  VALIDITY CHECKS")
    print("=" * 72)

    # Check: e_mitigated should be closer to e_exact than e_raw for ZNE configs
    zne_configs = ["C3_full_gf", "C5_full_pea_balanced", "C6_full_pea_heavy"]
    improvement_count = 0
    total_zne = 0
    for cid in zne_configs:
        if cid not in by_config:
            continue
        for d in by_config[cid]:
            if d["e_mit"] is not None and d["e_raw"] is not None and d["e_exact"] is not None:
                err_raw = abs(d["e_raw"] - d["e_exact"])
                err_mit = abs(d["e_mit"] - d["e_exact"])
                total_zne += 1
                if err_mit < err_raw:
                    improvement_count += 1

    if total_zne > 0:
        print(
            f"  ZNE improves over raw: {improvement_count}/{total_zne} "
            f"({improvement_count / total_zne * 100:.0f}%)"
        )
    else:
        print("  ZNE improvement check: no data")

    # Check: R² for ZNE configs
    r2_vals = []
    for cid in zne_configs:
        if cid not in by_config:
            continue
        for d in by_config[cid]:
            if d.get("zne_r2") is not None:
                r2_vals.append(d["zne_r2"])
    if r2_vals:
        print(
            f"  ZNE R² stats: mean={np.mean(r2_vals):.4f}, "
            f"min={np.min(r2_vals):.4f}, >0.9 rate={sum(1 for r in r2_vals if r > 0.9) / len(r2_vals) * 100:.0f}%"
        )

    # Check: monotonicity (more mitigation = lower error)
    means = {cid: np.mean([d["de"] for d in points]) for cid, points in by_config.items()}
    expected_order = ["C6_full_pea_heavy", "C5_full_pea_balanced", "C3_full_gf", "C0_raw"]
    available = [c for c in expected_order if c in means]
    if len(available) >= 3:
        monotonic = all(
            means[available[i]] <= means[available[i + 1]] for i in range(len(available) - 1)
        )
        print(
            f"  Monotonicity (PEA_heavy < PEA_balanced < GF < raw): {'✅ YES' if monotonic else '❌ NO'}"
        )
        for c in available:
            print(f"    {c}: {means[c] * 100:.2f}%")

    print("\n" + "=" * 72)
    print("  ANALYSIS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
