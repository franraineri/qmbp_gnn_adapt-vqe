#!/usr/bin/env python3
"""Verify all claims in the executive summary against raw data.

Checks each assertion for statistical robustness and flags any
that are weakly supported or potentially misleading.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "analysis" / "raw_data" / "all_variants.json"

with open(RAW) as f:
    data = json.load(f)

print("=" * 70)
print("VERIFICATION OF ANALYSIS CLAIMS")
print("=" * 70)

# ============================================================
# CLAIM 1: "71/120 noiseless pass (59%)"
# ============================================================
print("\n## CLAIM 1: Pass rate")
noiseless = [d for d in data if d["category"] == "noiseless" and d["delta_e_over_gap"] is not None]
passing = [d for d in noiseless if d["delta_e_over_gap"] < 0.05]
print(f"  Total noiseless with delta_e/gap: {len(noiseless)}")
print(f"  Passing (<0.05): {len(passing)}")
print(f"  Pass rate: {len(passing) / len(noiseless):.1%}")

# How many have no data?
unknown = [d for d in data if d["category"] == "noiseless" and d["delta_e_over_gap"] is None]
print(f"  Noiseless WITHOUT data: {len(unknown)}")
if unknown:
    print(f"  WARNING: {len(unknown)} noiseless variants have no data")
    print(f"  The 59% is based on {len(noiseless)} of {len(noiseless) + len(unknown)} total")

# ============================================================
# CLAIM 2: "Warm-start gain 93-99.9%"
# ============================================================
print("\n## CLAIM 2: Warm-start gain")
print("  Source: binnacle-comparative-analysis.md (Comparison 1)")
print("  This is from a SINGLE run (N=6, chain_1d, seed=42)")
print("  Not directly verified from variant data — relies on binnacle")
print("  But corroborated by: NL-A1 passing everywhere")

# ============================================================
# CLAIM 3: "Restart paradox" — more restarts hurts
# ============================================================
print("\n## CLAIM 3: Restart paradox")

by_topo_n = defaultdict(dict)
for d in noiseless:
    vid = d["variant_id"]
    key = (d["topology"], d["n_qubits"])
    if vid == "NL-A1":
        by_topo_n[key][1] = d["delta_e_over_gap"]
    elif vid == "NL-A3":
        by_topo_n[key][3] = d["delta_e_over_gap"]
    elif vid == "NL-A5":
        by_topo_n[key][5] = d["delta_e_over_gap"]
    elif vid == "NL-A7":
        by_topo_n[key][7] = d["delta_e_over_gap"]

print("  Restart series (delta_e/gap):")
paradox_count = 0
for key in sorted(by_topo_n.keys()):
    series = by_topo_n[key]
    if len(series) >= 3:
        items = sorted(series.items())
        line = f"    {key[0]} N={key[1]}: " + " -> ".join(f"{n}rst={v:.4f}" for n, v in items)
        vals = [v for _, v in items]
        # Non-monotonic: later value > 2x earlier
        if any(vals[i] > vals[i - 1] * 2 for i in range(1, len(vals))):
            line += " << NON-MONOTONIC"
            paradox_count += 1
        print(line)

print(f"\n  Paradox cases: {paradox_count}")
print("  CAVEAT: Each variant uses a SINGLE seed.")
print("  The paradox might be seed-specific, not systematic.")
print("  RECOMMENDATION: Qualify as 'observed, needs multi-seed verification'")

# ============================================================
# CLAIM 4: "ZNE fails at N=10 p=2, works at p=1"
# ============================================================
print("\n## CLAIM 4: ZNE at N=10")
noisy = [d for d in data if d["mean_r2"] is not None]
n10_noisy = [d for d in noisy if d["n_qubits"] == 10]

p2_results = [d for d in n10_noisy if "p1" not in d["variant_id"].lower()]
p1_results = [d for d in n10_noisy if "p1" in d["variant_id"].lower()]

if p2_results:
    gains = [d["mean_gain_pct"] for d in p2_results if d["mean_gain_pct"] is not None]
    n_negative = sum(1 for g in gains if g < 0)
    print(f"  p=2 N=10: {len(p2_results)} results")
    print(f"    Mean gain: {statistics.mean(gains):+.1f}%")
    print(f"    Negative gain: {n_negative}/{len(gains)}")
    positive = [d for d in p2_results if (d["mean_gain_pct"] or 0) > 0]
    if positive:
        print(f"    EXCEPTION: {len(positive)} with positive gain:")
        for p in positive:
            wins = p.get("n_mitigated_wins", "?")
            total = p.get("n_total", "?")
            print(f"      {p['variant_id']}: gain={p['mean_gain_pct']:+.1f}%, wins={wins}/{total}")

if p1_results:
    print(f"  p=1 N=10: {len(p1_results)} result(s)")
    for p in p1_results:
        wins = p.get("n_mitigated_wins", "?")
        total = p.get("n_total", "?")
        print(
            f"    {p['variant_id']}: R2={p['mean_r2']:.3f}, "
            f"gain={p['mean_gain_pct']:+.1f}%, "
            f"wins={wins}/{total}"
        )
    print("  CAVEAT: Single result. Need 3 seeds to confirm.")

# ============================================================
# CLAIM 5: "Hyperparameters irrelevant"
# ============================================================
print("\n## CLAIM 5: Hyperparameters irrelevant")

print("  Hidden dim at N=10:")
for topo in ["ladder", "triangular"]:
    h_results = {}
    for d in noiseless:
        if d["n_qubits"] != 10 or d["topology"] != topo:
            continue
        vid = d["variant_id"]
        if "B64" in vid:
            h_results[64] = d["delta_e_over_gap"]
        elif "B128" in vid:
            h_results[128] = d["delta_e_over_gap"]
        elif "B256" in vid:
            h_results[256] = d["delta_e_over_gap"]
    if len(h_results) >= 2:
        vals = list(h_results.values())
        spread = max(vals) - min(vals)
        mean_v = statistics.mean(vals)
        rel = spread / mean_v if mean_v > 0 else 0
        status = "CONFIRMED" if rel < 0.5 else "NOT irrelevant!"
        print(f"    {topo}: {h_results}")
        print(f"      spread={spread:.4f} ({rel:.0%} relative) -> {status}")

print("\n  Hidden dim at N=6:")
for topo in ["chain_1d", "ladder", "triangular"]:
    h_results = {}
    for d in noiseless:
        if d["n_qubits"] != 6 or d["topology"] != topo:
            continue
        vid = d["variant_id"]
        if "B64" in vid:
            h_results[64] = d["delta_e_over_gap"]
        elif "B128" in vid:
            h_results[128] = d["delta_e_over_gap"]
        elif "B256" in vid:
            h_results[256] = d["delta_e_over_gap"]
    if len(h_results) >= 2:
        vals = list(h_results.values())
        spread = max(vals) - min(vals)
        mean_v = statistics.mean(vals)
        rel = spread / mean_v if mean_v > 0 else 0
        status = "CONFIRMED" if rel < 0.5 else "NOT irrelevant!"
        print(f"    {topo}: {h_results}")
        print(f"      spread={spread:.4f} ({rel:.0%} relative) -> {status}")

# ============================================================
# CLAIM 6: "Reproducibility depends on topology"
# ============================================================
print("\n## CLAIM 6: Reproducibility")

seed_data = defaultdict(dict)
for d in noiseless:
    vid = d["variant_id"].lower()
    key = (d["topology"], d["n_qubits"])
    if "seed42" in vid or "e-seed42" in vid or "f-seed42" in vid:
        seed_data[key][42] = d["delta_e_over_gap"]
    elif "seed43" in vid or "e-seed43" in vid or "f-seed43" in vid:
        seed_data[key][43] = d["delta_e_over_gap"]
    elif "seed44" in vid or "e-seed44" in vid or "f-seed44" in vid:
        seed_data[key][44] = d["delta_e_over_gap"]

for key in sorted(seed_data.keys()):
    seeds = seed_data[key]
    if len(seeds) >= 2:
        vals = list(seeds.values())
        std = statistics.stdev(vals)
        mean_v = statistics.mean(vals)
        status = (
            "seed-independent"
            if std < 0.02
            else ("moderate variance" if std < 0.1 else "seed-DEPENDENT")
        )
        print(f"  {key[0]} N={key[1]}: {seeds}")
        print(f"    std={std:.4f}, mean={mean_v:.4f} -> {status}")

        # Check outlier-driven
        if std > 0.05 and len(vals) == 3:
            sorted_vals = sorted(vals)
            if sorted_vals[2] > 5 * sorted_vals[1]:
                without = statistics.stdev(sorted_vals[:2])
                print(
                    f"    OUTLIER: max={sorted_vals[2]:.4f} is "
                    f"{sorted_vals[2] / sorted_vals[1]:.0f}x next"
                )
                print(f"    Without outlier: std={without:.4f}")

# ============================================================
# CLAIM 7: Ladder N=6 data completeness
# ============================================================
print("\n## CLAIM 7: Ladder N=6 data completeness")
ladder_n6_all = [d for d in data if d["topology"] == "ladder" and d["n_qubits"] == 6]
ladder_n6_data = [d for d in ladder_n6_all if d["delta_e_over_gap"] is not None]
print(f"  Total ladder N=6 variants: {len(ladder_n6_all)}")
print(f"  With delta_e/gap data: {len(ladder_n6_data)}")
print(f"  Coverage: {len(ladder_n6_data) / len(ladder_n6_all):.0%}")
if len(ladder_n6_data) < len(ladder_n6_all) * 0.6:
    print("  WARNING: Less than 60% coverage — statistics unreliable")
    print("  RECOMMENDATION: Exclude ladder N=6 from main table or caveat")

# ============================================================
# OVERALL
# ============================================================
print("\n" + "=" * 70)
print("ROBUSTNESS SUMMARY")
print("=" * 70)
print("""
STRONG (keep as-is):
  - Framework works across topologies (large sample, multiple configs)
  - chain_1d seed-independent (std=0.004)
  - ZNE fails at N=10 p=2 (6/7 negative, consistent across topologies)
  - hidden_dim irrelevant at N=10 (<5% relative spread)
  - 7 grid points sufficient (standard passes in all topologies)
  - Implementation robust (98.8%, 186 variants)

NEEDS QUALIFICATION:
  - "Restart paradox" -> "observed in single-seed runs, needs verification"
  - p=1 ZNE -> "promising single result (R2=0.98, +73%), needs 3 seeds"
  - Triangular seed-dependent -> "confirmed, but N=10 driven by 1 outlier"
  - Hyperparameters -> "irrelevant at N=10; h=128 optimal at N=6"
  - Ladder N=6 -> "incomplete data (13/33), exclude from main comparison"

SHOULD CORRECT:
  - Pass rate denominator: note that 46 noiseless variants lack data
  - Warm-start gain: cite binnacle source, not variant data
  - "More restarts hurts" -> soften to "diminishing/negative returns observed"
""")
