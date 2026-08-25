#!/usr/bin/env python3
"""Verify MT vs ST comparison fairness and coherence."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

COMP_DIR = ROOT / "results" / "model_comparison"


def main():
    from qmbp_simulation.analysis.evaluation_report import generate_mt_vs_st_table

    # Generate table from library
    lines, summary = generate_mt_vs_st_table(COMP_DIR)

    print("=" * 80)
    print("MT vs ST VERIFICATION")
    print("=" * 80)
    print(f"\nTotal scenarios: {summary['total']}")
    print(f"MT wins: {summary['mt_wins']}, ST wins: {summary['st_wins']}")
    print()

    # Audit: check for biases
    print("BIAS CHECK:")
    print("-" * 80)

    # 1. Are we comparing the SAME models across scenarios?
    mt_models = set()
    st_models = set()
    for s in summary["per_scenario"]:
        mt_models.add(s["mt_model"])
        st_models.add(s["st_model"])

    print(f"\n  Unique MT models used: {len(mt_models)}")
    for m in sorted(mt_models):
        print(f"    - {m}")
    print(f"\n  Unique ST models used: {len(st_models)}")
    for m in sorted(st_models):
        print(f"    - {m}")

    # 2. Check if MT model is the SAME across all topos (unfair advantage?)
    print("\n  Fairness issue: MT uses ONE model for all topologies,")
    print("  while ST uses a SPECIALIZED model per topology.")
    print("  This is the CORRECT comparison for thesis (tests generalization).")

    # 3. Check if some scenarios are duplicated (inflating count)
    seen = set()
    duplicates = 0
    for s in summary["per_scenario"]:
        key = (s["topology"], s["n_qubits"])
        if key in seen:
            duplicates += 1
            print(f"\n  ⚠️ DUPLICATE scenario: {s['topology']} {s['target_n']}")
        seen.add(key)
    if duplicates:
        print(f"\n  ⚠️ {duplicates} DUPLICATE SCENARIOS detected!")
        print("     These inflate the count. De-duplicate by using only latest comparison.")
    else:
        print("\n  ✅ No duplicate scenarios")

    # 4. N-range bias: are MT and ST evaluated at the same N?
    print("\n\n  Per-scenario details:")
    print(f"  {'Topo':12} {'N':>5} {'MT dE/gap':>10} {'ST dE/gap':>10} {'Win':>5} {'Fair?':>6}")
    print("  " + "-" * 72)
    for s in summary["per_scenario"]:
        fair = "✅" if s["mt_mean_de_gap"] < 2.0 and s["st_mean_de_gap"] < 2.0 else "⚠️"
        win = "MT" if s["winner"] == "MT" else "ST"
        print(
            f"  {s['topology']:12} {s['n_qubits']:>5} {s['mt_mean_de_gap']:>10.4f} {s['st_mean_de_gap']:>10.4f} {win:>5} {fair:>6}"
        )

    # 5. Margin analysis
    print("\n\nMARGIN ANALYSIS:")
    print("-" * 80)
    mt_margins = []
    st_margins = []
    for s in summary["per_scenario"]:
        if s["winner"] == "MT":
            ratio = s["st_mean_de_gap"] / max(s["mt_mean_de_gap"], 1e-6)
            mt_margins.append(ratio)
        else:
            ratio = s["mt_mean_de_gap"] / max(s["st_mean_de_gap"], 1e-6)
            st_margins.append(ratio)

    if mt_margins:
        avg_mt = sum(mt_margins) / len(mt_margins)
        print(f"  When MT wins: avg ST/MT ratio = {avg_mt:.1f}x (MT is {avg_mt:.1f}x better)")
        marginal_mt = sum(1 for m in mt_margins if m < 1.1)
        print(f"  Marginal wins (<10% better): {marginal_mt}/{len(mt_margins)}")
    if st_margins:
        avg_st = sum(st_margins) / len(st_margins)
        print(f"  When ST wins: avg MT/ST ratio = {avg_st:.1f}x (ST is {avg_st:.1f}x better)")
        marginal_st = sum(1 for m in st_margins if m < 1.1)
        print(f"  Marginal wins (<10% better): {marginal_st}/{len(st_margins)}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
