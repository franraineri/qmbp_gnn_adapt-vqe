#!/usr/bin/env python3
"""Analyze existing results and suggest complementary experiments.

Goes beyond gap-filling to identify novel experiments that would
strengthen the thesis narrative.
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THESIS = ROOT / "results" / "thesis"


def scan_all_p1_results():
    """Get all p=1 pipeline results with diagnostics."""
    records = []
    for folder in [
        "p1_variants_N10",
        "variants_N6_N10_1D_linnear",
        "variants_N6_ladder",
        "variants_N6_triangular",
        "variants_N10_ladder",
        "variants_N10_triangular",
    ]:
        folder_path = THESIS / folder
        if not folder_path.exists():
            continue
        for subdir in sorted(folder_path.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            for pf in sorted(subdir.glob("pipeline_run_*.json"), reverse=True):
                try:
                    with open(pf) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                config = data.get("config", {})
                system = data.get("system", {})
                p_layers = config.get("p_layers") or system.get("p_layers", 2)
                if p_layers != 1:
                    continue
                p4 = data.get("phase4_results", [])
                diag = data.get("diagnostics", {})
                records.append(
                    {
                        "variant": subdir.name,
                        "folder": folder,
                        "topology": config.get("topology") or system.get("topology"),
                        "n_qubits": config.get("n_qubits") or system.get("n_qubits"),
                        "h_values": config.get("h_values", []),
                        "h_test": p4[0].get("h_test") if p4 else None,
                        "de_gap": p4[0].get("delta_e_over_gap") if p4 else None,
                        "seed": config.get("seed"),
                        "n_restarts": config.get("n_restarts"),
                        "theta_smoothness": diag.get("phase2", {}).get("theta_smoothness"),
                        "gen_gap": diag.get("phase3", {}).get("generalization_gap"),
                    }
                )
                break
    return records


def scan_p2_coverage():
    """Get p=2 coverage summary."""
    by_config = defaultdict(list)
    for folder in [
        "variants_N6_N10_1D_linnear",
        "variants_N6_ladder",
        "variants_N6_triangular",
        "variants_N10_ladder",
        "variants_N10_triangular",
    ]:
        folder_path = THESIS / folder
        if not folder_path.exists():
            continue
        for subdir in sorted(folder_path.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            for pf in sorted(subdir.glob("pipeline_run_*.json"), reverse=True):
                try:
                    with open(pf) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                config = data.get("config", {})
                system = data.get("system", {})
                p_layers = config.get("p_layers") or system.get("p_layers", 2)
                if p_layers != 2:
                    continue
                p4 = data.get("phase4_results", [])
                topo = config.get("topology") or system.get("topology")
                n = config.get("n_qubits") or system.get("n_qubits")
                de = p4[0].get("delta_e_over_gap") if p4 else None
                by_config[(topo, n)].append(de)
                break
    return by_config


def scan_noisy_coverage():
    """Get noisy/ZNE coverage."""
    records = defaultdict(list)
    for folder in [
        "variants_N6_N10_1D_linnear",
        "variants_N6_ladder",
        "variants_N6_triangular",
        "variants_N10_ladder",
        "variants_N10_triangular",
        "n6_noisy",
        "analysis_p1_zne",
    ]:
        folder_path = THESIS / folder
        if not folder_path.exists():
            continue
        for subdir in sorted(folder_path.iterdir()):
            if not subdir.is_dir():
                continue
            for nf in sorted(subdir.glob("noisy_*.json"), reverse=True):
                try:
                    with open(nf) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
                config = data.get("config", {})
                system = data.get("system", {})
                summary = data.get("summary", {})
                p = config.get("p_layers") or system.get("p_layers", 2)
                topo = config.get("topology") or system.get("topology", "unknown")
                n = config.get("n_qubits") or system.get("n_qubits")
                records[(topo, n, p)].append(
                    {
                        "gain": summary.get("mean_gain_pct"),
                        "r2": summary.get("mean_r2"),
                        "seed": config.get("seed"),
                    }
                )
                break
    return records


def main():
    print("=" * 80)
    print("  COMPLEMENTARY EXPERIMENT SUGGESTIONS")
    print("  Based on current data inventory + thesis needs")
    print("=" * 80)

    p1_results = scan_all_p1_results()
    scan_p2_coverage()
    noisy_coverage = scan_noisy_coverage()

    # ─── Current p=1 state ───
    print("\n  ── Current p=1 N=10 Results ──\n")
    p1_n10 = [r for r in p1_results if r["n_qubits"] == 10]
    by_topo = defaultdict(list)
    for r in p1_n10:
        by_topo[r["topology"]].append(r)

    for topo in sorted(by_topo.keys()):
        group = by_topo[topo]
        passes = [r for r in group if r["de_gap"] is not None and r["de_gap"] < 0.05]
        fails = [r for r in group if r["de_gap"] is not None and r["de_gap"] >= 0.10]
        print(f"  {topo}: {len(group)} runs, {len(passes)} pass, {len(fails)} fail")
        for r in sorted(group, key=lambda x: x.get("de_gap") or 999):
            de = f"{r['de_gap']:.4f}" if r["de_gap"] is not None else "N/A"
            v = (
                "PASS"
                if r["de_gap"] and r["de_gap"] < 0.05
                else ("MARG" if r["de_gap"] and r["de_gap"] < 0.10 else "FAIL")
            )
            seed_s = str(r["seed"]) if r["seed"] is not None else "—"
            theta_s = f" θ={r['theta_smoothness']:.3f}" if r.get("theta_smoothness") else ""
            print(f"    seed={seed_s:<5} h_test={r['h_test']} ΔE/gap={de} [{v}]{theta_s}")

    # ─── What's NOT covered ───
    print("\n" + "=" * 80)
    print("  NOVEL EXPERIMENTS (not gap-filling, genuinely new)")
    print("=" * 80)

    suggestions = []

    # 1. p=1 vs p=2 DIRECT comparison at same h_test
    print("\n  [1] p=1 vs p=2 direct comparison at SAME h_test")
    print("      Status: p=2 uses h_test=1.5 or 2.5; p=1 uses 2.25/2.75/4.25")
    print("      Gap: No direct apple-to-apple comparison exists")
    print("      Value: Table showing p=1 vs p=2 at identical conditions")
    suggestions.append(
        {
            "id": "COMP-1",
            "type": "p=2 at p=1 h_test values",
            "description": "Run p=2 pipeline at h_test=2.75 (ladder) and 4.25 (triangular) for direct comparison",
            "topologies": ["ladder", "triangular"],
            "n_qubits": 10,
            "estimated_time": "~6 min (2 topos × 3 seeds)",
        }
    )

    # 2. p=1 with denser training grid (addresses chain_1d/ladder failures)
    print("\n  [2] p=1 with DENSER training grid")
    print("      Status: Current uses 4-5 training points")
    print("      Gap: Chain_1d and ladder failures may be MPNN underfitting")
    print("      Value: Determines if failure is physics or data-limited")
    suggestions.append(
        {
            "id": "COMP-2",
            "type": "p=1 dense grid",
            "description": "Run p=1 chain_1d N=10 with 9 training points [4.0,3.75,3.5,...,2.0]",
            "topologies": ["chain_1d"],
            "n_qubits": 10,
            "estimated_time": "~5 min (1 topo × 3 seeds)",
        }
    )

    # 3. p=1 N=10 noisy with select_layouts_low_ces
    print("\n  [3] p=1 ZNE with layout selection strategy")
    print("      Status: 12 p=1 noisy runs exist but use random layouts")
    print("      Gap: select_layouts_low_ces() implemented but not validated")
    print("      Value: Validates the hardware deployment strategy")
    # Check if already done
    has_layout_selection = any(
        "low_ces" in str(r) for records in noisy_coverage.values() for r in records
    )
    if not has_layout_selection:
        suggestions.append(
            {
                "id": "COMP-3",
                "type": "p=1 ZNE with layout selection",
                "description": "Run p=1 noisy N=10 chain_1d with select_layouts_low_ces",
                "topologies": ["chain_1d"],
                "n_qubits": 10,
                "estimated_time": "~3 min (1 topo × 3 seeds)",
            }
        )

    # 4. p=2 N=10 with SAME config as p=1 (for fair comparison)
    print("\n  [4] p=2 N=10 with identical training grid as p=1")
    print("      Status: p=2 runs use different h-grids than p=1")
    print("      Gap: Can't attribute differences to p alone (confounded by grid)")
    print("      Value: Isolates the effect of p on pipeline performance")
    suggestions.append(
        {
            "id": "COMP-4",
            "type": "p=2 matched config",
            "description": "Run p=2 triangular N=10 with h=[5.0,4.5,4.0,3.5], h_test=4.25, seeds 42-44",
            "topologies": ["triangular"],
            "n_qubits": 10,
            "estimated_time": "~8 min (1 topo × 3 seeds, p=2 slower)",
        }
    )

    # 5. Multi-h_test deployment (test at multiple unseen points)
    print("\n  [5] Multi-h_test deployment (robustness check)")
    print("      Status: All p=1 runs test at single h_test")
    print("      Gap: Don't know if MPNN generalizes across the valid regime")
    print("      Value: Shows pipeline works at multiple deployment points")
    suggestions.append(
        {
            "id": "COMP-5",
            "type": "p=1 multi-h_test",
            "description": "Run p=1 triangular N=10 with h_test=[3.75, 4.25, 4.75] (3 unseen points)",
            "topologies": ["triangular"],
            "n_qubits": 10,
            "estimated_time": "~3 min (1 topo × 1 seed × 3 h_test)",
        }
    )

    # ─── Final recommendations ───
    print("\n" + "=" * 80)
    print("  FINAL RECOMMENDATIONS FOR run_p1_pipeline_variants_r2.py")
    print("=" * 80)

    print("""
  Add to the R2 script (in addition to the 6 corrected runs):

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ COMP-4: p=2 triangular N=10 matched config (for direct comparison)          │
  │   h_values=[5.0, 4.5, 4.0, 3.5], h_test=4.25, seeds 42,43,44              │
  │   Purpose: Same conditions as p=1 → isolates effect of ansatz depth         │
  │   Time: ~8 min                                                              │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ COMP-5: p=1 triangular N=10 multi-h_test (robustness)                       │
  │   h_values=[5.0, 4.5, 4.0, 3.5], h_test=[3.75, 4.25, 4.75], seed=42       │
  │   Purpose: Validates MPNN generalizes across valid regime                   │
  │   Time: ~3 min                                                              │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ COMP-2: p=1 chain_1d N=10 dense grid (diagnosis)                            │
  │   h_values=[4.0, 3.75, 3.5, 3.25, 3.0, 2.75, 2.5, 2.25, 2.0]             │
  │   h_test=2.75, seeds 42,43,44                                              │
  │   Purpose: Determines if R1 failures were data-limited                      │
  │   Time: ~5 min                                                              │
  └─────────────────────────────────────────────────────────────────────────────┘

  Total additional time: ~16 min (9 extra runs)
  Combined with R2 corrections: ~24 min total (15 runs)
""")


if __name__ == "__main__":
    main()
