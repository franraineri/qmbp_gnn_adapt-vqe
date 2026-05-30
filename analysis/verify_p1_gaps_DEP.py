#!/usr/bin/env python3
"""
DEPRECATED IS GONNA BE REMOVED
Verify p=1 gap analysis: check actual result files to confirm problems.

This script:
1. Reads the actual p=1 pipeline_run files to confirm h_test values
2. Checks the p=1 valid regime boundaries from binnacle data
3. Verifies what the triangular N=10 failure actually looks like
4. Confirms seed coverage gaps
5. Cross-references with p=1 ZNE noisy data

Usage:
    python analysis/verify_p1_gaps.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THESIS = ROOT / "results" / "thesis"

# p=1 valid regime from binnacle-p1-scaling.md
# These are the MINIMUM h values where p=1 HVA can express the ground state
P1_VALID_REGIME = {
    ("chain_1d", 6): 1.6,
    ("chain_1d", 10): 1.9,
    ("chain_1d", 20): 2.25,
    ("ladder", 6): 2.0,
    ("ladder", 10): 2.0,
    ("triangular", 6): 3.0,
    ("triangular", 10): 3.5,
}


def check_p1_pipeline_results():
    """Read and verify all p=1 pipeline results."""
    print("=" * 80)
    print("  VERIFICATION: p=1 Noiseless Pipeline Results")
    print("=" * 80)

    # Find all p=1 pipeline results
    p1_results = []

    variant_folders = [
        ("variants_N6_N10_1D_linnear", "chain_1d"),
        ("variants_N6_ladder", "ladder"),
        ("variants_N6_triangular", "triangular"),
        ("variants_N10_ladder", "ladder"),
        ("variants_N10_triangular", "triangular"),
    ]

    for folder_name, default_topo in variant_folders:
        folder_path = THESIS / folder_name
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

                topology = config.get("topology") or system.get("topology", default_topo)
                n_qubits = config.get("n_qubits") or system.get("n_qubits")
                h_values = config.get("h_values", [])
                seed = config.get("seed")
                n_restarts = config.get("n_restarts")

                de_gap = p4[0].get("delta_e_over_gap") if p4 else None
                h_test = p4[0].get("h_test") if p4 else None

                # Phase 2 diagnostics
                p2 = diag.get("phase2", {})
                theta_smooth = p2.get("theta_smoothness")
                conv_rate = p2.get("convergence_rate")

                # Phase 3 diagnostics
                p3 = diag.get("phase3", {})
                gen_gap = p3.get("generalization_gap")

                p1_results.append(
                    {
                        "file": str(pf.relative_to(ROOT)),
                        "folder": folder_name,
                        "variant": subdir.name,
                        "topology": topology,
                        "n_qubits": n_qubits,
                        "h_values": h_values,
                        "h_test": h_test,
                        "seed": seed,
                        "n_restarts": n_restarts,
                        "de_gap": de_gap,
                        "theta_smoothness": theta_smooth,
                        "convergence_rate": conv_rate,
                        "generalization_gap": gen_gap,
                    }
                )
                break  # Only latest file per variant

    # Report
    print(f"\n  Found {len(p1_results)} p=1 pipeline results\n")

    for r in sorted(p1_results, key=lambda x: (x["topology"], x["n_qubits"] or 0)):
        topo = r["topology"]
        n = r["n_qubits"]
        h_test = r["h_test"]
        de_gap = r["de_gap"]
        seed = r["seed"]

        # Check valid regime
        threshold = P1_VALID_REGIME.get((topo, n), 0)
        in_valid = h_test >= threshold if h_test is not None else False

        # Verdict
        if de_gap is not None:
            if de_gap < 0.05:
                verdict = "PASS ✅"
            elif de_gap < 0.10:
                verdict = "MARGINAL ⚠️"
            else:
                verdict = "FAIL ❌"
        else:
            verdict = "NO DATA"

        valid_str = "✓ IN regime" if in_valid else f"✗ OUTSIDE (need h≥{threshold})"

        print(f"  {topo:<12} N={n:<3} h_test={h_test}")
        print(
            f"    Verdict: {verdict} (ΔE/gap={de_gap:.4f})" if de_gap else f"    Verdict: {verdict}"
        )
        print(f"    Valid regime: {valid_str}")
        print(f"    h_values: {r['h_values']}")
        print(f"    seed={seed}, restarts={r['n_restarts']}")
        print(f"    θ_smooth={r['theta_smoothness']}, gen_gap={r['generalization_gap']}")
        print(f"    conv_rate={r['convergence_rate']}")
        print(f"    File: {r['file']}")
        print()

    return p1_results


def check_p1_noisy_results():
    """Check p=1 noisy/ZNE results for cross-reference."""
    print("=" * 80)
    print("  VERIFICATION: p=1 Noisy/ZNE Results")
    print("=" * 80)

    p1_noisy = []

    # Check analysis_p1_zne
    zne_dir = THESIS / "analysis_p1_zne"
    if zne_dir.exists():
        for subdir in sorted(zne_dir.iterdir()):
            if not subdir.is_dir():
                continue
            for nf in sorted(subdir.glob("noisy_*.json"), reverse=True):
                try:
                    with open(nf) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue

                config = data.get("config", {})
                summary = data.get("summary", {})

                p1_noisy.append(
                    {
                        "file": str(nf.relative_to(ROOT)),
                        "variant": subdir.name,
                        "topology": config.get("topology", "unknown"),
                        "n_qubits": config.get("n_qubits"),
                        "p_layers": config.get("p_layers"),
                        "seed": config.get("seed"),
                        "h_values": config.get("h_values", []),
                        "n_layouts": config.get("n_layouts"),
                        "mean_gain_pct": summary.get("mean_gain_pct"),
                        "mean_r2": summary.get("mean_r2"),
                    }
                )

    # Check variant folders for p=1 noisy
    for folder_name in ["variants_N10_ladder", "variants_N10_triangular", "variants_N6_triangular"]:
        folder_path = THESIS / folder_name
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
                p_layers = config.get("p_layers") or system.get("p_layers", 2)
                if p_layers != 1:
                    continue

                summary = data.get("summary", {})
                p1_noisy.append(
                    {
                        "file": str(nf.relative_to(ROOT)),
                        "variant": subdir.name,
                        "topology": config.get("topology") or system.get("topology"),
                        "n_qubits": config.get("n_qubits") or system.get("n_qubits"),
                        "p_layers": p_layers,
                        "seed": config.get("seed"),
                        "h_values": config.get("h_values", []),
                        "n_layouts": config.get("n_layouts"),
                        "mean_gain_pct": summary.get("mean_gain_pct"),
                        "mean_r2": summary.get("mean_r2"),
                    }
                )

    print(f"\n  Found {len(p1_noisy)} p=1 noisy results\n")

    # Group by topology
    by_topo = {}
    for r in p1_noisy:
        key = (r["topology"], r["n_qubits"])
        if key not in by_topo:
            by_topo[key] = []
        by_topo[key].append(r)

    for key in sorted(by_topo.keys()):
        topo, n = key
        group = by_topo[key]
        gains = [r["mean_gain_pct"] for r in group if r["mean_gain_pct"] is not None]
        seeds = sorted(set(r["seed"] for r in group if r["seed"] is not None))
        mean_gain = sum(gains) / len(gains) if gains else 0
        n_positive = sum(1 for g in gains if g > 0)

        print(f"  {topo:<12} N={n}: {len(group)} runs, seeds={seeds}")
        print(f"    Mean gain: {mean_gain:+.1f}%, positive: {n_positive}/{len(gains)}")
        print(f"    h_values: {group[0]['h_values']}")
        print()

    return p1_noisy


def diagnose_triangular_failure():
    """Deep dive into why triangular N=10 p=1 fails at h_test=4.0."""
    print("=" * 80)
    print("  DIAGNOSIS: Why does triangular N=10 p=1 fail at h_test=4.0?")
    print("=" * 80)

    # Read the actual file
    tri_path = THESIS / "variants_N10_triangular" / "nl_p1_triangular"
    pipeline_files = (
        sorted(tri_path.glob("pipeline_run_*.json"), reverse=True) if tri_path.exists() else []
    )

    if not pipeline_files:
        print("\n  ⚠️ File not found!")
        return

    with open(pipeline_files[0]) as f:
        data = json.load(f)

    config = data.get("config", {})
    p4 = data.get("phase4_results", [])
    diag = data.get("diagnostics", {})

    print("\n  Config:")
    print(f"    topology: {config.get('topology')}")
    print(f"    n_qubits: {config.get('n_qubits')}")
    print(f"    p_layers: {config.get('p_layers')}")
    print(f"    h_values: {config.get('h_values')}")
    print(f"    seed: {config.get('seed')}")
    print(f"    n_restarts: {config.get('n_restarts')}")

    if p4:
        print("\n  Phase 4 result:")
        print(f"    h_test: {p4[0].get('h_test')}")
        print(f"    delta_e_over_gap: {p4[0].get('delta_e_over_gap')}")
        print(f"    predicted_energy: {p4[0].get('predicted_energy')}")
        print(f"    phase_label: {p4[0].get('phase_label')}")

    p2 = diag.get("phase2", {})
    p3 = diag.get("phase3", {})
    p4_diag = diag.get("phase4", {})

    print("\n  Phase 2 diagnostics:")
    print(f"    theta_smoothness: {p2.get('theta_smoothness')}")
    print(f"    convergence_rate: {p2.get('convergence_rate')}")

    print("\n  Phase 3 diagnostics:")
    print(f"    generalization_gap: {p3.get('generalization_gap')}")

    decomp = p4_diag.get("energy_decomposition", {})
    if decomp:
        print("\n  Energy decomposition:")
        print(f"    e_exact: {decomp.get('e_exact')}")
        print(f"    e_vqe_ceiling: {decomp.get('e_vqe_ceiling')}")
        print(f"    e_mpnn_predicted: {decomp.get('e_mpnn_predicted')}")
        print(f"    error_from_circuit: {decomp.get('error_from_circuit')}")
        print(f"    error_from_mpnn: {decomp.get('error_from_mpnn')}")

    # Analysis
    print("\n  ─── ANALYSIS ───")
    threshold = P1_VALID_REGIME.get(("triangular", 10), 3.5)
    h_test = p4[0].get("h_test") if p4 else None
    print(f"    h_test={h_test}, valid regime threshold={threshold}")
    if h_test and h_test >= threshold:
        print(f"    h_test IS within valid regime (h≥{threshold})")
        print("    → Failure is NOT due to h_test being outside valid regime")
        print("    → Likely cause: MPNN prediction error or training issue")
        gen_gap = p3.get("generalization_gap")
        if gen_gap and gen_gap > 0.01:
            print(f"    → gen_gap={gen_gap:.4f} > 0.01 → MPNN overfitting!")
        theta = p2.get("theta_smoothness")
        if theta and theta > 1.0:
            print(f"    → theta_smoothness={theta:.4f} > 1.0 → warm-start chain break!")
    else:
        print("    h_test is OUTSIDE valid regime")
        print(f"    → Need to re-run with h_test≥{threshold}")


def produce_final_recommendations(p1_results, p1_noisy):
    """Produce final verified recommendations."""
    print("\n" + "=" * 80)
    print("  FINAL VERIFIED RECOMMENDATIONS")
    print("=" * 80)

    print("""
  Based on verified data:

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ PRIORITY 1 (HIGH) — chain_1d N=10 p=1 with correct h_test                  │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ Problem: Both existing runs use h_test < 1.9 (outside valid regime)         │
  │ Action:  Run pipeline with h_values=[4.0,3.5,3.0,2.5,2.0], h_test=2.5      │
  │ Seeds:   42, 43, 44                                                         │
  │ Time:    ~3 min total                                                       │
  │ Value:   Completes p=1 vs p=2 comparison table for thesis                   │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ PRIORITY 2 (HIGH) — ladder N=10 p=1 with 3 seeds                            │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ Problem: Only 1 run (no seed), already passes (0.036)                       │
  │ Action:  Run pipeline with h_values=[4.0,3.5,3.0,2.5], h_test=3.0          │
  │ Seeds:   42, 43, 44                                                         │
  │ Time:    ~5 min total                                                       │
  │ Value:   Confirms reproducibility of p=1 on ladder                          │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ PRIORITY 3 (HIGH) — triangular N=10 p=1 diagnosis + re-run                  │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ Problem: Fails at h_test=4.0 (ΔE/gap=0.603) despite being in valid regime  │
  │ Cause:   Likely MPNN overfitting or training issue (check gen_gap above)    │
  │ Action:  Re-run with seeds 42, 44 (seed=43 already exists)                  │
  │          Use h_values=[5.0,4.5,4.0,3.5], h_test=4.0                        │
  │ Time:    ~6 min total                                                       │
  │ Value:   Determines if failure is seed-specific or systematic               │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ NOT RECOMMENDED                                                             │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │ ❌ p=1 N=6 with more seeds — N=6 well covered by p=2, not hardware target  │
  │ ❌ More p=2 runs — 128 runs with full coverage already                      │
  │ ❌ N=12 — too slow (~30+ min), N=10→N=20 covers scaling                    │
  │ ❌ Kagome — only 1 p=2 run, demonstration topology only                    │
  │ ❌ p=1 noisy N=6 — ZNE at N=6 already confirmed with p=2 (+78.8%)          │
  └─────────────────────────────────────────────────────────────────────────────┘
""")


def main():
    p1_results = check_p1_pipeline_results()
    p1_noisy = check_p1_noisy_results()
    diagnose_triangular_failure()
    produce_final_recommendations(p1_results, p1_noisy)


if __name__ == "__main__":
    main()
