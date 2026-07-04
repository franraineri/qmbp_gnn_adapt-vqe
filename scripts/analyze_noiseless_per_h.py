#!/usr/bin/env python3
"""Per-h-point deep analysis of noiseless pipeline results.

Extracts VQE (section_2) and Deploy (section_4) data at specific h-values
from run_*.json files. Provides the detailed per-point view that the
standard noiseless_pipeline_analyzer summary lacks.

Usage:
    python scripts/analyze_noiseless_per_h.py <result_json> [--h 1.0 2.0 3.0]
    python scripts/analyze_noiseless_per_h.py results/experiments/exp_noiseless_tfim_longitudinal_v2/
    python scripts/analyze_noiseless_per_h.py <result_json> --all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_run(path: Path) -> dict:
    """Load a single run JSON."""
    return json.loads(path.read_text())


def find_closest(values: list[float], target: float, tol: float = 0.15):
    """Find index of closest value within tolerance."""
    best_idx, best_diff = None, 999.0
    for i, v in enumerate(values):
        d = abs(v - target)
        if d < best_diff:
            best_idx, best_diff = i, d
    return best_idx if best_diff <= tol else None


def analyze_run(data: dict, h_targets: list[float] | None = None) -> None:
    """Print detailed per-h analysis for a single run."""
    cfg = data["config"]
    sys_cfg = cfg["system"]
    results = data["results"]

    print("=" * 95)
    print(
        f"  N={sys_cfg['n_qubits']} p={sys_cfg['p_layers']} "
        f"topo={sys_cfg['topologies'][0]} model={sys_cfg['model']} "
        f"h=[{cfg['h_grid']['h_min']}, {cfg['h_grid']['h_max']}] "
        f"pts={cfg['h_grid']['h_points']}"
    )
    print(f"  Elapsed: {data.get('elapsed_s', 0):.1f}s")
    print("=" * 95)

    # ── Section 2: VQE summary ──
    s2 = results.get("section_2", {})
    s2_data = s2.get("data", {})
    topo = sys_cfg["topologies"][0]
    topo_s2 = s2_data.get("topologies", {}).get(topo, {})

    print(f"\n  Section 2 (VQE): {'✅ PASS' if s2.get('success') else '❌ FAIL'}")
    if topo_s2:
        print(f"    pass_rate: {topo_s2.get('n_pass_5pct')}/{topo_s2.get('n_points')}")
        print(
            f"    mean_F={topo_s2.get('mean_fidelity', 0):.5f}  "
            f"min_F={topo_s2.get('min_fidelity', 0):.4f}"
        )
        print(
            f"    mean_ΔE/gap={topo_s2.get('mean_de_gap', 0):.4e}  "
            f"max_ΔE/gap={topo_s2.get('max_de_gap', 0):.4e}"
        )
        print(
            f"    θ_smooth_max={topo_s2.get('theta_smoothness_max', 0):.4f}  "
            f"θ_smooth_mean={topo_s2.get('theta_smoothness_mean', 0):.4f}"
        )
        print(f"    mean_entropy={topo_s2.get('mean_entanglement_entropy', 0):.4f}")

    # ── Section 3: MPNN ──
    s3 = results.get("section_3", {})
    s3_data = s3.get("data", {})
    print(f"\n  Section 3 (MPNN): {'✅ PASS' if s3.get('success') else '❌ FAIL'}")
    if s3_data:
        print(
            f"    final_mse={s3_data.get('final_mse', '?'):.6e}  "
            f"final_de_gap={s3_data.get('final_de_gap', '?'):.4e}"
        )
        print(
            f"    n_params={s3_data.get('n_output_params')}  "
            f"n_training={s3_data.get('n_training_points')}"
        )
        per_h_mse = s3_data.get("per_h_mse", [])
        if per_h_mse:
            arr = np.array(per_h_mse)
            print(f"    per_h_mse: mean={arr.mean():.4e} max={arr.max():.4e} min={arr.min():.4e}")

    # ── Section 4: Deploy per-point ──
    s4 = results.get("section_4", {})
    s4_data = s4.get("data", {})
    pp = s4_data.get("per_point", [])

    print(f"\n  Section 4 (Deploy): {'✅ PASS' if s4.get('success') else '❌ FAIL'}")
    if s4_data:
        print(
            f"    n_test={s4_data.get('n_test_points')}  "
            f"pass_energy={s4_data.get('n_pass_energy')}  "
            f"correct_labels={s4_data.get('n_correct_label')}"
        )
        print(
            f"    mean_ΔE/gap={s4_data.get('mean_de_gap', 0):.4e}  "
            f"max_ΔE/gap={s4_data.get('max_de_gap', 0):.4e}"
        )
        print(f"    mean_F={s4_data.get('mean_fidelity', 0):.5f}")
        sf = s4_data.get("speedup_factor")
        if sf:
            print(
                f"    speedup_factor={sf:.1f}x  "
                f"mpnn_wins_vs_random={s4_data.get('mpnn_wins_vs_random')}"
            )

    if not pp:
        print("    (No per-point data)")
        return

    # ── Per-point table ──
    show_all = h_targets is None
    if show_all:
        points_to_show = sorted(pp, key=lambda x: x["h_test"])
    else:
        points_to_show = []
        for ht in sorted(h_targets):
            best = min(pp, key=lambda p: abs(p["h_test"] - ht))
            if abs(best["h_test"] - ht) <= 0.15:
                points_to_show.append(best)

    print(f"\n  {'─' * 90}")
    print(
        f"  {'h':>5} {'ΔE/gap':>10} {'F':>8} {'S_ent':>6} "
        f"{'E_pred':>12} {'E_exact':>12} {'⟨X⟩_err':>8} {'⟨ZZ⟩_err':>8} {'Lbl':>4}"
    )
    print(f"  {'─' * 90}")

    for p in points_to_show:
        h = p["h_test"]
        de = p["de_gap"]
        f = p.get("fidelity", 0)
        s = p.get("entanglement_entropy", 0)
        ep = p["e_pred"]
        ee = p["e_exact"]
        mx = p.get("mag_x_error", 0)
        zz = p.get("corr_zz_error", 0)
        lbl = "✓" if p.get("correct_label") else "✗"
        marker = " ⚠️" if de > 0.05 else ""
        print(
            f"  {h:>5.2f} {de:>10.4e} {f:>8.4f} {s:>6.4f} "
            f"{ep:>12.4f} {ee:>12.4f} {mx:>8.4e} {zz:>8.4e} {lbl:>4}{marker}"
        )

    # ── Statistics ──
    de_gaps = [p["de_gap"] for p in pp]
    fids = [p.get("fidelity", 0) for p in pp]
    n_pass = sum(1 for d in de_gaps if d < 0.05)
    n_labels = sum(1 for p in pp if p.get("correct_label"))

    print(f"\n  Statistics ({len(pp)} deploy points):")
    print(f"    Pass ΔE/gap<5%: {n_pass}/{len(pp)} ({100 * n_pass / len(pp):.0f}%)")
    print(
        f"    ΔE/gap: mean={np.mean(de_gaps):.4e} median={np.median(de_gaps):.4e} "
        f"max={np.max(de_gaps):.4e}"
    )
    print(f"    Fidelity: mean={np.mean(fids):.5f} min={np.min(fids):.5f}")
    print(f"    Labels: {n_labels}/{len(pp)}")

    # ── Failures ──
    failures = [(p["h_test"], p["de_gap"], p.get("fidelity", 0)) for p in pp if p["de_gap"] > 0.05]
    if failures:
        print(f"\n  Failures (ΔE/gap > 5%): {len(failures)} points")
        for h, de, f in sorted(failures):
            print(f"    h={h:.3f}: ΔE/gap={de:.4e}, F={f:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-h deep analysis of noiseless pipeline results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a run_*.json file OR a directory containing them",
    )
    parser.add_argument(
        "--h",
        "-H",
        type=float,
        nargs="+",
        default=None,
        help="Specific h-values to show (default: show all deploy points)",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Show all per-point data (same as omitting --h)",
    )
    args = parser.parse_args()

    target = Path(args.path)
    h_targets = args.h if not args.all else None

    if target.is_file():
        files = [target]
    elif target.is_dir():
        files = sorted(target.glob("run_*.json"))
        if not files:
            print(f"No run_*.json files found in {target}")
            return 1
    else:
        print(f"Path not found: {target}")
        return 1

    for f in files:
        try:
            data = load_run(f)
            print(f"\n  File: {f.name}")
            analyze_run(data, h_targets)
            print()
        except Exception as e:
            print(f"  ERROR processing {f.name}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
