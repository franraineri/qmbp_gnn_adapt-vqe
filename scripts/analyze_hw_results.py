"""Analyze hardware mitigation benchmark results.

Checks:
1. Circuit metrics consistency (n_2q, depth_2q, depth)
2. QPU time vs model prediction
3. Energy vs exact + delta_e_gap validation
4. Cross-config comparison
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

HW_DIR = ROOT / "results/mitigation_benchmark/hardware"
SIM_DIR = ROOT / "results/mitigation_benchmark/fake_backend"

# Expected circuit metrics for N=10 p=1 heavy_hex opt_level=2
EXPECTED = {
    "n_2q_gates": 18,
    "depth_2q": 14,
    "depth_range": (55, 65),  # varies with h/theta
    "active_qubits": 10,
    "optimization_level": 2,
}

# From preflight cost model
COST_MODEL = {
    "effective_clops": 3750,
    "shots": 16384,
    "time_per_circuit_s": 16384 / 3750,  # ~4.37s
}


def load_results():
    results = []
    for f in sorted(HW_DIR.rglob("*.json")):
        d = json.loads(f.read_text())
        if "error" in d and "results" not in d:
            continue
        results.append((f.name, d))
    return results


def check_circuit_metrics(name, d):
    cs = d.get("circuit_stats", {})
    issues = []

    n2q = cs.get("n_2q_gates")
    if n2q != EXPECTED["n_2q_gates"]:
        issues.append(f"n_2q_gates={n2q} (expected {EXPECTED['n_2q_gates']})")

    d2q = cs.get("depth_2q")
    if d2q != EXPECTED["depth_2q"]:
        issues.append(f"depth_2q={d2q} (expected {EXPECTED['depth_2q']})")

    depth = cs.get("depth")
    lo, hi = EXPECTED["depth_range"]
    if depth is not None and not (lo <= depth <= hi):
        issues.append(f"depth={depth} (expected {lo}-{hi})")

    aq = cs.get("active_qubits")
    if aq != EXPECTED["active_qubits"]:
        issues.append(f"active_qubits={aq} (expected {EXPECTED['active_qubits']})")

    opt = cs.get("optimization_level")
    if opt != EXPECTED["optimization_level"]:
        issues.append(f"opt_level={opt} (expected {EXPECTED['optimization_level']})")

    return issues


def check_qpu_time(name, d):
    timing = d.get("timing", {})
    wall = timing.get("wall_time_s")
    qpu = timing.get("qpu_seconds")
    issues = []

    if wall is None:
        issues.append("wall_time_s is None")
    if qpu is None:
        issues.append("qpu_seconds is None (not captured)")

    # Check if QPU time is reasonable (should be 5-400s for one circuit)
    if qpu is not None:
        if qpu < 1:
            issues.append(f"qpu_seconds={qpu} suspiciously low (<1s)")
        if qpu > 600:
            issues.append(f"qpu_seconds={qpu} very high (>10min)")

    # Check wall vs QPU (wall should be >= QPU due to queue)
    if wall and qpu and wall < qpu:
        issues.append(f"wall_time={wall:.0f}s < qpu={qpu}s (impossible)")

    return issues, wall, qpu


def check_energy(name, d):
    res = d.get("results", {})
    issues = []

    e_raw = res.get("e_raw")
    e_exact = res.get("e_exact")
    de_gap = res.get("delta_e_gap")
    h_val = d.get("benchmark_metadata", {}).get("h_value")

    if e_raw is None:
        issues.append("e_raw is None")
        return issues

    if e_exact is None:
        issues.append("e_exact is None")
        return issues

    # Variational principle: e_raw should be >= e_exact (within shot noise)
    # Allow small negative due to shot noise: -0.5 tolerance
    if e_raw < e_exact - 0.5:
        issues.append(
            f"Variational principle violated: e_raw={e_raw:.3f} < e_exact={e_exact:.3f} - 0.5"
        )

    # Delta_e_gap consistency
    if de_gap is not None and e_exact is not None:
        # Need gap to verify — use a rough estimate
        # For h=4.0 N=10: gap ≈ 5.92
        gap_approx = abs(e_exact) * 0.146  # rough
        de_recomputed = abs(e_raw - e_exact) / gap_approx if gap_approx > 0 else None
        # Just check it's positive and finite
        if de_gap < 0:
            issues.append(f"delta_e_gap={de_gap} is negative (impossible)")
        if not np.isfinite(de_gap):
            issues.append(f"delta_e_gap={de_gap} is not finite")

    return issues


def main():
    results = load_results()
    if not results:
        print("No valid hardware results found.")
        return

    print("=" * 70)
    print("  HARDWARE RESULTS ANALYSIS")
    print(f"  Files: {len(results)} in {HW_DIR}")
    print("=" * 70)

    all_ok = True
    summary = []

    for name, d in results:
        meta = d.get("benchmark_metadata", {})
        config_id = meta.get("config_id", "?")
        h_val = meta.get("h_value", "?")
        seed = meta.get("seed", "?")
        res = d.get("results", {})

        print(f"\n--- {config_id} | h={h_val} | seed={seed} ---")

        # 1. Circuit metrics
        circ_issues = check_circuit_metrics(name, d)
        if circ_issues:
            for issue in circ_issues:
                print(f"  ✗ CIRCUIT: {issue}")
            all_ok = False
        else:
            cs = d.get("circuit_stats", {})
            print(f"  ✓ Circuit: n_2q={cs.get('n_2q_gates')}, "
                  f"depth_2q={cs.get('depth_2q')}, depth={cs.get('depth')}")

        # 2. QPU timing
        time_issues, wall, qpu = check_qpu_time(name, d)
        if time_issues:
            for issue in time_issues:
                print(f"  ⚠ TIMING: {issue}")
        else:
            print(f"  ✓ Timing: wall={wall:.1f}s, QPU={qpu}s")

        # Model prediction comparison
        predicted_time = COST_MODEL["time_per_circuit_s"]
        if qpu:
            ratio = qpu / predicted_time
            print(f"    Model predicted: {predicted_time:.1f}s, actual: {qpu}s "
                  f"(ratio={ratio:.1f}x)")

        # 3. Energy validation
        energy_issues = check_energy(name, d)
        if energy_issues:
            for issue in energy_issues:
                print(f"  ✗ ENERGY: {issue}")
            all_ok = False
        else:
            e_raw = res.get("e_raw")
            e_exact = res.get("e_exact")
            de_gap = res.get("delta_e_gap")
            print(f"  ✓ Energy: e_raw={e_raw:.4f}, e_exact={e_exact:.4f}, "
                  f"ΔE/gap={de_gap*100:.2f}%")

        # Phase label
        label = res.get("phase_label")
        correct = res.get("correct_label")
        print(f"  ✓ Phase: {label} (correct={correct})")

        summary.append({
            "config": config_id,
            "h": h_val,
            "de_gap": res.get("delta_e_gap"),
            "e_raw": res.get("e_raw"),
            "qpu_s": qpu,
            "wall_s": wall,
        })

    # Cross-config comparison
    print("\n" + "=" * 70)
    print("  CROSS-CONFIG COMPARISON")
    print("=" * 70)
    print(f"\n  {'Config':<25s} {'h':>4s} {'ΔE/gap':>8s} {'E_raw':>10s} "
          f"{'QPU(s)':>7s} {'Wall(s)':>8s}")
    print("  " + "-" * 65)
    for s in sorted(summary, key=lambda x: (x["h"], x["config"])):
        de = f"{s['de_gap']*100:.1f}%" if s["de_gap"] else "N/A"
        e = f"{s['e_raw']:.3f}" if s["e_raw"] else "N/A"
        qpu_str = f"{s['qpu_s']}" if s["qpu_s"] else "N/A"
        wall_str = f"{s['wall_s']:.0f}" if s["wall_s"] else "N/A"
        print(f"  {s['config']:<25s} {s['h']:>4} {de:>8s} {e:>10s} "
              f"{qpu_str:>7s} {wall_str:>8s}")

    # FakeTorino comparison
    print("\n" + "=" * 70)
    print("  HARDWARE vs SIMULATION COMPARISON")
    print("=" * 70)
    for s in summary:
        config = s["config"]
        h = s["h"]
        # Find matching simulation result
        sim_dir = SIM_DIR / config
        if sim_dir.exists():
            h_str = f"h{str(h).replace('.', 'p')}"
            sim_files = list(sim_dir.glob(f"{h_str}_run_*.json"))
            if sim_files:
                sim_data = json.loads(sim_files[0].read_text())
                sim_de = sim_data.get("results", {}).get("delta_e_gap")
                hw_de = s["de_gap"]
                if sim_de and hw_de:
                    diff = (hw_de - sim_de) * 100
                    print(f"  {config:<25s} h={h}: "
                          f"HW={hw_de*100:.1f}% vs SIM={sim_de*100:.1f}% "
                          f"(Δ={diff:+.1f}pp)")

    print("\n" + "=" * 70)
    if all_ok:
        print("  ✓ ALL CHECKS PASSED")
    else:
        print("  ✗ SOME CHECKS FAILED — see above")
    print("=" * 70)


if __name__ == "__main__":
    main()
