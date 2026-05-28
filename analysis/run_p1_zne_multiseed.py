#!/usr/bin/env python3
"""Multi-seed verification of p=1 ZNE at N=10 triangular.

Hypothesis: p=1 reduces CX count enough for ZNE to work at N=10.
Evidence so far: 1 result (seed=42) shows R²=0.979, gain=+73%, 3/3 wins.
This script runs seeds 43 and 44 to confirm.

Expected runtime: ~2 min per seed (p=1 is fast).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(ROOT / ".venv" / "bin" / "python")
OUTPUT_BASE = ROOT / "analysis" / "verification" / "p1_zne_multiseed"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]
TOPOLOGY = "triangular"
N_QUBITS = 10
P_LAYERS = 1
H_VALUES = ["5.0", "4.5", "4.0"]
N_LAYOUTS = 3
SHOTS = 16384


def run_seed(seed: int) -> dict:
    """Run noisy pipeline for one seed."""
    output_dir = OUTPUT_BASE / f"seed_{seed}"
    output_dir.mkdir(exist_ok=True)

    cmd = [
        PYTHON,
        "scripts/run_noisy_pipeline.py",
        "--n-qubits",
        str(N_QUBITS),
        "--p",
        str(P_LAYERS),
        "--topology",
        TOPOLOGY,
        "--h-values",
        *H_VALUES,
        "--n-layouts",
        str(N_LAYOUTS),
        "--shots",
        str(SHOTS),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]

    print(f"  Running seed={seed}...", file=sys.stderr, flush=True)
    start = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(ROOT))

    elapsed = time.time() - start
    print(f"    Done in {elapsed:.1f}s (exit={result.returncode})", file=sys.stderr)

    if result.returncode != 0:
        print(f"    STDERR: {result.stderr[-500:]}", file=sys.stderr)
        return {"seed": seed, "success": False, "error": result.stderr[-200:]}

    # Find the result file
    noisy_files = sorted(output_dir.glob("noisy_*.json"), reverse=True)
    if not noisy_files:
        return {"seed": seed, "success": False, "error": "No output file found"}

    with open(noisy_files[0]) as f:
        data = json.load(f)

    summary = data.get("summary", {})
    return {
        "seed": seed,
        "success": True,
        "mean_r2": summary.get("mean_r2"),
        "mean_gain_pct": summary.get("mean_gain_pct"),
        "n_mitigated_wins": summary.get("n_mitigated_wins"),
        "n_total": summary.get("n_total"),
        "elapsed_s": elapsed,
    }


def main():
    print("=" * 60, file=sys.stderr)
    print("P=1 ZNE Multi-Seed Verification", file=sys.stderr)
    print(f"Config: N={N_QUBITS}, p={P_LAYERS}, {TOPOLOGY}", file=sys.stderr)
    print(f"Seeds: {SEEDS}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    results = []
    for seed in SEEDS:
        r = run_seed(seed)
        results.append(r)
        if r["success"]:
            print(
                f"    R²={r['mean_r2']:.3f}, gain={r['mean_gain_pct']:+.1f}%, "
                f"wins={r['n_mitigated_wins']}/{r['n_total']}",
                file=sys.stderr,
            )

    # Save results
    output_file = OUTPUT_BASE / "multiseed_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 60, file=sys.stderr)
    print("RESULTS:", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    n_success = sum(1 for r in results if r["success"])
    n_positive_gain = sum(1 for r in results if r["success"] and (r.get("mean_gain_pct") or 0) > 0)

    for r in results:
        if r["success"]:
            status = "✅" if (r.get("mean_gain_pct") or 0) > 0 else "❌"
            print(
                f"  Seed {r['seed']}: R²={r['mean_r2']:.3f}, "
                f"gain={r['mean_gain_pct']:+.1f}%, "
                f"wins={r['n_mitigated_wins']}/{r['n_total']} {status}",
                file=sys.stderr,
            )
        else:
            print(f"  Seed {r['seed']}: FAILED — {r.get('error', 'unknown')}", file=sys.stderr)

    print(f"\nVERDICT: {n_positive_gain}/{n_success} seeds show positive ZNE gain", file=sys.stderr)
    if n_positive_gain >= 2:
        print("✅ CONFIRMED: p=1 ZNE works at N=10 triangular (multi-seed)", file=sys.stderr)
    else:
        print("❌ NOT CONFIRMED: p=1 ZNE result was seed-specific", file=sys.stderr)

    print(f"\nResults saved to: {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
