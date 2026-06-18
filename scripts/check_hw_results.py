"""Check hardware benchmark results."""

import json
from pathlib import Path

hw_dir = Path("results/mitigation_benchmark/hardware")

for f in sorted(hw_dir.rglob("*.json")):
    print(f"=== {f.name} ===")
    d = json.loads(f.read_text())

    has_error = "error" in d
    results = d.get("results", {})
    e_raw = results.get("e_raw")
    e_exact = results.get("e_exact")
    de_gap = results.get("delta_e_gap")
    label = results.get("phase_label")
    correct = results.get("correct_label")
    meta = d.get("benchmark_metadata", {})
    config_id = meta.get("config_id", "?")
    h_val = meta.get("h_value", "?")
    mode = meta.get("execution_mode", "?")

    hw_cal = d.get("hardware_calibration", {})
    backend_name = hw_cal.get("backend_name", "?") if hw_cal else "?"

    print(f"  Config: {config_id}, h={h_val}, mode={mode}")
    print(f"  Backend: {backend_name}")
    print(f"  e_raw={e_raw}, e_exact={e_exact}")
    if de_gap is not None:
        print(f"  delta_e_gap = {de_gap * 100:.2f}%")
    else:
        print("  delta_e_gap = None")
    print(f"  Phase: {label}, correct={correct}")
    print(f"  Has error: {has_error}")
    if has_error:
        print(f"  Error: {d['error']}")

    cs = d.get("circuit_stats", {})
    if cs:
        n2q = cs.get("n_2q_gates")
        d2q = cs.get("depth_2q")
        dep = cs.get("depth")
        print(f"  Circuit: n_2q={n2q}, depth_2q={d2q}, depth={dep}")

    timing = d.get("timing", {})
    if timing:
        wall = timing.get("wall_time_s")
        qpu = timing.get("qpu_seconds")
        if wall:
            print(f"  Wall: {wall:.1f}s")
        if qpu:
            print(f"  QPU: {qpu}s")

    shots = d.get("shots", "?")
    print(f"  Shots: {shots}")
    print()
