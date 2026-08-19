#!/usr/bin/env python3
"""Edge case tests for the new validation functions and update script."""
import json
import os
import tempfile
import sys

sys.path.insert(0, "src")

from qmbp_simulation.analysis.metrics import (
    detect_h_frontier_anomalies,
    detect_training_zoo_incoherence,
    detect_pass_rate_regression,
    H_FRONTIER_MONOTONICITY_TOLERANCE,
    PASS_RATE_REGRESSION_THRESHOLD,
    GAP_MASKING_THRESHOLD,
)

failures = []

def check(name, condition, detail=""):
    if condition:
        print(f"  PASS: {name}")
    else:
        print(f"  FAIL: {name} {detail}")
        failures.append(name)


print("=== detect_h_frontier_anomalies ===")

# Empty input
check("empty input returns []", detect_h_frontier_anomalies([]) == [])

# Single config per topology — no pairs
single = [{"topology": "ladder", "n_qubits": 10, "h_frontier": 2.5}]
check("single config per topo returns []", detect_h_frontier_anomalies(single) == [])

# None h_frontier values skipped
with_none = [
    {"topology": "ladder", "n_qubits": 10, "h_frontier": None},
    {"topology": "ladder", "n_qubits": 20, "h_frontier": 3.5},
]
check("None h_frontier skipped", detect_h_frontier_anomalies(with_none) == [])

# Monotonically increasing — no anomaly
monotonic = [
    {"topology": "chain_1d", "n_qubits": 6, "h_frontier": 2.0},
    {"topology": "chain_1d", "n_qubits": 10, "h_frontier": 2.5},
    {"topology": "chain_1d", "n_qubits": 20, "h_frontier": 3.0},
]
check("monotonic increasing returns []", detect_h_frontier_anomalies(monotonic) == [])

# Drop exactly at tolerance — should NOT flag (strict >)
tol = H_FRONTIER_MONOTONICITY_TOLERANCE
at_tol = [
    {"topology": "chain_1d", "n_qubits": 6, "h_frontier": 2.5},
    {"topology": "chain_1d", "n_qubits": 10, "h_frontier": 2.5 - tol},
]
result = detect_h_frontier_anomalies(at_tol)
check("drop exactly at tolerance not flagged", len(result) == 0, f"got {result}")

# Drop slightly below tolerance — should flag
below = [
    {"topology": "chain_1d", "n_qubits": 6, "h_frontier": 2.5},
    {"topology": "chain_1d", "n_qubits": 10, "h_frontier": 2.5 - tol - 0.01},
]
result = detect_h_frontier_anomalies(below)
check("drop below tolerance flagged", len(result) == 1, f"got {len(result)}")

# Multiple topologies — only one has anomaly
multi_topo = [
    {"topology": "ladder", "n_qubits": 6, "h_frontier": 2.0},
    {"topology": "ladder", "n_qubits": 10, "h_frontier": 2.5},
    {"topology": "square", "n_qubits": 6, "h_frontier": 3.0},
    {"topology": "square", "n_qubits": 10, "h_frontier": 2.0},  # drop of 1.0 — anomaly
]
result = detect_h_frontier_anomalies(multi_topo)
check("only anomalous topology flagged", len(result) == 1 and result[0]["topology"] == "square")

# N values out of order in input — function should sort by N
unordered = [
    {"topology": "ladder", "n_qubits": 20, "h_frontier": 3.5},
    {"topology": "ladder", "n_qubits": 6, "h_frontier": 2.0},
    {"topology": "ladder", "n_qubits": 10, "h_frontier": 2.5},
]
check("unsorted N values handled correctly", detect_h_frontier_anomalies(unordered) == [])


print("\n=== detect_pass_rate_regression ===")

# No prev file
check("no prev file returns []",
      detect_pass_rate_regression([], previous_dashboard_path="/nonexistent/path.json") == [])

# Create prev with higher pass rates
prev_configs = [
    {"topology": "ladder", "n_qubits": 10, "pass_rate_dual_criterion": 0.95},
    {"topology": "heavy_hex", "n_qubits": 10, "pass_rate_dual_criterion": 0.80},
]
# Ladder drops hard across ALL its N — topology max drops from 0.95 to 0.60
curr_configs_bad = [
    {"topology": "ladder", "n_qubits": 10, "pass_rate_dual_criterion": 0.60},  # drop 0.35 → regression
    {"topology": "heavy_hex", "n_qubits": 10, "pass_rate_dual_criterion": 0.85},  # improvement → ok
]
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump({"configs": prev_configs}, f)
    prev_path = f.name

result = detect_pass_rate_regression(curr_configs_bad, previous_dashboard_path=prev_path)
check("regression detected when topology max drops 0.35", len(result) == 1 and result[0]["topology"] == "ladder")
check("improvement not flagged (heavy_hex)", all(r["topology"] != "heavy_hex" for r in result))

# Small drop (0.05) below threshold — should NOT flag
curr_small_drop = [
    {"topology": "ladder", "n_qubits": 10, "pass_rate_dual_criterion": 0.90},  # drop 0.05 → ok
]
result = detect_pass_rate_regression(curr_small_drop, previous_dashboard_path=prev_path)
check("small drop (0.05) not flagged", result == [])

# Both prev and curr missing dual criterion field — should handle gracefully
curr_no_dual = [
    {"topology": "ladder", "n_qubits": 10},  # missing pass_rate_dual_criterion
]
result = detect_pass_rate_regression(curr_no_dual, previous_dashboard_path=prev_path)
check("missing pass_rate_dual_criterion handled gracefully", isinstance(result, list))

os.unlink(prev_path)

# Corrupted prev JSON
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    f.write("{invalid json}")
    corrupt_path = f.name
result = detect_pass_rate_regression(curr_configs_bad, previous_dashboard_path=corrupt_path)
check("corrupted prev JSON returns []", result == [])
os.unlink(corrupt_path)


print("\n=== detect_training_zoo_incoherence (no NPZ write — just API check) ===")

# No zoo pass rate — should skip
no_zoo = [
    {"topology": "ladder", "n_qubits": 10, "zoo_pass_rate": None, "file": "ladder_N10_p1.npz"}
]
result = detect_training_zoo_incoherence(no_zoo)
check("None zoo_pass_rate skipped", result == [])

# Zoo pass below threshold — should skip
low_zoo = [
    {"topology": "ladder", "n_qubits": 10, "zoo_pass_rate": 0.40, "file": "ladder_N10_p1.npz"}
]
result = detect_training_zoo_incoherence(low_zoo)
check("zoo_pass below ZOO_PASS_FOR_INCOHERENCE_FLAG skipped", result == [])

# NPZ file doesn't exist — should skip gracefully
missing_npz = [
    {"topology": "ladder", "n_qubits": 999, "zoo_pass_rate": 0.99, "file": "ladder_N999_p1.npz"}
]
result = detect_training_zoo_incoherence(missing_npz)
check("missing NPZ file skipped gracefully", result == [])


print("\n=== update_cross_n_coverage.py edge cases ===")

from scripts.maintenance.update_cross_n_coverage import (
    generate_executive_summary,
    generate_topology_table,
    generate_gap_masking_table,
    generate_h_frontier_table,
    update_section,
)

# Empty dashboard
empty_dash = {"topology_summary": {}, "configs": []}
summary = generate_executive_summary(empty_dash)
check("empty dashboard executive summary is string", isinstance(summary, str))

# Dashboard with no gap masking
no_mask = [
    {"topology": "chain_1d", "n_qubits": 8, "pass_rate_5pct": 0.9, "pass_rate_dual_criterion": 0.9}
]
result = generate_gap_masking_table(no_mask)
check("no gap masking shows placeholder", "No significant gap masking detected" in result)

# h_frontier table with all None values
all_none = [
    {"topology": "chain_1d", "n_qubits": 8, "h_frontier": None},
    {"topology": "chain_1d", "n_qubits": 20, "h_frontier": None},
]
result = generate_h_frontier_table(all_none)
check("all-None h_frontier table is string (no crash)", isinstance(result, str))

# update_section with missing markers
doc = "## Section A\n\nsome content\n\n## Section B\n\ncontent"
updated, found = update_section(doc, "missing_section", "new content")
check("update_section with missing marker returns found=False", not found)
check("update_section with missing marker returns doc unchanged", updated == doc)

# update_section with markers present
doc_with_markers = (
    "prefix\n"
    "<!-- AUTO-GENERATED-BEGIN:my_section -->\n"
    "old content\n"
    "<!-- AUTO-GENERATED-END:my_section -->\n"
    "suffix"
)
updated, found = update_section(doc_with_markers, "my_section", "new content")
check("update_section with markers found=True", found)
check("update_section replaces content", "new content" in updated)
check("update_section preserves prefix", "prefix" in updated)
check("update_section preserves suffix", "suffix" in updated)
check("update_section removes old content", "old content" not in updated)

# Topology table with empty data for that topology
result = generate_topology_table("unknown_topo", [])
check("empty topology table returns placeholder", "*No data for unknown_topo*" in result)

# Topology table with gap masking config
configs_with_mask = [
    {
        "topology": "ladder", "n_qubits": 14, "n_points": 10,
        "h_range": [3.0, 3.5], "pass_rate_5pct": 0.8,
        "pass_rate_dual_criterion": 0.0,
        "h_frontier": 3.1, "theta_smoothness": 0.07,
        "zoo_vs_npz_divergence": 0.2, "model_stale": False,
    }
]
result = generate_topology_table("ladder", configs_with_mask)
check("gap masking flag appears in topology table", "GAP MASK" in result)


print()
if failures:
    print(f"FAILURES ({len(failures)}):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print(f"All {26 + len([l for l in open(__file__).read().split('check(') if l])} checks PASS")
