#!/usr/bin/env python3
"""Campaign Data Extractor — Auto-extract metrics from experiment results.

Scans results from the quantum advantage campaign (A.1-A.6) and produces
consolidated tables ready for the campaign document. No AI or manual
inspection required — just run after each experiment completes.

Usage:
    # After running experiments:
    python scripts/analysis/campaign_extractor.py

    # Specific step:
    python scripts/analysis/campaign_extractor.py --step A1
    python scripts/analysis/campaign_extractor.py --step A5

    # Output as markdown table:
    python scripts/analysis/campaign_extractor.py --format md

    # Output as JSON (for programmatic use):
    python scripts/analysis/campaign_extractor.py --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def find_latest_result(experiment_dir: str) -> Path | None:
    """Find the most recent run_*.json with actual data in an experiment directory."""
    exp_path = RESULTS / "experiments" / experiment_dir
    if not exp_path.exists():
        return None
    runs = sorted(exp_path.glob("run_*.json"), reverse=True)
    # Find first run that has section_1 data (skip partial/failed runs)
    for run in runs:
        try:
            with open(run) as f:
                data = json.load(f)
            results = data.get("results", {})
            if "section_1" in results and results["section_1"].get("data"):
                return run
        except (json.JSONDecodeError, KeyError):
            continue
    return runs[0] if runs else None


def extract_a1_qpu_scaling() -> dict | None:
    """Extract QPU time scaling data (Step A.1)."""
    path = find_latest_result("exp_scaling/qpu_time")
    if not path:
        print("  A.1: No results found in exp_scaling/qpu_time/")
        return None

    with open(path) as f:
        data = json.load(f)

    results = data.get("results", {})

    # Section 1: Transpilation
    s1 = results.get("section_1", {}).get("data", {})
    transpile_data = s1.get("per_n", [])

    # Section 2: CLOPS model
    s2 = results.get("section_2", {}).get("data", {})
    cost_data = s2.get("per_n", [])

    # Section 3: FakeTorino timing (may not exist)
    s3 = results.get("section_3", {}).get("data", {})
    timing_data = s3.get("per_n", [])

    # Section 4: Scaling fit
    s4 = results.get("section_4", {}).get("data", {})
    exponents = s4.get("scaling_exponents", {})

    # Merge into consolidated table
    table = []
    for i, td in enumerate(transpile_data):
        row = {
            "N": td["n_qubits"],
            "CX_pre": td["cx_count_pre_transpile"],
            "CX_post": td["cx_count_post_transpile"],
            "routing_x": round(td["routing_overhead_ratio"], 2),
            "depth_2q": td["depth_2q"],
            "transpile_s": td["transpile_time_s"],
        }
        if i < len(cost_data):
            cd = cost_data[i]
            row["CLOPS_eff"] = cd["effective_clops"]
            row["T_est_s"] = round(cd["est_total_expected_s"], 1)
            row["T1_ratio"] = round(cd["t1_budget_ratio"], 4)
            row["SNR"] = round(cd["snr_at_critical"], 2)
        if i < len(timing_data):
            row["T_fake_s"] = timing_data[i]["fake_backend_time_s"]
        table.append(row)

    return {
        "source_file": str(path.name),
        "table": table,
        "exponents": exponents,
    }


def extract_a5_mps_precision() -> dict | None:
    """Extract MPS precision study data (Step A.5)."""
    path = find_latest_result("exp_scaling/mps_precision")
    if not path:
        print("  A.5: No results found in exp_scaling/mps_precision/")
        return None

    with open(path) as f:
        data = json.load(f)

    results = data.get("results", {})
    s2 = results.get("section_2", {}).get("data", {})
    s3 = results.get("section_3", {}).get("data", {})

    per_config = s2.get("per_config", [])
    chi_values = s2.get("chi_values", [])

    # Build table: per (topology, N, h)
    table = []
    for rec in per_config:
        chi_results = rec.get("chi_results", {})
        e_exact = rec.get("e_exact")
        gap = rec.get("gap", 1.0)

        row = {
            "topology": rec["topology"],
            "N": rec["n_qubits"],
            "h": rec["h"],
            "E_exact": round(e_exact, 8) if e_exact else None,
            "gap": round(gap, 4),
        }

        # Add energy at each chi
        for chi in chi_values:
            chi_key = str(chi) if str(chi) in chi_results else chi
            if chi_key in chi_results:
                e_chi = chi_results[chi_key]["energy"]
                row[f"E_chi{chi}"] = round(e_chi, 8)
                if e_exact:
                    row[f"dE_gap_chi{chi}"] = round(
                        abs(e_chi - e_exact) / max(gap, 1e-10), 6
                    )

        # Truncation error (chi=64 vs chi_max)
        chi_max = max(chi_values)
        e_64 = chi_results.get(str(64), chi_results.get(64, {})).get("energy")
        e_max = chi_results.get(str(chi_max), chi_results.get(chi_max, {})).get("energy")
        if e_64 is not None and e_max is not None:
            row["trunc_64_vs_max"] = abs(e_64 - e_max)
        table.append(row)

    # Per-topology summary from section 3
    per_topo = s3.get("per_topology", {}) if s3 else {}

    return {
        "source_file": str(path.name),
        "chi_values": chi_values,
        "table": table,
        "per_topology_summary": per_topo,
    }


def extract_a3_deployment() -> dict | None:
    """Extract parametric deployment results (Step A.3)."""
    deploy_dir = RESULTS / "hardware" / "parametric"
    if not deploy_dir.exists():
        print("  A.3: No results found in hardware/parametric/")
        return None

    # Find latest deployment_N20_heavy_hex_*.json
    files = sorted(deploy_dir.glob("deployment_N*heavy_hex*.json"), reverse=True)
    if not files:
        files = sorted(deploy_dir.glob("deployment_N*.json"), reverse=True)
    if not files:
        return None

    with open(files[0]) as f:
        data = json.load(f)

    return {
        "source_file": str(files[0].name),
        "summary": data.get("summary", {}),
        "preflight": data.get("preflight", {}),
        "mps_comparison": data.get("mps_comparison", {}),
        "analysis": data.get("analysis", {}),
    }


def format_table_md(rows: list[dict], title: str = "") -> str:
    """Format a list of dicts as a markdown table."""
    if not rows:
        return f"{title}\n(no data)\n"

    keys = list(rows[0].keys())
    lines = []
    if title:
        lines.append(f"\n### {title}\n")

    # Header
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "|".join(["---"] * len(keys)) + "|")

    # Rows
    for row in rows:
        vals = []
        for k in keys:
            v = row.get(k)
            if v is None:
                vals.append("—")
            elif isinstance(v, float):
                if abs(v) < 0.001 and v != 0:
                    vals.append(f"{v:.2e}")
                else:
                    vals.append(f"{v}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Campaign Data Extractor")
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "A1", "A5", "A3"],
                        help="Which step to extract")
    parser.add_argument("--format", type=str, default="md",
                        choices=["md", "json"],
                        help="Output format")
    args = parser.parse_args()

    print("=" * 60)
    print("  QUANTUM ADVANTAGE CAMPAIGN — Data Extraction")
    print("=" * 60)

    extracted = {}

    if args.step in ("all", "A1"):
        print("\n[A.1] QPU Time Scaling:")
        a1 = extract_a1_qpu_scaling()
        if a1:
            extracted["A1"] = a1
            if args.format == "md":
                print(format_table_md(a1["table"], "A.1 QPU Time Scaling"))
                print(f"\nExponents: {json.dumps(a1['exponents'], indent=2)}")
            print(f"  Source: {a1['source_file']}")

    if args.step in ("all", "A5"):
        print("\n[A.5] MPS Precision:")
        a5 = extract_a5_mps_precision()
        if a5:
            extracted["A5"] = a5
            if args.format == "md":
                print(format_table_md(a5["table"], "A.5 MPS Chi-Convergence"))
                if a5["per_topology_summary"]:
                    print("\nPer-topology summary:")
                    for topo, summary in a5["per_topology_summary"].items():
                        print(f"  {topo}: verdict={summary.get('precision_verdict')}, "
                              f"mean_dE/gap_64={summary.get('mean_chi64_de_gap')}")
            print(f"  Source: {a5['source_file']}")

    if args.step in ("all", "A3"):
        print("\n[A.3] Parametric Deployment:")
        a3 = extract_a3_deployment()
        if a3:
            extracted["A3"] = a3
            if args.format == "md":
                print(f"  Summary: {json.dumps(a3['summary'], indent=2)}")
            print(f"  Source: {a3['source_file']}")

    if args.format == "json":
        print(json.dumps(extracted, indent=2, default=str))

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
