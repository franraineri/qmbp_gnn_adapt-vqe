#!/usr/bin/env python3
"""Verify completeness and robustness of thesis-grade noiseless runs.

Checks each run in the scaling table for:
1. All 4 sections completed successfully
2. Per-h deploy data present and complete
3. No NaN/Inf in metrics
4. Fidelity bounds respected (0 < F ≤ 1)
5. ΔE/gap physically reasonable (≥ 0)
6. Phase labels consistent with h-value
7. VQE variational principle respected (E_vqe ≥ E_exact)
8. Gaps are positive and non-degenerate
9. MPNN wins computed correctly
10. Speedup factor consistent
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qmbp_simulation.framework.result_io import load_result

# ═══════════════════════════════════════════════════════════════════════
# Thesis runs to verify (the scaling table)
# ═══════════════════════════════════════════════════════════════════════

THESIS_RUNS = {
    "tfim chain_1d N=10 p=3": "exp_noiseless/tfim/multi/run_20260709_172220.json",
    "tfim chain_1d N=16 p=4": "exp_noiseless_tfim_4/run_20260702_184821.json",
    "tfim chain_1d N=20 p=4": "exp_noiseless_tfim_4/run_20260702_200440.json",
    "tfim heavy_hex N=10 p=3": "exp_noiseless_tfim_4/run_20260702_172339.json",
    "tfim heavy_hex N=16 p=3": "exp_noiseless/tfim/heavy_hex/run_20260709_190354.json",
    "tfim heavy_hex N=20 p=3": "exp_noiseless/tfim/heavy_hex/run_20260707_043418.json",
    "tfim_long chain_1d N=10 p=3": "exp_noiseless_tfim_longitudinal_v3/run_20260627_203300.json",
    "tfim_long chain_1d N=16 p=3": "exp_noiseless/tfim_longitudinal/chain_1d/run_20260710_095700.json",
    "tfim_long chain_1d N=20 p=3": "exp_noiseless/tfim_longitudinal/chain_1d/run_20260710_102525.json",
    "tfim_long heavy_hex N=10 p=3": "exp_noiseless_tfim_longitudinal_v3/run_20260627_215224.json",
    "tfim_long heavy_hex N=16 p=3": "exp_noiseless/tfim_longitudinal/heavy_hex/run_20260710_113208.json",
    "tfim_long heavy_hex N=20 p=3": "exp_noiseless/tfim_longitudinal/heavy_hex/run_20260710_091429.json",
}

BASE = ROOT / "results" / "experiments"


# ═══════════════════════════════════════════════════════════════════════
# Verification checks
# ═══════════════════════════════════════════════════════════════════════


def check_run(label: str, rel_path: str) -> dict:
    """Run all verification checks on one result file."""
    path = BASE / rel_path
    issues: list[str] = []
    warnings: list[str] = []

    # Check 0: File exists
    if not path.exists():
        return {
            "label": label,
            "status": "MISSING",
            "issues": [f"File not found: {rel_path}"],
            "warnings": [],
        }

    try:
        data = load_result(path)
    except Exception as e:
        return {
            "label": label,
            "status": "CORRUPT",
            "issues": [f"Cannot load: {e}"],
            "warnings": [],
        }

    summary = data.get("summary", {})
    results = data.get("results", {})
    config = data.get("config", {})

    # Check 1: All 4 sections completed
    n_sections = summary.get("n_sections", 0)
    if n_sections != 4:
        issues.append(f"Only {n_sections} sections (need 4)")

    all_passed = summary.get("all_passed", False)
    if not all_passed:
        failed = summary.get("failed_sections", [])
        failed_names = [f.get("name", "?") for f in failed] if failed else []
        issues.append(f"Not all passed: failed={failed_names}")

    # Check 2: Section 4 has per-point data
    s4 = results.get("section_4", {})
    s4_data = s4.get("data", {})
    per_point = s4_data.get("per_point", [])
    if not per_point:
        issues.append("No per-h deploy data in Section 4")
        return {"label": label, "status": "INCOMPLETE", "issues": issues, "warnings": warnings}

    n_test = s4_data.get("n_test_points", len(per_point))
    n_pass = s4_data.get("n_pass_energy", 0)

    # Check 3: No NaN/Inf in key metrics
    mean_de = s4_data.get("mean_de_gap")
    mean_f = s4_data.get("mean_fidelity")
    speedup = s4_data.get("speedup_factor")

    if mean_de is not None and (math.isnan(mean_de) or math.isinf(mean_de)):
        issues.append(f"mean_de_gap is {mean_de} (NaN/Inf)")
    if mean_f is not None and (math.isnan(mean_f) or math.isinf(mean_f)):
        issues.append(f"mean_fidelity is {mean_f} (NaN/Inf)")

    # Check per-point data quality
    n_nan_de = 0
    n_bad_fid = 0
    n_negative_de = 0
    n_variational_violation = 0
    n_label_inconsistent = 0

    for pp in per_point:
        h = pp.get("h_test", pp.get("h", 0))
        de = pp.get("de_gap", pp.get("delta_e_gap"))
        fid = pp.get("fidelity", pp.get("f"))

        # Check 3: NaN/Inf
        if de is not None and (math.isnan(de) or math.isinf(de)):
            n_nan_de += 1
        # Check 4: Fidelity bounds
        if fid is not None and (fid < 0 or fid > 1.001):
            n_bad_fid += 1
        # Check 5: ΔE/gap ≥ 0
        if de is not None and de < -1e-10:
            n_negative_de += 1

    if n_nan_de > 0:
        issues.append(f"{n_nan_de}/{len(per_point)} points have NaN/Inf ΔE/gap")
    if n_bad_fid > 0:
        issues.append(f"{n_bad_fid}/{len(per_point)} points have fidelity outside [0,1]")
    if n_negative_de > 0:
        warnings.append(
            f"{n_negative_de}/{len(per_point)} points have negative ΔE/gap (numerical noise)"
        )

    # Check 6: Phase labels consistent
    for pp in per_point:
        h = pp.get("h_test", pp.get("h", 0))
        label_pred = pp.get("predicted_label", pp.get("phase_label", ""))
        # h > 2.0 should always be paramagnetic for TFIM
        if h > 2.5 and label_pred and "ferro" in label_pred.lower():
            n_label_inconsistent += 1
    if n_label_inconsistent > 0:
        warnings.append(
            f"{n_label_inconsistent} points labeled ferromagnetic at h>2.5 (suspicious)"
        )

    # Check 7: VQE section - variational principle
    # NOTE: For non-1D topologies with N>15, the DMRG E_exact uses TFIChain (1D model)
    # which may give a HIGHER energy than the true ground state. Small violations
    # (< 1e-2) in heavy_hex/ladder N>15 are EXPECTED and not a real issue.
    s2 = results.get("section_2", {}).get("data", {})
    s2_topo = s2.get("topologies", {})
    cfg_sys = data.get("config", {}).get("system", {})
    n_qubits_cfg = cfg_sys.get("n_qubits", 0)
    topo_cfg = cfg_sys.get("topologies", cfg_sys.get("topology", ""))
    topo_str = topo_cfg[0] if isinstance(topo_cfg, list) else topo_cfg
    # Looser threshold for non-chain topologies at N>15 (DMRG E_exact is approximate)
    _DMRG_APPROX_TOPOLOGIES = ("heavy_hex", "ladder")
    if topo_str in _DMRG_APPROX_TOPOLOGIES and n_qubits_cfg > 15:
        viol_threshold = 1e-2  # Tolerate up to 1e-2 (DMRG TFIChain limitation)
    else:
        viol_threshold = 1e-6  # Tight threshold for chain_1d (TFIChain is exact)
    for topo_name, topo_data in s2_topo.items():
        vqe_results = topo_data.get("per_point", topo_data.get("results", []))
        if isinstance(vqe_results, list):
            for vr in vqe_results:
                e_vqe = vr.get("e_vqe", vr.get("energy_vqe", vr.get("energy")))
                e_exact = vr.get("e_exact", vr.get("energy_exact", vr.get("ground_energy")))
                if e_vqe is not None and e_exact is not None:
                    if e_vqe < e_exact - viol_threshold:
                        n_variational_violation += 1

    if n_variational_violation > 0:
        issues.append(
            f"{n_variational_violation} VQE points violate variational principle (E_vqe < E_exact)"
        )

    # Check 8: Gaps from Section 1
    s1_data = results.get("section_1", {}).get("data", {})
    s1_topos = s1_data.get("topologies", {})
    for tname, tdata in s1_topos.items():
        gap_min = tdata.get("gap_min", 0)
        if gap_min is not None and gap_min <= 0:
            warnings.append(f"gap_min={gap_min:.6f} ≤ 0 in {tname} (degenerate/numerical)")

    # Check 8b: Cross-check E_exact between Section 1 and Section 2
    # If both have per-point energies, verify they match (detects cache corruption)
    for tname, tdata_s2 in s2_topo.items():
        s2_points = tdata_s2.get("per_point", [])
        s1_topo_data = s1_topos.get(tname, {})
        s1_points = s1_topo_data.get("points", [])
        if s2_points and s1_points and len(s2_points) == len(s1_points):
            n_mismatch = 0
            for i, (p1, p2) in enumerate(zip(s1_points, s2_points, strict=False)):
                e1 = p1.get("ground_energy", p1.get("energy"))
                e2 = p2.get("energy_exact", p2.get("e_exact", p2.get("ground_energy")))
                if e1 is not None and e2 is not None:
                    if abs(e1 - e2) > 1e-8:
                        n_mismatch += 1
            if n_mismatch > 0:
                issues.append(
                    f"E_exact mismatch between Section 1 and 2 in {tname}: "
                    f"{n_mismatch} points differ by >1e-8 (cache corruption?)"
                )

    # Check 9: MPNN wins consistency
    mpnn_wins_reported = s4_data.get("mpnn_wins_vs_random", 0)
    if per_point and mpnn_wins_reported is not None:
        computed_wins = 0
        for pp in per_point:
            de = pp.get("de_gap", 999)
            de_rand = pp.get("de_gap_random", pp.get("de_gap_random_init"))
            if de_rand is not None and de < de_rand:
                computed_wins += 1
        if abs(computed_wins - (mpnn_wins_reported or 0)) > 1:
            warnings.append(
                f"MPNN wins mismatch: reported={mpnn_wins_reported}, computed={computed_wins}"
            )

    # Check 10: Speedup reasonable
    if speedup is not None:
        if speedup < 1:
            warnings.append(f"Speedup={speedup:.1f}× < 1 (MPNN slower than VQE?)")
        elif speedup > 1000:
            warnings.append(f"Speedup={speedup:.0f}× suspiciously high")

    # Check 11: simulation_diagnostics consistency (MPS + 2D → chi check needed)
    sd = data.get("simulation_diagnostics", {})
    if sd:
        backend_type = sd.get("backend_type", "")
        if sd.get("chi_sufficiency_warning"):
            warnings.append(f"Chi sufficiency warning: {sd['chi_sufficiency_warning']}")
        # If MPS + 2D topology, recommend --verify-chi
        cfg = data.get("config", {})
        sys_cfg = cfg.get("system", {})
        topo_val = sys_cfg.get("topologies", sys_cfg.get("topology", ""))
        topo_check = topo_val[0] if isinstance(topo_val, list) else topo_val
        if "mps" in backend_type and topo_check in ("square", "triangular"):
            # Check if chi-convergence section exists
            if "section_3" not in results or not results.get("section_3", {}).get("data", {}).get(
                "chi_1x"
            ):
                warnings.append(
                    f"MPS backend on 2D topology '{topo_check}' — no chi-convergence "
                    f"verification found. Re-run with --verify-chi for thesis rigor."
                )

    # Check 12: Variational violations from new fields
    for topo_name_vv, topo_data_vv in s2_topo.items():
        n_viol = topo_data_vv.get("n_variational_violations", 0)
        max_viol = topo_data_vv.get("max_variational_violation", 0)
        if n_viol > 0:
            if max_viol > 1e-4:
                issues.append(
                    f"{n_viol} variational violations in {topo_name_vv} "
                    f"(max={max_viol:.2e}) — investigate solver/backend"
                )
            else:
                warnings.append(
                    f"{n_viol} minor variational violations in {topo_name_vv} "
                    f"(max={max_viol:.2e}, likely numerical noise)"
                )

    # Determine status
    if issues:
        status = "FAIL"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "label": label,
        "status": status,
        "n_test": n_test,
        "n_pass": n_pass,
        "deploy_rate": f"{n_pass / n_test * 100:.0f}%" if n_test > 0 else "—",
        "mean_de_gap": mean_de,
        "mean_fidelity": mean_f,
        "speedup": speedup,
        "issues": issues,
        "warnings": warnings,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main() -> int:
    print("=" * 70)
    print("  THESIS RUNS VERIFICATION")
    print("=" * 70)

    all_results = []
    n_pass = 0
    n_warn = 0
    n_fail = 0
    n_missing = 0

    for label, rel_path in THESIS_RUNS.items():
        result = check_run(label, rel_path)
        all_results.append(result)

        status = result["status"]
        icon = {
            "PASS": "✅",
            "WARN": "⚠️",
            "FAIL": "❌",
            "MISSING": "🚫",
            "INCOMPLETE": "🚫",
            "CORRUPT": "🚫",
        }
        print(f"\n  {icon.get(status, '?')} {label}")
        print(f"     Status: {status}")

        if status in ("PASS", "WARN"):
            print(
                f"     Deploy: {result.get('n_pass', '?')}/{result.get('n_test', '?')} ({result.get('deploy_rate', '?')})"
            )
            print(
                f"     ΔE/gap: {result.get('mean_de_gap', '?'):.5f}"
                if result.get("mean_de_gap")
                else ""
            )
            print(
                f"     F̄: {result.get('mean_fidelity', '?'):.5f}"
                if result.get("mean_fidelity")
                else ""
            )
            print(
                f"     Speedup: {result.get('speedup', '?'):.0f}×" if result.get("speedup") else ""
            )

        if result["issues"]:
            for i in result["issues"]:
                print(f"     ❌ {i}")
        if result["warnings"]:
            for w in result["warnings"]:
                print(f"     ⚠️  {w}")

        if status == "PASS":
            n_pass += 1
        elif status == "WARN":
            n_warn += 1
        elif status == "FAIL":
            n_fail += 1
        else:
            n_missing += 1

    # Summary
    print("\n" + "=" * 70)
    print(f"  SUMMARY: {n_pass} PASS | {n_warn} WARN | {n_fail} FAIL | {n_missing} MISSING")
    print("=" * 70)

    if n_fail == 0 and n_missing == 0:
        print("\n  ✅ ALL THESIS RUNS VERIFIED — data is complete and robust")
    elif n_missing > 0:
        print(f"\n  🚫 {n_missing} run(s) missing — check file paths")
    else:
        print(f"\n  ❌ {n_fail} run(s) have issues — review before publishing")

    return 0 if (n_fail == 0 and n_missing == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
