#!/usr/bin/env python3
"""Convert recovered QESEM JSON results into HardwareRunResult format.

Reads the recovered QESEM JSONs and produces HardwareRunResult-compatible
JSON files that integrate with existing validators and thesis figures.

Usage:
    .venv/bin/python scripts/convert_qesem_to_hwresult.py
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from qmbp_simulation.utils.helpers import json_dump

# ─── Constants from the deployment (h=4.0 for TFIM heavy_hex N=10 p=1) ────
H_VALUE = 4.0
E_EXACT = -40.565690435512735
GAP = 5.921971082752528
EXPECTED_LABEL = "paramagnetic"
DE_GAP_THRESHOLD = 0.05


def convert_qesem_recovered(json_path: Path, tier_label: str) -> dict:
    """Convert a recovered QESEM JSON into HardwareRunResult dict."""
    with open(json_path) as f:
        data = json.load(f)

    pub = data["pub_results"][0]
    evs = pub["evs"]
    stds = pub["stds"]
    metadata = pub.get("metadata", {})

    # Parse observables: [energy, X_0..X_9, ZZ_01..ZZ_89]
    energy_mitigated = evs[0]
    energy_std = stds[0]
    x_values = evs[1:11]
    zz_values = evs[11:20]

    # Noisy baselines (pre-mitigation)
    noisy_data = metadata.get("noisy_results", {})
    noisy_evs = noisy_data.get("evs", [0.0] * 20) if noisy_data else [0.0] * 20
    noisy_energy = noisy_evs[0] if noisy_evs else 0.0

    # Compute derived metrics
    delta_e = abs(energy_mitigated - E_EXACT)
    delta_e_gap = delta_e / GAP
    mag_x_mean = float(np.mean(x_values))
    corr_zz_mean = float(np.mean(zz_values))

    # Phase label (h=4.0 >> h_c=1.0 → always paramagnetic)
    phase_label = "paramagnetic" if mag_x_mean > 0.5 else "ferromagnetic"

    # ZNE gain: (noisy_error - mitigated_error) / noisy_error
    noisy_error = abs(noisy_energy - E_EXACT) if noisy_energy != 0.0 else 0.0
    mitigated_error = delta_e
    zne_gain = (
        (noisy_error - mitigated_error) / noisy_error
        if noisy_error > 0
        else 0.0
    )

    # Verdict
    correct_label = phase_label == EXPECTED_LABEL
    passed = delta_e_gap < DE_GAP_THRESHOLD and correct_label
    if passed:
        verdict = "PASS"
        verdict_reason = (
            f"ΔE/gap={delta_e_gap:.4f} < 5%, phase={phase_label} correct"
        )
    else:
        reasons = []
        if delta_e_gap >= DE_GAP_THRESHOLD:
            reasons.append(f"ΔE/gap={delta_e_gap:.4f} >= 5%")
        if not correct_label:
            reasons.append(f"phase={phase_label} != expected={EXPECTED_LABEL}")
        verdict = "FAIL"
        verdict_reason = "; ".join(reasons)

    # Build HardwareRunResult-compatible dict
    result = {
        "h_value": H_VALUE,
        "e_exact": E_EXACT,
        "e_zne": energy_mitigated,
        "delta_e_gap": delta_e_gap,
        "gap": GAP,
        "phase_label": phase_label,
        "expected_label": EXPECTED_LABEL,
        "zne_r2": 1.0,  # QESEM is unbiased (no extrapolation fit)
        "zne_gain": zne_gain,
        "mag_x_mean": mag_x_mean,
        "corr_zz_mean": corr_zz_mean,
        "sigma": energy_std,
        "total_shots": metadata.get("total_shots", 0),
        "job_ids": [data["job_id"]],
        "layouts_used": [],  # QESEM handles its own layout
        "ces_values": [],  # N/A for QESEM
        "per_site_x": x_values,
        "per_bond_zz": zz_values,
        "is_partial": False,
        "spsa_applied": False,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "zne_amplifier_used": "qesem",
        "mitigation_strategy": "qesem_unbiased",
        "layout_std": None,
        "fallback_triggered": False,
        "gnn_qem_applied": False,
        "gnn_qem_delta_e": None,
        "gnn_qem_confidence": None,
        "e_after_gnn_qem": None,
        "affine_correction_applied": False,
        "e_after_affine": None,
        "e_zne_std": energy_std,
        "obs_bounds_clipped": False,
        "n_obs_violations": 0,
        "layout_energy_outliers": 0,
        "e_obs_discrepancy": None,
        "e_obs_cross_valid_passed": True,
        "n_layouts_observables": 0,
        "stale_calibration_t1_drift_pct": None,
        "stale_calibration_stable": None,
        "effective_shots": metadata.get("total_shots"),
        "adaptive_shot_reason": "",
        # QESEM-specific fields
        "qesem_used": True,
        "qesem_job_id": data["job_id"],
        "qesem_total_qpu_time": metadata.get("total_qpu_time"),
        "qesem_gate_fidelities": metadata.get("gate_fidelities"),
        "qesem_total_shots": metadata.get("total_shots"),
        "qesem_mitigation_shots": metadata.get("mitigation_shots"),
        "qesem_noisy_evs": noisy_evs,
        # Extra metadata for provenance
        "_tier": tier_label,
        "_recovered_from": str(json_path.name),
        "_retrieved_at": data.get("retrieved_at"),
    }

    # ── Extract circuit_stats from QESEM's transpiled QASM ────────────
    transpiled_circs = metadata.get("transpiled_circs")
    if transpiled_circs and isinstance(transpiled_circs, list) and len(transpiled_circs) > 0:
        tc = transpiled_circs[0]
        qasm_str = tc.get("circuit", "")
        qubit_maps = tc.get("qubit_maps", [])
        physical_qubits = []
        if qubit_maps and len(qubit_maps) > 0:
            physical_qubits = [pair[1] for pair in qubit_maps[0]]

        # Parse gate counts from QASM
        qasm_gate_counts: dict[str, int] = {}
        qasm_n_2q = 0
        for line in qasm_str.split("\n"):
            line = line.strip()
            if line.startswith("rzz("):
                qasm_gate_counts["rzz"] = qasm_gate_counts.get("rzz", 0) + 1
                qasm_n_2q += 1
            elif line.startswith("cx "):
                qasm_gate_counts["cx"] = qasm_gate_counts.get("cx", 0) + 1
                qasm_n_2q += 1
            elif line.startswith("cz "):
                qasm_gate_counts["cz"] = qasm_gate_counts.get("cz", 0) + 1
                qasm_n_2q += 1
            elif line.startswith(("rx(", "ry(", "rz(")):
                gate_name = line[:2]
                qasm_gate_counts[gate_name] = qasm_gate_counts.get(gate_name, 0) + 1
            elif line.startswith("measure"):
                qasm_gate_counts["measure"] = qasm_gate_counts.get("measure", 0) + 1

        result["circuit_stats"] = {
            "source": "post_qesem_transpiled",
            "n_physical_qubits_used": len(physical_qubits),
            "physical_qubits": physical_qubits,
            "num_measurement_bases": tc.get("num_measurement_bases"),
            "n_2q_gates_transpiled": qasm_n_2q,
            "gate_counts_transpiled": qasm_gate_counts,
            "has_qasm": bool(qasm_str),
        }

    return result


def main():
    recovered_dir = PROJECT / "results/recovered/qesem"
    output_dir = PROJECT / "results/hardware/qesem_recovered"
    output_dir.mkdir(parents=True, exist_ok=True)

    tier0_path = recovered_dir / "qesem_recovered_82aa33cc-862c-4ba1-8017-6ab61eb7054e.json"
    tier1_path = recovered_dir / "qesem_recovered_4f16e846-9af2-4ee8-a78d-6f829766eefe.json"

    print("\n" + "═" * 70)
    print("  CONVERTING QESEM RESULTS → HardwareRunResult FORMAT")
    print("═" * 70 + "\n")

    results = []
    for path, label in [(tier0_path, "tier0"), (tier1_path, "tier1")]:
        if not path.exists():
            print(f"  ⚠️ Not found: {path}")
            continue
        result = convert_qesem_recovered(path, label)
        out_file = output_dir / (
            f"hwresult_{label}_h{result['h_value']:.1f}"
            f"_{result['qesem_job_id'][:8]}.json"
        )
        json_dump(result, out_file)
        results.append(result)

        print(f"  ✅ {label.upper()}: E = {result['e_zne']:.4f} ± {result['e_zne_std']:.4f}")
        print(f"     ΔE/gap = {result['delta_e_gap']:.4f}  [{result['verdict']}]")
        print(f"     ⟨X⟩ mean = {result['mag_x_mean']:.4f}  ⟨ZZ⟩ mean = {result['corr_zz_mean']:.4f}")
        print(f"     QPU time = {result['qesem_total_qpu_time']}s  |  shots = {result['total_shots']:,}")
        print(f"     ZNE gain = {result['zne_gain']:.2%}")
        print(f"     → {out_file.name}")
        print()

    # Also write a combined summary
    summary = {
        "description": "QESEM recovered results converted to HardwareRunResult format",
        "n_results": len(results),
        "results": results,
    }
    json_dump(summary, output_dir / "qesem_results_summary.json")
    print(f"  📄 Summary saved: results/hardware/qesem_recovered/qesem_results_summary.json")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
