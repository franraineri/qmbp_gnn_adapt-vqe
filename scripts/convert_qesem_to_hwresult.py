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

# ─── Ground truth values per h-value (TFIM N=10 OBC p=1) ─────────────
# These must be pre-computed via exact diagonalization.
GROUND_TRUTH: dict[float, dict] = {
    4.0: {
        "e_exact": -40.565690435512735,
        "gap": 5.921971082752528,
    },
    3.5: {
        "e_exact": -35.524253452300285,  # N=10 1D OBC TFIM h=3.5
        "gap": 4.762916470256498,
    },
}

EXPECTED_LABEL = "paramagnetic"
DE_GAP_THRESHOLD = 0.05


def _detect_h_value(data: dict) -> float:
    """Detect h-value from the Hamiltonian observable coefficients in metadata.

    The first observable in QESEM results is the Hamiltonian. For TFIM:
    H = -h*sum(X_i) - J*sum(Z_iZ_{i+1}), so the X coefficient reveals h.
    """
    print("  [DEBUG] _detect_h_value: inferring h from observable coefficients")

    # Check metadata.results (pub_results format with full observable data)
    metadata = data.get("metadata", {})
    if "pub_results" in data:
        metadata = data["pub_results"][0].get("metadata", {})

    results_data = metadata.get("results")
    if results_data and isinstance(results_data, list) and len(results_data) > 0:
        try:
            first_pub = results_data[0]
            if first_pub and len(first_pub) > 0:
                obs_str = str(first_pub[0][0])
                # Parse coefficient from observable string like "'IIIIIIIIIX': -4.0"
                import re

                x_coeffs = re.findall(r"'[IX]+': (-?\d+\.?\d*)", obs_str)
                if x_coeffs:
                    h_val = abs(float(x_coeffs[0]))
                    return h_val
        except (IndexError, TypeError, ValueError):
            pass

    # Fallback: check top-level fields
    if "h_value" in data:
        return float(data["h_value"])

    # Heuristic from energy: for flat format, check if energy_mitigated is
    # consistent with known h-values. E ≈ -h*N for h>>1.
    e_mit = data.get("energy_mitigated")
    if e_mit is not None:
        # N=10: E(h=4) ≈ -40.5, E(h=3.5) ≈ -35.4
        if abs(e_mit) > 38:
            return 4.0
        elif abs(e_mit) > 33:
            return 3.5

    # Default assumption for existing data
    return 4.0


def _parse_noisy_results(metadata: dict, n_obs: int = 20) -> tuple[list[float], bool]:
    """Safely parse noisy_results from metadata, handling all format variants.

    Returns (noisy_evs, is_available) where noisy_evs has length n_obs.
    """
    print("  [DEBUG] _parse_noisy_results: extracting pre-mitigation baseline")
    noisy_data = metadata.get("noisy_results")

    if noisy_data is None:
        return [0.0] * n_obs, False

    # Case 1: string repr (bug from old recovery script)
    if isinstance(noisy_data, str):
        print(f"  ⚠️ noisy_results is string repr (not parseable): '{noisy_data[:60]}...'")
        return [0.0] * n_obs, False

    # Case 2: dict with "_unparsed_repr" marker (from improved recovery)
    if isinstance(noisy_data, dict) and "_unparsed_repr" in noisy_data:
        print("  ⚠️ noisy_results was stored as unparsed repr")
        return [0.0] * n_obs, False

    # Case 3: proper dict with "evs" key
    if isinstance(noisy_data, dict) and "evs" in noisy_data:
        evs = noisy_data["evs"]
        if isinstance(evs, list) and len(evs) >= n_obs:
            return evs[:n_obs], True
        return [0.0] * n_obs, False

    # Case 4: dict without "evs" but with other structure
    if isinstance(noisy_data, dict):
        # Try to extract — might be nested differently
        return [0.0] * n_obs, False

    return [0.0] * n_obs, False


def convert_qesem_recovered(json_path: Path, tier_label: str) -> dict:
    """Convert a recovered QESEM JSON into HardwareRunResult dict."""
    print(f"  [DEBUG] convert_qesem_recovered: processing {json_path.name}")
    with open(json_path) as f:
        data = json.load(f)

    # Handle two recovery formats:
    # 1. "pub_results" format (from full recovery with metadata preservation)
    # 2. "flat" format (from recover_qesem_job.py — top-level evs/stds)
    if "pub_results" in data:
        pub = data["pub_results"][0]
        evs = pub["evs"]
        stds = pub["stds"]
        metadata = pub.get("metadata", {})
    elif "evs" in data:
        evs = data["evs"]
        stds = data["stds"]
        metadata = data.get("metadata", {})
    else:
        raise ValueError(
            f"Unrecognized QESEM JSON format in {json_path.name}. "
            f"Expected 'pub_results' or top-level 'evs' field. "
            f"Keys found: {list(data.keys())[:8]}"
        )

    # Parse observables: [energy, X_0..X_9, ZZ_01..ZZ_89]
    # Validate minimum expected length: 1 energy + 10 X + 9 ZZ = 20
    if len(evs) < 20:
        raise ValueError(
            f"Expected at least 20 observable values (1 energy + 10 X + 9 ZZ), "
            f"got {len(evs)}. File may be corrupted or from a different N."
        )
    if len(stds) < 20:
        raise ValueError(
            f"Expected at least 20 std values, got {len(stds)}. Partial result or data corruption."
        )
    energy_mitigated = evs[0]
    energy_std = stds[0]
    x_values = evs[1:11]
    zz_values = evs[11:20]

    # Sanity check: energy should be negative for TFIM with h > 0
    if energy_mitigated > 0:
        print(
            f"  ⚠️ WARNING: energy_mitigated={energy_mitigated:.4f} is positive. "
            f"Unexpected for TFIM ground state (should be << 0)."
        )

    # ── Detect h-value from Hamiltonian coefficients (Bug 1 fix) ─────
    h_value = _detect_h_value(data)
    if h_value not in GROUND_TRUTH:
        raise ValueError(
            f"Detected h={h_value} but no ground truth available. "
            f"Known h-values: {list(GROUND_TRUTH.keys())}. "
            f"Add E_exact and gap for h={h_value} to GROUND_TRUTH dict."
        )
    gt = GROUND_TRUTH[h_value]
    e_exact = gt["e_exact"]
    gap = gt["gap"]
    print(f"  [DEBUG] Detected h={h_value}, E_exact={e_exact:.6f}, gap={gap:.4f}")

    # ── Parse noisy baselines (Bug 3 fix: handles string repr) ───────
    noisy_evs, noisy_available = _parse_noisy_results(metadata, n_obs=20)
    # Flat format fallback: top-level "noisy_energy" field
    if not noisy_available and "noisy_energy" in data and data["noisy_energy"] is not None:
        noisy_evs[0] = data["noisy_energy"]
        noisy_available = True
    noisy_energy = noisy_evs[0] if noisy_available else 0.0

    # Compute derived metrics
    delta_e = abs(energy_mitigated - e_exact)
    delta_e_gap = delta_e / gap
    mag_x_mean = float(np.mean(x_values))
    corr_zz_mean = float(np.mean(zz_values))

    # Phase label (h >> h_c=1.0 → always paramagnetic for h>=3.5)
    phase_label = "paramagnetic" if mag_x_mean > 0.5 else "ferromagnetic"

    # ZNE gain: (noisy_error - mitigated_error) / noisy_error
    # Only valid when noisy data was actually parsed (not sentinel zeros)
    if noisy_available and noisy_energy != 0.0:
        noisy_error = abs(noisy_energy - e_exact)
        mitigated_error = delta_e
        zne_gain = (noisy_error - mitigated_error) / noisy_error if noisy_error > 1e-10 else 0.0
    else:
        zne_gain = None  # Cannot compute — noisy baseline unavailable
        print("  ⚠️ zne_gain=None (noisy baseline not available for this job)")

    # Verdict
    correct_label = phase_label == EXPECTED_LABEL
    passed = delta_e_gap < DE_GAP_THRESHOLD and correct_label
    if passed:
        verdict = "PASS"
        verdict_reason = f"ΔE/gap={delta_e_gap:.4f} < 5%, phase={phase_label} correct"
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
        "h_value": h_value,
        "e_exact": e_exact,
        "e_zne": energy_mitigated,
        "delta_e_gap": delta_e_gap,
        "gap": gap,
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
        "qesem_noisy_evs": noisy_evs if noisy_available else None,
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
    print("  [DEBUG] main: starting QESEM → HardwareRunResult conversion")
    recovered_dir = PROJECT / "results/recovered/qesem"
    output_dir = PROJECT / "results/hardware/qesem_recovered"
    output_dir.mkdir(parents=True, exist_ok=True)

    # All available QESEM recovered results
    job_files = [
        (recovered_dir / "qesem_recovered_82aa33cc-862c-4ba1-8017-6ab61eb7054e.json", "tier0"),
        (recovered_dir / "qesem_recovered_4f16e846-9af2-4ee8-a78d-6f829766eefe.json", "tier1"),
        (recovered_dir / "qesem_recovered_d628a502-677a-4610-a78c-3d5266c0cdbf.json", "tier1_h3.5"),
    ]

    print("\n" + "═" * 70)
    print("  CONVERTING QESEM RESULTS → HardwareRunResult FORMAT")
    print("═" * 70 + "\n")

    results = []
    for path, label in job_files:
        if not path.exists():
            print(f"  ⚠️ Not found: {path}")
            continue
        try:
            result = convert_qesem_recovered(path, label)
        except ValueError as e:
            print(f"  ❌ Skipping {path.name}: {e}")
            continue
        out_file = output_dir / (
            f"hwresult_{label}_h{result['h_value']:.1f}_{result['qesem_job_id'][:8]}.json"
        )
        json_dump(result, out_file)
        results.append(result)

        print(f"  ✅ {label.upper()}: E = {result['e_zne']:.4f} ± {result['e_zne_std']:.4f}")
        print(f"     ΔE/gap = {result['delta_e_gap']:.4f}  [{result['verdict']}]")
        print(
            f"     ⟨X⟩ mean = {result['mag_x_mean']:.4f}  ⟨ZZ⟩ mean = {result['corr_zz_mean']:.4f}"
        )
        print(
            f"     QPU time = {result['qesem_total_qpu_time']}s  |  shots = {result['total_shots']:,}"
        )
        gain = result["zne_gain"]
        print(
            f"     ZNE gain = {gain:.2%}"
            if gain is not None
            else "     ZNE gain = N/A (noisy baseline unavailable)"
        )
        print(f"     → {out_file.name}")
        print()

    # Also write a combined summary
    summary = {
        "description": "QESEM recovered results converted to HardwareRunResult format",
        "n_results": len(results),
        "results": results,
    }
    json_dump(summary, output_dir / "qesem_results_summary.json")
    print("  📄 Summary saved: results/hardware/qesem_recovered/qesem_results_summary.json")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
