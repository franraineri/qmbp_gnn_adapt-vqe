#!/usr/bin/env python3
"""Recover QESEM job results from IBM Qiskit Functions catalog.

QESEM jobs are submitted via QiskitFunctionsCatalog (not QiskitRuntimeService).
They must be recovered via the catalog API, not the standard runtime job retrieval.

Usage:
    .venv/bin/python scripts/recover_qesem_job.py 82aa33cc-862c-4ba1-8017-6ab61eb7054e
    .venv/bin/python scripts/recover_qesem_job.py 82aa33cc-862c-4ba1-8017-6ab61eb7054e --save results/recovered/
"""

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from qmbp_simulation.utils.helpers import json_dump


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Recover QESEM job results")
    parser.add_argument("job_id", help="QESEM job ID from QiskitFunctionsCatalog")
    parser.add_argument(
        "--save", type=str, default="results/recovered", help="Directory to save recovered results"
    )
    parser.add_argument(
        "--pub-format",
        action="store_true",
        help="Also save in pub_results format (for convert script)",
    )
    args = parser.parse_args()

    # Validate job_id format (UUID4)
    import re

    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
    )
    if not uuid_pattern.match(args.job_id):
        print(
            f"  ⚠️ job_id '{args.job_id}' doesn't match UUID format. "
            f"QESEM jobs use UUID4 IDs (e.g., 82aa33cc-862c-4ba1-8017-6ab61eb7054e)."
        )
        # Don't abort — some jobs may have non-standard IDs

    token = os.environ.get("IBM_KEY")
    instance = os.environ.get("IBM_INSTANCE_CRN")

    if not token or not instance:
        print("ERROR: Set IBM_KEY and IBM_INSTANCE_CRN env vars")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  QESEM Job Recovery — via QiskitFunctionsCatalog")
    print(f"{'=' * 60}")
    print(f"  Job ID: {args.job_id}")
    print("  [DEBUG] main: starting recovery from catalog")

    from qiskit_ibm_catalog import QiskitFunctionsCatalog

    catalog = QiskitFunctionsCatalog(
        channel="ibm_quantum_platform",
        token=token,
        instance=instance,
    )

    print("  Retrieving job...")
    # The catalog provides get_job_by_id() to retrieve Qiskit Function jobs
    job = catalog.get_job_by_id(args.job_id)

    status = job.status()
    print(f"  Status: {status}")

    if str(status).upper() not in ("DONE", "COMPLETED"):
        print(f"\n  ⚠️  Job not completed (status={status})")
        if str(status).upper() in ("ERROR", "CANCELLED"):
            try:
                err = job.result()
                print(f"  Error details: {err}")
            except Exception as e:
                print(f"  Could not retrieve error: {e}")
        sys.exit(1)

    # Retrieve result
    try:
        result = job.result()
        pub_result = result[0]
        evs = np.atleast_1d(pub_result.data.evs)
        stds = np.atleast_1d(pub_result.data.stds)
        metadata = pub_result.metadata

        print("\n  ✅ QESEM Results Retrieved Successfully!")
        print(f"  {'─' * 50}")
        print(f"  N observables: {len(evs)}")
        print(f"  Energy (mitigated): {evs[0]:.6f} ± {stds[0]:.6f}")

        if len(evs) >= 11:
            print(f"  ⟨X⟩ mean: {np.mean(evs[1:11]):.4f}")
        if len(evs) >= 20:
            print(f"  ⟨ZZ⟩ mean: {np.mean(evs[11:20]):.4f}")

        # Metadata
        print("\n  ── QESEM Metadata ──")
        for key in ("total_qpu_time", "total_shots", "mitigation_shots"):
            val = metadata.get(key)
            if val is not None:
                print(f"  {key}: {val}")

        gate_fid = metadata.get("gate_fidelities")
        if gate_fid:
            print(f"  gate_fidelities: {gate_fid}")

        # Noisy results
        noisy = metadata.get("noisy_results")
        noisy_energy = None
        if noisy is not None:
            if hasattr(noisy, "evs"):
                noisy_evs = np.atleast_1d(noisy.evs)
                noisy_energy = float(noisy_evs[0])
                print("\n  ── Noisy (pre-mitigation) ──")
                print(f"  Noisy energy: {noisy_energy:.6f}")
                print(f"  Mitigation delta: {abs(noisy_energy - evs[0]):.6f}")

        # Save
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)

        output = {
            "job_id": args.job_id,
            "status": str(status),
            "retrieved_at": datetime.now(UTC).isoformat(),
            "strategy": "qesem_unbiased",
            "energy_mitigated": float(evs[0]),
            "energy_std": float(stds[0]),
            "evs": evs.tolist(),
            "stds": stds.tolist(),
            "x_values": evs[1:11].tolist() if len(evs) >= 11 else [],
            "zz_values": evs[11:20].tolist() if len(evs) >= 20 else [],
            "noisy_energy": noisy_energy,
            "metadata": {},
        }
        # Safe metadata extraction (skip non-serializable objects)
        for k, v in metadata.items():
            if k == "transpiled_circs":
                # Preserve transpiled circuit info for provenance:
                # qubit_maps, QASM, num_measurement_bases are all JSON-serializable.
                if v and isinstance(v, list):
                    serialized_circs = []
                    for circ_info in v:
                        if isinstance(circ_info, dict):
                            serialized_circs.append(circ_info)
                        elif hasattr(circ_info, "__dict__"):
                            # Object — try to extract dict-like representation
                            serialized_circs.append(str(circ_info))
                        else:
                            serialized_circs.append(str(circ_info))
                    output["metadata"][k] = serialized_circs
                else:
                    output["metadata"][k] = None
            elif k == "noisy_results":
                # Preserve noisy_results as a proper dict with evs/stds arrays.
                # Without this, downstream convert scripts cannot compute zne_gain.
                if hasattr(v, "evs"):
                    noisy_dict = {
                        "evs": np.atleast_1d(v.evs).tolist(),
                    }
                    if hasattr(v, "stds"):
                        noisy_dict["stds"] = np.atleast_1d(v.stds).tolist()
                    output["metadata"][k] = noisy_dict
                elif isinstance(v, dict):
                    output["metadata"][k] = v
                else:
                    # Last resort: store string repr with a warning marker
                    output["metadata"][k] = {"_unparsed_repr": str(v)[:500]}
                    print(f"  ⚠️ noisy_results stored as string repr (type={type(v).__name__})")
            elif hasattr(v, "tolist"):
                output["metadata"][k] = v.tolist()
            elif isinstance(v, (int, float, str, bool, list, dict, type(None))):
                output["metadata"][k] = v
            else:
                output["metadata"][k] = str(v)

        out_path = save_dir / f"qesem_recovered_{args.job_id}.json"
        json_dump(output, out_path)
        print(f"\n  💾 Saved to: {out_path}")

        # Also save in pub_results format (compatible with convert script)
        if args.pub_format:
            pub_output = {
                "job_id": args.job_id,
                "status": str(status),
                "retrieved_at": datetime.now(UTC).isoformat(),
                "pub_results": [
                    {
                        "pub_idx": 0,
                        "evs": evs.tolist(),
                        "stds": stds.tolist(),
                        "n_observables": len(evs),
                        "metadata": output["metadata"],
                    }
                ],
                "n_pubs": 1,
            }
            pub_dir = save_dir / "qesem"
            pub_dir.mkdir(parents=True, exist_ok=True)
            pub_path = pub_dir / f"qesem_recovered_{args.job_id}.json"
            json_dump(pub_output, pub_path)
            print(f"  💾 pub_results format: {pub_path}")

    except Exception as e:
        print(f"\n  ❌ Failed to retrieve result: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
