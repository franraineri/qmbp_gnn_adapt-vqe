#!/usr/bin/env python3
"""Recover results from completed QESEM (Qiskit Functions) jobs.

QESEM jobs are Qiskit Serverless workloads — they use QiskitFunctionsCatalog,
NOT QiskitRuntimeService. The recovery mechanism is different from standard
Runtime jobs.

Usage:
    # Recover specific job IDs
    python scripts/recover_qesem_jobs.py 82aa33cc-862c-4ba1-8017-6ab61eb7054e

    # Recover multiple jobs
    python scripts/recover_qesem_jobs.py \
        82aa33cc-862c-4ba1-8017-6ab61eb7054e \
        4f16e846-9af2-4ee8-a78d-6f829766eefe

    # Save to directory
    python scripts/recover_qesem_jobs.py --save results/recovered/qesem/ \
        82aa33cc-862c-4ba1-8017-6ab61eb7054e

    # List recent QESEM jobs (find IDs you forgot)
    python scripts/recover_qesem_jobs.py --list-recent 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from qmbp_simulation.utils.helpers import json_dump, json_serialize


def connect_catalog():
    """Connect to QiskitFunctionsCatalog using environment credentials."""
    from qiskit_ibm_catalog import QiskitFunctionsCatalog

    token = os.environ.get("IBM_KEY")
    instance = os.environ.get("IBM_INSTANCE_CRN")
    if not token or not instance:
        print("ERROR: Set IBM_KEY and IBM_INSTANCE_CRN environment variables.")
        sys.exit(1)

    catalog = QiskitFunctionsCatalog(
        channel="ibm_quantum_platform",
        token=token,
        instance=instance,
    )
    return catalog


def list_recent_jobs(catalog, n: int = 10):
    """List recent QESEM (qedma/qesem) jobs from the catalog."""
    print(f"\n{'═' * 70}")
    print(f"  RECENT QESEM JOBS (last {n})")
    print(f"{'═' * 70}\n")

    # Load the QESEM function to access its jobs
    qesem_fn = catalog.load("qedma/qesem")

    # List jobs — the API may vary, try common patterns
    jobs = []
    if hasattr(qesem_fn, "jobs"):
        jobs = qesem_fn.jobs()
    elif hasattr(catalog, "jobs"):
        jobs = catalog.jobs()

    if not jobs:
        print("  No jobs found (or API does not support listing).")
        print("  Try recovering by job ID directly if you have it.")
        return

    for i, job in enumerate(jobs[:n]):
        job_id = job.job_id if hasattr(job, "job_id") else str(job)
        status = job.status() if hasattr(job, "status") else "?"
        print(f"  [{i}] {job_id}  status={status}")

    print(f"\n  Total shown: {min(n, len(jobs))}")
    print(f"{'═' * 70}\n")


def recover_qesem_job(catalog, job_id: str, verbose: bool = False) -> dict:
    """Recover a QESEM job result by ID.

    QESEM jobs are Qiskit Serverless workloads. We retrieve them via the
    catalog's job retrieval mechanism (not QiskitRuntimeService).

    Parameters
    ----------
    catalog : QiskitFunctionsCatalog
        Connected catalog instance.
    job_id : str
        QESEM job UUID (e.g., '82aa33cc-862c-4ba1-8017-6ab61eb7054e').
    verbose : bool
        Print detailed per-observable values.

    Returns
    -------
    dict
        Complete job data including status, results, and metadata.
    """
    print(f"\n{'═' * 70}")
    print(f"  RECOVERING QESEM JOB: {job_id}")
    print(f"{'═' * 70}")

    # Retrieve the job object from the catalog
    # QiskitFunctionsCatalog uses .job(job_id) or the function's .job(job_id)
    job = None
    qesem_fn = catalog.load("qedma/qesem")

    # Try multiple retrieval patterns (API may vary by version)
    for retrieval_fn in [
        lambda: qesem_fn.job(job_id),
        lambda: catalog.job(job_id),
    ]:
        try:
            job = retrieval_fn()
            if job is not None:
                break
        except (AttributeError, TypeError):
            continue
        except Exception as e:
            print(f"  Retrieval attempt failed: {e}")
            continue

    if job is None:
        print(f"\n  ❌ Could not retrieve job {job_id}")
        print("     Possible causes:")
        print("     - Job ID is incorrect")
        print("     - Job was submitted under a different instance/account")
        print("     - The catalog API version doesn't support job retrieval")
        return {"job_id": job_id, "status": "NOT_FOUND", "results": None}

    # ── Status check ──────────────────────────────────────────────────
    status = "UNKNOWN"
    try:
        status = job.status()
        if hasattr(status, "name"):
            status = status.name
        status = str(status).upper()
    except Exception as e:
        print(f"  (status check failed: {e})")

    print(f"\n  Status: {status}")

    if status not in ("DONE", "COMPLETED"):
        print(f"\n  ⚠️ Job is not completed (status={status}).")
        if status in ("QUEUED", "RUNNING", "INITIALIZING"):
            print("     The job is still executing. Wait and retry later.")
        elif status in ("ERROR", "CANCELLED", "FAILED"):
            print("     The job failed or was cancelled.")
            # Try to get logs
            try:
                logs = job.logs() if hasattr(job, "logs") else None
                if logs:
                    print(f"\n  ── Logs (last 500 chars) ──")
                    print(f"  {logs[-500:]}")
            except Exception:
                pass
        return {"job_id": job_id, "status": status, "results": None}

    # ── Retrieve result ───────────────────────────────────────────────
    print(f"\n  Fetching result (this may take a moment)...")
    try:
        result = job.result()
    except Exception as e:
        print(f"\n  ❌ Failed to retrieve result: {e}")
        print("     The job may still be running, or there was a network error.")
        print("     Try again in a few minutes.")
        return {"job_id": job_id, "status": status, "results": None, "error": str(e)}

    # ── Parse QESEM result ────────────────────────────────────────────
    print(f"\n  ── QESEM Results ──")

    pub_results = []
    try:
        # QESEM returns PubResult objects similar to Estimator
        for idx, pub_result in enumerate(result):
            evs = np.atleast_1d(pub_result.data.evs)
            stds = np.atleast_1d(
                pub_result.data.stds
                if hasattr(pub_result.data, "stds")
                else np.zeros_like(evs)
            )
            metadata = pub_result.metadata if hasattr(pub_result, "metadata") else {}

            pub_data = {
                "pub_idx": idx,
                "evs": evs.tolist(),
                "stds": stds.tolist(),
                "n_observables": len(evs),
                "metadata": {},
            }

            # Extract QESEM-specific metadata
            if isinstance(metadata, dict):
                for key in [
                    "total_qpu_time", "gate_fidelities", "total_shots",
                    "mitigation_shots", "transpiled_circs", "noisy_results",
                ]:
                    val = metadata.get(key, None)
                    if val is not None:
                        if hasattr(val, "tolist"):
                            pub_data["metadata"][key] = val.tolist()
                        elif hasattr(val, "evs"):
                            # noisy_results object
                            pub_data["metadata"][key] = {
                                "evs": np.atleast_1d(val.evs).tolist(),
                                "stds": (
                                    np.atleast_1d(val.stds).tolist()
                                    if hasattr(val, "stds") else None
                                ),
                            }
                        else:
                            pub_data["metadata"][key] = val

            pub_results.append(pub_data)

            # ── Compute circuit_stats from QESEM's transpiled circuit ─────
            # QESEM returns the transpiled QASM in metadata. Parse it to get
            # post-QESEM circuit stats for validator compatibility.
            transpiled_circs = pub_data["metadata"].get("transpiled_circs")
            if transpiled_circs and isinstance(transpiled_circs, list) and len(transpiled_circs) > 0:
                tc = transpiled_circs[0]
                qasm_str = tc.get("circuit", "")
                qubit_maps = tc.get("qubit_maps", [])
                physical_qubits = []
                if qubit_maps and len(qubit_maps) > 0:
                    physical_qubits = [pair[1] for pair in qubit_maps[0]]

                # Parse gate counts from QASM (lightweight, no Qiskit import needed)
                qasm_gate_counts: dict[str, int] = {}
                qasm_n_2q = 0
                qasm_depth_lines = 0
                for line in qasm_str.split("\n"):
                    line = line.strip()
                    if line.startswith("rzz("):
                        qasm_gate_counts["rzz"] = qasm_gate_counts.get("rzz", 0) + 1
                        qasm_n_2q += 1
                    elif line.startswith("cx ") or line.startswith("cx("):
                        qasm_gate_counts["cx"] = qasm_gate_counts.get("cx", 0) + 1
                        qasm_n_2q += 1
                    elif line.startswith("cz "):
                        qasm_gate_counts["cz"] = qasm_gate_counts.get("cz", 0) + 1
                        qasm_n_2q += 1
                    elif line.startswith("rx(") or line.startswith("ry(") or line.startswith("rz("):
                        gate_name = line[:2]
                        qasm_gate_counts[gate_name] = qasm_gate_counts.get(gate_name, 0) + 1
                    elif line.startswith("measure"):
                        qasm_gate_counts["measure"] = qasm_gate_counts.get("measure", 0) + 1
                    if line and not line.startswith(("//", "OPENQASM", "include", "gate", "bit", "qubit", "}", "{")):
                        qasm_depth_lines += 1

                pub_data["circuit_stats"] = {
                    "source": "post_qesem_transpiled",
                    "n_physical_qubits_used": len(physical_qubits),
                    "physical_qubits": physical_qubits,
                    "num_measurement_bases": tc.get("num_measurement_bases"),
                    "n_2q_gates_transpiled": qasm_n_2q,
                    "gate_counts_transpiled": qasm_gate_counts,
                    "has_qasm": bool(qasm_str),
                }

            # Print summary
            print(f"\n  PUB {idx}: {len(evs)} expectation values")
            # Interpret: [energy, X_0..X_9, ZZ_01..ZZ_89]
            if len(evs) >= 1:
                print(f"    Energy (mitigated):  {evs[0]:.6f} ± {stds[0]:.6f}")
            if len(evs) >= 11:
                x_vals = evs[1:11]
                print(f"    ⟨X⟩ mean:            {np.mean(x_vals):.6f}")
                if verbose:
                    for i, x in enumerate(x_vals):
                        print(f"      X_{i} = {x:.6f} ± {stds[1+i]:.6f}")
            if len(evs) >= 20:
                zz_vals = evs[11:20]
                print(f"    ⟨ZZ⟩ mean:           {np.mean(zz_vals):.6f}")
                if verbose:
                    for i, zz in enumerate(zz_vals):
                        print(f"      ZZ_{i},{i+1} = {zz:.6f} ± {stds[11+i]:.6f}")

            # QPU time
            qpu_time = pub_data["metadata"].get("total_qpu_time")
            if qpu_time:
                print(f"    QPU time:            {qpu_time:.1f}s")
            total_shots = pub_data["metadata"].get("total_shots")
            if total_shots:
                print(f"    Total shots:         {total_shots:,}")

    except Exception as e:
        print(f"\n  ❌ Error parsing results: {e}")
        pub_results = [{"error": str(e)}]

    # ── Logs (for debugging) ──────────────────────────────────────────
    logs = None
    try:
        logs = job.logs() if hasattr(job, "logs") else None
        if logs and verbose:
            print(f"\n  ── Server Logs (last 1000 chars) ──")
            print(f"  {logs[-1000:]}")
    except Exception:
        pass

    # ── Assemble output ───────────────────────────────────────────────
    output = {
        "job_id": job_id,
        "status": status,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "pub_results": pub_results,
        "n_pubs": len(pub_results),
        "logs_tail": logs[-500:] if logs else None,
    }

    print(f"\n{'═' * 70}")
    print(f"  ✅ Successfully recovered QESEM job {job_id}")
    print(f"     {len(pub_results)} PUB result(s)")
    print(f"{'═' * 70}\n")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Recover results from completed QESEM (Qiskit Functions) jobs"
    )
    parser.add_argument(
        "job_ids",
        nargs="*",
        help="QESEM job UUIDs to recover",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="results/recovered/qesem",
        help="Directory to save recovered results (default: results/recovered/qesem/)",
    )
    parser.add_argument(
        "--list-recent",
        type=int,
        default=None,
        metavar="N",
        help="List N most recent QESEM jobs instead of recovering",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed per-observable values and logs",
    )
    args = parser.parse_args()

    catalog = connect_catalog()

    # List mode
    if args.list_recent is not None:
        list_recent_jobs(catalog, n=args.list_recent)
        return

    # Recovery mode
    if not args.job_ids:
        # Default: recover the two known jobs from today's runs
        args.job_ids = [
            "82aa33cc-862c-4ba1-8017-6ab61eb7054e",  # Tier 0
            "4f16e846-9af2-4ee8-a78d-6f829766eefe",  # Tier 1
        ]
        print("  No job IDs specified. Recovering today's known QESEM jobs:")
        for jid in args.job_ids:
            print(f"    - {jid}")

    all_results = []
    for job_id in args.job_ids:
        result = recover_qesem_job(catalog, job_id.strip(), verbose=args.verbose)
        all_results.append(result)

    # Save results
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)
    for result in all_results:
        if result.get("pub_results"):
            jid = result["job_id"]
            out_path = save_dir / f"qesem_recovered_{jid}.json"
            json_dump(result, out_path)
            print(f"  💾 Saved: {out_path}")

    # Summary
    if len(all_results) > 1:
        print(f"\n{'─' * 50}")
        print(f"  Summary: {len(all_results)} jobs")
        for r in all_results:
            icon = "✅" if r.get("pub_results") else "❌"
            print(f"    {icon} {r['job_id']}: {r['status']}")


if __name__ == "__main__":
    main()
