#!/usr/bin/env python3
"""Recover and analyze results from a completed IBM Runtime job.

When a local process times out but the QPU job finished server-side,
this script retrieves the result using the job ID.

Usage:
    python scripts/recover_job_result.py d8tche5bh0os73epdphg
    python scripts/recover_job_result.py d8tche5bh0os73epdphg --save results/recovered/
    python scripts/recover_job_result.py d8tche5bh0os73epdphg --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure project root is in path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from qmbp_simulation.utils.helpers import json_dump

# ═══════════════════════════════════════════════════════════════════════════════
# Reusable function for in-memory job saving (called from hardware runners)
# ═══════════════════════════════════════════════════════════════════════════════


def save_raw_job_output(job, save_dir: str | Path, label: str = "") -> Path | None:
    """Save the complete raw QPU output from an in-memory job object.

    Call this from any hardware runner right after a job completes.
    Extracts all available data (results, metrics, options, ZNE data)
    and saves to a single JSON file for later analysis.

    Parameters
    ----------
    job : RuntimeJobV2 or PrimitiveJob
        The completed job object (must have .result() available).
    save_dir : str | Path
        Directory to save the raw output JSON.
    label : str
        Optional label prefix for the filename (e.g., "tier1_h3p25").

    Returns
    -------
    Path | None
        Path to the saved file, or None if saving failed.
    """
    import numpy as np

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get job ID
    job_id = job.job_id() if hasattr(job, "job_id") else "local"

    # ── Status ────────────────────────────────────────────────────────
    status = "DONE"
    if hasattr(job, "status"):
        raw_status = job.status()
        status = raw_status.name if hasattr(raw_status, "name") else str(raw_status).upper()

    # ── Metrics ───────────────────────────────────────────────────────
    metrics_data = {}
    if hasattr(job, "metrics"):
        try:
            metrics = job.metrics()
            if isinstance(metrics, dict):
                usage = metrics.get("usage", {})
                timestamps = metrics.get("timestamps", {})
                metrics_data = {
                    "qpu_seconds": usage.get("quantum_seconds"),
                    "billed_seconds": usage.get("seconds"),
                    "created": timestamps.get("created"),
                    "running": timestamps.get("running"),
                    "finished": timestamps.get("finished"),
                }
                created = timestamps.get("created")
                running = timestamps.get("running")
                if created and running:
                    from datetime import datetime as _dt

                    try:
                        t_c = _dt.fromisoformat(created.replace("Z", "+00:00"))
                        t_r = _dt.fromisoformat(running.replace("Z", "+00:00"))
                        metrics_data["queue_wait_s"] = (t_r - t_c).total_seconds()
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass

    # ── Job metadata ──────────────────────────────────────────────────
    job_metadata = {}
    for attr in ("program_id", "session_id", "tags"):
        val = getattr(job, attr, None)
        if val is not None:
            job_metadata[attr] = val
    try:
        inputs = getattr(job, "inputs", None)
        if isinstance(inputs, dict):
            options_input = inputs.get("options", {})
            if options_input:
                job_metadata["submitted_options"] = options_input
    except Exception:
        pass

    # ── Results ───────────────────────────────────────────────────────
    pub_results = []
    try:
        job_result = job.result()
        for idx, pub_result in enumerate(job_result):
            evs = pub_result.data.evs
            stds = getattr(pub_result.data, "stds", None)
            evs_val = float(evs) if np.isscalar(evs) else evs.tolist()
            stds_val = (
                float(stds)
                if stds is not None and np.isscalar(stds)
                else (stds.tolist() if stds is not None else None)
            )
            pub_data = {"pub_idx": idx, "evs": evs_val, "stds": stds_val}
            # ZNE per-noise-factor and ensemble data
            for extra in ("evs_noise_factors", "stds_noise_factors", "ensemble_stds", "metadata"):
                val = getattr(pub_result.data, extra, None)
                if val is not None:
                    pub_data[extra] = val.tolist() if hasattr(val, "tolist") else val
            pub_results.append(pub_data)
    except Exception as e:
        pub_results = [{"error": str(e)}]

    # ── Backend ───────────────────────────────────────────────────────
    backend_name = None
    try:
        backend_name = job.backend().name
    except Exception:
        pass

    # ── Assemble and save ─────────────────────────────────────────────
    output = {
        "job_id": job_id,
        "status": status,
        "backend": backend_name,
        "saved_at": datetime.now(UTC).isoformat(),
        "metrics": metrics_data,
        "job_metadata": job_metadata,
        "pub_results": pub_results,
        "n_pubs": len(pub_results),
    }

    prefix = f"{label}_" if label else ""
    out_path = save_dir / f"{prefix}raw_qpu_{job_id}.json"
    try:
        json_dump(output, out_path)
        return out_path
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# CLI recovery (reconnects to IBM by job_id)
# ═══════════════════════════════════════════════════════════════════════════════


def connect_service():
    """Connect to IBM Runtime service using environment credentials."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    key = os.environ.get("IBM_KEY")
    crn = os.environ.get("IBM_INSTANCE_CRN")
    if not key or not crn:
        print("ERROR: Set IBM_KEY and IBM_INSTANCE_CRN environment variables.")
        sys.exit(1)

    return QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=key,
        instance=crn,
    )


def retrieve_job(service, job_id: str, verbose: bool = False) -> dict:
    """Retrieve job results and metadata from IBM Runtime.

    Parameters
    ----------
    service : QiskitRuntimeService
        Connected service instance.
    job_id : str
        IBM Runtime job ID (e.g., 'd8tche5bh0os73epdphg').
    verbose : bool
        Print detailed info during retrieval.

    Returns
    -------
    dict
        Complete job data including status, results, metrics, and timestamps.
    """
    import numpy as np

    print(f"\n{'═' * 60}")
    print(f"  Recovering job: {job_id}")
    print(f"{'═' * 60}")

    job = service.job(job_id)

    # ── Status check ──────────────────────────────────────────────────
    raw_status = job.status()
    status = raw_status.name if hasattr(raw_status, "name") else str(raw_status).upper()
    print(f"\n  Status: {status}")

    if status not in ("DONE", "COMPLETED"):
        print(f"\n  ⚠️ Job is not completed (status={status}).")
        if status in ("QUEUED", "RUNNING", "VALIDATING"):
            print("     The job is still executing. Wait and retry.")
        elif status in ("ERROR", "CANCELLED"):
            print("     The job failed or was cancelled.")
            # Try to get error info
            try:
                error_msg = getattr(job, "error_message", None)
                if error_msg:
                    print(f"     Error: {error_msg}")
            except Exception:
                pass
        return {"job_id": job_id, "status": status, "results": None}

    # ── Metrics (QPU time, timestamps, usage) ─────────────────────────
    metrics_data = {}
    try:
        metrics = job.metrics()
        if isinstance(metrics, dict):
            usage = metrics.get("usage", {})
            timestamps = metrics.get("timestamps", {})
            metrics_data = {
                "qpu_seconds": usage.get("quantum_seconds"),
                "billed_seconds": usage.get("seconds"),
                "created": timestamps.get("created"),
                "running": timestamps.get("running"),
                "finished": timestamps.get("finished"),
            }
            # Compute queue wait
            created = timestamps.get("created")
            running = timestamps.get("running")
            if created and running:
                from datetime import datetime as _dt

                try:
                    t_c = _dt.fromisoformat(created.replace("Z", "+00:00"))
                    t_r = _dt.fromisoformat(running.replace("Z", "+00:00"))
                    metrics_data["queue_wait_s"] = (t_r - t_c).total_seconds()
                except (ValueError, TypeError):
                    pass

            print("\n  ── Timing ──")
            print(f"  QPU seconds:  {metrics_data.get('qpu_seconds', '?')}")
            print(f"  Billed:       {metrics_data.get('billed_seconds', '?')}s")
            if metrics_data.get("queue_wait_s") is not None:
                print(f"  Queue wait:   {metrics_data['queue_wait_s']:.0f}s")
            print(f"  Created:      {metrics_data.get('created', '?')}")
            print(f"  Running:      {metrics_data.get('running', '?')}")
            print(f"  Finished:     {metrics_data.get('finished', '?')}")
    except Exception as e:
        print(f"  (metrics unavailable: {e})")

    # ── Job metadata (inputs, options, session) ─────────────────────
    job_metadata = {}
    try:
        # Program ID (estimator / sampler)
        program_id = getattr(job, "program_id", None)
        if program_id:
            job_metadata["program_id"] = program_id
            print(f"\n  Program: {program_id}")
    except Exception:
        pass

    try:
        # Session/Batch ID — links related jobs together
        session_id = getattr(job, "session_id", None)
        if session_id:
            job_metadata["session_id"] = session_id
            if verbose:
                print(f"  Session: {session_id}")
    except Exception:
        pass

    try:
        # Tags
        tags = getattr(job, "tags", None)
        if tags:
            job_metadata["tags"] = tags
    except Exception:
        pass

    try:
        # Input parameters (shots, resilience options, etc.)
        inputs = job.inputs
        if isinstance(inputs, dict):
            # Extract options (shots, mitigation settings) without the circuit data
            options_input = inputs.get("options", {})
            if options_input:
                job_metadata["submitted_options"] = options_input
                if verbose:
                    print("\n  ── Submitted Options ──")
                    print(f"  {json.dumps(options_input, indent=4, default=str)[:500]}")
    except Exception:
        pass

    # ── Results ───────────────────────────────────────────────────────
    print("\n  ── Results ──")
    job_result = job.result()
    pub_results = []

    for idx, pub_result in enumerate(job_result):
        evs = pub_result.data.evs
        stds = pub_result.data.stds if hasattr(pub_result.data, "stds") else None

        # Handle both scalar and array results
        evs_val = float(evs) if np.isscalar(evs) else evs.tolist()
        stds_val = (
            float(stds)
            if stds is not None and np.isscalar(stds)
            else (stds.tolist() if stds is not None else None)
        )

        pub_data = {
            "pub_idx": idx,
            "evs": evs_val,
            "stds": stds_val,
        }

        # ── ZNE per-noise-factor data (gold for post-hoc analysis) ────
        # When ZNE is active, IBM Runtime may expose the raw extrapolation
        # data: energies at each noise factor before extrapolation.
        for attr in ("evs_noise_factors", "stds_noise_factors", "ensemble_stds", "metadata"):
            val = getattr(pub_result.data, attr, None)
            if val is not None:
                if hasattr(val, "tolist"):
                    pub_data[attr] = val.tolist()
                else:
                    pub_data[attr] = val

        pub_results.append(pub_data)

        if np.isscalar(evs):
            std_str = f" ± {float(stds):.6f}" if stds is not None else ""
            print(f"  PUB {idx}: E = {float(evs):.6f}{std_str}")
            # Show ZNE noise factor data if available
            if "evs_noise_factors" in pub_data:
                print(f"         ZNE noise factors: {pub_data['evs_noise_factors']}")
        else:
            evs_arr = np.atleast_1d(evs)
            print(f"  PUB {idx}: {len(evs_arr)} expectation values")
            if verbose:
                for i, ev in enumerate(evs_arr[:10]):
                    print(f"    [{i}] = {ev:.6f}")
                if len(evs_arr) > 10:
                    print(f"    ... ({len(evs_arr) - 10} more)")

    # ── Backend info ──────────────────────────────────────────────────
    backend_name = None
    try:
        backend_name = job.backend().name
        print(f"\n  Backend: {backend_name}")
    except Exception:
        pass

    # ── Assemble output ───────────────────────────────────────────────
    output = {
        "job_id": job_id,
        "status": status,
        "backend": backend_name,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "metrics": metrics_data,
        "job_metadata": job_metadata,
        "pub_results": pub_results,
        "n_pubs": len(pub_results),
    }

    print(f"\n{'═' * 60}")
    print(f"  ✅ Successfully recovered {len(pub_results)} PUB result(s)")
    print(f"{'═' * 60}\n")

    return output


def main():
    parser = argparse.ArgumentParser(description="Recover results from a completed IBM Runtime job")
    parser.add_argument(
        "job_ids",
        nargs="+",
        help="One or more IBM Runtime job IDs to recover",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Directory to save recovered results as JSON (default: print only)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed expectation values",
    )
    args = parser.parse_args()

    service = connect_service()

    all_results = []
    for job_id in args.job_ids:
        result = retrieve_job(service, job_id.strip(), verbose=args.verbose)
        all_results.append(result)

    # Save if requested
    if args.save:
        save_dir = Path(args.save)
        save_dir.mkdir(parents=True, exist_ok=True)
        for result in all_results:
            jid = result["job_id"]
            out_path = save_dir / f"recovered_{jid}.json"
            json_dump(result, out_path)
            print(f"  Saved: {out_path}")

    # Print summary
    if len(all_results) > 1:
        print(f"\n{'─' * 40}")
        print(f"  Summary: {len(all_results)} jobs recovered")
        for r in all_results:
            status_icon = "✅" if r["status"] == "DONE" else "❌"
            qpu = r.get("metrics", {}).get("qpu_seconds", "?")
            print(f"    {status_icon} {r['job_id']}: {r['status']}, QPU={qpu}s")


if __name__ == "__main__":
    main()
