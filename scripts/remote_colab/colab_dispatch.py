#!/usr/bin/env python3
"""
Colab Dispatch — Send compute jobs to Colab and collect results.

Orchestrates the local→remote→local workflow:
1. Reads local GroundTruthCache, identifies missing points
2. Creates a job JSON with the tasks to compute
3. Uploads job + worker to Colab, launches async
4. (Later) Downloads results and inserts into local cache

Usage:
    # Dispatch DMRG jobs for missing cache points
    .venv/bin/python scripts/remote/colab_dispatch.py dmrg \
        --topology ladder --n-qubits 26 --model tfim \
        --h-min 2.5 --h-max 3.5 --h-points 5

    # Dispatch VQE jobs
    .venv/bin/python scripts/remote/colab_dispatch.py vqe \
        --topology ladder --n-qubits 10 --model tfim --p-layers 1 \
        --h-min 2.0 --h-max 3.5 --h-points 10

    # Check status of running job
    .venv/bin/python scripts/remote/colab_dispatch.py status

    # Collect results and insert into local cache
    .venv/bin/python scripts/remote/colab_dispatch.py collect
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLAB_BIN = PROJECT_ROOT / ".venv" / "bin" / "colab"
SESSION_NAME = "qmbp"
REMOTE_BASE = "/content/qmbp"
JOB_PATH_REMOTE = "/content/job.json"
RESULTS_PATH_REMOTE = "/content/results.json"
LOG_PATH_REMOTE = "/content/worker.log"


# --- Colab helpers ---


def colab(*args: str, timeout: float = 60.0, check: bool = True) -> subprocess.CompletedProcess:
    """Run a colab CLI command."""
    cmd = [str(COLAB_BIN), *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        print(f"✗ colab {' '.join(args[:2])}: {result.stderr[:200]}", file=sys.stderr)
        sys.exit(1)
    return result


def colab_exec(code: str, timeout: float = 30.0) -> subprocess.CompletedProcess:
    """Execute Python code on the remote session."""
    cmd = [str(COLAB_BIN), "exec", "-s", SESSION_NAME, "--timeout", str(timeout)]
    return subprocess.run(cmd, input=code, capture_output=True, text=True, timeout=timeout + 30)


def ensure_session():
    """Ensure Colab session exists and is ready."""
    result = colab("sessions", check=False)
    if SESSION_NAME not in (result.stdout or ""):
        print("No active session. Creating...")
        colab("new", "-s", SESSION_NAME)
        time.sleep(3)
    print(f"  Session '{SESSION_NAME}' active.")


def ensure_env():
    """Ensure project is installed on VM. Uses pre-built tarball if available."""
    # Quick check: can we import the DMRG-critical modules?
    check = colab_exec(
        "import sys; sys.path.insert(0, '/content/qmbp/src')\n"
        "from qmbp_simulation.solvers.classical import ClassicalSolver\n"
        "print('ENV_OK')\n",
        timeout=15,
    )
    if "ENV_OK" in (check.stdout or ""):
        print("  ✓ Environment already set up")
        return

    print("  Setting up environment on VM...")

    # Use existing tarball if available, otherwise build it
    tarball_local = PROJECT_ROOT / "data" / ".colab_src.tar.gz"
    if not tarball_local.exists() or (time.time() - tarball_local.stat().st_mtime > 3600):
        print("  Building source tarball...")
        tarball_local.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "tar",
                "czf",
                str(tarball_local),
                "--exclude=__pycache__",
                "--exclude=*.pyc",
                "--exclude=*.pyo",
                "--exclude=*.so",
                "--exclude=*.egg-info",
                "-C",
                str(PROJECT_ROOT),
                "src/",
                "pyproject.toml",
            ],
            check=True,
            capture_output=True,
        )

    print("  Uploading source...")
    colab("upload", "-s", SESSION_NAME, str(tarball_local), "/content/qmbp_src.tar.gz")

    # Step 1: Extract tarball + install qiskit (can be slow, high timeout)
    print("  Installing dependencies (qiskit ~60s)...")
    install_code = (
        "import subprocess, os\n"
        "os.makedirs('/content/qmbp', exist_ok=True)\n"
        "subprocess.run('tar xzf /content/qmbp_src.tar.gz -C /content/qmbp/', shell=True)\n"
        "r = subprocess.run('pip install qiskit --quiet', shell=True, "
        "capture_output=True, text=True)\n"
        "print('QISKIT_INSTALLED' if r.returncode == 0 else 'QISKIT_FAIL:' + r.stderr[-100:])\n"
    )
    result = colab_exec(install_code, timeout=600)
    if "QISKIT_INSTALLED" not in (result.stdout or ""):
        print(f"  ⚠ Qiskit install issue: {result.stdout[-200:]}")
        # Continue anyway — might already be present

    # Step 2: Verify imports
    print("  Verifying imports...")
    verify_code = (
        "import sys; sys.path.insert(0, '/content/qmbp/src')\n"
        "from qmbp_simulation.models.hamiltonian import make_lattice\n"
        "from qmbp_simulation.solvers.classical import ClassicalSolver\n"
        "print('ENV_OK')\n"
    )
    result = colab_exec(verify_code, timeout=30)
    if "ENV_OK" not in (result.stdout or ""):
        print("✗ Environment setup failed", file=sys.stderr)
        print(result.stdout[-500:] if result.stdout else "no stdout")
        print(result.stderr[-200:] if result.stderr else "")
        sys.exit(1)
    print("  ✓ Environment ready")


# --- Job generation ---


def generate_h_grid(h_min: float, h_max: float, h_points: int) -> list[float]:
    """Generate uniform h grid."""
    return [round(float(h), 6) for h in np.linspace(h_min, h_max, h_points)]


def find_missing_dmrg(
    topology: str, n_qubits: int, model: str, h_values: list[float]
) -> list[float]:
    """Check local GroundTruthCache and return h-values not yet cached."""
    from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

    cache = GroundTruthCache()
    missing = []
    for h in h_values:
        if cache.get(topology, n_qubits, model, h) is None:
            missing.append(h)
    print(f"  Cache: {len(h_values) - len(missing)}/{len(h_values)} hits, {len(missing)} missing")
    return missing


# --- Commands ---


def cmd_dmrg(args: argparse.Namespace) -> None:
    """Dispatch DMRG ground truth computation to Colab."""
    h_values = generate_h_grid(args.h_min, args.h_max, args.h_points)

    # Check what's missing from local cache
    missing = find_missing_dmrg(args.topology, args.n_qubits, args.model, h_values)
    if not missing:
        print("✓ All points already cached locally. Nothing to dispatch.")
        return

    print(f"\nDispatching {len(missing)} DMRG tasks to Colab:")
    print(f"  {args.topology} N={args.n_qubits} model={args.model}")
    print(f"  h = {missing}")

    # Build job JSON
    job = {
        "task_type": "dmrg",
        "metadata": {
            "topology": args.topology,
            "n_qubits": args.n_qubits,
            "model": args.model,
            "dispatched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "tasks": [
            {
                "topology": args.topology,
                "n_qubits": args.n_qubits,
                "model": args.model,
                "h": h,
                "chi_max": args.chi_max,
            }
            for h in missing
        ],
    }

    _dispatch_job(job)


def cmd_vqe(args: argparse.Namespace) -> None:
    """Dispatch VQE optimization tasks to Colab."""
    h_values = generate_h_grid(args.h_min, args.h_max, args.h_points)

    print(f"\nDispatching {len(h_values)} VQE tasks to Colab:")
    print(f"  {args.topology} N={args.n_qubits} model={args.model} p={args.p_layers}")
    print(f"  h = {h_values}")

    job = {
        "task_type": "vqe",
        "metadata": {
            "topology": args.topology,
            "n_qubits": args.n_qubits,
            "model": args.model,
            "p_layers": args.p_layers,
            "dispatched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "tasks": [
            {
                "topology": args.topology,
                "n_qubits": args.n_qubits,
                "model": args.model,
                "h": h,
                "p_layers": args.p_layers,
                "maxiter": args.maxiter,
                "n_restarts": args.n_restarts,
                "method": args.method,
            }
            for h in h_values
        ],
    }

    _dispatch_job(job)


def _dispatch_job(job: dict) -> None:
    """Upload job and worker, launch async on Colab."""
    ensure_session()
    ensure_env()

    # Save job locally and upload
    local_job = PROJECT_ROOT / "data" / "colab_job.json"
    local_job.parent.mkdir(parents=True, exist_ok=True)
    with open(local_job, "w") as f:
        json.dump(job, f, indent=2)

    print("\n  Uploading job + worker...")
    colab("upload", "-s", SESSION_NAME, str(local_job), JOB_PATH_REMOTE)
    worker_path = PROJECT_ROOT / "scripts" / "remote" / "colab_worker.py"
    colab("upload", "-s", SESSION_NAME, str(worker_path), "/content/colab_worker.py")

    # Launch worker async
    print("  Launching worker (async)...")
    launch_code = (
        "import subprocess, os\n"
        f"cmd = 'nohup python -u /content/colab_worker.py "
        f"--job {JOB_PATH_REMOTE} --output {RESULTS_PATH_REMOTE} "
        f"> {LOG_PATH_REMOTE} 2>&1 &'\n"
        "subprocess.Popen(cmd, shell=True)\n"
        "print('DISPATCHED')\n"
    )
    result = colab_exec(launch_code, timeout=10)
    if "DISPATCHED" in (result.stdout or ""):
        print("\n✓ Job dispatched! Worker running in background.")
        print(f"  Tasks: {job['n_tasks'] if 'n_tasks' in job else len(job['tasks'])}")
        print("\n  Check progress: .venv/bin/python scripts/remote/colab_dispatch.py status")
        print("  Collect results: .venv/bin/python scripts/remote/colab_dispatch.py collect")
    else:
        print("⚠ Dispatch may have failed. Check with 'status' command.")
        print(result.stdout[-200:])


def cmd_status(args: argparse.Namespace) -> None:
    """Check if the remote worker is still running."""
    result = colab_exec(
        "import subprocess\n"
        "ps = subprocess.run('ps aux | grep colab_worker | grep -v grep', "
        "shell=True, capture_output=True, text=True)\n"
        "print('RUNNING' if ps.stdout.strip() else 'DONE')\n"
        f"r = subprocess.run('tail -5 {LOG_PATH_REMOTE}', shell=True, "
        "capture_output=True, text=True)\n"
        "print(r.stdout)\n",
        timeout=15,
    )
    print(result.stdout)


def cmd_collect(args: argparse.Namespace) -> None:
    """Download results from Colab and insert into local cache."""
    # Download results JSON
    local_results = PROJECT_ROOT / "data" / "colab_results.json"
    colab("download", "-s", SESSION_NAME, RESULTS_PATH_REMOTE, str(local_results), check=False)

    if not local_results.exists():
        print("✗ No results file found. Is the job still running? Try 'status' first.")
        return

    with open(local_results) as f:
        output = json.load(f)

    task_type = output.get("task_type", "unknown")
    results = output.get("results", [])
    metadata = output.get("metadata", {})
    n_success = output.get("n_success", 0)
    n_fail = output.get("n_fail", 0)

    print(f"Collected {n_success} results ({n_fail} failures)")
    print(f"  Task: {task_type}, {metadata}")

    if task_type == "dmrg":
        _insert_dmrg_results(results, metadata)
    elif task_type == "vqe":
        _save_vqe_results(results, metadata)
    else:
        print(f"  Results saved to {local_results} (manual processing needed)")


def _insert_dmrg_results(results: list[dict], metadata: dict) -> None:
    """Insert DMRG results into local GroundTruthCache."""
    from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

    cache = GroundTruthCache()
    topology = metadata.get("topology", "unknown")
    n_qubits = metadata.get("n_qubits", 0)
    model = metadata.get("model", "tfim")
    inserted = 0

    for r in results:
        if r.get("status") != "ok":
            continue
        cache.put(
            topology=topology,
            n_qubits=n_qubits,
            model=model,
            h=r["h"],
            energy=r["energy"],
            gap=r["gap"],
            method=r.get("gap_method", "colab_dmrg"),
            mag_x=r.get("mag_x"),
            corr_zz=r.get("corr_zz"),
        )
        inserted += 1

    cache.flush()
    print(f"\n✓ Inserted {inserted} entries into GroundTruthCache")
    print(f"  Cache now has {len(cache)} total entries")


def _save_vqe_results(results: list[dict], metadata: dict) -> None:
    """Save VQE results as NPZ for MPNN training."""
    ok_results = [r for r in results if r.get("status") == "ok"]
    if not ok_results:
        print("  No successful VQE results to save.")
        return

    topology = metadata.get("topology", "unknown")
    n_qubits = metadata.get("n_qubits", 0)
    model = metadata.get("model", "tfim")
    p_layers = metadata.get("p_layers", 1)

    h_values = np.array([r["h"] for r in ok_results])
    energies = np.array([r["energy"] for r in ok_results])
    thetas = np.array([r["theta_opt"] for r in ok_results])

    outdir = PROJECT_ROOT / "data" / "vqe_colab"
    outdir.mkdir(parents=True, exist_ok=True)
    fname = f"{model}_{topology}_n{n_qubits}_p{p_layers}.npz"
    outpath = outdir / fname

    np.savez(outpath, h_values=h_values, energies=energies, thetas=thetas)
    print(f"\n✓ Saved {len(ok_results)} VQE results to {outpath}")


# --- Main ---


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Colab Dispatch — send compute jobs to Colab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # dmrg
    p_dmrg = sub.add_parser("dmrg", help="Dispatch DMRG ground truth tasks")
    p_dmrg.add_argument("--topology", required=True)
    p_dmrg.add_argument("--n-qubits", type=int, required=True)
    p_dmrg.add_argument("--model", default="tfim")
    p_dmrg.add_argument("--h-min", type=float, required=True)
    p_dmrg.add_argument("--h-max", type=float, required=True)
    p_dmrg.add_argument("--h-points", type=int, required=True)
    p_dmrg.add_argument("--chi-max", type=int, default=None)
    p_dmrg.set_defaults(func=cmd_dmrg)

    # vqe
    p_vqe = sub.add_parser("vqe", help="Dispatch VQE optimization tasks")
    p_vqe.add_argument("--topology", required=True)
    p_vqe.add_argument("--n-qubits", type=int, required=True)
    p_vqe.add_argument("--model", default="tfim")
    p_vqe.add_argument("--p-layers", type=int, required=True)
    p_vqe.add_argument("--h-min", type=float, required=True)
    p_vqe.add_argument("--h-max", type=float, required=True)
    p_vqe.add_argument("--h-points", type=int, required=True)
    p_vqe.add_argument("--maxiter", type=int, default=500)
    p_vqe.add_argument("--n-restarts", type=int, default=3)
    p_vqe.add_argument("--method", default="L-BFGS-B")
    p_vqe.set_defaults(func=cmd_vqe)

    # status
    p_status = sub.add_parser("status", help="Check worker progress")
    p_status.set_defaults(func=cmd_status)

    # collect
    p_collect = sub.add_parser("collect", help="Download results into local cache")
    p_collect.set_defaults(func=cmd_collect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
