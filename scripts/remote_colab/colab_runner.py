#!/usr/bin/env python3
"""
Colab Runner — Manage persistent Colab sessions for qmbp experiments.

WHEN TO USE COLAB (vs local Mac):
  ✅ MPNN training (GPU-bound, benefits from T4)
  ✅ Large-N MPS runs (N>22, needs 12GB+ RAM)
  ✅ Parallel sweeps (offload while using Mac for other work)
  ✅ Long VQE sweeps with many h-points (>30 min)
  ❌ Small statevector runs (N≤16, p≤2) — Mac CPU is faster
  ❌ Quick iteration/debugging — latency overhead not worth it

COLAB LIMITS (free tier):
  - CPU: Intel Xeon 2 vCPUs @ 2.2GHz (weaker than Mac M-series)
  - RAM: ~12.7GB system, 15GB GPU VRAM (T4)
  - Session: 12h max, 90min idle timeout (keep-alive daemon handles this)
  - Quota: ~30h/week GPU, unlimited CPU

Usage (from project root):
    .venv/bin/python scripts/remote/colab_runner.py setup [--gpu T4]
    .venv/bin/python scripts/remote/colab_runner.py run <script> [args...]
    .venv/bin/python scripts/remote/colab_runner.py fetch
    .venv/bin/python scripts/remote/colab_runner.py stop
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLAB_BIN = PROJECT_ROOT / ".venv" / "bin" / "colab"
SESSION_NAME = "qmbp"
REMOTE_BASE = "/content/qmbp"
REPO_URL = "https://github.com/franraineri/qmbp_gnn_adapt-vqe.git"


# --- Colab CLI wrappers ---


def colab(*args: str, timeout: float = 300.0, check: bool = True) -> subprocess.CompletedProcess:
    """Run a colab CLI command."""
    cmd = [str(COLAB_BIN), *args]
    print(f"  → {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    if check and result.returncode != 0:
        print(f"  ✗ Command failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(1)
    return result


def colab_exec(code: str, timeout: float = 600.0) -> subprocess.CompletedProcess:
    """Execute Python code on the remote session via stdin pipe."""
    cmd = [str(COLAB_BIN), "exec", "-s", SESSION_NAME, "--timeout", str(timeout)]
    print(f"  → colab exec -s {SESSION_NAME} [snippet, {len(code)} chars, timeout={timeout}s]")
    result = subprocess.run(cmd, input=code, capture_output=True, text=True, timeout=timeout + 60)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        # Filter IPython SystemExit warnings (normal for scripts that call sys.exit(0))
        stderr_lines = [
            line
            for line in result.stderr.strip().splitlines()
            if "To exit: use 'exit'" not in line and "interactiveshell.py" not in line
        ]
        if stderr_lines:
            print("\n".join(stderr_lines), file=sys.stderr)
    return result


# --- Commands ---


def cmd_setup(args: argparse.Namespace) -> None:
    """Create a Colab session and install the project environment."""
    print("=" * 60)
    print("STEP 1: Checking/creating Colab session")
    print("=" * 60)

    # Check if session already exists
    result = colab("sessions", check=False)
    if SESSION_NAME in (result.stdout or ""):
        print(f"  Session '{SESSION_NAME}' already exists, reusing it.")
    else:
        new_args = ["new", "-s", SESSION_NAME]
        if args.gpu:
            new_args.extend(["--gpu", args.gpu])
        colab(*new_args)
        # Give the session a moment to initialize
        time.sleep(3)

    print("\n" + "=" * 60)
    print("STEP 2: Installing project on remote VM")
    print("=" * 60)

    branch = args.branch or _get_current_branch()

    setup_code = textwrap.dedent(f"""\
        import subprocess, sys, os, time

        def run(cmd, allow_fail=False):
            print(f"  [vm] {{cmd}}")
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if r.stdout.strip():
                print(r.stdout.strip())
            if r.returncode != 0:
                if allow_fail:
                    print(f"  (non-fatal: {{r.stderr.strip()[:200]}})")
                    return False
                print(f"FAILED: {{r.stderr.strip()[:500]}}", file=sys.stderr)
                sys.exit(1)
            return True

        # Clone or pull repo
        if os.path.exists("{REMOTE_BASE}/src"):
            print("Repo exists, pulling latest...")
            run("cd {REMOTE_BASE} && git fetch origin && git reset --hard origin/{branch}")
        else:
            print("Cloning repository...")
            run("git clone --branch {branch} --depth 1 {REPO_URL} {REMOTE_BASE}")

        # Install package
        print("Installing qmbp-simulation + deps...")
        run("pip install -e {REMOTE_BASE} --quiet 2>&1 | tail -3")

        # Verify
        run("python -c 'import qmbp_simulation; print(qmbp_simulation.__file__)'")

        # Show VM specs for reference
        import multiprocessing
        print(f"\\nVM specs: {{multiprocessing.cpu_count()}} vCPUs, ", end="")
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line:
                    gb = int(line.split()[1]) / 1024 / 1024
                    print(f"{{gb:.1f}} GB RAM")
                    break

        # Check GPU
        run("python -c \\"import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')\\"", allow_fail=True)
        print("\\nSETUP_OK")
    """)

    result = colab_exec(setup_code, timeout=600)
    if "SETUP_OK" not in (result.stdout or ""):
        print("✗ Setup failed!", file=sys.stderr)
        sys.exit(1)

    print("\n✓ Session ready!")
    print("  Run:   .venv/bin/python scripts/remote/colab_runner.py run <script> [args...]")
    print("  Sync:  .venv/bin/python scripts/remote/colab_runner.py sync")
    print("  Fetch: .venv/bin/python scripts/remote/colab_runner.py fetch")


def cmd_run(args: argparse.Namespace) -> None:
    """Run a local script on the remote Colab session."""
    script_path = args.script
    script_args = args.script_args

    # Resolve script relative to project root
    local_script = Path(script_path)
    if not local_script.is_absolute():
        local_script = PROJECT_ROOT / local_script
    if not local_script.exists():
        print(f"✗ Script not found: {local_script}", file=sys.stderr)
        sys.exit(1)

    # Convert local path to remote path (script is in the cloned repo)
    try:
        relative = local_script.relative_to(PROJECT_ROOT)
        remote_script = f"{REMOTE_BASE}/{relative}"
    except ValueError:
        # Script outside project — upload it
        print(f"Uploading {local_script.name} to VM...")
        colab("upload", "-s", SESSION_NAME, str(local_script), f"/content/{local_script.name}")
        remote_script = f"/content/{local_script.name}"

    # Build execution code — stream output in real-time via subprocess
    args_str = " ".join(script_args) if script_args else ""
    run_code = textwrap.dedent(f"""\
        import subprocess, sys, os
        os.chdir("{REMOTE_BASE}")
        os.environ["PYTHONUNBUFFERED"] = "1"

        cmd = ["python", "-u", "{remote_script}"] + {repr(script_args)}
        print(f"Running: {{' '.join(cmd)}}")
        print("-" * 60)
        sys.stdout.flush()

        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    """)

    timeout = args.timeout
    print(f"Running on Colab: {script_path} {args_str}")
    print(f"  timeout: {timeout}s")
    print("-" * 60)

    t0 = time.time()
    result = colab_exec(run_code, timeout=timeout)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n✗ Script failed (exit {result.returncode}, {elapsed:.0f}s)", file=sys.stderr)
        sys.exit(result.returncode)
    print(f"\n✓ Done in {elapsed:.0f}s")


def cmd_fetch(args: argparse.Namespace) -> None:
    """Download results from the remote session to local."""
    remote_results = f"{REMOTE_BASE}/results"
    local_results = PROJECT_ROOT / "results" / "colab"
    local_results.mkdir(parents=True, exist_ok=True)

    print("Listing remote results...")
    list_code = textwrap.dedent(f"""\
        import os, json
        results_dir = "{remote_results}"
        if not os.path.exists(results_dir):
            print("NO_RESULTS")
        else:
            files = []
            for root, dirs, filenames in os.walk(results_dir):
                for f in filenames:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, results_dir)
                    size = os.path.getsize(full)
                    files.append({{"path": rel, "size": size}})
            print("RESULTS_JSON:" + json.dumps(files))
    """)

    result = colab_exec(list_code, timeout=30)
    if "NO_RESULTS" in (result.stdout or ""):
        print("No results found on remote VM.")
        return

    files = []
    for line in (result.stdout or "").splitlines():
        if line.startswith("RESULTS_JSON:"):
            files = json.loads(line[len("RESULTS_JSON:") :])
            break

    if not files:
        print("No result files found.")
        return

    total_size = sum(f["size"] for f in files)
    print(f"Found {len(files)} files ({total_size / 1024:.1f} KB total):")
    for f in files[:10]:
        print(f"  {f['path']} ({f['size'] / 1024:.1f} KB)")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")

    print(f"\nDownloading to {local_results}/...")
    downloaded = 0
    for f in files:
        remote_path = f"{remote_results}/{f['path']}"
        local_path = local_results / f["path"]
        local_path.parent.mkdir(parents=True, exist_ok=True)
        r = colab("download", "-s", SESSION_NAME, remote_path, str(local_path), check=False)
        if r.returncode == 0:
            downloaded += 1
    print(f"\n✓ Downloaded {downloaded}/{len(files)} files to {local_results}/")


def cmd_status(args: argparse.Namespace) -> None:
    """Show status of the Colab session."""
    colab("status", "-s", SESSION_NAME, check=False)


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop the Colab session."""
    print("Stopping session...")
    colab("stop", "-s", SESSION_NAME)
    print("✓ Session stopped. GPU hours preserved.")


def cmd_sync(args: argparse.Namespace) -> None:
    """Pull latest code changes on the remote VM."""
    branch = args.branch or _get_current_branch()
    sync_code = textwrap.dedent(f"""\
        import subprocess, sys
        def run(cmd):
            print(f"  [vm] {{cmd}}")
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if r.stdout.strip():
                print(r.stdout.strip())
            if r.returncode != 0:
                print(f"FAILED: {{r.stderr.strip()[:300]}}", file=sys.stderr)
                sys.exit(1)

        run("cd {REMOTE_BASE} && git fetch origin && git reset --hard origin/{branch}")
        run("pip install -e {REMOTE_BASE} --quiet")
        print("SYNC_OK")
    """)
    result = colab_exec(sync_code, timeout=120)
    if "SYNC_OK" not in (result.stdout or ""):
        print("✗ Sync failed!", file=sys.stderr)
        sys.exit(1)
    print(f"✓ Synced to latest {branch}")


def cmd_bench(args: argparse.Namespace) -> None:
    """Quick benchmark to compare Colab vs local speed."""
    bench_code = textwrap.dedent("""\
        import time, numpy as np, sys, multiprocessing

        print(f"vCPUs: {multiprocessing.cpu_count()}")

        # CPU benchmark: matrix operations (simulates statevector)
        n = 1024
        t0 = time.time()
        for _ in range(10):
            a = np.random.randn(n, n)
            b = np.random.randn(n, n)
            _ = a @ b
        cpu_time = time.time() - t0
        print(f"CPU bench (10x 1024x1024 matmul): {cpu_time:.2f}s")

        # Check GPU
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.device("cuda")
                t0 = time.time()
                for _ in range(100):
                    a = torch.randn(2048, 2048, device=device)
                    b = torch.randn(2048, 2048, device=device)
                    _ = a @ b
                torch.cuda.synchronize()
                gpu_time = time.time() - t0
                print(f"GPU bench (100x 2048x2048 matmul): {gpu_time:.2f}s")
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("No GPU available")
        except ImportError:
            print("torch not installed")
        print("BENCH_OK")
    """)
    print("Running benchmark on Colab VM...")
    result = colab_exec(bench_code, timeout=60)
    if "BENCH_OK" not in (result.stdout or ""):
        print("✗ Benchmark failed", file=sys.stderr)


# --- Helpers ---


def _get_current_branch() -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        return result.stdout.strip() or "main"
    except Exception:
        return "main"


# --- Main ---


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Colab Runner — run qmbp experiments on Google Colab",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Setup session with T4 GPU
              python scripts/remote/colab_runner.py setup --gpu T4

              # Run MPNN training (good Colab use case — GPU-bound)
              python scripts/remote/colab_runner.py run \\
                scripts/experiment_runners/noiseless/run_noiseless_pipeline.py \\
                --topology chain_1d --n-qubits 20 --model tfim --p-layers 2

              # Quick benchmark to compare Colab vs local
              python scripts/remote/colab_runner.py bench

              # Sync latest code (after git push)
              python scripts/remote/colab_runner.py sync

              # Fetch results
              python scripts/remote/colab_runner.py fetch

              # Stop session (preserves GPU quota)
              python scripts/remote/colab_runner.py stop
        """),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # setup
    p_setup = sub.add_parser("setup", help="Create session and install environment")
    p_setup.add_argument("--gpu", choices=["T4", "L4", "G4", "H100", "A100"], default=None)
    p_setup.add_argument("--branch", default=None, help="Git branch (default: current)")
    p_setup.set_defaults(func=cmd_setup)

    # run
    p_run = sub.add_parser("run", help="Run a script on the remote session")
    p_run.add_argument("script", help="Path to script (relative to project root)")
    p_run.add_argument("script_args", nargs=argparse.REMAINDER, help="Args forwarded to script")
    p_run.add_argument("--timeout", type=float, default=7200.0, help="Timeout seconds (default 2h)")
    p_run.set_defaults(func=cmd_run)

    # sync
    p_sync = sub.add_parser("sync", help="Pull latest code on the remote VM")
    p_sync.add_argument("--branch", default=None)
    p_sync.set_defaults(func=cmd_sync)

    # fetch
    p_fetch = sub.add_parser("fetch", help="Download results from remote")
    p_fetch.set_defaults(func=cmd_fetch)

    # bench
    p_bench = sub.add_parser("bench", help="Benchmark Colab VM speed")
    p_bench.set_defaults(func=cmd_bench)

    # status
    p_status = sub.add_parser("status", help="Show session status")
    p_status.set_defaults(func=cmd_status)

    # stop
    p_stop = sub.add_parser("stop", help="Stop the Colab session")
    p_stop.set_defaults(func=cmd_stop)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
