"""One-shot setup + async launch on Colab VM."""

import subprocess
import sys
from pathlib import Path

COLAB = str(Path(__file__).resolve().parents[2] / ".venv" / "bin" / "colab")
SESSION = "qmbp"


def colab_exec(code: str, timeout: float = 300.0) -> subprocess.CompletedProcess:
    cmd = [COLAB, "exec", "-s", SESSION, "--timeout", str(timeout)]
    return subprocess.run(cmd, input=code, capture_output=True, text=True, timeout=timeout + 60)


def main():
    # Step 1: Setup repo
    print("=" * 50)
    print("STEP 1: Setting up repo on Colab VM")
    print("=" * 50)

    setup_code = (
        "import subprocess, sys, os\n"
        "def run(cmd):\n"
        "    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)\n"
        "    if r.stdout.strip(): print(r.stdout[-400:])\n"
        "    if r.returncode != 0:\n"
        "        print('FAIL:', r.stderr[-300:], file=sys.stderr)\n"
        "        sys.exit(1)\n"
        "\n"
        "BASE = '/content/qmbp'\n"
        "if os.path.exists(f'{BASE}/src'):\n"
        "    print('Pulling latest...')\n"
        "    run(f'cd {BASE} && git fetch origin && git reset --hard origin/changes-re-order-nb')\n"
        "else:\n"
        "    print('Cloning...')\n"
        "    run(f'git clone --branch changes-re-order-nb --depth 1 "
        "https://github.com/franraineri/qmbp_gnn_adapt-vqe.git {BASE}')\n"
        "\n"
        "print('Installing...')\n"
        "run(f'pip install -e {BASE} --quiet 2>&1 | tail -3')\n"
        "run(\"python -c 'import qmbp_simulation; print(qmbp_simulation.__file__)'\")\n"
        "print('SETUP_OK')\n"
    )

    result = colab_exec(setup_code, timeout=300)
    print(result.stdout[-800:])
    if "SETUP_OK" not in (result.stdout or ""):
        print("Setup FAILED:", result.stderr[-300:], file=sys.stderr)
        sys.exit(1)
    print("\n✓ Setup complete")

    # Step 2: Launch experiment asynchronously with nohup
    print("\n" + "=" * 50)
    print("STEP 2: Launching experiment (async, nohup)")
    print("=" * 50)

    experiment_cmd = (
        "python /content/qmbp/scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py "
        "--topology ladder --target-n 26 --h-min 2.5 --h-max 3.5 --h-points 5 "
        "--iterative-improve --max-iterations 3 --maxiter 50 --n-restarts 3"
    )

    launch_code = (
        "import subprocess, os\n"
        "os.chdir('/content/qmbp')\n"
        f"cmd = '{experiment_cmd}'\n"
        "log = '/content/qmbp/results/colab_run.log'\n"
        "os.makedirs('/content/qmbp/results', exist_ok=True)\n"
        "full_cmd = f'nohup {cmd} > {log} 2>&1 &'\n"
        "print(f'Launching: {full_cmd}')\n"
        "r = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)\n"
        "print(r.stdout)\n"
        "# Verify it started\n"
        "import time; time.sleep(2)\n"
        "ps = subprocess.run('ps aux | grep run_accelerated | grep -v grep', "
        "shell=True, capture_output=True, text=True)\n"
        "if ps.stdout.strip():\n"
        "    print('LAUNCH_OK')\n"
        "    print(ps.stdout.strip())\n"
        "else:\n"
        "    print('LAUNCH_FAILED')\n"
        "    # Check if it already finished or crashed\n"
        "    cat = subprocess.run(f'tail -20 {log}', shell=True, capture_output=True, text=True)\n"
        "    print(cat.stdout)\n"
    )

    result = colab_exec(launch_code, timeout=30)
    print(result.stdout)
    if "LAUNCH_OK" in (result.stdout or ""):
        print("\n✓ Experiment launched asynchronously!")
        print("  It will continue running even if you disconnect.")
        print("\n  Check progress:")
        print(
            "    .venv/bin/python scripts/remote/colab_runner.py run --timeout 10 "
            "scripts/remote/_check_progress.py"
        )
        print("\n  Or:")
        print(
            "    .venv/bin/colab exec -s qmbp --timeout 10 <<< "
            "\"import subprocess; print(subprocess.run('tail -30 /content/qmbp/results/colab_run.log', "
            'shell=True, capture_output=True, text=True).stdout)"'
        )
    else:
        print("\n✗ Launch may have failed. Check logs.", file=sys.stderr)
        if result.stderr:
            print(result.stderr[-300:])


if __name__ == "__main__":
    main()
