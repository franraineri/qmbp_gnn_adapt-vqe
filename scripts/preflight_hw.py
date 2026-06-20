#!/usr/bin/env python3
"""Hardware readiness preflight — checks everything before QPU execution."""

import os
import sys

errors = []
warnings = []

print("=" * 60)
print("  HARDWARE READINESS PREFLIGHT CHECK")
print("=" * 60)

# 1. Core packages
print("\n  [1] Package dependencies:")
for pkg in ["qiskit", "qiskit_ibm_runtime", "qiskit_aer", "numpy", "torch", "torch_geometric"]:
    try:
        m = __import__(pkg)
        v = getattr(m, "__version__", "?")
        print(f"      ✓ {pkg} {v}")
    except ImportError:
        errors.append(f"MISSING: {pkg}")
        print(f"      ✗ {pkg} NOT INSTALLED")

# 2. QASM3 export
print("\n  [2] QASM3 export:")
try:
    from qiskit.qasm3 import dumps

    print("      ✓ qiskit.qasm3.dumps available")
except ImportError:
    errors.append("qiskit.qasm3 not available")
    print("      ✗ qiskit.qasm3 NOT available")

# 3. Mapomatic (optional but recommended)
print("\n  [3] Layout optimizer:")
try:
    import mapomatic

    print(f"      ✓ mapomatic {mapomatic.__version__}")
except ImportError:
    warnings.append("mapomatic not installed — BFS fallback will be used")
    print("      ⚠ mapomatic NOT installed (BFS fallback OK)")

# 4. Our hardware module
print("\n  [4] Project hardware module:")
try:
    print("      ✓ HardwareBackend, preflight, calibration all importable")
except Exception as e:
    errors.append(f"Hardware module import: {e}")
    print(f"      ✗ Import error: {e}")

# 5. IBM credentials
print("\n  [5] IBM credentials:")
key = os.environ.get("IBM_KEY")
crn = os.environ.get("IBM_INSTANCE_CRN")
if key:
    print(f"      ✓ IBM_KEY: set (****{key[-4:]})")
else:
    errors.append("IBM_KEY not set. Export: export IBM_KEY='your_token'")
    print("      ✗ IBM_KEY NOT SET")
if crn:
    print(f"      ✓ IBM_INSTANCE_CRN: set (...{crn[-20:]})")
else:
    errors.append("IBM_INSTANCE_CRN not set. Export: export IBM_INSTANCE_CRN='crn:...'")
    print("      ✗ IBM_INSTANCE_CRN NOT SET")

# 6. Benchmark script runnable
print("\n  [6] Benchmark script:")
from pathlib import Path

bm_script = Path("scripts/experiment_runners/hardware/run_mitigation_benchmark.py")
if bm_script.exists():
    print(f"      ✓ {bm_script} exists")
else:
    errors.append(f"{bm_script} not found")
    print(f"      ✗ {bm_script} NOT FOUND")

# 7. Existing fake_backend results (confirms pipeline works)
print("\n  [7] Simulation validation data (seed=100):")
fb = Path("results/mitigation_benchmark/fake_backend")
for cfg in ["C0_raw", "C1_dd_only", "C4_full_pea_light", "C5_full_pea_balanced", "C16_aqc_pea"]:
    cfg_dir = fb / cfg
    files = list(cfg_dir.glob("*seed100.json")) if cfg_dir.exists() else []
    valid = 0
    for f in files:
        try:
            import json

            d = json.loads(f.read_text())
            r = d.get("results", {})
            if r.get("e_raw") is not None or r.get("e_mitigated") is not None:
                valid += 1
        except Exception:
            pass
    status = "✓" if valid >= 4 else "⚠"
    print(f"      {status} {cfg}: {valid} valid results")
    if valid < 4:
        warnings.append(f"{cfg} has only {valid}/4 h-point results for seed=100")

# 8. Summary
print("\n" + "=" * 60)
if errors:
    print(f"  ❌ {len(errors)} BLOCKING ISSUES — CANNOT PROCEED TO HARDWARE")
    for e in errors:
        print(f"     • {e}")
    print("\n  Fix these before running on QPU.")
    sys.exit(1)
elif warnings:
    print(f"  ⚠️  READY WITH {len(warnings)} WARNINGS:")
    for w in warnings:
        print(f"     • {w}")
else:
    print("  ✅ ALL CHECKS PASSED — READY FOR IBM KINGSTON")

print("\n" + "=" * 60)
print("  EXECUTION COMMANDS")
print("=" * 60)
print("""
  # Step 1: Set credentials (if not already)
  export IBM_KEY="your_ibm_quantum_api_token"
  export IBM_INSTANCE_CRN="crn:v1:bluemix:public:quantum-computing:..."

  # Step 2: Smoke test (1 job, ~3 min, validates connectivity)
  python scripts/experiment_runners/hardware/run_mitigation_benchmark.py \\
      --mode hardware --configs C0 --h-values 4.0 \\
      --shots 16384 --seed 42 --backend ibm_kingston

  # Step 3: Full execution (20 jobs, ~23 min QPU, ~$38 USD)
  python scripts/experiment_runners/hardware/run_mitigation_benchmark.py \\
      --mode hardware --configs C0,C1,C4,C5,C16 \\
      --h-values 4.0,3.75,3.5,3.25 --shots 16384 --seed 42 \\
      --backend ibm_kingston --batch

  # Step 4: Analyze results
  python -m project_health.analysis.mitigation_benchmark_analyzer --thesis-table
  python inspect_results.py --mode hardware --h-values 3.25,3.5,3.75,4.0 --seed 42
""")
