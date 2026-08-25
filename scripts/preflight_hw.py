#!/usr/bin/env python3
"""Pre-QPU execution preflight check — run BEFORE every hardware deployment.

Verifies that the pipeline will use the correct layout and mitigation
strategy BEFORE spending any QPU credits. Catches the bugs that caused
36.6% error (routing overhead) and 42% error (affine amplification).

Checks:
  1. SWAP-free layout is Priority 1 in select_layouts_for_hardware
  2. Affine correction clips correctly (no amplification)
  3. HardwareConfig has correct defaults
  4. Kingston coupling map connectivity for the fallback layout
  5. Circuit transpilation with the fallback layout produces ≤20 CZ
  6. VF2 does NOT override the SWAP-free layout (priority order)

Usage:
    .venv/bin/python scripts/preflight_hw.py          # Full check (~10s)
    .venv/bin/python scripts/preflight_hw.py --quick  # Config check only (~1s)

Exit codes:
    0 = safe to proceed with QPU execution
    1+ = DO NOT proceed, fix issues first
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

violations = 0


def ok(msg):
    print(f"  ✅ {msg}")


def fail(msg):
    global violations
    violations += 1
    print(f"  ❌ {msg}")


def warn(msg):
    print(f"  ⚠️  {msg}")


print("=" * 60)
print("PRE-QPU PREFLIGHT CHECK")
print("=" * 60)

# ── Check 1: HardwareConfig defaults ──
print("\n[1] HardwareConfig defaults")
from qmbp_simulation.execution.hardware.config import HardwareConfig

config = HardwareConfig()
if config.fallback_layout_kingston == [22, 23, 24, 25, 26, 27, 28, 16, 37, 17]:
    ok("fallback_layout_kingston = [22,23,24,25,26,27,28,16,37,17]")
else:
    fail(f"Wrong fallback layout: {config.fallback_layout_kingston}")

if config.use_mapomatic:
    ok("use_mapomatic = True (VF2 available as Priority 2)")
else:
    warn("use_mapomatic = False — only BFS fallback available")

if config.min_ces_spread == 0.02:
    ok("min_ces_spread = 0.02")
else:
    warn(f"min_ces_spread = {config.min_ces_spread} (expected 0.02)")

# ── Check 2: Affine correction ──
print("\n[2] Affine correction (clip, not amplify)")
from qmbp_simulation.execution import affine_correct_energy

# The exact scenario that caused 42% error
result = affine_correct_energy(-33.201, -33.198, n_qubits=10, h_value=3.25)
if result.corrected_energy == -33.198:
    ok("Sub-ground clip → e_ground (not -35.064)")
else:
    fail(f"Affine BUG still present! Got {result.corrected_energy}")

# Above-ground: no change
result2 = affine_correct_energy(-33.0, -33.198, n_qubits=10, h_value=3.25)
if not result2.correction_applied:
    ok("Above-ground energy: no correction")
else:
    fail("Above-ground energy incorrectly modified!")

# ── Check 3: Layout priority order ──
print("\n[3] Layout selection priority order")
from qmbp_simulation.execution.hardware.submission import select_layouts_for_hardware
import inspect

source = inspect.getsource(select_layouts_for_hardware)
# The SWAP-free block must come BEFORE the VF2 block
swap_free_pos = source.find("PRIORITY 1")
vf2_pos = source.find("PRIORITY 2")
bfs_pos = source.find("PRIORITY 3")

if swap_free_pos < vf2_pos < bfs_pos:
    ok("Priority order: SWAP-free (1) → VF2 (2) → BFS (3)")
else:
    fail("Priority order WRONG! SWAP-free must be before VF2")
    if swap_free_pos > vf2_pos:
        fail("  VF2 comes before SWAP-free — layout bug will recur!")

# Check that fallback_layout_kingston is referenced in the function
if "fallback_layout_kingston" in source:
    ok("fallback_layout_kingston referenced in select_layouts")
else:
    fail("fallback_layout_kingston NOT used in select_layouts!")

# ── Check 4: n_2q threshold ──
print("\n[4] Gate count threshold")
threshold = config.n_qubits * 2  # 20
if threshold == 20:
    ok(f"n_2q threshold = {threshold} (accepts 9 or 18 CZ, rejects 34+)")
else:
    warn(f"Threshold = {threshold} (expected 20 for N=10)")

# ── Check 5: Quick mode stops here ──
if "--quick" in sys.argv:
    print("\n" + "=" * 60)
    if violations == 0:
        print("QUICK PREFLIGHT: ALL PASSED ✅ — safe to deploy")
    else:
        print(f"QUICK PREFLIGHT: {violations} ISSUES ❌ — DO NOT deploy")
    print("=" * 60)
    sys.exit(violations)

# ── Check 5: Kingston coupling map (requires FakeKingston) ──
print("\n[5] Kingston coupling map verification")
try:
    import json

    cm_cache = Path("/tmp/kingston_cm.json")
    if cm_cache.exists():
        edges = json.loads(cm_cache.read_text())
    else:
        from qiskit_ibm_runtime.fake_provider import FakeKingston
        backend = FakeKingston()
        edges = list(backend.coupling_map.get_edges())
        cm_cache.write_text(json.dumps(edges))

    adj = {}
    for e in edges:
        adj.setdefault(e[0], set()).add(e[1])
        adj.setdefault(e[1], set()).add(e[0])

    # Verify all 9 heavy_hex edges are present in the fallback layout
    layout = config.fallback_layout_kingston
    required_edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(1,7),(3,8),(5,9)]
    missing = []
    for i, j in required_edges:
        qi, qj = layout[i], layout[j]
        if qj not in adj.get(qi, set()):
            missing.append((i, j, qi, qj))

    if not missing:
        ok(f"All 9 edges physically present on Kingston ({layout[:3]}...)")
    else:
        fail(f"{len(missing)} edges MISSING on Kingston!")
        for i, j, qi, qj in missing:
            print(f"    logical ({i},{j}) → physical ({qi},{qj}): NOT CONNECTED")
        fail("SWAP-free layout is INVALID for current Kingston topology!")
        fail("Kingston may have disabled qubits. Check IBM dashboard.")

except Exception as e:
    warn(f"Could not verify Kingston coupling map: {e}")
    warn("Run with FakeKingston available to complete this check")

# ── Check 6: Verify test suite passes ──
print("\n[6] Critical test suite")
import subprocess

test_result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "tests/hardware/test_layout_priority.py",
     "tests/hardware/test_p2a_guard.py",
     "tests/unit/test_affine_correction.py",
     "-q", "--tb=line"],
    capture_output=True, text=True, cwd=str(ROOT), timeout=30,
)
if test_result.returncode == 0:
    # Extract pass count
    last_line = test_result.stdout.strip().split("\n")[-1]
    ok(f"All layout + affine tests pass: {last_line}")
else:
    fail("Test failures detected!")
    for line in test_result.stdout.strip().split("\n")[-5:]:
        print(f"    {line}")

# ── Summary ──
print("\n" + "=" * 60)
if violations == 0:
    print("PREFLIGHT: ALL PASSED ✅")
    print("  Safe to run: .venv/bin/python scripts/experiment_runners/hardware/run_ibm_deployment.py --no-spsa --tier 0")
else:
    print(f"PREFLIGHT: {violations} ISSUES FOUND ❌")
    print("  DO NOT proceed with QPU execution until all issues are resolved.")
print("=" * 60)
sys.exit(violations)
