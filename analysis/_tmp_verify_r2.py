#!/usr/bin/env python3
"""Pre-flight check for run_p1_pipeline_variants_r2.py.

Verifies:
1. h_test values are NOT in training h_values (must be unseen)
2. h_test values are within valid regime
3. Output directories don't already exist (no collision)
4. PIPELINE_SCRIPT exists and is executable
5. All training h_values are within valid regime (no wasted VQE)
6. COMP-5 h_test=4.75 is within training range (interpolation, not extrapolation)
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# From the script
PIPELINE_SCRIPT = "scripts/experiment_runners/experiment_run_helpers_CHECK/run_pipeline.py"

P1_VALID_REGIME = {
    ("chain_1d", 10): 1.9,
    ("ladder", 10): 2.0,
    ("triangular", 10): 3.5,
}

P2_VALID_REGIME = {
    ("chain_1d", 10): 1.5,
    ("ladder", 10): 2.0,
    ("triangular", 10): 2.5,
}

# All variants defined in the script
VARIANTS = [
    # Noiseless corrections
    {
        "id": "P1R2-chain",
        "topo": "chain_1d",
        "n": 10,
        "p": 1,
        "h_values": [4.0, 3.5, 3.0, 2.5, 2.0],
        "h_test": [2.75],
        "output": "results/thesis/p1_variants_N10_r2/chain_1d_seed{seed}",
    },
    {
        "id": "P1R2-ladder",
        "topo": "ladder",
        "n": 10,
        "p": 1,
        "h_values": [4.0, 3.5, 3.0, 2.5],
        "h_test": [3.25],
        "output": "results/thesis/p1_variants_N10_r2/ladder_seed{seed}",
    },
    # Extended
    {
        "id": "COMP4-tri-p2",
        "topo": "triangular",
        "n": 10,
        "p": 2,
        "h_values": [5.0, 4.5, 4.0, 3.5],
        "h_test": [4.25],
        "output": "results/thesis/p1_variants_N10_r2/comp4_tri_p2_seed{seed}",
    },
    {
        "id": "COMP5-tri-multi",
        "topo": "triangular",
        "n": 10,
        "p": 1,
        "h_values": [5.0, 4.5, 4.0, 3.5],
        "h_test": [3.75, 4.25, 4.75],
        "output": "results/thesis/p1_variants_N10_r2/comp5_tri_multi_htest",
    },
    {
        "id": "COMP2-chain-dense",
        "topo": "chain_1d",
        "n": 10,
        "p": 1,
        "h_values": [4.0, 3.75, 3.5, 3.25, 3.0, 2.75, 2.5, 2.25, 2.0],
        "h_test": [2.75],
        "output": "results/thesis/p1_variants_N10_r2/comp2_chain_dense_seed{seed}",
    },
]

errors = []
warnings = []

print("=" * 70)
print("  PRE-FLIGHT CHECK: run_p1_pipeline_variants_r2.py")
print("=" * 70)

# Check 1: Pipeline script exists
print("\n[1] Pipeline script exists?")
script_path = ROOT / PIPELINE_SCRIPT
if script_path.exists():
    print(f"  ✅ {PIPELINE_SCRIPT}")
else:
    errors.append(f"Pipeline script not found: {PIPELINE_SCRIPT}")
    print(f"  ❌ NOT FOUND: {PIPELINE_SCRIPT}")

# Check 2: h_test NOT in h_values
print("\n[2] h_test values are UNSEEN (not in training set)?")
for v in VARIANTS:
    for ht in v["h_test"]:
        if ht in v["h_values"]:
            errors.append(f"{v['id']}: h_test={ht} IS in h_values={v['h_values']}")
            print(f"  ❌ {v['id']}: h_test={ht} IS in training set!")
        else:
            print(f"  ✅ {v['id']}: h_test={ht} not in {v['h_values']}")

# Check 3: h_test within valid regime
print("\n[3] h_test values within valid regime?")
for v in VARIANTS:
    regime = P1_VALID_REGIME if v["p"] == 1 else P2_VALID_REGIME
    threshold = regime.get((v["topo"], v["n"]), 0)
    for ht in v["h_test"]:
        if ht >= threshold:
            print(f"  ✅ {v['id']}: h_test={ht} ≥ {threshold} ({v['topo']} p={v['p']})")
        else:
            errors.append(f"{v['id']}: h_test={ht} < {threshold} (outside valid regime)")
            print(f"  ❌ {v['id']}: h_test={ht} < {threshold} OUTSIDE VALID REGIME!")

# Check 4: All training h_values within valid regime
print("\n[4] All training h_values within valid regime?")
for v in VARIANTS:
    regime = P1_VALID_REGIME if v["p"] == 1 else P2_VALID_REGIME
    threshold = regime.get((v["topo"], v["n"]), 0)
    below = [h for h in v["h_values"] if h < threshold]
    if below:
        warnings.append(
            f"{v['id']}: training h_values {below} are below valid regime {threshold} "
            f"(VQE may not converge well there)"
        )
        print(f"  ⚠️  {v['id']}: h_values {below} < {threshold} (may not converge)")
    else:
        print(f"  ✅ {v['id']}: all h_values ≥ {threshold}")

# Check 5: h_test is INTERPOLATION (within training range)
print("\n[5] h_test is interpolation (within training h range)?")
for v in VARIANTS:
    h_min = min(v["h_values"])
    h_max = max(v["h_values"])
    for ht in v["h_test"]:
        if h_min <= ht <= h_max:
            print(f"  ✅ {v['id']}: h_test={ht} in [{h_min}, {h_max}] (interpolation)")
        else:
            warnings.append(f"{v['id']}: h_test={ht} outside [{h_min}, {h_max}] (EXTRAPOLATION)")
            print(f"  ⚠️  {v['id']}: h_test={ht} OUTSIDE [{h_min}, {h_max}] (extrapolation!)")

# Check 6: Output directories don't exist
print("\n[6] Output directories are fresh (no collision)?")
out_base = ROOT / "results" / "thesis" / "p1_variants_N10_r2"
if out_base.exists():
    existing = list(out_base.iterdir())
    if existing:
        warnings.append(f"Output dir already has {len(existing)} entries: {out_base}")
        print(f"  ⚠️  {out_base} already exists with {len(existing)} entries")
        for e in existing[:5]:
            print(f"      → {e.name}")
    else:
        print(f"  ✅ {out_base} exists but is empty")
else:
    print(f"  ✅ {out_base} does not exist (will be created)")

# Summary
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"\n  Errors:   {len(errors)}")
print(f"  Warnings: {len(warnings)}")

if errors:
    print("\n  ❌ ERRORS (must fix before running):")
    for e in errors:
        print(f"    • {e}")

if warnings:
    print("\n  ⚠️  WARNINGS (review but may be acceptable):")
    for w in warnings:
        print(f"    • {w}")

if not errors and not warnings:
    print("\n  ✅ ALL CHECKS PASSED — safe to execute!")
elif not errors:
    print("\n  ✅ No blocking errors — safe to execute (review warnings above)")
else:
    print("\n  ❌ FIX ERRORS BEFORE EXECUTING!")
