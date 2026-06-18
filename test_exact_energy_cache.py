"""Quick verification: _get_exact_energy returns bit-exact values on repeated calls."""

import sys

sys.path.insert(0, "/Users/franco.raineri/devTools/QuantumDev/qmbp_gnn_adapt-vqe")
sys.path.insert(0, "/Users/franco.raineri/devTools/QuantumDev/qmbp_gnn_adapt-vqe/src")

from scripts.experiment_runners.hardware.run_mitigation_benchmark import (
    _classical_cache,
    _compute_upper_bound,
    _get_exact_energy,
)

# Test 1: First call populates cache
print("=== Test 1: First call to _get_exact_energy(3.5) ===")
e_exact_1, gap_1 = _get_exact_energy(3.5)
print(f"  e_exact = {e_exact_1:.15f}")
print(f"  gap     = {gap_1:.15f}")
print(f"  cache size = {len(_classical_cache)}")
assert 3.5 in _classical_cache, "Cache should contain h=3.5"

# Test 2: Second call returns bit-exact same values
print("\n=== Test 2: Second call to _get_exact_energy(3.5) ===")
e_exact_2, gap_2 = _get_exact_energy(3.5)
print(f"  e_exact = {e_exact_2:.15f}")
print(f"  gap     = {gap_2:.15f}")
assert e_exact_1 == e_exact_2, f"NOT bit-exact: {e_exact_1} != {e_exact_2}"
assert gap_1 == gap_2, f"NOT bit-exact: {gap_1} != {gap_2}"
print("  ✅ Bit-exact match confirmed (identity, not just close)")


# Test 3: Different h_value creates separate cache entry
print("\n=== Test 3: Different h_value (4.0) ===")
e_exact_3, gap_3 = _get_exact_energy(4.0)
print(f"  e_exact = {e_exact_3:.15f}")
print(f"  gap     = {gap_3:.15f}")
print(f"  cache size = {len(_classical_cache)}")
assert len(_classical_cache) == 2, "Cache should have 2 entries"
assert e_exact_3 != e_exact_1, "Different h should give different energy"

# Test 4: Upper bound computation
print("\n=== Test 4: _compute_upper_bound(3.5) ===")
e_upper = _compute_upper_bound(3.5)
print(f"  e_upper = {e_upper:.15f}")
assert e_upper > e_exact_1, "Upper bound must be > ground energy"
print(f"  ✅ e_upper ({e_upper:.6f}) > e_ground ({e_exact_1:.6f})")

# Test 5: Upper bound is cached
print("\n=== Test 5: Upper bound cache ===")
e_upper_2 = _compute_upper_bound(3.5)
assert e_upper == e_upper_2, "Upper bound not bit-exact on second call"
print("  ✅ Upper bound bit-exact on repeated call")

# Test 6: Physical sanity — ground energy should be negative for h>0
print("\n=== Test 6: Physical sanity ===")
assert e_exact_1 < 0, f"Ground energy should be negative, got {e_exact_1}"
assert gap_1 > 0, f"Gap should be positive, got {gap_1}"
print("  ✅ e_exact < 0 and gap > 0")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✅")
