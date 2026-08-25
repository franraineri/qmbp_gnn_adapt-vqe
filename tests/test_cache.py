"""Comprehensive EvalCache reliability test.

Verifies:
1. Key includes J → different J = different key (no stale cache hits)
2. Energy sanity rejection (|E| > 1e6)
3. NaN/Inf rejection 
4. Full precision hash (no false hits at 1e-10 diff)
5. CachedBackend with real pipeline (end-to-end)
6. Validate entry catches corruption
7. Key format includes all discriminating fields
"""
import sys
import numpy as np
from pathlib import Path
from qmbp_simulation.execution.eval_cache import EvalCache, CachedBackend
from qmbp_simulation.execution import NoiselessBackend
from qmbp_simulation import make_lattice, HamiltonianBuilder, HVACircuitBuilder

CACHE_PATH = Path("/tmp/qmbp_reliability_test.json")
CACHE_PATH.unlink(missing_ok=True)

errors = []

def check(name, condition, msg=""):
    if not condition:
        errors.append(f"{name}: {msg}")
        print(f"  ❌ {name}: {msg}")
    else:
        print(f"  ✅ {name}")

print("=" * 60)
print("  EvalCache Reliability & Data Integrity Test")
print("=" * 60)

cache = EvalCache(path=CACHE_PATH)
theta = np.array([0.1, -0.05, 0.2, -0.15], dtype=np.float64)

# ─── Test 1: J in key ────────────────────────────────────────────────
print("\n[1] J parameter in cache key...")
k1 = cache.make_key("chain_1d", 6, 2.0, theta, model="tfim", p_layers=1, J=1.0)
k2 = cache.make_key("chain_1d", 6, 2.0, theta, model="tfim", p_layers=1, J=0.5)
check("Different J → different key", k1 != k2, f"\n  k1={k1}\n  k2={k2}")
check("J visible in key", "J1.0000" in k1, k1)

# ─── Test 2: Energy sanity rejection ─────────────────────────────────
print("\n[2] Energy sanity guard...")
pre_len = len(cache)
cache.put("sane_key", -7.5)
check("Normal energy cached", len(cache) == pre_len + 1)
cache.put("insane_key", 1e7)
check("|E| > 1e6 rejected", len(cache) == pre_len + 1)
cache.put("neg_insane", -2e6)
check("Large negative rejected", len(cache) == pre_len + 1)

# ─── Test 3: NaN/Inf guard ───────────────────────────────────────────
print("\n[3] NaN/Inf guard...")
pre_len2 = len(cache)
cache.put("nan1", float("nan"))
cache.put("inf1", float("inf"))
cache.put("neginf1", float("-inf"))
check("Non-finite all rejected", len(cache) == pre_len2)

# ─── Test 4: Full precision (no false hits) ──────────────────────────
print("\n[4] Full precision hash...")
theta_a = np.array([0.123456789012345, 0.987654321098765])
theta_b = theta_a.copy()
theta_b[0] += 1e-15  # Smallest representable difference
k_a = cache.make_key("chain_1d", 4, 1.0, theta_a)
k_b = cache.make_key("chain_1d", 4, 1.0, theta_b)
check("1e-15 diff → different key", k_a != k_b)

# Same theta, different instance (should match)
theta_c = np.array([0.123456789012345, 0.987654321098765])
k_c = cache.make_key("chain_1d", 4, 1.0, theta_c)
check("Same values → same key", k_a == k_c)

# ─── Test 5: End-to-end with real backend ────────────────────────────
print("\n[5] End-to-end real backend...")
lattice = make_lattice("chain_1d", 4, J=1.0, h=2.5)
builder = HamiltonianBuilder()
H = builder.build(lattice)
hva = HVACircuitBuilder()
circuit, _ = hva.create(4, 1, lattice)
real_theta = np.random.default_rng(42).uniform(-0.1, 0.1, circuit.num_parameters)

backend = NoiselessBackend()
e2e_cache = EvalCache(path=Path("/tmp/qmbp_e2e.json"))
cached_be = CachedBackend(backend, topology="chain_1d", n_qubits=4, p_layers=1,
                           J=1.0, cache=e2e_cache)
cached_be.set_h(2.5)

# Compute fresh
e_fresh = cached_be.evaluate(circuit, H, real_theta)
check("Fresh compute works", np.isfinite(e_fresh), f"E={e_fresh}")

# Verify it's cached
e_cached = cached_be.evaluate(circuit, H, real_theta)
check("Cached returns identical", e_fresh == e_cached)

# Verify with different J → miss
lattice_j2 = make_lattice("chain_1d", 4, J=2.0, h=2.5)
H_j2 = builder.build(lattice_j2)
cached_j2 = CachedBackend(backend, topology="chain_1d", n_qubits=4, p_layers=1,
                           J=2.0, cache=e2e_cache)
cached_j2.set_h(2.5)
e_j2 = cached_j2.evaluate(circuit, H_j2, real_theta)
check("J=2 gives different energy", e_j2 != e_fresh, f"J1={e_fresh}, J2={e_j2}")
Path("/tmp/qmbp_e2e.json").unlink(missing_ok=True)

# ─── Test 6: Validate entry catches corruption ───────────────────────
print("\n[6] Validate entry detects corruption...")
# Manually inject wrong value
corrupt_cache = EvalCache(path=Path("/tmp/qmbp_corrupt.json"))
corrupt_key = corrupt_cache.make_key("chain_1d", 4, 2.5, real_theta, p_layers=1, J=1.0)
corrupt_cache.put(corrupt_key, -999.0)  # Wrong value
valid = corrupt_cache.validate_entry(corrupt_key, backend, circuit, H, real_theta)
check("Corruption detected", not valid)
# Entry should be removed after failed validation
check("Stale entry removed", corrupt_cache.get(corrupt_key) is None)
Path("/tmp/qmbp_corrupt.json").unlink(missing_ok=True)

# ─── Test 7: Key format completeness ─────────────────────────────────
print("\n[7] Key format includes all discriminating fields...")
key = cache.make_key("heavy_hex", 10, 1.5, theta, model="heisenberg", p_layers=3, J=0.8)
check("model in key", "heisenberg" in key, key)
check("topology in key", "heavy_hex" in key, key)
check("n_qubits in key", "|10|" in key, key)
check("p_layers in key", "|3|" in key, key)
check("J in key", "J0.8000" in key, key)
check("h in key", "1.50000000" in key, key)
check("theta hash in key", len(key.split("|")[-1]) == 32, f"hash part: {key.split('|')[-1]}")

# ─── Summary ─────────────────────────────────────────────────────────
CACHE_PATH.unlink(missing_ok=True)
print("\n" + "=" * 60)
if errors:
    print(f"  FAILED: {len(errors)} error(s)")
    for e in errors:
        print(f"    - {e}")
    if __name__ == "__main__":
        sys.exit(1)
else:
    print("  ALL RELIABILITY TESTS PASSED ✅")
    print("  Data integrity verified: what goes in = what comes out")
    if __name__ == "__main__":
        sys.exit(0)
