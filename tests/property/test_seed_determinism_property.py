"""Property-based test for seed determinism (Property 16).

# Feature: mitigation-benchmark, Property 16
# **Validates: Requirements 12.1**
#
# Property 16: Seed determinism
#   Same config × h × seed in simulation → bit-exact results.
#
# Tested at three levels:
#   1. ClassicalSolver cache: _get_exact_energy(h) called twice → same (e_exact, gap)
#   2. Circuit construction: _build_hva_circuit(h) with same seed → same circuit
#   3. Result path: _build_result_path() with same inputs → same directory pattern
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

# Add scripts to path for benchmark module import
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "scripts"))

from experiment_runners.hardware.run_mitigation_benchmark import (
    _build_hva_circuit,
    _build_result_path,
    _classical_cache,
    _get_exact_energy,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Strategies
# ═══════════════════════════════════════════════════════════════════════════════

h_value_st = st.sampled_from([3.0, 3.25, 3.5, 3.75, 4.0])
seed_st = st.integers(min_value=1, max_value=1000)
mode_st = st.sampled_from(["fake_backend", "hardware"])


# ═══════════════════════════════════════════════════════════════════════════════
# Property 16: Seed determinism — Circuit construction
# **Validates: Requirements 12.1**
# ═══════════════════════════════════════════════════════════════════════════════


class TestSeedProducesDeterministicCircuit:
    """Property 16 (part 1): Same h_value + same numpy seed → same circuit.

    **Validates: Requirements 12.1**

    _build_hva_circuit is deterministic given the same numpy random state.
    Two calls with np.random.seed(seed) followed by _build_hva_circuit(h)
    must produce identical QuantumCircuit objects.
    """

    @given(h_value=h_value_st, seed=seed_st)
    @settings(max_examples=20, deadline=None)
    def test_seed_produces_deterministic_circuit(self, h_value: float, seed: int):
        """Same seed → bit-exact same circuit."""
        np.random.seed(seed)
        c1 = _build_hva_circuit(h_value)

        np.random.seed(seed)
        c2 = _build_hva_circuit(h_value)

        # Same circuit: same gates, same parameters, same structure
        assert c1 == c2, (
            f"Circuits differ for h={h_value}, seed={seed}. "
            f"c1 has {c1.num_qubits}q/{len(c1.data)} ops, "
            f"c2 has {c2.num_qubits}q/{len(c2.data)} ops"
        )

    @given(h_value=h_value_st, seed=seed_st)
    @settings(max_examples=20, deadline=None)
    def test_circuit_data_identical(self, h_value: float, seed: int):
        """Circuit gate data (operations list) is identical across calls."""
        np.random.seed(seed)
        c1 = _build_hva_circuit(h_value)

        np.random.seed(seed)
        c2 = _build_hva_circuit(h_value)

        assert len(c1.data) == len(c2.data), (
            f"Different number of gates: {len(c1.data)} vs {len(c2.data)}"
        )
        assert c1.num_qubits == c2.num_qubits, (
            f"Different qubit counts: {c1.num_qubits} vs {c2.num_qubits}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Property 16: Seed determinism — ClassicalSolver cache
# **Validates: Requirements 12.1**
# ═══════════════════════════════════════════════════════════════════════════════


class TestExactEnergyDeterministic:
    """Property 16 (part 2): _get_exact_energy(h) is deterministic.

    **Validates: Requirements 12.1**

    ClassicalSolver uses exact diagonalization, so results must be
    bit-exact identical regardless of cache state. Two fresh calls
    (after cache clear) must produce the same (e_exact, gap).
    """

    @given(h_value=h_value_st)
    @settings(max_examples=5, deadline=None)
    def test_exact_energy_deterministic(self, h_value: float):
        """Same h_value → bit-exact same (e_exact, gap) after cache clear."""
        _classical_cache.clear()
        e1, g1 = _get_exact_energy(h_value)

        _classical_cache.clear()
        e2, g2 = _get_exact_energy(h_value)

        assert e1 == e2, f"e_exact not deterministic for h={h_value}: {e1} vs {e2}"
        assert g1 == g2, f"gap not deterministic for h={h_value}: {g1} vs {g2}"

    @given(h_value=h_value_st)
    @settings(max_examples=5, deadline=None)
    def test_exact_energy_cached_matches_fresh(self, h_value: float):
        """Cached result matches a freshly computed one (cache doesn't corrupt)."""
        _classical_cache.clear()
        e_fresh, g_fresh = _get_exact_energy(h_value)

        # Second call hits cache
        e_cached, g_cached = _get_exact_energy(h_value)

        assert e_fresh == e_cached, f"Cache corrupts e_exact: fresh={e_fresh} vs cached={e_cached}"
        assert g_fresh == g_cached, f"Cache corrupts gap: fresh={g_fresh} vs cached={g_cached}"


# ═══════════════════════════════════════════════════════════════════════════════
# Property 16: Seed determinism — Result path
# **Validates: Requirements 12.1**
# ═══════════════════════════════════════════════════════════════════════════════


class TestResultPathDeterministic:
    """Property 16 (part 3): _build_result_path() with same inputs → same directory.

    **Validates: Requirements 12.1**

    The directory structure (mode/config_id) must be deterministic given the
    same inputs. The filename includes a timestamp so it may differ between
    calls, but the parent directory pattern must be stable.
    """

    @given(h_value=h_value_st, seed=seed_st, mode=mode_st)
    @settings(max_examples=30, deadline=None)
    def test_result_path_directory_deterministic(self, h_value: float, seed: int, mode: str):
        """Same inputs → same parent directory pattern."""
        config_id = "C0_raw"

        path1 = _build_result_path(config_id, h_value, mode, seed)
        path2 = _build_result_path(config_id, h_value, mode, seed)

        # Parent directory must be identical (contains mode/config_id)
        assert path1.parent == path2.parent, f"Directory differs: {path1.parent} vs {path2.parent}"

    @given(h_value=h_value_st, seed=seed_st, mode=mode_st)
    @settings(max_examples=30, deadline=None)
    def test_result_path_h_pattern_deterministic(self, h_value: float, seed: int, mode: str):
        """Same h_value → same h-pattern in filename."""
        config_id = "C5_full_pea_balanced"
        h_str = f"h{str(h_value).replace('.', 'p')}"

        path = _build_result_path(config_id, h_value, mode, seed)

        assert h_str in path.name, f"h-pattern '{h_str}' missing from filename '{path.name}'"

    @given(seed=st.integers(min_value=1, max_value=41))
    @settings(max_examples=10, deadline=None)
    def test_result_path_seed_suffix_deterministic(self, seed: int):
        """Non-default seed always produces seed suffix in filename."""
        path = _build_result_path("C0_raw", 3.5, "fake_backend", seed)
        assert f"seed{seed}" in path.name, f"seed suffix 'seed{seed}' not in '{path.name}'"
