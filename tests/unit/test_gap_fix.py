"""Tests for H2 gap fix, gap_method tracing, exact_ground_state caching, and ResultIndex gap_methods.

Covers:
- ClassicalSolver eigsh fallback for non-chain topologies
- gap_method field on GroundTruthResult
- ValidationRunner.exact_ground_state caching
- ResultIndex query with gap_method filter
- Edge cases: near-degenerate gaps, large N floor fallback, invalid inputs
"""

import numpy as np
import pytest

from qmbp_simulation import ClassicalSolver, make_lattice
from qmbp_simulation.models.constants import EXACT_GAP_QUBIT_LIMIT
from qmbp_simulation.models.data_models import GroundTruthResult
from qmbp_simulation.models.model_registry import get_model_spec

# ═══════════════════════════════════════════════════════════════════════════════
# gap_method tracing
# ═══════════════════════════════════════════════════════════════════════════════


class TestGapMethod:
    """Verify gap_method is correctly set for different solver paths."""

    @pytest.fixture
    def solver(self):
        return ClassicalSolver()

    @pytest.fixture
    def tfim_spec(self):
        return get_model_spec("tfim")

    def test_exact_dense_gap_method(self, solver, tfim_spec):
        """N<13 uses dense eigh → gap_method='exact_dense'."""
        lattice = make_lattice("chain_1d", 6, J=1.0, h=2.0)
        H = tfim_spec.build_hamiltonian(lattice, **tfim_spec.hamiltonian_kwargs)
        gt = solver.solve(H, lattice, method="exact")
        assert gt.gap_method == "exact_dense"

    def test_exact_sparse_gap_method(self, solver, tfim_spec):
        """N>=13 uses sparse eigsh → gap_method='exact_sparse'."""
        lattice = make_lattice("chain_1d", 14, J=1.0, h=2.0)
        H = tfim_spec.build_hamiltonian(lattice, **tfim_spec.hamiltonian_kwargs)
        gt = solver.solve(H, lattice, method="exact")
        assert gt.gap_method == "exact_sparse"

    def test_dmrg_chain_analytical_gap_method(self, solver, tfim_spec):
        """DMRG on chain_1d falls back to analytical → gap_method='analytical_1d'."""
        import warnings

        lattice = make_lattice("chain_1d", 16, J=1.0, h=2.0)
        H = tfim_spec.build_hamiltonian(lattice, **tfim_spec.hamiltonian_kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            gt = solver.solve(H, lattice, method="dmrg")
        # Could be 'dmrg_excitation' (if excitation succeeds) or 'analytical_1d'
        assert gt.gap_method in ("dmrg_excitation", "analytical_1d")

    def test_dmrg_heavy_hex_eigsh_fallback(self, solver, tfim_spec):
        """heavy_hex N=10 DMRG uses eigsh fallback → gap_method='eigsh_fallback'."""
        lattice = make_lattice("heavy_hex", 10, J=1.0, h=2.0)
        H = tfim_spec.build_hamiltonian(lattice, **tfim_spec.hamiltonian_kwargs)
        gt = solver.solve(H, lattice, method="dmrg")
        assert gt.gap_method == "eigsh_fallback"

    def test_gap_method_default_on_dataclass(self):
        """GroundTruthResult has gap_method='unknown' by default."""
        gt = GroundTruthResult(
            h_value=1.0,
            ground_energy=-5.0,
            gap=0.5,
            ground_state=None,
            mag_x=0.5,
            corr_zz=-0.3,
            per_site_mag_x=np.array([0.5]),
            per_bond_corr_zz=np.array([-0.3]),
        )
        assert gt.gap_method == "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# Eigsh fallback correctness
# ═══════════════════════════════════════════════════════════════════════════════


class TestEigshFallback:
    """Verify eigsh fallback produces correct gaps for non-chain topologies."""

    @pytest.fixture
    def solver(self):
        return ClassicalSolver()

    @pytest.fixture
    def tfim_spec(self):
        return get_model_spec("tfim")

    def test_heavy_hex_gap_differs_from_floor(self, solver, tfim_spec):
        """heavy_hex gap from eigsh should differ significantly from 2π/N floor."""
        N = 10
        lattice = make_lattice("heavy_hex", N, J=1.0, h=2.0)
        H = tfim_spec.build_hamiltonian(lattice, **tfim_spec.hamiltonian_kwargs)
        gt = solver.solve(H, lattice, method="dmrg")
        floor = 2 * np.pi / N
        # Real gap should be much larger than floor at h=2.0 (far from h_c=1)
        assert gt.gap > floor * 1.5, f"Gap {gt.gap} too close to floor {floor}"

    def test_eigsh_matches_exact_diag(self, solver, tfim_spec):
        """For N≤15, eigsh fallback should match exact diag gap."""
        N = 10
        lattice = make_lattice("heavy_hex", N, J=1.0, h=2.0)
        H = tfim_spec.build_hamiltonian(lattice, **tfim_spec.hamiltonian_kwargs)
        gt_exact = solver.solve(H, lattice, method="exact")
        gt_dmrg = solver.solve(H, lattice, method="dmrg")
        # Both should give same gap (eigsh on same matrix)
        assert abs(gt_exact.gap - gt_dmrg.gap) < 1e-6

    def test_ladder_also_uses_eigsh_fallback(self, solver, tfim_spec):
        """ladder topology should also use eigsh (not chain analytical)."""
        lattice = make_lattice("ladder", 8, J=1.0, h=2.0)
        H = tfim_spec.build_hamiltonian(lattice, **tfim_spec.hamiltonian_kwargs)
        gt = solver.solve(H, lattice, method="dmrg")
        assert gt.gap_method == "eigsh_fallback"
        # Verify gap is physically reasonable (not floor)
        floor = 2 * np.pi / 8
        assert gt.gap > floor

    def test_exact_gap_qubit_limit_boundary(self):
        """Verify the constant is 18 (reduced from 20 to avoid macOS ARM64 segfaults)."""
        assert EXACT_GAP_QUBIT_LIMIT == 18
        # N=18 → 2^18 = 262k states → eigsh k=2 is safe on all platforms
        # N=19+ → 2^19 = 524k+ states → segfaults on macOS ARM64 (ARPACK/Accelerate)


# ═══════════════════════════════════════════════════════════════════════════════
# exact_ground_state caching
# ═══════════════════════════════════════════════════════════════════════════════


class TestExactGroundStateCache:
    """Verify caching behavior of ValidationRunner.exact_ground_state."""

    @pytest.fixture
    def runner(self):
        """Create a minimal ValidationRunner instance for testing."""
        import sys

        sys.argv = ["test", "--dry-run"]
        from qmbp_simulation.framework.runner_base import Section, ValidationRunner

        class _TestRunner(ValidationRunner):
            runner_id = "cache_test"
            experiment_id = "test"
            description = "test"
            hypothesis = "test"

            def define_sections(self):
                return [Section(id=1, name="t", fn=lambda: {}, hypothesis="t")]

        return _TestRunner()

    def test_cache_returns_same_values(self, runner):
        """Two calls with same args return identical results."""
        e1, g1 = runner.exact_ground_state("chain_1d", 6, 2.0)
        e2, g2 = runner.exact_ground_state("chain_1d", 6, 2.0)
        assert e1 == e2
        assert g1 == g2

    def test_cache_stores_entry(self, runner):
        """After a call, _gt_cache has one entry."""
        runner.exact_ground_state("chain_1d", 6, 2.0)
        assert len(runner._gt_cache) == 1

    def test_different_h_creates_new_entry(self, runner):
        """Different h-values create separate cache entries."""
        runner.exact_ground_state("chain_1d", 6, 2.0)
        runner.exact_ground_state("chain_1d", 6, 1.5)
        assert len(runner._gt_cache) == 2

    def test_different_topology_creates_new_entry(self, runner):
        """Different topologies create separate cache entries."""
        runner.exact_ground_state("chain_1d", 6, 2.0)
        runner.exact_ground_state("ladder", 6, 2.0)
        assert len(runner._gt_cache) == 2

    def test_cache_key_rounds_h(self, runner):
        """h values rounded to 6 decimals share cache entry."""
        runner.exact_ground_state("chain_1d", 6, 2.0000001)
        runner.exact_ground_state("chain_1d", 6, 2.0000002)
        # Both round to 2.0 at 6 decimals
        assert len(runner._gt_cache) == 1

    def test_cached_call_is_fast(self, runner):
        """Second call should be near-instant (< 1ms)."""
        import time

        runner.exact_ground_state("chain_1d", 6, 2.0)
        t0 = time.time()
        runner.exact_ground_state("chain_1d", 6, 2.0)
        elapsed = time.time() - t0
        assert elapsed < 0.01, f"Cached call took {elapsed:.4f}s — not cached?"


# ═══════════════════════════════════════════════════════════════════════════════
# ResultIndex gap_method querying
# ═══════════════════════════════════════════════════════════════════════════════


class TestResultIndexGapMethod:
    """Verify ResultIndex can filter by gap_method field."""

    def test_extract_gap_methods_from_envelope(self):
        """extract_run_metadata_summary extracts gap_methods correctly."""
        from qmbp_simulation.framework.result_io import extract_run_metadata_summary

        # Simulate a post-fix result envelope
        envelope = {
            "config": {
                "system": {
                    "model": "tfim",
                    "topologies": ["heavy_hex"],
                    "n_qubits": 10,
                    "p_layers": 3,
                },
                "experiment_id": "noiseless/tfim/heavy_hex",
            },
            "summary": {"all_passed": True, "pass_rate": 1.0, "n_sections": 4},
            "timestamp": "2026-07-13T00:00:00",
            "elapsed_s": 100.0,
            "schema_version": "2.0",
            "results": {
                "section_1": {
                    "data": {
                        "topologies": {
                            "heavy_hex": {
                                "points": [
                                    {"h": 2.0, "gap": 1.97, "gap_method": "eigsh_fallback"},
                                    {"h": 1.5, "gap": 1.02, "gap_method": "eigsh_fallback"},
                                ]
                            }
                        }
                    }
                }
            },
        }
        summary = extract_run_metadata_summary(envelope)
        assert summary["gap_methods"] == ["eigsh_fallback"]

    def test_extract_gap_methods_missing(self):
        """Pre-fix envelopes without gap_method → gap_methods=None."""
        from qmbp_simulation.framework.result_io import extract_run_metadata_summary

        envelope = {
            "config": {
                "system": {
                    "model": "tfim",
                    "topologies": ["chain_1d"],
                    "n_qubits": 6,
                    "p_layers": 2,
                },
                "experiment_id": "",
            },
            "summary": {"all_passed": True, "pass_rate": 1.0, "n_sections": 4},
            "timestamp": "2026-06-01T00:00:00",
            "elapsed_s": 50.0,
            "schema_version": "2.0",
            "results": {
                "section_1": {
                    "data": {
                        "topologies": {
                            "chain_1d": {
                                "points": [{"h": 2.0, "gap": 2.0}]  # no gap_method field
                            }
                        }
                    }
                }
            },
        }
        summary = extract_run_metadata_summary(envelope)
        assert summary["gap_methods"] is None

    def test_extract_mixed_gap_methods(self):
        """Envelope with multiple gap methods lists all of them."""
        from qmbp_simulation.framework.result_io import extract_run_metadata_summary

        envelope = {
            "config": {
                "system": {
                    "model": "tfim",
                    "topologies": ["chain_1d"],
                    "n_qubits": 16,
                    "p_layers": 3,
                },
                "experiment_id": "",
            },
            "summary": {"all_passed": True, "pass_rate": 1.0, "n_sections": 4},
            "timestamp": "2026-07-13T00:00:00",
            "elapsed_s": 100.0,
            "schema_version": "2.0",
            "results": {
                "section_1": {
                    "data": {
                        "topologies": {
                            "chain_1d": {
                                "points": [
                                    {"h": 2.0, "gap": 2.0, "gap_method": "analytical_1d"},
                                    {"h": 1.5, "gap": 1.0, "gap_method": "dmrg_excitation"},
                                ]
                            }
                        }
                    }
                }
            },
        }
        summary = extract_run_metadata_summary(envelope)
        assert sorted(summary["gap_methods"]) == ["analytical_1d", "dmrg_excitation"]


# ═══════════════════════════════════════════════════════════════════════════════
# Speedup fit edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpeedupFit:
    """Test fit_speedup_models from analyze_noiseless_scaling.py."""

    def test_fit_with_two_points(self):
        """With only 2 points, all 2-param models have R²=1 (underdetermined)."""
        import sys

        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
        from scripts.analyze_noiseless_scaling import fit_speedup_models

        data = [
            {
                "model": "tfim",
                "topology": "heavy_hex",
                "p_layers": 3,
                "n_scaling": [
                    {"n_qubits": 10, "speedup": 36.0},
                    {"n_qubits": 16, "speedup": 45.0},
                ],
            }
        ]
        results = fit_speedup_models(data, verbose=False)
        assert len(results) == 1
        assert results[0]["confidence"] == "underdetermined"
        assert results[0]["n_points"] == 2

    def test_fit_with_constant_speedup(self):
        """Flat speedup data should select 'constant' model."""
        import sys

        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
        from scripts.analyze_noiseless_scaling import fit_speedup_models

        data = [
            {
                "model": "tfim",
                "topology": "chain_1d",
                "p_layers": 4,
                "n_scaling": [
                    {"n_qubits": 8, "speedup": 60.0},
                    {"n_qubits": 10, "speedup": 58.0},
                    {"n_qubits": 16, "speedup": 62.0},
                    {"n_qubits": 20, "speedup": 59.0},
                ],
            }
        ]
        results = fit_speedup_models(data, verbose=False)
        assert len(results) == 1
        assert results[0]["best_model"] == "constant"
        assert results[0]["confidence"] == "low"  # R²≈0 for constant

    def test_fit_with_no_speedup_data(self):
        """Entries without speedup values are skipped."""
        import sys

        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
        from scripts.analyze_noiseless_scaling import fit_speedup_models

        data = [
            {
                "model": "tfim",
                "topology": "chain_1d",
                "p_layers": 2,
                "n_scaling": [
                    {"n_qubits": 10, "speedup": None},
                    {"n_qubits": 16, "speedup": None},
                ],
            }
        ]
        results = fit_speedup_models(data, verbose=False)
        assert len(results) == 0  # Skipped due to no valid speedup data

    def test_fit_single_point_skipped(self):
        """Only 1 point → skipped (need ≥2)."""
        import sys

        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
        from scripts.analyze_noiseless_scaling import fit_speedup_models

        data = [
            {
                "model": "tfim",
                "topology": "chain_1d",
                "p_layers": 1,
                "n_scaling": [{"n_qubits": 10, "speedup": 30.0}],
            }
        ]
        results = fit_speedup_models(data, verbose=False)
        assert len(results) == 0

    def test_fit_aicc_null_in_output(self):
        """AICc=inf should be serialized as None in output."""
        import sys

        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
        from scripts.analyze_noiseless_scaling import fit_speedup_models

        data = [
            {
                "model": "tfim",
                "topology": "heavy_hex",
                "p_layers": 3,
                "n_scaling": [
                    {"n_qubits": 10, "speedup": 36.0},
                    {"n_qubits": 16, "speedup": 45.0},
                    {"n_qubits": 20, "speedup": 48.0},
                ],
            }
        ]
        results = fit_speedup_models(data, verbose=False)
        # With 3 points, AICc for 2-param models is inf → stored as None
        assert results[0]["AICc"] is None
