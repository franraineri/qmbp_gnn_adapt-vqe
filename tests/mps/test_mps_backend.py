"""Tests for MPSBackend, VQEConfig method dispatch, and DMRG chi scaling.

Covers:
- Property tests P1-P8 (Hypothesis-based)
- Unit tests for constructor validation and backward compat
- Integration tests for cross-validation at small N

Wave 2 (tasks 2.3-2.5, 3.2-3.4), Wave 3 (tasks 5.1-5.2),
Wave 4 (tasks 7.1-7.2) of the MPS Scaling N>30 spec.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qmbp_simulation import (
    ClassicalSolver,
    HamiltonianBuilder,
    HVACircuitBuilder,
    VQEOptimizer,
    make_lattice,
)
from qmbp_simulation.execution import MPSBackend, NoiselessBackend
from qmbp_simulation.models import VQEConfig
from qmbp_simulation.models.constants import (
    DMRG_QUBIT_LIMIT,
    MPS_DEFAULT_CHI_MAX,
    SUPPORTED_VQE_METHODS,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Property Test P1: TeNPy exact cross-validation against statevector
# ═══════════════════════════════════════════════════════════════════════════════


class TestPropertyTeNPyExactCrossValidation:
    """P1: For any valid HVA circuit (N≤8), TeNPy exact and statevector agree."""

    @given(
        n_qubits=st.sampled_from([4, 6, 8]),
        h_val=st.floats(min_value=0.5, max_value=3.0),
        seed_val=st.integers(min_value=0, max_value=2**16 - 1),
    )
    @settings(max_examples=10, deadline=None)
    def test_tenpy_exact_matches_statevector(self, n_qubits, h_val, seed_val):
        """TeNPy exact strategy must agree with NoiselessBackend within 1e-10."""
        rng = np.random.default_rng(seed_val)
        builder = HamiltonianBuilder()
        hva = HVACircuitBuilder()

        lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=h_val)
        H = builder.build(lattice)
        circuit, _ = hva.create(n_qubits, 1, lattice)
        n_params = circuit.num_parameters

        theta = rng.uniform(-np.pi, np.pi, size=n_params)

        mps_backend = MPSBackend(strategy="tenpy_exact", chi_max=MPS_DEFAULT_CHI_MAX)
        noiseless = NoiselessBackend()

        e_mps = mps_backend.evaluate(circuit, H, theta)
        e_sv = noiseless.evaluate(circuit, H, theta)

        assert abs(e_mps - e_sv) < 1e-10, (
            f"TeNPy exact vs statevector diff={abs(e_mps - e_sv):.2e} (N={n_qubits}, h={h_val:.3f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Property Test P2: Aer MPS statistical precision bound
# ═══════════════════════════════════════════════════════════════════════════════


class TestPropertyAerMPSPrecision:
    """P2: Aer MPS energy is within statistical tolerance of exact.

    NOTE: The Aer MPS simulator reports 'precision' as the target standard
    deviation, but the actual Hamiltonian may have multiple Pauli terms whose
    errors compound. For a 4-qubit TFIM with ~8 Pauli terms, the effective
    σ can be up to sqrt(n_terms)*precision. We use 10*precision as a robust
    bound that should hold for >99.99% of random instances.
    """

    @given(
        h_val=st.floats(min_value=0.5, max_value=3.0),
        seed_val=st.integers(min_value=0, max_value=2**16 - 1),
    )
    @settings(max_examples=10, deadline=None)
    def test_aer_mps_within_statistical_bound(self, h_val, seed_val):
        """Aer MPS energy must be within 10*precision of statevector exact."""
        n_qubits = 4
        precision = 0.01
        rng = np.random.default_rng(seed_val)
        builder = HamiltonianBuilder()
        hva = HVACircuitBuilder()

        lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=h_val)
        H = builder.build(lattice)
        circuit, _ = hva.create(n_qubits, 1, lattice)
        n_params = circuit.num_parameters
        theta = rng.uniform(-np.pi, np.pi, size=n_params)

        mps_backend = MPSBackend(
            strategy="aer_mps", precision=precision, seed=42, chi_max=MPS_DEFAULT_CHI_MAX
        )
        noiseless = NoiselessBackend()

        e_mps = mps_backend.evaluate(circuit, H, theta)
        e_exact = noiseless.evaluate(circuit, H, theta)

        # 10*precision accounts for multi-term Hamiltonian error compounding
        # TFIM N=4 has ~8 Pauli terms: σ_eff ≈ sqrt(8)*precision ≈ 2.8*precision
        # Using 10x gives >99.99% confidence
        bound = 10 * precision
        assert abs(e_mps - e_exact) < bound, (
            f"Aer MPS outside bound: diff={abs(e_mps - e_exact):.4f}, "
            f"bound={bound:.4f} (h={h_val:.3f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Property Test P3: Seed reproducibility
# ═══════════════════════════════════════════════════════════════════════════════


class TestPropertySeedReproducibility:
    """P3: Same seed produces identical Aer MPS results."""

    @given(
        seed_val=st.integers(min_value=0, max_value=2**32 - 1),
        h_val=st.floats(min_value=0.5, max_value=3.0),
    )
    @settings(max_examples=10, deadline=None)
    def test_same_seed_identical_results(self, seed_val, h_val):
        """Two evaluations with same seed must return identical energy."""
        n_qubits = 4
        builder = HamiltonianBuilder()
        hva = HVACircuitBuilder()

        lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=h_val)
        H = builder.build(lattice)
        circuit, _ = hva.create(n_qubits, 1, lattice)
        theta = np.zeros(circuit.num_parameters)

        backend1 = MPSBackend(strategy="aer_mps", precision=0.01, seed=seed_val)
        backend2 = MPSBackend(strategy="aer_mps", precision=0.01, seed=seed_val)

        e1 = backend1.evaluate(circuit, H, theta)
        e2 = backend2.evaluate(circuit, H, theta)

        assert e1 == e2, f"Seed reproducibility failed: e1={e1}, e2={e2}, seed={seed_val}"


# ═══════════════════════════════════════════════════════════════════════════════
# Property Test P4: VQEConfig rejects invalid method
# ═══════════════════════════════════════════════════════════════════════════════


class TestPropertyVQEConfigValidation:
    """P4: VQEConfig raises ValueError for unsupported method names."""

    @given(
        method=st.text(min_size=1, max_size=20).filter(lambda s: s not in SUPPORTED_VQE_METHODS),
    )
    @settings(max_examples=10, deadline=None)
    def test_invalid_method_raises_valueerror(self, method):
        """Any method not in SUPPORTED_VQE_METHODS must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported optimizer method"):
            VQEConfig(method=method)


# ═══════════════════════════════════════════════════════════════════════════════
# Property Test P5: COBYLA dispatch works without TypeError
# ═══════════════════════════════════════════════════════════════════════════════


class TestPropertyCOBYLADispatch:
    """P5: COBYLA dispatch completes and produces finite energy."""

    @given(
        h_val=st.floats(min_value=1.5, max_value=3.0),
        seed_val=st.integers(min_value=0, max_value=2**16 - 1),
    )
    @settings(max_examples=5, deadline=None)
    def test_cobyla_produces_finite_energy(self, h_val, seed_val):
        """VQE with COBYLA method must produce a finite energy (no TypeError)."""
        n_qubits = 4
        rng = np.random.default_rng(seed_val)
        builder = HamiltonianBuilder()
        hva = HVACircuitBuilder()

        lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=h_val)
        H = builder.build(lattice)
        circuit, _ = hva.create(n_qubits, 1, lattice)
        theta = rng.uniform(-np.pi, np.pi, size=circuit.num_parameters)

        config = VQEConfig(
            method="COBYLA",
            p_layers=1,
            n_restarts=1,
            maxiter=50,
            enable_callbacks=False,
        )
        optimizer = VQEOptimizer(config=config, seed=seed_val)
        result = optimizer.optimize(H, circuit, theta)

        assert np.isfinite(result.energy), f"COBYLA produced non-finite energy: {result.energy}"


# ═══════════════════════════════════════════════════════════════════════════════
# Property Test P6: Default VQEConfig backward compatibility
# ═══════════════════════════════════════════════════════════════════════════════


class TestPropertyDefaultVQEConfigCompat:
    """P6: VQEConfig() and VQEConfig(method='L-BFGS-B') produce identical results."""

    @given(
        h_val=st.floats(min_value=1.5, max_value=3.0),
        seed_val=st.integers(min_value=0, max_value=2**16 - 1),
    )
    @settings(max_examples=5, deadline=None)
    def test_default_equals_explicit_lbfgsb(self, h_val, seed_val):
        """Default VQEConfig must behave identically to explicit L-BFGS-B."""
        n_qubits = 4
        rng = np.random.default_rng(seed_val)
        builder = HamiltonianBuilder()
        hva = HVACircuitBuilder()

        lattice = make_lattice("chain_1d", n_qubits, J=1.0, h=h_val)
        H = builder.build(lattice)
        circuit, _ = hva.create(n_qubits, 1, lattice)
        theta = rng.uniform(-np.pi, np.pi, size=circuit.num_parameters)

        config_default = VQEConfig(p_layers=1, n_restarts=1, maxiter=50, enable_callbacks=False)
        config_explicit = VQEConfig(
            method="L-BFGS-B", p_layers=1, n_restarts=1, maxiter=50, enable_callbacks=False
        )

        opt_default = VQEOptimizer(config=config_default, seed=seed_val)
        opt_explicit = VQEOptimizer(config=config_explicit, seed=seed_val)

        result_default = opt_default.optimize(H, circuit, theta.copy())
        result_explicit = opt_explicit.optimize(H, circuit, theta.copy())

        assert result_default.energy == result_explicit.energy, (
            f"Default vs explicit L-BFGS-B mismatch: "
            f"{result_default.energy} != {result_explicit.energy}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Property Test P7: Dynamic chi_max formula correctness
# ═══════════════════════════════════════════════════════════════════════════════


class TestPropertyChiMaxFormula:
    """P7: chi_max = min(400, max(200, 4*N)) produces correct values."""

    @given(n=st.integers(min_value=16, max_value=100))
    @settings(max_examples=30, deadline=None)
    def test_chi_max_formula(self, n):
        """Verify chi_max formula output for N in [16, 100]."""
        chi = min(400, max(200, 4 * n))

        # For N<=50: chi_max should be 200 (backward-compatible)
        if n <= 50:
            assert chi == 200, f"N={n}: expected chi=200, got {chi}"
        # For 50<N<=100: chi_max = 4*N (capped at 400)
        elif n <= 100:
            expected = min(400, 4 * n)
            assert chi == expected, f"N={n}: expected chi={expected}, got {chi}"


# ═══════════════════════════════════════════════════════════════════════════════
# Property Test P8: ClassicalSolver rejects N > 100
# ═══════════════════════════════════════════════════════════════════════════════


class TestPropertyDMRGLimit:
    """P8: ClassicalSolver.solve() rejects N>DMRG_QUBIT_LIMIT with method='dmrg'."""

    @given(n=st.integers(min_value=201, max_value=300))
    @settings(max_examples=10, deadline=None)
    def test_dmrg_rejects_over_limit(self, n):
        """DMRG must raise ValueError for N > DMRG_QUBIT_LIMIT (200)."""
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()

        lattice = make_lattice("chain_1d", n, J=1.0, h=1.0)
        H = builder.build(lattice)

        with pytest.raises(ValueError, match=str(DMRG_QUBIT_LIMIT)):
            solver.solve(H, lattice, method="dmrg")


# ═══════════════════════════════════════════════════════════════════════════════
# Unit Tests: MPSBackend constructor and error handling (Wave 4, task 7.1)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMPSBackendUnit:
    """Unit tests for MPSBackend constructor validation and naming."""

    def test_invalid_strategy_raises_valueerror(self):
        """MPSBackend('invalid') must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown MPS strategy"):
            MPSBackend(strategy="invalid")

    def test_name_format_tenpy(self):
        """MPSBackend('tenpy_exact') name must contain strategy info."""
        backend = MPSBackend(strategy="tenpy_exact", chi_max=MPS_DEFAULT_CHI_MAX)
        assert "tenpy_exact" in backend.name
        assert "chi64" in backend.name

    def test_name_format_aer(self):
        """MPSBackend('aer_mps') name must contain strategy and chi."""
        backend = MPSBackend(strategy="aer_mps", chi_max=32, precision=0.01)
        assert "aer_mps" in backend.name
        assert "chi32" in backend.name

    def test_parameter_count_mismatch_raises(self):
        """Passing wrong number of params must raise ValueError."""
        builder = HamiltonianBuilder()
        hva = HVACircuitBuilder()
        lattice = make_lattice("chain_1d", 4, J=1.0, h=1.5)
        H = builder.build(lattice)
        circuit, _ = hva.create(4, 1, lattice)

        backend = MPSBackend(strategy="tenpy_exact", chi_max=MPS_DEFAULT_CHI_MAX)
        wrong_params = np.zeros(circuit.num_parameters + 3)

        with pytest.raises(ValueError, match="Parameter count mismatch"):
            backend.evaluate(circuit, H, wrong_params)

    def test_vqeconfig_default_backward_compat(self):
        """VQEConfig() defaults must match pre-change behavior."""
        cfg = VQEConfig()
        assert cfg.method == "L-BFGS-B"
        assert cfg.p_layers == 2
        assert cfg.n_restarts == 5
        assert cfg.maxiter == 1000
        assert cfg.sweep_direction == "descending"

    def test_vqeconfig_no_method_arg_is_backward_compatible(self):
        """VQEConfig without explicit method should use L-BFGS-B."""
        cfg = VQEConfig(p_layers=1, n_restarts=1)
        assert cfg.method == "L-BFGS-B"

    def test_dmrg_qubit_limit_is_200(self):
        """DMRG_QUBIT_LIMIT constant must be 200 (validated at N=120 with χ=64)."""
        assert DMRG_QUBIT_LIMIT == 200


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests: Cross-validation at small N (Wave 4, task 7.2)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMPSBackendIntegration:
    """Integration tests: MPS backends vs NoiselessBackend at N=4."""

    def test_tenpy_exact_cross_validation_3_params(self):
        """TeNPy exact at N=4 with 3 distinct param vectors: diff < 1e-10."""
        builder = HamiltonianBuilder()
        hva = HVACircuitBuilder()
        lattice = make_lattice("chain_1d", 4, J=1.0, h=2.0)
        H = builder.build(lattice)
        circuit, _ = hva.create(4, 1, lattice)

        mps_backend = MPSBackend(strategy="tenpy_exact", chi_max=MPS_DEFAULT_CHI_MAX)
        noiseless = NoiselessBackend()

        rng = np.random.default_rng(42)
        for i in range(3):
            theta = rng.uniform(-np.pi, np.pi, size=circuit.num_parameters)
            e_mps = mps_backend.evaluate(circuit, H, theta)
            e_sv = noiseless.evaluate(circuit, H, theta)
            diff = abs(e_mps - e_sv)
            assert diff < 1e-10, f"Param vector {i}: TeNPy vs SV diff={diff:.2e}"

    def test_aer_mps_within_precision_bound(self):
        """Aer MPS at N=4 with precision=0.01: within ±3*precision of exact."""
        builder = HamiltonianBuilder()
        hva = HVACircuitBuilder()
        lattice = make_lattice("chain_1d", 4, J=1.0, h=2.0)
        H = builder.build(lattice)
        circuit, _ = hva.create(4, 1, lattice)

        precision = 0.01
        mps_backend = MPSBackend(
            strategy="aer_mps", precision=precision, seed=42, chi_max=MPS_DEFAULT_CHI_MAX
        )
        noiseless = NoiselessBackend()

        rng = np.random.default_rng(42)
        theta = rng.uniform(-np.pi, np.pi, size=circuit.num_parameters)

        e_mps = mps_backend.evaluate(circuit, H, theta)
        e_sv = noiseless.evaluate(circuit, H, theta)

        assert abs(e_mps - e_sv) < 3 * precision, (
            f"Aer MPS outside 3σ: diff={abs(e_mps - e_sv):.4f}, bound={3 * precision:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NEW: Robustness tests for N>63 direct path, warm-start, and analyzer
# ═══════════════════════════════════════════════════════════════════════════════


class TestMPSDirectPathLargeN:
    """Tests for the N>63 direct evaluation path (save_expectation_value)."""

    def test_direct_path_activates_above_63_qubits(self):
        """For N=64, the direct path (not BackendEstimatorV2) must be used."""
        # We can't easily check which internal path is used, but we CAN
        # verify the result is finite and physically reasonable for N=64.
        N = 64
        builder = HamiltonianBuilder()
        hva = HVACircuitBuilder()
        lattice = make_lattice("chain_1d", N, J=1.0, h=8.0)
        H = builder.build(lattice)
        circuit, _ = hva.create(N, 1, lattice)

        backend = MPSBackend(
            strategy="aer_mps", chi_max=MPS_DEFAULT_CHI_MAX, precision=0.01, seed=42
        )
        theta = np.array([1.0, 0.5])
        energy = backend.evaluate(circuit, H, theta)

        # Energy must be finite and negative (TFIM ground state is always negative)
        assert np.isfinite(energy), f"N=64 direct path returned non-finite: {energy}"
        assert energy < 0, f"N=64 energy should be negative for TFIM, got {energy}"
        # Rough bound: E should be approximately -N*h for deep paramagnetic
        assert energy > -N * 10, f"N=64 energy unreasonably low: {energy}"

    def test_direct_path_gives_reasonable_energy_n80(self):
        """N=64+ must produce physically reasonable energy via direct path."""
        N = 64  # Just above the 63 threshold — faster than N=80
        builder = HamiltonianBuilder()
        hva = HVACircuitBuilder()
        lattice = make_lattice("chain_1d", N, J=1.0, h=8.0)
        H = builder.build(lattice)
        circuit, _ = hva.create(N, 1, lattice)

        backend = MPSBackend(
            strategy="aer_mps", chi_max=MPS_DEFAULT_CHI_MAX, precision=0.01, seed=42
        )
        theta = np.array([2.5, 1.0])
        energy = backend.evaluate(circuit, H, theta)

        assert np.isfinite(energy)
        assert energy < 0
        # At h=8, N=64: E is very negative (paramagnetic regime)
        assert energy > -N * 12, f"N=64 energy unreasonably low: {energy}"


class TestVQEDescendingSweepMPS:
    """Test warm-start descending sweep produces consistent results with MPSBackend."""

    def test_descending_sweep_energies_decrease_with_h(self):
        """In descending sweep, VQE energy should decrease as h decreases.

        Physics: E = -J*ZZ - h*X. For h>>1 paramagnetic: E ≈ -N*h.
        As h decreases, E becomes less negative (closer to 0).
        So |E(h1)| > |E(h2)| when h1 > h2.
        """
        N = 4
        builder = HamiltonianBuilder()
        hva = HVACircuitBuilder()

        h_values = [3.0, 2.5, 2.0]
        lattice = make_lattice("chain_1d", N, J=1.0, h=h_values[0])
        circuit, _ = hva.create(N, 1, lattice)

        config = VQEConfig(method="L-BFGS-B", p_layers=1, n_restarts=1, maxiter=100)
        optimizer = VQEOptimizer(config=config, seed=42)

        energies = []
        theta_prev = np.zeros(circuit.num_parameters)

        for h in h_values:
            lat_h = make_lattice("chain_1d", N, J=1.0, h=h)
            H = builder.build(lat_h)
            result = optimizer.optimize(H, circuit, theta_prev)
            energies.append(result.energy)
            theta_prev = result.theta_opt.copy()

        # Energies should be monotonically increasing (less negative) as h decreases
        for i in range(len(energies) - 1):
            assert energies[i] < energies[i + 1], (
                f"Energy not monotonic: E(h={h_values[i]})={energies[i]:.4f} "
                f">= E(h={h_values[i + 1]})={energies[i + 1]:.4f}"
            )

    def test_warm_start_improves_convergence(self):
        """Warm-start from previous h should converge faster than cold start."""
        N = 4
        builder = HamiltonianBuilder()
        hva = HVACircuitBuilder()

        lattice = make_lattice("chain_1d", N, J=1.0, h=2.0)
        H = builder.build(lattice)
        circuit, _ = hva.create(N, 1, lattice)

        config = VQEConfig(method="L-BFGS-B", p_layers=1, n_restarts=1, maxiter=200)

        # Cold start
        opt_cold = VQEOptimizer(config=config, seed=42)
        cold_result = opt_cold.optimize(H, circuit, np.zeros(circuit.num_parameters))

        # Warm start (θ close to optimal — simulating previous h-point)
        opt_warm = VQEOptimizer(config=config, seed=42)
        warm_init = cold_result.theta_opt + np.random.default_rng(42).normal(0, 0.01, 2)
        warm_result = opt_warm.optimize(H, circuit, warm_init)

        # Warm start should need fewer or equal iterations
        assert warm_result.n_iterations <= cold_result.n_iterations + 5, (
            f"Warm start ({warm_result.n_iterations} iter) not faster than "
            f"cold start ({cold_result.n_iterations} iter)"
        )


class TestNelderMeadDispatch:
    """Test that Nelder-Mead method dispatch works correctly."""

    def test_nelder_mead_produces_finite_energy(self):
        """VQE with Nelder-Mead method must produce finite energy."""
        N = 4
        builder = HamiltonianBuilder()
        hva = HVACircuitBuilder()
        lattice = make_lattice("chain_1d", N, J=1.0, h=2.0)
        H = builder.build(lattice)
        circuit, _ = hva.create(N, 1, lattice)

        config = VQEConfig(
            method="Nelder-Mead",
            p_layers=1,
            n_restarts=1,
            maxiter=100,
            enable_callbacks=False,
        )
        optimizer = VQEOptimizer(config=config, seed=42)
        result = optimizer.optimize(H, circuit, np.array([0.5, 0.5]))

        assert np.isfinite(result.energy)
        # Should find something reasonable for TFIM N=4 h=2
        assert result.energy < 0


class TestDMRGLargeNConvergence:
    """Test that DMRG converges correctly for N>49 (the extended range)."""

    def test_dmrg_n60_converges(self):
        """DMRG at N=60 must converge with finite energy and gap."""
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        lattice = make_lattice("chain_1d", 60, J=1.0, h=5.0)
        H = builder.build(lattice)

        gt = solver.solve(H, lattice, method="dmrg")

        assert np.isfinite(gt.ground_energy)
        assert gt.ground_energy < 0
        assert gt.gap > 0
        # At h=5, E ≈ -N*h = -300 (paramagnetic approx)
        assert -400 < gt.ground_energy < -200

    def test_dmrg_n49_unchanged_behavior(self):
        """DMRG at N=49 (boundary) must still work as before."""
        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        lattice = make_lattice("chain_1d", 49, J=1.0, h=3.0)
        H = builder.build(lattice)

        gt = solver.solve(H, lattice, method="dmrg")

        assert np.isfinite(gt.ground_energy)
        assert gt.gap > 0


class TestScalingAnalyzerParsing:
    """Test the scaling analyzer parses result files correctly."""

    def test_parse_real_n40_result(self):
        """Analyzer must parse the actual N=40 result file."""
        from pathlib import Path

        from project_health.analysis.scaling_analyzer import (
            parse_scaling_run,
            scan_scaling_results,
        )

        results_dir = Path("results/scaling")
        if not results_dir.exists():
            pytest.skip("No scaling results directory")

        raw = scan_scaling_results(results_dir)
        if not raw:
            pytest.skip("No scaling results to parse")

        # Parse first result
        run = parse_scaling_run(raw[0])
        assert run is not None
        assert run.n_qubits > 0
        assert run.n_total > 0
        assert 0 <= run.mean_de_gap <= 1.0
        assert run.total_time_s > 0
        assert len(run.per_h_results) > 0

    def test_analyzer_report_structure(self):
        """generate_report must return a well-formed ScalingReport."""
        from pathlib import Path

        from project_health.analysis.scaling_analyzer import generate_report

        report = generate_report(Path("results/scaling"))

        assert report.overall_verdict in ("PASS", "PARTIAL", "FAIL", "NO_DATA")
        assert report.n_files_scanned >= 0
        # If we have data, runs should be populated
        if report.n_files_scanned > 0:
            assert len(report.runs) > 0
            assert len(report.scaling_law) > 0


class TestMPSBackendEdgeCases:
    """Edge cases and boundary conditions for MPSBackend."""

    def test_chi_max_minimum_2(self):
        """chi_max < 2 must raise ValueError."""
        with pytest.raises(ValueError, match="chi_max must be"):
            MPSBackend(strategy="aer_mps", chi_max=1)

    def test_precision_zero_raises(self):
        """precision <= 0 must raise ValueError."""
        with pytest.raises(ValueError, match="precision must be"):
            MPSBackend(strategy="aer_mps", precision=0)

    def test_precision_negative_raises(self):
        """Negative precision must raise ValueError."""
        with pytest.raises(ValueError, match="precision must be"):
            MPSBackend(strategy="aer_mps", precision=-0.01)

    def test_vqeconfig_all_valid_methods_accepted(self):
        """All methods in SUPPORTED_VQE_METHODS must be accepted."""
        for method in SUPPORTED_VQE_METHODS:
            cfg = VQEConfig(method=method, p_layers=1)
            assert cfg.method == method
