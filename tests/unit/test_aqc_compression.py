"""Unit tests for AQC-Tensor circuit compression module.

Tests validate:
- Config dataclass creation and defaults
- Compression of trivial circuits (identity check)
- Compression p=2 → p=1 on small system (N=4)
- Fidelity threshold enforcement
- Validation logic (CompressionValidation)
- Graceful skip if qiskit-addon-aqc-tensor not installed
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if AQC-Tensor not installed
pytest.importorskip("qiskit_addon_aqc_tensor", reason="qiskit-addon-aqc-tensor not installed")


from qmbp_simulation import (
    ClassicalSolver,
    HamiltonianBuilder,
    HVACircuitBuilder,
    VQEConfig,
    VQEOptimizer,
    make_lattice,
)
from qmbp_simulation.circuits.aqc_compression import (
    AQCCircuitCompressor,
    AQCCompressionConfig,
    AQCCompressionResult,
    CompressionValidation,
)


class TestAQCCompressionConfig:
    """Tests for configuration dataclass."""

    def test_default_config(self):
        config = AQCCompressionConfig()
        assert config.max_bond_dim == 64
        assert config.cutoff == 1e-8
        assert config.autodiff_backend == "jax"
        assert config.max_iterations == 200
        assert config.fidelity_threshold == 0.998
        assert config.ansatz_source == "auto"

    def test_custom_config(self):
        config = AQCCompressionConfig(max_bond_dim=128, fidelity_threshold=0.9999)
        assert config.max_bond_dim == 128
        assert config.fidelity_threshold == 0.9999


class TestAQCCircuitCompressor:
    """Integration tests for the compressor."""

    @pytest.fixture
    def small_system(self):
        """N=4 chain_1d system for fast tests."""
        lattice = make_lattice("chain_1d", 4, J=1.0, h=3.5)
        builder = HamiltonianBuilder()
        H = builder.build(lattice)
        solver = ClassicalSolver()
        exact = solver.solve(H, lattice)
        return lattice, H, exact

    def test_compressor_init(self):
        compressor = AQCCircuitCompressor()
        assert compressor.config.max_bond_dim == 64

    def test_compressor_custom_config(self):
        config = AQCCompressionConfig(max_bond_dim=32)
        compressor = AQCCircuitCompressor(config)
        assert compressor.config.max_bond_dim == 32

    def test_rejects_unbound_circuit(self, small_system):
        """Compression must reject circuits with free parameters."""
        lattice, H, exact = small_system
        circuit_builder = HVACircuitBuilder()
        circuit, _ = circuit_builder.create(4, 2, lattice)  # Has parameters

        compressor = AQCCircuitCompressor()
        with pytest.raises(ValueError, match="unbound parameters"):
            compressor.compress_circuit(circuit, lattice)

    def test_compress_p2_to_shallow_n4(self, small_system):
        """Compress HVA p=2 N=4 chain_1d. Core functionality test."""
        lattice, H, exact = small_system

        # Get θ_opt via VQE
        circuit_builder = HVACircuitBuilder()
        circuit_p2, _ = circuit_builder.create(4, 2, lattice)
        vqe_config = VQEConfig(n_restarts=3, maxiter=300)
        optimizer = VQEOptimizer(config=vqe_config, seed=42)
        rng = np.random.default_rng(42)
        init = rng.uniform(-0.01, 0.01, circuit_p2.num_parameters)
        vqe_result = optimizer.optimize(H, circuit_p2, init)

        target_circuit = circuit_p2.assign_parameters(vqe_result.theta_opt)

        # Compress
        config = AQCCompressionConfig(max_bond_dim=32, max_iterations=100)
        compressor = AQCCircuitCompressor(config)
        result = compressor.compress_circuit(target_circuit, lattice)

        # Validate result structure
        assert isinstance(result, AQCCompressionResult)
        assert result.compressed_circuit is not None
        assert result.fidelity > 0.99  # At least 99% fidelity on easy system
        assert result.n_2q_compressed < result.n_2q_original
        assert result.wall_clock_s > 0
        assert result.converged
        assert len(result.optimal_params) == result.n_params

    def test_validate_compression(self, small_system):
        """Test validation logic after compression."""
        lattice, H, exact = small_system

        # Get θ_opt via VQE
        circuit_builder = HVACircuitBuilder()
        circuit_p2, _ = circuit_builder.create(4, 2, lattice)
        vqe_config = VQEConfig(n_restarts=3, maxiter=300)
        optimizer = VQEOptimizer(config=vqe_config, seed=42)
        rng = np.random.default_rng(42)
        init = rng.uniform(-0.01, 0.01, circuit_p2.num_parameters)
        vqe_result = optimizer.optimize(H, circuit_p2, init)

        target_circuit = circuit_p2.assign_parameters(vqe_result.theta_opt)

        # Compress and validate
        config = AQCCompressionConfig(max_bond_dim=32, max_iterations=100)
        compressor = AQCCircuitCompressor(config)
        result = compressor.compress_circuit(target_circuit, lattice)

        validation = compressor.validate_compression(
            result,
            hamiltonian=H,
            energy_exact=exact.ground_energy,
            gap=exact.gap,
            energy_original=vqe_result.energy,
        )

        assert isinstance(validation, CompressionValidation)
        assert validation.energy_exact == exact.ground_energy
        assert validation.gap == exact.gap
        assert validation.delta_e_gap >= 0
        # For this easy system, should be acceptable
        assert validation.delta_e_gap < 0.05  # Within 5% for N=4 p=2

    def test_result_to_dict(self, small_system):
        """Ensure result serialization works."""
        lattice, H, exact = small_system
        circuit_builder = HVACircuitBuilder()
        circuit_p2, _ = circuit_builder.create(4, 2, lattice)
        vqe_config = VQEConfig(n_restarts=2, maxiter=200)
        optimizer = VQEOptimizer(config=vqe_config, seed=42)
        rng = np.random.default_rng(42)
        init = rng.uniform(-0.01, 0.01, circuit_p2.num_parameters)
        vqe_result = optimizer.optimize(H, circuit_p2, init)
        target_circuit = circuit_p2.assign_parameters(vqe_result.theta_opt)

        config = AQCCompressionConfig(max_bond_dim=32, max_iterations=50)
        compressor = AQCCircuitCompressor(config)
        result = compressor.compress_circuit(target_circuit, lattice)

        d = result.to_dict()
        assert "fidelity" in d
        assert "n_2q_original" in d
        assert "config" in d
        assert d["bond_dim_used"] == 32


class TestCompressionValidation:
    """Test CompressionValidation dataclass and serialization."""

    def test_to_dict(self):
        v = CompressionValidation(
            energy_original=-10.0,
            energy_compressed=-9.99,
            energy_exact=-10.01,
            delta_e=0.02,
            delta_e_gap=0.004,
            gap=5.0,
            fidelity=0.9995,
            depth_reduction_pct=35.0,
            n_2q_reduction_pct=50.0,
            acceptable=True,
            recommendation="use_compressed",
        )
        d = v.to_dict()
        assert d["acceptable"] is True
        assert d["recommendation"] == "use_compressed"
        assert d["delta_e_gap"] == 0.004


class TestAQCCompressionResultMethods:
    """Tests for AQCCompressionResult helper methods."""

    def _make_result(self, n_2q_compressed: int = 9) -> AQCCompressionResult:
        from qiskit.circuit import QuantumCircuit

        return AQCCompressionResult(
            compressed_circuit=QuantumCircuit(2),
            optimal_params=np.array([0.1, 0.2]),
            fidelity=0.999,
            depth_original=14,
            depth_compressed=11,
            depth_reduction_pct=21.4,
            n_2q_original=18,
            n_2q_compressed=n_2q_compressed,
            n_2q_reduction_pct=50.0,
            n_params=111,
            n_iterations=26,
            wall_clock_s=1.2,
            converged=True,
            bond_dim_used=64,
        )

    def test_is_zne_viable_pea_below_threshold(self):
        r = self._make_result(n_2q_compressed=9)
        assert r.is_zne_viable(amplifier="pea") is True  # 9 < 50

    def test_is_zne_viable_pea_above_threshold(self):
        r = self._make_result(n_2q_compressed=55)
        assert r.is_zne_viable(amplifier="pea") is False  # 55 > 50

    def test_is_zne_viable_gate_folding_below(self):
        r = self._make_result(n_2q_compressed=9)
        assert r.is_zne_viable(amplifier="gate_folding") is True  # 9 < 18

    def test_is_zne_viable_gate_folding_above(self):
        r = self._make_result(n_2q_compressed=20)
        assert r.is_zne_viable(amplifier="gate_folding") is False  # 20 > 18

    def test_is_zne_viable_adaptive(self):
        r = self._make_result(n_2q_compressed=45)
        assert r.is_zne_viable(amplifier="adaptive") is True  # 45 < 50

    def test_is_zne_viable_unknown_amplifier_defaults_50(self):
        r = self._make_result(n_2q_compressed=45)
        assert r.is_zne_viable(amplifier="future_method") is True  # default 50


class TestAQCCompressionCache:
    """Tests for the compression cache."""

    @pytest.fixture
    def cache(self, tmp_path):
        from qmbp_simulation.circuits.aqc_compression import AQCCompressionCache

        return AQCCompressionCache(cache_dir=tmp_path / "test_cache")

    def test_cache_empty_initially(self, cache):
        assert cache.size == 0

    def test_cache_miss_returns_none(self, cache):
        theta = np.array([1.0, 2.0, 3.0])
        result = cache.get("chain_1d", 10, 3.5, theta, 64)
        assert result is None

    def test_cache_put_and_get(self, cache):
        theta = np.array([1.0, 2.0, 3.0, 4.0])
        optimal = np.array([0.5, 0.6, 0.7])
        cache.put("chain_1d", 10, 3.5, theta, 64, optimal, fidelity=0.999)
        assert cache.size == 1

        retrieved = cache.get("chain_1d", 10, 3.5, theta, 64)
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, optimal)

    def test_cache_different_theta_misses(self, cache):
        theta1 = np.array([1.0, 2.0, 3.0, 4.0])
        theta2 = np.array([1.0, 2.0, 3.0, 5.0])  # Different
        optimal = np.array([0.5, 0.6])
        cache.put("chain_1d", 10, 3.5, theta1, 64, optimal, fidelity=0.999)

        result = cache.get("chain_1d", 10, 3.5, theta2, 64)
        assert result is None

    def test_cache_different_bond_dim_misses(self, cache):
        theta = np.array([1.0, 2.0, 3.0, 4.0])
        optimal = np.array([0.5, 0.6])
        cache.put("chain_1d", 10, 3.5, theta, 64, optimal, fidelity=0.999)

        result = cache.get("chain_1d", 10, 3.5, theta, 128)  # Different chi
        assert result is None

    def test_cache_get_with_metadata(self, cache):
        theta = np.array([1.0, 2.0])
        optimal = np.array([0.5])
        cache.put("heavy_hex", 10, 4.0, theta, 32, optimal, fidelity=0.9995, extra="info")

        got = cache.get_with_metadata("heavy_hex", 10, 4.0, theta, 32)
        assert got is not None
        params, meta = got
        np.testing.assert_array_almost_equal(params, optimal)
        assert meta["fidelity"] == 0.9995
        assert meta["topology"] == "heavy_hex"

    def test_cache_clear(self, cache):
        theta = np.array([1.0, 2.0])
        cache.put("chain_1d", 6, 2.0, theta, 64, np.array([0.1]), fidelity=0.99)
        cache.put("chain_1d", 6, 3.0, theta, 64, np.array([0.2]), fidelity=0.99)
        assert cache.size == 2
        removed = cache.clear()
        assert removed == 2
        assert cache.size == 0


class TestLazyImportsFromCircuits:
    """Test that AQC classes are accessible via circuits.__init__ lazy loading."""

    def test_import_compressor_from_circuits(self):
        from qmbp_simulation.circuits import AQCCircuitCompressor

        assert AQCCircuitCompressor is not None

    def test_import_config_from_circuits(self):
        from qmbp_simulation.circuits import AQCCompressionConfig

        cfg = AQCCompressionConfig()
        assert cfg.max_bond_dim == 64

    def test_import_cache_from_circuits(self):
        from qmbp_simulation.circuits import AQCCompressionCache

        assert AQCCompressionCache is not None

    def test_invalid_attr_raises(self):
        import qmbp_simulation.circuits as circuits_mod

        with pytest.raises(AttributeError):
            _ = circuits_mod.NonExistentClass
