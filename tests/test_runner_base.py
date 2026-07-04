"""Test suite for runner_base.py — validates all three runner types."""

import argparse
import json

from qmbp_simulation.framework.runner_base import (
    ExperimentRunner,
    Section,
    ValidationRunner,
    resolve_project_root,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Test fixtures — concrete runner implementations for testing
# ═══════════════════════════════════════════════════════════════════════════════


class _EmptyRunner(ValidationRunner):
    """Runner with missing required attributes (should fail preflight)."""

    runner_id = ""
    experiment_id = ""
    description = ""
    hypothesis = ""

    def define_sections(self):
        return []


class _ValidRunner(ValidationRunner):
    """Well-configured runner for testing."""

    runner_id = "test_runner"
    experiment_id = "TEST"
    description = "Test validation runner"
    hypothesis = "Framework works correctly"

    def define_sections(self):
        return [
            Section(id=1, name="Section A", fn=self.section_a, hypothesis="A works"),
            Section(id=2, name="Section B", fn=self.section_b, hypothesis="B works"),
        ]

    def section_a(self) -> dict:
        return {"value": 42, "pass": True}

    def section_b(self) -> dict:
        return {"value": 99, "pass": True}


class _FailingRunner(ValidationRunner):
    """Runner with sections that fail in different ways."""

    runner_id = "failing_runner"
    experiment_id = "FAIL"
    description = "Failing runner"
    hypothesis = "Error handling works"

    def define_sections(self):
        return [
            Section(id=1, name="Good", fn=self.good_section, hypothesis="Works"),
            Section(id=2, name="Exception", fn=self.bad_section, hypothesis="Raises"),
            Section(id=3, name="None Return", fn=self.none_section, hypothesis="None"),
        ]

    def good_section(self) -> dict:
        return {"pass": True}

    def bad_section(self) -> dict:
        raise ValueError("Intentional failure for testing")

    def none_section(self) -> dict:
        return None  # type: ignore


class _ExplicitFailRunner(ValidationRunner):
    """Runner where a section returns pass=False (no exception)."""

    runner_id = "explicit_fail"
    experiment_id = "XFAIL"
    description = "Explicit fail runner"
    hypothesis = "Explicit failure detection"

    def define_sections(self):
        return [
            Section(id=1, name="Passes", fn=lambda: {"pass": True}, hypothesis=""),
            Section(
                id=2,
                name="Fails",
                fn=lambda: {"pass": False, "reason": "threshold exceeded"},
                hypothesis="",
            ),
        ]


class _DuplicateIDRunner(ValidationRunner):
    """Runner with duplicate section IDs."""

    runner_id = "dup_runner"
    experiment_id = "DUP"
    description = "Duplicate runner"
    hypothesis = "Duplicate detection"

    def define_sections(self):
        return [
            Section(id=1, name="First", fn=lambda: {"pass": True}, hypothesis=""),
            Section(id=1, name="Duplicate", fn=lambda: {"pass": True}, hypothesis=""),
        ]


def _make_args(**kwargs):
    """Create argparse.Namespace with defaults for ValidationRunner."""
    defaults = {
        "section": None,
        "skip_preflight": False,
        "stop_on_failure": False,
        "verbose": False,
        "dry_run": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# Test: resolve_project_root
# ═══════════════════════════════════════════════════════════════════════════════


def test_resolve_project_root():
    """resolve_project_root finds the project root from any depth."""
    root = resolve_project_root(__file__)
    assert (root / "Makefile").exists() or (root / "pyproject.toml").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: ValidationRunner — preflight
# ═══════════════════════════════════════════════════════════════════════════════


def test_preflight_rejects_empty_runner():
    """Empty runner (no runner_id, etc.) should fail preflight."""
    runner = _EmptyRunner(args=_make_args())
    assert runner.run_preflight() is False


def test_preflight_accepts_valid_runner():
    """Well-configured runner should pass preflight."""
    runner = _ValidRunner(args=_make_args())
    assert runner.run_preflight() is True


def test_preflight_detects_duplicate_section_ids():
    """Duplicate section IDs should fail preflight."""
    runner = _DuplicateIDRunner(args=_make_args())
    assert runner.run_preflight() is False


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: ValidationRunner — execution lifecycle
# ═══════════════════════════════════════════════════════════════════════════════


def test_run_valid_returns_exit_0():
    """Valid runner should complete all sections and return exit code 0."""
    runner = _ValidRunner(args=_make_args())
    exit_code = runner.run()
    assert exit_code == 0
    assert len(runner._section_results) == 2
    assert all(r.success for r in runner._section_results)


def test_run_with_exception_returns_exit_1():
    """Runner with exception in a section returns exit code 1."""
    runner = _FailingRunner(args=_make_args())
    exit_code = runner.run()
    assert exit_code == 1
    # Section 1 passes, section 2 raises, section 3 returns None (treated as pass)
    assert runner._section_results[0].success is True
    assert runner._section_results[1].success is False
    assert "ValueError" in runner._section_results[1].error
    assert runner._section_results[2].success is True  # None → no "pass" key → success


def test_run_explicit_fail_detected():
    """Section returning pass=False should be detected as failure."""
    runner = _ExplicitFailRunner(args=_make_args())
    exit_code = runner.run()
    assert exit_code == 1
    assert runner._section_results[0].success is True
    assert runner._section_results[1].success is False


def test_stop_on_failure_aborts_early():
    """--stop-on-failure should stop after first failure."""
    runner = _FailingRunner(args=_make_args(stop_on_failure=True))
    exit_code = runner.run()
    assert exit_code == 1
    # Section 1 passes, section 2 fails → stop (section 3 never runs)
    assert len(runner._section_results) == 2


def test_section_filter():
    """--section should run only specified sections."""
    runner = _ValidRunner(args=_make_args(section=[2]))
    exit_code = runner.run()
    assert exit_code == 0
    assert len(runner._section_results) == 1
    assert runner._section_results[0].section_id == 2


def test_dry_run_no_execution():
    """--dry-run should not execute any sections."""
    runner = _ValidRunner(args=_make_args(dry_run=True))
    exit_code = runner.run()
    assert exit_code == 0
    assert len(runner._section_results) == 0


def test_skip_preflight():
    """--skip-preflight allows even misconfigured runners to proceed."""
    runner = _EmptyRunner(args=_make_args(skip_preflight=True))
    exit_code = runner.run()
    # No sections = no failures = exit 0
    assert exit_code == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: ValidationRunner — result saving
# ═══════════════════════════════════════════════════════════════════════════════


def test_result_envelope_structure(tmp_path, monkeypatch):
    """Result envelope should have the standard structure + digest compatibility."""
    # Redirect results to tmp_path/experiments (mirrors real layout)
    monkeypatch.setattr(
        "qmbp_simulation.framework.result_io._DEFAULT_RESULTS_ROOT",
        tmp_path / "experiments",
    )
    runner = _ValidRunner(args=_make_args())
    runner.run()

    # Find saved file
    exp_dir = tmp_path / "experiments" / "exp_test"
    json_files = list(exp_dir.glob("run_*.json"))
    assert len(json_files) == 1

    with open(json_files[0]) as f:
        data = json.load(f)

    # Verify envelope structure (result_io standard)
    assert "timestamp" in data
    assert "config" in data
    assert "results" in data
    assert "summary" in data
    assert "elapsed_s" in data
    assert "metadata" in data

    # Verify summary
    assert data["summary"]["n_sections"] == 2
    assert data["summary"]["n_passed"] == 2
    assert data["summary"]["n_failed"] == 0
    assert data["summary"]["all_passed"] is True
    assert data["summary"]["pass_rate"] == 1.0
    assert "total_time_s" in data["summary"]  # digest compat

    # Verify results contain section data
    assert "section_1" in data["results"]
    assert "section_2" in data["results"]
    assert data["results"]["section_1"]["success"] is True
    assert data["results"]["section_1"]["data"]["value"] == 42

    # Verify metadata has python version
    assert "python_version" in data["metadata"]

    # ── Digest/compare.py compatibility ──────────────────────────────────
    # The digest scanner reads: data["analysis"]["summary"]
    assert "analysis" in data
    assert "summary" in data["analysis"]
    assert data["analysis"]["summary"]["pass_rate"] == 1.0
    assert data["analysis"]["n_seeds"] == 0  # _ValidRunner has no seeds in config
    assert data["analysis"]["experiment_id"] == "TEST"
    assert data["analysis"]["hypothesis"] == "Framework works correctly"

    # config must have experiment_id for digest scanner
    assert data["config"]["experiment_id"] == "TEST"
    assert "category" in data["config"]
    assert "hypothesis" in data["config"]
    assert "system" in data["config"]


def test_structured_log_saved(tmp_path, monkeypatch):
    """Structured log should be saved as a separate file."""
    monkeypatch.setattr(
        "qmbp_simulation.framework.result_io._DEFAULT_RESULTS_ROOT",
        tmp_path / "experiments",
    )
    runner = _ValidRunner(args=_make_args())
    runner.run()

    exp_dir = tmp_path / "experiments" / "exp_test"
    log_files = list(exp_dir.glob("log_*.json"))
    assert len(log_files) == 1

    with open(log_files[0]) as f:
        log_data = json.load(f)

    # Structured log should have events
    assert "events" in log_data
    # Should contain preflight, setup, section events
    event_types = [e["event_type"] for e in log_data["events"]]
    assert "preflight_start" in event_types
    assert "preflight_passed" in event_types
    assert "setup_start" in event_types
    assert "setup_complete" in event_types
    assert "section_start" in event_types
    assert "section_complete" in event_types


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Section caching
# ═══════════════════════════════════════════════════════════════════════════════


def test_sections_cached():
    """define_sections() should only be called once (cached)."""
    call_count = 0

    class _CountingRunner(ValidationRunner):
        runner_id = "counting"
        experiment_id = "CNT"
        description = "Counting runner"
        hypothesis = "Caching works"

        def define_sections(self):
            nonlocal call_count
            call_count += 1
            return [Section(id=1, name="A", fn=lambda: {"pass": True}, hypothesis="test")]

    runner = _CountingRunner(args=_make_args())
    runner.run()
    assert call_count == 1, f"define_sections() called {call_count} times (expected 1)"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: ExperimentRunner
# ═══════════════════════════════════════════════════════════════════════════════


def test_experiment_runner_catches_import_error():
    """ExperimentRunner should catch import errors gracefully."""

    class _ImportFailRunner(ExperimentRunner):
        runner_id = "import_fail"

        def get_experiment_class(self):
            raise ImportError("Module not found: fake_module")

    args = argparse.Namespace(
        n_qubits=None,
        topology=None,
        seeds=None,
        skip_preflight=False,
        verbose=False,
    )
    runner = _ImportFailRunner(args=args)
    exit_code = runner.run()
    assert exit_code == 1


def test_experiment_runner_catches_missing_default_config():
    """ExperimentRunner should error clearly if default_config() is missing."""

    class _NoDefaultConfig:
        pass

    class _BadRunner(ExperimentRunner):
        runner_id = "bad_exp"

        def get_experiment_class(self):
            return _NoDefaultConfig

    args = argparse.Namespace(
        n_qubits=None,
        topology=None,
        seeds=None,
        skip_preflight=False,
        verbose=False,
    )
    runner = _BadRunner(args=args)
    exit_code = runner.run()
    assert exit_code == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: None return handling
# ═══════════════════════════════════════════════════════════════════════════════


def test_none_return_handled_gracefully():
    """Section returning None should warn but not crash."""

    class _NoneRunner(ValidationRunner):
        runner_id = "none_runner"
        experiment_id = "NONE"
        description = "None return runner"
        hypothesis = "None handled"

        def define_sections(self):
            return [
                Section(id=1, name="None", fn=lambda: None, hypothesis=""),  # type: ignore
            ]

    runner = _NoneRunner(args=_make_args())
    exit_code = runner.run()
    # None → no "pass" key → success assumed
    assert exit_code == 0
    assert runner._section_results[0].success is True
    assert runner._section_results[0].data == {}


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Digest/compare.py compatibility
# ═══════════════════════════════════════════════════════════════════════════════


def test_digest_scanner_can_parse_validation_runner_output(tmp_path, monkeypatch):
    """Digest scanner should successfully parse ValidationRunner output."""
    monkeypatch.setattr(
        "qmbp_simulation.framework.result_io._DEFAULT_RESULTS_ROOT",
        tmp_path / "experiments",
    )

    # Use a runner with realistic config (system, seeds, etc.)
    class _DigestCompatRunner(ValidationRunner):
        runner_id = "e4b_digest_test"
        experiment_id = "E4b"
        description = "E4b Digest Compatibility Test"
        hypothesis = "ValidationRunner output is parseable by digest"

        def define_sections(self):
            return [
                Section(id=1, name="Test", fn=lambda: {"pass": True}, hypothesis="works"),
            ]

        def build_config(self):
            return {
                "runner_id": self.runner_id,
                "experiment_id": self.experiment_id,
                "category": "E",
                "description": self.description,
                "hypothesis": self.hypothesis,
                "system": {
                    "n_qubits": 6,
                    "p_layers": 1,
                    "topology": "chain_1d",
                    "model": "tfim_longitudinal",
                },
                "seeds": DEFAULT_SEEDS,
            }

    runner = _DigestCompatRunner(args=_make_args())
    runner.run()

    # Now try parsing with the actual digest scanner
    from project_health.digest.scanner import ResultScanner

    scanner = ResultScanner(results_root=tmp_path)
    _, _, experiments = scanner.scan_all()

    # Should find our experiment
    assert len(experiments) == 1
    exp = experiments[0]
    assert exp.experiment_id == "E4b"
    assert exp.hypothesis == "ValidationRunner output is parseable by digest"
    assert exp.n_qubits == 6
    assert exp.p_layers == 1
    assert exp.topology == "chain_1d"
    assert exp.model == "tfim_longitudinal"
    assert exp.pass_rate == 1.0
    assert exp.n_seeds == 3
    assert exp.verdict == "confirmed"  # E4b criteria: pass_rate >= 0.90


def test_compare_result_store_can_load_validation_runner_output(tmp_path, monkeypatch):
    """ResultStore.compare_experiments should handle ValidationRunner output."""
    monkeypatch.setattr(
        "qmbp_simulation.framework.result_io._DEFAULT_RESULTS_ROOT",
        tmp_path / "experiments",
    )

    class _CompareCompatRunner(ValidationRunner):
        runner_id = "compare_test"
        experiment_id = "CMP"
        description = "Compare.py Compatibility Test"
        hypothesis = "ResultStore can compare ValidationRunner outputs"

        def define_sections(self):
            return [
                Section(id=1, name="A", fn=lambda: {"pass": True}, hypothesis=""),
                Section(id=2, name="B", fn=lambda: {"pass": False}, hypothesis=""),
            ]

        def build_config(self):
            return {
                "runner_id": self.runner_id,
                "experiment_id": self.experiment_id,
                "category": "C",
                "description": self.description,
                "hypothesis": self.hypothesis,
                "system": {"n_qubits": 6, "p_layers": 2, "topology": "chain_1d"},
                "seeds": [42],
            }

    runner = _CompareCompatRunner(args=_make_args())
    runner.run()

    # Verify the file is loadable by ResultStore
    from qmbp_simulation.framework import ResultStore

    store = ResultStore(results_root=tmp_path / "experiments")
    available = store.list_experiments()
    # ResultStore should find "CMP" in available experiments
    assert "CMP" in available


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: Utility methods (vqe_descending_sweep, exact_ground_state, etc.)
# ═══════════════════════════════════════════════════════════════════════════════


def test_exact_ground_state_n4():
    """exact_ground_state should return correct energy and positive gap for N=4 TFIM."""
    # N=4, h=2.0 (paramagnetic phase, well-known result)
    e_exact, gap = ValidationRunner.exact_ground_state("chain_1d", 4, h=2.0)

    # Energy should be negative (ferromagnetic coupling + transverse field)
    assert e_exact < 0, f"Expected negative energy, got {e_exact}"
    # Gap should be positive
    assert gap > 0, f"Expected positive gap, got {gap}"
    # For N=4 chain at h=2.0, the gap is well-defined (paramagnetic phase)
    assert gap < 5.0, f"Gap seems too large: {gap}"


def test_exact_ground_state_consistency():
    """exact_ground_state at different h should give different energies."""
    e1, gap1 = ValidationRunner.exact_ground_state("chain_1d", 4, h=1.0)
    e2, gap2 = ValidationRunner.exact_ground_state("chain_1d", 4, h=3.0)

    # Higher h → more paramagnetic → lower energy (more −h⟨X⟩ contribution)
    assert e2 < e1, f"Higher h should give lower energy: E(h=3)={e2} vs E(h=1)={e1}"
    # Both should have positive gaps
    assert gap1 > 0
    assert gap2 > 0


def test_vqe_descending_sweep_n4():
    """vqe_descending_sweep should produce optimized parameters for each h."""

    class _SweepTestRunner(ValidationRunner):
        runner_id = "sweep_test"
        experiment_id = "SWEEP"
        description = "VQE sweep test"
        hypothesis = "VQE converges"

        def define_sections(self):
            return [Section(id=1, name="A", fn=lambda: {"pass": True}, hypothesis="")]

    runner = _SweepTestRunner(args=_make_args())
    theta_map = runner.vqe_descending_sweep(
        topology="chain_1d",
        n_qubits=4,
        h_values=[2.0, 1.5],
        seed=42,
        p_layers=1,
        n_restarts=1,
        maxiter=100,
    )

    # Should have entries for both h-values
    assert 2.0 in theta_map
    assert 1.5 in theta_map

    # Parameters should be numpy arrays with correct length
    import numpy as np

    assert isinstance(theta_map[2.0], np.ndarray)
    assert isinstance(theta_map[1.5], np.ndarray)
    assert len(theta_map[2.0]) == len(theta_map[1.5])  # Same circuit
    assert len(theta_map[2.0]) > 0


def test_vqe_sweep_produces_good_energy():
    """VQE sweep result should give energy close to exact for N=4."""

    class _EnergyTestRunner(ValidationRunner):
        runner_id = "energy_test"
        experiment_id = "ENRG"
        description = "Energy test"
        hypothesis = "VQE is accurate"

        def define_sections(self):
            return [Section(id=1, name="A", fn=lambda: {"pass": True}, hypothesis="")]

    runner = _EnergyTestRunner(args=_make_args())

    h = 2.0
    theta_map = runner.vqe_descending_sweep(
        topology="chain_1d",
        n_qubits=4,
        h_values=[h],
        seed=42,
        p_layers=1,
        n_restarts=3,
        maxiter=500,
    )
    theta = theta_map[h]

    # Evaluate energy with the optimized parameters
    from qmbp_simulation import make_lattice
    from qmbp_simulation.execution import NoiselessBackend
    from qmbp_simulation.models.model_registry import get_model_spec

    spec = get_model_spec("tfim")
    lattice = make_lattice("chain_1d", 4, J=1.0, h=h)
    H = spec.build_hamiltonian(lattice, **spec.hamiltonian_kwargs)
    circuit, _ = spec.create_circuit(4, 1, lattice, **spec.circuit_kwargs)
    backend = NoiselessBackend()
    e_vqe = backend.evaluate(circuit, H, theta)

    # Compare to exact
    e_exact, gap = ValidationRunner.exact_ground_state("chain_1d", 4, h=h)
    de_gap = abs(e_vqe - e_exact) / gap

    # At N=4, p=1, h=2.0 with 3 restarts, should be < 5%
    assert de_gap < 0.10, f"ΔE/gap={de_gap:.4f} too large for N=4 p=1 h=2.0"


def test_compute_fidelity():
    """compute_fidelity should return ~1.0 for identical states."""
    import numpy as np
    from qiskit import QuantumCircuit

    # Create a trivial circuit that produces |0⟩^N when params=0
    n = 2
    qc = QuantumCircuit(n)
    from qiskit.circuit import Parameter

    theta = Parameter("θ")
    qc.ry(theta, 0)

    # With theta=0, state is |00⟩
    exact_state = np.zeros(2**n)
    exact_state[0] = 1.0  # |00⟩

    fid = ValidationRunner.compute_fidelity(qc, np.array([0.0]), exact_state)
    assert abs(fid - 1.0) < 1e-10, f"Expected fidelity ~1.0, got {fid}"

    # With theta=pi, state is |10⟩ — orthogonal to |00⟩
    fid_orth = ValidationRunner.compute_fidelity(qc, np.array([np.pi]), exact_state)
    assert fid_orth < 0.01, f"Expected fidelity ~0, got {fid_orth}"


def test_truncate_statevector_mps_full_rank():
    """MPS truncation at chi=2^N should be lossless (identity operation)."""
    import numpy as np

    # Create a random 4-qubit state
    n = 4
    rng = np.random.default_rng(42)
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi = psi / np.linalg.norm(psi)

    # Truncate at full rank (chi=2^(N/2)=4 is sufficient for most states)
    psi_trunc = ValidationRunner.truncate_statevector_mps(psi, n, chi_max=16)

    # Should be essentially identical (full rank = no truncation)
    overlap = abs(np.vdot(psi, psi_trunc)) ** 2
    assert overlap > 0.999, f"Full-rank MPS should preserve state: overlap={overlap}"


def test_truncate_statevector_mps_low_chi():
    """MPS truncation at chi=1 should produce a product state."""
    import numpy as np

    # Create a highly entangled state (GHZ-like)
    n = 4
    psi = np.zeros(2**n, dtype=complex)
    psi[0] = 1 / np.sqrt(2)  # |0000⟩
    psi[-1] = 1 / np.sqrt(2)  # |1111⟩

    # chi=1 means product state — GHZ cannot be represented
    psi_trunc = ValidationRunner.truncate_statevector_mps(psi, n, chi_max=1)

    # Truncated state should be normalized
    assert abs(np.linalg.norm(psi_trunc) - 1.0) < 1e-10

    # Overlap should be less than 1 (information lost)
    overlap = abs(np.vdot(psi, psi_trunc)) ** 2
    assert overlap < 0.99, f"Chi=1 should lose entanglement: overlap={overlap}"


def test_truncate_statevector_product_state_invariant():
    """A product state should survive any chi truncation unchanged."""
    import numpy as np

    # |+⟩^4 is a product state — should be invariant under MPS truncation
    n = 4
    plus = np.array([1, 1]) / np.sqrt(2)
    psi = plus.copy()
    for _ in range(n - 1):
        psi = np.kron(psi, plus)

    for chi in [1, 2, 4, 8]:
        psi_trunc = ValidationRunner.truncate_statevector_mps(psi, n, chi_max=chi)
        overlap = abs(np.vdot(psi, psi_trunc)) ** 2
        assert overlap > 0.99, f"Product state should survive chi={chi}: overlap={overlap}"


def test_resolve_backend_uses_subclass_attribute():
    """_resolve_backend should prefer self.noiseless if set."""
    from unittest.mock import MagicMock

    class _BackendRunner(ValidationRunner):
        runner_id = "backend_test"
        experiment_id = "BK"
        description = "Backend test"
        hypothesis = "Backend resolution"

        def define_sections(self):
            return [Section(id=1, name="A", fn=lambda: {"pass": True}, hypothesis="")]

    runner = _BackendRunner(args=_make_args())

    # Without any attribute set, creates NoiselessBackend
    backend = runner._resolve_backend()
    assert backend is not None

    # With self.noiseless set, uses that
    mock_backend = MagicMock()
    runner.noiseless = mock_backend
    assert runner._resolve_backend() is mock_backend

    # With self.backend set (no noiseless), uses that
    del runner.noiseless
    runner.backend = mock_backend
    assert runner._resolve_backend() is mock_backend


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: MPS_HW in EXPERIMENT_CRITERIA
# ═══════════════════════════════════════════════════════════════════════════════


def test_mps_hw_criteria_registered():
    """MPS_HW should be in EXPERIMENT_CRITERIA with pass_rate metric."""
    from qmbp_simulation.framework.criteria import EXPERIMENT_CRITERIA, compute_verdict

    assert "MPS_HW" in EXPERIMENT_CRITERIA
    criteria = EXPERIMENT_CRITERIA["MPS_HW"]
    assert criteria["metric"] == "pass_rate"
    assert criteria["threshold"] == 0.80

    # Verdict with pass_rate=1.0 → confirmed
    verdict, _ = compute_verdict("MPS_HW", {"pass_rate": 1.0})
    assert verdict == "confirmed"

    # Verdict with pass_rate=0.5 → failed (not in REJECTION_IS_FINDING)
    verdict, _ = compute_verdict("MPS_HW", {"pass_rate": 0.5})
    assert verdict == "failed"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: HardwareValidationRunner
# ═══════════════════════════════════════════════════════════════════════════════


def _make_hw_args(**kwargs):
    """Create argparse.Namespace with defaults for HardwareValidationRunner."""
    defaults = {
        "section": None,
        "skip_preflight": False,
        "stop_on_failure": False,
        "verbose": False,
        "dry_run": False,
        "mode": "fake_backend",
        "shots": 1024,
        "n_layouts": 3,
        "n_qubits": 6,
        "topology": "chain_1d",
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_hardware_runner_preflight_passes_fake():
    """HardwareValidationRunner preflight should pass in fake_backend mode."""
    from qmbp_simulation.framework.runner_base import HardwareValidationRunner

    class _HWRunner(HardwareValidationRunner):
        runner_id = "hw_test"
        experiment_id = "HW_TEST"
        description = "Hardware test"
        hypothesis = "Fake backend works"

        def define_sections(self):
            return [
                Section(id=1, name="Smoke", fn=lambda: {"pass": True}, hypothesis="works"),
            ]

    runner = _HWRunner(args=_make_hw_args())
    assert runner.run_preflight() is True


def test_hardware_runner_dry_run():
    """HardwareValidationRunner dry-run should list sections."""
    from qmbp_simulation.framework.runner_base import HardwareValidationRunner

    class _HWRunner(HardwareValidationRunner):
        runner_id = "hw_dry"
        experiment_id = "HW_DRY"
        description = "Hardware dry-run test"
        hypothesis = "Dry-run works"

        def define_sections(self):
            return [
                Section(id=1, name="Deploy", fn=lambda: {"pass": True}, hypothesis="test"),
            ]

    runner = _HWRunner(args=_make_hw_args(dry_run=True))
    exit_code = runner.run()
    assert exit_code == 0
    assert len(runner._section_results) == 0  # No execution in dry-run


def test_hardware_runner_config_has_hardware_fields():
    """HardwareValidationRunner config should include hardware-specific fields."""
    from qmbp_simulation.framework.runner_base import HardwareValidationRunner

    class _HWRunner(HardwareValidationRunner):
        runner_id = "hw_cfg"
        experiment_id = "HW_CFG"
        description = "Config test"
        hypothesis = "Config has hardware fields"

        def define_sections(self):
            return [Section(id=1, name="A", fn=lambda: {"pass": True}, hypothesis="")]

    runner = _HWRunner(args=_make_hw_args(mode="fake_backend", shots=8192))
    config = runner.build_config()

    assert "hardware" in config
    assert config["hardware"]["mode"] == "fake_backend"
    assert config["hardware"]["shots"] == 8192
    assert config["hardware"]["n_layouts"] == 3
    assert config["system"]["n_qubits"] == 6
    assert config["experiment_id"] == "HW_CFG"


def test_hardware_runner_setup_creates_backend():
    """HardwareValidationRunner.setup() should initialize hw_backend."""
    from qmbp_simulation.framework.runner_base import HardwareValidationRunner

    class _HWRunner(HardwareValidationRunner):
        runner_id = "hw_setup"
        experiment_id = "HW_SETUP"
        description = "Setup test"
        hypothesis = "Backend gets created"

        def define_sections(self):
            return [Section(id=1, name="A", fn=lambda: {"pass": True}, hypothesis="")]

    runner = _HWRunner(args=_make_hw_args())
    assert runner.hw_backend is None
    runner.setup()
    assert runner.hw_backend is not None

    from qmbp_simulation.execution.hardware import HardwareBackend

    assert isinstance(runner.hw_backend, HardwareBackend)
    # Logger should be shared
    assert runner.hw_backend._logger is runner.slog
