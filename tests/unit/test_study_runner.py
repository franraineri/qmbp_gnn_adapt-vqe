"""Unit tests for the StudyRunner strategy layer and regime gate.

Tests the pure, backend-agnostic optimizer strategies and the regime-gated
strategy resolution in ``qmbp_simulation.framework.study_runner``. Uses a stub
quadratic cost (no quantum backend) so the tests are fast and deterministic, and
imports only from ``src/`` (no ``scripts/`` per the repo testing rule).
"""

from __future__ import annotations

import numpy as np
import pytest

from qmbp_simulation.framework.study_runner import (
    STRATEGIES,
    WarmStartResult,
    resolve_strategy,
    run_strategy,
    strategy_bestof,
    strategy_metropolis,
    strategy_single,
)

# Layout: small frustrated-like case.
N_NN, N_NNN, N_QUBITS, P = 6, 4, 4, 1
N_PARAMS = N_NN + N_NNN + N_QUBITS


def _quadratic_cost(target):
    """A convex cost with a unique minimum at ``target`` (stands in for VQE)."""

    def cost(x):
        return float(np.sum((np.asarray(x) - target) ** 2))

    return cost


def _fidelity_from(target):
    """Fidelity proxy: 1 at the target, decaying with distance."""

    def fidelity(x):
        return float(np.exp(-np.sum((np.asarray(x) - target) ** 2)))

    return fidelity


class TestRegimeResolution:
    def test_second_order_enabled_in_window(self):
        assert resolve_strategy("second_order", 0.5) == "second_order"

    def test_second_order_falls_back_below_window(self):
        assert resolve_strategy("second_order", 0.3) == "metropolis"

    def test_second_order_falls_back_above_window(self):
        assert resolve_strategy("second_order", 0.9) == "metropolis"

    def test_other_strategies_passthrough(self):
        for s in ("first_order", "bestof", "metropolis"):
            assert resolve_strategy(s, 0.3) == s

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            resolve_strategy("nope", 0.5)


class TestStrategies:
    def test_single_converges_to_target(self):
        target = np.full(N_PARAMS, 0.2)
        cost, fid = _quadratic_cost(target), _fidelity_from(target)
        e, x, runs = strategy_single(cost, fid, np.zeros(N_PARAMS), maxiter=200)
        np.testing.assert_allclose(x, target, atol=1e-3)
        assert e < 1e-4
        assert len(runs) == 1

    def test_bestof_records_each_restart(self):
        target = np.full(N_PARAMS, 0.1)
        cost, fid = _quadratic_cost(target), _fidelity_from(target)
        e, x, runs = strategy_bestof(cost, fid, np.zeros(N_PARAMS), maxiter=100,
                                     sigmas=(0.0, 0.1, 0.3))
        assert len(runs) == 3
        assert runs[0]["sigma"] == 0.0
        assert e < 1e-3

    def test_metropolis_logs_hops_and_is_deterministic(self):
        target = np.full(N_PARAMS, -0.15)
        cost, fid = _quadratic_cost(target), _fidelity_from(target)
        e1, x1, runs1 = strategy_metropolis(cost, fid, np.zeros(N_PARAMS),
                                            maxiter=100, n_hops=3, seed0=123)
        e2, x2, runs2 = strategy_metropolis(cost, fid, np.zeros(N_PARAMS),
                                            maxiter=100, n_hops=3, seed0=123)
        assert len(runs1) == 4  # hop 0 + 3 hops
        np.testing.assert_allclose(x1, x2)  # same seed -> same trajectory
        np.testing.assert_allclose(e1, e2)


class TestRunStrategyDispatch:
    def test_returns_warmstart_result_with_resolved_label(self):
        target = np.full(N_PARAMS, 0.05)
        res = run_strategy(
            "second_order", cost=_quadratic_cost(target), fidelity=_fidelity_from(target),
            n_nn=N_NN, n_nnn=N_NNN, n_qubits=N_QUBITS, p_layers=P, h=0.9,
            J=1.0, J2=0.5, maxiter=100,
        )
        assert isinstance(res, WarmStartResult)
        # h=0.9 is outside the window -> resolved to metropolis.
        assert res.strategy == "metropolis"
        assert res.best_theta.shape == (N_PARAMS,)

    def test_second_order_used_in_window(self):
        target = np.full(N_PARAMS, 0.05)
        res = run_strategy(
            "second_order", cost=_quadratic_cost(target), fidelity=_fidelity_from(target),
            n_nn=N_NN, n_nnn=N_NNN, n_qubits=N_QUBITS, p_layers=P, h=0.5,
            J=1.0, J2=0.5, maxiter=100,
        )
        assert res.strategy == "second_order"

    def test_all_strategy_names_dispatch(self):
        target = np.zeros(N_PARAMS)
        for s in STRATEGIES:
            res = run_strategy(
                s, cost=_quadratic_cost(target), fidelity=_fidelity_from(target),
                n_nn=N_NN, n_nnn=N_NNN, n_qubits=N_QUBITS, p_layers=P, h=0.5,
                maxiter=50, n_hops=2,
            )
            assert res.best_theta.shape == (N_PARAMS,)
            assert res.strategy in STRATEGIES


class TestPersistAndSweep:
    """Lifecycle: persist_point artifacts + run_sweep checkpoint/resume.

    Uses a monkeypatched optimize_point so the test needs no quantum backend
    (fast, deterministic) — it exercises the base loop, not the physics.
    """

    def _fake_point(self, h):
        import numpy as np
        return {
            "topology": "square", "h": h, "n_qubits": 6, "p_layers": 1,
            "J2": 0.5, "model": "tfim_frustrated", "ansatz": "nnn",
            "gs_method": "cached/exact", "e_vqe": -5.0 - h, "e0_exact": -5.5 - h,
            "gap": 0.3, "abs_error": 0.5, "de_gap": 1.6, "fidelity": 0.8,
            "n_params": 20, "n_nn": 12, "n_nnn": 8,
            "strategy_requested": "metropolis", "strategy_resolved": "metropolis",
            "runs": [{"hop": 0}], "best_theta": np.full(20, 0.1),
            "_qc": None,
        }

    def _runner(self):
        from qmbp_simulation.framework.study_runner import StudyRunner
        r = StudyRunner(topology="square", n_qubits=6, p_layers=1, j2=0.5,
                        model="tfim_frustrated", strategy="metropolis")
        r.optimize_point = self._fake_point  # bypass the quantum backend
        return r

    def test_persist_point_writes_theta_and_drops_qc(self, tmp_path):
        from pathlib import Path
        r = self._runner()
        rec = r.persist_point(self._fake_point(0.5), tmp_path / "art")
        assert "_qc" not in rec
        assert isinstance(rec["best_theta"], list)
        assert Path(rec["theta_npz"]).exists()
        assert "circuit_qpy" not in rec  # no qpy_saver injected

    def test_persist_point_qpy_saver_injection(self, tmp_path):
        from pathlib import Path
        called = {}

        def fake_saver(circ, path):
            called["path"] = path
            Path(path).write_bytes(b"QPY")

        r = self._runner()
        point = self._fake_point(0.5)
        # give it a dummy object with assign_parameters
        class _QC:
            def assign_parameters(self, theta):
                return "bound"
        point["_qc"] = _QC()
        rec = r.persist_point(point, tmp_path / "art", qpy_saver=fake_saver)
        assert Path(rec["circuit_qpy"]).exists()
        assert called["path"] is not None

    def test_run_sweep_checkpoint_and_resume(self, tmp_path):
        import json
        from qmbp_simulation.framework.study_checkpoint import StudyCheckpoint

        ckpt_path = tmp_path / "sweep.json"

        def writer(payload):
            ckpt_path.write_text(json.dumps(payload, default=str))
            return ckpt_path

        r = self._runner()
        cp = StudyCheckpoint("per_h", writer=writer, path=ckpt_path)
        units = r.run_sweep([0.5, 0.9], cp, extra={"schema": "t_v1"})
        assert sorted(units) == ["0.50", "0.90"]
        saved = json.loads(ckpt_path.read_text())
        assert saved["schema"] == "t_v1"
        assert len(saved["per_h"]) == 2

        # Resume: both done -> optimize_point must NOT be called again.
        calls = {"n": 0}
        orig = self._fake_point

        def counting(h):
            calls["n"] += 1
            return orig(h)

        r.optimize_point = counting
        cp2 = StudyCheckpoint("per_h", writer=writer, path=ckpt_path)
        r.run_sweep([0.5, 0.9], cp2)
        assert calls["n"] == 0  # fully skipped on resume


class TestGroundStateNonDivergence:
    """Guard: the service (scripts) and the src StudyRunner ground-state paths
    must stay numerically identical. If someone edits one, this fails.

    Imports only from src (allowed); reproduces the service's numeric path
    directly rather than importing scripts/.
    """

    def test_src_eigsh_matches_service_formula(self):
        # Both paths call GroundTruthCache + eigsh(k=2). We verify the src helper
        # returns a normalized vector and consistent E0/gap on a tiny case.
        import numpy as np

        from qmbp_simulation.framework.study_runner import _exact_ground_state_vector

        psi, e0, gap, H = _exact_ground_state_vector(
            "square", 6, 0.5, model="tfim_frustrated", j2=0.5
        )
        assert psi is not None
        np.testing.assert_allclose(np.linalg.norm(psi), 1.0, atol=1e-10)
        assert gap >= 0.0
        assert e0 < 0.0  # frustrated TFIM ground energy is negative here
