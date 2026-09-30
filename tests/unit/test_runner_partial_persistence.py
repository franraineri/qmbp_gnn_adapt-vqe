"""Tests for the resumable-partial persistence contract on ValidationRunner.

Covers the base-class contract every runner inherits: ``save_partial`` /
``load_partial`` / ``on_restart_persister``. The contract must persist angles
(θ) crash-safely and let a later run continue from them.

Core behaviors:
- θ round-trips through save/load as ``best_theta``.
- A config-mismatched partial is discarded on load (not resumed).
- The ``resumable`` flag and ``completed`` count are recorded.
- ``on_restart_persister`` returns a callback matching the study_core hook
  shape ``(runs, best)`` and persists the best restart's angles.

Pure ``src/`` module, tested with ``tmp_path`` + ``monkeypatch.chdir`` so no
real ``results/`` entries are created.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from qmbp_simulation.framework.runner_base import Section, ValidationRunner


class _MinimalRunner(ValidationRunner):
    runner_id = "test_partial_runner"
    experiment_id = "TEST_PARTIAL"
    description = "Test runner for partial persistence"
    hypothesis = "Partials round-trip with angles"

    def define_sections(self):
        return [Section(id=1, name="noop", fn=lambda: {}, hypothesis="noop")]


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        n_qubits=6,
        p_layers=2,
        topology="square",
        model="tfim_frustrated",
        h=0.5,
        verbose=False,
        section=None,
        dry_run=False,
        skip_preflight=False,
        stop_on_failure=False,
        validate_vqe=False,
        validate_theta=False,
        theta_validation_level=4,
        strict_validation=False,
        resume=None,
        save_artifacts="never",
        no_bidirectional=False,
        force_bidirectional=False,
        preset=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """Runner whose checkpoint dir resolves under tmp_path (chdir isolation)."""
    monkeypatch.chdir(tmp_path)
    return _MinimalRunner(_args())


class TestSavePartialRoundTrip:
    """save_partial → load_partial preserves rows, angles, and metadata."""

    def test_theta_roundtrips_as_best_theta(self, runner):
        theta = np.array([0.1, -0.2, 0.3, 0.4])
        runner.save_partial("v1", rows=[{"restart": 0}], theta=theta)
        loaded = runner.load_partial("v1")
        assert loaded is not None
        np.testing.assert_allclose(loaded["best_theta"], theta, atol=1e-12)

    def test_resumable_flag_set(self, runner):
        runner.save_partial("v1", rows=[{"restart": 0}], theta=[0.1, 0.2])
        loaded = runner.load_partial("v1")
        assert loaded["resumable"] is True

    def test_completed_defaults_to_row_count(self, runner):
        runner.save_partial("v1", rows=[{"restart": 0}, {"restart": 1}], theta=None)
        loaded = runner.load_partial("v1")
        assert loaded["completed"] == 2

    def test_completed_explicit_override(self, runner):
        runner.save_partial("v1", rows=[{"restart": 0}], theta=None, completed=5)
        assert runner.load_partial("v1")["completed"] == 5

    def test_rows_preserved(self, runner):
        rows = [{"restart": 0, "energy": -1.0}, {"restart": 1, "energy": -2.0}]
        runner.save_partial("v1", rows=rows, theta=None)
        loaded = runner.load_partial("v1")
        assert loaded["rows"][1]["energy"] == -2.0

    def test_meta_merged(self, runner):
        runner.save_partial("v1", rows=[], theta=None, meta={"best_fidelity": 0.92})
        loaded = runner.load_partial("v1")
        assert loaded["meta"]["best_fidelity"] == 0.92

    def test_returns_written_path(self, runner):
        path = runner.save_partial("v1", rows=[], theta=None)
        assert path is not None
        assert path.name == ".partial_v1.json"
        assert path.exists()


class TestLoadPartialConfigGuard:
    """A partial from a different config is discarded, not resumed."""

    def test_missing_returns_none(self, runner):
        assert runner.load_partial("does_not_exist") is None

    def test_matching_config_loads(self, runner):
        runner.save_partial("v1", rows=[{"restart": 0}], theta=[0.1, 0.2])
        assert runner.load_partial("v1") is not None

    def test_n_qubits_mismatch_discarded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r_save = _MinimalRunner(_args(n_qubits=6))
        r_save.save_partial("v1", rows=[{"restart": 0}], theta=[0.1, 0.2])
        # A new runner at a different N must not resume the N=6 partial.
        r_load = _MinimalRunner(_args(n_qubits=18))
        assert r_load.load_partial("v1") is None

    def test_p_layers_mismatch_discarded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _MinimalRunner(_args(p_layers=2)).save_partial("v1", rows=[], theta=[0.1])
        assert _MinimalRunner(_args(p_layers=3)).load_partial("v1") is None

    def test_model_mismatch_discarded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _MinimalRunner(_args(model="tfim")).save_partial("v1", rows=[], theta=[0.1])
        assert _MinimalRunner(_args(model="tfim_frustrated")).load_partial("v1") is None

    def test_stale_partial_removed_on_mismatch(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r_save = _MinimalRunner(_args(n_qubits=6))
        path = r_save.save_partial("v1", rows=[], theta=[0.1])
        _MinimalRunner(_args(n_qubits=18)).load_partial("v1")
        assert not path.exists()


class TestOnRestartPersister:
    """on_restart_persister callback matches study_core.optimize_bestof hook."""

    def test_callback_persists_best_restart_theta(self, runner):
        cb = runner.on_restart_persister("v1")
        runs = [
            {"restart": 0, "energy": -1.0, "theta_final": [0.1, 0.2]},
            {"restart": 1, "energy": -3.0, "theta_final": [0.9, 0.8]},
        ]
        cb(runs, {"fidelity": 0.9, "energy": -3.0, "restart": 1})
        loaded = runner.load_partial("v1")
        np.testing.assert_allclose(loaded["best_theta"], [0.9, 0.8], atol=1e-12)

    def test_callback_records_all_runs(self, runner):
        cb = runner.on_restart_persister("v1")
        runs = [{"restart": 0, "energy": -1.0, "theta_final": [0.1]}]
        cb(runs, {"fidelity": 0.5, "energy": -1.0, "restart": 0})
        assert runner.load_partial("v1")["completed"] == 1

    def test_callback_falls_back_to_min_energy_when_no_index(self, runner):
        cb = runner.on_restart_persister("v1")
        runs = [
            {"restart": 0, "energy": -5.0, "theta_final": [1.0, 1.0]},
            {"restart": 1, "energy": -2.0, "theta_final": [2.0, 2.0]},
        ]
        cb(runs, {"fidelity": 0.9, "energy": -5.0})  # no "restart" key
        loaded = runner.load_partial("v1")
        np.testing.assert_allclose(loaded["best_theta"], [1.0, 1.0], atol=1e-12)

    def test_callback_surfaces_best_fidelity_in_meta(self, runner):
        cb = runner.on_restart_persister("v1")
        cb([{"restart": 0, "energy": -1.0, "theta_final": [0.1]}], {"fidelity": 0.88, "energy": -1.0, "restart": 0})
        assert runner.load_partial("v1")["meta"]["best_fidelity"] == 0.88


class TestPartialPersistenceEdgeScenarios:
    """Six edge scenarios that stress the contract's robustness boundaries.

    Each targets a distinct failure mode that a resumable θ-persistence layer
    must survive: corrupt angles, corrupt files on disk, absent config, a
    resume that overwrites an earlier partial, angle-type polymorphism, and a
    save that never breaks the run even when persistence itself fails.
    """

    def test_edge_1_non_finite_theta_dropped_not_persisted(self, runner):
        """θ with NaN/Inf is never persisted (best_theta stays None), but the
        partial still saves so completed units are not lost."""
        theta = np.array([0.1, np.nan, np.inf, 0.4])
        path = runner.save_partial("v1", rows=[{"restart": 0}], theta=theta)
        assert path is not None and path.exists()
        loaded = runner.load_partial("v1")
        assert loaded["best_theta"] is None
        assert loaded["completed"] == 1

    def test_edge_2_corrupt_partial_file_returns_none(self, runner):
        """A truncated/corrupt partial on disk loads as None (never raises)."""
        path = runner._checkpoint_dir() / ".partial_v1.json"
        path.write_text("{not valid json")
        assert runner.load_partial("v1") is None

    def test_edge_3_no_args_empty_fingerprint_still_roundtrips(self, tmp_path, monkeypatch):
        """A runner without _args (empty fingerprint) can still save/load a
        partial — the config guard simply has nothing to reject on."""
        monkeypatch.chdir(tmp_path)
        r = _MinimalRunner(_args())
        r._args = None  # simulate a runner constructed without parsed args
        r.save_partial("v1", rows=[{"restart": 0}], theta=[0.5, 0.6])
        loaded = r.load_partial("v1")
        assert loaded is not None
        np.testing.assert_allclose(loaded["best_theta"], [0.5, 0.6], atol=1e-12)

    def test_edge_4_resume_overwrites_earlier_partial_with_more_progress(self, runner):
        """A second save at the same label supersedes the first — a resume that
        made more progress must replace the stale floor, keeping the latest θ."""
        runner.save_partial("v1", rows=[{"restart": 0}], theta=[0.1, 0.2], meta={"best_fidelity": 0.80})
        runner.save_partial("v1", rows=[{"restart": 0}, {"restart": 1}], theta=[0.9, 0.8], meta={"best_fidelity": 0.92})
        loaded = runner.load_partial("v1")
        assert loaded["completed"] == 2
        assert loaded["meta"]["best_fidelity"] == 0.92
        np.testing.assert_allclose(loaded["best_theta"], [0.9, 0.8], atol=1e-12)

    def test_edge_5_theta_list_and_ndarray_are_equivalent(self, tmp_path, monkeypatch):
        """Angles passed as a Python list or a numpy array persist identically —
        callers must not have to pre-convert."""
        monkeypatch.chdir(tmp_path)
        r_list = _MinimalRunner(_args())
        r_list.save_partial("as_list", rows=[], theta=[0.3, -0.4, 0.5])
        r_arr = _MinimalRunner(_args())
        r_arr.save_partial("as_arr", rows=[], theta=np.array([0.3, -0.4, 0.5]))
        a = r_list.load_partial("as_list")["best_theta"]
        b = r_arr.load_partial("as_arr")["best_theta"]
        np.testing.assert_allclose(a, b, atol=1e-12)

    @pytest.mark.parametrize("exc", [OSError("disk full"), TypeError("not serializable"), ValueError("bad value")])
    def test_edge_6_save_failure_never_raises(self, runner, monkeypatch, exc):
        """If the atomic write fails for ANY of the expected reasons, save_partial
        returns None and never raises — persistence must never break the
        computation it protects (OSError, TypeError, ValueError all swallowed)."""
        import qmbp_simulation.framework.runner_base as rb

        def _boom(_path, _payload):
            raise exc

        monkeypatch.setattr(rb, "atomic_write_json", _boom, raising=False)
        # atomic_write_json is imported inside save_partial from study_checkpoint;
        # patch there so the import inside the method picks up the boom.
        import qmbp_simulation.framework.study_checkpoint as sc

        monkeypatch.setattr(sc, "atomic_write_json", _boom)
        result = runner.save_partial("v1", rows=[{"restart": 0}], theta=[0.1])
        assert result is None


class TestIncrementalRowAccumulation:
    """append_row accumulates without the caller rebuilding an O(n) list."""

    def test_append_row_accumulates(self, runner):
        runner.save_partial("v1", append_row={"restart": 0, "energy": -1.0})
        runner.save_partial("v1", append_row={"restart": 1, "energy": -2.0})
        runner.save_partial("v1", append_row={"restart": 2, "energy": -3.0})
        loaded = runner.load_partial("v1")
        assert loaded["completed"] == 3
        assert [r["restart"] for r in loaded["rows"]] == [0, 1, 2]

    def test_rows_snapshot_replaces_accumulator(self, runner):
        runner.save_partial("v1", append_row={"restart": 0})
        runner.save_partial("v1", rows=[{"restart": 9}])  # snapshot replaces
        loaded = runner.load_partial("v1")
        assert loaded["completed"] == 1
        assert loaded["rows"][0]["restart"] == 9

    def test_append_after_snapshot_extends(self, runner):
        runner.save_partial("v1", rows=[{"restart": 0}], append_row={"restart": 1})
        loaded = runner.load_partial("v1")
        assert [r["restart"] for r in loaded["rows"]] == [0, 1]

    def test_theta_tracks_latest_append(self, runner):
        runner.save_partial("v1", append_row={"restart": 0}, theta=[0.1, 0.2])
        runner.save_partial("v1", append_row={"restart": 1}, theta=[0.9, 0.8])
        loaded = runner.load_partial("v1")
        np.testing.assert_allclose(loaded["best_theta"], [0.9, 0.8], atol=1e-12)


class TestSharedResumablePayloadHelper:
    """build_resumable_payload / sanitize_theta are the shared base contract
    used by both ValidationRunner and the standalone study runners."""

    def test_sanitize_drops_non_finite(self):
        from qmbp_simulation.framework.study_checkpoint import sanitize_theta

        assert sanitize_theta([1.0, np.nan]) is None
        assert sanitize_theta([1.0, np.inf]) is None
        assert sanitize_theta(None) is None

    def test_sanitize_preserves_finite(self):
        from qmbp_simulation.framework.study_checkpoint import sanitize_theta

        np.testing.assert_allclose(sanitize_theta([0.1, -0.2]), [0.1, -0.2], atol=1e-12)

    def test_payload_has_resumable_core(self):
        from qmbp_simulation.framework.study_checkpoint import build_resumable_payload

        p = build_resumable_payload(rows=[{"a": 1}], theta=[0.3], fingerprint={"n_qubits": 6})
        assert p["resumable"] is True
        assert p["completed"] == 1
        assert p["config_fingerprint"] == {"n_qubits": 6}
        np.testing.assert_allclose(p["best_theta"], [0.3], atol=1e-12)

    def test_payload_extra_merges_study_fields(self):
        from qmbp_simulation.framework.study_checkpoint import build_resumable_payload

        p = build_resumable_payload(rows=[], theta=None, extra={"topology": "square", "schema": "v1"})
        assert p["topology"] == "square"
        assert p["schema"] == "v1"
        assert p["best_theta"] is None
