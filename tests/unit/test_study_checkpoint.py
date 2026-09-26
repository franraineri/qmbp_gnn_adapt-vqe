"""Unit tests for the crash-safe study checkpoint helper.

Covers the round-trip + resume + skip-completed behavior that the ``vl_vs_hva``
study runners rely on (per-hop / per-seed checkpointing). Pure ``src/`` module,
tested with ``tmp_path`` and no ``scripts/`` import.
"""

from __future__ import annotations

import json

import pytest

from qmbp_simulation.framework.study_checkpoint import (
    StudyCheckpoint,
    atomic_write_json,
    read_json,
    resume_ordered_list,
)


class TestAtomicJson:
    def test_write_and_read_roundtrip(self, tmp_path):
        path = tmp_path / "sub" / "ckpt.json"
        payload = {"a": 1, "nested": {"x": [1, 2, 3]}}
        written = atomic_write_json(path, payload)
        assert written == path
        assert read_json(path) == payload

    def test_read_missing_returns_none(self, tmp_path):
        assert read_json(tmp_path / "nope.json") is None

    def test_read_corrupt_returns_none(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json")
        assert read_json(p) is None

    def test_no_tmp_files_left_behind(self, tmp_path):
        path = tmp_path / "ckpt.json"
        atomic_write_json(path, {"k": "v"})
        leftovers = [p for p in tmp_path.iterdir() if ".tmp." in p.name]
        assert leftovers == []


class TestStudyCheckpointKeyedUnits:
    def test_record_and_persist(self, tmp_path):
        path = tmp_path / "seeds.json"
        cp = StudyCheckpoint("per_seed", path=path)
        assert cp.resume() == 0
        cp.record("0.50|0", {"fid1": 0.8, "fid2": 0.9})
        cp.record("0.50|1", {"fid1": 0.81, "fid2": 0.91})
        cp.persist(extra={"schema": "test_v1"})

        raw = json.loads(path.read_text())
        assert set(raw["per_seed"].keys()) == {"0.50|0", "0.50|1"}
        assert raw["schema"] == "test_v1"

    def test_resume_restores_completed_units(self, tmp_path):
        path = tmp_path / "seeds.json"
        cp = StudyCheckpoint("per_seed", path=path)
        cp.record("0.50|0", {"fid1": 0.8})
        cp.persist()

        # New instance simulating a restart.
        cp2 = StudyCheckpoint("per_seed", path=path)
        n = cp2.resume()
        assert n == 1
        assert cp2.is_done("0.50|0")
        assert not cp2.is_done("0.50|1")

    def test_pending_skips_completed(self, tmp_path):
        cp = StudyCheckpoint("per_seed", path=tmp_path / "s.json")
        cp.record("0.50|0", {})
        all_units = ["0.50|0", "0.50|1", "0.50|2"]
        assert cp.pending(all_units) == ["0.50|1", "0.50|2"]

    def test_injected_writer_is_used(self, tmp_path):
        seen = {}

        def writer(payload):
            seen["payload"] = payload
            return "written"

        cp = StudyCheckpoint("cache", writer=writer)
        cp.record("k", 123)
        result = cp.persist(extra={"params": {"n": 9}})
        assert result == "written"
        assert seen["payload"]["cache"] == {"k": 123}
        assert seen["payload"]["params"] == {"n": 9}

    def test_requires_writer_or_path(self):
        with pytest.raises(ValueError):
            StudyCheckpoint("units")

    def test_resume_without_path_returns_zero(self):
        cp = StudyCheckpoint("units", writer=lambda p: None)
        assert cp.resume() == 0


class TestResumeOrderedList:
    def test_reads_nested_list(self, tmp_path):
        path = tmp_path / "point.json"
        atomic_write_json(path, {"point": {"runs": [{"hop": 0}, {"hop": 1}]}})
        runs = resume_ordered_list(path, "runs", inside="point")
        assert len(runs) == 2
        assert runs[1]["hop"] == 1

    def test_reads_top_level_list(self, tmp_path):
        path = tmp_path / "flat.json"
        atomic_write_json(path, {"runs": [1, 2, 3]})
        assert resume_ordered_list(path, "runs") == [1, 2, 3]

    def test_missing_returns_empty(self, tmp_path):
        assert resume_ordered_list(tmp_path / "nope.json", "runs") == []

    def test_absent_key_returns_empty(self, tmp_path):
        path = tmp_path / "p.json"
        atomic_write_json(path, {"point": {"done": True}})
        assert resume_ordered_list(path, "runs", inside="point") == []
