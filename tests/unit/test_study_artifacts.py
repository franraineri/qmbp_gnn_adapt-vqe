"""Unit tests for the study artifact naming / tagging / status / writer core.

Covers the experiment-agnostic mechanics in
``qmbp_simulation.framework.study_artifacts``: the canonical filename scheme,
the metadata tag block, file-level status + atomic partial→final promotion,
binary sidecars, and the legacy-path read shim. No ``scripts/`` import.
"""

from __future__ import annotations

import json

import pytest

from qmbp_simulation.framework.study_artifacts import (
    STATUS_DEPRECATED,
    STATUS_FINAL,
    STATUS_PARTIAL,
    PathShim,
    StudyArtifactWriter,
    build_artifact_name,
    build_meta,
    infer_status_from_payload,
    meta_sidecar_path,
    promote_status,
)


class TestBuildArtifactName:
    def test_full_scheme_with_h(self):
        name = build_artifact_name(
            model="tfim_frustrated", topology="square", n=9, p=2, h=0.5,
            j2=0.5, method="metropolis", kind="final",
        )
        assert name == "tfim_frustrated_square_N9_p2_h0.50_J20.50_metropolis_final.json"

    def test_h_precision_is_two_decimals(self):
        name = build_artifact_name(model="tfim", topology="chain_1d", n=8, p=1,
                                   h=0.5, method="vqe", kind="theta", ext="npz")
        assert "_h0.50_" in name and name.endswith(".npz")

    def test_np_convention_preserved(self):
        name = build_artifact_name(model="tfim", topology="square", n=16, p=2,
                                   h=1.0, kind="final")
        assert "_N16_p2_" in name

    def test_sweep_omits_h_token(self):
        name = build_artifact_name(model="tfim", topology="square", n=9, p=2,
                                   method="sweep", kind="analysis")
        assert "_h" not in name
        assert "_N9_p2_J20.00_sweep_analysis.json" in name

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            build_artifact_name(model="m", topology="t", n=4, p=1, kind="bogus")


class TestBuildMeta:
    def test_required_fields_present(self):
        meta = build_meta(experiment="hva_nnn_sweep", run_id="r1",
                          physics={"N": 9, "h": 0.5}, metrics={"fidelity": 0.9})
        for k in ("schema", "status", "experiment", "run_id", "created_utc",
                  "updated_utc", "source_script", "git_commit", "physics",
                  "method", "metrics", "provenance", "artifacts"):
            assert k in meta
        assert meta["status"] == STATUS_PARTIAL
        assert meta["physics"]["N"] == 9

    def test_created_utc_preserved_when_provided(self):
        meta = build_meta(experiment="e", run_id="r", created_utc="2020-01-01T00:00:00+00:00")
        assert meta["created_utc"] == "2020-01-01T00:00:00+00:00"
        assert meta["updated_utc"] != meta["created_utc"]

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError):
            build_meta(experiment="e", run_id="r", status="bogus")


class TestStatusTransitions:
    def test_partial_to_final(self):
        assert promote_status(STATUS_PARTIAL, STATUS_FINAL) == STATUS_FINAL

    def test_partial_to_deprecated(self):
        assert promote_status(STATUS_PARTIAL, STATUS_DEPRECATED) == STATUS_DEPRECATED

    def test_final_to_deprecated(self):
        assert promote_status(STATUS_FINAL, STATUS_DEPRECATED) == STATUS_DEPRECATED

    def test_final_to_partial_illegal(self):
        with pytest.raises(ValueError):
            promote_status(STATUS_FINAL, STATUS_PARTIAL)

    def test_deprecated_is_terminal(self):
        with pytest.raises(ValueError):
            promote_status(STATUS_DEPRECATED, STATUS_FINAL)

    def test_idempotent_same_status(self):
        assert promote_status(STATUS_FINAL, STATUS_FINAL) == STATUS_FINAL


class TestInferStatus:
    def test_done_true_is_final(self):
        assert infer_status_from_payload({"point": {"done": True}}) == STATUS_FINAL

    def test_done_false_is_partial(self):
        assert infer_status_from_payload({"point": {"done": False}}) == STATUS_PARTIAL

    def test_top_level_done(self):
        assert infer_status_from_payload({"done": False}) == STATUS_PARTIAL

    def test_explicit_meta_status_wins(self):
        assert infer_status_from_payload(
            {"meta": {"status": "deprecated"}, "point": {"done": True}}
        ) == STATUS_DEPRECATED

    def test_aggregate_defaults_final(self):
        assert infer_status_from_payload({"per_seed": {"0.50|0": {}}}) == STATUS_FINAL


class TestStudyArtifactWriter:
    def test_partial_then_final_promotion_atomic(self, tmp_path):
        w = StudyArtifactWriter(experiment="hva_nnn_sweep", run_dir=tmp_path / "r1",
                                run_id="r1", source_script="scripts/x.py")
        p1 = w.write_partial("res.json", {"rows": [1]}, physics={"N": 9})
        d1 = json.loads(p1.read_text())
        created = d1["meta"]["created_utc"]
        assert d1["meta"]["status"] == STATUS_PARTIAL
        assert d1["meta"]["result_path"].endswith("res.json")

        p2 = w.write_final("res.json", {"rows": [1, 2]}, physics={"N": 9})
        d2 = json.loads(p2.read_text())
        assert p2 == p1  # same file, promoted in place
        assert d2["meta"]["status"] == STATUS_FINAL
        assert d2["meta"]["created_utc"] == created  # creation time preserved
        assert d2["rows"] == [1, 2]

    def test_no_tmp_files_left(self, tmp_path):
        w = StudyArtifactWriter(experiment="e", run_dir=tmp_path / "r", run_id="r")
        w.write_final("res.json", {"x": 1})
        assert [p for p in (tmp_path / "r").iterdir() if ".tmp." in p.name] == []

    def test_tag_artifact_sidecar(self, tmp_path):
        w = StudyArtifactWriter(experiment="e", run_dir=tmp_path / "r", run_id="r")
        (tmp_path / "r").mkdir(parents=True, exist_ok=True)
        npz = tmp_path / "r" / "theta.npz"
        npz.write_bytes(b"data")
        sp = w.tag_artifact(npz, physics={"N": 9}, status=STATUS_FINAL)
        assert sp == meta_sidecar_path(npz)
        meta = json.loads(sp.read_text())
        assert meta["status"] == STATUS_FINAL
        assert meta["result_path"].endswith("theta.npz")


class TestPathShim:
    def test_resolve_remapped_relative(self, tmp_path):
        shim = PathShim({"old/a.json": "runs/r1/final.json"}, root=tmp_path)
        resolved = shim.resolve("old/a.json")
        assert resolved == tmp_path / "runs/r1/final.json"

    def test_resolve_by_basename(self, tmp_path):
        shim = PathShim({"a.json": "runs/r1/final.json"}, root=tmp_path)
        assert shim.resolve(tmp_path / "somewhere" / "a.json").name == "final.json"

    def test_identity_when_not_mapped(self, tmp_path):
        shim = PathShim({}, root=tmp_path)
        assert shim.resolve("kept.json") == pathlib_Path("kept.json")

    def test_exists_reflects_disk(self, tmp_path):
        (tmp_path / "runs").mkdir()
        target = tmp_path / "runs" / "final.json"
        target.write_text("{}")
        shim = PathShim({"old.json": "runs/final.json"}, root=tmp_path)
        assert shim.exists("old.json") is True
        assert shim.exists("missing.json") is False


# small helper so the identity test doesn't need a top-level import shadow
from pathlib import Path as pathlib_Path  # noqa: E402
