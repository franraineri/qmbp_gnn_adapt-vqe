"""Unit tests for the study catalog builder (qmbp_simulation.framework.study_index).

Covers coordinate extraction across the study's JSON shapes, status inference
(incl. deprecated markers), idempotent index building, query filtering, and the
INDEX.md rendering. No ``scripts/`` import; uses tmp_path fixtures.
"""

from __future__ import annotations

import json

from qmbp_simulation.framework.study_index import (
    build_index,
    build_entry,
    extract_coordinates,
    query_index,
    render_markdown,
    write_index,
)


class TestExtractCoordinates:
    def test_from_point_dict(self):
        payload = {"point": {"topology": "square", "N": 9, "p_layers": 2, "h": 0.5,
                             "J2": 0.5, "gap": 0.22, "e0_exact": -6.9,
                             "fidelity": 0.7, "de_over_gap": 5.3, "cx": 42,
                             "optimizer": "warmstart+metropolis_basinhop", "maxiter": 150}}
        c = extract_coordinates(payload)
        assert c["physics"]["N"] == 9
        assert c["physics"]["h"] == 0.5
        assert c["physics"]["gap"] == 0.22
        assert c["metrics"]["fidelity"] == 0.7
        assert c["metrics"]["cx"] == 42
        assert "metropolis" in c["method"]["optimizer"]

    def test_from_params_block(self):
        payload = {"params": {"N": 18, "p_layers": 2, "J2": 0.5,
                             "model": "tfim_frustrated", "maxiter": 40,
                             "strategy": "second_order", "sigma": 0.15}}
        c = extract_coordinates(payload)
        assert c["physics"]["N"] == 18
        assert c["method"]["strategy"] == "second_order"
        assert c["method"]["sigma"] == 0.15

    def test_from_flat_analysis_dict(self):
        payload = {"topology": "triangular", "N": 9, "p_layers": 2, "h": 1.0,
                   "J2": 0.5, "E0": -5.0, "gap": 0.4}
        c = extract_coordinates(payload)
        assert c["physics"]["topo"] == "triangular"
        assert c["physics"]["e0"] == -5.0

    def test_from_rows_first_element(self):
        payload = {"rows": [{"topology": "square", "N": 6, "p_layers": 1, "h": 2.5,
                            "fidelity": 0.99}]}
        c = extract_coordinates(payload)
        assert c["physics"]["N"] == 6
        assert c["metrics"]["fidelity"] == 0.99

    def test_missing_fields_are_none(self):
        c = extract_coordinates({})
        assert c["physics"]["N"] is None
        assert c["metrics"]["fidelity"] is None


class TestBuildEntryStatus:
    def test_partial_from_done_false(self, tmp_path):
        f = tmp_path / "x.json"
        f.write_text(json.dumps({"point": {"done": False, "N": 18}}))
        e = build_entry(f, tmp_path)
        assert e["status"] == "partial"

    def test_final_from_done_true(self, tmp_path):
        f = tmp_path / "x.json"
        f.write_text(json.dumps({"point": {"done": True, "N": 8}}))
        e = build_entry(f, tmp_path)
        assert e["status"] == "final"

    def test_deprecated_marker(self, tmp_path):
        f = tmp_path / "n9_improve_efg_frustrated.json"
        f.write_text(json.dumps({"data": {}}))
        e = build_entry(f, tmp_path)
        assert e["status"] == "deprecated"

    def test_explicit_meta_status_wins(self, tmp_path):
        f = tmp_path / "x.json"
        f.write_text(json.dumps({"meta": {"status": "final", "experiment": "exp1",
                                          "run_id": "r1"}, "point": {"done": False}}))
        e = build_entry(f, tmp_path)
        assert e["status"] == "final"
        assert e["experiment"] == "exp1"
        assert e["run_id"] == "r1"


class TestBuildIndex:
    def _make_tree(self, root):
        (root / "expA").mkdir(parents=True)
        (root / "expA" / "a.json").write_text(json.dumps(
            {"point": {"done": True, "N": 9, "h": 0.5, "fidelity": 0.8}}))
        (root / "expA" / "b.json").write_text(json.dumps(
            {"point": {"done": False, "N": 9, "h": 0.3}}))
        (root / "n9_improve_efg_frustrated.json").write_text(json.dumps({"data": {}}))
        # sidecar + index must be skipped
        (root / "a.npz.meta.json").write_text(json.dumps({"meta": 1}))

    def test_scan_and_skip_sidecars(self, tmp_path):
        self._make_tree(tmp_path)
        idx = build_index(tmp_path)
        paths = {e["path"] for e in idx["entries"]}
        assert "a.npz.meta.json" not in paths
        assert idx["n_entries"] == 3

    def test_idempotent_entries(self, tmp_path):
        self._make_tree(tmp_path)
        a = build_index(tmp_path)["entries"]
        b = build_index(tmp_path)["entries"]
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

    def test_status_breakdown(self, tmp_path):
        self._make_tree(tmp_path)
        idx = build_index(tmp_path)
        statuses = sorted(e["status"] for e in idx["entries"])
        assert statuses == ["deprecated", "final", "partial"]

    def test_write_index_creates_both_files(self, tmp_path):
        self._make_tree(tmp_path)
        ij, im = write_index(tmp_path)
        assert ij.exists() and im.exists()
        assert "vl_vs_hva Study" in im.read_text()


class TestQuery:
    def _idx(self, tmp_path):
        (tmp_path / "e.json").write_text(json.dumps(
            {"point": {"done": True, "N": 9, "h": 0.5, "fidelity": 0.8,
                       "optimizer": "warmstart+metropolis_basinhop"}}))
        (tmp_path / "f.json").write_text(json.dumps(
            {"point": {"done": True, "N": 18, "h": 1.0, "strategy": "second_order"}}))
        return build_index(tmp_path)

    def test_filter_by_n(self, tmp_path):
        idx = self._idx(tmp_path)
        assert len(query_index(idx, n=9)) == 1

    def test_filter_by_h_tolerance(self, tmp_path):
        idx = self._idx(tmp_path)
        assert len(query_index(idx, h=0.5)) == 1
        assert len(query_index(idx, h=0.501)) == 1  # within tolerance

    def test_filter_by_method_substring(self, tmp_path):
        idx = self._idx(tmp_path)
        assert len(query_index(idx, method="metropolis")) == 1
        assert len(query_index(idx, method="second_order")) == 1

    def test_filter_by_status(self, tmp_path):
        idx = self._idx(tmp_path)
        assert len(query_index(idx, status="final")) == 2


class TestRenderMarkdown:
    def test_grouped_tables_and_status_line(self, tmp_path):
        (tmp_path / "exp1").mkdir()
        (tmp_path / "exp1" / "a.json").write_text(json.dumps(
            {"point": {"done": True, "N": 9, "h": 0.5, "fidelity": 0.8}}))
        md = render_markdown(build_index(tmp_path))
        assert "## exp1" in md
        assert "Status:" in md
        assert "| status | N | p | h |" in md
