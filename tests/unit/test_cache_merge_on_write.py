"""Concurrency robustness: caches merge-on-write instead of clobbering.

Regression guard for the corruption that concatenated / overwrote the ground
truth and eval caches when two runners flushed near-simultaneously. Both caches
now delegate their _save() to utils.helpers.merge_write_json_dict, which re-reads
the on-disk state under a cross-process lock and merges before the atomic write.

These tests simulate the race with two in-process instances (same file) writing
DIFFERENT keys: with the old rename-only save the second flush lost the first's
keys; with merge-on-write both survive and the file stays a single valid JSON.
"""

from __future__ import annotations

import json

import numpy as np
import pytest


class TestGroundTruthCacheMergeOnWrite:
    def test_two_instances_accumulate(self, tmp_path):
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        path = tmp_path / "gt.json"
        a = GroundTruthCache(path=path)
        b = GroundTruthCache(path=path)

        a.put("chain_1d", 4, "tfim", 2.5, energy=-4.0, gap=0.5)
        b.put("chain_1d", 6, "tfim", 3.0, energy=-6.0, gap=0.6)
        a.flush()
        b.flush()  # must NOT clobber A's key

        c = GroundTruthCache(path=path)
        assert c.get("chain_1d", 4, "tfim", 2.5) is not None, "A's key was clobbered"
        assert c.get("chain_1d", 6, "tfim", 3.0) is not None, "B's key missing"

    def test_same_key_conflict_keeps_lower_energy(self, tmp_path):
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        path = tmp_path / "gt.json"
        x = GroundTruthCache(path=path)
        x.put("chain_1d", 8, "tfim", 2.0, energy=-8.0, gap=0.5)  # worse
        x.flush()

        y = GroundTruthCache(path=path)
        y.put("chain_1d", 8, "tfim", 2.0, energy=-8.5, gap=0.5)  # better (lower)
        y.flush()

        z = GroundTruthCache(path=path)
        got = z.get("chain_1d", 8, "tfim", 2.0)
        np.testing.assert_allclose(got["energy"], -8.5, atol=1e-9)

    def test_result_is_single_valid_json(self, tmp_path):
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        path = tmp_path / "gt.json"
        a = GroundTruthCache(path=path)
        b = GroundTruthCache(path=path)
        a.put("chain_1d", 4, "tfim", 2.5, energy=-4.0, gap=0.5)
        b.put("chain_1d", 6, "tfim", 3.0, energy=-6.0, gap=0.6)
        a.flush()
        b.flush()
        doc = json.loads(path.read_text())  # must not raise → no concatenation
        assert isinstance(doc, dict) and "entries" in doc

    def test_flush_clears_dirty(self, tmp_path):
        from qmbp_simulation.solvers.ground_truth_cache import GroundTruthCache

        gt = GroundTruthCache(path=tmp_path / "gt.json")
        gt.put("chain_1d", 4, "tfim", 2.0, energy=-4.0, gap=0.5)
        gt.flush()
        assert gt._dirty is False


class TestEvalCacheMergeOnWrite:
    @pytest.mark.parametrize("p_layers", [1, 2])
    def test_two_instances_accumulate(self, tmp_path, p_layers):
        from qmbp_simulation.execution.eval_cache import EvalCache

        path = tmp_path / f"eval_p{p_layers}.json"
        n_params = 8 if p_layers == 1 else 14
        a = EvalCache(path=path)
        b = EvalCache(path=path)
        ka = a.make_key("chain_1d", 4, 2.5, np.zeros(n_params),
                        model="tfim_bond_resolved", p_layers=p_layers)
        kb = b.make_key("chain_1d", 6, 3.0, np.ones(n_params),
                        model="tfim_bond_resolved", p_layers=p_layers)
        a.put(ka, -4.0)
        b.put(kb, -6.0)
        a.flush()
        b.flush()  # must NOT clobber A's key

        c = EvalCache(path=path)
        assert c.get(ka) is not None, "A's key was clobbered"
        assert c.get(kb) is not None, "B's key missing"

    def test_result_is_single_valid_json(self, tmp_path):
        from qmbp_simulation.execution.eval_cache import EvalCache

        path = tmp_path / "eval.json"
        a = EvalCache(path=path)
        b = EvalCache(path=path)
        ka = a.make_key("chain_1d", 4, 2.5, np.zeros(8), model="tfim_bond_resolved", p_layers=1)
        kb = b.make_key("square", 6, 3.0, np.ones(8), model="tfim_bond_resolved", p_layers=1)
        a.put(ka, -4.0)
        b.put(kb, -6.0)
        a.flush()
        b.flush()
        doc = json.loads(path.read_text())  # must not raise
        assert isinstance(doc, dict) and "entries" in doc


class TestMergeWriteJsonDict:
    """Direct unit tests for the reusable helper."""

    def test_union_no_key_lost(self, tmp_path):
        from qmbp_simulation.utils.helpers import merge_write_json_dict

        path = tmp_path / "store.json"
        merge_write_json_dict(path, {"a": 1, "b": 2})
        n = merge_write_json_dict(path, {"c": 3})  # different keys → union
        assert n == 3
        doc = json.loads(path.read_text())
        assert set(doc["entries"]) == {"a", "b", "c"}

    def test_resolve_conflict(self, tmp_path):
        from qmbp_simulation.utils.helpers import merge_write_json_dict

        path = tmp_path / "store.json"
        merge_write_json_dict(path, {"k": 10})
        # keep the smaller value on conflict
        merge_write_json_dict(path, {"k": 5}, resolve=lambda old, new: min(old, new))
        doc = json.loads(path.read_text())
        assert doc["entries"]["k"] == 5

    def test_no_entries_key_writes_top_level(self, tmp_path):
        from qmbp_simulation.utils.helpers import merge_write_json_dict

        path = tmp_path / "flat.json"
        merge_write_json_dict(path, {"x": 1}, entries_key=None)
        doc = json.loads(path.read_text())
        assert doc == {"x": 1}
