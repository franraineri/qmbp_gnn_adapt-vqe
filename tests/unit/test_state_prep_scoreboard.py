"""Unit tests for the incremental state-prep scoreboard.

Covers the upsert contract (add / improve-only / keep-on-tie), deterministic
ordering, h-precision matching, and the JSON↔markdown roundtrip. Pure logic in
``src``, no I/O.
"""

from __future__ import annotations

from qmbp_simulation.analysis.state_prep_scoreboard import (
    build_document,
    config_key,
    make_entry,
    sort_entries,
    to_markdown,
    upsert,
    upsert_many,
)


def _e(method, n, h, variant, fidelity, **kw):
    return make_entry(method=method, n_qubits=n, h=h, variant=variant, fidelity=fidelity, **kw)


class TestUpsert:
    def test_add_new_config(self):
        entries, action = upsert([], _e("HVA", 9, 0.5, "nnn p=2", 0.95))
        assert action == "added"
        assert len(entries) == 1
        assert entries[0]["fidelity"] == 0.95

    def test_improves_only_when_higher(self):
        entries, _ = upsert([], _e("HVA", 9, 0.5, "nnn p=2", 0.90), now="2026-01-01T00:00:00Z")
        entries, action = upsert(entries, _e("HVA", 9, 0.5, "nnn p=2", 0.96), now="2026-01-02T00:00:00Z")
        assert action == "improved"
        assert entries[0]["fidelity"] == 0.96
        assert entries[0]["previous_fidelity"] == 0.90
        assert entries[0]["updated_utc"] == "2026-01-02T00:00:00Z"

    def test_keeps_incumbent_when_not_better(self):
        entries, _ = upsert([], _e("HVA", 9, 0.5, "nnn p=2", 0.96))
        entries, action = upsert(entries, _e("HVA", 9, 0.5, "nnn p=2", 0.90))
        assert action == "kept"
        assert entries[0]["fidelity"] == 0.96

    def test_tie_keeps_incumbent(self):
        entries, _ = upsert([], _e("HVA", 9, 0.5, "nnn p=2", 0.95))
        entries, action = upsert(entries, _e("HVA", 9, 0.5, "nnn p=2", 0.95))
        assert action == "kept"
        assert len(entries) == 1

    def test_distinct_configs_coexist(self):
        entries, _ = upsert([], _e("HVA", 9, 0.5, "nnn p=2", 0.95))
        entries, _ = upsert(entries, _e("VL", 9, 0.5, "L8/F50", 0.94, loader="vector"))
        entries, _ = upsert(entries, _e("HVA", 18, 0.5, "p2_base", 0.65))
        assert len(entries) == 3

    def test_h_matched_at_two_decimals(self):
        # 0.5 and 0.50 are the same config; 0.51 is different.
        entries, _ = upsert([], _e("HVA", 9, 0.5, "nnn p=2", 0.90))
        entries, action = upsert(entries, _e("HVA", 9, 0.50, "nnn p=2", 0.95))
        assert action == "improved"
        assert len(entries) == 1
        entries, action = upsert(entries, _e("HVA", 9, 0.51, "nnn p=2", 0.10))
        assert action == "added"
        assert len(entries) == 2

    def test_does_not_mutate_input(self):
        original = [_e("HVA", 9, 0.5, "nnn p=2", 0.90)]
        snapshot = list(original)
        upsert(original, _e("HVA", 9, 0.5, "nnn p=2", 0.99))
        assert original == snapshot  # unchanged


class TestUpsertMany:
    def test_batch_counts(self):
        cands = [
            _e("HVA", 9, 0.5, "nnn p=2", 0.90),
            _e("HVA", 9, 0.5, "nnn p=2", 0.95),  # improves the first
            _e("VL", 9, 0.5, "L8/F50", 0.94),  # new
            _e("VL", 9, 0.5, "L8/F50", 0.80),  # kept
        ]
        entries, counts = upsert_many([], cands)
        assert counts["added"] == 2
        assert counts["improved"] == 1
        assert counts["kept"] == 1
        # final fidelities reflect the best seen
        by_key = {config_key(e): e["fidelity"] for e in entries}
        assert by_key[("HVA", 9, 0.5, "nnn p=2", None)] == 0.95
        assert by_key[("VL", 9, 0.5, "L8/F50", None)] == 0.94


class TestOrderingAndDocument:
    def test_sort_order_deterministic(self):
        entries = [
            _e("VL", 18, 1.0, "L8/F50", 0.79),
            _e("HVA", 9, 0.5, "nnn p=2", 0.95),
            _e("HVA", 9, 0.5, "nn p=1", 0.72),
            _e("VL", 9, 0.5, "L8/F50", 0.94),
        ]
        ordered = sort_entries(entries)
        # Primary key N asc: the two N=9 come before N=18.
        assert ordered[0]["n_qubits"] == 9
        assert ordered[-1]["n_qubits"] == 18
        # Within N=9 h=0.5, higher fidelity first within a method group.
        n9 = [e for e in ordered if e["n_qubits"] == 9]
        hva9 = [e for e in n9 if e["method"] == "HVA"]
        assert hva9[0]["fidelity"] >= hva9[-1]["fidelity"]

    def test_document_shape(self):
        entries, _ = upsert([], _e("HVA", 9, 0.5, "nnn p=2", 0.95))
        doc = build_document(entries)
        assert doc["schema"] == "state_prep_scoreboard_v1"
        assert doc["n_entries"] == 1
        assert doc["entries"][0]["method"] == "HVA"

    def test_markdown_renders_entries(self):
        entries, _ = upsert([], _e("VL", 18, 0.5, "L8/F50", 0.86, loader="mps", chi=64, n_2q=264, total_gates=1768))
        md = to_markdown(build_document(entries))
        assert "Best-results scoreboard" in md
        assert "mps χ=64" in md
        assert "0.8600" in md
