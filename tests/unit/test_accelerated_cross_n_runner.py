"""Tests for AcceleratedCrossNRunner control logic (no heavy compute).

Covers the runner's pure decision logic that does NOT require running VQE/MPNN:
- define_sections: which sections run for each flag combination
  (standard / multi-n-train / force-retrain / from-zoo / --checkpoint /
   iterative-improve / budget-only)
- run_preflight: ladder even-N constraint, train-h-min>max guard
- _resolve_p_layers: int / list / absent
- _training_h_range + _train_h_range_note: zoo metadata h-range

The runner lives under scripts/ but its branching logic is the single source
of truth for how experiments are staged, so we import and test it directly.
"""

from __future__ import annotations

import argparse
import importlib

import pytest

_mod = importlib.import_module(
    "scripts.experiment_runners.bond_resolved.run_accelerated_cross_n"
)
AcceleratedCrossNRunner = _mod.AcceleratedCrossNRunner


def _make_runner(**overrides):
    """Instantiate the runner WITHOUT running it, with sane arg defaults.

    Uses __new__ + manual _args assignment to bypass __init__ side effects
    (logging setup, artifact store). define_sections / run_preflight only read
    from self._args, so this is sufficient and fast.
    """
    defaults = dict(
        train_n=10,
        target_n=[20],
        p_layers=[1],
        topology="chain_1d",
        h_min=2.0,
        h_max=4.5,
        h_points=15,
        train_h_min=None,
        train_h_max=None,
        from_zoo=False,
        checkpoint=None,
        multi_n_train=False,
        force_retrain=False,
        iterative_improve=False,
        budget_only=False,
        active_rounds=0,
        skip_retrain=False,
        refine_all=True,
        max_refine_per_iter=None,
    )
    defaults.update(overrides)
    runner = AcceleratedCrossNRunner.__new__(AcceleratedCrossNRunner)
    runner._args = argparse.Namespace(**defaults)
    return runner


def _section_names(runner):
    return [s.name for s in runner.define_sections()]


def _section_ids(runner):
    return [s.id for s in runner.define_sections()]


# ═══════════════════════════════════════════════════════════════════════════
# define_sections — staging logic per flag combination
# ═══════════════════════════════════════════════════════════════════════════


class TestDefineSections:
    def test_standard_train_predict(self):
        # Default: quality check → train(N) → cross-N predict
        r = _make_runner()
        names = _section_names(r)
        assert len(names) == 3
        assert "Quality Check" in names[0]
        assert "Train" in names[1] and "N=10" in names[1]
        assert "Cross-N Predict" in names[2]

    def test_multi_n_train_replaces_single_train(self):
        r = _make_runner(multi_n_train=True)
        names = _section_names(r)
        assert any("Multi-N Train" in n for n in names)
        assert not any(n.startswith("Train (") for n in names)

    def test_force_retrain_uses_multi_n_train_section(self):
        r = _make_runner(force_retrain=True)
        assert any("Multi-N Train" in n for n in _section_names(r))

    def test_from_zoo_is_predict_only(self):
        # --from-zoo → no training section at all
        r = _make_runner(from_zoo=True)
        names = _section_names(r)
        assert not any("Train" in n for n in names)
        assert any("Cross-N Predict" in n for n in names)

    def test_explicit_checkpoint_is_predict_only(self):
        # --checkpoint → treated like --from-zoo (use only this model)
        r = _make_runner(checkpoint="h_0p5_1p5")
        names = _section_names(r)
        assert not any("Train (" in n for n in names)
        assert any("Cross-N Predict" in n for n in names)

    def test_iterative_improve_sections(self):
        r = _make_runner(iterative_improve=True)
        names = _section_names(r)
        assert any("Quality Check" in n for n in names)
        assert any("Budget Estimation" in n for n in names)
        assert any("Iterative Improvement" in n for n in names)
        # No standard train / cross-N predict in iterative mode
        assert not any("Cross-N Predict" in n for n in names)

    def test_budget_only_stops_before_loop(self):
        r = _make_runner(iterative_improve=True, budget_only=True)
        names = _section_names(r)
        assert any("Budget Estimation" in n for n in names)
        assert not any("Iterative Improvement" in n for n in names)

    def test_section_ids_unique_within_mode(self):
        # IDs should not collide within a single mode's section list.
        for r in (
            _make_runner(),
            _make_runner(multi_n_train=True),
            _make_runner(from_zoo=True),
            _make_runner(iterative_improve=True),
        ):
            ids = _section_ids(r)
            assert len(ids) == len(set(ids)), f"duplicate section ids: {ids}"


# ═══════════════════════════════════════════════════════════════════════════
# run_preflight — topology / h-range constraints
# ═══════════════════════════════════════════════════════════════════════════


class TestRunPreflight:
    def test_chain_1d_passes(self):
        assert _make_runner().run_preflight() is True

    def test_ladder_rejects_odd_target_n(self):
        r = _make_runner(topology="ladder", train_n=10, target_n=[15])
        assert r.run_preflight() is False

    def test_ladder_rejects_odd_train_n(self):
        r = _make_runner(topology="ladder", train_n=9, target_n=[20])
        assert r.run_preflight() is False

    def test_ladder_accepts_even_n(self):
        r = _make_runner(topology="ladder", train_n=10, target_n=[16, 20])
        assert r.run_preflight() is True

    def test_ladder_odd_train_n_ok_when_from_zoo(self):
        # from_zoo skips training → odd train_n is irrelevant
        r = _make_runner(topology="ladder", train_n=9, target_n=[20], from_zoo=True)
        assert r.run_preflight() is True

    def test_train_h_min_greater_than_max_fails(self):
        r = _make_runner(train_h_min=3.0, train_h_max=2.0, multi_n_train=True)
        assert r.run_preflight() is False

    def test_train_h_range_valid_passes(self):
        r = _make_runner(train_h_min=2.0, train_h_max=3.0, multi_n_train=True)
        assert r.run_preflight() is True


# ═══════════════════════════════════════════════════════════════════════════
# _resolve_p_layers
# ═══════════════════════════════════════════════════════════════════════════


class TestResolvePLayers:
    def test_list_returns_first(self):
        assert _make_runner(p_layers=[2, 1])._resolve_p_layers() == 2

    def test_int_returns_int(self):
        assert _make_runner(p_layers=1)._resolve_p_layers() == 1

    def test_absent_returns_none(self):
        r = AcceleratedCrossNRunner.__new__(AcceleratedCrossNRunner)
        r._args = argparse.Namespace()  # no p_layers
        assert r._resolve_p_layers() is None


# ═══════════════════════════════════════════════════════════════════════════
# _training_h_range + _train_h_range_note (zoo metadata)
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainingHRange:
    def test_falls_back_to_sweep_range(self):
        r = _make_runner(h_min=2.0, h_max=4.5)
        assert r._training_h_range() == (2.0, 4.5)
        assert r._train_h_range_note() == ""

    def test_uses_explicit_train_range(self):
        r = _make_runner(train_h_min=1.0, train_h_max=2.5)
        assert r._training_h_range() == (1.0, 2.5)
        assert "train_h_range=[1.0, 2.5]" in r._train_h_range_note()

    def test_partial_train_range_note(self):
        r = _make_runner(train_h_min=1.0, train_h_max=None, h_max=4.5)
        lo, hi = r._training_h_range()
        assert lo == 1.0 and hi == 4.5
        assert "1.0" in r._train_h_range_note()
