"""Training NPZs are namespaced by model — no cross-model data mixing.

Regression guards for the design that separates training data per physics
model via a subdirectory: data/multi_n_training/{model}/{topo}_N{n}_p{p}.npz.

Every model — INCLUDING the default (tfim_bond_resolved) — WRITES under its
own per-model subdirectory so two Hamiltonians never share an NPZ (different
physics, often different θ dimension → would otherwise crash the aggregator
with an "Inconsistent node_features" error or silently mix data). READS for
the default model additionally fall back to the legacy data-root, so the
pre-migration corpus keeps loading until it is migrated into the subdir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qmbp_simulation.framework.result_io import (
    DEFAULT_MODEL_NAMESPACE,
    build_data_dir,
    iter_all_training_npzs,
    training_npz_glob,
    training_npz_path,
    training_npz_read_dirs,
)
from qmbp_simulation.utils.helpers import load_theta_from_npz


class TestBuildDataDir:
    def test_default_model_writes_to_subdir(self):
        root = Path("data/multi_n_training")
        # The default model now WRITES under its own subdir (not the root).
        assert build_data_dir(root) == root / DEFAULT_MODEL_NAMESPACE
        assert build_data_dir(root, model=DEFAULT_MODEL_NAMESPACE) == root / DEFAULT_MODEL_NAMESPACE

    def test_non_default_model_gets_subdir(self):
        root = Path("data/multi_n_training")
        assert build_data_dir(root, model="tfim_frustrated") == root / "tfim_frustrated"
        assert build_data_dir(root, model="xy") == root / "xy"

    def test_frustrated_namespace_stacks(self):
        root = Path("data/multi_n_training")
        # Default model + frustrated → {model}/frustrated
        assert build_data_dir(root, frustrated=True) == root / DEFAULT_MODEL_NAMESPACE / "frustrated"
        # model subdir + frustrated stack in order model/frustrated
        assert build_data_dir(root, model="xy", frustrated=True) == root / "xy" / "frustrated"


class TestTrainingNpzReadDirs:
    def test_default_model_reads_subdir_then_root(self):
        root = Path("R")
        dirs = training_npz_read_dirs(root, model=DEFAULT_MODEL_NAMESPACE)
        # Subdir first (canonical write location), legacy root second (fallback).
        assert dirs == [root / DEFAULT_MODEL_NAMESPACE, root]

    def test_non_default_model_reads_only_subdir(self):
        root = Path("R")
        dirs = training_npz_read_dirs(root, model="tfim_frustrated")
        # Non-default models never fall back to the root (belongs to default).
        assert dirs == [root / "tfim_frustrated"]


class TestTrainingNpzPath:
    def test_default_model_write_path_is_subdir(self):
        # WRITE path is always the per-model subdir now.
        p = training_npz_path("chain_1d", 10, 1, root=Path("R"))
        assert p == Path("R") / DEFAULT_MODEL_NAMESPACE / "chain_1d_N10_p1.npz"

    def test_non_default_model_path_subdir(self):
        p = training_npz_path("chain_1d", 10, 1, model="tfim_frustrated", root=Path("R"))
        assert p == Path("R/tfim_frustrated/chain_1d_N10_p1.npz")

    def test_glob_matches_path_dir(self):
        # The glob dir must equal the path's parent, so the aggregator scans
        # exactly where the runner writes.
        path = training_npz_path("chain_1d", 10, 2, model="xy", root=Path("R"))
        gdir, pattern = training_npz_glob("chain_1d", 2, model="xy", root=Path("R"))
        assert gdir == path.parent
        assert pattern == "chain_1d_N*_p2.npz"
        # The path's filename must match the glob pattern.
        import fnmatch

        assert fnmatch.fnmatch(path.name, pattern)


def _write_npz(path: Path, n_params: int, model_field: str | None = None):
    """Write a minimal valid training NPZ at path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h = np.array([2.5, 3.0])
    theta = np.zeros((2, n_params))
    e_vqe = np.array([-10.0, -11.0])
    e_exact = np.array([-10.02, -11.02])
    gaps = np.array([0.5, 0.6])
    de_gaps = np.abs(e_vqe - e_exact) / gaps
    payload = dict(
        h_values=h, theta_opt=theta, e_vqe=e_vqe, e_exact=e_exact,
        gaps=gaps, de_gaps=de_gaps,
        quality_tier=np.array(["verified", "verified"], dtype=object),
    )
    if model_field is not None:
        payload["model"] = model_field
    np.savez(path, **payload)


class TestLoadThetaFromNpzModel:
    def test_default_reads_root_corpus(self):
        # Uses the real repo corpus: default model at root should be found.
        d = load_theta_from_npz("chain_1d", 10, p_layers=1)
        assert d is not None and len(d) > 0

    def test_nonexistent_model_subdir_returns_none(self):
        d = load_theta_from_npz("chain_1d", 10, p_layers=1, model="tfim_frustrated")
        assert d is None  # subdir empty (or absent) → no crossing with default


class TestDefaultReadFallback:
    """The default model reads its subdir first, falling back to the legacy root."""

    def test_write_lands_in_subdir_read_finds_it(self, tmp_path):
        root = tmp_path / "data" / "multi_n_training"
        # Write via the canonical write path (must be the subdir).
        wpath = training_npz_path("chain_1d", 18, 1, root=root, for_write=True)
        assert wpath.parent == root / DEFAULT_MODEL_NAMESPACE
        _write_npz(wpath, n_params=7)
        # A read resolves to the same subdir file.
        rpath = training_npz_path("chain_1d", 18, 1, root=root, for_write=False)
        assert rpath == wpath

    def test_legacy_root_file_still_read(self, tmp_path):
        root = tmp_path / "data" / "multi_n_training"
        # Simulate an UN-migrated file sitting at the legacy root.
        legacy = root / "chain_1d_N18_p1.npz"
        _write_npz(legacy, n_params=7)
        # No subdir file exists → read falls back to the legacy root.
        rpath = training_npz_path("chain_1d", 18, 1, root=root, for_write=False)
        assert rpath == legacy

    def test_subdir_wins_over_root_when_both_exist(self, tmp_path):
        root = tmp_path / "data" / "multi_n_training"
        legacy = root / "chain_1d_N18_p1.npz"
        subdir = root / DEFAULT_MODEL_NAMESPACE / "chain_1d_N18_p1.npz"
        _write_npz(legacy, n_params=7)
        _write_npz(subdir, n_params=7)
        # Read prefers the migrated subdir copy.
        rpath = training_npz_path("chain_1d", 18, 1, root=root, for_write=False)
        assert rpath == subdir


class TestIterAllTrainingNpzs:
    """The model-agnostic corpus scanner sees subdirs + legacy root, deduped."""

    def test_collects_subdirs_and_root_dedup(self, tmp_path):
        root = tmp_path / "data" / "multi_n_training"
        _write_npz(root / "chain_1d_N18_p1.npz", n_params=7)  # legacy root
        _write_npz(root / DEFAULT_MODEL_NAMESPACE / "chain_1d_N20_p1.npz", n_params=7)
        _write_npz(root / "tfim_frustrated" / "chain_1d_N18_p1.npz", n_params=11)
        # A same-named file in a subdir shadows the root one (dedup by name).
        _write_npz(root / DEFAULT_MODEL_NAMESPACE / "chain_1d_N18_p1.npz", n_params=7)

        files = iter_all_training_npzs(root)
        names = sorted(f.name for f in files)
        # chain_1d_N18_p1.npz appears once (deduped), plus the N20 file.
        assert names.count("chain_1d_N18_p1.npz") == 1
        assert "chain_1d_N20_p1.npz" in names

    def test_skips_underscore_dirs(self, tmp_path):
        root = tmp_path / "data" / "multi_n_training"
        _write_npz(root / "_quarantine" / "chain_1d_N18_p1.npz", n_params=7)
        _write_npz(root / DEFAULT_MODEL_NAMESPACE / "chain_1d_N20_p1.npz", n_params=7)
        files = iter_all_training_npzs(root)
        # The _quarantine file is excluded; only the real subdir file remains.
        assert [f.name for f in files] == ["chain_1d_N20_p1.npz"]


class TestAggregatorNoCrossModel:
    """The aggregator must never mix two models' NPZs at the same (topo,N,p)."""

    def test_two_models_do_not_cross(self, tmp_path, monkeypatch):
        import qmbp_simulation.predictors.multi_n_aggregator as agg_mod

        # Redirect the aggregator's project root to tmp so it scans our fixtures.
        monkeypatch.setattr(agg_mod, "_PROJECT_ROOT", tmp_path)

        # Neutralize the real exclusion registry / dashboard so the test is
        # not affected by production exclusions of specific (topo,N).
        monkeypatch.setattr(
            "qmbp_simulation.predictors.multi_n_aggregator.MultiNAggregator._load_not_useful_files",
            lambda self: set(),
        )
        monkeypatch.setattr(
            "qmbp_simulation.predictors.multi_n_aggregator.MultiNAggregator._load_exclusion_registry",
            lambda self: set(),
        )

        train_root = tmp_path / "data" / "multi_n_training"
        # Use N=18 (not in the production exclusion list; below any max_n).
        # Default model at root: NN-only θ dim (7 here).
        _write_npz(train_root / "chain_1d_N18_p1.npz", n_params=7)
        # A different model in its subdir: different θ dim (11).
        _write_npz(
            train_root / "tfim_frustrated" / "chain_1d_N18_p1.npz",
            n_params=11,
            model_field="tfim_frustrated",
        )

        from qmbp_simulation.predictors.multi_n_aggregator import MultiNAggregator

        # Aggregator for the default model must see ONLY the root file.
        agg_default = MultiNAggregator(
            topology="chain_1d", model="tfim_bond_resolved", p_layers=1
        )
        agg_default.scan()
        pts_default = agg_default._data_by_n.get(18, [])
        assert pts_default, "default model should find its root NPZ"
        assert all(len(p["theta"]) == 7 for p in pts_default), (
            "default aggregator leaked the frustrated model's 11-dim θ"
        )

        # Aggregator for tfim_frustrated must see ONLY its subdir file.
        agg_frust = MultiNAggregator(
            topology="chain_1d", model="tfim_frustrated", p_layers=1
        )
        agg_frust.scan()
        pts_frust = agg_frust._data_by_n.get(18, [])
        assert pts_frust, "frustrated model should find its subdir NPZ"
        assert all(len(p["theta"]) == 11 for p in pts_frust), (
            "frustrated aggregator leaked the default model's 7-dim θ"
        )


class TestMigrationAudit:
    """The migration/audit tool must import and detect layout without moving."""

    def test_audit_runs_readonly(self, capsys):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "migrate_npz_by_model",
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "general_project_maintenance"
            / "migrate_npz_by_model.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        rc = mod.audit()
        assert rc == 0
        out = capsys.readouterr().out
        assert "default model at root" in out
