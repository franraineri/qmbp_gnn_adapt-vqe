"""Coverage recommendations must generate EXECUTABLE, p-aware commands.

Regression guards for two integration bugs where recommended commands would
silently corrupt data or fail to run:

1. _gap_low_dashboard_pass_rate emitted an iterative-improve command WITHOUT
   --p-layers, so acting on a p=2 gap would retrain/refine in p=1.
2. compute_retrain_queue (surfaced via _gap_retrain_queue) emitted invalid
   flags (--n-qubits, --retrain) making the command un-runnable.

These tests exercise the real functions and assert the emitted command strings
carry --p-layers and only valid runner flags.
"""

from __future__ import annotations

import json

import pytest


def _write_dashboard(tmp_path, monkeypatch, configs):
    """Point coverage's dashboard path at a synthetic dashboard in tmp."""
    import project_health.core.coverage as cov

    dash = tmp_path / "data" / "model_quality_dashboard.json"
    dash.parent.mkdir(parents=True, exist_ok=True)
    dash.write_text(json.dumps({"configs": configs}))

    # The function resolves the path from its module file location; redirect it
    # by monkeypatching Path resolution is fragile — instead patch the function's
    # root via the module-level attribute if present, else chdir into tmp.
    monkeypatch.chdir(tmp_path)
    return dash


class TestLowPassRateRecommendation:
    def test_recommendation_includes_p_layers_for_p2(self, tmp_path, monkeypatch):
        """A p=2 low-pass-rate config must yield a command with --p-layers 2."""
        import builtins

        import project_health.core.coverage as cov

        # A p=2 config below the 50% pass-rate threshold with enough points.
        configs = [
            {
                "topology": "chain_1d",
                "n_qubits": 10,
                "p_layers": 2,
                "n_points": 12,
                "n_params": 28,
                "model": "tfim_bond_resolved",
                "pass_rate_dual_criterion": 0.20,
                "h_range": [2.0, 3.0],
            }
        ]
        dash = tmp_path / "model_quality_dashboard.json"
        dash.write_text(json.dumps({"configs": configs}))

        # The function resolves the dashboard path internally and opens it.
        # Redirect ONLY that specific open to our synthetic dashboard; delegate
        # everything else to the real open so unrelated I/O is unaffected.
        real_open = builtins.open

        def _fake_open(path, *a, **k):
            if str(path).endswith("model_quality_dashboard.json"):
                return real_open(dash, *a, **k)
            return real_open(path, *a, **k)

        monkeypatch.setattr(builtins, "open", _fake_open)
        # The real data/model_quality_dashboard.json exists in the repo, so the
        # function's dashboard_path.exists() check passes; our _fake_open then
        # feeds it the synthetic p=2 config instead. No exists() patch needed.

        gaps = cov._gap_low_dashboard_pass_rate()
        p2 = [g for g in gaps if g.topology == "chain_1d" and g.p_layers == 2]
        assert p2, "expected a p=2 low-pass-rate gap"
        rec = p2[0].recommendation
        assert "--p-layers 2" in rec, f"recommendation missing --p-layers 2: {rec}"
        assert "--target-n 10" in rec
        assert "--iterative-improve" in rec


class TestRetrainQueueCommandExecutable:
    def test_gap_commands_use_valid_flags(self):
        """Retrain-queue-derived gaps must carry executable commands with p."""
        from qmbp_simulation.predictors.model_zoo import compute_retrain_queue

        queue = compute_retrain_queue()
        if not queue:
            pytest.skip("retrain queue empty — nothing to validate")
        for r in queue:
            cmd = r["command"]
            assert "--n-qubits" not in cmd
            assert not cmd.rstrip().endswith("--retrain")
            assert " --retrain " not in cmd
            assert "--target-n" in cmd and "--force-retrain" in cmd
            assert f"--p-layers {r['p_layers']}" in cmd
