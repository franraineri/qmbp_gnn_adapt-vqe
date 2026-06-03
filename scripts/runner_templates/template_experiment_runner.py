#!/usr/bin/env python3
"""Template: BaseExperiment lifecycle wrapper.

Use this when your experiment already has a BaseExperiment subclass
and you just need a script to run it with proper CLI, preflight, and exit codes.

The full lifecycle (preflight → setup → run → analyze → report → save)
is handled by BaseExperiment.execute() — this runner just provides the
entry point with CLI args and error handling.

Usage:
    python scripts/my_experiment.py
    python scripts/my_experiment.py --n-qubits 10
    python scripts/my_experiment.py --seeds 42 43
    python scripts/my_experiment.py --topology ladder
    python scripts/my_experiment.py --verbose
    python scripts/my_experiment.py --skip-preflight
"""

from __future__ import annotations

import sys

from qmbp_simulation.framework.runner_base import ExperimentRunner, resolve_project_root

# ─── Project root setup (works from any script depth) ────────────────────────
_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class RunMyExperiment(ExperimentRunner):
    """Runner for <ExperimentName>.

    The BaseExperiment lifecycle is handled automatically:
    1. Preflight validation (regime, sweep order, model compatibility).
    2. setup() — build circuits, solver, HVA.
    3. run() — sweep across seeds with checkpointing.
    4. analyze() — compute summary statistics.
    5. report() — print human-readable results.
    6. save() — persist to results/experiments/exp_{id}/run_{ts}.json.

    This runner adds:
    - CLI argument parsing (--n-qubits, --seeds, --topology, --verbose).
    - Proper exit codes (0=success, 1=failure).
    - Exception handling with clear error messages.
    - Optional post_execute() hook for extra analysis.
    """

    runner_id = "<RUNNER_ID>"  # e.g., "run_e4b"

    def get_experiment_class(self):
        """Lazy import of the experiment class.

        IMPORTANT: Put the import here (not at module level) so that:
        - Preflight failures are fast (no torch/qiskit loading).
        - Import errors produce clear messages.
        """
        from experiments.generalization.exp_e4b_longitudinal_hva_extended import (
            ExperimentE4b,
        )

        return ExperimentE4b

    def build_config(self):
        """Customize config with CLI overrides.

        The base class handles --n-qubits, --seeds, --topology automatically.
        Override to add experiment-specific config tweaks.
        """
        config = super().build_config()

        # Example: apply custom overrides from _add_custom_args
        # if self._args.p_layers is not None:
        #     config.system.p_layers = self._args.p_layers

        return config

    def post_execute(self, analysis):
        """Optional post-processing after execute() completes.

        Called only on success. Use for printing extra tables,
        computing derived metrics, etc.
        """
        summary = analysis.get("summary", {})
        if summary and "pass_rate" in summary:
            rate = summary["pass_rate"]
            verdict = "CONFIRMED" if rate >= 0.9 else "REJECTED"
            print(f"\n  Overall verdict: {verdict} (pass_rate={rate:.1%})")

    @classmethod
    def _add_custom_args(cls, parser):
        """Add experiment-specific CLI arguments."""
        # parser.add_argument("--p-layers", type=int, default=None)
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    RunMyExperiment.main()
