#!/usr/bin/env python3
"""Template: Multi-section validation runner.

Copy this file and customize for your validation suite.
Replace all <PLACEHOLDER> markers with your values.

Guarantees provided by the framework:
- Preflight validates runner_id, experiment_id, hypothesis, sections before run.
- Each section is isolated: exceptions don't abort the suite.
- Results auto-saved to results/experiments/exp_{id}/run_{timestamp}.json.
- Structured log saved alongside for post-hoc timing analysis.
- Exit code is non-zero if any section fails.
- --dry-run lists sections without executing.
- --stop-on-failure aborts on first failure.
- --section N M runs only selected sections.

Best practices (from ZNE validation suite, 2026-06-04):
- ALWAYS provide a per-section `hypothesis` string (avoids preflight warnings).
- Use self.vqe_descending_sweep() for noiseless baselines (avoids code duplication).
- Use self.exact_ground_state() for (e_exact, gap) lookup.
- For noisy sections, cache layout selection in setup/section_2 and reuse in
  sections 3/4 — ensures fair comparison across methods (same physical layout).
- Include build_config() with system params for digest/compare.py compatibility.

Usage:
    python scripts/my_validation.py
    python scripts/my_validation.py --section 1 2
    python scripts/my_validation.py --dry-run
    python scripts/my_validation.py --stop-on-failure
    python scripts/my_validation.py --verbose
    python scripts/my_validation.py --skip-preflight
"""

from __future__ import annotations

import logging
import sys

from qmbp_simulation.framework.runner_base import (
    Section,
    ValidationRunner,
    resolve_project_root,
)

# ─── Project root setup (works from any script depth) ────────────────────────
_ROOT = resolve_project_root(__file__)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Runner Implementation
# ═══════════════════════════════════════════════════════════════════════════════


class MyValidationRunner(ValidationRunner):
    """<DESCRIPTION> — replace with your validation suite description.

    Sections:
        1. <Section 1 name>
        2. <Section 2 name>
        ...
    """

    # ── Required class attributes ────────────────────────────────────────────
    runner_id = "<RUNNER_ID>"  # e.g., "e4b_hw_readiness"
    experiment_id = "<EXP_ID>"  # e.g., "E4b" (matches result_io naming)
    description = "<ONE LINE DESC>"  # e.g., "E4b Hardware Readiness Suite"
    hypothesis = "<OVERALL HYPOTHESIS>"  # What the suite collectively tests

    # ── Optional: custom CLI args ────────────────────────────────────────────
    @classmethod
    def _add_custom_args(cls, parser):
        """Add experiment-specific CLI arguments.

        These are accessible as self._args.<name> in section methods.
        """
        parser.add_argument(
            "--n-qubits",
            type=int,
            default=6,
            help="Number of qubits (default: %(default)s)",
        )
        parser.add_argument(
            "--topology",
            type=str,
            default="chain_1d",
            help="Lattice topology (default: %(default)s)",
        )

    # ── Optional: custom config for result envelope ──────────────────────────
    def build_config(self) -> dict:
        """Build the config dict saved in the result JSON.

        This is what appears under the "config" key in the output file.
        Include everything needed to reproduce this run.
        """
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "system": {
                "n_qubits": self._args.n_qubits,
                "p_layers": 2,
                "topology": self._args.topology,
            },
            "seeds": [42, 43, 44],
        }

    # ── Optional: custom preflight ───────────────────────────────────────────
    def run_preflight(self) -> bool:
        """Custom preflight validation.

        Call super() first to retain structural checks (runner_id, etc.).
        Add experiment-specific checks after.
        """
        if not super().run_preflight():  # noqa: SIM103
            return False

        # Example: validate h_test is within valid regime
        # from qmbp_simulation.framework.preflight import get_regime_threshold
        # threshold = get_regime_threshold(
        #     self._args.topology, self._args.n_qubits, P_LAYERS
        # )
        # if H_TEST < threshold:
        #     logger.error(f"h_test={H_TEST} < threshold={threshold}")
        #     return False

        return True

    # ── Optional: shared setup (lazy imports go here) ────────────────────────
    def setup(self):
        """One-time initialization after preflight passes.

        Put expensive imports and object construction here — NOT at module level.
        This way, preflight failures are fast and don't require heavy deps.

        PATTERNS:
        - Store expensive-to-import modules as self._minimize, self._make_lattice
        - Initialize shared state for cross-section data passing:
              self._phase2_data = None   # Populated in section 1, used in section 2
        - Build circuits/objects that are reused across sections:
              self._circuit, _ = self.hva.create(...)
              self._n_params = self._circuit.num_parameters
        - Resolve CLI args into instance state:
              self._seed = self._args.seed
        """
        from qmbp_simulation import HamiltonianBuilder
        from qmbp_simulation.circuits import HVACircuitBuilder
        from qmbp_simulation.execution import NoiselessBackend

        self.builder = HamiltonianBuilder()
        self.hva = HVACircuitBuilder()
        self.backend = NoiselessBackend()

    # ── Section definitions (REQUIRED) ───────────────────────────────────────
    def define_sections(self) -> list[Section]:
        """Define all validation sections.

        Rules:
        - Each section fn MUST return a dict.
        - Include "pass": False to signal failure explicitly.
        - If "pass" key is absent, success is assumed (unless exception).
        - Sections are isolated: one exception doesn't abort others.
        """
        return [
            Section(
                id=1,
                name="Reproducibility Check",
                fn=self.section_reproducibility,
                hypothesis="Same seed produces identical VQE parameters",
            ),
            Section(
                id=2,
                name="Cross-Topology Validation",
                fn=self.section_cross_topology,
                hypothesis="Framework achieves fid≥0.90 on ladder and triangular",
            ),
            # Add more sections...
        ]

    # ── Section implementations ──────────────────────────────────────────────

    def section_reproducibility(self) -> dict:
        """Section 1: Verify deterministic execution.

        Pattern:
        1. Run computation.
        2. Log results via logger.info() for console output.
        3. Return dict with metrics + "pass" key.

        IMPORT RULES for sections:
        - Put per-section imports at the TOP of the method body (not in loops).
        - Only import what this section needs (not all heavy deps).
        - If a section depends on another section's output, check and raise:
              if self._phase2_data is None:
                  raise RuntimeError("Section 1 must run first")
        """
        logger.info("  Running two identical VQE executions...")

        # ... your computation here ...
        diff = 0.0  # np.max(np.abs(results[0] - results[1]))

        passed = diff < 1e-12
        logger.info(f"  Max parameter diff: {diff:.2e}")
        logger.info(f"  {'[PASS]' if passed else '[FAIL]'} Reproducibility")

        return {
            "max_diff": diff,
            "threshold": 1e-12,
            "pass": passed,
        }

    def section_cross_topology(self) -> dict:
        """Section 2: Validate across topologies."""
        logger.info("  Testing ladder and triangular lattices...")

        # ... your computation here ...
        results_by_topo = {}

        # Example: use built-in VQE sweep helper
        # theta_map = self.vqe_descending_sweep(
        #     topology="ladder", n_qubits=6,
        #     h_values=[2.0, 1.75, 1.5, 1.25],
        #     seed=42, p_layers=2,
        # )

        # Example: use built-in exact energy helper
        # e_exact, gap = self.exact_ground_state("ladder", 6, h=1.5)

        # Log table
        logger.info(f"  {'Topology':<12} | {'Mean Fid':>8} | {'ΔE/gap':>7}")
        logger.info(f"  {'-' * 12}-+-{'-' * 8}-+-{'-' * 7}")

        all_pass = True  # Compute from actual results
        return {
            "topologies": results_by_topo,
            "pass": all_pass,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    MyValidationRunner.main()
