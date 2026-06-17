#!/usr/bin/env python3
"""Ext1b: Standalone p=1 revalidation for CONDITIONALLY_VIABLE h-points."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from qmbp_simulation import ClassicalSolver, HamiltonianBuilder, make_lattice
from qmbp_simulation.analysis.extension_models import ExtensionClassification
from qmbp_simulation.framework import (
    Section,
    ValidationRunner,
)
from qmbp_simulation.framework.runner_base import resolve_project_root
from qmbp_simulation.pipeline import PipelineRunner

_ROOT = resolve_project_root(__file__)
logger = logging.getLogger(__name__)

CONDITIONALLY_VIABLE = ExtensionClassification.CONDITIONALLY_VIABLE


class Ext1bP1ValidationRunner(ValidationRunner):
    """Standalone p=1 revalidation for CONDITIONALLY_VIABLE h-points from §10."""

    runner_id = "EXT1B_P1"
    experiment_id = "EXT1B"
    description = "Ext1b: p=1 revalidation for CONDITIONALLY_VIABLE h-points"
    hypothesis = (
        "CONDITIONALLY_VIABLE h-points from §10 achieve ΔE/gap < 5% at p=1, "
        "confirming borderline results before thesis reporting."
    )

    @classmethod
    def _add_custom_args(cls, parser: argparse.ArgumentParser) -> None:
        """Extend base CLI with Ext1b-specific arguments."""
        parser.add_argument(
            "--phase3-results",
            required=True,
            help="Path to Phase 3 JSON result file.",
        )
        parser.add_argument("--n-qubits", type=int, default=6)
        parser.add_argument("--p-layers", type=int, default=1)
        parser.add_argument("--topology", type=str, default="chain_1d")
        parser.add_argument("--h-train", type=float, nargs="+", default=None)
        parser.add_argument("--h-test", type=float, nargs="+", default=None)

    def build_config(self) -> dict:
        """Return config dict for the result envelope."""
        return {
            "runner_id": self.runner_id,
            "experiment_id": self.experiment_id,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "system": {
                "topology": self._args.topology,
                "n_qubits": self._args.n_qubits,
                "p_layers": self._args.p_layers,
            },
            "phase3_results_path": self._args.phase3_results,
        }

    def setup(self) -> None:
        """Enforce p≤2 constraint and load CONDITIONALLY_VIABLE h-points."""
        if self._args.p_layers > 2:
            raise ValueError(
                f"--p-layers={self._args.p_layers} > 2 violates HVA depth "
                "constraint. Maximum allowed: p=2."
            )
        self._cv_h_points = self._load_cv_h_points()

    def _load_cv_h_points(self) -> list[float]:
        """Load CONDITIONALLY_VIABLE h-points from --phase3-results JSON."""
        path = Path(self._args.phase3_results)
        with open(path) as f:
            data = json.load(f)
        cv_h: list[float] = []
        results = data.get("results", data)
        for key, val in results.items() if isinstance(results, dict) else []:
            if isinstance(val, dict):
                clf = val.get("classification", "")
                if clf == CONDITIONALLY_VIABLE.value:
                    h_val = val.get("h")
                    if h_val is None:
                        try:
                            h_val = float(key)
                        except (ValueError, TypeError):
                            logger.warning(f"  Cannot parse h from key '{key}', skipping.")
                            continue
                    cv_h.append(float(h_val))
        return sorted(set(cv_h), reverse=True)

    def define_sections(self) -> list[Section]:
        """Define validation sections for Ext1b p=1 revalidation."""
        return [
            Section(
                id=1,
                name="p=1 Revalidation of CONDITIONALLY_VIABLE h-points",
                fn=self.section_p1_revalidation,
                hypothesis=("All CONDITIONALLY_VIABLE h-points achieve ΔE/gap < 5% at p=1"),
            ),
        ]

    def section_p1_revalidation(self) -> dict:
        """Re-run p=1 VQE for each CONDITIONALLY_VIABLE h-point."""
        if not self._cv_h_points:
            logger.info("  No CONDITIONALLY_VIABLE points found. Skipping revalidation.")
            return {"pass": True, "skipped": True, "reason": "no_cv_points"}

        topology = self._args.topology
        n_qubits = self._args.n_qubits
        p_layers = self._args.p_layers

        per_h: dict[float, dict] = {}
        passed_count = 0

        for h in self._cv_h_points:
            logger.info(f"  Revalidating h={h} at p={p_layers}...")
            import time as _time

            t0 = _time.time()
            lattice = make_lattice(topology, n_qubits, J=1.0, h=h)
            runner = PipelineRunner(lattice=lattice, p_layers=p_layers)
            phase2 = runner.run_phase2(seed=42)
            e_vqe = phase2.energy
            solver = ClassicalSolver()
            H = HamiltonianBuilder().build(lattice)
            exact = solver.solve(H, lattice)
            de_gap = abs(e_vqe - exact.ground_energy) / abs(exact.gap)
            elapsed = _time.time() - t0
            passed = de_gap < 0.05
            if passed:
                passed_count += 1
            per_h[h] = {
                "de_gap": de_gap,
                "pass": passed,
                "e_vqe": e_vqe,
                "e_exact": exact.ground_energy,
                "gap": exact.gap,
                "elapsed_s": elapsed,
                "seed": 42,
                "n_iters": getattr(phase2, "n_iterations", None),
            }
            logger.info(
                f"    h={h}: ΔE/gap={de_gap:.4f} ({'PASS' if passed else 'FAIL'}) [{elapsed:.1f}s]"
            )

        summary = {
            "n_cv_points": len(self._cv_h_points),
            "n_passed": passed_count,
            "pass_rate": passed_count / len(self._cv_h_points),
            "pass": passed_count == len(self._cv_h_points),
        }
        return {"per_h": per_h, **summary}


if __name__ == "__main__":
    runner = Ext1bP1ValidationRunner()
    result = runner.run()
    sys.exit(0 if result == 0 else 1)
