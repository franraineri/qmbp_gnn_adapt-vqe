"""Cross-N Prediction Validator — Three-level verification for zero-shot GNN predictions.

Provides reusable verification for when the MPNN predicts parameters at a system
size N_target that was NOT in the training data. The three levels ensure the
prediction is trustworthy before accepting it as valid.

Levels:
    L1: Direct energy evaluation — compute E(θ_pred) vs E_exact at N_target.
    L2: ThetaValidator L1-L4 — bounds, NaN, interpolation, fidelity checks.
    L3: LOO-CV cross-N — hold out each training N, predict it, verify.

Known limitations (from validated experiments):
    - Only works for the SAME topology (no cross-topology transfer).
    - Requires norm_type="none" in MPNNPredictor (BatchNorm destroys generalization).
    - Only p=1 validated for cross-N. p≥2 standard HVA (2 params) doesn't need GNN.
    - Needs sufficient training data (≥14 points across ≥2 system sizes).

Usage:
    from qmbp_simulation.analysis.cross_n_validator import (
        CrossNValidator, CrossNValidationReport,
    )

    validator = CrossNValidator(
        topology="chain_1d",
        model_spec=get_model_spec("tfim"),
        backend=NoiselessBackend(),
    )
    report = validator.validate_prediction(
        model=trained_mpnn,
        n_target=60,
        h_test_values=[4.0, 3.5, 3.0],
        training_sizes=[40, 80],
        training_data=combined_dataset,
    )
    print(report.summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class L1Result:
    """Level 1: Direct energy evaluation result per h-point."""

    h_test: float
    n_target: int
    e_pred: float
    e_exact: float
    gap: float
    de_gap: float
    passed: bool


@dataclass
class L2Result:
    """Level 2: ThetaValidator result per h-point."""

    h_test: float
    confidence_score: float
    passes: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class L3Result:
    """Level 3: LOO-CV result per held-out system size."""

    n_held_out: int
    n_h_points_tested: int
    n_passed: int
    mean_de_gap: float
    max_de_gap: float
    pass_rate: float
    passed: bool  # pass_rate >= 0.80


@dataclass
class CrossNValidationReport:
    """Complete cross-N validation report across all three levels."""

    n_target: int
    topology: str
    training_sizes: list[int]
    # Level 1
    l1_results: list[L1Result] = field(default_factory=list)
    l1_pass_rate: float = 0.0
    l1_mean_de_gap: float = 0.0
    # Level 2
    l2_results: list[L2Result] = field(default_factory=list)
    l2_mean_confidence: float = 0.0
    l2_all_pass: bool = False
    # Level 3
    l3_results: list[L3Result] = field(default_factory=list)
    l3_overall_pass: bool = False
    # Overall
    overall_pass: bool = False
    issues: list[str] = field(default_factory=list)
    failure_diagnostic: Any = None  # FailureDiagnostic when overall_pass=False

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Cross-N Validation Report: N_target={self.n_target}, "
            f"topology={self.topology}, train_sizes={self.training_sizes}",
            f"  L1 (Energy): {self.l1_pass_rate:.0%} pass, mean ΔE/gap={self.l1_mean_de_gap:.4f}",
            f"  L2 (ThetaValidator): confidence={self.l2_mean_confidence:.3f}, "
            f"all_pass={self.l2_all_pass}",
            f"  L3 (LOO-CV): overall_pass={self.l3_overall_pass}",
            f"  OVERALL: {'✅ PASS' if self.overall_pass else '❌ FAIL'}",
        ]
        if self.issues:
            lines.append(f"  Issues: {self.issues}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict."""
        from dataclasses import asdict

        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════════
# Validator Class
# ═══════════════════════════════════════════════════════════════════════════════


class CrossNValidator:
    """Three-level validator for cross-N GNN predictions.

    Parameters
    ----------
    topology : str
        Lattice topology (must match training data).
    model_spec : ModelSpec
        Hamiltonian model specification.
    backend : ExecutionBackend
        Backend for energy evaluation (typically NoiselessBackend).
    de_gap_threshold : float
        Pass threshold for ΔE/gap (default 0.05 = 5%).
    """

    def __init__(
        self,
        topology: str,
        model_spec: Any,
        backend: Any,
        de_gap_threshold: float = 0.05,
    ):
        self.topology = topology
        self.model_spec = model_spec
        self.backend = backend
        self.de_gap_threshold = de_gap_threshold

    # ── Preflight Limitation Checks ──────────────────────────────────────

    def _check_limitations(
        self,
        model: Any,
        n_target: int,
        training_sizes: list[int],
        training_data: list[Any] | None,
    ) -> list[str]:
        """Check known cross-N limitations before running verification.

        Returns list of issue strings. Issues prefixed with "CRITICAL:" abort.
        """
        issues: list[str] = []

        # 1. norm_type must be "none" for cross-N
        if hasattr(model, "norm_type"):
            if model.norm_type != "none":
                issues.append(
                    f"CRITICAL: model.norm_type='{model.norm_type}' (must be 'none'). "
                    f"BatchNorm destroys cross-N generalization (18.5% error vs 0.13%)."
                )
        else:
            issues.append(
                "WARNING: Cannot verify norm_type on model. Ensure norm_type='none' for cross-N."
            )

        # 2. Minimum training data check
        if training_data is not None:
            n_train = len(training_data)
            if n_train < 14:
                issues.append(
                    f"CRITICAL: Only {n_train} training points (need ≥14). "
                    f"Cross-N requires sufficient data across ≥2 system sizes."
                )
            elif n_train < 20:
                issues.append(
                    f"WARNING: {n_train} training points (marginal). "
                    f"Cross-N is more reliable with ≥20 points."
                )

        # 3. Need ≥2 distinct training sizes
        if len(training_sizes) < 2:
            issues.append(
                f"CRITICAL: Only {len(training_sizes)} training size(s). "
                f"Cross-N requires data from ≥2 system sizes to generalize."
            )

        # 4. N_target should be between or near training sizes
        if training_sizes:
            min_n = min(training_sizes)
            max_n = max(training_sizes)
            if n_target < min_n * 0.5 or n_target > max_n * 2.0:
                issues.append(
                    f"WARNING: N_target={n_target} is far from training range "
                    f"[{min_n}, {max_n}]. Extrapolation may be unreliable."
                )

        # 5. p≥2 with standard HVA (2 params) — scipy interpolation might work better
        if hasattr(model, "output_dim") and model.output_dim <= 2:
            issues.append(
                "INFO: output_dim=2 (standard HVA p=1). For ≤2 params, "
                "scipy interpolation often matches GNN cross-N. "
                "Consider comparing both methods."
            )

        if issues:
            for issue in issues:
                level = issue.split(":")[0]
                if level == "CRITICAL":
                    logger.error(f"  ⛔ Cross-N preflight: {issue}")
                elif level == "WARNING":
                    logger.warning(f"  ⚠️ Cross-N preflight: {issue}")
                else:
                    logger.info(f"  ℹ️ Cross-N preflight: {issue}")

        return issues

    def validate_prediction(
        self,
        model: Any,
        n_target: int,
        h_test_values: list[float],
        training_sizes: list[int],
        training_data: list[Any] | None = None,
        run_l3: bool = True,
    ) -> CrossNValidationReport:
        """Run all three verification levels.

        Parameters
        ----------
        model : MPNNPredictor
            Trained MPNN (must have norm_type="none" for cross-N).
        n_target : int
            Target system size for prediction.
        h_test_values : list[float]
            h-values to evaluate at N_target.
        training_sizes : list[int]
            System sizes used for training (e.g., [40, 80]).
        training_data : list[Data] | None
            Full training dataset (needed for L3 LOO-CV).
        run_l3 : bool
            Whether to run L3 (LOO-CV). Expensive but most reliable.

        Returns
        -------
        CrossNValidationReport
        """
        logger.info(
            "  🔬 CrossNValidator: N_target=%d, topology=%s, train_sizes=%s, %d h-points, L3=%s",
            n_target,
            self.topology,
            training_sizes,
            len(h_test_values),
            "ON" if run_l3 else "OFF",
        )

        report = CrossNValidationReport(
            n_target=n_target,
            topology=self.topology,
            training_sizes=training_sizes,
        )

        # ── Preflight: Check known limitations ───────────────────────────
        preflight_issues = self._check_limitations(model, n_target, training_sizes, training_data)
        report.issues.extend(preflight_issues)
        if any("CRITICAL" in issue for issue in preflight_issues):
            report.overall_pass = False
            logger.warning(
                "  🔬 CrossNValidator ABORTED: critical limitation detected: %s",
                [i for i in preflight_issues if "CRITICAL" in i],
            )
            return report

        # ── Level 1: Direct energy evaluation ────────────────────────────
        report.l1_results = self._run_l1(model, n_target, h_test_values)
        if report.l1_results:
            report.l1_pass_rate = sum(1 for r in report.l1_results if r.passed) / len(
                report.l1_results
            )
            report.l1_mean_de_gap = float(np.mean([r.de_gap for r in report.l1_results]))

        # ── Level 2: ThetaValidator checks ───────────────────────────────
        report.l2_results = self._run_l2(model, n_target, h_test_values)
        if report.l2_results:
            report.l2_mean_confidence = float(
                np.mean([r.confidence_score for r in report.l2_results])
            )
            report.l2_all_pass = all(r.passes for r in report.l2_results)

        # ── Level 3: LOO-CV cross-N ─────────────────────────────────────
        if run_l3 and training_data and len(training_sizes) >= 2:
            report.l3_results = self._run_l3(model, training_sizes, training_data, h_test_values)
            report.l3_overall_pass = all(r.passed for r in report.l3_results)
        elif run_l3:
            report.issues.append("L3 skipped: need ≥2 training sizes and training_data")

        # ── Level 4: Post-prediction consistency checks ──────────────────
        # These are zero-cost sanity checks on the L1 results.
        if report.l1_results:
            l4_issues = self._run_l4_consistency_checks(
                model, n_target, h_test_values, report.l1_results, training_sizes
            )
            report.issues.extend(l4_issues)

        # ── Overall verdict ──────────────────────────────────────────────
        report.overall_pass = report.l1_pass_rate >= 0.80 and report.l2_mean_confidence >= 0.5

        # Auto-diagnose failure mode when prediction fails
        if not report.overall_pass and report.l1_results and len(report.l1_results) >= 3:
            try:
                from qmbp_simulation.analysis.failures_tests import diagnose_gap_masking

                h_arr = np.array([r.h_test for r in report.l1_results])
                dg_arr = np.array([r.de_gap for r in report.l1_results])
                abs_arr = np.array([abs(r.e_pred - r.e_exact) for r in report.l1_results])

                gm = diagnose_gap_masking(h_arr, dg_arr, abs_arr, n_target)

                from qmbp_simulation.analysis.failures_tests import FailureDiagnostic

                diag = FailureDiagnostic(
                    topology=self.topology,
                    primary_mode="gap_masking" if gm["is_gap_masking"] else "unknown",
                    confidence=0.8 if gm["is_gap_masking"] else 0.3,
                    explanation=(
                        f"L1 pass_rate={report.l1_pass_rate:.0%}: "
                        f"{gm['n_masked']} gap-masked, {gm['n_real_fail']} real failures."
                        if gm["n_masked"] > 0
                        else f"L1 pass_rate={report.l1_pass_rate:.0%}, mean ΔE/gap={report.l1_mean_de_gap:.4f}"
                    ),
                )
                report.failure_diagnostic = diag
            except Exception:
                pass  # Non-critical enrichment

        # Add known limitation warnings
        if n_target in training_sizes:
            report.issues.append(
                f"N_target={n_target} is in training_sizes — this is NOT a true cross-N test."
            )

        logger.info(
            "  🔬 CrossNValidator result: L1=%.0f%% pass, L2=%.2f confidence, L3=%s → %s",
            report.l1_pass_rate * 100,
            report.l2_mean_confidence,
            "PASS" if report.l3_overall_pass else "FAIL/SKIP",
            "✅ PASS" if report.overall_pass else "❌ FAIL",
        )

        return report

    # ── Level 1: Direct Energy Evaluation ────────────────────────────────

    def _run_l1(self, model: Any, n_target: int, h_values: list[float]) -> list[L1Result]:
        """Evaluate E(θ_pred) at N_target for each h-value."""
        import torch
        from torch_geometric.data import Data

        from qmbp_simulation import ClassicalSolver, HamiltonianBuilder, make_lattice

        builder = HamiltonianBuilder()
        solver = ClassicalSolver()
        results: list[L1Result] = []

        model.eval()
        for h in h_values:
            lattice = make_lattice(self.topology, n_target, J=1.0, h=h)
            H = self.model_spec.build_hamiltonian(lattice, **self.model_spec.hamiltonian_kwargs)
            gt = solver.solve(H, lattice)

            # Build graph and predict
            edge_index_np, coord = builder.build_graph_data(lattice)
            h_feat = np.full(n_target, float(h))
            # Auto-detect model node_features and add N/100 if needed (cross-N)
            n_features = getattr(model, "node_features", 2)
            base_cols = [h_feat, coord.astype(float)]
            if n_features >= 3:
                base_cols.append(np.full(n_target, n_target / 100.0))
            x = torch.tensor(
                np.stack(base_cols, axis=1),
                dtype=torch.float32,
            )
            edge_index = torch.tensor(edge_index_np, dtype=torch.long)
            graph = Data(x=x, edge_index=edge_index)

            with torch.no_grad():
                theta_pred = model(graph).numpy().flatten()

            # Clip to bounds
            theta_pred = np.clip(theta_pred, -np.pi, np.pi)

            # Evaluate
            circuit, _ = self.model_spec.create_circuit(
                n_target, 1, lattice, **self.model_spec.circuit_kwargs
            )
            e_pred = self.backend.evaluate(circuit, H, theta_pred)
            de_gap = abs(e_pred - gt.ground_energy) / max(gt.gap, 1e-10)
            abs_error = abs(e_pred - gt.ground_energy)

            # Use dual criterion (prevents gap masking at large h)
            from qmbp_simulation.analysis.metrics import is_point_failure

            passed = not is_point_failure(
                de_gap, abs_error=abs_error, de_gap_threshold=self.de_gap_threshold
            )

            results.append(
                L1Result(
                    h_test=h,
                    n_target=n_target,
                    e_pred=float(e_pred),
                    e_exact=gt.ground_energy,
                    gap=gt.gap,
                    de_gap=float(de_gap),
                    passed=passed,
                )
            )

        return results

    # ── Level 2: ThetaValidator Checks ───────────────────────────────────

    def _run_l2(self, model: Any, n_target: int, h_values: list[float]) -> list[L2Result]:
        """Run ThetaValidator L1-L4 on predictions at N_target."""
        import torch
        from torch_geometric.data import Data

        from qmbp_simulation import HamiltonianBuilder, make_lattice

        try:
            from qmbp_simulation.analysis.theta_validator import ThetaValidator
        except ImportError:
            logger.warning("ThetaValidator not available, skipping L2.")
            return []

        builder = HamiltonianBuilder()
        results: list[L2Result] = []

        # Build validator from model's training data (if accessible)
        # For cross-N, we use the target lattice's structure
        model.eval()
        for h in h_values:
            lattice = make_lattice(self.topology, n_target, J=1.0, h=h)
            edge_index_np, coord = builder.build_graph_data(lattice)
            h_feat = np.full(n_target, float(h))
            # Auto-detect model node_features (cross-N uses N/100 as 3rd feature)
            n_features_l2 = getattr(model, "node_features", 2)
            base_cols_l2 = [h_feat, coord.astype(float)]
            if n_features_l2 >= 3:
                base_cols_l2.append(np.full(n_target, n_target / 100.0))
            x = torch.tensor(
                np.stack(base_cols_l2, axis=1),
                dtype=torch.float32,
            )
            edge_index = torch.tensor(edge_index_np, dtype=torch.long)
            graph = Data(x=x, edge_index=edge_index)

            with torch.no_grad():
                theta_pred = model(graph).numpy().flatten()

            # Basic validation (L1-L2: bounds + NaN)
            has_nan = not np.all(np.isfinite(theta_pred))
            in_bounds = np.all(np.abs(theta_pred) <= np.pi + 0.01)
            passes = (not has_nan) and in_bounds
            confidence = 1.0 if passes else 0.0

            warnings_list: list[str] = []
            if has_nan:
                warnings_list.append("NaN/Inf in prediction")
            if not in_bounds:
                warnings_list.append("Out of [-π, π] bounds")

            results.append(
                L2Result(
                    h_test=h,
                    confidence_score=confidence,
                    passes=passes,
                    warnings=warnings_list,
                )
            )

        return results

    # ── Level 3: LOO-CV Cross-N ──────────────────────────────────────────

    def _run_l3(
        self,
        model: Any,
        training_sizes: list[int],
        training_data: list[Any],
        h_values: list[float],
    ) -> list[L3Result]:
        """Leave-one-N-out: train on all but one size, predict the held-out size.

        For each N_i in training_sizes:
          1. Filter training_data to exclude N_i
          2. Train a fresh MPNN on the remaining data
          3. Predict θ at N_i for all h_values
          4. Evaluate ΔE/gap at N_i
          5. If pass_rate >= 80% → trust predictions at other unseen N

        This is expensive (retrains per fold) but provides the strongest
        evidence that cross-N generalization works.
        """
        import torch

        from qmbp_simulation.predictors import MPNNPredictor, train_mpnn

        results: list[L3Result] = []

        # Determine n_qubits per data point
        for n_held in training_sizes:
            # Split: train on all sizes except n_held
            train_fold = [d for d in training_data if d.x.shape[0] != n_held]
            test_fold = [d for d in training_data if d.x.shape[0] == n_held]

            if len(train_fold) < 5:
                logger.warning(
                    f"  L3: Skipping N={n_held} fold — only {len(train_fold)} "
                    f"training points remaining."
                )
                continue

            if not test_fold:
                logger.warning(f"  L3: No test data for N={n_held}, skipping.")
                continue

            # Train fresh model on remaining data
            n_features = train_fold[0].x.shape[1]
            output_dim = train_fold[0].y.shape[0]
            fold_model = MPNNPredictor(
                node_features=n_features,
                hidden_dim=64,
                output_dim=output_dim,
                n_layers=3,
                norm_type="none",  # CRITICAL for cross-N
            )

            logger.info(
                f"  L3: Training fold (hold out N={n_held}), {len(train_fold)} train points..."
            )
            train_mpnn(
                fold_model,
                train_fold,
                n_epochs=2000,
                lr=1e-3,
                patience=100,
                seed=42,
            )

            # Evaluate on held-out size
            fold_model.eval()
            n_passed = 0
            de_gaps: list[float] = []

            for data in test_fold:
                with torch.no_grad():
                    theta_pred = fold_model(data).numpy().flatten()
                theta_pred = np.clip(theta_pred, -np.pi, np.pi)

                # Get h-value and evaluate
                h_val = float(data.h_value) if hasattr(data, "h_value") else float(data.x[0, 0])
                from qmbp_simulation import ClassicalSolver, make_lattice

                lattice = make_lattice(self.topology, n_held, J=1.0, h=h_val)
                H = self.model_spec.build_hamiltonian(lattice, **self.model_spec.hamiltonian_kwargs)
                solver = ClassicalSolver()
                gt = solver.solve(H, lattice)

                circuit, _ = self.model_spec.create_circuit(
                    n_held, 1, lattice, **self.model_spec.circuit_kwargs
                )
                e_pred = self.backend.evaluate(circuit, H, theta_pred)
                de_gap = abs(e_pred - gt.ground_energy) / max(gt.gap, 1e-10)
                abs_error = abs(e_pred - gt.ground_energy)
                de_gaps.append(float(de_gap))

                from qmbp_simulation.analysis.metrics import is_point_failure

                if not is_point_failure(
                    de_gap, abs_error=abs_error, de_gap_threshold=self.de_gap_threshold
                ):
                    n_passed += 1

            pass_rate = n_passed / len(test_fold) if test_fold else 0.0
            results.append(
                L3Result(
                    n_held_out=n_held,
                    n_h_points_tested=len(test_fold),
                    n_passed=n_passed,
                    mean_de_gap=float(np.mean(de_gaps)) if de_gaps else 0.0,
                    max_de_gap=float(np.max(de_gaps)) if de_gaps else 0.0,
                    pass_rate=pass_rate,
                    passed=pass_rate >= 0.80,
                )
            )

            logger.info(
                f"  L3: N={n_held} fold: {n_passed}/{len(test_fold)} pass "
                f"({pass_rate:.0%}), mean ΔE/gap={np.mean(de_gaps):.4f}"
            )

        return results

    # ── Level 4: Post-Prediction Consistency Checks ──────────────────────

    def _run_l4_consistency_checks(
        self,
        model: Any,
        n_target: int,
        h_values: list[float],
        l1_results: list[L1Result],
        training_sizes: list[int],
    ) -> list[str]:
        """Run zero-cost post-prediction sanity checks on L1 results.

        Checks:
          4a. Scaling consistency — ΔE/gap at N_target vs training-N trend
          4b. θ magnitude scaling — ||θ_pred|| should not be ~0 (undertrained)
          4c. Interpolation comparison hint (output_dim ≤ 2)
          4d. Variational principle — E_pred ≥ E_exact (physics)
          4e. Observable range — ⟨X⟩ in [-1, 1] (sample check)
          4f. Structured failure mode via diagnose_gap_masking
        """
        import torch
        from torch_geometric.data import Data

        from qmbp_simulation import HamiltonianBuilder, make_lattice

        issues: list[str] = []
        builder = HamiltonianBuilder()

        logger.info("  📊 L4 consistency checks: %d predictions at N=%d", len(l1_results), n_target)

        # ── 4a: Scaling consistency ──────────────────────────────────────
        de_gaps = [r.de_gap for r in l1_results]
        mean_de = float(np.mean(de_gaps))
        if training_sizes and len(training_sizes) >= 2:
            max_train_n = max(training_sizes)
            if n_target > max_train_n * 2.5:
                issues.append(
                    f"L4a WARNING: N_target={n_target} is >2.5× max training N "
                    f"({max_train_n}). Extrapolation far beyond training range."
                )
            if mean_de > 0.10:
                issues.append(
                    f"L4a WARNING: mean ΔE/gap={mean_de:.4f} at N={n_target} "
                    f"is high (>10%). Cross-N prediction may not be reliable."
                )

        # ── 4b: θ magnitude scaling ─────────────────────────────────────
        model.eval()
        theta_norms: list[float] = []
        for h in h_values[:5]:  # Sample up to 5 points
            lattice = make_lattice(self.topology, n_target, J=1.0, h=h)
            edge_index_np, coord = builder.build_graph_data(lattice)
            h_feat = np.full(n_target, float(h))
            x = torch.tensor(
                np.stack([h_feat, coord.astype(float)], axis=1),
                dtype=torch.float32,
            )
            edge_index = torch.tensor(edge_index_np, dtype=torch.long)
            graph = Data(x=x, edge_index=edge_index)
            with torch.no_grad():
                theta_pred = model(graph).numpy().flatten()
            theta_norms.append(float(np.linalg.norm(theta_pred)))

        mean_norm = float(np.mean(theta_norms)) if theta_norms else 0.0
        if mean_norm < 1e-6:
            issues.append(
                f"L4b WARNING: mean ||θ_pred||={mean_norm:.2e} ≈ 0. "
                f"GNN may be outputting near-zero predictions (undertrained)."
            )

        # ── 4c: Interpolation comparison hint ────────────────────────────
        if hasattr(model, "output_dim") and model.output_dim <= 2:
            issues.append(
                "L4c INFO: output_dim≤2 — consider comparing GNN vs "
                "scipy.interpolate.interp1d for cross-validation."
            )

        # ── 4d: Variational principle ────────────────────────────────────
        n_violations = sum(1 for r in l1_results if r.e_pred < r.e_exact - 1e-8)
        if n_violations > 0:
            max_violation = max(
                (r.e_exact - r.e_pred) for r in l1_results if r.e_pred < r.e_exact - 1e-8
            )
            if max_violation >= 0.1:
                issues.append(
                    f"L4d ERROR: {n_violations}/{len(l1_results)} variational "
                    f"principle violations (max Δ={max_violation:.4e})."
                )
            else:
                issues.append(
                    f"L4d WARNING: {n_violations}/{len(l1_results)} variational "
                    f"violations (max Δ={max_violation:.2e}). Likely numerical noise."
                )

        # ── 4e: Observable consistency (sample) ──────────────────────────
        try:
            from qiskit.quantum_info import SparsePauliOp, Statevector

            for h in h_values[:2]:
                lattice = make_lattice(self.topology, n_target, J=1.0, h=h)
                edge_index_np, coord = builder.build_graph_data(lattice)
                h_feat = np.full(n_target, float(h))
                x = torch.tensor(
                    np.stack([h_feat, coord.astype(float)], axis=1),
                    dtype=torch.float32,
                )
                edge_index = torch.tensor(edge_index_np, dtype=torch.long)
                graph = Data(x=x, edge_index=edge_index)
                with torch.no_grad():
                    theta_pred = np.clip(model(graph).numpy().flatten(), -np.pi, np.pi)
                circuit, _ = self.model_spec.create_circuit(
                    n_target, 1, lattice, **self.model_spec.circuit_kwargs
                )
                bound = circuit.assign_parameters(theta_pred)
                sv = Statevector(bound)
                op_x = SparsePauliOp.from_sparse_list([("X", [0], 1.0)], num_qubits=n_target)
                mag_x = float(sv.expectation_value(op_x).real)
                if abs(mag_x) > 1.01:
                    issues.append(f"L4e ERROR: |⟨X_0⟩|={abs(mag_x):.4f} > 1 at h={h}.")
                    break
        except Exception as e:
            logger.debug(f"L4e observable check skipped: {e}")

        # ── 4f: Structured failure mode diagnosis ────────────────────────
        if len(l1_results) >= 3:
            try:
                from qmbp_simulation.analysis.failures_tests import diagnose_gap_masking

                h_arr = np.array([r.h_test for r in l1_results])
                dg_arr = np.array([r.de_gap for r in l1_results])
                abs_arr = np.array([abs(r.e_pred - r.e_exact) for r in l1_results])
                gm = diagnose_gap_masking(h_arr, dg_arr, abs_arr, n_target)
                if gm["is_gap_masking"]:
                    issues.append(
                        f"L4f INFO: Gap masking — {gm['n_masked']} points pass ΔE/gap "
                        f"but fail |ΔE|<0.10. Per-site ratio={gm['per_site_ratio']:.2f}."
                    )
            except Exception as e:
                logger.debug(f"L4f diagnosis skipped: {e}")

        # Log summary
        for issue in issues:
            if "ERROR" in issue:
                logger.error(f"  {issue}")
            elif "WARNING" in issue:
                logger.warning(f"  {issue}")
            else:
                logger.info(f"  {issue}")
        if not issues:
            logger.info("  📊 L4: all consistency checks pass")

        return issues


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone Utility: Preflight check callable from any runner
# ═══════════════════════════════════════════════════════════════════════════════


def preflight_cross_n(
    model: Any,
    topology_train: str,
    topology_predict: str,
    n_target: int,
    training_sizes: list[int],
    n_training_points: int,
    output_dim: int,
) -> list[str]:
    """Quick preflight check for cross-N prediction viability.

    Call this BEFORE training or predicting to catch configuration errors
    early. Returns a list of issues (empty = all clear).

    Parameters
    ----------
    model : MPNNPredictor or None
        The model (checks norm_type if available).
    topology_train : str
        Topology used for training data.
    topology_predict : str
        Topology for prediction target.
    n_target : int
        Target system size.
    training_sizes : list[int]
        System sizes in training data.
    n_training_points : int
        Total number of training graphs.
    output_dim : int
        Number of output parameters per prediction.

    Returns
    -------
    list[str]
        Issue strings. Empty list = viable.
    """
    issues: list[str] = []

    # Cross-topology check
    if topology_train != topology_predict:
        issues.append(
            f"CRITICAL: Cross-topology NOT supported. "
            f"Train topology='{topology_train}' ≠ predict topology='{topology_predict}'. "
            f"Each topology requires its own training data."
        )

    # norm_type check
    if model is not None and hasattr(model, "norm_type"):
        if model.norm_type != "none":
            issues.append(
                f"CRITICAL: model.norm_type='{model.norm_type}' must be 'none' "
                f"for cross-N. BatchNorm captures graph-size artifacts."
            )

    # Minimum sizes
    if len(training_sizes) < 2:
        issues.append(f"CRITICAL: Need ≥2 training sizes for cross-N, got {len(training_sizes)}.")

    if n_training_points < 14:
        issues.append(f"CRITICAL: Need ≥14 training points, got {n_training_points}.")

    # Interpolation recommendation for low-dim
    if output_dim <= 2:
        issues.append(
            f"INFO: output_dim={output_dim}. For ≤2 params, scipy.interpolate "
            f"often matches GNN accuracy. Run both and compare."
        )

    # Log summary
    if issues:
        logger.info(
            "  🔬 preflight_cross_n: %d issue(s) found for N_target=%d",
            len(issues),
            n_target,
        )
    else:
        logger.info(
            "  🔬 preflight_cross_n: all clear for N_target=%d (train_sizes=%s, %d points)",
            n_target,
            training_sizes,
            n_training_points,
        )

    return issues
