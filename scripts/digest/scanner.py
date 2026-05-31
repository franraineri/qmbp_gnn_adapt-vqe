"""Result scanner — discovers and parses all result files into typed objects.

Scans results/experiments/ and results/thesis/ uniformly, classifying
each JSON file by its kind (noiseless pipeline, noisy/ZNE, or experiment).

No external dependencies — stdlib only.

Fixes applied (2026-05-30):
- Bug #1: Now reports worst-case ΔE/gap across ALL phase4_results points
  (previously only read phase4[0], hiding failures at other h_test values).
- Bug #4: Now parses ALL pipeline_run_*.json files per folder
  (previously only took the latest, missing multi-seed runs in same dir).
- Bug #2: Logs a warning instead of silently defaulting p_layers to 2.
- Skips hidden directories (starting with '.') in recursive scan.
- Handles UnicodeDecodeError in JSON loading.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from scripts.digest.models import (
    EXPERIMENT_CRITERIA,
    REJECTION_IS_FINDING,
    ExperimentResult,
    NoiselessResult,
    NoisyResult,
)

logger = logging.getLogger(__name__)

# Topologies we recognize from folder names
_KNOWN_TOPOLOGIES = ("chain_1d", "heavy_hex", "ladder", "triangular", "kagome", "linnear")
# "linnear" is a typo in the actual folder name (variants_N6_N10_1D_linnear)

# Folders to skip during recursive scanning
_SKIP_FOLDERS = frozenset({"checkpoints", "__pycache__", ".DS_Store"})


class ResultScanner:
    """Scans all result directories and classifies files by kind."""

    def __init__(self, results_root: Path = Path("results")) -> None:
        self.root = results_root

    def scan_all(
        self,
    ) -> tuple[list[NoiselessResult], list[NoisyResult], list[ExperimentResult]]:
        """Scan all result areas and return typed results."""
        noiseless: list[NoiselessResult] = []
        noisy: list[NoisyResult] = []
        experiments: list[ExperimentResult] = []

        # Scan results/experiments/
        exp_root = self.root / "experiments"
        if exp_root.exists():
            exp_dirs = [
                d
                for d in sorted(exp_root.iterdir())
                if d.is_dir() and d.name.startswith("exp_") and d.name != "exp_noisy_variants"
            ]
            logger.info("Scanning %d experiment directories...", len(exp_dirs))
            for exp_dir in exp_dirs:
                result = self._parse_experiment_dir(exp_dir)
                if result:
                    experiments.append(result)
            logger.info("  → %d experiments parsed", len(experiments))

        # Scan results/thesis/
        thesis_root = self.root / "thesis"
        if thesis_root.exists():
            folders = [
                f
                for f in sorted(thesis_root.iterdir())
                if f.is_dir() and not f.name.startswith(".")
            ]
            logger.info("Scanning %d thesis folders...", len(folders))
            for folder in folders:
                context = _infer_topology_from_name(folder.name)
                self._scan_folder_recursive(folder, noiseless, noisy, parent_context=context)
            logger.info("  → %d noiseless, %d noisy results", len(noiseless), len(noisy))

        return noiseless, noisy, experiments

    def scan_folder(
        self, folder_name: str
    ) -> tuple[list[NoiselessResult], list[NoisyResult], list[ExperimentResult]]:
        """Scan a specific folder by name (searches experiments/ then thesis/)."""
        noiseless: list[NoiselessResult] = []
        noisy: list[NoisyResult] = []
        experiments: list[ExperimentResult] = []

        # Try experiments/
        exp_path = self.root / "experiments" / folder_name
        if exp_path.exists() and exp_path.is_dir():
            result = self._parse_experiment_dir(exp_path)
            if result:
                experiments.append(result)
            return noiseless, noisy, experiments

        # Try thesis/ exact match
        thesis_path = self.root / "thesis" / folder_name
        if thesis_path.exists() and thesis_path.is_dir():
            context = _infer_topology_from_name(folder_name)
            self._scan_folder_recursive(thesis_path, noiseless, noisy, parent_context=context)
            return noiseless, noisy, experiments

        # Try thesis/ substring match
        thesis_root = self.root / "thesis"
        if thesis_root.exists():
            for folder in sorted(thesis_root.iterdir()):
                if folder.is_dir() and folder_name.lower() in folder.name.lower():
                    context = _infer_topology_from_name(folder.name)
                    self._scan_folder_recursive(folder, noiseless, noisy, parent_context=context)

        return noiseless, noisy, experiments

    # ── Recursive folder scanning ────────────────────────────────────────

    def _scan_folder_recursive(
        self,
        root: Path,
        noiseless: list[NoiselessResult],
        noisy: list[NoisyResult],
        *,
        parent_context: str = "",
    ) -> None:
        """Recursively scan a directory for pipeline and noisy results.

        Parameters
        ----------
        root : Path
            Directory to scan.
        noiseless : list
            Accumulator for noiseless results (mutated in place).
        noisy : list
            Accumulator for noisy results (mutated in place).
        parent_context : str
            Topology inferred from parent folder name (used as fallback).

        Fix #4: Now parses ALL pipeline_run files (not just the latest).
        """
        # Direct files in this folder — parse ALL (Fix #4)
        for pf in sorted(root.glob("pipeline_run_*.json")):
            result = self._parse_pipeline_file(pf, root.name)
            if result:
                _apply_topology_fallback(result, parent_context)
                noiseless.append(result)

        for nf in sorted(root.glob("noisy_*.json")):
            result = self._parse_noisy_file(nf, root.name)
            if result:
                _apply_topology_fallback(result, parent_context)
                noisy.append(result)

        # Recurse into subfolders
        for subfolder in sorted(root.iterdir()):
            if not subfolder.is_dir():
                continue
            # Skip hidden dirs and known non-result dirs
            if subfolder.name.startswith(".") or subfolder.name in _SKIP_FOLDERS:
                continue

            # Parse ALL pipeline_run files in subfolder (Fix #4)
            for pf in sorted(subfolder.glob("pipeline_run_*.json")):
                result = self._parse_pipeline_file(pf, subfolder.name)
                if result:
                    _apply_topology_fallback(result, parent_context)
                    noiseless.append(result)

            for nf in sorted(subfolder.glob("noisy_*.json")):
                result = self._parse_noisy_file(nf, subfolder.name)
                if result:
                    _apply_topology_fallback(result, parent_context)
                    noisy.append(result)

    # ── File parsers ─────────────────────────────────────────────────────

    def _parse_pipeline_file(self, path: Path, folder: str) -> NoiselessResult | None:
        """Parse a pipeline_run_*.json into a NoiselessResult.

        Fix #1: Reports worst-case ΔE/gap across ALL phase4_results entries.
        Previously only read phase4[0], which hid failures at other h_test values.
        """
        data = _load_json(path)
        if not data:
            return None

        config = data.get("config", {})
        mpnn_cfg = config.get("mpnn", {})
        system = data.get("system", {})
        diagnostics = data.get("diagnostics", {})
        phase4 = data.get("phase4_results", [])

        # Phase 4 primary metric — use worst-case across all test points (Fix #1)
        delta_e = None
        phase_label = ""
        phase_correct = None
        mag_x_error = None
        corr_zz_error = None

        if phase4:
            # Find the worst (highest) ΔE/gap across all test points
            worst_idx = 0
            worst_de = phase4[0].get("delta_e_over_gap")
            for i, entry in enumerate(phase4):
                de = entry.get("delta_e_over_gap")
                if de is not None and (worst_de is None or de > worst_de):
                    worst_de = de
                    worst_idx = i

            # Report the worst-case point as the primary metric
            p4 = phase4[worst_idx]
            delta_e = p4.get("delta_e_over_gap")
            phase_label = p4.get("phase_label", "")
            checklist = p4.get("metrics_checklist", {})
            phase_correct = checklist.get("correct_phase")
            mag_x_error = p4.get("mag_x_error")
            corr_zz_error = p4.get("corr_zz_error")

        phase2_diag = diagnostics.get("phase2", {})
        phase3_diag = diagnostics.get("phase3", {})

        # Extract p_layers with warning (Fix #2)
        p_layers = _extract_p_layers(config, system, path)

        return NoiselessResult(
            source_file=str(path),
            folder=folder,
            n_qubits=_get_int(config, "n_qubits") or _get_int(system, "n_qubits"),
            p_layers=p_layers,
            topology=config.get("topology") or system.get("topology", ""),
            n_restarts=config.get("n_restarts", 5),
            seed=config.get("seed"),
            h_values=config.get("h_values", []),
            h_test=config.get("h_test", []),
            hidden_dim=mpnn_cfg.get("hidden_dim", 128),
            n_epochs=mpnn_cfg.get("n_epochs", 6000),
            patience=mpnn_cfg.get("patience", 500),
            delta_e_over_gap=delta_e,
            phase_label=phase_label,
            phase_correct=phase_correct,
            mag_x_error=mag_x_error,
            corr_zz_error=corr_zz_error,
            convergence_rate=phase2_diag.get("convergence_rate"),
            theta_smoothness=phase2_diag.get("theta_smoothness"),
            worst_convergence_h=phase2_diag.get("worst_convergence_h"),
            generalization_gap=phase3_diag.get("generalization_gap"),
            theta_zz_mse=phase3_diag.get("theta_zz_mse"),
            elapsed_s=data.get("elapsed_s", 0),
            variant_id=folder,
        )

    def _parse_noisy_file(self, path: Path, folder: str) -> NoisyResult | None:
        """Parse a noisy_*.json into a NoisyResult."""
        data = _load_json(path)
        if not data:
            return None

        config = data.get("config", {})
        system = data.get("system", {})
        summary = data.get("summary", {})
        per_h = data.get("results_per_h", [])

        # Extract p_layers with warning (Fix #2)
        p_layers = _extract_p_layers(config, system, path)

        return NoisyResult(
            source_file=str(path),
            folder=folder,
            n_qubits=_get_int(config, "n_qubits") or _get_int(system, "n_qubits"),
            p_layers=p_layers,
            topology=config.get("topology") or system.get("topology", ""),
            seed=config.get("seed", 42),
            n_layouts=config.get("n_layouts", 3),
            shots=config.get("shots", 16384),
            h_values=config.get("h_values", []),
            mean_r2=summary.get("mean_r2", 0),
            mean_gain_pct=summary.get("mean_gain_pct", 0),
            n_mitigated_wins=summary.get("n_mitigated_wins", 0),
            n_total=summary.get("n_total", 0),
            success_criteria_met=summary.get("success_criteria_met", False),
            mean_de_noiseless=summary.get("mean_de_noiseless", 0),
            mean_de_noisy_raw=summary.get("mean_de_noisy_raw", 0),
            mean_de_zne=summary.get("mean_de_zne", 0),
            per_h_r2=[p.get("r_squared", 0) for p in per_h],
            per_h_gain=[p.get("gain_pct", 0) for p in per_h],
            elapsed_s=summary.get("elapsed_s", 0),
            variant_id=folder,
        )

    def _parse_experiment_dir(self, exp_dir: Path) -> ExperimentResult | None:
        """Parse an experiment directory (exp_<id>/) into ExperimentResult."""
        run_files = sorted(exp_dir.glob("run_*.json"), reverse=True)
        if not run_files:
            return None

        data = _load_json(run_files[0])
        if not data:
            return None

        config = data.get("config", {})
        analysis = data.get("analysis", {})
        summary = analysis.get("summary", {})
        system = config.get("system", {})

        if not summary or "error" in summary:
            return None

        exp_id = config.get("experiment_id", exp_dir.name.replace("exp_", "").upper())

        # Compute verdict
        criteria = EXPERIMENT_CRITERIA.get(exp_id, {})
        metric_name = criteria.get("metric", "mean_de_gap")
        threshold = criteria.get("threshold", 0.05)
        criteria_desc = criteria.get("desc", "ΔE/gap < 5%")

        if metric_name == "mean_de_gap":
            value = summary.get("mean_de_gap", float("inf"))
            passed = value < threshold
        else:
            value = summary.get("pass_rate", 0.0)
            passed = value >= threshold

        if passed:
            verdict = "confirmed"
        else:
            verdict = "rejected" if exp_id in REJECTION_IS_FINDING else "failed"

        # Experiment-specific extras
        extras: dict[str, Any] = {}
        if exp_id == "A3":
            extras["scaling_law_fit"] = analysis.get("scaling_law_fit", {})
        elif exp_id == "G1":
            extras["data_efficiency"] = analysis.get("data_efficiency", {})

        # Extract p_layers (experiments use system dict)
        p_layers_val = system.get("p_layers") or config.get("p_layers")
        p_layers = int(p_layers_val) if p_layers_val is not None else 2

        return ExperimentResult(
            source_file=str(run_files[0]),
            folder=exp_dir.name,
            experiment_id=exp_id,
            category=config.get("category", exp_id[0] if exp_id else ""),
            hypothesis=config.get("hypothesis", ""),
            description=config.get("description", ""),
            n_qubits=system.get("n_qubits", 0),
            p_layers=p_layers,
            topology=system.get("topology", ""),
            h_values=system.get("h_values", []),
            seeds=config.get("seeds", []),
            verdict=verdict,
            criteria=criteria_desc,
            mean_de_gap=summary.get("mean_de_gap"),
            std_de_gap=summary.get("std_de_gap"),
            pass_rate=summary.get("pass_rate"),
            n_seeds=analysis.get("n_seeds", 0),
            total_time_s=summary.get("total_time_s", 0),
            extras=extras,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════


def _load_json(path: Path) -> dict[str, Any] | None:
    """Safely load a JSON file, returning None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        logger.debug("Failed to load %s: %s", path, e)
        return None


def _get_int(d: dict, key: str) -> int:
    """Get an integer from a dict, returning 0 for None/missing."""
    val = d.get(key)
    return int(val) if val is not None else 0


def _extract_p_layers(config: dict, system: dict, path: Path) -> int:
    """Extract p_layers from config/system dicts.

    Fix #2: Logs a warning instead of silently defaulting to 2.
    Still returns 2 as default (for backward compat with NoiselessResult dataclass),
    but the warning makes it visible in --verbose mode.
    """
    p = config.get("p_layers") or system.get("p_layers")
    if p is not None:
        return int(p)
    logger.warning("p_layers not found in %s; defaulting to 2", path.name)
    return 2


def _infer_topology_from_name(name: str) -> str:
    """Infer topology from a folder name (e.g., 'variants_N10_ladder' → 'ladder')."""
    name_lower = name.lower()
    for topo in _KNOWN_TOPOLOGIES:
        if topo in name_lower:
            # Normalize the typo
            return "chain_1d" if topo == "linnear" else topo
    return ""


def _apply_topology_fallback(result: NoiselessResult | NoisyResult, context: str) -> None:
    """Set topology from parent context if the result file didn't specify one."""
    if not result.topology and context:
        result.topology = context
