"""Persistence of hardware execution results.

Saves all artifacts from a hardware run into a timestamped directory
using result_io utilities and StructuredLogger for execution logs.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qmbp_simulation.utils.helpers import json_serialize

if TYPE_CHECKING:
    from qmbp_simulation.execution.hardware.config import HardwareConfig, HardwareRunResult
    from qmbp_simulation.framework.logging import StructuredLogger


def _collect_metadata(seed: int | None = None) -> dict[str, Any]:
    """Collect run metadata with lazy import to avoid circular dependency."""
    from qmbp_simulation.framework.result_io import collect_run_metadata

    return collect_run_metadata(seed=seed)


def _write_json(data: Any, path: Path) -> None:
    """Write data to JSON with numpy/Path/datetime-safe serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=json_serialize)


def save_run(
    result: HardwareRunResult,
    config: HardwareConfig,
    logger: StructuredLogger,
    calibration_info: dict[str, Any],
    options_dict: dict[str, Any],
    execution_mode_name: str,
    raw_per_layout: list[dict],
    zne_data: dict[str, Any],
    input_params: Any | None = None,
) -> Path:
    """Persist all artifacts from a single hardware execution.

    Creates ``{output_dir}/run_YYYYMMDD_HHMMSS/`` with config.json, provenance.json,
    raw_results.json, zne_analysis.json, summary.json, input_params.json, and
    execution_log.json.

    Parameters
    ----------
    input_params : array-like | None
        The input parameters used for this execution. Saved for full
        reproducibility — allows re-running the exact same point.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(asdict(config), run_dir / "config.json")
    _write_json(
        {
            "execution_timestamp": datetime.now(UTC).isoformat(),
            "backend_name": config.backend_name,
            **_collect_metadata(config.layout_seed),
            "execution_mode": execution_mode_name,
            "job_ids": result.job_ids,
            "layouts_used": result.layouts_used,
            "ces_values": result.ces_values,
            "calibration_error_rates": calibration_info,
            "options_snapshot": options_dict,
            "n_qubits": config.n_qubits,
            "layout_selection_seed": config.layout_seed,
        },
        run_dir / "provenance.json",
    )
    _write_json(
        {"n_layouts": len(raw_per_layout), "per_layout": raw_per_layout},
        run_dir / "raw_results.json",
    )
    _write_json(zne_data, run_dir / "zne_analysis.json")
    _write_json(
        {
            "h_test": result.h_value,
            "e_exact": result.e_exact,
            "e_zne": result.e_zne,
            "gap": result.gap,
            "delta_e_gap": result.delta_e_gap,
            "phase_label": result.phase_label,
            "expected_label": result.expected_label,
            "mag_x_mean": result.mag_x_mean,
            "corr_zz_mean": result.corr_zz_mean,
            "sigma": result.sigma,
            "total_shots_consumed": result.total_shots,
            "zne_r2": result.zne_r2,
            "zne_gain": result.zne_gain,
            "spsa_applied": result.spsa_applied,
            "is_partial": result.is_partial,
            "verdict": result.verdict,
        },
        run_dir / "summary.json",
    )

    # Save input parameters for exact reproducibility
    if input_params is not None:
        import numpy as np

        _write_json(
            {"params": np.asarray(input_params).tolist(), "n_params": len(input_params)},
            run_dir / "input_params.json",
        )

    logger.log("run_saved", data={"run_dir": str(run_dir)})
    logger.save(run_dir / "execution_log.json")
    return run_dir


def save_partial_before_error(
    partial_results: list[dict],
    logger: StructuredLogger,
    config: HardwareConfig,
    error_msg: str,
) -> Path:
    """Save partial results before propagating an exception.

    Creates ``{output_dir}/run_YYYYMMDD_HHMMSS_PARTIAL/`` with partial_results.json,
    config.json, and execution_log.json. Never lose data on failure.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_dir) / f"run_{ts}_PARTIAL"
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        {"partial_results": partial_results, "error": error_msg}, run_dir / "partial_results.json"
    )
    _write_json(asdict(config), run_dir / "config.json")
    logger.log("execution_abort", data={"error": error_msg, "run_dir": str(run_dir)})
    logger.save(run_dir / "execution_log.json")
    return run_dir


def save_sweep_summary(
    results: list[HardwareRunResult],
    config: HardwareConfig,
    logger: StructuredLogger,
) -> Path:
    """Save consolidated sweep summary with per-h-point verdicts and overall pass rate.

    Writes ``{output_dir}/sweep_summary.json``.
    """
    per_h = [
        {
            "h_value": r.h_value,
            "e_exact": r.e_exact,
            "e_zne": r.e_zne,
            "delta_e_gap": r.delta_e_gap,
            "phase_label": r.phase_label,
            "expected_label": r.expected_label,
            "verdict": r.verdict,
            "is_partial": r.is_partial,
        }
        for r in results
    ]
    n_pass = sum(1 for r in results if r.verdict == "PASS")
    summary = {
        "n_points": len(results),
        "n_pass": n_pass,
        "pass_rate": n_pass / len(results) if results else 0.0,
        "per_h_point": per_h,
    }
    out_path = Path(config.output_dir) / "sweep_summary.json"
    _write_json(summary, out_path)
    logger.log(
        "sweep_summary_saved", data={"path": str(out_path), "pass_rate": summary["pass_rate"]}
    )
    return out_path
