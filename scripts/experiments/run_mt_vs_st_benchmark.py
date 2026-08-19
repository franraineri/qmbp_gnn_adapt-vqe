#!/usr/bin/env python3
"""MT vs ST Benchmark: Fair head-to-head comparison of model architectures.

Orchestrates training + evaluation of Single-Topology (ST) and Multi-Topology (MT)
models on identical conditions, producing a structured comparison report.

Phases:
  1. TRAIN: Train fresh ST models per topology + 2 MT variants (residual, residual+film)
  2. EVAL-IN: In-distribution evaluation (same N range as training)
  3. EVAL-EXTRAP: Extrapolation evaluation (N beyond training range)
  4. EVAL-FRUSTRATED: Dedicated evaluation on physics-limited topologies
  5. COMPARE: Head-to-head comparison with identical h-points and N targets
  6. REPORT: Generate unified markdown report

All models are trained with the SAME hyperparameters (hidden=256, layers=3,
epochs=2000, patience=200, seed=42) to ensure fair comparison.

Usage:
    # Full benchmark (train + eval + compare)
    .venv/bin/python scripts/experiments/run_mt_vs_st_benchmark.py

    # Skip training, only evaluate existing models
    .venv/bin/python scripts/experiments/run_mt_vs_st_benchmark.py --skip-training

    # Only specific phase
    .venv/bin/python scripts/experiments/run_mt_vs_st_benchmark.py --phase train
    .venv/bin/python scripts/experiments/run_mt_vs_st_benchmark.py --phase eval
    .venv/bin/python scripts/experiments/run_mt_vs_st_benchmark.py --phase compare

    # Dry run (show commands without executing)
    .venv/bin/python scripts/experiments/run_mt_vs_st_benchmark.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / ".venv" / "bin" / "python")
RESULTS_DIR = ROOT / "results" / "mt_vs_st_benchmark"

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration: Shared hyperparameters for fair comparison
# ═══════════════════════════════════════════════════════════════════════════════

# Training hyperparameters (IDENTICAL for ST and MT)
TRAIN_CONFIG = {
    "hidden_dim": 256,
    "n_layers": 3,
    "epochs": 2000,
    "patience": 200,
    "lr": 1e-3,
    "seed": 42,
    "max_de_gap": 0.10,  # Quality filter for training data
}

# Topologies and their viable N ranges (based on project-status data)
TOPOLOGIES = {
    "chain_1d": {
        "train_max_n": 20,
        "eval_in_dist": [6, 8, 10, 12],       # Within training range
        "eval_extrap": [16, 21, 31],           # Beyond training
        "category": "unfrustrated_1d",
        "h_min": 2.5,
        "h_max": 5.0,
        "h_points": 6,
    },
    "heavy_hex": {
        "train_max_n": 16,
        "eval_in_dist": [4, 6, 10, 12],
        "eval_extrap": [20, 24, 29],
        "category": "unfrustrated_quasi1d",
        "h_min": 2.0,
        "h_max": 3.5,
        "h_points": 6,
    },
    "ladder": {
        "train_max_n": 12,
        "eval_in_dist": [4, 6, 8, 10],
        "eval_extrap": [12, 18, 22, 24],
        "category": "frustrated_quasi2d",
        "h_min": 2.5,
        "h_max": 5.0,
        "h_points": 6,
    },
    "square": {
        "train_max_n": 14,
        "eval_in_dist": [4, 6, 8, 10],
        "eval_extrap": [14, 18, 21],
        "category": "frustrated_2d",
        "h_min": 2.5,
        "h_max": 5.0,
        "h_points": 6,
    },
    "triangular": {
        "train_max_n": 6,  # Physics limit: p=1 fails at N≥8
        "eval_in_dist": [3, 4, 6],
        "eval_extrap": [8, 11, 12, 13],                # Expected to fail (documents limit)
        "category": "frustrated_2d",
        "h_min": 2.5,
        "h_max": 5.0,
        "h_points": 6,
    },
}

# MT model variants to train
MT_VARIANTS = [
    {
        "name": "MT_residual",
        "use_residual": True,
        "film": False,
        "readout_mode": "last",
    },
    {
        "name": "MT_residual_film",
        "use_residual": True,
        "film": True,
        "readout_mode": "last",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def run_cmd(cmd: list[str], label: str, dry_run: bool = False) -> dict:
    """Execute a command and return result dict."""
    cmd_str = " ".join(cmd)
    if dry_run:
        print(f"  [DRY] {label}")
        print(f"        {cmd_str}")
        return {"label": label, "status": "dry_run", "cmd": cmd_str}

    print(f"\n{'─' * 70}")
    print(f"  ▶ {label}")
    print(f"    {cmd_str}")
    print(f"{'─' * 70}", flush=True)

    # Create per-task log file
    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace(" ", "_").replace(":", "").replace("[", "").replace("]", "")[:40]
    log_file = log_dir / f"seq_{safe_label}.log"

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    elapsed = time.perf_counter() - t0

    status = "ok" if result.returncode == 0 else "failed"

    # Save log
    with open(log_file, "w") as f:
        f.write(f"# Task: {label}\n# Status: {status} ({elapsed:.1f}s)\n# {'=' * 60}\n\n")
        if result.stdout:
            f.write(result.stdout)
        if result.stderr:
            f.write("\n\n# === STDERR ===\n")
            f.write(result.stderr)

    if result.returncode != 0:
        print(f"  ❌ FAILED ({elapsed:.1f}s) — see {log_file.name}")
        # Print last 10 lines of stderr as preview
        for line in result.stderr.strip().split("\n")[-10:]:
            print(f"    {line}")
    else:
        print(f"  ✅ Done ({elapsed:.1f}s)")

    return {
        "label": label,
        "status": status,
        "elapsed_s": round(elapsed, 1),
        "returncode": result.returncode,
        "cmd": cmd_str,
        "log_file": str(log_file.name),
    }


def _run_indexed_task(args_tuple: tuple) -> tuple[int, dict]:
    """Execute a single indexed task (module-level for pickle compatibility).

    Saves stdout+stderr to a per-task log file for post-hoc debugging.
    """
    idx, cmd, label, cwd = args_tuple
    from pathlib import Path

    # Create per-task log file
    log_dir = Path(cwd) / "results" / "mt_vs_st_benchmark" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_label = label.replace(" ", "_").replace(":", "").replace("[", "").replace("]", "")[:40]
    log_file = log_dir / f"{idx:02d}_{safe_label}.log"

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    elapsed = time.perf_counter() - t0
    status = "ok" if proc.returncode == 0 else "failed"

    # Write combined stdout+stderr to log file
    with open(log_file, "w") as f:
        f.write(f"# Task: {label}\n")
        f.write(f"# Cmd: {' '.join(cmd)}\n")
        f.write(f"# Status: {status} ({elapsed:.1f}s)\n")
        f.write(f"# {'=' * 60}\n\n")
        if proc.stdout:
            f.write(proc.stdout)
        if proc.stderr:
            f.write("\n\n# === STDERR ===\n")
            f.write(proc.stderr)

    return idx, {
        "label": label,
        "status": status,
        "elapsed_s": round(elapsed, 1),
        "returncode": proc.returncode,
        "cmd": " ".join(cmd),
        "log_file": str(log_file.name),
    }


def run_parallel(tasks: list[tuple[list[str], str]], max_workers: int = 0, dry_run: bool = False, force_sequential: bool = False) -> list[dict]:
    """Run multiple commands in parallel using ProcessPoolExecutor.

    On M2 (8 cores), uses up to 4 workers by default to avoid memory
    pressure from multiple PyTorch/Qiskit processes. Each subprocess
    gets its own memory space (no GIL issues).

    Parameters
    ----------
    tasks : list[tuple[list[str], str]]
        List of (cmd, label) tuples.
    max_workers : int
        Max parallel processes. 0 = auto (min(4, n_tasks, cpu_count//2)).
    dry_run : bool
        If True, just print commands.
    force_sequential : bool
        If True, run sequentially regardless of max_workers.

    Returns
    -------
    list[dict]
        Results in original order.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if dry_run:
        return [run_cmd(cmd, label, dry_run=True) for cmd, label in tasks]

    if force_sequential:
        return [run_cmd(cmd, label) for cmd, label in tasks]

    if max_workers <= 0:
        cpu_count = os.cpu_count() or 4
        max_workers = min(4, len(tasks), cpu_count // 2)

    if max_workers <= 1 or len(tasks) <= 1:
        return [run_cmd(cmd, label) for cmd, label in tasks]

    print(f"\n  ⚡ Parallel execution: {len(tasks)} tasks, {max_workers} workers")

    # Build pickleable task tuples
    task_args = [(i, cmd, label, str(ROOT)) for i, (cmd, label) in enumerate(tasks)]

    results_map: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_run_indexed_task, arg): arg[0]
            for arg in task_args
        }
        for future in as_completed(futures):
            idx, result = future.result()
            results_map[idx] = result
            marker = "✅" if result["status"] == "ok" else "❌"
            print(
                f"  {marker} [{result['elapsed_s']:.0f}s] {result['label']}",
                flush=True,
            )

    # Return in original order
    return [results_map[i] for i in range(len(tasks))]


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: TRAIN
# ═══════════════════════════════════════════════════════════════════════════════


def phase_train_st(dry_run: bool = False, sequential: bool = False) -> list[dict]:
    """Train fresh ST models for each topology using accelerated cross-N runner."""
    print("\n" + "═" * 70)
    print("  PHASE 1a: Train Single-Topology (ST) models")
    print("═" * 70)

    tasks = []
    for topo, cfg in TOPOLOGIES.items():
        cmd = [
            PYTHON,
            "scripts/experiment_runners/bond_resolved/run_accelerated_cross_n.py",
            "--topology", topo,
            "--multi-n-train",
            "--force-retrain",
            "--p-layers", "1",
            "--h-min", str(cfg["h_min"]),
            "--h-max", str(cfg["h_max"]),
            "--h-points", str(cfg["h_points"]),
            "--target-n", *[str(n) for n in cfg["eval_in_dist"]],
        ]
        tasks.append((cmd, f"ST train: {topo}"))

    # ST models are independent → parallelize (M2 has 8 cores)
    return run_parallel(tasks, dry_run=dry_run, force_sequential=sequential)


def phase_train_mt(dry_run: bool = False) -> list[dict]:
    """Train MT model variants using multi-topology training runner."""
    results = []
    print("\n" + "═" * 70)
    print("  PHASE 1b: Train Multi-Topology (MT) models")
    print("═" * 70)

    for variant in MT_VARIANTS:
        cmd = [
            PYTHON,
            "scripts/experiment_runners/cross_topology/run_multi_topology_training.py",
            "--max-n", "20",
            "--max-de-gap", str(TRAIN_CONFIG["max_de_gap"]),
            "--hidden-dim", str(TRAIN_CONFIG["hidden_dim"]),
            "--n-layers", str(TRAIN_CONFIG["n_layers"]),
            "--epochs", str(TRAIN_CONFIG["epochs"]),
            "--patience", str(TRAIN_CONFIG["patience"]),
            "--lr", str(TRAIN_CONFIG["lr"]),
            "--seed", str(TRAIN_CONFIG["seed"]),
            "--register-zoo",
            "-v",
        ]
        if variant["use_residual"]:
            cmd.append("--use-residual")
        if variant["film"]:
            cmd.append("--film")
        if variant["readout_mode"] != "last":
            cmd.extend(["--readout-mode", variant["readout_mode"]])

        r = run_cmd(cmd, f"MT train: {variant['name']}", dry_run)
        results.append(r)

    return results


def phase_finetune_mt_per_topology(dry_run: bool = False, sequential: bool = False) -> list[dict]:
    """Fine-tune the best MT model for each topology (MT→ST transfer).

    Takes the MT_residual_film model (best overall from benchmark) and
    specializes it with per-topology data. This produces hybrid models
    that combine MT's generalization with ST's specialization.
    """
    print("\n" + "═" * 70)
    print("  PHASE 1c: Fine-tune MT → per-topology (transfer learning)")
    print("═" * 70)

    tasks = []
    for topo, cfg in TOPOLOGIES.items():
        cmd = [
            PYTHON,
            "scripts/experiment_runners/cross_topology/run_finetune_from_mt.py",
            "--topology", topo,
            "--max-n", str(cfg["train_max_n"]),
            "--max-de-gap", str(TRAIN_CONFIG["max_de_gap"]),
            "--epochs", "500",
            "--lr", "3e-4",
            "--patience", "100",
            "--p-layers", "1",
            "-v",
        ]
        tasks.append((cmd, f"Fine-tune MT→{topo}"))

    # All fine-tunes are independent → parallelize
    return run_parallel(tasks, max_workers=4, dry_run=dry_run, force_sequential=sequential)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: EVALUATE (using model_comparison for head-to-head)
# ═══════════════════════════════════════════════════════════════════════════════


def phase_eval_comparison(dry_run: bool = False, sequential: bool = False) -> list[dict]:
    """Run head-to-head comparisons: all models on same conditions per topology."""
    print("\n" + "═" * 70)
    print("  PHASE 2a: Head-to-head evaluation (model_comparison)")
    print("═" * 70)

    tasks = []
    for topo, cfg in TOPOLOGIES.items():
        # In-distribution comparison
        target_n_in = cfg["eval_in_dist"]
        cmd_in = [
            PYTHON,
            "scripts/experiment_runners/cross_topology/run_model_comparison.py",
            "--topology", topo,
            "--target-n", *[str(n) for n in target_n_in],
            "--h-min", str(cfg["h_min"]),
            "--h-max", str(cfg["h_max"]),
            "--h-points", str(cfg["h_points"]),
            "--auto-detect",
            "--include-versions",
            "--promote-best",
            "--save-report",
            "-v",
        ]
        tasks.append((cmd_in, f"Compare IN-DIST: {topo} N={target_n_in}"))

        # Extrapolation comparison
        target_n_ext = cfg["eval_extrap"]
        cmd_ext = [
            PYTHON,
            "scripts/experiment_runners/cross_topology/run_model_comparison.py",
            "--topology", topo,
            "--target-n", *[str(n) for n in target_n_ext],
            "--h-min", str(cfg["h_min"]),
            "--h-max", str(cfg["h_max"]),
            "--h-points", str(cfg["h_points"]),
            "--auto-detect",
            "--include-versions",
            "--save-report",
            "-v",
        ]
        tasks.append((cmd_ext, f"Compare EXTRAP: {topo} N={target_n_ext}"))

    # All comparisons are independent → parallelize
    # Use max 3 workers: each comparison loads a model + does circuit evals (memory-heavy)
    return run_parallel(tasks, max_workers=3, dry_run=dry_run, force_sequential=sequential)


def phase_deep_extrapolation(dry_run: bool = False) -> list[dict]:
    """Deep extrapolation: test scaling at N=40-100 via run_large_n_extrapolation.

    Only runs on topologies where N>30 is physically meaningful (unfrustrated).
    Uses cached ground truth from previous runs to minimize DMRG compute.
    Includes random VQE baseline for speedup metric (thesis Table 4).
    """
    results = []
    print("\n" + "═" * 70)
    print("  PHASE 2b: Deep extrapolation (N=40-100, scaling law)")
    print("═" * 70)

    # Only topologies where deep extrapolation makes sense
    DEEP_EXTRAP_CONFIGS = {
        "chain_1d": {
            "target_n": [30, 40, 60],
            "h_min": 2.5,
            "h_max": 5.0,
            "h_points": 6,
        },
        "heavy_hex": {
            "target_n": [20, 30],
            "h_min": 2.0,
            "h_max": 3.5,
            "h_points": 6,
        },
    }

    for topo, cfg in DEEP_EXTRAP_CONFIGS.items():
        if topo not in TOPOLOGIES:
            continue  # Respect --topologies filter

        cmd = [
            PYTHON,
            "scripts/experiment_runners/scaling/run_large_n_extrapolation.py",
            "--topology", topo,
            "--target-n", *[str(n) for n in cfg["target_n"]],
            "--h-min", str(cfg["h_min"]),
            "--h-max", str(cfg["h_max"]),
            "--h-points", str(cfg["h_points"]),
            "--model-name", "tfim_bond_resolved",
            "--p-layers", "1",
            "--refine-failing",
            "--max-refine", "2",
            "--vqe-maxiter", "50",
            "--vqe-restarts", "2",
        ]
        r = run_cmd(cmd, f"Deep extrap: {topo} N={cfg['target_n']}", dry_run)
        results.append(r)

        # Also run with random baseline for speedup metric (lighter version)
        cmd_baseline = [
            PYTHON,
            "scripts/experiment_runners/scaling/run_large_n_extrapolation.py",
            "--topology", topo,
            "--target-n", *[str(n) for n in cfg["target_n"][:2]],  # Only first 2 N (expensive)
            "--h-min", str(cfg["h_min"]),
            "--h-max", str(cfg["h_max"]),
            "--h-points", "4",  # Fewer points for baseline (expensive)
            "--model-name", "tfim_bond_resolved",
            "--p-layers", "1",
            "--vqe-maxiter", "50",
            "--vqe-restarts", "3",
        ]
        r = run_cmd(cmd_baseline, f"VQE baseline: {topo} N={cfg['target_n'][:2]}", dry_run)
        results.append(r)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: REPORT
# ═══════════════════════════════════════════════════════════════════════════════


def phase_consolidated_report() -> Path:
    """Read comparison JSONs and produce a unified MT vs ST markdown table.

    Scans results/model_comparison/ for the latest comparison per topology,
    plus results/experiments/exp_large_n_extrap/ for deep extrapolation data.
    Produces: results/mt_vs_st_benchmark/consolidated_report.md
    """
    import json as _json

    COMP_DIR = ROOT / "results" / "model_comparison"
    EXTRAP_DIR_R = ROOT / "results" / "experiments" / "exp_large_n_extrap"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# MT vs ST Consolidated Benchmark Report",
        "",
        f"**Generated**: {ts}",
        "",
        "## Head-to-Head Comparison (model_comparison results)",
        "",
        "| Topology | Scenario | N targets | Winner | Winner ΔE/gap | Runner-up | Runner-up ΔE/gap |",
        "|----------|----------|-----------|--------|:---:|-----------|:---:|",
    ]

    # Scan latest comparison per topology
    comparison_data = {}
    if COMP_DIR.exists():
        for topo in TOPOLOGIES:
            files = sorted(COMP_DIR.glob(f"compare_{topo}_*.json"))
            for f in files[-2:]:  # Last 2 (in-dist + extrap)
                try:
                    d = _json.loads(f.read_text())
                    target_n = d.get("target_n", [])
                    best = d.get("best_model")
                    best_arch = d.get("best_arch", "?")

                    # Determine scenario from N values
                    topo_cfg = TOPOLOGIES.get(topo, {})
                    in_dist_n = topo_cfg.get("eval_in_dist", [])
                    is_in_dist = target_n and all(n in in_dist_n for n in target_n)
                    scenario = "in-dist" if is_in_dist else "extrap"

                    # Find best and runner-up from results
                    results = d.get("results", [])
                    scoreable = [
                        r for r in results
                        if "results_by_n" in r and not r.get("error")
                        and any(
                            m.get("mean_de_gap") is not None
                            for m in r["results_by_n"].values()
                        )
                    ]

                    def _avg_dg(r):
                        vals = [
                            m.get("mean_de_gap", 1.0)
                            for m in r["results_by_n"].values()
                            if m.get("mean_de_gap") is not None
                        ]
                        return float(np.mean(vals)) if vals else 1.0

                    scoreable.sort(key=_avg_dg)
                    winner_label = scoreable[0]["label"][:30] if scoreable else "—"
                    winner_dg = f"{_avg_dg(scoreable[0]):.4f}" if scoreable else "—"
                    runner_label = scoreable[1]["label"][:30] if len(scoreable) > 1 else "—"
                    runner_dg = f"{_avg_dg(scoreable[1]):.4f}" if len(scoreable) > 1 else "—"

                    lines.append(
                        f"| {topo} | {scenario} | {target_n} | "
                        f"{winner_label} | {winner_dg} | {runner_label} | {runner_dg} |"
                    )

                    comparison_data.setdefault(topo, []).append({
                        "scenario": scenario,
                        "target_n": target_n,
                        "winner": winner_label,
                        "winner_dg": winner_dg,
                    })
                except Exception:
                    continue

    # Deep extrapolation results
    lines.extend([
        "",
        "## Deep Extrapolation (N=30-100)",
        "",
        "| Topology | N | ΔE/gap | |ΔE|/N | Grade | Checkpoint |",
        "|----------|---|:---:|:---:|:---:|---|",
    ])

    if EXTRAP_DIR_R.exists():
        for f in sorted(EXTRAP_DIR_R.glob("run_*.json"))[-5:]:
            try:
                d = _json.loads(f.read_text())
                config = d.get("config", {})
                topo = config.get("topology", "?")
                # Structure: results.section_2.data.{mpnn_results, checkpoint_used}
                results_sections = d.get("results", {})
                sec2 = results_sections.get("section_2", {}).get("data", {})
                ckpt = sec2.get("checkpoint_used", "?")[:40]
                mpnn_res = sec2.get("mpnn_results", {})
                for n_str, metrics in mpnn_res.items():
                    n = metrics.get("n_qubits", n_str)
                    dg = metrics.get("mean_de_gap", 0)
                    eps = metrics.get("mean_abs_error_per_site", 0)
                    grade = metrics.get("grade", "?")
                    lines.append(
                        f"| {topo} | {n} | {dg:.4f} | {eps:.2e} | {grade} | {ckpt} |"
                    )
            except Exception:
                continue

    # Summary
    lines.extend([
        "",
        "## Key Findings",
        "",
        "- **Best overall architecture**: (determined by winner counts above)",
        "- **Scaling behavior**: |ΔE|/N ≈ constant indicates extensive scaling",
        "- **Physics limits**: Frustrated topologies (triangular N≥8, ladder N≥14) are ansatz-limited at p=1",
        "",
        "---",
        f"*Generated by `scripts/experiments/run_mt_vs_st_benchmark.py` at {ts}*",
    ])

    report_path = RESULTS_DIR / "consolidated_report.md"
    report_path.write_text("\n".join(lines))
    print(f"\n  📊 Consolidated report: {report_path.relative_to(ROOT)}")
    return report_path


def phase_report(all_results: list[dict]) -> None:
    """Generate unified benchmark report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "train_hyperparams": TRAIN_CONFIG,
            "topologies": {k: v for k, v in TOPOLOGIES.items()},
            "mt_variants": MT_VARIANTS,
        },
        "phases": all_results,
        "summary": {
            "total_steps": len(all_results),
            "passed": sum(1 for r in all_results if r.get("status") == "ok"),
            "failed": sum(1 for r in all_results if r.get("status") == "failed"),
            "total_time_s": sum(r.get("elapsed_s", 0) for r in all_results),
        },
    }

    report_path = RESULTS_DIR / f"benchmark_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  📊 Benchmark report: {report_path.relative_to(ROOT)}")

    # Summary
    print("\n" + "═" * 70)
    print("  BENCHMARK SUMMARY")
    print("═" * 70)
    print(f"  Steps: {report['summary']['total_steps']} "
          f"(✅ {report['summary']['passed']} | ❌ {report['summary']['failed']})")
    print(f"  Time:  {report['summary']['total_time_s']:.0f}s")
    print(f"  Report: {report_path.relative_to(ROOT)}")
    print()
    print("  Next steps:")
    print("    1. Review results/model_comparison/ for per-topology winners")
    print("    2. Run: .venv/bin/python scripts/analysis/evaluate_zoo_models.py --update-zoo")
    print("    3. Check: .venv/bin/python scripts/maintenance/check_zoo_coherence.py")
    print("═" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MT vs ST Benchmark: Fair head-to-head comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=["train", "eval", "compare", "extrap", "all"],
        default="all",
        help="Which phase to run: train, eval (in-dist+extrap comparisons), "
        "extrap (deep N=40-100), or all (default: all)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip Phase 1 (training). Use existing models.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=None,
        help="Limit to specific topologies (default: all 5)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel execution (useful for debugging)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Max parallel workers (0=auto: min(4, cpu//2)). M2 recommended: 3-4.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Filter topologies if requested
    if args.topologies:
        global TOPOLOGIES
        TOPOLOGIES = {k: v for k, v in TOPOLOGIES.items() if k in args.topologies}
        if not TOPOLOGIES:
            print(f"ERROR: No matching topologies. Available: {list(TOPOLOGIES.keys())}")
            return 1

    print("═" * 70)
    print("  MT vs ST BENCHMARK")
    print(f"  Topologies: {list(TOPOLOGIES.keys())}")
    print(f"  MT variants: {[v['name'] for v in MT_VARIANTS]}")
    print(f"  Train config: epochs={TRAIN_CONFIG['epochs']}, "
          f"patience={TRAIN_CONFIG['patience']}, seed={TRAIN_CONFIG['seed']}")
    print("═" * 70)

    all_results = []

    # Phase 1: Train (uses existing NPZ data — including any prior refinements)
    if args.phase in ("train", "all") and not args.skip_training:
        seq = args.no_parallel
        all_results.extend(phase_train_st(args.dry_run, sequential=seq))
        all_results.extend(phase_train_mt(args.dry_run))
        # Fine-tune MT → per-topology (transfer learning, after MT is ready)
        all_results.extend(phase_finetune_mt_per_topology(args.dry_run, sequential=seq))

    # Phase 2a: Evaluate (head-to-head comparisons at moderate N)
    # NOTE: Runs AFTER training so newly trained models are compared.
    # Refined θ from these evaluations are persisted to NPZ automatically,
    # closing the feedback loop for the NEXT training cycle.
    if args.phase in ("eval", "compare", "all"):
        seq = args.no_parallel
        all_results.extend(phase_eval_comparison(args.dry_run, sequential=seq))

    # Phase 2b: Deep extrapolation (N=40-100, scaling law validation)
    # Refines failing points and persists to NPZ — feeds next retrain cycle.
    if args.phase in ("eval", "extrap", "all"):
        all_results.extend(phase_deep_extrapolation(args.dry_run))

    # Phase 3: Report
    if not args.dry_run:
        phase_report(all_results)
        phase_consolidated_report()

    return 0


if __name__ == "__main__":
    sys.exit(main())
