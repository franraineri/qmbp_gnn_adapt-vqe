#!/usr/bin/env python3
"""Generate a consolidated summary of the full MT pipeline execution.

Reads results from all pipeline stages and produces a single comparative
report showing:
- Architecture ablation winner
- MT training convergence and quality
- Per-topology model comparison: MT vs single-topology vs fine-tuned
- Large-N extrapolation grades
- Uncertainty calibration status
- Multi-topology vs single-topology head-to-head analysis

Usage:
    .venv/bin/python scripts/analysis/pipeline_summary.py
    .venv/bin/python scripts/analysis/pipeline_summary.py --json
    .venv/bin/python scripts/analysis/pipeline_summary.py --save
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidated pipeline summary")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--save", action="store_true", help="Save to results/pipeline_summary.md")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading (graceful — never crashes on missing data)
# ═══════════════════════════════════════════════════════════════════════════════


def load_latest_ablation() -> dict | None:
    """Load the most recent ablation result."""
    ablation_dir = ROOT / "results" / "arch_ablation"
    if not ablation_dir.exists():
        return None
    files = sorted(ablation_dir.glob("ablation_*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text())
    except (json.JSONDecodeError, OSError):
        return None


def load_model_comparisons() -> list[dict]:
    """Load all model comparison results."""
    comp_dir = ROOT / "results" / "model_comparison"
    if not comp_dir.exists():
        return []
    results = []
    for f in sorted(comp_dir.glob("compare_*.json")):
        try:
            results.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return results


def load_extrapolation_grades() -> dict[str, dict]:
    """Load extrapolation results keyed by topology with N→grade mapping."""
    extrap_dir = ROOT / "data" / "large_n_extrapolation"
    if not extrap_dir.exists():
        return {}

    results: dict[str, dict] = {}
    for f in sorted(extrap_dir.glob("*.npz")):
        try:
            parts = f.stem.rsplit("_N", 1)
            if len(parts) != 2:
                continue
            topo = parts[0]
            n_str = parts[1].split("_")[0]
            data = np.load(f, allow_pickle=True)
            de_gaps = data["de_gaps"] if "de_gaps" in data else None
            if de_gaps is None or len(de_gaps) == 0:
                continue
            n_pts = len(de_gaps)
            pass_rate = float(np.mean(de_gaps < 0.05))
            mean_dg = float(np.mean(de_gaps))
            p90_dg = float(np.percentile(de_gaps, 90))

            from qmbp_simulation.analysis.constants import compute_quality_score, grade_from_score
            score = compute_quality_score(mean_dg, p90_dg, None, n_pts)
            grade = grade_from_score(score)

            results.setdefault(topo, {})[int(n_str)] = {
                "n_points": n_pts,
                "pass_rate": pass_rate,
                "mean_de_gap": mean_dg,
                "grade": grade,
            }
        except Exception:
            continue
    return results


def load_dashboard() -> dict | None:
    """Load the model quality dashboard."""
    path = ROOT / "data" / "model_quality_dashboard.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def load_zoo_models() -> list[dict]:
    """Load all zoo entries with their metadata."""
    try:
        from qmbp_simulation.predictors.model_zoo import _load_manifest
        entries = _load_manifest()
        return [
            {
                "checkpoint": e.checkpoint_file,
                "topology": e.topology,
                "n_qubits": e.n_qubits,
                "pass_rate": e.pass_rate,
                "n_training_points": e.n_training_points,
                "is_multi_topology": e.is_multi_topology,
                "is_multi_n": e.is_multi_n,
                "is_evaluated": e.is_evaluated,
            }
            for e in entries
        ]
    except Exception:
        return []


def load_training_curves() -> list[dict]:
    """Summarize training curve files."""
    curve_dir = ROOT / "results" / "training_curves"
    if not curve_dir.exists():
        return []
    curves = []
    for f in sorted(curve_dir.glob("*.npz"))[-10:]:
        try:
            data = np.load(f)
            mse = data["mse_history"]
            val_mse = data.get("val_mse_history", np.array([]))
            curves.append({
                "file": f.name,
                "epochs": len(mse),
                "final_mse": float(mse[-1]) if len(mse) > 0 else None,
                "initial_mse": float(mse[0]) if len(mse) > 0 else None,
                "final_val_mse": float(val_mse[-1]) if len(val_mse) > 0 else None,
            })
        except Exception:
            continue
    return curves


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis: Multi-Topology vs Single-Topology Comparison
# ═══════════════════════════════════════════════════════════════════════════════


def analyze_mt_vs_single(zoo_models: list[dict], comparisons: list[dict]) -> dict:
    """Compare multi-topology model performance against per-topology models.

    Uses TWO data sources:
    1. Model comparison JSONs (real ΔE/gap evaluation on extrapolation grids)
    2. Zoo pass_rate (evaluated on training data — less reliable for cross-N)

    Priority: comparison data > zoo pass_rate.

    Returns
    -------
    dict with:
        per_topology: {topo: {mt_de_gap, single_de_gap, mt_pass, single_pass, winner, delta}}
        summary: {mt_wins, single_wins, ties, mt_advantage_mean}
    """
    per_topo_results: dict[str, dict] = {}

    # Source 1: Model comparison JSONs (gold standard — real extrapolation eval)
    for comp in comparisons:
        topo = comp.get("topology", "?")
        results = comp.get("results", [])

        mt_de_gaps = []
        single_de_gaps = []
        mt_pass_rates = []
        single_pass_rates = []

        for r in results:
            if r.get("error"):
                continue
            source = r.get("source", "")
            results_by_n = r.get("results_by_n", {})
            if not results_by_n:
                continue

            # Collect mean_de_gap and pass_rate across N values
            de_gaps = [v.get("mean_de_gap", 1.0) for v in results_by_n.values() if isinstance(v, dict)]
            pass_rates = [v.get("pass_rate_dual", 0) for v in results_by_n.values() if isinstance(v, dict)]
            avg_dg = float(np.mean(de_gaps)) if de_gaps else 1.0
            avg_pr = float(np.mean(pass_rates)) if pass_rates else 0.0

            if "multi_topology" in source or "multi-topo" in source.lower():
                mt_de_gaps.append(avg_dg)
                mt_pass_rates.append(avg_pr)
            elif "per_topology" in source or "per-topo" in source.lower():
                single_de_gaps.append(avg_dg)
                single_pass_rates.append(avg_pr)
            elif "orphan" in source:
                # Orphan multitopo checkpoints → treat as MT variant
                mt_de_gaps.append(avg_dg)
                mt_pass_rates.append(avg_pr)

        if mt_de_gaps or single_de_gaps:
            best_mt_dg = min(mt_de_gaps) if mt_de_gaps else float("inf")
            best_single_dg = min(single_de_gaps) if single_de_gaps else float("inf")
            best_mt_pr = max(mt_pass_rates) if mt_pass_rates else 0.0
            best_single_pr = max(single_pass_rates) if single_pass_rates else 0.0

            # Winner determined by ΔE/gap (lower is better)
            if best_mt_dg < best_single_dg * 0.9:
                winner = "MT"
            elif best_single_dg < best_mt_dg * 0.9:
                winner = "single"
            else:
                winner = "tie"

            per_topo_results[topo] = {
                "mt_de_gap": best_mt_dg,
                "single_de_gap": best_single_dg,
                "mt_pass_rate": best_mt_pr,
                "single_pass_rate": best_single_pr,
                "winner": winner,
                "delta_de_gap": best_single_dg - best_mt_dg,  # Positive = MT better
                "source": "comparison",
            }

    # Source 2: Zoo pass_rate for topologies without comparison data
    mt_zoo = [m for m in zoo_models if m["is_multi_topology"] and m["is_evaluated"]]
    best_mt_zoo_pr = max((m["pass_rate"] for m in mt_zoo), default=0.0)

    for m in zoo_models:
        topo = m["topology"]
        if not m["is_multi_topology"] and m["is_evaluated"] and topo not in per_topo_results:
            per_topo_results[topo] = {
                "mt_de_gap": None,
                "single_de_gap": None,
                "mt_pass_rate": best_mt_zoo_pr,
                "single_pass_rate": m["pass_rate"],
                "winner": "MT" if best_mt_zoo_pr > m["pass_rate"] + 0.10 else (
                    "single" if m["pass_rate"] > best_mt_zoo_pr + 0.10 else "tie"
                ),
                "delta_de_gap": None,
                "source": "zoo_only",
            }

    # Summary
    mt_wins = sum(1 for v in per_topo_results.values() if v["winner"] == "MT")
    single_wins = sum(1 for v in per_topo_results.values() if v["winner"] == "single")
    ties = sum(1 for v in per_topo_results.values() if v["winner"] == "tie")
    deltas = [v["delta_de_gap"] for v in per_topo_results.values() if v.get("delta_de_gap") is not None]
    mt_advantage_de_gap = float(np.mean(deltas)) if deltas else 0.0

    return {
        "per_topology": per_topo_results,
        "summary": {
            "mt_wins": mt_wins,
            "single_wins": single_wins,
            "ties": ties,
            "mt_advantage_de_gap": mt_advantage_de_gap,
            "n_topologies_compared": len(per_topo_results),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Report Formatting
# ═══════════════════════════════════════════════════════════════════════════════


def format_text_report(data: dict) -> str:
    """Format a human-readable text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("  PIPELINE EXECUTION SUMMARY")
    lines.append(f"  Generated: {datetime.now(timezone.utc).isoformat()[:19]}Z")
    lines.append("=" * 80)

    # 1. Ablation
    abl = data.get("ablation")
    lines.append("\n┌─ ARCHITECTURE ABLATION ─────────────────────────────────────────────┐")
    if abl:
        lines.append(f"│ Topology: {abl.get('topology', '?')} | "
                     f"Epochs: {abl.get('max_epochs', '?')} | "
                     f"Graphs: {abl.get('n_training_graphs', '?')} | "
                     f"Filter: max_de_gap={abl.get('max_de_gap', '?')}")
        lines.append(f"│ Best: {abl.get('best_variant', '?')}")
        lines.append(f"│ {'Variant':<20} {'val_MSE':>10} {'MSE':>10} {'Epochs':>7} {'Stop':>15}")
        for r in abl.get("results", []):
            if "name" not in r:
                continue
            flag = " ★" if r["name"] == abl.get("best_variant") else ""
            val = f"{r['val_mse']:.2e}" if r.get("val_mse") else "N/A"
            mse = f"{r['final_mse']:.2e}" if r.get("final_mse") else "(ref)"
            ep = str(r.get("n_epochs", "—"))
            stop = r.get("stop_reason", "—")[:15]
            lines.append(f"│ {r['name']:<20} {val:>10} {mse:>10} {ep:>7} {stop:>15}{flag}")
    else:
        lines.append("│ No ablation results found")
    lines.append("└────────────────────────────────────────────────────────────────────┘")

    # 2. Training convergence
    curves = data.get("training_curves", [])
    lines.append("\n┌─ TRAINING CONVERGENCE ─────────────────────────────────────────────┐")
    if curves:
        for c in curves[-5:]:
            reduction = ""
            if c.get("initial_mse") and c.get("final_mse") and c["initial_mse"] > 0:
                red = (1 - c["final_mse"] / c["initial_mse"]) * 100
                reduction = f" (↓{red:.0f}%)"
            val_str = f" val={c['final_val_mse']:.2e}" if c.get("final_val_mse") else ""
            lines.append(
                f"│ {c['file'][:42]:<42} {c['epochs']:>5}ep MSE={c['final_mse']:.2e}{val_str}{reduction}"
            )
    else:
        lines.append("│ No training curves found")
    lines.append("└────────────────────────────────────────────────────────────────────┘")

    # 3. MT vs Single-Topology (THE KEY COMPARISON)
    mt_analysis = data.get("mt_vs_single", {})
    per_topo = mt_analysis.get("per_topology", {})
    summary = mt_analysis.get("summary", {})
    lines.append("\n┌─ MULTI-TOPOLOGY vs SINGLE-TOPOLOGY (head-to-head) ─────────────────┐")
    if per_topo:
        lines.append(f"│ {'Topology':<12} {'MT ΔE/gap':>10} {'Single ΔE/gap':>14} {'Winner':>8} {'Source':>10}")
        lines.append(f"│ {'─'*12} {'─'*10} {'─'*14} {'─'*8} {'─'*10}")
        for topo, v in sorted(per_topo.items()):
            w_icon = "🟢" if v["winner"] == "MT" else ("🔴" if v["winner"] == "single" else "⚪")
            mt_dg = f"{v['mt_de_gap']:.4f}" if v.get("mt_de_gap") is not None else "N/A"
            s_dg = f"{v['single_de_gap']:.4f}" if v.get("single_de_gap") is not None else "N/A"
            src = v.get("source", "?")
            lines.append(
                f"│ {topo:<12} {mt_dg:>10} {s_dg:>14} "
                f"{w_icon} {v['winner']:<5} {src:>10}"
            )
        lines.append(f"│")
        lines.append(
            f"│ Score: MT wins {summary.get('mt_wins', 0)} | "
            f"Single wins {summary.get('single_wins', 0)} | "
            f"Ties {summary.get('ties', 0)}"
        )
        if summary.get("mt_advantage_de_gap") is not None:
            adv = summary["mt_advantage_de_gap"]
            lines.append(f"│ MT ΔE/gap advantage: {adv:+.4f} (positive = MT has lower error)")
        verdict = (
            "MT model generalizes BETTER across topologies"
            if summary.get("mt_wins", 0) > summary.get("single_wins", 0)
            else "Single-topology models are MORE ACCURATE per topology"
            if summary.get("single_wins", 0) > summary.get("mt_wins", 0)
            else "No clear winner — consider hybrid deployment (MT + fine-tune)"
        )
        lines.append(f"│ Verdict: {verdict}")
    else:
        lines.append("│ No comparison data available (run model_comparison first)")
    lines.append("└────────────────────────────────────────────────────────────────────┘")

    # 4. Zoo status
    zoo = data.get("zoo_models", [])
    lines.append("\n┌─ ZOO MODEL STATUS ────────────────────────────────────────────────┐")
    if zoo:
        by_topo: dict[str, list] = {}
        for m in zoo:
            by_topo.setdefault(m["topology"], []).append(m)
        for topo, models in sorted(by_topo.items()):
            evaluated = [m for m in models if m["is_evaluated"]]
            best_pr = max((m["pass_rate"] for m in models), default=0)
            total_pts = sum(m["n_training_points"] for m in models)
            lines.append(
                f"│ {topo:<15} {len(models)} models | "
                f"{len(evaluated)} eval'd | best={best_pr:>4.0%} | {total_pts:>5} pts"
            )
    else:
        lines.append("│ No zoo models found")
    lines.append("└────────────────────────────────────────────────────────────────────┘")

    # 5. Large-N extrapolation
    extrap = data.get("extrapolation", {})
    lines.append("\n┌─ LARGE-N EXTRAPOLATION ────────────────────────────────────────────┐")
    if extrap:
        lines.append(f"│ {'Topology':<12} {'N':>4} {'Pts':>4} {'Pass%':>6} {'ΔE/gap':>8} {'Grade':>6}")
        for topo, n_data in sorted(extrap.items()):
            for n, info in sorted(n_data.items()):
                lines.append(
                    f"│ {topo:<12} {n:>4} {info['n_points']:>4} "
                    f"{info['pass_rate']:>5.0%} {info['mean_de_gap']:>8.4f} "
                    f"{info['grade']:>6}"
                )
    else:
        lines.append("│ No extrapolation data found (run step 5)")
    lines.append("└────────────────────────────────────────────────────────────────────┘")

    # 6. Model comparisons detail
    comps = data.get("comparisons", [])
    lines.append("\n┌─ MODEL COMPARISONS (latest per topology) ─────────────────────────┐")
    if comps:
        # Show latest per topology
        latest_by_topo: dict[str, dict] = {}
        for comp in comps:
            topo = comp.get("topology", "?")
            latest_by_topo[topo] = comp
        for topo, comp in sorted(latest_by_topo.items()):
            best = comp.get("best_model", "?")
            arch = comp.get("best_arch", "?")
            n_models = comp.get("n_models", 0)
            lines.append(f"│ {topo:<12} | {n_models} models → Winner: {best} (arch={arch})")
    else:
        lines.append("│ No comparison results found")
    lines.append("└────────────────────────────────────────────────────────────────────┘")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    args = parse_args()

    # Gather all data
    zoo_models = load_zoo_models()
    comparisons = load_model_comparisons()

    data = {
        "ablation": load_latest_ablation(),
        "comparisons": comparisons,
        "extrapolation": load_extrapolation_grades(),
        "training_curves": load_training_curves(),
        "zoo_models": zoo_models,
        "dashboard": load_dashboard(),
        "mt_vs_single": analyze_mt_vs_single(zoo_models, comparisons),
    }

    if args.json:
        print(json.dumps(data, indent=2, default=str))
    else:
        report = format_text_report(data)
        print(report)

        if args.save:
            output = ROOT / "results" / "pipeline_summary.md"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(f"```\n{report}\n```\n")
            print(f"\nSaved to: {output.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
