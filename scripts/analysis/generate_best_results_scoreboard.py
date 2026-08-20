#!/usr/bin/env python3
"""Generate Best Results Scoreboard — best-ever ΔE/gap per (topology × N).

Parses all evaluation reports in results/extrapolation_evals/ and for each
(topology, N) combination, finds the best result achieved at h≈2.5
(the hardest h-value near h_critical where the gap is smallest).

Output: results/best_results_scoreboard.md

Integration:
    - Called by post_experiment_sync() as Step 3b
    - Can be run standalone: .venv/bin/python scripts/analysis/generate_best_results_scoreboard.py

Usage:
    .venv/bin/python scripts/analysis/generate_best_results_scoreboard.py
    .venv/bin/python scripts/analysis/generate_best_results_scoreboard.py --json
    .venv/bin/python scripts/analysis/generate_best_results_scoreboard.py --target-h 3.0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

EVAL_DIR = ROOT / "results" / "extrapolation_evals"
COMPARISON_DIR = ROOT / "results" / "model_comparison"
OUTPUT_PATH = ROOT / "results" / "best_results_scoreboard.md"
OUTPUT_JSON_PATH = ROOT / "results" / "best_results_scoreboard.json"

TARGET_H: float = 2.5
H_TOLERANCE: float = 0.15  # Accept h within ±0.15 of target


@dataclass
class PerHResult:
    """A single per-h measurement extracted from an eval report."""

    h: float
    e_pred: float
    e_exact: float
    abs_error: float
    abs_error_per_site: float
    gap: float
    de_gap: float


@dataclass
class EvalEntry:
    """One (topology, N) result from a single eval report."""

    topology: str
    n_qubits: int
    checkpoint: str
    is_mt: bool
    date: str  # ISO timestamp from report
    h_used: float  # actual h closest to target
    result: PerHResult
    report_file: str  # relative path to source report


@dataclass
class BestResult:
    """The best-ever result for a (topology, N) pair."""

    topology: str
    n_qubits: int
    best_de_gap: float
    best_abs_error: float
    best_abs_error_per_site: float
    gap_at_best: float
    h_used: float
    grade: str
    score: float
    checkpoint: str
    model_type: str  # "ST" or "MT"
    date: str
    report_file: str
    run_json: str = ""  # Path to corresponding JSON envelope (traceability)
    n_reports_scanned: int = 0
    # Additional context
    e_pred: float = 0.0
    e_exact: float = 0.0


def parse_eval_report(report_path: Path, target_h: float = TARGET_H) -> list[EvalEntry]:
    """Parse a single eval report markdown and extract per-N results at h≈target.

    Returns one EvalEntry per N section that contains a row with h close to target_h.
    """
    text = report_path.read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")

    entries: list[EvalEntry] = []

    # Extract metadata from header
    checkpoint = "unknown"
    date_str = ""
    is_mt = False
    topology = ""

    # Infer topology from directory name: {topo}_p{N}/
    parent_dir = report_path.parent.name  # e.g. "chain_1d_p1"
    topo_match = re.match(r"(.+)_p\d+$", parent_dir)
    if topo_match:
        topology = topo_match.group(1)

    for line in lines[:20]:  # Header is in first 20 lines
        if line.startswith("**Model**:"):
            checkpoint = line.split(":", 1)[1].strip()
        elif line.startswith("**Date**:"):
            date_str = line.split(":", 1)[1].strip()
        elif line.startswith("**Multi-topology**:"):
            mt_val = line.split(":", 1)[1].strip().lower()
            is_mt = mt_val in ("yes", "true")
        elif "_MT_" in report_path.name:
            is_mt = True

    # Parse per-N sections
    current_n: int | None = None
    in_table = False

    for line in lines:
        # Detect N section: "## N = 60 (119 params)"
        n_match = re.match(r"^## N\s*=\s*(\d+)", line)
        if n_match:
            current_n = int(n_match.group(1))
            in_table = False
            continue

        # Detect table header
        if current_n is not None and "| h |" in line and "E_pred" in line:
            in_table = True
            continue

        # Skip separator
        if in_table and line.startswith("|---"):
            continue

        # Parse table row
        if in_table and current_n is not None and line.startswith("|"):
            parts = [p.strip() for p in line.split("|")[1:-1]]  # strip empty first/last
            if len(parts) < 7:
                continue
            try:
                h_val = float(parts[0])
            except (ValueError, IndexError):
                continue

            # Check if this h is close to target
            if abs(h_val - target_h) > H_TOLERANCE:
                continue

            try:
                e_pred = float(parts[1])
                e_exact = float(parts[2])
                abs_error = float(parts[3])
                abs_error_per_site = float(parts[4])
                gap = float(parts[5])
                de_gap = float(parts[6])
            except (ValueError, IndexError):
                continue

            # ── Physical validity checks ─────────────────────────────────
            # Reject entries with non-physical or corrupt values
            # (aligned with ExperimentMetrics.validate() from framework.metrics)
            if de_gap < 0 or gap < 0 or abs_error < 0:
                continue  # Non-physical: negative error or gap
            if de_gap > 1e6 or abs_error > 1e6:
                continue  # Corrupt data (overflow or parse error)
            if abs(e_exact) < 1e-10 and abs(e_pred) < 1e-10:
                continue  # Zero energies = likely parse error
            if e_pred > 0 and e_exact < -1.0:
                continue  # Energy sign mismatch (same check as ExperimentMetrics)
            # Verify consistency: |ΔE| ≈ |e_pred - e_exact|
            recomputed_abs_error = abs(e_pred - e_exact)
            if abs_error > 0 and abs(recomputed_abs_error - abs_error) / max(abs_error, 1e-8) > 0.1:
                # More than 10% discrepancy — likely stale e_exact in report
                # Use recomputed value (e_pred and e_exact are authoritative)
                abs_error = recomputed_abs_error
                abs_error_per_site = abs_error / max(current_n, 1)
                if gap > 1e-10:
                    de_gap = abs_error / gap

            result = PerHResult(
                h=h_val,
                e_pred=e_pred,
                e_exact=e_exact,
                abs_error=abs_error,
                abs_error_per_site=abs_error_per_site,
                gap=gap,
                de_gap=de_gap,
            )

            rel_path = str(report_path.relative_to(ROOT))
            entries.append(
                EvalEntry(
                    topology=topology,
                    n_qubits=current_n,
                    checkpoint=checkpoint,
                    is_mt=is_mt,
                    date=date_str,
                    h_used=h_val,
                    result=result,
                    report_file=rel_path,
                )
            )

        # End of table (empty line or next section)
        if in_table and (line.strip() == "" or line.startswith("#")):
            in_table = False

    return entries


def scan_all_reports(target_h: float = TARGET_H) -> list[EvalEntry]:
    """Scan all eval reports from extrapolation_evals/ AND model_comparison/.

    Both directories use the same markdown format (generated by
    generate_evaluation_report from analysis.evaluation_report).
    """
    all_entries: list[EvalEntry] = []

    for scan_dir in (EVAL_DIR, COMPARISON_DIR):
        if not scan_dir.exists():
            continue

        for topo_dir in sorted(scan_dir.iterdir()):
            if not topo_dir.is_dir() or topo_dir.name.startswith("_"):
                continue

            for report_file in sorted(topo_dir.glob("eval_*.md")):
                try:
                    entries = parse_eval_report(report_file, target_h=target_h)
                    all_entries.extend(entries)
                except Exception as e:
                    print(f"  ⚠️ Error parsing {report_file.name}: {e}", file=sys.stderr)

    return all_entries


def _find_run_json(report_file: str) -> str:
    """Find the JSON run envelope corresponding to a markdown eval report.

    Naming conventions:
    - model_comparison: compare_{topo}_{timestamp}.json in same parent dir
    - extrapolation_evals: no direct JSON sibling, look in results/experiments/
    - experiments dir: run_{timestamp}.json under matching topology

    Returns relative path to JSON if found, else empty string.
    """
    report_path = ROOT / report_file
    parent = report_path.parent  # e.g. results/model_comparison/chain_1d_p1/

    # Extract timestamp from filename: eval_{topo}[_MT]_{YYYYMMDD_HHMMSS}.md
    stem = report_path.stem  # e.g. "eval_chain_1d_20260819_141529"
    ts_match = re.search(r"(\d{8}_\d{6})", stem)
    if not ts_match:
        return ""
    timestamp = ts_match.group(1)

    # Extract topology
    topo_match = re.match(r"(.+)_p\d+$", parent.name)
    topology = topo_match.group(1) if topo_match else ""

    # Strategy 1: JSON sibling in model_comparison/ directory
    #   results/model_comparison/compare_{topo}_{timestamp}.json
    if "model_comparison" in report_file:
        comparison_dir = parent.parent  # results/model_comparison/
        candidate = comparison_dir / f"compare_{topology}_{timestamp}.json"
        if candidate.exists():
            try:
                return str(candidate.relative_to(ROOT))
            except ValueError:
                return str(candidate)

    # Strategy 2: results/experiments/ directory
    #   results/experiments/exp_model_comparison/tfim_bond_resolved/{topo}/run_{timestamp}.json
    exp_dir = ROOT / "results" / "experiments"
    for pattern_dir in [
        exp_dir / "exp_model_comparison" / "tfim_bond_resolved" / topology,
        exp_dir / "exp_bond_resolved_scaling",
        exp_dir / f"exp_large_n_extrapolation",
    ]:
        if pattern_dir.exists():
            candidate = pattern_dir / f"run_{timestamp}.json"
            if candidate.exists():
                try:
                    return str(candidate.relative_to(ROOT))
                except ValueError:
                    return str(candidate)

    # Strategy 3: Timestamp-based fuzzy search (±2 minutes tolerance)
    # The eval report and JSON may have slightly different timestamps
    # due to generation delay
    ts_base = timestamp[:8]  # YYYYMMDD
    for pattern_dir in [
        ROOT / "results" / "model_comparison",
        exp_dir / "exp_model_comparison" / "tfim_bond_resolved" / topology,
    ]:
        if not pattern_dir.exists():
            continue
        for json_file in pattern_dir.glob(f"*{ts_base}*.json"):
            json_ts_match = re.search(r"(\d{8}_\d{6})", json_file.name)
            if json_ts_match:
                json_ts = json_ts_match.group(1)
                # Within 5 minutes of each other
                try:
                    t1 = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                    t2 = datetime.strptime(json_ts, "%Y%m%d_%H%M%S")
                    if abs((t1 - t2).total_seconds()) < 300:
                        try:
                            return str(json_file.relative_to(ROOT))
                        except ValueError:
                            return str(json_file)
                except ValueError:
                    continue

    return ""


def compute_best_per_topology_n(
    entries: list[EvalEntry],
) -> dict[str, dict[int, BestResult]]:
    """For each (topology, N), find the entry with the lowest ΔE/gap.

    Returns nested dict: {topology: {n_qubits: BestResult}}.
    """
    from qmbp_simulation.analysis.constants import compute_quality_score, grade_from_score

    # Group by (topology, N)
    grouped: dict[tuple[str, int], list[EvalEntry]] = {}
    for entry in entries:
        key = (entry.topology, entry.n_qubits)
        grouped.setdefault(key, []).append(entry)

    # Deduplicate: same (checkpoint, N, h, de_gap) from different directories
    # Keep the one with the lower ΔE/gap (they should be identical, but prefer
    # the entry from extrapolation_evals as canonical source if tied).
    for key in grouped:
        seen: dict[tuple[str, int, float], EvalEntry] = {}
        deduped = []
        for entry in grouped[key]:
            dedup_key = (entry.checkpoint, entry.n_qubits, round(entry.result.de_gap, 6))
            if dedup_key not in seen:
                seen[dedup_key] = entry
                deduped.append(entry)
            else:
                # Keep the one with lower de_gap (prefer extrapolation_evals source)
                existing = seen[dedup_key]
                if entry.result.de_gap < existing.result.de_gap:
                    deduped.remove(existing)
                    deduped.append(entry)
                    seen[dedup_key] = entry
        grouped[key] = deduped

    results: dict[str, dict[int, BestResult]] = {}

    for (topo, n), group in sorted(grouped.items()):
        # Find entry with lowest ΔE/gap
        best_entry = min(group, key=lambda e: e.result.de_gap)
        r = best_entry.result

        # Grade based on ΔE/gap thresholds directly (more intuitive for
        # single-point best-ever tracking than the composite quality score
        # which penalizes low n_points via confidence).
        # Thresholds aligned with DE_GAP_THRESHOLD=0.05 dual criterion:
        #   A: ΔE/gap < 0.03 (well below threshold)
        #   B: ΔE/gap < 0.05 (passes dual criterion)
        #   C: ΔE/gap < 0.10 (near-pass, within 2× threshold)
        #   D: ΔE/gap < 0.50 (moderate error)
        #   F: ΔE/gap ≥ 0.50
        if r.de_gap < 0.03:
            grade = "A"
        elif r.de_gap < 0.05:
            grade = "B"
        elif r.de_gap < 0.10:
            grade = "C"
        elif r.de_gap < 0.50:
            grade = "D"
        else:
            grade = "F"

        # Also compute the composite score for reference
        score = compute_quality_score(
            mean_de_gap=r.de_gap,
            p90_de_gap=r.de_gap,
            mean_abs_error_per_site=r.abs_error_per_site,
            n_points=8,  # treat as full-confidence for scoreboard display
        )

        model_type = "MT" if best_entry.is_mt else "ST"

        # Find corresponding JSON for traceability
        run_json = _find_run_json(best_entry.report_file)

        best = BestResult(
            topology=topo,
            n_qubits=n,
            best_de_gap=r.de_gap,
            best_abs_error=r.abs_error,
            best_abs_error_per_site=r.abs_error_per_site,
            gap_at_best=r.gap,
            h_used=r.h,
            grade=grade,
            score=score,
            checkpoint=best_entry.checkpoint,
            model_type=model_type,
            date=best_entry.date,
            report_file=best_entry.report_file,
            run_json=run_json,
            n_reports_scanned=len(group),
            e_pred=r.e_pred,
            e_exact=r.e_exact,
        )

        results.setdefault(topo, {})[n] = best

    return results


def format_markdown(
    best_by_topo: dict[str, dict[int, BestResult]],
    target_h: float,
    n_reports_total: int,
) -> str:
    """Format the scoreboard as markdown."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Best Results Scoreboard",
        "",
        f"**Updated**: {now}",
        f"**Reference h-value**: {target_h:.2f} "
        "(hardest region near h_critical; actual h used noted per entry)",
        f"**Reports scanned**: {n_reports_total}",
        f"**Criterion**: Best ΔE/gap achieved at h≈{target_h:.1f} per (topology × N)",
        "",
        "> This report shows the **best single-point result ever achieved** at h≈"
        f"{target_h:.1f} for each",
        "> (topology, N) combination in the **extrapolation regime** (N values tested "
        "with MPNN zero-shot prediction).",
        "> It does NOT average over h — it tracks the "
        "hardest operating point near h_critical.",
        "> For in-distribution quality (training N), see `model_evaluation_report.md`.",
        "> Grade thresholds: A (<3%), B (<5%), C (<10%), D (<50%), F (≥50%).",
        "",
        "---",
        "",
    ]

    # ── Global summary table ─────────────────────────────────────────────
    lines.append("## Summary: Best Grade per Topology")
    lines.append("")
    lines.append(
        "| Topology | Max N evaluated | Best grade | "
        "Best ΔE/gap (any N) | Best model type | N trained up to |"
    )
    lines.append("|---|---|---|---|---|---|")

    for topo in sorted(best_by_topo.keys()):
        n_results = best_by_topo[topo]
        max_n = max(n_results.keys())
        # Best grade across all N
        best_grade_entry = min(n_results.values(), key=lambda r: r.best_de_gap)
        best_grade = best_grade_entry.grade
        best_dg = best_grade_entry.best_de_gap
        best_type = best_grade_entry.model_type
        # Max N with reasonable results
        reasonable_ns = [n for n, r in n_results.items() if r.best_de_gap < 10.0]
        max_n_trained = max(reasonable_ns) if reasonable_ns else max_n

        lines.append(
            f"| {topo} | {max_n} | {best_grade} | "
            f"{best_dg:.4f} | {best_type} | {max_n_trained} |"
        )

    lines.extend(["", "---", ""])

    # ── Per-topology detailed tables ─────────────────────────────────────
    for topo in sorted(best_by_topo.keys()):
        n_results = best_by_topo[topo]

        lines.append(f"## {topo}")
        lines.append("")

        # Determine actual h used (should be same for all, but might vary slightly)
        h_values_used = sorted(set(r.h_used for r in n_results.values()))
        if len(h_values_used) == 1:
            lines.append(f"**h used**: {h_values_used[0]:.3f}")
        else:
            lines.append(
                f"**h used**: varies ({min(h_values_used):.3f} – {max(h_values_used):.3f})"
            )
        lines.append("")

        lines.append(
            "| N | ΔE/gap | |ΔE| | |ΔE|/N | gap | Grade | "
            "Model | Checkpoint | Date | Source |"
        )
        lines.append(
            "|--:|-------:|-----:|------:|----:|:-----:|"
            ":-----:|-----------|------|--------|"
        )

        for n in sorted(n_results.keys()):
            r = n_results[n]
            # Truncate checkpoint for readability
            ckpt_short = r.checkpoint
            if len(ckpt_short) > 40:
                ckpt_short = ckpt_short[:37] + "..."

            # Source: link to report (and JSON if available)
            source_link = f"`{Path(r.report_file).name}`"
            if r.run_json:
                source_link += f" ([json]({r.run_json}))"

            lines.append(
                f"| {n} | {r.best_de_gap:.4f} | {r.best_abs_error:.4f} | "
                f"{r.best_abs_error_per_site:.2e} | {r.gap_at_best:.4f} | "
                f"{r.grade} | {r.model_type} | "
                f"{ckpt_short} | {r.date[:10] if r.date else '—'} | "
                f"{source_link} |"
            )

        lines.extend(["", ""])

    # ── Footer ───────────────────────────────────────────────────────────
    lines.extend(
        [
            "---",
            "",
            "*Auto-generated by `scripts/analysis/generate_best_results_scoreboard.py`*",
            f"*Data sources: `{EVAL_DIR.relative_to(ROOT)}/` + `{COMPARISON_DIR.relative_to(ROOT)}/`*",
        ]
    )

    return "\n".join(lines)


def cross_validate_with_registry(
    best_by_topo: dict[str, dict[int, BestResult]],
) -> list[str]:
    """Cross-validate scoreboard results against ModelRegistryDB evaluations.

    Checks:
    1. Does the zoo pass_rate_by_n align with what we found as "best"?
    2. Are there discrepancies between eval reports and DB-recorded pass rates?
    3. Is the registered "best model" the same we identified?

    Returns a list of warning/info lines to include in the report.
    """
    validation_lines: list[str] = []

    try:
        from qmbp_simulation.predictors.model_zoo import _load_manifest
        from qmbp_simulation.predictors.model_registry_db import ModelRegistryDB

        db = ModelRegistryDB()
        manifest = _load_manifest()

        # Build a map: checkpoint_file → zoo entry
        zoo_map = {e.checkpoint_file: e for e in manifest}

        for topo, n_results in sorted(best_by_topo.items()):
            for n, best in sorted(n_results.items()):
                ckpt_name = Path(best.checkpoint).name
                if ckpt_name not in zoo_map:
                    # Try partial match
                    matches = [k for k in zoo_map if ckpt_name.startswith(k[:20])]
                    if not matches:
                        continue
                    ckpt_name = matches[0]

                zoo_entry = zoo_map[ckpt_name]

                # Check 1: pass_rate_by_n consistency
                if zoo_entry.pass_rate_by_n:
                    n_str = str(n)
                    if n_str in zoo_entry.pass_rate_by_n:
                        zoo_pass_at_n = float(zoo_entry.pass_rate_by_n[n_str])
                        # If zoo says 0% pass but we show grade A/B, flag discrepancy
                        if best.grade in ("A", "B") and zoo_pass_at_n < 0.20:
                            validation_lines.append(
                                f"⚠️ {topo} N={n}: scoreboard grade={best.grade} but "
                                f"zoo pass_rate_by_n[{n}]={zoo_pass_at_n:.0%} — "
                                f"possible stale zoo data"
                            )
                        # If zoo says good pass but we show F, flag
                        elif best.grade == "F" and zoo_pass_at_n > 0.50:
                            validation_lines.append(
                                f"⚠️ {topo} N={n}: scoreboard grade=F (ΔE/gap={best.best_de_gap:.3f}) "
                                f"but zoo pass_rate_by_n[{n}]={zoo_pass_at_n:.0%} — "
                                f"possible inflated zoo or h-grid difference"
                            )

                # Check 2: Registry evaluation records
                record = db.get_model(ckpt_name)
                if record and record.evaluations:
                    latest_eval = record.evaluations[-1]
                    # If latest eval target_n includes this N, cross-check
                    if n in (latest_eval.target_n_values or []):
                        if (latest_eval.mean_de_gap > 0 and
                                abs(best.best_de_gap - latest_eval.mean_de_gap) / max(latest_eval.mean_de_gap, 1e-8) > 1.0):
                            # Our per-h best is very different from the registry's mean
                            # This is expected (we pick best h, registry averages all h)
                            # Only flag if direction is inverted
                            if best.best_de_gap > latest_eval.mean_de_gap * 3:
                                validation_lines.append(
                                    f"⚠️ {topo} N={n}: scoreboard ΔE/gap@h=2.5 = {best.best_de_gap:.3f} "
                                    f">> registry mean ΔE/gap = {latest_eval.mean_de_gap:.3f} — "
                                    f"h=2.5 is anomalously hard for this config"
                                )

    except ImportError:
        validation_lines.append("ℹ️ ModelRegistryDB not available — cross-validation skipped")
    except Exception as e:
        validation_lines.append(f"ℹ️ Cross-validation error: {e}")

    return validation_lines


def generate_scoreboard(
    target_h: float = TARGET_H,
    output_json: bool = False,
) -> dict:
    """Main entry point: scan reports, compute best, write markdown.

    Returns summary dict for programmatic use.
    """
    entries = scan_all_reports(target_h=target_h)

    if not entries:
        print("  ⚠️ No evaluation reports found in results/extrapolation_evals/")
        return {"n_entries": 0, "best_by_topo": {}}

    best_by_topo = compute_best_per_topology_n(entries)

    # Count unique reports
    unique_reports = set(e.report_file for e in entries)
    n_reports = len(unique_reports)

    # Write markdown
    md_content = format_markdown(best_by_topo, target_h, n_reports)

    # Cross-validate with ModelRegistryDB
    validation_warnings = cross_validate_with_registry(best_by_topo)
    if validation_warnings:
        # Append validation section to markdown
        validation_section = [
            "",
            "---",
            "",
            "## Cross-Validation (vs ModelRegistryDB)",
            "",
        ]
        if any(w.startswith("⚠️") for w in validation_warnings):
            validation_section.append("| Issue | Detail |")
            validation_section.append("|---|---|")
            for w in validation_warnings:
                validation_section.append(f"| {w[:2]} | {w[2:].strip()} |")
        else:
            for w in validation_warnings:
                validation_section.append(f"- {w}")
        validation_section.append("")
        md_content += "\n".join(validation_section)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(md_content)
    print(f"  📊 Best Results Scoreboard: {OUTPUT_PATH.relative_to(ROOT)}")

    # Prepare JSON summary
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_h": target_h,
        "n_reports_scanned": n_reports,
        "n_entries_parsed": len(entries),
        "best_by_topology": {},
    }

    for topo, n_results in sorted(best_by_topo.items()):
        summary["best_by_topology"][topo] = {
            str(n): asdict(r) for n, r in sorted(n_results.items())
        }

    if output_json:
        OUTPUT_JSON_PATH.write_text(json.dumps(summary, indent=2, default=str))
        print(f"  📄 JSON output: {OUTPUT_JSON_PATH.relative_to(ROOT)}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Best Results Scoreboard from extrapolation eval reports"
    )
    parser.add_argument(
        "--target-h",
        type=float,
        default=TARGET_H,
        help=f"Reference h-value to track (default: {TARGET_H})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also write JSON output",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=H_TOLERANCE,
        help=f"Tolerance for h-matching (default: ±{H_TOLERANCE})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global H_TOLERANCE
    H_TOLERANCE = args.tolerance

    summary = generate_scoreboard(target_h=args.target_h, output_json=args.json)

    if summary["n_entries_parsed"] == 0:
        return 1

    # Print quick summary
    print(f"\n  Scanned {summary['n_reports_scanned']} reports, "
          f"found {summary['n_entries_parsed']} entries at h≈{args.target_h}")

    for topo, n_results in sorted(summary["best_by_topology"].items()):
        n_vals = sorted(int(k) for k in n_results.keys())
        best_grade = min(n_results.values(), key=lambda r: r["best_de_gap"])
        print(
            f"  {topo:14s}: N={n_vals}, "
            f"best ΔE/gap={best_grade['best_de_gap']:.4f} ({best_grade['grade']})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
